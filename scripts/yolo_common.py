"""Pinned YOLO experiment helpers. No model weights are distributed here."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from cpg import load_pipeline
from cpg.validation import photo_cases, check_expected_object, box_iou

WEIGHTS_SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
LIMITS = {'tensor_atol': 0.0005, 'dense_box_atol': 4., 'dense_box_rtol': .01,
          'dense_score_atol': .03, 'dense_score_rtol': .03,
          'detection_iou': .95, 'detection_score_atol': .03,
          'cross_engine_box_score_floor': .01, 'confidence': .5, 'nms_iou': .7, 'max_detections': 300}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(weights, *, fuse=True):
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != '8.3.200':
        raise ValueError('This experiment requires ultralytics==8.3.200')
    if digest(weights) != WEIGHTS_SHA256:
        raise ValueError('YOLO weights do not match the pinned checksum')
    model = YOLO(str(weights)).model.float().eval()
    # Match the pinned upstream export preparation before any FP16 conversion.
    return model.fuse(verbose=False) if fuse else model


def pipeline():
    pipe = load_pipeline('examples/yolo.yaml')
    plan = pipe.plan((1080, 1920, 3))
    size, norm, out = pipe.operations
    if (pipe.input_encoding != 'bgr8' or plan['output_shape'] != [1,3,640,640]
            or size.value != 114 or out.dtype != 'float16'
            or norm.mean != (0.,0.,0.) or norm.std != (1.,1.,1.) or norm.scale != 1/255):
        raise ValueError('The experiment requires the documented fixed YOLO contract')
    return pipe


class Dense(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        result = self.model(image)
        return result[0] if isinstance(result, tuple) else result


def decode(dense):
    """Shared single-image, single-label class-aware NMS, on the input device."""
    if tuple(dense.shape) != (1,84,8400):
        raise ValueError('Expected the fixed YOLOv8n dense output shape')
    rows = dense[0].T.float()
    scores, labels = rows[:,4:].max(dim=1)
    keep = scores >= LIMITS['confidence']
    xywh, scores, labels = rows[keep,:4], scores[keep], labels[keep]
    boxes = torch.cat((xywh[:,:2]-xywh[:,2:]/2, xywh[:,:2]+xywh[:,2:]/2), dim=1)
    selected = batched_nms(boxes, scores, labels, LIMITS['nms_iou'])[:LIMITS['max_detections']]
    return {'boxes':boxes[selected], 'scores':scores[selected], 'labels':labels[selected]}


def detection_record(detection, geometry, expectation):
    values = {key:value.detach().cpu().tolist() for key,value in detection.items()}
    values['source_boxes'] = [geometry.source_box(box) for box in values['boxes']]
    # Fixture annotations use sparse COCO IDs. YOLO uses contiguous class indices.
    fixture_labels = [{0:1, 15:17}.get(label, -1) for label in values['labels']]
    values['expected_object_iou'] = check_expected_object(
        fixture_labels, values['scores'], values['source_boxes'], expectation)
    return values


def compare_detections(reference, actual):
    """Require a one-to-one match, including no missing or extra confident detections."""
    a = {k:v.detach().cpu().numpy() for k,v in reference.items()}
    b = {k:v.detach().cpu().numpy() for k,v in actual.items()}
    if len(a['labels']) != len(b['labels']):
        raise AssertionError('Confident detection counts differ')
    unused = set(range(len(b['labels'])))
    matches = []
    for box, score, label in zip(a['boxes'], a['scores'], a['labels']):
        candidates = [(box_iou(box,b['boxes'][j]),j) for j in unused if b['labels'][j] == label]
        overlap,j = max(candidates, default=(0.,-1))
        if overlap < LIMITS['detection_iou'] or abs(float(score)-float(b['scores'][j])) > LIMITS['detection_score_atol']:
            raise AssertionError('Confident detections exceed declared IoU/score limits')
        unused.remove(j)
        matches.append(float(overlap))
    return matches


def compare_dense(reference, actual, *, cross_engine=False):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError('Non-finite dense output')
    if reference.shape != actual.shape or tuple(actual.shape) != (1,84,8400):
        raise AssertionError('Unexpected dense output shape')
    for value in (reference,actual):
        if (value[:,:4][:,2:] < 0).any() or (value[:,4:] < 0).any() or (value[:,4:] > 1).any():
            raise AssertionError('Invalid box dimensions or class probabilities')
    result = {}
    for name, part in [('box',slice(0,4)), ('score',slice(4,None))]:
        a,b = reference[:,part].float(),actual[:,part].float()
        result[name+'_max_abs_error'] = float((a-b).abs().max())
        if cross_engine and name == 'box':
            # Cross-engine FP16 background coordinates can drift despite unchanged detections.
            # Keep all scores checked, and keep same-engine CPG comparisons unmasked.
            relevant = torch.maximum(reference[:,4:].float().amax(1),actual[:,4:].float().amax(1)) >= LIMITS['cross_engine_box_score_floor']
            violations = (a-b).abs() > (LIMITS['dense_box_atol']+LIMITS['dense_box_rtol']*b.abs())
            result['background_box_limit_violations'] = int((violations & ~relevant[:,None,:]).sum())
            result['cross_engine_relevant_proposals'] = int(relevant.sum())
            a,b = a.permute(0,2,1)[relevant],b.permute(0,2,1)[relevant]
        torch.testing.assert_close(a,b,atol=LIMITS['dense_'+name+'_atol'],rtol=LIMITS['dense_'+name+'_rtol'])
    return result


def cases():
    originals = []
    for item in photo_cases(Path('tests/fixtures/images/manifest.json')):
        yield item
        if item[3]['transform'] == 'none':
            originals.append(item)
    for name,rgb,expectation,provenance in originals:
        enlarged = np.repeat(np.repeat(rgb,2,axis=0),2,axis=1)
        h,w,_ = enlarged.shape
        top,left = (1080-h)//2,(1920-w)//2
        canvas = np.full((1080,1920,3),114,np.uint8)
        canvas[top:top+h,left:left+w] = enlarged
        expected = dict(expectation)
        expected['box_xyxy'] = (np.asarray(expectation['box_xyxy'])*2+[left,top,left,top]).tolist()
        yield name+'-1080p',canvas,expected,dict(provenance,transform=f'repeat 2x; pad top {top}, left {left} to 1080x1920')

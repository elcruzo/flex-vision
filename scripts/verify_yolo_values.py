"""Recheck saved GPU tensors, dense scores, and detections with NumPy on the Mac."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpg.reference import numpy_reference
from cpg.validation import box_iou, check_expected_object
from yolo_common import cases, pipeline, WEIGHTS_SHA256


def decode_numpy(head):
    rows=head[0].T.astype(np.float32)
    labels=rows[:,4:].argmax(1)
    scores=rows[np.arange(len(rows)),labels+4]
    keep=scores >= .5
    xywh,scores,labels=rows[keep,:4],scores[keep],labels[keep]
    boxes=np.concatenate((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2),axis=1)
    pending=np.argsort(-scores,kind='stable').tolist()
    selected=[]
    while pending and len(selected)<300:
        index=pending.pop(0);selected.append(index)
        pending=[j for j in pending if labels[j] != labels[index] or box_iou(boxes[index],boxes[j]) <= .7]
    return boxes[selected],scores[selected],labels[selected]


def verify(root):
    report=json.loads((root/'results.json').read_text())
    if report['status'] != 'passed' or report['weights_sha256'] != WEIGHTS_SHA256:
        raise ValueError('Require a passed run with pinned model weights')
    checked=[]
    pipe=pipeline()
    for name,rgb,expectation,provenance in cases():
        expected,g=numpy_reference(pipe,np.ascontiguousarray(rgb[...,::-1]))
        controls=np.load(root/(name+'-controls.npz'))
        np.testing.assert_array_equal(controls['tensor'],expected)
        reference=controls['tensorrt_dense'].astype(np.float32)
        rb,rs,rl=decode_numpy(reference)
        for candidate in ('cpg','torch','cvcuda'):
            archive=np.load(root/(name+'-'+candidate+'.npz'))
            tensor,head=archive['tensor'],archive['dense'].astype(np.float32)
            if tensor.shape != (1,3,640,640) or tensor.dtype != np.float16 or head.shape != (1,84,8400):
                raise ValueError('Unexpected tensor/dense contract')
            np.testing.assert_allclose(tensor,expected,atol=.0005,rtol=0)
            if (not np.isfinite(head).all() or (head[:,2:4]<0).any()
                    or (head[:,4:]<0).any() or (head[:,4:]>1).any()):
                raise ValueError('Invalid dense values')
            if candidate == 'cpg':
                np.testing.assert_array_equal(tensor.view(np.uint16),expected.view(np.uint16))
            score_tol = .003 if candidate == 'cpg' else .03
            np.testing.assert_allclose(head[:,4:],reference[:,4:],atol=score_tol,rtol=score_tol)
            relevant = np.ones((1,8400),dtype=bool) if candidate == 'cpg' else np.maximum(head[:,4:].max(1),reference[:,4:].max(1)) >= .01
            a,b=head[:,:4].transpose(0,2,1)[relevant],reference[:,:4].transpose(0,2,1)[relevant]
            np.testing.assert_allclose(a,b,atol=4,rtol=.01)
            boxes,scores,labels=decode_numpy(head)
            if len(boxes) != len(rb): raise ValueError('Detection counts differ')
            unused=set(range(len(rb)));ious=[]
            for box,score,label in zip(boxes,scores,labels):
                overlap,index=max([(box_iou(box,rb[j]),j) for j in unused if rl[j]==label],default=(0,-1))
                if overlap < .95 or abs(float(score)-float(rs[index])) > .03:
                    raise ValueError('Detection class/box/score mismatch')
                unused.remove(index);ious.append(overlap)
            source_boxes=[g.source_box(box.tolist()) for box in boxes]
            fixture_labels=[{0:1,15:17}.get(int(label),-1) for label in labels]
            semantic_iou=check_expected_object(fixture_labels,scores,source_boxes,expectation)
            checked.append({'fixture':name,'candidate':candidate,'tensor_max_abs_error':float(np.abs(tensor.astype(np.float32)-expected.astype(np.float32)).max()),
                'matched_ious':ious,'expected_object_iou':semantic_iou})
    return {'status':'independently_verified','checks':checked,
            'scope':'NumPy recheck of exported tensors, same-engine dense outputs, and independent CPU NMS; not a new GPU run'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify(args.root)
    with args.output.open('x') as file: json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()

"""Validate inspection preprocessing through real classifier inference.

CPU/MPS use CPU inference. CUDA uses resident CUDA inference before validation downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from cpg.validation import photo_cases
from inspection_reference import numpy_inspection, torch_inspection, gaussian_coefficients, SHARPEN, MEAN, STD, TorchInspectionBaseline
from local_detector import digest

WEIGHTS_URL = 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
WEIGHTS_SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'


def cases():
    for name,rgb,_,provenance in photo_cases(Path('tests/fixtures/images/manifest.json')):
        if name == 'chelsea':
            h,w,_ = rgb.shape
            yield name,rgb,(0,0,w,h),provenance
            odd = np.full((h+33,w+39,3),114,dtype=np.uint8)
            odd[11:11+h,13:13+w] = rgb
            yield 'chelsea-odd-roi',odd,(13,11,w,h),dict(provenance,transform='pad top/left 11/13, bottom/right 22/26')
            enlarged = np.repeat(np.repeat(rgb,6,axis=0),6,axis=1)
            frame = np.full((2160,3840,3),114,dtype=np.uint8)
            eh,ew,_ = enlarged.shape
            top,left = (2160-eh)//2,(3840-ew)//2
            frame[top:top+eh,left:left+ew] = enlarged
            yield 'chelsea-4k-roi',frame,(left,top,ew,eh),dict(provenance,transform='repeat pixels 6x; centered 4K padding value 114')
            return
    raise ValueError('Pinned Chelsea fixture is missing')


def check_semantics(logits, *, negative=False):
    probabilities = logits.softmax(dim=1)[0]
    labels = probabilities.topk(5).indices.tolist()
    allowed = set(range(151,269) if negative else range(281,286))
    mass = float(probabilities[list(sorted(allowed))].sum())
    if not allowed.intersection(labels) or mass < .2:
        raise AssertionError(f'Expected {"dog (negative control)" if negative else "cat"} in top five with group probability >= 0.2; top5={labels}, mass={mass:.6f}')
    return labels,mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=3,help='CUDA warmup passes per fixture, outside the NVTX range')
    parser.add_argument('--negative-control',action='store_true',help='Require an absent dog class; the run must fail')
    parser.add_argument('--weights',type=Path,default=Path('.cache/models/mobilenet_v3_small-047dcff4.pth'))
    args = parser.parse_args()
    if not 0 <= args.warmup <= 100:
        parser.error('warmup must be between 0 and 100')
    if not args.weights.is_file() or digest(args.weights) != WEIGHTS_SHA256:
        parser.error(f'Expected pinned model weights: {WEIGHTS_URL} (SHA256 {WEIGHTS_SHA256})')
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        parser.error('Disable MPS CPU fallback for this explicit device experiment')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'failed','scope':'inspection baseline and classifier smoke; no industrial accuracy, TensorRT, or performance claim',
              'warmup_per_case':args.warmup if args.device == 'cuda' else 0,
              'preprocessing_device':args.device,'inference_device':'cuda' if args.device == 'cuda' else 'cpu','negative_control':args.negative_control,
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
              'weights_url':WEIGHTS_URL,'weights_sha256':WEIGHTS_SHA256,'model':'MobileNet_V3_Small.IMAGENET1K_V1',
              'runner_sha256':digest(Path(__file__)),'reference_sha256':digest(Path('scripts/inspection_reference.py')),
              'fixture_manifest_sha256':digest(Path('tests/fixtures/images/manifest.json')),
              'recipe':{'gaussian':{'size':5,'sigma':1.2,'coefficients':gaussian_coefficients().tolist()},
                        'sharpen':SHARPEN.tolist(),'border':'replicate','clamp':[0,255],
                        'order':['bgr-to-rgb','gaussian-horizontal','gaussian-vertical','sharpen','clamp','crop','resize','normalize','nchw'],
                        'resize':{'width':224,'height':224,'coordinates':'half-pixel','antialias':False},
                        'intermediate_dtype':'float32','output_dtype':'float32','scale':1/255,'mean':MEAN,'std':STD},
              'tensor_atol':2e-4,'logit_atol':.001,'logit_rtol':1e-4,'cases':[]}
    try:
        torch.set_num_threads(4)
        model = mobilenet_v3_small(weights=None).eval()
        model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
        gpu_model = None
        prepared = None
        if args.device == 'cuda':
            import copy
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            gpu_model = copy.deepcopy(model).cuda().eval()
            prepared = TorchInspectionBaseline('cuda')
            report['gpu'] = torch.cuda.get_device_name()
            report['cuda'] = torch.version.cuda
            report['tf32'] = False
        categories = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta['categories']
        for name,rgb,roi,provenance in cases():
            frame = rgb[...,::-1]  # Deliberately noncontiguous BGR input.
            case = {'id':name,'shape':list(frame.shape),'roi_xywh':roi,'provenance':provenance,
                    'input_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'status':'failed'}
            report['cases'].append(case)
            started = time.perf_counter()
            expected = numpy_inspection(frame,roi)
            if prepared is not None:
                # Explicit fixture upload precedes the GPU preprocessing/inference boundary.
                resident = torch.from_numpy(np.ascontiguousarray(frame)).cuda()
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        gpu_model(prepared.from_bgr8(resident,roi))
                    torch.cuda.synchronize()
                with torch.inference_mode(), torch.cuda.nvtx.range('inspection_to_classifier'):
                    actual_device = prepared.from_bgr8(resident,roi)
                    candidate_device = gpu_model(actual_device)
                    torch.cuda.synchronize()
                # Downloads below are validation only, after real resident inference.
                actual = actual_device.cpu()
            else:
                actual = torch_inspection(frame,roi,device=args.device).cpu()
            np.testing.assert_allclose(actual.numpy(),expected,atol=report['tensor_atol'],rtol=0)
            case['tensor_max_abs_error'] = float(np.abs(actual.numpy()-expected).max())
            with torch.inference_mode():
                baseline = model(torch.from_numpy(expected))
                candidate = candidate_device.cpu() if prepared is not None else model(actual)
            torch.testing.assert_close(candidate,baseline,atol=report['logit_atol'],rtol=report['logit_rtol'])
            case['logit_max_abs_error'] = float((candidate-baseline).abs().max())
            baseline_top,baseline_mass = check_semantics(baseline,negative=args.negative_control)
            candidate_top,candidate_mass = check_semantics(candidate,negative=args.negative_control)
            if baseline_top != candidate_top:
                raise AssertionError('Top-five class ordering differs')
            case.update(top5=candidate_top,top5_names=[categories[i] for i in candidate_top],
                        baseline_group_probability=baseline_mass,candidate_group_probability=candidate_mass)
            np.savez_compressed(args.output/f'{name}.npz',expected=expected,actual=actual.numpy(),
                                baseline_logits=baseline.numpy(),candidate_logits=candidate.numpy())
            case['diagnostic_seconds'] = time.perf_counter()-started
            case['status'] = 'passed'
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))


if __name__ == '__main__':
    main()

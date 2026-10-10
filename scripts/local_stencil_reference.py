"""Explicit CPU/MPS stencil references through pinned classifier inference."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from cpg import Pipeline, load_pipeline
from cpg.reference import numpy_reference, torch_reference
from cpg.validation import photo_cases
from local_classifier import WEIGHTS_SHA256, check_semantics
from local_detector import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise ValueError('Disable MPS CPU fallback for this explicit reference experiment')
    weights=Path('.cache/models/mobilenet_v3_small-047dcff4.pth')
    if digest(weights)!=WEIGHTS_SHA256:raise ValueError('Require pinned classifier weights')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='failed',revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                torch_version=torch.__version__,platform=platform.platform(),python=platform.python_version(),
                mps_fallback_environment=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                fixture_manifest_sha256=digest(Path('tests/fixtures/images/manifest.json')),
                preprocessing_device=args.device,inference_device='cpu',weights_sha256=WEIGHTS_SHA256,
                scope='Explicit stencil reference and real classifier checks; no CUDA or performance acceptance',
                tensor_atol=2e-4,logit_atol=.001,logit_rtol=1e-4,checks=[])
    try:
        if report['dirty']:raise ValueError('Require clean committed source')
        torch.set_num_threads(4)
        model=mobilenet_v3_small(weights=None).eval()
        model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        rgb=next(rgb for name,rgb,_,_ in photo_cases(Path('tests/fixtures/images/manifest.json')) if name=='chelsea')
        frame=rgb[...,::-1]
        report['input_sha256']=hashlib.sha256(frame.tobytes()).hexdigest()
        for size in (3,5,7,9):
            kernel=[[0.]*size for _ in range(size)]
            kernel[0][0]=.125;kernel[size//2][size//2]=.75;kernel[-1][0]=.125
            pipe=(Pipeline(input_encoding='bgr8').conv2d(kernel).letterbox(224,224)
                  .normalize([.485,.456,.406],[.229,.224,.225],scale=1/255).to(dtype='float32',layout='NCHW'))
            # Exercise relative file loading as well as the literal frontend.
            (args.output/f'kernel-{size}.yaml').write_text('kernel: '+json.dumps(kernel)+'\n')
            config=args.output/f'pipeline-{size}.yaml'
            config.write_text('input: {encoding: bgr8}\npipeline:\n  - convolution: {kernel: kernel-'+str(size)+'.yaml}\n  - letterbox: {width: 224, height: 224}\n  - normalize: {mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], scale: '+str(1/255)+'}\noutput: {dtype: float32, layout: nchw}\n')
            loaded=load_pipeline(config)
            if loaded!=pipe:raise AssertionError('Literal and kernel-file graphs differ')
            expected,_=numpy_reference(pipe,frame)
            actual,_=torch_reference(loaded,frame,device=args.device)
            actual=actual.cpu()
            np.testing.assert_allclose(actual.numpy(),expected,atol=report['tensor_atol'],rtol=0)
            with torch.inference_mode():
                baseline=model(torch.from_numpy(expected));candidate=model(actual)
            torch.testing.assert_close(candidate,baseline,atol=report['logit_atol'],rtol=report['logit_rtol'])
            labels,mass=check_semantics(candidate)
            if check_semantics(baseline)[0]!=labels:raise AssertionError('Classifier top-five order differs')
            report['checks'].append(dict(size=size,tensor_max_abs_error=float(np.abs(actual.numpy()-expected).max()),
                                         logit_max_abs_error=float((candidate-baseline).abs().max()),top5=labels,cat_probability=mass,
                                         literal_and_file_equal=True,plan=pipe.plan(frame.shape)))
        report['status']='passed'
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

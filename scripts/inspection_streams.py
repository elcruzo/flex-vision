"""Validate producer/preprocessor/consumer streams through real inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

from cpg import InspectionPipeline
from inspection_reference import numpy_inspection
from local_classifier import cases, check_semantics, WEIGHTS_SHA256
from local_detector import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('.cache/models/mobilenet_v3_small-047dcff4.pth'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required; no CPU fallback')
    if digest(args.weights) != WEIGHTS_SHA256:
        parser.error('Expected pinned MobileNetV3 weights')
    if subprocess.check_output(['git','status','--porcelain'],text=True):
        parser.error('Commit the measured source first')
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'failed','revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'runtime_sha256':digest(Path('src/cpg/inspection.py')),'runner_sha256':digest(Path(__file__)),
              'weights_sha256':WEIGHTS_SHA256,'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(),'cases':[],
              'scope':'same-device stream ordering and allocator lifetime stress; no throughput claim'}
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        cpu_model = mobilenet_v3_small(weights=None).eval()
        cpu_model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
        import copy
        model = copy.deepcopy(cpu_model).cuda().eval()
        streams = [torch.cuda.Stream() for _ in range(4)]
        setup, producer, processor, consumer = streams
        model_ready = torch.cuda.Event()
        model_ready.record()
        consumer.wait_event(model_ready)
        with torch.inference_mode():
            for name,rgb,roi,_ in cases():
                variants = [np.ascontiguousarray(rgb[...,::-1]),np.ascontiguousarray(rgb[:,::-1,::-1])]
                expected = [numpy_inspection(frame,roi) for frame in variants]
                expected_logits = [cpu_model(torch.from_numpy(x)) for x in expected]
                sources = [torch.from_numpy(frame).cuda() for frame in variants]
                uploaded = torch.cuda.Event()
                uploaded.record()
                producer.wait_event(uploaded)
                pending = []
                for strategy in ('full','roi'):
                    with torch.cuda.stream(setup):
                        prepared = InspectionPipeline(roi,strategy).prepare(variants[0].shape)
                    for iteration in range(4):
                        variant = iteration % 2
                        with torch.cuda.stream(producer):
                            frame = sources[variant].clone()
                            input_ready = torch.cuda.Event()
                            input_ready.record()
                        with torch.cuda.stream(processor):
                            result = prepared.submit(frame,ready_event=input_ready)
                        del frame
                        with torch.cuda.stream(consumer):
                            tensor = result.wait()
                            logits = model(tensor)
                            snapshot = tensor.clone()
                        retained = tensor if iteration % 2 == 0 else None
                        pending.append((strategy,iteration,variant,logits,snapshot,retained))
                        del tensor,result
                        # Reuse similarly sized allocator blocks on both origin streams.
                        with torch.cuda.stream(producer):
                            churn = torch.empty_like(sources[variant]).zero_()
                            del churn
                        with torch.cuda.stream(processor):
                            churn = torch.empty((1,3,224,224),device='cuda').zero_()
                            del churn
                    # Constants must remain valid for already queued work after deletion.
                    del prepared
                consumer.synchronize()
                for strategy,iteration,variant,logits,snapshot,retained in pending:
                    actual = snapshot.cpu().numpy()
                    np.testing.assert_allclose(actual,expected[variant],atol=2e-4,rtol=0)
                    got = logits.cpu()
                    torch.testing.assert_close(got,expected_logits[variant],atol=.001,rtol=1e-4)
                    labels,_ = check_semantics(got)
                    wanted,_ = check_semantics(expected_logits[variant])
                    assert labels == wanted, 'Top-five ordering differs'
                    if retained is not None:
                        torch.testing.assert_close(retained.cpu(),snapshot.cpu(),atol=0,rtol=0)
                    report['cases'].append({'fixture':name,'strategy':strategy,'iteration':iteration,
                        'variant':variant,'input_sha256':hashlib.sha256(variants[variant].tobytes()).hexdigest(),
                        'tensor_max_abs_error':float(np.abs(actual-expected[variant]).max()),
                        'logit_max_abs_error':float((got-expected_logits[variant]).abs().max()),
                        'retained':retained is not None,'top5':labels,'status':'passed'})
                for stream in streams:
                    stream.synchronize()
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))


if __name__ == '__main__':
    main()

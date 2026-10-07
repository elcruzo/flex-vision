"""Measure CUDA inspection through real classifier inference.

Validate each selected implementation with local_classifier.py --device cuda first.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torchvision
from torchvision.models import mobilenet_v3_small
from cpg import InspectionPipeline

from inspection_reference import TorchInspectionBaseline
from inspection_roi import TorchInspectionROI
from local_classifier import cases, WEIGHTS_SHA256
from local_detector import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--candidate-validation',type=Path,help='Compare the ROI candidate using its matching CUDA validation')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=30)
    parser.add_argument('--weights',type=Path,default=Path('.cache/models/mobilenet_v3_small-047dcff4.pth'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required; no CPU fallback')
    if not 1 <= args.iterations <= 10000 or not 1 <= args.repeats <= 20 or not 1 <= args.warmup <= 1000:
        parser.error('Invalid sample or warmup count')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True):
        parser.error('Commit the measured source first')
    validations = {'baseline': json.loads(args.validation.read_text())}
    if args.candidate_validation:
        candidate = json.loads(args.candidate_validation.read_text())
        candidate_name = candidate.get('implementation')
        if candidate_name not in ('roi','planned-full','planned-roi'):
            parser.error('Candidate validation must identify roi, planned-full, or planned-roi')
        validations[candidate_name] = candidate
    for implementation, validation in validations.items():
      if (validation.get('status') != 'passed' or validation.get('negative_control') or
        validation.get('implementation', 'baseline') != implementation or
        validation.get('runner_sha256') != digest(Path('scripts/local_classifier.py')) or
        (implementation == 'roi' and validation.get('candidate_sha256') != digest(Path('scripts/inspection_roi.py'))) or
        (implementation.startswith('planned-') and validation.get('inspection_runtime_sha256') != digest(Path('src/cpg/inspection.py'))) or
        validation.get('preprocessing_device') != 'cuda' or validation.get('inference_device') != 'cuda' or
        validation.get('revision') != revision or validation.get('dirty') or
        validation.get('torch') != torch.__version__ or validation.get('torchvision') != torchvision.__version__ or
        validation.get('gpu') != torch.cuda.get_device_name() or validation.get('cuda') != torch.version.cuda or
        validation.get('fixture_manifest_sha256') != digest(Path('tests/fixtures/images/manifest.json')) or
        validation.get('reference_sha256') != digest(Path('scripts/inspection_reference.py')) or
        validation.get('weights_sha256') != WEIGHTS_SHA256 or digest(args.weights) != WEIGHTS_SHA256):
        parser.error('Require passing CUDA classifier validation from this clean revision and model')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = mobilenet_v3_small(weights=None).eval()
    model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
    model = model.cuda()
    implementations = {'baseline': TorchInspectionBaseline('cuda')}
    if args.candidate_validation:
        implementations[candidate_name] = TorchInspectionROI('cuda') if candidate_name == 'roi' else None
    report = {'status':'failed','scope':'serial unfused CUDA baseline through PyTorch classifier; not a CPG comparison',
              'revision':revision,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
              'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'weights_sha256':WEIGHTS_SHA256,
              'validation_sha256':digest(args.validation),'runner_sha256':digest(Path(__file__)),
              'iterations':args.iterations,'repeats':args.repeats,'warmup':args.warmup,'tf32':False,
              'timing':'CUDA events for stream stages; perf_counter_ns for completed host call; no outlier removal',
              'cases':[]}
    if args.candidate_validation:
        report.update(scope='matched full-frame versus ROI reference through PyTorch classifier',
                      candidate_validation_sha256=digest(args.candidate_validation),
                      candidate_implementation=candidate_name,
                      candidate_sha256=digest(Path('scripts/inspection_roi.py') if candidate_name == 'roi' else Path('src/cpg/inspection.py')),
                      order='baseline/candidate on even repeats; candidate/baseline on odd repeats')
    metrics = ['preprocess_ms','inference_ms','stream_total_ms','host_total_ms']
    columns = ['case','implementation','repeat','iteration',*metrics]
    try:
        with (args.output/'samples.csv').open('x',newline='') as file, torch.inference_mode():
            writer = csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()
            for name,rgb,roi,_ in cases():
                frame = np.ascontiguousarray(rgb[...,::-1])
                for validation in validations.values():
                    checked = {c['id']:c for c in validation['cases']}.get(name,{})
                    if checked.get('status') != 'passed' or checked.get('input_sha256') != hashlib.sha256(frame.tobytes()).hexdigest() or checked.get('roi_xywh') != list(roi):
                        raise ValueError('Fixture is not covered by the passing validation')
                resident = torch.from_numpy(frame).cuda()
                executors = {}
                for implementation, prepared in implementations.items():
                    if implementation.startswith('planned-'):
                        executors[implementation] = InspectionPipeline(roi,implementation.removeprefix('planned-')).prepare(frame.shape)
                    else:
                        executors[implementation] = lambda image, prepared=prepared, roi=roi: prepared.from_bgr8(image,roi)
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                rows = []
                for repeat in range(args.repeats):
                  order = list(implementations)
                  if repeat % 2:
                    order.reverse()
                  for implementation in order:
                    execute = executors[implementation]
                    for _ in range(args.warmup):
                        model(execute(resident))
                    torch.cuda.synchronize()
                    start_allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    for iteration in range(args.iterations):
                        started = time.perf_counter_ns()
                        events[0].record()
                        tensor = execute(resident)
                        events[1].record()
                        logits = model(tensor)
                        events[2].record()
                        events[2].synchronize()
                        host_ms = (time.perf_counter_ns()-started)/1e6
                        row = dict(case=name,implementation=implementation,repeat=repeat,iteration=iteration,
                                   preprocess_ms=events[0].elapsed_time(events[1]),
                                   inference_ms=events[1].elapsed_time(events[2]),
                                   stream_total_ms=events[0].elapsed_time(events[2]),host_total_ms=host_ms)
                        writer.writerow(row)
                        rows.append(row)
                        del tensor, logits
                    report['cases'].append({'case':name,'repeat':repeat,'implementation':implementation,
                        'allocated_before_bytes':start_allocated,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_above_start_bytes':torch.cuda.max_memory_allocated()-start_allocated,
                        'reserved_after_bytes':torch.cuda.memory_reserved()})
                report.setdefault('percentiles',{})[name] = {
                    implementation: {
                        metric:dict(zip(('p50','p95','p99'),np.percentile(
                            [r[metric] for r in rows if r['implementation'] == implementation],
                            [50,95,99],method='linear').tolist()))
                        for metric in metrics}
                    for implementation in implementations}
                del resident
        report['samples_sha256'] = digest(args.output/'samples.csv')
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'percentiles':report['percentiles']}))


if __name__ == '__main__':
    main()

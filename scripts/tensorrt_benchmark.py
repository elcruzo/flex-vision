"""Compare synchronous CPG and unfused PyTorch through the same TensorRT engine.

Run tensorrt_detector.py on this host first. This runner is a serial replay
experiment, not camera latency or a multi-camera throughput benchmark.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from cpg import load_pipeline
from cpg.reference import numpy_reference
from cpg.validation import photo_cases, check_expected_object
from local_detector import WEIGHTS_SHA256, digest
from tensorrt_detector import Consumer, DenseDetector


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Value must be positive')
    return number


def statistics(rows):
    return {key: dict(zip(('p50', 'p95', 'p99'), np.percentile(
        [row[key] for row in rows], [50, 95, 99], method='linear').tolist()))
        for key in ('preprocess_gpu_ms', 'network_gpu_ms', 'decode_gpu_ms', 'total_gpu_ms', 'total_host_ms')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validated-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=positive, default=1000, help='Samples per candidate per repeat')
    parser.add_argument('--repeats', type=positive, default=10)
    parser.add_argument('--warmup', type=positive, default=100)
    parser.add_argument('--trace', action='store_true', help='Diagnostic run only; timings are not benchmark evidence')
    parser.add_argument('--weights', type=Path, default=Path('.cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'))
    args = parser.parse_args()
    previous = json.loads((args.validated_run/'report.json').read_text())
    engine = args.validated_run/'ssdlite.engine'
    if previous['status'] != 'passed' or digest(engine) != previous['engine_sha256']:
        parser.error('Expected a passed detector run and its unchanged engine')
    if digest(args.weights) != WEIGHTS_SHA256:
        parser.error('Model weights do not match the pinned hash')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'scope': 'serial 1080p photographic replay to hybrid SSDLite detections',
              'instrumented': args.trace, 'engine_sha256': digest(engine),
              'validated_report_sha256': digest(args.validated_run/'report.json'),
              'runner_sha256': digest(Path(__file__)),
              'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'dirty': bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'samples_per_repeat': args.samples, 'repeats': args.repeats, 'warmup': args.warmup,
              'percentile_method': 'numpy linear', 'validation': [], 'runs': [],
              'missing_metrics': ['hardware bandwidth counters', 'peak temporary memory', 'CPU utilization',
                                  'Jetson power', 'live camera latency', 'multi-camera throughput'],
              'policy': 'Same explicit stream and engine; fresh per-call outputs; warmed allocator pools; baseline constants cached on device'}
    try:
        pipeline = load_pipeline('examples/local-detector.yaml')
        if digest(Path('examples/local-detector.yaml')) != previous['config_sha256']:
            raise ValueError('Configuration changed since the validated engine run')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
        model.cuda()
        consumer = Consumer(engine)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), cp.cuda.ExternalStream(stream.cuda_stream, device_id=0), torch.inference_mode():
            anchors = DenseDetector(model).anchors(torch.zeros((1,3,320,320),device='cuda'))
            # Known photographed object, enlarged by exact pixel repetition and placed in a 1080p frame.
            _, rgb, expectation, provenance = next(photo_cases(Path('tests/fixtures/images/manifest.json')))
            enlarged = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
            host_rgb = np.full((1080,1920,3),114,dtype=np.uint8)
            h,w = enlarged.shape[:2]
            top,left = (1080-h)//2,(1920-w)//2
            host_rgb[top:top+h,left:left+w] = enlarged
            expectation = dict(expectation)
            expectation['box_xyxy'] = (np.asarray(expectation['box_xyxy'])*2+[left,top,left,top]).tolist()
            host = np.ascontiguousarray(host_rgb[...,::-1])
            expected,g = numpy_reference(pipeline,host)
            report['input'] = {'shape':list(host.shape),'sha256':hashlib.sha256(host.tobytes()).hexdigest(),
                               'provenance':provenance,'transform':'repeat pixels 2x; centered 1080p padding value 114',
                               'expectation':expectation}
            frame = cp.asarray(host)
            tensor = torch.from_dlpack(frame)
            norm = pipeline.operations[1]
            mean = torch.tensor(norm.mean,device='cuda').view(1,3,1,1)
            std = torch.tensor(norm.std,device='cuda').view(1,3,1,1)

            def unfused():
                x = tensor.flip(-1).permute(2,0,1)[None].float()
                x = F.interpolate(x,size=(g.resized_height,g.resized_width),mode='bilinear',align_corners=False,antialias=False)
                x = F.pad(x,(g.left,g.output_width-g.resized_width-g.left,g.top,g.output_height-g.resized_height-g.top),value=pipeline.operations[0].value)
                x = ((x*norm.scale-mean)/std).contiguous()
                # Match the production CPG call's synchronous preprocessing boundary.
                stream.synchronize()
                return cp.from_dlpack(x)

            paths = {'torch_unfused': unfused, 'cpg': lambda: pipeline(frame)}
            def decode(head):
                return model.postprocess_detections(head,anchors,[(320,320)])[0]

            reference = model([torch.from_numpy(expected[0]).cuda()])[0]
            reference = {k:v.cpu() for k,v in reference.items()}
            for name,preprocess in paths.items():
                actual = preprocess()
                output = decode(consumer(actual))
                stream.synchronize()
                actual_cpu = cp.asnumpy(actual)
                np.testing.assert_allclose(actual_cpu,expected,atol=2e-4,rtol=0)
                output = {k:v.cpu() for k,v in output.items()}
                keep,other = output['scores']>=.5,reference['scores']>=.5
                for key,atol in [('labels',0),('scores',.001),('boxes',.1)]:
                    torch.testing.assert_close(output[key][keep],reference[key][other],atol=atol,rtol=0)
                iou = check_expected_object(output['labels'][keep].tolist(),output['scores'][keep].tolist(),
                      [g.source_box(b.tolist()) for b in output['boxes'][keep]],expectation)
                report['validation'].append({'candidate':name,'tensor_max_abs_error':float(np.abs(actual_cpu-expected).max()),'expected_object_iou':iou})
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            # Initialize event resources before collecting any sample.
            for event in events:
                event.record(stream)
            stream.synchronize()
            fields = ['repeat','candidate','sample','preprocess_gpu_ms','network_gpu_ms','decode_gpu_ms','total_gpu_ms','total_host_ms']
            if args.trace:
                torch.cuda.profiler.start()
            with (args.output/'samples.csv').open('x',newline='') as dest:
                writer = csv.DictWriter(dest,fieldnames=fields)
                writer.writeheader()
                for repeat in range(args.repeats):
                    order = list(paths) if repeat%2 == 0 else list(reversed(paths))
                    for name in order:
                        preprocess = paths[name]
                        for _ in range(args.warmup):
                            output = decode(consumer(preprocess()))
                            stream.synchronize()
                        rows = []
                        for sample in range(args.samples):
                            # Previous sample completed before the next start. No queue overlap.
                            started = time.perf_counter_ns()
                            with torch.cuda.nvtx.range(name+'_hybrid_pipeline'):
                                events[0].record(stream)
                                image = preprocess()
                                events[1].record(stream)
                                head = consumer(image)
                                events[2].record(stream)
                                output = decode(head)
                                events[3].record(stream)
                                stream.synchronize()
                            host_ms = (time.perf_counter_ns()-started)/1e6
                            rows.append(dict(zip(fields,[repeat,name,sample,events[0].elapsed_time(events[1]),
                                        events[1].elapsed_time(events[2]),events[2].elapsed_time(events[3]),
                                        events[0].elapsed_time(events[3]),host_ms])))
                        writer.writerows(rows)
                        dest.flush()
                        report['runs'].append({'repeat':repeat,'candidate':name,'count':len(rows),'milliseconds':statistics(rows)})
            if args.trace:
                torch.cuda.profiler.stop()
        report['samples_sha256'] = digest(args.output/'samples.csv')
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'runs':len(report['runs'])}))


if __name__ == '__main__':
    main()

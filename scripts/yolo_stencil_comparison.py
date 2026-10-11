"""Matched resident stencil pipelines through one fixed FP16 TensorRT engine.

This prepared experiment has no measured result until it passes on NVIDIA hardware.
"""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True, help='Passed 192-case stencil report from this source and engine')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    if args.blocks < 2 or args.blocks % 2 or args.samples < 100:
        parser.error('Require an even number of blocks, at least two, and 100 samples')
    from verify_yolo_stencil import verify
    acceptance_bytes = args.acceptance.read_bytes()
    accepted = verify(json.loads(acceptance_bytes))
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if accepted['revision'] != revision:
        raise ValueError('Stencil acceptance must use the measured source revision')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'acceptance.json').write_bytes(acceptance_bytes)
    import cupy as cp
    import numpy as np
    import torch
    import torch.nn.functional as F
    from cpg.pipeline import Conv2d
    from cpg.reference import numpy_reference
    from yolo_common import pipeline, cases, digest, decode, compare_dense, compare_detections, LIMITS
    from yolo_gpu import Consumer
    report = dict(status='failed', mode='stencil-comparison-v1',
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                  engine_sha256=digest(args.engine), acceptance_sha256=digest(args.acceptance), blocks=args.blocks, samples=args.samples,
                  warmup=100, limits=LIMITS, checks=[],
                  validation_policy='Every timed inference checked after its interval; both paths include host synchronization. Following-memory-state effects remain possible.',
                  versions=dict(cupy=cp.__version__,torch=torch.__version__),
                  gpu=torch.cuda.get_device_name(0),
                  scope='Resident 1080p, positive asymmetric stencils, owned synchronous outputs, one fixed TensorRT engine.',
                  unmeasured=['transfer trace','launch counts','power','memory bandwidth','peak workspace','multi-camera throughput'])
    stream = torch.cuda.Stream()
    try:
        if report['dirty']:
            raise ValueError('Require clean committed source')
        if accepted['engine_sha256'] != report['engine_sha256']:
            raise ValueError('Stencil acceptance must use the measured TensorRT engine')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        consumer = Consumer(args.engine)
        base = pipeline()
        with torch.inference_mode(), torch.cuda.stream(stream), cp.cuda.ExternalStream(stream.cuda_stream, device_id=0), \
                (args.output/'timings.csv').open('w',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['fixture','size','block','sample','path','preprocess_ms','inference_ms','decode_ms','gpu_total_ms','host_ms'])
            for name,rgb,_,_ in cases():
                if rgb.shape != (1080,1920,3):
                    continue
                host = np.ascontiguousarray(rgb[...,::-1])
                frame = cp.asarray(host)
                for size in (3,5,7,9):
                    report['active_scenario'] = dict(fixture=name,size=size)
                    coefficients = [[0.]*size for _ in range(size)]
                    coefficients[0][0] = coefficients[-1][0] = .125
                    coefficients[size//2][size//2] = .75
                    pipe = replace(base, operations=(Conv2d(tuple(tuple(r) for r in coefficients)),)+base.operations)
                    expected,g = numpy_reference(pipe,host)
                    # Upload constants once. Both timed paths receive the same resident source.
                    weights = torch.tensor(pipe.operations[0].kernel,device='cuda',dtype=torch.float32).flip((0,1))
                    weights = weights.view(1,1,size,size).repeat(3,1,1,1).contiguous()
                    def baseline():
                        image = torch.from_dlpack(frame).flip(-1).permute(2,0,1)[None].float()
                        image = F.conv2d(F.pad(image,(size//2,)*4,mode='replicate'),weights,groups=3)
                        image = F.interpolate(image,size=(g.resized_height,g.resized_width),mode='bilinear',align_corners=False,antialias=False)
                        image = F.pad(image,(g.left,640-g.resized_width-g.left,g.top,640-g.resized_height-g.top),value=114)
                        image = (image*(1/255)).half().contiguous()
                        stream.synchronize()
                        return cp.from_dlpack(image)
                    runners = {'cpg':lambda:pipe(frame), 'pytorch':baseline}
                    oracle = cp.asarray(expected)
                    stream.synchronize()
                    reference = consumer(oracle)
                    reference_detection = decode(reference)
                    stream.synchronize()
                    for path,run in runners.items():
                        tensor = run()
                        actual = cp.asnumpy(tensor)
                        if path == 'cpg':
                            np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
                        else:
                            np.testing.assert_allclose(actual,expected,atol=LIMITS['tensor_atol'],rtol=0)
                        head = consumer(tensor)
                        compare_dense(reference,head,preprocessing=True,strict_output=path=='cpg')
                        compare_detections(reference_detection,decode(head))
                        stream.synchronize()
                        report['checks'].append(dict(fixture=name,size=size,path=path,tensor_max_error=float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32)))),inference='passed'))
                        for _ in range(report['warmup']):
                            warm = run(); decoded = decode(consumer(warm)); stream.synchronize()
                        del tensor,head,warm,decoded
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    for block in range(args.blocks):
                        order = ('cpg','pytorch') if block%2 == 0 else ('pytorch','cpg')
                        for path in order:
                            for sample in range(args.samples):
                                start = time.perf_counter_ns()
                                events[0].record()
                                tensor = runners[path]()
                                events[1].record()
                                head = consumer(tensor)
                                events[2].record()
                                detections = decode(head)
                                events[3].record()
                                stream.synchronize()
                                host_ms = (time.perf_counter_ns()-start)/1e6
                                writer.writerow([name,size,block,sample,path,
                                    *[events[i].elapsed_time(events[i+1]) for i in range(3)],
                                    events[0].elapsed_time(events[3]),host_ms])
                                # Validate every completed inference outside the timed interval.
                                compare_dense(reference,head,preprocessing=True,strict_output=path=='cpg')
                                compare_detections(reference_detection,detections)
                                stream.synchronize()
                                del tensor,head,detections
                        file.flush()
                        print('stencil_timed',name,size,block,flush=True)
                    del oracle,reference,reference_detection,weights,events
            if len(report['checks']) != 16:
                raise ValueError('Require two 1080p fixtures, four sizes, and two validated paths')
            report['active_scenario'] = None
            report['status'] = 'passed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        try:
            stream.synchronize()
            report['drain_status'] = 'passed'
        except Exception as exc:
            report.update(status='failed',drain_status='failed',drain_error=repr(exc))
            raise
        finally:
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

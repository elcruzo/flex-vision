"""Matched serial synchronous/submission inference. No overlap claim."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


def require_gate(report, revision, engine_sha256, dirty):
    from verify_yolo_delayed_consumer import verify
    gate = verify(report)
    if dirty or gate['revision'] != revision or gate['engine_sha256'] != engine_sha256:
        raise ValueError('Require clean matching delayed-consumer source and engine')
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--delayed-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from yolo_common import digest
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    gate = require_gate(json.loads(args.delayed_report.read_text()), revision, digest(args.engine), dirty)
    args.output.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    import numpy as np
    import torch
    from cpg.reference import numpy_reference
    from yolo_common import pipeline, cases, decode
    from pending_yolo_consumer import PendingConsumer
    from yolo_sustained import valid
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    pre, downstream = streams
    consumer = PendingConsumer(args.engine, downstream)
    pipe = pipeline()
    report = {'status': 'failed', 'revision': revision, 'dirty': False,
              'engine_sha256': gate['engine_sha256'], 'delayed_report_sha256': digest(args.delayed_report),
              'mode': 'serial-async-v1', 'blocks': 10, 'samples_per_block': 1000, 'warmup': 100,
              'scope': 'Resident 1080p input; fresh outputs; serial FP16 TensorRT and CUDA NMS; no camera acquisition or overlap',
              'fixtures': [], 'p99_guard_fraction': 0.05}
    pending = job = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), (args.output / 'samples.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['fixture', 'block', 'sample', 'candidate', 'dispatch_host_ms',
                             'preprocess_gpu_ms', 'complete_gpu_ms', 'complete_host_ms', 'correct'])
            events = [cp.cuda.Event() for _ in range(3)]
            for name, rgb, _, _ in cases():
                if not name.endswith('1080p'):
                    continue
                host = np.ascontiguousarray(rgb[..., ::-1])
                expected_host, _ = numpy_reference(pipe, host)
                with np.load(args.controls / (name + '-controls.npz')) as controls:
                    np.testing.assert_array_equal(expected_host, controls['tensor'])
                    with downstream, torch.cuda.stream(consumer._torch_stream):
                        expected = torch.from_numpy(expected_host).cuda()
                        reference = torch.from_numpy(controls['tensorrt_dense']).cuda()
                        reference_detection = decode(reference)
                        warm = consumer(cp.from_dlpack(expected))
                        torch.testing.assert_close(warm, reference, atol=0, rtol=0)
                downstream.synchronize()
                with pre:
                    source = cp.asarray(host)
                pre.synchronize()
                record = {'name': name, 'checked_frames': 0, 'failed_frames': 0}
                report['fixtures'].append(record)
                # Warm the full checked path before collecting either candidate.
                for block in range(-1, 10):
                    order = ('sync', 'submit') if block % 2 == 0 else ('submit', 'sync')
                    for candidate in order:
                        for index in range(100 if block == -1 else 1000):
                            start = time.perf_counter_ns()
                            events[0].record(pre)
                            with pre:
                                if candidate == 'sync':
                                    image = pipe(source)
                                else:
                                    pending = pipe.submit(source, stream=pre)
                                    image = pending.wait_on(downstream)
                                events[1].record(pre)
                            dispatch_ms = (time.perf_counter_ns() - start) / 1e6
                            # Order the timing marker before both consumer paths.
                            # submit also exercises its public readiness handoff.
                            downstream.wait_event(events[1])
                            job = consumer.enqueue(image)
                            head = job.wait() if candidate == 'sync' else job._output
                            with downstream, torch.cuda.stream(consumer._torch_stream):
                                decoded = decode(head)
                                events[2].record(downstream)
                            downstream.synchronize()
                            job.wait()
                            job = None
                            if pending is not None:
                                pending.close()
                                pending = None
                            host_ms = (time.perf_counter_ns() - start) / 1e6
                            # Validation follows the complete timing boundary on both paths.
                            with downstream, torch.cuda.stream(consumer._torch_stream):
                                correct = bool(valid(image, head, decoded, expected, reference, reference_detection, 'cpg').item())
                            if block >= 0:
                                writer.writerow([name, block, index, candidate, dispatch_ms,
                                    cp.cuda.get_elapsed_time(events[0], events[1]),
                                    cp.cuda.get_elapsed_time(events[0], events[2]), host_ms, correct])
                                record['checked_frames'] += 1
                                record['failed_frames'] += not correct
                            if not correct:
                                raise AssertionError('Complete inference correctness failed')
                            del image, head, decoded
                    handle.flush()
                    print('async_comparison', name, block, flush=True)
        if len(report['fixtures']) != 2:
            raise ValueError('Require both fixed 1080p fixtures')
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        try:
            if pending is not None: pending.close()
            if job is not None: job.close()
            for stream in streams: stream.synchronize()
            report['drain_status'] = 'passed'
        except Exception as exc:
            report['status'] = 'failed'
            report['drain_status'] = 'failed'
            report['drain_error'] = repr(exc)
            raise
        finally:
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

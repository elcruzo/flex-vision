"""Focused CAI-only complete-inference capture. No latency selection."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--cai-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Output already exists')
    from yolo_common import digest, pipeline, cases, decode, compare_dense, compare_detections
    from verify_yolo_cai import verify
    gate = verify(json.loads(args.cai_report.read_text()))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    if dirty or revision != gate['revision'] or digest(args.engine) != gate['engine_sha256']:
        raise ValueError('Require clean matching CAI correctness source and engine')
    import cupy as cp
    import numpy as np
    import torch
    from cpg.reference import numpy_reference
    from yolo_cai import Exporter
    from pending_yolo_consumer import PendingConsumer
    producer, pre, downstream = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    streams = [producer, pre, downstream, cp.cuda.Stream.null, cp.cuda.Stream.ptds]
    consumer, pipe = PendingConsumer(args.engine, downstream), pipeline()
    report = dict(status='failed', mode='cai-trace-v1', dirty=False, revision=revision,
                  engine_sha256=gate['engine_sha256'], cai_report_sha256=digest(args.cai_report), checks=[],
                  scope='Resident contiguous 1080p CAI-only input through completed TensorRT and CUDA NMS; no performance claim')
    pending = job = None
    profiling = False

    def execute(frame, execution):
        nonlocal pending, job
        with pre:
            if execution == 'sync':
                image = pipe(frame)
                downstream.wait_event(pre.record())
            else:
                pending = pipe.submit(frame, stream=pre)
                image = pending.wait_on(downstream)
        job = consumer.enqueue(image)
        head = job._output
        with downstream, torch.cuda.stream(consumer._torch_stream):
            detections = decode(head)
        downstream.synchronize()
        job.wait()
        job = None
        if pending is not None:
            pending.close()
            pending = None
        return image, head, detections

    try:
        with torch.inference_mode():
            prepared = []
            for name, rgb, _, _ in cases():
                if not name.endswith('1080p'): continue
                host = np.ascontiguousarray(rgb[..., ::-1])
                expected, _ = numpy_reference(pipe, host)
                with np.load(args.controls / (name + '-controls.npz')) as controls:
                    np.testing.assert_array_equal(expected, controls['tensor'])
                    with downstream, torch.cuda.stream(consumer._torch_stream):
                        reference = torch.from_numpy(controls['tensorrt_dense']).cuda()
                        expected_detections = decode(reference)
                downstream.synchronize()
                for label, stream, handle in [('explicit', producer, producer.ptr),
                        ('legacy', cp.cuda.Stream.null, 1), ('ptds', cp.cuda.Stream.ptds, 2), ('ready', producer, None)]:
                    with stream: source = cp.asarray(host)
                    stream.synchronize()
                    frame = Exporter(source, stream, handle)
                    # Compile and warm each complete execution path outside measured ranges.
                    for execution in ('sync', 'submit'):
                        image, head, detections = execute(frame, execution)
                        np.testing.assert_array_equal(cp.asnumpy(image).view(np.uint16), expected.view(np.uint16))
                        with torch.cuda.stream(consumer._torch_stream): compare_dense(reference, head, strict_output=True)
                        del image, head, detections
                    prepared.append((name, label, frame, expected, reference, expected_detections))
            torch.cuda.profiler.start()
            profiling = True
            control = cp.arange(1024, dtype=cp.float32)
            with torch.cuda.nvtx.range('known_4096_byte_d2h_control'): cp.asnumpy(control)
            del control
            for name, label, frame, expected, reference, expected_detections in prepared:
                for execution in ('sync', 'submit'):
                    report['active_scenario'] = dict(fixture=name, producer=label, execution=execution)
                    with torch.cuda.nvtx.range('yolo_cai_' + label + '_' + execution + '_complete'):
                        image, head, detections = execute(frame, execution)
                    # Diagnostic downloads and comparisons are outside the completed capture range.
                    np.testing.assert_array_equal(cp.asnumpy(image).view(np.uint16), expected.view(np.uint16))
                    with torch.cuda.stream(consumer._torch_stream):
                        compare_dense(reference, head, strict_output=True)
                        compare_detections(expected_detections, detections)
                    downstream.synchronize()
                    report['checks'].append(dict(fixture=name, producer=label, execution=execution,
                                                tensor_exact=True, dense_output=True, detections=True))
                    del image, head, detections
            report['active_scenario'] = None
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
            try:
                if profiling: torch.cuda.profiler.stop()
            finally:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()

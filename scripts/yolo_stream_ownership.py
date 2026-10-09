"""Validate explicit stream handoffs through the synchronous detector baseline."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output report already exists')
    import cupy as cp
    import numpy as np
    import torch
    from cpg.reference import numpy_reference
    from yolo_common import pipeline, cases, decode, compare_dense, compare_detections, digest
    from yolo_gpu import Consumer
    report = {'status': 'failed', 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
              'engine_sha256': digest(args.engine), 'cases': [],
              'scope': 'Explicit three-stream handoffs with synchronous CPG and TensorRT; not asynchronous runtime acceptance'}
    # Keep stream, event, source, and destination owners alive through completion.
    streams = [torch.cuda.Stream() for _ in range(3)]
    producer, preprocess, consumer_stream = streams
    try:
        if report['dirty']:
            raise ValueError('Commit measured source first')
        pipe = pipeline()
        with torch.cuda.stream(consumer_stream):
            consumer = Consumer(args.engine)
        for name, rgb, _, _ in cases():
            host = np.ascontiguousarray(rgb[..., ::-1])
            expected, _ = numpy_reference(pipe, host)
            with np.load(args.controls / (name + '-controls.npz')) as controls:
                np.testing.assert_array_equal(expected, controls['tensor'])
                with torch.cuda.stream(consumer_stream):
                    reference = torch.from_numpy(controls['tensorrt_dense']).cuda()
            for input_kind in ('cupy', 'torch'):
                retained = []
                for cycle in range(4):
                    ready = torch.cuda.Event()
                    done = torch.cuda.Event()
                    with torch.cuda.stream(producer), cp.cuda.ExternalStream(producer.cuda_stream):
                        # Build the producer payload on its stream. No host wait here.
                        source = cp.asarray(host)
                        source = source.copy()
                        frame = source if input_kind == 'cupy' else torch.from_dlpack(source)
                        ready.record(producer)
                    with torch.cuda.stream(preprocess), cp.cuda.ExternalStream(preprocess.cuda_stream):
                        preprocess.wait_event(ready)
                        result = pipe(frame)
                    with torch.cuda.stream(consumer_stream), cp.cuda.ExternalStream(consumer_stream.cuda_stream):
                        # CPG is synchronous, so its return establishes readiness.
                        head = consumer(result)
                        compare_dense(reference, head, strict_output=True)
                        compare_detections(decode(reference), decode(head))
                        done.record(consumer_stream)
                    with torch.cuda.stream(preprocess), cp.cuda.ExternalStream(preprocess.cuda_stream):
                        preprocess.wait_event(done)
                        # Check a retained default output after later calls finish.
                        snapshot = result.copy()
                        preprocess.synchronize()
                        for old, saved in retained:
                            np.testing.assert_array_equal(cp.asnumpy(old), cp.asnumpy(saved))
                        np.testing.assert_array_equal(cp.asnumpy(result).view(np.uint16), expected.view(np.uint16))
                        retained.append((result, snapshot))
                    # References are intentionally retained through the consumer fence.
                    done.synchronize()
                report['cases'].append({'name': name, 'input': input_kind, 'cycles': 4,
                                        'retention': 'passed', 'status': 'passed'})
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        # Drain outstanding reads before Python releases stream and buffer owners.
        try:
            for stream in streams:
                stream.synchronize()
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

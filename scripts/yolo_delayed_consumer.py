"""Check preprocessing close while a delayed TensorRT consumer still owns input."""
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
    if args.output.exists(): raise ValueError('Output report already exists')
    import cupy as cp
    import numpy as np
    import torch
    from cpg.reference import numpy_reference
    from yolo_common import pipeline, cases, digest, decode, compare_dense, compare_detections
    from pending_yolo_consumer import PendingConsumer
    report = {'status': 'failed', 'revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'dirty': bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
              'engine_sha256': digest(args.engine), 'cases': [], 'delay_cycles': 500000000,
              'scope': 'Owned-output pending-consumer correctness; artificial delay; no performance claim'}
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    preprocess, consumer_stream = streams
    delay = cp.RawKernel(r'''
extern "C" __global__ void delay(unsigned long long cycles) {
    unsigned long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cycles) { }
}
''', 'delay')
    job = None
    pending = None
    try:
        if report['dirty']: raise ValueError('Commit measured source first')
        pipe = pipeline()
        consumer = PendingConsumer(args.engine, consumer_stream)
        # Compile and exercise the delay outside the ownership scenario.
        with consumer_stream: delay((1,), (1,), (np.uint64(1),))
        consumer_stream.synchronize()
        for name, rgb, _, _ in cases():
            host = np.ascontiguousarray(rgb[...,::-1])
            expected, _ = numpy_reference(pipe, host)
            with np.load(args.controls / (name + '-controls.npz')) as controls:
                np.testing.assert_array_equal(expected, controls['tensor'])
                with torch.cuda.stream(consumer._torch_stream):
                    reference = torch.from_numpy(controls['tensorrt_dense']).cuda()
                consumer_stream.synchronize()
            # Warm the execution context before introducing an artificial delay.
            with torch.cuda.stream(consumer._torch_stream), consumer_stream:
                expected_input = torch.from_numpy(expected).cuda()
                warm = consumer(cp.from_dlpack(expected_input))
                torch.testing.assert_close(warm, reference, atol=0, rtol=0)
            checks = []
            for kind in ('cupy','torch'):
                for cycle in range(4):
                    with preprocess:
                        source = cp.asarray(host)
                        frame = source if kind == 'cupy' else torch.from_dlpack(source)
                        pending = pipe.submit(frame, stream=preprocess)
                    del frame, source
                    image = pending.wait_on(consumer_stream)
                    with consumer_stream:
                        delay((1,), (1,), (np.uint64(report['delay_cycles']),))
                    job = consumer.enqueue(image)
                    pending.close()
                    del pending, image
                    pending = None
                    # A completed reader would not test the intended boundary.
                    if job._event.query():
                        raise AssertionError('Consumer completed before lifetime observation')
                    try: consumer.enqueue(job._image)
                    except RuntimeError: pass
                    else: raise AssertionError('Pending context was rebound')
                    # Start another preprocessing allocation while the first reader is pending.
                    with preprocess:
                        other_source = cp.asarray(host)
                        other = pipe(other_source)
                    with preprocess:
                        np.testing.assert_array_equal(cp.asnumpy(job._image).view(np.uint16), expected.view(np.uint16))
                    head = job.wait()
                    with torch.cuda.stream(consumer._torch_stream), consumer_stream:
                        compare_dense(reference, head, strict_output=True)
                        compare_detections(decode(reference), decode(head))
                        np.testing.assert_array_equal(cp.asnumpy(other).view(np.uint16), expected.view(np.uint16))
                    job = None
                    checks.append({'input': kind, 'cycle': cycle, 'pending_after_close': True,
                                   'rebind_rejected': True, 'dense_output': True, 'detections': True,
                                   'input_tensor_exact': True, 'subsequent_tensor_exact': True})
            report['cases'].append({'name': name, 'checks': checks, 'status': 'passed'})
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
            report['status'] = 'failed'; report['drain_status'] = 'failed'; report['drain_error'] = repr(exc)
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__': main()

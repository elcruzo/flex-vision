"""Execute CUDA numerical, input-stride, DLPack, and retained-output checks."""
import argparse
import json
from pathlib import Path
import cupy as cp
import numpy as np
import torch
from cpg import Pipeline
from cpg.reference import numpy_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as dest:
        report = {'status': 'failed', 'cases': []}
        try:
            rng = np.random.default_rng(27)
            for shape in [(1, 1, 3), (1, 17, 3), (17, 1, 3), (33, 17, 3), (481, 639, 3), (1080, 1920, 3)]:
                host = rng.integers(0, 256, shape, dtype=np.uint8)
                device = cp.asarray(host)
                for dtype in ('float16', 'float32'):
                    for encoding in ('rgb8', 'bgr8'):
                        pipe = Pipeline(input_encoding=encoding).letterbox(63, 65).normalize([.485,.456,.406], [.229,.224,.225], scale=1/255).to(dtype=dtype, layout='NCHW')
                        for slicing in (False, True):
                            source = device[::-1, ::2, ::-1] if slicing else device
                            reference = host[::-1, ::2, ::-1] if slicing else host
                            expected, _ = numpy_reference(pipe, reference)
                            actual = pipe(source)
                            np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=0.002 if dtype == 'float16' else 2e-6, rtol=0)
                            retained = cp.asnumpy(actual)
                            second = pipe(source)
                            assert actual.data.ptr != second.data.ptr
                            np.testing.assert_array_equal(cp.asnumpy(actual), retained)
                            report['cases'].append({'shape': list(reference.shape), 'dtype': dtype, 'encoding': encoding, 'strided': slicing, 'status': 'passed'})
            pipe = Pipeline().letterbox(31, 29).normalize([0,0,0],[1,1,1],scale=1/255).to(dtype='float32',layout='NCHW')
            # Producer and consumer use distinct streams; object DLPack orders the handoff.
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                source = torch.full((19,23,3), 173, device='cuda', dtype=torch.uint8)
                result = pipe(source)
            expected, _ = numpy_reference(pipe, np.full((19,23,3),173,np.uint8))
            np.testing.assert_allclose(cp.asnumpy(result),expected,atol=2e-6,rtol=0)
            report['torch_dlpack'] = 'passed'
            try:
                pipe(cp.zeros((2,2,3),dtype=cp.float32))
            except TypeError:
                report['invalid_dtype'] = 'rejected'
            else:
                raise AssertionError('float input was accepted')
            report['status'] = 'passed'
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            json.dump(report,dest,indent=2)
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))


if __name__ == '__main__':
    main()

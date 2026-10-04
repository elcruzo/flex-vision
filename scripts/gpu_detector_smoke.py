"""Validate a CUDA development host with CuPy interoperability and real inference.

Select the unfused reference or the experimental CPG CUDA backend.
Fixture uploads and validation downloads are deliberate. No residency claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from cpg import load_pipeline
from cpg.reference import numpy_reference
from cpg.validation import photo_cases, check_expected_object
from local_detector import WEIGHTS_SHA256, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('reference', 'cpg'), default='reference')
    parser.add_argument('--weights', type=Path, default=Path('.cache/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'))
    args = parser.parse_args()
    if digest(args.weights) != WEIGHTS_SHA256:
        parser.error('The model weights do not match the pinned SHA256')
    if not torch.cuda.is_available():
        parser.error('A CUDA device is required')
    args.output.mkdir(parents=True, exist_ok=False)
    config = Path('examples/local-detector.yaml')
    pipeline = load_pipeline(config)
    norm = pipeline.operations[1]
    if (pipeline.input_encoding != 'bgr8' or pipeline.plan((1080, 1920, 3))['output_shape'] != [1, 3, 320, 320]
            or pipeline.operations[2].dtype != 'float32' or norm.scale != 1/255
            or norm.mean != (0., 0., 0.) or norm.std != (1., 1., 1.)):
        parser.error('This smoke fixture requires 320x320 RGB FP32 NCHW with scale 1/255, mean 0, std 1')
    report = {'status': 'failed', 'scope': 'CUDA detector smoke; not complete G1, TensorRT, or performance acceptance',
              'backend': args.backend,
              'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cupy': cp.__version__,
              'numpy': np.__version__, 'cuda_build': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'weights_sha256': WEIGHTS_SHA256,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
              'source_files': {str(p): digest(p) for p in sorted(Path('src/cpg').glob('*.py'))},
              'runner_sha256': digest(Path(__file__)), 'config_sha256': digest(config),
              'fixture_manifest_sha256': digest(Path('tests/fixtures/images/manifest.json')),
              'tensor_atol': 2e-4, 'dense_atol': 0.01, 'dense_rtol': 1e-4, 'cases': []}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
        model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
        model.cuda()
        captures = []
        hook = model.head.register_forward_hook(lambda _m, _i, out: captures.append({k: v.detach().clone() for k, v in out.items()}))
        stream = torch.cuda.Stream()
        # Initialize model weights on the default stream before using the test stream.
        stream.wait_stream(torch.cuda.current_stream())
        # Torch 2.9.1 lacks __cuda_stream__; retain this compatibility API.
        # Keep the owning Torch stream alive until all submitted work completes.
        with torch.cuda.stream(stream), cp.cuda.ExternalStream(stream.cuda_stream, device_id=0), torch.inference_mode():
            values = torch.arange(1024, device='cuda', dtype=torch.float32)
            shared = cp.from_dlpack(values)
            if shared.data.ptr != values.data_ptr():
                raise AssertionError('Torch to CuPy DLPack did not share the allocation')
            kernel = cp.RawKernel('extern "C" __global__ void twice(float* p) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<1024) p[i]*=2; }', 'twice')
            kernel((4,), (256,), (shared,))
            returned = torch.from_dlpack(shared)
            if returned.data_ptr() != values.data_ptr():
                raise AssertionError('CuPy to Torch DLPack did not share the allocation')
            torch.testing.assert_close(returned, torch.arange(1024, device='cuda', dtype=torch.float32)*2, rtol=0, atol=0)
            report['interop'] = 'same-stream DLPack pointer identity and NVRTC kernel passed'
            for case_id, rgb, expectation, provenance in photo_cases(Path('tests/fixtures/images/manifest.json')):
                bgr = np.ascontiguousarray(rgb[..., ::-1])
                expected, g = numpy_reference(pipeline, bgr)
                case = {'id': case_id, 'provenance': provenance, 'input_sha256': hashlib.sha256(bgr.tobytes()).hexdigest()}
                report['cases'].append(case)
                # Explicit fixture upload. All following image operations execute on CUDA.
                frame = cp.asarray(bgr)
                tensor = torch.from_dlpack(frame)
                if tensor.data_ptr() != frame.data.ptr:
                    raise AssertionError('Input interoperability did not share the allocation')
                with torch.cuda.nvtx.range('unfused_reference_preprocess' if args.backend == 'reference' else 'cpg_preprocess'):
                    if args.backend == 'cpg':
                        processed = pipeline(frame)
                        actual = torch.from_dlpack(processed)
                        if actual.data_ptr() != processed.data.ptr:
                            raise AssertionError('Output interoperability did not share the allocation')
                    else:
                        actual = reference_preprocess(tensor, pipeline, g)
                captures.clear()
                with torch.cuda.nvtx.range('candidate_inference'):
                    candidate = model([actual[0]])[0]
                baseline = model([torch.from_numpy(expected[0]).cuda()])[0]
                # Downloads occur after inference, for evidence only.
                actual_cpu = actual.cpu().numpy()
                np.testing.assert_allclose(actual_cpu, expected, atol=2e-4, rtol=0)
                case['tensor_max_abs_error'] = float(np.abs(actual_cpu-expected).max())
                case['dense_max_abs_error'] = {}
                for key in captures[0]:
                    torch.testing.assert_close(captures[0][key], captures[1][key], atol=0.01, rtol=1e-4)
                    case['dense_max_abs_error'][key] = float((captures[0][key]-captures[1][key]).abs().max().item())
                candidate = {k: v.cpu() for k, v in candidate.items()}
                baseline = {k: v.cpu() for k, v in baseline.items()}
                keep = candidate['scores'] >= .5
                expected_keep = baseline['scores'] >= .5
                for key, atol in [('labels', 0), ('scores', .001), ('boxes', .1)]:
                    torch.testing.assert_close(candidate[key][keep], baseline[key][expected_keep], atol=atol, rtol=0)
                boxes = [g.source_box(b.tolist()) for b in candidate['boxes'][keep]]
                case['expected_object_iou'] = check_expected_object(candidate['labels'][keep].tolist(), candidate['scores'][keep].tolist(), boxes, expectation)
                np.savez_compressed(args.output/f'{case_id}.npz', actual=actual_cpu, expected=expected,
                                    **{'candidate_'+k: v.numpy() for k, v in candidate.items()},
                                    **{'baseline_'+k: v.numpy() for k, v in baseline.items()})
                case['status'] = 'passed'
            stream.synchronize()
        hook.remove()
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases'])}))


def reference_preprocess(tensor, pipeline, g):
    actual = tensor.flip(-1).permute(2, 0, 1)[None].float()
    actual = F.interpolate(actual, size=(g.resized_height, g.resized_width), mode='bilinear', align_corners=False, antialias=False)
    actual = F.pad(actual, (g.left, g.output_width-g.resized_width-g.left, g.top, g.output_height-g.resized_height-g.top), value=pipeline.operations[0].value)
    norm = pipeline.operations[1]
    mean = torch.tensor(norm.mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device='cuda').view(1, 3, 1, 1)
    return ((actual*norm.scale-mean)/std).contiguous()


if __name__ == '__main__':
    main()

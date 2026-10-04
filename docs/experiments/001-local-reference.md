# Iteration 001 — local detector reference

Code revision: `693157e07a21fbbd9316e2dddba0be77c69ff4f3`.
Scope: synthetic local reference validation, not CUDA/TensorRT or a performance claim.

The isolated Mac environment passed `pip check` and 20 focused tests.
Both preprocessing references fed the same pretrained SSDLite CPU detector.
The MPS path includes an explicit download for comparison and inference.

| Preprocessing | Input H×W | Maximum tensor error | Maximum dense head error | Result |
| --- | --- | --- | --- | --- |
| cpu | 1080×1920 | 0 | 0 | passed |
| cpu | 481×639 | 2.4497509e-05 | 4.1484833e-05 | passed |
| cpu | 33×17 | 1.1920929e-06 | 2.5272369e-05 | passed |
| mps | 1080×1920 | 0 | 0 | passed |
| mps | 481×639 | 2.4497509e-05 | 4.1007996e-05 | passed |
| mps | 33×17 | 1.1445954e-06 | 2.3603439e-05 | passed |

Tensor tolerance was 0.0002 absolute. Dense output tolerance was 0.01 absolute plus 0.0001 relative.
Postprocessed labels, scores, and boxes also passed the declared checks.
Synthetic detections do not establish model accuracy.

Full local bundles are in benchmark-results/iteration-001-isolated-cpu and benchmark-results/iteration-001-isolated-mps.
They contain manifests, input hashes, plans, numerical results, and tensor/detection archives.
Large local artifacts remain outside Git. The runner reproduces them from the pinned weights and synthetic fixture generator.
See [local development](../LOCAL_DEVELOPMENT.md) for commands and model provenance.

Next: strict YAML equivalence through the same inference runner, then representative image fixtures and the first NVIDIA experiment.

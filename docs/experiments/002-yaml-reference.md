# Iteration 002 — YAML through real inference

Code revision: `d95699efc7fae659f620d67b224c5a73673de564`.
Configuration: examples/local-detector.yaml.
Configuration SHA256: `ad6c1314b69f2546183349d463b88406ba4928221e861c06eced930f2b9cb424`.

The isolated environment passed its dependency check and 34 focused tests.
The configuration loader rejects invalid fields, duplicate keys, unsupported operations, and unsafe YAML tags.
Python and YAML graph descriptions produce equivalent pixels.
The inspector emits a plan without executing preprocessing.

| Preprocessing | Input H×W | Maximum tensor error | Full consumer check |
| --- | --- | --- | --- |
| cpu | 1080×1920 | 0 | passed |
| cpu | 481×639 | 2.4497509e-05 | passed |
| cpu | 33×17 | 1.1920929e-06 | passed |
| mps | 1080×1920 | 0 | passed |
| mps | 481×639 | 2.4497509e-05 | passed |
| mps | 33×17 | 1.1445954e-06 | passed |

Every run compared dense model outputs and postprocessed detections against the independent NumPy reference input.
The model is the same pretrained SSDLite CPU consumer used in iteration 001.
MPS preprocessing includes a deliberate download for CPU inference and validation.
This is synthetic local integration evidence, not CUDA, TensorRT, accuracy, or speed evidence.

Full local bundles are in benchmark-results/iteration-002-committed-cpu and benchmark-results/iteration-002-committed-mps.
The manifest stores exact configuration text, input/model/source hashes, versions, and declared tolerances.
See [local development](../LOCAL_DEVELOPMENT.md) to reproduce the runs.

Next: a representative image fixture with clear provenance and expected outputs, then NVIDIA CUDA/TensorRT execution.

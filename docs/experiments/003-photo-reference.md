# Iteration 003 — photographs and semantic checks

Code revision: `95ade40650cc9d85275dc7c1d50291188ab5ffd8`.
The isolated Mac environment passed 43 focused tests.
Both CPU and MPS preprocessing passed seven full detector cases each.
Inference remained on the CPU with the pinned pretrained SSDLite model.

| Preprocessing | Photo case | Maximum tensor error | Expected object source-box IoU |
| --- | --- | --- | --- |
| cpu | astronaut | 1.5318394e-05 | 0.930434 |
| cpu | astronaut-padded | 1.3589859e-05 | 0.863174 |
| cpu | chelsea | 7.0631504e-06 | 0.944010 |
| cpu | chelsea-padded | 4.6491623e-06 | 0.916463 |
| mps | astronaut | 1.5318394e-05 | 0.930434 |
| mps | astronaut-padded | 1.3589859e-05 | 0.863174 |
| mps | chelsea | 7.0631504e-06 | 0.944010 |
| mps | chelsea-padded | 4.6491623e-06 | 0.916463 |

Each photo requires the expected person/cat label above score 0.5 and source-space IoU of at least 0.5.
Broad hand-reviewed boxes define the smoke expectations before candidate execution.
All cases also passed tensor, dense-head, and postprocessed baseline comparisons.

The absent-class negative control ran actual inference and exited nonzero with a failed result bundle.
The hardware preflight on this Mac exited nonzero and reported CUDA blocked, rather than counting MPS as CUDA.
Its NVIDIA execution branch remains untested until a remote GPU is available.

These photographs are integration fixtures, not representative robotics accuracy data.
This run makes no CUDA, TensorRT, residency, or performance claim.

Reproduce using the commands in [local development](../LOCAL_DEVELOPMENT.md).
Full local bundles are in benchmark-results/iteration-003-committed-cpu and benchmark-results/iteration-003-committed-mps.
The negative-control and Mac-preflight bundles are separate diagnostic artifacts.

Next external dependency: one NVIDIA GPU session with CUDA and profiling access.
The user has no GPU yet. [Runpod setup](../RUNPOD_SETUP.md) supplies the preparation and preflight steps.

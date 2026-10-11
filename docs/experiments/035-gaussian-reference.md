# Experiment 035: Gaussian reference through real classifier inference

All four required Gaussian sizes passed explicit CPU and MPS preprocessing through the pinned CPU MobileNet classifier.
The clean source revision was 16e0e48ccab8e1e9bb955f21251dc74eb91a93cd.
Each run used sigma 1.2, the pinned Chelsea photograph, and the existing ImageNet normalization contract.
The independent NumPy oracle and grouped PyTorch convolution tensors agreed within the predeclared 2e-4 absolute tolerance.
Classifier logits agreed within 0.001 absolute and 1e-4 relative tolerance.
Top-five classes matched and the existing cat probability check passed for each size.

The maximum CPU tensor error was 1.327321e-5 and the maximum CPU logit error was 2.765656e-5.
See [CPU report](data/035/cpu.json) and [MPS host report](data/035/mps-host.json) for per-size measurements, input/model identities, environment, and plans.
The first MPS attempt reported the device unavailable inside the sandbox.
Retain [that failed report](data/035/mps-sandbox-failure.json).
The rerun with host hardware access passed. CPU fallback was not enabled.
MPS output downloads and CPU classifier inference are explicit reference behavior.

Separate local checks verified Gaussian symmetry, coefficient normalization, replicated constant-image borders, and the narrow-sigma identity limit.
All four Gaussian YAML descriptions matched their Python graphs.
The existing stencil/configuration checks passed 24 cases.
These checks do not validate arbitrary sigma/model accuracy, CUDA execution, GPU residency, or performance.

Gaussian graphs currently lower to one full two-dimensional stencil.
A separable CUDA strategy, multiple stencil stages, and complete detector comparisons remain pending.
The prepared hardware gate must include Gaussian coefficients before claiming Gaussian GPU acceptance.
No paid resources were created for this local experiment.

## Reproducible Gaussian YAML repeat

Clean revision 4c5c775 repeats CPU and MPS preprocessing using the Gaussian YAML frontend for actual inference input.
Both runs passed the same four sizes, tensor tolerance, real classifier logits, top-five ordering, and cat probability checks.
The harness also preserves Python and relative convolution-file graph equality.
See [CPU YAML repeat](data/035/cpu-yaml-repeat.json) and [MPS YAML repeat](data/035/mps-yaml-repeat.json).
Each record includes the Gaussian configuration hash and explicit graph-equality result.
The four retained gaussian-N.yaml files match those hashes.
MPS fallback remained disabled. CPU inference and MPS output download remain explicit.
This repeat improves configuration-path reproducibility without extending CUDA or performance acceptance.

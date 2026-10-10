# Experiment 033: configurable stencils through local classifier inference

All required stencil sizes passed explicit CPU and MPS preprocessing through the pinned MobileNet classifier.
Each path checked 3×3, 5×5, 7×7, and 9×9 asymmetric kernels on the pinned Chelsea photograph.
The independent NumPy tensor and grouped PyTorch convolution tensor agreed within the predeclared 2e-4 absolute tolerance.
Real classifier logits agreed within 0.001 absolute and 1e-4 relative tolerance.
Top-five class order matched and the expected cat group passed the existing semantic check.
Literal Python graphs and relative YAML kernel-file graphs were equal for every size.

The original clean revision was `f8168dcea5b00c217b0c058c08afa29ee4f520a0`.
The manifest/fallback-policy repeat used clean revision `da22d12357f36b870e0405611589939b032d7d77`.
All four original and repeated reports passed. See [CPU repeat](data/033/cpu-repeat.json) and [MPS repeat](data/033/mps-repeat.json).
The repeat records PyTorch, Python, platform, fixture manifest hash, input hash, model hash, parameters, and plans.
MPS fallback was not enabled. MPS output downloads and CPU inference are explicit reference behavior.
No CUDA, TensorRT, GPU residency, or performance claim follows from these local results.

Maximum repeated CPU tensor error was 2.0951e-5. Maximum CPU logit error was 4.1008e-5.
Maximum repeated MPS tensor error was 2.0951e-5. Maximum MPS logit error was 4.3869e-5.
These measurements apply to these kernels and input, not arbitrary numerical stability or industrial model accuracy.

The local suite passed 211 checks after the implementation correction.
An oversized-coefficient check initially failed because this Python build packed an FP32 infinity without raising.
The frontend now explicitly requires the converted coefficient to remain finite.
Direction checks use an asymmetric corner impulse and a manually shifted edge-replicated expected image for every stencil size.
A negative-intermediate check rejects accidental clamping.

CUDA stencil execution remains explicitly unsupported.
The next implementation must specialize coefficients and preserve this semantic contract through complete real inference.
The existing detector graphs without stencils remain unchanged.
See [the contract](../STENCILS.md). New implementation source does not import cuda-conv code.

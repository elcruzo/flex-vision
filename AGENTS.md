# Repository guidance

## Goal

Build a GPU preprocessing runtime that converts camera frames into inference-ready tensors with measured reductions in overhead.
Read docs/PLAN.md before choosing implementation work.
Use the latest CUDA Preprocess Graph direction when older FluxVision requirements differ.

## Work rules

- Keep GPU pixel payloads on the GPU during the normal GPU path.
- Define numerical semantics before optimizing an operation.
- Preserve buffer ownership and stream dependencies across framework boundaries.
- Measure complete pipelines against credible GPU baselines.
- Use docs/TESTING.md for end-to-end acceptance through real inference.
- Use docs/HARDWARE.md for target setup and environment limits.
- Do not count unit-test passes or skipped GPU jobs as full-system validation.
- Preserve required scope and candidate ideas using docs/REQUIREMENTS.md and docs/OPTIONS.md.
- Mark unsupported platforms and unmeasured results explicitly.
- Check current primary sources before choosing version-sensitive APIs.
- Update docs/DECISIONS.md when evidence changes a design choice.
- Commit in small, coherent steps with relevant validation.
- Preserve unrelated user changes.

Do not expand the operator catalog before testing the performance hypothesis.
Do not treat proposed APIs or platform targets as implemented support.
Do not publish packages or create releases without a user request.

## Writing skill

Read .agents/skills/asd-ste100/SKILL.md when writing repository documentation or technical instructions.
Use its prose guidance without claiming certified STE compliance.
Preserve qualifications and technical meaning when simplifying text.
Keep the upstream license and source revision with the bundled skill.

## Current state

The repository contains a graph frontend and explicit NumPy/PyTorch reference paths.
Read docs/LOCAL_DEVELOPMENT.md and docs/SOURCE_AUDIT.md before continuing.
An RTX 4090 CUDA reference/inference smoke experiment passed. Read docs/experiments/004-cuda-host.md.
An initial synchronous fused CUDA backend exists for the detector graph. Read docs/CUDA_BACKEND.md.
Full G0/G1 acceptance and the two-workload performance gate remain pending.

A fixed FP32 TensorRT consumer passed on an L4. Read docs/experiments/006-tensorrt.md.
The first matched latency comparison exists in docs/experiments/007-latency.md.
Inference controls in docs/experiments/008-controls.md support a preceding-memory-state effect.
Before another paid rental, verify independent termination. The local timer failed during a long Mac-session pause.
The inspection reference passed CPU and MPS preprocessing through real classifier inference. Read docs/INSPECTION_REFERENCE.md and docs/experiments/009-inspection-reference.md.
The second GPU baseline passed. Read docs/experiments/012-inspection-cuda.md.
The ROI candidate passed local classifier checks. Read docs/experiments/013-inspection-roi.md.
The matched CUDA comparison passed for the 4K case. Read docs/experiments/014-inspection-comparison.md.
The experimental InspectionPipeline exposes explicit full and ROI strategies. Read docs/INSPECTION_RUNTIME.md.
Integrated CUDA correctness, ROI timing, and traces passed on an L4. Read docs/experiments/016-integrated-inspection.md.
Next, validate stream ownership and retained outputs through inference before broadening graph support. Small-image preprocessing was slower.
Reference operators are not production CUDA support.

GitHub Actions is optional. The default rental path uses a disposable Runpod CPU controller. Read docs/GPU_LEASE.md.
The controller cleanup test passed. Read docs/experiments/011-direct-lease.md for evidence and remaining failure limits.
An unavailable GitHub Actions runner is not a project prerequisite.

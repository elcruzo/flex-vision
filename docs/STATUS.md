# Product progress

Updated October 9, 2026. This assessment uses the full scope in REQUIREMENTS.md.
No milestone percentage is assigned. The remaining gates differ substantially in effort and hardware needs.

## Where we are

The project has evidence for its performance hypothesis on two fixed workloads.
It remains an experimental runtime, with substantial work before the complete ROS/Jetson MVP.
The detector and inspection measurements justify continued development. They do not complete the release criteria.

| Goal | Evidence now | What remains |
| --- | --- | --- |
| Correct detector input through actual inference | Fused RGB/BGR letterbox, normalization, FP16/FP32 NCHW, pinned YOLO/TensorRT controls | Complete interoperability, stride, stream, and platform matrix |
| Lower overhead on two realistic workloads | About 26% lower detector preprocessing p50 versus CV-CUDA; about 42% lower 4K inspection preprocessing versus full-frame PyTorch | Full success-gate acceptance, broader baselines and tradeoff checks |
| Stable long-running inspection | L4 30-minute soak: 338,155 correct completed frames | Target-platform repeats and live-camera coverage |
| Stable detector load | RTX 4090 short matrix: 187,279 correct completed frames, zero declared post-drain growth | Prepared 30-minute detector soak, service-tail investigation, unresolved earlier vendor-pool increases |
| Reusable asynchronous runtime | Synchronous owned detector output; experimental caller-owned output awaits GPU acceptance; inspection stream handoff experiments | Explicit reusable workspace, asynchronous lifetimes, batches, and concurrent detector streams |
| Configurable pipeline compiler | Immutable graph, strict subset YAML, fixed detector plan, experimental inspection plans | User stencils, full required operator matrix, general fusion and backend selection |
| Measured tuning and profiling | Experiment scripts, retained raw samples and sanitized transfer traces | Public tune/profile/benchmark commands, compatible plan cache, safe invalidation |
| ROS camera-to-model path | Required contracts and tests documented | ROS node, CUDA-backed buffer integration, CPU-image upload path, live camera and downstream transport |
| Jetson and distribution | x86 NVIDIA development evidence, local Mac references | Orin/Thor validation, CSI demo, clean binary installation, license decision, PyPI and ROS release work |

## What the improvements mean

The detector preprocessing improvement reduces a small part of total inference time.
On the latest person fixture, complete-host p50 changed from about 0.889 ms with PyTorch to 0.869 ms with CPG.
Complete-host p99 was slightly worse for CPG in that comparison.
Overload throughput varied between repeats. There is no consistent detector throughput advantage to claim.

The inspection rewrite avoids filtering irrelevant pixels outside the requested ROI and its necessary halo.
On the measured 4K workload, complete-host p50 fell from 9.0664 ms to 5.6988 ms.
Small-image preprocessing was slower. This strategy is useful when the input and ROI justify it.
It is not evidence that every custom CUDA kernel beats a vendor primitive.

The resident replay inputs omit acquisition and upload costs.
Logical camera feeds are synthetic schedules, not live cameras or concurrent detector streams.
The results do not establish industrial model accuracy, ROS transport behavior, Jetson power, or every platform's performance.

Evidence: [inspection comparison](experiments/014-inspection-comparison.md),
[inspection soak](experiments/019-inspection-soak.md), and
[detector repeatability and load](experiments/021-detector-repeatability.md).

## Next sequence

1. Reconcile pending cloud charges within the authorized budget.
2. Run the prepared thermally conditioned detector soak and independently audit its evidence.
3. Investigate service tails using the retained per-frame, telemetry, and allocator records.
4. Define output-buffer ownership and asynchronous execution semantics before changing the runtime.
5. Implement one measured runtime change, then repeat the complete detector inference scenario.
6. Add the configurable stencil path and robot workload as the plan requires.
7. Develop and validate ROS transport and Jetson capture before claiming the flagship product path.

The existing two-workload evidence permits focused runtime development.
It does not remove any required operator, integration, or release deliverable.
Do not expand the filter catalog to substitute for unresolved ownership, tail-latency, or transport work.

## Current hardware constraint

The Mac supports editing, CPU/MPS references, model checks, and offline evidence audits.
It cannot validate CUDA, TensorRT execution, or NVIDIA performance.
The 30-minute detector soak is prepared, not measured.
Pending cloud reservations currently leave too little unreserved budget for another guarded lease.
No GPU rental is active or scheduled for this preparation.

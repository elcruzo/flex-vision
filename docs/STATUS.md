# Product progress

Updated October 10, 2026. This assessment uses the full scope in REQUIREMENTS.md.
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
| Stable detector load | RTX 4090 30-minute soak: 432,000 correct completed frames, zero drops, zero declared allocator growth | Broader platform/load coverage, service-tail investigation, unresolved earlier vendor-pool increases |
| Reusable asynchronous runtime | Synchronous owned detector output; experimental caller-owned FP16 output passed fixed TensorRT checks; inspection stream handoff experiments | Explicit reusable workspace, asynchronous lifetimes, batches, and concurrent detector streams |
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
[detector repeatability and load](experiments/021-detector-repeatability.md), and
[detector soak and output reuse](experiments/022-detector-soak.md).

## Next sequence

1. Reconcile rental reservations within the existing $15 authorization and verify reliable evidence export.
2. Preserve the passed delayed-consumer gate from experiment 027, including the missed export from experiment 026.
3. Repeat the matched serial comparison after async trace and allocator evidence. The first comparison passed in experiment 027.
4. Verify async-specific traces, allocator behavior, repeatability, and the complete-host p99 guard before selecting a strategy.
5. Evaluate bounded concurrent execution and reusable workspace with explicit consumer completion.
6. Add the configurable stencil path, general planning, and measured tuning as the plan requires.
7. Develop and validate ROS transport and Jetson capture for the flagship camera-to-TensorRT path.

Keep caller-owned output optional. Its measured allocation benefit did not produce a latency improvement.
The submission API, fixed ownership, delayed-consumer gate, and first serial comparison passed their scoped checks.

The existing two-workload evidence permits focused runtime development.
It does not remove any required operator, integration, or release deliverable.
Do not expand the filter catalog to substitute for unresolved ownership, tail-latency, or transport work.

## Current hardware constraint

The Mac supports editing, CPU/MPS references, model checks, and offline evidence audits.
It cannot validate CUDA, TensorRT execution, or NVIDIA performance.
The 30-minute detector soak and fixed caller-owned FP16 output scenario passed on RTX 4090.
Lifetime-bounded reconciliation permitted the completed lease within the existing $15 authorization.
No GPU rental remains active.

Matched output reuse passed 40,000 inference checks. Complete-host p99 increased about 0.2%, within the fixed 5% guard.
Fresh owned output remains the default. See [experiment 023](experiments/023-output-reuse.md).

The fixed detector stream scenario passed 48 TensorRT executions across six fixtures and CuPy/PyTorch inputs.
Three distinct streams used explicit handoffs. Retained outputs and shutdown passed.
Both preprocessing and TensorRT remain synchronous. Async execution remains open.
See [experiment 024](experiments/024-detector-streams.md).

The owned-output submission API passed 48 fixed TensorRT executions with early caller-source release.
See [experiment 025](experiments/025-async-submission.md), including the first pinned stream compatibility failure.
This is integration correctness progress, not a measured async latency gain.
Delayed-consumer behavior, async traces, and matched performance are the next gates.

The delayed-consumer rental in experiment 026 missed export during a local session pause.
No delayed-consumer acceptance was obtained. Exact cloud resource absence was independently verified.
The coordinator now checks the independent wall deadline before polling and export.
See [experiment 026](experiments/026-expired-delayed-consumer.md). Reconcile reservations before another paid run.

Independent evidence export is prepared through an optional GPU-side signed PUT URL.
The local archive/upload/retrieval check passed. The full local suite passed 139 checks.
See [export setup](DURABLE_EXPORT.md). No storage destination or cloud export acceptance exists yet.
Experiment 026 remains unvalidated. Provider export needs validation only if that optional route is selected.

The recent four ended leases now have lifetime-bounded replacement reservations.
The cumulative conservative bound is $13.295168122393484 against the existing $15 authorization.
No new rental was created. See RUNPOD_SETUP.md and its calculation record.
The delayed-consumer runner now regenerates complete numerical controls without repeating baseline timing and trace capture first.
Experiment 027 passed the fixed delayed-consumer gate. Numerical-only controls cannot count as performance evidence.


Experiment 027 passed 48 delayed-consumer checks and 40,000 matched serial inference checks on RTX 4090.
Complete-host p50 improved 9.38–9.42%, with p99 improved 8.99–10.37% in this run.
See [experiment 027](experiments/027-delayed-and-async.md).
Keep the synchronous default. Async traces, allocator observations, independent repeats, and concurrency remain pending.


The async diagnostic capture and allocator gate is prepared in `scripts/run_yolo_async_diagnostics.sh`.
It runs 4,000 checked serial frames and audits capture controls, transfers, and post-drain pool observations.
See ASYNC_BENCHMARK.md and D052/D053. Experiment 028 passed its fixed GPU diagnostic protocol.
Local fault checks reject incomplete frames, missing checkpoints, engine mismatch, and tensor-sized host transfers.
They do not establish hardware acceptance or production async support.


Experiment 028 passed 4,000 async diagnostic inference checks, transfer capture, and declared post-drain pool observations.
See [experiment 028](experiments/028-async-diagnostics.md). No latency selection follows from instrumented samples.
The synchronous default remains unchanged. Independent performance repeats and bounded concurrency remain pending.


An experimental version-3 CAI-only detector adapter now implements explicit producer and completion ordering.
DLPack remains preferred for dual-protocol objects. Local checks passed, but real GPU inference acceptance remains pending.
See CUDA_BACKEND.md and D054 before claiming CUDA Array Interface interoperability.

The CAI-only TensorRT harness now prepares 144 bidirectional-ordering and retained-output scenarios.
See CAI_VALIDATION.md and D055. Local coverage fault checks passed.
Hardware execution, CAI-specific transfer capture, invalid-device controls, and broader interoperability remain pending.


Experiment 029 failed in the CAI harness before its first import because CuPy Event lacks query().
The corrected harness uses Event.done and preserves the required pending boundaries.
The DLPack preference also avoids evaluating an unused CAI descriptor.
See experiments/029-cai-event-failure.md. CAI GPU acceptance remains pending and all rental resources are absent.

Experiment 030 passed seven CAI scenarios before the harness violated stream=None input reuse ordering.
Protocol cai-v2 now separates caller completion from advertised-stream completion fencing.
See experiments/030-cai-ready-reuse.md and D057. All rental resources are absent.
The full 144-case gate, CAI transfer capture, and broader interoperability remain pending.

Experiment 031 passed all 144 fixed CAI-only TensorRT correctness and ownership scenarios on RTX 4090.
Advertised streams fenced immediate reuse. Ready-input callers completed preprocessing before writing again.
See experiments/031-cai-inference.md. CAI transfer capture and broader interoperability remain pending.
All rental resources are absent. Keep the synchronous default and existing async limits.

A focused CAI transfer-capture runner and evidence verifier are prepared.
They bind 16 complete-inference ranges to the same clean 144-case gate and engine.
The verifier rejects missing preprocessing, missing inference work, and aggregate pixel-sized host transfers.
Local checks cannot establish residency. Hardware capture remains pending. See D059.

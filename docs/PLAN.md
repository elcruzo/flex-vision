# Delivery plan

## Outcome

Help a perception engineer move camera data into a model with less latency, memory movement, and integration code.
The unit of success is a complete pipeline, not an isolated filter.

This plan defines outcomes and evidence. It does not freeze the implementation.
G0 source review and the initial local G1 reference slice exist.
Full G0/G1 acceptance remains pending. An RTX 4090 CUDA reference/inference smoke run passed.
The initial synchronous CUDA backend and fixed FP32 TensorRT consumer passed real inference checks.
See [TensorRT evidence](experiments/006-tensorrt.md). Broader runtime acceptance remains pending.
[Experiment 007](experiments/007-latency.md) records the first matched latency comparison on one L4 workload.
[Experiment 008](experiments/008-controls.md) tested fixed inputs, output reuse, and common memory conditioning.
It supports a preceding-memory-state effect. The two-workload success gate remains open.
Independent Runpod controller cleanup passed in [experiment 011](experiments/011-direct-lease.md).
Use that verified path for the next GPU experiment and verify cleanup after each rental.
Use the [direct Runpod controller](GPU_LEASE.md) for rentals. GitHub Actions is an optional alternative, not a prerequisite.
[Experiment 009](experiments/009-inspection-reference.md) validates the complete inspection reference through a pretrained classifier locally.
[Experiment 012](experiments/012-inspection-cuda.md) now establishes the matched CUDA reference, repeated latency baseline, and zero-copy Nsight range for that workload.
That baseline identified 4K preprocessing as the next optimization target. Subsequent experiments below measured the ROI improvement.
[Experiment 013](experiments/013-inspection-roi.md) validates a crop-aware candidate locally through real classifier inference.
[Experiment 014](experiments/014-inspection-comparison.md) passed the matched L4 comparison: 42% lower 4K preprocessing latency through real classifier inference.
Small-image preprocessing was slower. The [experimental fixed inspection API](INSPECTION_RUNTIME.md) now exposes explicit full and ROI plans.
[Experiment 016](experiments/016-integrated-inspection.md) validated integrated CUDA correctness, ROI timing, and transfer traces on an L4.
[Experiment 017](experiments/017-inspection-streams.md) passed 24 cross-stream classifier cases with explicit handoffs and retained-output checks.
Its first timing run failed the small-image p99 guard. A revised run passed, with both results retained.
The [sustained inspection experiment](experiments/018-sustained-inspection.md) passed its short CUDA correctness and performance checks.
Its revised fixed-stream run measured 63.48% higher overload throughput and stable post-drain allocations across repeats.
The initial run exposed allocation growth when the harness introduced new streams between blocks. Both runs remain in the evidence.
The [sustained inspection protocol](SUSTAINED_INSPECTION.md) records the load, queue policy, validation overhead, and stream-lifetime correction.
Treat its short runs as development evidence, not the comparison or soak tiers in TESTING.md.
The [continuous inspection soak](experiments/019-inspection-soak.md) now passes the 30-minute tier on one L4 configuration.
All 338,155 completed frames passed, with zero declared allocator growth. The overload schedule dropped 21.72% of offered frames.
This does not complete live-camera, Jetson, ROS, or detector soak coverage.
The measured SSDLite FP32 workload does not complete the required 640×640 FP16 YOLO/TensorRT scenario.
The fixed inspection API does not complete configurable stencils, reusable workspace, batching, or general graph planning.
The [FP16 YOLO/TensorRT experiment](experiments/020-fp16-yolo.md) passed six GPU cases, 60,000 matched samples, and an independent transfer capture.
The cached path measured 24.77% lower preprocessing p50 versus CV-CUDA and 37.10–38.66% lower p99 versus PyTorch.
Complete host p50 improved by 2.19–3.83%. This is fixed second-workload development evidence, not full release acceptance.
The [detector repeatability and load experiment](experiments/021-detector-repeatability.md) repeated the preprocessing gain.
Its final twelve-block matrix checked 187,279 completed frames with zero declared post-drain allocator growth.
Two earlier vendor-pool failures remain failed. Sustained throughput and tail gains varied by block.
This is short serial load evidence, not concurrent camera streams or a detector soak.
The [30-minute detector soak](experiments/022-detector-soak.md) passed: all 432,000 offered frames completed correctly, with zero drops and zero declared allocator growth.
The synchronous caller-owned FP16 output scenario also passed six TensorRT fixtures. Experiment 023 measured zero output-pool requests with reuse, but preprocessing host p50 increased about 5.2%.
Keep caller-owned output optional. Define consumer ordering before asynchronous execution.
[The asynchronous contract](ASYNC_EXECUTION.md) defines ownership, slot states, shutdown, and acceptance.
The three-stream detector harness passed 48 real TensorRT executions on RTX 4090.
See [experiment 024](experiments/024-detector-streams.md). Asynchronous execution remains pending.
See [experiment 023](experiments/023-output-reuse.md) for the matched inference result.
Keep broader sustained coverage, vendor comparisons, reusable workspace, batching, and the robot/ROS workloads open.
See [the backend contract](CUDA_BACKEND.md), [host evidence](experiments/004-cuda-host.md), and [fused execution evidence](experiments/005-fused-cuda.md).
See [local development](LOCAL_DEVELOPMENT.md) and [source audit](SOURCE_AUDIT.md) for current evidence.

[REQUIREMENTS.md](REQUIREMENTS.md) preserves the complete product scope and maps each requirement to its milestone.
It includes target APIs, configuration, every MVP operator, ROS contracts, tuning records, research experiments, and distribution deliverables.
[TESTING.md](TESTING.md) defines the complete inference feedback loop and scenario acceptance checks.
[HARDWARE.md](HARDWARE.md) defines cloud and Jetson setup, access needs, and the limits of each environment.
[OPTIONS.md](OPTIONS.md) preserves candidate approaches and earlier FluxVision possibilities without changing their priority.
An early milestone subset does not replace the complete release scope.
Changes to required outcomes need an explicit scope decision. Backend choices remain open to evidence.

See [the current product assessment](STATUS.md) for completed evidence and remaining release gates.

## Rules for progress

1. Start each milestone with its hypothesis and acceptance checks.
2. Check current primary documentation before choosing version-sensitive APIs.
3. Implement the smallest complete workload that tests the hypothesis.
4. Record correctness, performance, and limitations together.
5. Update the decision log when evidence changes the approach.
6. Commit each coherent change with its relevant checks.

A failed experiment is useful evidence. Do not expand the filter catalog to hide a failed performance hypothesis.

## G0 — Establish a reproducible starting point

**Outcome:** Another engineer can reproduce the source baseline and understand the target environment.

Work:

- Audit cuda-conv at a recorded commit before code reuse.
- Check its license and preserve required attribution.
- Identify reusable stencil kernels, presets, correctness tests, and CUDA event measurements.
- Select a project license before importing code or publishing packages.
- Record available GPU hardware, drivers, toolkit, operating system, and Python versions.
- Check the actual compatibility of Jetson, CUDA, ROS, CuPy, PyTorch, and TensorRT versions.
- Define image semantics and model input requirements before optimizing them.

Exit evidence:

- A source audit records what to retain, replace, or defer.
- A minimal GPU test runs on at least one supported development target.
- A compatibility table distinguishes tested, planned, and unavailable environments.
- Correctness checks can run separately from performance captures without replacing full inference validation.
- A target passes the hardware preflight, including real inference and trace export.
- The selected hardware has a recorded access path, environment manifest, and run budget where relevant.

Target environments are Jetson AGX Orin, Jetson AGX Thor, and x86-64 NVIDIA GPUs.
The requested targets include CUDA 13 and ROS 2 Lyrical, subject to actual platform compatibility.
The local planning environment does not prove GPU support.

## G1 — Make the detector input contract correct

**Outcome:** A GPU camera frame becomes the exact tensor that a detector expects.

Start with RGB8 or BGR8 HWC input and one image.
Implement letterbox, channel conversion, scaling, normalization, and FP16/FP32 output in NCHW format.
Keep the graph description separate from execution from the start.
An initial planner can choose a fixed strategy.

Exit evidence:

- Independent reference tests cover pixels, output shape, channel order, and dtype.
- Tests cover odd dimensions, extreme aspect ratios, padding, strides, and invalid inputs.
- Tests define resize coordinates, rounding, normalization, and border behavior.
- CuPy input and PyTorch CUDA input stay on the GPU.
- Tests prove ownership and producer/consumer stream synchronization.
- Repeated calls do not overwrite output that the consumer still owns.
- A minimal TensorRT consumer accepts the output without a host copy.
- CuPy, PyTorch CUDA, CUDA Array Interface, and DLPack each have explicit interoperability tests.
- Python and YAML examples have an equivalence fixture for later configuration delivery.

G1 requires scenario E01 from TESTING.md: actual model inference, checked output, and an exported result bundle.
Unit-test success or a tensor with the expected shape is insufficient.
Prefer a simple correct implementation before a specialized kernel.
Do not label a CPU reference as the production GPU backend.

## G2 — Establish credible pipeline measurements

**Outcome:** A reproducible benchmark identifies the actual overhead and useful optimization targets.

Implement detector and inspection baselines with supported GPU libraries.
Define the multi-camera workload now, even if its full ROS transport comes later.
Use the same image semantics, precision, and output contract for each comparison.

Exit evidence:

- Raw samples and environment metadata reproduce p50, p95, p99, and throughput.
- Reports separate cold compilation, warm execution, and end-to-end latency.
- Reports identify allocations, transfers, launches, and synchronization points.
- At least two realistic workloads have a working GPU baseline through their actual inference consumer.
- Missing baselines have a reason and a follow-up condition.
- The robot workload specifies camera count, offered load, queues, calibration, and inference endpoint.
- Each required baseline has a per-workload applicability record.

Use [the benchmark protocol](BENCHMARKS.md) for all comparisons.
Do not derive a speed claim from a CPU-only comparison.

## G3 — Prove the runtime advantage

**Outcome:** Optimize two complete workloads without changing their defined numerical behavior.

Candidate approaches:

- Fuse letterbox, normalization, type conversion, and output layout.
- Reuse temporary buffers with explicit lifetimes.
- Specialize constant stencils for 3×3, 5×5, 7×7, and 9×9 kernels.
- Compare separable Gaussian passes against direct and tiled implementations.
- Fuse arithmetic after a stencil when semantics permit it.
- Compare vendor primitives against internal implementations at the graph level.

Exit evidence:

- The optimized result passes the same independent reference checks.
- A trace supports the reported transfer and launch counts.
- Concurrent streams and batches pass full-inference ownership, output-retention, and correctness scenarios.
- Memory use reaches a bounded steady state during a sustained run.
- Two realistic pipelines meet the success gate below.
- All stencil sizes accept literal coefficients and configuration files.
- Common constant kernels have compile-time specialization and unrolling evidence.
- The full operator matrix in REQUIREMENTS.md has implementation and correctness checks before release.

### Success gate

On each of at least two pipelines, demonstrate one of these improvements against a credible GPU baseline:

- At least 20% lower preprocessing p50 latency.
- At least 20% lower preprocessing p99 latency.
- At least 20% higher sustained multi-camera throughput.
- Materially lower GPU or CPU use.
- Fewer memory copies or temporary allocations.

Report the practical effect of fewer copies or allocations without adding a new success threshold.
Define “materially lower” before collecting the comparison results.
Record acceptable tradeoffs in accuracy, memory, power, throughput, and tail latency before the run.
A faster median must not conceal an unacceptable tail regression.

If the gate fails, inspect the bottleneck and revise the architecture.
Possible outcomes include a narrower operator scope, a vendor-backed planner, or a dedicated stencil extension.
Do not proceed to broad operator expansion without evidence.

## G4 — Make plans repeatable and inspectable

**Outcome:** Engineers can explain a plan and reuse a measured strategy on compatible hardware.

Proposed commands are `cpg profile`, `cpg benchmark`, and `cpg tune`.
These commands do not exist yet.

Exit evidence:

- The profiler shows original operations, selected implementations, and fusion boundaries.
- Tuning considers only numerically valid candidates.
- Each candidate records latency distribution, workspace size, and launch count.
- Cache keys include device, software versions, input contract, graph, and numerical policy.
- A cache mismatch triggers a safe default or retuning.
- Production execution does not tune unexpectedly on the frame path.
- The execution plan reports unsupported metrics without invented values.

Autotuning is a required deliverable. Compare it with a stable heuristic planner to measure its benefit.
The tuning record must include GPU architecture, image size, pixel format, pipeline, chosen implementation, p50, p95, p99, workspace, and launches.
Test that production loads a compatible cached plan without repeating tuning.
The profiler must compare measured original stages and optimized groups, including totals and transfer counts.
CUDA Graph capture is a separate candidate optimization, not a requirement implied by the project name.

## G5 — Preserve GPU data through ROS and inference

**Outcome:** A camera-to-model example preserves ownership and avoids unnecessary payload transfers across component boundaries.

Implement the requested CUDA-backed `rosidl::Buffer` path for ROS 2 Lyrical after checking its current API and downstream support.
Track incompatible or unavailable APIs as explicit blockers or scope decisions.
Do not silently substitute a generic CUDA message for this requirement.
Build a small transport experiment before committing to a ROS interface.
Document where GPU residency starts for each camera source.

Exit evidence:

- A CUDA-backed message reaches preprocessing and inference without a host payload copy.
- An ordinary CPU Image follows an explicit, measured upload path.
- Message metadata preserves timestamps, encoding, dimensions, and frame identity.
- Tensor output carries shape, dtype, layout, ownership, and synchronization information.
- Ordinary ROS tools have a documented inspection path and explicit copy costs.
- Backpressure, queue limits, dropped frames, and shutdown behavior have tests.
- A TensorRT example records preprocessing, inference, and total latency separately.
- An Nsight Systems trace supports the residency claim.
- The package accepts sensor_msgs/Image and exposes Image or downstream-compatible tensor output.
- Tests cover the documented preprocess_node executable and cuda_preprocess_node component names.
- The yolo_demo.launch.py entry point runs the Jetson camera-to-TensorRT chain.
- The demo reports camera dimensions/FPS, preprocessing, inference, total latency, CPU copies, GPU memory, and dropped frames.

Do not force an NCHW tensor into an Image message with misleading image metadata.
Choose a tensor message only after checking the inference consumer's contract.
Rectification requires calibration and its own reference tests. It can remain a vendor stage in the robot workload.

## G6 — Make the release reproducible

**Outcome:** A new engineer can install, run, inspect, and reproduce the supported pipeline.

Exit evidence:

- Packaging works in clean environments from the tested compatibility table.
- Supported binary packages need no manual CUDA compilation.
- The package documents any runtime compilation, cache, and toolkit requirements.
- The Python API and YAML schema provide versioned validation and actionable errors.
- The release includes batch, stream, and supported dtype coverage.
- The README opens with real camera-to-model benchmark evidence.
- The Jetson demo reports measured values and includes trace instructions.
- License, attribution, release artifacts, and supported platforms are explicit.
- Every requirement R01–R21 has evidence or an explicit, accepted scope change.
- Detector, inspection, and multi-camera robot reports include their actual inference endpoints.
- ROS indexing and clean PyPI installation have release tasks and verification records.
- Research requirement R22 links to experiments or states its pending research status.
- E01–E12 have scenario results, target coverage, and any explicit blockers.
- Clean-install, sustained-load, live-camera, and trace artifacts accompany release claims.

GitHub is the project home. PyPI distributes Python packages.
ROS packaging follows validated integration.
Hugging Face hosts a secondary model demo with a pinned model and exact preprocessing configuration.
Kaggle is a secondary education channel. Upstream OpenCV, Kornia, or ROS contributions remain optional.
Check current issue status before planning an upstream contribution.
Do not publish package names, releases, or third-party contributions as part of repository initialization.

## First implementation sequence

1. Complete the source and environment audit from G0.
2. Define a detector fixture, pinned pretrained model, and its numerical reference.
   Run reference preprocessing through real PyTorch inference locally on CPU or supported MPS operations.
3. Add graph construction, validation, and an inspectable fixed plan.
   Move the first CUDA-dependent experiment to NVIDIA hardware without waiting for the full frontend.
4. Execute the unfused GPU detector path.
5. Add stream and ownership tests for CuPy and PyTorch.
6. Run real TensorRT inference and export baseline samples, correctness results, and a focused trace.
7. Add the fused detector candidate and compare it.
8. Add the inspection workload and the first custom stencil.

Each item should produce a reviewable commit or a small series of commits.
The sequence can change when evidence changes the dependencies.

## Deferred scope

Defer a stable C++ ABI, broad backend coverage, and general graph scheduling until demand or measurements justify them.
Defer morphology, thresholding, debayering, and a large filter catalog.
Keep the research paper as an evidence summary after reproducible experiments.
Check hardware access and compatibility before promising a deadline.


## Feedback-loop execution policy

Each implementation milestone must exercise the affected complete workload on its required hardware.
Use tests to diagnose failures, then rerun the full scenario after the fix.
Measure one changed strategy at a time against the same baseline and fixture.
Record unsuccessful experiments and revise the next hypothesis from the trace.
A missing GPU means blocked hardware validation, not a passed milestone.
A cloud-only result does not complete Jetson camera, power, or transport acceptance.

Do not spend on remote hardware during planning.
Before paid provisioning, confirm account access, allowed spending, storage retention, and shutdown responsibility.
Build the reproducible harness so the first rented session can test the real pipeline immediately.

## Detector soak gate

The continuous detector soak passed on one RTX 4090 configuration. See [experiment 022](experiments/022-detector-soak.md).
See [the frozen soak protocol](SUSTAINED_YOLO.md#continuous-detector-soak).
It requires measured thermal conditioning, thirty continuous minutes through TensorRT, per-frame checks, and independent evidence verification.
Local fault checks do not complete this gate. Reconcile cloud reservations before another rental under the existing $15 budget.

The experimental owned-output submission passed 48 fixed TensorRT executions in [experiment 025](experiments/025-async-submission.md).
Caller input references were released before consumption. The completion object retained their owners.
Keep the synchronous default. Delayed consumers, slot recycling, async traces, and matched performance remain pending.

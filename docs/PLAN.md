# Delivery plan

## Outcome

Help a perception engineer move camera data into a model with less latency, memory movement, and integration code.
The unit of success is a complete pipeline, not an isolated filter.

This plan defines outcomes and evidence. It does not freeze the implementation.
All runtime milestones are pending. No hardware result exists yet.

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
- Correctness tests can run separately from hardware performance tests.

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
- A minimal inference consumer accepts the output without a host copy.

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
- At least two realistic workloads have a working GPU baseline.
- Missing baselines have a reason and a follow-up condition.

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
- Concurrent streams and batches pass ownership and correctness tests.
- Memory use reaches a bounded steady state during a sustained run.
- Two realistic pipelines meet the success gate below.

### Success gate

On each of at least two pipelines, demonstrate one of these improvements against a credible GPU baseline:

- At least 20% lower preprocessing p50 latency.
- At least 20% lower preprocessing p99 latency.
- At least 20% higher sustained multi-camera throughput.
- Materially lower GPU or CPU use.
- Fewer transfers or temporary allocations with a useful measured system benefit.

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

Autotuning must justify its complexity against a stable heuristic planner.
CUDA Graph capture is a separate candidate optimization, not a requirement implied by the project name.

## G5 — Preserve GPU data through ROS and inference

**Outcome:** A camera-to-model example preserves ownership and avoids unnecessary payload transfers across component boundaries.

First validate current ROS 2 Lyrical buffer APIs and downstream message support.
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

GitHub is the project home. PyPI distributes Python packages.
ROS packaging follows validated integration. Model-hosting demos and educational notebooks are secondary.
Do not publish package names, releases, or third-party contributions as part of repository initialization.

## First implementation sequence

1. Complete the source and environment audit from G0.
2. Define a detector fixture and its numerical reference.
3. Add graph construction, validation, and an inspectable fixed plan.
4. Execute the unfused GPU detector path.
5. Add stream and ownership tests for CuPy and PyTorch.
6. Measure the baseline with raw samples.
7. Add the fused detector candidate and compare it.
8. Add the inspection workload and the first custom stencil.

Each item should produce a reviewable commit or a small series of commits.
The sequence can change when evidence changes the dependencies.

## Deferred scope

Defer a stable C++ ABI, broad backend coverage, and general graph scheduling until demand or measurements justify them.
Defer morphology, thresholding, debayering, and a large filter catalog.
Keep the research paper as an evidence summary after reproducible experiments.
Check hardware access and compatibility before promising a deadline.

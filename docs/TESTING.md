# End-to-end feedback loop

Status: implementation plan. The harness and commands described below do not exist yet.
This is the primary validation strategy for the runtime.
Unit tests can help isolate defects, but they cannot satisfy a pipeline milestone by themselves.

## The development loop

```text
pin code + environment + model + input + expected behavior
                         ↓
run credible baseline and current CPG through real inference
                         ↓
check pixels + model outputs + ownership + stream ordering
                         ↓
measure latency + throughput + memory + transfers + power
                         ↓
inspect traces and identify the limiting stage
                         ↓
change one strategy and rerun the same experiment
                         ↓
keep or reject the change, record evidence, commit
```

A run ends at a real model result, not merely a tensor allocation or successful kernel launch.
Include detection coordinate mapping or classifier outputs in the check.
The final robotics run includes capture, ROS transport, preprocessing, inference, and the downstream result subscriber.

## Harness deliverables

Build one reusable runner with these responsibilities:

| Part | Required behavior |
| --- | --- |
| Environment preflight | Check actual GPU execution, dependencies, profiling access, and available memory |
| Input source | Deterministic fixtures, recorded frames, timestamped replay, ROS bags, and live camera adapters |
| Independent reference | Produce expected tensor values and model inputs independently of optimized code |
| Baseline adapters | Execute applicable CuPy, PyTorch, vendor, and ROS alternatives |
| CPG adapter | Run unfused, fused, heuristic, and tuned plans through the same consumer |
| Real inference consumer | Load a pinned model and execute TensorRT with the produced device tensor |
| Load controller | Apply fixed arrival rates, bounded queues, concurrent streams, and overload |
| Correctness evaluator | Check numerical error, model outputs, frame identity, and coordinate mapping |
| Instrumentation | Record frame events, GPU events, allocations, transfer evidence, and telemetry |
| Artifact writer | Save samples, manifests, results, traces, failures, and comparison reports |
| Comparator | Compare matching runs and identify correctness or performance regressions |

Keep workload definitions separate from runner logic.
Use the same model, input corpus, output requirements, and arrival schedule across compared strategies.
Keep the baseline independent enough to detect errors in CPG semantics.
Do not make a baseline call CPG's optimized kernel underneath.

## Preflight before a paid benchmark session

1. Record the environment manifest from HARDWARE.md.
2. Execute a small GPU operation and check its result.
3. Exercise a non-default stream with an explicit consumer dependency.
4. Pass real data across CuPy and PyTorch without a host staging array.
5. Build or load the target model engine under verified compatibility rules.
6. Execute one full frame through preprocessing and TensorRT.
7. Collect a small trace with a deliberately introduced transfer outside the candidate production run.
8. Confirm that the measurement tools detect that transfer.
9. Write and export a complete result bundle.

A failed preflight prevents a performance claim, not all useful debugging.
Classify restricted counters separately from broken CUDA tracing or broken inference.
Do not spend a long benchmark session discovering that trace collection is unavailable.

## Fixtures and numerical contract

Use both deterministic synthetic inputs and a representative recorded image corpus.
Include channel-coded patterns, ramps, impulses, edges, constant images, and seeded noise.
Include tiny images, odd dimensions, extreme aspect ratios, border pixels, and real 1080p/4K frames.
Pin fixtures and model artifacts by checksum, source, and license.

Before timing, define tolerances for every output dtype and operation sequence.
Check maximum and distribution of absolute/relative tensor errors.
Test correlation versus convolution with an asymmetric stencil.
Use reference cases that reveal channel swaps, padding offsets, normalization errors, and incorrect resize coordinates.

Check model results as well as tensor values.
For detection, compare classes, scores, boxes, and the mapping back to the camera image.
For classification, compare logits or probabilities and label consistency.
Define acceptable numerical differences before collecting optimized results.
Do not accept a tensor mismatch only because the model predicts the same label.

Correctness checks may download outputs in a separate validation run.
Exclude those downloads from the normal GPU residency claim and performance capture.
Explicitly label diagnostic transfers in trace artifacts.

## Complete test scenarios

| ID | Scenario | Pass evidence |
| --- | --- | --- |
| E01 | 1080p RGB/BGR → letterbox → normalize → FP16 NCHW → TensorRT YOLO | Correct tensor, detections, coordinate mapping, and measured boundary |
| E02 | 4K → Gaussian → sharpen → ROI → resize → normalize → classifier | Correct ROI and model result, stable workspace, baseline comparison |
| E03 | Multi-camera → rectify → stencil → resize → normalize → TensorRT | Per-camera ordering, calibration, fairness, drops, tails, and throughput |
| E04 | Equivalent Python and YAML pipelines with kernel files | Equivalent graph semantics and model outputs, useful invalid-config errors |
| E05 | CuPy, PyTorch CUDA, CUDA Array Interface, DLPack inputs | Actual inference after each handoff with valid ownership and dependencies |
| E06 | Batches, concurrent streams, retained outputs | No stale frames, overwritten tensors, cross-stream races, or unbounded memory |
| E07 | Tune → save → restart process → production execution | Matching cached plan loads without tuning on the frame path |
| E08 | ROS CPU Image → upload → preprocessing → inference | Correct metadata and outputs, measured and disclosed transfer cost |
| E09 | ROS CUDA-backed rosidl::Buffer → preprocessing → inference | Verified buffer path, correct lifetime, payload remains GPU-resident |
| E10 | Jetson CSI capture → ROS → detector → result subscriber | Live camera metrics, trace, end-to-end timing, and launch reproduction |
| E11 | Clean installation on each claimed platform | Documented install and complete example succeed without manual CUDA compilation |
| E12 | Shutdown, restart, overload, malformed input, unavailable backend | Defined errors or recovery, bounded resources, no silent wrong results |

E01–E03 must run through the real inference consumer.
A replay-only E03 result does not complete E10.
Use every required MVP operator in at least one end-to-end fixture, including Sobel, clamp, and arithmetic variants.
Test all stencil sizes through the pipeline, not only isolated filter calls.

## Streams, lifetimes, and faults

Intentionally delay a consumer while the next frame arrives.
Retain several outputs across subsequent calls and check that earlier contents remain valid.
Drop the caller's input reference after submission and check ownership until work completes.
Run producers and consumers on different non-default streams with explicit dependencies.
Exercise strided inputs, unsupported layouts, and wrong-device inputs.
Require either correct support or an explicit rejection for each case.

Change resolution and batch size during a sustained session.
Test cache eviction, workspace limits, allocation failure, and clean shutdown after errors.
Restart the ROS subscriber and interrupt a camera source.
Test malformed kernel files, missing model assets, and stale tuning records.
A fallback must appear in the result metadata and plan explanation.
It must not silently introduce a CPU pixel path.

## Multi-camera and sustained runs

Start with one stream, then test two, four, and six concurrent streams where the target permits them.
Use 1080p robot inputs and a separate six-stream 4K inspection experiment.
The six-stream 4K case preserves the customer scenario. It is a stress workload, not a promised throughput result.
Sweep offered load through sustainable operation and deliberate overload.
Record actual frame rate rather than assuming the illustrative 60 FPS demo target.

Use bounded queues with a declared drop or backpressure policy.
Keep frame IDs and timestamps through the full graph.
Report per-camera latency and completion counts so aggregate throughput cannot conceal starvation.
Measure input-to-inference and input-to-result-subscriber latency separately.
Synchronize clock domains or restrict subtraction to timestamps from the same monotonic clock.

Proposed run tiers:

- Developer check: short complete-inference run after a change.
- Comparison run: at least 10,000 measured frames per candidate across repeated runs.
- Soak run: at least 30 minutes after thermal stabilization, including memory and queue monitoring.
- Release run: repeat the comparison and soak on every platform claimed by the release.

These durations are initial protocol choices, not PRD performance requirements.
Record amendments and their statistical reason before publishing a comparison.
Run cold-start compilation, warm-cache startup, and steady-state execution separately.

## Trace proof and attribution

Annotate frame IDs and stage boundaries so host events, GPU work, and inference can be correlated.
Inspect kernel launches, allocations, transfers, synchronization, idle gaps, and overlap.
Collect short focused traces separately from uninstrumented latency runs.
Keep the original and optimized traces for the same workload.

Absence of an obvious copy call alone does not prove GPU residency on shared-memory systems.
Check the buffer allocator, pointer ownership, CPU access, and any memory migration or staging path.
Document the exact boundary where the GPU-residency claim starts.
Capture uploads may occur before that boundary and must remain visible in the end-to-end report.

Use a known-transfer control to check instrumentation sensitivity.
Use baseline transport with identical kernels to isolate the value of CUDA-backed ROS messages.
Use identical transport with fusion disabled to isolate kernel and memory-planning improvements.

## Required result bundle

Proposed artifact layout:

```text
benchmark-results/<run-id>/
  manifest.json          # revision, environment, models, inputs, hashes
  workload.yaml          # exact operations, semantics, load, queue policy
  plan.json              # groups, implementations, cache status, workspace
  samples.csv            # per-frame events, stream, camera, latency, status
  correctness.json       # tolerances, errors, inference comparisons
  telemetry.csv          # memory, CPU/GPU use, clocks, power, temperature
  summary.json           # metrics, unavailable fields, comparison identity
  report.md              # findings, limitations, decision, reproduction
  logs/                  # preflight, build, execution, failures
  traces/                # focused profiling captures
```

This layout is a harness contract, not an existing output format.
Preserve unsuccessful runs and include reasons for exclusion.
Upload large artifacts to the chosen artifact store and record hashes in the report.
Do not commit raw traces or private camera data to normal Git history.

## Decision after each experiment

A change passes only when the relevant full scenarios produce correct results.
Compare it with the same baseline under the same environment and workload.
Check p50, p95, p99, throughput, allocations, transfers, memory, utilization, and available power metrics.
Use the original success gate in PLAN.md without weakening it or adding hidden thresholds.

If a change regresses a required behavior, reject or repair it before keeping the optimization.
If a benchmark is inconclusive, collect better evidence rather than declaring a win.
Record the bottleneck, chosen change, before/after artifacts, and next experiment.
Then commit the implementation and evidence summary in a reviewable change.

## CI and release feedback

Use lightweight checks for quick feedback, but make hardware scenario results the runtime acceptance evidence.
Run the short end-to-end suite on an available GPU after relevant implementation changes.
Run sustained and target-hardware suites at milestone and release boundaries.
Do not create recurring paid GPU jobs without an agreed budget.

Mark a GPU job as blocked when missing hardware prevents execution. Do not mark it as passed.
A release matrix must show each required target and scenario as passed, failed, blocked, or not applicable.
An unavailable Jetson leaves Jetson validation pending. It does not remove Jetson from the plan.

The first useful implementation checkpoint is E01 with baseline, real inference, correctness output, and a trace.
After that checkpoint, repeat the loop for fusion, E02, and the robot/ROS scenarios.


## Model artifacts and the local reference loop

CPG does not require an LLM, model training, Ollama, or a paid model API.
Use pretrained vision models as real consumers of preprocessing output.
Begin with one small detector. Add one classifier for the inspection workload.
Reuse the detector for concurrent camera streams.
YOLO and RT-DETR remain detector candidates. Select the exact model after checking its license and export path.
Segmentation, depth, and VLM encoders remain later consumer possibilities.

Hugging Face is an optional source for model artifacts and a later demo channel.
An upstream model repository is also acceptable.
Runtime inference should use the pinned local artifacts rather than depend on a hosted endpoint.
Do not download weights until the selected model and artifact license are clear.

The model manifest must record:

- Source URL, immutable revision, artifact checksum, and license.
- Architecture, weight file, class labels, and expected output interpretation.
- Input size, dynamic-shape policy, batch limits, dtype, layout, and channel order.
- Resize/letterbox rules, padding, scaling, mean/std, and coordinate transforms.
- Output decoding and NMS rules for detection, including thresholds.
- Exporter version, export options, and target engine build settings.
- Reference inference environment, precision, and output tolerances.

Check whether the model already includes preprocessing to avoid applying it twice.
Use identical weights and postprocessing across baseline and CPG comparisons.
Check the exported model against the reference model before attributing differences to CPG.
Keep engine conversion errors separate from preprocessing errors.

Before a CUDA implementation exists, run this local development scenario:

```text
recorded/synthetic image → independent reference preprocessing
                         → real PyTorch model on CPU or MPS
                         → tensor/output checks → saved result bundle
```

Record the actual backend and any fallback in the bundle.
MPS availability does not prove that every operator in a selected model works.
A passing local scenario validates fixture semantics and harness behavior.
It does not pass E01's TensorRT acceptance or the CUDA/ROS hardware scenarios.
Reuse the exact artifacts and checks when moving the experiment to NVIDIA hardware.

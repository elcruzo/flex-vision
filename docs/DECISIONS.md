# Decision log

Each decision needs a status, reason, and condition for reconsideration.
Proposed implementation details can change without changing the product goal.

## D001 — Optimize the complete pipeline

Status: accepted direction.

CPG targets the camera-to-model boundary.
The original cuda-conv work provides a possible stencil component.
A faster isolated convolution is insufficient evidence of product value.

Reconsider scope if two realistic workloads fail the performance gate after targeted experiments.

## D002 — Keep the current repository name

Status: accepted for initialization.

Use `elcruzo/flex-vision` as the repository.
Use CUDA Preprocess Graph as the current product direction.
Treat `cpg`, `cuda-preprocess-graph`, and `cuda_preprocess` as proposed distribution names.

Check name availability and maintainer preference before publication.
Do not rename the remote as part of implementation.

## D003 — Define a graph before adding many operators

Status: accepted direction, implementation provisional.

Record operations before execution.
Start with a small planner and one complete detector workload.
Add complexity only when measurements or required semantics justify it.

Reconsider planner structure after the detector and inspection comparisons.

## D004 — Defer backend and platform claims

Status: pending experiments.

Generated CUDA, internal kernels, and vendor libraries are candidate implementations.
The requested targets include CUDA 13, ROS 2 Lyrical, Orin, Thor, and desktop GPUs.
This document does not claim that all requested versions work together.

Check current primary documentation and test actual combinations before declaring support.
Record the source URL, date checked, installed version, and experiment result for version-sensitive decisions.

## D005 — Preserve evidence and plain language

Status: accepted workflow.

Use small commits with relevant validation.
Use the bundled ASD-STE100 skill for clear technical prose.
Keep required technical terms and preserve uncertainty.
The skill does not certify compliance with the official STE dictionary.

## D006 — Choose a license before reuse or distribution

Status: pending.

The intended project is open source.
Select the license after auditing cuda-conv and candidate dependencies.
The writing skill retains its upstream MIT license independently.
Do not infer a project-wide license from that bundled dependency.


## D007 — Preserve detailed requirements separately from delivery order

Status: accepted correction to the initial documentation.

The first plan compressed several concrete deliverables into broad milestone language.
REQUIREMENTS.md now records the full operator matrix, interfaces, transport contract, tuning records, demo, research questions, and distribution priorities.
Autotuning remains required. The full robot benchmark remains required even when two other workloads pass the early gate.
Fewer copies or allocations can satisfy the original success criterion without an added system-benefit threshold.

Implementation details can change with evidence. Required outcomes need an explicit scope decision before removal.
External release claims and issue status remain unverified research leads until checked against current sources.


## D008 — Validate through complete inference and target hardware

Status: accepted testing direction.

The user requires the full development feedback loop, not unit tests as the main completion signal.
TESTING.md defines scenarios, real model consumers, traces, sustained runs, faults, and result bundles.
HARDWARE.md separates cloud GPU evidence from Jetson camera and power evidence.
OPTIONS.md retains alternatives without copying the source brief verbatim.

This documentation change does not provision hardware.
Paid setup needs account access and an agreed spending limit.


## D009 — Use the Mac for reference inference before CUDA experiments

Status: accepted development direction.

The local Apple M5 has working PyTorch MPS support in a small arithmetic check.
Use CPU/MPS reference inference to develop fixtures and the complete feedback harness.
Select one pretrained detector, then one classifier. No LLM service or training pipeline is required.

Move CUDA-dependent experiments to NVIDIA hardware early.
Local reference success does not satisfy CUDA, TensorRT, or Jetson acceptance.
A production Metal backend is not part of this decision.


## D010 — Start with an explicit local reference slice

Status: implemented for local validation, CUDA acceptance pending.

The frontend records an immutable letterbox/normalize/output graph without execution.
Independent NumPy sampling and PyTorch interpolation implement the documented image contract.
The first full consumer is pretrained SSDLite on CPU, with CPU or MPS preprocessing.
This small consumer exercises the harness without claiming YOLO/TensorRT coverage.

No cuda-conv code is imported. Its MIT kernels remain migration candidates after the source audit.
The first observed hazards concern host-copy defaults, per-channel dispatch, and a strided kernel-timing input.
GPU tests must establish the behavior before migration.


## D011 — Validate configuration through the real consumer

Status: implemented for the initial detector subset.

YAML loading uses a restricted schema, rejects duplicate/unknown fields, and requires explicit normalization scale.
The inspector emits a reference-only plan and does not invent CUDA launch counts.
The detector runner loads the YAML graph and checks the model input contract before inference.
The wider PRD configuration remains required and will expand with operator implementations.


## D012 — Add semantic photograph checks before the NVIDIA step

Status: implemented locally. The subsequent remote experiment is recorded in D013.

Two pinned photographs supplement numerical fixtures, with original and padded variants.
Semantic class and source-box overlap checks prevent empty-output agreement from passing the photograph suite.
The broad annotations are smoke expectations, not a detection accuracy dataset.

The user has no NVIDIA hardware yet and requested setup instructions.
RUNPOD_SETUP.md defines account, SSH, template, budget, and preflight steps without starting paid compute.
The Mac preflight must report blocked rather than pretend MPS supplies CUDA.


## D013 — Validate a rented CUDA host before backend development

Status: hardware smoke and real PyTorch CUDA inference passed. Full G0/G1 acceptance remains pending.

The first session used one RTX 4090 under the user-approved $15 total budget.
It used CUDA 13.0, PyTorch 2.9.1+cu130, TorchVision 0.24.1+cu130, and CuPy 14.2.0.
Pin the framework pair before installing optional dependencies. An unconstrained install selected a different PyTorch release.
A separate environment retained the template's framework and passed `pip check`.

The smoke runner keeps the CUDA backend separate from the production API.
It checks same-stream DLPack pointer identity, compiled CUDA execution, photograph tensors, dense detector outputs, and semantic detections.
It deliberately uploads fixtures and downloads validation evidence.
Do not infer zero-copy camera transport or optimized pipeline performance from this experiment.

PyTorch 2.9.1 streams lack the CUDA stream protocol required by CuPy's newer `Stream.from_external` API.
Use CuPy's deprecated `ExternalStream` compatibility API in this pinned experiment, while keeping its Torch owner alive.
Recheck this choice when updating the framework pair.

Export and verify artifacts before terminating temporary compute.
No persistent volume was created. The Pod was terminated and the account list returned no Pods.
See [experiment 004](experiments/004-cuda-host.md) for evidence and remaining checks.


## D014 — Start CUDA execution with a synchronous fused detector kernel

Status: implemented and validated on one RTX 4090. Full G1 remains incomplete.

The first CUDA backend fuses the fixed detector graph and returns a new CuPy output.
Calls reject CPU input and synchronize the current CuPy stream before returning.
This initial restriction preserves imported input ownership until completion and avoids hidden output reuse.
It does not replace the required asynchronous runtime or reusable-buffer work.

Double-precision sampling coordinates and disabled multiply/add contraction match the independent oracle before FP32 interpolation.
Keep these correctness choices until measured evidence supports a change.
The kernel supports byte-strided CuPy uint8 input and CUDA object DLPack input.
CUDA Array Interface-only objects remain unsupported.

The trace showed one kernel and no memory-copy events in each preprocessing range.
The full detector passed on all four pinned photograph cases.
No latency comparison or TensorRT result exists yet. See [experiment 005](experiments/005-fused-cuda.md).

## D015 — Validate a fixed TensorRT network before general integration

Status: the FP32 SSDLite hybrid consumer passed on an L4.

TensorRT executes trained normalization, backbone, and dense heads. TorchVision CUDA retains box decoding and NMS.
The consumer binds the existing CuPy input pointer and retains buffers through synchronous completion.
This experiment does not establish dynamic shapes, FP16 TensorRT, portable engine caching, or a production adapter.

Use TensorRT 10.13.3.9.post1 with the recorded runtime pin for this environment. The original package failed dependency installation.
The next experiment must address the observed default-stream warning and compare equivalent warm pipeline latency.
See [experiment 006](experiments/006-tensorrt.md). No performance advantage is claimed.

## D016 — Separate preprocessing gains from complete detector gains

Status: first matched serial replay measured on one L4.

CPG reduced the median preprocessing stream interval by about 42% against the straightforward PyTorch baseline.
Complete host latency fell by about 6.7% in the pooled samples. Decode/NMS dominated this hybrid detector.
The full trace contained small host transfers, despite the copy-free preprocessing-to-dense-network boundary.
Keep these two residency scopes separate in documentation.

Identical image tensors still produced different downstream stage timings between candidate runs.
Use identical-input controls and inspect allocation/dispatch behavior before attributing those changes to fusion.
Do not expand performance claims or declare the two-workload success gate complete from this single experiment.
See [experiment 007](experiments/007-latency.md) for raw-sample hashes, scope, and limitations.

## D017 — Treat downstream memory state as part of the graph experiment

Status: six attribution controls completed on one L4.

The network difference persisted with identical input pointers and reused dense outputs.
It nearly disappeared when both labels skipped preprocessing, or when both paths performed a common memory write before inference.
This supports a preceding-memory-state effect, consistent with cache behavior. It does not prove a specific cache mechanism.
Keep the scratch-memory control out of production. Keep operator and complete-graph measurements distinct.
See [experiment 008](experiments/008-controls.md).

## D018 — Verify independent expiration before another rental

Status: required after an observed shutdown-control failure.

A Mac-side long-sleep timer failed during a long pause, leaving a Pod running idle.
The image also lacked the executable and credential required by the attempted Pod-side timer.
Do not rely on either unverified mechanism for future rentals.
Verify provider-side expiration or an independent termination mechanism, including readback and an actual cleanup test.
Keep total spend accounting across sessions and include unposted reserves.

## D019 — Establish the full inspection reference before adding CUDA operators

Status: CPU and MPS preprocessing passed through a real pretrained classifier on three fixtures, including 4K input.

Use a fixed Gaussian → sharpen → clamp → crop → resize → normalize contract for the next workload.
Filtering precedes ROI extraction. Preserve global border semantics and FP32 intermediates without uint8 quantization.
Keep independent NumPy accumulation and PyTorch grouped-convolution references outside the public graph API for now.

MobileNetV3 Small with pinned ImageNet weights provides an actual inference consumer and a predefined cat-class smoke check.
The photograph does not establish industrial defect accuracy, and the custom ROI transform is not canonical ImageNet evaluation.
The absent-class negative control must fail. Pixel agreement alone is insufficient.
See [experiment 009](experiments/009-inspection-reference.md). GPU performance and the two-workload gate remain pending.

## D020 — Do not use Runpod expiration flags as a shutdown guarantee

Status: native expiration is unavailable in the reviewed client/backend path.

Runpod removed `--stop-after` and `--terminate-after` because the backend accepted but ignored their deadlines.
The REST API has no corresponding scheduling field or deadline readback.
See [the merged Runpod change](https://github.com/runpod/runpodctl/pull/330), checked on 2026-10-05.
An independent watchdog still needs implementation and an actual termination test before another experiment rental.
A hosted job timeout alone does not terminate a cloud Pod.
Continue local harness preparation while this requirement remains open.

## D021 — Create and expire experimental Pods from an independent runner

Status: implemented, live proof pending because the hosted job did not start.

Use a manually dispatched GitHub Actions job to create and watch one disposable Pod.
Start the deadline before provisioning. Bind deletion to the exact creation receipt and verify provider absence.
Reject delayed dispatch, overlapping rentals, unavailable capacity, and an increased GPU rate.
Keep account credentials out of the GPU container and use encrypted Actions secret storage.

The runner remains subject to infrastructure failures and API outages. This is not a provider-enforced cost cap.
A passing mock test or a committed workflow does not satisfy D018. Require the two-minute live test first.
See [experiment 010](experiments/010-lease-preparation.md).

## D022 — Make GitHub Actions optional

The user requested development without a GitHub Actions dependency.
Use a disposable Runpod CPU controller to create and expire the GPU lease.
It removes its temporary secret and deletes itself after GPU cleanup.
Keep Actions as an optional independent host for the same lease code. No CPG runtime or test requires Actions.

The controller uses a temporary encrypted Runpod account secret. It never sends that credential to the GPU workload.
Only controller code runs on the CPU host. Account credentials are broader than a Pod-scoped token.
Record both CPU and GPU charges, and verify cleanup through provider reads.
The independent termination requirement remains. A failed boot or provider outage still needs manual intervention.
The direct controller cleanup test passed. See [experiment 011](experiments/011-direct-lease.md) and [the lease procedure](GPU_LEASE.md).

## D023 — Test crop-aware execution before adding inspection operators

The 4K baseline in experiment 012 spends most of its measured time in preprocessing.
Test a restricted work rectangle with a three-pixel halo before adding public operators.
Keep the filter stages and numerical tolerances unchanged.
Local classifier checks passed in [experiment 013](experiments/013-inspection-roi.md).
Experiment 014 supplied CUDA correctness, matched latency, and trace evidence.
The 4K candidate reduced preprocessing p50 and p99 by approximately 42%, with unchanged launches and no measured copies.
Small-image preprocessing medians regressed approximately 5%. Retain the rewrite as a selective candidate, not a universal default.
Public runtime integration and representative industrial evaluation remain pending.

## D024 — Expose explicit inspection plans before automatic selection

Add an experimental fixed `InspectionPipeline` with immutable plans and prepared PyTorch execution.
Keep full-frame execution as the default. Require an explicit ROI strategy until tuning supports a general selection policy.
The measured three fixture sizes are insufficient to infer a portable size threshold.
Preserve the independent references and compare the integrated runtime through the existing classifier harness.
Local checks passed. Experiment 016 also passed integrated CUDA validation, matched ROI performance, and transfer traces on an L4.
Keep strategy selection explicit. Small-image preprocessing remains slower, and cross-stream inference acceptance remains pending.
See [the runtime contract](INSPECTION_RUNTIME.md).

## D025 — Separate stream ordering from storage lifetime

Use CUDA events to order preparation, producer input, preprocessing, and downstream inference.
Use PyTorch storage tracking when a tensor is used on another stream.
Avoid redundant constant tracking on their allocation stream. Input allocation streams are not inferred.
Expose `submit` and a completion handle, while retaining the plain-tensor same-stream call.
The caller remains responsible for input-buffer writes and model initialization dependencies.
Experiment 017 passed 24 cross-stream inference cases. Its initial timing guard failed and the revised run passed.
Keep both results and do not attribute variable small-image tails solely to the implementation change.
Sustained concurrent throughput and memory bounds remain unmeasured.

## D026 — Reuse camera streams in sustained comparisons

Experiment 018 measured the inspection path under bounded resident-input arrivals through real classifier inference.
The initial harness introduced new CUDA streams between blocks and showed increasing allocation baselines.
No measured block grew above its own post-warmup allocation after drain.
The revised harness reuses four streams across strategies and repeats, without clearing allocator caches.
Post-drain allocations remained stable across the revised repeats, with a 63.48% median paired overload throughput gain for ROI execution.
Both runs retain numerical checks, raw frames, drop records, and per-camera results.

Use persistent streams for fixed camera lanes. Do not interpret a library's retained stream resources as a per-frame leak without attribution.
The comparison used different hosts, so between-run timing changes do not isolate stream-lifetime effects.
This short test includes GPU validation overhead. It does not establish uninstrumented deployment throughput or the required long soak.
The public runtime still needs reusable workspace, batch support, and the broader graph contracts.
Complete the FP16 YOLO/TensorRT detector path next, preserving the full remaining requirements.

## D027 — Complete thermal conditioning before counting soak time

Experiment 019 passed a continuous 1,800-second inspection run after the declared thermal gate passed.
The completed L4 run checked all 338,155 completed frames. All three declared allocator growth checks measured zero.
The offered overload schedule dropped 21.72% of arrivals. Keep those drops and the measured 2.86% tail drift visible.
This establishes stability for the fixed resident inspection workload, not a general memory bound or release-wide acceptance.

The first rental could not finish conditioning and the soak within its remaining deadline. The second expired before experiment submission.
Use automatic SSH submission, verified export, and exact GPU termination under the independent controller for subsequent prepared rentals.
The local coordinator still depends on an awake Mac for submission and export. The cloud controller bounds the lease independently.
The third lease completed and exported successfully during a chat-session pause, then verified all cleanup without another turn.

## D028 — Match model preparation before comparing FP16 consumers

The first FP16 YOLO TensorRT experiment failed its dense-output comparison before timing.
PyTorch retained batch normalization. ONNX export folded that normalization into FP16 convolution weights.
The pinned Ultralytics exporter explicitly fuses the FP32 model before converting it to FP16.
Use that preparation for both the PyTorch reference and TensorRT export.
Retain the original failure. Keep the original numerical limits for the rerun.
Six local photo cases pass the corrected preparation, including fused/unfused FP32 and FP16 comparisons.
GPU acceptance remains dependent on the complete rerun, not on those local results.
See [the fixed YOLO protocol](YOLO_VALIDATION.md).

## D029 — Isolate preprocessing equivalence from cross-engine background boxes

The corrected YOLO model passed both person fixtures through every candidate.
The cat fixture retained 12 background-coordinate mismatches between FP16 PyTorch and TensorRT.
All affected proposals scored below 0.000001. The actual detection scores were equal and the box IoU was 0.99801.

Use the same TensorRT consumer for independent NumPy input and every preprocessing candidate.
Retain the original unmasked dense-output limits for this comparison, which isolates preprocessing changes.
For cross-engine checks, retain all score checks and apply box limits at a 0.01 score floor in either engine.
Keep the original failures, report excluded background-coordinate violations, and preserve all final detection checks.
This revises the experimental acceptance policy. It does not establish general model accuracy or equality between FP16 engines.
See [the revised protocol](YOLO_VALIDATION.md).

## D030 — Treat background box drift consistently across preprocessing paths

Revision 2 failed the PyTorch baseline after CPG passed the same padded person input.
The baseline tensor satisfied the FP16 tolerance. A proposal scoring about 0.00014 changed its width by seven pixels.
A universal background-coordinate limit therefore confounds preprocessing equivalence with FP16 model sensitivity.

Apply the explicit 0.01 score floor to box comparisons for every path.
Retain all background errors and original-limit violations. Tighten same-engine dense class-score tolerance from 0.03 to 0.003, including relative tolerance.
Preserve tensor, final detection, finiteness, probability, and box-validity checks.
This is acceptance revision 3, not a retroactive pass for previous experiments.
The next comparison also uses a two-pass planar CV-CUDA candidate to avoid an unnecessary layout pass.

## D031 — Keep candidate equivalence stricter than approximate framework interpolation

The original person fixture produced exact CPG tensor and TensorRT outputs.
The PyTorch interpolation baseline differed by at most one FP16 step at 2,453 tensor elements.
Its largest background class-score change was 0.01343, while the final detection matched at IoU 0.99944.

Require bitwise FP16 tensor equality for CPG, unmasked dense box checks, and 0.003 dense score tolerances.
For the library baselines, restore the original 0.03 score tolerances and preserve confidence-aware box checks.
Keep every approximation, failed prior policy, and task-level detection check visible.
Collect every numerical fixture before rejecting the experiment. Do not benchmark a failed run.
This acceptance revision validates the fixed CPG contract and bounds the baselines' approximate interpolation behavior separately.

## D032 — Measure immutable metadata caching after the fixed YOLO comparison

The first completed YOLO comparison passed correctness but CPG preprocessing p50 was 8.4–8.5% slower than PyTorch.
CPG rebuilt its inspection dictionary and scalar launch arguments on each call.
Cache immutable metadata by the frozen pipeline and input shape, with at most 64 entries.
Never cache arrays, strides, imported owners, or streams. Preserve allocation and synchronization behavior.
The hypothesis requires another identical GPU comparison. It is not a measured speed claim yet.
Also preserve the profiler target after capture ends so the trace report and SQLite export can complete.
See [experiment 020](experiments/020-fp16-yolo.md).

The follow-up passed unchanged numerical checks and measured 24.77% lower preprocessing p50 versus CV-CUDA.
Its complete host p50 improvement was 2.19–3.83%, depending on baseline.
The trace recorded small metadata transfers but no frame-sized host transfer in completed ranges.
Keep repeatability and sustained detector acceptance open. Do not infer broader support from this fixed workload.

## D033 — Warm the complete checked load path before allocator baselines

The detector comparison repeated its scoped preprocessing improvement on another RTX 4090 rental.
The initial load experiment passed all 21,591 completed-frame checks but failed CV-CUDA's retained CuPy pool guard.
The retained pool grew by one output allocation, while live used bytes remained unchanged.
Warmup omitted the GPU correctness path used during every measured frame.
Exercise that complete checked path during all 100 warmup frames. Keep the original 1 MiB growth allowance and preserve the failed attempt.
This is a first-use allocation hypothesis, not evidence that the failure is harmless or fixed.
See [experiment 021](experiments/021-detector-repeatability.md).

## D034 — Separate main-stream completion from vendor resource release

Checked warmup removed the first control-block pool increase, but the CV-CUDA overload block still added one output allocation late in measurement.
Pinned source shows resource holds released through auxiliary callbacks. Main-stream synchronization does not drain that auxiliary stream.
Test an explicit auxiliary drain only as a separate diagnostic control. Keep the normal comparison and 1 MiB guard unchanged.
Collect all load blocks before failing acceptance so one baseline failure cannot hide later candidate evidence.
A complete failed matrix remains failed, even when its raw records are independently verified.
See the [investigation protocol](SUSTAINED_YOLO.md#saturation-resource-lifetime-investigation).

The complete-matrix run passed all normal allocator guards and 187,279 completed-frame checks.
The auxiliary-drain diagnostic also passed, but ordinary CV-CUDA had zero growth in the same rental.
This control therefore does not isolate the earlier resource-lifetime failure. Keep that explanation provisional.
The lower preprocessing median repeated, while sustained throughput and tails varied. Do not claim a repeatable throughput gain.

## D035 — Prepare the detector soak before another rental

Use one continuous 30-minute CPG run after measured thermal conditioning.
Reuse the existing detector controls, inference endpoint, GPU checks, and bounded admission policy.
Retain five-minute latency summaries, per-frame rows, telemetry, and both post-drain and sampled allocator growth.
This is a stability gate. It does not establish comparative throughput, concurrent CUDA streams, or live-camera behavior.
Pending cloud charges leave insufficient unreserved room for another guarded lease under the existing $15 authorization.
Prepare and check the harness locally. Mark GPU validation pending until a budget-safe rental can run it.

## D036 — Caller-owned output before asynchronous reuse

Add an optional synchronous CuPy `out` destination to the fixed detector backend.
Keep the existing owned-output default and numerical kernel unchanged.
Validate destination metadata and conservative input aliasing before dispatch.
The caller owns consumer-completion ordering and output lifetime.
This step isolates output allocation from asynchronous execution and future workspace scheduling.
Local metadata checks passed. Real TensorRT reuse validation is prepared and remains pending.
Do not change the frozen soak strategy or claim a performance gain from this implementation alone.

## D037 — Accept the fixed detector soak and synchronous FP16 output reuse

The 30-minute RTX 4090 soak completed all 432,000 offered frames with passing checks and zero drops.
All declared post-drain and sampled allocator growth counters were zero.
The local audit reproduced the remote report from retained compressed records and telemetry.
Caller-owned FP16 output passed all six TensorRT fixtures, including retention and invalid-destination checks.
Accept these fixed scenarios. Keep FP32 output, asynchronous consumers, broader platforms, and release acceptance open.
Measure output reuse before selecting it as a performance strategy. Do not infer its benefit from avoiding a call to `empty`.
The first dispatch failed before CUDA setup. Preserve that failure and the corrected deadline handoff.
Lifetime-bounded reconciliation allowed the experiment within the existing $15 budget, while retaining delayed-charge headroom.

## D038 — Measure synchronous output reuse before asynchronous changes

Compare fresh owned output and one caller-owned FP16 slot through matched TensorRT inference.
Use alternating order, 40,000 checked samples, pooled and per-block statistics, and separate allocation-hook observations.
Keep the slot alive for both candidates to preserve common memory state.
Require the fresh-output hook as a positive control. Distinguish pool requests from actual device allocations.
Reject adoption if complete-host p99 regresses more than 5% on either fixture.
A neutral or slower result remains valid evidence. Do not require a speedup to accept correct optional output semantics.

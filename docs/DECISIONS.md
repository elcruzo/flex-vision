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

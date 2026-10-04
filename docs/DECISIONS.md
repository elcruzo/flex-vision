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

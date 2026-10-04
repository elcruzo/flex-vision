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

The repository contains planning documents and the writing skill.
No CUDA runtime or benchmark result exists yet.
Start implementation with the G0 audit and G1 detector contract.

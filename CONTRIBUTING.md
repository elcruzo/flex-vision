# Contributing

## Start from a goal

Read [the delivery plan](docs/PLAN.md).
Choose a pending goal and state the evidence your change should produce.
Check current primary documentation before using version-sensitive CUDA, ROS, or interoperability APIs.
Record important decisions in [the decision log](docs/DECISIONS.md).

## Keep changes reviewable

Use one purpose per commit.
Separate source import, API behavior, optimization, and benchmark results when each needs independent review.
Include relevant correctness checks with implementation changes.
Record which checks ran and which need GPU hardware.
Never present a skipped hardware check as a pass.

Do not change numerical behavior to obtain a faster benchmark without documenting the new contract.
Do not introduce CPU payload copies into the GPU path.
Do not publish performance claims without reproducible GPU comparisons.

## Write clear documentation

Apply [the bundled ASD-STE100 skill](.agents/skills/asd-ste100/SKILL.md).
Use short sentences, active voice, and explicit conditions.
Keep technical names and uncertainty when they carry meaning.
Treat vocabulary advice as guidance, not certified STE compliance.

Run the structural check from the repository root:

```bash
python3 .agents/skills/asd-ste100/scripts/ste-lint.py README.md CONTRIBUTING.md AGENTS.md docs/*.md --disable synonym-rotation
```

Review terminology consistency manually.
The synonym check confuses CUDA launches with project starts and fixed plans with correctness.
Review advisory findings manually. A clean linter result does not prove technical accuracy.
The skill is available to repository agents on their next turn.
[UPSTREAM.md](.agents/skills/asd-ste100/UPSTREAM.md) records its source revision and license.

## Validate according to scope

For documentation, check links, examples, status labels, and consistency.
For runtime changes, add independent numerical references and ownership tests.
For performance changes, follow [the benchmark protocol](docs/BENCHMARKS.md).
For distribution changes, test a clean installation on each claimed platform.

This repository currently has no runtime, package build, or GPU test suite.
Commands in design documents describe planned interfaces unless explicitly marked as available.

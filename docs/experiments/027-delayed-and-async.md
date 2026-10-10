# Experiment 027: pending-reader correctness and matched serial submission

## Outcome

All 48 delayed-consumer TensorRT executions passed on RTX 4090.
All 40,000 measured frames passed exact tensor, dense-output, validity, and detection checks.
Submission reduced complete-host p50 by 9.38–9.42% versus synchronous execution in this run.
Complete-host p99 improved by 8.99–10.37%, within the frozen 5% regression guard.
Keep the synchronous default. Independent repeatability, async traces, allocator observations, and concurrency remain pending.

| Fixture | Sync host p50 / p99, ms | Submit host p50 / p99, ms |
| --- | --- | --- |
| Person, 1080p | 0.859294 / 0.897535 | 0.778708 / 0.804467 |
| Cat, 1080p | 0.858818 / 0.884005 | 0.777916 / 0.804547 |

These measurements include preprocessing, TensorRT, CUDA NMS, complete drainage, and handle cleanup.
They exclude acquisition, upload, and correctness checks after the measured boundary.
Every frame drains before the next begins. No overlap or multi-camera throughput claim follows.
See [the frozen protocol](../ASYNC_BENCHMARK.md) for timing and instrumentation boundaries.

## Correctness and environment

The delayed harness observed a genuinely pending consumer after preprocessing closed in every cycle.
It rejected busy-context reuse and preserved correct output across subsequent allocations.
Six fixtures, CuPy/PyTorch inputs, and four cycles each produced 48 checks.
The finite artificial delay applies only to this correctness gate, not the latency comparison.
PendingConsumer remains a test harness, not production inference integration.

Source: `a3ccdbcedccbc064fecc94bbac0d08789c7281f8`, clean checkout.
Engine SHA-256: `7af545f08980aae33f581abd43ddaa21cfb3b76a69482d4d5b3c2cba3d5f23ab`.
Hardware: RTX 4090, driver 580.126.20, 24,564 MiB memory.
Pinned packages and environment remain in [the compact bundle](data/027/preflight.json).
The six-fixture controls used numerical-only validation, not the 60,000-sample baseline timing protocol.
The independent local NumPy and CPU NMS audit passed all 18 candidate/fixture combinations.
The delayed coverage audit verified 48 records. It does not independently recompute delayed tensors.
Local and remote async sample audits matched exactly across all 40,000 ordered rows.

## Dispatch correction and evidence

The optional comparison initially lacked CPG_RESULTS in its parent dispatch shell.
The child script's export cannot propagate into that parent.
Before the child returned, the wrapper received the missing export. Measured repository source stayed unchanged.
Both wrapper hashes and the change remain in [the patch record](data/027/dispatch-wrapper-patch.json).
The final experiment exited zero. No ambiguous SSH submission was repeated.

The coordinator exported 40 hash-verified files at 04:54:36 UTC on October 10, 2026.
Compressed raw samples, numerical reports, independent audits, and environment records remain in data/027.
Engines, weights, and dense arrays remain in the ignored local bundle.
No async-specific trace or peak-memory observation was captured in this experiment.

## Cleanup and budget

GPU z7re87uacyl1by was absent at 04:54:37 UTC.
Controller 8nv3n6zv03d514 and secret o82dkpjhvc cleanup passed at 04:54:58 UTC.
A fresh inventory returned no Pods or secrets. See [cleanup evidence](data/027/cleanup.json).
Retain the $0.65 reservation until billing reconciliation.
The cumulative conservative bound remains $13.945168122393484 within the original $15 authorization.

## Product consequence

This result advances ownership correctness and measured camera-to-model overhead reduction.
It supports further async development but does not select a production strategy.
Next, capture async transfer/ordering and allocator evidence, then repeat the matched comparison.
Bounded concurrent scheduling, reusable slots, configurable stencils, tuning, ROS, and Jetson remain required work.

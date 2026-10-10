# Matched asynchronous detector comparison

Status: first NVIDIA comparison passed. Independent repeats and broader async acceptance remain pending.
This experiment advances G1 ownership and G3 pipeline overhead in PLAN.md.
The product goal remains camera frame to inference-ready tensor, with measured reductions in overhead.

## Question and prerequisites

Does removing intermediate host waits improve complete detector service time without changing tensors or worsening p99?
The first comparison is serial. It isolates host-wait placement before investigating multiple frames in flight.
It cannot establish overlap or multi-camera throughput.

Require a passing delayed-consumer report from the same clean revision and TensorRT engine.
The runner audits all 48 delayed executions before measurement.
It rejects mismatched source or engine identity. Experiment 026 supplies no passing report.
Regenerate the six-fixture controls using the established fixed YOLO protocol.
Do not replace independent tensor and model controls with outputs from the candidate itself.

## Frozen strategies

Both strategies use resident BGR8 input, identical numerical semantics, fresh output allocation, and the same TensorRT context.
Both retain two native streams throughout the run. Input upload finishes before measurement.
Both use explicit dependencies between preprocessing and the consumer.

- `sync`: call the existing synchronous pipeline, enqueue TensorRT, wait for inference, then run CUDA NMS.
- `submit`: submit preprocessing, queue consumer readiness, enqueue TensorRT and CUDA NMS, then wait for complete results.

The submission and inference handles retain their owners until successful completion.
Each frame drains before the next begins. Caller-owned slots and consumer recycling are excluded.
The experimental PendingConsumer remains a harness, not production TensorRT integration.
No runtime default changes as part of this preparation.

## Samples and boundaries

Use the existing person and cat 1080p fixtures.
Warm each strategy for 100 fully checked frames per fixture.
Collect ten blocks of 1,000 frames per strategy per fixture: 40,000 measured frames in total.
Alternate strategy order between blocks. Keep streams, engine, and allocator pools unchanged.
Do not remove outliers or discard failed frames.

Record these boundaries:

| Field | Meaning |
| --- | --- |
| dispatch_host_ms | Host time through preprocessing dispatch and its timing marker. Sync includes its preprocessing wait. Submit does not promise completed preprocessing. |
| preprocess_gpu_ms | CUDA event interval around preprocessing on its stream. It can include host scheduling gaps. |
| complete_gpu_ms | Ordered CUDA interval through preprocessing, inference, and NMS. It includes scheduling gaps, not just kernel execution. |
| complete_host_ms | Host entry through complete consumer drain and explicit handle cleanup, before correctness checks. |

Do not label dispatch time as completed preprocessing latency.
Every frame receives exact tensor, dense-output, validity, and detection checks after the measured boundary.
Those checks can condition the next sample. This is an instrumented development comparison.
The report and raw rows preserve that scope. Camera arrival, transport, uploads, and dataset accuracy remain outside it.

## Decision rule

Reject strategy selection if complete-host p99 regresses more than 5% on either fixture.
Report p50, p95, p99, and per-block distributions even when the rule fails.
A passing guard alone does not establish a speedup. Report neutral or slower results explicitly.
Repeat a favorable comparison on an independent rental before a default-selection decision.
Require a separate async-specific Nsight trace and allocator observations before broader async acceptance.
Power, bandwidth, utilization, peak memory, and comparative throughput remain unmeasured by this harness.

## Hardware commands

Run these only inside an independently bounded NVIDIA session with verified export and reconciled budget.
Use the pinned environment from YOLO_VALIDATION.md.

```bash
.venv-gpu/bin/python scripts/yolo_delayed_consumer.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --output "$CPG_RESULTS/delayed-consumer.json"
.venv-gpu/bin/python scripts/yolo_async_benchmark.py \
  --engine "$CPG_RESULTS/measured/yolo.engine" --controls "$CPG_RESULTS/measured" \
  --delayed-report "$CPG_RESULTS/delayed-consumer.json" --output "$CPG_RESULTS/async-comparison"
.venv-gpu/bin/python scripts/summarize_yolo_async.py "$CPG_RESULTS/async-comparison" \
  --output "$CPG_RESULTS/async-comparison/audit.json"
```

The audit also accepts compressed samples.csv.gz.
It rejects wrong protocol labels, missing source or gate identity, missing frames, duplicate rows, incorrect results, and nonfinite timings.
Local generated evidence checks validate these rejection rules. They do not establish hardware correctness or performance.

## Follow-through to the product

Use the measured result to decide whether submission reduces overhead before adding managed reusable slots.
Then evaluate bounded concurrent execution, configurable stencils, general planning, and measured tuning.
ROS buffer transport and Jetson camera-to-TensorRT acceptance remain required for the flagship product.
Supporting export and rental controls serve those measurements. They do not replace product milestones.


Experiment 027 passed 48 delayed-consumer checks and 40,000 matched serial inference checks on RTX 4090.
Complete-host p50 improved 9.38–9.42%, with p99 improved 8.99–10.37% in this run.
See [experiment 027](experiments/027-delayed-and-async.md).
Keep the synchronous default. Async traces, allocator observations, independent repeats, and concurrency remain pending.

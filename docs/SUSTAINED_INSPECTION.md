# Sustained inspection experiment

Status: the short CUDA run passed correctness and performance checks. See [experiment 018](experiments/018-sustained-inspection.md).
The revised fixed-stream run showed stable post-drain allocations across repeats.
Experiment 018 is the developer tier. It does not satisfy the 10,000-frame comparison or 30-minute soak tiers.
The separate [experiment 019](experiments/019-inspection-soak.md) passed the 30-minute inspection soak on an L4.
It retains the initial failed thermal gate and idle lease, plus all completed-run frames and drop records.
Other platforms, workloads, and broader acceptance remain pending.

## Question

Does ROI preprocessing improve completed throughput under bounded concurrent load?
Does the runtime preserve numerical results without increasing live allocations after each run?

Use the pinned 4K Chelsea fixture and MobileNetV3 classifier from experiment 017.
Alternate the original and mirrored image. Compare every accepted frame against independent NumPy preprocessing and CPU classifier results.
This is a synthetic resident-input experiment. It does not represent live cameras, ROS transport, or industrial defect accuracy.

## Protocol

Run from a clean commit on the guarded CUDA host:

```bash
python scripts/inspection_streams.py --output benchmark-results/018-streams
python scripts/inspection_sustained.py --output benchmark-results/018-sustained
```

Use the pinned model weights already required by the classifier harness.
Record the environment, GPU driver, source revision, and artifact hashes with the experiment.
Export the reports and raw CSV files before terminating the rental.
Recompute counts, latency, throughput, and per-camera results from the exported CSV files:

```bash
python scripts/summarize_sustained.py benchmark-results/018-sustained --output benchmark-results/018-summary.json
```

This check rejects missing or duplicate frames, inconsistent timings, and a report that fails correctness.
It preserves a failed performance verdict when correctness passes.
Reconcile provider charges against the existing $15 total budget before provisioning.

The default experiment uses these settings:

| Setting | Value |
| --- | --- |
| Input | Resident 4K BGR uint8, batch one |
| Control load | One camera, 30 arrivals per second |
| Stress load | Four cameras, 60 arrivals per second each |
| Duration | 20 seconds per strategy per repeat |
| Repeats | Four, with alternating baseline/ROI order |
| Pending limit | Two frames per camera |
| Streams | One processing/inference stream per camera |
| Warmup | Ten frames per stream, outside measurement |

Reuse the same four-stream pool across strategies and repeats. The one-camera workload uses its first stream.
The first GPU run created streams per block and showed rising post-drain allocations across blocks.
Retain that result. The revised protocol changes stream lifetime only, with the same loads, thresholds, and checks.
PyTorch documents retained [cuBLAS workspaces per handle and stream](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cublas-workspaces).
Stream reuse tests that explanation without clearing caches or changing the production runtime.

Arrivals are evenly staggered across cameras. Rotate the first camera between repeats.
Drop an arrival when its dispatch slot expires or its camera queue is full.
Do not replay missed arrivals as a burst. Drain accepted frames after the arrival window.
Abort if the drain exceeds 30 seconds. The independent lease still limits rental duration if a CUDA call stalls.

The baseline uses the independent full-frame PyTorch implementation.
The candidate uses the public ROI implementation. Both use the same model and admission policy.
Inputs, model weights, references, and preparation precede measurement.
Each accepted frame runs preprocessing, inference, and GPU correctness checks.
Only completion metadata reaches the CPU during measurement. Download validation flags after the queues drain.

## Acceptance and interpretation

Require all of the following for correctness:

- Every offered frame has exactly one completion or drop record.
- Every completed tensor meets absolute tolerance 0.0002 against the independent reference.
- Every completed logit meets absolute tolerance 0.001 and relative tolerance 0.0001.
- Ordered top-five labels match the reference, which must pass the existing cat semantic check.
- The number of pending frames never exceeds cameras multiplied by queue depth.
- All accepted frames finish within the drain limit.

Record completed throughput, both drop reasons, p50/p95/p99 latency, and each repeat separately.
The performance hypothesis requires at least 20% higher stress-load throughput in the median paired comparison.
Also require no more than 5% worse control-load p99 in the median paired comparison.
Keep failures and per-repeat variability. A passing correctness status does not imply a passing performance hypothesis.

Host latency starts at scheduled arrival and ends when the controller observes completion.
It includes CPU dispatch, queue delay, validation kernels, polling, and contention.
CUDA event latency spans preprocessing and inference, excluding subsequent validation kernels.
Validation still competes with other streams and affects throughput. This experiment does not measure an uninstrumented deployment.
Report throughput with drain time and the count completed within the arrival window.
Do not remove slow samples or calculate latency for dropped frames.

Sample allocated and reserved GPU bytes every 100 milliseconds.
Record allocation baselines, peaks, and allocated bytes after drain for each run.
Flag increasing post-drain allocations across repeats of the same strategy and camera count for investigation.
Do not call a fixed queue depth a hard GPU memory bound. Cached allocator blocks and library workspaces need separate interpretation.
Host metadata and validation flags have fixed maximum sizes derived from the bounded experiment duration.

## Continuous 30-minute soak

Run `python scripts/inspection_sustained.py --soak --output benchmark-results/019-soak` on the guarded CUDA host.
This mode fixes four cameras at 60 arrivals per second each, ROI strategy, and two pending frames per camera.
It retains the same numerical checks, source images, model, and persistent streams.
It does not run a new matched speed comparison.

First, condition the GPU under the same load for five minutes.
Require the last 120 one-second temperature samples to span at most 2 degrees Celsius.
Their successive one-minute medians must differ by at most 1 degree Celsius.
Allow one additional five-minute conditioning interval. Abort if temperature remains unstable.
Then run one uninterrupted 1,800-second arrival window. Do not clear caches or restart the model during measurement.

Keep all 432,000 offered-frame records, including drops, and sample GPU allocation every 100 milliseconds.
Record temperature, SM clock, power, utilization, and device memory once per second.
Require every accepted frame to pass inference checks and finish within the existing drain allowance.
Require allocation growth of at most 1 MiB for each of these checks:

- Allocated bytes after drain compared with the measured start.
- Minimum sampled allocation in the last five minutes compared with the first five minutes.
- Maximum reserved bytes in the last five minutes compared with the first five minutes.

These checks detect growth in this workload. They do not establish a reusable workspace bound.
Report per-minute and per-camera latency, throughput, drops, and thermal behavior without removing slow samples.
Host result metadata accumulates within the fixed experiment size. This harness does not test bounded host memory for indefinite operation.
Use a 60-minute direct-controller lease and terminate it after verified export.

## Limits and next evidence

The shared read-only model and immutable resident sources avoid camera upload costs.
The finite stream test remains the separate test for producer/consumer handoffs and allocator lifetime.
This experiment uses one stream per camera. It does not validate overlapping stages within one camera.
Record unsuccessful runs before changing the schedule, queue depth, or performance threshold.
Extend duration and add real camera arrival traces only after this protocol passes.
Report per-camera completion counts, drops, and latency from the recorded lane identifiers before interpreting aggregate throughput.
An aggregate gain cannot establish fairness or exclude camera starvation.

Event polling follows the [PyTorch 2.9 Event contract](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html).
`query()` reports completion without a host synchronization call.

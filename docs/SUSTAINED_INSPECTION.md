# Sustained inspection experiment

Status: harness prepared. CUDA execution and throughput acceptance remain pending.

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

## Limits and next evidence

The shared read-only model and immutable resident sources avoid camera upload costs.
The finite stream test remains the separate test for producer/consumer handoffs and allocator lifetime.
This experiment uses one stream per camera. It does not validate overlapping stages within one camera.
Record unsuccessful runs before changing the schedule, queue depth, or performance threshold.
Extend duration and add real camera arrival traces only after this protocol passes.

Event polling follows the [PyTorch 2.9 Event contract](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html).
`query()` reports completion without a host synchronization call.

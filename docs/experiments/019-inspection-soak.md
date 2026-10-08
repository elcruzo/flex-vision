# Experiment 019 — Continuous inspection soak

Status: the third lease completed the full continuous soak and passed the declared correctness and allocator checks.
The initial failed thermal gate and the idle second lease remain in this report.

## Protocol

Run measured source `26ed35b` with `scripts/inspection_sustained.py --soak`.
Use four persistent camera streams, resident 4K BGR inputs, ROI preprocessing, and the pinned MobileNetV3 classifier.
Alternate the original and mirrored Chelsea fixture. Check every accepted tensor, logit vector, and ordered top-five result.
Each camera offers 60 frames per second. Each queue holds at most two pending frames.
Drop expired dispatch slots or arrivals to full queues. Preserve every offered frame in the raw CSV.

The [soak protocol](../SUSTAINED_INSPECTION.md#continuous-30-minute-soak) defines thermal stabilization and memory thresholds before measurement.
Condition under the same load for five minutes, with one additional five-minute interval if needed.
Then measure one uninterrupted 1,800-second arrival window, followed by a bounded drain.
This is a candidate stability test, not a new matched speed comparison.

## Initial attempt

GPU `mlrt00axvzkm05` in EUR-IS-2 ran the preflight and all 24 stream-handoff cases successfully.
The controller took approximately five minutes to start. A subsequent session delay consumed additional lease time before setup execution.
The independent deadline remained 2026-10-08 02:53:50 UTC.

Temperature reached approximately 88 degrees Celsius during the first conditioning interval.
The last 120 samples spanned 3 degrees Celsius. Their one-minute median change was 1 degree.
The fixed gate requires a range of at most 2 degrees, so it correctly rejected that interval.
The remaining deadline could not cover a second complete conditioning interval, the 30-minute soak, and verified export.
The second conditioning interval was interrupted. No measured soak began.

All ten completed artifact files matched the remote hashes before termination.
The [conditioning report](data/019/initial/sustained/conditioning-0.json), compressed raw CSV, and telemetry retain the failed gate evidence.
The remote manifest hashes the original CSV bytes. Decompress the committed CSV before comparing its hash.
Both Pods and temporary secret `txilc0n911` were absent after GPU termination.

## Retry

The retry uses the same source, workload, thresholds, and 60-minute independent lease.
Controller `gu2mqlrfefs9y0` uses EU-NL-1. Its requested GPU region is EUR-IS-1.
Reserve a further $0.65 within the original $15 authorization. Do not count this as new budget.
The retry expired during a session pause before SSH submission. No experiment ran.
Subsequent reads found no Pods and no temporary secret `jrdmqti3ir`.
The next attempt uses an automatic local coordinator for SSH submission, verified export, and termination.
Its cloud controller still enforces the independent deadline. The experiment source and thermal thresholds remain unchanged.

## Completed run

GPU `czpobr6j1rnf3t` in EUR-IS-1 used an NVIDIA L4 with driver 580.159.04 and 23,034 MiB reported capacity.
The environment used PyTorch 2.9.1+cu130, TorchVision 0.24.1+cu130, and NumPy 2.2.6.
The preflight and all 24 separate stream-handoff cases passed.
The first five-minute conditioning interval again had a 3-degree temperature range and failed the fixed gate.
The second interval had a 2-degree range and zero median change. It passed without changing the thresholds.

Measurement ran continuously from 08:38:28.628 to 09:08:28.635 UTC on October 8, 2026.
That is 04:38 to 05:08 Eastern. The arrival window was 1,800 seconds, with 0.0069 seconds of drain.
The harness did not restart the model, replace the stream pool, or clear allocator caches during measurement.

| Result | Measured value |
| --- | ---: |
| Offered frames | 432,000 |
| Completed frames | 338,155 |
| Failed completed-frame checks | 0 |
| Dropped frames | 93,845 (21.72%) |
| Expired dispatch slots | 37,843 |
| Full-queue drops | 56,002 |
| Completed throughput, including drain | 187.863 FPS |
| Arrival-to-completion p50 / p95 / p99 | 34.957 / 42.960 / 46.667 ms |
| GPU preprocessing plus inference p50 / p95 / p99 | 20.888 / 31.348 / 36.266 ms |
| Maximum pending frames | 8 |

Every completed tensor met the independent NumPy tolerance. Every classifier logit vector and ordered top-five result met the reference checks.
The offered 240 FPS exceeded measured capacity. Passing this overload soak does not mean four cameras can sustain 60 completed FPS each.
Host latency includes dispatch, queues, GPU checks, and completion polling.
GPU event latency excludes the following validation kernels, although concurrent checks still compete for the GPU.

Per-minute completed throughput ranged from 185.733 to 188.450 FPS.
Per-minute p99 ranged from 44.210 to 47.249 ms.
The last five minutes had 2.86% higher p99 than the first five minutes.
This is a measured drift observation, not a new baseline speed comparison or an undeclared pass threshold.

| Camera | Completed / offered | Arrival-to-completion p99 |
| --- | ---: | ---: |
| 0 | 84,647 / 108,000 | 46.616 ms |
| 1 | 84,432 / 108,000 | 46.643 ms |
| 2 | 84,657 / 108,000 | 46.702 ms |
| 3 | 84,419 / 108,000 | 46.716 ms |

No camera starved in this fixed staggered schedule. This does not establish fairness for arbitrary arrivals.

## Memory and thermal evidence

Allocated bytes were 100,540,928 at measurement start and after drain.
Peak allocated bytes were 276,995,072. Reserved bytes stayed at 891,289,600, including the peak and post-drain value.
All three declared growth checks measured zero bytes: post-drain allocation, sampled allocation floor, and sampled reservation ceiling.
The maximum gap between allocation samples was 0.1074 seconds.

The measured window contains 1,790 telemetry samples. Their largest gap was 1.486 seconds.
GPU temperature ranged from 59 to 61 degrees Celsius, with a median of 60.
Median GPU utilization was 100%, median SM clock was 1,515 MHz, and median reported power was 72 W.
Reported whole-device memory had a median of 1,106 MiB and transient samples up to 1,300 MiB.
Those spikes occurred in minutes 7 and 22. Later minutes returned to 1,106 MiB.
Thus, stable PyTorch allocations do not imply identical whole-device memory at every sample.

These measurements include the model, resident sources, references, and GPU validation overhead.
They do not prove a reusable CPG workspace bound. Host result metadata accumulates within the finite experiment.

## Reproduction, export, and cleanup

The [report](data/019/completed/sustained/report.json), raw compressed CSV files, telemetry, package manifest, and execution log are committed.
The [independent summary](data/019/completed/independent-summary.json) recomputes counts, timings, camera distributions, and memory checks.
Run it against the committed files with a new output path:

```bash
python scripts/summarize_soak.py docs/experiments/data/019/completed/sustained --output /tmp/soak-summary.json
```

The verifier reads either original CSV files or their gzip copies. Sample hashes refer to the decompressed bytes.
All 15 exported files matched their remote SHA-256 values before GPU termination.
The coordinator verified export at 09:09:43 UTC and GPU termination at 09:09:44 UTC.
It verified controller and secret cleanup at 09:09:55 UTC, without another chat turn.
A later account read again found no Pods and no temporary secret.
The successful lease cost $0.4014476068987278 including its controller and disk.
Total project spending is $7.141572519794863 of the $15 authorization. See [the ledger](../RUNPOD_SETUP.md).

## Scope of acceptance

This completes the 30-minute resident inspection soak on this L4 configuration.
It does not complete a Jetson soak, live-camera capture, ROS transport, YOLO/TensorRT, or representative industrial accuracy validation.
There is no new concurrent Nsight capture in this experiment. Earlier synchronous trace evidence remains separately scoped.
The broader release, two-workload performance gate, reusable workspace, batching, and vendor comparisons remain open.

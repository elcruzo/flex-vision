# Experiment 022: continuous detector soak and caller-owned output

## Outcome

The thermally conditioned 30-minute detector soak passed on one RTX 4090 configuration.
All 432,000 offered frames completed with passing GPU correctness checks. No frames dropped.
Each of four logical feeds completed 108,000 frames.
Post-drain and sampled first-to-last-window growth were zero in all four declared allocator counters.
The separate caller-owned FP16 output scenario passed through real TensorRT inference on all six fixtures.

These results close this configuration's detector soak gate. They do not complete the full runtime or release gates.
The feeds are synthetic resident-input schedules on one persistent serial stream, not concurrent GPU streams or live cameras.

## Failed dispatch and corrected handoff

The first dispatch rejected the missing lease deadline in its SSH environment before CUDA setup.
The Pod configuration had the deadline. The SSH shell did not inherit it.
GPU `jdq1fhfhe339we` was removed at 08:01:34 UTC on October 9, 2026.
The subsequent inventory was empty and its temporary secret was absent.
No numerical, reuse, or soak acceptance came from that dispatch.
Its missing result directory exposed an early-failure export limitation.

The coordinator now explicitly exports the validated deadline, creates an early-failure artifact directory, and checks cleanup on exceptions.
The corrected measured revision was `d69928ec62db70a918785e05436b038f90a5bf21`.
The independent GPU lease remained 60 minutes. Setup left about 3,360 seconds before its deadline.
The soak command had a 3,059-second execution timeout and a five-minute export reserve.

## Hardware and controls

The host used NVIDIA GeForce RTX 4090, driver 580.126.20, and 24,564 MiB device memory.
The pinned environment used Torch 2.9.1+cu130, CuPy 14.2.0, TensorRT 10.13.3.9.post1, and CV-CUDA 0.18.0.
The model and fixed FP16 engine preparation follow YOLO_VALIDATION.md.
The engine hash, package manifest, source revision, and GPU preflight are in [the evidence bundle](data/022/results.json).
Six numerical fixtures and 60,000 matched latency samples preceded the soak.
A separate transfer capture passed its known 4,096-byte device-to-host control.
The local NumPy audit checked the 18 saved candidate/fixture combinations and independently decoded their detections.

## Caller-owned output correctness

Two caller-owned CuPy FP16 slots alternated through four reuse cycles per fixture.
All six fixtures passed bitwise tensor equality, dense-output checks, and final detection checks.
The scenario checked that writing one slot preserved the alternate retained slot.
It also checked default owned-output retention and Torch-input interoperability.
Wrong dtype, shape, layout, and destination type were rejected for each fixture.
Input-overlapping destinations were rejected for both 1080p fixtures. The source frames remained unchanged.
See [the reuse report](data/022/output-reuse.json).

This is synchronous FP16 output acceptance on one device and stream configuration.
It does not validate FP32 destinations, arbitrary consumer streams, automatic consumer tracking, or asynchronous workspace reuse.
It does not measure allocation or latency savings from `out=`.
The soak kept the previously measured default-output path rather than changing allocation strategy during the stability test.

## Conditioning and continuous load

One five-minute conditioning block passed the thermal rule.
Its final 120 samples had 0°C spread and no median change between consecutive 60-sample windows.
The measured schedule then ran continuously for 1,800.000026 seconds at four logical feeds of 60 FPS each.
Each feed had two pending slots. The dispatcher selected the oldest admitted arrival across feeds.
The context, stream, resident fixtures, and allocator pools stayed alive. No cache clearing or restart occurred.
Every completed frame received tensor, dense-output, validity, and final-detection checks on the GPU.
The scalar check result was transferred to the host for every completed frame.
The arrival-to-completion boundary includes that validation overhead.

| Five-minute window | Arrival latency p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| 0–5 minutes | 1.304 ms | 1.377 ms | 1.435 ms |
| 5–10 minutes | 1.305 ms | 1.378 ms | 1.434 ms |
| 10–15 minutes | 1.304 ms | 1.377 ms | 1.426 ms |
| 15–20 minutes | 1.304 ms | 1.377 ms | 1.441 ms |
| 20–25 minutes | 1.304 ms | 1.378 ms | 1.442 ms |
| 25–30 minutes | 1.304 ms | 1.377 ms | 1.428 ms |

No new tail-drift threshold was introduced after measurement.
The measured temperature ranged from 35°C to 36°C across 1,801 soak telemetry rows.
Telemetry also records clocks, desktop GPU power, utilization, and device memory.
It does not establish Jetson power or thermal behavior.

All declared counters showed zero post-drain growth: Torch allocated/reserved and CuPy used/total bytes.
Their sampled ceilings also showed zero growth between the first and last five-minute windows.
These counters are not whole-device peak memory or TensorRT workspace measurements.
Two Linux process observations recorded resident memory of 1,536,112 and 1,676,720 kB as host arrival records accumulated.
The late observed process high-water mark was 1,676,720 kB. This is not a final whole-host peak measurement.

## Matched comparison before the soak

Times below are milliseconds. Complete host includes preprocessing, TensorRT, CUDA NMS, and completion synchronization.
It excludes acquisition, upload, and the soak's per-frame validation overhead.

| Fixture | Candidate | Preprocess p50 / p99 | Complete host p50 / p99 |
| --- | --- | ---: | ---: |
| Person 1080p | CPG | 0.0843 / 0.0942 | 0.8704 / 0.8954 |
| Person 1080p | PyTorch | 0.0983 / 0.1126 | 0.8874 / 0.9132 |
| Person 1080p | CV-CUDA | 0.1097 / 0.1229 | 0.9062 / 0.9275 |
| Cat 1080p | CPG | 0.0847 / 0.0942 | 0.8706 / 0.8939 |
| Cat 1080p | PyTorch | 0.0985 / 0.1117 | 0.8881 / 0.9127 |
| Cat 1080p | CV-CUDA | 0.1096 / 0.1258 | 0.9054 / 0.9309 |

CPG preprocessing p50 was about 22.7–23.2% lower than CV-CUDA in this rental.
The complete-host p50 benefit remained small. This comparison does not establish a universal throughput or tail advantage.
Earlier varied tails and vendor-pool failures remain in experiments 020 and 021.
This CPG-only soak does not resolve those vendor allocation failures.
Two photo fixtures are not two product workloads. Inspection remains the separate second workload.

## Independent evidence, export, and cleanup

All 78 remote artifacts matched their hashes before GPU termination at 08:44:52 UTC on October 9.
Controller `rzexvi6tq4jb0p` and secret `7tx1ljwerw` were absent at 08:45:13 UTC.
A later read at 14:05:48 UTC again found no Pods and confirmed both absent.
See [cleanup evidence](data/022/cleanup.json).

The local soak audit reproduced the remote report exactly from compressed arrivals and telemetry.
The reader initially required an uncompressed telemetry file. It now accepts the compressed export too.
This export-reader correction did not change acceptance thresholds or GPU execution.
The compact bundle retains raw compressed samples, telemetry, manifests, verifier results, and a sanitized SQLite capture.
The sanitized trace passed its known-transfer and complete-range checks.
Small metadata transfers remain visible. Do not describe the trace as zero host copies.
Model weights, engines, dense tensor dumps, and raw environment-bearing traces remain in the ignored local bundle.
Full numerical rechecking requires those local tensor dumps or regeneration. The compact report alone cannot reproduce that check.

Reproduce the retained soak audit:

```bash
python scripts/summarize_yolo_soak.py docs/experiments/data/022/soak --output local-soak-verification.json
```

The actual GPU entry point is `scripts/run_yolo_soak_experiment.sh` under the verified independent controller.
Use a new result directory and a committed source revision. Do not provision outside the reconciled spending bound.

## Next decision

Measure caller-owned output against the unchanged owned-output path through the same inference consumer.
Define consumer completion, event handoff, and lifetime rules before asynchronous execution changes.
Keep the full stencil, planning, tuning, ROS, Jetson, and packaging requirements open.

# Experiment 023: matched caller-owned output

## Outcome

All 40,000 sampled frames passed through real FP16 TensorRT inference on RTX 4090.
Caller-owned output removed the observed CuPy pool requests. It did not improve latency.
Preprocessing host p50 increased by 5.18% for the person fixture and 5.25% for the cat fixture.
Complete-host p99 increased by 0.23% and 0.22%, respectively.
Both fixtures passed the predeclared 5% p99 regression guard.
Keep `out=` optional for output placement. Keep fresh owned output as the default.
Do not select reuse as a speed optimization from this evidence.

## Environment and fixed protocol

Measured source: `e8cb4bf0dc3aa5b5df8e9b14ea2c117a2417e7f6`, with a clean checkout.
The host used RTX 4090, driver 580.173.02, and 24,564 MiB GPU memory.
The environment used Torch 2.9.1+cu130, CuPy 14.2.0, CV-CUDA 0.18.0, and TensorRT 10.13.3.9.post1.
The fixed engine SHA-256 was `0969580b5e352f49bb7de7e38d8811c4b73e5fc2490c0a798aff4fe2b3911040`.
Model licensing and preparation remain as specified in YOLO_VALIDATION.md.

Each candidate ran 100 checked warmups and ten blocks of 1,000 timed frames per fixture.
Candidate order alternated by block. A common preallocated slot remained alive during both candidates.
Inputs stayed resident. The same persistent serial stream, TensorRT consumer, and CUDA NMS served both candidates.
Outputs were released after consumption and validation.
Every sampled frame received the existing tensor, dense-output, validity, and final-detection checks.
Validation and its scalar download occurred after timing. They can condition later samples for both candidates.
CUDA-event intervals include host launch gaps. They are not isolated kernel timings.

## Matched latency

All times are milliseconds. Complete host includes preprocessing, TensorRT, CUDA NMS, and synchronization.
It excludes camera acquisition, upload, and per-frame validation.

| Fixture | Output | Preprocess host p50 / p99 | Complete host p50 / p99 |
| --- | --- | ---: | ---: |
| Person 1080p | Fresh | 0.088206 / 0.097182 | 0.866488 / 0.892602 |
| Person 1080p | Reuse | 0.092774 / 0.101430 | 0.870379 / 0.894685 |
| Cat 1080p | Fresh | 0.088105 / 0.097583 | 0.865550 / 0.898052 |
| Cat 1080p | Reuse | 0.092734 / 0.101410 | 0.869307 / 0.900055 |

The complete-host p50 increase was about 0.45% and 0.43%.
The retained report contains pooled and per-block p50, p95, and p99 for all four timing boundaries.
The destination validation adds work, but this experiment does not isolate the cause of the latency increase.
Do not remove ownership or overlap checks based on this result.

## Allocation observations

Separate hooks observed 100 preprocessing calls per candidate and fixture, outside timed measurement.
Fresh output made 100 pool requests for 245,760,000 total requested bytes per fixture.
Reuse made zero pool requests. Both candidates made zero observed underlying device allocations.
The fresh-output positive control passed for both fixtures.
These counters cover CuPy preprocessing only. They exclude Torch and TensorRT allocations.
A warm pool can satisfy requests without allocating device memory.
The common live slot prevents independent peak-memory comparison. No whole-device memory reduction is established.

## Controls and independent audit

The six-fixture caller-owned output scenario passed again, including retention and invalid-destination checks.
A separate 60,000-sample comparison regenerated CPG, Torch, and CV-CUDA controls on the same host.
The local NumPy audit passed all 18 candidate/fixture combinations and independently decoded the detections.
The local reuse audit reproduced the remote report exactly from all 40,000 retained compressed rows.
The sanitized transfer capture passed its known 4,096-byte device-to-host control and complete-range checks.
Small metadata transfers remain visible. This separate default-output trace does not establish reuse-specific transfer behavior.

Reproduce the retained measurement audit:

```bash
python scripts/summarize_yolo_reuse.py docs/experiments/data/023/reuse-benchmark --output local-reuse-verification.json
```

The compact bundle includes raw timing rows, allocation observations, manifests, audit reports, and sanitized trace evidence.
Weights, engines, dense tensor dumps, and the original environment-bearing trace remain ignored.
Full numerical rechecking requires the local dense dumps or regeneration.

## Export, cleanup, and budget

The coordinator verified all 71 exported remote files at 16:22:11 UTC on October 9, 2026.
GPU `f2sti5i5x4xnc1` was absent at 16:22:13 UTC.
Controller `spmvmy3tfpxhhb` and temporary secret `gakflfeydr` were absent at 16:22:24 UTC.
A fresh read at 16:22:52 UTC found no Pods and confirmed all three resources absent.
See [cleanup evidence](data/023/cleanup.json).

Posted billing remained $8.70408385734845 at the final read.
Keep the $0.65 reservation until charges reconcile. The conservative cumulative bound remains $12.014050660732413, below $15.
Reservations are not actual spend. No rental remains active.

## Decision and limits

Keep synchronous caller-owned output as an optional storage contract, with the default unchanged.
This is one GPU, fixed FP16 output, two fixtures, and serial resident-input execution.
It does not validate asynchronous consumers, FP32 output, live capture, ROS, or Jetson.
Define consumer completion and buffer lifetime before asynchronous execution changes.
The result supports fewer allocator requests, not a material latency breakthrough or completion of the release gates.

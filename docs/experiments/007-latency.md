# First matched TensorRT latency comparison

Result: CPG reduced the preprocessing interval in this L4 experiment, while the complete hybrid detector improved by much less.
The two-workload success gate remains open. This single replay does not establish production or Jetson performance.

The timed run used clean revision `5a4a538`. Diagnostic trace and summary tools ran at revision `383a9fb`.
The benchmark implementation did not change between these revisions.

## Workload and procedure

A pinned astronaut photograph was enlarged by exact pixel repetition and centered in a 1920×1080 BGR8 frame.
Both candidates produced FP32 RGB NCHW 320×320 input for the same TensorRT SSDLite engine.
TorchVision CUDA decoded boxes and applied NMS for both candidates.
This is a serial photographic replay, not the required 640×640 FP16 YOLO workload or a live camera test.

The baseline used separate PyTorch CUDA operations for color conversion, layout, float conversion, resize, padding, and normalization.
Normalization constants remained on the device. Both paths used fresh outputs through warmed framework allocation pools.
Both used the same explicit stream and synchronous preprocessing boundary.

Each candidate collected 10,000 samples across ten repeats, with 100 warmup calls before each repeat.
Candidate order alternated between repeats. No sample was excluded.
The independent reference and actual detection checks passed before timing began.
The earlier four-photo detector suite also passed with the explicit stream.

CUDA events delimit stream intervals. Those intervals include dispatch gaps and synchronization effects, not only kernel duration.
Host latency starts before preprocessing and ends after decoded detections complete.
File decoding, upload, model construction, verification downloads, warmup, and sample-file writes lie outside that interval.
Read [the runnable procedure](../TENSORRT_EXPERIMENT.md) for commands and the allocation policy.

## Measured latency

Values below are pooled milliseconds from the uninstrumented samples, using NumPy's linear percentile method.

| Boundary | PyTorch p50 | CPG p50 | PyTorch p95 | CPG p95 | PyTorch p99 | CPG p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preprocessing stream interval | 0.231872 | 0.134176 | 0.247552 | 0.144706 | 0.260288 | 0.156034 |
| Dense network stream interval | 0.929184 | 0.852960 | 0.939651 | 0.864064 | 0.950880 | 0.874880 |
| Decode/NMS stream interval | 12.148896 | 11.425072 | 12.340197 | 11.607841 | 12.484407 | 11.813471 |
| Complete host latency | 13.321868 | 12.430273 | 13.517700 | 12.617615 | 13.675793 | 12.849157 |

Per-repeat preprocessing p50 reductions ranged from 41.62% to 42.68%, with a median of 42.00%.
Per-repeat preprocessing p99 reductions ranged from 31.87% to 43.16%, with a median of 40.29%.
Complete host p50 reductions ranged from 6.37% to 7.22%, with a median of 6.62%.
Complete host p99 reductions ranged from 3.12% to 8.89%, with a median of 6.41%.
These observed repeat ranges are not confidence intervals.

Decode/NMS dominates this implementation's latency. A preprocessing improvement does not translate into an equal complete-pipeline improvement.
Network and decoder intervals also changed, despite using the same engine and decoder code.
Do not attribute these downstream changes solely to the fused kernel.
A fixed-input control and allocation/dispatch analysis remain necessary to explain them.

The baseline is a straightforward PyTorch path. CV-CUDA, VPI, OpenCV CUDA, Kornia, and reused-buffer alternatives were not measured here.
Do not interpret this result as a comparison against the fastest available vendor pipeline.

## Trace findings

A separate short diagnostic capture avoids profiler overhead in the reported timing samples.
The strict four-photo preprocessing-to-TensorRT check found 166 kernels and zero memory-copy events in each completed range.
The engine was rebuilt for this session. Its launch count need not match the previous engine's count.

Two complete diagnostic ranges per candidate produced identical counts within each candidate:

| Complete hybrid detector range | PyTorch | CPG |
| --- | ---: | ---: |
| CUDA kernels | 1252 | 1245 |
| Host-to-device events / bytes | 3 / 12 | 3 / 12 |
| Device-to-host events / bytes | 181 / 724 | 181 / 724 |
| Device-to-device events / bytes | 1 / 43804 | 1 / 43804 |

The complete detector is not copy-free. Small host transfers occur outside the previously validated dense-network boundary.
Their sizes are consistent with scalar metadata. The trace counts alone do not prove the content of each transfer.
They do not show an image-sized host transfer inside these ranges.
CUPTI kinds 1, 2, and 8 identify these transfer directions.
[NVIDIA CUPTI binding definitions](https://docs.nvidia.com/cupti-python/api-reference/topics/bindings.html)

## Limits and next experiment

This is one input composition, model, precision, cloud GPU, and synchronous allocation policy.
Hardware bandwidth counters, peak temporary memory, CPU utilization, and multi-camera throughput remain unmeasured.
GPU telemetry is saved separately. It does not establish Jetson power consumption or a controlled power study.

The next experiment should isolate identical-input inference and inspect decoder synchronization before expanding operators.
Then validate a production-oriented inference consumer and a second workload against credible GPU baselines.
Asynchronous ownership, reusable buffers, FP16 TensorRT, YOLO, inspection, Jetson, and ROS requirements remain unchanged.

## Environment, artifacts, and cleanup

The session used an NVIDIA L4 with driver 595.91.07 and the pinned CUDA 13 environment from experiment 006.
PyTorch 2.9.1+cu130, TorchVision 0.24.1+cu130, CuPy 14.2.0, and TensorRT 10.13.3.9.post1 passed dependency checks.
Both 1080p preprocessing outputs matched the independent reference exactly. Both expected-object IoUs were 0.950994.
All 43 local checks passed. Summary controls rejected instrumented and changed sample files.

All 26 remote artifacts passed SHA256 verification after transfer to the Mac.
The Mac independently reproduced the GPU host summary.
Local bundle: `benchmark-results/iteration-007-latency/`. Large traces and raw samples remain outside Git.
The bundle includes uninstrumented samples, separate diagnostic samples, reports, engines, traces, and telemetry.

| Artifact | SHA256 |
| --- | --- |
| latency/samples.csv | 1f00108893e0b21d09191aa7df6a552e018252fe1cffe4d83094527c872cb1f2 |
| latency/report.json | f4d31daaf6fa42f0e4991674cd196117f548cce723880418e613296fe2669054 |
| latency-summary.json | da10c38078bbdbc97ee81ee554b024ef73cdfaf70d94f8ef0424e09467cd5a79 |
| latency-trace.nsys-rep | 5760868a248227180f1170d3d0a57c44f2211187bd74d9ba61bd48ebd488d570 |
| latency-trace.sqlite | dfad945d3f070164d803e61432e4d4acb86f9b5926b0556ef29685547599c0b6 |
| tensorrt-detector/ssdlite.engine | 5c6be16d5134d4b812c7bff18bf5ede6c8fae333a4dc1f8fb47ba81b1c5810a0 |

Telemetry includes initialization as well as timed execution. Temperatures ranged from 39 to 44°C.
Clock and power transitions during initialization remain in the raw telemetry. No manual clock lock was applied.

Pod `slkpedxcxg5nj0` ran in EU-RO-1 at $0.49/hour with 50 GB temporary disk and no persistent volume.
It started at 17:40:08 UTC and was terminated at approximately 17:53:16 UTC on 2026-10-04.
Elapsed compute cost is approximately $0.107. Reserve $0.12 including disk until billing posts.
The cleanup read returned no Pods. The local fallback timer was cancelled.
Billing reads still showed no posted records for experiments 006 and 007. They were not free sessions.
Posted earlier charges plus both session reserves total approximately $0.423 of the authorized $15.

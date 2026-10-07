# Experiment 018: sustained inspection through real inference

## Protocol and scope

Use the [sustained protocol](../SUSTAINED_INSPECTION.md) with unchanged tensor and classifier tolerances.
The input is the pinned 4K Chelsea canvas, alternating original and mirrored pixels.
The consumer is the pinned pretrained MobileNetV3 Small on CUDA.
Compare independent full-frame PyTorch preprocessing with the public ROI path.

Offer one stream at 30 frames per second, then four streams at 60 frames per second each.
Limit each camera to two pending frames. Drop expired dispatch slots or arrivals whose queue is full.
Each block lasts 20 seconds, followed by drain. Run four alternating-order repeats per strategy and workload.
There are 43,200 offered arrivals across the complete protocol.

Every accepted frame checks its full tensor, all logits, and ordered top-five labels against independent CPU references.
Reference logits must pass the predefined cat-class semantic check.
GPU validation kernels remain in the measured load for both implementations.
Host completion times include those checks and event-polling delay.
This is an instrumented developer experiment, not an uninstrumented deployment or the 30-minute soak.

The preset performance hypothesis requires at least 20% higher median paired overload throughput.
It also requires no more than 5% higher median paired single-camera p99.
Memory growth and per-camera behavior need separate inspection. A correctness pass does not imply memory acceptance.

## Initial run: performance gain with changing allocation baselines

Revision `5109835` ran on an L4 in EUR-IS-1 with driver 595.91.07.
The environment used CUDA 13, PyTorch 2.9.1+cu130, and TorchVision 0.24.1+cu130.
The initial harness created new CUDA streams for each comparison block.
All 24 separate stream-handoff cases and all 29,542 completed sustained frames passed correctness checks.

| Load | Baseline completed / offered | ROI completed / offered |
| --- | ---: | ---: |
| One camera, four repeats | 2,398 / 2,400 | 2,400 / 2,400 |
| Four cameras, four repeats | 9,407 / 19,200 | 15,337 / 19,200 |

Median paired overload throughput improved 62.99%.
The median paired single-camera p99 ratio was 0.9501, within the preset 1.05 limit.
Overload baseline throughput ranged from 116.76 to 117.81 completed frames per second.
ROI throughput ranged from 191.20 to 192.25 completed frames per second.
These figures include drain time and GPU validation overhead.

All cameras completed work in every block.
Baseline per-camera counts ranged from 585 to 590 per block, with p99 between 69.52 and 71.00 ms.
ROI counts ranged from 947 to 972, with p99 between 43.77 and 44.22 ms.
These observed ranges do not establish fairness under different arrival patterns.
Most baseline drops came from full queues. ROI drops included 1,609 missed dispatch slots and 2,254 full queues.
CPU dispatch therefore remains relevant even when preprocessing becomes faster.

Post-drain allocated bytes rose from 71,397,376 in the first block to 368,025,088 in later blocks.
Each block returned to its own starting allocation. Growth occurred between blocks that introduced streams, rather than within measured frame processing.
This observation requires investigation before a steady-memory claim.
It is not proof of a per-frame leak or unlimited growth.

The [initial report and raw CSV files](data/018/initial/sustained/report.json) retain the full run.
The [independent summary](data/018/initial/independent-summary.json) reproduces timings and records each camera.
The [export check](data/018/initial/export-verification.json) matched all 22 completed result files to remote hashes.
Telemetry was still changing during export and did not match its earlier hash. Exclude it from verified evidence.
Later trace files were not exported before expiration. This initial run adds no retained trace proof.

## Revised stream policy

Revision `fd42d5a` reuses four CUDA streams across strategies and repeats.
The one-camera control uses the first stream from that pool.
The public runtime, model, inputs, arrival rates, queue depth, checks, and performance limits are unchanged.
No allocator cache is cleared between blocks.

PyTorch documents retained [cuBLAS workspaces per handle and stream](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cublas-workspaces).
That mechanism is consistent with the initial growth. Stream reuse tests the explanation without changing preprocessing.
The rerun uses another host, so timing differences between the two leases cannot establish a causal effect of stream reuse.
Compare baseline and ROI within each run.

## Revised results

The revised run used an L4 in EUR-IS-2 with driver 580.159.04 and the same pinned framework versions.
All 24 separate stream-handoff cases passed again.
All 28,965 completed sustained frames passed tensor, logit, and ordered-label checks.
The [revised report](data/018/revised/sustained/report.json) retains every block and its allocation samples.
The [independent summary](data/018/revised/independent-summary.json) reproduces the raw CSV results and per-camera distributions.

| Load | Baseline completed / offered | ROI completed / offered |
| --- | ---: | ---: |
| One camera, four repeats | 2,399 / 2,400 | 2,400 / 2,400 |
| Four cameras, four repeats | 9,178 / 19,200 | 14,988 / 19,200 |

| Repeat | Baseline overload FPS | ROI overload FPS | Baseline overload p99 ms | ROI overload p99 ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 115.784 | 188.883 | 71.080 | 45.558 |
| 1 | 114.617 | 188.022 | 72.004 | 45.597 |
| 2 | 114.399 | 186.304 | 72.344 | 45.638 |
| 3 | 113.372 | 185.726 | 72.533 | 45.662 |

The median paired throughput ratio was 1.6348: 63.48% higher completed throughput.
The median paired single-camera p99 ratio was 0.7007, below the predefined 1.05 limit.
Both performance checks passed. Single-camera baseline p99 ranged from 9.120 to 9.258 ms, versus 6.351 to 6.931 ms for ROI.
Per-camera completion counts differed by at most one frame within each overload block.
No camera starved. The pending count stayed within the declared limit of eight for four cameras.

Post-drain allocations were identical across repeats and strategies for each camera count:

| Camera count | Allocated bytes after warmup and after drain | Baseline peak allocated bytes | ROI peak allocated bytes |
| --- | ---: | ---: | ---: |
| One | 71,397,376 | 470,659,072 | 247,851,520 |
| Four | 100,113,920 | 499,375,616 | 276,568,064 |

There were no within-block or across-repeat allocation-growth flags in the revised run.
These values include the model, inputs, references, validation flags, and library allocations tracked by PyTorch.
They are not isolated CPG workspace measurements or total device memory.
The runtime still allocates fresh outputs and lacks a bounded reusable workspace pool.
Stream reuse removes the observed changing allocation baseline in this short experiment. It does not prove a general memory bound.

## Interpretation and remaining work

Retain the ROI strategy and use persistent camera streams for this workload.
The improvement follows reduced preprocessing work, not a new fused stencil kernel.
The fixed stream pool is a harness correction, not a production runtime optimization.
Keep both runs, including the initial allocation finding and dropped arrivals.

This is batch-one resident replay with GPU correctness checks, a fixed input size, and an ImageNet smoke photograph.
It excludes capture, upload, ROS, TensorRT, representative defect accuracy, and input-resolution changes.
It does not complete the 30-minute soak or the required multi-camera rectification/TensorRT robot workload.
CPU utilization, physical memory bandwidth, allocation-call counts, and Jetson power were not measured.
The required 640×640 FP16 YOLO/TensorRT detector remains a separate delivery task.

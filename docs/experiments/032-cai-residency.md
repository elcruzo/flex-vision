# Experiment 032: fixed CAI complete-inference transfer capture

The repeated 144-case cai-v2 correctness gate passed on RTX 4090.
The separate 16-range capture then passed real FP16 YOLO TensorRT inference and CUDA NMS.
Both 1080p fixtures passed across explicit, legacy-default, per-thread-default, and ready-input contracts in sync and submit modes.
Exact tensor bits, dense outputs, and detections passed after every captured range.
The independent local audit accepted all report coverage and selected transfer observations.
The separate NumPy/CPU NMS audit passed all 18 exported baseline comparisons. Those baseline inputs use DLPack.

Clean measured revision: `fabc8fea87e4f8bb45ecb21b6c240fda9209b159`.
Engine SHA-256: `df9cd7e8fb41318178955c96a1c1ddd2728b64c5105877c147df6931672dc6f0`.
The capture records and verifies its matching CAI correctness-report hash.
Uploads, compilation, warmup, and diagnostic downloads remain outside selected complete ranges.
This trace uses ready contiguous resident inputs. The separate artificial-delay correctness gate checks producer reuse and signed strides.

## Captured observations

Every completed range contained 168 CUDA kernels, including exactly one CPG preprocessing kernel.
Every range recorded one four-byte H2D transfer and four four-byte D2H transfers.
The aggregate host bytes were 20, below the 2,457,600-byte FP16 output tensor size.
Do not report zero host copies. These small complete-path metadata copies remain part of the observations.
Each range also recorded three CUPTI kind-8 copies, totaling 144 bytes for astronaut and 160 bytes for chelsea.
The known 4096-byte D2H control appeared exactly once with its expected size and direction.
No selected range contained a frame/tensor-sized aggregate host transfer.
These observations support the fixed resident CAI path. They do not prove absence of every possible partial-payload transfer in another workload.

The verifier rejects a missing preprocessing kernel, missing downstream GPU work, incomplete inference, and aggregate tensor-sized host bytes.
It includes chunked transfers in that aggregate. Fault checks do not substitute for the hardware capture.
No latency, overlap, or allocator-stability selection follows from these instrumented ranges.

## Artifacts and cleanup

The coordinator exported 42 hash-verified files at 05:44:28.587998 UTC on October 10, 2026.
It verified GPU termination at 05:44:29.854919 UTC.
Controller and temporary-secret cleanup passed at 05:44:40.642008 UTC.
Fresh inventories returned no Pods or secrets. See [cleanup](data/032/cleanup.json).
Keep the $0.65 reservation until lifetime and billing reconciliation.
The conservative cumulative reservation remains $14.98270978906015 within the original $15 authorization.

The complete approximately 64 MiB bundle remains ignored locally in benchmark-results/iteration-032-yolo.
The repository retains the sanitized 456 KiB range projection and reports.
The projection contains 17 NVTX spans, 2,688 kernels, 129 copies, 4,697 runtime events, and 451 sanitized strings.
The reusable projection tool at revision `7c223f9` produced this local artifact after measured-source execution.
Its selected counts and control matched the full sanitized capture exactly.
The CAI trace verifier also passed independently on the projection.
See [projection provenance](data/032/projection.json), [local full-trace audit](data/032/local-cai-trace-audit.json), and [projection audit](data/032/local-projection-audit.json).
A projection is not a whole-process trace. Raw traces, weights, engines, and dense arrays remain local.

## Remaining product work

This completes the first fixed CAI correctness and transfer-capture protocol.
Read-only and zero-stride input, invalid-device/pointer controls, external producers, and broader dtype/platform coverage remain pending.
The synchronous default and current owned-output async limits remain unchanged.
Configurable stencils, general planning/tuning, batching/workspace, ROS, and Jetson remain required MVP work.

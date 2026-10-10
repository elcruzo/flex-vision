# Experiment 028: async capture and allocator observations

## Outcome

The prepared diagnostic protocol passed 4,000 real TensorRT inference checks on RTX 4090.
All 48 regenerated delayed-consumer checks passed first.
CuPy used/total and PyTorch allocated/reserved bytes showed zero growth above each fixture's warm post-drain baseline.
This short serial run does not prove TensorRT internal memory stability, transient peak bounds, or concurrent capacity.

Four completed inference ranges captured sync and submit for person and cat 1080p fixtures.
Each range contained 168 CUDA kernels, including one preprocessing kernel.
Observed consumer kernels began after preprocessing completed, on a different stream.
The [ordering observations](data/028/kernel-order-observations.json) preserve actual stream IDs and timing gaps.
These observations do not independently reconstruct every event dependency or establish overlap.

Each range recorded one 4-byte host-to-device copy and four 4-byte device-to-host copies.
Small device-side copies also appeared. No host copy approached the 2,457,600-byte output size.
The known 4,096-byte device-to-host control passed.
Do not report zero host copies or infer the absence of smaller pixel transfers solely from this threshold.
These ranges exclude input uploads and validation downloads.

## Reproduction and audit

Clean measured revision: `07dc5e31e3ee2161e812511d30d4e91b590ed91d`.
Engine SHA-256: `10435fa6358ebc96209bd6d2c739be34d92986464597e6921aa046252115a5e5`.
Hardware: RTX 4090, driver 580.126.20, 24,564 MiB memory.
The pinned environment and numerical semantics match the established YOLO protocol.
Use `scripts/run_yolo_async_diagnostics.sh` under the independent lease controller.
See [the diagnostic protocol](../ASYNC_BENCHMARK.md) for exact boundaries and exclusions.

The remote and local diagnostic audits matched, including every raw frame and checkpoint.
The independent NumPy and CPU NMS audit passed all 18 baseline candidate/fixture combinations.
Correctness checks after the timed boundary condition later frames. No timings from this capture select a strategy.
The uninstrumented comparison still needs an independent repeat after experiment 027.

The audit now hashes decompressed CSV bytes so compression preserves sample identity.
The original remote audit hashed its uncompressed CSV before export compression.
The corrected local verifier produced the identical hash from the exported gzip file.
A compression fault test passed. The measured source and GPU protocol did not change.

## Evidence and cleanup

The coordinator exported 43 hash-verified files at 05:05:20 UTC on October 10, 2026.
The full 639 MiB bundle remains ignored locally in benchmark-results/iteration-028-yolo.
The repository retains reports, compressed samples, environment records, and a sanitized range projection.
The projection contains five selected NVTX spans, overlapping kernels/copies/runtime calls, and sanitized strings.
Its control and all per-range kernel/copy observations match the full capture exactly.
See [projection provenance](data/028/projection.json). It is not a whole-process trace.
Raw environment-bearing traces, weights, engines, and dense arrays remain local.

GPU gpfuyl1s3zlfe4 was absent at 05:05:21 UTC.
Controller uadt87wrabbbso and secret zi0su5n07w cleanup passed at 05:05:32 UTC.
The fresh 05:05:39 UTC inventory returned no Pods or secrets.
Keep the $0.65 reservation until reconciliation.
The cumulative conservative bound remains $14.595168122393484 within the original $15 authorization.

## Next product work

This completes the first fixed async diagnostic capture and declared allocator gate.
It does not complete async production support or change the synchronous default.
Repeat the favorable uninstrumented comparison independently, then evaluate bounded concurrent scheduling with correct consumer lifetimes.
CUDA Array Interface producers, configurable stencils, general planning/tuning, ROS, and Jetson remain required work.

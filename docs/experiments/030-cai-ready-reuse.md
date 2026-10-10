# Experiment 030: ready-input reuse ordering failure

The CAI gate failed after seven passing scenarios on RTX 4090.
The ready-input submission then failed exact tensor comparison because the harness queued a producer overwrite before preprocessing completed.
This exporter advertised stream=None. The adapter had no producer stream on which to order the future write.
Retaining the allocation prevents release. It does not prevent caller mutation.

Clean measured revision: `e0c9f5a5f04f8d6c92c02e3d1bdfaeeae3c756c4`.
Engine SHA-256: `46bdab5bfa26c5ade5916ea883d6c2c1ac83779b2dd01aa5767b5a8d7a53d1a5`.
The passing cases cover the contiguous astronaut fixture across explicit, legacy, and per-thread streams in both execution modes, plus ready-input synchronous execution.
The failed case mismatched 1,106,304 of 1,228,800 output elements. Stream drainage passed.
This partial result does not establish full CAI acceptance or an adapter failure under compliant input reuse.

The corrected cai-v2 harness waits for completion before stream=None reuse.
Advertised-stream submissions still require an immediate queued overwrite behind the adapter's completion fence.
A second pending observation occurs immediately before that overwrite, after garbage collection.
The report records reuse ordering, pending observations, and the active scenario and stage on failure.
The full 144-case real-inference gate remains required.
All 187 local checks passed after this correction.
The independent local baseline audit passed all 18 exported candidate/fixture comparisons.
Those baseline paths use DLPack and do not validate the CAI adapter.

The coordinator exported 35 hash-verified files at 05:25:01.734913 UTC on October 10, 2026.
It verified GPU termination at 05:25:03.376836 UTC.
Fresh Runpod inventories returned no Pods and no secrets. See [cleanup evidence](data/030/cleanup.json).
Preserve the $0.65 reservation until lifetime and billing reconciliation.
The conservative cumulative reservation remains $14.540987566837929 under the original $15 authorization.
See [failed report](data/030/cai.json). No performance or CAI transfer claim follows from this run.

# Experiment 031: CAI-only real-inference correctness

All 144 cai-v2 scenarios passed on RTX 4090 through the fixed FP16 YOLO TensorRT consumer.
The local report audit independently checked scenario coverage and required observations.
The separate NumPy and CPU NMS audit passed all 18 exported baseline candidate/fixture comparisons.
Those baseline comparisons use DLPack. They do not independently recompute the CAI-specific outputs, which the remote harness checked against the same oracle.

Clean measured revision: `0f8c12cf7bc56548e8daa4014a50f2fac8c48433`.
Engine SHA-256: `01078815d14db395b8dba0c7f7abb94db4e1bc72f1d8759ecb926971ff83fa7b`.
The matrix covers six fixtures, contiguous/reversed-row/reversed-channel layouts, four producer contracts, and synchronous/submitted execution.
Every case checked exact FP16 tensor bits, dense inference outputs, detections, and producer reuse.
Every later output also checked that the preceding retained output stayed exact.
All submitted cases observed pending preprocessing and exporter retention after caller release.
Advertised-stream submissions observed pending preprocessing again immediately before queuing producer reuse.
Ready-input submissions completed preprocessing before the caller queued future writes.
Stream drainage passed.

The coordinator exported 36 hash-verified files at 05:34:15.341910 UTC on October 10, 2026.
It verified GPU termination at 05:34:16.980199 UTC.
Controller and temporary-secret cleanup passed at 05:34:38.136664 UTC.
Fresh inventories returned no Pods or secrets. See [cleanup](data/031/cleanup.json).
Keep the $0.65 reservation until lifetime and billing reconciliation.
The conservative cumulative reservation remains $14.781640344615706 within the original $15 authorization.

See [CAI evidence](data/031/cai.json), [local coverage audit](data/031/local-cai-audit.json), and [baseline value audit](data/031/local-values.json).
Experiments 029 and 030 remain retained failed attempts.

This result establishes the fixed CAI correctness and ownership scenario, not all interoperability acceptance.
CAI-specific transfer capture, invalid-device/pointer controls, read-only and zero-stride inputs, external libraries, and broader dtypes/platforms remain pending.
Artificial delays provide no performance or overlap evidence.
Keep the synchronous default and owned-output submission limits unchanged.

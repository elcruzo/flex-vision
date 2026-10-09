# Experiment 021 — Detector repeatability and bounded load

Status: the final twelve-block matrix passed its short correctness and allocator guards.
Both earlier retained-pool failures remain failed. The matched preprocessing improvement repeated, but sustained throughput gains varied.

## Repeated comparison

Run source `6d396a3` with the same fixed detector backend and protocol as experiment 020.
All six numerical cases, 18 independent NumPy checks, and 60,000 matched samples passed again on an RTX 4090.
The transfer verifier passed on a sanitized SQLite export, including the known 4,096-byte control.
The original capture hash remains in the remote manifest. Raw environment-bearing captures stay local.

| Fixture | Candidate | Preprocess p50 / p99, ms | Complete host p50 / p99, ms |
| --- | --- | --- | --- |
| Person 1080p | CPG | 0.0850 / 0.0965 | 0.8749 / 0.9257 |
| Person 1080p | PyTorch | 0.1023 / 0.1290 | 0.8990 / 0.9852 |
| Person 1080p | CV-CUDA | 0.1158 / 0.1423 | 0.9155 / 0.9919 |
| Cat 1080p | CPG | 0.0850 / 0.0994 | 0.8742 / 1.0087 |
| Cat 1080p | PyTorch | 0.1004 / 0.1432 | 0.8918 / 1.0597 |
| Cat 1080p | CV-CUDA | 0.1147 / 0.1601 | 0.9112 / 1.0839 |

CPG preprocessing p50 improved by 25.92–26.59% versus CV-CUDA, compared with 24.77% in the preceding optimized rental.
Its preprocessing p99 improved by 25.22–30.58% versus PyTorch. The preceding rental measured 37.10–38.66%.
Complete host p50 improved by 1.97–2.68% versus PyTorch and 4.07–4.44% versus CV-CUDA.
Both rentals support the scoped preprocessing improvement. Their differing tails require continued reporting rather than a universal latency claim.
The two photos remain one detector workload. They are not two independent product workloads.

## Initial bounded-load result

The [sustained protocol](../SUSTAINED_YOLO.md) uses four logical feeds and one persistent serial stream.
The first three 30-second control blocks offered 7,200 frames each at an aggregate 240 FPS.
CPG completed 7,191 frames and dropped nine. PyTorch and CV-CUDA completed all 7,200.
All 21,591 completed frames passed their GPU tensor, dense-output, and detection checks.
The partial CSV audit verified arrival coverage and completed-frame checks. It does not pass full sustained acceptance.

Torch allocated/reserved and CuPy live used counters showed zero post-drain growth in these three blocks.
CPG and PyTorch also showed zero retained CuPy growth.
CV-CUDA's retained CuPy pool grew by 2,457,600 bytes, exactly one FP16 1×3×640×640 output allocation.
The declared allowance is 1,048,576 bytes. This failed the guard and stopped the experiment before overload and second-repeat coverage.
No sustained throughput advantage is claimed from this incomplete attempt.

The initial warmup omitted the validation used on every measured frame.
Source `6b0af27` changes warmup to 100 fully checked frames on each candidate.
It preserves the stream, allocation pools, admission policy, and growth threshold. No cache is cleared.
This tests a first-use retained-allocation explanation. It does not retroactively pass the initial attempt.

## Evidence and cleanup

The [initial evidence](data/021/initial/results.json) includes raw compressed comparison and load rows, independent verifier results, and sanitized trace data.
All 72 remote artifacts matched their hashes before termination at 05:50:04 UTC on October 9, 2026.
The controller and temporary credential were absent at 05:50:15 UTC.
The retry uses a new independently bounded lease within the original $15 authorization.

This experiment does not complete detector soak, concurrent streams, batching, reusable workspace, live-camera, Jetson, or ROS acceptance.


## Checked-warmup attempt

Source `6b0af27` again passed the numerical, matched timing, and sanitized transfer-capture checks.
All six control blocks and the first three overload blocks completed without a failed frame check.
The nine completed blocks offered 187,200 frames and completed 114,761.
Every declared post-drain allocator counter was unchanged except the retained CuPy pool in the CV-CUDA overload block.
That block added 2,457,600 bytes at approximately 28.04 seconds, again exceeding the unchanged 1 MiB guard.
Warmup correction therefore did not resolve the saturation result. Keep the overall sustained verdict failed.

CPG completed 7,188 and 7,200 frames in its two control blocks, with twelve total drops.
PyTorch and CV-CUDA completed all 14,400 control frames each.
The first overload blocks completed 23,744 CPG, 24,069 PyTorch, and 23,760 CV-CUDA frames out of 48,000 offered per candidate.
These checked, serial load results do not demonstrate a CPG throughput advantage.
The second overload repeat was hidden by the first failure. The next harness retains all twelve blocks before rejecting a failed matrix.

Pinned vendor source shows auxiliary-stream resource callbacks and cached external-buffer wrappers.
This is a concrete lifetime hypothesis for the extra retained allocation, not proof of an unbounded leak.
A separate private auxiliary-drain diagnostic will test it without changing the timed vendor baseline or clearing caches.
See the [investigation protocol](../SUSTAINED_YOLO.md#saturation-resource-lifetime-investigation).

All 77 artifacts were hash-verified before GPU removal at 06:01:23 UTC on October 9.
The controller and temporary credential were absent at 06:01:34 UTC. The [checked-warmup evidence](data/021/checked-warmup/load-results.json) preserves the failed verdict.


## Completed matrix

Source `27e6a56` retained all twelve normal blocks and a separate auxiliary-drain diagnostic.
The numerical runner passed all six cases. Independent checks verified all 18 candidate/fixture comparisons and 60,000 timing samples again.
The backend and normal comparison remained unchanged. The new control did not enter the timed baseline.

The normal load matrix offered 331,200 frames and completed 187,279. All completed frames passed their GPU checks.
CPG completed 63,019, PyTorch 61,712, and CV-CUDA 62,548 frames across their control and overload blocks.
Every feed completed frames in every block. Pending depth remained within two per feed, with at most eight queued frames in total.
All twelve blocks showed zero post-drain growth in Torch allocated/reserved and CuPy used/total counters.
These are declared allocator counters, not whole-device peak memory or a reusable-workspace guarantee.

| Load per feed | Candidate | Repeat | Completed / offered | Completed FPS with drain | Arrival p50 / p99, ms |
| --- | --- | ---: | ---: | ---: | --- |
| 60 FPS | CPG | 0 | 7,189 / 7,200 | 239.63 | 1.343 / 1.845 |
| 60 FPS | CPG | 1 | 7,200 / 7,200 | 240.00 | 1.346 / 7.489 |
| 60 FPS | PyTorch | 0 | 7,200 / 7,200 | 240.00 | 1.380 / 8.427 |
| 60 FPS | PyTorch | 1 | 7,200 / 7,200 | 240.00 | 1.373 / 7.823 |
| 60 FPS | CV-CUDA | 0 | 7,200 / 7,200 | 240.00 | 1.396 / 9.570 |
| 60 FPS | CV-CUDA | 1 | 7,200 / 7,200 | 240.00 | 1.390 / 1.929 |
| 400 FPS | CPG | 0 | 23,450 / 48,000 | 781.44 | 9.679 / 21.611 |
| 400 FPS | CPG | 1 | 25,180 / 48,000 | 839.09 | 9.468 / 11.174 |
| 400 FPS | PyTorch | 0 | 24,212 / 48,000 | 806.85 | 9.757 / 15.915 |
| 400 FPS | PyTorch | 1 | 23,100 / 48,000 | 769.79 | 9.921 / 21.595 |
| 400 FPS | CV-CUDA | 0 | 24,064 / 48,000 | 801.93 | 9.950 / 11.865 |
| 400 FPS | CV-CUDA | 1 | 24,084 / 48,000 | 802.54 | 9.955 / 11.688 |

CPG dropped eleven control arrivals in its first block and none in the second. Both library baselines dropped none at control load.
The first CPG overload block had lower completion rate and worse p99 than either baseline. The second block improved both metrics.
Do not average away this reversal or claim a repeatable sustained-throughput advantage.
GPU validation competes for resources and includes a scalar completion transfer per accepted frame.
The checks also differ by numerical policy: CPG checks all boxes, while approximate baselines use the documented confidence floor.
This is checked serial service capacity, not concurrent GPU streams or uninstrumented production capacity.
The raw summary retains per-feed counts, per-feed p99, and early/late p99 for every block.

## Final matched timing

| Fixture | Candidate | Preprocess p50 / p99, ms | Complete host p50 / p99, ms |
| --- | --- | --- | --- |
| Person 1080p | CPG | 0.0840 / 0.1061 | 0.8694 / 1.0787 |
| Person 1080p | PyTorch | 0.0999 / 0.1620 | 0.8888 / 1.0705 |
| Person 1080p | CV-CUDA | 0.1134 / 0.1749 | 0.9067 / 1.1171 |
| Cat 1080p | CPG | 0.0840 / 0.1128 | 0.8695 / 1.2335 |
| Cat 1080p | PyTorch | 0.0995 / 0.1792 | 0.8888 / 1.3649 |
| Cat 1080p | CV-CUDA | 0.1134 / 0.2099 | 0.9071 / 1.4710 |

The lower preprocessing median repeated. Complete person p99 was slightly worse than PyTorch in this rental.
A preprocessing gain does not establish a universal complete-pipeline tail improvement.
Keep this limit visible alongside the preceding repetitions.

## Auxiliary-stream diagnostic

The private auxiliary-drain control completed 22,284 of 48,000 offered frames with zero failed checks and zero declared allocator growth.
Its rows passed a separate coverage, completion-boundary, and allocator audit.
The normal CV-CUDA blocks also showed zero growth in this rental, so this control does not isolate the cause of earlier increases.
The pinned source supports the resource-lifetime hypothesis. The observed saturation failures remain unresolved across rentals.
Do not add the private hook to production execution or use its slower checked capacity as a competitive baseline.

## Final evidence and next gate

The [completed bundle](data/021/completed/load-verification.json) contains raw compressed load and timing samples, independent numerical and trace checks, and a sanitized SQLite capture.
All 81 remote artifacts matched their hashes before GPU termination at 06:22:38 UTC on October 9, 2026.
The controller and temporary credential were absent at 06:22:49 UTC. The independent GPU lease was 45 minutes.
A fresh API read at 06:32:54 UTC found no Pods and confirmed that the controller and secret were absent. See [cleanup evidence](data/021/completed/cleanup.json).
Spending reconciliation remains in RUNPOD_SETUP.md.

The retained load rows independently reproduce `load-verification.json`. To repeat that audit, use a temporary directory with the retained CSV files and expose `load-results.json` there as `results.json`. Run `scripts/summarize_yolo_load.py` on that directory.
The combined bundle uses `results.json` for the separate numerical and timing experiment.
The sanitized SQLite capture also passed integrity and environment-removal checks. Full numerical rechecking requires the ignored local tensor artifacts; the compact published numerical report alone cannot reproduce that check.

This completes the short serial detector load matrix. It does not complete detector soak, asynchronous streams, batching, Jetson, ROS, or release acceptance.
Next, establish a thermally conditioned 30-minute detector soak and investigate the observed service-tail variation.
Prepared output ownership and asynchronous scheduling remain architectural work. Broad operator expansion remains gated by the complete product plan.

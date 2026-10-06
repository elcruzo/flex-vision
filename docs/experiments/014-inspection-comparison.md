# Experiment 014: matched inspection comparison

## Protocol fixed before measurement

Compare the full-frame PyTorch baseline and ROI candidate on the same GPU and model.
Require passing CUDA classifier reports from the same clean revision for both implementations.
Keep the existing tensor and inference tolerances. Do not relax them after seeing results.

Use four repeats of 300 frames per fixture and implementation, with 30 warmups per block.
Alternate block order: baseline then ROI for even repeats, ROI then baseline for odd repeats.
Both implementations use the same resident BGR8 input and CUDA classifier.
Record preprocessing, inference, stream total, completed host latency, and PyTorch allocator observations.
Preserve all samples. Report pooled and per-repeat p50, p95, and p99 using linear percentiles.
Do not mix profiled timing with ordinary measurements.

The primary hypothesis is at least 20% lower 4K preprocessing p50 or p99.
Reject adoption if pooled complete-host p99 regresses by more than 5% on any fixture.
Inspect repeat-level results before accepting a pooled improvement.
The full-frame fixture is a control where the candidate cannot avoid filter pixels.
Use separate traces to inspect launches and transfers for both implementations.

This experiment evaluates a graph rewrite implemented with PyTorch primitives.
It does not establish production CPG support, industrial accuracy, or a full two-workload release gate.

## Results

The comparison passed its stated thresholds on October 6, 2026.
The clean measured revision was `5595217d50c00f88354241a0913125cae880eaac`.
The host used an NVIDIA L4, driver 595.91.07, CUDA 13.0, PyTorch 2.9.1+cu130, and TorchVision 0.24.1+cu130.
TF32 and cuDNN benchmark selection were disabled. See the [package manifest](data/014/packages.txt).

All six unprofiled classifier cases passed, as did all six cases in separate profiling runs.
The maximum tensor error was 0.0000461 for both implementations, below the unchanged 0.0002 limit.
All logits, ordered top-five classes, and cat-class checks passed.
Another 19 CUDA edge cases passed with maximum tensor error 0.00000382.
Retained outputs remained unchanged, and input frames remained unchanged.

Times below are milliseconds. Host total includes preprocessing, classifier inference, and completion synchronization.
It excludes camera acquisition and input upload.

| Fixture | Implementation | Preprocess p50 | Preprocess p99 | Host total p50 | Host total p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chelsea | Baseline | 0.1497 | 0.2318 | 2.2486 | 2.9416 |
| Chelsea | ROI | 0.1571 | 0.2217 | 2.2399 | 2.9553 |
| Odd ROI | Baseline | 0.1517 | 0.2087 | 2.2909 | 3.1431 |
| Odd ROI | ROI | 0.1600 | 0.2231 | 2.2861 | 3.1196 |
| 4K ROI | Baseline | 8.0238 | 8.1225 | 9.0664 | 9.2037 |
| 4K ROI | ROI | 4.6608 | 4.7131 | 5.6988 | 5.7767 |

The 4K candidate reduced preprocessing p50 by 41.91% and p99 by 41.97%.
Complete-host p50 fell by 37.14% and p99 by 37.23%.
The largest complete-host p99 regression was 0.46% on Chelsea, within the predefined 5% limit.
Small-image preprocessing medians regressed approximately 5%. Do not apply this candidate universally.

| Repeat | 4K baseline preprocessing p50 / p99 | 4K ROI preprocessing p50 / p99 |
| --- | ---: | ---: |
| 0 | 7.9832 / 8.1436 | 4.6480 / 4.7112 |
| 1 | 8.0252 / 8.1043 | 4.6532 / 4.7070 |
| 2 | 8.0280 / 8.0928 | 4.6673 / 4.7210 |
| 3 | 8.0314 / 8.0956 | 4.6653 / 4.7071 |

The PyTorch peak allocation above the block's starting allocation fell from 399,261,696 to 235,228,672 bytes for 4K.
These are allocator observations for the complete measured path, not total GPU memory or a general workspace bound.
Each implementation recorded 199 kernels and zero memcpy events in each completed inference range.
Launch count is unchanged. The measured advantage comes from restricting the work region.

## Saved evidence and reproduction

The [raw samples](data/014/samples.csv) contain 7,200 rows.
The [latency report](data/014/latency.json) includes p95, inference times, and allocator observations.
Independent local recomputation matched all reported pooled percentiles.
See [baseline validation](data/014/baseline.json), [candidate validation](data/014/roi.json), and [edge checks](data/014/edges.json).
The [baseline trace summary](data/014/baseline-trace-summary.json) and [candidate trace summary](data/014/roi-trace-summary.json) record copies and launches.

All 28 exported artifacts matched their remote SHA256 hashes before GPU termination.
The [full manifest](data/014/sha256.txt) uses original paths under `benchmark-results/iteration-014-inspection/`.
Large NPZ arrays, SQLite exports, and Nsight reports remain in that ignored local directory.
The small reports and CSV are committed here for independent review.

```bash
python scripts/local_classifier.py --device cuda --output benchmark-results/baseline
python scripts/local_classifier.py --device cuda --implementation roi --output benchmark-results/roi
python scripts/inspection_benchmark.py --validation benchmark-results/baseline/report.json --candidate-validation benchmark-results/roi/report.json --output benchmark-results/comparison
```

Use new output directories inside ignored `benchmark-results/` when reproducing these commands.
The benchmark requires a clean revision, pinned model weights, and matching validation hashes.

## Interpretation and limits

Retain this rewrite as a measured planner candidate for sufficiently large crops and images.
The three fixture sizes do not establish a universal selection threshold.
The result is an experimental PyTorch graph optimization through a real classifier.
Public CPG inspection integration, representative industrial data, concurrency, Jetson, and ROS validation remain pending.
This run alone does not close the full two-workload release gate.

GPU `h1svmmclpuiqhz` used the independent CPU controller `7abcbhr2jj52iw`.
The controller armed a deadline of 21:08:26 UTC, or 2:08:26 p.m. Pacific, before GPU provisioning.
After verified export, the GPU was terminated early. Provider reads confirmed both Pods and the temporary secret were absent.
Current-session billing had not posted at cleanup. Reserve $0.10 including disk until reconciliation.

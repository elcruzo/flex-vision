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

## Status

Prepared. GPU measurements pending.

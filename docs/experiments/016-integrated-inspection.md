# Experiment 016: integrated inspection CUDA acceptance

## Protocol

Validate `planned-full` and `planned-roi` through the real CUDA classifier at the same clean revision as the independent baseline.
Retain the existing tensor, logit, top-five, and semantic tolerances.
Compare the integrated ROI path with the baseline using experiment 014's alternating four-repeat, 300-frame protocol.
Require at least 20% lower 4K preprocessing p50 or p99 and no pooled host p99 regression above 5% on any fixture.
Capture separate traces for both integrated strategies. Do not use profiled latency for the comparison.
Check CUDA edge crops, retained outputs, and precision rejection through the public API.
Record all results, including failed criteria. No automatic strategy threshold follows from three fixture sizes.

## Status

Prepared. Integrated CUDA evidence is pending.

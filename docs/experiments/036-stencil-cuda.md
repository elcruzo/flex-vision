# Experiment 036: specialized stencils through FP16 TensorRT

The complete v2 gate passed 288 executions on an RTX 4090.
All four stencil sizes passed positive and negative asymmetric kernels and dense sigma-1.2 Gaussian coefficients.
Six fixtures, contiguous and reversed-channel layouts, and synchronous and submit execution were covered.
Exact FP16 tensor bits matched the independent NumPy oracle.
Dense TensorRT outputs, decoded detections, retained earlier outputs, and stream drainage passed.
The measured clean revision was 014e17298a9fdf4f656ed90d79cd0144b52ebb62.
The engine hash is recorded in the retained reports.
This establishes fixed CUDA stencil correctness, including Gaussian, rather than general stream or platform acceptance.

## Matched complete-inference timing

The timing gate passed 6,400 completed inference checks across two resident 1080p photographs, four positive asymmetric stencil sizes, and CPG/PyTorch paths.
Four alternating blocks contained 100 samples per path and scenario, after 100 warmup executions.
Both paths used the same fixed FP16 TensorRT engine and synchronous owned-output behavior.
Every timed inference passed dense and detection checks outside its measured interval.
The independent local verifier reproduced complete coverage and timing percentiles from exported raw samples.

CPG preprocessing p50 was 25–31% lower than grouped PyTorch convolution followed by unfused detector preprocessing.
Complete host p50 improved by approximately 6–9%.
These are fixed-run observations. Gaussian timing, vendor stencil comparisons, independent repeats, and sustained multi-camera behavior remain pending.
Post-interval validation can influence the following memory state. No causal isolation or universal fastest-strategy claim follows.
The full two-dimensional direct kernel remains the selected candidate; shared-memory and separable implementations remain unevaluated.
See data/036/stencil-comparison-audit.json for exact per-scenario p50, p95, and p99 values.

## Export and cleanup

All 40 remote artifacts matched their hashes before termination.
GPU vvv9bap8a9u75x termination was verified at 02:10:55 UTC on October 11.
Controller 7ok804tzxpls0k and temporary credential cleanup passed at 02:11:16 UTC.
Fresh API reads returned no pods and no secrets.
Retained reports and raw timing rows exclude model weights, engines, and environment-bearing execution logs.
Full ignored artifacts remain local in benchmark-results/iteration-036-yolo.
See data/budget-after-036.json for the conservative lifetime ceiling under the existing $15 authorization.

No stencil-specific Nsight residency capture was collected in this run.
No Jetson, ROS, live-camera, reusable-workspace, batching, general planner, or autotuning acceptance follows.
Earlier expired and startup failures remain in experiments 026 and 034.

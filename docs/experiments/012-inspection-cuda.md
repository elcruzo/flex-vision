# Experiment 012: CUDA inspection baseline

This run establishes a complete GPU reference for the inspection workload. It is a baseline, not a CPG performance claim.

## Workload

Pinned Chelsea fixtures were processed as BGR8 HWC frames through Gaussian filtering, sharpen, ROI crop, resize, ImageNet normalization, and FP32 NCHW conversion. The resulting tensor was consumed directly by CUDA MobileNetV3 Small (`IMAGENET1K_V1`). Three fixtures were used: the original image, an odd-sized ROI canvas, and a 4K ROI canvas.

The measured revision was `6605257894e1f7b47057428af2993eab5bc3f817`. The model, fixture, runner, and reference hashes are recorded in `benchmark-results/iteration-012-inspection/correctness/report.json`.

## Correctness

All three cases passed the independent NumPy reference checks and real CUDA classifier inference. The tensor absolute-error limit was `0.0002`; the largest observed tensor error was `0.0000461`. Logit limits were `0.001` absolute and `0.0001` relative; ordered top-five predictions also matched. The absent-dog semantic negative control failed as designed, confirming that the semantic assertion is active.

## Baseline measurements

The latency collection used four repeats of 300 frames per fixture, with 30 warmups before each repeat: 3,600 samples total. It measures a resident-input PyTorch reference, excludes camera acquisition and upload, and does not claim a CPG improvement.

| fixture | preprocess p50 / p99 (ms) | inference p50 / p99 (ms) | stream total p50 / p99 (ms) |
| --- | --- | --- | --- |
| Chelsea | 0.144 / 0.170 | 1.988 / 2.251 | 2.134 / 2.395 |
| odd ROI | 0.146 / 0.177 | 2.003 / 2.171 | 2.149 / 2.336 |
| 4K ROI | 8.003 / 8.096 | 1.010 / 1.044 | 9.012 / 9.141 |

The 4K fixture makes preprocessing the dominant measured stage. These values are workload and environment evidence, not portable performance promises.

## Residency trace

The separate Nsight Systems capture contains three completed `inspection_to_classifier` ranges. Each range contains 199 GPU kernels, zero memcpy events, and zero recorded memcpy bytes. Upload, model setup, validation downloads, and warmups were outside the range. The raw report, SQLite export, and machine-readable summary are in this directory.

## Reproduction and limits

The full artifact set is accompanied by `remote-sha256sum.txt`. The run used an NVIDIA L4, CUDA 13.0, PyTorch 2.9.1+cu130, TorchVision 0.24.1+cu130, Python 3.12.3, and Linux. It validates a PyTorch reference path only; it does not establish production CUDA operator support, ROS transport, TensorRT integration, power, bandwidth, or a two-workload success gate. The next optimization decision should target the 4K preprocessing cost while preserving the global-border and finite-precision semantics.

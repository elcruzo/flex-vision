# Experiment 024: synchronous detector stream ownership

## Outcome

All 48 real TensorRT executions passed on RTX 4090.
The twelve cases cover six fixtures with CuPy and PyTorch inputs, each with four cycles.
Every cycle passed exact FP16 tensor, dense-output, final-detection, and retained-output checks.
Three distinct non-default streams performed production, preprocessing, and consumption.
All streams drained successfully before owner release.
Accept this fixed synchronous stream scenario. Do not claim asynchronous execution or overlap.

## Environment and ordering

Measured source: `aeb370fd24a5a237455592fb7e8a93be5ca7bb0b`, with a clean checkout.
The host used RTX 4090, driver 580.178.04, and 24,564 MiB device memory.
Packages were Torch 2.9.1+cu130, CuPy 14.2.0, CV-CUDA 0.18.0, and TensorRT 10.13.3.9.post1.
The engine SHA-256 was `f3fc87a450b85f8478334173c35709dc53aaebe99ca29409358e05a05d45878e`.
Model preparation and licensing follow YOLO_VALIDATION.md.

The producer created the GPU input and recorded its readiness event.
The preprocessing stream waited on that event before reading CuPy or DLPack input.
CPG synchronized before returning. That return established output readiness for the separate consumer stream.
The TensorRT consumer also synchronized before returning.
A consumer completion event ordered later snapshot work on the preprocessing stream.
The harness retained input owners through completion and checked prior owned outputs after later calls.

This scenario checks correct explicit ordering around a synchronous implementation.
It does not isolate DLPack's implicit handoff because both input paths also use the explicit producer event.
It does not test pending consumers, caller-owned slot recycling, early source-owner release, concurrent TensorRT contexts, or cancellation.
Those remain acceptance requirements for the proposed asynchronous contract.

## Controls and audits

The same host regenerated six numerical fixtures and 60,000 serial baseline samples.
The local NumPy audit passed all 18 candidate/fixture combinations and independently decoded detections.
The local stream coverage audit exactly matched the remote report: twelve cases and 48 checked cycles.
That report audit validates recorded coverage. It does not independently recompute the stream scenario's tensors.

The separate default-path trace passed its known 4,096-byte device-to-host control and complete-range checks.
Small metadata transfers remain visible. It is not a capture of the three-stream scenario or proof of overlap.
Retain these qualifications when discussing GPU residency.

| Fixture | Candidate | Preprocess p50, ms | Complete host p50 / p99, ms |
| --- | --- | ---: | ---: |
| Person 1080p | CPG | 0.084640 | 0.871679 / 0.898481 |
| Person 1080p | PyTorch | 0.098112 | 0.888085 / 0.915112 |
| Person 1080p | CV-CUDA | 0.109696 | 0.905483 / 0.934307 |
| Cat 1080p | CPG | 0.084800 | 0.872556 / 0.899512 |
| Cat 1080p | PyTorch | 0.098368 | 0.888991 / 0.915082 |
| Cat 1080p | CV-CUDA | 0.109824 | 0.905712 / 0.934167 |

These are serial resident-input baseline timings, not three-stream performance or live-camera latency.

## Retained evidence and reproduction

The compact bundle retains raw compressed baseline samples, stream reports, environment manifests, audit results, and sanitized trace evidence.
Weights, engines, dense tensor dumps, and the original environment-bearing trace remain ignored.
Full numerical recomputation needs those local tensor dumps or regeneration.

```bash
python scripts/verify_yolo_stream_ownership.py docs/experiments/data/024/stream-ownership.json --output local-stream-verification.json
```

The GPU entry point is `scripts/run_yolo_stream_experiment.sh` under the independent direct controller.
Local focused checks passed eleven evidence/coordinator checks before hardware execution.
Local checks alone did not establish this acceptance.

## Cleanup and budget

The coordinator verified all 69 exported files at 23:03:51 UTC on October 9, 2026.
GPU `d3p1cf4fjk2v30` was absent at 23:03:53 UTC.
Controller `s3zi8nijcosqli` and secret `vwd2rxt8fq` cleanup was verified at 23:04:14 UTC.
The fresh 23:04:18 UTC inventory found no Pods and confirmed all three resources absent.
See [cleanup evidence](data/024/cleanup.json).

Posted billing was $8.783250520227739 at the final read.
Keep previous reservation headroom and this lease's $0.65 reservation until reconciliation.
Counting the posted increase without offsetting old reservations gives a conservative bound of $12.743217323611702, below the authorized $15.
Reservations are not actual spend. No rental remains active.

## Next decision

Implement an explicit completion object with owner retention before removing the preprocessing host wait.
Keep the public synchronous default unchanged.
Validate delayed producers and consumers, source lifetime, output recycling, and shutdown through actual inference before performance selection.
This experiment does not complete asynchronous, ROS, Jetson, or release acceptance.

# Experiment 025: owned-output asynchronous submission correctness

## Outcome

The corrected run passed all 48 real FP16 TensorRT executions on RTX 4090.
Twelve cases cover six fixtures and CuPy/PyTorch inputs, with four cycles each.
Every cycle passed exact tensor, dense-output, final-detection, and retained-output checks.
Caller source references were released after submission and before consumption.
The completion object retained the original and imported input owners until preprocessing completed.
Three distinct native CuPy streams remained alive through successful shutdown.

Accept this fixed owned-output correctness scenario. Keep the synchronous default unchanged.
This does not establish asynchronous inference, delayed-consumer behavior, slot recycling, or performance benefit.

## First failed run and stream compatibility

The first run failed before async inference because Torch 2.9.1 streams lacked the CUDA stream protocol method `__cuda_stream__`.
CuPy Stream.from_external rejected those objects. The default-path controls ran, but no async acceptance came from that attempt.
The coordinator exported 67 hash-verified files and removed GPU `4q0ttle5butxnf` at 23:49:37 UTC on October 9, 2026.
Controller `ks669mn6w8rpm5` and secret `zc1i6yfy81` were absent at 23:49:48 UTC.
The subsequent 23:49:58 UTC inventory was empty. The compact failure evidence is in [025-failed](data/025-failed/failure.json).
Repeated SSH polling failures delayed progress collection. The coordinator did not resubmit the experiment.

The corrected harness owns native CuPy streams and wraps their pointers with Torch ExternalStream.
It retains the native owners until all stream work drains.
[PyTorch 2.9 documentation](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.ExternalStream.html) requires the caller to retain the external stream owner.
This pinned-version correction changes the harness, not the preprocessing numerical policy.

## Corrected environment and ordering

Measured source: `003da1c3d2ecb1b2ef6e1b61d7aea4eb7dc1169e`, with a clean checkout.
The host used RTX 4090, driver 595.91.07, and 24,564 MiB device memory.
The pinned environment used Torch 2.9.1+cu130, CuPy 14.2.0, CV-CUDA 0.18.0, and TensorRT 10.13.3.9.post1.
The engine SHA-256 was `8b95d4dbab4127f9340f8bed3d4ed5f659716b2bd6cbe8c619e918eb6f3e6678`.
Model preparation and licensing remain as specified in YOLO_VALIDATION.md.

The producer records a ready event. The preprocessing stream waits before submission.
`Pipeline.submit` queues preprocessing and records its completion event while retaining owners.
The caller deletes its frame/source references before consumer work.
`wait_on` queues the consumer dependency. The TensorRT consumer then executes and synchronizes before returning.
The harness closes the submission after consumer completion and checks retained default outputs across subsequent calls.
The explicit producer event orders both input paths. This does not isolate DLPack's implicit handoff.

This tests the implemented event dependency through real inference.
It does not deliberately keep a consumer pending during close, prove that preprocessing was still pending at source deletion, or test device failures.
Cold compilation, DLPack internals, and host scheduling can introduce delays outside the explicit runtime host wait.
No trace from this scenario proves the absence of all host waits.

## Independent audits and retained evidence

The local report audit matched the remote result exactly: twelve cases and 48 checked async-submit cycles.
Its mode check rejects synchronous evidence presented as async acceptance.
This coverage audit does not independently recompute the async tensors.
The separate baseline numerical audit passed all 18 candidate/fixture combinations and independently decoded detections.
The regenerated 60,000 baseline timing rows also passed their audit.
Those timings exercise the synchronous path. They are not an async comparison.
Absolute baseline times varied from earlier hosts. Do not attribute that difference to async submission.

The sanitized baseline trace passed its known 4,096-byte device-to-host control and complete-range checks.
Small metadata transfers remain visible. This trace does not capture the async submission scenario or demonstrate overlap.
Raw compressed baseline samples, environment manifests, stream reports, audit results, and the sanitized trace remain in the compact bundle.
Weights, engines, dense dumps, and original environment-bearing traces remain ignored.
Full independent tensor rechecking needs the local dense dumps or regeneration.

```bash
python scripts/verify_yolo_stream_ownership.py docs/experiments/data/025/async-ownership.json --mode async-submit --output local-async-verification.json
```

The hardware entry point is `scripts/run_yolo_async_experiment.sh` under the direct independent controller.
The twelve focused local lifecycle/evidence checks passed after the correction. They did not replace hardware execution.

## Export, cleanup, and budget

The corrected run exported 69 hash-verified files at 23:58:49 UTC on October 9.
GPU `0e3cultzi45uw0` was absent at 23:58:51 UTC.
Controller `kbemgn020opvxw` and secret `cg6dkmn5rq` cleanup was verified at 23:59:03 UTC.
The fresh 23:59:19 UTC inventory was empty and confirmed all three resources absent.
See [cleanup evidence](data/025/cleanup.json).

Posted billing remained $8.783250520227739 at the final read.
Retain two $0.65 reservations for the failed and corrected attempts, plus previous delayed-charge headroom.
The conservative cumulative bound is $14.043217323611703, within the authorized $15.
Reservations are not actual spending. No GPU rental remains active.
Reconcile billing before any further rental.

## Product implication and next gate

This closes the first real-inference correctness check for the explicit owned-output submission API.
It advances the camera-to-model integration boundary without weakening tensor semantics or source lifetime.
It is not a new measured performance breakthrough or the complete product.
Next, test delayed consumers and completion cleanup, then compare matched synchronous and submission execution with an async-specific trace.
Define timing boundaries and the p99 guard before measuring.
Reusable slots, batches, configurable stencils, tuning, ROS transport, Jetson, and live-camera acceptance remain open.

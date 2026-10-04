# Hardware and environment plan

Status: setup plan. This document does not configure a remote GPU, account, paid instance, or Jetson.
Source check date: 2026-10-04. Recheck availability and compatibility before provisioning.

## Recommended starting setup

Use the local computer for editing, reference pipelines, supported model inference, and reviewing artifacts.
Use one dedicated NVIDIA GPU on a Linux machine for the first complete detector experiment.
That machine can be owned, borrowed, or rented. Runpod Pods are a suitable candidate for remote development through SSH.
Runpod documents SSH as an option for long-running development sessions. [Runpod connection options](https://docs.runpod.io/pods/connect-to-a-pod)

Start with one GPU, not a cluster.
Choose capacity after fixing the model, image sizes, concurrency, and workspace budget.
A planning allowance of 16–24 GB GPU memory, 32 GB host memory, and 100 GB storage is a starting estimate.
It is not a measured minimum or a requirement to rent the most expensive GPU.
Leave space for datasets, engines, compiler caches, containers, and traces.

Choose an on-demand session for the first reproducible measurements.
Test actual tracing permissions before committing to longer rental time.
A persistent development machine is the preferred workflow for debugging streams, compilers, and traces.
Serverless inference endpoints are not the initial test environment.

## What each environment can prove

| Environment | Appropriate evidence | Evidence it cannot replace |
| --- | --- | --- |
| Local computer without NVIDIA CUDA | Documentation, graph/configuration logic, reference generation, report inspection | Actual CUDA execution, GPU performance, CUDA interoperability |
| Dedicated Linux cloud GPU | CUDA runtime, framework handoffs, real TensorRT inference, replay load, tuning, available profiler traces | Jetson power, CSI transport, Jetson shared-memory behavior |
| Desktop Blackwell GPU | Desktop architecture comparison and plan selection | Orin or Thor results |
| Jetson AGX Orin | Orin runtime, ROS transport, live camera behavior, power and thermal runs | Thor performance |
| Jetson AGX Thor | Thor runtime, ROS transport, live camera behavior, power and thermal runs | Orin performance |
| Remote Jetson lab with attached cameras | Target results when capture, controls, telemetry, and access are sufficient | Physical-camera claims if the lab only offers file replay |

Keep Orin and Thor in the required platform matrix even if only one is initially available.
DGX Spark is a possible later comparison platform from the ecosystem discussion, not a replacement for either Jetson target.
Cloud replay is useful early evidence. Label it as replay in every report.

## Cloud instance selection checklist

Before renting, record:

1. GPU model, architecture, memory, and sharing or partitioning policy.
2. Driver version and compatibility with the selected CUDA toolkit and frameworks.
3. Host CPU, RAM, storage, and expected CPU contention.
4. SSH access, image choice, and package-install permissions.
5. CUDA tracing support and any restricted profiling counters.
6. Persistent storage, artifact export, and restart behavior.
7. Current compute, storage, and transfer prices.
8. Maximum run duration, interruption policy, and shutdown procedure.

Compare providers using these capabilities, not only the GPU model or hourly price.
If a Runpod instance cannot provide required profiling access, use another instance or a controlled GPU workstation.
Do not interpret unavailable counters as zero activity.

Runpod offers network volumes that persist independently of a Pod.
Choose storage deliberately and export valuable results before deleting compute. [Runpod network volumes](https://docs.runpod.io/storage/network-volumes)

This plan does not fix a price quote.
Calculate session cost from current rates, planned compute hours, retained storage, and transfer charges.
Before paid provisioning, obtain the user's provider/account choice and spending limit.
Credentials, billing details, and private keys must stay outside Git.
This planning task does not require purchasing a GPU session.

## Reproducible software environment

Create a separate pinned environment for each platform family.
Do not assume an x86 image works on Jetson.
Record the base image digest or host installation manifest, not only a mutable image tag.

The manifest must include:

- OS, CPU architecture, kernel, driver, GPU identity, and compute capability.
- CUDA toolkit/runtime, compiler, Python, and package lock.
- CuPy, PyTorch, TensorRT, model exporter, and engine build options.
- ROS distribution, middleware, Isaac ROS, VPI, CV-CUDA, and OpenCV where used.
- Nsight Systems and other measurement tool versions.
- JetPack, power mode, clocks, and cooling settings on Jetson.
- Repository revision, configuration hashes, and model/dataset hashes.

An installed CUDA toolkit does not establish driver compatibility.
A successful import does not establish successful kernel execution or interoperability.
Run the preflight and complete inference checks in [TESTING.md](TESTING.md).
Build model engines on the intended target unless verified portability rules permit reuse.

NVIDIA's Isaac ROS 5.0 setup table lists Orin and Thor with JetPack 7.2 and at least 128 GB NVMe storage.
Use that table as a setup reference, then verify the installed combination. [Isaac ROS setup](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html)
Do not automatically replace system packages on a working Jetson to match a generic container recipe.
The exact CUDA 13, Lyrical, and framework combination remains a G0 installation check.

## Jetson bench equipment

Arrange access to both required Jetson platforms over the project, through purchase, borrowing, or a lab.
The first live-camera bench needs:

- A Jetson board with supported power supply and active cooling.
- NVMe capacity for the environment, captures, models, and traces.
- Wired network and SSH access for development and artifact collection.
- A supported CSI camera for the flagship capture path.
- Calibration data and a repeatable scene or recorded corpus.
- A USB or GigE camera when evaluating those input paths.
- A camera hub, carrier, or network arrangement suitable for the intended multi-camera load.
- Platform power telemetry and, when available, an external power meter.

Check camera driver, connector, encoding, capture rate, timestamp, and buffer ownership before buying accessories.
One USB camera does not validate the CSI zero-copy path.
Six concurrent replay streams do not validate six physical camera interfaces.
Use replay to isolate compute, then validate the physical configuration separately.

Nsight Systems has different target packages for Jetson and workstation/cloud systems.
Use the matching edition and verify collection on the target. [Nsight Systems installation](https://docs.nvidia.com/nsight-systems/InstallationGuide/)

## Setup completion record

A target is ready only when its setup record links to:

1. The environment manifest and exact repository revision.
2. A successful CUDA operation and asynchronous stream check.
3. A CuPy/PyTorch handoff followed by actual TensorRT inference.
4. A trace containing expected kernels and a known transfer.
5. A persistent artifact write and successful export.
6. The allowed run budget and resource cleanup procedure for rented hardware.

Mark each field as passed, failed, blocked, or not applicable with evidence.
Do not declare the environment ready from `nvidia-smi` alone.


## Current Mac and local-first development

Observed on 2026-10-04:

| Item | Local observation |
| --- | --- |
| Computer | MacBook Air, Apple M5 |
| CPU / GPU | 10 CPU cores / 10 GPU cores |
| Memory | 16 GB unified memory |
| OS / architecture | macOS 26.5.1 / arm64 |
| Current python3 environment | PyTorch 2.8.0 |
| MPS checks | Built and available, small GPU arithmetic check passed |

These observations describe the development machine, not a supported CPG deployment target.
The smoke check does not establish model compatibility or benchmark performance.
Do not store machine serial numbers, device UUIDs, or private account identifiers in environment reports.

PyTorch's MPS backend can execute supported operations on the Apple GPU.
Check availability and actual model execution in the chosen environment. [PyTorch MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html)
The Apple GPU does not execute CPG's NVIDIA CUDA kernels.
MPS results do not establish CUDA stream behavior, CUDA Array Interface support, TensorRT compatibility, or Jetson performance.

### Work to complete locally

1. Define graph construction, operator semantics, YAML validation, and plan inspection.
2. Create deterministic frames and the independent CPU reference pipeline.
3. Run a small pretrained vision model through reference preprocessing and real PyTorch inference.
4. Try MPS for that same reference pipeline and model where its operators work.
5. Compare tensors, model outputs, and detection coordinate mapping against the CPU reference.
6. Build the replay source, result bundle, comparison report, and failure diagnostics.
7. Test model/configuration loading and prepare pinned artifacts for the NVIDIA run.

This local loop runs real model inference. It is more than isolated unit tests.
Keep it explicitly labeled as a reference path, not the CPG CUDA backend.
Do not build a production Metal backend merely to occupy the local GPU.

Use one small model and modest batches initially.
The 16 GB memory is shared by the OS, applications, CPU work, and GPU work.
Monitor memory pressure and reduce input buffering before trying large batches or six 4K streams.
MPS acceleration is optional for the reference path. CPU execution remains useful when an operator is unsupported.
Record any CPU fallback. Do not report a mixed path as entirely GPU-resident.
Use an isolated project environment when establishing the reproducible setup rather than relying on the observed global installation.

### When to move work to NVIDIA hardware

Move to a rented or accessible NVIDIA GPU as soon as the next experiment needs CUDA compilation, dispatch, or framework interoperability.
Also move when testing TensorRT, CUDA fusion, tuning decisions, or NVIDIA performance.
Do not wait until the entire frontend is complete to discover backend constraints.
A suitable first remote checkpoint is the smallest detector path with matching reference tensors and real TensorRT output.

Move to actual Jetson hardware when testing camera buffers, ROS CUDA transport, power, thermals, and the live flagship demo.
Keep the Mac as the editor, SSH client, reference runner, and report viewer throughout the project.
This sequence limits rental time without postponing the GPU evidence that drives the design.

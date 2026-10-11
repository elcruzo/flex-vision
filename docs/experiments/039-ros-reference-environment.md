# Experiment 039: ROS-compatible local CPU inference environment

The separate ARM64 ROS Lyrical reference image built and passed framework/ROS imports.
The pinned base image uses Python 3.14.4. It cannot reuse the existing Mac PyTorch 2.8 environment.
The derived environment uses PyTorch 2.10.0+cpu, TorchVision 0.25.0+cpu, and NumPy 2.4.2 inside a virtual environment with installed ROS bindings visible.
Existing Mac and CUDA experiment environments were not changed.
The [upstream 2.10 announcement](https://dev-discuss.pytorch.org/t/pytorch-2-10-0-general-availability/3291) and [Python 3.14 support notice](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276) informed this explicit ABI choice.
Actual official CPU wheels installed successfully, and torch, torchvision, and rclpy imported together.

The existing Gaussian reference harness then passed all four sizes through real pinned MobileNet classifier inference.
It checked NumPy tensors, grouped PyTorch tensors, classifier logits, top-five ordering, cat probability, and Gaussian YAML/file/Python graph equality.
The previous declared tolerances remained unchanged.
See data/039/classifier.json for exact source, model, image, configuration hashes, numerical errors, and scope.
Resolved Python and system packages and the local image identity are retained.
Top-level package versions and the base are pinned; apt/transitive dependency resolution is recorded, not fully locked.

The initial base-image virtual environment attempt failed because ensurepip was absent.
The derived image installs python3-venv explicitly. Its apt step installed four packages and upgraded none.
No ROS package upgrade was requested.
This reference image is local and is not a published package or release.

This accepts environment compatibility and an explicit classifier reference case.
The classifier input was not received from a ROS subscription in this experiment.
Actual ROS Image to CPG to classifier integration is the next gate.
No GPU execution, CUDA-backed transport, camera capture, TensorRT, or performance acceptance follows.
The local disposable execution container ended. No paid resources were used.

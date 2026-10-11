# Standard Image transport probe

This experimental package is not the cuda_preprocess production node.
It first checks the ordinary CPU Image path on ROS 2 Lyrical before adding native CUDA handles.
No image preprocessing, model inference, CUDA transport, camera capture, or performance advantage is implemented here.

The independent source publishes 30 deterministic RGB8 frames at a nominal ten-per-second timer rate.
Each frame is 1920×1080 with 16 padding bytes per row.
The subscriber checks every pixel, all padding, encoding, dimensions, timestamp interval, and sequence identity.
It records raw publish-to-validation latency samples. Those samples include CPU payload validation and are not preprocessing latency.
Discovery, shutdown, and a 30-second process deadline are explicit.
It uses reliable QoS with queue depth four. Any missing sequence prevents acceptance.

The source requires sensor_msgs/Image.data to have the native rosidl::Buffer type at compile time.
This establishes schema compatibility only. CPU buffers still carry and serialize CPU pixel data.
A subsequent CUDA probe must check the native read/write handle path, promotion copies, delayed readers, and transfer traces.
See docs/ROS_BUFFER_AUDIT.md for the pinned API and remaining acceptance requirements.

Build with the installed ROS development tools. The local container harness is scripts/run_ros_cpu_probe.sh.
The harness requires a new CPG_ROS_RESULTS directory inside the container and the repository mounted at /workspace.
No package or release is published. Project licensing remains undecided; the package license marker records that state.

The publisher accepts optional pixel, padding, stride, or sequence fault modes.
It corrupts the sixth message before publication. The subscriber must reject the transported payload and return failure.
The harness records both process exit codes. A build error or timeout is not a passing fault control.
Experiments 037 and 038 record the valid CPU run and four actual transport rejection controls.

Dockerfile.reference prepares a separate CPU inference environment for the container's Python 3.14 ABI.
It uses PyTorch 2.10/TorchVision 0.25 CPU wheels in a virtual environment that can access installed ROS bindings.
It does not alter the existing Mac or CUDA benchmark environments.
The image checks ROS/framework imports and records resolved Python/system packages.
A successful image build would establish environment compatibility only; real ROS preprocessing and model acceptance remain separate.
The initial base-image virtual-environment attempt lacked ensurepip. The derived image installs python3-venv explicitly.

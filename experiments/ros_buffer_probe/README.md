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

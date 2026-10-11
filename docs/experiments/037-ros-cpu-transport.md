# Experiment 037: native standard Image CPU transport on Lyrical

The bounded native publisher/subscriber probe built and passed in a pinned official ARM64 ROS Lyrical container on the Mac.
Two separate processes using rmw_fastrtps_cpp exchanged 30 deterministic RGB8 frames.
Each frame was 1920×1080 with a 5,776-byte row stride and 16 padding bytes per row.
The subscriber checked every pixel and padding byte, exact sequence identity, encoding, dimensions, storage size, and nonnegative timestamp interval.
All 30 frames passed with no observed sequence drops.
Both processes exited successfully within their bounded deadlines.

The C++ compile-time assertion confirmed that this installed sensor_msgs/Image data field has the native rosidl::Buffer<uint8_t> type.
This validates the standard-message CPU schema and transport step. It does not establish CUDA backend activation or GPU residency.
Payload generation, transport, and validation all used CPU buffers in this experiment.
No CPG preprocessing or classifier/TensorRT inference was connected.

See data/037/provenance.json for the clean measured revision, source hash, image digest, middleware, and exact scope.
The build log and installed ROS package versions are retained with the publisher/subscriber reports.
Raw latency samples measure publication timestamp through full CPU pixel validation.
They are container observations, not preprocessing latency, live-camera latency, or a performance comparison.
The source was built without a requested optimization profile; do not use these values as an optimized CPU baseline.

The initial setup attempt failed on an optional unset ROS shell variable before build.
The harness now lets ROS initialize optional variables before restoring strict shell checks.
The first compilation attempt then failed because Lyrical removed ament_target_dependencies.
Current exported dependency targets corrected that build. Both failures are retained.
The successful clean revision is 769f5f7; its full hash is in the provenance record.

The disposable container ended and was removed automatically. No paid infrastructure was used.
Project licensing is undecided; this experimental package is not published.
The next native CUDA probe must retain read/write handles, count CPU promotion copies, test delayed consumers, and collect controlled transfer traces.
CPU-to-GPU preprocessing, stamped tensor output, real inference, queue overload/shutdown cases, and Jetson camera acceptance remain required.

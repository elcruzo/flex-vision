# Experiment 045: letterbox geometry through the installed ROS node

Eight real classifier frames passed the configured preprocessing executable with geometry-bearing stamped tensors.
The clean measured source was 1759dd1.
The envelope carries the actual rounded resized width and height, plus left and top padding.
The inference consumer reconstructed the geometry and compared it with the independent NumPy reference.
It also mapped the full resized image rectangle back to the exact source-frame rectangle.
All eight geometry and coordinate checks passed alongside the unchanged tensor, classifier, and semantic checks.
The independent audit again matched timestamps and frame identity across all three processes.
All process exit codes were zero. Reports and build logs remain in data/045.

This is actual CPU ROS transport and classifier inference in the isolated ARM64 Lyrical environment.
It validates letterbox metadata for this fixed fixture, not real detector outputs or arbitrary geometry.
The reference host-copy path remains explicit. No CUDA transport, native buffer handle, live camera, or Jetson acceptance is claimed.
Geometry validation rejects invalid dimensions, layouts, and resized-image bounds before coordinate mapping.
A later detector consumer still requires matched real detections and independent source-coordinate checks.
The envelope schema remains experimental and is not a stable public interface.

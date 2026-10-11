# Experiment 044: installed configured preprocessing executable

The cuda_preprocess package built and registered preprocess_node through ROS package discovery.
The actual ros2 run executable loaded examples/ros-classifier.yaml with explicit cpu-reference execution.
A separate publisher and classifier subscriber completed eight padded BGR8 frames.
The fixed graph used Gaussian 5, letterbox 224, ImageNet normalization, and FP32 NCHW output.

All tensors matched the independent NumPy oracle at unchanged tolerance.
Classifier logits, top-five classes, and the cat semantic check passed.
The independent audit matched original timestamps and frame IDs across publisher, installed node, and inference subscriber.
All three processes returned zero at clean source revision 29627e8.
Reports, executable registration, and build logs remain in data/044.

This validates the configured experimental CPU executable through real inference.
It does not validate clean Python packaging, native cuda_preprocess_node composition, CUDA transport, TensorRT, live cameras, or Jetson.
The local ARM64 Lyrical reference environment remains isolated from the pinned CUDA environments.
No paid resources were used.

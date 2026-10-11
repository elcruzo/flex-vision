# Experimental configured ROS node

The cuda_preprocess package installs the preprocess_node executable.
Its current backend is explicit CPU reference execution. It does not implement CUDA-backed transport or the native component API.
The stamped output uses the experimental cpg_tensor_probe envelope around standard ExperimentalTensor.
See [configured-node evidence](experiments/044-configured-ros-node.md).

Build the pinned tensor_msgs dependency, experiments/ros_tensor_probe, and ros/cuda_preprocess with colcon.
Source each overlay. Install the CPG Python source and its reference dependencies in the interpreter used by the executable.
The tested local container supplies the interpreter through PATH and the source through PYTHONPATH.
This is not a clean-distribution installation claim. Project licensing remains undecided.

```bash
ros2 run cuda_preprocess preprocess_node \
  --config examples/ros-classifier.yaml \
  --backend cpu-reference
```

The default input topic is cpg_reference/image. The default output topic is cpg_reference/tensor.
Override them with --input-topic and --output-topic.
The reliable QoS depth defaults to two. Change it with --queue-depth.
The node waits for an output subscriber before subscribing to input.
A missing consumer during a callback stops the node with an error.
Discovery and a bounded DDS queue do not prove downstream completion or a production drop policy.

Input is validated RGB8 or BGR8 host storage with padded rows supported.
The YAML graph must fit the currently implemented fixed pipeline contract.
The node preserves the source header, encoding, and dimensions in the output envelope.
CPU preprocessing and tensor serialization perform host payload copies.
The reference path must not be described as GPU-resident or a measured CUDA upload fallback.

Use --max-frames and --deadline for bounded replay experiments.
Use --report for per-frame headers and shape records. Existing report paths are rejected.
After the declared frame limit, the node allows one second for publication drainage.
This delay is not an acknowledgment from the inference consumer.
Errors stop execution. Production recovery, completion tracking, live-camera backpressure, and CUDA ownership remain pending.

# ROS CUDA buffer integration audit

Status: upstream API review only. No CPG ROS package, transport build, or robot acceptance is claimed.
The inspected ros2/rosidl_buffer_backends source revision is d7cd9642d77a1d64fd85f25ba0bf96e108401900.
The temporary checkout was inspected without importing upstream implementation code into CPG.

The [Lyrical release notes](https://github.com/ros2/ros2_documentation/blob/rolling/source/Releases/Release-Lyrical-Luth.rst) describe C++ uint8 array fields as rosidl::Buffer and identify Fast DDS and Zenoh middleware support.
Pin and record the middleware as part of transport acceptance. An installed distribution name alone does not prove CUDA transport.

## Checked APIs

The [pinned CUDA buffer API](https://github.com/ros2/rosidl_buffer_backends/blob/d7cd9642d77a1d64fd85f25ba0bf96e108401900/cuda_buffer_backend/cuda_buffer/include/cuda_buffer/cuda_buffer_api.hpp) exposes allocation and stream-specific input/output handles.
The current source directory is cuda_buffer_backend/cuda_buffer; an earlier guessed top-level cuda_buffer path returned 404.
The installed include remains cuda_buffer/cuda_buffer_api.hpp.

For an already CUDA-backed input, from_input_buffer returns a read handle on that buffer.
For a CPU-backed input, it allocates a CUDA buffer and queues a host-to-device payload copy.
The promoted buffer is retained by the handle.
The CPG CPU fallback must therefore label and measure this upload. It must not report the promotion as resident zero-copy input.

The [handle implementation](https://github.com/ros2/rosidl_buffer_backends/blob/d7cd9642d77a1d64fd85f25ba0bf96e108401900/cuda_buffer_backend/cuda_buffer/include/cuda_buffer/cuda_buffer_handle.hpp) records read/write completion events on the supplied stream.
Keep preprocessing reads on that stream, or establish explicit dependencies before destruction.
A Python raw-pointer view alone cannot replace the native message and handle owners.
A bridge must retain those owners through the final queued read and drain on shutdown/error.

## Tensor consumer contract

The [pinned ExperimentalTensor schema](https://github.com/ros2/rosidl_buffer_backends/blob/d7cd9642d77a1d64fd85f25ba0bf96e108401900/tensor_msgs/msg/ExperimentalTensor.msg) carries DLPack-style dtype, shape, strides, byte offset, and buffer-backed storage.
It can describe the required FP16 NCHW model input without pretending that NCHW is a conventional Image encoding.
It has no camera timestamp or frame header.
A stamped wrapper is a candidate, not a selected production interface. Check downstream inference compatibility before fixing the message schema.
The [conversion bridge](https://github.com/ros2/rosidl_buffer_backends/blob/d7cd9642d77a1d64fd85f25ba0bf96e108401900/torch_conversions/torch_conversions/include/torch_conversions/torch_conversions.hpp) is another integration candidate; CPG does not yet expose it.

## Next transport acceptance

Run a bounded publisher/subscriber probe on Linux with an NVIDIA GPU and the pinned ROS/backend sources.
Test separate processes, explicit producer/consumer streams, padded image strides, delayed readers, and shutdown while reads remain pending.
Run the same pixel fixtures through CUDA-backed and ordinary CPU Image messages.
Verify metadata and numerical results independently, then capture transfers with a known-copy control.
Record promotion bytes, drops, queue depth, and complete latency instead of inferring them from handle construction.
Only after this probe passes, connect the preprocessing runtime and real TensorRT inference.
Jetson camera capture, target builds, power, and the flagship launch remain separate required gates.
No build was attempted on this Mac: its MPS GPU cannot execute the CUDA transport path.

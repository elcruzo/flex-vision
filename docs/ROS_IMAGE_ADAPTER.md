# CPU ROS Image adapter

`cpg.ros.cpu_image_view(message, expected_encoding="bgr8")` borrows host pixels from a standard Image message.
It returns a read-only HWC uint8 NumPy view, an immutable metadata snapshot, and the retained message owner.
It does not import ROS, upload pixels, or implement a native CUDA bridge.

The adapter accepts RGB8 and BGR8 with explicit matching encoding.
Dimensions and row stride must be positive uint32 values.
Each row must contain all three channels. The byte count must equal height times row stride.
Row padding remains outside the pixel view.
The timestamp requires signed int32 seconds and nanoseconds below one billion.
The frame identifier remains available in the metadata snapshot.

Accepted payload storage is bytes, bytearray, or an unsigned-byte array.
Foreign buffer objects are rejected without array conversion.
A CUDA-backed message needs a separate native bridge that retains its buffer and stream handles.

The caller must not mutate the underlying message while a consumer reads it.
A read-only view prevents writes through that view. It does not lock the original mutable payload.
The retained owner preserves storage lifetime, but does not provide asynchronous consumer completion.

Experiment 041 passed this adapter through actual ROS subscription, CPU reference preprocessing, and classifier inference.
See [the evidence](experiments/041-ros-image-adapter.md).
Production preprocessing nodes, tensor publication, GPU transport, and Jetson acceptance remain pending.

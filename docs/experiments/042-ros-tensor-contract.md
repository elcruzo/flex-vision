# Experiment 042: standard CPU tensor serialization contract

The pinned upstream tensor_msgs package built in the existing ARM64 Lyrical reference container.
The base image did not contain this package, and its local apt index returned no matching package.
The source revision was d7cd9642d77a1d64fd85f25ba0bf96e108401900.
Upstream source remained in a temporary external checkout. No implementation source was imported into CPG.

Actual generated ROS serialization preserved every FP16 and FP32 tensor bit for shape [1, 3, 224, 224].
The borrowed CPU views retained their message owner and rejected writes through the view.
Ten malformed-message controls rejected mismatched shapes, dtype lanes, explicit strides, offsets, and truncated storage.
The checks ran again after strict integer metadata validation changed.
The source hashes, reports, and build logs are retained in data/042.
This development run used a working tree based on 120635a, not a clean experiment revision.

The writer explicitly copies NumPy host pixels into the standard tensor message.
The reader accepts only contiguous host FP16/FP32 storage with zero offset and an exact consumer shape.
This does not implement a general DLPack view or native CUDA bridge.
The message lacks camera metadata and layout fields. A separate checked integration contract remains necessary.
No separate-process transport, model inference, CUDA ownership, or camera identity acceptance is claimed by this serialization check.
The next integration must retain the Image header and validate the complete tensor-to-inference consumer path.

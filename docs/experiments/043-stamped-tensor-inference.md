# Experiment 043: Image to stamped tensor to classifier

Three separate processes passed eight complete CPU frames through ROS transport, preprocessing, and actual classifier inference.
The publisher sent padded BGR8 Image messages. The preprocessing process published a stamped standard tensor envelope.
The inference process consumed FP32 NCHW tensors through the pinned MobileNet classifier.
Gaussian sizes 3, 5, 7, and 9 each ran twice.

Independent NumPy tensor values, classifier logits, top-five ordering, and the cat semantic check passed at unchanged tolerances.
The audit compared sequence, frame identity, seconds, and nanoseconds across all three process reports.
Every header matched exactly from publisher through preprocessing to inference.
All three processes returned zero at clean revision 050d09e.
Reports, build logs, source identities, model checksum, per-frame errors, and the independent audit remain in data/043.

The first run failed because preprocessing accepted an Image before the classifier subscriber became available.
The classifier then reached its bounded deadline without an output.
All initial reports and failures remain in data/043/initial.
The correction delays Image subscription until downstream tensor discovery. Consumer loss after startup remains a failure.
The corrected run passed. Discovery does not prove completion acknowledgment or production backpressure.

This is ARM64 Docker, Lyrical/Fast DDS, and explicit CPU reference execution.
The tensor envelope uses standard ExperimentalTensor plus the original header, layout, source dimensions, and encoding.
Its schema remains experimental. The package is not published and its project license remains undecided.
CPU serialization and the inference consumer's CPU copy are explicit payload copies, not zero-copy GPU transport.
Native CUDA handles, live cameras, TensorRT ROS handoff, production preprocess_node packaging, geometry for detections, and Jetson remain pending.
No paid hardware was used.

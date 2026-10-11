# Experiment 041: owner-retaining Image adapter through classifier inference

The extracted CPU Image adapter passed eight subscribed BGR8 messages through preprocessing and actual MobileNet inference.
The clean measured revision was c852461949f243f8ce6510b1c8e9c9c667ed3806.
Gaussian sizes 3, 5, 7, and 9 each ran twice.
All transported pixels and padding matched the pinned original photograph.
Each decoded view retained its message owner and rejected writes through the view.

The unchanged NumPy tensor oracle, classifier logits, class order, and cat semantic checks passed.
Both independent processes returned zero. No timeout or skipped inference counted as acceptance.
Reports and logs are retained in data/041, including model checksum, timestamps, source identity, and per-frame errors.
The environment is the local ARM64 Lyrical/Fast DDS CPU reference image documented in experiment 039.
The disposable container ended and no paid resources were used.

This accepts the fixed CPU adapter integration.
It does not validate native CUDA buffer handles, live cameras, TensorRT, tensor publication, or production ROS nodes.
A retained CPU owner does not prove asynchronous GPU consumer safety.

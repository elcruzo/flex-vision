# Experiment 040: subscribed ROS Image through CPG reference and classifier

Two complete local runs passed actual ROS Image subscription, explicit CPG CPU preprocessing, and pinned MobileNet classifier inference.
Each run processed eight messages: Gaussian sizes 3, 5, 7, and 9 repeated twice.
The publisher sent the pinned Chelsea photograph as BGR8 with 17 padding bytes per row.
The subscriber verified the complete transported payload and padding against the original fixture before constructing a strided HWC view.
Encoding, dimensions, row stride, sequence frame identity, and received timestamps were checked or retained.

Each subscribed-image PyTorch reference tensor matched the independent NumPy oracle within 2e-4 absolute tolerance.
Classifier logits matched within 0.001 absolute and 1e-4 relative tolerance.
Top-five class ordering matched and the expected cat group passed the existing minimum-probability check.
The original and repeated clean source hashes, model checksum, framework versions, payload identities, per-frame errors, and timestamps are retained in data/040.
The positive repeat used revision 6bd085f. The initial run used dadbfb1.
The host audit separately regenerated the padded payload identity from the pinned original photograph.
Both runs returned zero from both processes, totaling 16 completed inference frames.

The semantic negative control used the same transported image and preprocessing but required the absent dog class.
It failed specifically at the classifier semantic check, with cat classes present and dog group probability near zero.
The subscriber returned code 1. The publisher detected consumer disappearance and also returned code 1.
No build error or timeout was counted as a successful control.

The scope is ARM64 Docker, ROS Lyrical/Fast DDS, PyTorch 2.10 CPU, and explicit reference preprocessing.
No pixels were GPU-backed. Downloads/uploads, TensorRT, native CUDA handle lifetime, live camera capture, tensor-message publication, and production preprocess_node integration are not validated.
This is a complete subscribed-image reference experiment, not a production CPU fallback implementation or a performance comparison.
The graph contains one Gaussian stencil followed by letterbox, normalization, and FP32 NCHW output.
The library's normal CUDA call path and earlier GPU benchmark sources remain unchanged.

The bounded harness has a 45-second node deadline and a 60-second outer publisher timeout.
Its two processes run separately inside the same pinned local image from experiment 039.
The disposable containers ended and were removed. No paid resources were used.

# Experiment 038: corrupted standard Image transport controls

The valid CPU transport control again passed 30 complete padded 1080p frames.
Four separate native publisher/subscriber runs then rejected an intentionally corrupted sixth message.
A pixel-byte corruption produced pixel mismatch. A row-padding corruption produced row padding mismatch.
Wrong stride metadata and wrong sequence identity each produced the metadata/sequence failure.
The corruptions occurred at the publisher before ordinary ROS serialization.
Each subscriber rejected the actual transported message, not an in-process mock.

All failing subscribers exited with code 1. Their publishers detected subscriber loss before completion and also exited with code 1.
No case was accepted merely because a timeout or build failed.
The valid control returned code 0 from both processes.
These outcomes are independently checked in data/038/audit.json with retained logs and exact source/container identities.
The clean source revision was a452f0a; its full hash is in the audit.
The pinned ARM64 Lyrical container and Fast DDS configuration match experiment 037.

The harness now records both child exit codes, including failure cases.
The publisher stops when a subscriber disappears after delivery has begun, while allowing initial discovery and normal final drainage.
The existing bounded process deadlines remain active.
This does not establish overload handling, recovery from a transient discovery loss, or a production queue policy.

This validates four error-detection controls for ordinary CPU Image transport.
It does not validate CUDA buffers, promotion uploads, asynchronous owner retention, CPG preprocessing, inference, or Jetson.
No paid resources were used. The disposable container was removed.

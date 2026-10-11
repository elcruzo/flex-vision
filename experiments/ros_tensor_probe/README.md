# Experimental stamped tensor probe

This package defines an experimental envelope around the pinned standard tensor_msgs/ExperimentalTensor schema.
It preserves the source Image header, layout, source dimensions, and encoding in one transport message.
It is not a production ROS interface or a CUDA owner bridge.

The three-process CPU classifier gate is scripts/ros_classifier_reference.py with --tensor-path.
Run the subscribe, preprocess, and publish roles in separate processes with matching ROS domain and middleware.
The preprocessing process waits for the tensor subscriber before accepting Image input.
Each role has a 45-second node deadline. The publisher requires a 60-second external timeout.
The fixture uses eight frames and Gaussian sizes 3, 5, 7, and 9 twice.

Build the standard tensor package at revision d7cd9642d77a1d64fd85f25ba0bf96e108401900 first.
Source its overlay before building this package with colcon.
No upstream implementation is bundled here. Keep its upstream license when obtaining that dependency.
See docs/experiments/043-stamped-tensor-inference.md for measured acceptance and retained initial failure.
Project licensing remains undecided. Do not distribute this package as a release.

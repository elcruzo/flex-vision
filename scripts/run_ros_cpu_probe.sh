#!/usr/bin/env bash
set -euo pipefail
: "${CPG_ROS_RESULTS:?Set a new result directory}"
mkdir "$CPG_ROS_RESULTS"
set +u
source /opt/ros/lyrical/setup.bash
set -u
mkdir -p /tmp/cpg-ros-probe/src
cp -R /workspace/experiments/ros_buffer_probe /tmp/cpg-ros-probe/src/
cd /tmp/cpg-ros-probe
colcon build --event-handlers console_direct+ > "$CPG_ROS_RESULTS/build.log" 2>&1
set +u
source install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=73
ros2 run cpg_buffer_probe image_probe subscribe > "$CPG_ROS_RESULTS/subscriber.log" 2>&1 &
CPG_SUBSCRIBER_PID=$!
trap 'kill "$CPG_SUBSCRIBER_PID" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=3 35 ros2 run cpg_buffer_probe image_probe publish > "$CPG_ROS_RESULTS/publisher.log" 2>&1
wait "$CPG_SUBSCRIBER_PID"
trap - EXIT
dpkg-query -W 'ros-lyrical-*' > "$CPG_ROS_RESULTS/packages.txt"
uname -a > "$CPG_ROS_RESULTS/platform.txt"

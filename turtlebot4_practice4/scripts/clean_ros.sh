#!/usr/bin/env bash
# ==============================================================================
# Script dọn dẹp triệt để các tiến trình ROS 2 & Gazebo chạy ngầm bị kẹt
# ==============================================================================
echo "Đang dọn dẹp toàn bộ tiến trình ROS 2, Gazebo và bridges chạy ngầm..."

pids=$(ps -eo pid,cmd | grep -E "parameter_bridge|irobot_create|slam_toolbox|robot_state_publisher|joint_state_publisher|static_transform_publisher|turtlebot4_node|turtlebot4_ignition|ign gazebo|nav2_|controller_server|planner_server|behavior_server|bt_navigator|lifecycle_manager|amcl|map_server|velocity_smoother|smoother_server|waypoint_follower|tf_relay|auto_undock|voice_navigation_node|voice_command_node|gesture_recognition_node|robot_behavior_node|emotion_recognition_node|emotion_control_node|rviz2|ros2 launch" | grep -v grep | grep -v "clean_ros" | awk -v self="$$" -v ppid="$PPID" '$1 != self && $1 != ppid {print $1}')

if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
fi

pkill -9 -f "[i]gn gazebo" 2>/dev/null || true
pkill -9 -f "[r]uby.*gazebo" 2>/dev/null || true
ros2 daemon stop 2>/dev/null || true

# Dọn dẹp shared memory và semaphore của FastDDS bị kẹt
rm -rf /dev/shm/fastrtps* /dev/shm/sem.* /dev/shm/sem.fastrtps* 2>/dev/null || true
rm -f /tmp/launch_params* 2>/dev/null || true

sleep 1
echo "======================================================="
echo ">>> ĐÃ DỌN DẸP SẠCH TOÀN BỘ TIẾN TRÌNH THÀNH CÔNG! <<<"
echo "Bây giờ bạn có thể khởi chạy lại bình thường."
echo "======================================================="

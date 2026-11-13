#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file khởi chạy mô phỏng TurtleBot 4 trong Gazebo (Ignition Fortress)
Đã tối ưu tách pha khởi tạo:
- Pha 1 (t = 0.0s): Khởi chạy Ignition Gazebo Server & GUI nạp xong hoàn toàn cảnh 3D
  warehouse.sdf và phát /world/warehouse/scene/info (tránh lỗi Gazebo bị đen màn hình).
- Pha 2 (t = 4.5s): Spawn robot TurtleBot 4, Dock và khởi tạo gz_ros2_control + diffdrive_controller
  khi Gazebo Server đã ổn định (tránh deadlock khi lấy URDF robot_description).
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('world', default_value='warehouse',
                          description='Gazebo world: warehouse, depot, or maze'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Mô hình TurtleBot 4 (standard hoặc lite)'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'],
                          description='Mở RViz2 trực tiếp cùng simulator'),
    DeclareLaunchArgument('x', default_value='0.0', description='Toạ độ X khởi đầu'),
    DeclareLaunchArgument('y', default_value='0.0', description='Toạ độ Y khởi đầu'),
    DeclareLaunchArgument('z', default_value='0.0', description='Toạ độ Z khởi đầu'),
    DeclareLaunchArgument('yaw', default_value='0.0', description='Góc Yaw khởi đầu (rad)'),
]


def generate_launch_description():
    pkg_tb4_ign = get_package_share_directory('turtlebot4_ignition_bringup')
    pkg_practice = get_package_share_directory('turtlebot4_practice4')

    ignition_launch = PathJoinSubstitution([
        pkg_tb4_ign, 'launch', 'ignition.launch.py'
    ])
    robot_spawn_launch = PathJoinSubstitution([
        pkg_tb4_ign, 'launch', 'turtlebot4_spawn.launch.py'
    ])
    rviz_config = PathJoinSubstitution([
        pkg_practice, 'config', 'turtlebot4_sim.rviz'
    ])

    # Cố định kết nối Ignition Transport trên Loopback, X11 XCB & NVIDIA GPU Offload để tránh lỗi đen màn hình GUI
    set_ign_ip = SetEnvironmentVariable('IGN_IP', '127.0.0.1')
    set_qt_xcb = SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')
    set_nv_offload = SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1')
    set_glx_nvidia = SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

    # Pha 1 (t = 0.0s): Khởi chạy thế giới Gazebo trước để GUI nhận được /scene/info ngay lập tức
    ignition_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world', default='warehouse')),
            ('model', LaunchConfiguration('model', default='standard')),
        ]
    )

    # Pha 2 (t = 4.5s): Spawn robot TurtleBot 4 và kích hoạt gz_ros2_control sau khi Gazebo đã sẵn sàng
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace', default='')),
            ('model', LaunchConfiguration('model', default='standard')),
            ('rviz', 'false'),
            ('x', LaunchConfiguration('x', default='0.0')),
            ('y', LaunchConfiguration('y', default='0.0')),
            ('z', LaunchConfiguration('z', default='0.0')),
            ('yaw', LaunchConfiguration('yaw', default='0.0')),
        ]
    )
    delayed_robot_spawn = TimerAction(
        period=4.5,
        actions=[GroupAction(actions=[robot_spawn], scoped=True)]
    )

    # Khởi chạy RViz2 với cấu hình chuẩn cho mô phỏng cơ bản (Fixed Frame: odom)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_sim',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ],
        condition=IfCondition(LaunchConfiguration('rviz', default='false'))
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(set_ign_ip)
    ld.add_action(set_qt_xcb)
    ld.add_action(set_nv_offload)
    ld.add_action(set_glx_nvidia)
    ld.add_action(ignition_world)
    ld.add_action(delayed_robot_spawn)
    ld.add_action(rviz_node)
    return ld

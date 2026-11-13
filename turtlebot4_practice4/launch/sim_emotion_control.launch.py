#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file khởi chạy hệ thống Điều khiển TurtleBot 4 bằng Cảm xúc Khuôn mặt AI
(DeepFace Emotion Recognition + OpenCV + Gazebo Simulation + Map Server & AMCL trên RViz2)

Bao gồm:
1. Gazebo Simulation (TurtleBot 4 Standard trong thế giới warehouse)
2. TF Relay & Mock Hardware Services
3. Localization (Map Server nạp bản đồ maps/map.yaml + AMCL định vị trên bản đồ)
4. RViz2 hiển thị bản đồ (/map), RobotModel, LaserScan (/scan) và ParticleCloud
5. Auto Undock + Di chuyển robot ra vùng không gian mở trung tâm (0.90 m)
6. emotion_control_node (subscribe /detected_emotion -> publish /cmd_vel, /emotion_robot_status)
7. emotion_recognition_node (chạy trong .venv_emotion nếu tồn tại -> publish /detected_emotion)

Khi launch_sim:=false: chỉ khởi chạy ngay emotion_control_node và emotion_recognition_node.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


VENV_PYTHON_PATH = '/home/tuan/turtelbot4_ws/.venv_emotion/bin/python3'


def generate_launch_description():
    pkg_practice = get_package_share_directory('turtlebot4_practice4')
    pkg_tb4_nav = get_package_share_directory('turtlebot4_navigation')

    sim_launch = PathJoinSubstitution([pkg_practice, 'launch', 'sim_robot.launch.py'])
    default_map = PathJoinSubstitution([pkg_practice, 'maps', 'map.yaml'])
    loc_params = PathJoinSubstitution([pkg_practice, 'config', 'localization_params.yaml'])
    rviz_config = PathJoinSubstitution([pkg_practice, 'config', 'turtlebot4_emotion.rviz'])
    tb4_loc_launch = PathJoinSubstitution([pkg_tb4_nav, 'launch', 'localization.launch.py'])
    fastdds_profile = PathJoinSubstitution([pkg_practice, 'config', 'fastdds_udp_only.xml'])

    set_fastdds = SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile)

    recognition_prefix = VENV_PYTHON_PATH if os.path.exists(VENV_PYTHON_PATH) else None

    arguments = [
        DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
        DeclareLaunchArgument(
            'launch_sim',
            default_value='true',
            choices=['true', 'false'],
            description='Khởi chạy Gazebo Simulation, Localization, RViz2 và Auto Undock',
        ),
        DeclareLaunchArgument(
            'world',
            default_value='warehouse',
            description='Thế giới Gazebo (warehouse, depot, maze)',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            choices=['true', 'false'],
            description='Sử dụng simulation time cho các node mô phỏng/định vị',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Đường dẫn file bản đồ map.yaml đã lưu trước đó',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            choices=['true', 'false'],
            description='Khởi chạy giao diện RViz2 hiển thị bản đồ và robot',
        ),
        DeclareLaunchArgument(
            'auto_undock',
            default_value='true',
            choices=['true', 'false'],
            description='Tự động Undock và đưa robot ra vùng rộng thoáng không vật cản',
        ),
        DeclareLaunchArgument(
            'enable_recognition_node',
            default_value='true',
            choices=['true', 'false'],
            description='Khởi chạy Node nhận diện cảm xúc khuôn mặt (emotion_recognition_node)',
        ),
        DeclareLaunchArgument(
            'enable_control_node',
            default_value='true',
            choices=['true', 'false'],
            description='Khởi chạy Node điều khiển robot theo cảm xúc (emotion_control_node)',
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Chỉ số Webcam laptop (/dev/videoX)',
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            choices=['true', 'false'],
            description='Tắt cửa sổ OpenCV GUI (chạy headless)',
        ),
        DeclareLaunchArgument(
            'sample_images_dir',
            default_value='',
            description='Thư mục chứa 6 ảnh khuôn mặt mẫu để demo tự động thay cho webcam',
        ),
        DeclareLaunchArgument(
            'sample_hold_sec',
            default_value='8.5',
            description='Thời gian giữ mỗi ảnh khuôn mặt mẫu (giây) trong chế độ demo tự động',
        ),
    ]

    def make_control_node():
        return Node(
            package='turtlebot4_practice4',
            executable='emotion_control_node',
            name='emotion_control_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            condition=IfCondition(LaunchConfiguration('enable_control_node')),
        )

    def make_recognition_node():
        node_kwargs = {
            'package': 'turtlebot4_practice4',
            'executable': 'emotion_recognition_node',
            'name': 'emotion_recognition_node',
            'output': 'screen',
            'parameters': [
                {'camera_id': ParameterValue(LaunchConfiguration('camera_id'), value_type=int)},
                {'headless': ParameterValue(LaunchConfiguration('headless'), value_type=bool)},
                {'sample_images_dir': LaunchConfiguration('sample_images_dir')},
                {'sample_hold_sec': ParameterValue(LaunchConfiguration('sample_hold_sec'), value_type=float)},
                {'use_sim_time': False},
            ],
            'condition': IfCondition(LaunchConfiguration('enable_recognition_node')),
        }
        if recognition_prefix is not None:
            node_kwargs['prefix'] = recognition_prefix
        return Node(**node_kwargs)

    # Nhóm chạy đầy đủ khi launch_sim:=true
    sim_include = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([sim_launch]),
                launch_arguments=[
                    ('world', LaunchConfiguration('world')),
                    ('namespace', LaunchConfiguration('namespace')),
                    ('rviz', 'false'),
                ],
            )
        ],
        scoped=True,
    )

    tf_relay_node = Node(
        package='turtlebot4_practice4',
        executable='tf_relay',
        name='tf_relay',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    delayed_tf_relay = TimerAction(period=3.0, actions=[tf_relay_node])

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([tb4_loc_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('map', LaunchConfiguration('map')),
            ('params', loc_params),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )
    delayed_localization = TimerAction(period=10.0, actions=[localization])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_emotion_map',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )
    delayed_rviz = TimerAction(period=10.0, actions=[rviz_node])

    undock_node = Node(
        package='turtlebot4_practice4',
        executable='auto_undock',
        name='auto_undocker',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'move_to_open_space': True},
            {'open_space_distance': 0.90},
        ],
        condition=IfCondition(LaunchConfiguration('auto_undock')),
    )
    delayed_undock = TimerAction(period=16.0, actions=[undock_node])

    # Khởi chạy emotion_control_node sau khi auto_undock hoàn tất để tránh tranh chấp /cmd_vel
    delayed_control_node = TimerAction(period=30.0, actions=[make_control_node()])
    delayed_recognition_node = TimerAction(period=32.0, actions=[make_recognition_node()])

    with_sim_group = GroupAction(
        actions=[
            sim_include,
            delayed_tf_relay,
            delayed_localization,
            delayed_rviz,
            delayed_undock,
            delayed_control_node,
            delayed_recognition_node,
        ],
        condition=IfCondition(LaunchConfiguration('launch_sim')),
    )

    # Nhóm chạy ngay lập tức (không delay) khi launch_sim:=false
    without_sim_group = GroupAction(
        actions=[
            make_control_node(),
            make_recognition_node(),
        ],
        condition=UnlessCondition(LaunchConfiguration('launch_sim')),
    )

    ld = LaunchDescription(arguments)
    ld.add_action(set_fastdds)
    ld.add_action(with_sim_group)
    ld.add_action(without_sim_group)
    return ld

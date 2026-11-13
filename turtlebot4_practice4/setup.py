import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'turtlebot4_practice4'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), [f for f in glob('config/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'maps'), [f for f in glob('maps/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'scripts'), [f for f in glob('scripts/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'docs'), [f for f in glob('docs/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'docs', 'figures'), [f for f in glob('docs/figures/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tuan',
    maintainer_email='student@example.com',
    description='TurtleBot 4 Practice 4: Simulation, SLAM, AMCL Localization and Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'check_topics = turtlebot4_practice4.check_topics:main',
            'auto_nav_goals = turtlebot4_practice4.auto_nav_goals:main',
            'auto_undock = turtlebot4_practice4.auto_undock:main',
            'tf_relay = turtlebot4_practice4.tf_relay:main',
            'yolo_detector = turtlebot4_practice4.yolo_detector:main',
            'goal_survey = turtlebot4_practice4.goal_survey:main',
            'spawn_yolo_objects = turtlebot4_practice4.spawn_yolo_objects:main',
            'webcam_cmdvel_check = turtlebot4_practice4.webcam_cmdvel_check:main',
            'gesture_recognition_node = turtlebot4_practice4.gesture_recognition_node:main',
            'robot_behavior_node = turtlebot4_practice4.robot_behavior_node:main',
            'map_segmentation = turtlebot4_practice4.map_segmentation:main',
            'voice_command_node = turtlebot4_practice4.voice_command_node:main',
            'voice_navigation_node = turtlebot4_practice4.voice_navigation_node:main',
            'emotion_recognition_node = turtlebot4_practice4.emotion_recognition_node:main',
            'emotion_control_node = turtlebot4_practice4.emotion_control_node:main',
        ],
    },
)


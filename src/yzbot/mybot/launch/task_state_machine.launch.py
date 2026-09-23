import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('mybot')
    state_params = os.path.join(share, 'config', 'task_state_machine.yaml')
    executor_params = os.path.join(share, 'config', 'fixed_joint_pick_place.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=state_params,
            description='Task state-machine parameter file.',
        ),
        DeclareLaunchArgument(
            'executor_params_file', default_value=executor_params,
            description='Single-cube executor parameter file.',
        ),
        Node(
            package='mybot',
            executable='task_state_machine.py',
            name='task_state_machine',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'executor_params_file': LaunchConfiguration('executor_params_file')},
            ],
        ),
    ])

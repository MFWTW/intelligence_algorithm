import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mybot_share = get_package_share_directory('mybot')
    default_params = os.path.join(mybot_share, 'config', 'task_parser.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Task parser parameter file (mapping rules and API setup).',
        ),
        DeclareLaunchArgument(
            'generator_path',
            default_value=os.path.expanduser('~/dev_ws/TMSCQtest_x86_x64.bin'),
            description='Path to the problem generator executable.',
        ),
        DeclareLaunchArgument(
            'problem_override',
            default_value='',
            description='Parse this text instead of running the generator.',
        ),
        Node(
            package='mybot',
            executable='task_parser.py',
            name='task_parser',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'generator_path': LaunchConfiguration('generator_path'),
                    'problem_override': LaunchConfiguration('problem_override'),
                },
            ],
        ),
    ])

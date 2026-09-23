import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mybot_share = get_package_share_directory('mybot')
    default_params = os.path.join(
        mybot_share, 'config', 'fixed_joint_pick_place.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Fixed pick-and-place trajectory parameter file.',
        ),
        DeclareLaunchArgument(
            'cube_name',
            default_value='red_cube_1',
            description='Gazebo cube model to pick and deliver.',
        ),
        DeclareLaunchArgument(
            'warehouse',
            default_value='A',
            choices=['A', 'B', 'C'],
            description='Destination warehouse zone.',
        ),
        Node(
            package='mybot',
            executable='fixed_joint_pick_place.py',
            name='fixed_joint_pick_place',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'cube_name': LaunchConfiguration('cube_name'),
                    'warehouse': LaunchConfiguration('warehouse'),
                },
            ],
        ),
    ])

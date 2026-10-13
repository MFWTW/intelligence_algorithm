import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

GENERATOR_NAME = 'TMSCQtest_x86_x64.bin'


def default_generator_path():
    """The generator copy that travels with the workspace, without guessing where.

    An installed package sits at <ws>/install/mybot/share/mybot, so the workspace
    root is four levels up - the same on any machine, no user name involved.
    ~/dev_ws and the working directory stay as fallbacks for the layouts that
    predate this. task_parser.py applies the same search at run time, so a wrong
    guess here is still recoverable instead of fatal.
    """
    share = Path(get_package_share_directory('mybot'))
    ws_root = share.parents[3] if len(share.parents) >= 4 else Path.home() / 'dev_ws'
    candidates = (ws_root / GENERATOR_NAME,
                  Path.home() / 'dev_ws' / GENERATOR_NAME,
                  Path.cwd() / GENERATOR_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


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
            default_value=default_generator_path(),
            description='Path to the problem generator executable. Defaults to '
                        'the copy in this workspace; override with '
                        'generator_path:=/abs/path or MYBOT_WS.',
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

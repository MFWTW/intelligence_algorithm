"""Internal Nav2 experiment bringup; use nav_bringup_gazebo2's public flags."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_NAVIGATION_LAUNCHES = {
    'udp': 'navigation_udp_experiment.launch.py',
    'trace': 'navigation_trace_experiment.launch.py',
    'trace_udp': 'navigation_trace_udp_experiment.launch.py',
    'yield_pause': 'navigation_yield_pause.launch.py',
}


def _bringup_experiment(context):
    mode = LaunchConfiguration('_nav_experiment_mode', default='').perform(context)
    if mode not in _NAVIGATION_LAUNCHES:
        raise RuntimeError('Start Nav2 experiments through nav_bringup_gazebo2.launch.py')

    nav2_launch = Path(get_package_share_directory('nav2_bringup')) / 'launch'
    experiment_launch = Path(__file__).parent / _NAVIGATION_LAUNCHES[mode]
    common_args = {
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'params_file': LaunchConfiguration('_nav_experiment_params_file'),
        'use_composition': LaunchConfiguration('use_composition'),
        'use_respawn': LaunchConfiguration('use_respawn'),
    }
    navigation_args = dict(common_args)
    if mode in ('trace', 'trace_udp'):
        navigation_args.update({
            'smoother_trace_file': LaunchConfiguration('smoother_trace_file'),
            'smoother_poll_fallback_sec': LaunchConfiguration('smoother_poll_fallback_sec'),
        })

    # Keep stock localization and choose exactly one navigation variant.
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav2_launch / 'localization_launch.py')),
            launch_arguments={**common_args, 'map': LaunchConfiguration('map')}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(experiment_launch)),
            launch_arguments=navigation_args.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_bringup_experiment)])

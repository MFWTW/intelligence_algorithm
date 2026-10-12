#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, OrSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    navigation_share = get_package_share_directory('bot_navigation')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    default_map = os.path.join(navigation_share, 'maps', 'mapn3.yaml')
    default_params = os.path.join(
        navigation_share, 'param', 'originbot_nav2_2.yaml'
    )
    rviz_config = os.path.join(
        nav2_bringup_share, 'rviz', 'nav2_default_view.rviz'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    slam = LaunchConfiguration('slam')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    smoother_udp = LaunchConfiguration('smoother_udp_experiment')
    smoother_trace = LaunchConfiguration('smoother_trace_experiment')
    smoother_trace_udp = LaunchConfiguration('smoother_trace_udp_experiment')
    smoother_trace_file = LaunchConfiguration('smoother_trace_file')
    smoother_poll_fallback_sec = LaunchConfiguration('smoother_poll_fallback_sec')
    experimental = OrSubstitution(
        smoother_udp, OrSubstitution(smoother_trace, smoother_trace_udp))

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'slam': slam,
            'use_composition': use_composition,
            'use_respawn': use_respawn,
        }.items(),
        condition=UnlessCondition(experimental),
    )

    def guard_experiment(context):
        # `IfCondition`/`OrSubstitution` accept '1' (and coerce other strings),
        # so a literal `== 'true'` test would treat `:=1` as "no experiment",
        # skip every check below, and still let two experiment includes fire.
        # Accept exactly the two true/false vocabularies and fail closed on
        # anything else.
        def enabled(cfg):
            value = cfg.perform(context).strip().lower()
            if value in ('false', '0', ''):
                return False
            if value in ('true', '1'):
                return True
            raise RuntimeError(f'Unrecognized smoother experiment value: {value!r}')

        modes = [name for name, cfg in (
            ('udp', smoother_udp), ('trace', smoother_trace),
            ('trace_udp', smoother_trace_udp)) if enabled(cfg)]
        if not modes:
            return []
        if len(modes) > 1:
            raise RuntimeError(
                f'Smoother experiments are mutually exclusive; got {modes}')
        if use_composition.perform(context).strip().lower() not in ('false', '0', '') \
                or slam.perform(context).strip().lower() not in ('false', '0', ''):
            raise RuntimeError('Smoother experiment requires non-composed, map-based Nav2')
        if 'trace' in modes[0] and not smoother_trace_file.perform(context).strip():
            raise RuntimeError('Trace experiments require a non-empty smoother_trace_file')
        return []

    # Stock bringup includes these two launches. Duplicate only their includes,
    # not the Nav2 node definitions; the pinned stock navigation script itself
    # supplies every node. Exactly one smoother is chosen by the mode guard.
    nav2_experimental_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_yaml, 'use_sim_time': use_sim_time,
            'params_file': params_file, 'use_composition': use_composition,
            'use_respawn': use_respawn,
        }.items(),
        condition=IfCondition(experimental),
    )
    nav2_udp_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch', 'navigation_udp_experiment.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time, 'params_file': params_file,
            'use_composition': use_composition, 'use_respawn': use_respawn,
        }.items(),
        condition=IfCondition(smoother_udp),
    )
    nav2_trace_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch', 'navigation_trace_experiment.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time, 'params_file': params_file,
            'use_composition': use_composition, 'use_respawn': use_respawn,
            'smoother_trace_file': smoother_trace_file,
            'smoother_poll_fallback_sec': smoother_poll_fallback_sec,
        }.items(),
        condition=IfCondition(smoother_trace),
    )
    nav2_trace_udp_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch',
                         'navigation_trace_udp_experiment.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time, 'params_file': params_file,
            'use_composition': use_composition, 'use_respawn': use_respawn,
            'smoother_trace_file': smoother_trace_file,
            'smoother_poll_fallback_sec': smoother_poll_fallback_sec,
        }.items(),
        condition=IfCondition(smoother_trace_udp),
    )

    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_ground_truth',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='nav2_rviz',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the Gazebo simulation clock.',
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz with the Nav2 configuration.',
        ),
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Full path to the map YAML file.',
        ),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Full path to the Nav2 parameter file.',
        ),
        DeclareLaunchArgument(
            'slam', default_value='False',
            description='Run SLAM instead of map-based localization.',
        ),
        DeclareLaunchArgument(
            'use_composition', default_value='False',
            description='Use separate Nav2 processes; avoids container-wide stalls.',
        ),
        DeclareLaunchArgument(
            'use_respawn', default_value='True',
            description='Respawn a crashed Nav2 process in non-composed mode.',
        ),
        DeclareLaunchArgument(
            'smoother_udp_experiment', default_value='false',
            description='Opt-in isolated test: only velocity_smoother uses UDP transport.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_experiment', default_value='false',
            description='Opt-in isolated test: one process writes internal-only telemetry.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_udp_experiment', default_value='false',
            description='Opt-in single-variable test: instrumented smoother AND UDP transport.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_file', default_value='',
            description='Absolute trace output path, required by diagnostic binary.',
        ),
        DeclareLaunchArgument(
            'smoother_poll_fallback_sec', default_value='',
            description='Opt-in P0 probe: seconds of input silence before the '
                        'smoother timer polls its own subscription directly. '
                        'Empty disables it and leaves the control path untouched.',
        ),
        OpaqueFunction(function=guard_experiment),
        map_to_odom,
        nav2,
        nav2_experimental_localization,
        nav2_udp_navigation,
        nav2_trace_navigation,
        nav2_trace_udp_navigation,
        rviz,
    ])

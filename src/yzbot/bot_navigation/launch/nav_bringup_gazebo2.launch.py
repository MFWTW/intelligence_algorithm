#!/usr/bin/python3
"""Standalone Nav2 bringup and compatibility entry point for experiment flags."""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def resolve_nav_bt_xml():
    """Path to mybot's tuned recovery tree as installed on THIS machine, else ''.

    Module level (not nested in generate_launch_description) so the portability
    test can call it directly without building a launch graph. An explicit
    MYBOT_NAV_TO_POSE_BT_XML override is honoured when it names a real file, so
    an alternative tree (the A/B stable-path variant) can be tested without
    editing this launch or the production YAML.
    """
    override = os.environ.get('MYBOT_NAV_TO_POSE_BT_XML', '').strip()
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f'MYBOT_NAV_TO_POSE_BT_XML points at a missing file: {override}')
        return override
    try:
        share = get_package_share_directory('mybot')
    except PackageNotFoundError:
        # bot_navigation stays usable on its own; Nav2 then keeps its stock tree.
        return ''
    candidate = os.path.join(share, 'config', 'nav_to_pose_short_recovery.xml')
    return candidate if os.path.isfile(candidate) else ''


def nav_params_with_local_bt_xml(params_file, bt_xml=None):
    """Return params_file with default_nav_to_pose_bt_xml set for THIS machine.

    The parameter cannot be shipped as a path in the YAML (2026-10-13):
      * an absolute one exists only on the machine that wrote it, and a clone
        elsewhere silently loses the tuned recovery tree;
      * "$HOME/..." does not work either - ROS 2 does not expand environment
        variables inside a parameter file, so Nav2 receives the literal string;
      * a bare relative path is opened against the node's own CWD, which depends
        on where the operator happened to run `ros2 launch`.

    So the shipped YAML carries an EMPTY value and the real path is injected here,
    resolved from the installed mybot share directory of whichever machine is
    running. Empty result (mybot not installed) leaves the file untouched: Nav2
    then uses its stock recovery tree - the same graceful fallback as before, so
    this can degrade but never break the stack.
    """
    if bt_xml is None:
        bt_xml = resolve_nav_bt_xml()
    if not bt_xml:
        return params_file
    return RewrittenYaml(
        source_file=params_file,
        param_rewrites={'default_nav_to_pose_bt_xml': bt_xml},
        convert_types=False,
    )


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
    # Stock and experimental bringup must use the same BT path rewrite.
    nav_params_file = nav_params_with_local_bt_xml(params_file)
    slam = LaunchConfiguration('slam')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    experiment_mode = '_nav_experiment_mode'
    experiment_flags = {
        'udp': LaunchConfiguration('smoother_udp_experiment'),
        'trace': LaunchConfiguration('smoother_trace_experiment'),
        'trace_udp': LaunchConfiguration('smoother_trace_udp_experiment'),
        'yield_pause': LaunchConfiguration('yield_pause_mode'),
    }
    smoother_trace_file = LaunchConfiguration('smoother_trace_file')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': nav_params_file,
            'slam': slam,
            'use_composition': use_composition,
            'use_respawn': use_respawn,
        }.items(),
        condition=LaunchConfigurationEquals(experiment_mode, ''),
    )

    def select_experiment(context):
        # Resolve the old public flags once, before starting any nodes. Both
        # 'true' and '1' must go through the same validation.
        def enabled(cfg):
            value = cfg.perform(context).strip().lower()
            if value in ('false', '0', ''):
                return False
            if value in ('true', '1'):
                return True
            raise RuntimeError(f'Unrecognized smoother experiment value: {value!r}')

        modes = [name for name, cfg in experiment_flags.items() if enabled(cfg)]
        if len(modes) > 1:
            raise RuntimeError(
                f'Smoother experiments are mutually exclusive; got {modes}')
        mode = modes[0] if modes else ''
        if mode:
            off_values = ('false', '0', '')
            if any(cfg.perform(context).strip().lower() not in off_values
                   for cfg in (use_composition, slam)):
                raise RuntimeError('Smoother experiment requires non-composed, map-based Nav2')
            if mode in ('trace', 'trace_udp') and not smoother_trace_file.perform(context).strip():
                raise RuntimeError('Trace experiments require a non-empty smoother_trace_file')
        return [SetLaunchConfiguration(experiment_mode, mode)]

    experiments = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch', 'navigation_experiments.launch.py')
        ),
        launch_arguments={'_nav_experiment_params_file': nav_params_file}.items(),
        condition=LaunchConfigurationNotEquals(experiment_mode, ''),
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
            'yield_pause_mode', default_value='false',
            description='Opt-in guarded Nav2 motion pipeline for retained-goal yield tests.',
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
        OpaqueFunction(function=select_experiment),
        map_to_odom,
        nav2,
        experiments,
        rviz,
    ])

import os

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mybot_share = get_package_share_directory('mybot')
    navigation_share = get_package_share_directory('bot_navigation')
    navigation_map = os.path.join(
        navigation_share, 'maps', 'room_from_world.yaml'
    )
    # The validated fast profile lives IN THE REPO (config/profiles) rather than
    # in run_logs/tools, which is gitignored: the whole提速配置 used to be
    # reachable only through env vars pointing at that ignored directory, so a
    # plain `start_competition.sh` silently ran the slow MPPI baseline while the
    # measured rounds ran RPP at 1.5 m/s.
    _profiles = os.path.join(mybot_share, 'config', 'profiles')
    _fast_nav = os.path.join(_profiles, 'nav_rpp15.yaml')
    navigation_params = _fast_nav if os.path.isfile(_fast_nav) else os.path.join(
        navigation_share, 'param', 'originbot_nav2_2.yaml'
    )
    _fast_state = os.path.join(_profiles, 'state_machine_fast.yaml')
    state_machine_params = _fast_state if os.path.isfile(_fast_state) else os.path.join(
        mybot_share, 'config', 'task_state_machine.yaml'
    )
    _fast_executor = os.path.join(_profiles, 'executor_fast.yaml')
    executor_params = _fast_executor if os.path.isfile(_fast_executor) else os.path.join(
        mybot_share, 'config', 'fixed_joint_pick_place.yaml'
    )
    parser_params = os.path.join(mybot_share, 'config', 'task_parser.yaml')

    def _yaml_default(env_name, yaml_path, node_key, param, fallback):
        """Read a switch's default from its YAML, with an env override.

        These arguments used to hard-code their default ('2' for mapping_rule,
        'false' for competition_mode and require_start_button). Because the launch
        ALWAYS passes the parameter, that value silently OVERRODE the YAML:编辑
        task_parser.yaml 的 competition_mode: true 完全没效果，仍然是练习模式;
        same for mapping_rule before it was fixed. The YAML is the source of
        truth here; the environment variable and an explicit `param:=value` on
        the command line still win.
        """
        override = os.environ.get(env_name)
        if override:
            return str(override)
        try:
            import yaml
            with open(yaml_path, encoding='utf-8') as handle:
                params = yaml.safe_load(handle)[node_key]['ros__parameters']
            value = params.get(param)
        except Exception:                                  # noqa: BLE001
            return fallback
        if value is None:
            return fallback
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    def _mapping_rule_default() -> str:
        return _yaml_default('MYBOT_MAPPING_RULE', parser_params, 'task_parser',
                             'mapping_rule', '2')

    def _competition_mode_default() -> str:
        return _yaml_default('MYBOT_COMPETITION_MODE', parser_params,
                             'task_parser', 'competition_mode', 'false')

    def _require_start_button_default() -> str:
        return _yaml_default('MYBOT_REQUIRE_START_BUTTON', state_machine_params,
                             'task_state_machine', 'require_start_button', 'false')

    gui = LaunchConfiguration('gui')
    start_moveit = LaunchConfiguration('start_moveit')
    start_nav2 = LaunchConfiguration('start_nav2')
    moveit_rviz = LaunchConfiguration('moveit_rviz')
    nav_rviz = LaunchConfiguration('nav_rviz')
    smoother_udp_experiment = LaunchConfiguration('smoother_udp_experiment')
    smoother_trace_experiment = LaunchConfiguration('smoother_trace_experiment')
    smoother_trace_udp_experiment = LaunchConfiguration('smoother_trace_udp_experiment')
    smoother_trace_file = LaunchConfiguration('smoother_trace_file')
    smoother_poll_fallback_sec = LaunchConfiguration('smoother_poll_fallback_sec')
    health_timeout = LaunchConfiguration('health_timeout')
    nav_map = LaunchConfiguration('nav_map')
    nav_params_file = LaunchConfiguration('nav_params_file')
    executor_params_file = LaunchConfiguration('executor_params_file')
    state_machine_params_file = LaunchConfiguration('state_machine_params_file')
    require_start_button = LaunchConfiguration('require_start_button')
    start_task_system = LaunchConfiguration('start_task_system')
    start_task_parser = LaunchConfiguration('start_task_parser')
    mapping_rule = LaunchConfiguration('mapping_rule')
    competition_mode = LaunchConfiguration('competition_mode')
    moving_obstacles = LaunchConfiguration('moving_obstacles')
    parser_linger_sec = LaunchConfiguration('parser_linger_sec')
    start_dashboard = LaunchConfiguration('start_dashboard')
    foxglove = LaunchConfiguration('foxglove')
    foxglove_port = LaunchConfiguration('foxglove_port')

    # The judging dashboard is optional infrastructure: a fresh machine that
    # has not installed ros-humble-foxglove-bridge yet must still be able to
    # start the whole competition stack, so the include is dropped (with a
    # warning) instead of failing the launch.
    try:
        get_package_share_directory('foxglove_bridge')
        foxglove_installed = True
    except PackageNotFoundError:
        foxglove_installed = False

    if foxglove_installed:
        foxglove_bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(mybot_share, 'launch', 'foxglove_bridge.launch.py')
            ),
            launch_arguments={'port': foxglove_port}.items(),
            condition=IfCondition(foxglove),
        )
    else:
        foxglove_bridge = LogInfo(
            msg='foxglove:=true but ros-humble-foxglove-bridge is not '
                'installed; skipping the visualization bridge. Install it '
                'with: sudo apt install ros-humble-foxglove-bridge',
            condition=IfCondition(foxglove),
        )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mybot_share, 'launch', 'gazebo_world2.launch.py')
        ),
        launch_arguments={'gui': gui}.items(),
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mybot_share, 'launch', 'my_moveit_rviz.launch.py')
        ),
        launch_arguments={'rviz': moveit_rviz}.items(),
        condition=IfCondition(start_moveit),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch', 'nav_bringup_gazebo2.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'use_rviz': nav_rviz,
            'map': nav_map,
            'params_file': nav_params_file,
            'slam': 'False',
            'smoother_udp_experiment': smoother_udp_experiment,
            'smoother_trace_experiment': smoother_trace_experiment,
            'smoother_trace_udp_experiment': smoother_trace_udp_experiment,
            'smoother_trace_file': smoother_trace_file,
            'smoother_poll_fallback_sec': smoother_poll_fallback_sec,
        }.items(),
        condition=IfCondition(start_nav2),
    )

    startup_gate = Node(
        package='mybot',
        executable='startup_gate.py',
        name='startup_gate',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'timeout_sec': ParameterValue(health_timeout, value_type=float),
        }],
    )

    def start_after_gate(event, _context):
        if event.returncode == 0:
            return [moveit, navigation]
        # Without this the launch just goes quiet: MoveIt/Nav2 are skipped, the
        # health check keeps reporting "not ready", and the task queue waits in
        # IDLE with no hint of why the robot never moves.
        return [LogInfo(msg=(
            'STARTUP GATE FAILED (exit code '
            f'{event.returncode}): MoveIt and Nav2 were NOT started, so '
            '/competition/system_ready stays false and the task queue waits in '
            'IDLE. Look above for "Switch controller timed out" or missing '
            'topics, then stop everything and run: '
            'ros2 run mybot reset_dds_cache.sh'
        ))]

    start_stack_when_ready = RegisterEventHandler(
        OnProcessExit(target_action=startup_gate, on_exit=start_after_gate)
    )

    # Re-stamp the laser scan with the current sim time before the costmaps see
    # it: the Gazebo lidar's own stamp is ~10 ms ahead of the newest
    # odom->base_footprint, so tf2 extrapolates, the costmap cannot raytrace and
    # phantom obstacles accumulate until the base fails every goal with
    # "collision ahead" (measured 2026-10-13). See scripts/scan_relay.py.
    scan_relay = Node(
        package='mybot',
        executable='scan_relay.py',
        name='scan_relay',
        output='screen',
        condition=IfCondition(start_nav2),
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/scan',
            'output_topic': '/scan_stamped_now',
        }],
    )

    # Emergency avoidance. Nav2's collision_monitor has been configured in
    # nav_rpp15.yaml all along (FootprintApproach polygon, 1.2 s of look-ahead,
    # /scan) but NOTHING ever launched it, so the base had no reactive layer at
    # all: measured 2026-10-13 a patrol box crossed the base's path at 1.2 m/s
    # closing speed, the base drove into it and was flung at 8.24 m/s (16 m in
    # one second), after which every navigation goal failed. Round record:
    # run_logs/moverprio_20261013-140648 (monitor.jsonl in run_logs/monitor_moverprio2).
    #
    # It needs its own lifecycle manager: nav2_bringup's navigation_launch.py
    # hardcodes its node_names list and does not include collision_monitor, so
    # the standard manager would never configure or activate it. The extra
    # manager runs with bond_timeout 0.0 to disable the bond - the collision
    # monitor creates a bond on activation and Nav2's manager is not its
    # primary, so leaving bonds enabled would make the two managers fight over
    # /bond.
    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        condition=IfCondition(start_nav2),
        parameters=[nav_params_file, {'use_sim_time': True}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )
    collision_monitor_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_collision_monitor',
        output='screen',
        condition=IfCondition(start_nav2),
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['collision_monitor'],
        }],
    )

    health_check = Node(
        package='mybot',
        executable='system_health_check.py',
        name='system_health_check',
        output='screen',
        parameters=[{
            # Use wall time so timeout/retry checks work before /clock appears.
            'use_sim_time': False,
            'timeout_sec': ParameterValue(health_timeout, value_type=float),
            'require_moveit': ParameterValue(start_moveit, value_type=bool),
            'require_nav2': ParameterValue(start_nav2, value_type=bool),
            # The lifecycle managers normally own every transition, but they can
            # wedge for good: measured 2026-10-07, map_server's CONFIGURE
            # response was dropped and lifecycle_manager_localization blocked
            # inside that call forever, leaving system_ready false and the task
            # idle for 201 s. The health check now re-issues the transition
            # (CONFIGURE or ACTIVATE as needed) once a node whose state it has
            # actually observed stays non-active for recover_grace_sec - long
            # enough that a healthy bringup, which configures every node within
            # a few seconds, never reaches it.
            'auto_recover_nav2': True,
            'recover_grace_sec': 45.0,
        }],
    )

    task_state_machine = Node(
        package='mybot',
        executable='task_state_machine.py',
        name='task_state_machine',
        output='screen',
        condition=IfCondition(start_task_system),
        parameters=[
            state_machine_params_file,
            {'executor_params_file': executor_params_file},
            # Opt-in manual start for the layout's 「开始任务」 button; off by
            # default so a task still starts by itself when the system is ready.
            {'require_start_button': ParameterValue(require_start_button,
                                                    value_type=bool)},
        ],
    )

    task_parser = Node(
        package='mybot',
        executable='task_parser.py',
        name='task_parser',
        output='screen',
        condition=IfCondition(start_task_parser),
        parameters=[
            parser_params,
            {'linger_sec': ParameterValue(parser_linger_sec, value_type=float)},
            # Which of the three published mapping rules this run uses. The
            # judge picks one before timing, so it must be switchable per launch
            # without editing the YAML:
            #   ros2 launch mybot competition_bringup.launch.py mapping_rule:=3
            {'mapping_rule': ParameterValue(mapping_rule, value_type=int)},
            # Competition mode appends 11470507 to the generator, which makes the
            # problem change on a 30-minute wall-clock window instead of being
            # random per run.
            {'competition_mode': ParameterValue(competition_mode,
                                                value_type=bool)},
        ],
    )

    # Opt-in moving obstacles: boxes that patrol the base's own routes so
    # avoidance is exercised while it fetches and delivers. Off by default so
    # timed rounds are unchanged; the matching world variant is
    # offic_room_movers.world (select it with MYBOT_WORLD_NAME).
    moving_obstacles_node = Node(
        package='mybot',
        executable='moving_obstacles.py',
        name='moving_obstacles',
        output='screen',
        condition=IfCondition(moving_obstacles),
        parameters=[{'use_sim_time': True}],
    )

    # Rule sheet section 三: the judges' layout reads the task, progress and
    # work/end-effector state from standard topics. The supervisor publishes
    # those as JSON strings, which a Foxglove Indicator cannot index into, so
    # this node derives one small topic per displayed value.
    dashboard = Node(
        package='mybot',
        executable='dashboard_bridge.py',
        name='dashboard_bridge',
        output='screen',
        condition=IfCondition(start_dashboard),
    )

    return LaunchDescription([
        # Same WSL/D3D12 story as gazebo_world2.launch.py: keep every GL
        # process (Gazebo, MoveIt RViz, Nav2 RViz) on the discrete adapter so
        # the renderer does not lose its device and drop parts of the scene.
        SetEnvironmentVariable(
            'MESA_D3D12_DEFAULT_ADAPTER_NAME',
            os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA'),
        ),
        DeclareLaunchArgument(
            'gui', default_value='true', choices=['true', 'false'],
            description='Start the Gazebo graphical client.',
        ),
        DeclareLaunchArgument(
            'start_moveit', default_value='true', choices=['true', 'false'],
            description='Start MoveIt after Gazebo initialization.',
        ),
        DeclareLaunchArgument(
            'start_nav2', default_value='true', choices=['true', 'false'],
            description='Start Nav2 after Gazebo initialization.',
        ),
        DeclareLaunchArgument(
            'moveit_rviz', default_value='true', choices=['true', 'false'],
            description='Start the MoveIt RViz window.',
        ),
        DeclareLaunchArgument(
            'nav_rviz', default_value='false', choices=['true', 'false'],
            description='Start a second RViz window for Nav2.',
        ),
        DeclareLaunchArgument(
            'smoother_udp_experiment', default_value='false', choices=['true', 'false'],
            description='Isolated single-process DDS transport test; never enable by default.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_experiment', default_value='false', choices=['true', 'false'],
            description='Isolated instrumented smoother test; never enable by default.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_udp_experiment', default_value='false', choices=['true', 'false'],
            description='Isolated instrumented smoother on UDP transport; never enable by default.',
        ),
        DeclareLaunchArgument(
            'smoother_poll_fallback_sec', default_value='',
            description='Opt-in P0 input-poll fallback; empty keeps stock behaviour.',
        ),
        DeclareLaunchArgument(
            'smoother_trace_file', default_value='',
            description='Absolute JSONL output for internal smoother telemetry.',
        ),
        DeclareLaunchArgument(
            'nav_map', default_value=navigation_map,
            description='Nav2 static map YAML. Defaults to the map generated '
                        'from the Gazebo world so AMCL has matching features.',
        ),
        DeclareLaunchArgument(
            'state_machine_params_file', default_value=state_machine_params,
            description='Supervisor parameter file. Overriding it is how the '
                        'base-plausibility limit is raised in step with a higher '
                        'commanded speed, without editing the production YAML.',
        ),
        DeclareLaunchArgument(
            'executor_params_file', default_value=executor_params,
            description='Executor (pick-and-place) parameter file. Overriding it '
                        'is how an opt-in, reversible behaviour experiment runs '
                        'without editing the production YAML.',
        ),
        DeclareLaunchArgument(
            'nav_params_file', default_value=navigation_params,
            description='Nav2 parameter file. Overriding it is how an opt-in, '
                        'reversible speed experiment is run without editing the '
                        'production YAML.',
        ),
        DeclareLaunchArgument(
            'competition_mode',
            default_value=_competition_mode_default(),
            choices=['true', 'false'],
            description='Pass 11470507 to the problem generator so the problem '
                        'matches the judges\' 30-minute window.',
        ),
        DeclareLaunchArgument(
            'require_start_button',
            default_value=_require_start_button_default(),
            choices=['true', 'false'],
            description='Hold the run in WAITING_FOR_START until true arrives on '
                        '/competition/start (the Foxglove 「开始任务」 button).',
        ),
        DeclareLaunchArgument(
            'moving_obstacles',
            # ON by default (requested 2026-10-12): the patrol boxes must be part
            # of the scene whenever `ros2 run mybot start_competition.sh` starts,
            # so a competition run never silently rehearses on an empty floor.
            # Set MYBOT_MOVING_OBSTACLES=false to get the old behaviour.
            default_value=os.environ.get('MYBOT_MOVING_OBSTACLES') or 'true',
            choices=['true', 'false'],
            description='Start the patrolling obstacle boxes (needs the '
                        'offic_room_movers.world variant to have the models).',
        ),
        DeclareLaunchArgument(
            'mapping_rule',
            default_value=_mapping_rule_default(),
            choices=['1', '2', '3'],
            description='Competition mapping rule (1, 2 or 3). The judge picks '
                        'one before timing; defaults to whatever '
                        'task_parser.yaml says, and can be overridden with '
                        'MYBOT_MAPPING_RULE or mapping_rule:=N.',
        ),
        DeclareLaunchArgument(
            'parser_linger_sec', default_value='45.0',
            description='How long task_parser keeps /competition/task latched '
                        'after a successful parse so a late supervisor still '
                        'receives it. 0 exits immediately.',
        ),
        DeclareLaunchArgument(
            'health_timeout', default_value='420.0',
            description='Seconds before the health checker reports a startup timeout.',
        ),
        DeclareLaunchArgument(
            'start_task_system', default_value='true', choices=['true', 'false'],
            description='Start the competition task state machine.',
        ),
        DeclareLaunchArgument(
            'start_task_parser', default_value='true', choices=['true', 'false'],
            description='Run the problem generator and the LLM task parser. '
                        'Requires DEEPSEEK_API_KEY in the environment.',
        ),
        DeclareLaunchArgument(
            'start_dashboard', default_value='true', choices=['true', 'false'],
            description='Publish /competition/dashboard/* for the scoring '
                        'layout (task counts, zones, progress, work state).',
        ),
        DeclareLaunchArgument(
            'foxglove', default_value='true', choices=['true', 'false'],
            description='Serve the Foxglove WebSocket dashboard on '
                        'ws://localhost:<foxglove_port>. Silently skipped when '
                        'ros-humble-foxglove-bridge is not installed.',
        ),
        DeclareLaunchArgument(
            'foxglove_port', default_value='8765',
            description='Foxglove WebSocket port.',
        ),
        gazebo,
        # Start the heavy stacks only after sensors and all controllers are
        # actually ready; fixed delays race on slower machines.
        startup_gate,
        start_stack_when_ready,
        scan_relay,
        collision_monitor,
        collision_monitor_lifecycle,
        health_check,
        task_state_machine,
        task_parser,
        moving_obstacles_node,
        dashboard,
        # Cheap (a few MB) and harmless while the stack is still coming up:
        # starting it here means the dashboard can be opened before the run
        # begins, and latched /competition/* values are not missed.
        foxglove_bridge,
    ])

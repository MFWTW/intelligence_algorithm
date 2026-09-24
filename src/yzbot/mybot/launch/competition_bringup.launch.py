import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
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
    navigation_params = os.path.join(
        navigation_share, 'param', 'originbot_nav2_2.yaml'
    )
    state_machine_params = os.path.join(
        mybot_share, 'config', 'task_state_machine.yaml'
    )
    executor_params = os.path.join(
        mybot_share, 'config', 'fixed_joint_pick_place.yaml'
    )
    parser_params = os.path.join(mybot_share, 'config', 'task_parser.yaml')

    gui = LaunchConfiguration('gui')
    start_moveit = LaunchConfiguration('start_moveit')
    start_nav2 = LaunchConfiguration('start_nav2')
    moveit_rviz = LaunchConfiguration('moveit_rviz')
    nav_rviz = LaunchConfiguration('nav_rviz')
    health_timeout = LaunchConfiguration('health_timeout')
    nav_map = LaunchConfiguration('nav_map')
    start_task_system = LaunchConfiguration('start_task_system')
    start_task_parser = LaunchConfiguration('start_task_parser')
    parser_linger_sec = LaunchConfiguration('parser_linger_sec')

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
            'params_file': navigation_params,
            'slam': 'False',
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
        return []

    start_stack_when_ready = RegisterEventHandler(
        OnProcessExit(target_action=startup_gate, on_exit=start_after_gate)
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
            # Nav2 lifecycle transitions remain owned by its lifecycle managers.
            'auto_recover_nav2': False,
        }],
    )

    task_state_machine = Node(
        package='mybot',
        executable='task_state_machine.py',
        name='task_state_machine',
        output='screen',
        condition=IfCondition(start_task_system),
        parameters=[
            state_machine_params,
            {'executor_params_file': executor_params},
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
        ],
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
            'nav_map', default_value=navigation_map,
            description='Nav2 static map YAML. Defaults to the map generated '
                        'from the Gazebo world so AMCL has matching features.',
        ),
        DeclareLaunchArgument(
            'parser_linger_sec', default_value='45.0',
            description='How long task_parser keeps /competition/task latched '
                        'after a successful parse so a late supervisor still '
                        'receives it. 0 exits immediately.',
        ),
        DeclareLaunchArgument(
            'health_timeout', default_value='180.0',
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
        gazebo,
        # Start the heavy stacks only after sensors and all controllers are
        # actually ready; fixed delays race on slower machines.
        startup_gate,
        start_stack_when_ready,
        health_check,
        task_state_machine,
        task_parser,
    ])

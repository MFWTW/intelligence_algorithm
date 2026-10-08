import os
import re
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro

def remove_comments(text):
    """移除XML/HTML注释"""
    pattern = r'<!--(.*?)-->'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def generate_launch_description():
    robot_name_in_model = 'six_arm'
    model_pkg_name = 'mybot_description'
    urdf_name = "originbot_with_rgbd_gazebo_arm.xacro"
    #world_name = 'room_aboxa3.world'
    world_name = 'offic_room.world'

    model_pkg_share = FindPackageShare(package=model_pkg_name).find(model_pkg_name)
    gazebo_ros_share = FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    urdf_model_path = os.path.join(model_pkg_share, f'urdf/{urdf_name}')
    world_file_path = os.path.join(model_pkg_share, f'worlds/{world_name}')

    # 使用 gazebo_ros 提供的启动文件，以便正确设置 Gazebo 的模型、资源和插件路径。
    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'true',
        }.items(),
    )
    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    # 解析 xacro 并去除注释。process_file 会正确管理文件读取过程。
    robot_description = xacro.process_file(urdf_model_path).toxml()
    params = {'robot_description': remove_comments(robot_description)}

    # robot_state_publisher 节点
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, params, {"publish_frequency": 15.0}],
        output='screen'
    )

    # 在 Gazebo 中生成机器人（通过 robot_description 话题）
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', robot_name_in_model,
            '-topic', 'robot_description',
            '-timeout', '120.0',
            '-x', '0.0', '-y', '0.0', '-z', '0.0', '-Y', '0.0'
        ],
        output='screen'
    )

    # ========== 使用 spawner 加载控制器（替代 ros2 control 命令） ==========
    # Three things were learned the hard way on this WSL host:
    #   * --switch-timeout matters as much as the discovery timeout; with the
    #     default 5 s the activate call fails as "Switch controller timed out
    #     after 5.000000 seconds!".
    #   * Even then a spawner can fail to reach /controller_manager at all
    #     ("Could not contact service /controller_manager/list_controllers"
    #     after waiting --controller-manager-timeout), while the very next
    #     spawner one second later succeeds. Fast DDS on WSL randomly loses
    #     that endpoint.
    #   * A single failed spawner used to abort the whole bringup: no
    #     controllers -> the startup gate times out -> MoveIt/Nav2 never start
    #     -> /competition/system_ready stays false and the task queue waits in
    #     IDLE, which looks exactly like "the robot does not move".
    # So each spawner is retried a few times before giving up.
    def controller_spawner(controller: str) -> Node:
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                controller, '-c', '/controller_manager',
                '--controller-manager-timeout', '120',
                '--service-call-timeout', '60',
                '--switch-timeout', '60',
            ],
            output='screen',
        )

    def start_after(event, _context, action, label):
        """Start ``action`` once the previous step succeeded."""
        if event.returncode == 0:
            return [action]
        return [LogInfo(msg=(
            f'{label} exited with code {event.returncode}; not starting the '
            'next startup step. Check the lines above.'
        ))]

    def retry_on_failure(controller, next_action, attempts=4):
        """Handler for a spawner's exit: retry the same controller, else move on.

        The retry is what makes startup survive this host: a spawner can fail
        with "Could not contact service /controller_manager/list_controllers"
        even though the controller manager is alive and the *next* spawner one
        second later succeeds (Fast DDS on WSL loses that endpoint). A fresh
        Node is created per attempt because a launch action may only be
        executed once.
        """
        remaining = {'left': attempts}

        def on_exit(event, _context):
            if event.returncode == 0:
                return [] if next_action is None else [next_action]
            if remaining['left'] <= 0:
                return [LogInfo(msg=(
                    f'{controller}: still failing after {attempts} attempts, '
                    'giving up. MoveIt/Nav2 will not start, /competition/'
                    'system_ready stays false and the task queue waits in IDLE. '
                    'Stop everything, run ros2 run mybot reset_dds_cache.sh, '
                    'then start again.'
                ))]
            remaining['left'] -= 1
            attempt = attempts - remaining['left']
            return [
                LogInfo(msg=(
                    f'{controller} failed (exit {event.returncode}); retrying '
                    f'{attempt}/{attempts} in 3 s.'
                )),
                TimerAction(
                    period=3.0,
                    actions=[controller_spawner(controller)],
                ),
            ]

        return on_exit

    load_joint_state_broadcaster = controller_spawner('joint_state_broadcaster')
    load_arm_controller = controller_spawner('arm_controller')
    load_gripper_controller = controller_spawner('gripper_controller')

    # ========== 事件顺序（依次加载，每个控制器失败自动重试 4 次） ==========
    # 机器人 spawn 成功 -> joint_state_broadcaster -> arm_controller
    # -> gripper_controller；任一环节失败只重试它自己，不跳过下一步。
    evt1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=lambda event, context: start_after(
                event, context, load_joint_state_broadcaster, 'spawn_entity'
            ),
        )
    )
    evt2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=retry_on_failure('joint_state_broadcaster', load_arm_controller),
        )
    )
    evt3 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_arm_controller,
            on_exit=retry_on_failure('arm_controller', load_gripper_controller),
        )
    )
    evt4 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_gripper_controller,
            on_exit=retry_on_failure('gripper_controller', None),
        )
    )

    # Gazebo 差速驱动插件已经发布 /odom 和 odom -> base_footprint。
    # 这里不再重复启动 EKF，避免依赖不存在的 config/ekf.yaml，
    # 同时避免两个节点竞争发布同一条 TF。
    # 这台机器上 gzserver/gzclient 默认走 Mesa 的 D3D12 渲染后端，
    # 而它默认选中核显（AMD Radeon iGPU，共享内存）。实测核显路径会
    # 报 "Vertex Buffer: Out of memory" -> "D3D12: Removing Device" 并让
    # gzserver/gzclient 段错误退出，渲染残缺（例如看不到车体）。
    # 指定 NVIDIA 独显（这里有 8 GB 独立显存）后同一场景稳定渲染；
    # 没有 NVIDIA 显卡时 Mesa 会忽略该名字并回落到默认适配器。
    # 如需强制别的适配器，运行前自行 export 同名环境变量即可。
    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable(
        'MESA_D3D12_DEFAULT_ADAPTER_NAME',
        os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA'),
    ))
    ld.add_action(DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Whether to start the Gazebo graphical client.',
    ))
    ld.add_action(start_gazebo_server)
    ld.add_action(start_gazebo_client)
    ld.add_action(node_robot_state_publisher)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(evt1)
    ld.add_action(evt2)
    ld.add_action(evt3)
    ld.add_action(evt4)

    return ld

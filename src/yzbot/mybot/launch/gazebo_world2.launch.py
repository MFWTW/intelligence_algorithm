import os
import re
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
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
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_prefix
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
    # Opt-in world override, used to A/B a world-level setting (the physics step)
    # without editing the production world. Unset means offic_room.world, so the
    # default launch is byte-for-byte the same as before.
    # The patrol boxes are NOT part of any world file: with their planar_move
    # plugins present at load time, the robot's own spawn fails ("Entity pushed
    # to spawn queue, but spawn service timed out waiting for entity to appear
    # in simulation under the name [six_arm]"), while the same world without the
    # plugins spawns fine. moving_obstacles.py inserts them at runtime instead,
    # so the world selection stays exactly as it was.
    world_name = os.environ.get('MYBOT_WORLD_NAME') or 'offic_room.world'

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
    # Opt-in camera-rate override for the render-load A/B. The vision servo's
    # own loop was measured at 2.4-3.1 Hz (it spends ~0.3 s per 1280x720 frame
    # on HSV), so a 10 Hz camera feeds it ~3x more frames than it consumes;
    # render load is what caps this host's real-time factor. Unset keeps 10 Hz.
    robot_description = xacro.process_file(
        urdf_model_path,
        mappings={'camera_update_rate': os.environ.get('MYBOT_CAMERA_RATE') or '10.0'},
    ).toxml()
    safe_drive = os.environ.get('MYBOT_YIELD_PAUSE_MODE', '').lower() in ('1', 'true')
    if safe_drive:
        stock_plugin = 'libgazebo_ros_diff_drive.so'
        if robot_description.count(stock_plugin) != 1:
            raise RuntimeError('Expected exactly one stock drive plugin for yield pause')
        robot_description = robot_description.replace(
            stock_plugin, 'libmybot_safe_diff_drive.so')
        if robot_description.count('<command_topic>cmd_vel</command_topic>') != 1:
            raise RuntimeError('Expected exactly one Gazebo drive command topic')
        robot_description = robot_description.replace(
            '<command_topic>cmd_vel</command_topic>',
            '<command_timeout>0.5</command_timeout><command_topic>cmd_vel</command_topic>',
            1,
        )
    params = {'robot_description': remove_comments(robot_description)}

    # robot_state_publisher 节点
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, params, {"publish_frequency": 15.0}],
        output='screen'
    )

    # 在 Gazebo 中生成机器人（通过 robot_description 话题）
    # Retry INSIDE the process, for the same reason as the controller spawners
    # below: the whole bringup hangs off this action's exit, so it must not be
    # replaced by a new one. Gazebo sometimes answers the spawn call almost
    # immediately with "Entity pushed to spawn queue, but spawn service timed out
    # waiting for entity to appear in simulation under the name [six_arm]"
    # (measured 2026-10-12 +3.53 s, 0.17 s after the call) and spawn_entity.py
    # then exits 1 - with no robot there are no controllers, no /scan, no /map,
    # and the task sits in IDLE forever. One retry turns that into a few seconds.
    _spawn_cmd = (
        'ros2 run gazebo_ros spawn_entity.py'
        f' -entity {robot_name_in_model} -topic robot_description'
        ' -timeout 120.0 -x 0.0 -y 0.0 -z 0.0 -Y 0.0'
    )
    spawn_entity_cmd = ExecuteProcess(
        cmd=['bash', '-c', (
            'for attempt in 1 2 3 4; do '
            'echo "[spawn_entity] attempt $attempt/4"; '
            f'out=$({_spawn_cmd} 2>&1); code=$?; printf "%s\\n" "$out"; '
            # "already exists" means the FIRST call did create the robot - Gazebo
            # just failed to confirm it in time. Measured 2026-10-12: attempt 1
            # reported "Entity pushed to spawn queue, but spawn service timed out
            # waiting for entity to appear", attempts 2-4 reported "Entity
            # [six_arm] already exists", and /gazebo/link_states listed
            # six_arm::base_footprint - the robot was there all along. Treating
            # that as failure exhausted the retries, aborted the chain, left
            # every controller unloaded and /competition/system_ready false, so
            # the task waited in IDLE forever looking like "the robot does not
            # move". The same lesson was already applied to the patrol boxes;
            # this is the robot's turn.
            'if [ $code -eq 0 ] || printf "%s" "$out" | grep -q "already exists"; '
            'then echo "[spawn_entity] ok on attempt $attempt"; exit 0; fi; '
            'echo "[spawn_entity] attempt $attempt failed; retrying in 5s"; '
            'sleep 5; done; '
            'echo "[spawn_entity] FAILED after 4 attempts - run: '
            'ros2 run mybot reset_dds_cache.sh, then start again"; exit 1'
        )],
        output='screen',
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
    #   * --service-call-timeout was 60 s, which made ONE lost response cost a
    #     full minute: measured 2026-10-14 15:21, the gripper spawner's
    #     list_controllers reply was dropped ("failed to send response ...
    #     (timeout)") and the spawner then sat until 65.9 s before retrying
    #     successfully at 66.2 s. It is 12 s now - a normal reply arrives in
    #     well under 1 s (joint_state_broadcaster and arm_controller configure
    #     and activate in 0.04-1.0 s each), and the in-process retry loop still
    #     gives four attempts, so a genuinely dead controller_manager still ends
    #     in the same loud failure.
    # So each spawner retries INSIDE its own process (see below); the retry
    # must never be a new launch action or the chain breaks.
    def controller_spawner(controller: str) -> ExecuteProcess:
        """Load and activate one controller, retrying INSIDE the process.

        The retry must not be a new launch action. The previous version had the
        OnProcessExit handler start a fresh spawner Node on failure, but launch
        only fires that handler for the exact action object it was registered
        with - so when the replacement succeeded nobody was listening and the
        chain stopped dead. Measured 2026-10-11 (movers_live/startbtn_round.log):
        joint_state_broadcaster failed once, its retry succeeded, and
        arm_controller / gripper_controller were then never even attempted -
        /competition/system_ready stayed false and the task sat in IDLE forever,
        which looks exactly like "the robot does not move".

        Keeping the retry inside one shell process makes the exit code mean
        "succeeded after up to 4 tries", so the chain's target action never
        changes and the next step always runs on success.
        """
        spawner = (
            f'ros2 run controller_manager spawner {controller} '
            '-c /controller_manager --controller-manager-timeout 120 '
            '--service-call-timeout 12 --switch-timeout 60'
        )
        script = (
            f'for attempt in 1 2 3 4; do '
            f'echo "[{controller}] attempt $attempt/4"; '
            f'{spawner} && exit 0; '
            f'echo "[{controller}] attempt $attempt failed; retrying in 3s"; '
            f'sleep 3; done; '
            f'echo "[{controller}] FAILED after 4 attempts - stop everything, '
            f'run: ros2 run mybot reset_dds_cache.sh, then start again"; exit 1'
        )
        return ExecuteProcess(cmd=['bash', '-c', script], output='screen')

    def start_after(event, _context, action, label):
        """Start ``action`` once the previous step succeeded."""
        if event.returncode == 0:
            return [] if action is None else [action]
        return [LogInfo(msg=(
            f'{label} exited with code {event.returncode}; not starting the '
            'next startup step. Check the lines above.'
        ))]

    load_joint_state_broadcaster = controller_spawner('joint_state_broadcaster')
    load_arm_controller = controller_spawner('arm_controller')
    load_gripper_controller = controller_spawner('gripper_controller')

    # ========== 事件顺序（依次加载；每个 spawner 在自己的进程内重试 4 次） ==========
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
            on_exit=lambda event, context: start_after(
                event, context, load_arm_controller, 'joint_state_broadcaster'
            ),
        )
    )
    evt3 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_arm_controller,
            on_exit=lambda event, context: start_after(
                event, context, load_gripper_controller, 'arm_controller'
            ),
        )
    )
    evt4 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_gripper_controller,
            on_exit=lambda event, context: start_after(
                event, context, None, 'gripper_controller'
            ),
        )
    )
    # The GUI client is deliberately NOT started next to gzserver any more.
    # Measured 2026-10-14 over 12 rounds: gzserver advertised /spawn_entity after
    # 5.8 s in the fast rounds and after 34.1 s in the slow ones, and that single
    # wait accounted for the whole startup difference (READY 24-27 s vs 48-100 s)
    # because nothing else can start before the robot exists. gzclient renders
    # the same world on the same host and competes with the loader for CPU and
    # GPU, so it now starts once the robot is in the world.
    evt_gui = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=lambda event, context: (
                [start_gazebo_client] if event.returncode == 0 else []
            ),
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
    if safe_drive:
        plugin_dir = os.path.join(get_package_prefix('mybot'), 'lib')
        if not os.path.isfile(os.path.join(plugin_dir, 'libmybot_safe_diff_drive.so')):
            raise RuntimeError('Guarded drive plugin is not installed; rebuild mybot')
        ld.add_action(SetEnvironmentVariable(
            'GAZEBO_PLUGIN_PATH',
            plugin_dir + os.pathsep + os.environ.get('GAZEBO_PLUGIN_PATH', ''),
        ))
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
    ld.add_action(node_robot_state_publisher)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(evt1)
    ld.add_action(evt2)
    ld.add_action(evt3)
    ld.add_action(evt4)
    ld.add_action(evt_gui)

    return ld

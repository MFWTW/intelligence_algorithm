
环境
vmware 17
英文 ubuntu 22.04 ros humble

依赖
sudo apt install ros-humble-desktop-full

sudo apt install gazebo
sudo apt install ros-humble-gazebo-*

sudo apt install ros-humble-moveit
sudo apt install ros-humble-moveit-setup-assistant
sudo apt install ros-humble-moveit-*

sudo apt install ros-humble-controller-manager -y
sudo apt install ros-humble-joint-trajectory-controller ros-humble-joint-state-broadcaster -y

sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-nav2*

启动机械臂moveit
ros2 launch mybot demo.launch.py

启动机械臂moveit和gazebo仿真
ros2 launch mybot gazebo.launch.py
ros2 launch mybot my_moveit_rviz.launch.py

启动机械臂moveit和gazebo 底盘+机械臂+传感器仿真
ros2 launch mybot gazebo_world.launch.py
ros2 launch mybot my_moveit_rviz.launch.py

note:
moveit 配置时只以臂模型做参考，臂+底盘整体运行 moveit会出现link不匹配的Waring，规划会出错，但是可以执行动作，
可能需要把合并后的模型注释掉gazebo的部分重新通过moveit_setup_assistant 配置


## 导航入口

| 用途 | 启动入口 | 默认地图 / Nav2 参数 | 节点与依赖 |
| --- | --- | --- | --- |
| **比赛默认** | `ros2 run mybot start_competition.sh`（带预检和就绪等待），或 `ros2 launch mybot competition_bringup.launch.py` | [room_from_world.yaml](bot_navigation/maps/room_from_world.yaml) / [nav_rpp15.yaml](mybot/config/profiles/nav_rpp15.yaml)（RPP 快档；文件缺失才回退 MPPI） | 入口负责 Gazebo、机器人和控制器；启动门检查通过后拉起 Nav2 和 MoveIt。快档局部代价地图用 `scan_relay` 提供的 `/scan_stamped_now`，也接收巡逻箱发布的 `/movers/obstacles`（启用巡逻箱时）。 |
| **独立仿真导航** | `ros2 launch bot_navigation nav_bringup_gazebo2.launch.py` | [mapn3.yaml](bot_navigation/maps/mapn3.yaml) / [originbot_nav2_2.yaml](bot_navigation/param/originbot_nav2_2.yaml)（MPPI） | 须另起与地图匹配的 Gazebo 场景和机器人（例如 `ros2 launch mybot gazebo_world2.launch.py`，按场景传 `map:=…`），提供 `/clock`、`/scan`、`/odom`、`odom → base_footprint` 等 TF，并接收底盘速度命令。入口不启动 MoveIt、雷达中继或比赛任务节点。 |
| **旧版，仅保留兼容** | `ros2 launch bot_navigation nav_bringup_gazebo.launch.py` | [map.yaml](bot_navigation/maps/map.yaml) / [originbot_nav2.yaml](bot_navigation/param/originbot_nav2.yaml)（MPPI） | 自带 Nav2 和 RViz；须另起旧地图对应的仿真和机器人。默认由 AMCL 发布 `map → odom`，需有效初始位姿（必要时在 RViz 设置）。不与新版入口混跑。 |
| **实验，显式开启** | 在比赛或独立导航命令后追加下列实验参数 | 沿用所选入口的地图和参数；让行暂停比赛模式还应用对应参数覆盖 | 必须使用非组合、基于地图的 Nav2；其余依赖沿用所选入口，附加要求见下。 |

比赛和独立导航的默认参数都关闭了 AMCL 的 TF 广播，由新版导航入口发布静态 `map → odom`；AMCL 初始位姿由参数设定。若改用 SLAM 或自定义定位配置，须先核对这条 TF 的发布者，不能让两套节点同时发布。比赛 RPP 快档的 `collision_monitor` 读取 `/cmd_vel`，输出 `/cmd_vel_monitored`，**不在实际底盘控制链上**。不要同时运行两套导航，也不要用独立入口仅替换成 RPP 参数来复现比赛。

比赛不需要图形界面时，可运行 `ros2 launch mybot competition_bringup.launch.py gui:=false moveit_rviz:=false nav_rviz:=false`。就绪与健康状态分别见 `/competition/system_ready`、`/competition/system_health`。

### 导航实验参数（四选一，默认均关闭）

- `smoother_udp_experiment:=true`：仅让 `velocity_smoother` 使用 UDP，不等同于比赛脚本的全局 `--udp`。
- `smoother_trace_experiment:=true`：使用 `mybot_velocity_smoother_trace` 记录内部追踪，须传 `smoother_trace_file:=/tmp/smoother.jsonl`。
- `smoother_trace_udp_experiment:=true`：同一追踪版 smoother，仅该进程用 UDP，同样须传追踪文件。
- `yield_pause_mode:=true`：让行暂停。完整仿真用 `MYBOT_YIELD_PAUSE_MODE=1 ros2 launch mybot competition_bringup.launch.py yield_pause_mode:=true`，由比赛入口配套启动 `yield_velocity_gate` 并应用参数覆盖；独立导航入口只切换 Nav2，不会启动配套 gate。

例如：`ros2 launch bot_navigation nav_bringup_gazebo2.launch.py smoother_udp_experiment:=true`。原命令和参数不变，实验节点的编排由内部的 [navigation_experiments.launch.py](bot_navigation/launch/navigation_experiments.launch.py) 负责，不需要另起一个 launch。

任务状态机默认随比赛入口启动；如不需要调用云端大模型，显式关闭解析器：
ros2 launch mybot competition_bringup.launch.py start_task_parser:=false

需要自动出题并解析时，请先设置 DEEPSEEK_API_KEY，再使用：
ros2 launch mybot competition_bringup.launch.py start_task_parser:=true

也可单独启动状态机：
ros2 launch mybot task_state_machine.launch.py

状态机订阅 /competition/task 与 /competition/system_ready，并发布：
/competition/state、/competition/progress、/competition/current_object、
/competition/error、/competition/retry_count。

### 比赛导航卡墙与回归验证

比赛默认地图、局部 footprint 和全局规划余量保持不变。普通航段由执行器统一决定是否恢复；
Nav2 在路径失败时不自动清图、旋转或倒车。目标应答、让行、重试和停靠共用同一航段的墙钟预算。
只有全局起点是内接膨胀格（99）且新鲜的局部代价地图和 TF 确认无碰撞时，才允许一次仅调整
全局半径的尝试；实体障碍（100）、地图未知或陈旧时不放宽。退让还要求实时雷达及两张代价
地图均通过扫掠检查，并以实测位移、无进展时间和截止时间为限，不缩小本地 footprint。
窄口外侧先限速至 1.20 m/s，再在原预刹车边界限速至 0.90 m/s；通畅路段仍可到 1.50 m/s。

发现 `NAV_STATE_UNKNOWN` 或 `GEOMETRY_STATE_UNKNOWN` 时，先停任务并检查 Nav2 是否仍有
活动目标、两张代价地图的半径/膨胀参数是否符合配置，再重启导航栈和执行器；不要直接重试。
通过同一任务与路线分别在 `moving_obstacles:=false` 和默认巡逻箱模式下复跑；对照失败时的
全局/局部格值、实际位移与截至恢复的时间。若仍出现实体碰撞，先复核仿真中折叠臂、相机及货物
的实际碰撞包络，再调整正常行驶的 footprint；不要通过缩小半径来绕过检查。
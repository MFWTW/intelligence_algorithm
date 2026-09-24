# 依赖清单 / 新机器搭建指南

> 面向：第一次拉到本仓库、要在自己机器上把「仿真 + 出题解析 + 导航抓取」跑起来的开发者。
>
> 目标环境：**Ubuntu 22.04 (jammy) + ROS 2 Humble + Gazebo Classic 11**。
> Windows 建议用 WSL2 + WSLg（本仓库就是在 WSL2 上开发与验证的）。

配套文件：

- [`requirements-ros.txt`](requirements-ros.txt) —— apt / ROS 2 包清单（可一键装）
- [`requirements.txt`](requirements.txt) —— pip 的 Python 第三方库

## 0. 实测版本基线（照着装不会踩版本坑）

| 组件 | 本仓库验证过的版本 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| ROS 2 | Humble（`ros-humble-desktop` 0.10.0） |
| Gazebo | Classic 11.10.2 |
| Nav2 | `ros-humble-navigation2` 1.1.20 |
| MoveIt 2 | `ros-humble-moveit` 2.5.10 |
| ros2_control | 2.54.0 + `gazebo_ros2_control` 0.4.10 |
| Python / OpenCV / NumPy | 3.10.12 / 4.5.4（apt `python3-opencv`）/ ≥1.21 |

## 1. 快速安装（三条命令）

```bash
# 0) ROS 2 Humble 的 apt 源要先配好（官方文档）：
#    https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
sudo apt update

# 1) 系统 / ROS 包：清单直装
grep -vE '^\s*(#|$)' requirements-ros.txt | xargs sudo apt install -y

#    （等价做法，推荐：从各 package.xml 自动推导，跟代码保持一致）
sudo rosdep init && rosdep update      # 只需第一次，已初始化过会报错，忽略即可
rosdep install --from-paths src --ignore-src -r -y

# 2) Python 第三方库
python3 -m pip install -r requirements.txt
```

## 2. 这些依赖各自是干什么用的

| 类别 | 关键包 | 谁在用 |
|---|---|---|
| ROS 2 基础 | `ros-humble-desktop`、`xacro`、`robot-state-publisher`、`tf2-ros`、`rviz2` | 所有节点、URDF/TF、可视化 |
| 仿真 | `gazebo`、`gazebo-ros`、`gazebo-plugins`、`gazebo-dev` | 比赛世界、相机/雷达/差速插件、LinkAttacher 插件编译 |
| 控制 | `ros2-control`、`ros2-controllers`、`gazebo-ros2-control` | 六轴臂 + 夹爪的 `FollowJointTrajectory` |
| 导航 | `navigation2`、`nav2-bringup`、`nav2-mppi-controller`、`nav2-amcl` | 取物/送仓导航、定位（地图 + AMCL） |
| 机械臂 | `moveit`、`moveit-configs-utils`、`moveit-kinematics` | `move_group` + MoveIt RViz |
| 视觉 | `python3-opencv`、`cv-bridge`、`image-transport` | HSV 找方块、`/camera/image_raw` 订阅（`numpy` 做掩码） |

各类的完整逐条清单和注释都在 `requirements-ros.txt` 里。

## 3. 仓库自带、不需要另外下载

- **LinkAttacher**（吸附/释放方块用的 Gazebo 插件与消息）就在
  `src/yzbot/IFRA_LinkAttacher/`，跟着 `colcon build` 一起编，别单独去找。
- **比赛世界与地图**：`mybot_description/worlds/offic_room.world`、
  `bot_navigation/maps/room_from_world.pgm/.yaml`、`worlds/dumpster/`（模型+贴图）都在仓库里，
  `offic_room.world` 不引用任何 `model://` 外部模型，离线可用。
- **出题程序** `TMSCQtest_x86_x64.bin`（x86_64 Linux ELF）已在仓库根目录。

## 4. 明确「不需要装」的东西

- `ai_msgs`：只有 `src/yzbot/tools_demo` 的示例节点 import 它，比赛流程完全不用
  （它是 `exec_depend`，所以缺了也能 `colcon build`，只是别去跑那个 demo）。
- `cartographer_ros` / `rtabmap_ros`：只有 `bot_navigation/launch/` 里那两套
  **备选** SLAM launch 才用，正常流程用的是静态地图 + AMCL。
- `opennav_docking`：`originbot_nav2_2.yaml` 里留了一段 `docking_server`
  参数，但 Humble 的 `nav2_bringup` 不会启动它，本流程不需要这两个包。
- `laser_filters`：配置里没有用到。
- **不要**用 pip 装 `rclpy` / `cv_bridge` / `tf2` / `moveit` / `nav2` 等 ROS 包，
  一律走 apt 的 `ros-humble-*`，否则 ABI 与环境会错乱。

## 5. 编译与启动

```bash
cd ~/dev_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch mybot competition_bringup.launch.py
```

- `scripts/*.py` 必须有可执行位，否则 launch 报 `executable ... not found`：
  ```bash
  chmod +x src/yzbot/mybot/scripts/*.py
  ```
- 无图形界面（长时间自动跑、省资源）：
  ```bash
  ros2 launch mybot competition_bringup.launch.py gui:=false moveit_rviz:=false
  ```
- 跑出题 + 云端解析需要环境变量（Key 不落盘）：
  ```bash
  export DEEPSEEK_API_KEY=sk-xxxx
  ```

## 6. 装完自检

```bash
source /opt/ros/humble/setup.bash && source ~/dev_ws/install/setup.bash

# a) 包和可执行文件都在
ros2 pkg list | grep -E "^(mybot|mybot_description|bot_navigation|linkattacher_msgs|ros2_linkattacher)$"
ros2 pkg executables mybot

# b) Python 库能导入
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"

# c) 仿真 + 传感器 + 控制器
ros2 launch mybot gazebo_world2.launch.py gui:=false     # 另开终端：
ros2 topic list | grep -E "camera/image_raw|scan|odom|joint_states"
ros2 control list_controllers        # 三个都应为 active

# d) 整套（出题/解析可关）直到就绪信号为 true
ros2 launch mybot competition_bringup.launch.py gui:=false start_moveit:=false \
    start_task_parser:=false start_task_system:=false nav_rviz:=false
ros2 topic echo /competition/system_ready --once
```

## 7. 环境相关的坑（不是装包，但会让你跑不起来）

1. **WSL 渲染**：Gazebo 默认可能选到核显（共享显存的 AMD/Intel iGPU），
   会报 `Vertex Buffer: Out of memory` / `D3D12: Removing Device`，
   表现为模型残缺（例如看不到车体）。两个 launch 已经默认
   `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`，没有 N 卡会自动回落。
2. **Fast DDS 残留**：反复 Ctrl-C / `kill -9` 之后，新起的节点会随机发现不到
   `/navigate_to_pose`、代价地图服务等，动作目标响应也可能丢
   （`Failed to send goal response ... (timeout)`）。重拉整套仿真前先跑：
   ```bash
   ros2 run mybot reset_dds_cache.sh
   ```
   详见 [README 的故障排查章节](README.md)。
3. **首次启动慢**：Gazebo 加载 12 万行的世界文件 + MoveIt/Nav2 起来大约 1~2 分钟，
   等 `/competition/system_ready` 变 `true` 再发任务。
4. **`--symlink-install`**：改 launch / 参数 / 脚本后不用重新 build，但要保住可执行位。

更细的开发计划与历史结论见 [`比赛后续开发任务清单.md`](比赛后续开发任务清单.md)
和 `run_logs/` 下的复现报告（`run_logs/` 不进版本库）。

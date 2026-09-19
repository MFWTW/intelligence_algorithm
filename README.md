# intelligence_algorithm

基于 ROS 2 Humble 的移动机器人智能任务项目，包含 Gazebo 仿真、Nav2 导航、MoveIt 2 机械臂控制及大模型交互示例。

## 环境

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo
- Nav2
- MoveIt 2

## 源码结构

- `src/yzbot/mybot`：机械臂 MoveIt 2 配置与启动文件
- `src/yzbot/mybot_description`：机器人模型与 Gazebo 场景
- `src/yzbot/bot_navigation`：Nav2、Cartographer、地图与 RViz 配置
- `src/yzbot/tools_demo`：大模型交互示例
- `src/yzbot/IFRA_LinkAttacher`：Gazebo link attach/detach 服务

## 编译

```bash
cd ~/dev_ws
colcon build --symlink-install
source install/setup.bash
```

## 启动

```bash
# Gazebo：底盘、机械臂及传感器
ros2 launch mybot gazebo_world.launch.py

# MoveIt 2
ros2 launch mybot my_moveit_rviz.launch.py

# Nav2
ros2 launch bot_navigation nav_bringup_gazebo.launch.py
```

更详细的开发计划见 [`比赛后续开发任务清单.md`](比赛后续开发任务清单.md)。


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


一键启动 Gazebo、控制器、MoveIt2、Nav2 和健康检查：
ros2 launch mybot competition_bringup.launch.py

无图形界面启动：
ros2 launch mybot competition_bringup.launch.py gui:=false moveit_rviz:=false nav_rviz:=false

也可以分别启动：
ros2 launch mybot gazebo_world2.launch.py
ros2 launch mybot my_moveit_rviz.launch.py
ros2 launch bot_navigation nav_bringup_gazebo2.launch.py

仿真初始位姿已由 AMCL 参数自动设置，不再需要在 RViz 中手动点击 2D Pose Estimate。
系统就绪状态：/competition/system_ready
健康检查详情：/competition/system_health

任务状态机默认随 competition_bringup 启动（不会自动调用云端大模型）：
ros2 launch mybot competition_bringup.launch.py start_task_parser:=false

需要自动出题并解析时，请先设置 DEEPSEEK_API_KEY，再使用：
ros2 launch mybot competition_bringup.launch.py start_task_parser:=true

也可单独启动状态机：
ros2 launch mybot task_state_machine.launch.py

状态机订阅 /competition/task 与 /competition/system_ready，并发布：
/competition/state、/competition/progress、/competition/current_object、
/competition/error、/competition/retry_count。

导航失败会在确认代价地图清理完成后重试；动作、导航及吸附状态无法确认时，
系统进入 FAILED 并禁止自动重试，避免旧目标与新目标同时执行。
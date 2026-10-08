# intelligence_algorithm

基于 ROS 2 Humble 的移动机器人智能任务项目，包含 Gazebo 仿真、Nav2 导航、MoveIt 2 机械臂控制及大模型交互示例。

## 环境

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Nav2
- MoveIt 2

**新机器请先看 [`DEPENDENCIES.md`](DEPENDENCIES.md)**：里面有实测版本基线、
一键安装命令、每个依赖的用途，以及「哪些不用装」。对应的清单文件是
[`requirements-ros.txt`](requirements-ros.txt)（apt / ROS 包）和
[`requirements.txt`](requirements.txt)（pip 的 Python 库）。

## 源码结构

- `src/yzbot/mybot`：机械臂 MoveIt 2 配置与启动文件
- `src/yzbot/mybot_description`：机器人模型与 Gazebo 场景
- `src/yzbot/bot_navigation`：Nav2、Cartographer、地图与 RViz 配置
- `src/yzbot/tools_demo`：大模型交互示例
- `src/yzbot/IFRA_LinkAttacher`：Gazebo link attach/detach 服务

## 编译

```bash
cd ~/dev_ws
# 依赖没装齐时先补（详见 DEPENDENCIES.md）
#   grep -vE '^\s*(#|$)' requirements-ros.txt | xargs sudo apt install -y
#   python3 -m pip install -r requirements.txt
# 或者用 rosdep 自动推导：
#   rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 启动

**这台 WSL 机器上最稳的启动方式**（自动预检残留进程 → 清 Fast DDS 缓存 → 无 GUI
启动 → 等到 `system_ready` 并报告缺什么）：

```bash
cd ~/dev_ws && source install/setup.bash
ros2 run mybot start_competition.sh
```

| 需求 | 命令 |
| --- | --- |
| 需要 Gazebo 窗口 | `ros2 run mybot start_competition.sh --gui` |
| 需要 MoveIt RViz | `ros2 run mybot start_competition.sh --with-rviz` |
| 上一次崩了、进程还残留 | `ros2 run mybot start_competition.sh --kill` |
| 单纯 `ros2 launch`（不推荐，少了预检与等待） | `ros2 launch mybot competition_bringup.launch.py gui:=false moveit_rviz:=false` |

它比裸 `ros2 launch` 多做四件事，都是实测踩出来的：

1. **预检残留进程**：被 kill 的 launch 常留下孤儿 Nav2/gzserver，两套会抢
   `/cmd_vel`，表现为机器人完全不动；有残留时脚本直接拒绝启动并给出清理命令。
2. **清 Fast DDS 残留**：反复 Ctrl-C 之后 `/dev/shm/fastrtps_*` 会累积（实测一次
   积到 97 个），新节点随机匹配不上端点 —— `Switch controller timed out after
   5.000000 seconds!`、`Could not contact service /controller_manager/...`、
   costmap 死等 `map -> base_footprint` 都是这个根因。
3. **默认不带 GUI/RViz**：Gazebo GUI 会把场景再渲染一遍，实测满栈 RTF 只有
   0.18～0.43（`ros2 run mybot rtf_probe.py` 可自测），可视化建议用 Foxglove。
4. **等到就绪再交给你**：看门狗订阅 latched 的 `/competition/system_ready`，
   就绪打印 `✅ 系统就绪`，超时则打印还缺什么。

系统就绪状态发布到 `/competition/system_ready`，详细检查结果发布到
`/competition/system_health`。也可以分别启动：

```bash
ros2 launch mybot gazebo_world2.launch.py
ros2 launch mybot my_moveit_rviz.launch.py
ros2 launch bot_navigation nav_bringup_gazebo2.launch.py
```

## 出题程序 + 云端大模型任务解析

出题程序（`TMSCQtest_x86_x64.bin`，x86_64 Linux ELF）每次运行输出一行中文应用题。
任务解析节点用 DeepSeek 云端 API 把它解析成结构化任务，并在本地独立复核算术。

API Key **存一次，以后不用管**（运行时自动加载，且绝不进入版本库）：

```bash
cd ~/dev_ws
source install/setup.bash
ros2 run mybot set_deepseek_key.sh     # 交互粘贴，写进 ~/.config/mybot/deepseek_api_key (600)
ros2 launch mybot competition_bringup.launch.py
```

查找顺序（第一个命中的生效）：`$DEEPSEEK_API_KEY` → `api_key_file` 参数 →
`$DEEPSEEK_API_KEY_FILE` → `~/.config/mybot/deepseek_api_key` → `~/.deepseek_api_key`
→ `./.secrets/deepseek_api_key` → `./.env`（后两个相对启动目录，已在 `.gitignore` 里）。
`ros2 run mybot set_deepseek_key.sh --check` 可以查看当前会加载哪一个。

只想手动导环境变量也行（临时用）：

```bash
export DEEPSEEK_API_KEY='sk-...'
ros2 launch mybot task_parser.launch.py
```

解析一道题并退出，日志里会打印题目、结构化任务和状态：

```
Task status: GENERATING
题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小美需要2件T恤,1条裤子;...
Task status: PARSING
deepseek-chat replied in 1.2s (attempt 1/4).
结构化任务: {"capacity":5,"totals":{"first":18,"second":3},"x":4,"y":1,
            "tasks":[{"color":"red","count":4,"zone":"A"},
                     {"color":"blue","count":1,"zone":"C"}]}
Task status: SUCCESS
```

发布的话题：

| Topic                           | 内容                                                                      |
| ------------------------------- | ------------------------------------------------------------------------- |
| `/competition/task_problem`   | 原始题目                                                                  |
| `/competition/task_rules`     | 映射规则                                                                  |
| `/competition/task_raw_reply` | 大模型原始回复                                                            |
| `/competition/task`           | 最终结构化任务（JSON）                                                    |
| `/competition/task_status`    | `GENERATING` / `PARSING` / `REPAIRING` / `SUCCESS` / `FAILED:*` |

调试时可以用 `-p problem_override:="题目文本"` 跳过出题程序直接解析。
映射规则、模型名、超时与重试次数都在
`src/yzbot/mybot/config/task_parser.yaml` 里改，不用动代码。

**分工**：大模型只做**信息抽取**（把题面里每个人需要的数量逐行列出），
**求和、向上取整、颜色、区域全部由代码算**，不让模型做心算。
实测这样把"9 只猫数成 11 只"这类错误彻底消掉了。

模型输出：

```json
{"capacity":5,"items":[{"who":"小爱","first":2,"second":5},
                       {"who":"小宠","first":2,"second":0},
                       {"who":"小动","first":2,"second":0},
                       {"who":"小乐","first":3,"second":5},
                       {"who":"小欢","first":0,"second":4}]}
```

节点自己求和得 `totals={first:9,second:14}`，再算 `x=ceil(9/5)=2, y=ceil(14/5)=3`。

**颜色和区域来自赛前公布的映射规则**，在 `config/task_parser.yaml` 配置：

```yaml
color_x: red              # x代表红色
color_y: blue             # y代表蓝色
# 出题程序保证 x,y >= 1 且 x + y = 5，所以数量只可能是 1/2/3/4
# 按数量 1,2,3,4 依次给区域，可表达任意规则（含 B 区）
zone_by_count: ['C', 'C', 'A', 'A']   # 即“>=3去A，其余去C”
zone_rule_text: '数量大于等于3的去A区，其他颜色去C区'
```

判定是**逐颜色**查表。因为 x+y=5，两个数量不可能同时 ≥3，所以正好一个去 A
一个去 C。四种可能结果（x=红, y=蓝）：

| x,y   | 红 | 蓝 |
| ----- | -- | -- |
| (1,4) | C  | A  |
| (2,3) | C  | A  |
| (3,2) | A  | C  |
| (4,1) | A  | C  |

赛前规则换成三档（如 ≥5→A、≥3→B、其余→C）时，把 `zone_by_count` 改成
`['C','B','B','A']` 之类即可，代码不用动。

**四层校验**：

1. 结构校验（`capacity` 正整数、`items` 非空、数量非负整数）；
2. 不变量校验：`x,y >= 1` 且 `x + y = 5`（出题程序的背景约束，即使遇到
   没见过的题型也能挡住数错）；
3. 确定性复核：用正则从题面独立重算总数（只在能完整解析时启用）；
4. 越界校验：单个颜色数量不超过场景里的 5 个方块。

任何一层不过都会把错误回灌给模型要求修复（`REPAIRING` → `REPAIRED`）。
实测连跑 8 次随机出题 **8/8 通过、0 次修复**，单次解析延迟 1～2 s。

> 注意：`scripts/*.py` 需要可执行位。若用 `--symlink-install` 且报
> `No executable found`，执行 `chmod +x src/yzbot/mybot/scripts/*.py`。

## 导航到预抓取点、视觉精定位、抓取并配送到仓库

先启动完整仿真，并等待 `/competition/system_ready` 变为 `true`：

```bash
ros2 launch mybot competition_bringup.launch.py
```

在另一个终端指定物块和目标仓库（无需人工把车摆到抓取位）：

```bash
cd ~/dev_ws
source install/setup.bash
ros2 launch mybot fixed_joint_pick_place.launch.py \
  cube_name:=red_cube_1 warehouse:=A
```

节点依次执行：

1. 机械臂回到运输位；
2. Nav2 导航到该物块的**预抓取底盘位姿**（失败时清代价地图重试，再退到备用接近位姿）；
3. **视觉精定位**：HSV 找出对应颜色方块，横向居中 + 按底边像素行伺服前进，
   再以标定步进补齐最后几厘米；
4. 张开夹爪 → 预抓取 → 下探 → `ATTACHLINK` → 闭合夹爪 → 抬升 → 运输位；
5. Nav2 导航至仓库 → 预放置 → 下降 → 张开 → `DETACHLINK` → 退回 → 回安全位；
6. **落点校验**：读取方块落点并判断是否在 1.0 × 0.5 m 区域内。

`warehouse` 支持 `A`、`B`、`C`。`/competition/vision_alignment` 发布视觉
状态字符串，`/vision/debug_image` 发布带标注的调试图。

全部关节角、位姿与阈值位于
`src/yzbot/mybot/config/fixed_joint_pick_place.yaml`，均已在 Gazebo 中标定：

- **抓取/放置关节角**：`grasp_joints` 使指爪正好夹住 0.03 m 方块
  （`link6` 距方块中心约 0.063 m）；`place_joints` 抬高，使方块落在
  0.02 m 厚的区域标牌**上表面**而不是嵌进去（嵌进去会被物理引擎缓慢弹出区域）。
- **夹爪**：两指**中心距** = `0.075 − finger_joint1`（Gazebo TF 实测），指块本身厚 10 mm，
  所以**内壁开口 = `0.065 − finger_joint1`**。方块 30 mm，因此
  `gripper_open: 0.0`（内壁 65 mm）、`gripper_closed: 0.022`（内壁 43 mm，每侧约 6.5 mm 余量）。
  曾经写在这里的 `0.048` 实际是内壁 17 mm，会每侧压进方块 6.5 mm——与固定关节对抗，
  正是"抖动把机器人弹飞"的配置。方块由 Gazebo 固定关节承载，指爪只做"包住"动作。
  收紧依据是运行日志里的 `Pre-attach cube offset in link6 frame`：实测方块在
  link6 坐标系 `dy = +0.010 m`，不接触的安全上限约 `0.027`，改这个值前先看这行。
- **视觉测距**：0.093 m 高、水平安装的相机满足
  `行 = 360 + 66.0 / (相机到方块距离 − 0.015)`；`visual_approach_row` 为停止行，
  剩余距离由 `visual_final_creep_m` 补齐。
- **仓库停靠位**：`warehouse_*_pose` 已按实测落点误差做预补偿。

节点在吸附前会检查 `link6` 与方块的距离（标定值 0.063 m，上限 0.09 m），
距离过大时拒绝远距离吸附；放置后会校验落点是否在区域内。
注意已放置的方块仍会被后续导航路径推挤，连续作业需为已放置方块预留绕行路径。

## 比赛可视化：Foxglove 状态面板

`competition_bringup.launch.py` 默认带 `foxglove:=true`，会额外起
`foxglove_bridge`（WebSocket，8765）。Windows 上的 Foxglove 桌面版选
**Foxglove WebSocket** 连 `ws://localhost:8765` 即可（WSL 是 mirrored 网络模式，
localhost 直接互通），再导入仓库里的布局文件
`src/yzbot/mybot/config/foxglove/competition_layout.json`：题目、大模型回复、
结构化任务、进度、状态迁移、3D（雷达/路径/机器人）、视觉识别画面、日志
九块面板全部绑定真实话题，没有占位文字。

只想起桥（仿真已在跑）：

```bash
ros2 launch mybot foxglove_bridge.launch.py
```

安装、面板清单、双机位端口放行和排错见 [`FOXGLOVE.md`](FOXGLOVE.md)。

## 性能：墙钟时间的瓶颈是仿真实时率（RTF），不是导航参数

赛题按墙钟计时，而 Gazebo 在这台 WSL 机器上跑不满实时。实测：

| 场景 | RTF | 说明 |
| --- | --- | --- |
| 只起 Gazebo（机器人+传感器） | **0.93** | 基本实时 |
| 完整 bringup 空闲（+MoveIt+Nav2） | **0.43** | 只剩四成 |
| 完整 bringup + 抓取执行器导航中 | **0.18** | 机器人"仿真内"0.33 m/s，墙钟只有 **0.061 m/s** |

也就是说 198 s 的一次抓取配送，机器人实际只"经历"了约 34 s；5 分钟墙钟预算在这台
机器上只买到约 55 s 的机器人时间。**单块 45 s 的目标靠调 Nav2 参数达不到**，要先把
实时率提上去，或者接受完成数量少。

已经做的降载改动（都是仿真侧，不影响算法标定）：

- 相机 15 Hz → **10 Hz**（视觉伺服本来就是 10 Hz 控制周期，15 Hz 只多花渲染）
- 雷达 20 Hz → **10 Hz**（局部代价地图 5 Hz、全局 1 Hz，10 Hz 已翻倍富余）
- 视觉 HSV 管线只在视觉伺服窗口内运行（原来在两条导航腿上也逐帧处理 1280×720，
  实测占约 27% 单核 + 图像话题流量，抢的就是 gzserver 的 CPU）
- `vx_max` 0.5 → 0.6（仿真内有效，墙钟收益有限，风险最高，出问题就回 0.5）

跑之前先自检，别在低实时率上做计时测试：

```bash
ros2 run mybot rtf_probe.py          # 打印 RTF + 机器人墙钟/仿真速度
```

> **必须知道的坑**：`Ctrl-C`/`kill` 一个 `ros2 launch` 后，子进程（Nav2、MoveIt、
> gzserver）经常变成孤儿继续跑。再起一套就会**两套 Nav2 抢 `/cmd_vel`**，表现为
> 机器人不动、TF 混乱、RTF 测出来是假数据。重拉之前确认：
> ```bash
> pgrep -c -f nav2_ ; pgrep -c -f gzserver     # 都应为 0 或 1
> pkill -f 'nav2_'; pkill -f 'gzserv[e]r'; pkill -f 'move_grou[p]'
> ros2 run mybot reset_dds_cache.sh
> ```

更快的做法：`gui:=false` 跑仿真（Gazebo GUI 会把场景再渲染一遍），可视化用 Foxglove
（见 [`FOXGLOVE.md`](FOXGLOVE.md)），需要录制时录 Foxglove 的 3D 面板而不是 Gazebo 窗口。

## 故障排查：动作目标响应超时 / 节点互相发现不到

典型日志：

```
[arm_controller]: Received new action goal
[arm_controller]: Accepted new action goal
[arm_controller.rclcpp_action]: Failed to send goal response ... (timeout): client will not receive response
[fixed_joint_pick_place]: arm transport raised an exception: MOTION_STATE_UNKNOWN: goal response timed out
[task_state_machine]: UNSAFE_OBJECT_STATE:PRE_GRASP:arm transport:FAILED:STATE_UNKNOWN
```

控制器**已经接受**了目标，但 Fast DDS 只给服务响应约 100 ms 的投递窗口；
响应被丢掉后客户端等满 `action_timeout_sec`，只能判定状态未知并安全停机。
同类症状还有：新起的节点看不到 `/navigate_to_pose`、代价地图服务等
（`Timed out waiting for ...`），而服务端其实活着。

根因是 WSL 上 Fast DDS 的残留状态：反复 Ctrl-C / `kill -9` 之后，
`/dev/shm/fastrtps_*` 会留下上百个死进程的共享内存段，`ros2 daemon`
也可能长期挂着旧 participant，新进程就会随机匹配不上端点。

处理办法（**必须在没有 ROS/Gazebo 进程运行时**执行，然后重新拉起仿真）：

```bash
ros2 run mybot reset_dds_cache.sh
```

脚本会停掉 ros2 daemon 并清掉残留段。同一份代码实测对比：清理前连续三次启动
执行器分别卡在 10 s / 55 s / 55 s 的三个**不同**端点上；清理后 0.7 s 通过全部
10 项依赖检查，并完整跑完一次抓取配送。

想彻底不走共享内存（用一点拷贝开销换稳定），可选用仓库里的纯 UDPv4 profile：

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=\
  $(ros2 pkg prefix mybot)/share/mybot/config/fastdds_wsl_udp.xml
```

另外 `fixed_joint_pick_place.yaml` 里 `action_retry_count` 会对“目标响应丢失”
做有限次重试（重发同一个目标，幂等安全），`dependency_timeout_sec` 控制启动时
等待各 action/service 出现的时长。

更详细的开发计划见 [`比赛后续开发任务清单.md`](比赛后续开发任务清单.md)。

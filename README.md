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

## 完整运行流程（从冷启动到一轮跑完）

> **只想照着做？先看 [`启动流程.md`](启动流程.md)** —— 一页速查（确认 → 启动 → 看 → 停 → 排错），
> 比赛当天照着做即可。下面是同样内容的展开版，带每一步的原因。

这一节是**按顺序照做即可**的主流程。每一步都实测可用；每一步失败时该看哪里也写在里面。

### 第 0 步：确认工作区干净（每次跑之前都做）

上一次 `Ctrl-C` 或 `kill` 常留下孤儿进程，**两套 Nav2 会抢 `/cmd_vel`**，表现为机器人完全不动。

```bash
cd ~/dev_ws
pgrep -c -f nav2_ ; pgrep -c -f gzserver     # 两个都应为 0
```

不为 0 就先清（**必须在没有 ROS/Gazebo 进程时执行**）：

```bash
ros2 run mybot start_competition.sh --kill    # 预检 + 清 Fast DDS 缓存 + 启动
```

### 第 1 步：确认已编译，且脚本有可执行位

```bash
cd ~/dev_ws
colcon build --symlink-install
source install/setup.bash
```

> 改过 `worlds/*.world` 或 `urdf/*.xacro` 时，**必须连 `mybot_description` 一起重编**：
> `colcon build --packages-select mybot_description mybot --symlink-install`。
> 只编 `mybot` 的话新 world 不会进 `share/`，Gazebo 会**回退到 `empty.world`**——
> 机器人、`/ATTACH`、`/set_entity_state` 全都不存在，报出的错误却和 DDS 发现故障
> **一模一样**。**排查时先看加载的是哪个 world**（日志里搜 `worlds/`）。

```bash
chmod +x src/yzbot/mybot/scripts/*.py       # edit 工具会丢可执行位
```

### 第 2 步：配置 DeepSeek Key（只做一次）

```bash
ros2 run mybot set_deepseek_key.sh          # 交互粘贴，写入 ~/.config/mybot/deepseek_api_key (600)
ros2 run mybot set_deepseek_key.sh --check  # 查看当前会加载哪一个
```

不配也能跑：解析器会退回到**离线算术**（只认已识别的题型模板），但覆盖面窄，比赛别赌。

### 第 3 步：确定映射规则与出题模式

**这两个开关的默认值都从 YAML 读**（`config/task_parser.yaml`），所以不设环境变量也生效；
环境变量或 `xxx:=值` 命令行仍然优先。

```bash
# 映射规则：1 / 2 / 3 —— YAML 当前默认 = 1
export MYBOT_MAPPING_RULE=2

# 出题模式：比赛模式给生成器追加 11470507，题目按 30 分钟窗口变化
# YAML 当前默认 = true（常开），要临时回练习模式：
export MYBOT_COMPETITION_MODE=false
```

> **两个开关都踩过同一个坑**：launch 里曾把默认值硬编码（`mapping_rule` 写死 `'2'`、
> `competition_mode` 写死 `'false'`），而 launch **总会**把参数传给节点，于是
> **静默覆盖了 YAML** —— 改配置完全没效果。现在三个开关统一通过 `_yaml_default()`
> 以 YAML 为默认来源，并有静态测试防止再犯。

比赛模式的输出比练习模式**多三行**（种子行 + 上下两条分隔线），解析器两种都能正确取到题目行：

```
使用种子: 995389                     ← 同一 30 分钟窗口内种子相同
------------------------------------------------------------
题目: 每个书箱有5本书籍;...
------------------------------------------------------------
```

种子会记进日志（`Generator seed: 995389`），方便和裁判公布的题目对号。

### 第 4 步：启动整栈

```bash
cd ~/dev_ws && source install/setup.bash
ros2 run mybot start_competition.sh
```

它做四件事：预检残留 → 清 Fast DDS 缓存 → 无 GUI 启动 → **等到 `/competition/system_ready` 为 true 并打印还缺什么**。
看到 `✅ 系统就绪` 再往下走。

**默认就是实测过的快配置**：三个参数文件指向仓库里的
[`config/profiles/`](src/yzbot/mybot/config/profiles)（RPP 控制器 @ **1.5 m/s**、机械臂节拍加速、
泊位旋转修正、三条减速区），整轮墙钟实测 **239–355 s**。

> **这里踩过一个后果很严重的坑**：这套快配置原先只存在于 `run_logs/tools/`（被 gitignore）
> 的变体文件里，**只有带上 `NAV_PARAMS_FILE` 等环境变量跑测量脚本才会生效**。
> 于是直接 `start_competition.sh` 跑的是仓库原始配置（**MPPI 控制器 @ 0.8 m/s、
> smoother 上限 1.0**），慢一倍以上。现在快配置已经进仓库并成为 launch 默认值。

要回退到原始配置（MPPI）：

```bash
ros2 run mybot start_competition.sh \
  nav_params_file:=$(ros2 pkg prefix bot_navigation)/share/bot_navigation/param/originbot_nav2_2.yaml \
  executor_params_file:=$(ros2 pkg prefix mybot)/share/mybot/config/fixed_joint_pick_place.yaml
```

| 需求 | 加的参数 |
| --- | --- |
| 要 Gazebo 窗口 | `--gui` |
| 要 MoveIt RViz | `--with-rviz` |
| 先清残留再启动 | `--kill` |
| 自定义就绪超时（秒） | `--wait 600` |

### 第 5 步：任务自动开跑（或按按钮开跑）

系统就绪后，出题程序 → 大模型解析 → 状态机**自动开始**抓取配送，无需人工干预。

**想改成"裁判说开始、你再按"**（对应 Foxglove 布局里的「开始任务」按钮）：

```bash
MYBOT_REQUIRE_START_BUTTON=true ros2 run mybot start_competition.sh
```

打开后状态机会停在 `WAITING_FOR_START`，直到 `/competition/start` 收到 `true`：

```bash
# 命令行模拟按钮
ros2 topic pub --once /competition/start std_msgs/msg/Bool "{data: true}"
```

**默认关闭**，保持自动开跑。按钮可以提前按（会记住），但**每接受一个新任务都会重置**。

### 第 6 步：用 Foxglove 看（推荐）

`start_competition.sh` 默认已带 `foxglove_bridge`（WebSocket 8765）。
Windows 上的 Foxglove 选 **Foxglove WebSocket** 连 `ws://localhost:8765`，再导入布局：

```bash
cp "$(ros2 pkg prefix mybot)/share/mybot/config/foxglove/competition_dashboard_layout.json" \
   /mnt/c/Users/26807/Downloads/
```

三个布局的用途：

| 布局文件 | 用途 |
| --- | --- |
| `competition_dashboard_layout.json` | **评分用大盘**：题目、红/蓝数量、区域、识别画面×2、三维（含地图）、任务进度、当前任务、状态、开始按钮 |
| `competition_layout.json` | 九宫格：原始题目、大模型回复、结构化任务、进度、异常、状态迁移、3D、视觉画面、日志 |
| `competition_indicators_layout.json` | 指标为主的全量话题版 |

详见 [`FOXGLOVE.md`](FOXGLOVE.md)。**面板只有在仿真运行时才有数据**，不跑就是空的。

### 第 7 步：判断这一轮是否成功

三个地方任一即可：

```bash
ros2 topic echo /competition/state --once      # 期望 COMPLETED
ros2 topic echo /competition/progress --once   # 期望 {"completed":5,"total":5}
ros2 topic echo /competition/error --once      # 期望空
```

Foxglove 里看「任务进度」到 5/5、「状态」为 `已释放`、`/competition/error` 无内容。

### 第 8 步：停栈与清理

```bash
# Ctrl-C 之后仍然要确认没有孤儿
pgrep -c -f nav2_ ; pgrep -c -f gzserver       # 都应为 0
ros2 run mybot reset_dds_cache.sh              # 反复启停后清 Fast DDS 残留
```

### 可选：让小车在途中避障（移动障碍）

地图上放 2–3 个**巡逻方块**，逼小车在导航时避障：

```bash
MYBOT_MOVING_OBSTACLES=true ros2 run mybot start_competition.sh
```

**一个开关就够**——它会自动选用 `offic_room_movers.world` 变体。
（早先需要同时设 `MYBOT_WORLD_NAME`，只设一个会得到"开关开着、场景里却没有方块"的
困惑状态；现在 `MYBOT_MOVING_OBSTACLES=true` 单独就生效，显式指定 world 仍优先。）

- **黄色** 方块，尺寸 0.22×0.22×**0.30 m**（高于雷达所在的 0.06 m，所以代价图看得见）
- **运行时生成**，不需要配套 world 变体。**注意**：方块及其 `planar_move` 插件若写进
  `.world`，会在世界加载时**阻塞 Gazebo 的生成队列**，导致机器人自己 spawn 失败
  （`Entity pushed to spawn queue, but spawn service timed out waiting for entity to appear`）。
  节点改为启动后用 `/spawn_entity` 插入，并对失败重试。
- 巡逻段取自小车自己的历史轨迹并做过净空校核（见 `run_logs/tools/make_movers_world.py` 注释）
- **速度是精确的**：方块由**位姿伺服**驱动（节点按里程推算后直接设定位姿），
  不是发 `cmd_vel`。实测设定 0.16/0.30/0.16 m/s 时真实速度为
  0.151/0.283/0.150 m/s——用速度控制时只能达到 0.031/0.118（摩擦每个周期把方块刹停）。
- **推不动**：位置每帧被覆盖，实测最大垂偏 0.7 mm，永远严格贴在自己的巡逻线上
- `mover_2` 刻意调快（0.30 m/s，其余 0.16），让障碍是"横穿"而不是"同向陪跑"

**默认关闭**。实测在该配置下任务仍能 5/5 完成。

### 可选：本地测量与验证（`run_logs/` 是 gitignore 的，新克隆里没有）

日常复现一整轮并落盘全部证据：

```bash
bash run_logs/tools/run_one_round.sh <标签> [最大墙钟秒]
```

常用环境变量：`NAV_PARAMS_FILE` / `EXECUTOR_PARAMS_FILE` / `STATE_MACHINE_PARAMS_FILE` 指定参数变体，
`MAPPING_RULE`、`COMPETITION_MODE`、`MOVING_OBSTACLES`、`REQUIRE_START_BUTTON`、`MYBOT_WORLD_NAME`、
`READY_TIMEOUT`、`DRY_RUN=on`（只预检不真跑）。结果落在 `run_logs/<标签>_<时间戳>/`。

跑完核验与拆解：

```bash
python3 run_logs/tools/verify_round_evidence.py run_logs/<目录>   # 证据完整性（5/5、清理、哈希…）
python3 run_logs/tools/mono_stage_breakdown.py run_logs/<目录>    # 每块物块的分段耗时
python3 run_logs/tools/op_timings.py run_logs/<目录>              # 逐操作计时明细
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

出题程序（`TMSCQtest_x86_x64.bin`，x86_64 Linux ELF）每次运行输出一道中文应用题。
**练习模式输出一行；比赛模式（追加参数 `11470507`）输出四行**（种子行 + 分隔线 + 题目 + 分隔线），
解析器两种都能正确取到题目行。
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

### 赛前公布的映射规则：三条全部实现，可随时切换

规则公布原文（省赛自选一条，**国赛由裁判在计时前挑选**）：

| # | 规则原文 | 颜色由谁定 | 区域由谁定 |
| - | -------- | ---------- | ---------- |
| 1 | "x 代表红色，y 代表蓝色；数值大于等于 3 的去 A 区，其他颜色去 C 区" | **字母**（x=红，y=蓝） | **数值**（≥3→A，否则 C） |
| 2 | "xy 中，数值大于等于 3 值的代表红色，否则代表蓝色；x 去 B 区，y 去 A 区" | **数值**（≥3→红，否则蓝） | **字母**（x→B，y→A） |
| 3 | "蓝色去 B 区，红色去 C 区；xy 中，数值大的是蓝色，数值少的是红色" | **比较**（大=蓝，小=红） | **颜色**（蓝→B，红→C） |

**注意三条规则是结构不同的**，不只是常数不同：颜色分别由字母 / 数值 / 比较决定，
区域分别由数值 / 字母 / 颜色决定。因此代码里按规则分派（`task_parser.py` 的
`_rule1/_rule2/_rule3`），不是改一个查表。

四种数量分配下的全部结果（已逐条对照原文验过）：

| x,y   | 规则 1 | 规则 2 | 规则 3 |
| ----- | ------ | ------ | ------ |
| (1,4) | red1→C, blue4→A | blue1→B, red4→A | red1→C, blue4→B |
| (2,3) | red2→C, blue3→A | blue2→B, red3→A | red2→C, blue3→B |
| (3,2) | red3→A, blue2→C | red3→B, blue2→A | blue3→B, red2→C |
| (4,1) | red4→A, blue1→C | red4→B, blue1→A | blue4→B, red1→C |

出题程序保证 `x,y >= 1` 且 `x + y = 5`，所以 **x ≠ y 恒成立**，规则 3 不会遇到并列；
真出现并列时解析器会**报错而不是猜**。

**切换方式（三种，优先级从高到低）**：

```bash
# 1) 命令行（临时，最高优先）
ros2 launch mybot competition_bringup.launch.py mapping_rule:=3
# 2) 环境变量
MYBOT_MAPPING_RULE=3 ros2 run mybot start_competition.sh
# 3) 配置文件（默认来源，改完一直生效）
#    src/yzbot/mybot/config/task_parser.yaml 里的 mapping_rule
```

> **踩过的坑**：这个 launch 参数曾经把默认值**硬编码成 `'2'`**，而 launch 总会把参数
> 传给节点，于是它**静默覆盖了 YAML**——改了 `mapping_rule` 却毫无效果。现在
> launch 的默认值改为**读取 YAML 本身**，YAML 是唯一默认来源，上面两种方式仍可覆盖。

生效的规则会**打印在日志里**并发布到 `/competition/task_rules`，方便计时前与裁判核对：

```
映射规则 2 生效：x=2, y=3 -> 2件blue去B区, 3件red去A区
```

各规则用到的参数（`config/task_parser.yaml`）：

| 规则 | 参数 |
| ---- | ---- |
| 1 | `color_x`、`color_y`、`zone_by_count`（按数量 1,2,3,4 查表，可表达任意分档） |
| 2 | `red_min_count`（默认 3）、`zone_x`、`zone_y` |
| 3 | `zone_blue`、`zone_red` |

**大模型只负责抽取两个数量，颜色和区域一律由代码算**，所以模型读错规则也传不到任务里。

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

## 改了物块/取物点的位置？先画出地图来看

世界文件用米、地图是 PGM 加一个 yaml（原点与分辨率），两者混起来很容易搞错。
仓库里带了一个工具，把物块位置**直接画在地图上**：

```bash
python3 tools/show_positions.py            # 写出 tools/positions.png
```

图上标出 10 个物块（红/蓝）、它们的取物点、以及 A/B/C 三个区，
同时打印世界坐标、地图像素、是否可通行、离哪个区最近。

**它会替你检查两个最容易改错的地方：**

1. **每个物块名在世界文件里出现两次，生效的是 `<state>` 存档** —— 而且是其中的
   **链接位姿** `<model><link><pose>`，不是模型位姿。实测：模型定义写 (−8,−3) 时仿真仍报
   (−9.995, 0.015)，即 state 的值。**只改模型定义完全不起作用，且不报错。**
   用 `--write-back` 会一次改全三处；手工改要三处都改。
2. **光改世界文件不够** —— 执行器开到 `config/fixed_joint_pick_place.yaml` 的
   `pickup_<物块名>_pose`，到了再靠视觉找方块。物块挪了、取物点没挪，车就会停在旧位置
   然后找不到方块。两者相差超过 0.8 m 时工具直接告警。

改完世界文件记得重编：`colcon build --packages-select mybot_description mybot --symlink-install`

## 比赛可视化：Foxglove 状态面板

`competition_bringup.launch.py` 默认带 `foxglove:=true`，会额外起
`foxglove_bridge`（WebSocket，8765）。Windows 上的 Foxglove 桌面版选
**Foxglove WebSocket** 连 `ws://localhost:8765` 即可（WSL 是 mirrored 网络模式，
localhost 直接互通），再导入仓库里的布局文件。**三个布局都全部绑定真实话题，没有占位文字**：

| 布局文件 | 面板 |
| -------- | ---- |
| `competition_dashboard_layout.json` | **评分大盘**：题目、红/蓝数量、红/蓝区域、识别画面×2、三维、任务进度、当前任务、状态、**开始按钮** |
| `competition_layout.json` | 九宫格：原始题目、大模型回复、结构化任务、进度、异常、状态迁移、3D、视觉画面、日志 |
| `competition_indicators_layout.json` | 指标为主的全量话题版 |

三维面板的**固定坐标系是 `map`**，并绑定 `/map`（静态地图）、`/global_costmap/costmap`
（全局代价图）与 `/local_costmap/costmap`（默认隐藏）。

> **踩过的坑**：3D 面板曾经**只绑了雷达和路径、没绑地图**，而文档标题却写着"3D（地图/…）"。
> 排查时先分清两件事：**面板里没数据** 与 **话题根本没桥接出去**。确认方法：
> `foxglove_bridge` 的 `topic_whitelist` 默认是"全部"，且 `/map` 确实出现在 bridge 的
> 频道列表里——**数据一直有，是布局没绑**。另外**不跑仿真时 3D 必然是空的**，这不是布局问题。

「开始按钮」绑 `/competition/start`（`std_msgs/Bool`），用法见
[完整运行流程 第 5 步](#第-5-步任务自动开跑或按按钮开跑)。

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
| 完整 bringup + 抓取执行器导航中 | **0.18** | 满载导航时最差 |

上表最后一行的 **0.18 是旧基线（底盘限速 0.33 m/s）时的读数**。同一台机器后来实测：
**RTF 在各阶段基本平稳在 ~0.90**，总 CPU 只用 8.3/16 核，所以"RTF 是瓶颈"这个结论
当时下早了——真正的瓶颈是**几何**（走廊宽度限制了可用速度），不是实时率。

当时的错误结论「单块 45 s 靠调参数达不到」也已被推翻：把底盘限速从 0.33 提到 1.5 m/s、
配三条减速区（消除 365 次 `collision ahead`）修掉泊位多圈旋转之后，实测
**单块 50–70 s、整轮 239–355 s**（5 块全成）。详细的分阶段提速与全部负结果见
`run_logs/tools/整轮提速审计-20261009.md`（该目录 gitignore）。

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

### 三个容易误判的坑（实测）

**① world 没装 → Gazebo 静默回退到 `empty.world`。**
改了 `worlds/*.world` 只编 `mybot` 的话，新文件不会进 `share/`，Gazebo 找不到就退回空场景：
机器人、`/ATTACH`、`/set_entity_state` 全都不存在。**症状与 DDS 发现故障完全一样**
（`client never matched; rebuilt for DDS discovery`），但根因完全不同。
**排查第一步永远是：日志里搜 `worlds/`，看加载的到底是哪个 world。**

```bash
grep -oE "worlds/[a-z_0-9]+\.world" run_logs/<目录>/sim.log | sort -u
```

**② 数组型 ROS 参数没有"两全"的声明写法。** 实测四种组合：

| 写法 | 参数文件给了数组 | 参数文件没给（生产） |
| ---- | --- | --- |
| `declare_parameter(name, rclpy.Parameter.Type.X)` | ✅ | ❌ `ParameterUninitializedException` |
| `declare_parameter(name, [])` | ❌ `InvalidParameterTypeException` | ✅ |
| **非空且类型正确的默认值**（如 `[0.0]` 哨兵） | ✅ | ✅ |

所以 `slow_zones` 用哨兵 `[0.0]`（长度 < 4 视为未启用），`generator_args` 干脆改成
**字符串**（空格分隔）以避开类型歧义。**这个坑曾经潜伏在执行器里**：生产 YAML 没有
`slow_zones`，平时都靠变体文件提供值才没暴露，一旦按生产配置启动就会崩。
已加静态测试（AST 扫描禁止把 `Parameter.Type` 当值传入）防回归。

**③ 改了配置却"不生效"，先怀疑 launch 覆盖。** `mapping_rule` 就发生过：launch 参数
默认值被硬编码，而 launch 总会传参，于是静默压过 YAML。**判断方法**：看节点启动时
打印的实际生效值（如 `映射规则 N 生效`），而不是看你改了哪个文件。

更详细的开发计划见 [`比赛后续开发任务清单.md`](比赛后续开发任务清单.md)。

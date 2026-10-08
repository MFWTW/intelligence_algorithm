# Foxglove 可视化接入（赛题第七项）

目标：评委在 Windows 上打开 Foxglove，就能看到**真实 ROS 2 话题**里的
题目、大模型回复、结构化任务、任务状态、进度、地图/路径/雷达、视觉识别画面
和异常日志——没有一行静态占位文字。

```
Gazebo + Nav2 + MoveIt + 任务状态机 (WSL2)
        │  ROS 2 DDS
        ▼
foxglove_bridge  (WebSocket, 默认 8765)
        │  ws://localhost:8765
        ▼
Foxglove 桌面版 (Windows) ── 布局文件 competition_layout.json
```

本机环境已经满足两个前提，不用再折腾网络：

- WSL 是 **mirrored 网络模式**（`C:\Users\26807\.wslconfig` 里的
  `networkingMode=mirrored`），Windows 和 WSL 共用 localhost，所以 Foxglove
  直接连 `ws://localhost:8765`，**不需要**记 WSL 的 IP，也不需要端口转发。
- Foxglove 桌面版已装在 Windows：
  `C:\Users\26807\AppData\Local\Programs\foxglove\Foxglove.exe`。

---

## 1. 安装 bridge（每台机器一次）

```bash
sudo apt update
sudo apt install -y ros-humble-foxglove-bridge
```

选 Foxglove 而不是 `rosbridge_server` 的原因：`/competition/task`、`/competition/state`
这类话题是 **RELIABLE + TRANSIENT_LOCAL（latched）**，foxglove_bridge 会按发布者
QoS 订阅，后打开的 Foxglove 仍能立刻看到最后一帧任务；rosbridge 走固定 volatile
QoS，晚连接就只能等下一次发布。图像话题上 foxglove_bridge 也走二进制通道，
比 rosbridge 的 base64 JSON 轻得多。

> 如果赛题/评委明确要求 rosbridge（例如用 coStudio），再补装：
> `sudo apt install -y ros-humble-rosbridge-suite`，
> 启动 `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`，
> 在 Foxglove 里选 **Rosbridge** 连接 `ws://localhost:9090` 即可；本仓库的
> 话题命名不用改。

## 2. 启动

推荐：跟仿真一起起（一条命令，赛前就用这条）：

```bash
cd ~/dev_ws
source install/setup.bash
ros2 launch mybot competition_bringup.launch.py foxglove:=true
```

`foxglove` 默认就是 `true`；没装 bridge 的机器只会打印一条警告然后照常启动，
不会把整个 bringup 拖挂。不想开就 `foxglove:=false`，换端口用
`foxglove_port:=8765`。

仿真已经在跑、只想补一个 bridge：

```bash
ros2 launch mybot foxglove_bridge.launch.py
```

自检（bridge 起来后）：

```bash
ss -tlnp | grep 8765          # 应该看到 0.0.0.0:8765 在 LISTEN
ros2 topic list | grep competition
```

## 3. 在 Foxglove 里连接

1. 打开 Windows 上的 Foxglove。
2. `Open connection…` → 选 **Foxglove WebSocket**。
3. URL 填 `ws://localhost:8765`，`Open`。
4. 左侧 Topics 里应立刻出现 `/competition/…`、`/scan`、`/plan`、`/tf` 等。

## 4. 导入布局（可复用布局文件）

布局在仓库里：`src/yzbot/mybot/config/foxglove/competition_layout.json`，
编译后也会装到 `$(ros2 pkg prefix mybot)/share/mybot/config/foxglove/`。

复制到 Windows 能选到的地方：

```bash
cp "$(ros2 pkg prefix mybot)/share/mybot/config/foxglove/competition_layout.json" \
   /mnt/c/Users/26807/Downloads/foxglove_competition_layout.json
```

Foxglove → 右上 `Layouts` → `Import from file…` → 选上面那个文件。

导入后是九宫格式的九块面板，全部已绑定真实话题：

| 面板              | 话题                        | 类型                  | 发布者               |
| ----------------- | --------------------------- | --------------------- | -------------------- |
| 原始题目          | `/competition/task_problem` | `std_msgs/String`     | `task_parser`        |
| 大模型原始回复    | `/competition/task_raw_reply` | `std_msgs/String`   | `task_parser`        |
| 结构化任务        | `/competition/task`         | `std_msgs/String`（JSON，latched） | `task_parser` |
| 进度              | `/competition/progress`     | `std_msgs/String`（JSON，latched） | `task_state_machine` |
| 异常              | `/competition/error`        | `std_msgs/String`（latched） | `task_state_machine` |
| 状态迁移          | `/competition/state`        | `std_msgs/String`（latched） | `task_state_machine` |
| 3D（地图/路径/雷达/机器人） | `/scan`、`/plan`、`/local_plan`、`/amcl_pose`、`/odom`、`/tf`、`/joint_states`、`/robot_description` | LaserScan / Path / Pose / Odometry / TF / JointState | Nav2、Gazebo、`robot_state_publisher` |
| 视觉识别画面      | `/vision/debug_image`       | `sensor_msgs/Image`   | `fixed_joint_pick_place.py` |
| 日志              | `/rosout`                   | `rcl_interfaces/Log`  | 全部节点             |

3D 面板里已经配好两层：`foxglove.Grid` 网格，以及 `foxglove.Urdf`
（`sourceType: topic`，取 `/robot_description`，关节跟随 `/joint_states`），
所以机器人模型会跟着 Gazebo 一起动。

## 5. 赛题还要求、但没塞进布局的面板

`Add panel` 里点两下、在设置里选话题即可，都是现成话题：

- **拿不准就加 Indicator**：话题 `/competition/system_ready`（`std_msgs/Bool`），
  系统自检通过变绿——评委最关心这个。
- `/competition/task_rules`（映射规则）、`/competition/task_status`
  （`GENERATING/PARSING/REPAIRING/SUCCESS/FAILED:*`）、`/competition/system_health`
  （十项自检明细）：Raw Messages 面板。
- `/competition/current_object`（当前方块）、`/competition/vision_alignment`
  （视觉对位状态）、`/competition/executor_state`、`/competition/executor_error`
  （机械臂执行器状态/错误）：Raw Messages 面板。
- `/competition/retry_count`（`std_msgs/Int32`）：Plot 面板，路径填
  `/competition/retry_count.data`。
- `/camera/image_raw`（原始相机）：Image 面板。
- `/joint_states`、`/tf`：Plot / Raw Messages 面板，用于展示机械臂与夹爪状态。

> 排错提示：任务清单第七节里写的 `/competition/question`、`/competition/mapping_rule`、
> `/competition/llm_raw_response`、`/competition/tasks` 是**当初的计划名**，
> 代码里实际发布的是上表的名字（`task_problem` / `task_rules` /
> `task_raw_reply` / `task`）。以真实话题为准，规则要求的是"绑定真实 topic"，
> 不是绑定某个名字。

## 6. 双机位/第二台电脑看

如果评委用另一台机器看，把地址换成 Windows 的局域网 IP
（`ws://192.168.x.x:8765`），并在**管理员 PowerShell** 放行端口：

```powershell
New-NetFirewallRule -DisplayName "Foxglove Bridge 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| Foxglove 连不上 | `ss -tlnp \| grep 8765` 确认 bridge 在跑；确认启动的是 `foxglove:=true`；确认服务端装在这台 WSL 里 |
| 话题列表是空的 | 仿真没起来（bridge 只在有客户端订阅时才订阅话题）；先 `ros2 topic list` 看有没有数据 |
| `/competition/*` 面板没内容 | 那些话题是 latched，只有任务流程真的跑过才有值；跑一次 `ros2 topic pub` 或走完整流程 |
| 图像卡顿 | 降低相机分辨率/帧率，或调大 `send_buffer_limit:=100000000`，或在 `foxglove_bridge.launch.py` 里把 `topic_whitelist` 收窄到只用得上的话题 |
| 时间轴/曲线时间对不上 | `/competition/*` 是**无 header** 的 String，Foxglove 用到达时间打戳；bridge 故意用 `use_sim_time:=false`，别改成 true |
| bridge 起来后 DDS 又开始抽风 | 按 README 的《故障排查》先 `ros2 run mybot reset_dds_cache.sh`；仍不稳就 `foxglove:=false` 单独在另一个终端起 bridge |

### 现场提醒：让「结构化任务」面板全程可见

`task_parser` 解析完只把 `/competition/task_problem`、`/competition/task_raw_reply`、
`/competition/task` 这些 latched 话题保持 `parser_linger_sec` 秒（默认 **45 s**），
之后节点退出、DDS 里的 latched 缓存随之消失。也就是说**评委如果在解析 45 秒之后
才打开 Foxglove，题目和大模型回复这两块面板会是空的**（状态机的 `/competition/state`、
`/competition/progress` 不受影响，它全程活着）。

想让任务文本整场都可读，把 linger 拉长到覆盖全程即可（不影响出题，解析只做一次）：

```bash
ros2 launch mybot competition_bringup.launch.py parser_linger_sec:=900
```

## 8. 本机实测记录（2026-10-03，foxglove_bridge 3.5.0）

用自写的 Foxglove-WebSocket 客户端直连 bridge 压测了一遍，结论都是实测、不是推断：

| 验证项 | 结果 |
| --- | --- |
| WebSocket 握手 | `HTTP/1.1 101 Switching Protocols`，子协议协商为 `foxglove.sdk.v1` |
| serverInfo | `supportedEncodings=['cdr','json']`，capabilities 六项齐全（`assets`/`clientPublish`/`connectionGraph`/`parameters`/`parametersSubscribe`/`services`） |
| 连接即推频道表 | 一次推送 8 个频道：`/competition/*` 五个 + `/rosout` + `/parameter_events` + sysinfo |
| 按 `channelId` 订阅后收数据 | 数据走**二进制帧**，帧内是 CDR，String/Bool/Int32 均正确解出 |
| **晚加入也能拿到 latched 值** | **5/5**——`/competition/state`、`/competition/task`、`/competition/progress`、`/competition/system_ready`、`/competition/retry_count` 在一个"发布之后才连接"的客户端里全部重现 |
| bridge 订阅 QoS | `ros2 topic info /competition/state -v` 显示 `foxglove_bridge` 为 `RELIABLE + TRANSIENT_LOCAL` 订阅者，与发布者完全匹配 |
| launch 参数注入 | `ros2 param get /foxglove_bridge topic_whitelist` → 字符串数组 `['.*']`，`send_buffer_limit` = 50000000，`capabilities` 六项 |

> 这条实测同时说明：rosbridge 的对比结论（latched 话题晚连接会丢）不是这个 bridge 的问题——
> foxglove_bridge 会按发布者 QoS 建订阅，所以**评委随时打开 Foxglove 都能看到最后一帧任务状态**，
> 唯一的例外是上面那条 `parser_linger_sec`（那是发布者退出导致的，跟 bridge 无关）。

## 9. 相关文件

- `src/yzbot/mybot/launch/foxglove_bridge.launch.py`：bridge 启动文件（端口、
  白名单、仿真时间等参数）。
- `src/yzbot/mybot/launch/competition_bringup.launch.py`：`foxglove:=` /
  `foxglove_port:=` 参数。
- `src/yzbot/mybot/config/foxglove/competition_layout.json`：可复用布局文件。

---

## 10. 赛题评分面板（按赛题「三、评分系统」做的那一版）

赛题要求画布上实时显示：**任务题目 / 放置区域 / 每种资源抓取数量 / 任务进度 /
工作状态 / 末端状态（抓取、放置）**，并且数据来自「参赛队伍发布的标准化 ROS topics」。

难点在于 `/competition/task`、`/competition/progress` 是 **JSON 字符串**，
Foxglove 的 Indicator 只能取 `话题.字段`，没法从字符串里取字段。所以按赛题要求加了一个
桥接节点，把字段拆成标准话题：

```bash
ros2 run mybot dashboard_bridge.py     # 已默认随 competition_bringup 启动
```

| 看板话题（全部 latched） | 类型 | 内容 |
| --- | --- | --- |
| `/competition/dashboard/problem` | String | 任务题目（去掉重复的「题目:」前缀） |
| `/competition/dashboard/red_count` | Int32 | 红色需要抓几个 |
| `/competition/dashboard/blue_count` | Int32 | 蓝色需要抓几个 |
| `/competition/dashboard/red_zone` | String | `红色到A区`（放置区域） |
| `/competition/dashboard/blue_zone` | String | `蓝色到C区` |
| `/competition/dashboard/grab_index` | Int32 | 当前第几个（1 起，封顶为总数） |
| `/competition/dashboard/total` / `completed` | Int32 | 总数 / 已完成数 |
| `/competition/dashboard/work_state` | String | 中文工作状态：识别资源 / 前往资源点 / 执行抓取 / 前往放置区 … |
| `/competition/dashboard/gripper_state` | String | 末端状态：未抓取 / 抓取中 / 已抓取 / 携带中 / 放置中 / 已释放 |
| `/competition/dashboard/current_cube` | String | 当前方块（`red_cube_2`） |

布局文件：`src/yzbot/mybot/config/foxglove/competition_dashboard_layout.json`，
版式与赛题示意图一致：

```text
┌──────────┬──────────┬──────────┬──────────┐
│ 红色数量 │ 蓝色数量 │ 蓝色     │ 红色     │  ← 数量 + 目标区域
├──────────┴──────────┼──────────┴──────────┤
│ 识别画面（相机）    │ 识别画面（识别框）  │  任务进度：第 N 个
├─────────────────────┴─────────────────────┤  当前任务：前往资源点
│                                           │  状态：已抓取
│              三维（雷达/路径/机器人）     │  任务题目
└───────────────────────────────────────────┘
```

导入：Foxglove → `Layouts` → `Import from file…` → 选该文件（也在
`C:\Users\26807\Downloads\competition_dashboard_layout.json`）。

说明两点：

- 「识别画面（识别框）」绑 `/vision/debug_image`，只在视觉精定位那几秒有画面（省 CPU
  的刻意设计）；要它常亮可以让我加 `visual_debug_always` 开关。
- 示意图里的「开始按钮」在本工程没有对应语义（任务由出题程序 + 大模型自动生成并自动
  开跑），所以那一格换成了赛题要求的**任务题目**面板。

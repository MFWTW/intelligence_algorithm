# 三轮实测（mapping_rule 1/2/3）— 2026-11-25 最终版

代码：`MFWTW/intelligence_algorithm` main@09e2470 + 三个补丁（见 §5），`colcon build --symlink-install` 通过。
题面：同一 seed（997547）连跑三轮，只有 `mapping_rule` 不同，所以三轮的"时间差"只来自映射规则与现场，不来自题面。
每轮：`WATCH=1 COLLISION_PROBE=1 bash tools/run_round.sh <rule> <out>`，外部 watcher 只观察不停轮。

| 轮 | 目录 | 映射 | 交付 | 墙钟 | 5 块区间合计 | 分段累加 | nav | creep | vision | arm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | `runs/2026-11-25_1415_rule1_final` | 1 | 5/5 | 290.1 s | 287.0 s | 313.5 | 198.3 | 33.6 | 34.2 | 47.4 |
| r2 | `runs/2026-11-25_1415_rule2_final` | 2 | 5/5 | 264.9 s | 262.6 s | 284.7 | 178.6 | 29.4 | 29.1 | 47.6 |
| r3 | `runs/2026-11-25_1410_rule3_probe` | 3 | 5/5 | 281.3 s | 278.8 s | 294.9 | 194.1 | 23.3 | 30.0 | 47.5 |

三个口径的区别（`analysis.json` 的 `totals.wall_s` / `objects[].wall_s` / `timings`）：

- **墙钟** = 状态机进入 `PARSING_TASK`（按下开始、开始读题）→ `State -> COMPLETED`。含读题约 3 s 和 5 块全流程，**不含** Gazebo/Nav2 的 1–3 分钟启动。
- **5 块区间合计** = 每块 `Started <obj>` → `DELIVERY COMPLETED: <obj>` 的区间相加，比墙钟小约 3 s（读题 + 收尾）。
- **分段累加（"物块合计"）** = 每块的 `TIMING` 标签相加，比区间合计大 16–27 s：导航段与视觉对齐/让行段的计时器有重叠，同一段墙钟被两个标签各算一次（每块 2–6 s）。**所以它是"各段之和"，不是一局的总时长，不能当比赛用时看。**

nav 占 62–66%，arm 每块恒定 9.5 s（夹取/放置节拍不变），vision 5.1–10.9 s。

## 1. 逐物块（秒）

| 轮 | 物块 | 仓库 | 合计 | nav | creep | vision | arm |
|---|---|---|---:|---:|---:|---:|---:|
| r1 | red_cube_1 | C | 57.7 | 29.6 | 7.6 | 10.9 | 9.5 |
| r1 | blue_cube_3 | A | 52.4 | 32.1 | 5.0 | 5.6 | 9.5 |
| r1 | blue_cube_4 | A | 60.9 | 39.4 | 6.3 | 5.5 | 9.5 |
| r1 | blue_cube_5 | A | 71.8 | 49.3 | 6.9 | 6.0 | 9.5 |
| r1 | blue_cube_1 | A | 71.2 | 47.8 | 7.9 | 6.1 | 9.4 |
| r2 | blue_cube_1 | B | 41.6 | 22.9 | 3.6 | 5.4 | 9.5 |
| r2 | red_cube_1 | A | 56.6 | 33.9 | 7.3 | 5.8 | 9.5 |
| r2 | red_cube_2 | A | 48.5 | 25.8 | 7.5 | 5.6 | 9.5 |
| r2 | red_cube_5 | A | 61.9 | 39.7 | 6.9 | 5.7 | 9.5 |
| r2 | red_cube_4 | A | 76.5 | 56.3 | 4.0 | 6.6 | 9.5 |
| r3 | red_cube_1 | C | 65.1 | 42.0 | 6.3 | 7.3 | 9.5 |
| r3 | blue_cube_1 | B | 46.5 | 26.7 | 4.2 | 5.9 | 9.5 |
| r3 | blue_cube_2 | B | 54.5 | 34.8 | 4.5 | 5.7 | 9.5 |
| r3 | blue_cube_5 | B | 65.3 | 45.0 | 4.7 | 6.0 | 9.5 |
| r3 | blue_cube_3 | B | 63.9 | 45.7 | 3.5 | 5.1 | 9.5 |

图：`analysis/fig1_per_object.png`（每块 nav/creep/vision/arm 堆叠）。

## 2. 车"停在那儿"的时间去哪了

`tools/dwell_report.py`：用 Gazebo 真值，车身在同一 0.5 m 范围内待满 2.5 s 记一段，再用 TIMING 行把每段归到 phase。

| 轮 | 停留段 | 合计 | 占墙钟 | 泊位作业（arm/creep） | 途中停等 |
|---|---:|---:|---:|---:|---:|
| r1 | 10 | 82.7 s | 30% | 78.6 s（95%） | 4.6 s + 4.1 s |
| r2 | 8 | 73.6 s | 30% | 73.6 s（100%） | 0 |
| r3 | 10 | 92.1 s | 35% | 77.7 s（84%） | 14.4 s |

- **泊位作业是大头，且是设计内的**：每个仓库 10–11 s（`Open gripper` / `Lower to place`），抓取 6.6–8.1 s（`Move to grasp` + `Close gripper`，含 2 次视觉对齐）。arm 节拍恒定 9.5 s/块，说明机械臂不是瓶颈，也不是异常。
- **真正的"途中久留"只有 3 段**：r1 4.6 s @(+1.74,-5.18)（collision_ahead×3 + band_start×2）；r1 4.1 s @(-4.86,-5.95)（蠕行抓取段，无异常事件）；r3 14.4 s @(-1.25,-4.78)（yield_stop×3 + band_start×1，等巡逻箱通过）。
- 全轮 `stale odometry` **0 次**（打补丁前同题是 9/3/3 次），说明 odom 时效那个洞已堵上。

图：`analysis/fig2_dwell.png`（蓝=泊位作业、红=途中停等，柱右标主导原因）。

## 3. 避障

| 轮 | collision ahead | planner 失败 | yield 事件 | 进入 band | 最小 hull 间隙 | 最小 policy 间隙 | 实际接触 |
|---|---:|---:|---:|---:|---:|---:|---|
| r1 | 12 | 2 | 29 | 2 | +0.138 m（mover_2 @69.8 s） | -0.032 m | 无 |
| r2 | 24 | 0 | 29 | 2 | +0.067 m（mover_3 @211.6 s） | -0.083 m | 无 |
| r3 | 7 | 0 | 52 | 2 | +0.166 m（mover_2 @243.4 s） | +0.009 m | 无 |

- **"障碍物擦着车过去"确实存在，但没有接触**：r2 的 mover_3 在 211.6 s 从停住的车旁通过，**车体外壳最近只差 6.7 cm**（policy 层已侵入 8.3 cm，所以 collision monitor 一直在报警）；`contact_analysis.py` 判定 `contacts: none`、`pushed while stopped: none`。
- 这是设计使然：`moving_obstacles.py` 的 keep-out 是 0.75 m，箱在里面**保持自己的优先权和运动**（日志原文 `the box keeps its priority and its motion unchanged`），车必须停/让。所以"压着车过去"= 车已让到静止、箱按原轨迹擦过，**不是车被推开**。
- r1 的 mover_2 中心距最低 0.46 m（低于 0.47 m 的 shell 触线），但 hull 间隙仍 +0.138 m：hull 是 0.41×0.37 的方盒，半对角 0.276 m，箱半宽 0.11 m，几何上不接触。

图：`analysis/fig3_movers.png`（每条 mover 的中心距时间序列，横线是 shell touch 0.47 m / guard 0.42 m）。

## 4. 三轮里最贵的一次：为什么规则 3 之前必死

证据（`runs/2026-11-25_watch_rule3`、`runs/2026-11-25_watch_rule2`，两次同一根因）：

1. `GridBased: failed to create plan with tolerance 0.50` 连续 9–15 次；`behavior_server` 的 backup / spin 全部 `Collision Ahead - Exiting ...` 失败；18 s 后 `bt_navigator: Goal failed` → `status=6` → `warehouse X: no verified recovery for unclassified failure.` → 载货 → `UNSAFE_OBJECT_STATE` → 整轮报废。
2. 解码 `probe.jsonl` 里的 local costmap：车开进某点后，**车自身 0.7 m 内出现 287–290 个致命格**（最近 0.02 m），而 4 s 前是 0。同期 scan 最小量程被削到 0.50 m（= `range_min`），即真有东西贴到 0.5 m 内。
3. 画图（`analysis/debug_costmap_rule3.png`）确认：不是假障碍，车确实挤进了西侧墙的膨胀带，NavFn 起点非法，而 Nav2 拒绝后退 —— 死锁。

## 5. 改了什么（三个补丁，均已构建、单测通过）

1. **odom 未来时间戳**（`fixed_joint_pick_place.py:5065` `_recovery_stamp_fresh`）：`-0.05 <= age` → `-1.0 <= age`。负 age 只可能是最新样本，原来超过 50 ms 就直接判 `stale odometry`，把受检后退变成"0.00 m 退让"，band 内白等 6–12 s。
2. **载货壳收窄**（`config/fixed_joint_pick_place.yaml` 与 `config/profiles/executor_fast.yaml` **两处都要改**，profile 后加载会覆盖）：`loaded_robot_radius_m 0.35 → 0.28`、`loaded_inflation_radius_m 0.6 → 0.45`，`global_radius_margin_m` 保持 0.10（余量 0 会让 NavFn 贴着内切带走，这是受控试验结论）。日志随之变成 `local radius=0.28 m, global radius=0.38 m`。
3. **卡死自救 `_wedged_retreat` / `_wedged_retry`**：attempt 耗尽、`no verified recovery` 之后，**不再直接判腿失败**，而是用激光净空（不依赖代价栅格）向宽的一侧后退 0.25 m，再重入该腿一次；每条腿最多一次；净空 <0.35 m 只告警不硬退。
4. **总时长上限架空**：`navigation_timeout_sec 300 → 1800`（两处配置），只当兜底，不再当判据；真正的停滞判据交给外部 watcher（15 s 内没走出 0.5 m 方框）。
5. **`tools/watcher.py`（外部监督）**：订阅 `/odom`、`/cmd_vel`、`/competition/state` + tail `stack.log`；15 s / 0.5 m 方框判停滞；**跳过泊位装卸状态**（DETECTING/PRE_GRASP/GRASPING/VERIFY_\*/PLACING/NEXT_OBJECT/LIFTING，这些状态车身本来就该不动）；输出 `watcher/timeline.csv`、`stall_NN.md`（判 JAM 还是 WAIT、主导日志原因、最后一条 Begin navigating、120 行日志窗口）。`run_round.sh` 的 `WATCH=1` 已经接上，`WATCH_KILL=0` 只观察。

## 6. 复现

```bash
cd ~/intelligence_algorithm-main
source /opt/ros/humble/setup.bash && source install/setup.bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
WATCH=1 COLLISION_PROBE=1 bash tools/run_round.sh 1 runs/2026-11-25_1415_rule1_final
python3 tools/per_object_timings.py --dir runs/2026-11-25_1415_rule1_final --json > runs/.../objects.json
MPLCONFIGDIR=$PWD/.mplconfig python3 tools/plot_round_analysis.py --dir <d1> --dir <d2> --dir <d3> --out analysis
```

## 7. 仍未闭环

- 14:00 那次规则 3（`runs/2026-11-25_1405_rule3_wedge2`）在 7.05 s 处失败：`YIELD stop` → `holding still` → 13 ms 后 `YIELD arc (swept-stop)` → 直接 `Pick-and-place stopped`。已在 `_navigate_once__timed` 的两条静默出口（hold 后 `deadline_expired()`、`_resume_nav_goal()` 返回 None）加了 WARN。随后同代码连跑两轮规则 3 都是 5/5，**该路径未再触发，尚未确证**是哪一条。
- watcher 目前只做到"发现并报告"；`--kill` 默认在 `run_round.sh` 里是开的，但本次三轮都用 `WATCH_KILL=0` 观察，停轮动作还没在真实轮次里验证过。

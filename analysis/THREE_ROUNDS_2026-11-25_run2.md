# 三轮实测（mapping_rule 1/2/3）— 2026-11-25 复跑

## 1. 运行条件

- 代码：`main@09e2470` + 本轮已合入的四个补丁（odom 未来时戳窗口、载货壳 0.28/0.38、`_wedged_retreat`/`_wedged_retry`、`navigation_timeout_sec=1800` 兜底）。
- `colcon build --symlink-install` → `Summary: 7 packages finished [1.43s]`。
- 三轮同一题面 **seed 997549**（"每个衣柜有5件衣物…小美想要3件T恤2条裤子"），只改 `config/task_parser.yaml` 的 `mapping_rule`：
  - `映射规则 1 生效：x=3, y=2 -> 3件red去A区, 2件blue去C区`
  - `映射规则 2 生效：x=3, y=2 -> 3件red去B区, 2件blue去A区`
  - `映射规则 3 生效：x=3, y=2 -> 3件blue去B区, 2件red去C区`
- 每轮：`WATCH=1 WATCH_KILL=0 COLLISION_PROBE=1 bash tools/run_round.sh <rule> runs/2026-11-25_1440_m<rule>`；外部 watcher 只观察（不停轮），另跑 collision probe 记录 costmap/scan/plan/cmd_vel。
- 三轮 **全部 `State -> COMPLETED`，5/5，watcher 0 stall**。

## 2. 结果

| 轮 | 目录 | 映射 | 交付 | 墙钟 | 分段累加 | nav | creep | vision | arm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| m1 | `runs/2026-11-25_1440_m1` | 3 red→A / 2 blue→C | 5/5 | 274.5 s | 298.9 | 190.4 | 31.9 | 28.7 | 47.6 |
| m2 | `runs/2026-11-25_1440_m2` | 3 red→B / 2 blue→A | 5/5 | 293.2 s | 315.4 | 197.3 | 30.2 | 37.5 | 50.0 |
| m3 | `runs/2026-11-25_1440_m3` | 3 blue→B / 2 red→C | 5/5 | 247.1 s | 263.9 | 161.0 | 25.2 | 29.7 | 47.5 |

口径见 `THREE_ROUNDS_2026-11-25_final.md` 顶部：墙钟 = `PARSING_TASK`→`COMPLETED`；分段累加 = 5 块 `TIMING` 相加（段间有 2–6 s/块 重叠，故大于墙钟）。

## 3. 逐物块（秒）

| 轮 | 物块 | 仓 | 合计 | nav | creep | vision | arm |
|---|---|---|---:|---:|---:|---:|---:|
| m1 | red_cube_2 | A | 48.6 | 25.8 | 7.8 | 5.5 | 9.5 |
| m1 | red_cube_5 | A | 62.8 | 40.0 | 7.7 | 5.6 | 9.5 |
| m1 | red_cube_4 | A | 67.4 | 47.1 | 4.2 | 6.4 | 9.5 |
| m1 | blue_cube_3 | C | 61.7 | 40.0 | 6.6 | 5.6 | 9.5 |
| m1 | blue_cube_5 | C | 58.4 | 37.6 | 5.6 | 5.6 | 9.5 |
| m2 | red_cube_1 | B | 49.6 | 26.0 | 6.8 | 7.3 | 9.5 |
| m2 | red_cube_3 | B | 59.8 | 31.7 | 7.3 | 11.2 | 9.5 |
| m2 | red_cube_5 | B | 63.7 | 44.2 | 4.0 | 5.9 | 9.5 |
| m2 | blue_cube_3 (第 1 次，失败) | A | 25.0 | 22.5 | 0.0 | 0.0 | 2.5 |
| m2 | blue_cube_3 (第 2 次) | A | 47.8 | 26.9 | 5.2 | 6.3 | 9.5 |
| m2 | blue_cube_4 | A | 69.5 | 46.1 | 6.9 | 6.9 | 9.5 |
| m3 | blue_cube_1 | B | 46.2 | 26.2 | 4.4 | 5.9 | 9.5 |
| m3 | blue_cube_2 | B | 60.3 | 40.6 | 3.5 | 6.7 | 9.5 |
| m3 | blue_cube_5 | B | 64.9 | 45.3 | 4.1 | 5.9 | 9.5 |
| m3 | red_cube_1 | C | 46.6 | 23.6 | 7.7 | 5.7 | 9.5 |
| m3 | red_cube_5 | C | 45.9 | 25.4 | 5.5 | 5.5 | 9.5 |

每条腿里 nav 占 50–70%，creep（贴位蠕行）4–8 s，vision（视觉对齐）5.5–11.2 s，arm 恒定 9.5 s。

## 4. 车停着的时间

| 轮 | 停留段 | 合计 | 占墙钟 | 泊位作业 | 途中停等 |
|---|---:|---:|---:|---:|---|
| m1 | 10 | 88.2 s | 32% | 79.6 s | 8.6 s @(+13.17,-3.70) |
| m2 | 10 | 88.3 s | 30% | 84.6 s | 3.7 s @(-8.86,-9.64) |
| m3 | 11 | 90.0 s | 36% | 87.2 s | 2.8 s @(-2.36,0.00) |

- 泊位段：`Open gripper` / `Lower to place` 每仓 9.6–11.6 s；抓取 `Move to grasp` + `Close gripper` 6.6–8.1 s（含 2 次视觉对齐）。三轮 arm 恒 9.5 s/块，说明机械臂节拍没退化。
- 三轮中途停等合计 15.1 s（占墙钟 2%）：m1 是让行巡逻箱（yield_hold×2 + yield_stop×2）；m2/m3 是短促的蠕行/清障。
- `stale odometry` 三轮全 0。

## 5. 避障

| 轮 | collision ahead | planner 失败 | YIELD 行 | 最小 hull 间隙 | 最小 policy 间隙 | 接触 | 被推走 |
|---|---:|---:|---:|---:|---:|---|---|
| m1 | 18 | 0 | 68 | **+0.364 m**（mover_2 @233.0 s） | +0.193 m | 无 | 无 |
| m2 | 17 | 0 | 68 | **+0.214 m**（mover_2 @184.0 s） | +0.046 m | 无 | 无 |
| m3 | 9 | 1 | 51 | **+0.214 m**（mover_2 @148.0 s） | +0.169 m | 无 | 无 |

- 三轮**零接触、零被推**（`contact_analysis.py`，SAT 几何，hull 0.41×0.37 m，mover 半箱 0.11 m）。
- policy 层最小间隙 m2 只有 **+0.046 m**：车已停下让行，箱子按原轨迹擦过（keep-out 0.75 m 内箱子保持优先权），所以 collision monitor 会持续报警，但车身外壳始终没碰到。
- collision ahead 主要出现在仓库门口的窄通道和泊位蠕行段，都是拒动（把速度压到 0），不是撞上。

## 6. 唯一一次失败（m2 的 blue_cube_3 第 1 次，25.0 s）

时间线（`runs/2026-11-25_1440_m2/stack.log`）：

1. `Started blue_cube_3 -> A attempt 1` → `navigate to blue_cube_3 visual handoff` 正常推进；
2. 视觉交接点前 `yield arc`（让行绕行）→ `yielding around a moving obstacle via (0.80,-3.08)`；
3. 侧移到位后 `backup visual handoff` 段：`YIELD stop (swept-stop) box (+0.50,-0.35) v=(-0.11,+0.03) world=0.286`，`mover_2` 进到 0.68 m（keep-out 0.75 m）；
4. `YIELD stop (predict-hold)` 连续 5 次，wait 从 0.1 s 涨到 4.2 s；
5. `TIMING nav:blue_cube_3 backup visual handoff 5.201` → 立刻 `Pick-and-place stopped at step: navigate to pickup.` → `Object executor failed (1); retrying after safety stop.` → `attempt 2` 成功。

代码位置：`src/yzbot/mybot/scripts/fixed_joint_pick_place.py:3461`

```python
                if not self._hold_until_clear(label, x, y, deadline):
                    deadline_expired()
                    return False
```

`_hold_until_clear`（`:3012`）在 `REASON_WAIT_EXPIRED` 时 `break`，随后 `cleared = ... and not holds` 为 False → 这里直接 `return False`，**没有任何日志**，于是整腿失败。这是本轮唯一一次中断，代价 25.0 s（含重试），round 仍 5/5。

之前给另外两条静默出口（hold 后 `deadline_expired()`、`_resume_nav_goal() is None`）加的 WARN 这次没触发，正好说明这条路是第三条、也是最短的那条。

## 7. 结论

- **导航**：三轮全部 5/5，逐块 nav 23.6–47.1 s，仓库间路线差别（A/B/C 远近）直接体现在 nav 上；本轮没有出现此前那种"起点落在膨胀带里 → NavFn 连续拒图"的死锁，`failed to create plan` 合计 1 次且自愈。
- **避障**：动态障碍最近车身间隙 0.214 m、零接触；让行会停下来等箱子过去，不硬挤；`collision ahead` 的拒动都发生在窄通道/泊位，属于预期。
- **稳定性**：唯一问题是"让行等待预算用完"被当成"导航腿失败"（上面 §6），当前靠外层 attempt 重试兜住，代价 25 s；把这条路改成"可重试/重新规划"而不是直接判腿失败，是本轮剩下的唯一硬伤。

## 8. 复现

```bash
cd ~/intelligence_algorithm-main
source /opt/ros/humble/setup.bash && source install/setup.bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
WATCH=1 WATCH_KILL=0 COLLISION_PROBE=1 bash tools/run_round.sh 1 runs/2026-11-25_1440_m1
python3 tools/per_object_timings.py --dir runs/2026-11-25_1440_m1
python3 tools/contact_analysis.py runs/2026-11-25_1440_m1/events.jsonl
python3 tools/dwell_report.py --dir runs/2026-11-25_1440_m1
```

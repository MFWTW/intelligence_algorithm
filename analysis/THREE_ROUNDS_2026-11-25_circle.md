# 动态障碍：单一半径圈（2026-11-25 下午）

这份记录的是**换规则**之后的三组跑批：把原来那套"扫掠带 + 4 s 预测 + 车道判定 + 绕行弧"换成一个
圆心在车体上的单一半径圈。所有跑批都在这份代码上：`WATCH=1 WATCH_KILL=0 COLLISION_PROBE=1`。

## 1. 规则与参数（都已落库）

| 项 | 值 | 位置 |
|---|---|---|
| 圈半径 | **1.5 m** | `yield_simple_radius_m`（`obstacle_yield.YieldConfig.simple_radius_m`） |
| 只算车前方 | **是**（base 系 `x > 0`） | `yield_simple_front_only: true` |
| 圈内 | 停车等待（`ACTION_STOP` / `REASON_WAIT`） | `obstacle_yield.decide()` 顶部闸门 |
| 圈外 | 直行，忽略该箱子 | 同上 |
| 等待额度 | **15 s**（与外部停滞看门狗同一个数） | `yield_wait_budget_sec`（`config/profiles/executor_fast.yaml:29`，脚本默认同步改 15.0） |
| 额度用完 | **判这条腿失败**（空载会重试；载货则整轮 FAILED） | `_hold_until_clear` / `_navigate_once__timed` |
| 接触守卫（`contact_guard_m`） | **0.0，关闭** | `moving_obstacles.py:277` |

代码入口：`src/yzbot/mybot/scripts/obstacle_yield.py`（`decide()` 的 `simple_radius_m` 闸门、
`YieldConfig.simple_radius_front_only`）、`src/yzbot/mybot/scripts/fixed_joint_pick_place.py:857-861`
（参数声明）与 `:1231-1234`（装配）。

## 2. 三组跑批

| 组 | 规则 | 轮次 | 交付 | 墙钟 | 物块合计 | nav | 停车次数 | 最长 hold |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 16:40 | 整圆 1.5 m，额度 20 s | m1 | 5/5 | 281.3 | 305.3 | 194.8 | 43 | 8.9 |
| | | m2 | 5/5 | 368.4 | 390.7 | 279.5 | 107 | 19.2 |
| | | m3 | 5/5 | 353.2 | 369.2 | 265.9 | 118 | 15.0 |
| 17:00 | 半圆 1.5 m，额度 20 s | m1 | 5/5 | 312.2 | 334.8 | 225.5 | 43 | 13.7 |
| | | m2 | **1/5** | 100.9 | 101.9 | 71.4 | 32 | 4.0 |
| | | m3 | 5/5 | 303.5 | 314.9 | 216.2 | 62 | 13.1 |
| 17:20 | 半圆 1.5 m，额度 15 s | m1 | **4/5** | 301.7 | 315.8 | 214.2 | 52 | 9.9 |
| | | m2 | **1/5** | 89.2 | 91.6 | 59.8 | 20 | 4.2 |
| | | m3 | 5/5 | 353.6 | 366.9 | 266.3 | 73 | 11.0 |

**注意口径**：题面每 30 分钟换一次（seed 随之变），所以三组之间**不是 A/B 对照**——每组的 5 个物块、
仓库归属都不同。组内可比，组间只能看"有没有整轮报废"这个粗信号。

对照（同样的题面机制、但换规则之前，半圆前一轮）：`runs/2026-11-25_1440_m1/m2/m3` 是
274.5/293.2/247.1 s 墙钟，`runs/2026-11-25_1605_rule3_noguard` 254 s。

## 3. 为什么"只算车前方"

整圆那组里 43/107/118 次停车中有 **15/45/29 次是为了身后的箱子**停的；车后方不可能被撞到，
这类停车只有代价。改成半圆后，三组的"身后停车"计数都是 **0**。

## 4. 等待时长的分布（决定 15 s 额度会不会咬人）

成功放行的 hold 时长（秒，`corridor clear after N s`）：

- `1720_front15_m1`：0.5 0.6 0.7 0.7 0.8 0.9 1.1 2.2 3.0 3.6 3.9 3.9 4.6 6.8 7.5 9.9
- `1720_front15_m3`：0.5 0.6 0.6 0.7 0.7 0.7 0.8 1.0 1.5 1.9 2.1 2.1 2.2 2.3 2.6 3.3 4.0 5.4 5.6 5.8 7.0 9.6 11.0
- `1700_front_m3`：… 4.9 5.6 6.0 6.1 6.7 **13.1**

绝大多数 <3 s，尾巴 7–14 s。也就是说 15 s 额度**只咬尾巴**，但一旦咬在"载货腿"上就是整轮报废。

## 5. 三次整轮报废的现场

### 5.1 `1700_front_m2`（半圆，20 s 额度）
`warehouse A: the hold never confirmed a clear corridor (held_for=20.0s holds=True cleared=False reason='wait-expired')`
→ `Pick-and-place stopped at step: navigate to warehouse.` → `UNSAFE_OBJECT_STATE:NAV_TO_ZONE:...FAILED:CUBE_ATTACHED`
→ `FAILED SUMMARY: completed=1/5`。
现场：base (-3.02,-6.89)，箱子在其正前方 0.85→0.55 m 反复穿行（`YIELD stop (wait) … inside the 1.50 m circle`），
正处在 `mover_2` 巡逻线 `(-2.90,-5.13)→(3.19,-5.79)` 的端头附近。

### 5.2 `1720_front15_m2`（半圆，15 s 额度）
`warehouse A: the hold never confirmed a clear corridor (held_for=7.6s holds=True cleared=False reason='wait')`
——7.6 s 就退出了，说明这条腿的等待额度已经在前一次 hold 里花掉了一部分（额度是**按 episode**算的，
不是每次 hold 重新计数），剩下的 7.6 s 用完即判失败。随后 `TIMING nav:warehouse A 11.348`
→ `FAILED SUMMARY: completed=1/5`。

### 5.3 `1720_front15_m1`（半圆，15 s 额度）
`warehouse A: the leg gave up with no verified recovery; retreating 0.25 m on lidar clearance (1.29 m) before ...`
→ `Pick-and-place stopped at step: navigate to warehouse.` → `FAILED SUMMARY: completed=4/5`。
不是等待额度，是 Nav2 侧没有可行恢复路径（`no verified recovery`）后走了一次 `_wedged_retreat`
（新加的自救后退），但这次后退没能把腿救回来。

## 6. 顺带修的工具

`tools/watcher.py` 的 `_area_hold()` 原实现"锚点只在样本离锚点超过 radius 时才重置"，
一条直线上匀速开 7 m 也可能判成"在同一小块地方待了 15 s"（`runs/2026-11-25_1640_circle_m2/watcher/stall_01.md`
里 displacement=7.136 m 与 held=15.0 s 自相矛盾）。改成对样本做**包围盒**判断：只统计"最近一段
能塞进 2×radius 方框"的样本，移动会让盒子长大、跨度自然塌缩。离线回放 m2 的 timeline，同一位置仍判 21.9 s
（说明那次停车是真的），但不再出现"边开边判停滞"的自相矛盾。

## 7. 复现命令

```bash
cd ~/intelligence_algorithm-main
source /opt/ros/humble/setup.bash && source install/setup.bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
WATCH=1 WATCH_KILL=0 WATCH_KILL_AFTER=600 COLLISION_PROBE=1 \
  bash tools/run_round.sh 3 runs/2026-11-25_XXXX_circle_m3
python3 tools/per_object_timings.py --dir runs/2026-11-25_XXXX_circle_m3
python3 tools/dwell_report.py --dir runs/2026-11-25_XXXX_circle_m3
python3 tools/contact_analysis.py runs/2026-11-25_XXXX_circle_m3/events.jsonl
```

## 8. 仍然开着的口子

1. **等待额度按 episode 累计**：一条腿上第一次 hold 花掉的秒数会从后面的 hold 里扣，所以"15 s"在
   实际现场可能只剩几秒（`1720_front15_m2` 的 7.6 s 就是这样）。要不要改成"每次 hold 各 15 s"，
   还没定。
2. **额度用完就判腿失败**：这是当前的操作员决定（"保持 15 s 不变、直接判失败"），代价是载货腿
   一旦遇到长尾就整轮报废。
3. **圈内等待 = 原地不动**：箱子若真压过来，车不会让（`contact_guard_m=0` 已经不拦箱子），
   2026-11-25 上午 `1615_rule1_noguard` 就是这样被弹到 3.2 m/s 判整轮死的。

## 9. 90° 扇形（2026-11-25 17:48 之后）

操作员的判断：车停下可能不是"挡路"，而是**半圆边缘扫到了路过箱子的边**。把"半个圆"收窄成
"车头正前方 ±45° 的 90° 扇形"。

- 新参数 `yield_simple_fov_deg`（`obstacle_yield.YieldConfig`，节点 `fixed_joint_pick_place.py:827`
  声明 90.0、`:1228` 装配）。判定改为 `bearing = atan2(box_y, box_x)`，
  `abs(bearing) <= 45°` 才算"在路线上"；`0 < fov < 180` 之外的行为：`fov=180` 恢复整半圆，
  `simple_radius_front_only=False` 仍是字面整圆。
- 新增/改写的测试在 `src/yzbot/mybot/test/test_yield_simple_radius.py`（13 passed）：45° 边界算在圈内、
  60° 放行、90° 扇形可加宽、节点确实声明了 90.0。

### 9.1 证据：旧数据里一半的停车是"侧面擦过"

按日志里 `YIELD stop (wait) box (x,y)` 的方位角重新统计（正前方 ±45° 内算"真挡路"）：

| 轮次 | 停车总数 | 正前方 ±45° | 侧面 45–90° | 车后 |
|---|---:|---:|---:|---:|
| 1640_circle_m1 | 43 | 9 | 19 | 15 |
| 1640_circle_m2 | 107 | 25 | 37 | 45 |
| 1640_circle_m3 | 118 | 48 | 41 | 29 |
| 1700_front_m1 | 43 | 16 | 27 | 0 |
| 1700_front_m3 | 62 | 24 | 38 | 0 |
| 1720_front15_m1 | 52 | 19 | 33 | 0 |
| 1720_front15_m2 | 20 | 3 | 17 | 0 |
| 1720_front15_m3 | 73 | 46 | 27 | 0 |

半圆时代（1700/1720 系列）**40–85% 的停车是侧面 45–90°**，正是操作员描述的"箱子刚好蹭到半圆边缘"。

### 9.2 收窄后三轮

| 轮次 | 交付 | 墙钟 | nav | 扇形内停车 | 侧面停车 | hold 次数 / 最长 |
|---|---:|---:|---:|---:|---:|---|
| `runs/2026-11-25_1748_cone_m1` | **2/5** | 224 | 182 | 47 | 0 | 5 / 14.3 s |
| `..._1748_cone_m2` | 5/5 | 259 | 174 | 16 | 0 | 6 / 7.8 s |
| `..._1748_cone_m3` | 5/5 | 288 | 201 | 37 | 0 | 7 / 13.1 s |

侧面停车按定义归零（扇形外的箱子直接放行）。与 1720 系列相比停车数下降（m2 只有 16 次），
且车"带着箱子在 1.5 m 内继续行驶"的时长为 9.8–12.7 s（1720_m3 是 7.2 s），说明确实从箱子旁边开过去了。

### 9.3 代价

`tools/contact_analysis.py`：

- `cone_m1`：min hull **-0.011 m**（mover_4 @98.9 s），`contacts: 1`、`pushed while stopped: 1`。
- `cone_m2`：min hull +0.208 m，`contacts: none`。
- `cone_m3`：min hull +0.072 m，`contacts: none`，但 `pushed while stopped: 2`。

即扇形收窄后，**偏离车头 50–60° 的箱子仍可能擦到车角**（m1 有一次 1.1 cm 的重叠，没把仿真撞爆）。
如果后续出现真正的碰撞，优先把 `yield_simple_fov_deg` 收到 70–75°，而不是回到 180。

`cone_m1` 的整轮失败与箱子无关：`Navigation to blue_cube_4 backup visual handoff failed; status=6`
→ `planning algorithm GridBased failed to generate a valid path to (10.73, 4.00)` 连续 35 s
→ `_wedged_retreat` 因为激光净空 0.51 m ≤ 0.60 m 拒绝后退 → 腿失败 → `OBJECT_FAILED:executor exited 1`。

### 9.4 +/-50 度（fov 100）

操作员在 9.3 之后决定把扇形从 +/-45 放宽到 **+/-50**（`yield_simple_fov_deg: 100`，总张角）。
理由：`cone_m1` 那次 1.1 cm 的重叠发生在 50–60 度方位，+/-45 把它漏掉了。

验证一轮 `runs/2026-11-25_1811_cone50_m1`（rule 1，`WATCH=1 WATCH_KILL=0`）：

| 交付 | 墙钟 | nav | vision | arm | 停车 | 最大方位角 | hold | 最长 hold | 接触 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5/5 | 281 s | 192 s | 33 s | 48 s | 30 | 49 度 | 8 | 7.8 s | 无 |

`contact_analysis`：min hull **+0.023 m**（mover_4 @103.7 s），`contacts: none`、`pushed while stopped: none`。
停车方位角最大值 49 度，正好贴着新的扇形边界，说明规则按预期生效；但 2.3 cm 的车身间隙仍然很薄。

### 9.5 +/-50 度三题目整套

| 轮次 | 规则 | 交付 | 墙钟 | nav | vision | arm | nav 占比 | 停车 | 最大方位 | hold / 最长 | 接触 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `runs/2026-11-25_1811_cone50_m1` | 1 | 5/5 | 281 | 192 | 33 | 48 | 68% | 30 | 49 | 8 / 7.8 | 无（+0.023 m） |
| `runs/2026-11-25_1820_cone50_m2` | 2 | 5/5 | 265 | 181 | 29 | 48 | 68% | 21 | 50 | 10 / 3.2 | 无（+0.320 m） |
| `runs/2026-11-25_1826_cone50_m3` | 3 | 5/5 | 286 | 199 | 31 | 48 | 70% | 37 | 48 | 8 / 10.0 | 无（+0.118 m） |

稳定性计数：`planner_fail` 1 / 0 / 0，`collision_ahead` 12 / 0 / 8，`stale odometry` **0 / 0 / 0**，
watcher `0 stall` ×3，`contacts: none` ×3，`pushed while stopped: none` ×3。
三题都是 5/5，nav 占比 68–70%，arm 恒定 48 s。图：`analysis/run_cone50/fig1..fig4`。

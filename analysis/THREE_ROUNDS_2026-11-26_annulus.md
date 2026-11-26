# 三轮跑批（2026-11-26 13:45）：环带让行规则

## 这一版改了什么

| 项 | 值 | 位置 |
|---|---|---|
| 让行半径 | 1.8 m | `yield_simple_radius_m`，`scripts/fixed_joint_pick_place.py:966` |
| 前方扇形 | ±50°（fov 100°） | `yield_simple_fov_deg`，`:974` |
| 环带下界（commit） | **1.0 m 以内不再停车** | `yield_simple_commit_m`，`:975` |
| 等待预算 | 15 s 每次 hold | `yield_wait_budget_sec`，`config/profiles/executor_fast.yaml` |
| 接触守卫 | 关（`contact_guard_m: 0.0`） | `scripts/moving_obstacles.py` |
| 放置门控 / 视觉伺服 | L1 开、L2 开 | `place_visual_gate*` / `place_servo*` |

规则：箱子在 **1.0–1.8 m 的前方扇形**内 → 停车等；**<1.0 m** → 不等，按原路径继续走；圈外/身后 → 忽略。

## 结果总表

| 轮次 | 规则 | 终态 | 墙钟 | 5 块合计 | nav | creep | vision | arm | YIELD 停车 | commit 分支 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runs/2026-11-26_1345_ann_m1` | 1 | COMPLETED 5/5 | **258.4 s** | 277.7 | 170.8 (61%) | 28.6 | 30.4 | 47.6 | 15 | 0 |
| `runs/2026-11-26_1351_ann_m2` | 2 | COMPLETED 5/5 | **253.5 s** | 264.6 | 161.1 (61%) | 26.0 | 29.4 | 47.5 | 10 | 0 |
| `runs/2026-11-26_1356_ann_m3` | 3 | COMPLETED 5/5 | **271.8 s** | 285.2 | 179.7 (63%) | 22.8 | 34.7 | 47.6 | 11 | 0 |

`no verified recovery` 0/0/0，`failed to create plan` 3/0/6（都自行恢复），watcher 停顿时长 ≥15 s 的段数 0/0/1。

## 逐块耗时（秒：合计 / nav / creep / vision / arm）

### rule 1 — 1 红→C、4 蓝→A

| 物块 | 仓库 | 合计 | nav | creep | vision | arm |
|---|---|---:|---:|---:|---:|---:|
| red_cube_1 | C | 53.5 | 30.1 | 5.9 | 7.8 | 9.5 |
| red_cube_5 | C | 47.3 | 26.8 | 5.4 | 5.5 | 9.5 |
| blue_cube_3 | A | 43.8 | 24.1 | 4.5 | 5.6 | 9.5 |
| blue_cube_4 | A | 65.2 | 42.9 | 7.0 | 5.6 | 9.5 |
| blue_cube_5 | A | 68.0 | 46.8 | 5.8 | 5.8 | 9.5 |

### rule 2 — 3 红→A、2 蓝→B

| 物块 | 仓库 | 合计 | nav | creep | vision | arm |
|---|---|---:|---:|---:|---:|---:|
| blue_cube_1 | B | 45.0 | 25.1 | 4.2 | 6.0 | 9.5 |
| blue_cube_2 | B | 54.6 | 35.0 | 3.8 | 6.3 | 9.5 |
| red_cube_1 | A | 56.7 | 34.8 | 6.4 | 5.9 | 9.5 |
| red_cube_2 | A | 45.4 | 24.3 | 6.1 | 5.4 | 9.5 |
| red_cube_5 | A | 62.9 | 41.9 | 5.6 | 5.8 | 9.4 |

### rule 3 — 2 红→C、3 蓝→B

| 物块 | 仓库 | 合计 | nav | creep | vision | arm |
|---|---|---:|---:|---:|---:|---:|
| red_cube_1 | C | 69.7 | 44.3 | 5.4 | 10.5 | 9.5 |
| red_cube_5 | C | 57.4 | 36.2 | 5.7 | 5.9 | 9.5 |
| blue_cube_1 | B | 44.6 | 25.0 | 4.2 | 5.8 | 9.5 |
| blue_cube_2 | B | 51.8 | 32.2 | 3.7 | 6.2 | 9.5 |
| blue_cube_5 | B | 61.8 | 42.0 | 3.9 | 6.3 | 9.5 |

## 避障

| 轮次 | 最小车身间隙 | 接触 | 静止时被推 | 最近一次 |
|---|---:|---|---|---|
| m1 | +0.230 m | none | none | mover_2 @38.9 s |
| m2 | +0.146 m | none | none | mover_4 @87.6 s |
| m3 | +0.123 m | none | none | mover_4 @181.5 s |

三轮零接触、零推挤。commit 分支（<1 m 直行）三轮共 0 次触发——箱子没有真的贴到 1 m 以内，所以这条新规则**还没被真实事件验证过**，只有单测覆盖。

## 停留

- m1、m2 没有 ≥15 s 的停留。
- m3 有 1 段：`(-3.92,-6.31)`，`NAV_TO_ZONE`，15 s，`WAIT`（没发指令、栈停住），日志主因 `planner×24 / controller_refusal×20 / behaviour×11`，该腿目标是 C 泊位 (0.02,-2.70)，从 (-5.15,-5.77) 出发——穿过南侧窄走廊时规划器反复失败，最后自行绕出、腿完成。

## 与前一天/当天上午的对比

| 版本 | rule1 | rule2 | rule3 |
|---|---|---|---|
| 单一半圆 1.5 m / 20 s（11-25 16:40） | 5/5 281 s | 5/5 368 s | 5/5 353 s |
| ±50° 1.5 m / 15 s（11-25 18:11 起） | 5/5 281 s | 5/5 265 s | 5/5 286 s |
| 1.8 m / ±50°、守卫关（11-26 上午） | 5/5 224 s | **4/5**（11.3 m/s 被撞爆） | 5/5 236 s |
| **1.8 m / ±50° + commit 1.0 m（本轮）** | **5/5 258 s** | **5/5 254 s** | **5/5 272 s** |

YIELD 停车次数从 22–46（单圆）降到 10–15（环带），等待造成的空转明显减少。

## 图

- `analysis/run_annulus/fig1_per_object.png` — 逐块 nav/creep/vision/arm 堆叠
- `analysis/run_annulus/fig2_dwell.png` — 停留段与归因
- `analysis/run_annulus/fig3_movers.png` — 动态障碍中心距
- `analysis/run_annulus/fig4_paths.png` — 真值轨迹与停留点

## 复现

```bash
cd ~/intelligence_algorithm-main
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --symlink-install
for r in 1 2 3; do
  OUT="runs/$(date +%F_%H%M)_ann_m$r"
  WATCH=1 WATCH_KILL=0 WATCH_KILL_AFTER=600 COLLISION_PROBE=1 \
    bash tools/run_round.sh $r "$OUT"
  python3 tools/per_object_timings.py --dir "$OUT" > "$OUT/timings.txt"
  python3 tools/dwell_report.py --dir "$OUT" > "$OUT/dwells.txt"
done
MPLCONFIGDIR=$PWD/.mplconfig python3 tools/plot_round_analysis.py \
  --dir runs/<m1> --dir runs/<m2> --dir <m3> --out analysis/run_annulus
```

## 仍未闭环

1. commit 分支（<1 m 直行）三轮 0 次触发，只有单测覆盖；需要一次真实近距离事件才能确认它不会把车送进箱子里。
2. m3 那段 15 s 停留的窄走廊（静态净空 0.47 m 量级）仍是规划器的高风险区，本轮靠自行绕出，没有兜底。
3. `runs/2026-11-25_1427_r2` 里 `/cmd_vel` 末端通道 t=25.2 后停发（`cmd_vel_nav` 仍在发），仍未查清。

---

# 第二批（2026-11-26 14:07，同一份代码，新题面）

| 轮次 | 规则 | 终态 | 墙钟 | 5 块合计 | nav | creep | vision | arm | YIELD 停车 | 停留段/合计 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `runs/2026-11-26_1407_ann2_m1` | 1 | COMPLETED 5/5 | 280.1 s | 294.2 | 183.2 (62%) | 28.3 | 34.6 | 47.6 | 16 | 10 / 88.3 s |
| `runs/2026-11-26_1413_ann2_m2` | 2 | COMPLETED 5/5 | 292.6 s | 307.4 | 200.5 (65%) | 22.3 | 36.6 | 47.6 | 28 | 11 / 83.8 s |
| `runs/2026-11-26_1419_ann2_m3` | 3 | COMPLETED 5/5 | 285.2 s | 297.6 | 198.8 (67%) | 20.8 | 30.0 | 47.6 | 42 | 12 / 91.5 s |

`no verified recovery` 0/0/0，watcher 长停留 0/0/0，commit 分支 **0/0/0**。

## 逐块（合计 / nav / creep / vision / arm）

- **m1**：red_cube_2→A 48.5 (26.6/6.7/5.6)、red_cube_5→A 61.4 (39.3/6.5/6.0)、red_cube_4→A 59.6 (40.7/2.7/6.5)、red_cube_1→A 65.4 (38.5/6.6/10.7)、blue_cube_3→C 59.3 (38.2/5.7/5.8)
- **m2**：red_cube_1→B 49.0 (24.9/6.3/8.2)、red_cube_3→B 57.4 (30.5/6.1/11.3)、red_cube_5→B 66.4 (48.4/2.8/5.6)、red_cube_2→B 63.5 (45.2/3.4/5.3)、blue_cube_3→A 71.0 (51.5/3.8/6.1)
- **m3**：blue_cube_1→B 48.4 (28.2/4.4/6.1)、blue_cube_2→B 56.4 (37.4/2.8/6.6)、blue_cube_5→B 66.5 (47.2/4.2/5.6)、blue_cube_3→B 68.4 (49.2/4.0/5.6)、red_cube_1→C 57.9 (36.9/5.4/6.1)

## 避障（这一批出现了第一次真实擦碰）

| 轮次 | 最小车身间隙 | 接触 | 静止时被推 |
|---|---:|---|---|
| m1 | +0.206 m（mover_3 @153.8 s） | none | none |
| m2 | **−0.028 m**（mover_2 @251.3 s） | **1 段（0.7 s）** | **1 次：t=250.7 s 车静止、cmd=0，被推动 0.026 m** |
| m3 | +0.133 m（mover_2 @193.7 s） | none | none |

m2 那次：车按规则停着（`YIELD stop (wait) box (+1.23,-1.13) v=(-1.64,+0.45)`），箱子以约 1.6 m/s 从右前方擦过来，车身被挤进 2.8 cm、车被推了 2.6 cm，随后箱子走开、这一腿正常完成（没有 11.3 m/s 那种物理爆炸）。**结论：车停着不动时，飞驰的箱子仍可能蹭到车身——因为接触守卫已经关掉，箱子不会为我们让路。**

## 两批对比（同一份代码）

| | rule1 | rule2 | rule3 |
|---|---:|---:|---:|
| 第一批 13:45 | 258.4 s | 253.5 s | 271.8 s |
| 第二批 14:07 | 280.1 s | 292.6 s | 285.2 s |
| YIELD 停车（1/2 批） | 15 / 16 | 10 / 28 | 11 / 42 |

题面每 30 分钟换一次，两批不是同一套物块分布；两批都是 5/5，说明这一版在两种题面下都能完赛，但让行次数波动很大（10–42），时间也随题面波动 250–293 s。

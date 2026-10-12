# 已验证的比赛用参数快档（fast profile）

这三份文件是**整轮实测通过**的参数配置，放在仓库里（而不是 `run_logs/tools/`，
那个目录被 gitignore）。`competition_bringup.launch.py` 的三个参数文件默认就指向它们，
所以 `start_competition.sh` **不加任何参数**跑出来的就是快配置。

| 文件 | 对应 launch 参数 | 关键改动 |
| ---- | ---------------- | -------- |
| `nav_rpp15.yaml` | `nav_params_file` | 控制器 **MPPI → RPP**；`desired_linear_vel 1.5`；`velocity_smoother.max_velocity [1.5,0,2.0]`；开启 `use_collision_detection` 与 `use_cost_regulated_linear_velocity_scaling`；lookahead `0.3/0.6/0.9` |
| `executor_fast.yaml` | `executor_params_file` | 机械臂节拍加速（`arm_motion_sec 1.0→0.6`、`gripper_motion_sec 1.0→0.5`、`settle_sec 0.4→0.2`）；泊位旋转修正（`dock_creep_align_distance_m 0.12→0.25`）；`dock_creep_timeout_sec 14→26`；三条减速区 `slow_zones` |
| `state_machine_fast.yaml` | `state_machine_params_file` | `max_plausible_speed_mps 3.0`（配合 1.5 m/s 底盘） |

**实测成绩**（5 块全部完成、落点全部在区内）：整轮墙钟 239–355 s，中位约 320 s；
对照生产配置（MPPI @ 0.8/1.0 m/s）约 400–800 s。

要回到仓库原始配置（MPPI）：

```bash
ros2 run mybot start_competition.sh \
  nav_params_file:=$(ros2 pkg prefix bot_navigation)/share/bot_navigation/param/originbot_nav2_2.yaml \
  executor_params_file:=$(ros2 pkg prefix mybot)/share/mybot/config/fixed_joint_pick_place.yaml
```

> 注意：这三份是**部分覆盖**文件（只含需要改的键），Nav2 其余参数走默认值。
> 这是实测跑通几十轮的形式，**不要**把它们当成完整配置去和 `originbot_nav2_2.yaml` 逐键对比。

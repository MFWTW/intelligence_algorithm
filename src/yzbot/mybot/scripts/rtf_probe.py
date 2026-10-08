#!/usr/bin/env python3
"""Report the simulation real-time factor (RTF) and the effective base speed.

The competition budget is wall-clock: five minutes is five minutes no matter
how slow Gazebo runs. On this WSL host the full stack measured RTF 0.18-0.43,
i.e. the robot only experiences 18-43% of a second per wall second, and that -
not the Nav2 speed limits - is what makes a run long. Run this before a scored
attempt to know what the machine is currently delivering:

    ros2 run mybot rtf_probe.py            # 15 s sample
    ros2 run mybot rtf_probe.py --seconds 30

It prints:

* RTF from /clock (best effort QoS, as Gazebo publishes it);
* the robot's displacement per wall second and per simulated second, so a
  "the robot crawled" impression can be told apart from "the sim crawled";
* the cheap wins to check when the numbers are low (Gazebo GUI off, no leftover
  Nav2 stacks, camera/lidar rates).
"""

import argparse
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock


class RtfProbe(Node):
    def __init__(self) -> None:
        super().__init__('rtf_probe')
        self.clocks = []
        self.pose = None
        self.create_subscription(
            Clock, '/clock', self._clock_cb, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, '/odom', self._odom_cb, 50)

    def _clock_cb(self, msg: Clock) -> None:
        self.clocks.append(
            (time.monotonic(), msg.clock.sec + msg.clock.nanosec * 1e-9)
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15.0)
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = RtfProbe()

    # Warm up so the first /clock pair and an initial pose are available.
    end = time.monotonic() + 3.0
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)

    start_wall = time.monotonic()
    start_pose = node.pose
    start_clocks = len(node.clocks)
    end = start_wall + args.seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)

    wall = time.monotonic() - start_wall
    clocks = node.clocks[start_clocks:]
    if len(clocks) < 5:
        print('/clock 采样不足：Gazebo 没在跑，或 /clock 被过滤掉了。')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    sim = clocks[-1][1] - clocks[0][1]
    rtf = sim / wall if wall > 0 else 0.0
    print(f'RTF = {rtf:.3f}   （仿真 {sim:.2f}s / 墙钟 {wall:.2f}s，{len(clocks)} 条 /clock）')

    if start_pose is not None and node.pose is not None:
        moved = math.hypot(
            node.pose[0] - start_pose[0], node.pose[1] - start_pose[1]
        )
        if moved > 0.01:
            print(f'机器人在此期间走了 {moved:.2f} m')
            print(f'  墙钟速度 {moved / wall:.3f} m/s   ← 比赛计时看到的')
            print(f'  仿真速度 {moved / sim if sim > 0 else 0:.3f} m/s   ← Nav2 参数控制的')
        else:
            print('机器人没有移动（只测到仿真速率）。')

    if rtf < 0.6:
        print('\n实时率偏低，按收益排序检查：')
        print('  1. 是否带了 Gazebo GUI：gui:=false，可视化改用 Foxglove（见 FOXGLOVE.md）')
        print('  2. 是否有上一次 launch 残留的 Nav2 节点：pgrep -c -f nav2_ 应为 1 组')
        print('  3. 传感器频率：相机 10 Hz / 雷达 10 Hz 已够用（本仓库已调好）')
        print('  4. 跑之前 ros2 run mybot reset_dds_cache.sh')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Send exactly ONE arm trajectory goal and report what comes back.

Purpose: decide whether the duplicated ``Received new action goal`` lines in
the gzserver log (2026-11-23, rounds 11d/11f/11g/11h/11k) are caused by this
side sending twice, or by two action servers answering one client request.

The driver counts the ``Received new action goal`` lines in ``stack.log``
before and after this script runs: one send must produce exactly one line.  A
delta of 2 means the write was delivered to two servers, i.e. the duplication
lives in the gzserver/controller side, not in the pick-and-place client.
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]


class ArmGoalProbe(Node):
    def __init__(self, response_timeout: float, hold: float) -> None:
        super().__init__('arm_goal_probe')
        self._response_timeout = response_timeout
        self._hold = hold
        self._positions: list[float] = [0.0] * len(ARM_JOINTS)
        self._have_state = False
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)
        self._client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )

    def _on_state(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position))
        if all(name in by_name for name in ARM_JOINTS):
            self._positions = [float(by_name[name]) for name in ARM_JOINTS]
            self._have_state = True

    def run(self) -> int:
        started = time.monotonic()
        while not self._have_state and time.monotonic() - started < 10.0:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info(
            f'joint state: have_state={self._have_state} '
            f'positions={[round(p, 3) for p in self._positions]}'
        )

        server_ready = self._client.wait_for_server(timeout_sec=20.0)
        self.get_logger().info(f'server_ready={server_ready}')
        if not server_ready:
            return 3

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = list(self._positions)
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]

        sent_at = time.monotonic()
        self.get_logger().info('sending exactly one goal')
        future = self._client.send_goal_async(goal)
        deadline = sent_at + self._response_timeout
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            self.get_logger().error('no goal response; the request was sent once')
            return 4
        handle = future.result()
        self.get_logger().info(
            f'goal response accepted={handle.accepted} '
            f'after {time.monotonic() - sent_at:.3f}s'
        )

        # Any late duplicate would arrive inside this window; keep spinning so
        # the log stays complete for the driver-side count.
        end = time.monotonic() + self._hold
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(
            f'probe finished; sent=1 hold={self._hold:.1f}s'
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--response-timeout', type=float, default=5.0)
    parser.add_argument('--hold', type=float, default=3.0)
    args = parser.parse_args()
    rclpy.init()
    node = ArmGoalProbe(args.response_timeout, args.hold)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())

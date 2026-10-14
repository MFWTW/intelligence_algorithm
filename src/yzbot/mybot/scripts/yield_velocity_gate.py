#!/usr/bin/env python3
"""Dual-stage, fail-closed motion gate around Nav2's velocity smoother.

Nav2/manual commands pass through the input gate into the smoother. Its output
then passes through the *same hold latch* on its way to Gazebo. When a hold
starts both linear and angular commands at the actuator are immediately zero;
waiting for the smoother's deceleration would leave the base sliding for ~0.6 s
(measured in the first guarded simulation). A separate Gazebo plugin with a
0.5 s command deadman stops the wheels if this process dies while moving.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class VelocityGateState:
    """Pure decision layer; no ROS state, so rejection and latch are testable.

    The hold engages on the first True (a mover in the lane must stop the base
    immediately) but a release needs ``release_debounce_sec`` of continuous
    False. Two writers, or one stuttering writer, therefore cannot flap the
    latch at message rate: measured 2026-10-14 with two publishers, the flag
    toggled every 50 ms and each "released" window forwarded a command.
    """

    def __init__(self, stale_sec=0.3, release_debounce_sec=0.15):
        self.stale_sec = float(stale_sec)
        self.release_debounce_sec = float(release_debounce_sec)
        self.held = False
        self.pending_release_since = None
        self.last_input = None
        self.command = Twist()

    def hold(self, active, now):
        if active:
            self.held = True
            self.pending_release_since = None
            self.command = Twist()
            self.last_input = None
            return
        if not self.held:
            return
        if self.pending_release_since is None:
            self.pending_release_since = now
            return
        if now - self.pending_release_since >= self.release_debounce_sec:
            self.held = False
            self.pending_release_since = None

    def input(self, command, now):
        if not all(math.isfinite(value) for value in (
            command.linear.x, command.linear.y, command.linear.z,
            command.angular.x, command.angular.y, command.angular.z,
        )):
            return
        if not self.held:
            self.command = command
            self.last_input = now

    def output(self, now):
        if (self.held or self.last_input is None
                or now - self.last_input > self.stale_sec):
            return Twist()
        return self.command


class YieldVelocityGate(Node):
    def __init__(self):
        super().__init__('yield_velocity_gate')
        # Every existing Nav2, supervisor and manual-servo writer uses the raw
        # cmd_vel_nav topic. Only this gate may write the smoother's redirected
        # cmd_vel_nav_guarded input (validated against the live ROS graph).
        self.declare_parameter('input_topic', '/cmd_vel_nav')
        self.declare_parameter('output_topic', '/cmd_vel_nav_guarded')
        self.declare_parameter('stale_sec', 0.3)
        self.declare_parameter('release_debounce_sec', 0.15)
        stale = self.get_parameter('stale_sec').value
        self.state = VelocityGateState(
            stale, self.get_parameter('release_debounce_sec').value)
        self.drive_state = VelocityGateState(
            stale, self.get_parameter('release_debounce_sec').value)
        self.output_pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.drive_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(Bool, '/competition/yield_gate_held', 10)
        self.create_subscription(
            Twist, self.get_parameter('input_topic').value,
            lambda command: self.state.input(command, time.monotonic()), 10)
        self.create_subscription(
            Twist, '/cmd_vel_smoothed_raw',
            lambda command: self.drive_state.input(command, time.monotonic()), 10)
        self.create_subscription(Bool, '/competition/yield_hold', self._hold, 10)
        self.create_timer(0.05, self._tick)

    def _hold(self, state):
        now = time.monotonic()
        was_held = self.state.held
        self.state.hold(state.data, now)
        self.drive_state.hold(state.data, now)
        if self.state.held != was_held:
            # The gate's own transition log is the authority on hold windows: a
            # subscriber's samples can arrive late and look like a command leak.
            self.get_logger().info(
                'yield hold ' + ('engaged' if self.state.held else 'released')
                + f' at {now:.3f}')
        if self.state.held:
            # Do not wait for the next timer tick when a mover enters the lane.
            self.output_pub.publish(Twist())
            self.drive_pub.publish(Twist())

    def _tick(self):
        now = time.monotonic()
        self.output_pub.publish(self.state.output(now))
        self.drive_pub.publish(self.drive_state.output(now))
        status = Bool()
        status.data = self.state.held
        self.status_pub.publish(status)


def main():
    rclpy.init()
    node = YieldVelocityGate()
    try:
        rclpy.spin(node)
    finally:
        node.output_pub.publish(Twist())
        node.drive_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

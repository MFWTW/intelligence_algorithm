#!/usr/bin/env python3
"""Record one competition round to JSONL for offline timing/route analysis.

Subscribes to the pose/plan/velocity/competition-status streams of a running
competition stack and appends one JSON object per message to ``--out``::

    {"t": 12.345, "w": 1763000000.123, "topic": "odom", "d": {...}}

``t`` is seconds since the recorder started (monotonic wall clock), ``w`` the
absolute wall-clock time, so a later analysis can align the launch log (which
carries HH:MM:SS) with the samples.

    python3 tools/round_recorder.py --out /tmp/events.jsonl

Stops on its own ``--after-terminal`` seconds after /competition/state reaches a
terminal value, or after ``--max-duration`` seconds, whichever comes first.
"""
import argparse
import json
import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from gazebo_msgs.msg import LinkStates, ModelStates
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Int32, String

TERMINAL_STATES = {'COMPLETED', 'FAILED', 'ABORTED'}


def make_qos(depth=10, reliable=True, transient=False):
    qos = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST)
    qos.reliability = (
        ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
    )
    qos.durability = (
        DurabilityPolicy.TRANSIENT_LOCAL if transient else DurabilityPolicy.VOLATILE
    )
    return qos


def yaw_of(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RoundRecorder(Node):
    def __init__(self, args):
        super().__init__('round_recorder')
        self._args = args
        self._file = open(args.out, 'w', encoding='utf-8', buffering=1)
        self._t0 = time.time()
        self._lines = 0
        self._terminal_at = None
        self._stopped = False
        self._stop_requested = False
        self._mover_throttle = max(0.0, 1.0 / max(0.1, args.model_rate))
        self._last_model_emit = -1e9
        self._last_link_emit = -1e9

        # Streams used for timing + trajectory reconstruction.
        self.create_subscription(
            Odometry, '/odom', self._on_odom, make_qos(50, reliable=False))
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, make_qos(50, reliable=False))
        self.create_subscription(
            Twist, '/cmd_vel_nav', self._on_cmd_nav, make_qos(50, reliable=False))
        self.create_subscription(
            Path, '/plan', self._on_plan, make_qos(1, reliable=True))
        self.create_subscription(
            Path, '/local_plan', self._on_local_plan, make_qos(1, reliable=True))
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl,
            make_qos(1, reliable=True))
        self.create_subscription(
            LinkStates, '/gazebo/link_states', self._on_link_states, 10)
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self._on_model_states,
            make_qos(1, reliable=False))

        # Competition control/status streams (latched).
        latched = make_qos(1, reliable=True, transient=True)
        for topic in (
            '/competition/executor_state',
            '/competition/executor_goal',
            '/competition/executor_result',
            '/competition/executor_error',
            '/competition/progress',
            '/competition/current_object',
            '/competition/error',
            '/competition/task_problem',
            '/competition/task_rules',
            '/competition/task_seed',
            '/competition/task_status',
        ):
            self.create_subscription(
                String, topic,
                lambda msg, name=topic: self._emit(name.rsplit('/', 1)[-1],
                                                   {'data': msg.data}),
                latched)
        self.create_subscription(
            Int32, '/competition/retry_count',
            lambda msg: self._emit('retry_count', {'data': msg.data}), latched)
        self.create_subscription(
            String, '/competition/vision_alignment',
            lambda msg: self._emit('vision_alignment', {'data': msg.data}),
            make_qos(10))
        for topic in ('/competition/yield_hold', '/competition/yield_gate_held'):
            self.create_subscription(
                Bool, topic,
                lambda msg, name=topic: self._emit(
                    name.rsplit('/', 1)[-1], {'data': bool(msg.data)}),
                make_qos(10))

        self.create_timer(2.0, self._check_stop)
        self.create_timer(60.0, self._heartbeat)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        self.get_logger().info(f'recording to {args.out}')

    # ------------------------------------------------------------------ emit
    def _emit(self, topic, payload):
        if self._stopped:
            return
        now = time.time()
        self._file.write(json.dumps(
            {'t': round(now - self._t0, 3), 'w': round(now, 3),
             'topic': topic, 'd': payload},
            separators=(',', ':')) + '\n')
        self._lines += 1

    def _on_odom(self, msg):
        pose = msg.pose.pose
        self._emit('odom', {
            'x': round(pose.position.x, 4),
            'y': round(pose.position.y, 4),
            'yaw': round(yaw_of(pose.orientation), 4),
            'vx': round(msg.twist.twist.linear.x, 4),
            'wz': round(msg.twist.twist.angular.z, 4),
        })

    def _on_cmd_vel(self, msg):
        self._emit('cmd_vel', {'vx': round(msg.linear.x, 4),
                               'wz': round(msg.angular.z, 4)})

    def _on_cmd_nav(self, msg):
        self._emit('cmd_vel_nav', {'vx': round(msg.linear.x, 4),
                                   'wz': round(msg.angular.z, 4)})

    def _on_plan(self, msg):
        self._emit('plan', self._path_payload(msg))

    def _on_local_plan(self, msg):
        self._emit('local_plan', self._path_payload(msg))

    @staticmethod
    def _path_payload(msg):
        pts = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)]
               for p in msg.poses]
        return {'frame': msg.header.frame_id, 'stamp': round(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, 3),
            'n': len(pts), 'pts': pts}

    def _on_amcl(self, msg):
        pose = msg.pose.pose
        self._emit('amcl_pose', {
            'x': round(pose.position.x, 4),
            'y': round(pose.position.y, 4),
            'yaw': round(yaw_of(pose.orientation), 4),
        })

    def _on_link_states(self, msg):
        """Ground-truth world pose of the robot base (proves world == map)."""
        now = time.monotonic()
        if now - self._last_link_emit < self._mover_throttle:
            return
        self._last_link_emit = now
        for name, pose in zip(msg.name, msg.pose):
            if name.endswith('base_link') or name.endswith('base_footprint'):
                self._emit('link_states', {
                    'name': name,
                    'x': round(pose.position.x, 4),
                    'y': round(pose.position.y, 4),
                    'yaw': round(yaw_of(pose.orientation), 4)})
                return

    def _on_model_states(self, msg):
        # Entites we care about: the robot, the four moving obstacles, cubes.
        now = time.monotonic()
        if now - self._last_model_emit < self._mover_throttle:
            return
        self._last_model_emit = now
        if not getattr(self, '_saw_model_names', False):
            self._saw_model_names = True
            self._emit('model_all', {
                n: [round(p.position.x, 3), round(p.position.y, 3)]
                for n, p in zip(msg.name, msg.pose)})
        wanted = {}
        for name, pose in zip(msg.name, msg.pose):
            if (name == 'robot' or name.startswith('mover')
                    or 'cube' in name):
                wanted[name] = [round(pose.position.x, 3),
                                round(pose.position.y, 3),
                                round(yaw_of(pose.orientation), 3)]
        self._emit('model_states', wanted)

    # ------------------------------------------------------------------ stop
    def _check_stop(self):
        if self._stopped:
            return
        if self._terminal_at is None:
            state = self._last_state if hasattr(self, '_last_state') else None
            if state in TERMINAL_STATES:
                self._terminal_at = time.time()
        elapsed = time.time() - self._t0
        if (self._terminal_at is not None
                and time.time() - self._terminal_at >= self._args.after_terminal):
            self._finish('terminal state reached')
        elif elapsed >= self._args.max_duration:
            self._finish('max duration reached')

    def _heartbeat(self):
        if not self._stopped:
            self.get_logger().info(
                f't={time.time() - self._t0:.0f}s lines={self._lines} state='
                f'{getattr(self, "_last_state", "?")}')

    def _on_signal(self, signum, frame):
        # Python runs this between bytecodes in the main thread; the main loop
        # notices the flag and shuts down from there (calling shutdown from a
        # signal handler deadlocks inside rclpy).
        self._stop_requested = True

    def _finish(self, reason):
        if self._stopped:
            return
        self._stopped = True
        self._terminal_at = self._terminal_at or time.time()
        self._file.flush()
        self._file.close()
        self.get_logger().info(
            f'stopped after {time.time() - self._t0:.1f}s, {self._lines} '
            f'lines ({reason})')

    @property
    def stop_requested(self):
        return self._stopped or self._stop_requested


class TerminalWatcher(RoundRecorder):
    """Track the competition state so the recorder can stop by itself."""

    def __init__(self, args):
        super().__init__(args)
        self.create_subscription(
            String, '/competition/state', self._on_state,
            make_qos(1, reliable=True, transient=True))

    def _on_state(self, msg):
        self._last_state = msg.data
        self._emit('state', {'data': msg.data})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-duration', type=float, default=1500.0)
    parser.add_argument('--after-terminal', type=float, default=8.0)
    parser.add_argument('--model-rate', type=float, default=5.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = TerminalWatcher(args)
    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node._finish('main loop exit')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

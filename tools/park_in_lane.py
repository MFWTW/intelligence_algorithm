#!/usr/bin/env python3
"""Park the base in a mover's lane and let the patrol box drive at it.

The contact guard in moving_obstacles is only a safety net until a box is
actually driven into the base; the three contactfix rounds never got closer
than 0.51 m, so the guard was never exercised. This experiment forces the
case: the base is teleported a short distance ahead of a patrol box (no
start press, so nothing commands the wheels) and the box walks into it.

Phase A keeps contact_guard_m at its configured value and the box should stop
about one guard radius short of the base. Phase B (--guard-off-phase) sets
contact_guard_m to zero at runtime and re-parks the base in the lane, which
reproduces the historical failure: the kinematic box is driven through the
dynamic base and Gazebo's contact solver runs away.

Evidence goes to --out as JSONL: every sample carries the base pose, the box
pose, the closest /movers/obstacles point and the base speed from
/gazebo/model_states (odometry is not trusted after a teleport).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

PARK_POSE_TOL = 0.05


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ParkInLane(Node):
    def __init__(self, args):
        super().__init__('park_in_lane')
        self.args = args
        self.models: dict[str, tuple[float, float, float, float]] = {}
        self.cloud_min = math.inf
        self.base_speed = 0.0
        self.base_xy = None
        self.odom_speed = 0.0
        self.out = open(args.out, 'w', encoding='utf-8')
        self.set_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(ModelStates, '/gazebo/model_states', self._models_cb, 20)
        self.create_subscription(PointCloud2, '/movers/obstacles', self._cloud_cb, 10)
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._mover_track: deque = deque(maxlen=40)
        self.samples: list[dict] = []
        self.worst = {'speed': 0.0, 'cloud_min': math.inf}

    # ---------------------------------------------------------------- inputs
    def _models_cb(self, msg: ModelStates):
        for name, pose in zip(msg.name, msg.pose):
            self.models[name] = (pose.position.x, pose.position.y,
                                 yaw_of(pose.orientation),
                                 pose.position.z)
        if self.args.robot in self.models:
            x, y, yaw, _z = self.models[self.args.robot]
            if self.base_xy is not None:
                dt = max(1e-3, time.monotonic() - self._last_step)
                step = math.hypot(x - self.base_xy[0], y - self.base_xy[1])
                self.base_speed = max(self.base_speed * 0.6, step / dt)
                self.worst['speed'] = max(self.worst['speed'], self.base_speed)
            self.base_xy = (x, y, yaw)
            self._last_step = time.monotonic()
        if self.args.mover in self.models:
            self._mover_track.append(
                (time.monotonic(), *self.models[self.args.mover][:2]))

    def _cloud_cb(self, msg: PointCloud2):
        if self.base_xy is None or msg.width == 0:
            return
        bx, by = self.base_xy[0], self.base_xy[1]
        step = msg.point_step
        field_x = next((f.offset for f in msg.fields if f.name == 'x'), 0)
        field_y = next((f.offset for f in msg.fields if f.name == 'y'), 4)
        raw = msg.data
        best = math.inf
        for i in range(msg.width * msg.height):
            off = i * step
            if off + 8 > len(raw):
                break
            x = struct.unpack_from('<f', raw, off + field_x)[0]
            y = struct.unpack_from('<f', raw, off + field_y)[0]
            d = math.hypot(x - bx, y - by)
            if d < best:
                best = d
        self.cloud_min = best
        self.worst['cloud_min'] = min(self.worst['cloud_min'], best)

    def _odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.odom_speed = math.hypot(v.x, v.y)

    # ---------------------------------------------------------------- helpers
    def wait_for(self, predicate, timeout_s: float, label: str) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        self.get_logger().error(f'timed out waiting for {label}')
        return False

    def mover_heading(self) -> tuple[float, float]:
        if len(self._mover_track) >= 2:
            (t0, x0, y0), (t1, x1, y1) = self._mover_track[0], self._mover_track[-1]
            if t1 - t0 > 0.4:
                dx, dy = x1 - x0, y1 - y0
                norm = math.hypot(dx, dy)
                if norm > 1e-3:
                    return dx / norm, dy / norm
        return 1.0, 0.0

    def park_ahead(self, ahead: float) -> tuple[float, float, float] | None:
        if self.args.mover not in self.models:
            return None
        mx, my = self.models[self.args.mover][:2]
        hx, hy = self.mover_heading()
        x, y = mx + hx * ahead, my + hy * ahead
        yaw = math.atan2(hy, hx)
        req = SetEntityState.Request()
        req.state.name = self.args.robot
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = self.models.get(
            self.args.robot, (0.0, 0.0, 0.0, 0.0))[3]
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = 'world'
        req.set_pose = True
        future = self.set_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.base_xy = (x, y, yaw)
        return x, y, yaw

    def set_param(self, name: str, value: str) -> bool:
        try:
            proc = subprocess.run(
                ['ros2', 'param', 'set', self.args.guard_node, name, value],
                capture_output=True, text=True, timeout=20)
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            self.get_logger().error(f'param set failed: {exc}')
            return False
        ok = 'Successful' in (proc.stdout or '')
        self.get_logger().info(
            f'ros2 param set {self.args.guard_node} {name} {value}: '
            f'{(proc.stdout or proc.stderr).strip()[:120]}')
        return ok

    def record(self, phase: str, note: str = ''):
        if self.base_xy is None:
            return
        bx, by, byaw = self.base_xy
        mover = self.models.get(self.args.mover, (math.nan, math.nan, math.nan,
                                                  math.nan))
        center = math.hypot(mover[0] - bx, mover[1] - by)
        row = {
            't': round(time.time(), 3),
            'phase': phase,
            'base': [round(bx, 3), round(by, 3), round(byaw, 3)],
            'mover': [round(mover[0], 3), round(mover[1], 3)],
            'center_d': round(center, 3) if math.isfinite(center) else None,
            'surface_d': (round(center - 0.157, 3)
                          if math.isfinite(center) else None),
            'cloud_min': (round(self.cloud_min, 3)
                          if math.isfinite(self.cloud_min) else None),
            'base_speed': round(self.base_speed, 3),
            'odom_speed': round(self.odom_speed, 3),
            'note': note,
        }
        self.out.write(json.dumps(row) + '\n')
        self.out.flush()
        self.samples.append(row)

    def hold(self, phase: str, seconds: float, stop_when_touching=True):
        """Keep the wheels at zero and sample; stop early on a blow-up."""
        zero = Twist()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_pub.publish(zero)
            self.record(phase)
            if self.args.mover in self.models:
                self._mover_track.append(
                    (time.monotonic(), *self.models[self.args.mover][:2]))
            bx, by = self.base_xy[0], self.base_xy[1]
            left_map = bx < -15.0 or bx > 17.0 or by < -16.0 or by > 6.2
            blow_up = self.base_speed > 1.5 or left_map
            if blow_up:
                self.record(phase, note='BLOW-UP')
                self.get_logger().error(
                    f'{phase}: base at ({bx:.2f},{by:.2f}) with speed '
                    f'{self.base_speed:.2f} m/s - the contact solver blew up')
                return 'blow-up'
            if stop_when_touching and self.cloud_min < 0.20:
                self.record(phase, note='box inside the footprint')
                return 'touching'
            time.sleep(0.05)
        return 'timeout'

    def summary(self) -> dict:
        phase_rows = {}
        for row in self.samples:
            phase_rows.setdefault(row['phase'], []).append(row)
        out = {}
        for phase, rows in phase_rows.items():
            cloud = [r['cloud_min'] for r in rows if r['cloud_min'] is not None]
            surf = [r['surface_d'] for r in rows if r['surface_d'] is not None]
            out[phase] = {
                'samples': len(rows),
                'min_cloud': min(cloud) if cloud else None,
                'min_surface': min(surf) if surf else None,
                'max_base_speed': max(r['base_speed'] for r in rows),
                'min_center': min(r['center_d'] for r in rows
                                  if r['center_d'] is not None) if rows else None,
            }
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--robot', default='six_arm')
    parser.add_argument('--mover', default='mover_2')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--ahead', type=float, default=0.9,
                        help='park this far in front of the box along its lane')
    parser.add_argument('--phase-a', type=float, default=45.0)
    parser.add_argument('--phase-b', type=float, default=25.0)
    parser.add_argument('--guard-node', default='/moving_obstacles')
    parser.add_argument('--guard-param', default='contact_guard_m')
    parser.add_argument('--guard-value', default='0.42')
    parser.add_argument('--no-park', action='store_true',
                        help='leave the base where it spawned (its odom is '
                             'consistent, so the guard sees it)')
    parser.add_argument('--guard-off-phase', action='store_true',
                        help='phase B sets the guard to 0 and reproduces the old bug')
    args = parser.parse_args()

    rclpy.init()
    node = ParkInLane(args)
    rc = 0
    try:
        ready = node.wait_for(
            lambda: args.robot in node.models and args.mover in node.models
            and node.set_client.service_is_ready(), 120.0, 'model states')
        if not ready:
            rc = 2
        else:
            node.get_logger().info(
                f'robot {node.models[args.robot][:2]}, '
                f'mover {node.models[args.mover][:2]}')
            node.set_param(args.guard_param, args.guard_value)
            pose = None if args.no_park else node.park_ahead(args.ahead)
            node.get_logger().info(f'phase A: parked at {pose}, guard on')
            outcome = node.hold('A', args.phase_a)
            node.get_logger().info(f'phase A outcome: {outcome}')
            if args.guard_off_phase and rclpy.ok():
                node.set_param(args.guard_param, '0.0')
                node._mover_track.clear()
                pose = None if args.no_park else node.park_ahead(args.ahead)
                node.get_logger().info(f'phase B: parked at {pose}, guard off')
                outcome = node.hold('B', args.phase_b)
                node.get_logger().info(f'phase B outcome: {outcome}')
                node.set_param(args.guard_param, args.guard_value)
            summary = node.summary()
            print('PARK_SUMMARY ' + json.dumps(summary, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - never leave the runner hanging
        node.get_logger().error(f'park_in_lane failed: {exc}')
        rc = 3
    finally:
        node.out.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

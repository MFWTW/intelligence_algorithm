#!/usr/bin/env python3
"""Record the local costmap, RPP collision arc and raw scan during a round.

Why this exists
---------------
``round_recorder.py`` logs the three velocity streams but not the two things
that decide whether a "detected collision ahead" episode is real geometry or a
costmap artefact: the exact cells the controller's arc swept, and the laser
returns that could have produced them. This probe records both, plus the
footprint and the patrol-box poses, so an episode can be reconstructed offline:

    python3 tools/collision_probe.py --out <out_dir>/probe.jsonl

One JSON object per message:

    {"t": <s since probe start>, "w": <wall clock>, "topic": "...", "d": {...}}

Heavy payloads (costmap, scan) are zlib-compressed then base64-encoded; the
decoder lives in ``tools/analyze_collision_episodes.py``. Everything is
best-effort: a missing topic is reported once and the probe keeps recording.
"""
import argparse
import base64
import json
import math
import signal
import struct
import time
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PolygonStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String

try:
    from nav2_msgs.msg import CollisionMonitorState
except ImportError:  # pragma: no cover - only if nav2_msgs is not sourced
    CollisionMonitorState = None

TERMINAL_STATES = {'COMPLETED', 'FAILED', 'ABORTED'}


def make_qos(depth=10, reliable=False, transient=False):
    qos = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST)
    qos.reliability = (
        ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
    )
    qos.durability = (
        DurabilityPolicy.TRANSIENT_LOCAL if transient else DurabilityPolicy.VOLATILE
    )
    return qos


def yaw_of(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def stamp_of(msg):
    s = msg.header.stamp
    return round(s.sec + s.nanosec * 1e-9, 4)


def pack_floats(values):
    return base64.b64encode(zlib.compress(
        struct.pack('<%df' % len(values), *values), 6)).decode('ascii')


class CollisionProbe(Node):
    def __init__(self, args):
        super().__init__('collision_probe')
        self._args = args
        self._out = open(args.out, 'w', encoding='utf-8', buffering=1)
        self._t0 = time.time()
        self._lines = 0
        self._stopped = False
        self._stop_requested = False
        self._terminal_at = None
        self._last_state = None
        self._seen = {}

        self._last_scan = -1e9
        self._last_movers = -1e9
        self._last_voxel = -1e9
        self._last_costmap = -1e9
        self._last_global_costmap = -1e9
        self._last_footprint = -1e9
        self._last_plan = -1e9
        self._last_local_plan = -1e9

        # Velocity/pose chain: full rate, tiny messages.
        self.create_subscription(
            Odometry, '/odom', self._on_odom, make_qos(50))
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, make_qos(50))
        self.create_subscription(
            Twist, '/cmd_vel_nav', self._on_cmd_nav, make_qos(50))

        # RPP publishes the exact arc it tested for collision as a Path.
        self.create_subscription(
            Path, '/lookahead_collision_arc', self._on_arc, make_qos(20))
        if CollisionMonitorState is not None:
            self.create_subscription(
                CollisionMonitorState, '/collision_monitor_state',
                self._on_monitor, make_qos(20))
        else:
            self.get_logger().warning('nav2_msgs not available: no monitor state')

        # Costmaps: the local one publishes full costmaps at 2 Hz.
        self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap', self._on_costmap,
            make_qos(2, reliable=True))
        # The global grid is what the planner actually starts from, so a leg
        # that dies with "failed to create plan" is only explained by this one:
        # the local grid is built from the laser and can call a pose clear while
        # the static layer still has the base inside a lethal band.
        self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self._on_global_costmap,
            make_qos(1, reliable=True))
        self.create_subscription(
            PolygonStamped, '/local_costmap/published_footprint',
            self._on_footprint, make_qos(2, reliable=True))

        # Raw laser: what the relay/costmap was actually offered.
        self.create_subscription(
            LaserScan, '/scan', self._on_scan, make_qos(10))

        # The patrol boxes are stamped into the voxel layer *as a sensor* in
        # base_link; recording the cloud is the only way to tell a mark the
        # boxes still justify from one they have already left behind.
        self.create_subscription(
            PointCloud2, '/movers/obstacles', self._on_movers, make_qos(10))
        # Occupied voxels with their z: tells a mark left by the box cloud at
        # z=0.3 from a mark the laser made at z=0.1, which decides whether the
        # low scan ray can ever clear it.
        self.create_subscription(
            PointCloud2, '/local_costmap/voxel_grid', self._on_voxel, make_qos(2))

        # Planning output for context.
        self.create_subscription(
            Path, '/plan', self._on_plan, make_qos(2, reliable=True))
        self.create_subscription(
            Path, '/local_plan', self._on_local_plan, make_qos(2, reliable=True))

        self.create_subscription(
            String, '/competition/state', self._on_state,
            make_qos(1, reliable=True, transient=True))

        self.create_timer(1.0, self._check_stop)
        self.create_timer(15.0, self._report)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        self.get_logger().info(f'collision probe -> {args.out}')

    # ---------------------------------------------------------------- emitting
    def _emit(self, topic, payload):
        if self._stopped:
            return
        now = time.time()
        self._out.write(json.dumps(
            {'t': round(now - self._t0, 3), 'w': round(now, 3),
             'topic': topic, 'd': payload},
            separators=(',', ':')) + '\n')
        self._lines += 1
        self._seen[topic] = self._seen.get(topic, 0) + 1

    # ------------------------------------------------------------------ streams
    def _on_odom(self, msg):
        p = msg.pose.pose
        self._emit('odom', {
            'x': round(p.position.x, 4), 'y': round(p.position.y, 4),
            'yaw': round(yaw_of(p.orientation), 4),
            'vx': round(msg.twist.twist.linear.x, 4),
            'wz': round(msg.twist.twist.angular.z, 4)})

    def _on_cmd_vel(self, msg):
        self._emit('cmd_vel', {'vx': round(msg.linear.x, 4),
                               'wz': round(msg.angular.z, 4)})

    def _on_cmd_nav(self, msg):
        self._emit('cmd_vel_nav', {'vx': round(msg.linear.x, 4),
                                   'wz': round(msg.angular.z, 4)})

    def _on_arc(self, msg):
        pts = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)]
               for p in msg.poses]
        self._emit('collision_arc', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'n': len(pts), 'pts': pts})

    def _on_monitor(self, msg):
        self._emit('collision_monitor', {
            'state': int(msg.state), 'action_type': int(msg.action_type)})

    def _on_costmap(self, msg):
        now = time.monotonic()
        if now - self._last_costmap < 1.0 / max(0.1, self._args.costmap_rate):
            return
        self._last_costmap = now
        raw = bytes((int(v) & 0xFF) for v in msg.data)
        self._emit('costmap', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'res': round(msg.info.resolution, 4),
            'w': msg.info.width, 'h': msg.info.height,
            'ox': round(msg.info.origin.position.x, 4),
            'oy': round(msg.info.origin.position.y, 4),
            'data': base64.b64encode(zlib.compress(raw, 6)).decode('ascii')})

    def _on_global_costmap(self, msg):
        now = time.monotonic()
        if now - self._last_global_costmap < 1.0 / max(0.1, self._args.global_costmap_rate):
            return
        self._last_global_costmap = now
        raw = bytes((int(v) & 0xFF) for v in msg.data)
        self._emit('global_costmap', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'res': round(msg.info.resolution, 4),
            'w': msg.info.width, 'h': msg.info.height,
            'ox': round(msg.info.origin.position.x, 4),
            'oy': round(msg.info.origin.position.y, 4),
            'data': base64.b64encode(zlib.compress(raw, 6)).decode('ascii')})

    def _on_footprint(self, msg):
        now = time.monotonic()
        if now - self._last_footprint < 0.5:
            return
        self._last_footprint = now
        self._emit('footprint', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'pts': [[round(p.x, 3), round(p.y, 3)] for p in msg.polygon.points]})

    def _on_scan(self, msg):
        now = time.monotonic()
        if now - self._last_scan < 1.0 / max(0.1, self._args.scan_rate):
            return
        self._last_scan = now
        r = list(msg.ranges)
        self._emit('scan', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'amin': round(msg.angle_min, 4),
            'ainc': round(msg.angle_increment, 4),
            'rmin': round(msg.range_min, 3), 'rmax': round(msg.range_max, 3),
            'n': len(r), 'ranges': pack_floats(r)})

    def _on_movers(self, msg):
        now = time.monotonic()
        if now - self._last_movers < 1.0 / max(0.1, self._args.movers_rate):
            return
        self._last_movers = now
        off = {f.name: f.offset for f in msg.fields}
        if not {'x', 'y', 'z'} <= set(off):
            return
        pts = []
        for i in range(msg.width * msg.height):
            base = i * msg.point_step
            try:
                x, y, z = (struct.unpack_from('<f', msg.data, base + off[k])[0]
                           for k in ('x', 'y', 'z'))
            except struct.error:
                break
            pts.append([round(x, 3), round(y, 3), round(z, 3)])
        self._emit('movers', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'n': len(pts), 'pts': pts})

    def _on_voxel(self, msg):
        now = time.monotonic()
        if now - self._last_voxel < 1.0 / max(0.1, self._args.voxel_rate):
            return
        self._last_voxel = now
        off = {f.name: f.offset for f in msg.fields}
        if not {'x', 'y', 'z'} <= set(off):
            return
        pts = []
        n = msg.width * msg.height
        for i in range(n):
            if len(pts) >= self._args.voxel_cap:
                break
            base = i * msg.point_step
            try:
                x, y, z = (struct.unpack_from('<f', msg.data, base + off[k])[0]
                           for k in ('x', 'y', 'z'))
            except struct.error:
                break
            pts.append([round(x, 2), round(y, 2), round(z, 2)])
        self._emit('voxel', {
            'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
            'n': n, 'pts': pts})

    def _on_plan(self, msg):
        now = time.monotonic()
        if now - self._last_plan < 0.5:
            return
        self._last_plan = now
        self._emit('plan', self._path(msg))

    def _on_local_plan(self, msg):
        now = time.monotonic()
        if now - self._last_local_plan < 0.5:
            return
        self._last_local_plan = now
        self._emit('local_plan', self._path(msg))

    @staticmethod
    def _path(msg):
        return {'frame': msg.header.frame_id, 'stamp': stamp_of(msg),
                'pts': [[round(p.pose.position.x, 3),
                         round(p.pose.position.y, 3)] for p in msg.poses]}

    def _on_state(self, msg):
        self._last_state = msg.data
        self._emit('state', {'data': msg.data})
        if msg.data in TERMINAL_STATES:
            self._terminal_at = self._terminal_at or time.time()

    # -------------------------------------------------------------------- stop
    def _check_stop(self):
        if self._stopped:
            return
        elapsed = time.time() - self._t0
        if (self._terminal_at is not None
                and time.time() - self._terminal_at >= self._args.after_terminal):
            self._finish('terminal state reached')
        elif elapsed >= self._args.max_duration:
            self._finish('max duration reached')

    def _report(self):
        if self._stopped:
            return
        parts = ' '.join(f'{k}={v}' for k, v in sorted(self._seen.items()))
        self.get_logger().info(
            f't={time.time() - self._t0:.0f}s lines={self._lines} '
            f'state={self._last_state} | {parts}')

    def _on_signal(self, signum, frame):
        self._stop_requested = True

    def _finish(self, reason):
        if self._stopped:
            return
        self._stopped = True
        self._out.flush()
        self._out.close()
        self.get_logger().info(
            f'stopped after {time.time() - self._t0:.1f}s, {self._lines} '
            f'lines ({reason})')

    @property
    def stop_requested(self):
        return self._stopped or self._stop_requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-duration', type=float, default=1500.0)
    ap.add_argument('--after-terminal', type=float, default=10.0)
    ap.add_argument('--costmap-rate', type=float, default=2.0)
    ap.add_argument('--global-costmap-rate', type=float, default=1.0)
    ap.add_argument('--scan-rate', type=float, default=5.0)
    ap.add_argument('--movers-rate', type=float, default=5.0)
    ap.add_argument('--voxel-rate', type=float, default=1.0)
    ap.add_argument('--voxel-cap', type=int, default=1500)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = CollisionProbe(args)
    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node._finish('main loop exit')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

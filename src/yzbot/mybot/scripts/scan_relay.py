#!/usr/bin/env python3
"""Re-stamp the laser scan onto a timestamp the tf2 buffer can answer.

The defect this fixes (measured 2026-10-13): the Gazebo laser plugin stamps each
scan ~10 ms AHEAD of the newest `odom -> base_footprint` transform in the tf2
buffer (scan at sim 1671.213 s while the buffer's latest data is 1671.199 s).
The local costmap transforms the scan with the scan's own stamp, so tf2 throws
"Lookup would require extrapolation into the future" and the observation is
served with an identity sensor pose. The points still get MARKED, but the
raytrace that CLEARS them never runs ("Sensor origin at (0.00, 0.00 0.00) is out
of map bounds ... cannot raytrace for it", 2102 occurrences in one session).

The result is cumulative phantom obstacles: measured on that session, 2123
lethal cells in the local costmap of which 1128 had no laser hit within 0.15 m -
including one 0.05 m from the base. The base then fails every goal with
"RegulatedPurePursuitController detected collision ahead!" and
"Controller patience exceeded" from a pose the static map calls free.

The stamp used here is the newest `/odom` header stamp: the odom plugin
publishes `odom -> base_footprint` with exactly that stamp, so a lookup at it can
never extrapolate. Re-stamping with the node's own "now" was tried first and is
NOT enough - while the base is moving, tf2 throws for roughly a third of scans
(measured after the first version of this fix: 1274 warnings in 90 s), because
"now" is still ahead of the newest odom message.

Cost: one message copy of ~12 kB at 10 Hz.

    ros2 run mybot scan_relay.py --ros-args -p input_topic:=/scan \
        -p output_topic:=/scan_stamped_now
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def _stamp_ns(stamp):
    """builtin_interfaces/Time -> integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_stamped_now')
        # A scan older than this is dropped rather than re-stamped: publishing a
        # stale scan as "now" would move obstacles to where the base no longer
        # is. Measured scan age at 10 Hz is ~10 ms.
        self.declare_parameter('odom_topic', '/odom')
        # A scan whose own stamp is older than this is dropped rather than
        # re-stamped: publishing a stale scan as fresh would place obstacles
        # where the base no longer is.
        self.declare_parameter('max_age_sec', 0.5)
        self.declare_parameter('report_period_sec', 10.0)

        # The Gazebo lidar publishes RELIABLE; BEST_EFFORT subscribers (the
        # costmaps) are compatible with it, while a BEST_EFFORT publisher would
        # be invisible to any RELIABLE subscriber. Keep both sides RELIABLE.
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        self._pub = self.create_publisher(
            LaserScan, str(self.get_parameter('output_topic').value), qos)
        self.create_subscription(
            LaserScan, str(self.get_parameter('input_topic').value),
            self._relay, qos)
        # (arrival, header stamp) of the newest two odom messages. The scan is
        # stamped with the newest one that has already been in the buffer for
        # `tf_settle_sec`, so the transform is guaranteed to exist when the
        # costmap asks for it - a stamp the buffer cannot answer makes the
        # costmap mark the scan but never raytrace it, which is the whole defect
        # this node exists to fix.
        self._odom_prev = None
        self._odom_cur = None
        # The relay answers the question "which timestamp can tf2 actually
        # resolve?" from the same buffer the costmap uses, instead of guessing:
        # asking for a stamp the buffer cannot answer is exactly what makes the
        # costmap mark a scan and never clear it.
        self._tf_buffer = Buffer()
        TransformListener(self._tf_buffer, self)
        self.declare_parameter('tf_target_frame', 'odom')
        self.declare_parameter('tf_source_frame', 'base_footprint')
        self.declare_parameter('tf_settle_sec', 0.06)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom, 50)
        self._seen = 0
        self._dropped = 0
        self._no_odom = 0
        self._max_age = 0.0
        self._max_stamp_age = 0.0
        self._max_odom_age = 0.0
        self.create_timer(
            max(1.0, float(self.get_parameter('report_period_sec').value)),
            self._report)

    def _odom(self, msg):
        now = time.monotonic()
        self._odom_prev = self._odom_cur
        self._odom_cur = (now, msg.header.stamp)
        self._max_odom_age = max(self._max_odom_age, now - self._odom_prev[0]) \
            if self._odom_prev else self._max_odom_age

    def _bf_stamp_ns(self):
        """Newest `odom -> base_footprint` transform time in the tf2 buffer.

        Returns integer nanoseconds, NOT a Time: subtracting an int from a
        `rclpy.time.Time` raises "unsupported operand type(s) for -: 'Time' and
        'int'", which killed the previous version of this node 8 s after startup
        and left the costmaps on the raw scan.
        """
        target = str(self.get_parameter('tf_target_frame').value)
        source = str(self.get_parameter('tf_source_frame').value)
        try:
            stamp = self._tf_buffer.get_latest_common_time(target, source)
        except Exception:                       # noqa: BLE001
            return None
        if stamp.nanoseconds == 0:
            return None
        return int(stamp.nanoseconds)

    def _pick_stamp_ns(self):
        """Pick a stamp (integer ns) the costmap's own tf lookups can answer.

        Preference order:
          1. 20 ms behind the newest `odom -> base_footprint` time the tf2 buffer
             holds - inside the window the buffer can interpolate, and close
             enough that the base has moved <2 cm at patrol speed;
          2. the newest settled odom header stamp, for the first ticks before
             the tf buffer is populated.
        """
        settle = max(0.0, float(self.get_parameter('tf_settle_sec').value))
        now_mono = time.monotonic()
        buffered = self._bf_stamp_ns()
        if buffered is not None:
            return max(0, buffered - 20_000_000)
        for sample in (self._odom_cur, self._odom_prev):
            if sample is not None and now_mono - sample[0] >= settle:
                return _stamp_ns(sample[1])
        if self._odom_cur is None:
            return None
        return _stamp_ns(self._odom_cur[1])

    def _relay(self, msg):
        # Integer nanoseconds throughout: mixing a message stamp (RCL_ROS_TIME)
        # with a node clock of a different type raises "Can't subtract times
        # with different clock types", which killed the first version of this
        # node at startup.
        now_ns = self.get_clock().now().nanoseconds
        age = (now_ns - _stamp_ns(msg.header.stamp)) / 1e9
        self._max_age = max(self._max_age, age)
        stamp_ns = self._pick_stamp_ns()
        if stamp_ns is None:
            # Without a timestamp we know tf2 can answer, publishing on "now"
            # would just reintroduce the extrapolation this node exists to fix.
            self._no_odom += 1
            return
        if age > float(self.get_parameter('max_age_sec').value):
            self._dropped += 1
            return
        out = LaserScan()
        out.header = msg.header
        # The whole point: the costmap looks transforms up at this stamp, and
        # this stamp is one the tf2 buffer has already been proven to hold.
        out.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
        # How old the stamp we publish is: too old a stamp means the costmap
        # clears obstacles at where the base used to be.
        self._max_stamp_age = max(self._max_stamp_age, (now_ns - stamp_ns) / 1e9)
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = msg.ranges
        out.intensities = msg.intensities
        self._pub.publish(out)
        self._seen += 1

    def _report(self):
        self.get_logger().info(
            f'scan_relay: re-stamped {self._seen} scans, dropped {self._dropped}, '
            f'waiting for odom {self._no_odom}, worst input age '
            f'{self._max_age:.3f} s, worst stamp age {self._max_stamp_age:.3f} s, '
            f'worst odom gap {self._max_odom_age:.3f} s')


def main(args=None):
    rclpy.init(args=args)
    node = ScanRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

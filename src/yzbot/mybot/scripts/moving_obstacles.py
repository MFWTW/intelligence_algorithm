#!/usr/bin/env python3
"""Drive the moving-obstacle boxes so the base has to avoid them mid-task.

Each box is a normal Gazebo model that is inserted at runtime; this node serves
its pose every tick with /gazebo/set_entity_state, so the box cannot be shoved
off its segment and its speed is exactly the commanded one (see
_Patrol.advance for why a cmd_vel box could not hold either).

The boxes have the highest priority (requested 2026-10-13): their motion is a
pure function of time. They never yield, never slow, never leave their segment
and cannot be displaced - the base has no influence on them whatsoever, and it is
the base's job to keep out of their way. Every box is spawned KINEMATIC so base
contact cannot shove it either.

To make avoidance possible the node also publishes the boxes' footprints on
`obstacle_cloud_topic` (PointCloud2, base_link frame, one flat square per
box) so both Nav2 costmaps carry them as a real obstacle source and the planner
routes around them instead of discovering them with the lidar at the last moment.
The frame matters: a cloud in `map` makes its sensor origin (0,0), outside the
rolling local costmap for most of the route, so clearing cannot erase old marks.

Patrols are declared as one string so the parameter has no array-typing problem:

    patrols: "mover_1:-1.16,-0.84,-0.40,0.58,0.16; mover_2:1.30,-5.72,-0.10,-4.94,0.16"

i.e. `name:x_start,y_start,x_end,y_end,speed_mps`. The box's yaw in the world is
set to the segment direction, so the served pose drives the whole segment.

The segments themselves come from the robot's recorded trajectories and were
clearance-checked against the static map, so a box always leaves the base room
to get past. See run_logs/tools/make_movers_world.py.
"""
import math
import time

import rclpy
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
from std_msgs.msg import Header, String
from sensor_msgs_py import point_cloud2

# mover_2 is deliberately quicker than the others: a faster crossing obstacle
# forces an actual avoidance decision instead of letting the base pace behind it.
DEFAULT_PATROLS = (
    # 2026-10-12: mover_1 used to patrol (-1.16,-0.84)<->(-0.40,0.58), which
    # crosses the robot's own start pose at (0,0). With the boxes now loaded on
    # every competition start that flung the base ("ROBOT_LAUNCHED:base covered
    # ground at ... exceeding 3.0 m/s") on the very first leg. Moved to open
    # floor at x=-12, where the static map gives 0.87-1.39 m of clearance and the
    # nearest cube, zone plate or start pose is 6.3 m away.
    'mover_1:-12.00,-3.00,-10.50,-3.00,0.16;'
    'mover_2:1.30,-5.72,-0.10,-4.94,0.30;'
    # mover_3 carries the `k` flag: measured on movers_live/stack2.log it was
    # shoved off its own segment and wedged there for 60+ s. Kinematic bodies
    # ignore contact forces, so the base cannot displace it.
    'mover_3:11.76,-4.53,12.62,-5.87,0.16,k'
)


# Footprint served to the costmaps (and the X/Y size in BOX_SDF).
BOX_SIZE_M = 0.22


def transform_point_xyz(point, transform):
    """Apply a map->base_link TF to one simulated obstacle return.

    Use the entire quaternion, not only yaw: the robot can pitch under load.
    The cloud's frame must place its *origin* at the robot, or Nav2's rolling
    local costmap cannot raytrace stale box marks as the patrol moves away.
    """
    x, y, z = point
    q = transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-9:
        raise ValueError('invalid zero quaternion for moving-obstacle cloud')
    qx, qy, qz, qw = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    offset = transform.translation
    return (
        x + qw * tx + (qy * tz - qz * ty) + offset.x,
        y + qw * ty + (qz * tx - qx * tz) + offset.y,
        z + qw * tz + (qx * ty - qy * tx) + offset.z,
    )


# Reporting threshold only: a warning is logged when the base comes this close
# to a box. It has NO effect on the box - the box never yields.
DEFAULT_CLEARANCE_M = 0.75

def parse_patrols(text):
    """Parse the patrol spec into [(name, x0, y0, x1, y1, speed, kinematic)].

    An entry may still end with a ``k`` flag (kept for compatibility with
    existing configs). It no longer selects anything: EVERY box is inserted
    kinematic now, because a kinematic body ignores contact forces and the base
    must not be able to displace a highest-priority obstacle.
    """
    patrols = []
    for chunk in str(text or '').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'patrol {chunk!r} needs "name:x0,y0,x1,y1,speed[,k]"')
        name, values = chunk.split(':', 1)
        parts = [v.strip() for v in values.split(',')]
        kinematic = parts and parts[-1].lower() in ('k', 'kinematic', 'true')
        if kinematic:
            parts = parts[:-1]
        numbers = [float(v) for v in parts]
        if len(numbers) != 5:
            raise ValueError(
                f'patrol {name!r} needs 5 numbers (x0,y0,x1,y1,speed), '
                f'got {len(numbers)}'
            )
        x0, y0, x1, y1, speed = numbers
        if speed <= 0.0:
            raise ValueError(f'patrol {name!r} speed must be > 0, got {speed}')
        if math.hypot(x1 - x0, y1 - y0) < 0.2:
            raise ValueError(f'patrol {name!r} segment is shorter than 0.2 m')
        patrols.append((name.strip('/ '), x0, y0, x1, y1, speed, kinematic))
    if not patrols:
        raise ValueError('patrols is empty; nothing to move')
    return patrols


BOX_SDF = """<sdf version='1.7'>
  <model name='{name}'>
{kinematic}    <link name='link'>
      <inertial>
        <mass>6</mass>
        <inertia>
          <ixx>0.08</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.08</iyy><iyz>0</iyz><izz>0.08</izz>
        </inertia>
      </inertial>
      <collision name='collision'>
        <geometry><box><size>0.22 0.22 0.30</size></box></geometry>
        <max_contacts>10</max_contacts>
        <surface>
          <contact><ode><soft_cfm>0.001</soft_cfm><soft_erp>0.2</soft_erp>
            <max_vel>0.1</max_vel><min_depth>0.001</min_depth></ode></contact>
          <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
        </surface>
      </collision>
      <visual name='visual'>
        <geometry><box><size>0.22 0.22 0.30</size></box></geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Yellow</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class _Patrol:
    def __init__(self, name, x0, y0, x1, y1, speed, kinematic=True):
        # `s` is the commanded distance travelled along the segment. Driving the
        # pose from it (rather than commanding a velocity) is what makes the
        # speed exact and the box unshovable - see advance().
        self.name = name
        self.x0, self.y0 = x0, y0
        self.dx, self.dy = x1 - x0, y1 - y0
        self.length = math.hypot(self.dx, self.dy)
        self.ux, self.uy = self.dx / self.length, self.dy / self.length
        self.speed = speed
        # Always true: see _spawn_boxes. Kept as an attribute so the record can
        # still report what was asked for.
        self.kinematic = True
        self.direction = 1.0
        self.x, self.y = x0, y0
        self.seen = True
        self.yaw = math.atan2(self.dy, self.dx)
        # Progress the box was last seen making, for the stuck watchdog.
        self.s = self.length / 2.0      # start at the segment centre

    def centre(self):
        return (self.x0 + self.dx / 2.0, self.y0 + self.dy / 2.0)

    def advance(self, dt):
        """Move along the segment at exactly `speed`, ping-ponging at the ends.

        A velocity command cannot do this: measured 2026-10-11 the planar_move
        boxes achieved only 0.031 m/s for a 0.16 command (and 0.118 for 0.30),
        because ground friction stops the 6 kg box every control cycle - it only
        coasts v^2/(2*mu*g) per cycle. Serving the pose removes friction from the
        loop, so the commanded speed is the actual speed.
        """
        self.s += self.speed * dt * self.direction
        if self.s >= self.length:
            self.s = self.length - (self.s - self.length)
            self.direction = -1.0
        elif self.s <= 0.0:
            self.s = -self.s
            self.direction = 1.0
        self.s = min(max(self.s, 0.0), self.length)
        self.x = self.x0 + self.ux * self.s
        self.y = self.y0 + self.uy * self.s

    def progress(self):
        """Position along the segment, 0 at the start and 1 at the end."""
        return self.s / self.length



class MovingObstacles(Node):
    def __init__(self):
        super().__init__('moving_obstacles')
        self.declare_parameter('patrols', DEFAULT_PATROLS)
        self.declare_parameter('publish_rate_hz', 10.0)
        # Highest priority (requested 2026-10-13): a box's motion is a pure
        # function of time. It never yields, never slows, never leaves its
        # segment and is never displaced - nothing the base does changes where a
        # box is or where it goes next. Every box is served as a KINEMATIC body
        # (see _spawn_boxes), so base contact cannot shove it off its line
        # either. Keeping the base out of the way is the base's job.
        #
        # This parameter is only a diagnostic threshold: below it a warning is
        # logged so the round record shows the base came inside the box, but the
        # box's motion is not affected in any way. It used to gate the motion
        # (the "yield to the base" bubble), which froze mover_2 for 110 s of a
        # 130 s trace (run_logs/yield3_131041) and deadlocked the base in front
        # of it.
        self.declare_parameter('robot_clearance_m', DEFAULT_CLEARANCE_M)
        self.declare_parameter('spawn_on_start', True)
        # The first box is requested while Gazebo is still inserting the robot
        # and its plugins, and that request loses the race ("Entity pushed to
        # spawn queue, but ... timed out waiting for entity to appear",
        # measured on movers_live/stack5.log for mover_1 while mover_2 and
        # mover_3 succeeded). Retry rather than shipping a missing obstacle.
        # Inserting the boxes while Gazebo is still adding the robot wins the
        # spawn queue and the ROBOT never appears. Measured 2026-10-11 on
        # comp_mode_round.log: the boxes started at t+0.00 s, all three were in
        # by t+8.45 s, and spawn_entity sat on "Waiting for entity xml on
        # robot_description" forever - no /scan, no /map, no controllers. Wait
        # for the robot's own links to show up first.
        self.declare_parameter('wait_for_robot_sec', 120.0)
        self.declare_parameter('robot_model_prefix', 'six_arm')
        self.declare_parameter('spawn_attempts', 4)
        self.declare_parameter('spawn_retry_sec', 5.0)
        self.patrols = [ _Patrol(*spec) for spec in
                         parse_patrols(self.get_parameter('patrols').value) ]
        # No cmd_vel publisher: the boxes carry no plugin any more, their pose
        # is served directly. See _Patrol.advance for why.
        for patrol in self.patrols:
            self.get_logger().warning(
                f'{patrol.name}: patrolling ({patrol.x0:.2f},{patrol.y0:.2f})'
                f'<->({patrol.x0 + patrol.dx:.2f},{patrol.y0 + patrol.dy:.2f}) '
                f'at {patrol.speed:.2f} m/s, kinematic, highest priority (never '
                f'yields, never displaced)')
        self._reset_client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')
        self._spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self._robot_xy = None
        self.create_subscription(
            Odometry, '/odom',
            lambda msg: setattr(
                self, '_robot_xy',
                (msg.pose.pose.position.x, msg.pose.pose.position.y)),
            10)
        self._spawned = False
        self._last_tick = None
        rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.create_timer(1.0 / rate, self._tick)
        # Periodic proof of motion: without this the round record cannot show
        # whether the boxes actually patrolled.
        self.declare_parameter('report_period_sec', 5.0)
        period = max(0.5, float(self.get_parameter('report_period_sec').value))
        self.create_timer(period, self._report)
        # Costmap source: the boxes are stamped into both Nav2 costmaps so the
        # planner routes around them. The lidar sees them too, but only inside
        # the local costmap's 5 m window and only once they are already on the
        # planned path - measured 2026-10-13 the base drove straight at a box
        # and then paced behind it.
        self.declare_parameter('obstacle_cloud_topic', '/movers/obstacles')
        self.declare_parameter('publish_obstacle_cloud', True)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cloud_warn_last = -math.inf
        cloud_qos = QoSProfile(depth=1)
        cloud_qos.reliability = ReliabilityPolicy.RELIABLE
        cloud_qos.durability = DurabilityPolicy.VOLATILE
        self._cloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter('obstacle_cloud_topic').value),
            cloud_qos)
        # The centre lines themselves, latched, so a consumer joining later (the
        # executor is spawned per task) still learns where the lanes are. Without
        # this the yield policy cannot tell "stop beside the box" from "stop
        # inside its lane", and a base parked in a lane is trapped there: the box
        # comes back every few seconds (measured 2026-10-13, 18 s frozen, 30
        # hold/resume cycles). Same spec string as the `patrols` parameter.
        self.declare_parameter('patrols_topic', '/movers/patrols')
        patrols_qos = QoSProfile(depth=1)
        patrols_qos.reliability = ReliabilityPolicy.RELIABLE
        patrols_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._patrols_pub = self.create_publisher(
            String, str(self.get_parameter('patrols_topic').value), patrols_qos)
        self._patrols_msg = String()
        self._patrols_msg.data = str(self.get_parameter('patrols').value)
        # Latched topics only deliver on publish, so send it now and on every
        # report tick: a subscriber that starts later must still receive it.
        self._patrols_pub.publish(self._patrols_msg)

    def _publish_obstacle_cloud(self):
        """Publish each box in the robot-centred lidar frame, never map-origin.

        A PointCloud2 is interpreted as a *sensor*: Nav2 raytraces from the
        message frame's origin. `map` gives that sensor (0,0), which is outside
        the rolling local costmap over most of the route; 5244 raytrace warnings
        were recorded in yield_trial_1 and old mover marks could not clear.
        Using base_link provides a valid origin to both costmaps. The TF
        timestamp is also used for the cloud, so points and origin correspond
        to exactly the same pose; missing/stale TF skips publication rather than
        lying about where an obstacle is.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                'base_link', 'map', Time())
            stamp = tf.header.stamp
            stamp_seconds = stamp.sec + stamp.nanosec * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp_seconds
            if stamp_seconds <= 0.0 or not -0.5 <= age <= 0.5:
                raise ValueError(f'map-to-base transform age {age:.2f} s')
        except (TransformException, ValueError) as exc:
            now = time.monotonic()
            if now - self._cloud_warn_last >= 5.0:
                self._cloud_warn_last = now
                self.get_logger().warning(
                    f'Mover cloud skipped until a fresh map-to-base transform '
                    f'is available: {exc}')
            return
        half = BOX_SIZE_M / 2.0
        grid = 0.05
        cells = int(round(BOX_SIZE_M / grid))
        offsets = [(-half + ix * grid, -half + iy * grid)
                   for ix in range(cells + 1) for iy in range(cells + 1)]
        points = []
        for patrol in self.patrols:
            cos_yaw, sin_yaw = math.cos(patrol.yaw), math.sin(patrol.yaw)
            for dx, dy in offsets:
                world = (patrol.x + cos_yaw * dx - sin_yaw * dy,
                         patrol.y + sin_yaw * dx + cos_yaw * dy,
                         BOX_SIZE_M / 2.0)
                points.append(transform_point_xyz(world, tf.transform))
        header = Header()
        header.stamp = stamp
        header.frame_id = 'base_link'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self._cloud_pub.publish(point_cloud2.create_cloud(header, fields, points))

    def _tick(self):
        if not self._spawned:
            return                     # nothing to drive until the boxes exist
        now = self.get_clock().now().nanoseconds / 1e9
        dt = 0.0 if self._last_tick is None else max(0.0, min(now - self._last_tick, 0.5))
        self._last_tick = now
        if dt <= 0.0:
            return
        self._last_dt = dt
        for patrol in self.patrols:
            # The obstacle never yields, never slows and never leaves its
            # segment: the base has to keep out of its way (requested
            # 2026-10-13). Before this, the box held position for as long as the
            # base stayed inside a 0.75 m bubble - measured on
            # run_logs/yield3_131041, mover_2 sat frozen for 110 s of a 130 s
            # trace while the base paced in front of it, and neither could get
            # through.
            patrol.advance(dt)
            if not self._reset_client.service_is_ready():
                continue
            request = SetEntityState.Request()
            request.state.name = patrol.name
            request.state.pose.position.x = float(patrol.x)
            request.state.pose.position.y = float(patrol.y)
            request.state.pose.position.z = 0.15
            request.state.pose.orientation.z = math.sin(patrol.yaw / 2.0)
            request.state.pose.orientation.w = math.cos(patrol.yaw / 2.0)
            # Match the twist to the motion so contacts behave sensibly even
            # though the pose is authoritative.
            request.state.twist.linear.x = float(patrol.speed * patrol.direction)
            self._reset_client.call_async(request)
            self._warn_if_base_inside_box(patrol)
        if bool(self.get_parameter('publish_obstacle_cloud').value):
            self._publish_obstacle_cloud()

    def _report(self):
        self._patrols_pub.publish(self._patrols_msg)
        for patrol in self.patrols:
            if not patrol.seen:
                self.get_logger().warning(f'MOVER {patrol.name} no-odom')
                continue
            self.get_logger().info(
                f'MOVER {patrol.name} {patrol.x:.3f} {patrol.y:.3f} '
                f'progress={patrol.progress():.3f} dir={patrol.direction:+.0f}')

    def _warn_if_base_inside_box(self, patrol):
        """Diagnostic only: the base must never be inside a box.

        This changes NOTHING about the box: it has already been advanced, its
        pose is served regardless, and the warning exists so the round record
        shows a base that drove into an obstacle instead of leaving it to be
        inferred from a violent pose jump. The box is kinematic, so the base
        cannot move it; the base has to keep out of its way.
        """
        if self._robot_xy is None:
            return
        limit = max(0.0, float(self.get_parameter('robot_clearance_m').value))
        if limit <= 0.0:
            return
        rx, ry = self._robot_xy
        gap = math.hypot(patrol.x - rx, patrol.y - ry)
        if gap < limit:
            self.get_logger().warning(
                f'MOVER {patrol.name}: the base is {gap:.2f} m from the box '
                f'(keep-out {limit:.2f} m); the box keeps its priority and its '
                'motion unchanged.')

    def _wait_for_robot(self):
        """Block until Gazebo reports the robot's own links, or the timeout."""
        prefix = str(self.get_parameter('robot_model_prefix').value)
        limit = max(0.0, float(self.get_parameter('wait_for_robot_sec').value))
        if not prefix or limit <= 0.0:
            return True
        seen = {'robot': False}

        def _cb(msg):
            if any(name.startswith(prefix + '::') for name in msg.name):
                seen['robot'] = True

        subscription = self.create_subscription(
            LinkStates, '/gazebo/link_states', _cb, 10)
        deadline = time.monotonic() + limit
        while not seen['robot'] and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.destroy_subscription(subscription)
        if seen['robot']:
            self.get_logger().warning(
                f'{prefix} is in the scene; inserting the patrol boxes.')
        else:
            self.get_logger().error(
                f'{prefix} never appeared within {limit:.0f}s; inserting the '
                'boxes anyway (the robot spawn may still be pending).')
        return seen['robot']

    def _spawn_boxes(self):
        """Insert the boxes at runtime instead of putting them in the world.

        Measured 2026-10-11: with the boxes and their planar_move plugins present
        in the .world file, the robot's own spawn fails -
        "Entity pushed to spawn queue, but spawn service timed out waiting for
        entity to appear in simulation under the name [six_arm]" - while the same
        world with the plugins stripped spawns fine. The plugins block Gazebo's
        spawn queue during world load. Spawning the boxes afterwards avoids the
        race entirely and also removes the need for a world variant.
        """
        if not bool(self.get_parameter('spawn_on_start').value):
            return
        self._wait_for_robot()
        if not self._spawn_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(
                '/spawn_entity never became available; no boxes this run.')
            return
        attempts = max(1, int(self.get_parameter('spawn_attempts').value))
        retry = max(0.0, float(self.get_parameter('spawn_retry_sec').value))
        for patrol in self.patrols:
            centre = patrol.centre()
            for attempt in range(1, attempts + 1):
                request = SpawnEntity.Request()
                request.name = patrol.name
                # Every box is inserted kinematic: a kinematic body ignores
                # contact forces, so the base cannot shove it off its segment
                # even when it drives straight into it (measured 2026-10-11 the
                # base displaced a dynamic box ~0.7 mm per tick before this).
                # The base is a dynamic body, so the box still pushes IT.
                request.xml = BOX_SDF.format(
                    name=patrol.name,
                    kinematic='    <kinematic>true</kinematic>\n')
                request.robot_namespace = ''
                request.reference_frame = 'world'
                request.initial_pose.position.x = float(centre[0])
                request.initial_pose.position.y = float(centre[1])
                request.initial_pose.position.z = 0.15
                request.initial_pose.orientation.z = math.sin(patrol.yaw / 2.0)
                request.initial_pose.orientation.w = math.cos(patrol.yaw / 2.0)
                future = self._spawn_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
                result = future.result()
                message = getattr(result, 'status_message', '') if result else 'no reply'
                # "already exists" means the FIRST request did create the box -
                # Gazebo just failed to confirm it in time. Measured
                # 2026-10-11 on check_round.log: mover_1 reported a timeout on
                # attempt 1, then "Entity [mover_1] already exists" on attempts
                # 2-4, so treating that as failure wrongly declared a box missing
                # that was in fact in the scene.
                if (bool(result) and result.success) or 'already exists' in message:
                    self.get_logger().warning(
                        f'{patrol.name}: spawn ok at '
                        f'({centre[0]:.2f},{centre[1]:.2f}) on attempt {attempt}'
                        + (' (already present)' if 'already exists' in message else ''))
                    break
                if attempt == attempts:
                    self.get_logger().error(
                        f'{patrol.name}: spawn FAILED after {attempts} attempts - '
                        f'{message}')
                else:
                    self.get_logger().warning(
                        f'{patrol.name}: spawn attempt {attempt} failed '
                        f'({message[:60]}); retrying in {retry:.0f}s')
                    time.sleep(retry)
        self._spawned = True

def main(args=None):
    rclpy.init(args=args)
    try:
        node = MovingObstacles()
    except Exception as exc:
        rclpy.logging.get_logger('moving_obstacles').error(
            f'Moving obstacles failed to start: {exc}')
        rclpy.shutdown()
        raise SystemExit(1)
    try:
        node.get_logger().warning('Spawning the patrol boxes...')
        node._spawn_boxes()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

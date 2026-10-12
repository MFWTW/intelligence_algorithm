#!/usr/bin/env python3
"""Drive the moving-obstacle boxes so the base has to avoid them mid-task.

Each box is a normal Gazebo model with libgazebo_ros_planar_move.so, so it moves
with real physics (it is not teleported, and it can be bumped). This node only
sends `cmd_vel` on `/<name>/cmd_vel` and watches `/<name>/odom` to know where the
box actually is.

Patrols are declared as one string so the parameter has no array-typing problem:

    patrols: "mover_1:-1.16,-0.84,-0.40,0.58,0.16; mover_2:1.30,-5.72,-0.10,-4.94,0.16"

i.e. `name:x_start,y_start,x_end,y_end,speed_mps`. The box's yaw in the world is
set to the segment direction (planar_move drives along the model's own +x), so a
single linear.x drives the whole segment; direction is flipped when the measured
position passes an endpoint, which self-corrects if the base shoves a box.

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


# Base radius 0.35 m + box half-diagonal ~0.16 m + margin. Shared with the
# parameter default so a test can assert the two cannot drift apart.
# Raised from 0.55 on 2026-10-12: the loaded robot's own safety radius is 0.40 m
# and Nav2 inflates it to 0.60 m, so a box parked 0.55 m away already sits inside
# the envelope the planner refuses to enter. 0.75 keeps the box clear of it.
DEFAULT_CLEARANCE_M = 0.75


def parse_patrols(text):
    """Parse the patrol spec into [(name, x0, y0, x1, y1, speed, kinematic)].

    An entry may end with an optional ``k`` flag marking the box kinematic:
    a kinematic body ignores contact forces, so the base cannot shove it off its
    segment (and it does not fight ground friction while coasting).
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
    def __init__(self, name, x0, y0, x1, y1, speed, kinematic=False):
        # `s` is the commanded distance travelled along the segment. Driving the
        # pose from it (rather than commanding a velocity) is what makes the
        # speed exact and the box unshovable - see advance().
        self.name = name
        self.x0, self.y0 = x0, y0
        self.dx, self.dy = x1 - x0, y1 - y0
        self.length = math.hypot(self.dx, self.dy)
        self.ux, self.uy = self.dx / self.length, self.dy / self.length
        self.speed = speed
        self.kinematic = bool(kinematic)
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
        # A served pose is authoritative and the box cannot be shoved, so if it
        # is moved onto the base ODE resolves the overlap by flinging the base:
        # measured 2026-10-11 "base covered ground at 4.4 m/s (reported twist
        # 1.3 m/s)" -> ROBOT_LAUNCHED aborted the round. Hold a box while the
        # base is close instead of driving through it.
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
                f'at {patrol.speed:.2f} m/s')
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

    def _tick(self):
        if not self._spawned:
            return                     # nothing to drive until the boxes exist
        now = self.get_clock().now().nanoseconds / 1e9
        dt = 0.0 if self._last_tick is None else max(0.0, min(now - self._last_tick, 0.5))
        self._last_tick = now
        if dt <= 0.0:
            return
        clearance = max(0.0, float(self.get_parameter('robot_clearance_m').value))
        for patrol in self.patrols:
            if self._robot_too_close(patrol, clearance):
                # Yield to the base rather than materialising inside it.
                continue
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

#!/usr/bin/env python3

    def _report(self):
        for patrol in self.patrols:
            if not patrol.seen:
                self.get_logger().warning(f'MOVER {patrol.name} no-odom')
                continue
            self.get_logger().info(
                f'MOVER {patrol.name} {patrol.x:.3f} {patrol.y:.3f} '
                f'progress={patrol.progress():.3f} dir={patrol.direction:+.0f}')

    def _robot_too_close(self, patrol, clearance):
        """True while the base is within `clearance` of where the box would go."""
        if self._robot_xy is None or clearance <= 0.0:
            return False
        step = patrol.direction * patrol.speed
        ahead = min(max(patrol.s + step * 0.2, 0.0), patrol.length)
        bx = patrol.x0 + patrol.ux * ahead
        by = patrol.y0 + patrol.uy * ahead
        rx, ry = self._robot_xy
        return math.hypot(bx - rx, by - ry) < clearance

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
                request.xml = BOX_SDF.format(
                    name=patrol.name,
                    kinematic='    <kinematic>true</kinematic>\n'
                    if patrol.kinematic else '')
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

#!/usr/bin/env python3

"""Competition task supervisor with explicit, recoverable execution states.

The node consumes the validated JSON published by ``task_parser.py``, expands it
into a deterministic cube queue, and runs the existing single-cube executor for
one object at a time.  A child process boundary prevents a failed action client
or stale callback from contaminating the next object attempt.
"""

import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import rclpy
import yaml
from action_msgs.srv import CancelGoal
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32, String


VALID_STATES = {
    'IDLE', 'PARSING_TASK', 'VALIDATING_TASK', 'DETECTING_OBJECT',
    'NAV_TO_OBJECT', 'PRE_GRASP', 'GRASPING', 'VERIFY_GRASP',
    'NAV_TO_ZONE', 'PLACING', 'VERIFY_PLACE', 'NEXT_OBJECT',
    'COMPLETED', 'FAILED',
}


class TaskStateMachine(Node):
    """Supervise parsing readiness and sequential pick/place subprocesses."""

    def __init__(self) -> None:
        super().__init__('task_state_machine')
        self.declare_parameter('executor_params_file', '')
        self.declare_parameter('max_object_retries', 1)
        self.declare_parameter('skip_failed_object', False)
        self.declare_parameter('object_timeout_sec', 300.0)
        # The generated problem guarantees x + y = 5, so every competition run
        # carries five cubes and cannot fit in a fixed short budget: measured
        # per-cube wall time is about 184 s. 0 means "derive the budget from the
        # queue length"; a positive value is used verbatim as an override.
        self.declare_parameter('task_timeout_sec', 0.0)
        self.declare_parameter('task_timeout_per_object_sec', 360.0)
        self.declare_parameter('task_startup_grace_sec', 180.0)
        self.declare_parameter('health_loss_grace_sec', 3.0)
        self.declare_parameter('autostart', True)
        # A finished or failed run must be repeatable: the parser republishes the
        # same JSON when the same problem is generated again.
        self.declare_parameter('allow_task_replay', True)
        # Resident executor (opt-in, default off). When enabled the supervisor
        # spawns fixed_joint_pick_place.py ONCE per task with resident_mode:=true
        # and streams one goal per cube over /competition/executor_goal, instead
        # of forking a new process (and a new DDS participant) for every cube.
        # Measured 2026-10-07: a fresh participant can burn 55-60 s discovering
        # the Nav2 action and costmap services, and round13 lost the whole run at
        # cube 3 when it never discovered them at all.
        self.declare_parameter('use_resident_executor', False)
        # Nearest-first cube selection (default on). The task JSON fixes only the
        # colour, the count and the zone, so any cube of that colour may be used -
        # but the queue used to take cube numbers 1..N, which on the x=4, y=1 task
        # sent the robot to blue_cube_1 30.2 m away while blue_cube_3 sat 14.4 m
        # away, and to red_cube_3 (16.2 m from the zone) while red_cube_5 (7.1 m)
        # was never touched: 33.9 m / ~56 s of avoidable driving, more than every
        # controller change of this session. It also skips blue_cube_1's Wall_57
        # clearance route, the one leg that has failed a whole round with
        # "No valid trajectories out of 419" (round19).
        self.declare_parameter('select_nearest_cubes', True)
        # Base sanity watchdog. Gazebo can blow up instead of failing politely:
        # on 2026-10-07 a cube-carrying base that had been jammed against a wall
        # for 29 s was ejected at 36 m/s, flew 72 m and landed outside the map
        # (2.5 m high, still holding the cube) while /cmd_vel was zero. Nothing
        # noticed: the BT kept replanning for another 10 s (ComputePathToPose
        # even reported SUCCESS mid-flight) and the run only ended at the nav
        # timeout, leaving the cube attached and the base off-map.
        # Two cheap invariants catch that immediately. The bounds are the static
        # map in the *map* frame; /odom is compared against them with a margin
        # that absorbs the AMCL map->odom correction (measured ~0.1 m).
        self.declare_parameter('base_sanity_enabled', True)
        self.declare_parameter('max_plausible_speed_mps', 2.0)
        self.declare_parameter('map_bounds', [-15.05, -16.00, 16.99, 6.15])
        self.declare_parameter('map_bounds_margin_m', 1.5)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._state_pub = self.create_publisher(String, '/competition/state', qos)
        self._progress_pub = self.create_publisher(String, '/competition/progress', qos)
        self._object_pub = self.create_publisher(String, '/competition/current_object', qos)
        self._error_pub = self.create_publisher(String, '/competition/error', qos)
        self._retry_pub = self.create_publisher(Int32, '/competition/retry_count', qos)
        self._stop_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        # Killing the executor subprocess does NOT cancel the NavigateToPose goal
        # it sent: Nav2 keeps driving (and running recovery behaviours) until the
        # goal finishes on its own. Measured 2026-10-06: after the task had
        # already been declared FAILED, the leftover goal kept the base busy for
        # another 5.4 minutes. An empty CancelGoal request means "cancel every
        # goal" for that action server; the done-callback keeps it non-blocking
        # because this node is spun by a single-threaded executor.
        self._cancel_nav_client = None
        try:
            self._cancel_nav_client = self.create_client(
                CancelGoal, '/navigate_to_pose/_action/cancel_goal'
            )
        except Exception as exc:  # noqa: BLE001 - cancel support is best effort
            self.get_logger().warning(
                f'Could not create the Nav2 cancel client: {exc}'
            )

        self.create_subscription(Bool, '/competition/system_ready', self._ready_cb, qos)
        self.create_subscription(String, '/competition/task', self._task_cb, qos)
        self.create_subscription(
            String, '/competition/task_status', self._parser_status_cb, qos
        )
        self.create_subscription(
            String, '/competition/executor_state', self._executor_state_cb, qos
        )
        self.create_subscription(
            String, '/competition/executor_error', self._executor_error_cb, qos
        )
        # Resident-executor handshake. Both endpoints are TRANSIENT_LOCAL so a
        # goal published right after the resident process starts is still
        # delivered when its subscription appears a moment later.
        self._goal_pub = self.create_publisher(
            String, '/competition/executor_goal', qos
        )
        self.create_subscription(
            String, '/competition/executor_result', self._executor_result_cb, qos
        )
        # Best-effort: the watchdog must never block on a missing /odom.
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self._system_ready = False
        self._not_ready_since = time.monotonic()
        self._queue = []
        self._total = 0
        self._completed = 0
        self._index = 0
        self._attempt = 0
        self._process = None
        self._process_started = 0.0
        self._task_started = None
        self._terminal = False
        self._state = ''
        self._last_task_json = ''
        self._unsafe_to_retry = False
        self._last_executor_error = ''
        # Base sanity watchdog: (x, y, speed, monotonic) of the last odom sample,
        # plus the previous pose so the speed can be derived from the pose delta
        # as well as the reported twist (a blow-up can leave the twist stale for
        # a few frames). Deliberately not reset per cube, so a sample that
        # straddles two cubes cannot look like a teleport.
        self._odom = None
        self._odom_prev = None
        # Resident-executor bookkeeping: _object_active mirrors "a cube is in
        # flight" independently of _process, because in resident mode _process is
        # the long-lived executor rather than a per-cube child.
        self._object_active = False
        self._resident_process = None
        self._object_id = 0
        self._pending_object_id = 0
        self._timer = self.create_timer(0.2, self._tick)
        self._set_state('IDLE')
        self._publish_progress()

    @staticmethod
    def _string(text: str) -> String:
        msg = String()
        msg.data = text
        return msg

    def _set_state(self, state: str) -> None:
        if state not in VALID_STATES:
            self.get_logger().warning(f'Ignoring unknown state {state!r}.')
            return
        if state != self._state:
            self._state = state
            self.get_logger().info(f'State -> {state}')
        self._state_pub.publish(self._string(state))

    def _publish_progress(self) -> None:
        payload = {
            'completed': self._completed,
            'total': self._total,
            'current_index': self._index,
            'attempt': self._attempt,
        }
        self._progress_pub.publish(
            self._string(json.dumps(payload, separators=(',', ':')))
        )
        retry = Int32()
        retry.data = self._attempt
        self._retry_pub.publish(retry)

    def _task_budget(self) -> float:
        """Wall-clock budget for the whole queue.

        Measured subprocess time is ~184 s per cube and the generated problem
        always carries five cubes, so a fixed 290 s task timeout killed the run
        during the second cube. The budget is therefore derived from the queue
        length unless the operator overrides it explicitly.
        """
        override = float(self.get_parameter('task_timeout_sec').value)
        if override > 0.0:
            return override
        per_object = float(
            self.get_parameter('task_timeout_per_object_sec').value
        )
        grace = float(self.get_parameter('task_startup_grace_sec').value)
        return grace + per_object * max(1, self._total)

    def _fail(self, code: str, detail: str) -> None:
        self._stop_child()
        for _ in range(3):
            self._stop_pub.publish(Twist())
        message = f'{code}:{detail}'
        self._error_pub.publish(self._string(message))
        self.get_logger().error(message)
        self._terminal = True
        self._set_state('FAILED')
        self._publish_progress()

    def _ready_cb(self, msg: Bool) -> None:
        ready = bool(msg.data)
        if ready:
            self._not_ready_since = None
        elif self._system_ready or self._not_ready_since is None:
            self._not_ready_since = time.monotonic()
        self._system_ready = ready

    def _odom_cb(self, msg: Odometry) -> None:
        now = time.monotonic()
        position = msg.pose.pose.position
        linear = msg.twist.twist.linear
        speed = math.hypot(linear.x, linear.y)
        if self._odom_prev is not None:
            prev_x, prev_y, prev_t = self._odom_prev
            dt = now - prev_t
            if dt > 1e-3:
                speed = max(
                    speed,
                    math.hypot(position.x - prev_x, position.y - prev_y) / dt,
                )
        self._odom_prev = (position.x, position.y, now)
        self._odom = (position.x, position.y, speed, now)

    def _base_sanity_violation(self, now: float):
        """Return (code, detail) once the base stops obeying physics.

        Both checks are deliberately generous: the base is commanded at most
        vx_max (0.6-1.0 m/s) and the map margin absorbs the AMCL map->odom
        correction, so a healthy run cannot trip them.
        """
        if not bool(self.get_parameter('base_sanity_enabled').value):
            return None
        if self._odom is None:
            return None
        x, y, speed, stamp = self._odom
        if now - stamp > 1.0:          # stale odom: nothing trustworthy to judge
            return None
        limit = float(self.get_parameter('max_plausible_speed_mps').value)
        if speed > limit:
            return (
                'ROBOT_LAUNCHED',
                f'base speed {speed:.1f} m/s exceeds {limit:.1f} m/s '
                '(simulation blew up; /cmd_vel was not the cause)',
            )
        bounds = [float(v) for v in self.get_parameter('map_bounds').value]
        margin = float(self.get_parameter('map_bounds_margin_m').value)
        if not (
            bounds[0] - margin <= x <= bounds[2] + margin
            and bounds[1] - margin <= y <= bounds[3] + margin
        ):
            return (
                'ROBOT_OFF_MAP',
                f'base at ({x:.1f}, {y:.1f}) left the map '
                f'[{bounds[0]:.1f},{bounds[1]:.1f}]-[{bounds[2]:.1f},{bounds[3]:.1f}]',
            )
        return None

    def _task_active(self) -> bool:
        """True while a task is running (an object in flight, or more queued).

        In resident mode the executor process stays alive after the task ends, so
        "a child process exists" no longer means "a task is running" - the task
        itself decides, which keeps task replay and parser status working.
        """
        return self._task_started is not None and not self._terminal

    def _parser_status_cb(self, msg: String) -> None:
        status = msg.data.strip()
        if self._task_active() or self._terminal:
            return
        if status in {'GENERATING', 'PARSING', 'REPAIRING', 'REPAIRED'}:
            self._set_state('PARSING_TASK')
        elif status.startswith('FAILED:'):
            self._fail('TASK_PARSE', status)

    def _cube_catalogue(self):
        """Read the calibrated pickup poses: ({colour: [(cube, (x, y))]}, zones).

        The executor's parameter file is the source of truth for where each cube
        is, and it already lists all ten candidates.
        """
        try:
            _, params_path = self._executor_paths()
            with open(params_path, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, ValueError, yaml.YAMLError) as exc:  # noqa: BLE001
            self.get_logger().warning(f'Cube catalogue unavailable: {exc}')
            return {}, {}
        section = data.get('fixed_joint_pick_place', {}).get('ros__parameters', {})
        catalogue = {'red': [], 'blue': []}
        zones = {}
        for key, value in section.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            pickup = re.match(r'pickup_(red|blue)_cube_(\d+)_pose$', key)
            if pickup:
                name = key[len('pickup_'):-len('_pose')]
                catalogue[pickup.group(1)].append(
                    (name, (float(value[0]), float(value[1])))
                )
                continue
            zone = re.match(r'warehouse_([abc])_pose$', key)
            if zone:
                zones[zone.group(1).upper()] = (float(value[0]), float(value[1]))
        return catalogue, zones

    def _nearest_first_queue(self, queue):
        """Replace cube numbers with the cheapest available cubes of that colour.

        Cost per object is (current position -> pickup) + (pickup -> its zone),
        with the robot assumed to sit at the previous delivery zone (its start
        position for the first object). Any cube of the right colour is legal, so
        this is pure route shortening.
        """
        catalogue, zones = self._cube_catalogue()
        if not catalogue or not zones:
            self.get_logger().warning(
                'Nearest-first selection unavailable; keeping cube numbering.'
            )
            return queue
        chosen = set()
        reference = (0.0, 0.0)
        resolved = []
        for item in queue:
            colour = 'blue' if 'blue' in item['cube'] else 'red'
            zone_point = zones.get(item['zone'])
            candidates = [
                candidate
                for candidate in catalogue.get(colour, [])
                if candidate[0] not in chosen
            ]
            if not candidates or zone_point is None:
                resolved.append(item)
                continue
            cube, point = min(
                candidates,
                key=lambda candidate: math.dist(reference, candidate[1])
                + math.dist(candidate[1], zone_point),
            )
            chosen.add(cube)
            resolved.append({'cube': cube, 'zone': item['zone']})
            reference = zone_point
        self.get_logger().info(
            'Nearest-first cube selection: '
            + ' -> '.join(f"{item['cube']}:{item['zone']}" for item in resolved)
        )
        return resolved

    def _task_cb(self, msg: String) -> None:
        raw = msg.data.strip()
        if not raw:
            return
        if raw == self._last_task_json:
            # Re-running the same problem is only meaningful once the previous
            # run has settled; while a run is active the duplicate is ignored.
            if not (
                bool(self.get_parameter('allow_task_replay').value)
                and self._terminal
            ):
                return
            self.get_logger().info('Replaying the previous task on request.')
        if self._task_active():
            self.get_logger().warning('Ignoring a new task while execution is active.')
            return
        self._set_state('VALIDATING_TASK')
        try:
            payload = json.loads(raw)
            groups = payload['tasks']
            if not isinstance(groups, list) or not groups:
                raise ValueError('tasks must be a non-empty list')
            queue = []
            used = {'red': 0, 'blue': 0}
            seen_colours = set()
            for group in groups:
                colour = str(group['color']).lower()
                zone = str(group['zone']).upper()
                count = int(group['count'])
                if colour not in used or zone not in {'A', 'B', 'C'}:
                    raise ValueError(f'invalid color/zone: {colour}/{zone}')
                if colour in seen_colours:
                    raise ValueError(f'duplicate task color: {colour}')
                seen_colours.add(colour)
                if count < 0 or used[colour] + count > 5:
                    raise ValueError(f'invalid {colour} count: {count}')
                for number in range(used[colour] + 1, used[colour] + count + 1):
                    queue.append({'cube': f'{colour}_cube_{number}', 'zone': zone})
                used[colour] += count
            if bool(self.get_parameter('select_nearest_cubes').value):
                queue = self._nearest_first_queue(queue)
            if not queue:
                raise ValueError('task expands to an empty object queue')
            if 'x' in payload and 'y' in payload:
                expected_total = int(payload['x']) + int(payload['y'])
                if len(queue) != expected_total:
                    raise ValueError(
                        f'task count {len(queue)} does not match x+y={expected_total}'
                    )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._fail('INVALID_TASK', str(exc))
            return

        self._last_task_json = raw
        self._queue = queue
        self._total = len(queue)
        self._completed = 0
        self._index = 0
        self._attempt = 0
        # The budget starts when the first executor actually launches, not when
        # the JSON arrives: the parser runs at bringup time, so charging the
        # task for the Gazebo/Nav2 startup would waste real execution time.
        self._task_started = None
        self._terminal = False
        self._error_pub.publish(self._string(''))
        self._publish_progress()
        self.get_logger().info(
            f'Accepted deterministic task queue ({len(queue)} objects, '
            f'budget {self._task_budget():.0f}s): {queue}'
        )

    def _executor_state_cb(self, msg: String) -> None:
        if self._object_active and msg.data in VALID_STATES:
            self._set_state(msg.data)

    def _note_executor_error(self, text: str) -> None:
        """Classify one executor error: some states forbid an automatic retry."""
        self._last_executor_error = text
        self._unsafe_to_retry = any(
            marker in text
            for marker in ('CUBE_ATTACHED', 'CUBE_RELEASED', 'STATE_UNKNOWN')
        )

    def _executor_error_cb(self, msg: String) -> None:
        if self._object_active and msg.data:
            self._note_executor_error(msg.data)
            self._error_pub.publish(msg)

    def _executor_paths(self):
        prefix = Path(get_package_prefix('mybot'))
        executable = prefix / 'lib' / 'mybot' / 'fixed_joint_pick_place.py'
        params = str(self.get_parameter('executor_params_file').value).strip()
        if not params:
            params = os.path.join(
                get_package_share_directory('mybot'),
                'config',
                'fixed_joint_pick_place.yaml',
            )
        if not executable.is_file():
            raise FileNotFoundError(f'executor is not installed: {executable}')
        if not os.path.isfile(params):
            raise FileNotFoundError(f'executor parameter file not found: {params}')
        return str(executable), params

    def _executor_command(self, item: dict) -> list[str]:
        executable, params = self._executor_paths()
        return [
            executable, '--ros-args', '--params-file', params,
            '-p', f"cube_name:={item['cube']}",
            '-p', f"warehouse:={item['zone']}",
        ]

    def _resident_command(self) -> list[str]:
        executable, params = self._executor_paths()
        return [
            executable, '--ros-args', '--params-file', params,
            '-p', 'resident_mode:=true',
        ]

    def _resident_enabled(self) -> bool:
        return bool(self.get_parameter('use_resident_executor').value)

    def _ensure_resident(self) -> bool:
        """Start the long-lived executor if it is not already running."""
        process = self._resident_process
        if process is not None and process.poll() is None:
            self._process = process
            return True
        try:
            process = subprocess.Popen(
                self._resident_command(), start_new_session=True
            )
        except (OSError, ValueError) as exc:
            self.get_logger().error(f'Could not start the resident executor: {exc}')
            self._resident_process = None
            self._process = None
            return False
        self._resident_process = process
        self._process = process
        # Drop whatever goal the previous resident still had retained: the goal
        # topic is TRANSIENT_LOCAL so a late-joining subscriber is handed the last
        # sample, and that must never be an object this task already finished.
        self._goal_pub.publish(self._string(json.dumps({'cancel': True, 'object_id': 0})))
        self.get_logger().info(
            'Started the resident executor; it will serve every remaining cube.'
        )
        return True

    def _teardown_resident(self) -> None:
        process = self._resident_process
        self._resident_process = None
        if self._process is process:
            self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass

    def _release_process(self) -> None:
        """Forget the running executor after an object finished or died."""
        self._object_active = False
        if self._resident_enabled():
            # The resident process only disappears when it really exited.
            if self._process is not None and self._process.poll() is not None:
                self._resident_process = None
                self._process = None
            return
        self._process = None

    def _executor_result_cb(self, msg: String) -> None:
        """Resident-mode replacement for waiting on a child's exit code."""
        if not self._resident_enabled() or not self._object_active:
            return
        raw = msg.data.strip()
        if not raw:
            return
        try:
            result = json.loads(raw)
        except ValueError:
            self.get_logger().warning(f'Unusable executor result: {raw}')
            return
        if int(result.get('object_id', 0)) != self._pending_object_id:
            # A cancelled or already timed-out object reporting late.
            self.get_logger().warning(
                'Ignoring a stale executor result for object_id '
                f"{result.get('object_id')} (waiting for "
                f'{self._pending_object_id}).'
            )
            return
        success = bool(result.get('success'))
        if not success and result.get('error'):
            self._last_executor_error = str(result['error'])
            self._note_executor_error(self._last_executor_error)
        self._handle_child_exit(0 if success else 1)

    def _start_current(self) -> None:
        if self._index >= self._total:
            self._terminal = True
            self._object_pub.publish(self._string(''))
            self._set_state('COMPLETED')
            self._publish_progress()
            return
        item = self._queue[self._index]
        self._attempt += 1
        self._object_pub.publish(self._string(item['cube']))
        self._publish_progress()
        self._set_state('DETECTING_OBJECT')
        if self._resident_enabled():
            if not self._ensure_resident():
                self._fail('EXECUTOR_START', 'resident executor unavailable')
                return
            self._object_id += 1
            self._pending_object_id = self._object_id
            self._object_active = True
            self._goal_pub.publish(
                self._string(
                    json.dumps(
                        {
                            'cube': item['cube'],
                            'zone': item['zone'],
                            'attempt': self._attempt,
                            'object_id': self._object_id,
                        }
                    )
                )
            )
        else:
            try:
                command = self._executor_command(item)
                self._process = subprocess.Popen(command, start_new_session=True)
            except (OSError, ValueError) as exc:
                self._fail('EXECUTOR_START', str(exc))
                return
            self._object_active = True
        self._process_started = time.monotonic()
        if self._task_started is None:
            self._task_started = self._process_started
        self._unsafe_to_retry = False
        self._last_executor_error = ''
        self.get_logger().info(
            f"Started {item['cube']} -> {item['zone']} attempt {self._attempt}."
        )

    def _cancel_nav2_goals(self) -> None:
        """Cancel every leftover navigate_to_pose goal on the Nav2 server.

        Non-blocking on purpose: this node runs a single-threaded executor, so
        nested spinning (wait_for_service / spin_until_future_complete) inside a
        callback is not safe. A cancel request costs nothing when no goal is
        running, so it is safe to fire on every abnormal child exit.
        """
        client = self._cancel_nav_client
        if client is None:
            return
        if not client.service_is_ready():
            self.get_logger().warning(
                'Nav2 cancel service is not ready; a leftover navigation goal '
                'may keep running.'
            )
            return
        client.call_async(CancelGoal.Request()).add_done_callback(
            self._on_nav_cancel_done
        )

    def _on_nav_cancel_done(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 - best effort cleanup
            self.get_logger().warning(f'Nav2 cancel request failed: {exc}')
            return
        count = len(response.goals_canceling) if response is not None else 0
        if count:
            self.get_logger().info(
                f'Cancelled {count} leftover Nav2 goal(s) after stopping the executor.'
            )

    def _stop_child(self) -> None:
        if self._resident_enabled() and self._resident_process is not None:
            # The resident executor must survive an aborted object: ask it to
            # abandon the current cube, then cancel the goal it left in Nav2 so
            # the base stops driving behind the task logic.
            self._goal_pub.publish(
                self._string(
                    json.dumps({'cancel': True, 'object_id': self._pending_object_id})
                )
            )
            self.get_logger().warning(
                'Asked the resident executor to abandon the current object.'
            )
            self._object_active = False
            self._cancel_nav2_goals()
            return
        process = self._process
        self._process = None
        self._object_active = False
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
        self._cancel_nav2_goals()

    def _handle_child_exit(self, returncode: int) -> None:
        self._release_process()
        if returncode != 0:
            # The executor is gone; any goal it left in Nav2 must not keep
            # driving the base (and running recoveries) behind the task logic.
            self._cancel_nav2_goals()
        if returncode == 0:
            self._completed += 1
            self._index += 1
            self._attempt = 0
            self._set_state('NEXT_OBJECT')
            self._publish_progress()
            return

        if self._unsafe_to_retry:
            detail = self._last_executor_error or f'executor exited {returncode}'
            self._fail('UNSAFE_OBJECT_STATE', detail)
            return

        retries = max(0, int(self.get_parameter('max_object_retries').value))
        if self._attempt <= retries:
            self._error_pub.publish(
                self._string(f'OBJECT_RETRY:executor exited {returncode}')
            )
            self.get_logger().warning(
                f'Object executor failed ({returncode}); retrying after safety stop.'
            )
            self._stop_pub.publish(Twist())
            self._set_state('DETECTING_OBJECT')
            return
        if bool(self.get_parameter('skip_failed_object').value):
            self.get_logger().error('Retry budget exhausted; skipping object by policy.')
            self._index += 1
            self._attempt = 0
            self._set_state('NEXT_OBJECT')
            self._publish_progress()
            return
        self._fail('OBJECT_FAILED', f'executor exited {returncode}')

    def _tick(self) -> None:
        if self._terminal or not bool(self.get_parameter('autostart').value):
            return
        now = time.monotonic()
        if (
            self._queue
            and self._task_started is not None
            and now - self._task_started > self._task_budget()
        ):
            self._fail('TASK_TIMEOUT', 'competition task exceeded its time budget')
            return
        if self._object_active or self._task_started is not None:
            violation = self._base_sanity_violation(now)
            if violation is not None:
                code, detail = violation
                # A launched base is an unknown mechanical state: never retry it
                # automatically, and leave the cube alone for a manual reset.
                self._unsafe_to_retry = True
                self._last_executor_error = f'{code}:{detail}'
                self._fail(code, detail)
                return
        if self._object_active:
            if (
                not self._system_ready
                and self._not_ready_since is not None
                and now - self._not_ready_since > float(
                    self.get_parameter('health_loss_grace_sec').value
                )
            ):
                self._fail(
                    'SYSTEM_NOT_READY',
                    'health gate stayed false during execution',
                )
                return
            returncode = (
                self._process.poll() if self._process is not None else None
            )
            if returncode is not None:
                self._handle_child_exit(returncode)
                return
            if now - self._process_started > float(
                self.get_parameter('object_timeout_sec').value
            ):
                if self._state in {
                    'GRASPING', 'VERIFY_GRASP', 'NAV_TO_ZONE',
                    'PLACING', 'VERIFY_PLACE',
                }:
                    self._unsafe_to_retry = True
                    self._last_executor_error = (
                        f'OBJECT_TIMEOUT:{self._state}:state may be unsafe'
                    )
                self._stop_child()
                self._handle_child_exit(124)
            return
        if not self._queue:
            return
        if not self._system_ready:
            self._set_state('IDLE')
            return
        self._start_current()

    def destroy_node(self):
        self._stop_child()
        self._teardown_resident()
        if rclpy.ok():
            self._stop_pub.publish(Twist())
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskStateMachine()
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

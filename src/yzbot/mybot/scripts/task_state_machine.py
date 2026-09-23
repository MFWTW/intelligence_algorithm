#!/usr/bin/env python3

"""Competition task supervisor with explicit, recoverable execution states.

The node consumes the validated JSON published by ``task_parser.py``, expands it
into a deterministic cube queue, and runs the existing single-cube executor for
one object at a time.  A child process boundary prevents a failed action client
or stale callback from contaminating the next object attempt.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import Twist
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
        self.declare_parameter('object_timeout_sec', 240.0)
        self.declare_parameter('task_timeout_sec', 290.0)
        self.declare_parameter('health_loss_grace_sec', 3.0)
        self.declare_parameter('autostart', True)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._state_pub = self.create_publisher(String, '/competition/state', qos)
        self._progress_pub = self.create_publisher(String, '/competition/progress', qos)
        self._object_pub = self.create_publisher(String, '/competition/current_object', qos)
        self._error_pub = self.create_publisher(String, '/competition/error', qos)
        self._retry_pub = self.create_publisher(Int32, '/competition/retry_count', qos)
        self._stop_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

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

        self._system_ready = False
        self._not_ready_since = time.monotonic()
        self._queue = []
        self._total = 0
        self._completed = 0
        self._index = 0
        self._attempt = 0
        self._process = None
        self._process_started = 0.0
        self._task_started = 0.0
        self._terminal = False
        self._state = ''
        self._last_task_json = ''
        self._unsafe_to_retry = False
        self._last_executor_error = ''
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

    def _parser_status_cb(self, msg: String) -> None:
        status = msg.data.strip()
        if self._process is not None or self._terminal:
            return
        if status in {'GENERATING', 'PARSING', 'REPAIRING', 'REPAIRED'}:
            self._set_state('PARSING_TASK')
        elif status.startswith('FAILED:'):
            self._fail('TASK_PARSE', status)

    def _task_cb(self, msg: String) -> None:
        raw = msg.data.strip()
        if not raw or raw == self._last_task_json:
            return
        if self._process is not None:
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
        self._task_started = time.monotonic()
        self._terminal = False
        self._error_pub.publish(self._string(''))
        self._publish_progress()
        self.get_logger().info(f'Accepted deterministic task queue: {queue}')

    def _executor_state_cb(self, msg: String) -> None:
        if self._process is not None and msg.data in VALID_STATES:
            self._set_state(msg.data)

    def _executor_error_cb(self, msg: String) -> None:
        if self._process is not None and msg.data:
            self._last_executor_error = msg.data
            self._unsafe_to_retry = any(
                marker in msg.data
                for marker in ('CUBE_ATTACHED', 'CUBE_RELEASED', 'STATE_UNKNOWN')
            )
            self._error_pub.publish(msg)

    def _executor_command(self, item: dict) -> list[str]:
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
        return [
            str(executable), '--ros-args', '--params-file', params,
            '-p', f"cube_name:={item['cube']}",
            '-p', f"warehouse:={item['zone']}",
        ]

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
        try:
            command = self._executor_command(item)
            self._process = subprocess.Popen(command, start_new_session=True)
        except (OSError, ValueError) as exc:
            self._fail('EXECUTOR_START', str(exc))
            return
        self._process_started = time.monotonic()
        self._unsafe_to_retry = False
        self._last_executor_error = ''
        self.get_logger().info(
            f"Started {item['cube']} -> {item['zone']} attempt {self._attempt}."
        )

    def _stop_child(self) -> None:
        process = self._process
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

    def _handle_child_exit(self, returncode: int) -> None:
        self._process = None
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
        if self._queue and now - self._task_started > float(
            self.get_parameter('task_timeout_sec').value
        ):
            self._fail('TASK_TIMEOUT', 'competition task exceeded its time budget')
            return
        if self._process is not None:
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
            returncode = self._process.poll()
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

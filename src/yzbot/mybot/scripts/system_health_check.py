#!/usr/bin/env python3

import json
import os
import signal
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from moveit_msgs.action import MoveGroup
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


# Nav2 nodes that NavigateToPose depends on. bt_navigator in particular can be
# left 'inactive' when the bringup races, while its action server is still
# discoverable - which makes a naive action-name check report a false ready.
NAV2_NODES = [
    'amcl',
    'map_server',
    'planner_server',
    'controller_server',
    'bt_navigator',
    'behavior_server',
    'smoother_server',
    'waypoint_follower',
    'velocity_smoother',
]


def nav2_node_process_matches(cmdline: str, name: str) -> bool:
    """Does this process command line belong to Nav2 node ``name``?

    Module level so the restart watchdog's targeting is unit testable without
    touching /proc. The ``__node:=`` form is what Nav2's launch writes for the
    node itself; the ``lib/nav2_*`` path is the executable. A lifecycle
    manager's ``-p node_names:=[...]`` list mentions the name without either, and
    matching it would kill the manager instead of the node.
    """
    if '--ros-args' not in cmdline:
        return False
    if f'__node:={name}' in cmdline:
        return True
    return 'lib/nav2_' in cmdline and f'/{name}' in cmdline


class SystemHealthCheck(Node):
    """Publish a latched readiness flag after core simulation services are healthy."""

    def __init__(self):
        super().__init__('system_health_check')
        self.declare_parameter('timeout_sec', 180.0)
        self.declare_parameter('require_moveit', True)
        self.declare_parameter('require_nav2', True)
        # Observe lifecycle state and repair a bringup that wedged. A healthy
        # bringup finishes well inside recover_grace_sec, so the repair only
        # ever fires on a stack that is genuinely stuck (see _recover_nav2).
        self.declare_parameter('auto_recover_nav2', True)
        self.declare_parameter('recover_interval_sec', 5.0)
        self.declare_parameter('recover_grace_sec', 45.0)
        self.declare_parameter('action_rebuild_grace_sec', 20.0)
        self.declare_parameter('controller_query_timeout_sec', 5.0)
        # A node whose lifecycle services NEVER match cannot be repaired by
        # rebuilding the client: measured 2026-10-14 17:11, smoother_server's
        # GetState service stayed undiscovered through four rebuilds (61.9 s,
        # 79.9 s, 97.9 s, 117.9 s) and Nav2 could never activate, so
        # /competition/system_ready stayed false and the task sat in IDLE. Two of
        # the last fourteen launches died exactly this way. Restarting the node's
        # process gives it a fresh DDS participant (the Nav2 launch respawns it),
        # and the transition repair above then walks it back to active. The
        # restart is one attempt per node, only BEFORE the stack is ever ready,
        # so it can never fire while the robot is running.
        self.declare_parameter('restart_unmatched_nav2_node', True)
        self.declare_parameter('unmatched_rebuilds_before_restart', 3)
        # The second way a node livelocks: its service IS discovered but its
        # replies are not, so the health check keeps deciding from a stale state.
        # Measured 2026-10-14 17:18 (run_logs/resume_remaining_20261014-171824):
        # smoother_server configured fine at 1791969537, yet 18 CONFIGURE
        # requests later it was still reported non-active and ACTIVATE was never
        # requested. Six repair rounds is ~30 s at recover_interval_sec 5 s.
        self.declare_parameter('stuck_repairs_before_restart', 6)

        self._timeout_sec = float(self.get_parameter('timeout_sec').value)
        self._require_moveit = bool(self.get_parameter('require_moveit').value)
        self._require_nav2 = bool(self.get_parameter('require_nav2').value)
        self._started_at = time.monotonic()
        self._timed_out = False
        self._last_ready = None
        self._controller_future = None
        self._controller_requested_at = 0.0
        self._active_controllers = set()
        self._auto_recover = bool(
            self.get_parameter('auto_recover_nav2').value
        )
        self._recover_interval = float(
            self.get_parameter('recover_interval_sec').value
        )
        self._last_recover = 0.0
        self._controller_query_timeout = float(
            self.get_parameter('controller_query_timeout_sec').value
        )
        self._nav2_states = {}
        self._inactive_since = {}
        self._nav2_client_missing_since = {}
        self._state_clients = {}
        self._activate_clients = {}
        self._state_client_created = {}
        self._unmatched_rebuilds = {}
        self._restarted_nodes = set()
        self._restart_unmatched = bool(
            self.get_parameter('restart_unmatched_nav2_node').value
        )
        self._rebuilds_before_restart = max(
            1, int(self.get_parameter('unmatched_rebuilds_before_restart').value)
        )
        self._stuck_repairs = {}
        self._stuck_repairs_before_restart = max(
            1, int(self.get_parameter('stuck_repairs_before_restart').value)
        )
        self._ever_ready = False
        if self._require_nav2:
            for name in NAV2_NODES:
                self._state_clients[name] = self.create_client(
                    GetState, f'/{name}/get_state'
                )
                self._activate_clients[name] = self.create_client(
                    ChangeState, f'/{name}/change_state'
                )
                self._state_client_created[name] = time.monotonic()

        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._ready_pub = self.create_publisher(
            Bool, '/competition/system_ready', ready_qos
        )
        self._status_pub = self.create_publisher(
            String, '/competition/system_health', ready_qos
        )

        self._controller_client = self.create_client(
            ListControllers, '/controller_manager/list_controllers'
        )
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self._navigate_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose'
        )
        self._timer = self.create_timer(1.0, self._check_health)
        self.get_logger().info(
            'System health check started; task execution remains blocked until ready.'
        )

    def _request_controllers(self):
        if not self._controller_client.service_is_ready():
            return
        if self._controller_future is None:
            self._controller_future = self._controller_client.call_async(
                ListControllers.Request()
            )
            self._controller_requested_at = time.monotonic()
            return
        if not self._controller_future.done():
            # A single lost response used to wedge this checker forever: the
            # future stayed pending, _active_controllers stayed empty, and
            # /competition/system_ready never became true even though every
            # controller was active. Drop a stale request and ask again.
            if (
                time.monotonic() - self._controller_requested_at
                > self._controller_query_timeout
            ):
                self.get_logger().warning(
                    'Controller health query timed out; retrying.'
                )
                self._controller_future = None
            return

        try:
            response = self._controller_future.result()
            self._active_controllers = {
                controller.name
                for controller in response.controller
                if controller.state == 'active'
            }
        except Exception as exc:  # ROS service failures must not stop retries.
            self.get_logger().warning(f'Controller health query failed: {exc}')
        finally:
            self._controller_future = None

    def _poll_nav2_states(self):
        """Query each Nav2 node's lifecycle state without blocking the timer."""
        now = time.monotonic()
        for name, client in list(self._state_clients.items()):
            if not client.service_is_ready():
                # These clients are created before MoveIt/Nav2 exist (they start
                # only after the startup gate). On WSL/Fast DDS such a client can
                # stay unmatched *forever*, so the node is reported inactive even
                # though `ros2 lifecycle get` says active - and the health check
                # then blocks the task forever. Rebuild the client so discovery
                # runs again against the now-existing server.
                if now - self._state_client_created.get(name, now) > 20.0:
                    self.destroy_client(client)
                    self._state_clients[name] = self.create_client(
                        GetState, f'/{name}/get_state'
                    )
                    self._state_client_created[name] = now
                    rebuilds = self._unmatched_rebuilds.get(name, 0) + 1
                    self._unmatched_rebuilds[name] = rebuilds
                    self.get_logger().warning(
                        f'{name}: lifecycle service never matched; client rebuilt '
                        f'({rebuilds}/{self._rebuilds_before_restart}).'
                    )
                    if (rebuilds >= self._rebuilds_before_restart
                            and self._should_restart_node(name)):
                        self._restart_nav2_node(name, reason=(
                    f'its lifecycle service stayed undiscovered after '
                    f'{self._unmatched_rebuilds.get(name, 0)} client rebuilds'
                    if self._unmatched_rebuilds.get(name, 0)
                    else f'it stayed non-active through {self._stuck_repairs.get(name, 0)} '
                         'repair attempts'))
                continue
            self._unmatched_rebuilds[name] = 0
            self._stuck_repairs.pop(name, None)
            self._state_client_created[name] = now
            future = client.call_async(GetState.Request())
            future.add_done_callback(
                lambda done, node_name=name: self._store_state(node_name, done)
            )

    def _store_state(self, name, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f'{name} state query failed: {exc}')
            return
        if response is not None:
            self._nav2_states[name] = response.current_state.id

    def _should_restart_node(self, name: str) -> bool:
        """May this node be restarted right now?

        Only before the stack was ever ready (the robot cannot be running a task
        then) and only once per node, so a restart loop is impossible.
        """
        return (self._restart_unmatched and not self._ever_ready
                and name not in self._restarted_nodes)

    def _restart_nav2_node(self, name: str, reason: str = '') -> bool:
        """Terminate an undiscoverable Nav2 node so the launch respawns it.

        The Nav2 bringup starts every node with respawn=True, so killing the
        process yields a fresh DDS participant - which is what fixes a discovery
        failure that re-creating the client cannot (see the parameter comment).
        """
        self._restarted_nodes.add(name)
        victim = None
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in (os.getpid(), os.getppid()):
                continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as handle:
                    cmdline = handle.read().replace(b'\0', b' ').decode(
                        errors='ignore')
            except OSError:
                continue
            if nav2_node_process_matches(cmdline, name):
                victim = (pid, cmdline.strip()[:120])
                break
        if victim is None:
            self.get_logger().error(
                f'{name}: lifecycle services stayed undiscovered and no matching '
                'process was found to restart; the stack needs a manual restart.'
            )
            return False
        try:
            os.kill(victim[0], signal.SIGTERM)
        except OSError as exc:
            self.get_logger().error(f'{name}: could not restart pid {victim[0]}: {exc}')
            return False
        self.get_logger().error(
            f'{name}: {reason or "unrecoverable lifecycle state"}; terminated '
            f'pid {victim[0]} ({victim[1]}) so the launch respawns it with a '
            'fresh DDS participant.'
        )
        return True

    def _recover_nav2(self, inactive):
        """Nudge Nav2 nodes the bringup left behind (rate limited).

        The lifecycle managers own the normal bringup, but they can wedge
        permanently. Measured 2026-10-07: the Fast DDS response to
        map_server's CONFIGURE was dropped, lifecycle_manager_localization
        stayed blocked inside that one service call forever - no error, no
        retry, no timeout - so map_server stayed 'inactive', amcl stayed
        'unconfigured', /competition/system_ready stayed false, and the task
        sat in IDLE for 201 s until a human activated the nodes by hand.
        Re-issuing the transition recovers exactly that state, and the
        recover_grace_sec gate keeps this away from a healthy bringup.
        """
        now = time.monotonic()
        if now - self._last_recover < self._recover_interval:
            return
        self._last_recover = now
        for name in inactive:
            state = self._nav2_states.get(name)
            # Pick the transition the node actually needs: an 'inactive' node
            # wants ACTIVATE, an 'unconfigured' one wants CONFIGURE first.
            if state == State.PRIMARY_STATE_UNCONFIGURED:
                transition, label = Transition.TRANSITION_CONFIGURE, 'CONFIGURE'
            elif state == State.PRIMARY_STATE_INACTIVE:
                transition, label = Transition.TRANSITION_ACTIVATE, 'ACTIVATE'
            else:
                # Unknown state (never answered a query) or FINALIZED (needs a
                # cleanup we should not race): leave it to the managers.
                continue
            client = self._activate_clients.get(name)
            if client is None or not client.service_is_ready():
                continue
            request = ChangeState.Request()
            request.transition.id = transition
            attempts = self._stuck_repairs.get(name, 0) + 1
            self._stuck_repairs[name] = attempts
            self.get_logger().warning(
                f'{name} is stuck non-active; requesting {label} '
                f'({attempts}/{self._stuck_repairs_before_restart}).'
            )
            client.call_async(request)
            if (attempts >= self._stuck_repairs_before_restart
                    and self._should_restart_node(name)):
                # Repeating the same transition forever is a livelock when the
                # node's own replies are the thing being dropped: measured
                # 2026-10-14 17:18, smoother_server configured fine at 1791969537
                # yet the health check asked it to CONFIGURE another 18 times and
                # never reached ACTIVATE, because the state it read back stayed
                # stale. A fresh participant breaks that loop.
                self.get_logger().error(
                    f'{name}: still non-active after {attempts} repair attempts; '
                    'restarting it so the transitions can make progress again.'
                )
                self._restart_nav2_node(name, reason=(
                    f'it stayed non-active through '
                    f'{self._stuck_repairs.get(name, 0)} repair attempts'))

    def _check_health(self):
        now = time.monotonic()
        self._request_controllers()
        topic_names = {name for name, _ in self.get_topic_names_and_types()}

        required_topics = {
            '/clock', '/joint_states', '/odom', '/scan', '/tf', '/tf_static',
            '/camera/image_raw',
        }
        if self._require_nav2:
            required_topics.add('/map')

        missing_topics = sorted(required_topics - topic_names)
        required_controllers = {
            'joint_state_broadcaster', 'arm_controller', 'gripper_controller'
        }
        missing_controllers = sorted(
            required_controllers - self._active_controllers
        )

        moveit_ready = (
            not self._require_moveit
            or self._move_group_client.wait_for_server(timeout_sec=0.0)
        )

        # An inactive bt_navigator still advertises its action server, so the
        # lifecycle state - not action discovery - decides whether Nav2 works.
        inactive_nav2 = []
        if self._require_nav2:
            self._poll_nav2_states()
            inactive_nav2 = sorted(
                name for name in NAV2_NODES
                if self._nav2_states.get(name) != State.PRIMARY_STATE_ACTIVE
            )
            # Remember since when each node has been non-active, so the repair
            # below never races a bringup that is simply still in progress.
            # Only nodes whose state was actually observed may start the clock:
            # one that has not answered a GetState yet (the manager's configure
            # order has not reached it) must not count, or the repair fires in
            # the middle of a healthy bringup - measured 2026-10-07, it
            # CONFIGUREd smoother_server, velocity_smoother and
            # waypoint_follower while the manager was still walking the list.
            known_inactive = [
                name for name in inactive_nav2
                if self._nav2_states.get(name) is not None
            ]
            for name in known_inactive:
                self._inactive_since.setdefault(name, now)
            for name in list(self._inactive_since):
                if name not in known_inactive:
                    del self._inactive_since[name]
            if known_inactive and self._auto_recover:
                grace = float(self.get_parameter('recover_grace_sec').value)
                stuck = [
                    name for name in known_inactive
                    if now - self._inactive_since.get(name, now) >= grace
                ]
                if stuck:
                    self._recover_nav2(stuck)
        nav2_action_up = self._navigate_client.wait_for_server(timeout_sec=0.0)
        if self._require_nav2 and not nav2_action_up and not inactive_nav2:
            # Every Nav2 node is ACTIVE and the action server exists, yet this
            # client cannot see it: the client was built at node construction,
            # before Nav2 existed, and the discovery never completed. It does not
            # recover on its own - measured 2026-10-11 the stack sat with
            # /competition/system_ready false and the state machine IDLE forever
            # while `ros2 action list` showed /navigate_to_pose. Rebuild the
            # client after a grace period so a lost discovery costs seconds.
            self._nav2_client_missing_since.setdefault(now, now)
            grace = float(self.get_parameter('action_rebuild_grace_sec').value)
            if now - self._nav2_client_missing_since[now] >= grace:
                self.get_logger().warning(
                    f'/navigate_to_pose has been unreachable for {grace:.0f}s '
                    'while every Nav2 node is active; rebuilding the action '
                    'client for DDS discovery.')
                self._navigate_client.destroy()
                self._navigate_client = ActionClient(
                    self, NavigateToPose, '/navigate_to_pose')
                del self._nav2_client_missing_since[now]
        else:
            self._nav2_client_missing_since.clear()
        nav2_ready = (
            not self._require_nav2
            or (nav2_action_up and not inactive_nav2)
        )
        ready = not missing_topics and not missing_controllers
        ready = ready and moveit_ready and nav2_ready

        details = {
            'ready': ready,
            'missing_topics': missing_topics,
            'missing_controllers': missing_controllers,
            'moveit_action_ready': moveit_ready,
            'nav2_action_ready': nav2_ready,
            'nav2_inactive_nodes': inactive_nav2,
        }
        self._ready_pub.publish(Bool(data=ready))
        self._status_pub.publish(String(data=json.dumps(details, ensure_ascii=False)))

        if ready != self._last_ready:
            if ready:
                self._ever_ready = True
                self.get_logger().info(
                    'SYSTEM READY: Gazebo, controllers, sensors, MoveIt and Nav2 are available.'
                )
            else:
                self.get_logger().info(
                    'System not ready yet: ' + json.dumps(details, ensure_ascii=False)
                )
            self._last_ready = ready

        elapsed = time.monotonic() - self._started_at
        if not ready and elapsed >= self._timeout_sec and not self._timed_out:
            self.get_logger().error(
                'STARTUP HEALTH CHECK TIMEOUT: task execution remains blocked. '
                + json.dumps(details, ensure_ascii=False)
            )
            self._timed_out = True
        elif ready:
            self._timed_out = False


def main(args=None):
    rclpy.init(args=args)
    node = SystemHealthCheck()
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

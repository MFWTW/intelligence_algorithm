#!/usr/bin/env python3

import json
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
        self.declare_parameter('controller_query_timeout_sec', 5.0)

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
        self._state_clients = {}
        self._activate_clients = {}
        self._state_client_created = {}
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
                    self.get_logger().warning(
                        f'{name}: lifecycle service never matched; client rebuilt.'
                    )
                continue
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
            self.get_logger().warning(
                f'{name} is stuck non-active; requesting {label}.'
            )
            client.call_async(request)

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
        nav2_ready = (
            not self._require_nav2
            or (
                self._navigate_client.wait_for_server(timeout_sec=0.0)
                and not inactive_nav2
            )
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

#!/usr/bin/env python3

"""Navigate to a cube, visually align, grasp, carry, and place it.

The coarse pickup and warehouse poses are loaded from YAML.  After Nav2 reaches
an object's pre-grasp pose, a bounded HSV image servo centers the requested
colour and corrects the final standoff before the fixed-joint grasp sequence.
"""

import math
import sys
import time
from typing import Iterable, Sequence

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseStamped, Twist
from linkattacher_msgs.srv import AttachLink, DetachLink
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
]
GRIPPER_JOINTS = ['finger_joint1']


class FixedJointPickPlace(Node):
    """Run a configurable, single-cube fixed joint trajectory."""

    def __init__(self) -> None:
        super().__init__('fixed_joint_pick_place')

        self.declare_parameter('cube_name', 'red_cube_1')
        self.declare_parameter('warehouse', 'A')
        self.declare_parameter('navigation_frame', 'map')
        # Approach the rows from the east so cube_1 is not occluded by cubes
        # 2..5, and park far enough back that the cube stays in the camera view
        # across Nav2's parking tolerance; vision closes the rest.
        pickup_defaults = {
            'red_cube_1': [-0.920, 1.0, math.pi],
            'red_cube_2': [-1.020, 1.0, math.pi],
            'red_cube_3': [-1.120, 1.0, math.pi],
            'red_cube_4': [-1.220, 1.0, math.pi],
            'red_cube_5': [-1.320, 1.0, math.pi],
            'blue_cube_1': [-0.920, -1.0, math.pi],
            'blue_cube_2': [-1.020, -1.0, math.pi],
            'blue_cube_3': [-1.120, -1.0, math.pi],
            'blue_cube_4': [-1.220, -1.0, math.pi],
            'blue_cube_5': [-1.320, -1.0, math.pi],
        }
        for cube, pose in pickup_defaults.items():
            self.declare_parameter(f'pickup_{cube}_pose', pose)
        self.declare_parameter('use_backup_pickup_pose', True)
        self.declare_parameter('pickup_backup_offset_m', 0.15)
        # While coarse Nav2 is active, monitor the live Gazebo base-to-cube
        # distance. At this threshold cancel Nav2 and let the camera servo own
        # the remaining approach. This also catches paths that pass the cube
        # before reaching the calibrated pre-grasp goal.
        self.declare_parameter('enable_visual_handoff', True)
        self.declare_parameter('visual_handoff_distance_m', 1.5)

        # Saved runtime state places A/B/C at x=5, y=-11/-12/-13 with
        # 1.0 x 0.5 m plates. The goals below are pre-compensated by the
        # measured docking shortfall and carried-cube offset.
        self.declare_parameter('warehouse_a_pose', [4.792, -10.74, 0.0])
        self.declare_parameter('warehouse_b_pose', [4.792, -11.74, 0.0])
        self.declare_parameter('warehouse_c_pose', [4.792, -12.74, 0.0])
        # blue_cube_1 -> B otherwise follows the long global path into a DWB
        # local minimum beside Wall_57. These two clearance poses force the
        # loaded robot around the open north end before cross-map transport.
        self.declare_parameter(
            'blue_cube_1_warehouse_b_via_1', [-4.2, 0.0, 0.0]
        )
        self.declare_parameter(
            'blue_cube_1_warehouse_b_via_2', [-2.3, 0.0, 0.0]
        )
        self.declare_parameter('robot_model_name', 'six_arm')
        self.declare_parameter('robot_base_link', 'base_footprint')
        # The folded arm already sweeps about 0.32 m from the base centre while
        # empty. Loaded transport uses a 0.40 m safety radius: the physical cube
        # reaches about 0.35 m, and the extra 5 cm keeps DWB from entering exact
        # footprint tangency at wall ends where every rollout becomes invalid.
        self.declare_parameter('empty_robot_radius_m', 0.32)
        self.declare_parameter('empty_inflation_radius_m', 0.42)
        self.declare_parameter('loaded_robot_radius_m', 0.40)
        self.declare_parameter('loaded_inflation_radius_m', 0.60)
        # link6 is a real Gazebo link. grasping_frame is a MoveIt/TF frame and
        # may be collapsed by Gazebo's fixed-joint reduction.
        self.declare_parameter('robot_attach_link', 'link6')
        self.declare_parameter('cube_link', 'link')
        self.declare_parameter('require_system_ready', True)
        self.declare_parameter('ready_timeout_sec', 180.0)
        self.declare_parameter('action_timeout_sec', 30.0)
        self.declare_parameter('navigation_timeout_sec', 120.0)
        self.declare_parameter('navigation_retry_count', 2)
        self.declare_parameter('service_timeout_sec', 10.0)
        self.declare_parameter('require_proximity_check', True)
        self.declare_parameter('verify_placement', True)
        # Map-frame centres of the 1.0 x 0.5 m warehouse plates.
        self.declare_parameter('zone_a_centre', [5.0, -11.0])
        self.declare_parameter('zone_b_centre', [5.0, -12.0])
        self.declare_parameter('zone_c_centre', [5.0, -13.0])
        self.declare_parameter('zone_half_length_m', 0.5)
        self.declare_parameter('zone_half_width_m', 0.25)
        self.declare_parameter('placement_tolerance_m', 0.015)
        self.declare_parameter('placement_settle_sec', 1.5)
        # Gazebo leaves a small residual velocity in a just-released cube, which
        # makes it creep out of the zone. Simulation-only stabiliser.
        self.declare_parameter('stabilize_release', True)

        # Coarse Nav2 is followed by colour-based closed-loop base alignment.
        self.declare_parameter('enable_visual_alignment', True)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('fine_cmd_vel_topic', '/cmd_vel_nav')
        self.declare_parameter('visual_timeout_sec', 45.0)
        self.declare_parameter('visual_detection_stale_sec', 0.5)
        self.declare_parameter('visual_min_contour_area', 800.0)
        # Deliberately strict pure-red gate: the Gazebo cube is nearly pure red,
        # while brick/wood walls are darker and contain much more green/blue.
        self.declare_parameter('visual_red_min_saturation', 200)
        self.declare_parameter('visual_red_min_value', 100)
        self.declare_parameter('visual_red_dominance_ratio', 2.5)
        # Gazebo calibration: bottom row = 360 + 66.0 / (standoff - 0.015).
        # Row 680 stops well clear of the 720 px frame edge; the band is wide
        # enough to survive bottom-row jitter but tight enough that the creep
        # never drives link6 past the cube.
        self.declare_parameter('visual_approach_row', 680.0)
        self.declare_parameter('visual_row_tolerance', 10.0)
        # The creep needed to reach the 0.173 m standoff is derived from the
        # measured bottom row, so stopping anywhere in the band still lands
        # link6 on the cube. visual_final_creep_m is only a manual trim.
        self.declare_parameter('visual_row_distance_k', 66.0)
        self.declare_parameter('visual_row_distance_offset_m', 0.015)
        # Reject red/brown furniture whose vertical position is inconsistent
        # with the known target-cube range. The dynamic handoff already uses
        # Gazebo link states, so this simulation-only gate prevents a false blob
        # from ending search before the real floor-level cube enters view.
        self.declare_parameter('camera_forward_offset_m', 0.18)
        self.declare_parameter('visual_expected_row_tolerance', 80.0)
        self.declare_parameter('visual_target_standoff_m', 0.187)
        self.declare_parameter('visual_final_creep_m', 0.0)
        self.declare_parameter('visual_creep_speed', 0.025)
        self.declare_parameter('visual_center_tolerance', 0.04)
        self.declare_parameter('visual_linear_gain', 0.25)
        self.declare_parameter('visual_angular_gain', 0.8)
        self.declare_parameter('visual_max_linear_speed', 0.06)
        self.declare_parameter('visual_max_angular_speed', 0.25)
        self.declare_parameter('visual_stable_frames', 3)
        # If the cube is not in view (e.g. Nav2 parked off-target because of
        # localization drift), rotate in place to find it before giving up.
        self.declare_parameter('visual_search_rate', 0.35)
        self.declare_parameter('visual_search_timeout_sec', 18.0)
        self.declare_parameter('proximity_timeout_sec', 5.0)
        self.declare_parameter('max_attach_distance_m', 0.16)
        self.declare_parameter('max_attached_distance_m', 0.20)
        self.declare_parameter('arm_motion_sec', 3.0)
        self.declare_parameter('gripper_motion_sec', 1.0)
        self.declare_parameter('settle_sec', 0.4)

        # Starting values are based on the repository's existing manual
        # trajectory scripts and SRDF home state. They are calibration values,
        # not universal Cartesian poses.
        self.declare_parameter(
            'home_joints', [0.0, 0.6102, 1.2593, 0.0, -1.4931, 0.0]
        )
        self.declare_parameter(
            'pre_grasp_joints', [0.0, 0.9, 1.17, 0.0, -0.3, 0.0]
        )
        self.declare_parameter(
            'grasp_joints', [0.0, 1.2, 1.17, 0.0, -0.3, 0.0]
        )
        self.declare_parameter(
            'lift_joints', [0.0, 0.9, 1.17, 0.0, -0.3, 0.0]
        )
        self.declare_parameter(
            'transport_joints', [0.0, 0.6102, 1.2593, 0.0, -1.4931, 0.0]
        )
        self.declare_parameter(
            'pre_place_joints', [0.0, 0.9, 1.17, 0.0, -0.3, 0.0]
        )
        # Raised relative to the grasp posture so the cube is released on top
        # of the 0.02 m zone plate instead of inside it.
        self.declare_parameter(
            'place_joints', [0.0, 1.035, 1.295, 0.0, -0.3, 0.0]
        )
        # 0.0 opens a 75 mm finger gap; 0.048 closes it to 27 mm around the cube.
        self.declare_parameter('gripper_open', 0.0)
        self.declare_parameter('gripper_closed', 0.048)

        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
        )
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
        )
        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose',
        )
        self._set_state_client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )
        self._attach_client = self.create_client(AttachLink, '/ATTACHLINK')
        self._detach_client = self.create_client(DetachLink, '/DETACHLINK')
        self._clear_global_client = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap'
        )
        self._clear_local_client = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap'
        )
        self._global_costmap_params_client = self.create_client(
            SetParameters, '/global_costmap/global_costmap/set_parameters'
        )
        self._local_costmap_params_client = self.create_client(
            SetParameters, '/local_costmap/local_costmap/set_parameters'
        )
        self._bridge = CvBridge()
        self._visual_detection = None
        self._visual_detection_time = 0.0
        self._visual_range_gate_active = False
        self._target_colour = 'red'
        camera_topic = str(self.get_parameter('camera_topic').value)
        self._image_sub = self.create_subscription(
            Image, camera_topic, self._image_callback, qos_profile_sensor_data
        )
        self._fine_cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter('fine_cmd_vel_topic').value), 10
        )
        self._vision_state_pub = self.create_publisher(
            String, '/competition/vision_alignment', 10
        )
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._executor_state_pub = self.create_publisher(
            String, '/competition/executor_state', status_qos
        )
        self._executor_error_pub = self.create_publisher(
            String, '/competition/executor_error', status_qos
        )
        self._debug_image_pub = self.create_publisher(
            Image, '/vision/debug_image', 10
        )

        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._system_ready = False
        self._latest_link_states = None
        self._ready_sub = self.create_subscription(
            Bool,
            '/competition/system_ready',
            self._ready_callback,
            ready_qos,
        )
        self._link_states_sub = self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self._link_states_callback,
            10,
        )

    def _ready_callback(self, msg: Bool) -> None:
        self._system_ready = msg.data

    def _link_states_callback(self, msg: LinkStates) -> None:
        self._latest_link_states = msg

    def _image_callback(self, msg: Image) -> None:
        """Track the requested colour blob and publish its image geometry.

        The five cubes of one colour form a single merged blob when seen from
        the low forward camera, so the blob is selected by area among contours
        that reach the lower half of the image (the floor-level cube row). The
        blob's *bottom* row is the nearest cube's floor contact: unlike the
        blob width it is unaffected by heading error, which is what makes it a
        usable range signal.
        """
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Camera conversion failed: {exc}')
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self._target_colour == 'blue':
            mask = cv2.inRange(hsv, np.array([95, 80, 45]), np.array([135, 255, 255]))
            box_colour = (255, 0, 0)
        else:
            min_saturation = int(
                self.get_parameter('visual_red_min_saturation').value
            )
            min_value = int(self.get_parameter('visual_red_min_value').value)
            low_red = cv2.inRange(
                hsv,
                np.array([0, min_saturation, min_value]),
                np.array([6, 255, 255]),
            )
            high_red = cv2.inRange(
                hsv,
                np.array([174, min_saturation, min_value]),
                np.array([180, 255, 255]),
            )
            mask = cv2.bitwise_or(low_red, high_red)
            # HSV hue alone cannot distinguish pure red from brick red. Require
            # the red channel to dominate both green and blue as well.
            blue, green, red = cv2.split(image)
            ratio = float(
                self.get_parameter('visual_red_dominance_ratio').value
            )
            red_float = red.astype(np.float32)
            dominant = np.where(
                (red_float >= ratio * green.astype(np.float32))
                & (red_float >= ratio * blue.astype(np.float32)),
                255,
                0,
            ).astype(np.uint8)
            mask = cv2.bitwise_and(mask, dominant)
            box_colour = (0, 0, 255)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        minimum = float(self.get_parameter('visual_min_contour_area').value)
        image_height, image_width = image.shape[:2]
        expected_bottom = None
        if self._visual_range_gate_active:
            cube = str(self.get_parameter('cube_name').value)
            base_distance = self._cached_planar_distance_to_cube(cube)
            if base_distance is not None:
                camera_offset = float(
                    self.get_parameter('camera_forward_offset_m').value
                )
                row_offset = float(
                    self.get_parameter('visual_row_distance_offset_m').value
                )
                row_k = float(self.get_parameter('visual_row_distance_k').value)
                camera_standoff = max(0.05, base_distance - camera_offset)
                expected_bottom = 360.0 + row_k / max(
                    0.05, camera_standoff - row_offset
                )
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / max(1.0, float(height))
            # The merged cube row is taller than wide once the near cube gets
            # close, so the aspect window must stay permissive here.
            bottom_row = float(y + height)
            if bottom_row < 0.50 * image_height or not 0.15 <= aspect <= 3.0:
                continue
            if expected_bottom is not None:
                row_gate = float(
                    self.get_parameter('visual_expected_row_tolerance').value
                )
                if abs(bottom_row - expected_bottom) > row_gate:
                    continue
            if width > 0.60 * image_width or height > 0.95 * image_height:
                continue
            if best is None or area > best[0]:
                best = (area, x, y, width, height)

        if best is not None:
            area, x, y, width, height = best
            # The five same-colour cubes merge into one blob, so the bounding
            # box centre is not the target cube. The lowest strip of the blob is
            # the nearest cube: its centre is the point to align on.
            strip = max(1, int(0.2 * height))
            sub = mask[y + height - strip:y + height, x:x + width]
            columns = np.nonzero(sub)[1]
            centre_x = (
                x + float(columns.mean()) if columns.size else x + width / 2.0
            )
            self._visual_detection = (
                centre_x, float(width), float(area), float(image_width),
                float(y + height),
            )
            self._visual_detection_time = time.monotonic()
            cv2.rectangle(image, (x, y), (x + width, y + height), box_colour, 2)
            cv2.circle(image, (int(centre_x), y + height - 1), 5, (0, 255, 0), -1)

        if self._debug_image_pub.get_subscription_count() > 0:
            debug = self._bridge.cv2_to_imgmsg(image, encoding='bgr8')
            debug.header = msg.header
            self._debug_image_pub.publish(debug)

    def _publish_vision_state(self, text: str) -> None:
        message = String()
        message.data = text
        self._vision_state_pub.publish(message)

    def _publish_executor_state(self, state: str) -> None:
        message = String()
        message.data = state
        self._executor_state_pub.publish(message)
        self.get_logger().info(f'Executor state: {state}')

    def _publish_executor_error(self, error: str) -> None:
        message = String()
        message.data = error
        self._executor_error_pub.publish(message)

    def _stop_base(self) -> None:
        self._fine_cmd_pub.publish(Twist())

    def _double_list(self, name: str, expected_length: int) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != expected_length:
            raise ValueError(
                f'Parameter {name} must contain {expected_length} values; '
                f'got {len(values)}.'
            )
        return values

    def _wait_until(self, predicate, timeout_sec: float, description: str) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(f'Timed out waiting for {description}.')
        return False

    def _wait_for_dependencies(self) -> bool:
        if bool(self.get_parameter('require_system_ready').value):
            timeout = float(self.get_parameter('ready_timeout_sec').value)
            self.get_logger().info('Waiting for /competition/system_ready=true ...')
            if not self._wait_until(
                lambda: self._system_ready, timeout, 'system readiness'
            ):
                return False

        timeout = float(self.get_parameter('service_timeout_sec').value)
        dependencies = [
            (
                lambda: self._arm_client.wait_for_server(timeout_sec=0.0),
                'arm trajectory action',
            ),
            (
                lambda: self._gripper_client.wait_for_server(timeout_sec=0.0),
                'gripper trajectory action',
            ),
            (
                lambda: self._navigate_client.wait_for_server(timeout_sec=0.0),
                'Nav2 navigate_to_pose action',
            ),
            (self._attach_client.service_is_ready, '/ATTACHLINK'),
            (self._detach_client.service_is_ready, '/DETACHLINK'),
            (
                self._clear_global_client.service_is_ready,
                'global costmap clear service',
            ),
            (
                self._clear_local_client.service_is_ready,
                'local costmap clear service',
            ),
            (
                self._global_costmap_params_client.service_is_ready,
                'global costmap parameter service',
            ),
            (
                self._local_costmap_params_client.service_is_ready,
                'local costmap parameter service',
            ),
            (
                self._set_state_client.service_is_ready,
                '/gazebo/set_entity_state',
            ),
        ]
        for predicate, description in dependencies:
            if not self._wait_until(predicate, timeout, description):
                return False
        return True

    def _execute_trajectory(
        self,
        label: str,
        client: ActionClient,
        joint_names: Sequence[str],
        positions: Iterable[float],
        motion_sec: float,
    ) -> bool:
        positions = [float(value) for value in positions]
        self.get_logger().info(f'{label}: target={positions}')

        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=motion_sec).to_msg()
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = client.send_goal_async(goal)
        timeout = float(self.get_parameter('action_timeout_sec').value)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done():
            raise RuntimeError('MOTION_STATE_UNKNOWN: goal response timed out')

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'{label}: controller rejected the goal.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error(f'{label}: execution timed out; cancelling goal.')
            cancel_future = goal_handle.cancel_goal_async()
            grace = min(5.0, timeout)
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=grace)
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=grace)
            if not cancel_future.done() or not result_future.done():
                raise RuntimeError(
                    'MOTION_STATE_UNKNOWN: cancellation was not confirmed'
                )

        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error(f'{label}: action returned no result.')
            return False
        result = wrapped_result.result
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self.get_logger().error(
                f'{label}: failed with status={wrapped_result.status}, '
                f'code={result.error_code}, message={result.error_string!r}.'
            )
            return False

        time.sleep(float(self.get_parameter('settle_sec').value))
        return True

    def _move_arm(self, label: str, parameter_name: str) -> bool:
        return self._execute_trajectory(
            label,
            self._arm_client,
            ARM_JOINTS,
            self._double_list(parameter_name, len(ARM_JOINTS)),
            float(self.get_parameter('arm_motion_sec').value),
        )

    def _move_gripper(self, label: str, position: float) -> bool:
        return self._execute_trajectory(
            label,
            self._gripper_client,
            GRIPPER_JOINTS,
            [position],
            float(self.get_parameter('gripper_motion_sec').value),
        )

    def _warehouse_goal(self) -> tuple[str, list[float]]:
        warehouse = str(self.get_parameter('warehouse').value).strip().upper()
        parameter_names = {
            'A': 'warehouse_a_pose',
            'B': 'warehouse_b_pose',
            'C': 'warehouse_c_pose',
        }
        if warehouse not in parameter_names:
            raise ValueError(
                f'warehouse must be A, B or C; got {warehouse!r}.'
            )
        return warehouse, self._double_list(parameter_names[warehouse], 3)

    def _pickup_goal(self) -> tuple[str, list[float]]:
        cube = str(self.get_parameter('cube_name').value).strip()
        parameter_name = f'pickup_{cube}_pose'
        if not self.has_parameter(parameter_name):
            raise ValueError(f'No calibrated pickup pose for cube {cube!r}.')
        return cube, self._double_list(parameter_name, 3)

    def _clear_costmaps(self) -> bool:
        for label, client in (
            ('global', self._clear_global_client),
            ('local', self._clear_local_client),
        ):
            future = client.call_async(ClearEntireCostmap.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or future.exception() is not None:
                self.get_logger().error(
                    f'Could not confirm {label} costmap clear; refusing to '
                    'start a new goal while a late clear may still arrive.'
                )
                return False
        return True

    def _set_costmap_geometry(
        self, radius: float, inflation_radius: float, label: str
    ) -> bool:
        """Update both active Nav2 costmaps for empty or loaded geometry."""
        if radius <= 0.0 or inflation_radius < radius:
            self.get_logger().error(
                f'Invalid {label} costmap geometry: robot radius={radius:.3f} m, '
                f'inflation radius={inflation_radius:.3f} m.'
            )
            return False
        parameters = [
            Parameter(
                name='robot_radius',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=radius,
                ),
            ),
            Parameter(
                name='inflation_layer.inflation_radius',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=inflation_radius,
                ),
            ),
        ]
        timeout = float(self.get_parameter('service_timeout_sec').value)
        for costmap_label, client in (
            ('global', self._global_costmap_params_client),
            ('local', self._local_costmap_params_client),
        ):
            request = SetParameters.Request(parameters=parameters)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done() or future.exception() is not None:
                self.get_logger().error(
                    f'Could not update {costmap_label} costmap to the {label} '
                    'geometry.'
                )
                return False
            response = future.result()
            if response is None or len(response.results) != len(parameters):
                self.get_logger().error(
                    f'{costmap_label.capitalize()} costmap returned an invalid '
                    f'response for the {label} geometry.'
                )
                return False
            for result in response.results:
                if not result.successful:
                    self.get_logger().error(
                        f'{costmap_label.capitalize()} costmap rejected the '
                        f'{label} geometry: {result.reason}'
                    )
                    return False
        self.get_logger().info(
            f'Nav2 costmap geometry set for {label}: robot radius={radius:.3f} m, '
            f'inflation radius={inflation_radius:.3f} m.'
        )
        return True

    def _cached_planar_distance_to_cube(self, cube: str) -> float | None:
        """Return live base-to-cube XY distance without blocking navigation."""
        msg = self._latest_link_states
        if msg is None:
            return None
        base_name = (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('robot_base_link').value}"
        )
        cube_name = f"{cube}::{self.get_parameter('cube_link').value}"
        try:
            base_pose = msg.pose[list(msg.name).index(base_name)]
            cube_pose = msg.pose[list(msg.name).index(cube_name)]
        except ValueError:
            return None
        return math.hypot(
            base_pose.position.x - cube_pose.position.x,
            base_pose.position.y - cube_pose.position.y,
        )

    def _cached_planar_distance_to_point(self, x: float, y: float) -> float | None:
        """Return live Gazebo base distance to a map/world XY point."""
        msg = self._latest_link_states
        if msg is None:
            return None
        base_name = (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('robot_base_link').value}"
        )
        try:
            base_pose = msg.pose[list(msg.name).index(base_name)]
        except ValueError:
            return None
        return math.hypot(base_pose.position.x - x, base_pose.position.y - y)

    def _navigate_once(
        self,
        label: str,
        values: Sequence[float],
        visual_handoff_cube: str | None = None,
        completion_radius_m: float | None = None,
    ) -> bool:
        x, y, yaw = [float(value) for value in values]
        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter('navigation_frame').value)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(
            f'Navigate to {label}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}.'
        )
        send_future = self._navigate_client.send_goal_async(goal)
        response_timeout = float(self.get_parameter('service_timeout_sec').value)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=response_timeout)
        if not send_future.done():
            self.get_logger().error(
                f'Nav2 did not acknowledge {label} within '
                f'{response_timeout:.1f}s; refusing to wait indefinitely.'
            )
            self._stop_base()
            raise RuntimeError('NAV_STATE_UNKNOWN: goal response timed out')
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'Nav2 rejected the {label} goal.')
            return False

        result_future = goal_handle.get_result_async()
        timeout = float(self.get_parameter('navigation_timeout_sec').value)
        deadline = time.monotonic() + timeout
        handoff_distance = float(
            self.get_parameter('visual_handoff_distance_m').value
        )
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if completion_radius_m is not None:
                point_distance = self._cached_planar_distance_to_point(x, y)
                if (
                    point_distance is not None
                    and point_distance <= completion_radius_m
                ):
                    self.get_logger().info(
                        f'Clearance handoff reached for {label}: '
                        f'base-to-goal distance={point_distance:.3f} m <= '
                        f'{completion_radius_m:.3f} m; cancelling exact-pose '
                        'alignment and continuing the route.'
                    )
                    cancel_future = goal_handle.cancel_goal_async()
                    cancel_timeout = min(5.0, response_timeout)
                    rclpy.spin_until_future_complete(
                        self, cancel_future, timeout_sec=cancel_timeout
                    )
                    rclpy.spin_until_future_complete(
                        self, result_future, timeout_sec=cancel_timeout
                    )
                    for _ in range(3):
                        self._stop_base()
                        rclpy.spin_once(self, timeout_sec=0.1)
                    if not cancel_future.done() or not result_future.done():
                        raise RuntimeError(
                            'NAV_STATE_UNKNOWN: clearance handoff cancellation '
                            'was not confirmed'
                        )
                    return True
            if visual_handoff_cube is None:
                continue
            distance = self._cached_planar_distance_to_cube(visual_handoff_cube)
            if distance is None or distance > handoff_distance:
                continue

            self.get_logger().info(
                f'Visual handoff reached for {visual_handoff_cube}: '
                f'base-to-cube distance={distance:.3f} m <= '
                f'{handoff_distance:.3f} m; cancelling Nav2 before camera control.'
            )
            cancel_future = goal_handle.cancel_goal_async()
            cancel_timeout = min(5.0, response_timeout)
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=cancel_timeout
            )
            rclpy.spin_until_future_complete(
                self, result_future, timeout_sec=cancel_timeout
            )
            for _ in range(3):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.1)
            if not cancel_future.done() or not result_future.done():
                raise RuntimeError(
                    'NAV_STATE_UNKNOWN: visual handoff cancellation was not confirmed'
                )
            self.get_logger().info(
                f'Nav2 control released at the {visual_handoff_cube} visual handoff.'
            )
            return True

        if not result_future.done():
            self.get_logger().error(f'Navigation to {label} timed out; cancelling.')
            cancel_future = goal_handle.cancel_goal_async()
            cancel_timeout = min(5.0, response_timeout)
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=cancel_timeout
            )
            rclpy.spin_until_future_complete(
                self, result_future, timeout_sec=cancel_timeout
            )
            self._stop_base()
            if not cancel_future.done():
                self.get_logger().error(
                    f'Nav2 cancellation for {label} was not acknowledged; '
                    'navigation state is uncertain.'
                )
            if not result_future.done():
                raise RuntimeError(
                    'NAV_STATE_UNKNOWN: cancellation was not confirmed'
                )
        wrapped_result = result_future.result()
        success = (
            wrapped_result is not None
            and wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        )
        if not success:
            status = None if wrapped_result is None else wrapped_result.status
            self.get_logger().error(f'Navigation to {label} failed; status={status}.')
        return success

    def _navigate_to(
        self,
        label: str,
        values: Sequence[float],
        visual_handoff_cube: str | None = None,
        completion_radius_m: float | None = None,
    ) -> bool:
        attempts = max(1, int(self.get_parameter('navigation_retry_count').value) + 1)
        for attempt in range(1, attempts + 1):
            if self._navigate_once(
                label, values, visual_handoff_cube, completion_radius_m
            ):
                self.get_logger().info(f'Navigation stage complete: {label}.')
                time.sleep(float(self.get_parameter('settle_sec').value))
                return True
            if attempt < attempts:
                self.get_logger().warning(
                    f'Retrying {label} navigation ({attempt}/{attempts - 1}) '
                    'after clearing costmaps.'
                )
                if not self._clear_costmaps():
                    return False
        return False

    def _navigate_to_pickup(self) -> bool:
        empty_radius = float(self.get_parameter('empty_robot_radius_m').value)
        empty_inflation = float(
            self.get_parameter('empty_inflation_radius_m').value
        )
        if not self._set_costmap_geometry(
            empty_radius, empty_inflation, 'empty pickup'
        ):
            return False
        if not self._clear_costmaps():
            return False
        cube, values = self._pickup_goal()
        use_handoff = bool(self.get_parameter('enable_visual_handoff').value)
        if use_handoff and not bool(
            self.get_parameter('enable_visual_alignment').value
        ):
            self.get_logger().error(
                'Visual handoff requires enable_visual_alignment=true.'
            )
            return False
        handoff_cube = cube if use_handoff else None
        label = f'{cube} visual handoff' if use_handoff else f'{cube} pre-grasp'
        if self._navigate_to(label, values, handoff_cube):
            return True
        if not bool(self.get_parameter('use_backup_pickup_pose').value):
            return False
        backup = list(values)
        offset = float(self.get_parameter('pickup_backup_offset_m').value)
        backup[0] -= math.cos(backup[2]) * offset
        backup[1] -= math.sin(backup[2]) * offset
        self.get_logger().warning('Primary pickup pose failed; trying backup approach.')
        return self._navigate_to(
            f'{cube} backup visual handoff', backup, handoff_cube
        )

    def _navigate_to_warehouse(self) -> bool:
        # The folded gripper and attached cube sweep about 0.34 m from the base
        # centre. Enlarge the active Nav2 footprint before asking for a route;
        # otherwise a path with 0.20 m wall clearance can physically strike the
        # payload, tip the robot, and launch it outside the occupancy map.
        loaded_radius = float(self.get_parameter('loaded_robot_radius_m').value)
        loaded_inflation = float(
            self.get_parameter('loaded_inflation_radius_m').value
        )
        if not self._set_costmap_geometry(
            loaded_radius, loaded_inflation, 'loaded transport'
        ):
            return False
        if not self._clear_costmaps():
            return False
        warehouse, values = self._warehouse_goal()
        cube = str(self.get_parameter('cube_name').value).strip()
        if cube == 'blue_cube_1' and warehouse == 'B':
            for index in (1, 2):
                parameter = f'blue_cube_1_warehouse_b_via_{index}'
                waypoint = self._double_list(parameter, 3)
                if not self._navigate_to(
                    f'blue_cube_1 to B clearance waypoint {index}',
                    waypoint,
                    completion_radius_m=0.35,
                ):
                    return False
                if not self._clear_costmaps():
                    return False
        return self._navigate_to(f'warehouse {warehouse}', values)

    def _visual_align(self) -> bool:
        """Servo the base using the colour contour, then hand control back.

        Nav2 publishes zero velocity through the smoother after a goal, so the
        servo writes to the smoother's *input* topic (``cmd_vel_nav``). The
        smoother then owns the single ``/cmd_vel`` output and the differential
        drive controller never sees two competing publishers.
        """
        if not bool(self.get_parameter('enable_visual_alignment').value):
            self.get_logger().warning('Visual fine alignment is disabled.')
            return True
        self._visual_range_gate_active = True
        try:
            return self._visual_align_active()
        finally:
            self._visual_range_gate_active = False
            # Several zero setpoints flush the servo command out of the
            # smoother's queue before the next Nav2 goal starts.
            for _ in range(3):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.1)

    def _visual_align_active(self) -> bool:
        cube = str(self.get_parameter('cube_name').value).lower()
        self._target_colour = 'blue' if cube.startswith('blue') else 'red'
        self._visual_detection = None
        self._publish_vision_state(f'SEARCHING:{self._target_colour}')
        deadline = time.monotonic() + float(
            self.get_parameter('visual_timeout_sec').value
        )
        stable = 0
        required = max(1, int(self.get_parameter('visual_stable_frames').value))
        approach_row = float(self.get_parameter('visual_approach_row').value)
        row_tolerance = float(self.get_parameter('visual_row_tolerance').value)
        centre_tolerance = float(
            self.get_parameter('visual_center_tolerance').value
        )
        stale_limit = float(
            self.get_parameter('visual_detection_stale_sec').value
        )
        linear_gain = float(self.get_parameter('visual_linear_gain').value)
        angular_gain = float(self.get_parameter('visual_angular_gain').value)
        max_linear = float(self.get_parameter('visual_max_linear_speed').value)
        max_angular = float(self.get_parameter('visual_max_angular_speed').value)
        # Phase 1: search. Turn slowly until the colour blob is in view, so a
        # coarse parking error does not abort the whole pick.
        search_rate = float(self.get_parameter('visual_search_rate').value)
        search_deadline = time.monotonic() + float(
            self.get_parameter('visual_search_timeout_sec').value
        )
        search_command = Twist()
        search_command.angular.z = search_rate
        while rclpy.ok() and time.monotonic() < search_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            detection = self._visual_detection
            if (
                detection is not None
                and time.monotonic() - self._visual_detection_time <= stale_limit
            ):
                break
            self._fine_cmd_pub.publish(search_command)
            self._publish_vision_state(f'SEARCHING:{self._target_colour}')
        self._stop_base()

        # Phase 2: centre and approach.
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            detection = self._visual_detection
            if (
                detection is None
                or time.monotonic() - self._visual_detection_time > stale_limit
            ):
                stable = 0
                self._stop_base()
                continue
            centre_x, width, area, image_width, bottom_row = detection
            centre_error = (centre_x - image_width / 2.0) / (image_width / 2.0)
            # Bottom row grows as the robot approaches: it is the range error.
            row_error = (approach_row - bottom_row) / approach_row
            centred = abs(centre_error) <= centre_tolerance
            at_range = abs(bottom_row - approach_row) <= row_tolerance
            if centred and at_range:
                stable += 1
                self._stop_base()
                if stable >= required:
                    self.get_logger().info(
                        f'Visual alignment complete: centre_error={centre_error:.3f}, '
                        f'bottom_row={bottom_row:.0f}.'
                    )
                    return self._creep_to_grasp(bottom_row)
                continue

            stable = 0
            command = Twist()
            command.linear.x = max(
                -0.5 * max_linear, min(max_linear, linear_gain * row_error)
            )
            command.angular.z = max(
                -max_angular, min(max_angular, -angular_gain * centre_error)
            )
            self._fine_cmd_pub.publish(command)
            self._publish_vision_state(
                f'ALIGNING:{self._target_colour}:x={centre_error:.3f}:'
                f'bottom={bottom_row:.0f}'
            )

        self._stop_base()
        self._publish_vision_state(f'FAILED:{self._target_colour}')
        self.get_logger().error('Visual fine alignment timed out; grasp is inhibited.')
        return False

    def _creep_to_grasp(self, bottom_row: float) -> bool:
        """Close the last centimetres the level camera cannot measure.

        Below roughly 0.20 m the nearest cube's floor contact leaves the image,
        so the range signal saturates before the gripper is over the cube. The
        remaining distance is computed from the last measured row instead of a
        fixed guess, so any stop inside the band lands link6 on the cube.
        """
        k = float(self.get_parameter('visual_row_distance_k').value)
        offset = float(self.get_parameter('visual_row_distance_offset_m').value)
        target = float(self.get_parameter('visual_target_standoff_m').value)
        standoff = offset + k / max(1.0, bottom_row - 360.0)
        distance = max(
            0.0,
            standoff - target
            + float(self.get_parameter('visual_final_creep_m').value),
        )
        speed = float(self.get_parameter('visual_creep_speed').value)
        self.get_logger().info(
            f'Measured standoff {standoff:.3f} m at row {bottom_row:.0f}; '
            f'creeping {distance:.3f} m to the {target:.3f} m grasp standoff.'
        )
        if distance <= 0.0 or speed <= 0.0:
            self._publish_vision_state(f'ALIGNED:{self._target_colour}')
            return True
        command = Twist()
        command.linear.x = speed
        end = time.monotonic() + distance / speed
        while rclpy.ok() and time.monotonic() < end:
            self._fine_cmd_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._stop_base()
        self.get_logger().info(
            f'Final approach creep: {distance:.3f} m at {speed:.3f} m/s.'
        )
        self._publish_vision_state(f'ALIGNED:{self._target_colour}')
        return True

    def _settle_after_release(self) -> bool:
        """Let the released cube come to rest before the arm retreats."""
        time.sleep(float(self.get_parameter('placement_settle_sec').value))
        return True

    def _stabilize_released_cube(self) -> bool:
        """Zero the residual release velocity Gazebo leaves in the cube."""
        if not bool(self.get_parameter('stabilize_release').value):
            return True
        pose = self._link_pose(
            str(self.get_parameter('cube_name').value),
            str(self.get_parameter('cube_link').value),
        )
        if pose is None:
            return True
        request = SetEntityState.Request()
        request.state.name = str(self.get_parameter('cube_name').value)
        request.state.pose = pose
        request.state.twist = Twist()
        request.state.reference_frame = 'world'
        future = self._set_state_client.call_async(request)
        timeout = float(self.get_parameter('service_timeout_sec').value)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.exception() is not None:
            self.get_logger().warning('Could not zero the released cube velocity.')
            return True
        self.get_logger().info('Release stabilised: cube velocity zeroed.')
        return True

    def _verify_placement(self) -> bool:
        """Confirm the released cube actually rests inside the target zone."""
        if not bool(self.get_parameter('verify_placement').value):
            return True
        # The arm is already home, so nothing will touch the cube again: clear
        # the residual velocity before measuring the final resting pose.
        self._stabilize_released_cube()
        time.sleep(0.5)
        warehouse, _ = self._warehouse_goal()
        centre = self._double_list(f'zone_{warehouse.lower()}_centre', 2)
        centre_x, centre_y = centre
        cube_pose = self._link_pose(
            str(self.get_parameter('cube_name').value),
            str(self.get_parameter('cube_link').value),
        )
        if cube_pose is None:
            return False
        offset_x = cube_pose.position.x - centre_x
        offset_y = cube_pose.position.y - centre_y
        half_length = float(self.get_parameter('zone_half_length_m').value)
        half_width = float(self.get_parameter('zone_half_width_m').value)
        tolerance = float(self.get_parameter('placement_tolerance_m').value)
        inside = (
            abs(offset_x) <= half_length - tolerance
            and abs(offset_y) <= half_width - tolerance
        )
        message = (
            f'Placement check: cube at ({cube_pose.position.x:.3f}, '
            f'{cube_pose.position.y:.3f}), zone {warehouse} centre at '
            f'({centre_x:.3f}, {centre_y:.3f}), offset '
            f'({offset_x:+.3f}, {offset_y:+.3f}) m, inside limits '
            f'[{half_length:.2f} x {half_width:.2f}] m.'
        )
        if inside:
            self.get_logger().info(message)
            return True
        self.get_logger().error(f'Cube missed warehouse {warehouse}. {message}')
        return False

    def _link_pose(self, model: str, link: str):
        """Return a Gazebo link pose, refreshing cached link states first."""
        self._latest_link_states = None
        timeout = float(self.get_parameter('proximity_timeout_sec').value)
        if not self._wait_until(
            lambda: self._latest_link_states is not None,
            timeout,
            f'{model}::{link}',
        ):
            return None
        try:
            index = list(self._latest_link_states.name).index(f'{model}::{link}')
        except ValueError:
            self.get_logger().error(f'Missing Gazebo link {model}::{link}.')
            return None
        return self._latest_link_states.pose[index]

    def _link_request_values(self) -> tuple[str, str, str, str]:
        return (
            str(self.get_parameter('robot_model_name').value),
            str(self.get_parameter('robot_attach_link').value),
            str(self.get_parameter('cube_name').value),
            str(self.get_parameter('cube_link').value),
        )

    def _link_distance(self) -> float | None:
        # Discard cached data so verification observes a state published after
        # the preceding attach, arm motion, or navigation operation.
        self._latest_link_states = None
        timeout = float(self.get_parameter('proximity_timeout_sec').value)
        if not self._wait_until(
            lambda: self._latest_link_states is not None,
            timeout,
            '/gazebo/link_states',
        ):
            return None

        model1, link1, model2, link2 = self._link_request_values()
        robot_link_name = f'{model1}::{link1}'
        cube_link_name = f'{model2}::{link2}'
        names = self._latest_link_states.name
        try:
            robot_pose = self._latest_link_states.pose[names.index(robot_link_name)]
            cube_pose = self._latest_link_states.pose[names.index(cube_link_name)]
        except ValueError:
            self.get_logger().error(
                'Cannot check attachment distance; missing Gazebo links: '
                f'{robot_link_name!r} or {cube_link_name!r}.'
            )
            return None

        dx = robot_pose.position.x - cube_pose.position.x
        dy = robot_pose.position.y - cube_pose.position.y
        dz = robot_pose.position.z - cube_pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _check_attach_proximity(self) -> bool:
        if not bool(self.get_parameter('require_proximity_check').value):
            self.get_logger().warning(
                'Attachment proximity check is disabled; remote objects can '
                'be attached by the Gazebo plugin.'
            )
            return True

        distance = self._link_distance()
        if distance is None:
            return False
        maximum = float(self.get_parameter('max_attach_distance_m').value)
        self.get_logger().info(
            f'Pre-attach distance: {distance:.3f} m (limit {maximum:.3f} m).'
        )
        if distance > maximum:
            self.get_logger().error(
                'Cube is too far from link6; refusing a non-physical remote '
                'attachment. Tune the grasp joints or move the base closer.'
            )
            return False
        return True

    def _verify_attachment(self) -> bool:
        distance = self._link_distance()
        if distance is None:
            return False
        maximum = float(self.get_parameter('max_attached_distance_m').value)
        self.get_logger().info(
            f'Attached-cube distance: {distance:.3f} m '
            f'(limit {maximum:.3f} m).'
        )
        if distance > maximum:
            self.get_logger().error(
                'Attachment verification failed: cube is no longer following '
                'the robot end link.'
            )
            return False
        return True

    def _attach(self) -> bool:
        if not self._check_attach_proximity():
            return False
        request = AttachLink.Request()
        (
            request.model1_name,
            request.link1_name,
            request.model2_name,
            request.link2_name,
        ) = self._link_request_values()
        return self._call_link_service('ATTACH', self._attach_client, request)

    def _detach(self) -> bool:
        request = DetachLink.Request()
        (
            request.model1_name,
            request.link1_name,
            request.model2_name,
            request.link2_name,
        ) = self._link_request_values()
        return self._call_link_service('DETACH', self._detach_client, request)

    def _call_link_service(self, label: str, client, request) -> bool:
        self.get_logger().info(
            f'{label}: {request.model1_name}::{request.link1_name} <-> '
            f'{request.model2_name}::{request.link2_name}'
        )
        future = client.call_async(request)
        timeout = float(self.get_parameter('service_timeout_sec').value)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            # ROS service requests cannot be cancelled. The request may still
            # complete later, so stop the workflow in an explicit unknown state
            # instead of hanging forever or starting unsafe recovery motion.
            raise RuntimeError(f'{label}_STATE_UNKNOWN')
        try:
            response = future.result()
        except Exception as exc:  # rclpy service failures surface via Future.
            self.get_logger().error(f'{label}: service call failed: {exc}')
            return False
        if response is None or not response.success:
            message = '<no response>' if response is None else response.message
            self.get_logger().error(f'{label}: rejected: {message}')
            return False
        self.get_logger().info(f'{label}: {response.message}')
        return True

    def run_once(self) -> bool:
        if not self._wait_for_dependencies():
            self._publish_executor_error('DEPENDENCY_TIMEOUT')
            return False

        cube_name = str(self.get_parameter('cube_name').value)
        warehouse, _ = self._warehouse_goal()
        self.get_logger().info(
            f'Starting fixed grasp of {cube_name}, then delivery to '
            f'warehouse {warehouse}.'
        )

        open_position = float(self.get_parameter('gripper_open').value)
        closed_position = float(self.get_parameter('gripper_closed').value)

        # Each operation carries the externally visible competition state.  The
        # supervisor can therefore report real progress instead of inferring it
        # from process lifetime or log text.
        steps = [
            ('PRE_GRASP', 'arm transport', lambda: self._move_arm('Arm transport', 'transport_joints')),
            ('NAV_TO_OBJECT', 'navigate to pickup', self._navigate_to_pickup),
            ('DETECTING_OBJECT', 'visual fine alignment', self._visual_align),
            ('PRE_GRASP', 'open gripper', lambda: self._move_gripper('Open gripper', open_position)),
            (
                'PRE_GRASP', 'pre-grasp',
                lambda: self._move_arm('Move to pre-grasp', 'pre_grasp_joints'),
            ),
            ('GRASPING', 'grasp', lambda: self._move_arm('Move to grasp', 'grasp_joints')),
            # Attach before closing: the 27 mm finger grip would otherwise shove
            # the free cube away from link6 before the joint is created.
            ('GRASPING', 'attach cube', self._attach),
            (
                'GRASPING', 'close gripper',
                lambda: self._move_gripper('Close gripper', closed_position),
            ),
            ('VERIFY_GRASP', 'lift', lambda: self._move_arm('Lift cube', 'lift_joints')),
            ('VERIFY_GRASP', 'verify grasp', self._verify_attachment),
            (
                'VERIFY_GRASP', 'transport posture',
                lambda: self._move_arm(
                    'Fold arm for base transport', 'transport_joints'
                ),
            ),
            ('VERIFY_GRASP', 'verify transport hold', self._verify_attachment),
            ('NAV_TO_ZONE', 'navigate to warehouse', self._navigate_to_warehouse),
            ('VERIFY_GRASP', 'verify carried cube', self._verify_attachment),
            (
                'PLACING', 'pre-place',
                lambda: self._move_arm('Move to pre-place', 'pre_place_joints'),
            ),
            ('PLACING', 'place', lambda: self._move_arm('Lower to place', 'place_joints')),
            # Open while the Gazebo fixed joint still supports the cube, then
            # detach after the fingers no longer pinch or eject it.
            ('PLACING', 'open gripper', lambda: self._move_gripper('Open gripper', open_position)),
            ('PLACING', 'detach cube', self._detach),
            ('VERIFY_PLACE', 'settle after release', self._settle_after_release),
            (
                'VERIFY_PLACE', 'retreat',
                lambda: self._move_arm('Retreat from place', 'pre_place_joints'),
            ),
            ('VERIFY_PLACE', 'return home', lambda: self._move_arm('Return home', 'home_joints')),
            ('VERIFY_PLACE', 'verify placement', self._verify_placement),
        ]

        attached = False
        released = False
        unsafe_state = False
        for state, name, operation in steps:
            self._publish_executor_state(state)
            try:
                succeeded = operation()
            except Exception as exc:
                self.get_logger().error(f'{name} raised an exception: {exc}')
                if 'STATE_UNKNOWN' in str(exc):
                    # A late action/service completion may still move the robot
                    # or change attachment. Forbid recovery motion and retry.
                    unsafe_state = True
                self._publish_executor_error(f'{state}:{name}:EXCEPTION:{exc}')
                succeeded = False
            if succeeded:
                attached = name == 'attach cube' or (
                    attached and name != 'detach cube'
                )
                released = released or name == 'detach cube'
                continue

            self._stop_base()
            error = f'{state}:{name}:FAILED'
            if unsafe_state:
                error += ':STATE_UNKNOWN'
            elif attached:
                error += ':CUBE_ATTACHED'
            elif released:
                error += ':CUBE_RELEASED'
            self._publish_executor_error(error)
            self.get_logger().error(f'Pick-and-place stopped at step: {name}.')
            if attached or unsafe_state:
                self.get_logger().error(
                    'Robot or attachment state is unsafe/unknown. Automatic '
                    'recovery and retry are disabled; stop motion and inspect '
                    'the robot before manual intervention.'
                )
            else:
                # Before attachment it is safe to make one bounded attempt to
                # leave the arm in its collision-safe transport posture.
                self._move_arm('Failure recovery transport', 'transport_joints')
                self._move_gripper('Failure recovery open gripper', open_position)
            return False

        self.get_logger().info(
            f'DELIVERY COMPLETED: {cube_name} placed in warehouse {warehouse}.'
        )
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixedJointPickPlace()
    exit_code = 1
    try:
        exit_code = 0 if node.run_once() else 1
    except (KeyboardInterrupt, ValueError) as exc:
        node.get_logger().error(f'Fixed pick-and-place aborted: {exc}')
    except Exception as exc:  # Keep an actionable terminal result for launch.
        node.get_logger().error(f'Unexpected fixed pick-and-place error: {exc}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

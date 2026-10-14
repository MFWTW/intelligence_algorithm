#!/usr/bin/env python3

"""Navigate to a cube, visually align, grasp, carry, and place it.

The coarse pickup and warehouse poses are loaded from YAML.  After Nav2 reaches
an object's pre-grasp pose, a bounded HSV image servo centers the requested
colour and corrects the final standoff before the fixed-joint grasp sequence.
"""

import contextlib
import json
import math
import os
import sys
import time
from collections import deque
from typing import Iterable, Sequence

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from cv_bridge import CvBridge
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseStamped, Twist
from linkattacher_msgs.srv import AttachLink, DetachLink
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import (
    Parameter,
    ParameterDescriptor,
    ParameterType,
    ParameterValue,
)
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import obstacle_yield


ARM_JOINTS = [
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
]
GRIPPER_JOINTS = ['finger_joint1']

# Speed limit used to hold the base still for a moving obstacle. Deliberately
# not 0.0: that is Nav2's "no limit" sentinel (see _tick_yield). 1 mm/s is far
# below anything the base can track and still performs a real clamp, because RPP
# applies the limit after its own minimum-speed floors.
YIELD_STOP_SPEED_LIMIT = 0.001

# A sweep stamp this far from the node's own clock is not staleness, it is two
# different time bases. 10^6 s is ~11.6 days: far beyond any real pause.
CLOCK_MISMATCH_SEC = 1.0e6


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
        # The camera servo needs a usable standoff: its range estimate is
        # row = 360 + 66.0 / (distance - 0.015), so a cube that is already almost
        # touching the base is outside the calibrated range. Measured
        # 2026-10-07: red_cube_2's pickup pose doubles as its clearance waypoint
        # and sits 0.58 m from the cube, so the 0.35 m completion radius let the
        # base arrive as close as 0.23 m and the servo then timed out - round39
        # handed off at 0.247 m, round43 at 0.260 m, while every other cube hands
        # off at ~1.18 m. Below the minimum the base is backed off first.
        self.declare_parameter('visual_handoff_min_distance_m', 0.85)
        self.declare_parameter('visual_handoff_retreat_target_m', 1.10)
        # 8 s x 0.06 m/s was not enough: measured 2026-10-07 (round45) the base
        # only opened 0.26 -> 0.473 m and the servo timed out again, because the
        # base moves far slower than commanded here (~0.026 m/s effective). The
        # budget below allows ~1.8 m of travel, and the loop exits the moment the
        # target is reached, so the headroom costs nothing on a healthy base.
        self.declare_parameter('visual_handoff_retreat_timeout_sec', 15.0)
        self.declare_parameter('visual_handoff_retreat_speed_mps', 0.12)

        # Saved runtime state places A/B/C at x=5, y=-11/-12/-13 with
        # 1.0 x 0.5 m plates. The goals below are pre-compensated by the
        # measured docking shortfall and carried-cube offset.
        # Every delivery used to aim at the SAME spot inside the zone, so the
        # 2nd/3rd/4th cube landed on top of the ones already there and slid off.
        # Measured 2026-10-12 (seed 995410, x=4 red -> A): delivery 1 placed at
        # (6.883,-2.519), delivery 3 was released at the nominal (6.886,-2.483)
        # - on top of delivery 1 - and ended at (6.334,-2.446), 0.552 m off and
        # outside the +/-0.485 acceptance box. The dock creep itself was fine
        # (0.020 m); the miss was stacking, not docking.
        # These offsets are applied to the dock pose by how many cubes have
        # already been delivered to that zone, and are chosen to stay inside the
        # 1.0 x 0.5 m plate: 0.12 m in y (box half-width 0.235) and 0.24 m in x
        # (box half-length 0.485). Cubes are 0.03 m, so 0.12 m apart never stacks
        # (0.09 m of clearance between neighbours).
        #
        # 2026-10-13: the y slots used to be +/-0.16, which left only 0.075 m of
        # the 0.235 acceptance half-width for the placement error, and the error
        # is a systematic ~0.02-0.08 m in -y (the cube comes to rest slightly
        # beyond the dock point). Measured on four slot-3 deliveries:
        #   y offset -0.182 / -0.217 / -0.231 / -0.244  ->  margins
        #   +0.053 / +0.018 / +0.004 / -0.009 m, i.e. the last one missed the
        #   zone by 9 mm and failed the round (VERIFY_PLACE:...CUBE_RELEASED).
        # Slots 1/2/4, which sit at 0.00/+0.16, always had 0.17-0.23 m of margin
        # and never failed. Pulling the y slots in to +/-0.12 raises the budget
        # to 0.115 m, which the worst measured error (0.084 m) still fits.
        self._zone_deliveries = {}
        self.declare_parameter(
            'zone_slot_offsets',
            [0.0, 0.0, 0.0, 0.12, 0.0, -0.12, -0.24, 0.0],
        )
        self.declare_parameter('warehouse_a_pose', [4.792, -10.74, 0.0])
        self.declare_parameter('warehouse_b_pose', [4.792, -11.74, 0.0])
        self.declare_parameter('warehouse_c_pose', [4.792, -12.74, 0.0])
        # Aligned final approach per zone (used by _navigate_to_warehouse).
        # Round16 measured 56 s of the five loaded legs - 28% of their total -
        # spent crawling and rotating inside the last 0.25 m, because the global
        # plan ended at an arbitrary heading (155 deg off on red_cube_4, 43 deg on
        # blue_cube_1) and DWB must rotate in place before it may report success.
        # Each point sits 1.5 m straight in front of its plate, facing the plate
        # yaw; all three are free space in room_from_world.pgm, as is the straight
        # run into the goal. Set false to go back to the direct route.
        self.declare_parameter('use_aligned_warehouse_approach', True)
        # Send the whole detour (every <cube>_warehouse_<zone>_via_<n> point
        # plus the leg's own target) as ONE NavigateThroughPoses goal instead
        # of one NavigateToPose goal per waypoint. Nav2 then plans and follows
        # a single path through the polyline: the base does not stop at each
        # waypoint, the route on screen is the configured one, and the yield
        # layer resumes one goal rather than three. Set false to go back to the
        # per-waypoint walk (kept as the fallback and for A/B rounds).
        self.declare_parameter('use_continuous_detour', True)
        # Experimental: retain the top-level Nav2 action during a mover hold.
        # Requires the guarded Nav2 launch, dual-axis gate, and drive watchdog.
        # The production profile leaves this off until an integration round passes.
        self.declare_parameter('retain_nav_goal_on_yield', False)
        # How close the base must get to a detour waypoint before the next one
        # becomes the no-progress reference (see _chain_progress_target).
        self.declare_parameter('chain_switch_radius_m', 0.75)
        self.declare_parameter(
            'warehouse_a_approach_pose', [3.142, -11.016, 0.0]
        )
        self.declare_parameter(
            'warehouse_b_approach_pose', [3.142, -12.016, 0.0]
        )
        self.declare_parameter(
            'warehouse_c_approach_pose', [3.142, -13.016, 0.0]
        )
        # blue_cube_1 -> B otherwise follows the long global path into a DWB
        # local minimum beside Wall_57. These two clearance poses force the
        # loaded robot around the open north end before cross-map transport.
        # DWB settles into a zero-progress local minimum when a route demands a
        # large heading correction right beside a concave corner (measured at
        # Wall_57 and at the doorway next to Wall_115). Instead of tuning the
        # controller per case, every cube may declare up to two clearance
        # waypoints that shape its approach; empty means "go direct". The
        # descriptor must allow dynamic typing: an empty list default is
        # inferred as BYTE_ARRAY and then rejects the DOUBLE_ARRAY from YAML.
        # blue_cube_1 sits behind Wall_57, whose only opening is its north end.
        # red_cube_2 is north of Wall_115 and is only reachable through the
        # 2.09 m doorway beside it (x ~ 9.7); going direct from warehouse C let
        # the path drift east of the doorway and then demand a ~180 deg turn at
        # the wall end, which DWB answered by twitching in place for minutes.
        pickup_via_defaults = {
            'blue_cube_1': [[-2.3, 0.0, math.pi], [-4.2, 0.0, math.pi]],
            'red_cube_2': [[9.65, -7.70, math.pi / 2.0], []],
        }
        waypoint_descriptor = ParameterDescriptor(dynamic_typing=True)
        for cube in pickup_defaults:
            first, second = pickup_via_defaults.get(cube, [[], []])
            self.declare_parameter(
                f'{cube}_pickup_via_1', first, waypoint_descriptor
            )
            self.declare_parameter(
                f'{cube}_pickup_via_2', second, waypoint_descriptor
            )
        self.declare_parameter(
            'blue_cube_1_warehouse_b_via_1', [-4.2, 0.0, 0.0]
        )
        self.declare_parameter(
            'blue_cube_1_warehouse_b_via_2', [-2.3, 0.0, 0.0]
        )
        # 2026-10-07: the blue cube needs the same north-end detour when its
        # target is C (one plate further south). Driving direct to C cut the
        # corner 1.7 m south of that corridor and wedged the loaded robot where
        # walls sat 0.50 m on both sides against a 0.40 m footprint, so DWB
        # rejected all 419 trajectories and the task aborted. Same validated
        # coordinates as the B detour, declared separately so C can be tuned.
        self.declare_parameter(
            'blue_cube_1_warehouse_c_via_1', [-4.2, 0.0, 0.0]
        )
        self.declare_parameter(
            'blue_cube_1_warehouse_c_via_2', [-2.3, 0.0, 0.0]
        )
        # 2026-10-07: red_cube_1 -> C wedged twice in the same pocket (round30
        # was flung out of the map from it, round32 produced 884 rejected
        # rollouts and died with the cube still attached). West of the wall at
        # x=-3.35 the base is boxed in: the pocket is closed to the north by the
        # wall at y=-4.4, and a flood fill with the loaded 0.35 m footprint
        # cannot reach the goal from inside it at all (the reachable set
        # collapses to the start cell). The gap the planner aims at is simply
        # too narrow to turn in, so the fix is to not go in: leave the pocket
        # southward, run east, and come back up the x=+0.03 corridor. Both
        # straight legs are clear at 0.35 m, verified by flood fill on the
        # static map rather than by eye.
        self.declare_parameter(
            'red_cube_1_warehouse_c_via_1', [-3.275, -6.725, -0.475695]
        )
        self.declare_parameter(
            'red_cube_1_warehouse_c_via_2', [0.025, -4.875, 0.510950]
        )
        # 2026-10-07 (round36): blue_cube_3 -> A is the second leg that cannot be
        # driven direct. It carried the cube correctly (traced z=+0.238 m, so not
        # dragging) yet still produced 383 rejected rollouts and "Controller
        # patience exceeded" on the A aligned approach - exactly where round29
        # died. A flood fill with the loaded 0.35 m footprint reaches A only by
        # looping east: out along y=-5.98 to x=+12.88, north to y=-3.5, then back
        # west. The middle point is kept because the original shortcut from
        # (12.88,-3.88) to (12.48,-3.18) had 11 blocked cells at 0.35 m.
        # Even the new third target cannot safely be cut to in a straight line
        # from via_1 (static occupied-edge clearance dips to 0.318 m).
        self.declare_parameter(
            'blue_cube_3_warehouse_a_via_1', [12.875, -3.875, 0.454302]
        )
        self.declare_parameter(
            'blue_cube_3_warehouse_a_via_2', [12.875, -3.525, 1.570796]
        )
        # Keep the standalone/default executor aligned with the validated YAML.
        # The old target's nominal via_2 chord entered the 0.35 m loaded
        # footprint; two same-task five-cube runs reached this open-area target
        # in ~3 s without the previously repeated 45 s stall.
        self.declare_parameter(
            'blue_cube_3_warehouse_a_via_3', [12.700, -2.800, 2.422763]
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
        # Global costmap radius = the local one + this margin; see
        # _set_costmap_geometry for why the two deliberately differ. 0.25 m of
        # clearance over the 0.35 m dead-zone radius is what makes a run
        # repeatable: with the two radii equal the planned route's tightest point
        # sat exactly at the robot radius, so a few centimetres of drift decided
        # between success and a permanent wedge. Measured on
        # room_from_world.pgm: all 16 cube x zone legs, and every dock, approach
        # and pickup pose, stay reachable up to a 0.75 m radius, so a 0.60 m
        # global radius is well inside what the map can support.
        self.declare_parameter('global_radius_margin_m', 0.25)
        # link6 is a real Gazebo link. grasping_frame is a MoveIt/TF frame and
        # may be collapsed by Gazebo's fixed-joint reduction.
        self.declare_parameter('robot_attach_link', 'link6')
        self.declare_parameter('cube_link', 'link')
        self.declare_parameter('require_system_ready', True)
        self.declare_parameter('ready_timeout_sec', 180.0)
        self.declare_parameter('action_timeout_sec', 30.0)
        self.declare_parameter('navigation_timeout_sec', 120.0)
        # Progress watchdog. A wedged base can sit out the whole
        # navigation_timeout_sec producing nothing at all: measured 2026-10-07
        # (round32) the loaded base thrashed +/-1.9 rad/s for 40 s trying to
        # settle its heading, walked itself into a corner it could not turn out
        # of, and then burned the remaining 260 s with 884 rejected rollouts and
        # 17 aborted spins. Giving up on the attempt after this long without
        # closing the distance hands the leg to the retry path (dead-zone escape
        # + jam break) while there is still budget left for the other cubes.
        # 0 disables the watchdog.
        self.declare_parameter('navigation_stall_timeout_sec', 45.0)
        self.declare_parameter('navigation_stall_progress_m', 0.15)
        # Hard "the base has simply stopped" limit, independent of the goal: if
        # the base does not move this far within this long, the attempt is
        # abandoned. The progress check above can be satisfied by a base that is
        # circling or creeping, and measured 2026-10-07 (round39) a carried base
        # covered 5 cm in 135 s with the wheels almost still (0.02 rad/s) and zero
        # rejected rollouts - it would otherwise sit out the full 300 s timeout
        # three times over. 0 disables it.
        self.declare_parameter('navigation_nomotion_timeout_sec', 60.0)
        self.declare_parameter('navigation_nomotion_radius_m', 0.05)
        # Last-resort guard: a loaded leg can burn its whole wall-clock budget in
        # progress-checker recoveries while the base is already standing on the
        # goal (measured 2026-10-06: cancelled 0.14 m from the warehouse plate).
        # Anything inside this radius counts as reached. Kept well below
        # zone_half_width_m (0.25 m) so the drop still lands on the plate and the
        # placement check can still pass.
        self.declare_parameter('navigation_accept_radius_m', 0.22)
        # Sustained "close enough + aligned" acceptance while the goal is still
        # running. Measured 2026-10-06 (round 2): the base parked at yaw 2.3 deg
        # but 0.19 m short of the plate; both costmaps were free, yet MPPI
        # commanded ~0 m/s for 122 s until the cube budget expired, because the
        # goal sat beside/behind the base and a diff-drive MPPI with
        # vx_min = 0.0 cannot trim that last piece. Accepting it after a short
        # hold hands the accuracy over to verify_placement.
        self.declare_parameter('navigation_accept_yaw_rad', 0.15)
        self.declare_parameter('navigation_accept_hold_sec', 1.0)
        # Final dock creep (2026-10-07): DWB cannot finish the last stretch to a
        # plate. Just outside xy_goal_tolerance the RotateToGoal critic has not
        # taken over yet, so a base that arrives with a heading error circles the
        # goal at max yaw rate instead of stopping - round17 measured a 16.6 m
        # path for a 1.48 m leg and 110-218 s per cube. The sustained-accept gate
        # above cannot latch either: at 1.9 rad/s the yaw error stays inside its
        # +-0.15 rad window for only ~0.16 s, but the gate needs a 1.0 s hold.
        # So release Nav2 a little further out and drive the rest directly.
        self.declare_parameter('enable_dock_creep', True)
        self.declare_parameter('dock_creep_trigger_m', 0.45)
        self.declare_parameter('dock_creep_speed_mps', 0.12)
        self.declare_parameter('dock_creep_forward_gain', 0.8)
        self.declare_parameter('dock_creep_tolerance_m', 0.03)
        self.declare_parameter('dock_creep_yaw_tolerance_rad', 0.06)
        self.declare_parameter('dock_creep_timeout_sec', 14.0)
        # A creep that ends further than this is refused instead of being handed
        # to the old 0.22 m accept gate: 0.165 m of docking error already put the
        # released cube 0.301 m off the plate centre in y (round21 verify_placement
        # failure). A refused leg is retried by _navigate_to with cleared costmaps,
        # which is much cheaper than an UNSAFE_OBJECT_STATE failure at placement.
        self.declare_parameter('dock_creep_accept_m', 0.10)
        # Two re-arming windows instead of one: the C dock can need a turn plus
        # 0.45 m of travel, and a cube-attached leg must not fail on the first
        # timeout (round22 ended the whole task that way).
        self.declare_parameter('dock_creep_attempts', 2)
        # Inside this range the creep switches from "point at the goal" to
        # "hold the plate yaw"; the arm's drop point is calibrated for the latter.
        self.declare_parameter('dock_creep_align_distance_m', 0.12)
        self.declare_parameter('navigation_retry_count', 2)
        # Tight-passage speed caps: flat [x, y, radius_m, speed_mps, ...]. Empty
        # means the executor never publishes a speed limit at all and the
        # controller keeps running with its own desired_linear_vel.
        # Default is the single-element sentinel [0.0], meaning "no zones".
        # Measured 2026-10-11: a Type enum default raises
        # ParameterUninitializedException when no parameter file supplies the
        # name (the production configuration), while a bare [] default raises
        # InvalidParameterTypeException when a file DOES supply a double array.
        # A non-empty, correctly typed default is the only form that works in
        # both cases; a list shorter than one group of four means "off".
        self.declare_parameter('slow_zones', [0.0])
        # How early the tight-passage cap engages, measured from the zone
        # boundary. 0.8 m covers 1.5 m/s of braking distance (0.45 m) plus the
        # sensing/control latency (0.45 m) at the profile's cruise speed.
        self.declare_parameter('slow_zone_brake_ahead_m', 0.8)
        # Hard-wedge detector. A base wedged by the controller's own collision
        # check cannot move at all, and the only thing that frees it is this
        # executor's escape - so waiting out navigation_stall_timeout_sec (45 s)
        # buys nothing. The signal below is not a proxy: the controller emits
        # "detected collision ahead" while refusing to move, and it is absent
        # (measured: exactly 0 messages) during every legitimate stationary
        # phase, which all happen at a dock AFTER Nav2 hands control back.
        # Disabled unless wedge_collision_window_sec is > 0.
        self.declare_parameter('wedge_collision_window_sec', 0.0)
        self.declare_parameter('wedge_collision_min_messages', 5)
        self.declare_parameter('wedge_collision_radius_m', 0.05)
        self.declare_parameter('wedge_collision_timeout_sec', 6.0)
        # Dead-zone escape. Nav2 marks every cell within robot_radius of an
        # obstacle as inscribed, and DWB's BaseObstacle rejects a rollout as soon
        # as the footprint touches one - including the rollout that *is* the
        # current pose. So once the base centre gets closer than robot_radius to a
        # wall it can neither drive nor even rotate in place, the Spin recovery
        # aborts with "Collision Ahead", and the leg can only die on its timeout.
        # Measured 2026-10-07 (round32, carried leg to C): the base thrashed
        # +/-1.9 rad/s for 40 s while aligning, walked itself from 0.69 m to
        # 0.37 m off a wall, then produced 884 rejected rollouts and 17 failed
        # spins before UNSAFE_OBJECT_STATE. Shrinking the radius for the retries
        # makes the current pose legal again so DWB can plan a way out; the
        # nominal geometry is restored before the next stage runs.
        # 0.24, not 0.28: measured against the map for the round32 wedge pose
        # (-3.64, -5.44), which stays BLOCKED all the way down to 0.30 and only
        # frees at <=0.26. The base half-diagonal is 0.226 m, so 0.24 still
        # covers the chassis with margin; only the carried cube/arm (which reach
        # ~0.32 m) may overlap geometry, and the escape is deliberately slow
        # (0.08 m/s) and short (0.12 m). A brief graze is the accepted cost of
        # leaving a pose where every rollout - including rotating in place - is
        # illegal, so the leg can otherwise only die on its timeout.
        self.declare_parameter('dead_zone_escape_enabled', True)
        self.declare_parameter('dead_zone_escape_radius_m', 0.24)
        # Never plan a carried cube through a gap narrower than its own envelope:
        # 0.317 m is the tucked-elbow reach measured on the real robot.
        self.declare_parameter('loaded_escape_min_radius_m', 0.32)
        # The shove has to actually restore legal clearance at the *nominal*
        # radius, so it is longer than the old 0.12 m nudge: a base parked 0.32 m
        # from a wall needs roughly 0.1-0.2 m of travel before a 0.35 m footprint
        # is legal again, and the loaded case relies on this motion alone.
        self.declare_parameter('escape_distance_m', 0.35)
        self.declare_parameter('escape_speed_mps', 0.08)
        # Fast DDS gives an action goal *response* only ~100 ms to be delivered,
        # so a transient transport stall makes the controller's answer vanish
        # even though it accepted the goal ("Failed to send goal response
        # (timeout)" appears in the controller's log). That used to abort the
        # whole task with an unknown motion state. Every retry re-sends exactly
        # the same target, so letting the possibly-running motion settle first
        # makes the retry idempotent and safe.
        self.declare_parameter('action_retry_count', 2)
        self.declare_parameter('service_timeout_sec', 10.0)
        # 2026-10-07: gzserver 偶发死锁（link attacher 在 ROS 回调里改物理场景）会让
        # ATTACH/DETACH 服务迟到；超时后不要立刻报废整轮，先重试一次，并用 Gazebo
        # link states 判定真实状态（详见 _call_link_service / _confirm_detached）。
        self.declare_parameter('link_service_attempts', 2)
        # Discovering every Nav2 service can take longer than a single service
        # call budget right after startup, especially on loaded WSL hosts; the
        # dependency gate gets its own, more generous timeout.
        self.declare_parameter('dependency_timeout_sec', 60.0)
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
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('fine_cmd_vel_topic', '/cmd_vel_nav')

        # Predictive yielding for moving obstacles (see obstacle_yield.py).
        # Off by default until it has been measured over a full round: with it
        # disabled the executor behaves exactly as it did before, which is the
        # only way a change to the navigation loop can be judged.
        self.declare_parameter('enable_obstacle_yield', False)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('yield_update_period_sec', 0.1)
        # Keep in step with obstacle_yield.YieldConfig defaults; a value that is
        # not declared here cannot be tuned from YAML.
        self.declare_parameter('yield_half_width_m', 0.24)
        self.declare_parameter('yield_obstacle_half_width_m', 0.11)
        self.declare_parameter('yield_detection_range_m', 2.5)
        self.declare_parameter('yield_track_range_m', 4.0)
        self.declare_parameter('yield_cap_range_m', 1.8)
        self.declare_parameter('yield_stop_range_m', 1.2)
        self.declare_parameter('yield_cap_speed_mps', 0.8)
        self.declare_parameter('yield_resume_speed_mps', 0.6)
        self.declare_parameter('yield_min_mover_speed_mps', 0.04)
        # The other end of the mover-speed window: a track faster than this is
        # association noise, not a patrol box (measured phantoms: 0.86-1.52 m/s
        # against real boxes at 0.16-0.30 m/s).
        self.declare_parameter('yield_max_mover_speed_mps', 0.6)
        # How much time must be left between "the base is past" and "the box
        # enters the corridor": the sensing/control latency of this stack.
        self.declare_parameter('yield_clear_margin_sec', 0.3)
        self.declare_parameter('yield_wait_budget_sec', 20.0)
        self.declare_parameter('yield_arc_offset_m', 0.55)
        self.declare_parameter('yield_arc_apex_m', 0.8)
        self.declare_parameter('yield_arc_goal_radius_m', 0.35)
        # The patrol boxes' centre lines, published latched by moving_obstacles.
        # Without them the policy cannot tell "stop beside the box" from "stop
        # inside the box's lane", and a base parked in a lane is trapped: the box
        # returns every few seconds (measured 2026-10-13, 18 s frozen).
        self.declare_parameter('mover_patrols_topic', '/movers/patrols')
        # Rate limit for the yield diagnostic below; 0.0 disables it, and it is
        # silent unless the yield feature itself is enabled. It only ever logs
        # (no motion), so it is safe to leave on during a round.
        self.declare_parameter('yield_diag_period_sec', 5.0)
        self.declare_parameter('visual_timeout_sec', 45.0)
        self.declare_parameter('visual_detection_stale_sec', 0.5)
        self.declare_parameter('visual_min_contour_area', 800.0)
        # The strict HSV gates are what keep the brick-red walls out, but a shaded
        # cube drops under them and the blob then vanishes for seconds. Measured
        # 2026-10-12 with the cubes in their new positions: vision_align went from
        # 8.8-9.2 s to 28-64 s with repeated "Visual blob lost for 3.1 s;
        # restarting the search". The relaxed tier is only consulted when the
        # strict one finds nothing, and every shape and expected-row gate still
        # applies to it. Set the three scales to 1.0 to disable the fallback.
        self.declare_parameter('visual_relaxed_saturation_scale', 0.65)
        self.declare_parameter('visual_relaxed_value_scale', 0.60)
        self.declare_parameter('visual_relaxed_dominance_scale', 0.80)
        # Deliberately strict pure-red gate: the Gazebo cube is nearly pure red,
        # while brick/wood walls are darker and contain much more green/blue.
        self.declare_parameter('visual_red_min_saturation', 200)
        self.declare_parameter('visual_red_min_value', 100)
        self.declare_parameter('visual_red_dominance_ratio', 2.5)
        # Blue needs the same treatment. Measured on this Gazebo world: the
        # blue cube renders as H=120, S=250, V=200 with B/G,B/R > 25, while the
        # shadowed brick/wood walls that used to be picked up as "blue" sit at
        # H=103..114, S=87..105, V=135..155 with B/G,B/R of only 1.2..1.6. The
        # former bare hue window (95..135 with S>=80, V>=45) therefore matched
        # whole wall patches that were larger than the cube itself, which is
        # what made the base servo onto walls instead of the cube.
        self.declare_parameter('visual_blue_hue_min', 110)
        self.declare_parameter('visual_blue_hue_max', 130)
        self.declare_parameter('visual_blue_min_saturation', 170)
        # Shadowed blue faces only lose value (S stays ~250), so this floor is
        # deliberately low; saturation and channel dominance do the rejecting.
        self.declare_parameter('visual_blue_min_value', 70)
        self.declare_parameter('visual_blue_dominance_ratio', 2.0)
        # Gazebo calibration: bottom row = 360 + 66.0 / (standoff - 0.015).
        # Row 680 stops well clear of the 720 px frame edge; the band is wide
        # enough to survive bottom-row jitter but tight enough that the creep
        # never drives link6 past the cube.
        # 2026-10-07: 相机降到 640x480 后，图像竖直中心从 720p 的 360 变成 240。
        # 下面两处原本写死 360，会让最后一段蠕动的距离算错 0.27 m（表现为
        # "Cube is too far from link6" → 拒绝吸附）。改为参数，分辨率可切。
        self.declare_parameter('visual_row_center', 240.0)
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
        self.declare_parameter('visual_min_linear_speed', 0.02)
        self.declare_parameter('visual_max_angular_speed', 0.25)
        self.declare_parameter('visual_stable_frames', 3)
        # The final standoff used to come from ONE frame - whichever happened to
        # satisfy the acceptance gate. Measured 2026-10-12: with
        # visual_row_tolerance 18 px the accepting row can be anywhere in 662-698
        # and standoff = 0.015 + 66/(row-360) sweeps 0.210-0.234 m, i.e. 23 mm of
        # spread from the gate alone; a single noisy frame (partial blob, shadow)
        # moved it further and the base stopped ~40 mm too far, giving
        # "Pre-attach distance 0.107 m (limit 0.090)" and a refused grasp in a
        # round that had already aligned (centre_error -0.002). So: average over
        # the agreeing frames instead of trusting the last one.
        self.declare_parameter('visual_standoff_frames', 5)
        self.declare_parameter('visual_stable_span_px', 8.0)
        # If the cube is not in view (e.g. Nav2 parked off-target because of
        # localization drift), rotate in place to find it before giving up.
        self.declare_parameter('visual_search_rate', 0.35)
        self.declare_parameter('visual_search_timeout_sec', 18.0)
        # 2026-10-10: with a bad coarse parking pose the blob can slip out of the
        # frame during centring. This loop used to sit still until the 75 s
        # timeout; now it re-enters the search after this many seconds without a
        # fresh detection. Measured cost of the old behaviour: 72 s of silence in
        # run_logs/speed_rpp15_trial1_20261010-143655.
        self.declare_parameter('visual_research_after_sec', 3.0)
        self.declare_parameter('proximity_timeout_sec', 5.0)
        # One missed /gazebo/link_states window used to end the whole round:
        # measured 2026-10-12 (official_run3 and vision_stab1), the run stopped at
        # "Timed out waiting for /gazebo/link_states" during VERIFY_GRASP after
        # 3/5 deliveries, with nothing wrong with the robot. A bounded retry rides
        # out a transient DDS/gzserver stall while still failing in finite time
        # (attempts x proximity_timeout_sec).
        self.declare_parameter('proximity_retry_attempts', 3)
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
        # The detour chain is sent as ONE goal here instead of one goal per
        # waypoint. Measured 2026-10-14: with a goal per waypoint the operator
        # watches the base "ignore the route" because every leg is cancelled and
        # re-planned from wherever the yield hold left it (11 planned goals
        # produced 38 goal cancellations in one round). A single
        # NavigateThroughPoses goal keeps one plan for the whole polyline.
        self._navigate_through_client = ActionClient(
            self,
            NavigateThroughPoses,
            '/navigate_through_poses',
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
        self._speed_limit_pub = None
        self._active_speed_limit = 0.0
        self._slow_zones = self._parse_slow_zones()
        self._controller_collisions = deque()
        self._wedge_log_sub = None
        self._wedge_enabled = bool(self.get_parameter(
            'wedge_collision_timeout_sec').value > 0.0
            and self.get_parameter('wedge_collision_window_sec').value > 0.0)
        if self._wedge_enabled:
            from rcl_interfaces.msg import Log as _Log
            self._wedge_log_sub = self.create_subscription(
                _Log, '/rosout', self._wedge_log, 50)
        # The HSV pipeline only runs while the servo owns the base. Processing
        # every 1280x720 frame during the two navigation legs cost ~27% of a
        # core and ~40 MB/s of image traffic for no benefit, and on this host
        # gzserver is the bottleneck (measured RTF 0.18 with the full stack).
        self._vision_enabled = False
        self._visual_range_gate_active = False
        self._target_colour = 'red'
        camera_topic = str(self.get_parameter('camera_topic').value)
        self._image_sub = self.create_subscription(
            Image, camera_topic, self._image_callback, qos_profile_sensor_data
        )
        self._fine_cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter('fine_cmd_vel_topic').value), 10
        )
        self._retain_nav_goal_on_yield = bool(
            self.get_parameter('retain_nav_goal_on_yield').value)
        self._yield_hold_pub = None
        self._yield_gate_held = False
        self._yield_gate_status_time = 0.0
        if self._retain_nav_goal_on_yield:
            self._yield_hold_pub = self.create_publisher(
                Bool, '/competition/yield_hold', 10)
            self.create_subscription(
                Bool, '/competition/yield_gate_held',
                self._on_yield_gate_held, 10)
        # Latest lidar sweep, used only to decide which way to escape a jam:
        # reversing blindly is what drove the round32 base 0.03 m *deeper* into
        # the wall it was already trapped against.
        self._scan = None
        # When this node last received a sweep, on the monotonic clock. Falls
        # back to it when the node clock and the scan stamp are on different
        # bases - see _yield_scan_points.
        self._scan_received_at = None
        self._clock_mismatch_reported = False
        # Nominal costmap geometry for the stage in progress, plus whether a
        # temporary dead-zone escape has shrunk it.
        self._nominal_geometry = None
        self._escaping = False
        self._scan_sub = self.create_subscription(
            LaserScan, str(self.get_parameter('scan_topic').value),
            self._scan_callback, qos_profile_sensor_data
        )
        # Patrol centre lines, learned from the moving-obstacle node. Empty
        # list = unknown = the pre-lane behaviour, which is the safe fallback.
        self._patrol_segments = []
        patrols_qos = QoSProfile(depth=1)
        patrols_qos.reliability = ReliabilityPolicy.RELIABLE
        patrols_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, str(self.get_parameter('mover_patrols_topic').value),
            self._patrols_callback, patrols_qos
        )
        # Predictive yielding. The tracker and policy live in obstacle_yield.py
        # so the decision rule can be unit-tested without ROS; this node only
        # supplies observations and owns the two actuators the policy has
        # (the controller's speed limit, and a lateral arc waypoint).
        self._yield_config = obstacle_yield.YieldConfig(
            enabled=bool(self.get_parameter('enable_obstacle_yield').value),
            half_width_m=float(self.get_parameter('yield_half_width_m').value),
            obstacle_half_width_m=float(
                self.get_parameter('yield_obstacle_half_width_m').value),
            detection_range_m=float(
                self.get_parameter('yield_detection_range_m').value),
            track_range_m=float(self.get_parameter('yield_track_range_m').value),
            cap_range_m=float(self.get_parameter('yield_cap_range_m').value),
            stop_range_m=float(self.get_parameter('yield_stop_range_m').value),
            cap_speed_mps=float(
                self.get_parameter('yield_cap_speed_mps').value),
            resume_speed_mps=float(
                self.get_parameter('yield_resume_speed_mps').value),
            min_mover_speed_mps=float(
                self.get_parameter('yield_min_mover_speed_mps').value),
            max_mover_speed_mps=float(
                self.get_parameter('yield_max_mover_speed_mps').value),
            clear_margin_sec=float(
                self.get_parameter('yield_clear_margin_sec').value),
            wait_budget_sec=float(
                self.get_parameter('yield_wait_budget_sec').value),
            arc_offset_m=float(self.get_parameter('yield_arc_offset_m').value),
            arc_apex_m=float(self.get_parameter('yield_arc_apex_m').value),
            arc_goal_radius_m=float(
                self.get_parameter('yield_arc_goal_radius_m').value),
        )
        self._yield_tracker = obstacle_yield.YieldTracker(self._yield_config)
        self._yield_decision = obstacle_yield.YieldDecision(
            obstacle_yield.ACTION_GO, obstacle_yield.REASON_CLEAR
        )
        self._yield_latched = False
        self._yield_wait_since = None
        self._yield_clear_since = None
        self._yield_arc_attempts = 0
        self._yield_arc_goal = None
        self._yield_arc_used = False
        self._yield_suppress = False
        # Why the last navigation attempt was abandoned. The retry escape uses
        # it to pick a direction that matches the failure: driving 0.70 m
        # FORWARD after the controller refused the path ahead re-enters the same
        # refusal (measured 2026-10-14 15:34: eight attempts, 74-144 refusals
        # each, base never moved 2 cm, round lost at 1/5).
        self._last_attempt_failure = None
        # Index of the detour waypoint the no-progress watchdog measures
        # against; reset per leg in _navigate_once__timed.
        self._chain_index = 0
        # True while Nav2's goal is cancelled for a deliberate mover hold. The
        # speed limit alone stops the base but NOT its rotation (measured
        # 2026-10-13: ~1.5 rad/s while "held"), so the hold takes the goal away
        # and stops both axes itself.
        self._yield_pause_active = False
        # One obstacle episode: started by a hold, ended only by real progress
        # (see _update_yield_episode). The wait budget is spent against this,
        # not against an individual hold.
        self._yield_episode_since = None
        self._yield_episode_anchor = None
        self._yield_free_since = None
        self._yield_last_update = 0.0
        self._yield_last_log = 0.0
        # The decision log only fires on a non-GO action, so "nothing in the log"
        # cannot be told apart from "the tracker never produced a track". This
        # rate-limited diagnostic closes that gap the first time it is asked.
        self._yield_diag_last = 0.0
        self._yield_diag_period = float(
            self.get_parameter('yield_diag_period_sec').value)
        # Mover positions to exclude from the room estimate: a box must not be
        # mistaken for the wall the base would be squeezing past.
        self._yield_excluded = []
        self._odom = None
        self._odom_sub = self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom_callback, 20
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
        self._last_carried_log = 0.0
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

        # Resident mode (opt-in, default off): instead of being spawned once per
        # cube, the supervisor starts this node once and streams one goal per
        # object over /competition/executor_goal, collecting
        # /competition/executor_result. Rationale: a freshly spawned executor
        # creates a new DDS participant, and on this host (WSL2 + Fast DDS) that
        # participant can spend 55-60 s discovering the Nav2 action and the
        # costmap services - or never find them, which is what killed round13 at
        # cube 3. Matching every endpoint once and reusing it removes both the
        # stall and the per-cube participant/SHM churn.
        self.declare_parameter('resident_mode', False)
        self._resident = bool(self.get_parameter('resident_mode').value)
        self._goal_queue = deque()
        self._cancel_requested = False
        self._last_error = ''
        self._goal_sub = None
        self._result_pub = None
        if self._resident:
            resident_qos = QoSProfile(depth=1)
            resident_qos.reliability = ReliabilityPolicy.RELIABLE
            resident_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self._goal_sub = self.create_subscription(
                String,
                '/competition/executor_goal',
                self._goal_callback,
                resident_qos,
            )
            self._result_pub = self.create_publisher(
                String, '/competition/executor_result', resident_qos
            )

    def resident(self) -> bool:
        return self._resident

    def _ready_callback(self, msg: Bool) -> None:
        self._system_ready = msg.data

    def _link_states_callback(self, msg: LinkStates) -> None:
        self._latest_link_states = msg

    def _wheel_speeds(self) -> str:
        """Wheel spin magnitude as 'l=..,r=..', or 'n/a'.

        This is the number that separates the two ways a carried base can refuse
        to move (round35 and round38 both died here, right after the grasp):
        wheels turning while the base stands still means the robot is physically
        caught on something, wheels still means nothing reached the wheels at all.
        Read from the wheel *links*, not /joint_states: the diff-drive plugin
        publishes only joint2/joint3 there, so the wheels never appear. Only the
        magnitude is reported because the twist frame is not guaranteed, and "is
        it spinning at all" is the whole question.
        """
        msg = self._latest_link_states
        if msg is None:
            return 'n/a'
        names = list(msg.name)
        model = self.get_parameter('robot_model_name').value
        parts = []
        for side in ('left', 'right'):
            try:
                angular = msg.twist[
                    names.index(f'{model}::{side}_wheel_link')
                ].angular
            except (ValueError, IndexError):
                return 'n/a'
            parts.append(
                f'{side[0]}='
                f'{math.sqrt(angular.x ** 2 + angular.y ** 2 + angular.z ** 2):.2f}'
            )
        return ','.join(parts) + ' rad/s'

    def _log_carried_cube_state(self, label: str) -> None:
        """Trace the carried cube while a loaded leg is running (rate limited).

        The 2026-10-07 round35 jam could not be diagnosed after the fact: the
        base reported "commanded but not moving" eight times with a completely
        clean costmap (zero rejected rollouts) and nothing had recorded where the
        cube actually was. These numbers separate the two candidates - dragging
        on the floor (cube z near its 0.015 m half-height) versus caught on
        geometry (large horizontal offset from the base). Purely observational:
        it reads the cached link states and never blocks.
        """
        now = time.monotonic()
        if now - self._last_carried_log < 2.0:
            return
        msg = self._latest_link_states
        if msg is None:
            return
        names = list(msg.name)
        cube = str(self.get_parameter('cube_name').value).strip()
        try:
            cube_pose = msg.pose[
                names.index(
                    f"{cube}::{self.get_parameter('cube_link').value}"
                )
            ]
            base_pose = msg.pose[
                names.index(
                    f"{self.get_parameter('robot_model_name').value}::"
                    f"{self.get_parameter('robot_base_link').value}"
                )
            ]
        except ValueError:
            return
        horizontal = math.hypot(
            cube_pose.position.x - base_pose.position.x,
            cube_pose.position.y - base_pose.position.y,
        )
        if horizontal > 1.0:
            return              # the cube is still on the floor: not carrying yet
        self._last_carried_log = now
        self.get_logger().info(
            f'{label}: carried {cube} z={cube_pose.position.z:+.3f} m '
            f'(base z={base_pose.position.z:+.3f}), horizontal={horizontal:.3f} m, '
            f'wheels {self._wheel_speeds()}.'
        )

    def _image_callback(self, msg: Image) -> None:
        """Track the requested colour blob and publish its image geometry.

        The five cubes of one colour form a single merged blob when seen from
        the low forward camera, so the blob is selected by area among contours
        that reach the lower half of the image (the floor-level cube row). The
        blob's *bottom* row is the nearest cube's floor contact: unlike the
        blob width it is unaffected by heading error, which is what makes it a
        usable range signal.
        """
        if not self._vision_enabled:
            # Navigation phases: drop the frame before any conversion work.
            return
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Camera conversion failed: {exc}')
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Two tiers (2026-10-12): the strict HSV gates below are what keep the
        # brick-red walls out, but a shaded cube drops under them and the blob
        # then disappears for seconds - measured with the cubes in their new
        # positions, vision_align went from 8.8-9.2 s to 28-64 s with repeated
        # "Visual blob lost for 3.1 s; restarting the search". The relaxed tier is
        # only consulted when the strict one finds NO candidate, and every shape
        # and expected-row gate below still applies to it, so a wall cannot sneak
        # in merely because the cube was dim. Set the three *_scale parameters to
        # 1.0 to disable the fallback entirely.
        image_height, image_width = image.shape[:2]
        box_colour = (0, 0, 255)

        def build_mask(sat_scale: float, val_scale: float, dom_scale: float):
            """Colour mask with the brightness gates optionally loosened."""
            nonlocal box_colour
            if self._target_colour == 'blue':
                min_saturation = int(
                    self.get_parameter('visual_blue_min_saturation').value
                    * sat_scale
                )
                min_value = int(
                    self.get_parameter('visual_blue_min_value').value
                    * val_scale
                )
                hue_min = int(self.get_parameter('visual_blue_hue_min').value)
                hue_max = int(self.get_parameter('visual_blue_hue_max').value)
                mask = cv2.inRange(
                    hsv,
                    np.array([hue_min, min_saturation, min_value]),
                    np.array([hue_max, 255, 255]),
                )
                # Hue alone also matches the desaturated blue-grey of shadowed
                # brick/wood walls, so require the blue channel to dominate both
                # red and green exactly as the red gate does.
                blue, green, red = cv2.split(image)
                ratio = float(
                    self.get_parameter('visual_blue_dominance_ratio').value
                ) * dom_scale
                blue_float = blue.astype(np.float32)
                dominant = np.where(
                    (blue_float >= ratio * red.astype(np.float32))
                    & (blue_float >= ratio * green.astype(np.float32)),
                    255,
                    0,
                ).astype(np.uint8)
                mask = cv2.bitwise_and(mask, dominant)
                box_colour = (255, 0, 0)
            else:
                min_saturation = int(
                    self.get_parameter('visual_red_min_saturation').value
                    * sat_scale
                )
                min_value = int(
                    self.get_parameter('visual_red_min_value').value * val_scale
                )
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
                ) * dom_scale
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
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        def pick_best(mask):
            """Largest contour passing the shape and expected-row gates."""
            # NOTE: the accumulator must be the SAME name the loop writes and the
            # function returns. A refactor once initialised `found`, let the loop
            # fill `best`, and returned `found` - so the detector silently returned
            # None for every frame, the vision search timed out after 18 s, and a
            # competition round failed 0/5 while every source-grep test passed.
            best = None
            minimum = float(self.get_parameter('visual_min_contour_area').value)
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
                    row_center = float(self.get_parameter('visual_row_center').value)
                    camera_standoff = max(0.05, base_distance - camera_offset)
                    expected_bottom = row_center + row_k / max(
                        0.05, camera_standoff - row_offset
                    )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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


            return best

        mask = build_mask(1.0, 1.0, 1.0)
        best = pick_best(mask)
        if best is None:
            relaxed_mask = build_mask(
                float(self.get_parameter(
                    'visual_relaxed_saturation_scale').value),
                float(self.get_parameter('visual_relaxed_value_scale').value),
                float(self.get_parameter('visual_relaxed_dominance_scale').value),
            )
            best = pick_best(relaxed_mask)
            if best is not None:
                # Only reached when the strict gates saw nothing at all, which is
                # exactly the shaded-cube case this fallback exists for.
                mask = relaxed_mask
                self.get_logger().warning(
                    'Strict colour gates found no blob; using the relaxed tier.'
                )

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
        # Remembered so the resident-mode result can hand the supervisor the same
        # diagnosis it already receives on /competition/executor_error.
        self._last_error = error
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

    def _dependency_specs(self):
        """Every remote endpoint this executor needs, as (key, description, ready).

        Every callable MUST look the self._*_client attribute up when it is
        called. The action entries always did; the service entries used to pass
        the bound method of whichever client object existed when this list was
        built. After _rebuild_dependency_client() destroyed and replaced that
        client, the stale bound method called into the destroyed object and
        rclpy raised "cannot use Destroyable because destruction was requested"
        straight out of the resident loop: measured 2026-10-10, both costmap
        service clients were rebuilt and the executor died with
        OBJECT_FAILED: executor exited 1 at 0/5, 20 s into the first cube
        (run_logs/speed_rpp15_combined_trial2_20261010-170156).
        """
        return [
            (
                'arm',
                'arm trajectory action',
                lambda: self._arm_client.wait_for_server(timeout_sec=0.0),
            ),
            (
                'gripper',
                'gripper trajectory action',
                lambda: self._gripper_client.wait_for_server(timeout_sec=0.0),
            ),
            (
                'navigate',
                'Nav2 navigate_to_pose action',
                lambda: self._navigate_client.wait_for_server(timeout_sec=0.0),
            ),
            (
                'navigate_through',
                'Nav2 navigate_through_poses action',
                lambda: self._navigate_through_client.wait_for_server(
                    timeout_sec=0.0
                ),
            ),
            ('attach', '/ATTACHLINK',
             lambda: self._attach_client.service_is_ready()),
            ('detach', '/DETACHLINK',
             lambda: self._detach_client.service_is_ready()),
            (
                'clear_global',
                'global costmap clear service',
                lambda: self._clear_global_client.service_is_ready(),
            ),
            (
                'clear_local',
                'local costmap clear service',
                lambda: self._clear_local_client.service_is_ready(),
            ),
            (
                'params_global',
                'global costmap parameter service',
                lambda: self._global_costmap_params_client.service_is_ready(),
            ),
            (
                'params_local',
                'local costmap parameter service',
                lambda: self._local_costmap_params_client.service_is_ready(),
            ),
            (
                'set_state',
                '/gazebo/set_entity_state',
                lambda: self._set_state_client.service_is_ready(),
            ),
        ]

    def _rebuild_dependency_client(self, key: str) -> None:
        """Drop and re-create one unmatched client so DDS discovery runs again."""
        if key == 'arm':
            self._arm_client.destroy()
            self._arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                '/arm_controller/follow_joint_trajectory',
            )
        elif key == 'gripper':
            self._gripper_client.destroy()
            self._gripper_client = ActionClient(
                self,
                FollowJointTrajectory,
                '/gripper_controller/follow_joint_trajectory',
            )
        elif key == 'navigate':
            self._navigate_client.destroy()
            self._navigate_client = ActionClient(
                self, NavigateToPose, '/navigate_to_pose'
            )
        elif key == 'navigate_through':
            self._navigate_through_client.destroy()
            self._navigate_through_client = ActionClient(
                self, NavigateThroughPoses, '/navigate_through_poses'
            )
        elif key == 'attach':
            self.destroy_client(self._attach_client)
            self._attach_client = self.create_client(
                AttachLink, '/ATTACHLINK'
            )
        elif key == 'detach':
            self.destroy_client(self._detach_client)
            self._detach_client = self.create_client(
                DetachLink, '/DETACHLINK'
            )
        elif key == 'clear_global':
            self.destroy_client(self._clear_global_client)
            self._clear_global_client = self.create_client(
                ClearEntireCostmap,
                '/global_costmap/clear_entirely_global_costmap',
            )
        elif key == 'clear_local':
            self.destroy_client(self._clear_local_client)
            self._clear_local_client = self.create_client(
                ClearEntireCostmap,
                '/local_costmap/clear_entirely_local_costmap',
            )
        elif key == 'params_global':
            self.destroy_client(self._global_costmap_params_client)
            self._global_costmap_params_client = self.create_client(
                SetParameters,
                '/global_costmap/global_costmap/set_parameters',
            )
        elif key == 'params_local':
            self.destroy_client(self._local_costmap_params_client)
            self._local_costmap_params_client = self.create_client(
                SetParameters,
                '/local_costmap/local_costmap/set_parameters',
            )
        elif key == 'set_state':
            self.destroy_client(self._set_state_client)
            self._set_state_client = self.create_client(
                SetEntityState, '/gazebo/set_entity_state'
            )

    def _wait_for_dependencies__timed(self) -> bool:
        if bool(self.get_parameter('require_system_ready').value):
            timeout = float(self.get_parameter('ready_timeout_sec').value)
            self.get_logger().info('Waiting for /competition/system_ready=true ...')
            if not self._wait_until(
                lambda: self._system_ready, timeout, 'system readiness'
            ):
                return False

        # All endpoints share ONE deadline. Waiting per endpoint used to allow a
        # 540 s worst case on its own, and round13 cube 3 died exactly there: a
        # freshly spawned executor participant never discovered
        # /navigate_to_pose (attempt 1) and then never discovered the local
        # costmap clear service (attempt 2), each burning the full
        # dependency_timeout_sec while the stack itself was healthy (sim clock
        # advancing, 32 nodes alive, health checker still reporting READY, and
        # nothing in the log except these two timeouts). On WSL/Fast DDS a new
        # client can stay unmatched for its whole lifetime, so - exactly as
        # system_health_check.py already does for the lifecycle services -
        # rebuild the client every few seconds and let discovery run again.
        timeout = float(self.get_parameter('dependency_timeout_sec').value)
        rebuild_interval = 8.0
        started_at = time.monotonic()
        deadline = started_at + timeout
        pending = self._dependency_specs()
        rebuilt_at = {}
        while pending and rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            for spec in list(pending):
                key, description, ready = spec
                if ready():
                    pending.remove(spec)
                    rebuilt_at.pop(key, None)
                    continue
                if now - rebuilt_at.get(key, started_at) >= rebuild_interval:
                    self._rebuild_dependency_client(key)
                    rebuilt_at[key] = now
                    self.get_logger().warning(
                        f'{description}: client never matched; rebuilt for DDS '
                        f'discovery ({deadline - now:.0f} s of budget left).'
                    )
            if pending:
                rclpy.spin_once(self, timeout_sec=0.1)

        if pending:
            self.get_logger().error(
                f'Timed out waiting for {timeout:.0f} s for: '
                + ', '.join(description for _, description, _ in pending)
                + ' (clients were rebuilt; DDS discovery stall).'
            )
            return False
        return True

    def _send_goal_tolerant(
        self,
        label: str,
        client: ActionClient,
        goal,
        response_timeout: float,
        settle_sec: float,
    ):
        """Send an action goal, retrying when the goal response is lost.

        The controller can accept a goal and still fail to deliver the response
        (Fast DDS allows roughly 100 ms for a service response). Because each
        retry re-sends the identical target, waiting for the possibly-executing
        motion to settle first keeps the retry idempotent. Returns ``None`` when
        every attempt lost its response.
        """
        attempts = max(1, int(self.get_parameter('action_retry_count').value) + 1)
        for attempt in range(1, attempts + 1):
            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(
                self, future, timeout_sec=response_timeout
            )
            if future.done():
                return future.result()
            self.get_logger().warning(
                f'{label}: no goal response within {response_timeout:.1f}s '
                f'(attempt {attempt}/{attempts}); the server may have accepted '
                'the goal and be executing it, so waiting before re-sending the '
                'same target.'
            )
            if attempt < attempts:
                end = time.monotonic() + settle_sec
                while rclpy.ok() and time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def _execute_trajectory__timed(
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
        timeout = float(self.get_parameter('action_timeout_sec').value)
        settle_sec = float(self.get_parameter('settle_sec').value)
        goal_handle = self._send_goal_tolerant(
            label, client, goal, timeout, motion_sec + settle_sec
        )
        if goal_handle is None:
            raise RuntimeError('MOTION_STATE_UNKNOWN: goal response timed out')

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
        goal = list(self._double_list(parameter_names[warehouse], 3))
        offsets = [float(v) for v in self.get_parameter('zone_slot_offsets').value]
        if len(offsets) >= 2:
            slots = [offsets[i:i + 2] for i in range(0, len(offsets) - 1, 2)]
            index = self._zone_deliveries.get(warehouse, 0) % len(slots)
            dx, dy = slots[index]
            goal[0] += dx
            goal[1] += dy
            self.get_logger().info(
                f'warehouse {warehouse}: slot {index + 1}/{len(slots)} '
                f'(offset {dx:+.2f}, {dy:+.2f}) -> dock ({goal[0]:.3f}, {goal[1]:.3f})'
            )
        return warehouse, goal

    def _pickup_goal(self) -> tuple[str, list[float]]:
        cube = str(self.get_parameter('cube_name').value).strip()
        parameter_name = f'pickup_{cube}_pose'
        if not self.has_parameter(parameter_name):
            raise ValueError(f'No calibrated pickup pose for cube {cube!r}.')
        return cube, self._double_list(parameter_name, 3)

    def _clear_costmaps__timed(self) -> bool:
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

    def _set_costmap_geometry__timed(
        self, radius: float, inflation_radius: float, label: str,
        remember: bool = True,
    ) -> bool:
        """Update both active Nav2 costmaps for empty or loaded geometry."""
        if radius <= 0.0 or inflation_radius < radius:
            self.get_logger().error(
                f'Invalid {label} costmap geometry: robot radius={radius:.3f} m, '
                f'inflation radius={inflation_radius:.3f} m.'
            )
            return False
        # The global costmap gets a deliberately larger footprint than the local
        # one. NavFn will happily route through the inscribed band (cost 253) that
        # DWB's BaseObstacle refuses outright, which is the root of the recurring
        # "old place" wedges: measured 2026-10-07, loaded legs to A produced 383
        # and 694 rejected rollouts even though the flood fill proved a compliant
        # route existed. Planning with a fatter robot keeps the global path in
        # corridors DWB can actually drive; the local costmap keeps the true
        # 0.35 m so the controller can still work the last tight metres.
        # Checked on room_from_world.pgm: every dock, approach point and pickup
        # pose stays free, and every delivery leg stays reachable, up to 0.55 m.
        margin = max(0.0, float(self.get_parameter('global_radius_margin_m').value))
        timeout = float(self.get_parameter('service_timeout_sec').value)
        for costmap_label, client, costmap_radius in (
            ('global', self._global_costmap_params_client, radius + margin),
            ('local', self._local_costmap_params_client, radius),
        ):
            parameters = [
                Parameter(
                    name='robot_radius',
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=costmap_radius,
                    ),
                ),
                Parameter(
                    name='inflation_layer.inflation_radius',
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=max(inflation_radius, costmap_radius),
                    ),
                ),
            ]
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
        if remember:
            # The leg's nominal geometry, so a temporary dead-zone escape can
            # always put back exactly what the stage expects.
            self._nominal_geometry = (radius, inflation_radius, label)
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

    def _retreat_to_handoff_standoff(self, cube: str, distance: float) -> float:
        """Open the base-to-cube standoff before the camera servo takes over.

        Drives along whichever axis points away from the cube - the base usually
        arrives nose-first, so it reverses, but the heading is checked rather
        than assumed. Bounded by a timeout; returns the best distance seen so the
        caller can log what actually happened.
        """
        target = float(
            self.get_parameter('visual_handoff_retreat_target_m').value
        )
        speed = abs(float(
            self.get_parameter('visual_handoff_retreat_speed_mps').value
        ))
        deadline = time.monotonic() + float(
            self.get_parameter('visual_handoff_retreat_timeout_sec').value
        )
        self.get_logger().warning(
            f'{cube} is only {distance:.3f} m away - too close for the camera '
            f'servo; retreating to {target:.2f} m before handing over.'
        )
        best = distance
        command = Twist()
        started = time.monotonic()
        flipped = False
        while rclpy.ok() and time.monotonic() < deadline:
            pose = self._cached_base_pose()
            current = self._cached_planar_distance_to_cube(cube)
            if pose is None or current is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            best = max(best, current)
            if current >= target:
                break
            cube_pose = self._cached_cube_position(cube)
            if cube_pose is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            # Heading that points straight away from the cube. atan2 takes y
            # first: swapping the arguments mirrors the escape direction, which
            # is why the first attempt at this retreat drove 0.26 -> 0.288 m.
            away = math.atan2(
                pose[1] - cube_pose[1], pose[0] - cube_pose[0]
            )
            error = abs(math.atan2(
                math.sin(away - pose[2]), math.cos(away - pose[2])
            ))
            # Facing the cube (error near pi) means reverse; facing away, drive.
            command.linear.x = speed if error < math.pi / 2.0 else -speed
            self._fine_cmd_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
            # Belt and braces: if the first half second of driving closed the
            # gap instead of opening it, the heading estimate is wrong - flip.
            if not flipped and time.monotonic() - started > 0.5:
                if current <= distance:
                    command.linear.x = -command.linear.x
                    flipped = True
                    self.get_logger().warning(
                        f'{cube}: retreat was closing the gap; reversing.'
                    )
        self._stop_base()
        for _ in range(3):
            self._stop_base()
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warning(
            f'{cube}: retreat finished at {best:.3f} m.'
        )
        return best

    def _cached_cube_position(self, cube: str) -> tuple[float, float] | None:
        """Live cube XY from the cached Gazebo link states, or None."""
        msg = self._latest_link_states
        if msg is None:
            return None
        key = f"{cube}::{self.get_parameter('cube_link').value}"
        try:
            pose = msg.pose[list(msg.name).index(key)]
        except ValueError:
            return None
        return pose.position.x, pose.position.y

    def _cached_base_pose(self) -> tuple[float, float, float] | None:
        """Live base (x, y, yaw) from the cached Gazebo link states, or None."""
        msg = self._latest_link_states
        if msg is None:
            return None
        key = (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('robot_base_link').value}"
        )
        try:
            pose = msg.pose[list(msg.name).index(key)]
        except ValueError:
            return None
        quat = pose.orientation
        yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
        )
        return pose.position.x, pose.position.y, yaw

    def _cached_base_position(self) -> tuple[float, float] | None:
        """Live base XY from the cached Gazebo link states, or None."""
        msg = self._latest_link_states
        if msg is None:
            return None
        key = (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('robot_base_link').value}"
        )
        try:
            pose = msg.pose[list(msg.name).index(key)]
        except ValueError:
            return None
        return pose.position.x, pose.position.y

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

    def _cached_goal_error(
        self, x: float, y: float, yaw: float
    ) -> tuple[float, float] | None:
        """Return live Gazebo (distance, yaw error) of the base to a map pose.

        The link states are world-frame and the goals are map-frame, which the
        rest of this executor already treats as interchangeable (the stack runs
        an identity map->odom transform next to AMCL).
        """
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
        distance = math.hypot(base_pose.position.x - x, base_pose.position.y - y)
        quat = base_pose.orientation
        base_yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
        )
        yaw_error = math.atan2(
            math.sin(base_yaw - yaw), math.cos(base_yaw - yaw)
        )
        return distance, yaw_error

    def _base_pose(self):
        """Return the Gazebo ground-truth base pose as (x, y, yaw), or None."""
        msg = self._latest_link_states
        if msg is None:
            return None
        base_name = (
            f"{self.get_parameter('robot_model_name').value}::"
            f"{self.get_parameter('robot_base_link').value}"
        )
        try:
            pose = msg.pose[list(msg.name).index(base_name)]
        except ValueError:
            return None
        quat = pose.orientation
        yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
        )
        return pose.position.x, pose.position.y, yaw

    def _creep_to_dock__timed(self, label: str, x: float, y: float, yaw: float) -> bool:
        """Drive the last few centimetres to the plate with Nav2 released.

        Rotate to the plate yaw first, then creep straight in with a small
        lateral correction. This replaces the controller's near-goal limit cycle
        (see enable_dock_creep) and, unlike waiting for the goal checker, it ends
        at the same pose every time, which is what the placement accuracy is
        calibrated against.
        """
        speed = float(self.get_parameter('dock_creep_speed_mps').value)
        forward_gain = float(
            self.get_parameter('dock_creep_forward_gain').value
        )
        position_tolerance = float(
            self.get_parameter('dock_creep_tolerance_m').value
        )
        yaw_tolerance = float(
            self.get_parameter('dock_creep_yaw_tolerance_rad').value
        )
        timeout = float(self.get_parameter('dock_creep_timeout_sec').value)
        accept_radius = float(self.get_parameter('dock_creep_accept_m').value)
        accept_yaw = float(
            self.get_parameter('navigation_accept_yaw_rad').value
        )
        attempts = max(1, int(self.get_parameter('dock_creep_attempts').value))
        align_distance = float(
            self.get_parameter('dock_creep_align_distance_m').value
        )
        command = Twist()
        final_distance = float('inf')
        final_yaw = float('inf')
        # 2026-10-08: release on POSITION, not on heading.
        # The C/A docks kept failing while the base was already at the plate:
        #   run_logs/acceptfix_20261008-002353  A: 0.163 m / 73.6 deg,
        #                                            0.172 m / -157.0 deg,
        #                                            0.166 m / 75.5 deg,
        #                                            0.171 m / 94.7 deg
        # Every one of those is inside dock_creep_accept_m 0.22 m; only the
        # heading was off, and the base rotates at just ~0.543 rad/s, so a
        # 120-160 deg correction needs 4-5 s of pure rotation plus spin-up and
        # the final translation - it does not fit the 14 s budget, and the leg
        # then dies with the cube still attached. Let the arm do the work
        # instead: the arm goes to a fixed posture and the placement zone is
        # large (|dx| <= 0.485 m, |dy| <= 0.235 m), so a base yaw error still
        # drops the cube inside the zone as long as it stays under the ~60 deg
        # cliff derived in navigation_accept_yaw_rad. So once the position is
        # inside dock_creep_accept_m we still POLISH the heading - it keeps the
        # drop closer to the zone centre - but we no longer refuse the arrival
        # for it: after position_hold_sec of holding position we accept.
        position_hold_sec = 2.0
        position_since = None
        for attempt in range(1, attempts + 1):
            deadline = time.monotonic() + timeout
            position_since = None
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                pose = self._base_pose()
                if pose is None:
                    continue
                base_x, base_y, base_yaw = pose
                dx = x - base_x
                dy = y - base_y
                distance = math.hypot(dx, dy)
                yaw_error = math.atan2(
                    math.sin(yaw - base_yaw), math.cos(yaw - base_yaw)
                )
                if (
                    distance <= position_tolerance
                    and abs(yaw_error) <= yaw_tolerance
                ):
                    break
                if distance <= accept_radius:
                    # Position is good; give the heading a short chance to settle,
                    # then take the pose - but ONLY while the heading is still
                    # inside the bound the drop geometry allows.
                    # 2026-10-08: an earlier version accepted at ANY heading and
                    # the cube landed outside the zone:
                    #   Cube missed warehouse C: cube at (0.149,-2.445), centre
                    #   (0.016,-3.060), offset (+0.133,+0.615), limits [0.50x0.25]
                    #   (run_logs/dockfix_20261008-004925)
                    # 136.6 deg of heading error threw the fixed-posture drop
                    # 0.615 m off, so my earlier ~60 deg estimate (from a 0.35 m
                    # lever) was far too generous. Back-solving that drop: a pure-y
                    # offset needs |sin| = 0.615/L and a pure-x one |1-cos| = 0.133/L,
                    # which no single L satisfies - the offset is mixed, so the
                    # tighter bound wins. Centring the cube needs <= ~8 deg and
                    # staying inside the 0.235 m half-width needs <= ~49 deg, so gate
                    # on accept_yaw (the 45.8 deg bound used elsewhere in the leg) and
                    # otherwise keep polishing: a retry with a fresh heading beats a
                    # cube off the plate.
                    now = time.monotonic()
                    if position_since is None:
                        position_since = now
                    elif now - position_since >= position_hold_sec:
                        if abs(yaw_error) <= accept_yaw:
                            self.get_logger().warning(
                                f'{label}: holding position for '
                                f'{position_hold_sec:.1f} s at {distance:.3f} m, '
                                f'heading {math.degrees(yaw_error):+.1f} deg off - '
                                f'inside the {math.degrees(accept_yaw):.1f} deg drop '
                                'bound, so accepting the arrival and letting the arm '
                                'place. verify_placement still decides.'
                            )
                            command.linear.x = 0.0
                            command.angular.z = 0.0
                            self._fine_cmd_pub.publish(command)
                            # Must RETURN, not break: breaking only leaves the inner
                            # `while`, so the outer `for attempt` loop ran on, hit its
                            # own deadline, and overwrote the acceptance with
                            #   "dock creep failed at 0.121 m, -134.6 deg off yaw"
                            self._stop_base()
                            final_distance = distance
                            final_yaw = yaw_error
                            return True
                        # Heading still outside the drop bound: keep creeping and
                        # re-arm the hold so the next check is a fresh 2 s window.
                        position_since = now
                else:
                    position_since = None
                if distance > align_distance:
                    # Far field: point at the goal and drive. The previous law
                    # drove along the plate axis instead, so an arrival whose
                    # error was mostly lateral collapsed to the 0.03 m/s floor and
                    # stalled about 0.2 m short (round21 C dock: 0.165 m in 8 s;
                    # round22: 0.198 m in 14 s, three times, cube attached).
                    bearing = math.atan2(dy, dx)
                    bearing_error = math.atan2(
                        math.sin(bearing - base_yaw),
                        math.cos(bearing - base_yaw),
                    )
                    if abs(bearing_error) > 0.6:
                        command.linear.x = 0.0
                        command.angular.z = max(
                            -1.2, min(1.2, 2.0 * bearing_error)
                        )
                    else:
                        command.linear.x = max(
                            0.03, min(speed, forward_gain * distance)
                        )
                        command.angular.z = max(
                            -1.0, min(1.0, 2.0 * bearing_error)
                        )
                else:
                    # Final centimetres: hold the plate yaw, because the arm's
                    # fixed offset - and with it the drop point - is calibrated
                    # against exactly this pose.
                    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
                    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
                    # 2026-10-07: inside the align distance the old law always
                    # creeped forward (>= 0.02 m/s floor) while the yaw term
                    # 2.0 * yaw_error + 1.5 * lateral saturated the 0.8 rad/s
                    # clamp. An arrival that was badly mis-yawed therefore could
                    # not stop turning: it circled the dock point instead of
                    # settling on it. Measured on run_logs/attachfix_...: the base
                    # orbited (0.02, -2.70) inside a +/-0.25 m box for 8 s and
                    # swept 2351 deg of yaw across the two attempts, passing the
                    # -1.571 rad target without ever stopping - dock creep then
                    # died on its 14 s timeout (0.171/0.169/0.080 m, 176/82/-32
                    # deg off). The caster-friction fix made rotation easier and
                    # amplified it (the same dock had already swept 826 deg
                    # before). So: when the heading is still far off, rotate in
                    # place and do not translate at all. Translation resumes once
                    # the yaw is inside the alignment band, which is also when
                    # the forward/lateral maths stops fighting itself.
                    if abs(yaw_error) > max(0.12, 4.0 * yaw_tolerance):
                        command.linear.x = 0.0
                        # One-sided clamp: a full 0.8 rad/s spin overshoots the
                        # small residual errors that matter for docking.
                        command.angular.z = max(
                            -0.8, min(0.8, 2.0 * yaw_error)
                        )
                    else:
                        command.linear.x = max(
                            0.02, min(speed, forward_gain * max(forward, 0.0))
                        )
                        command.angular.z = max(
                            -0.8, min(0.8, 2.0 * yaw_error + 1.5 * lateral)
                        )
                self._fine_cmd_pub.publish(command)
            self._stop_base()
            for _ in range(3):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.05)

            pose = self._base_pose()
            if pose is None:
                self.get_logger().error(f'{label}: dock creep lost the base pose.')
                return False
            final_distance = math.hypot(x - pose[0], y - pose[1])
            final_yaw = math.atan2(
                math.sin(yaw - pose[2]), math.cos(yaw - pose[2])
            )
            if (
                final_distance <= position_tolerance + 0.02
                and abs(final_yaw) <= yaw_tolerance + 0.05
            ):
                self.get_logger().info(
                    f'{label}: dock creep finished at {final_distance:.3f} m '
                    f'from the goal, {math.degrees(final_yaw):.1f} deg off yaw.'
                )
                return True
            if attempt < attempts:
                # Re-arm instead of failing outright: with a cube attached a
                # failed leg becomes an UNSAFE_OBJECT_STATE that ends the whole
                # task (round22).
                self.get_logger().warning(
                    f'{label}: dock creep still {final_distance:.3f} m short '
                    f'({math.degrees(final_yaw):.1f} deg off yaw); re-arming '
                    f'({attempt}/{attempts}).'
                )

        if final_distance <= accept_radius and abs(final_yaw) <= accept_yaw:
            self.get_logger().warning(
                f'{label}: dock creep stopped short at {final_distance:.3f} m, '
                f'{math.degrees(final_yaw):.1f} deg off yaw (target '
                f'{position_tolerance:.3f} m / '
                f'{math.degrees(yaw_tolerance):.1f} deg; anything worse than '
                f'{accept_radius:.3f} m is refused because a 0.165 m docking '
                'error already put the cube outside the zone).'
            )
            return True
        self.get_logger().error(
            f'{label}: dock creep failed at {final_distance:.3f} m from the '
            f'goal, {math.degrees(final_yaw):.1f} deg off yaw.'
        )
        return False

    def _release_nav_control(self, label, goal_handle, result_future, timeout):
        """Wait for a terminal Nav2 result before sending manual base commands.

        An acknowledged cancel request alone is not enough: FollowPath may still
        be publishing while cancellation is in flight. Fail closed if the
        action transport cannot establish that the previous goal has ended.
        """
        if not result_future.done():
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=timeout)
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            raise RuntimeError(
                f'NAV_STATE_UNKNOWN: {label} cancellation not confirmed'
            )
        result = result_future.result()
        if result is None or result.status not in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        ):
            raise RuntimeError(
                f'NAV_STATE_UNKNOWN: {label} has no terminal result'
            )
        self._stop_base()

    def _on_yield_gate_held(self, message: Bool) -> None:
        self._yield_gate_held = message.data
        self._yield_gate_status_time = time.monotonic()

    def _command_yield_gate(self, held: bool, timeout: float = 0.5) -> bool:
        """Wait for a fresh acknowledgement from the sole smoother-input gate."""
        if self._yield_hold_pub is None:
            return False
        since = time.monotonic()
        command = Bool()
        command.data = held
        while rclpy.ok() and time.monotonic() - since < timeout:
            self._yield_hold_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self._yield_gate_status_time >= since
                    and self._yield_gate_held == held):
                return True
        return False

    def _ensure_yield_gate_clear(self, timeout: float = 1.5) -> bool:
        """Bring the gate to a known released state before a leg drives.

        A previous leg can leave the latch held (it failed while a hold was
        active), so a stale True is cleared here instead of being treated as a
        dead gate. No fresh status at all still fails closed.
        """
        if not getattr(self, '_retain_nav_goal_on_yield', False):
            return True
        fresh = time.monotonic() - self._yield_gate_status_time <= 0.5
        if fresh and not self._yield_gate_held:
            return True
        if not self._command_yield_gate(False, timeout=timeout):
            return False
        return True

    def _hold_until_clear_retained(
        self, label, x, y, deadline, goal_handle, result_future, response_timeout,
    ) -> bool:
        """Keep Nav2's TOP-LEVEL goal but halt its BT and gate both velocities.

        The gate is released only once the leg is finished with Nav2: either the
        corridor is clear and the same goal continues, or the cancel has been
        confirmed. Otherwise every abandon path would leave the base gated off
        and the following retry could not drive at all.
        """
        if not self._command_yield_gate(True):
            self.get_logger().error(f'{label}: velocity gate did not acknowledge hold.')
            self._release_nav_control(
                label, goal_handle, result_future, min(5.0, response_timeout))
            self._command_yield_gate(False)
            return False
        self._yield_pause_active = True
        started = time.monotonic()
        self.get_logger().warning(
            f'{label}: holding both velocity axes with Nav2 goal retained.')
        try:
            while (rclpy.ok() and self._yield_decision.holds
                   and not result_future.done() and time.monotonic() < deadline):
                # Re-publish the hold: a BT plugin instantiated a tick late must
                # still observe it; the gate remains latched until explicit clear.
                message = Bool()
                message.data = True
                self._yield_hold_pub.publish(message)
                rclpy.spin_once(self, timeout_sec=0.1)
                self._tick_yield(self._cached_planar_distance_to_point(x, y))
            held_for = time.monotonic() - started
            if self._yield_decision.holds or self._yield_arc_goal is not None:
                # The loop left while the obstacle was still there: either the
                # leg budget expired or Nav2's own goal ended. Both mean this
                # attempt is over, so the goal must go before anything drives.
                if self._yield_decision.holds and held_for >= deadline - started:
                    reason = 'leg budget expired'
                elif self._yield_arc_goal is not None:
                    reason = 'taking the yield arc instead'
                else:
                    reason = 'Nav2 ended the goal during the hold'
                if result_future.done():
                    reason = 'Nav2 ended the goal during the hold'
                self.get_logger().warning(
                    f'{label}: releasing the retained goal ({reason}) after '
                    f'{held_for:.1f} s of holding.'
                )
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout))
                self._command_yield_gate(False, timeout=1.0)
                return False
            if not self._command_yield_gate(False, timeout=1.0):
                # Do not allow a late 'hold=False' or a dead gate to be mistaken
                # for a confirmed release; cancel the goal while still gated.
                self.get_logger().error(
                    f'{label}: velocity gate did not confirm the release; '
                    'cancelling Nav2 while still gated.'
                )
                message = Bool()
                message.data = True
                self._yield_hold_pub.publish(message)
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout))
                self._command_yield_gate(False, timeout=1.5)
                return False
            return True
        finally:
            self._yield_pause_active = False

    def _hold_until_clear(self, label, x, y, deadline) -> bool:
        """Stand still until the yield policy releases a mover hold.

        Runs with Nav2's goal already cancelled, so the controller publishes
        nothing and the zero Twist this loop sends to the smoother's input is
        the only command: BOTH axes stop, which is the part a Nav2 speed limit
        cannot do. Returns True when the corridor cleared inside the leg's own
        deadline, False when the leg should be failed instead.
        """
        self._yield_pause_active = True
        self.get_logger().warning(f'{label}: holding still for a moving obstacle.')
        try:
            while (rclpy.ok() and self._yield_decision.holds
                    and time.monotonic() < deadline):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.1)
                self._tick_yield(self._cached_planar_distance_to_point(x, y))
        finally:
            self._yield_pause_active = False
        return not self._yield_decision.holds

    def _resume_nav_goal(self, label, goal, response_timeout, client=None):
        """Re-send a leg's own goal after a yield pause.

        Returns ``(goal_handle, result_future)`` or None when Nav2 refuses the
        goal, in which case the caller fails the leg and the ordinary retry
        loop takes over. The goal object is the identical one the leg started
        with - one pose for a plain leg, the whole detour chain for a
        continuous-detour leg - so a resume is the same leg continuing.
        """
        goal_handle = self._send_goal_tolerant(
            label, client if client is not None else self._navigate_client,
            goal, response_timeout, 5.0
        )
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f'Nav2 did not accept the resumed {label} goal; failing the leg.'
            )
            return None
        self.get_logger().warning(
            f'{label}: mover hold clear; resuming the navigation goal.'
        )
        return goal_handle, goal_handle.get_result_async()

    def _pose_stamped(self, values: Sequence[float]) -> PoseStamped:
        x, y, yaw = [float(value) for value in values]
        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter('navigation_frame').value)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def _chain_progress_target(self, chain, x: float, y: float):
        """The point whose arrival is the next progress milestone.

        Without a chain this is the leg's own goal, exactly as before. With a
        chain it is the earliest waypoint not yet reached, so the no-progress
        watchdog still fires on a genuinely stuck base while the bypass is
        allowed to drive away from the final pose first.
        """
        if not chain:
            return (x, y)
        index = min(self._chain_index, len(chain) - 1)
        while index < len(chain) - 1:
            here = self._cached_planar_distance_to_point(
                float(chain[index][0]), float(chain[index][1])
            )
            ahead = self._cached_planar_distance_to_point(
                float(chain[index + 1][0]), float(chain[index + 1][1])
            )
            # Advance once the base is closer to the NEXT waypoint than to this
            # one, i.e. it has effectively passed it. A fixed arrival radius is
            # wrong here: Nav2's through-poses path rounds the corners and can
            # stay >0.75 m from a waypoint the base has already driven past,
            # which left the reference behind the robot and made the
            # no-progress watchdog abandon a healthy attempt (measured
            # 2026-10-14: "no progress for 45 s (still 2.12 m from the goal)",
            # a 0.70 m back-off, and 205 s for a leg that takes 52 s otherwise).
            if here is None or ahead is None or ahead > here:
                break
            index += 1
        self._chain_index = index
        return (float(chain[index][0]), float(chain[index][1]))

    def _nav_goal(self, values: Sequence[float], chain=None):
        """Build the Nav2 goal, and pick the client that must execute it.

        No ``chain`` means the original single-pose NavigateToPose goal. With a
        chain, it is ONE NavigateThroughPoses goal holding every detour waypoint
        in order followed by the leg's own target, so Nav2 plans and follows the
        configured polyline as a single path. A chain entry that repeats the
        target is dropped: two identical consecutive poses add nothing and some
        BT configurations stall on them.
        """
        if chain:
            goal = NavigateThroughPoses.Goal()
            poses = [
                self._pose_stamped(waypoint)
                for waypoint in chain
                if math.hypot(float(waypoint[0]) - float(values[0]),
                              float(waypoint[1]) - float(values[1])) > 1e-3
            ]
            poses.append(self._pose_stamped(values))
            goal.poses = poses
            if getattr(self, '_retain_nav_goal_on_yield', False):
                goal.behavior_tree = os.path.join(
                    get_package_share_directory('mybot'), 'config',
                    'nav_through_poses_yield_pause.xml')
            return goal, self._navigate_through_client
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(values)
        if getattr(self, '_retain_nav_goal_on_yield', False):
            goal.behavior_tree = os.path.join(
                get_package_share_directory('mybot'), 'config',
                'nav_to_pose_yield_pause.xml')
        return goal, self._navigate_client

    def _remaining_chain(self, chain):
        """The detour waypoints the base has NOT effectively passed yet.

        A re-sent goal makes Nav2 forget which waypoints were already reached:
        its RemovePassedGoals only drops the ones the base is within 0.7 m of
        *in that goal*, so re-sending the whole chain after every mover hold
        sends the base back to the first waypoint again. Measured 2026-10-14
        (run_logs/stable_topose_ab_20261014-161832, blue_cube_3 -> A): seven
        holds, seven resumes, and the base drove from via_1 to via_2 and back to
        via_1 twice before it could start climbing to via_3.
        """
        if not chain:
            return None
        remaining = list(chain[min(self._chain_index, len(chain)):])
        return remaining or None

    def _navigate_once__timed(
        self,
        label: str,
        values: Sequence[float],
        visual_handoff_cube: str | None = None,
        completion_radius_m: float | None = None,
        chain: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        x, y, yaw = [float(value) for value in values]
        goal, client = self._nav_goal(values, self._remaining_chain(chain))
        route_text = ''
        if self._remaining_chain(chain):
            route_text = ' via ' + ' -> '.join(
                f'({float(waypoint[0]):.3f}, {float(waypoint[1]):.3f})'
                for waypoint in self._remaining_chain(chain)
            )
        self.get_logger().info(
            f'Navigate to {label}: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
            f'{route_text}.'
        )
        response_timeout = float(self.get_parameter('service_timeout_sec').value)
        if not self._ensure_yield_gate_clear():
            self._stop_base()
            raise RuntimeError(
                'YIELD_GATE_NOT_READY: guarded smoother input unavailable')
        # Nav2 can lose a goal response the same way the arm controller does;
        # re-sending the identical pose preempts the old goal and is harmless.
        goal_handle = self._send_goal_tolerant(
            label, client, goal, response_timeout, 5.0
        )
        if goal_handle is None:
            self.get_logger().error(
                f'Nav2 did not acknowledge {label} within '
                f'{response_timeout:.1f}s; refusing to wait indefinitely.'
            )
            self._stop_base()
            raise RuntimeError('NAV_STATE_UNKNOWN: goal response timed out')
        if not goal_handle.accepted:
            self.get_logger().error(f'Nav2 rejected the {label} goal.')
            return False

        result_future = goal_handle.get_result_async()
        timeout = float(self.get_parameter('navigation_timeout_sec').value)
        deadline = time.monotonic() + timeout
        handoff_distance = float(
            self.get_parameter('visual_handoff_distance_m').value
        )
        min_handoff_distance = float(
            self.get_parameter('visual_handoff_min_distance_m').value
        )
        accept_radius = float(
            self.get_parameter('navigation_accept_radius_m').value
        )
        accept_yaw = float(
            self.get_parameter('navigation_accept_yaw_rad').value
        )
        accept_hold = float(
            self.get_parameter('navigation_accept_hold_sec').value
        )
        dock_creep_trigger = (
            float(self.get_parameter('dock_creep_trigger_m').value)
            if bool(self.get_parameter('enable_dock_creep').value)
            else 0.0
        )
        accept_since = None
        stall_timeout = float(
            self.get_parameter('navigation_stall_timeout_sec').value
        )
        stall_progress = float(
            self.get_parameter('navigation_stall_progress_m').value
        )
        best_distance = None
        stall_since = None
        # Do not drive an escape while a Nav2 goal is still active. On stall,
        # cancel the goal and let _navigate_to perform one escape before retry.
        nomotion_timeout = float(
            self.get_parameter('navigation_nomotion_timeout_sec').value
        )
        nomotion_radius = float(
            self.get_parameter('navigation_nomotion_radius_m').value
        )
        motion_anchor = None
        motion_since = None
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            # The yield policy subsumes the slow-zone cap (it merges both and
            # publishes the lower one), so it replaces the old call here rather
            # than following it, which would publish twice per tick.
            self._tick_yield(self._cached_planar_distance_to_point(x, y))
            if self._yield_decision.holds and not self._yield_suppress:
                # A deliberate mover hold must not be driven by Nav2, and it must
                # not end the leg either. The speed limit alone stops the base
                # but NOT its rotation (measured 2026-10-13: ~1.5 rad/s while
                # "held", which is what the operator saw as circling), so cancel
                # the goal and wait it out in a nested loop - the loop below
                # would otherwise see the cancelled future and fall straight
                # through to the jam escape, which moved the base 0.70 m
                # backwards on every single hold of that round.
                if getattr(self, '_retain_nav_goal_on_yield', False):
                    if not self._hold_until_clear_retained(
                        label, x, y, deadline, goal_handle, result_future,
                        response_timeout,
                    ):
                        return False
                    # A deliberate hold is not a jam. Resume the same top-level
                    # action with fresh no-motion and progress watch baselines.
                    motion_anchor = motion_since = None
                    best_distance = stall_since = accept_since = None
                    continue
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout),
                )
                if not self._hold_until_clear(label, x, y, deadline):
                    return False
                if self._yield_arc_goal is not None:
                    # The episode budget can expire inside the hold loop. Nav2
                    # is already cancelled; hand the pending arc to _navigate_to
                    # instead of resuming a goal just to cancel it immediately.
                    self.get_logger().warning(
                        f'{label}: mover hold budget spent; taking the yield arc.'
                    )
                    return False
                # Rebuild from the REMAINING waypoints: the base may have
                # reached one or two of them while it was held, and re-sending
                # the whole chain would send it back to the first one.
                goal, client = self._nav_goal(values, self._remaining_chain(chain))
                resumed = self._resume_nav_goal(
                    label, goal, response_timeout, client
                )
                if resumed is None:
                    return False
                goal_handle, result_future = resumed
                continue
            if self._yield_arc_goal is not None:
                # Do not wait for the ordinary no-motion watchdog: it can run
                # the jam/back-off recovery while a mover occupies this lane.
                self.get_logger().warning(
                    f'{label}: yield arc pending; releasing Nav2 without jam recovery.'
                )
                # A confirmed cancel resets the progress checker for the next
                # goal; an unconfirmed one raises NAV_STATE_UNKNOWN.
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout),
                )
                return False
            self._log_carried_cube_state(label)
            if nomotion_timeout > 0.0:
                # "The base has not moved at all", straight from its position -
                # the check the operator makes by eye from the Gazebo window.
                position = self._cached_base_position()
                if position is not None:
                    now = time.monotonic()
                    if (
                        motion_anchor is None
                        or math.hypot(
                            position[0] - motion_anchor[0],
                            position[1] - motion_anchor[1],
                        ) >= nomotion_radius
                    ):
                        motion_anchor = position
                        motion_since = now
                    elif (
                        self._wedge_enabled
                        and not self._yield_latched
                        and motion_since is not None
                        and now - motion_since > float(self.get_parameter(
                            'wedge_collision_timeout_sec').value)
                        and self._wedge_collisions_since(motion_since) >= int(
                            self.get_parameter(
                                'wedge_collision_min_messages').value)
                    ):
                        # Hard wedge: the controller is refusing to move because
                        # its own collision check sees the footprint intruding,
                        # and no amount of waiting changes that - only the escape
                        # below does. Measured cost of waiting instead: 42.7 s in
                        # run_logs/speed_rpp15_combined_trial5_20261010-173315,
                        # with 353 collision messages and zero motion.
                        self.get_logger().error(
                            f'{label}: controller reported '
                            f'{self._wedge_collisions_since(motion_since)} collision '
                            f'refusals while the base stood still for '
                            f'{now - motion_since:.1f} s; abandoning early so the '
                            'escape can free it.'
                        )
                        # The path AHEAD is what the controller refuses, so the
                        # retry must retreat instead of pressing forward again.
                        self._last_attempt_failure = 'wedge_refusals'
                        self._release_nav_control(
                            label, goal_handle, result_future,
                            min(5.0, response_timeout),
                        )
                        return False
                    elif (
                        motion_since is not None
                        and now - motion_since > nomotion_timeout
                    ):
                        self._last_attempt_failure = 'no_motion'
                        self.get_logger().error(
                            f'{label}: base has not moved more than '
                            f'{nomotion_radius:.2f} m in {nomotion_timeout:.0f} s; '
                            'stopping this attempt.'
                        )
                        self._release_nav_control(
                            label, goal_handle, result_future,
                            min(5.0, response_timeout),
                        )
                        return False
            if stall_timeout > 0.0:
                # Abandon an attempt that has stopped closing the distance, so
                # the retry (escape + jam break) still has budget to work with.
                # With a detour chain the base legitimately drives AWAY from the
                # final pose first (the bypass stages south before crossing), so
                # progress is measured against the next waypoint that has not
                # been reached yet, never against the final pose from the start.
                target = self._chain_progress_target(chain, x, y)
                live_distance = self._cached_planar_distance_to_point(*target)
                if live_distance is not None:
                    now = time.monotonic()
                    if (
                        best_distance is None
                        or live_distance < best_distance - stall_progress
                    ):
                        best_distance = live_distance
                        stall_since = now
                    elif stall_since is None:
                        stall_since = now
                    elif now - stall_since > stall_timeout:
                        # 2026-10-07: consult the acceptance gate before giving up.
                        # This path used to cancel the goal unconditionally, so a
                        # base that HAD arrived still failed the leg. Measured on
                        # run_logs/yawfix_20261008-000634 (blue_cube_4/5 -> A):
                        #   dock creep failed at 0.163 m, 153.1 deg off yaw
                        #   dock creep failed at 0.163 m, -125.4 deg off yaw
                        #   dock creep failed at 0.164 m, -141.2 deg off yaw
                        # all inside navigation_accept_radius_m 0.22 m, then the
                        # 45 s stall timer cancelled the goal and the leg died with
                        # the cube attached - the widened yaw gate never got a say,
                        # because _dock_creep's own timeout returns False without
                        # reaching its acceptance check. Widening a tolerance cannot
                        # help while the abandonment path skips it, so check here:
                        # for a stationary base the cached poses are current.
                        base_pose = self._base_pose()
                        if base_pose is not None:
                            here_distance = math.hypot(
                                x - base_pose[0], y - base_pose[1]
                            )
                            here_yaw = math.atan2(
                                math.sin(yaw - base_pose[2]),
                                math.cos(yaw - base_pose[2]),
                            )
                            if (
                                here_distance <= accept_radius
                                and abs(here_yaw) <= accept_yaw
                            ):
                                self.get_logger().warning(
                                    f'{label}: stopped closing the distance for '
                                    f'{stall_timeout:.0f} s but the pose is inside '
                                    f'the accept gate ({here_distance:.3f} m <= '
                                    f'{accept_radius:.3f} m, '
                                    f'{math.degrees(here_yaw):+.1f} deg <= '
                                    f'{math.degrees(accept_yaw):.1f} deg); '
                                    'accepting the arrival instead of abandoning '
                                    'the leg. verify_placement remains the gate.'
                                )
                                self._stop_base()
                                best_distance = live_distance
                                stall_since = now
                                continue
                        # Never publish an escape Twist against a live Nav2
                        # controller. The former in-goal _back_off raced with
                        # FollowPath on /cmd_vel_nav; the outer retry will move
                        # only after this goal's cancellation is confirmed.
                        self._last_attempt_failure = 'no_progress'
                        self.get_logger().warning(
                            f'{label}: no progress for {stall_timeout:.0f} s '
                            f'(still {live_distance:.2f} m from the goal); '
                            'abandoning this attempt so the retry can free the base.'
                        )
                        self._release_nav_control(
                            label, goal_handle, result_future,
                            min(5.0, response_timeout),
                        )
                        return False
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
                # Exact-pose goal (warehouse plate): accept a sustained
                # "close enough and aligned" pose instead of waiting for the
                # goal checker, which MPPI cannot satisfy from every approach
                # direction on this map.
                error = self._cached_goal_error(x, y, yaw)
                if error is not None:
                    goal_distance, yaw_error = error
                    if (
                        dock_creep_trigger > 0.0
                        # Exact-pose plate goals only: clearance waypoints pass a
                        # completion_radius_m and must not be crept, otherwise the
                        # creep times out ~8 s short of the waypoint (round18).
                        and (
                            completion_radius_m is None
                            or completion_radius_m <= 0.0
                        )
                        and goal_distance <= dock_creep_trigger
                    ):
                        self.get_logger().info(
                            f'{label}: base is {goal_distance:.3f} m from the '
                            'plate; releasing Nav2 for the final dock creep.'
                        )
                        self._release_nav_control(
                            label, goal_handle, result_future,
                            min(5.0, response_timeout),
                        )
                        return self._creep_to_dock(label, x, y, yaw)
                    if (
                        accept_radius > 0.0
                        and goal_distance <= accept_radius
                        and abs(yaw_error) <= accept_yaw
                    ):
                        if accept_since is None:
                            accept_since = time.monotonic()
                        elif time.monotonic() - accept_since >= accept_hold:
                            self.get_logger().warning(
                                f'{label} accepted as reached: base is '
                                f'{goal_distance:.3f} m from the goal '
                                f'(<= {accept_radius:.3f} m) and '
                                f'{math.degrees(yaw_error):.1f} deg off yaw '
                                f'(<= {math.degrees(accept_yaw):.1f} deg) for '
                                f'{accept_hold:.1f} s; releasing Nav2.'
                            )
                            cancel_future = goal_handle.cancel_goal_async()
                            cancel_timeout = min(5.0, response_timeout)
                            rclpy.spin_until_future_complete(
                                self, cancel_future, timeout_sec=cancel_timeout
                            )
                            rclpy.spin_until_future_complete(
                                self, result_future, timeout_sec=cancel_timeout
                            )
                            self._stop_base()
                            return True
                    else:
                        accept_since = None
                continue
            distance = self._cached_planar_distance_to_cube(visual_handoff_cube)
            if distance is None or distance > handoff_distance:
                continue

            self.get_logger().info(
                f'Visual handoff reached for {visual_handoff_cube}: '
                f'base-to-cube distance={distance:.3f} m <= '
                f'{handoff_distance:.3f} m; cancelling Nav2 before camera control.'
            )
            # The previous order reversed while FollowPath was still active:
            # two competing publishers drove /cmd_vel_nav at the same time.
            self._release_nav_control(
                label, goal_handle, result_future, min(5.0, response_timeout)
            )
            distance = self._cached_planar_distance_to_cube(visual_handoff_cube)
            if distance is None or distance > handoff_distance:
                self.get_logger().warning(
                    f'{visual_handoff_cube}: visual handoff pose changed while '
                    'cancelling Nav2; retrying navigation.'
                )
                return False
            if distance < min_handoff_distance:
                # Only retreat once cancellation/result have been confirmed.
                distance = self._retreat_to_handoff_standoff(
                    visual_handoff_cube, distance
                )
                if distance < min_handoff_distance or distance > handoff_distance:
                    self.get_logger().error(
                        f'{visual_handoff_cube}: retreat ended outside visual '
                        f'handoff range ({distance:.3f} m); refusing blind servo.'
                    )
                    return False
            for _ in range(3):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info(
                f'Nav2 control released at the {visual_handoff_cube} visual handoff.'
            )
            return True

        if not result_future.done():
            point_distance = self._cached_planar_distance_to_point(x, y)
            if (
                accept_radius > 0.0
                and point_distance is not None
                and point_distance <= accept_radius
            ):
                self.get_logger().warning(
                    f'Navigation to {label} used its whole {timeout:.0f}s budget '
                    f'but the base is only {point_distance:.3f} m from the goal '
                    f'(<= {accept_radius:.3f} m); accepting it as reached.'
                )
                cancel_future = goal_handle.cancel_goal_async()
                cancel_timeout = min(5.0, response_timeout)
                rclpy.spin_until_future_complete(
                    self, cancel_future, timeout_sec=cancel_timeout
                )
                rclpy.spin_until_future_complete(
                    self, result_future, timeout_sec=cancel_timeout
                )
                self._stop_base()
                return True
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

    def _patrols_callback(self, msg: String) -> None:
        """Remember the boxes' lanes so a hold is never taken inside one."""
        segments = obstacle_yield.parse_patrol_segments(msg.data)
        if segments and segments != self._patrol_segments:
            self._patrol_segments = segments
            self.get_logger().warning(
                f'Learned {len(segments)} moving-obstacle lane(s); the base will '
                'not park inside one.'
            )

    def _scan_callback(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_received_at = time.monotonic()

    def _clearer_axis(self) -> float:
        """+1.0 when the front half-plane is clearer than the rear, else -1.0.

        The lidar sits on the base looking forward, so 0 rad is straight ahead.
        Invalid returns are skipped; without a usable sweep this falls back to
        -1.0 (reverse), the historical behaviour.
        """
        scan = self._scan
        if scan is None:
            return -1.0
        front = rear = math.inf
        angle = scan.angle_min
        for value in scan.ranges:
            if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                if -math.pi / 2.0 <= angle <= math.pi / 2.0:
                    front = min(front, value)
                else:
                    rear = min(rear, value)
            angle += scan.angle_increment
        if not math.isfinite(front) and not math.isfinite(rear):
            return -1.0
        return 1.0 if front > rear else -1.0

    def _carried_cube_name(self) -> str | None:
        """Name of the cube currently on the arm, or None when empty-handed.

        Uses the cached link states only (never blocks), and treats "within 1 m
        of the base" as carried - the same test the carried-cube trace uses.
        """
        msg = self._latest_link_states
        if msg is None:
            return None
        names = list(msg.name)
        cube = str(self.get_parameter('cube_name').value).strip()
        try:
            cube_pose = msg.pose[
                names.index(
                    f"{cube}::{self.get_parameter('cube_link').value}"
                )
            ]
            base_pose = msg.pose[
                names.index(
                    f"{self.get_parameter('robot_model_name').value}::"
                    f"{self.get_parameter('robot_base_link').value}"
                )
            ]
        except ValueError:
            return None
        if math.hypot(
            cube_pose.position.x - base_pose.position.x,
            cube_pose.position.y - base_pose.position.y,
        ) <= 1.0:
            return cube
        return None

    def _enter_dead_zone_escape(self, label: str) -> None:
        """Shrink the costmap radius so a trapped pose becomes legal again.

        The radius is clamped while a cube is on the arm. The aggressive escape
        value is *below* the carried envelope (~0.317 m), so Nav2 would happily
        plan the payload straight through geometry it does not fit past -
        measured 2026-10-07 (round36), the base ended up on its side (roll 120
        deg) at warehouse A after an escape-assisted retry. Loaded legs therefore
        never shrink past the carried envelope; the direct, scan-guided shove in
        _back_off is what actually frees a carried base.
        """
        if not bool(self.get_parameter('dead_zone_escape_enabled').value):
            return
        nominal = self._nominal_geometry
        if nominal is None or self._escaping:
            return
        radius, inflation, _ = nominal
        escape = float(self.get_parameter('dead_zone_escape_radius_m').value)
        carried = self._carried_cube_name()
        if carried is not None:
            floor = float(
                self.get_parameter('loaded_escape_min_radius_m').value
            )
            if escape < floor:
                self.get_logger().warning(
                    f'{label}: carrying {carried}; keeping the costmap radius at '
                    f'{floor:.2f} m instead of {escape:.2f} m so the payload is '
                    'never planned through geometry it cannot pass.'
                )
                escape = floor
        if escape <= 0.0 or escape >= radius:
            return
        if not self._set_costmap_geometry(
            escape, max(inflation, escape), 'dead-zone escape', remember=False
        ):
            return
        self._escaping = True
        self.get_logger().warning(
            f'{label}: retry escape requested after navigation failure; '
            f'costmap radius temporarily {radius:.2f} -> {escape:.2f} m. '
            'No local clearance measurement was made; do not infer a dead zone.'
        )

    def _restore_nominal_geometry(self, label: str) -> None:
        if not self._escaping:
            return
        self._escaping = False
        nominal = self._nominal_geometry
        if nominal is None:
            return
        radius, inflation, _ = nominal
        # The leg is measured against this radius, so a failed restore must be
        # loud: the rest of the task would run with a footprint that is too small.
        if not self._set_costmap_geometry(radius, inflation, 'restored'):
            self.get_logger().error(
                f'{label}: could not restore the nominal costmap geometry '
                f'(robot radius {radius:.2f} m).'
            )
        else:
            self.get_logger().info(
                f'{label}: nominal costmap geometry restored '
                f'(robot radius {radius:.2f} m).'
            )

    def _back_off(
        self, label: str, distance_m: float = 0.12, speed: float = 0.08,
        direction: float | None = None,
    ) -> None:
        """Drive briefly along the clearer axis to free a jammed base.

        Published to the smoother input (the same channel the visual servo and
        dock creep use) only after the previous Nav2 goal has a confirmed
        terminal result. The direction comes from the
        lidar: measured 2026-10-07 (round32), always reversing pushed the base
        0.03 m *deeper* into the wall it was already trapped against.

        ``direction`` overrides the lidar choice when the failure itself says
        which way is useless: after the controller refused the path ahead, the
        lidar can still call the front clearer while every attempt to drive
        forward is refused again (measured 2026-10-14 15:34).
        """
        if direction is None:
            direction = self._clearer_axis()
        else:
            direction = 1.0 if direction > 0.0 else -1.0
        command = Twist()
        command.linear.x = direction * abs(speed)
        deadline = time.monotonic() + distance_m / max(abs(speed), 1e-3)
        while rclpy.ok() and time.monotonic() < deadline:
            self._fine_cmd_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._stop_base()
        for _ in range(3):
            self._stop_base()
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warning(
            f'{label}: moved {distance_m:.2f} m '
            f'{"forward" if direction > 0 else "backward"} to break a possible '
            'jam before retrying.'
        )

    def _parse_slow_zones(self):
        """Read the tight-passage speed caps and create the publisher lazily.

        Returns [] and creates nothing unless the parameter is set, so a default
        configuration behaves exactly as before: no extra publisher and no
        message ever sent to the controller.
        """
        try:
            raw = list(self.get_parameter('slow_zones').value or [])
        except Exception:
            raw = []
        zones = []
        if len(raw) < 4:
            # Includes the [0.0] "off" sentinel and an absent parameter.
            raw = []
        if len(raw) % 4 != 0:
            raise RuntimeError(
                f'slow_zones needs groups of 4 (x, y, radius_m, speed_mps); '
                f'got {len(raw)} values'
            )
        for index in range(0, len(raw), 4):
            x, y, radius, speed = (float(v) for v in raw[index:index + 4])
            if not (radius > 0.0) or not (0.0 < speed <= 1.5):
                raise RuntimeError(
                    f'slow_zones entry {index // 4} is invalid: radius must be '
                    f'>0 and speed in (0, 1.5], got radius={radius} speed={speed}'
                )
            zones.append((x, y, radius, speed))
        # Yield also uses this channel. Without this condition, an enabled
        # policy with no slow zones would claim to hold while publishing NO
        # speed-limit message at all.
        if zones or bool(self.get_parameter('enable_obstacle_yield').value):
            self._speed_limit_pub = self.create_publisher(
                SpeedLimit, 'speed_limit', 10
            )
        if zones:
            self.get_logger().warning(
                f'{len(zones)} slow zone(s) configured; the base will be capped '
                'while inside them.'
            )
        return zones

    def _wedge_log(self, msg) -> None:
        """Remember when the controller last refused to move for collision."""
        try:
            if msg.name != 'controller_server':
                return
            if 'detected collision ahead' not in msg.msg:
                return
        except AttributeError:
            return
        self._controller_collisions.append(time.monotonic())

    def _wedge_collisions_since(self, since: float) -> int:
        window = float(self.get_parameter('wedge_collision_window_sec').value)
        cutoff = max(since, time.monotonic() - window)
        while self._controller_collisions and self._controller_collisions[0] < cutoff:
            self._controller_collisions.popleft()
        return len(self._controller_collisions)

    # ------------------------------------------------------------------ timers
    def _clear_costmaps(self, *a, **k):
        with self._op('svc:clear_costmaps'):
            return self._clear_costmaps__timed(*a, **k)

    def _set_costmap_geometry(self, *a, **k):
        with self._op('svc:set_costmap_geometry'):
            return self._set_costmap_geometry__timed(*a, **k)

    def _attach(self, *a, **k):
        with self._op('svc:attach'):
            return self._attach__timed(*a, **k)

    def _detach(self, *a, **k):
        with self._op('svc:detach'):
            return self._detach__timed(*a, **k)

    def _wait_for_dependencies(self, *a, **k):
        with self._op('deps:discovery'):
            return self._wait_for_dependencies__timed(*a, **k)

    def _creep_to_grasp(self, *a, **k):
        with self._op('creep_to_grasp'):
            return self._creep_to_grasp__timed(*a, **k)
    # Thin wrappers so each discrete operation reports its own cost without
    # touching the body of the original implementation.
    def _execute_trajectory(self, label, *a, **k):
        with self._op(f'arm:{label}'):
            return self._execute_trajectory__timed(label, *a, **k)

    def _creep_to_dock(self, label, *a, **k):
        with self._op(f'creep:{label}'):
            return self._creep_to_dock__timed(label, *a, **k)

    def _visual_align_active(self, *a, **k):
        with self._op('vision_align'):
            return self._visual_align_active__timed(*a, **k)

    def _navigate_once(self, label, *a, **k):
        with self._op(f'nav:{label}'):
            return self._navigate_once__timed(label, *a, **k)

    def _spin_for_fresh_detection(self, deadline: float, stale_limit: float) -> None:
        """Spin until a new camera frame has been processed, or briefly give up.

        Bounded so a dead camera cannot hang the loop: it returns after
        `timeout` seconds even without a frame, and the caller then sees a stale
        detection and takes its existing lost-blob path.
        """
        last = self._visual_detection_time
        timeout = 0.25
        end = min(deadline, time.monotonic() + timeout)
        while rclpy.ok() and time.monotonic() < end:
            if self._visual_detection_time > last:
                return
            rclpy.spin_once(self, timeout_sec=0.02)

    def _log_cube_positions(self, when: str) -> None:
        """Log every cube's XY so displacement after placement is detectable.

        Requested to check whether the base shoves an already-placed cube while
        driving to the next one: verify_placement only looks at the cube it just
        released, so a cube moved later was previously invisible. Uses the cached
        Gazebo LinkStates the node already receives.
        """
        msg = self._latest_link_states
        if msg is None:
            return
        for name in msg.name:
            if 'cube' not in name:
                continue
            try:
                index = msg.name.index(name)
                pose = msg.pose[index]
            except (ValueError, IndexError):
                continue
            self.get_logger().info(
                f'CUBEPOS {when} {name} {pose.position.x:.4f} {pose.position.y:.4f}'
            )

    @contextlib.contextmanager
    def _op(self, label: str):
        """Time one discrete operation and log it as a parseable TIMING line.

        Requested so every operation's cost can be accounted for offline rather
        than inferred from stage totals. Deliberately logs at INFO with a fixed
        shape so run_logs/tools/op_timings.py can aggregate it; it changes no
        control behaviour.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self.get_logger().info(
                f'TIMING {label} {time.monotonic() - started:.3f}'
            )

    def _publish_speed_limit(self, limit: float, reason: str = '') -> None:
        """Send a speed cap to the controller, publishing only on change.

        Publishing only on change is not an optimisation: the controller server
        logs every limit it receives, and a 10 Hz stream of identical caps buries
        the round record it is supposed to explain.
        """
        if self._speed_limit_pub is None or limit == self._active_speed_limit:
            return
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.percentage = False
        message.speed_limit = limit
        self._speed_limit_pub.publish(message)
        previous = self._active_speed_limit
        self._active_speed_limit = limit
        if reason:
            position = self._cached_base_position()
            where = ('' if position is None
                     else f' near ({position[0]:.2f}, {position[1]:.2f})')
            self.get_logger().warning(
                f'{reason}: speed limit {previous:.2f} -> {limit:.2f} m/s{where}.'
            )

    def _slow_zone_limit(self) -> float:
        """Speed cap demanded by the configured tight passages, 0.0 for none.

        The cap applies ``slow_zone_brake_ahead_m`` *before* the boundary, not
        only inside it. Measured 2026-10-13 (run_logs/full_round_yield_20261013-195911):
        the base entered a tight passage at the full 1.5 m/s, the cap only bit
        1.2 s later, and the stop distance (0.45 m braking + 0.45 m of control
        latency) was about the distance left to the wall - the base hit it and
        was thrown at 3.3 m/s, failing the round on the first delivery.
        """
        if not self._slow_zones:
            return 0.0
        position = self._cached_base_position()
        if position is None:
            return 0.0
        lookahead = float(self.get_parameter('slow_zone_brake_ahead_m').value)
        limit = 0.0
        for x, y, radius, speed in self._slow_zones:
            distance = math.hypot(position[0] - x, position[1] - y) - radius
            if distance <= lookahead:
                limit = speed if limit == 0.0 else min(limit, speed)
        return limit

    def _apply_slow_zones(self) -> None:
        """Cap the controller's speed while the base is inside a tight passage.

        Nav2 exposes this properly: controller_server subscribes to
        ``nav2_msgs/msg/SpeedLimit`` and calls ``setSpeedLimit()`` on the active
        controller plugin (RegulatedPurePursuitController honours it by
        rewriting desired_linear_vel). A cap is always the cautious direction,
        but it exists for a concrete reason rather than as blanket caution:

        blue_cube_3 -> A can only reach zone A through one passage whose chord
        clearance is 0.318 m against the 0.35 m loaded footprint - the route was
        chosen by flood fill and the margin is documented in
        fixed_joint_pick_place.yaml. At 1.5 m/s the tracking error exceeds that
        margin, the footprint clips the wall, RPP then refuses to move (and the
        recovery behaviours refuse too, because their footprint test sees the
        same intrusion), and the leg burns the whole navigation stall timeout
        before the dead-zone escape frees it: measured
        (12.778,-3.790) at 0.340 m from the nearest lethal cell, i.e. 1 cm
        inside, costing 45 s (run_logs/speed_rpp15_combined_trial5_20261010-173315).
        Slow down there and the passage costs seconds instead.
        """
        limit = self._slow_zone_limit()
        self._publish_speed_limit(limit, 'Slow zone' if limit > 0.0 else '')

    def _clear_speed_limit(self) -> None:
        """Always hand the controller back its own speed limit."""
        self._reset_yield_state()
        if self._speed_limit_pub is None or self._active_speed_limit == 0.0:
            return
        message = SpeedLimit()
        message.header.stamp = self.get_clock().now().to_msg()
        message.percentage = False
        message.speed_limit = 0.0
        self._speed_limit_pub.publish(message)
        self._active_speed_limit = 0.0

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _yield_base_motion(self) -> tuple[float, float, float, float, float]:
        """(vx, vy, wz, position x, position y) in the odom frame.

        Ground-truth pose is used for the base position rather than the odom
        message for consistency with every other decision in this node, but the
        *velocity* comes from odometry because Gazebo's link states carry no
        twist and differencing them amplifies the link-state publish jitter.
        """
        pose = self._base_pose()
        if pose is None or self._odom is None:
            return 0.0, 0.0, 0.0, math.nan, math.nan
        twist = self._odom.twist.twist
        return twist.linear.x, twist.linear.y, twist.angular.z, pose[0], pose[1]

    def _yield_scan_points(self, yaw: float):
        """Latest sweep as base-frame (x, y, range), or None when unusable.

        Points belonging to a tracked mover are returned separately so the room
        estimate can ignore them: a box crossing on the left is not the wall the
        base would be squeezing past if it took the arc.
        """
        scan = self._scan
        if scan is None:
            return None, []
        if scan.range_max < 0.5 or scan.angle_increment == 0.0:
            return None, []
        now = self._clock_seconds()
        stamp = float(scan.header.stamp.sec) + float(scan.header.stamp.nanosec) * 1e-9
        # An unstamped sweep must not be treated as a fresh measurement.
        age = max(0.0, now - stamp) if stamp > 0.0 else math.inf
        if age > CLOCK_MISMATCH_SEC:
            # The node clock and the sweep are on different bases (this node was
            # started with use_sim_time false while the sweep carries Gazebo's
            # simulation time, or the reverse). Measured 2026-10-13: the age read
            # 1791884787 s, every tick took the "stale scan" exit, and the yield
            # policy was completely inert for a whole round WITHOUT ONE LOG LINE.
            # The question the guard below actually asks is "has this node heard
            # a sweep recently", so answer that with the monotonic receipt time
            # and say loudly what happened instead of failing silently.
            if not self._clock_mismatch_reported:
                self._clock_mismatch_reported = True
                self.get_logger().error(
                    f'Clock mismatch: sweep stamp {stamp:.3f} is '
                    f'{age:.0f} s from this node\'s clock {now:.3f}; using the '
                    'monotonic receipt age instead. Check use_sim_time on this '
                    'node (every Nav2 node in this stack has it true).'
                )
            age = (math.inf if self._scan_received_at is None
                   else max(0.0, time.monotonic() - self._scan_received_at))
        points = []
        for index, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            if value < scan.range_min or value > scan.range_max:
                continue
            angle = scan.angle_min + index * scan.angle_increment
            points.append((value * math.cos(angle), value * math.sin(angle), value))
        return age, points

    def _yield_room(self, points, excluded) -> tuple[float | None, float | None]:
        """Lateral room beside the current path, (left, right), from the scan.

        Deliberately measured from the *lidar* rather than the costmap: the
        local costmap is inflated for planning (0.36 m) and carries the movers
        themselves, so reading it would report the box as the wall. Sectors are
        sampled 60 deg either side of the beam, and the reported room is the
        perpendicular component `min_range * cos(30 deg)`, i.e. how far the base
        could move sideways before its own radius touches what is there.
        """
        half_width = math.radians(30.0)
        centres = {'left': math.radians(75.0), 'right': math.radians(-75.0)}
        room = {}
        for name, centre in centres.items():
            best = None
            for x, y, rng in points:
                angle = math.atan2(y, x)
                if abs(angle - centre) > half_width:
                    continue
                if any(math.hypot(x - ex, y - ey) < self._yield_config.obstacle_half_width_m * 2.0
                       for ex, ey in excluded):
                    continue
                best = rng if best is None else min(best, rng)
            if best is None:
                room[name] = None
            else:
                room[name] = max(0.0, best * math.cos(half_width) - 0.2)
        return room['left'], room['right']

    def _yield_step(self, distance_to_goal_m: float | None):
        """One policy tick. Returns (decision, latched, arc goal in map frame)."""
        motion_vx, motion_vy, wz, base_x, base_y = self._yield_base_motion()
        pose = self._base_pose()
        yaw = pose[2] if pose is not None else math.nan
        age, points = self._yield_scan_points(yaw)
        if (points is None or not self._yield_config.enabled
                or age > self._yield_config.track_timeout_sec
                or not all(math.isfinite(v) for v in (base_x, base_y, yaw))):
            # Losing a scan during a hold must NOT publish the 0.0 'release'
            # sentinel immediately: that would restore full cruise on no new
            # evidence. Continue holding only within the same finite budget;
            # once exhausted, hand control back to Nav2's collision checking.
            elapsed = (time.monotonic() - self._yield_wait_since
                       if self._yield_wait_since is not None else 0.0)
            self._yield_diag(age if age is not None else -1.0,
                             [] if points is None else points, [], [], None,
                             obstacle_yield.YieldDecision(
                                 obstacle_yield.ACTION_GO,
                                 obstacle_yield.REASON_CLEAR,
                                 detail='no fresh pose/scan'),
                             time.monotonic())
            if (self._yield_config.enabled and self._yield_latched
                    and elapsed < self._yield_config.wait_budget_sec):
                return (obstacle_yield.YieldDecision(
                    obstacle_yield.ACTION_STOP, obstacle_yield.REASON_WAIT,
                    wait_elapsed_sec=elapsed, detail='awaiting fresh pose/scan'
                ), True, None)
            return (obstacle_yield.YieldDecision(
                obstacle_yield.ACTION_GO, obstacle_yield.REASON_CLEAR,
                detail='no fresh pose/scan'), False, None)

        # Tracking every laser return makes a wall fragment the nearest 'box'
        # and can associate opposite ends of one obstacle across frames. Track
        # compact connected components instead; each becomes one centroid.
        # Only returns inside the tracking range are clustered: beyond it the
        # lidar's 1 deg spacing exceeds the cluster gap, so a distant wall
        # shatters into hundreds of tiny "box-like" clusters - measured
        # 2026-10-13 as 17-41 box-like clusters per sweep in a room holding
        # three boxes, and the noise tracks they created were what the policy
        # then reacted to.
        clusters = obstacle_yield.cluster_points(
            [(math.atan2(y, x), rng) for x, y, rng in points
             if rng <= self._yield_config.track_range_m])
        boxes = obstacle_yield.box_like_clusters(clusters)
        centroids = [(
            sum(r * math.cos(a) for a, r in c) / len(c),
            sum(r * math.sin(a) for a, r in c) / len(c),
        ) for c in boxes]
        now = time.monotonic()
        self._yield_tracker.update(
            centroids, now, (base_x, base_y, yaw, motion_vx, motion_vy, wz))
        track = self._yield_tracker.nearest(
            now, (base_x, base_y, yaw, motion_vx, motion_vy, wz),
            min_world_speed=self._yield_config.min_mover_speed_mps)
        excluded = [] if track is None else [
            (track['base_x'], track['base_y'])]
        left_room, right_room = self._yield_room(points, excluded)

        # The budget is spent against the whole EPISODE, not against the current
        # hold. A box whose lane the base is standing in re-enters the corridor
        # every few seconds, so each individual hold is short and the per-hold
        # timer never reaches the budget - measured 2026-10-13 the base sat
        # 0.30 m from mover_3's segment for 18 s of zero motion (98 controller
        # refusals) through 30 alternating hold/resume cycles, and the lateral
        # escape never fired because no single hold lasted 20 s.
        wait_elapsed = self._yield_episode_elapsed(now)
        if track is None:
            obs = None
        else:
            obs = obstacle_yield.YieldObservation(
                box_x=track['base_x'], box_y=track['base_y'],
                box_vx=track['rel_vx'], box_vy=track['rel_vy'],
                vx=motion_vx, wz=wz,
                left_room_m=left_room, right_room_m=right_room,
                distance_to_goal_m=distance_to_goal_m,
                scan_age_sec=age,
                box_world_speed_mps=track['world_speed'],
                latched=self._yield_latched,
                lane_distance_m=obstacle_yield.distance_to_lane(
                    base_x, base_y, self._patrol_segments),
                # A patrol box reverses at its endpoints; its measured velocity
                # says "leaving" right up to the turn (measured 2026-10-13).
                lane_end_distance_m=obstacle_yield.distance_to_lane_end(
                    track['x'], track['y'], self._patrol_segments),
                lane_end_closing=obstacle_yield.closing_on_lane_end(
                    track['x'], track['y'], track['vx'], track['vy'],
                    self._patrol_segments),
            )
        decision = obstacle_yield.decide(
            obs, self._yield_config,
            wait_elapsed_sec=wait_elapsed,
            arc_attempts=self._yield_arc_attempts,
        )
        self._yield_diag(age, points, clusters, boxes, track, decision, now)
        latched = decision.action == obstacle_yield.ACTION_STOP
        arc_goal = None
        if decision.action == obstacle_yield.ACTION_ARC and decision.arc_x is not None:
            static_returns = [
                (x, y) for x, y, _ in points
                if not any(math.hypot(x - ex, y - ey) <= 0.22
                           for ex, ey in excluded)
            ]
            if obstacle_yield.arc_scan_clear(
                static_returns, decision.arc_x, decision.arc_y,
                radius_m=self._yield_config.base_half_width_m + 0.05,
            ):
                arc_goal = (
                    base_x + math.cos(yaw) * decision.arc_x
                    - math.sin(yaw) * decision.arc_y,
                    base_y + math.sin(yaw) * decision.arc_x
                    + math.cos(yaw) * decision.arc_y,
                )
            else:
                decision = obstacle_yield.YieldDecision(
                    obstacle_yield.ACTION_GO,
                    obstacle_yield.REASON_WAIT_EXPIRED,
                    detail='arc approach blocked or unobserved by lidar',
                )
        if self._yield_config.enabled and decision.action != obstacle_yield.ACTION_GO:
            self._yield_log_decision(decision, obs)
        return decision, latched, arc_goal

    def _yield_diag(self, age, points, clusters, boxes, track, decision,
                    now) -> None:
        """One rate-limited line telling the tracker apart from the policy.

        2026-10-13: a full round with `enable_obstacle_yield=True` logged no
        YIELD line at all, but the decision log only fires on a non-GO action -
        so "the policy said go" and "the tracker never produced a track" looked
        identical. This line carries the counts that separate them: raw returns,
        clusters, box-like clusters, and whether a track cleared the mover-speed
        gate. Logging only; it commands nothing.
        """
        if (self._yield_diag_period <= 0.0
                or not self._yield_config.enabled        # off means off, silently
                or now - self._yield_diag_last < self._yield_diag_period):
            return
        self._yield_diag_last = now
        if track is None:
            detail = (f'scan_age={age:.2f} points={len(points)} '
                      f'clusters={len(clusters)} box_like={len(boxes)} track=None')
        else:
            detail = (f'scan_age={age:.2f} points={len(points)} '
                      f'clusters={len(clusters)} box_like={len(boxes)} '
                      f'track=({track["base_x"]:+.2f},{track["base_y"]:+.2f}) '
                      f'rel=({track["rel_vx"]:+.2f},{track["rel_vy"]:+.2f}) '
                      f'world={track["world_speed"]:.2f} life={track["life"]:.1f}')
        self.get_logger().warning(
            f'YIELD_DIAG {detail} decision={decision.action}'
            f'/{decision.reason}')

    def _yield_log_decision(self, decision, obs) -> None:
        """Rate-limited yield logging, in a shape the round record can grep."""
        now = time.monotonic()
        if now - self._yield_last_log < 1.0:
            return
        self._yield_last_log = now
        box = 'n/a' if obs is None else (
            f'({obs.box_x:+.2f},{obs.box_y:+.2f}) v=({obs.box_vx:+.2f},'
            f'{obs.box_vy:+.2f}) world={obs.box_world_speed_mps}'
        )
        self.get_logger().warning(
            f'YIELD {decision.action} ({decision.reason}) box {box} '
            f'wait={decision.wait_elapsed_sec:.1f}s: {decision.detail}'
        )

    def _clock_seconds(self) -> float:
        """The same clock the scan stamps use (sim time under Gazebo)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _yield_episode_elapsed(self, now: float) -> float:
        """Seconds spent in the current yield episode (0.0 when there is none)."""
        if self._yield_episode_since is None:
            return 0.0
        return max(0.0, now - self._yield_episode_since)

    def _update_yield_episode(self, now: float, latched: bool) -> None:
        """Track one obstacle episode, ended by real progress - not by a lull.

        A box whose lane the base is standing in leaves the corridor and comes
        back every few seconds, so the base alternates hold/resume indefinitely
        while its per-hold timer keeps restarting. Measured 2026-10-13: 30 such
        cycles, 18 s of frozen base (98 controller refusals) parked 0.30 m from
        mover_3's patrol segment, and the lateral escape never fired because no
        single hold ever reached the 20 s budget. The episode therefore ends
        only when the base has actually got somewhere: ``episode_reset_progress_m``
        of travel since the hold that started it.
        """
        position = self._cached_base_position()
        if not latched:
            if self._yield_episode_since is not None:
                if self._yield_free_since is None:
                    self._yield_free_since = now
                elif now - self._yield_free_since >= self._yield_config.episode_clear_sec:
                    self._yield_episode_since = None
                    self._yield_episode_anchor = None
                    self._yield_free_since = None
        else:
            self._yield_free_since = None
        if (self._yield_episode_since is not None
                and position is not None and self._yield_episode_anchor is not None):
            moved = math.hypot(position[0] - self._yield_episode_anchor[0],
                               position[1] - self._yield_episode_anchor[1])
            if moved >= self._yield_config.episode_reset_progress_m:
                self._yield_episode_since = None
                self._yield_episode_anchor = None
        if latched and self._yield_episode_since is None:
            self._yield_episode_since = now
            self._yield_episode_anchor = position

    def _tick_yield(self, distance_to_goal_m: float | None = None) -> None:
        """Run the yield policy and apply its speed decision to the controller.

        Both speed sources are merged rather than fought over: the tight-passage
        cap and the yield cap are each conservative, so the lower one wins.
        """
        if self._yield_suppress:
            self._apply_slow_zones()
            return
        now = time.monotonic()
        period = float(self.get_parameter('yield_update_period_sec').value)
        if now - self._yield_last_update < period:
            return
        self._yield_last_update = now
        decision, latched, arc_goal = self._yield_step(distance_to_goal_m)
        # A single unmatched scan is not evidence that the crossing is clear.
        # Keep the cap until it has been clear for a short, continuous window,
        # but never extend the overall 20 s budget for a missing sensor.
        if (self._yield_latched and decision.action == obstacle_yield.ACTION_GO
                and decision.reason != obstacle_yield.REASON_WAIT_EXPIRED
                and self._yield_wait_since is not None
                and now - self._yield_wait_since < self._yield_config.wait_budget_sec):
            if self._yield_clear_since is None:
                self._yield_clear_since = now
            if now - self._yield_clear_since < self._yield_config.clear_hold_sec:
                decision = obstacle_yield.YieldDecision(
                    obstacle_yield.ACTION_STOP, obstacle_yield.REASON_WAIT,
                    detail='confirming clear corridor',
                )
                latched = True
        else:
            self._yield_clear_since = None
        self._yield_decision = decision
        self._update_yield_episode(now, latched)
        if latched:
            if self._yield_wait_since is None:
                self._yield_wait_since = now
        elif decision.action == obstacle_yield.ACTION_ARC:
            self._yield_wait_since = None
            # One outstanding arc at a time, and a hard ceiling per leg: the
            # arc is consumed by the retry loop, and this flag is what keeps the
            # loop from consuming a fresh one before the previous has been driven.
            if (arc_goal is not None and self._yield_arc_goal is None
                    and not self._yield_arc_used
                    and self._yield_arc_attempts < self._yield_config.max_arc_attempts):
                self._yield_arc_goal = arc_goal
        else:
            self._yield_wait_since = None
        self._yield_latched = latched

        limit = self._slow_zone_limit()
        reason = 'Slow zone'
        if decision.action == obstacle_yield.ACTION_CAP:
            if limit == 0.0 or decision.speed_mps < limit:
                limit = decision.speed_mps
                reason = 'Yield cap'
        elif decision.action in (
            obstacle_yield.ACTION_STOP, obstacle_yield.ACTION_ARC
        ):
            # Keep the hold in force while the arc is being queued and the
            # original Nav2 goal is being canceled. Only release it after
            # cancellation is confirmed and the detour goal is ready.
            # A stop must NOT be sent as 0.0. In this Nav2 version
            # `speed_limit = 0.0` is `nav2_costmap_2d::NO_SPEED_LIMIT`, i.e.
            # "clear the cap and restore the controller's own
            # desired_linear_vel" - measured against the installed binary:
            # nav2_regulated_pure_pursuit_controller 1.1.20 `setSpeedLimit`
            # compares against 0.0 and copies `base_desired_linear_vel_` (1.5
            # m/s here) when it matches. Sending 0.0 to hold would therefore
            # make the base accelerate. The limit is applied as
            # `clamp(|linear_vel|, 0, desired_linear_vel_)` *after* RPP's
            # min-speed floors, so a small non-zero value really does bind.
            limit = YIELD_STOP_SPEED_LIMIT
            reason = 'Yield hold'
        self._publish_speed_limit(limit, reason if limit > 0.0 else '')

    def _pending_yield_arc(self):
        """Take the pending yield arc waypoint, if the policy asked for one.

        Consuming it here (rather than only reading it) is what makes the retry
        loop terminate: each arc is driven once, and the per-leg ceiling in
        ``_yield_arc_attempts`` stops a moving obstacle from being chased around
        the map one offset at a time.
        """
        arc = self._yield_arc_goal
        if arc is None:
            return None
        self._yield_arc_goal = None
        self._yield_arc_attempts += 1
        return arc

    def _reset_yield_state(self) -> None:
        """Forget the yield episode. Called when a navigation leg ends.

        The policy is deliberately amnesiac across legs: a wait budget that
        survived into the next leg would let a second obstacle inherit the first
        one's spent patience.
        """
        self._yield_latched = False
        self._yield_wait_since = None
        self._yield_clear_since = None
        self._yield_arc_attempts = 0
        self._yield_arc_goal = None
        self._yield_arc_used = False
        self._yield_pause_active = False
        self._yield_episode_since = None
        self._yield_episode_anchor = None
        self._yield_free_since = None
        self._yield_decision = obstacle_yield.YieldDecision(
            obstacle_yield.ACTION_GO, obstacle_yield.REASON_CLEAR
        )

    def _navigate_to(
        self,
        label: str,
        values: Sequence[float],
        visual_handoff_cube: str | None = None,
        completion_radius_m: float | None = None,
        chain: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        attempts = max(1, int(self.get_parameter('navigation_retry_count').value) + 1)
        # A canceled goal followed by one yield waypoint must still get a
        # chance to retry the original goal, even when retries are disabled.
        if self._yield_config.enabled:
            attempts += 1
        self._log_cube_positions(f'before:{label}')
        # Progress along the detour chain belongs to the LEG, not to one attempt:
        # a retry after a hold must not send the base back to a waypoint it has
        # already reached (measured 2026-10-14: seven holds in one leg, each
        # resume re-sending the full chain).
        self._chain_index = 0
        try:
            for attempt in range(1, attempts + 1):
                # Each attempt gets a clean failure fingerprint: the escape after
                # a failed attempt reads it to choose a direction.
                self._last_attempt_failure = None
                # A yield arc is not a failure: the leg was abandoned because a
                # moving obstacle owned the lane, not because the base was
                # wedged. Drive the lateral waypoint the policy picked, then
                # retry the same goal WITHOUT the dead-zone escape and jam break
                # below, which would shove a perfectly healthy base backwards
                # through the obstacle it just avoided.
                arc = self._pending_yield_arc()
                if arc is not None:
                    self.get_logger().warning(
                        f'{label}: yielding around a moving obstacle via '
                        f'({arc[0]:.2f}, {arc[1]:.2f}) before retrying.'
                    )
                    self._yield_arc_used = True
                    # The old goal has been canceled. Lift the near-zero hold
                    # before driving the lateral goal, and do not re-enter the
                    # same yield episode while that waypoint is active.
                    self._yield_suppress = True
                    try:
                        self._apply_slow_zones()
                        arc_ok = self._navigate_once(
                            f'{label} yield arc',
                            (arc[0], arc[1], values[2]), None,
                            float(self.get_parameter(
                                'yield_arc_goal_radius_m').value),
                        )
                    finally:
                        self._yield_suppress = False
                    if not arc_ok or not self._clear_costmaps():
                        return False
                    continue
                if self._navigate_once(
                    label, values, visual_handoff_cube, completion_radius_m,
                    chain,
                ):
                    self.get_logger().info(f'Navigation stage complete: {label}.')
                    time.sleep(float(self.get_parameter('settle_sec').value))
                    return True
                if self._yield_arc_goal is not None:
                    # This leg ended deliberately for a moving-obstacle waypoint,
                    # not because the base is wedged. The next iteration consumes
                    # the arc; never back off through the mover or shrink the
                    # loaded footprint on this path.
                    continue
                if attempt < attempts:
                    self.get_logger().warning(
                        f'Retrying {label} navigation ({attempt}/{attempts - 1}) '
                        'after clearing costmaps.'
                    )
                    # A failed leg usually means the base is wedged, and the
                    # commonest wedge is the costmap itself: inside robot_radius
                    # of a wall no rollout is legal, so the retry cannot move at
                    # all. Shrink the radius first, then shove along the clearer
                    # axis, then clear and retry against the relaxed costmap.
                    self._enter_dead_zone_escape(label)
                    # Break a physical jam before trying again. Measured 2026-10-07
                    # (round29, carried leg): 18 x "Failed to make progress" with ZERO
                    # "No valid trajectories" - the controller had valid paths, but
                    # the carried cube/arm was caught on a wall, and the plain retry
                    # only drove forward into the same jam until the task died with
                    # UNSAFE_OBJECT_STATE.
                    self._back_off(
                        label,
                        distance_m=float(
                            self.get_parameter('escape_distance_m').value
                        ),
                        speed=float(
                            self.get_parameter('escape_speed_mps').value
                        ),
                        # A refusal-abandoned attempt means the path ahead is what
                        # the controller rejects. Retreat along the heading
                        # instead of trusting a lidar front/rear comparison that
                        # keeps calling that same path clear.
                        direction=(-1.0 if self._last_attempt_failure
                                   == 'wedge_refusals' else None),
                    )
                    if not self._clear_costmaps():
                        return False
            return False
        finally:
            # Never leave a shrunk footprint behind: the next stage (and the
            # placement drop) is calibrated against the nominal radius.
            self._restore_nominal_geometry(label)
            # ...and never leave the controller throttled either: the next leg
            # must get its own desired_linear_vel back.
            self._clear_speed_limit()

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
        # Drive any declared clearance waypoints before the final approach so a
        # concave corner never forces DWB into a large turn at zero clearance.
        for index in (1, 2):
            parameter = f'{cube}_pickup_via_{index}'
            if not self.get_parameter(parameter).value:
                continue
            waypoint = self._double_list(parameter, 3)
            if not self._navigate_to(
                f'{cube} pickup clearance waypoint {index}',
                waypoint,
                completion_radius_m=0.35,
            ):
                return False
            if not self._clear_costmaps():
                return False
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
        zone = warehouse.lower()
        waypoints = self._detour_waypoints(cube, zone)
        if waypoints and bool(self.get_parameter('use_continuous_detour').value):
            # ONE goal for the whole detour: Nav2 plans a single path through
            # every waypoint and the leg's own target, so the base follows the
            # configured route without stopping at each point. This is also why
            # the route on screen matches the YAML: the per-waypoint walk used
            # to re-plan from scratch at every stop, and each yield hold
            # cancelled and resumed a different goal.
            if not self._navigate_to(
                f'{cube} to {warehouse} continuous detour',
                values,
                chain=waypoints,
            ):
                return False
            return True
        for index, waypoint in enumerate(waypoints, start=1):
            if not self._navigate_to(
                f'{cube} to {warehouse} detour waypoint {index}',
                waypoint,
                completion_radius_m=0.35,
            ):
                return False
            if not self._clear_costmaps():
                return False
        # Arrive aligned. Without this the global plan can end facing any
        # direction (round16: 155 deg off on red_cube_4, 43 deg on blue_cube_1)
        # and DWB then burns 8-24 s rotating in place inside the goal tolerance
        # before it may declare success - 56 s across the five loaded legs. The
        # approach point sits 1.5 m in front of the zone facing the zone yaw, so
        # the last leg is a straight run with almost no heading error.
        if bool(self.get_parameter('use_aligned_warehouse_approach').value):
            approach = self._double_list(
                f'warehouse_{warehouse.lower()}_approach_pose', 3
            )
            if not self._navigate_to(
                f'{warehouse} aligned approach',
                approach,
                completion_radius_m=0.30,
            ):
                return False
            if not self._clear_costmaps():
                return False
        return self._navigate_to(f'warehouse {warehouse}', values)

    def _detour_waypoints(self, cube: str, zone: str) -> list[list[float]]:
        """The configured detour points for this cube's delivery to a zone.

        Declared as <cube>_warehouse_<zone>_via_<n>. Any cube may carry them;
        the walk stops at the first undeclared (or too short) parameter, so a
        cube without waypoints keeps the direct route exactly as before.
        """
        waypoints = []
        for index in range(1, 5):
            parameter = f'{cube}_warehouse_{zone}_via_{index}'
            if not self.has_parameter(parameter):
                break
            raw = self.get_parameter(parameter).value
            if raw is None or len(list(raw)) != 3:
                break
            waypoints.append(self._double_list(parameter, 3))
        return waypoints

    def _visual_align(self) -> bool:
        """Servo the base using the colour contour, then hand control back.

        Nav2 publishes zero velocity through the smoother after a goal, so the
        servo writes to the smoother's *input* topic (``cmd_vel_nav``). The
        smoother forwards servo commands to ``/cmd_vel``. Nav2's recovery
        behavior_server can also publish directly to ``/cmd_vel``; this method
        relies on confirmed Nav2 goal termination, not exclusive topic ownership.
        """
        if not bool(self.get_parameter('enable_visual_alignment').value):
            self.get_logger().warning('Visual fine alignment is disabled.')
            return True
        self._visual_range_gate_active = True
        # Turn the HSV pipeline on only for the servo window.
        self._vision_enabled = True
        try:
            return self._visual_align_active()
        finally:
            self._vision_enabled = False
            self._visual_range_gate_active = False
            # Several zero setpoints flush the servo command out of the
            # smoother's queue before the next Nav2 goal starts.
            for _ in range(3):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=0.1)

    def _visual_align_active__timed(self) -> bool:
        cube = str(self.get_parameter('cube_name').value).lower()
        self._target_colour = 'blue' if cube.startswith('blue') else 'red'
        self._visual_detection = None
        self._publish_vision_state(f'SEARCHING:{self._target_colour}')
        deadline = time.monotonic() + float(
            self.get_parameter('visual_timeout_sec').value
        )
        stable = 0
        required = max(1, int(self.get_parameter('visual_stable_frames').value))
        standoff_frames = max(1, int(
            self.get_parameter('visual_standoff_frames').value
        ))
        span_limit = float(self.get_parameter('visual_stable_span_px').value)
        # Rows of the frames that currently agree, newest last.
        stable_rows: list[float] = []
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
        min_linear = float(self.get_parameter('visual_min_linear_speed').value)
        max_angular = float(self.get_parameter('visual_max_angular_speed').value)
        # Phase 1: search. Turn slowly until the colour blob is in view, so a
        # coarse parking error does not abort the whole pick.
        search_rate = float(self.get_parameter('visual_search_rate').value)
        search_timeout = float(
            self.get_parameter('visual_search_timeout_sec').value
        )
        search_command = Twist()
        search_command.angular.z = search_rate

        def search_until_visible() -> bool:
            """Rotate until a fresh blob is seen. True if one is in view."""
            search_deadline = min(deadline, time.monotonic() + search_timeout)
            while rclpy.ok() and time.monotonic() < search_deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                detection = self._visual_detection
                if (
                    detection is not None
                    and time.monotonic() - self._visual_detection_time <= stale_limit
                ):
                    return True
                self._fine_cmd_pub.publish(search_command)
                self._publish_vision_state(f'SEARCHING:{self._target_colour}')
            return False

        # 2026-10-10: the search is re-entered whenever the blob is lost, instead
        # of sitting still until the timeout. Measured on the 1.5 m/s rounds
        # (run_logs/speed_rpp15_trial1_20261010-143655 and
        # speed_rpp15_udp_trial1_20261010-144630): a fast handoff left the cube at
        # centre_error -0.80 (near the frame edge), the centring motion pushed it
        # out of view, and this loop then only called _stop_base() and waited -
        # a single ALIGNING sample followed by 72 s of silence and
        # "Visual fine alignment timed out". 29 slower rounds never lost the blob,
        # which is why the hole went unnoticed. Re-searching can only help find
        # the cube; every acceptance gate below is unchanged.
        lost_since = None
        research_after = float(
            self.get_parameter('visual_research_after_sec').value
        )
        # Phase timing, logging only: the alignment costs 6.5-12 s per cube
        # (40.6 s across a 5-object round, all of it inside the navigation
        # states) and the loop's own gates give no clue which phase owns it.
        search_started = time.monotonic()
        search_total = 0.0
        align_started = 0.0
        stable_frames_used = 0
        while rclpy.ok() and time.monotonic() < deadline:
            if not search_until_visible():
                break
            search_total += time.monotonic() - search_started
            self._stop_base()
            align_started = time.monotonic()
            # Phase 2: centre and approach.
            while rclpy.ok() and time.monotonic() < deadline:
                # Consume one CAMERA frame per iteration, not "the next callback
                # in the queue". Measured 2026-10-11: the detection pipeline
                # costs 12.1 ms/frame (cv_bridge 2.0 + HSV 10.1) so this node can
                # process ~82 Hz, the camera publishes at 10 Hz, yet this loop ran
                # at only 2.4-3.1 Hz because a single spin_once() serviced one
                # arbitrary callback from a queue dominated by the high-rate
                # gazebo LinkStates/odom subscriptions. The per-iteration travel is
                # speed/rate - 0.06 m/s at 3 Hz is 20 mm per step against an
                # acceptance band of +/-9 px (~6 mm) - so the rate, not the speed
                # cap, is what forces the crawl (raising visual_max_linear_speed
                # 0.15 -> 0.30 changed nothing). Waiting for a fresh frame lets the
                # same control law converge at the camera's own rate.
                self._spin_for_fresh_detection(deadline, stale_limit)
                detection = self._visual_detection
                if (
                    detection is None
                    or time.monotonic() - self._visual_detection_time > stale_limit
                ):
                    stable = 0
                    self._stop_base()
                    now = time.monotonic()
                    if lost_since is None:
                        lost_since = now
                    elif now - lost_since >= research_after:
                        # Stop waiting out the timeout: go back and look again.
                        self.get_logger().warning(
                            'Visual blob lost for '
                            f'{now - lost_since:.1f} s; restarting the search.'
                        )
                        lost_since = None
                        search_started = time.monotonic()
                        break
                    continue
                lost_since = None
                centre_x, width, area, image_width, bottom_row = detection
                centre_error = (centre_x - image_width / 2.0) / (image_width / 2.0)
                # Bottom row grows as the robot approaches: it is the range error.
                row_error = (approach_row - bottom_row) / approach_row
                centred = abs(centre_error) <= centre_tolerance
                at_range = abs(bottom_row - approach_row) <= row_tolerance
                if centred and at_range:
                    stable += 1
                    stable_rows.append(float(bottom_row))
                    if len(stable_rows) > standoff_frames:
                        stable_rows.pop(0)
                    self._stop_base()
                    if stable >= required:
                        span = max(stable_rows) - min(stable_rows)
                        # Accept once the agreeing frames really agree. If they
                        # never settle, take their median anyway rather than hold
                        # out for a gate that may not come: the median of several
                        # readings still beats the last one, and the deadline
                        # above bounds the loop, so this cannot hang.
                        if span <= span_limit or stable >= required * 3:
                            used_row = float(np.median(stable_rows))
                            stable_frames_used = stable
                            self.get_logger().info(
                                f'VISALIGN search={search_total:.1f}s '
                                f'align={time.monotonic() - align_started:.1f}s '
                                f'stable={stable} frames'
                            )
                            self.get_logger().info(
                                f'Visual alignment complete: '
                                f'centre_error={centre_error:.3f}, '
                                f'bottom_row={used_row:.0f} '
                                f'(median of {len(stable_rows)} frames, '
                                f'span {span:.0f} px, last {bottom_row:.0f}).'
                            )
                            return self._creep_to_grasp(used_row)
                        # Readings disagree: keep servoing and collect more.
                    continue

                stable = 0
                stable_rows.clear()
                command = Twist()
                # Two-stage approach (2026-10-07): d(row)/d(distance) is about
                # 1500 px/m at the 0.19 m standoff, so 0.15 m/s moves the bottom row
                # ~15 px per frame while the accepted band is only +-9.3 px: the
                # servo overshot the band and never collected 3 stable frames
                # (measured as a 75 s alignment timeout). Go fast while far, gentle
                # inside the last quarter of the row error.
                effective_max = max_linear
                if abs(row_error) < 0.25:
                    effective_max = min(max_linear, 0.06)
                vx = max(-0.5 * effective_max, min(effective_max, linear_gain * row_error))
                # A pure P term crawls below 1 cm/s over the last few centimetres,
                # which dominated the 40 s alignment measured on the red_cube_1 leg;
                # hold a floor while still outside the tolerance band so the tail
                # takes ~1 s instead of ~5 s. _creep_to_grasp re-measures the
                # standoff afterwards, so a small overshoot is recovered there.
                if row_error > 0.0 and vx < min_linear and effective_max > min_linear:
                    vx = min_linear
                command.linear.x = vx
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

    def _creep_to_grasp__timed(self, bottom_row: float) -> bool:
        """Close the last centimetres the level camera cannot measure.

        Below roughly 0.20 m the nearest cube's floor contact leaves the image,
        so the range signal saturates before the gripper is over the cube. The
        remaining distance is computed from the last measured row instead of a
        fixed guess, so any stop inside the band lands link6 on the cube.
        """
        k = float(self.get_parameter('visual_row_distance_k').value)
        row_center = float(self.get_parameter('visual_row_center').value)
        offset = float(self.get_parameter('visual_row_distance_offset_m').value)
        target = float(self.get_parameter('visual_target_standoff_m').value)
        standoff = offset + k / max(1.0, bottom_row - row_center)
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
            f'({offset_x:+.3f}, {offset_y:+.3f}) m, effective acceptance '
            f'limits [±{half_length - tolerance:.3f} x '
            f'±{half_width - tolerance:.3f}] m'
        )
        if inside:
            # Only a cube that actually landed inside counts towards the slot
            # rotation, so a failed delivery does not consume a slot.
            warehouse = str(self.get_parameter('warehouse').value).upper()
            self._zone_deliveries[warehouse] = \
                self._zone_deliveries.get(warehouse, 0) + 1
            self.get_logger().info(
                f'{message}, inside limits. '
                f'(zone {warehouse}: {self._zone_deliveries[warehouse]} placed)')
            return True
        self.get_logger().error(f'Cube missed warehouse {warehouse}. '
                                f'{message}, OUTSIDE limits.')
        return False

    def _refresh_link_states(self) -> bool:
        """Refresh the cached Gazebo LinkStates, with a bounded retry.

        Every caller that needs a FRESH link state must go through here. There
        used to be two independent refresh sites - _link_pose and _link_offset -
        and when only the first was given a retry, a round still died at
        "Timed out waiting for /gazebo/link_states" in VERIFY_GRASP while the
        topic was publishing happily at 18 Hz. One shared helper is the only way
        to keep that from being re-learned a third time.
        """
        timeout = float(self.get_parameter('proximity_timeout_sec').value)
        attempts = max(1, int(self.get_parameter('proximity_retry_attempts').value))
        for attempt in range(1, attempts + 1):
            self._latest_link_states = None
            if self._wait_until(
                lambda: self._latest_link_states is not None,
                timeout,
                '/gazebo/link_states',
            ):
                return True
            if attempt < attempts:
                self.get_logger().warning(
                    f'No fresh Gazebo link states for {timeout:.0f} s; '
                    f'retrying ({attempt}/{attempts - 1}).'
                )
        return False

    def _link_pose(self, model: str, link: str):
        """Return a Gazebo link pose, refreshing cached link states first."""
        if not self._refresh_link_states():
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

    def _link_offset(self) -> tuple[float, float, float] | None:
        """Cube centre relative to the robot end link, in the end-link frame.

        ``finger_joint1`` slides along the end link's y axis, so the y component
        of this offset is exactly what decides how far the finger may travel
        before it touches the cube: with 10 mm thick fingers the inner opening is
        ``0.065 - finger_joint1`` and the cube is 0.03 m wide.
        """
        if not self._refresh_link_states():
            return None

        model1, link1, model2, link2 = self._link_request_values()
        try:
            robot_index = list(self._latest_link_states.name).index(
                f'{model1}::{link1}'
            )
            cube_index = list(self._latest_link_states.name).index(
                f'{model2}::{link2}'
            )
        except ValueError:
            return None

        robot_pose = self._latest_link_states.pose[robot_index]
        cube_pose = self._latest_link_states.pose[cube_index]
        dx = cube_pose.position.x - robot_pose.position.x
        dy = cube_pose.position.y - robot_pose.position.y
        dz = cube_pose.position.z - robot_pose.position.z
        return self._rotate_by_inverse(robot_pose.orientation, (dx, dy, dz))

    @staticmethod
    def _rotate_by_inverse(quaternion, vector) -> tuple[float, float, float]:
        """Rotate ``vector`` by the inverse of ``quaternion`` (unit, xyzw)."""
        qx, qy, qz, qw = (
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
        # Conjugate == inverse for a unit quaternion.
        cx, cy, cz, cw = -qx, -qy, -qz, qw
        vx, vy, vz = vector
        tx = 2.0 * (cy * vz - cz * vy)
        ty = 2.0 * (cz * vx - cx * vz)
        tz = 2.0 * (cx * vy - cy * vx)
        return (
            vx + cw * tx + (cy * tz - cz * ty),
            vy + cw * ty + (cz * tx - cx * tz),
            vz + cw * tz + (cx * ty - cy * tx),
        )

    def _link_distance(self) -> float | None:
        # The helper discards cached data, so verification observes a state
        # published after the preceding attach, arm motion or navigation.
        if not self._refresh_link_states():
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
        offset = self._link_offset()
        if offset is not None:
            # dy is the finger-closing axis: the cube must satisfy
            # dy + 0.015 <= 0.0325 (fixed finger) for the grip not to squeeze it.
            self.get_logger().info(
                f'Pre-attach cube offset in link6 frame: dx={offset[0]:+.3f}, '
                f'dy={offset[1]:+.3f}, dz={offset[2]:+.3f} m.'
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

    def _attach__timed(self) -> bool:
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

    def _detach__timed(self) -> bool:
        request = DetachLink.Request()
        (
            request.model1_name,
            request.link1_name,
            request.model2_name,
            request.link2_name,
        ) = self._link_request_values()
        return self._call_link_service('DETACH', self._detach_client, request)

    def _confirm_detached(self) -> bool:
        """Decide from the world whether a timed-out DETACH already happened.

        Lifting link6 separates the two cases: a released cube stays on the
        plate while the link rises (distance grows past the attached limit), a
        still-gripped cube follows the link (distance stays small).
        """
        if not self._move_arm('Lift to confirm release', 'lift_joints'):
            return False
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.3)
        distance = self._link_distance()
        if distance is None:
            return False
        maximum = float(self.get_parameter('max_attached_distance_m').value)
        if distance > maximum:
            self.get_logger().warning(
                f'DETACH was not answered, but the cube stayed behind '
                f'({distance:.3f} m > {maximum:.3f} m) after lifting; treating '
                'it as released.'
            )
            return True
        return False

    def _call_link_service(self, label: str, client, request) -> bool:
        self.get_logger().info(
            f'{label}: {request.model1_name}::{request.link1_name} <-> '
            f'{request.model2_name}::{request.link2_name}'
        )
        timeout = float(self.get_parameter('service_timeout_sec').value)
        attempts = max(1, int(self.get_parameter('link_service_attempts').value))
        for attempt in range(1, attempts + 1):
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                break
            # ROS service requests cannot be cancelled, so the request may still
            # land later. Decide from the world state instead of giving up:
            if label == 'ATTACH':
                # The lift + attached-distance check that follows is the real
                # verification, so carrying on is safe: a cube that did not get
                # attached simply fails that check with a *known* state.
                self.get_logger().warning(
                    f'{label}: no answer within {timeout:.0f}s; continuing and '
                    'letting the lift/attachment check decide.'
                )
                return True
            if self._confirm_detached():
                return True
            if attempt < attempts:
                self.get_logger().warning(
                    f'{label}: no answer within {timeout:.0f}s '
                    f'(attempt {attempt}/{attempts}); retrying.'
                )
                continue
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

    def _goal_callback(self, msg: String) -> None:
        """Queue a supervisor goal (resident mode only)."""
        raw = msg.data.strip()
        if not raw:
            return
        try:
            goal = json.loads(raw)
        except ValueError as exc:
            self.get_logger().error(f'Rejected unusable executor goal: {exc}')
            return
        if goal.get('cancel'):
            self._cancel_requested = True
            self.get_logger().warning('Cancellation requested by the supervisor.')
            return
        self._goal_queue.append(goal)

    def _apply_goal(self, goal: dict):
        """Point the node's cube/warehouse parameters at the queued object."""
        cube = str(goal.get('cube', '')).strip()
        zone = str(goal.get('zone', '')).strip().upper()
        if not cube or zone not in {'A', 'B', 'C'}:
            self.get_logger().error(f'Rejected invalid executor goal: {goal}')
            return None
        self.set_parameters(
            [
                RclpyParameter('cube_name', RclpyParameter.Type.STRING, cube),
                RclpyParameter('warehouse', RclpyParameter.Type.STRING, zone),
            ]
        )
        # A detection from the previous object must never leak into this one.
        self._visual_detection = None
        self._visual_detection_time = 0.0
        return cube

    def _publish_result(self, goal: dict, success: bool) -> None:
        if self._result_pub is None:
            return
        payload = {
            'cube': str(goal.get('cube', '')),
            'zone': str(goal.get('zone', '')),
            'object_id': int(goal.get('object_id', 0)),
            'success': bool(success),
            'error': '' if success else self._last_error,
        }
        message = String()
        message.data = json.dumps(payload)
        self._result_pub.publish(message)

    def run_resident(self) -> None:
        """Serve one goal per cube without re-creating the process."""
        self.get_logger().info(
            'Executor resident mode: waiting for /competition/executor_goal ...'
        )
        while rclpy.ok():
            if not self._goal_queue:
                rclpy.spin_once(self, timeout_sec=0.2)
                continue
            goal = self._goal_queue.popleft()
            cube = self._apply_goal(goal)
            if cube is None:
                self._publish_result(goal, False)
                continue
            self._cancel_requested = False
            self._last_error = ''
            self.get_logger().info(
                f"Resident goal {cube} -> {goal.get('zone', '')} "
                f"(object_id={goal.get('object_id', 0)})."
            )
            success = False
            try:
                success = self.run_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.get_logger().error(f'Resident object raised an exception: {exc}')
                self._last_error = f'EXCEPTION:{exc}'
                success = False
            if self._cancel_requested:
                success = False
                self._last_error = self._last_error or 'CANCELLED_BY_SUPERVISOR'
            self._publish_result(goal, success)
        self.get_logger().info('Executor resident loop finished.')

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
            # Check the landing position IMMEDIATELY after release, before the
            # 40 s of retreat/return-home arm motion. Measured 2026-10-12
            # (twotier5, blue_cube_5 -> A): the carry was verified attached at
            # the dock (0.070 m of a 0.120 m limit), yet the single end-of-
            # sequence check found the cube 1.9 m away and could not say whether
            # it was placed badly or knocked away afterwards. This early check
            # answers that, and reports a bad placement ~40 s sooner.
            ('VERIFY_PLACE', 'verify placement', self._verify_placement),
            (
                'VERIFY_PLACE', 'retreat',
                lambda: self._move_arm('Retreat from place', 'pre_place_joints'),
            ),
            ('VERIFY_PLACE', 'return home', lambda: self._move_arm('Return home', 'home_joints')),
            # Final confirmation in the resting position: a cube that the
            # retreat later nudged out of the plate still counts as missed.
            ('VERIFY_PLACE', 'verify placement after retreat',
             self._verify_placement),
        ]

        attached = False
        released = False
        unsafe_state = False
        for state, name, operation in steps:
            if self._cancel_requested:
                self.get_logger().warning(
                    'Object cancelled by the supervisor; skipping the remaining '
                    'steps.'
                )
                self._stop_base()
                self._publish_executor_error('CANCELLED_BY_SUPERVISOR')
                return False
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
    if node.resident():
        try:
            node.run_resident()
        except (KeyboardInterrupt, ValueError) as exc:
            node.get_logger().error(f'Resident executor aborted: {exc}')
        finally:
            if rclpy.ok():
                node.destroy_node()
                rclpy.shutdown()
        return
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

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
import queue
import re
import sys
import threading
import time
from collections import deque
from dataclasses import fields, is_dataclass, replace
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
from std_srvs.srv import Empty
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import (
    Parameter,
    ParameterDescriptor,
    ParameterType,
    ParameterValue,
)
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter as RclpyParameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
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


def _yield_trace_json(value):
    """Strict JSON: non-finite numbers and unavailable values become null.

    No arbitrary object representation is emitted: records have an explicit
    allowlist and never contain tasks, API keys, or track history.
    """
    def clean(item):
        if item is None or isinstance(item, (bool, str, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if is_dataclass(item) and not isinstance(item, type):
            return {f.name: clean(getattr(item, f.name)) for f in fields(item)}
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        return None
    return json.dumps(clean(value), allow_nan=False, separators=(',', ':'))


class _YieldTraceWriter:
    """Bounded best-effort diagnostics; logger I/O never runs on the tick.

    Full queue / oversized lines are dropped. This is not a replay log: sample
    construction and enqueue still cost CPU, OS scheduling can perturb timing,
    and the daemon may lose queued records on exit.
    """
    def __init__(self, logger):
        self.queue = queue.Queue(maxsize=64)
        self.logger = logger
        self.dropped = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='yield-trace-writer')
        self._thread.start()

    def submit(self, record):
        if self._stopped.is_set():
            return
        try:
            # This count excludes serialization/logging failures and shutdown loss.
            record['queue_full_drops_before'] = self.dropped
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self):
        """Stop promptly; queued samples are intentionally discarded.

        A logger call already in progress cannot be cancelled by Python. If it
        blocks, the bounded join expires and its daemon can outlive the node.
        """
        self._stopped.set()
        try:
            self.queue.put_nowait(None)  # wake a sleeping consumer; full is OK
        except queue.Full:
            pass
        self._thread.join(timeout=0.05)

    def _run(self):
        try:
            while not self._stopped.is_set():
                record = self.queue.get()
                if record is None or self._stopped.is_set():
                    break
                try:
                    line = _yield_trace_json(record)
                    if (not self._stopped.is_set()
                            and len(line.encode('utf-8')) <= 16384):
                        self.logger.warning('YIELD_TICK_JSON ' + line)
                except Exception:
                    # Diagnostics must not terminate the control path.
                    pass
        finally:
            self.logger = None


def dock_creep_twist(
    dx, dy, yaw, base_yaw, distance, speed, forward_gain, yaw_tolerance,
    align_distance, lateral_band=0.06, visual=None,
):
    """One step of the final dock creep, as ``(linear_x, angular_z)``.

    Split out of ``_creep_to_dock__timed`` so the law can be tested without a
    live node. The shape matters more than it looks:

    * Far field (``distance > align_distance``): point at the goal and drive.
    * Near field: rotate in place while the heading is far off (a translating
      spin used to orbit the dock), then translate.
    * 2026-11-16: inside ``align_distance`` the old law only ever drove
      FORWARD, ``max(0.02, min(speed, gain * max(forward, 0)))``, so a residual
      that was mostly SIDEWAYS collapsed to the 0.02 m/s floor and the base
      parked short. Measured on run_logs/competition_lanes_v02: two warehouse A
      docks accepted at 0.210 m (two runs in a row, one at 0.210 m), which threw
      the released cube 0.153 m off in y and failed VERIFY_PLACE. While the
      lateral error is above ``lateral_band`` the creep now points at the goal
      again and drives with a 0.05 m/s floor, so a sideways 0.2 m closes in a
      couple of seconds; only the last few centimetres (lateral <= band, dock
      ahead) use the plate-yaw-hold law the drop point is calibrated against.
    """
    distance = max(0.0, float(distance))
    yaw_error = math.atan2(math.sin(yaw - base_yaw), math.cos(yaw - base_yaw))
    # 2026-11-25: the geometric law below re-aims at the goal every time the
    # lateral residual leaves `lateral_band`, and each re-aim turns the base -
    # that is the back-and-forth hunt at the berth ("小车会来回摆动几下").
    # When the warehouse plate is in view its centroid is a direct measurement
    # of the same lateral error the law is inferring from dead reckoning, so
    # use it: same heading and lateral gains, minus a centroid term.  `visual`
    # is (fill, centroid_x_normalised) or None; None keeps the old law exactly.
    if visual is not None and distance <= align_distance * 1.5:
        fill, centroid = visual
        if fill is not None and abs(yaw_error) <= 0.6:
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            wz = 2.0 * yaw_error + 1.5 * lateral - 0.8 * centroid
            return (
                max(0.03, min(speed, forward_gain * max(forward, 0.05))),
                max(-0.8, min(0.8, wz)),
            )
    if distance > align_distance:
        bearing = math.atan2(dy, dx)
        bearing_error = math.atan2(
            math.sin(bearing - base_yaw), math.cos(bearing - base_yaw)
        )
        if abs(bearing_error) > 0.6:
            return 0.0, max(-1.0, min(1.0, 1.5 * bearing_error))
        return (
            max(0.03, min(speed, forward_gain * distance)),
            max(-1.0, min(1.0, 2.0 * bearing_error)),
        )
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    # 2026-11-16: the base can arrive at the plate facing anything from -161 to
    # -21 deg (trace: competition_lanes_v02h). Unwinding 160 deg at the 0.45
    # rad/s fine-trim clamp took 20 s and pushed the base 0.2 m past the plate
    # on the way, so the trim clamp is reached in two stages: swing fast while
    # the error is large, trim slowly only near the end.
    if abs(yaw_error) > 0.6:
        return 0.0, max(-1.0, min(1.0, 1.2 * yaw_error))
    # 2026-11-16: the near-field turn used 2.0*yaw_error clamped at 0.8 rad/s.
    # The loop spins at 20 Hz in real time but the sim runs at ~0.2 RTF, so the
    # effective control period is ~0.25 s of sim time and one step of that law
    # turned ~11 deg past the target: run_logs/competition_lanes_v02b ended two
    # creeps at 22.6 and -41.1 deg off after 26 s of hunting. Gain 1.5 with a
    # 0.45 rad/s clamp and a 0.04 rad deadband converges instead of orbiting.
    # 2026-11-16: the lateral residual has to win while it is still large. With
    # the yaw-first order below, a dock that ended 0.106 m off to the side and
    # 19.7 deg off in heading stayed in "rotate only" (19.7 > 4 * tolerance),
    # and rotating can never close a sideways offset: competition_lanes_v02j
    # spent its whole 26 s budget at d=0.154 m. Aim at the plate and creep while
    # the heading is still sane, and let joint1 absorb the rest (limit 34 deg).
    if abs(lateral) > lateral_band and abs(yaw_error) < 1.0:
        bearing = math.atan2(dy, dx)
        bearing_error = math.atan2(
            math.sin(bearing - base_yaw), math.cos(bearing - base_yaw)
        )
        return (
            max(0.05, min(speed, forward_gain * distance)),
            max(-1.0, min(1.0, 2.0 * bearing_error)),
        )
    if abs(yaw_error) > max(0.04, 4.0 * yaw_tolerance):
        return 0.0, max(-0.45, min(0.45, 1.0 * yaw_error))
    if abs(lateral) > lateral_band or forward <= 0.02:
        bearing = math.atan2(dy, dx)
        bearing_error = math.atan2(
            math.sin(bearing - base_yaw), math.cos(bearing - base_yaw)
        )
        return (
            max(0.05, min(speed, forward_gain * distance)),
            max(-1.0, min(1.0, 2.0 * bearing_error)),
        )
    return (
        max(0.02, min(speed, forward_gain * max(forward, 0.0))),
        max(-0.8, min(0.8, 2.0 * yaw_error + 1.5 * lateral)),
    )


def entry_recovery_pose(raw) -> list[float] | None:
    """Parse `<x, y, yaw?>` for a zone-entrance waiting point.

    An empty list, the `[0.0]` "off" sentinel and anything non-finite mean
    "no waiting point is configured" and return None, so the caller keeps the
    old retry ladder exactly as it was.
    """
    try:
        values = [float(value) for value in list(raw or [])]
    except (TypeError, ValueError):
        return None
    if len(values) < 2:
        return None
    if not all(math.isfinite(value) for value in values[:3]):
        return None
    if len(values) == 2:
        values.append(0.0)
    return values[:3]


def describe_pose(pose) -> str:
    """One log field for a start/end pose, for post-round diagnosis.

    2026-11-21 review: a retry that re-enters from the pose the previous attempt
    failed at repeats the same approach, and that is invisible unless the pose
    and heading of every attempt are on the record.
    """
    if pose is None:
        return 'pose unknown'
    return (
        f'({pose[0]:.2f}, {pose[1]:.2f}) heading '
        f'{math.degrees(float(pose[2])):+.0f} deg'
    )


# 2026-11-25: the warehouse floor plates, read straight out of
# src/yzbot/mybot_description/worlds/offic_room.world - zone_a diffuse
# "0 1 0", zone_b "1 1 0", zone_c "0 1 1". Hue in degrees, so that the table
# can be checked against the world file without doing OpenCV arithmetic.
# (0 1 1) is 180 deg, not 90: 90 was the first guess and it cost a round -
# warehouse C measured fill=0.00 for 6 s while A measured 0.96.
ZONE_PLATE_HUE_DEG = {'A': 120.0, 'B': 60.0, 'C': 180.0}

# What the competition actually stands on: `zone_<x>_plate` in the executor
# config plus the letter that the plate carries.
ZONE_PLATE_COLOUR = {'A': 'green', 'B': 'yellow', 'C': 'cyan'}


def zone_plate_hue_window(house: str, tolerance_deg: float) -> tuple[int, int] | None:
    """OpenCV (0-179) hue window of one warehouse floor plate.

    Returns None for an unknown warehouse, which the caller treats as "no
    plate to check" rather than as a failure. A window that runs past 179 is
    returned wrapped (lo > hi) and the mask code ORs the two halves.
    """
    centre = ZONE_PLATE_HUE_DEG.get(str(house).strip().upper())
    if centre is None:
        return None
    hue = int(round(centre / 2.0))
    half = max(1, int(round(float(tolerance_deg) / 2.0)))
    return (hue - half) % 180, (hue + half) % 180


def plate_band_metrics(
    hsv, rows_fraction: float, band_fraction: float,
    hue_window, min_saturation: float, min_value: float,
):
    """Fill ratio and centroid of the plate colour in the bottom band.

    2026-11-25: the gate only needs two numbers - how much of the bottom of
    the frame is the warehouse colour (are we close enough / is it the right
    warehouse) and where its centroid sits horizontally (are we centred).
    The band is deliberately the bottom few percent of the image: the plate is
    0.01 m off the floor, so it is only in view when the base is nearly
    there, and the black letter standing on the plate is excluded by the hue
    window rather than by geometry.

    Returns (fill, centroid_x_normalised, rows, columns, dominant_hue) with
    centroid relative to the band centre in [-1, 1]. fill is None when there is
    no window to apply. dominant_hue is the most common hue among the pixels
    that pass the saturation/value gates, or -1 when none do - it is what makes
    a wrong-colour miss diagnosable from the log alone instead of from a
    replay.
    """
    if hue_window is None:
        return None, 0.0, 0, 0, -1
    height, width = hsv.shape[:2]
    rows = int(round(height * float(rows_fraction)))
    rows = max(1, min(height, rows))
    columns = int(round(width * float(band_fraction)))
    columns = max(1, min(width, columns))
    left = (width - columns) // 2
    roi = hsv[height - rows:height, left:left + columns]
    if roi.size == 0:
        return None, 0.0, rows, columns, -1
    hue = roi[:, :, 0].astype(np.int16)
    saturated = (roi[:, :, 1] >= min_saturation) & (roi[:, :, 2] >= min_value)
    dominant_hue = -1
    lit = hue[saturated]
    if lit.size:
        counts = np.bincount(lit, minlength=180)
        dominant_hue = int(counts.argmax())
    lo, hi = hue_window
    if lo <= hi:
        mask = (hue >= lo) & (hue <= hi)
    else:
        mask = (hue >= lo) | (hue <= hi)
    mask = mask & saturated
    total = int(mask.sum())
    fill = total / float(mask.size)
    if total == 0:
        return 0.0, 0.0, rows, columns, dominant_hue
    weights = mask.sum(axis=0).astype(np.float64)
    xs = np.arange(weights.size, dtype=np.float64)
    centroid = float((weights * xs).sum() / weights.sum())
    # Centre on the middle pixel, not on columns/2: an evenly filled band must
    # read exactly 0.0, otherwise every pass would carry a 1/columns bias that
    # eats into the tolerance.
    middle = (columns - 1) / 2.0
    half = max(1.0, middle)
    return fill, (centroid - middle) / half, rows, columns, dominant_hue


def travel_clearance(
    ranges, angle_min, angle_increment, range_min, range_max, direction,
):
    """Nearest valid return in the half-plane the base is about to travel.

    ``direction`` is +1.0 for forward and -1.0 for reverse, the same split
    ``_clearer_axis`` uses (the lidar looks forward along the base x axis).
    Returns None when no usable return falls in that half-plane, which the
    caller must read as "cannot verify" and never as "clear".

    Split out of the node so the retreat guard can be tested without a live
    scan: the 2026-11-21 failed round reversed 0.70 m on one stale comparison
    and moved 0.03 m, and the fix is a per-tick check against this function.
    """
    forward = float(direction) >= 0.0
    nearest = math.inf
    angle = float(angle_min)
    for value in ranges:
        if math.isfinite(value) and range_min <= value <= range_max:
            in_front = -math.pi / 2.0 <= angle <= math.pi / 2.0
            if in_front == forward:
                nearest = min(nearest, float(value))
        angle += angle_increment
    return nearest if math.isfinite(nearest) else None


def classify_entrance_failure(
    *, start_in_band: bool, failure: str | None, moved_m: float,
    elapsed_sec: float, plan_abort_sec: float,
) -> str:
    """Name what a failed narrow-entrance attempt actually was.

    Measured 2026-11-21 (red_cube_1 -> C): three different faults produced the
    same "navigation failed" line, and only the first one is fixed by backing
    the base up:

    * ``band``       - the base itself is inside the planner's inscribed band,
      so NavFn rejects the START cell before it looks at any route. Nothing
      about the corridor or the map can be inferred from this attempt; the base
      has to leave the band first.
    * ``controller`` - the controller refused to track a route it was given
      ("detected collision ahead"). Whether the planner produced that route is
      NOT inferred here: the caller probes the planner separately, because a
      NavigateToPose failure cannot tell the two apart on its own.
    * ``refused``    - the attempt returned quickly and the base did not move.
      On its own that is not proof of a planning refusal (a broken behaviour
      tree fails just as fast), so the caller re-probes and treats a second
      probe failure as a static block instead of waiting for movers.
    * ``stalled``    - the attempt ran for a while without moving. Same probe
      follows; it is separated only so the log says which shape it had.
    * ``progress``   - the base actually moved (`moved_m` exceeds the deadband)
      and then failed, so the entrance geometry was not what stopped it.
    """
    if start_in_band:
        return 'band'
    if failure == 'wedge_refusals':
        return 'controller'
    if float(moved_m) <= 0.10:
        return 'refused' if float(elapsed_sec) <= float(plan_abort_sec) else 'stalled'
    return 'progress'


class FixedJointPickPlace(Node):
    """Run a configurable, single-cube fixed joint trajectory."""

    def destroy_node(self):
        writer = getattr(self, '_yield_trace_writer', None)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            self._yield_trace_writer = None
        return super().destroy_node()

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
        # 停稳后到方块的距离低于此值就必须先退避：相机的行先验在这里已经
        # 跑出画面，探测会把所有轮廓都拒掉，症状只是 "Visual blob lost"。
        self.declare_parameter('visual_handoff_coast_min_distance_m', 0.45)
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
        # and never failed. The y slots stay at +/-0.12 because the heading term
        # is now compensated in joint1 and the position term is held to <=0.10 m
        # by dock_creep_accept_m, so 0.12+0.10 = 0.22 m fits the +/-0.235 half
        # width. 2026-11-16: the old 0.22 m acceptance let a 0.21 m residue stand
        # and put a cube at -0.257 m, 0.022 m outside the box.
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
        self.declare_parameter('guarded_navigation_recovery', False)
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
        # Red cube 1 -> C: operator requested a Nav2-planned route without
        # fixed intermediate points. Keep explicit numeric overrides possible.
        for index in (1, 2):
            self.declare_parameter(
                f'red_cube_1_warehouse_c_via_{index}', [], waypoint_descriptor
            )
        # No forced blue_cube_3 -> A detour: let Nav2 plan around the walls and
        # live obstacles. Dynamic typing still permits explicit numeric routes.
        for index in (1, 2, 3):
            self.declare_parameter(
                f'blue_cube_3_warehouse_a_via_{index}', [], waypoint_descriptor
            )
        # blue_cube_5 sits behind the 1.10 m mouth between the east end of the
        # wall at y ~ -1.12 and the north end of the wall at x ~ -2.85. A mover
        # that parks in that mouth blocks the local costmap, the direct leg
        # from warehouse B then inches at the wall tip until the 45 s progress
        # watchdog gives up (measured 2026-11-22, rule 3: two attempts, 241.8 s
        # and 184.6 s, 62 % of the round for one cube). This outer route leaves
        # B westwards through the 1.5 m gap at x ~ -12.9, turns south of the
        # wall at y ~ -4.4 and rejoins the goal from the west, never entering
        # the mouth. Verified on room_from_world.pgm: waypoint clearance
        # 1.40/1.01/0.75 m and 0.85/1.00/0.58 m straight-chord clearance, so
        # only the last leg back to the goal is left to the planner. Driven at
        # most once per pickup, and only after a failed attempt.
        pickup_fallback_defaults = {
            'blue_cube_5': [
                [-10.60, -3.00, math.pi],
                [-12.30, -3.40, -2.91],
                [-12.95, -4.45, -0.20],
            ],
        }
        for cube in pickup_defaults:
            route = pickup_fallback_defaults.get(cube, [])
            for index in (1, 2, 3):
                self.declare_parameter(
                    f'{cube}_pickup_fallback_via_{index}',
                    route[index - 1] if index <= len(route) else [],
                    waypoint_descriptor,
                )
        self.declare_parameter('pickup_fallback_enabled', True)
        # The fallback only has to carry the base past the mouth; the goal is
        # attacked again from there, so 0.35 m is plenty and matches the
        # pickup clearance waypoints.
        self.declare_parameter('pickup_fallback_completion_radius_m', 0.35)
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
        # Opt-in soft planning costs only. Zero preserves the per-leg inflation.
        self.declare_parameter('global_inflation_min_m', 0.0)
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
        # Experimental outer shoulder, off unless both values are positive.
        # Width must be <= brake-ahead distance. Set cruise to the verified
        # uncapped Nav2 desired_linear_vel; RPP's absolute limit replaces it.
        self.declare_parameter('slow_zone_approach_width_m', 0.0)
        self.declare_parameter('slow_zone_approach_cruise_mps', 0.0)
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
        # Never reduce the controller footprint or drive blind after a timeout.
        self.declare_parameter('dead_zone_escape_enabled', True)
        self.declare_parameter('escape_distance_m', 0.25)
        self.declare_parameter('escape_speed_mps', 0.08)
        # One replan when only the extra GLOBAL margin blocks a known local pose.
        self.declare_parameter('stuck_relax_global_radius', True)
        # 2026-11-21, P0: a zone whose entrance is narrower than the margined
        # footprint needs a recovery of its own. red_cube_1 -> C drove 0.30 m
        # from a wall tip (the mouth there is 0.89 m wide), which is inside the
        # inscribed band of BOTH the 0.45 m planned radius and the 0.35 m local
        # one; every retry then re-planned from that illegal start cell, so the
        # relax could not apply, the 0.70 m shove moved the base 0.03 m, and
        # both retries failed the same way (0/5, 174 s). With these parameters
        # the final dock leg instead retreats out of the band on a verified
        # lidar axis, stages at a clearance-verified waiting point outside the
        # mouth and retries inside a bounded window - without shrinking the
        # loaded footprint or clearing the costmaps again. Off by default: an
        # empty or single-value wait pose keeps the historical retry ladder
        # untouched.
        self.declare_parameter('entry_recovery_enabled', False)
        self.declare_parameter('entry_recovery_zone', 'C')
        self.declare_parameter('entry_recovery_stall_timeout_sec', 18.0)
        # [x, y, yaw]; validated against the static map by
        # src/yzbot/mybot/test/test_entry_recovery.py, which fails if the point
        # is closer to an obstacle than the loaded envelope.
        self.declare_parameter('entry_recovery_wait_pose', [0.0])
        # A second staging point, at least 0.4 m from the first. Attempts
        # alternate between them: the failed round re-entered C from a pose
        # 0.125 m from the previous one, i.e. it re-ran the same approach, and
        # a retry that starts the same way buys nothing. Empty keeps a single
        # staging point.
        self.declare_parameter('entry_recovery_alt_pose', [0.0])
        self.declare_parameter('entry_recovery_wait_budget_sec', 45.0)
        # One deadline covers staging, probing and every attempt. Measured on
        # the old code: a 45 s budget consumed 60 s because each stage started
        # its own timer.
        self.declare_parameter('entry_recovery_max_attempts', 3)
        # Long enough to finish a 4.4 m approach from the waiting point at
        # the measured 0.25-0.35 m/s; a shorter cap would abort a pass that
        # is about to succeed. Still capped by what is left of the budget.
        self.declare_parameter('entry_recovery_attempt_timeout_sec', 30.0)
        # Timeout of the plan-only probe: a ComputePathToPose-only behaviour
        # tree that never moves the base. An attempt that returns faster than
        # this without moving is recorded separately, but the probe - not the
        # timing - decides whether a route exists.
        self.declare_parameter('entry_recovery_plan_abort_sec', 5.0)
        # Arrive, stop and assess the entrance from this far out. Tightened
        # from 0.45 m: the pose the base failed from on 2026-11-21 was 0.125 m
        # from the waiting point, so a 0.45 m radius would have accepted it as
        # "staged" and driven the same approach again.
        self.declare_parameter('entry_recovery_reach_m', 0.20)
        # Position matching is not enough to say "a different approach": also
        # require this much heading difference before a retry counts as new.
        self.declare_parameter('entry_recovery_yaw_tol_rad', 0.35)
        # Retreat length and speed. 0.30 m of clearance plus the loaded 0.35 m
        # radius needs 0.45 m to reach the band edge, so 0.6 m clears it with
        # margin.
        self.declare_parameter('entry_recovery_retreat_m', 0.6)
        self.declare_parameter('entry_recovery_retreat_speed_mps', 0.10)
        # The retreat is open-loop only in the sense that its length is fixed:
        # every tick re-reads the lidar and stops when the travel direction
        # gets this close. Below the loaded 0.35 m radius plus a 0.05 m margin
        # the base would be driving itself into the band it is trying to leave.
        self.declare_parameter('entry_recovery_stop_clearance_m', 0.40)
        # A scan older than this is not used to authorise movement; the base
        # then holds still and fails the recovery closed.
        self.declare_parameter('entry_recovery_scan_max_age_sec', 0.6)
        # Test-only switch: hand the dock leg to the recovery BEFORE the first
        # ordinary attempt. A successful entrance never fails, so without this
        # a Gazebo check of the staging point, the plan-only probe and the
        # recovery-free drive tree would depend on catching a corner cut. It is
        # False everywhere except config/profiles/entry_recovery_test.yaml and
        # the scored profile is asserted not to carry it.
        self.declare_parameter('entry_recovery_force_first', False)
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
        # 2026-11-16: the attacher mutates joints from a ROS callback thread
        # while ODE steps in the physics thread. Freezing the world for the
        # width of the ATTACH/DETACH call keeps that mutation between two
        # physics steps; see _freeze_physics() for the crash it addresses.
        self.declare_parameter('freeze_physics_around_link_services', True)
        self.declare_parameter('physics_freeze_timeout_sec', 2.0)
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

        # 2026-11-25: the three warehouses are colour-coded floor plates in
        # offic_room.world - zone_a (0 1 0) green, zone_b (1 1 0) yellow,
        # zone_c (0 1 1) cyan, each 1.0 x 0.5 m with a 0.4 m black letter
        # standing on it. When the base reaches its dock the plate fills the
        # bottom of the forward camera, exactly like a cube reaching
        # visual_approach_row at pickup - so the bottom band is a direct
        # "this is warehouse X and we are close enough to place" signal that
        # the creep law, which only sees the goal in the map, cannot give.
        # Observational first: the gate logs fill and centroid for every
        # delivery and never fails a leg (fail_open), so the thresholds are
        # calibrated on real rounds before it may veto a placement.
        self.declare_parameter('place_visual_gate', True)
        self.declare_parameter('place_visual_gate_bottom_rows', 0.16)
        self.declare_parameter('place_visual_gate_centre_band', 0.70)
        # Calibrated on 2026-11-25 (two rounds, 8 deliveries that all sat on
        # their own plate): fill ranged 0.62-1.00, centroid -0.25..+0.38.
        # 0.30 would also accept a half-occluded plate; 0.45 keeps every
        # observed dock and still rejects a plate that is only just in view.
        self.declare_parameter('place_visual_gate_min_fill', 0.45)
        # 2026-11-25 round: a delivery that filled 0.96 of the band read
        # centroid +0.03, while two that filled 0.62/0.72 (same dock,
        # same plate) read +0.38/+0.27 with 20-30 deg of residual heading.
        # Centring is a servo target, not a pass/fail test, so the
        # tolerance stays wide and the number is logged for L2.
        self.declare_parameter('place_visual_gate_centre_tol', 0.60)
        self.declare_parameter('place_visual_gate_frames', 3)
        self.declare_parameter('place_visual_gate_timeout_sec', 6.0)
        self.declare_parameter('place_visual_gate_hue_tolerance', 14.0)
        self.declare_parameter('place_visual_gate_min_saturation', 90.0)
        self.declare_parameter('place_visual_gate_min_value', 60.0)
        self.declare_parameter('place_visual_gate_fail_open', True)
        # L2: hand the last stretch of the dock creep to the plate
        # centroid instead of to the bearing of a goal point, so the base
        # stops re-aiming (the rocking) while it closes the sideways
        # residual. The positional stop criterion is unchanged.

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
        # The whole mover policy: one circle around the base centre. Inside it
        # the base waits, outside it the base drives on. 0.0 restores the old
        # band/prediction/arc policy (kept for comparison, not for races).
        self.declare_parameter('yield_simple_radius_m', 1.8)
        # Only the half of that circle in front of the base is a reason to wait:
        # a box behind cannot be driven into. False restores the literal circle.
        self.declare_parameter('yield_simple_front_only', True)
        # Width of the cone ahead that counts as "in the way": +/-50 deg (100
        # total). A box grazing the rim of the half circle no longer stops the
        # base, while the 50-60 deg corner that cost one 1.1 cm overlap at +/-45
        # is still covered (2026-11-25). 180 restores the whole front half.
        self.declare_parameter('yield_simple_fov_deg', 100.0)
        # Inside this radius the box is already too close to wait out, so the
        # leg keeps its path instead of stopping: waiting at contact range let
        # the box arrive (2026-11-26 rule 2, base launched at 11.3 m/s) or burned
        # the hold budget on a box that was leaving (2026-11-25 21:33 rule 2,
        # 0/5). Operator rule: wait in the 1.0-1.8 m annulus, drive on inside 1.0.
        self.declare_parameter('yield_simple_commit_m', 1.0)
        self.declare_parameter('yield_cap_range_m', 1.8)
        self.declare_parameter('yield_stop_range_m', 1.2)
        self.declare_parameter('yield_cap_speed_mps', 0.8)
        self.declare_parameter('yield_resume_speed_mps', 0.6)
        # Static clusters jitter: the centroid of a wall or of the target cube
        # moved 0.05-0.09 m/s frame to frame, which the old 0.04 m/s floor
        # accepted as a mover - round 9 held still for 129 s on a "box" that was
        # 0.6 m away and had world=0.051 m/s in the log. Patrol boxes run at
        # 0.20 m/s, so 0.12 keeps every real mover and drops the jitter.
        self.declare_parameter('yield_min_mover_speed_mps', 0.12)
        # A hold that lasts must not pin the base inside an inscribed band: RPP
        # then refuses every arc and no Nav2 recovery can run (the goal is
        # cancelled). Retreating along the lidar-verified axis is the only exit.
        self.declare_parameter('yield_band_escape_enabled', True)
        self.declare_parameter('yield_band_escape_after_sec', 3.0)
        self.declare_parameter('yield_band_escape_attempts', 2)
        # NavFn cannot plan out of an inscribed band: a start pose closer than
        # the global robot radius to a wall is lethal and every GridBased request
        # is aborted (measured 2026-11-23: base 0.158 m from a wall after a hold,
        # "failed to generate a valid path" -> status=6, no recovery). Leave the
        # band physically before asking for a plan.
        self.declare_parameter('start_band_escape_enabled', True)
        # A band escape is a shove, not a drive: cap it and cap how often one
        # leg may use it (measured 2026-11-25: a 0.25 m shove with 0.51 m of
        # scan room is enough; the old 0.60 m gate refused it and the leg died).
        self.declare_parameter('place_servo', False)
        self.declare_parameter('place_servo_engage_m', 1.00)
        self.declare_parameter('place_servo_fill_start', 0.20)
        self.declare_parameter('start_band_escape_retreat_m', 0.25)
        self.declare_parameter('start_band_escape_attempts', 2)
        # Keep static-cluster phantoms (measured 0.86-1.52 m/s) outside the
        # tracking window while detecting the 0.20 m/s patrol boxes.
        self.declare_parameter('yield_max_mover_speed_mps', 0.6)
        # How much time must be left between "the base is past" and "the box
        # enters the corridor": the sensing/control latency of this stack.
        self.declare_parameter('yield_clear_margin_sec', 0.3)
        # 15 s matches the external stall watchdog (2026-11-25).
        self.declare_parameter('yield_wait_budget_sec', 15.0)
        self.declare_parameter('yield_arc_offset_m', 0.55)
        self.declare_parameter('yield_arc_apex_m', 0.8)
        self.declare_parameter('yield_arc_goal_radius_m', 0.35)
        # How close to a patrol centre line a hold is allowed to park, and how
        # far outside the swept band a held obstacle must be before the hold is
        # released. Both are pinned in the deployment profile: they decide how
        # close the base may come to a kinematic box that never yields.
        self.declare_parameter('yield_lane_hold_clearance_m', 0.62)
        self.declare_parameter('yield_lane_escape_delay_sec', 1.0)
        self.declare_parameter('yield_go_commit_sec', 0.8)
        self.declare_parameter('yield_go_commit_m', 0.40)
        self.declare_parameter('yield_hold_release_margin_m', 0.12)
        self.declare_parameter('yield_clear_hold_sec', 0.5)
        # The lateral exit is a Nav2 goal, so it is only worth issuing if the
        # planner will accept it. Without this check the arc landed on a cell
        # inside the inflated band and NavFn answered "Failed to create a plan
        # from potential when a legal potential was found" three times in a row
        # (2026-11-15, 16:13:09-16:13:38), each followed by a failed
        # backup/spin on the same costmap.
        self.declare_parameter('yield_arc_costmap_check', True)
        # A clear apex is not a drivable sidestep. On the 2026-11-21 round
        # (block 4, red_cube_1 -> A) the arc target was outside the inscribed
        # band, but the carried 0.35 m envelope could not cross the 0.55 m of
        # floor between the base and that cell: the arc leg drove 0.94 m in
        # 14.8 s, ended with controller status 6, and the leg failed a 3/5
        # round with the cube still attached. Sample the whole base->apex path,
        # not only its endpoint.
        self.declare_parameter('yield_arc_path_check', True)
        self.declare_parameter('yield_arc_path_step_m', 0.10)
        # Ignore the cells under the base itself: the loaded footprint is
        # already there, and a wall 0.34 m away would otherwise reject every
        # arc even though the base is standing there legally.
        self.declare_parameter('yield_arc_path_start_m', 0.20)
        # A lateral waypoint is short (<= ~1 m). If the base stops closing it,
        # abandon it in seconds instead of waiting for the 45 s leg watchdog:
        # measured 2026-11-21, the arc made 0.70 m in 36.9 s before the round
        # died. The total cap bounds a slow creep that never quite stops.
        self.declare_parameter('yield_arc_progress_timeout_sec', 8.0)
        self.declare_parameter('yield_arc_progress_m', 0.10)
        self.declare_parameter('yield_arc_timeout_sec', 20.0)
        # After a failed sidestep the base sits somewhere between the two
        # lanes. Give the mover a moment to clear and retry the ORIGINAL route
        # instead of declaring the carried leg unretryable.
        self.declare_parameter('yield_arc_recovery_hold_sec', 1.0)
        # Go round a crossing box instead of parking for it. Movers patrol at
        # 0.20 m/s, so the old order (stop, wait out the budget, then take the
        # lateral escape) spent 26.75 s / 31.83 s / 42.02 s in single episodes
        # across the 2026-11-18 rounds while the floor beside the base was
        # empty. The waypoint is the same one `yield_arc_costmap_check` vets;
        # only the moment it is taken changed. Head-on boxes and anything
        # inside the 0.75 m keep-out still stop.
        self.declare_parameter('yield_approach_trial', False)
        self.declare_parameter('yield_early_detour', True)
        self.declare_parameter('yield_early_detour_lead_sec', 0.6)
        self.declare_parameter('yield_early_detour_head_on_mps', 0.05)
        self.declare_parameter('yield_early_detour_range_m', 3.0)
        # Predicted-occupancy planning: compare the base's candidate future at
        # each future instant against the mover's, and take the fastest one that
        # clears. Defaults mirror obstacle_yield.YieldConfig; only the knobs a
        # round is likely to tune are exposed.
        self.declare_parameter('yield_predict_enabled', True)
        self.declare_parameter('yield_predict_range_m', 4.0)
        self.declare_parameter('yield_predict_horizon_sec', 4.0)
        self.declare_parameter('yield_predict_clearance_m', 0.15)
        self.declare_parameter('global_costmap_topic', 'global_costmap/costmap')
        # The patrol boxes' centre lines, published latched by moving_obstacles.
        # Without them the policy cannot tell "stop beside the box" from "stop
        # inside the box's lane", and a base parked in a lane is trapped: the box
        # returns every few seconds (measured 2026-10-13, 18 s frozen).
        self.declare_parameter('mover_patrols_topic', '/movers/patrols')
        # The patrol boxes publish their own footprints in base_link, so this
        # node can see a box the lidar cannot yet (occlusion at a doorway).
        # Measured detour1/detour2: the yield layer first saw the box that cost
        # an 11.3 s standstill at 1.7 m of range, already inside the choke.
        # Used only as an extra yield observation, never as costmap input, so it
        # cannot re-introduce the plan-failure storms.
        self.declare_parameter('mover_cloud_topic', '/movers/obstacles')
        self.declare_parameter('yield_mover_cloud_stage', True)
        self.declare_parameter('yield_mover_cloud_range_m', 6.0)
        # Wider than the swept band on purpose: a box that will enter the band
        # is still off to the side when it is 4-6 m away, and the policy's CAP
        # branch is what turns "it is coming" into "slow down now".
        self.declare_parameter('yield_mover_cloud_band_m', 1.2)
        self.declare_parameter('yield_mover_cloud_timeout_sec', 0.35)
        # Rate limit for the yield diagnostic below; 0.0 disables it, and it is
        # silent unless the yield feature itself is enabled. It only ever logs
        # (no motion), so it is safe to leave on during a round.
        self.declare_parameter('yield_diag_period_sec', 5.0)
        self.declare_parameter('yield_tick_trace', False)
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
        # Optional placement-only experiment; 0 falls back to arm_motion_sec.
        self.declare_parameter('place_arm_motion_sec', 0.0)
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
        # joint1 turns about the vertical axis (arm.xacro), so the heading the
        # dock ended with can be taken out of the release point instead of
        # being inherited by it: a 0.36 m leash turns 20 deg of base yaw into
        # 0.12 m of sideways throw, which is a whole slot (2026-11-16).
        self.declare_parameter('place_yaw_compensation', True)
        self.declare_parameter('place_yaw_comp_limit_rad', 0.6)
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
        self._link_service_frozen = False
        self._physics_clients = {}
        self._clear_global_client = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap'
        )
        self._clear_local_client = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap'
        )
        self._recovery_tf = Buffer()
        self._recovery_tf_listener = TransformListener(self._recovery_tf, self)
        self._make_geometry_clients('global')
        self._make_geometry_clients('local')
        self._geometry_fault = False
        self._navigation_control_fault = False
        self._nav_leg_deadline = None
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
        self._range_gate_warned = False
        self._target_colour = 'red'
        # Placement gate: while active the image callback only measures the
        # warehouse plate and returns, so the cube contour pipeline stays out
        # of the way (see _plate_sample).
        self._plate_gate_active = False
        self._plate_gate_house = ''
        self._plate_sample_ts = 0.0
        self._plate_sample_metrics = None
        self._plate_gate_logged = set()
        camera_topic = str(self.get_parameter('camera_topic').value)
        self._image_sub = self.create_subscription(
            Image, camera_topic, self._image_callback, qos_profile_sensor_data
        )
        self._fine_cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter('fine_cmd_vel_topic').value), 10
        )
        self._guarded_navigation_recovery = bool(
            self.get_parameter('guarded_navigation_recovery').value)
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
        self._nominal_geometry = None
        self._relaxed_global = False
        self._relax_used = False
        self._global_radius_before_relax = None
        # Re-entrancy guard for the narrow-entrance recovery: its own staging
        # and retry legs must not trigger a second recovery.
        self._entry_recovery_running = False
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
        self._mover_cloud = None
        self._mover_cloud_stamp = 0.0
        self.create_subscription(
            PointCloud2, str(self.get_parameter('mover_cloud_topic').value),
            self._mover_cloud_callback, qos_profile_sensor_data
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
            simple_radius_m=float(
                self.get_parameter('yield_simple_radius_m').value),
            simple_radius_front_only=bool(
                self.get_parameter('yield_simple_front_only').value),
            simple_radius_fov_deg=float(
                self.get_parameter('yield_simple_fov_deg').value),
            simple_radius_commit_m=float(
                self.get_parameter('yield_simple_commit_m').value),
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
            lane_hold_clearance_m=float(
                self.get_parameter('yield_lane_hold_clearance_m').value),
            lane_escape_delay_sec=float(
                self.get_parameter('yield_lane_escape_delay_sec').value),
            go_commit_sec=float(
                self.get_parameter('yield_go_commit_sec').value),
            go_commit_m=float(self.get_parameter('yield_go_commit_m').value),
            hold_release_margin_m=float(
                self.get_parameter('yield_hold_release_margin_m').value),
            clear_hold_sec=float(
                self.get_parameter('yield_clear_hold_sec').value),
            approach_trial_enabled=bool(
                self.get_parameter('yield_approach_trial').value),
            early_detour_enabled=bool(
                self.get_parameter('yield_early_detour').value),
            early_detour_lead_sec=float(
                self.get_parameter('yield_early_detour_lead_sec').value),
            early_detour_head_on_mps=float(
                self.get_parameter('yield_early_detour_head_on_mps').value),
            early_detour_range_m=float(
                self.get_parameter('yield_early_detour_range_m').value),
            predict_enabled=bool(
                self.get_parameter('yield_predict_enabled').value),
            predict_range_m=float(
                self.get_parameter('yield_predict_range_m').value),
            predict_horizon_sec=float(
                self.get_parameter('yield_predict_horizon_sec').value),
            predict_clearance_m=float(
                self.get_parameter('yield_predict_clearance_m').value),
        )
        # Latest global costmap, used only to reject an unusable lateral exit.
        # Unknown/absent map = no opinion (the lidar gate still applies).
        self._global_costmap = None
        self._global_costmap_received_at = None
        self._local_costmap = None
        self._local_costmap_received_at = None
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('global_costmap_topic').value),
            self._global_costmap_callback, costmap_qos
        )
        self.create_subscription(
            OccupancyGrid, 'local_costmap/costmap',
            self._local_costmap_callback, costmap_qos
        )
        self._yield_tracker = obstacle_yield.YieldTracker(self._yield_config)
        self._yield_decision = obstacle_yield.YieldDecision(
            obstacle_yield.ACTION_GO, obstacle_yield.REASON_CLEAR
        )
        self._dock_yaw_error = 0.0
        self._dock_distance = float('inf')
        self._yield_latched = False
        self._yield_release_from = None
        self._yield_release_since = None
        self._yield_resume_started = None
        self._yield_continuous_hold_since = None
        self._yield_wait_since = None
        self._yield_clear_since = None
        self._yield_arc_attempts = 0
        self._yield_arc_goal = None
        self._yield_arc_used = False
        self._yield_suppress = False
        # The arc leg that is currently being driven (map frame) and whether the
        # mid-arc watch has called it off. The watch keeps checking the mover's
        # prediction while _yield_suppress stops the ordinary policy from making
        # a second decision.
        self._yield_arc_target = None
        self._yield_arc_abort = None
        self._yield_arc_watch_speed = None
        self._yield_watch_last = 0.0
        # Why the last navigation attempt was abandoned. The retry escape uses
        # it to pick a direction that matches the failure: driving 0.70 m
        # FORWARD after the controller refused the path ahead re-enters the same
        # refusal (measured 2026-10-14 15:34: eight attempts, 74-144 refusals
        # each, base never moved 2 cm, round lost at 1/5).
        self._last_attempt_failure = None
        # One outer pickup route per pickup attempt: once the bypass has been
        # tried, a still-failing leg goes back to the ordinary retry ladder so
        # the base cannot be driven in circles around the same mouth.
        self._pickup_fallback_used = False
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
        self._yield_trace_writer = None
        if bool(self.get_parameter('yield_tick_trace').value):
            try:
                self._yield_trace_writer = _YieldTraceWriter(self.get_logger())
            except Exception:
                # Diagnostics are optional even if worker startup fails.
                pass
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

    def _carrying_now(self) -> bool:
        """True while the grasped cube rides on the base.

        The yield layer needs this to exclude the cube's own returns: at
        0.233 m from the lidar axis it sits outside the 0.12 m self-return
        cutoff, so without the flag it enters the tracker as a cluster that
        travels with the base (r3b: track world=0.15 m, decision=go/behind -
        cleared only because the geometry happened to put it behind).
        Reads the cached link states; never blocks.
        """
        msg = self._latest_link_states
        if msg is None:
            return False
        names = list(msg.name)
        try:
            cube = msg.pose[names.index(
                f"{self.get_parameter('cube_name').value}::"
                f"{self.get_parameter('cube_link').value}")]
            base = msg.pose[names.index(
                f"{self.get_parameter('robot_model_name').value}::"
                f"{self.get_parameter('robot_base_link').value}")]
        except (ValueError, IndexError):
            return False
        horizontal = math.hypot(cube.position.x - base.position.x,
                                cube.position.y - base.position.y)
        lift = cube.position.z - base.position.z
        return horizontal <= 1.0 and lift > 0.10

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

    def _plate_sample(self, hsv) -> None:
        """Measure the warehouse plate in the bottom band of one frame.

        2026-11-25: called from the image callback while the gate is open. It
        stores the last measurement and nothing else - the gate loop decides
        whether a run of them is good enough, so the callback stays cheap.
        """
        window = zone_plate_hue_window(
            self._plate_gate_house,
            float(self.get_parameter('place_visual_gate_hue_tolerance').value),
        )
        fill, centroid, rows, columns, dominant_hue = plate_band_metrics(
            hsv,
            float(self.get_parameter('place_visual_gate_bottom_rows').value),
            float(self.get_parameter('place_visual_gate_centre_band').value),
            window,
            float(self.get_parameter('place_visual_gate_min_saturation').value),
            float(self.get_parameter('place_visual_gate_min_value').value),
        )
        if fill is None:
            return
        self._plate_sample_metrics = (fill, centroid, rows, columns, dominant_hue)
        self._plate_sample_ts = time.monotonic()

    def _verify_place_target_plate(self) -> bool:
        """L1 gate: is the warehouse colour filling the bottom of the camera?

        2026-11-25 (user request): every warehouse owns a floor plate - A
        green, B yellow, C cyan - so "the bottom of the image is this
        warehouse's colour" is both an identity check (are we at the right
        one) and a range check (are we close enough to place). The gate is
        observational: with fail_open it logs PLACE_GATE with the measured
        fill/centroid and returns True, so a mis-set threshold costs a log
        line, not a round. Once the numbers are in, fail_open can be turned
        off and the gate may refuse.
        """
        if not bool(self.get_parameter('place_visual_gate').value):
            return True
        house = str(self.get_parameter('warehouse').value).strip().upper()
        window = zone_plate_hue_window(
            house, float(self.get_parameter('place_visual_gate_hue_tolerance').value)
        )
        if window is None:
            self.get_logger().warning(
                f'place gate: warehouse {house!r} has no plate colour on record; '
                'placing without the colour check.'
            )
            return True
        need = max(1, int(self.get_parameter('place_visual_gate_frames').value))
        budget = max(0.5, float(self.get_parameter('place_visual_gate_timeout_sec').value))
        min_fill = float(self.get_parameter('place_visual_gate_min_fill').value)
        centre_tol = float(self.get_parameter('place_visual_gate_centre_tol').value)
        self._plate_gate_house = house
        self._plate_sample_metrics = None
        self._plate_sample_ts = 0.0
        self._plate_gate_active = True
        self._vision_enabled = True
        started = time.monotonic()
        deadline = started + budget
        good = 0
        best = (0.0, 9.9, 0, -1)
        attempts = 0
        try:
            while rclpy.ok() and not self._cancel_requested and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                attempts += 1
                if self._plate_sample_metrics is None:
                    continue
                fill, centroid, rows, columns, dominant_hue = (
                    self._plate_sample_metrics)
                if abs(fill) > abs(best[0]):
                    best = (fill, centroid, rows, dominant_hue)
                if fill >= min_fill and abs(centroid) <= centre_tol:
                    good += 1
                else:
                    good = 0
                if good >= need:
                    self.get_logger().info(
                        f'PLACE_GATE:PASS warehouse {house} ({ZONE_PLATE_COLOUR[house]}) '
                        f'fill={fill:.2f} centroid={centroid:+.2f} rows={rows} '
                        f'hue={dominant_hue} window={window} '
                        f'spent={time.monotonic() - started:.1f}s'
                    )
                    return True
            self.get_logger().warning(
                f'PLACE_GATE:MISS warehouse {house} '
                f'({ZONE_PLATE_COLOUR[house]}) was not confirmed within '
                f'{budget:.1f}s: best fill={best[0]:.2f} centroid={best[1]:+.2f} '
                f'rows={best[2]} band hue={best[3]} (window {window}) '
                f'(need fill>={min_fill:.2f} |centroid|<={centre_tol:.2f} for '
                f'{need} frames, {attempts} frames seen)'
            )
            return bool(self.get_parameter('place_visual_gate_fail_open').value)
        finally:
            self._plate_gate_active = False
            self._vision_enabled = False
            self._plate_gate_house = ''


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
        if getattr(self, '_plate_gate_active', False):
            # The plate gate owns the frame while it is open: measuring the
            # warehouse plate is the whole job, and letting the cube pipeline
            # run would publish a blob for a colour nobody asked about.
            self._plate_sample(hsv)
            return
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
                    # 2026-11-17: below roughly 0.45 m this prior runs past the
                    # bottom of the frame (66 / 0.25 m = 264 px under the horizon,
                    # but 66 / 0.10 m = 660 px), so every contour fails the gate
                    # and the detector returns None for the whole servo window -
                    # the run only shows "Visual blob lost" and a 75 s timeout.
                    # Outside the range the prior was calibrated for, fall back to
                    # the area/aspect gates instead of rejecting everything.
                    if not (
                        0.35 * image_height <= expected_bottom <= 1.05 * image_height
                    ):
                        if not self._range_gate_warned:
                            self._range_gate_warned = True
                            self.get_logger().warning(
                                'Row prior '
                                f'{expected_bottom:.0f} px is outside the frame '
                                f'({image_height} px) at '
                                f'{base_distance:.2f} m; skipping the row gate '
                                'instead of rejecting every contour.'
                            )
                        expected_bottom = None
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
                lambda: (self._global_costmap_params_client.service_is_ready()
                         and self._global_costmap_get_client.service_is_ready()),
            ),
            (
                'params_local',
                'local costmap parameter service',
                lambda: (self._local_costmap_params_client.service_is_ready()
                         and self._local_costmap_get_client.service_is_ready()),
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
        elif key in ('params_global', 'params_local'):
            side = key.removeprefix('params_')
            self.destroy_client(getattr(self, f'_{side}_costmap_params_client'))
            self.destroy_client(getattr(self, f'_{side}_costmap_get_client'))
            self._make_geometry_clients(side)
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
        *,
        deadline: float | None = None,
    ):
        """Send a goal; never overlap unacknowledged navigation goals.

        The controller can accept a goal and still fail to deliver the response
        (Fast DDS allows roughly 100 ms for a service response). Because each
        retry re-sends the identical target, waiting for the possibly-executing
        motion to settle first keeps the retry idempotent. Returns ``None`` when
        every attempt lost its response. For navigation, instead confirm
        terminal release of any late acceptance or fail closed.
        """
        attempts = max(1, int(self.get_parameter('action_retry_count').value) + 1)
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = leg_deadline if deadline is None else min(leg_deadline, deadline)
        if deadline is not None:
            if time.monotonic() >= deadline or getattr(self, '_cancel_requested', False):
                return None
            future = client.send_goal_async(goal)
            response_wait = min(response_timeout, max(0.0, deadline - time.monotonic()))
            if response_wait > 0.0:
                rclpy.spin_until_future_complete(self, future, timeout_sec=response_wait)
            if not future.done() and attempts > 1:
                # A missing reply is not a rejection. Poll the same request;
                # another goal could overlap this unacknowledged one.
                settle_deadline = min(deadline, time.monotonic() + settle_sec)
                while (rclpy.ok() and not future.done()
                       and not getattr(self, '_cancel_requested', False)):
                    remaining = settle_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    rclpy.spin_once(self, timeout_sec=min(0.1, remaining))
            may_proceed = (future.done() and time.monotonic() < deadline
                           and not getattr(self, '_cancel_requested', False))
            # Share this grace between a late reply and terminal confirmation.
            grace_deadline = time.monotonic() + 0.5
            while not future.done() and rclpy.ok():
                remaining = grace_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
            try:
                handle = future.result() if future.done() else None
            except Exception:
                handle = None
            if handle is None:
                self._last_attempt_failure = 'nav_control_not_released'
                self._navigation_control_fault = True
                raise RuntimeError(f'NAV_STATE_UNKNOWN: {label} goal response unavailable')
            if may_proceed:
                return handle
            if handle.accepted:
                self._release_nav_control(
                    label, handle, None, response_timeout,
                    deadline=deadline, grace_deadline=grace_deadline,
                )
            return None

        for attempt in range(1, attempts + 1):
            if getattr(self, '_cancel_requested', False):
                return None
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
                while (rclpy.ok() and time.monotonic() < end
                       and not getattr(self, '_cancel_requested', False)):
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if future.done():
                        return future.result()
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
        cancelled_locally = False
        if not result_future.done():
            cancelled_locally = True
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
            wrapped_result.status == GoalStatus.STATUS_CANCELED
            and result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
            and not cancelled_locally
        ):
            # 2026-11-23 18:50 (probe11k): the controller logged "Received new
            # action goal" twice for one send (6 ms apart, both accepted) and
            # preempted our handle, yet its own "Goal reached, success!" shows
            # the same trajectory completed. The duplicate arrives on the wire,
            # not from this node: one send_goal_async produced exactly one
            # server-side receive in tools/arm_goal_probe.py. Treat a cancel we
            # did not ask for, with a successful controller result, as the
            # motion having happened - otherwise PRE_GRASP fails and the whole
            # round dies at attempt 1.
            self.get_logger().warning(
                f'{label}: goal was preempted by a duplicate request; the '
                f'controller reports the trajectory finished, continuing.'
            )
            time.sleep(float(self.get_parameter('settle_sec').value))
            return True
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

    def _place_arm_motion_sec(self) -> float:
        value = self.get_parameter('place_arm_motion_sec').value
        try:
            duration = float(value)
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        if math.isfinite(duration) and duration > 0.0:
            return duration
        return float(self.get_parameter('arm_motion_sec').value)

    def _move_arm(self, label: str, parameter_name: str, *, placing: bool = False) -> bool:
        return self._execute_trajectory(
            label,
            self._arm_client,
            ARM_JOINTS,
            self._double_list(parameter_name, len(ARM_JOINTS)),
            self._place_arm_motion_sec() if placing else
            float(self.get_parameter('arm_motion_sec').value),
        )

    def _move_arm_placing(self, label: str, parameter_name: str) -> bool:
        """Move the arm to a place posture, rotating joint1 to cancel the dock
        heading error.

        joint1 turns about the vertical axis (arm.xacro: axis 0 0 1), so adding
        the base heading error to it rotates the whole arm - and therefore the
        release point - back onto the plate's y axis. Without this the drop
        inherits the base's yaw: measured 2026-11-16, warehouse A docks
        accepted at 22.6 and -41.1 deg while the creep was refusing to converge
        (0.36 m leash x sin(22.6 deg) = 0.14 m of sideways throw, more than the
        whole slot allowance).
        """
        target = self._double_list(parameter_name, len(ARM_JOINTS))
        if not target:
            return self._move_arm(label, parameter_name, placing=True)
        if bool(self.get_parameter('place_yaw_compensation').value):
            limit = float(self.get_parameter('place_yaw_comp_limit_rad').value)
            yaw_error = float(getattr(self, '_dock_yaw_error', 0.0))
            comp = max(-limit, min(limit, yaw_error))
            if abs(comp) > 1.0e-3:
                target[0] = target[0] + comp
                self.get_logger().info(
                    f'{label}: joint1 {math.degrees(comp):+.1f} deg of dock '
                    f'heading compensation (dock ended '
                    f'{math.degrees(yaw_error):+.1f} deg off at '
                    f'{getattr(self, "_dock_distance", float("nan")):.3f} m).'
                )
        return self._execute_trajectory(
            label, self._arm_client, ARM_JOINTS, target,
            self._place_arm_motion_sec(),
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
            self.get_logger().info(f'EXECUTOR_COSTMAP_CLEAR request {label}')
            future = client.call_async(ClearEntireCostmap.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or future.exception() is not None:
                self.get_logger().error(
                    f'Could not confirm {label} costmap clear; refusing to '
                    'start a new goal while a late clear may still arrive.'
                )
                return False
        return True

    def _nav_budget_remaining(self, limit: float = math.inf) -> float:
        deadline = getattr(self, '_nav_leg_deadline', None)
        if deadline is None:
            return max(0.0, limit)
        return max(0.0, min(limit, deadline - time.monotonic()))

    def _make_geometry_clients(self, side: str) -> None:
        root = f'/{side}_costmap/{side}_costmap'
        setattr(self, f'_{side}_costmap_params_client', self.create_client(
            SetParametersAtomically, f'{root}/set_parameters_atomically'))
        setattr(self, f'_{side}_costmap_get_client', self.create_client(
            GetParameters, f'{root}/get_parameters'))

    def _geometry_call(self, client, request, deadline):
        if time.monotonic() >= deadline:
            return None, False  # Nothing was sent.
        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                self, future, timeout_sec=max(0.0, deadline - time.monotonic()))
            if future.done() and future.exception() is None:
                return future.result(), True
        except Exception as exc:
            self.get_logger().error(f'Costmap parameter service failed: {exc}')
        return None, True  # The request may still have taken effect.

    def _read_map_geometry(self, side, names, deadline):
        client = getattr(self, f'_{side}_costmap_get_client')
        response, _ = self._geometry_call(
            client, GetParameters.Request(names=list(names)), deadline)
        if response is None or len(response.values) != len(names):
            return None
        if any(value.type != ParameterType.PARAMETER_DOUBLE
               or not math.isfinite(value.double_value) for value in response.values):
            return None
        return dict(zip(names, (value.double_value for value in response.values)))

    def _write_map_geometry(self, side, values, deadline):
        client = getattr(self, f'_{side}_costmap_params_client')
        params = [Parameter(name=name, value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=value))
            for name, value in values.items()]
        response, sent = self._geometry_call(
            client, SetParametersAtomically.Request(parameters=params), deadline)
        if not sent:
            return False
        if response is None:
            # Do not race a rollback against an unresolved write. A restart of
            # the navigation stack is required before the executor moves again.
            self._geometry_fault = True
            raise RuntimeError(f'GEOMETRY_STATE_UNKNOWN: {side} write unconfirmed')
        if not response.result.successful:
            self.get_logger().error(
                f'{side} costmap rejected geometry: {response.result.reason}')
            return False
        actual = self._read_map_geometry(side, values, deadline)
        return actual == values

    def _update_costmap_geometry(self, desired, *, cleanup=False, expected=None):
        """Apply a cross-node change with exact readback and compensating rollback."""
        if getattr(self, '_geometry_fault', False):
            raise RuntimeError('GEOMETRY_STATE_UNKNOWN: restart Nav2 and executor')
        timeout = float(self.get_parameter('service_timeout_sec').value)
        # Cleanup is safety work, not permission to move after the leg expired.
        timeout = min(2.0, timeout) if cleanup else self._nav_budget_remaining(timeout)
        if not math.isfinite(timeout) or timeout <= 0.0:
            return None
        deadline = time.monotonic() + timeout
        before = {}
        for side, values in desired.items():
            before[side] = self._read_map_geometry(side, values, deadline)
            if before[side] is None:
                self.get_logger().error(f'Could not read {side} costmap geometry.')
                return None
        if expected is not None and before != expected:
            self._geometry_fault = True
            raise RuntimeError('GEOMETRY_STATE_UNKNOWN: nominal geometry changed externally')
        tried = []
        for side, values in desired.items():
            if before[side] == values:
                continue
            tried.append(side)
            if self._write_map_geometry(side, values, deadline):
                continue
            rollback_deadline = time.monotonic() + min(2.0, timeout)
            for changed in reversed(tried):
                if not self._write_map_geometry(changed, before[changed], rollback_deadline):
                    self._geometry_fault = True
                    raise RuntimeError(
                        f'GEOMETRY_STATE_UNKNOWN: {changed} rollback unconfirmed')
            self.get_logger().warning('Costmap geometry change reverted.')
            return None
        return before

    def _set_costmap_geometry__timed(
        self, radius: float, inflation_radius: float, label: str,
    ) -> bool:
        """Confirm nominal empty/loaded radii on both Nav2 costmaps."""
        margin = float(self.get_parameter('global_radius_margin_m').value)
        global_min = float(self.get_parameter('global_inflation_min_m').value)
        planned_radius = radius + margin
        if (not all(math.isfinite(v) for v in (radius, inflation_radius, margin,
                                               global_min, planned_radius))
                or radius <= 0.0 or inflation_radius < radius
                or margin < 0.0 or global_min < 0.0):
            self.get_logger().error(f'Invalid {label} costmap geometry.')
            return False
        inflation_name = 'inflation_layer.inflation_radius'
        before = self._update_costmap_geometry({
            'global': {'robot_radius': planned_radius,
                       inflation_name: max(inflation_radius, planned_radius, global_min)},
            'local': {'robot_radius': radius, inflation_name: inflation_radius},
        })
        if before is None:
            return False
        self._nominal_geometry = (radius, inflation_radius, label)
        self.get_logger().info(
            f'Nav2 geometry confirmed for {label}: local radius={radius:.2f} m, '
            f'global radius={planned_radius:.2f} m.')
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
        deadline = time.monotonic() + self._nav_budget_remaining(float(
            self.get_parameter('visual_handoff_retreat_timeout_sec').value
        ))
        self.get_logger().warning(
            f'{cube} is only {distance:.3f} m away - too close for the camera '
            f'servo; retreating to {target:.2f} m before handing over.'
        )
        best = distance
        command = Twist()
        started = time.monotonic()
        flipped = False
        while (rclpy.ok() and time.monotonic() < deadline
               and not getattr(self, '_cancel_requested', False)):
            pose = self._cached_base_pose()
            current = self._cached_planar_distance_to_cube(cube)
            if pose is None or current is None:
                rclpy.spin_once(self, timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())))
                continue
            best = max(best, current)
            if current >= target:
                break
            cube_pose = self._cached_cube_position(cube)
            if cube_pose is None:
                rclpy.spin_once(self, timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())))
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
            if time.monotonic() >= deadline:
                break
            self._fine_cmd_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())))
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
            rclpy.spin_once(self, timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())))
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


    def _creep_to_dock__timed(
        self, label: str, x: float, y: float, yaw: float, *,
        deadline: float | None = None,
    ) -> bool:
        """Drive the last few centimetres to the plate with Nav2 released.

        Rotate to the plate yaw first, then creep straight in with a small
        lateral correction. This replaces the controller's near-goal limit cycle
        (see enable_dock_creep) and, unlike waiting for the goal checker, it ends
        at the same pose every time, which is what the placement accuracy is
        calibrated against.
        """
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = leg_deadline if deadline is None else min(leg_deadline, deadline)
        shared_deadline = deadline if deadline is not None else math.inf

        def expired():
            if deadline is not None and time.monotonic() >= deadline:
                self._last_attempt_failure = 'deadline'
                self._stop_base()
                return True
            return getattr(self, '_cancel_requested', False)

        if expired():
            return False
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
        accept_yaw = min(
            float(self.get_parameter('navigation_accept_yaw_rad').value),
            float(self.get_parameter('place_yaw_comp_limit_rad').value),
        )
        attempts = max(1, int(self.get_parameter('dock_creep_attempts').value))
        align_distance = float(
            self.get_parameter('dock_creep_align_distance_m').value
        )
        servo_enabled = bool(self.get_parameter('place_servo').value)
        servo_engage_m = float(self.get_parameter('place_servo_engage_m').value)
        servo_fill_start = float(self.get_parameter('place_servo_fill_start').value)
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
        # inside dock_creep_accept_m and the arm's compensation limit, stop and
        # verify the measured pose remains in bounds for position_hold_sec.
        position_hold_sec = 2.0
        position_since = None
        # The placement posture is a fixed set of joint angles, so the drop
        # point sits a fixed ~0.36 m leash from the base. A base heading error
        # of theta therefore throws the release sideways by ~0.36*sin(theta):
        # 20 deg is 0.12 m, which is a whole slot. Record what the dock ended
        # with so the place step can rotate joint1 by the same angle and put the
        # drop back on the plate's axis instead of the base's.
        self._dock_yaw_error = 0.0
        self._dock_distance = float('inf')
        started = time.monotonic()
        # 2026-11-16: Nav2 hands over while the base is still doing up to
        # 1.5 m/s, and with max_decel 2.5 m/s^2 it coasts ~0.4 m. Starting the
        # law straight away made it overshoot the plate and turn around
        # (competition_lanes_v02h, warehouse B: t=0.0 d=0.281 m, t=1.0 s
        # odom=0.368 m/s -> d=0.044 m, then 0.500 m and a 175 deg heading
        # error). Hold still until the coast is spent instead.
        settle_until = min(shared_deadline, time.monotonic() + 2.0)
        while rclpy.ok() and time.monotonic() < settle_until:
            if expired():
                return False
            self._stop_base()
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, settle_until - time.monotonic())))
            if expired():
                return False
            first = self._base_pose()
            if first is None:
                continue
            sleep_sec = min(0.10, max(0.0, shared_deadline - time.monotonic()))
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            self._stop_base()
            if expired():
                return False
            second = self._base_pose()
            if second is None:
                continue
            coast = math.hypot(
                second[0] - first[0], second[1] - first[1]
            ) / 0.10
            if coast < 0.06:
                self.get_logger().info(
                    f'{label}: base settled before the dock creep '
                    f'(coast {coast:.3f} m/s).'
                )
                break
        for attempt in range(1, attempts + 1):
            if expired():
                return False
            attempt_deadline = min(shared_deadline, time.monotonic() + timeout)
            position_since = None
            # One line per second of the creep. Without it the docking phase is
            # a 12-28 s black box: measured 2026-11-16 (competition_lanes_v02g)
            # the creep burned 28.1/13.1/12.6/12.6 s to close <=0.35 m, and the
            # per-second trace is what tells a slow law from a base that cannot
            # track the command.
            trace_at = 0.0
            trace_pose = self._base_pose()
            while rclpy.ok() and time.monotonic() < attempt_deadline:
                if not getattr(self, '_cancel_requested', False):
                    remaining = max(0.0, attempt_deadline - time.monotonic())
                    rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
                if expired():
                    self._stop_base()
                    return False
                if time.monotonic() >= attempt_deadline:
                    break
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
                elapsed = time.monotonic() - started
                if elapsed >= trace_at and trace_pose is not None:
                    trace_at = elapsed + 1.0
                    moved = math.hypot(base_x - trace_pose[0], base_y - trace_pose[1])
                    trace_pose = (base_x, base_y, base_yaw)
                    self.get_logger().info(
                        f'{label}: creep t={elapsed:.1f}/{attempt} d={distance:.3f} '
                        f'yaw={math.degrees(yaw_error):+.1f} '
                        f'lateral={-dx * math.sin(base_yaw) + dy * math.cos(base_yaw):+.3f} '
                        f'odom={moved / max(1.0, elapsed - (trace_at - 1.0)):.3f} m/s '
                        f'cmd=({command.linear.x:.3f},{command.angular.z:+.2f})'
                    )
                if distance <= accept_radius and abs(yaw_error) <= accept_yaw:
                    # Stop correcting as soon as both placement bounds are met.
                    # Continue sampling the actual pose while stopped: if the base
                    # coasts out of bounds the hold is reset and docking resumes.
                    self._stop_base()
                    command.linear.x = 0.0
                    command.angular.z = 0.0
                    now = time.monotonic()
                    if position_since is None:
                        position_since = now
                    elif now - position_since >= position_hold_sec:
                        self.get_logger().info(
                            f'{label}: stopped within placement bounds for '
                            f'{position_hold_sec:.1f} s at {distance:.3f} m, '
                            f'heading {math.degrees(yaw_error):+.1f} deg off.'
                        )
                        self._dock_yaw_error = yaw_error
                        self._dock_distance = distance
                        return not expired()
                    continue
                else:
                    position_since = None
                visual = None
                if servo_enabled and distance <= servo_engage_m:
                    metrics = getattr(self, '_plate_sample_metrics', None)
                    stamp = float(getattr(self, '_plate_sample_ts', 0.0) or 0.0)
                    if (metrics and metrics[0] is not None
                            and time.monotonic() - stamp <= 0.6
                            and float(metrics[0]) >= servo_fill_start):
                        visual = (float(metrics[0]), float(metrics[1]))
                vx, wz = dock_creep_twist(
                    dx, dy, yaw, base_yaw, distance, speed, forward_gain,
                    yaw_tolerance, align_distance, visual=visual,
                )
                command.linear.x = vx
                command.angular.z = wz
                if expired():
                    return False
                if time.monotonic() >= attempt_deadline:
                    break
                self._fine_cmd_pub.publish(command)
            self._stop_base()
            for _ in range(3):
                if expired():
                    return False
                self._stop_base()
                remaining = max(0.0, shared_deadline - time.monotonic())
                rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
            if expired():
                return False

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
                return not expired()
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

    def _release_nav_control(
        self, label, goal_handle, result_future, timeout, *,
        deadline=None, grace_deadline=None,
    ):
        """Wait for a terminal Nav2 result before sending manual base commands.

        An acknowledged cancel request alone is not enough: FollowPath may still
        be publishing while cancellation is in flight. Fail closed if the
        action transport cannot establish that the previous goal has ended.
        """
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = leg_deadline if deadline is None else min(leg_deadline, deadline)
        try:
            if result_future is None:
                try:
                    result_future = goal_handle.get_result_async()
                    if result_future is None:
                        raise RuntimeError('get_result_async returned no future')
                except Exception:
                    # Without a result subscription we can request a stop but
                    # cannot prove ownership was released.
                    goal_handle.cancel_goal_async()
                    raise
            if not result_future.done():
                cancel_future = goal_handle.cancel_goal_async()
                if deadline is None:
                    rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=timeout)
                    rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
                else:
                    if grace_deadline is None:
                        now = time.monotonic()
                        grace_deadline = max(now, min(now + timeout, deadline)) + 0.5
                    # The terminal result is the authority, not cancel ACK.
                    while not result_future.done() and rclpy.ok():
                        remaining = grace_deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
            result = result_future.result() if result_future.done() else None
        except Exception as exc:
            self._last_attempt_failure = 'nav_control_not_released'
            self._navigation_control_fault = True
            raise RuntimeError(f'NAV_STATE_UNKNOWN: {label} terminal result unavailable') from exc
        if not result_future.done():
            self._last_attempt_failure = 'nav_control_not_released'
            self._navigation_control_fault = True
            raise RuntimeError(
                f'NAV_STATE_UNKNOWN: {label} cancellation not confirmed'
            )
        if result is None or getattr(result, 'status', None) not in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        ):
            self._last_attempt_failure = 'nav_control_not_released'
            self._navigation_control_fault = True
            raise RuntimeError(
                f'NAV_STATE_UNKNOWN: {label} has no terminal result'
            )
        self._stop_base()
        return result

    def _on_yield_gate_held(self, message: Bool) -> None:
        self._yield_gate_held = message.data
        self._yield_gate_status_time = time.monotonic()

    def _command_yield_gate(self, held: bool, timeout: float = 0.5) -> bool:
        """Wait for a fresh acknowledgement from the sole smoother-input gate."""
        if self._yield_hold_pub is None:
            return False
        since = time.monotonic()
        deadline = since + self._nav_budget_remaining(timeout)
        command = Bool()
        command.data = held
        while rclpy.ok() and (left := deadline - time.monotonic()) > 0.0:
            self._yield_hold_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=min(0.05, left))
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

    def _hold_outcome_text(self) -> str:
        """Describe why a traffic hold ended the way it did.

        A hold whose release is never verified used to fail the leg with no
        explanation at all: the only trace was a TIMING line. These four fields
        separate the cases that look identical from outside - the policy never
        released (holds=True), it released but the corridor was never verified
        (holds=False, cleared=False), or it gave up on its own budget
        (reason='timed out waiting for the corridor').
        """
        report = getattr(self, '_last_hold_report', None)
        if report is None:
            return 'held_for=? holds=? cleared=? reason=?'
        held_for, cleared, holds, reason = report
        return (f'held_for={held_for:.1f}s holds={holds} cleared={cleared} '
                f'reason={reason!r}')

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
        # Same rule as _hold_until_clear: traffic waiting gets its own budget
        # instead of the remains of the leg's, which can be a fraction of a
        # second by the time a box forces the hold (measured 2026-11-23).
        hold_budget = max(0.0, float(
            self.get_parameter('yield_wait_budget_sec').value))
        hold_deadline = started + hold_budget
        self.get_logger().warning(
            f'{label}: holding both velocity axes with Nav2 goal retained.')
        try:
            while (rclpy.ok() and self._yield_decision.holds
                   and not getattr(self, '_cancel_requested', False)
                   and not result_future.done() and time.monotonic() < hold_deadline):
                # Re-publish the hold: a BT plugin instantiated a tick late must
                # still observe it; the gate remains latched until explicit clear.
                message = Bool()
                message.data = True
                self._yield_hold_pub.publish(message)
                rclpy.spin_once(self, timeout_sec=max(0.0, min(0.1, hold_deadline - time.monotonic())))
                self._tick_yield(self._cached_planar_distance_to_point(x, y))
                if (self._yield_decision.holds and
                        self._yield_decision.reason == obstacle_yield.REASON_WAIT_EXPIRED):
                    break
            held_for = time.monotonic() - started
            self._last_hold_report = (
                held_for,
                bool(not self._yield_decision.holds
                     and not getattr(self, '_cancel_requested', False)
                     and rclpy.ok()),
                bool(self._yield_decision.holds),
                str(self._yield_decision.reason),
            )
            if (getattr(self, '_cancel_requested', False) or not rclpy.ok()
                    or time.monotonic() >= hold_deadline or result_future.done()
                    or self._yield_decision.holds or self._yield_arc_goal is not None):
                # A hold must never open the gate on an abandoned goal.
                if getattr(self, '_cancel_requested', False):
                    reason = 'supervisor cancelled the object'
                elif result_future.done():
                    reason = 'Nav2 ended the goal during the hold'
                elif self._yield_arc_goal is not None:
                    reason = 'taking the yield arc instead'
                else:
                    reason = 'hold ended without a clear corridor'
                self.get_logger().warning(
                    f'{label}: releasing the retained goal ({reason}) after '
                    f'{held_for:.1f} s of holding.'
                )
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout))
                self._command_yield_gate(False, timeout=1.0)
                return False
            if (not self._command_yield_gate(False, timeout=1.0)
                    or getattr(self, '_cancel_requested', False)):
                # A gate acknowledgement is not permission to continue a
                # goal if the supervisor cancelled it during the round trip.
                self.get_logger().error(
                    f'{label}: yield release interrupted; cancelling Nav2.'
                )
                message = Bool()
                message.data = True
                self._yield_hold_pub.publish(message)
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout))
                self._command_yield_gate(False, timeout=1.5)
                return False
            if held_for > 0.0:
                leg_deadline = getattr(self, '_nav_leg_deadline', None)
                if leg_deadline is not None:
                    self._nav_leg_deadline = leg_deadline + held_for
                self.get_logger().info(
                    f'{label}: corridor clear after {held_for:.1f} s of holding; '
                    'traffic time is not charged to the navigation budget.'
                )
            return True
        finally:
            self._yield_pause_active = False

    def _hold_until_clear(self, label, x, y, deadline) -> bool:
        """Stand still until the yield policy releases a mover hold.

        Runs with Nav2's goal already cancelled, so the controller publishes
        nothing and the zero Twist this loop sends to the smoother's input is
        the only command: BOTH axes stop, which is the part a Nav2 speed limit
        cannot do. Returns True when the corridor cleared, False when the leg
        should be failed instead.

        The wait is bounded by ``yield_wait_budget_sec`` and NOT by the leg's
        own deadline: measured 2026-11-23 (runs ..._postfix2/3), the leg budget
        was already down to 0.15-0.73 s when a patrol box forced a hold, so
        this call returned False within ~30 ms and both rounds died with the
        box still metres away. Waiting for traffic is not a navigation-budget
        failure, so the held time is also added back to the leg deadline.
        """
        self._yield_pause_active = True
        self.get_logger().warning(f'{label}: holding still for a moving obstacle.')
        started = time.monotonic()
        hold_budget = max(0.0, float(
            self.get_parameter('yield_wait_budget_sec').value))
        hold_deadline = started + hold_budget
        escape_after = float(self.get_parameter('yield_band_escape_after_sec').value)
        escapes_left = int(self.get_parameter('yield_band_escape_attempts').value)
        escapes = 0
        try:
            while (rclpy.ok() and self._yield_decision.holds
                    and not getattr(self, '_cancel_requested', False)
                    and time.monotonic() < hold_deadline):
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=max(0.0, min(0.1, hold_deadline - time.monotonic())))
                self._tick_yield(self._cached_planar_distance_to_point(x, y))
                if (self._yield_decision.holds and
                        self._yield_decision.reason == obstacle_yield.REASON_WAIT_EXPIRED):
                    break
                # Standing still inside an inscribed band is a deadlock, not a
                # hold: RPP cannot leave it (allow_reversing is false) and no
                # Nav2 recovery can run on a cancelled goal. Retry after the
                # hold has had its ordinary chance, and only when the footprint
                # is demonstrably in the band.
                if (escapes < escapes_left
                        and bool(self.get_parameter('yield_band_escape_enabled').value)
                        and time.monotonic() - started >= escape_after
                        and not self._local_recovery_pose_clear()):
                    escapes += 1
                    self.get_logger().warning(
                        f'{label}: held {time.monotonic() - started:.1f} s with the '
                        'footprint inside an inscribed band; attempting a guarded '
                        f'retreat (attempt {escapes}/{escapes_left}).'
                    )
                    self._back_off(
                        f'{label} band escape', 0.20, 0.06, -1.0,
                        leave_band=True, hold_escape=True, until=hold_deadline)
                    self._tick_yield(self._cached_planar_distance_to_point(x, y))
        finally:
            self._yield_pause_active = False
        held_for = time.monotonic() - started
        cleared = (rclpy.ok()
                   and not getattr(self, '_cancel_requested', False)
                   and not self._yield_decision.holds)
        self._last_hold_report = (
            held_for, bool(cleared), bool(self._yield_decision.holds),
            str(self._yield_decision.reason))
        if cleared and held_for > 0.0:
            leg_deadline = getattr(self, '_nav_leg_deadline', None)
            if leg_deadline is not None:
                self._nav_leg_deadline = leg_deadline + held_for
            self.get_logger().info(
                f'{label}: corridor clear after {held_for:.1f} s of holding; '
                'traffic time is not charged to the navigation budget.'
            )
        return cleared

    def _resume_nav_goal(self, label, goal, response_timeout, client=None, *, deadline=None):
        """Re-send a leg's own goal after a yield pause.

        Returns ``(goal_handle, result_future)`` or None when Nav2 refuses the
        goal, in which case the caller fails the leg and the ordinary retry
        loop takes over. The goal object is the identical one the leg started
        with - one pose for a plain leg, the whole detour chain for a
        continuous-detour leg - so a resume is the same leg continuing.
        """
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = leg_deadline if deadline is None else min(leg_deadline, deadline)
        if deadline is not None and time.monotonic() >= deadline:
            self._last_attempt_failure = 'deadline'
            self.get_logger().warning(
                f'{label}: resume refused, the leg deadline is already spent.')
            return None
        if getattr(self, '_cancel_requested', False):
            self._stop_base()
            return None
        goal_handle = self._send_goal_tolerant(
            label, client if client is not None else self._navigate_client,
            goal, response_timeout, 5.0, deadline=deadline,
        )
        expired = deadline is not None and time.monotonic() >= deadline
        if expired or getattr(self, '_cancel_requested', False):
            if goal_handle is not None and goal_handle.accepted:
                self._release_nav_control(
                    label, goal_handle, None,
                    min(5.0, response_timeout), deadline=deadline,
                )
            if expired:
                self._last_attempt_failure = 'deadline'
            return None
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f'Nav2 did not accept the resumed {label} goal; failing the leg.'
            )
            return None
        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            result_future = None
        if result_future is None:
            self._release_nav_control(
                label, goal_handle, None, min(5.0, response_timeout), deadline=deadline,
            )
            return None
        self.get_logger().warning(
            f'{label}: mover hold clear; resuming the navigation goal.'
        )
        return goal_handle, result_future

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
        index = min(self._chain_index, len(chain))
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
        if index == len(chain) - 1:
            last = self._cached_planar_distance_to_point(
                float(chain[index][0]), float(chain[index][1]))
            if last is not None and last <= 0.7:
                index += 1
        self._chain_index = index
        return (x, y) if index == len(chain) else (float(chain[index][0]), float(chain[index][1]))

    def _nav_goal(self, values: Sequence[float], chain=None, behavior_tree=None):
        """Build the Nav2 goal, and pick the client that must execute it.

        No ``chain`` means the original single-pose NavigateToPose goal. With a
        chain, it is ONE NavigateThroughPoses goal holding every detour waypoint
        in order followed by the leg's own target, so Nav2 plans and follows the
        configured polyline as a single path. A chain entry that repeats the
        target is dropped: two identical consecutive poses add nothing and some
        BT configurations stall on them.

        ``behavior_tree`` overrides the tree for this one goal. The
        narrow-entrance recovery passes a probe tree (plan only, no motion) and
        a recovery-free drive tree. The competition profile also uses
        recovery-free navigation for ordinary legs so nothing bypasses the
        executor's checks by clearing the costmap or moving inside a recovery.
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
            if behavior_tree:
                raise RuntimeError(
                    'a behaviour tree override cannot be combined with a '
                    'multi-pose chain')
            if getattr(self, '_retain_nav_goal_on_yield', False):
                goal.behavior_tree = os.path.join(
                    get_package_share_directory('mybot'), 'config',
                    'nav_through_poses_yield_pause.xml')
            elif getattr(self, '_guarded_navigation_recovery', False):
                goal.behavior_tree = self._entry_tree('nav_through_poses_guarded.xml')
            return goal, self._navigate_through_client
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(values)
        if behavior_tree:
            goal.behavior_tree = str(behavior_tree)
        elif getattr(self, '_retain_nav_goal_on_yield', False):
            goal.behavior_tree = os.path.join(
                get_package_share_directory('mybot'), 'config',
                'nav_to_pose_yield_pause.xml')
        elif getattr(self, '_guarded_navigation_recovery', False):
            goal.behavior_tree = self._entry_drive_tree()
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

    @staticmethod
    def _progress_watchdog_step(
        anchor: tuple[float, float] | None,
        since: float | None,
        position: tuple[float, float],
        now: float,
        distance_m: float,
        timeout_sec: float,
    ) -> tuple[tuple[float, float], float | None, bool]:
        """Update a no-progress baseline; return (anchor, since, expired).

        The timer restarts only when the base has moved ``distance_m`` from the
        baseline, not on pose noise. A caller that passes ``timeout_sec <= 0``
        never expires.
        """
        if (
            anchor is None
            or math.hypot(
                position[0] - anchor[0], position[1] - anchor[1]
            ) >= distance_m
        ):
            return position, now, False
        return anchor, since, (
            timeout_sec > 0.0
            and since is not None
            and now - since > timeout_sec
        )

    def _navigate_once__timed(
        self,
        label: str,
        values: Sequence[float],
        visual_handoff_cube: str | None = None,
        completion_radius_m: float | None = None,
        chain: Sequence[Sequence[float]] | None = None,
        progress_timeout_sec: float = 0.0,
        progress_distance_m: float = 0.10,
        max_duration_sec: float = 0.0,
        behavior_tree: str | None = None,
    ) -> bool:
        timeout = float(self.get_parameter('navigation_timeout_sec').value)
        if max_duration_sec > 0.0:
            timeout = min(timeout, float(max_duration_sec))
        deadline = time.monotonic() + timeout
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = min(deadline, leg_deadline)

        def deadline_expired():
            if time.monotonic() >= deadline:
                self._last_attempt_failure = 'deadline'
                return True
            return False

        if deadline_expired() or getattr(self, '_cancel_requested', False):
            self._stop_base()
            return False
        x, y, yaw = [float(value) for value in values]
        goal, client = self._nav_goal(
            values, self._remaining_chain(chain), behavior_tree=behavior_tree,
        )
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
        gate_clear = self._ensure_yield_gate_clear()
        if deadline_expired() or getattr(self, '_cancel_requested', False):
            self._stop_base()
            return False
        if not gate_clear:
            self._stop_base()
            raise RuntimeError(
                'YIELD_GATE_NOT_READY: guarded smoother input unavailable')
        goal_handle = self._send_goal_tolerant(
            label, client, goal, response_timeout, 5.0, deadline=deadline,
        )
        if goal_handle is None:
            deadline_expired()
            self._stop_base()
            return False
        if not goal_handle.accepted:
            self.get_logger().error(f'Nav2 rejected the {label} goal.')
            return False

        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            result_future = None
        if result_future is None:
            self._release_nav_control(
                label, goal_handle, None, min(5.0, response_timeout), deadline=deadline,
            )
            return False
        if deadline_expired() or getattr(self, '_cancel_requested', False):
            self._release_nav_control(
                label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline)
            return False
        handoff_distance = float(
            self.get_parameter('visual_handoff_distance_m').value
        )
        min_handoff_distance = float(
            self.get_parameter('visual_handoff_min_distance_m').value
        )
        accept_radius = float(
            self.get_parameter('navigation_accept_radius_m').value
        )
        accept_yaw = min(
            float(self.get_parameter('navigation_accept_yaw_rad').value),
            float(self.get_parameter('place_yaw_comp_limit_rad').value),
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
        if stall_timeout > 0.0 and self._entry_recovery_applies(label):
            entry_stall = float(self.get_parameter('entry_recovery_stall_timeout_sec').value)
            if entry_stall > 0.0:
                stall_timeout = min(stall_timeout, entry_stall)
        stall_progress = float(
            self.get_parameter('navigation_stall_progress_m').value
        )
        best_distance = None
        stall_since = None
        stall_waypoint = None
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
        progress_anchor = None
        progress_since = None
        progress_distance = max(0.01, float(progress_distance_m))
        progress_timeout = max(0.0, float(progress_timeout_sec))
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            if not getattr(self, '_cancel_requested', False):
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if deadline_expired() or getattr(self, '_cancel_requested', False):
                self._release_nav_control(
                    label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline)
                return False
            # The yield policy subsumes the slow-zone cap (it merges both and
            # publishes the lower one), so it replaces the old call here rather
            # than following it, which would publish twice per tick.
            if result_future.done():
                break
            self._tick_yield(self._cached_planar_distance_to_point(x, y))
            if deadline_expired():
                break
            if self._yield_suppress and self._yield_arc_abort is not None:
                # The mid-arc watch found the sidestep is no longer clear. Stop
                # here and hand the leg back: the policy re-runs on the next
                # attempt and may hold, crawl, or take the other side, which is
                # the whole point of not driving the arc blind.
                self.get_logger().warning(
                    f'{label}: abandoning the yield arc '
                    f'({self._yield_arc_abort}).')
                self._last_attempt_failure = 'yield_arc_aborted'
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout), deadline=deadline)
                self._stop_base()
                return False
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
                        self.get_logger().warning(
                            f'{label}: the retained hold never confirmed a '
                            'clear corridor; failing the attempt '
                            f'({self._hold_outcome_text()}).'
                        )
                        deadline_expired()
                        return False
                    # A cleared hold gave the waiting time back to the leg;
                    # refresh the local deadline before the loop re-checks it.
                    leg_deadline = getattr(self, '_nav_leg_deadline', None)
                    if leg_deadline is not None and leg_deadline > deadline:
                        deadline = leg_deadline
                    # A deliberate hold is not a jam. Resume the same top-level
                    # action with fresh no-motion and progress watch baselines.
                    motion_anchor = motion_since = None
                    best_distance = stall_since = accept_since = None
                    progress_anchor = progress_since = None
                    continue
                self._release_nav_control(
                    label, goal_handle, result_future,
                    min(5.0, response_timeout), deadline=deadline,
                )
                if not self._hold_until_clear(label, x, y, deadline):
                    self.get_logger().warning(
                        f'{label}: the hold never confirmed a clear corridor '
                        f'({self._hold_outcome_text()}); failing the attempt.'
                    )
                    deadline_expired()
                    return False
                # A cleared hold gives the waiting time back to the leg (see
                # _hold_until_clear): refresh the local deadline so the arc
                # check and the resume below do not fail on the old value.
                leg_deadline = getattr(self, '_nav_leg_deadline', None)
                if leg_deadline is not None and leg_deadline > deadline:
                    deadline = leg_deadline
                if self._yield_arc_goal is not None:
                    # The episode budget can expire inside the hold loop. Nav2
                    # is already cancelled; hand the pending arc to _navigate_to
                    # instead of resuming a goal just to cancel it immediately.
                    self.get_logger().warning(
                        f'{label}: mover hold budget spent; taking the yield '
                        f'arc ({self._hold_outcome_text()}).'
                    )
                    return False
                # Rebuild from the REMAINING waypoints: the base may have
                # reached one or two of them while it was held, and re-sending
                # the whole chain would send it back to the first one.
                if deadline_expired():
                    self.get_logger().warning(
                        f'{label}: leg deadline expired while resuming after a '
                        f'hold; failing the attempt ({self._hold_outcome_text()}).'
                    )
                    return False
                goal, client = self._nav_goal(
                    values, self._remaining_chain(chain), behavior_tree=behavior_tree,
                )
                resumed = self._resume_nav_goal(
                    label, goal, response_timeout, client, deadline=deadline,
                )
                if resumed is None:
                    self.get_logger().warning(
                        f'{label}: could not resume the goal after a hold '
                        f'(failure={self._last_attempt_failure!r}, '
                        f'{self._hold_outcome_text()}); failing the attempt.'
                    )
                    return False
                goal_handle, result_future = resumed
                # A deliberate yield is not evidence of a navigation stall.
                motion_anchor = motion_since = None
                best_distance = stall_since = accept_since = None
                progress_anchor = progress_since = None
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
                    min(5.0, response_timeout), deadline=deadline,
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
                            min(5.0, response_timeout), deadline=deadline,
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
                            min(5.0, response_timeout), deadline=deadline,
                        )
                        return False
            if progress_timeout > 0.0:
                # A sidestep is short; the ordinary 45 s stall window is far too
                # long to spend on one that has stopped moving. Measured
                # 2026-11-21: 0.70 m in 36.9 s, then the round failed carrying
                # the cube. Bound the arc itself so the caller can go back to
                # waiting or retry the original route.
                position = self._cached_base_position()
                if position is not None:
                    now = time.monotonic()
                    progress_anchor, progress_since, expired = (
                        self._progress_watchdog_step(
                            progress_anchor, progress_since, position, now,
                            progress_distance, progress_timeout,
                        )
                    )
                    if expired:
                        self._last_attempt_failure = 'arc_no_progress'
                        self.get_logger().error(
                            f'{label}: no progress beyond '
                            f'{progress_distance:.2f} m in '
                            f'{progress_timeout:.0f} s; abandoning this arc.'
                        )
                        self._release_nav_control(
                            label, goal_handle, result_future,
                            min(5.0, response_timeout), deadline=deadline,
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
                waypoint = getattr(self, '_chain_index', 0)
                if waypoint != stall_waypoint:
                    best_distance = stall_since = None
                    stall_waypoint = waypoint
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
                            min(5.0, response_timeout), deadline=deadline,
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
                    self._release_nav_control(
                        label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline,
                    )
                    for _ in range(3):
                        if deadline_expired() or getattr(self, '_cancel_requested', False):
                            return False
                        self._stop_base()
                        rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
                    return not deadline_expired() and not getattr(self, '_cancel_requested', False)
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
                            min(5.0, response_timeout), deadline=deadline,
                        )
                        if deadline_expired() or getattr(self, '_cancel_requested', False):
                            return False
                        reached = self._creep_to_dock(label, x, y, yaw, deadline=deadline)
                        return not deadline_expired() and reached
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
                            self._release_nav_control(
                                label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline,
                            )
                            return not deadline_expired() and not getattr(self, '_cancel_requested', False)
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
                label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline
            )
            if deadline_expired() or getattr(self, '_cancel_requested', False):
                return False
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
                if deadline_expired() or getattr(self, '_cancel_requested', False):
                    return False
                self._stop_base()
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
            if deadline_expired() or getattr(self, '_cancel_requested', False):
                return False
            # 2026-11-17: the retreat check above runs *before* the base has
            # stopped, so it only sees the distance at the moment Nav2 was
            # cancelled. Cancelling and flushing three zero setpoints costs
            # 0.3-1.0 s of travel, and the first attempts showed the base
            # arriving at 0.09 m (run_logs/legprobe_rule1_20261117-184656:
            # "... is only 0.093 m away - too close for the camera servo"),
            # while the handoff itself had been logged at 1.34 m. Below about
            # 0.45 m the cube sits under the level camera and the row prior in
            # pick_best runs past the bottom of the image, so every contour is
            # rejected and the only symptom is a 75 s "Visual blob lost"
            # timeout. Re-measure once the base is actually still.
            stopped_distance = self._cached_planar_distance_to_cube(
                visual_handoff_cube
            )
            coast_min = float(
                self.get_parameter('visual_handoff_coast_min_distance_m').value
            )
            if stopped_distance is not None and stopped_distance < coast_min:
                self.get_logger().warning(
                    f'{visual_handoff_cube}: base coasted to '
                    f'{stopped_distance:.3f} m '
                    f'(< {coast_min:.2f} m) after stopping; opening '
                    'the standoff before the camera servo.'
                )
                stopped_distance = self._retreat_to_handoff_standoff(
                    visual_handoff_cube, stopped_distance
                )
                if stopped_distance < coast_min:
                    self.get_logger().error(
                        f'{visual_handoff_cube}: retreat after stopping ended at '
                        f'{stopped_distance:.3f} m (< {coast_min:.2f} '
                        'm); refusing blind servo.'
                    )
                    return False
            shown = ('unknown' if stopped_distance is None
                     else f'{stopped_distance:.3f} m')
            self.get_logger().info(
                f'Nav2 control released at the {visual_handoff_cube} visual '
                f'handoff (base-to-cube {shown}).'
            )
            return not deadline_expired() and not getattr(self, '_cancel_requested', False)

        expired = deadline_expired()
        if expired:
            self.get_logger().error(f'Navigation to {label} timed out; cancelling.')
        wrapped_result = self._release_nav_control(
            label, goal_handle, result_future, min(5.0, response_timeout), deadline=deadline,
        )
        if expired or deadline_expired() or getattr(self, '_cancel_requested', False):
            return False
        status = wrapped_result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return True
        if status == GoalStatus.STATUS_ABORTED:
            # Use named constants from the actual result schema, never generic
            # status, log text, or a guessed numeric range. Humble's result may
            # not carry any diagnostic code at all.
            result = getattr(wrapped_result, 'result', None)
            code = getattr(result, 'error_code', None)
            for reason, codes in (
                ('planner_no_path', ('NO_VALID_PATH', 'PLANNER_NO_VALID_PATH')),
                ('controller_failure', ('NO_VALID_CONTROL', 'FAILED_TO_MAKE_PROGRESS',
                                        'CONTROLLER_NO_VALID_CONTROL',
                                        'CONTROLLER_FAILED_TO_MAKE_PROGRESS')),
            ):
                if code and any(code == getattr(type(result), name, None) for name in codes):
                    self._last_attempt_failure = reason
                    break
        self.get_logger().error(f'Navigation to {label} failed; status={status}.')
        return False

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

    def _start_cell_inside_global_band(self) -> bool:
        """Test a fresh published start cell; includes lethal for entry guards."""
        cost = self._global_recovery_cost()
        return cost is not None and cost >= 99

    def _relax_global_footprint(self, label: str) -> bool:
        """Remove only the extra global radius for at most one verified goal."""
        if (getattr(self, '_relax_used', False) or self._relaxed_global
                or not self.get_parameter('stuck_relax_global_radius').value
                or self._nominal_geometry is None):
            return False
        if self._global_recovery_cost() != 99 or not self._local_recovery_pose_clear():
            return False
        radius, _, _ = self._nominal_geometry
        margin = float(self.get_parameter('global_radius_margin_m').value)
        if not math.isfinite(margin) or margin <= 0.0:
            return False
        nominal_radius = radius + margin
        recovery_radius = self._recovery_envelope_radius()
        if recovery_radius >= nominal_radius or self._entry_yield_pending():
            return False
        self._relax_used = True
        before = self._update_costmap_geometry(
            {'global': {'robot_radius': recovery_radius}},
            expected={'global': {'robot_radius': nominal_radius}})
        if before is None:
            return False
        self._global_radius_before_relax = nominal_radius
        self._relaxed_global = True
        stamp = self.get_clock().now().nanoseconds
        deadline = time.monotonic() + self._nav_budget_remaining(7.0)
        while rclpy.ok() and not getattr(self, '_cancel_requested', False):
            left = deadline - time.monotonic()
            if left <= 0.0:
                break
            newer = self._grid_stamp(self._global_costmap)
            if newer is not None and (stamp is None or newer > stamp):
                cost = self._global_recovery_cost()
                if (cost is not None and 0 <= cost < 99 and self._local_recovery_pose_clear()
                        and not self._entry_yield_pending()):
                    self.get_logger().warning(
                        f'{label}: verified global-only inscribed band; this goal uses '
                        f'{recovery_radius:.2f} m planning radius instead of {nominal_radius:.2f} m.')
                    return True
                if cost is not None and cost != 99:
                    break
            rclpy.spin_once(self, timeout_sec=min(0.05, left))
        self.get_logger().warning(f'{label}: no verified clear start after global relax.')
        self._restore_nominal_geometry(label)
        return False

    def _restore_nominal_geometry(self, label: str) -> None:
        if not self._relaxed_global:
            return
        old_radius = self._global_radius_before_relax
        if (old_radius is None or self._update_costmap_geometry(
                {'global': {'robot_radius': old_radius}}, cleanup=True) is None):
            self._geometry_fault = True
            raise RuntimeError(f'GEOMETRY_STATE_UNKNOWN: {label} nominal restore unconfirmed')
        self._relaxed_global = False
        self._global_radius_before_relax = None
        self.get_logger().info(f'{label}: nominal global radius {old_radius:.2f} m restored.')

    def _back_off(
        self, label: str, distance_m: float = 0.25, speed: float = 0.08,
        direction: float | None = -1.0, *, leave_band: bool = False,
        until: float = math.inf, hold_escape: bool = False,
        clearance_floor: float | None = None,
    ) -> bool:
        """Attempt one verified straight retreat without changing either footprint.

        ``hold_escape`` relaxes the "never move while a yield holds" rule for the
        one case where standing still is the unsafe option: the footprint sits in
        an inscribed band, RPP refuses every arc, and the hold itself is what
        keeps the base there. The retreat stays guarded by the same per-step
        scan and swept-costmap checks as any other shove.

        ``clearance_floor`` replaces the default scan gate (``min_scan + 0.10``,
        i.e. 0.60 m) for callers that only need a short, band-clearing shove.
        Measured 2026-11-25 (runs/2026-11-25_1918_servo_m3): the base was wedged
        0.32 m from the wall with 0.50-0.51 m of room on the free side, so the
        0.60 m gate refused the very retreat that would have freed it and the
        leg died twice with "no verified recovery".
        """
        if (not self.get_parameter('dead_zone_escape_enabled').value
                or getattr(self, '_geometry_fault', False)
                or getattr(self, '_navigation_control_fault', False)):
            return False
        distance_m, speed = min(distance_m, 0.25), min(abs(speed), 0.08)
        if (not math.isfinite(distance_m) or distance_m <= 0.0
                or not math.isfinite(speed) or speed <= 0.0):
            return False
        axis = self._clearer_axis() if direction is None else (1.0 if direction > 0 else -1.0)
        command = Twist()
        command.linear.x = axis * speed
        # A retreat that frees the footprint from an inscribed band is recovery,
        # not progress: the leg budget is usually already spent when the band
        # trap appears (measured 2026-11-23, ..._holdfix2/3: three escapes
        # stopped at 0.00 m with "time budget" while 0.16 m from a wall). Give
        # the guarded escape its own short window instead.
        retreat_budget = (4.0 if (hold_escape or leave_band)
                          else self._nav_budget_remaining(7.0))
        # A caller that hands over an already-expired deadline (the leg budget
        # is usually spent by the time a band trap appears) must not turn the
        # escape into a zero-length window: measured 2026-11-23 on
        # ..._contactfix1, "band escape measured 0.00 m; stop: time budget"
        # twice, 1 ms apart, with the retreat window already in the past.
        budget_start = time.monotonic()
        window_end = budget_start + retreat_budget
        deadline = min(until, window_end) if until > budget_start else window_end
        start = self._recovery_pose('odom')
        travelled = 0.0
        # A None start is a refusal before the window opens, not a spent
        # window: name it so the record cannot be misread as "time budget".
        reason = 'time budget' if start is not None else 'stale odometry'
        begun = time.monotonic()
        last_progress, best = begun, 0.0
        min_scan = max(0.50, float(self.get_parameter('entry_recovery_stop_clearance_m').value))
        floor = min_scan + 0.10 if clearance_floor is None else float(clearance_floor)
        try:
            while (start is not None and rclpy.ok() and time.monotonic() < deadline
                   and not getattr(self, '_cancel_requested', False)):
                if self._yield_config.enabled and not hold_escape:
                    self._tick_yield()
                if not hold_escape and self._entry_yield_pending():
                    reason = 'moving obstacle/yield'
                    break
                pose = self._recovery_pose('odom')
                if pose is None:
                    reason = 'stale odometry'
                    break
                dx, dy = pose[0] - start[0], pose[1] - start[1]
                travelled = axis * (dx * math.cos(start[2]) + dy * math.sin(start[2]))
                yaw_change = math.atan2(math.sin(pose[2] - start[2]), math.cos(pose[2] - start[2]))
                if (travelled < -0.02 or abs(yaw_change) > 0.12
                        or abs(dy * math.cos(start[2]) - dx * math.sin(start[2])) > 0.04):
                    reason = 'unexpected motion'
                    break
                if travelled >= distance_m:
                    reason = 'measured distance'
                    break
                if leave_band and travelled >= 0.05:
                    cost = self._global_recovery_cost()
                    if cost is not None and 0 <= cost < 99:
                        reason = 'left planner band'
                        break
                if travelled >= best + 0.015:
                    best, last_progress = travelled, time.monotonic()
                if time.monotonic() - last_progress >= 3.0:
                    reason = 'no measured progress'
                    break
                clearance = self._entry_scan_clearance(axis)
                if clearance is None or clearance <= floor:
                    # Name the measured room, not just the verdict: on the
                    # 2026-11-25 rule-3 round both band-escape attempts reported
                    # this line while a patrol box stood 0.20 m off the flank, so
                    # "not verified" and "there really is no room on that side"
                    # were indistinguishable in the log.
                    reason = (
                        'scan clearance not verified'
                        if clearance is None else
                        f'scan clearance {clearance:.2f} m <= '
                        f'{floor:.2f} m along the retreat axis')
                    break
                if not self._recovery_corridor_clear(axis, 0.10):
                    # A footprint inside an inscribed band makes the swept
                    # check refuse the very retreat that would leave the band:
                    # the band is 99 all around it. Measured 2026-11-23 on
                    # ..._bandfix2, 167 collision refusals in 18 s and then
                    # "guarded retreat measured -0.00 m; stop: scan or swept
                    # costmap clearance not verified". For a band escape the
                    # scan above already proves nothing solid is in the way, so
                    # only a *static lethal* cell (the wall itself) may veto.
                    if not (leave_band
                            and self._band_escape_cell_free(axis, 0.18)):
                        reason = 'swept costmap clearance not verified'
                        break
                left = deadline - time.monotonic()
                if left <= 0.0:
                    break
                self._fine_cmd_pub.publish(command)
                rclpy.spin_once(self, timeout_sec=min(0.05, left))
        finally:
            self._stop_base()
        self.get_logger().warning(
            f'{label}: guarded retreat measured {travelled:.2f} m; stop: {reason}.')
        if (getattr(self, '_cancel_requested', False)
                or (not hold_escape and self._entry_yield_pending())
                or self._nav_budget_remaining() <= 0.0 or time.monotonic() >= until
                or reason in ('unexpected motion', 'stale odometry',
                              'scan clearance not verified',
                              'swept costmap clearance not verified')):
            return False
        if leave_band:
            cost = self._global_recovery_cost()
            return travelled >= 0.05 and cost is not None and 0 <= cost < 99
        return travelled >= 0.05

    # ------------------------------------------------------ narrow-entrance P0
    def _entry_tree(self, name: str) -> str:
        """Absolute path of one of the recovery's own behaviour trees."""
        path = os.path.join(
            get_package_share_directory('mybot'), 'config', name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f'ENTRY_RECOVERY behaviour tree is not installed: {path}')
        return path

    def _entry_probe_tree(self) -> str:
        return self._entry_tree('nav_to_pose_entry_probe.xml')

    def _entry_drive_tree(self) -> str:
        return self._entry_tree('nav_to_pose_entry_guard.xml')

    def _entry_wait_pose(self) -> list[float] | None:
        """The clearance-verified staging point for the configured zone."""
        return entry_recovery_pose(
            self.get_parameter('entry_recovery_wait_pose').value
        )

    def _entry_stage_poses(self) -> list[list[float]]:
        """Staging points, primary first; the alternate must really differ.

        A retry that re-enters from the pose the previous attempt failed at
        repeats the same approach - the 2026-11-21 round re-started C from
        0.125 m from the previous start. Anything closer than 0.40 m to the
        primary is dropped rather than pretending to be an alternative.
        """
        primary = self._entry_wait_pose()
        if primary is None:
            return []
        poses = [primary]
        alternate = entry_recovery_pose(
            self.get_parameter('entry_recovery_alt_pose').value
        )
        if alternate is not None and math.hypot(
            alternate[0] - primary[0], alternate[1] - primary[1],
        ) >= 0.40:
            poses.append(alternate)
        return poses

    def _entry_recovery_applies(self, label: str) -> bool:
        """True only for the final dock leg of the configured zone.

        The detour legs (`<cube> to <zone> continuous detour`) and the yield
        arcs keep the ordinary ladder: the recovery is about the last metres
        into a mouth, not about the whole route.
        """
        if self._entry_recovery_running:
            return False
        if not bool(self.get_parameter('entry_recovery_enabled').value):
            return False
        zone = str(
            self.get_parameter('entry_recovery_zone').value
        ).strip().upper()
        if not zone or not self._entry_stage_poses():
            return False
        return label.strip().upper() == f'WAREHOUSE {zone}'

    def _entry_yield_pending(self) -> bool:
        """A yield decision or a queued arc means the base must not be shoved."""
        decision = getattr(self, '_yield_decision', None)
        if bool(getattr(decision, 'holds', False)):
            return True
        return self._yield_arc_goal is not None

    def _note_scan_refusal(self, reason: str) -> None:
        """Say why a lidar-verified move was refused instead of refusing silently.

        Measured 2026-11-25 (runs/2026-11-25_1545_rule3_holdlog): every retreat
        reported "scan clearance not verified" while YIELD_DIAG showed
        scan_age=0.00 s and 360 points, so the sweep itself was arriving and one
        of the preconditions below was failing. Without this line the operator
        cannot tell a missing sweep from a stamp that is a second into the
        future, and the whole recovery path looks like it simply did nothing.
        Throttled: the retreat polls this at 20 Hz.
        """
        now = time.monotonic()
        if now - getattr(self, '_scan_refusal_logged_at', 0.0) < 1.0:
            return
        self._scan_refusal_logged_at = now
        self.get_logger().warning(
            f'lidar clearance not verified ({reason}); refusing to move on it.')

    def _stamp_age_sec(self, msg) -> float | None:
        stamp = self._grid_stamp(msg)
        if stamp is None:
            return None
        return (self.get_clock().now().nanoseconds - stamp) * 1e-9

    def _entry_scan_clearance(self, direction: float) -> float | None:
        """Travel-direction clearance from a fresh scan, or None when unknown.

        None covers both "no sweep" and "stale sweep" and the caller must treat
        it as a refusal to move: the failed round reversed on a single
        comparison taken before the shove started.
        """
        scan = self._scan
        if scan is None:
            self._note_scan_refusal('no sweep received yet')
            return None
        max_age = max(0.05, float(
            self.get_parameter('entry_recovery_scan_max_age_sec').value))
        if self._scan_received_at is None:
            self._note_scan_refusal('no arrival time recorded')
            return None
        arrival_age = time.monotonic() - self._scan_received_at
        if arrival_age > max_age:
            self._note_scan_refusal(
                f'arrival age {arrival_age:.2f} s > {max_age:.2f} s')
            return None
        if not self._recovery_stamp_fresh(scan, 0.5):
            self._note_scan_refusal(
                f'header stamp age {self._stamp_age_sec(scan)} s outside '
                '[-1.00, 0.50] s')
            return None
        if (not math.isfinite(scan.range_min) or scan.range_min < 0.0
                or not math.isfinite(scan.range_max)
                or scan.range_max <= scan.range_min):
            self._note_scan_refusal(
                f'range bounds min={scan.range_min} max={scan.range_max}')
            return None
        return travel_clearance(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, direction,
        )

    def _entry_travelled(self, start) -> float:
        current = self._cached_base_pose()
        if start is None or current is None:
            return 0.0
        return math.hypot(current[0] - start[0], current[1] - start[1])

    def _entry_retreat(self, label: str, direction, distance_m: float,
                       speed: float, deadline: float) -> bool:
        return self._back_off(f'ENTRY_RECOVERY {label}', distance_m, speed,
                              direction, leave_band=True, until=deadline)

    def _entry_leave_band(self, label: str, deadline: float) -> bool:
        """Clear the planner band with at most two guarded retreats.

        The lidar-chosen axis is tried first: on 2026-11-21 that is what freed
        this pose (0.71 m south-west, heading 218 deg), while reversing along
        the heading the base arrived on pointed at the wall tip.
        """
        cost = self._global_recovery_cost()
        if cost is None:
            return False
        if cost < 99:
            return True
        distance = max(0.0, float(
            self.get_parameter('entry_recovery_retreat_m').value))
        speed = max(0.02, float(
            self.get_parameter('entry_recovery_retreat_speed_mps').value))
        for direction, what in ((None, 'the clearer lidar axis'), (-1.0, 'reverse')):
            cost = self._global_recovery_cost()
            if cost is None:
                return False
            if cost < 99:
                return True
            if time.monotonic() >= deadline:
                break
            position = self._cached_base_pose()
            where = ('an unknown pose' if position is None else
                     f'({position[0]:.2f}, {position[1]:.2f}, '
                     f'{math.degrees(position[2]):.0f} deg)')
            self.get_logger().warning(
                f'ENTRY_RECOVERY {label}: start cell inside the global band at '
                f'{where}; guarded retreat along {what} ({distance:.2f} m).'
            )
            if self._entry_retreat(label, direction, distance, speed, deadline):
                return True
        cost = self._global_recovery_cost()
        return cost is not None and cost < 99

    def _entry_probe(self, label: str, values, timeout: float) -> str:
        """Ask the global planner for a route without moving the base.

        Returns 'path', 'no_path' or 'unknown'. 'unknown' covers a missing goal
        response, a rejected goal and a probe that timed out; it is never
        treated as "no path", because a behaviour tree that failed to load
        looks exactly like that.

        The distinction matters on this map: the global costmap carries only
        the static layer and the inflation layer (no moving obstacles), so a
        'no_path' answer cannot be fixed by waiting for a patrol box - it is a
        property of the mouth and the loaded envelope.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = min(deadline, leg_deadline)
        if time.monotonic() >= deadline or getattr(self, '_cancel_requested', False):
            return 'unknown'
        goal, client = self._nav_goal(
            values, None, behavior_tree=self._entry_probe_tree())
        response_timeout = float(
            self.get_parameter('service_timeout_sec').value)
        options = {} if leg_deadline is None else {'deadline': deadline}
        handle = self._send_goal_tolerant(
            f'{label} entry probe', client, goal,
            min(response_timeout, max(0.0, deadline - time.monotonic())),
            3.0, **options,
        )
        if handle is None or not handle.accepted:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: the plan-only probe was not accepted '
                f'({self._entry_probe_tree()}); treating as unknown.'
            )
            return 'unknown'
        try:
            future = handle.get_result_async()
        except Exception:
            future = None
        if future is None:
            if leg_deadline is not None:
                self._release_nav_control(
                    f'{label} entry probe', handle, None, response_timeout,
                    deadline=deadline,
                )
            else:
                handle.cancel_goal_async()
            return 'unknown'
        # Do not use rclpy.spin_until_future_complete here: it builds its own
        # executor and never delivered this action result, so a probe that the
        # bt_navigator finished in 150 ms (`Goal succeeded`) was scored as a
        # 5 s timeout and the recovery failed closed on a working probe. This
        # node already spins with rclpy.spin_once everywhere else; poll the
        # future with the same mechanism and keep one deadline.
        while (rclpy.ok() and not future.done()
               and not getattr(self, '_cancel_requested', False)):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
        if (not future.done() or time.monotonic() >= deadline
                or getattr(self, '_cancel_requested', False)):
            if leg_deadline is not None:
                self._release_nav_control(
                    f'{label} entry probe', handle, future, response_timeout,
                    deadline=deadline,
                )
            elif not future.done():
                handle.cancel_goal_async()
                self._stop_base()
            self.get_logger().warning(
                f'ENTRY_RECOVERY {label}: plan-only probe timed out after '
                f'{timeout:.1f} s; treating as unknown.'
            )
            return 'unknown'
        verified = (self._release_nav_control(
            f'{label} entry probe', handle, future, response_timeout, deadline=deadline,
        ) if leg_deadline is not None else future.result())
        status = getattr(verified, 'status', None)
        if status == GoalStatus.STATUS_SUCCEEDED:
            return 'path'
        if status == GoalStatus.STATUS_ABORTED:
            return 'no_path'
        return 'unknown'

    def _entry_stage(self, label: str, pose, reach: float, yaw_tol: float,
                     timeout: float, deadline: float) -> bool:
        """Put the base on a staging point and verify it is really there.

        Two things the old version got wrong: an unknown pose was read as
        "0 m away", i.e. as already staged, and the arrival radius (0.45 m)
        was larger than the distance between the failed pose and the waiting
        point (0.125 m), so the same approach could be replayed.
        """
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = min(deadline, leg_deadline)
        if time.monotonic() >= deadline or getattr(self, '_cancel_requested', False):
            return False
        current = self._cached_base_pose()
        if current is None:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: base pose unknown; cannot verify the '
                'staging point, so the recovery fails closed.'
            )
            return False
        distance = math.hypot(current[0] - pose[0], current[1] - pose[1])
        if distance <= reach:
            yaw_error = abs(math.atan2(
                math.sin(current[2] - float(pose[2])),
                math.cos(current[2] - float(pose[2])),
            ))
            if yaw_error > 2.0 * yaw_tol:
                self.get_logger().warning(
                    f'ENTRY_RECOVERY {label}: staged at '
                    f'({current[0]:.2f}, {current[1]:.2f}) but '
                    f'{math.degrees(yaw_error):.0f} deg off the configured '
                    'heading; continuing from the actual pose.'
                )
            return time.monotonic() < deadline and not getattr(self, '_cancel_requested', False)
        budget = min(float(timeout), max(0.0, deadline - time.monotonic()))
        if budget < 1.0:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: no budget left to reach the staging '
                'point.'
            )
            return False
        self.get_logger().warning(
            f'ENTRY_RECOVERY {label}: staging at '
            f'({pose[0]:.2f}, {pose[1]:.2f}), {distance:.2f} m away.'
        )
        ok = self._navigate_once(
            f'{label} entry wait', pose,
            completion_radius_m=reach,
            max_duration_sec=budget,
            behavior_tree=self._entry_drive_tree(),
        )
        if not ok:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: could not reach the staging point; '
                'failing closed.'
            )
            return False
        current = self._cached_base_pose()
        if current is None:
            return False
        distance = math.hypot(current[0] - pose[0], current[1] - pose[1])
        if distance > 1.5 * reach:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: staged {distance:.2f} m from the '
                'staging point (over the verified radius); failing closed.'
            )
            return False
        return time.monotonic() < deadline and not getattr(self, '_cancel_requested', False)

    def _entry_recovery(
        self, label: str, values, visual_handoff_cube, completion_radius_m,
        chain,
    ) -> bool:
        """Recover a narrow entrance with one deadline and no blind motion.

        Measured 2026-11-21 (red_cube_1 -> C, 0/5): the base cut the corner into
        a 0.89 m mouth and stopped 0.30 m from the wall tip. That pose is inside
        the inscribed band of both the 0.45 m planned radius and the 0.35 m
        local one, so NavFn rejected the start cell before searching, the
        relaxed attempt was skipped for the same reason, RPP reported
        "collision ahead", and the 0.70 m blind shove moved the base 0.03 m.
        Both retries repeated exactly that, and the leg burned 174 s.

        The pass, in order:

        1. Leave the band with `_entry_leave_band` - scan-guarded retreats, no
           map clear and no footprint shrink. Neither can change a physical
           intrusion, and the clearance that refused the motion is still there.
        2. Stage at one of the configured waiting points (they alternate between
           attempts) and verify the base is really within `reach` of it.
        3. Probe the global planner with a plan-only tree. A 'no_path' answer is
           a static property of the mouth for the loaded envelope, so the pass
           moves to the other staging point instead of waiting for a patrol box
           the global costmap does not even carry.
        4. Drive with a recovery-free tree, so the only thing that can retreat,
           wait or give up is this method. Each failure is classified with the
           distance the base actually covered (`classify_entrance_failure`):
           Nav2's own backup/spin used to move the base between attempts, which
           made "the same pose keeps failing" impossible to verify.
        5. Stop at `entry_recovery_max_attempts` or the shared deadline,
           whichever comes first, and fail closed.

        Returns True only when the dock leg completes; False is a bounded exit
        and the caller must not fall back to the shrink/clear/shove ladder.
        """
        poses = self._entry_stage_poses()
        if not poses:
            return False
        budget = max(0.0, float(
            self.get_parameter('entry_recovery_wait_budget_sec').value))
        attempts = max(1, int(
            self.get_parameter('entry_recovery_max_attempts').value))
        attempt_timeout = max(1.0, float(
            self.get_parameter('entry_recovery_attempt_timeout_sec').value))
        probe_timeout = max(0.5, float(
            self.get_parameter('entry_recovery_plan_abort_sec').value))
        reach = max(0.05, float(
            self.get_parameter('entry_recovery_reach_m').value))
        yaw_tol = max(0.05, float(
            self.get_parameter('entry_recovery_yaw_tol_rad').value))
        deadline = time.monotonic() + budget
        leg_deadline = getattr(self, '_nav_leg_deadline', None)
        if leg_deadline is not None:
            deadline = min(deadline, leg_deadline)
        self._nav_leg_deadline = deadline
        no_path_poses = 0
        controller_refusals = 0
        drive_refusals = 0
        failed_starts: list[tuple[float, float]] = []
        self._entry_recovery_running = True
        try:
            if time.monotonic() >= deadline or getattr(self, '_cancel_requested', False):
                return False
            if not self._entry_leave_band(label, deadline):
                self.get_logger().error(
                    f'ENTRY_RECOVERY {label}: still inside the planner band '
                    'after the guarded retreats; failing closed instead of '
                    'shrinking the loaded footprint.'
                )
                return False
            for attempt in range(1, attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining < 2.0:
                    break
                pose = poses[(attempt - 1) % len(poses)]
                if not self._entry_stage(
                        label, pose, reach, yaw_tol,
                        min(attempt_timeout, remaining), deadline):
                    return False
                start = self._cached_base_pose()
                if start is not None and any(
                    math.hypot(start[0] - fx, start[1] - fy) <= reach
                    for fx, fy in failed_starts
                ):
                    self.get_logger().warning(
                        f'ENTRY_RECOVERY {label}: attempt {attempt} would '
                        'start from a pose that already failed; skipping it.'
                    )
                    continue
                remaining = deadline - time.monotonic()
                if remaining < 2.0:
                    break
                verdict = self._entry_probe(
                    label, values, min(probe_timeout, remaining))
                if verdict == 'unknown':
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0 or getattr(self, '_cancel_requested', False):
                        return False
                    verdict = self._entry_probe(
                        label, values, min(probe_timeout, remaining))
                if verdict == 'no_path':
                    no_path_poses += 1
                    self.get_logger().warning(
                        f'ENTRY_RECOVERY {label}: no global route to the dock '
                        f'from the staging point on attempt {attempt}; the '
                        'global costmap carries no moving obstacles, so this is '
                        'treated as a static block, not as something to wait '
                        'out.'
                    )
                    if no_path_poses >= len(poses):
                        self.get_logger().error(
                            f'ENTRY_RECOVERY {label}: the planner reports no '
                            f'route from every staging point '
                            f'({no_path_poses}); failing closed.'
                        )
                        return False
                    continue
                if verdict == 'unknown':
                    self.get_logger().error(
                        f'ENTRY_RECOVERY {label}: the plan-only probe could not '
                        'be evaluated; failing closed rather than driving '
                        'unverified.'
                    )
                    return False
                remaining = deadline - time.monotonic()
                if remaining < 2.0:
                    break
                self._last_attempt_failure = None
                started = time.monotonic()
                before = self._cached_base_pose()
                ok = self._navigate_once(
                    label, values, visual_handoff_cube, completion_radius_m,
                    None,
                    max_duration_sec=min(attempt_timeout, remaining),
                    behavior_tree=self._entry_drive_tree(),
                )
                elapsed = time.monotonic() - started
                if (getattr(self, '_last_attempt_failure', None) == 'nav_control_not_released'
                        or time.monotonic() >= deadline
                        or getattr(self, '_cancel_requested', False)):
                    return False
                self._stop_base()
                moved = self._entry_travelled(before)
                if ok:
                    self.get_logger().info(
                        f'ENTRY_RECOVERY {label}: entrance cleared on attempt '
                        f'{attempt} after {elapsed:.1f} s, from '
                        f'{describe_pose(before)} to '
                        f'{describe_pose(self._cached_base_pose())}.'
                    )
                    return True
                if before is not None:
                    failed_starts.append((before[0], before[1]))
                verdict = classify_entrance_failure(
                    start_in_band=self._start_cell_inside_global_band(),
                    failure=self._last_attempt_failure,
                    moved_m=moved,
                    elapsed_sec=elapsed,
                    plan_abort_sec=probe_timeout,
                )
                self.get_logger().warning(
                    f'ENTRY_RECOVERY {label}: attempt {attempt} failed after '
                    f'{elapsed:.1f} s having moved {moved:.2f} m '
                    f'({verdict}) from {describe_pose(before)}.'
                )
                if verdict == 'band':
                    if not self._entry_leave_band(label, deadline):
                        self.get_logger().error(
                            f'ENTRY_RECOVERY {label}: could not leave the band '
                            'after the attempt; failing closed.'
                        )
                        return False
                    continue
                if verdict == 'controller':
                    controller_refusals += 1
                    if controller_refusals >= attempts:
                        self.get_logger().error(
                            f'ENTRY_RECOVERY {label}: the controller refused '
                            f'every route ({controller_refusals}); failing '
                            'closed.'
                        )
                        return False
                    continue
                if verdict == 'refused':
                    drive_refusals += 1
                    if drive_refusals > 1:
                        self.get_logger().error(
                            f'ENTRY_RECOVERY {label}: the planner reported a '
                            'route but the drive tree failed again without '
                            'moving; check that '
                            f'{self._entry_drive_tree()} loads. Failing closed.'
                        )
                        return False
                    continue
                continue
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: entrance still blocked after '
                f'{budget:.1f} s and {attempts} attempt(s); failing closed '
                'without shrinking the loaded footprint or clearing the '
                'costmaps.'
            )
            return False
        except RuntimeError as error:
            self.get_logger().error(
                f'ENTRY_RECOVERY {label}: aborted ({error}); failing closed.'
            )
            return False
        finally:
            if (time.monotonic() >= deadline and
                    getattr(self, '_last_attempt_failure', None) != 'nav_control_not_released'):
                self._last_attempt_failure = 'deadline'
            self._nav_leg_deadline = leg_deadline
            self._entry_recovery_running = False

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
            servo = self._start_place_servo(label)
            try:
                return self._creep_to_dock__timed(label, *a, **k)
            finally:
                if servo:
                    self._stop_place_servo()

    def _servo_house(self, label: str) -> str:
        """The warehouse letter of a creep label, or '' when there is none."""
        match = re.search(r'warehouse\s*([ABCabc])\b', label or '')
        return match.group(1).upper() if match else ''

    def _start_place_servo(self, label: str) -> bool:
        """Route the camera at the plate while the base closes the last metre.

        The plate is 0.02 m thick and the camera sits 0.1 m up, so it only
        enters the frame in the last ~0.4 m; sampling earlier is wasted.  The
        sampling shares the gate's plumbing (`_plate_gate_active`) so there is
        one producer of `_plate_sample_metrics` and one way to stop it.
        """
        if not bool(self.get_parameter('place_servo').value):
            return False
        house = self._servo_house(label)
        if not house:
            return False
        self._plate_gate_active = True
        self._plate_gate_house = house
        self._vision_enabled = True
        return True

    def _stop_place_servo(self) -> None:
        self._plate_gate_active = False
        self._plate_gate_house = ''
        self._vision_enabled = False

    def _visual_align_active(self, *a, **k):
        with self._op('vision_align'):
            return self._visual_align_active__timed(*a, **k)

    def _band_escape_axis(self) -> tuple[float, float | None]:
        """Widest of the two straight retreat directions, with the room the scan sees."""
        best_axis, best_room = -1.0, None
        for candidate in (-1.0, 1.0):
            try:
                room = self._entry_scan_clearance(candidate)
            except Exception:  # noqa: BLE001 - diagnostics must never break a leg
                room = None
            if room is not None and (best_room is None or room > best_room):
                best_axis, best_room = candidate, room
        return best_axis, best_room

    def _ensure_start_pose_outside_band(self, label: str) -> None:
        """Retreat before driving when the base itself stands in an inscribed band.

        Measured 2026-11-23 (runs/2026-11-23_rule3_holdfix3): after a yield hold
        the base stood 0.158 m from a wall, i.e. inside its own global inscribed
        band. From there every GridBased request was aborted ("failed to
        generate a valid path to (-2.84, -11.58)") because a lethal start pose
        has no admissible plan; the leg then failed as status=6 and the executor
        reported "no verified recovery". No planner can plan its way out of a
        band, so back off first, then let the leg plan from a clear pose.

        The shove now follows the scan instead of a fixed reverse direction and
        is bounded per leg: measured 2026-11-25 (runs/2026-11-25_1918_servo_m3)
        the base sat 0.32 m from the wall with 0.51 m of room on its free side,
        and the old fixed reverse - plus the 0.60 m scan gate - refused the one
        shove that would have freed the leg.
        """
        if not bool(self.get_parameter('start_band_escape_enabled').value):
            return
        local_clear = True
        global_cost = None
        try:
            local_clear = self._local_recovery_pose_clear()
            global_cost = self._global_recovery_cost()
        except Exception:  # noqa: BLE001 - diagnostics must never break a leg
            return
        # The planner plans in the *global* grid, so a clear local grid is not
        # enough: measured 2026-11-25 (runs/2026-11-25_2022_l2_m2) the base
        # stopped 0.21 m from a static wall, the laser saw open floor and this
        # gate returned, while every GridBased request aborted 58 times because
        # the start cell sat in the static lethal band. Global cost is None when
        # no grid is published yet, which must stay "no opinion", not "blocked".
        if local_clear and (global_cost is None or global_cost < 99):
            return
        where = ('local clear, global cost '
                 f'{global_cost}' if local_clear else 'local grid dirty')
        attempts = max(0, int(self.get_parameter('start_band_escape_attempts').value))
        used = int(getattr(self, '_start_band_escapes', 0))
        if used >= attempts:
            self.get_logger().warning(
                f'{label}: the start pose is still inside an inscribed band after '
                f'{used} shoves ({where}); asking for a plan anyway.')
            return
        self._start_band_escapes = used + 1
        best_axis, best_room = self._band_escape_axis()
        if best_room is None or best_room < 0.35:
            self.get_logger().warning(
                f'{label}: the start pose is inside an inscribed band ({where}), '
                'but the scan allows no 0.35 m retreat '
                f'(best {best_room if best_room is not None else float("nan"):.2f} m); '
                'asking for a plan anyway.')
            return
        distance, floor = self._band_escape_shove(best_room)
        self.get_logger().warning(
            f'{label}: the start pose is inside an inscribed band ({where}); '
            f'retreating {distance:.2f} m on lidar clearance ({best_room:.2f} m) '
            'before asking for a plan.')
        self._back_off(
            f'{label}: leaving the band at the start pose', distance, 0.08,
            best_axis, leave_band=True, clearance_floor=floor)

    def _wedged_retreat(self, label: str) -> bool:
        """One lidar-verified retreat for a base the costgrid calls clear.

        `_ensure_start_pose_outside_band` only runs when a leg starts, and it
        only moves the base when the published grid agrees the pose is bad.
        Measured 2026-11-25 (runs/2026-11-25_watch_rule3, loaded leg to
        warehouse B): the leg started from a clear pose, the base drove 1 m
        into the inflated band beside the west wall, and from there every
        GridBased request aborted for 18 s while the behavior server answered
        "Collision Ahead" to every backup and spin - so the leg died as
        status=6, the executor reported "no verified recovery", and the round
        ended with 3/5 delivered and a cube attached. The local costmap had
        lethal cells 0.02 m from the base centre at that moment.
        Clearance from the scan does not depend on that grid: retreat against
        what the lidar actually sees and let the leg plan again from there.
        """
        if not bool(self.get_parameter('start_band_escape_enabled').value):
            return False
        if getattr(self, '_wedged_retreat_used', False):
            return False
        best_axis, best_room = self._band_escape_axis()
        if best_room is None or best_room < 0.35:
            # A retreat the scan already refuses must not be attempted: that
            # refusal is what keeps a base in a band for the rest of the round.
            self.get_logger().warning(
                f'{label}: the leg gave up with no verified recovery, but the '
                'scan allows no 0.35 m retreat '
                f'(best {best_room if best_room is not None else float("nan"):.2f} m); '
                'not shoving the base.')
            return False
        self._wedged_retreat_used = True
        distance, floor = self._band_escape_shove(best_room)
        self.get_logger().warning(
            f'{label}: the leg gave up with no verified recovery; retreating '
            f'{distance:.2f} m on lidar clearance ({best_room:.2f} m) before one '
            'last attempt.')
        return self._back_off(
            f'{label}: wedged start pose', distance, 0.08, best_axis,
            leave_band=True, clearance_floor=floor)

    def _band_escape_shove(self, room: float) -> tuple[float, float]:
        """Distance and scan floor for one band-clearing shove.

        The shove is capped at ``start_band_escape_retreat_m`` and never eats
        more than 0.25 m of the room the scan reported, so a base with 0.50 m
        on its free side still gets a 0.25 m retreat instead of a refusal.
        The floor keeps 0.22 m of air: the lidar sits at the front of the base,
        so part of every range is the base itself.
        """
        limit = max(0.08, float(
            self.get_parameter('start_band_escape_retreat_m').value))
        distance = max(0.08, min(limit, float(room) - 0.25))
        return distance, max(0.30, distance + 0.22)

    def _wedged_retry(self, label, values, visual_handoff_cube,
                      completion_radius_m, chain) -> bool:
        """Retreat on lidar clearance, then run the leg once more.

        Bounded twice: `_wedged_retreat` spends its one retreat per leg, and
        the re-entry is flagged so the second pass cannot retreat again.
        """
        if getattr(self, '_wedged_retry_active', False):
            return False
        if getattr(self, '_cancel_requested', False):
            return False
        if not self._wedged_retreat(label):
            return False
        self._wedged_retry_active = True
        try:
            return self._navigate_to(label, values, visual_handoff_cube,
                                     completion_radius_m, chain)
        finally:
            self._wedged_retry_active = False

    def _navigate_once(self, label, *a, **k):
        previous = getattr(self, '_nav_leg_deadline', None)
        maximum = k.get('max_duration_sec')
        if maximum is not None and float(maximum) > 0.0:
            limit = time.monotonic() + float(maximum)
            self._nav_leg_deadline = limit if previous is None else min(limit, previous)
        use_relaxed_geometry = getattr(self, '_relaxed_global', False)
        try:
            if self._navigation_motion_fault() or self._nav_budget_remaining() <= 0.0:
                return False
            # A lethal start pose aborts every plan, so clear the band first.
            # (getattr: the unit-test stubs bind only the methods they exercise.)
            start_band_check = getattr(self, '_ensure_start_pose_outside_band', None)
            if start_band_check is not None:
                start_band_check(label)
            with self._op(f'nav:{label}'):
                return self._navigate_once__timed(label, *a, **k)
        finally:
            self._nav_leg_deadline = previous
            if use_relaxed_geometry:
                self._restore_nominal_geometry(label)

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

        The opt-in outer shoulder adds one intermediate cap without delaying
        the full pre-brake. Once engaged, each cap releases slightly farther
        out than it engages to avoid repeated messages at either boundary.
        """
        if not self._slow_zones:
            self._slow_zone_held_limit = 0.0
            return 0.0
        lookahead = float(self.get_parameter('slow_zone_brake_ahead_m').value)
        width = self.get_parameter('slow_zone_approach_width_m').value or 0.0
        cruise = (self.get_parameter('slow_zone_approach_cruise_mps').value or 0.0
                  if 0.0 < width <= lookahead < math.inf else 0.0)
        staged = 0.0 < cruise <= 1.5
        # Never use _active_speed_limit here: that may be a yield cap or hold.
        previous = getattr(self, '_slow_zone_held_limit', 0.0) if staged else 0.0
        position = self._cached_base_position()
        if position is None or (staged and not all(map(math.isfinite, position))):
            self._slow_zone_held_limit = previous
            return previous

        release_margin = min(0.10, width / 4.0) if staged else 0.0
        limit = 0.0
        for x, y, radius, speed in self._slow_zones:
            distance = math.hypot(position[0] - x, position[1] - y) - radius
            full_reach = lookahead + (release_margin if previous == speed else 0.0)
            if distance <= full_reach:
                candidate = speed
            elif staged and speed < cruise:
                shoulder = (cruise + speed) / 2.0
                shoulder_reach = lookahead + width
                if previous == shoulder:
                    shoulder_reach += release_margin
                if not distance <= shoulder_reach:  # also excludes NaN
                    continue
                candidate = shoulder
            else:
                continue
            limit = candidate if limit == 0.0 else min(limit, candidate)
        self._slow_zone_held_limit = limit if staged else 0.0
        return limit

    def _apply_slow_zones(self) -> None:
        """Publish the tight-passage cap before or inside a passage.

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

    def _mover_cloud_callback(self, msg) -> None:
        """Keep the patrol footprints in base_link for a long-range yield look.

        Only points inside the tracking window are kept, so the per-tick scan in
        `_mover_cloud_box` stays cheap; the cloud arrives at 10 Hz at most.
        """
        try:
            if msg.point_step not in (12, 16, 20, 24, 32):
                return
            import struct as _struct
            points = []
            limit = float(self.get_parameter(
                'yield_mover_cloud_range_m').value) + 0.5
            for offset in range(0, len(msg.data), msg.point_step):
                x, y = _struct.unpack_from('<ff', msg.data, offset)
                if 0.0 < x <= limit and abs(y) <= 1.5:
                    points.append((x, y))
        except Exception:                                  # noqa: BLE001
            return
        self._mover_cloud = points
        self._mover_cloud_stamp = time.monotonic()

    def _mover_cloud_box(self):
        """(x, y) of the nearest box in the swept band, or None.

        Only used when the laser tracker has no confirmed mover: a box already
        tracked is better evidence than a kinematic footprint.
        """
        if not bool(self.get_parameter('yield_mover_cloud_stage').value):
            return None
        points = self._mover_cloud
        if not points:
            return None
        timeout = float(self.get_parameter(
            'yield_mover_cloud_timeout_sec').value)
        if time.monotonic() - self._mover_cloud_stamp > timeout:
            return None
        return obstacle_yield.cloud_box_ahead(
            points,
            band_m=float(self.get_parameter(
                'yield_mover_cloud_band_m').value),
            max_range_m=float(self.get_parameter(
                'yield_mover_cloud_range_m').value),
        )

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
        if getattr(self, '_yield_trace_writer', None) is not None:
            self._yield_trace_scan_age_domain = None
        scan = self._scan
        if scan is None:
            return None, []
        if scan.range_max < 0.5 or scan.angle_increment == 0.0:
            return None, []
        now = self._clock_seconds()
        stamp = float(scan.header.stamp.sec) + float(scan.header.stamp.nanosec) * 1e-9
        # An unstamped sweep must not be treated as a fresh measurement.
        age = max(0.0, now - stamp) if stamp > 0.0 else math.inf
        if getattr(self, '_yield_trace_writer', None) is not None:
            self._yield_trace_scan_age_domain = 'node_clock'
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
            if getattr(self, '_yield_trace_writer', None) is not None:
                self._yield_trace_scan_age_domain = 'monotonic_receipt_fallback'
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

    def _global_costmap_callback(self, msg: OccupancyGrid) -> None:
        self._global_costmap = msg
        self._global_costmap_received_at = time.monotonic()

    def _local_costmap_callback(self, msg: OccupancyGrid) -> None:
        self._local_costmap = msg
        self._local_costmap_received_at = time.monotonic()

    @staticmethod
    def _grid_stamp(msg) -> int | None:
        if msg is None:
            return None
        stamp = msg.header.stamp
        ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        return ns if ns > 0 else None

    def _recovery_stamp_fresh(self, msg, max_age: float) -> bool:
        stamp = self._grid_stamp(msg)
        if stamp is None:
            return False
        age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
        # A negative age means the sample is stamped slightly in the future,
        # which for a monotonically increasing odom/TF stream can only be the
        # newest sample plus clock skew -- it is not stale data.  The 50 ms
        # window this replaces rejected exactly those samples: the 2026-11-25
        # rule-3 runs logged "guarded retreat measured 0.00 m; stop: stale
        # odometry" three times in a row on ages of -0.1 s to -0.28 s, so the
        # robot never left the costmap inscribed band and the leg was aborted.
        return -1.0 <= age <= max_age

    def _recovery_pose(self, frame: str):
        base = str(self.get_parameter('robot_base_link').value).lstrip('/')
        try:
            tf = self._recovery_tf.lookup_transform(frame, base, Time())
        except TransformException:
            return None
        if not self._recovery_stamp_fresh(tf, 0.35):
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        if not all(math.isfinite(value) for value in (tr.x, tr.y, q.x, q.y, q.z, q.w)):
            return None
        if not 0.99 <= q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w <= 1.01:
            return None
        yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                         1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        return (tr.x, tr.y, yaw)

    def _fresh_recovery_grid(self, side: str):
        msg = getattr(self, f'_{side}_costmap', None)
        received = getattr(self, f'_{side}_costmap_received_at', None)
        sim_age, wall_age = (1.6, 8.0) if side == 'global' else (0.85, 4.0)
        if (received is None or time.monotonic() - received > wall_age
                or not self._recovery_stamp_fresh(msg, sim_age)):
            return None
        info = msg.info
        q, origin = info.origin.orientation, info.origin.position
        if (not msg.header.frame_id or not 0.0 < info.resolution <= 0.25
                or info.width <= 0 or info.height <= 0
                or len(msg.data) != info.width * info.height
                or not all(math.isfinite(v) for v in (origin.x, origin.y, q.x, q.y, q.z, q.w))
                or max(abs(q.x), abs(q.y), abs(q.z), abs(abs(q.w) - 1.0)) > 1e-4):
            return None
        return msg

    def _band_escape_cell_free(self, direction: float, distance: float) -> bool:
        """Is the cell ``distance`` behind the base free of *static lethal* cost?

        Only a band escape calls this. Its footprint is inside the 99 ring that
        the swept check objects to, so the swept check can never approve the
        retreat that leaves the ring (measured 2026-11-23, ..._bandfix2: 167
        collision refusals in 18 s, then the retreat stopped at -0.00 m with
        "scan or swept costmap clearance not verified"). The lidar already
        proved nothing solid is within 0.6 m of the travel direction, so the
        only remaining veto is 100 - the wall itself, including the wall behind
        that the 0.5 m lidar blind spot cannot see.
        """
        pose = self._recovery_pose('odom')
        if pose is None:
            return False
        x = pose[0] + direction * distance * math.cos(pose[2])
        y = pose[1] + direction * distance * math.sin(pose[2])
        for side in ('global', 'local'):
            grid = self._fresh_recovery_grid(side)
            if grid is None:
                continue
            value = self._recovery_cell(grid, x, y)
            if value is not None and value >= 100:
                return False
        return True

    @staticmethod
    def _recovery_cell(grid, x: float, y: float) -> int | None:
        info = grid.info
        i = math.floor((x - info.origin.position.x) / info.resolution)
        j = math.floor((y - info.origin.position.y) / info.resolution)
        if 0 <= i < info.width and 0 <= j < info.height:
            value = int(grid.data[j * info.width + i])
            return value if 0 <= value <= 100 else None
        return None

    def _global_recovery_cost(self) -> int | None:
        grid = self._fresh_recovery_grid('global')
        pose = self._recovery_pose(grid.header.frame_id) if grid is not None else None
        return self._recovery_cell(grid, *pose[:2]) if pose is not None else None

    def _recovery_envelope_radius(self) -> float:
        # Recovery floor only: measured Xacro projection of the folded arm,
        # camera and attached cube is up to ~0.422 m; reserve an extra 8 mm.
        # This does not change normal planning or controller geometry.
        nominal = getattr(self, '_nominal_geometry', None)
        return max(0.43, nominal[0]) if nominal is not None else math.inf

    @staticmethod
    def _grid_sweep_clear(grid, pose, radius: float, travel: float) -> bool:
        """Check a straight swept disk against lethal cells and unknown space."""
        if not math.isfinite(radius) or radius <= 0.0:
            return False
        info = grid.info
        r = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        x, y, yaw = pose
        dx, dy = travel * math.cos(yaw), travel * math.sin(yaw)
        padded = radius + r / math.sqrt(2.0)
        lo_x = math.floor((min(x, x + dx) - padded - ox) / r)
        hi_x = math.floor((max(x, x + dx) + padded - ox) / r)
        lo_y = math.floor((min(y, y + dy) - padded - oy) / r)
        hi_y = math.floor((max(y, y + dy) + padded - oy) / r)
        if lo_x < 0 or lo_y < 0 or hi_x >= info.width or hi_y >= info.height:
            return False
        span_sq = dx * dx + dy * dy
        for j in range(lo_y, hi_y + 1):
            ry = oy + (j + 0.5) * r - y
            for i in range(lo_x, hi_x + 1):
                cost = grid.data[j * info.width + i]
                if 0 <= cost < 100:  # 99 is inscribed, not a physical obstacle.
                    continue
                rx = ox + (i + 0.5) * r - x
                t = min(1.0, max(0.0, (rx * dx + ry * dy) / span_sq)) if span_sq else 0.0
                if (rx - t * dx)**2 + (ry - t * dy)**2 <= padded**2:
                    return False
        return True

    def _local_recovery_pose_clear(self) -> bool:
        grid = self._fresh_recovery_grid('local')
        pose = self._recovery_pose(grid.header.frame_id) if grid is not None else None
        return (pose is not None and
                (cost := self._recovery_cell(grid, *pose[:2])) is not None and cost < 99 and
                self._grid_sweep_clear(grid, pose, self._recovery_envelope_radius(), 0.0))

    def _recovery_corridor_clear(self, direction: float, lookahead: float) -> bool:
        # Local adds sensed/moving obstacles; global also verifies the static
        # geometry even if the local layer was recently cleared by Nav2.
        for side in ('local', 'global'):
            grid = self._fresh_recovery_grid(side)
            pose = self._recovery_pose(grid.header.frame_id) if grid is not None else None
            if pose is None or not self._grid_sweep_clear(
                    grid, pose, self._recovery_envelope_radius(), direction * lookahead):
                return False
        return True

    def _global_cell_cost(self, x: float, y: float) -> int | None:
        """Published global costmap value at (x, y), or None when unjudgeable.

        None means "no map yet" or "outside the map": callers must treat that as
        no opinion, not as free space.
        """
        msg = self._global_costmap
        if msg is None:
            return None
        info = msg.info
        resolution = float(info.resolution)
        if resolution <= 0.0:
            return None
        mx = int(math.floor((x - (info.origin.position.x)) / resolution))
        my = int(math.floor((y - (info.origin.position.y)) / resolution))
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        index = my * info.width + mx
        data = msg.data
        if index < 0 or index >= len(data):
            return None
        return int(data[index])

    def _costmap_clearance_at(self, x: float, y: float) -> bool | None:
        """Is (x, y) outside the global costmap's lethal/inscribed band?

        Returns None when there is nothing to judge against (no map yet, or the
        point is outside it): callers must treat that as "no opinion", not as
        free space. Published values follow nav2_costmap_2d: cell 99 means the
        cell is inside the inscribed radius (cost 253), 100 is lethal (254), and
        -1 is unknown - a planner cannot seed a goal in unknown space either.
        """
        cost = self._global_cell_cost(x, y)
        if cost is None:
            return None
        return -1 < cost < 99

    def _yield_arc_path_clear(self, target, base_x, base_y):
        """Is the whole base->arc path outside the global costmap band?

        Returns ``(ok, detail)``. A missing costmap is "no opinion", matching
        :meth:`_costmap_clearance_at`, not a rejection: the caller has already
        run the lidar whole-segment gate. A cell that is definitely inside the
        inscribed or lethal band rejects the arc, because the loaded envelope
        cannot drive over it.
        """
        if not bool(self.get_parameter('yield_arc_path_check').value):
            return True, None
        dx = float(target[0]) - base_x
        dy = float(target[1]) - base_y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return False, 'arc has zero length'
        start = min(
            length,
            max(0.0, float(self.get_parameter(
                'yield_arc_path_start_m').value)),
        )
        step = max(
            0.02,
            float(self.get_parameter('yield_arc_path_step_m').value),
        )
        distance = start
        while True:
            fraction = min(1.0, distance / length)
            x = base_x + dx * fraction
            y = base_y + dy * fraction
            if self._costmap_clearance_at(x, y) is False:
                return False, (
                    f'arc path crosses the global costmap band at '
                    f'({x:.2f}, {y:.2f}), {distance:.2f} m from the base'
                )
            if fraction >= 1.0:
                break
            distance = min(length, distance + step)
        return True, None

    def _yield_arc_watch(self, obs, base_x: float, base_y: float, yaw: float):
        """Re-check the arc being driven against the current mover track.

        The arc goal was chosen in the base frame of the tick that decided it;
        what matters now is the part still ahead, so the target is re-expressed
        in the current base frame before it is planned.
        """
        target = self._yield_arc_target
        if target is None:
            return obstacle_yield.ArcWatch(True)
        dx = target[0] - base_x
        dy = target[1] - base_y
        ax = math.cos(yaw) * dx + math.sin(yaw) * dy
        ay = -math.sin(yaw) * dx + math.cos(yaw) * dy
        if obs is None:
            # A fresh scan with nothing inside the tracking range is not the
            # same evidence as a lost scan: there is no mover close enough to
            # occupy the arc, and treating the box leaving the field of view as
            # danger would abort every sidestep the moment it starts working.
            return obstacle_yield.ArcWatch(True)
        obs = replace(obs, arc_active=True,
                      arc_target_x=ax, arc_target_y=ay)
        return obstacle_yield.watch_arc(obs, self._yield_config)

    def _yield_step(self, distance_to_goal_m: float | None, watch: bool = False):
        """One policy tick. Returns (decision, latched, arc goal in map frame).

        With ``watch`` the tick only re-checks the arc that is being driven and
        returns an ``obstacle_yield.ArcWatch`` instead of a decision. The
        observation is built by the same code in both modes: while the arc runs
        the ordinary policy is suppressed, and this is the only thing still
        looking at the mover.
        """
        motion_vx, motion_vy, wz, base_x, base_y = self._yield_base_motion()
        pose = self._base_pose()
        yaw = pose[2] if pose is not None else math.nan
        age, points = self._yield_scan_points(yaw)
        if getattr(self, '_yield_trace_writer', None) is not None:
            try:
                self._yield_trace_capture(age, points, None, None, None, None,
                                          None, distance_to_goal_m)
            except Exception:
                pass
        if (points is None or age is None or not self._yield_config.enabled
                or age > self._yield_config.track_timeout_sec
                or not all(math.isfinite(v) for v in (base_x, base_y, yaw))):
            if watch:
                # The arc cannot be confirmed without a scan, and "no new
                # evidence" is not "clear": stop and decide again.
                return obstacle_yield.ArcWatch(
                    False, 0.0, 'lost the scan while driving the yield arc')
            # Losing a scan during a hold must NOT publish the 0.0 'release'
            # sentinel immediately: that would restore full cruise on no new
            # evidence. Once the budget is spent, an active hold fails closed.
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
                    and (elapsed < self._yield_config.wait_budget_sec
                         or getattr(self, '_yield_pause_active', False))):
                reason = (obstacle_yield.REASON_WAIT_EXPIRED
                          if elapsed >= self._yield_config.wait_budget_sec
                          else obstacle_yield.REASON_WAIT)
                return (obstacle_yield.YieldDecision(
                    obstacle_yield.ACTION_STOP, reason,
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
        carrying = self._carrying_now()
        self._yield_tracker.update(
            centroids, now, (base_x, base_y, yaw, motion_vx, motion_vy, wz),
            carrying=carrying)
        track = self._yield_tracker.nearest(
            now, (base_x, base_y, yaw, motion_vx, motion_vy, wz),
            min_world_speed=self._yield_config.min_mover_speed_mps,
            carrying=carrying)
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
        held_since = getattr(self, '_yield_continuous_hold_since', None)
        cloud_box = self._mover_cloud_box() if track is None else None
        if track is None and cloud_box is not None:
            # Seen by the footprints, not by the laser: same policy, weaker
            # evidence. Velocity is unknown, so the observation carries the
            # patrol boxes' nominal 0.20 m/s instead of pretending it is static
            # (a static reading makes the policy treat the box as leaving).
            obs = obstacle_yield.YieldObservation(
                box_x=cloud_box[0], box_y=cloud_box[1],
                box_vx=None, box_vy=None,
                vx=motion_vx, wz=wz,
                left_room_m=left_room, right_room_m=right_room,
                distance_to_goal_m=distance_to_goal_m,
                scan_age_sec=age,
                box_world_speed_mps=0.20,
                latched=self._yield_latched,
                lane_distance_m=obstacle_yield.distance_to_lane(
                    base_x, base_y, self._patrol_segments),
            )
        elif track is None:
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
        if watch:
            return self._yield_arc_watch(obs, base_x, base_y, yaw)
        if (self._yield_config.enabled
                and getattr(self._yield_config, 'approach_trial_enabled', False)
                and self._yield_latched and obs is None
                and wait_elapsed < self._yield_config.wait_budget_sec):
            # A fresh scan is not necessarily a fresh mover observation: the
            # tracker can lose a box behind a corner or in its speed gate.
            # Don't start the clear timer on that loss. Keep the same finite
            # episode budget; this is not an independent collision stop.
            decision = obstacle_yield.YieldDecision(
                obstacle_yield.ACTION_STOP, obstacle_yield.REASON_WAIT,
                wait_elapsed_sec=wait_elapsed, detail='awaiting mover observation',
            )
        else:
            decision = obstacle_yield.decide(
                obs, self._yield_config,
                wait_elapsed_sec=wait_elapsed,
                arc_attempts=self._yield_arc_attempts,
                continuous_hold_sec=0.0 if held_since is None else now - held_since,
            )
        if getattr(self, '_yield_trace_writer', None) is not None:
            try:
                self._yield_trace_capture(age, points, clusters, boxes, track, obs,
                                          carrying, distance_to_goal_m, decision)
            except Exception:
                pass
        self._yield_diag(age, points, clusters, boxes, track, decision, now,
                         cloud_box=cloud_box)
        latched = decision.action == obstacle_yield.ACTION_STOP
        arc_goal = None
        if decision.action == obstacle_yield.ACTION_ARC and decision.arc_x is not None:
            static_returns = [
                (x, y) for x, y, _ in points
                if not any(math.hypot(x - ex, y - ey) <= 0.22
                           for ex, ey in excluded)
            ]
            target = (
                base_x + math.cos(yaw) * decision.arc_x
                - math.sin(yaw) * decision.arc_y,
                base_y + math.sin(yaw) * decision.arc_x
                + math.cos(yaw) * decision.arc_y,
            )
            blocked = None
            arc_half_width = self._yield_config.base_half_width_m
            if carrying:
                # Keep the scan veto sized for the loaded footprint, even if
                # a navigation retry temporarily relaxes the costmap.
                arc_half_width = max(arc_half_width, float(
                    self.get_parameter('loaded_robot_radius_m').value))
            if not obstacle_yield.arc_scan_clear(
                static_returns, decision.arc_x, decision.arc_y,
                radius_m=arc_half_width + 0.05,
            ):
                blocked = 'arc approach blocked or unobserved by lidar'
            elif bool(self.get_parameter('yield_arc_costmap_check').value):
                clear = self._costmap_clearance_at(*target)
                if clear is False or (
                    clear is None and decision.reason == obstacle_yield.REASON_BLOCKED_DETOUR
                ):
                    blocked = ('arc target sits inside the global costmap\'s '
                               'inscribed band or unknown space')
            if blocked is None:
                # The apex can be outside the band while the floor between the
                # base and that cell is not: the carried envelope then drives
                # into the wall before reaching it (2026-11-21, 0.94 m in
                # 14.8 s). Vet every cell of the sidestep, or do not take it.
                path_ok, path_detail = self._yield_arc_path_clear(
                    target, base_x, base_y)
                if not path_ok:
                    blocked = path_detail
            if blocked is None:
                arc_goal = target
            else:
                # Do NOT release the hold over a rejected arc. Releasing hands
                # control back to Nav2 with the box still in the corridor, and
                # the controller then refuses the path every tick (the 18 s
                # stall of 2026-10-14). Keep waiting and spend this attempt, so
                # the per-leg ceiling still bounds how long the base waits.
                self._yield_arc_attempts += 1
                decision = obstacle_yield.YieldDecision(
                    obstacle_yield.ACTION_STOP,
                    obstacle_yield.REASON_WAIT,
                    detail=blocked,
                )
                latched = True
        if self._yield_config.enabled and decision.action != obstacle_yield.ACTION_GO:
            self._yield_log_decision(decision, obs)
        return decision, latched, arc_goal

    def _yield_trace_capture(self, age, points, clusters, boxes, track, obs,
                             carrying, distance_to_goal_m, decision=None):
        """Snapshot only policy inputs, not raw scans, task data or histories."""
        self._yield_trace_sample = None
        try:
            track_keys = ('born', 't', 'x', 'y', 'vx', 'vy', 'base_x',
                          'base_y', 'rel_vx', 'rel_vy', 'world_speed', 'age', 'life')
            self._yield_trace_sample = {
                'schema': 'yield_tick_v1',
                'clock': {'tick_domain': 'python_monotonic_seconds',
                          'sample_monotonic_sec': time.monotonic(),
                          'scan_age_domain': getattr(
                              self, '_yield_trace_scan_age_domain', None),
                          'node_clock_type': str(self.get_clock().clock_type)},
                'scan': {'age_sec': age,
                         'point_count': (None if points is None or age is None
                                         else len(points)),
                         'cluster_count': None if clusters is None else len(clusters),
                         'box_count': None if boxes is None else len(boxes)},
                'observation': obs,
                'observation_missing': obs is None,
                'track': None if track is None else {
                    'stable_id': None,
                    'identity_note': 'born is monotonic track birth, not a unique ID',
                    **{key: track.get(key) for key in track_keys}},
                'carrying': carrying, 'distance_to_goal_m': distance_to_goal_m,
                'config_thresholds': self._yield_config,
                'caller_thresholds': {
                    'update_period_sec': self.get_parameter(
                        'yield_update_period_sec').value,
                    'arc_costmap_check': self.get_parameter(
                        'yield_arc_costmap_check').value,
                    'clock_mismatch_sec': CLOCK_MISMATCH_SEC,
                    'arc_mover_exclusion_radius_m': 0.22,
                    'arc_scan_radius_extra_m': 0.05},
                'policy_decision': decision,
                'before': {'latched': self._yield_latched,
                           'wait_since': self._yield_wait_since,
                           'clear_since': self._yield_clear_since,
                           'episode_since': self._yield_episode_since,
                           'free_since': self._yield_free_since,
                           'episode_anchor': self._yield_episode_anchor,
                           'release_since': self._yield_release_since,
                           'release_from': self._yield_release_from,
                           'arc_attempts': self._yield_arc_attempts,
                           'arc_used': self._yield_arc_used},
            }
        except Exception:
            # Diagnostics are observational, including when ROS is shutting down.
            pass

    def _yield_trace_emit(self, now, decision, latched, arc_goal):
        try:
            sample = getattr(self, '_yield_trace_sample', None)
            if sample is None:
                return
            sample['clock']['tick_monotonic_sec'] = now
            sample['final_decision'] = decision
            sample['after'] = {'latched': latched,
                               'wait_since': self._yield_wait_since,
                               'clear_since': self._yield_clear_since,
                               'episode_since': self._yield_episode_since,
                               'free_since': self._yield_free_since,
                               'episode_anchor': self._yield_episode_anchor,
                               'release_since': self._yield_release_since,
                               'release_from': self._yield_release_from,
                               'arc_attempts': self._yield_arc_attempts,
                               'arc_used': self._yield_arc_used,
                               'candidate_arc_goal': arc_goal,
                               'queued_arc_goal': self._yield_arc_goal}
            self._yield_trace_writer.submit(sample)
        except Exception:
            pass

    def _yield_diag(self, age, points, clusters, boxes, track, decision,
                    now, cloud_box=None) -> None:
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
        cloud = ('none' if cloud_box is None
                 else f'({cloud_box[0]:+.2f},{cloud_box[1]:+.2f})')
        self.get_logger().warning(
            f'YIELD_DIAG {detail} cloud={cloud} decision={decision.action}'
            f'/{decision.reason}')

    def _yield_log_decision(self, decision, obs) -> None:
        """Rate-limited yield logging, in a shape the round record can grep."""
        now = time.monotonic()
        if now - self._yield_last_log < 1.0:
            return
        self._yield_last_log = now
        def fmt(value):
            # A footprint-only observation has no velocity; logging must not
            # raise inside the control loop (round band047r1 did, one line
            # before the same None reached the conflict arithmetic).
            return 'n/a' if value is None else f'{value:+.2f}'

        box = 'n/a' if obs is None else (
            f'({obs.box_x:+.2f},{obs.box_y:+.2f}) v=({fmt(obs.box_vx)},'
            f'{fmt(obs.box_vy)}) world={obs.box_world_speed_mps}'
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
        Trace records cover policy ticks only: suppressed and period-limited calls
        do not run a decision and do not produce a trace record.
        """
        if self._yield_suppress:
            # The arc leg disables the ordinary policy, deliberately: a second
            # lateral goal must not be created while the first one is driven.
            # The mover, however, does not stop changing, so the arc is
            # re-checked on the policy's period and its verdict caps the arc
            # speed or calls it off. Suppressing the danger check together with
            # the decision was the bug this replaced: the base drove the sidestep
            # blind for up to the arc timeout (20.10 s in the 2026-11-21 round).
            if self._yield_arc_target is not None:
                now = time.monotonic()
                period = float(self.get_parameter('yield_update_period_sec').value)
                if now - self._yield_watch_last >= period:
                    self._yield_watch_last = now
                    watch = self._yield_step(distance_to_goal_m, watch=True)
                    self._yield_arc_watch_speed = (
                        watch.speed_mps if watch.ok else 0.0)
                    if not watch.ok:
                        self._yield_arc_abort = watch.detail
                    elif watch.detail:
                        self.get_logger().info(
                            f'Yield arc watch: {watch.detail}.',
                            throttle_duration_sec=2.0)
            limit = self._slow_zone_limit()
            reason = 'Slow zone' if limit > 0.0 else ''
            watch_speed = self._yield_arc_watch_speed
            if watch_speed is not None:
                if watch_speed <= 0.0:
                    limit = YIELD_STOP_SPEED_LIMIT
                    reason = 'Yield arc hold'
                elif limit == 0.0 or watch_speed < limit:
                    limit = watch_speed
                    reason = 'Yield arc watch'
            self._publish_speed_limit(limit, reason)
            return
        now = time.monotonic()
        period = float(self.get_parameter('yield_update_period_sec').value)
        if now - self._yield_last_update < period:
            return
        self._yield_last_update = now
        decision, latched, arc_goal = self._yield_step(distance_to_goal_m)
        # GO and CAP both move the stopped base. Require both releases to be
        # supported by a continuous window of policy ticks during a real hold.
        paused = getattr(self, '_yield_pause_active', False)
        expired = (decision.action == obstacle_yield.ACTION_GO
                   and decision.reason == obstacle_yield.REASON_WAIT_EXPIRED)
        unusable_arc = (
            decision.action == obstacle_yield.ACTION_ARC
            and self._yield_arc_goal is None
            and (arc_goal is None or self._yield_arc_used
                 or self._yield_arc_attempts >= self._yield_config.max_arc_attempts)
        )
        if paused and (expired or unusable_arc):
            decision = obstacle_yield.YieldDecision(
                obstacle_yield.ACTION_STOP, obstacle_yield.REASON_WAIT_EXPIRED,
                detail='no verified release or usable side-step',
            )
            latched = True
        if (self._yield_latched
                and (decision.action == obstacle_yield.ACTION_GO
                     or (paused and decision.action == obstacle_yield.ACTION_CAP))
                and decision.reason != obstacle_yield.REASON_WAIT_EXPIRED
                and self._yield_wait_since is not None
                and (paused or now - self._yield_wait_since < self._yield_config.wait_budget_sec)):
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
        # A release has to survive the base's own acceleration (see
        # YieldConfig.go_commit_sec): keep the latch - never the hold - until the
        # base has actually moved, so the next tick judges the corridor at the
        # speed the base is about to drive instead of at "stopped, so 0.05 m/s".
        if (self._yield_latched and not latched
                and decision.action == obstacle_yield.ACTION_GO):
            if self._yield_release_from is None:
                self._yield_release_from = self._cached_base_position()
                self._yield_release_since = now
            position = self._cached_base_position()
            moved = 0.0
            if position is not None and self._yield_release_from is not None:
                moved = math.hypot(position[0] - self._yield_release_from[0],
                                   position[1] - self._yield_release_from[1])
            commit_m = float(getattr(self._yield_config, 'go_commit_m', 0.0))
            commit_sec = float(getattr(self._yield_config, 'go_commit_sec', 0.0))
            if (moved < commit_m
                    and now - (self._yield_release_since or now) < commit_sec):
                latched = True
            else:
                self._yield_release_from = None
                self._yield_release_since = None
        else:
            self._yield_release_from = None
            self._yield_release_since = None
        if getattr(self._yield_config, 'approach_trial_enabled', False):
            if decision.holds:
                self._yield_resume_started = None
                if not self._yield_decision.holds:
                    self._yield_continuous_hold_since = now
            else:
                self._yield_continuous_hold_since = None
                if self._yield_decision.holds and decision.action != obstacle_yield.ACTION_ARC:
                    self._yield_resume_started = now
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
        resumed_at = getattr(self, '_yield_resume_started', None)
        if resumed_at is not None and not decision.holds:
            resume_limit = obstacle_yield.resume_speed_limit(now - resumed_at)
            if resume_limit is None:
                self._yield_resume_started = None
            elif limit == 0.0 or resume_limit < limit:
                limit, reason = resume_limit, 'Yield staged resume'
        self._publish_speed_limit(limit, reason if limit > 0.0 else '')
        if getattr(self, '_yield_trace_writer', None) is not None:
            try:
                self._yield_trace_emit(now, decision, latched, arc_goal)
            except Exception:
                pass

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
        self._yield_release_from = None
        self._yield_release_since = None
        self._yield_resume_started = None
        self._yield_continuous_hold_since = None
        self._yield_wait_since = None
        self._yield_clear_since = None
        self._yield_arc_attempts = 0
        self._yield_arc_goal = None
        self._yield_arc_used = False
        self._yield_arc_target = None
        self._yield_arc_abort = None
        self._yield_arc_watch_speed = None
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
        if fault := FixedJointPickPlace._navigation_motion_fault(self):
            raise RuntimeError(f'{fault}: new navigation goal blocked')
        duration = float(self.get_parameter('navigation_timeout_sec').value)
        if not math.isfinite(duration) or duration <= 0.0:
            self.get_logger().error(f'{label}: invalid navigation_timeout_sec.')
            return False
        previous_deadline = getattr(self, '_nav_leg_deadline', None)
        deadline = time.monotonic() + duration
        self._nav_leg_deadline = deadline if previous_deadline is None else min(deadline, previous_deadline)

        def remaining(limit=math.inf):
            return FixedJointPickPlace._nav_budget_remaining(self, limit)

        def interrupted():
            if fault := FixedJointPickPlace._navigation_motion_fault(self):
                raise RuntimeError(f'{fault}: {label} recovery blocked')
            if remaining() <= 0.0:
                self._last_attempt_failure = 'deadline'
                self.get_logger().warning(f'{label}: shared navigation budget exhausted.')
                return True
            return getattr(self, '_cancel_requested', False)

        attempts = max(1, int(self.get_parameter('navigation_retry_count').value) + 1)
        # A canceled goal followed by one yield waypoint must still get a
        # chance to retry the original goal, even when retries are disabled.
        if self._yield_config.enabled:
            attempts += 1
        # Progress along the detour chain belongs to the LEG, not to one attempt:
        # a retry after a hold must not send the base back to a waypoint it has
        # already reached (measured 2026-10-14: seven holds in one leg, each
        # resume re-sending the full chain).
        self._chain_index = 0
        self._relax_used = False
        if not getattr(self, '_wedged_retry_active', False):
            # One lidar-verified retreat per leg: the retry that consumes it
            # re-enters here and must not arm a second one.
            self._wedged_retreat_used = False
            # Same cap for the pre-flight shove that clears an inscribed start
            # pose: a base parked in a band must not be pushed on every attempt.
            self._start_band_escapes = 0
            # Drop the previous leg's hold report so a leg that fails without
            # any hold cannot quote the leg before it.
            self._last_hold_report = None
        try:
            self._log_cube_positions(f'before:{label}')
            attempt = 0
            retreat_used = False
            unchanged_retry_used = False
            arc_failure_recovered = False
            no_retreat = False
            while attempt < attempts:
                if interrupted():
                    return False
                attempt += 1
                # Each attempt gets a clean failure fingerprint: the escape after
                # a failed attempt reads it to choose a direction.
                self._last_attempt_failure = None
                # A yield arc is a convenience, not a promise: the leg was
                # abandoned because a moving obstacle owned the lane. Drive the
                # lateral waypoint the policy picked, then retry the same goal
                # WITHOUT the dead-zone escape and jam break below, which would
                # shove a perfectly healthy base backwards through the obstacle
                # it just avoided. If the sidestep cannot be driven, recover to a
                # hold and re-evaluate the original route; the sidestep itself
                # must not be the terminal failure.
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
                    # The watch needs the target in map coordinates; it is
                    # re-expressed in the base frame on every tick.
                    self._yield_arc_target = arc
                    self._yield_arc_abort = None
                    self._yield_arc_watch_speed = None
                    self._yield_watch_last = 0.0
                    arc_progress_timeout = float(self.get_parameter(
                        'yield_arc_progress_timeout_sec').value)
                    arc_progress_distance = float(self.get_parameter(
                        'yield_arc_progress_m').value)
                    arc_max_duration = float(self.get_parameter(
                        'yield_arc_timeout_sec').value)
                    try:
                        self._apply_slow_zones()
                        arc_ok = self._navigate_once(
                            f'{label} yield arc',
                            (arc[0], arc[1], values[2]), None,
                            float(self.get_parameter(
                                'yield_arc_goal_radius_m').value),
                            progress_timeout_sec=arc_progress_timeout,
                            progress_distance_m=arc_progress_distance,
                            max_duration_sec=arc_max_duration,
                        )
                        if interrupted():
                            return False
                        # The detour target was already vetted by the yield
                        # policy (scan + costmap), so an arc leg that dies with
                        # the start cell inside the planner's band failed on the
                        # START, not on the detour: measured 2026-11-18, the arc
                        # goal had 1.12 m clearance and the leg still burned
                        # 17.7 s to status 6 from a pose 0.355 m off a wall. One
                        # relaxed retry recovers the detour. An arc the watch
                        # abandoned must not be retried at all.
                        if (not arc_ok and self._yield_arc_abort is None
                                and self._start_cell_inside_global_band()
                                and self._relax_global_footprint(
                                    f'{label} yield arc')):
                            arc_ok = self._navigate_once(
                                f'{label} yield arc',
                                (arc[0], arc[1], values[2]), None,
                                float(self.get_parameter(
                                    'yield_arc_goal_radius_m').value),
                                progress_timeout_sec=arc_progress_timeout,
                                progress_distance_m=arc_progress_distance,
                                max_duration_sec=arc_max_duration,
                            )
                    finally:
                        self._yield_suppress = False
                        self._yield_arc_target = None
                    if interrupted():
                        return False
                    if not arc_ok:
                        if self._yield_arc_abort is not None:
                            # The watch called the sidestep off: the mover's
                            # predicted sweep now cuts the arc. Keep current
                            # obstacle observations for the original route. This is
                            # not a planning failure, so the rest of the arc
                            # budget is kept, and the base must not back off
                            # through the mover.
                            no_retreat = True
                            self.get_logger().warning(
                                f'{label}: resuming the original route after '
                                'the sidestep was abandoned.')
                            continue
                        # Keep the fail-closed property: the ORIGINAL route is
                        # still retried with its own watchdogs and final
                        # UNSAFE_OBJECT_STATE path. The arc consumed the extra
                        # attempt the yield policy granted, so do not refund it:
                        # this must not become an unbounded clear-map retry
                        # loop. Spend each arc the policy grants only once per
                        # leg.
                        max_arc = int(getattr(
                            self._yield_config, 'max_arc_attempts', 0))
                        if max_arc > 0 and self._yield_arc_attempts < max_arc:
                            self._yield_arc_attempts = max_arc
                        if not arc_failure_recovered and attempt < attempts:
                            arc_failure_recovered = True
                            # The arc consumed the extra attempt the yield
                            # policy granted (attempts += 1). Do not refund it:
                            # the original route still has its configured retry
                            # budget, and this must not become a way to buy more
                            # clear-map retries. The base is between two lanes
                            # with the cube on it, and no rear-clearance check
                            # has verified a retreat, so the ordinary jam
                            # recovery must not shove it backwards either.
                            no_retreat = True
                            self.get_logger().warning(
                                f'{label}: yield arc did not complete; holding '
                                'and retrying the original route.'
                            )
                            time.sleep(remaining(float(self.get_parameter(
                                'yield_arc_recovery_hold_sec').value)))
                            continue
                        # No recovery budget left: the loop bound ends the leg,
                        # and the caller still owns the UNSAFE decision.
                        continue
                    # The canceled attempt that granted this arc and the arc
                    # itself are ONE recovery, and `attempts += 1` above bought
                    # exactly one extra slot for it - but the cancellation
                    # already spent that slot on the ordinary attempt it
                    # displaced. Refund the arc's: otherwise the leg ends at the
                    # waypoint it just reached, without ever re-driving the goal
                    # (measured 2026-11-22, rule 3 round B, blue_cube_5 -> B:
                    # the sidestep completed at (0.40, -7.79) and the loaded leg
                    # still failed UNSAFE because no attempt was left, killing
                    # the round at 2/5 with the cube attached).
                    attempt -= 1
                    continue
                # Test-only switch: skip the ordinary first attempt so the
                # recovery is exercised even when the entrance is healthy. It
                # must be scoped to the leg the recovery actually owns - in the
                # 2026-11-21 forced round it also skipped the first attempt of
                # `red_cube_1 visual handoff`, which has nothing to do with the
                # mouth, because applicability was not part of the condition.
                forced_entry = (
                    attempt == 1
                    and self._entry_recovery_applies(label)
                    and bool(self.get_parameter(
                        'entry_recovery_force_first').value)
                )
                if forced_entry:
                    self.get_logger().warning(
                        f'{label}: entry recovery forced for this run; the '
                        'ordinary first attempt is skipped.'
                    )
                if not forced_entry and self._navigate_once(
                    label, values, visual_handoff_cube, completion_radius_m,
                    chain,
                ):
                    if interrupted():
                        return False
                    self.get_logger().info(f'Navigation stage complete: {label}.')
                    time.sleep(remaining(float(self.get_parameter('settle_sec').value)))
                    return not interrupted()
                if interrupted():
                    return False
                if self._yield_arc_goal is not None:
                    # This leg ended deliberately for a moving-obstacle waypoint,
                    # not because the base is wedged. The next iteration consumes
                    # the arc; never back off through the mover or shrink the
                    # loaded footprint on this path.
                    continue
                if getattr(getattr(self, '_yield_decision', None), 'holds', False):
                    return False
                if not no_retreat and self._entry_recovery_applies(label):
                    if self._entry_recovery(
                        label, values, visual_handoff_cube,
                        completion_radius_m, chain,
                    ):
                        if interrupted():
                            return False
                        self.get_logger().info(
                            f'Navigation stage complete: {label} (entry '
                            'recovery).'
                        )
                        time.sleep(remaining(float(self.get_parameter('settle_sec').value)))
                        return not interrupted()
                    if self._yield_arc_goal is not None:
                        # The yield policy granted a lateral waypoint while the
                        # recovery was running. Drive it through the ordinary
                        # path instead of discarding a decision the route
                        # planner already made, then retry the dock leg.
                        self.get_logger().warning(
                            f'{label}: the yield policy granted a sidestep '
                            'during the entry recovery; driving it before the '
                            'next attempt.'
                        )
                        continue
                    # The recovery owns the decision once it starts: its bounded
                    # exit must not fall through to the shrink/clear/shove
                    # ladder it exists to replace.
                    return False
                if attempt >= attempts:
                    break
                failure = self._last_attempt_failure
                if failure == 'nav_control_not_released' or interrupted():
                    return False
                if (not no_retreat and failure not in ('wedge_refusals', 'controller_failure')
                        and self._relax_global_footprint(label)):
                    # The next goal owns this temporary margin; it restores it
                    # even if that goal fails or is cancelled.
                    continue
                if interrupted():
                    return False
                outer_route = self._pickup_fallback_route(label)
                if outer_route and not no_retreat and self._pickup_fallback_needed(outer_route, values):
                    self._pickup_fallback_used = True
                    if self._drive_pickup_fallback(label, outer_route):
                        continue
                    if interrupted():
                        return False
                    if self._yield_arc_goal is not None:
                        continue
                    if getattr(getattr(self, '_yield_decision', None), 'holds', False):
                        return False
                if not no_retreat and failure in ('wedge_refusals', 'controller_failure', 'no_motion'):
                    if retreat_used:
                        break
                    retreat_used = True
                    if self._back_off(
                        label, distance_m=float(self.get_parameter('escape_distance_m').value),
                        speed=float(self.get_parameter('escape_speed_mps').value),
                        direction=-1.0,
                    ):
                        continue
                    if self._yield_arc_goal is not None:
                        continue
                    break
                if (unchanged_retry_used or not self._local_recovery_pose_clear()
                        or (cost := self._global_recovery_cost()) is None or cost >= 99):
                    self.get_logger().warning(
                        f'{label}: no verified recovery for {failure or "unclassified failure"}.')
                    break
                # One unchanged retry may ride out a transient refusal. A
                # generic Nav2 abort is not evidence for a geometry change.
                unchanged_retry_used = True
            # Every configured attempt is spent. The grid cannot be trusted
            # here - this is exactly the state where NavFn refuses to start and
            # Nav2 refuses to back out - so ask the scan instead, free the pose
            # and give the leg one more pass before the round is lost.
            # (getattr: the unit-test stubs bind only the methods they exercise.)
            wedged_retry = getattr(self, '_wedged_retry', None)
            if (wedged_retry is not None
                    and wedged_retry(label, values, visual_handoff_cube,
                                     completion_radius_m, chain)):
                return True
            return False
        finally:
            self._nav_leg_deadline = previous_deadline
            try:
                if getattr(self, '_relaxed_global', False):
                    self._restore_nominal_geometry(label)
            finally:
                if FixedJointPickPlace._navigation_motion_fault(self):
                    self._stop_base()
                    self._publish_speed_limit(YIELD_STOP_SPEED_LIMIT, 'navigation state unknown')
                else:
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
        # One outer route per pickup attempt, however many legs it takes; see
        # _pickup_fallback_route.
        self._pickup_fallback_used = False
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

    def _pickup_fallback_route(self, label: str) -> list[list[float]]:
        """The configured outer route for one pickup leg, or [].

        Stored as ``<cube>_pickup_fallback_via_<n>``. It is never part of the
        normal approach: it only runs after an attempt failed, and it stops
        short of the goal so the goal itself is attacked again from there with
        the last leg planned by Nav2 as usual.
        """
        if not bool(self.get_parameter('pickup_fallback_enabled').value):
            return []
        cube = ''
        for suffix in (' visual handoff', ' pre-grasp'):
            if label.endswith(suffix):
                cube = label[: -len(suffix)].strip()
                break
        if not cube:
            return []
        route = []
        for index in (1, 2, 3):
            parameter = f'{cube}_pickup_fallback_via_{index}'
            if not self.has_parameter(parameter):
                break
            raw = self.get_parameter(parameter).value
            if raw is None or len(list(raw)) != 3:
                break
            route.append(self._double_list(parameter, 3))
        return route

    def _pickup_fallback_needed(self, route, values) -> bool:
        """Whether this failure should switch to the outer route, once.

        Only what a bypass can cure qualifies: a stall, a controller refusing
        every path, or a plan that fails from a start cell the planner itself
        considers free (an obstacle in the middle of the route, which is what a
        parked mover looks like). An abandoned sidestep is excluded: there the
        base is already wedged beside the mouth and must be freed first. The
        detour must also be the shorter way out, so a failure close to the goal
        stays with the ordinary ladder instead of driving back across the map.
        """
        if self._pickup_fallback_used:
            return False
        failure = self._last_attempt_failure
        if failure == 'yield_arc_aborted':
            return False
        if failure not in ('no_progress', 'planner_no_path'):
            return False
        cost = self._global_recovery_cost()
        if cost is None or cost >= 99 or not self._local_recovery_pose_clear():
            return False
        position = self._cached_base_position()
        if position is None:
            return False
        to_route = math.hypot(
            route[0][0] - position[0], route[0][1] - position[1]
        )
        to_goal = math.hypot(values[0] - position[0], values[1] - position[1])
        return to_route < to_goal

    def _drive_pickup_fallback(self, label: str, route) -> bool:
        """Drive the outer route once and stop on its last point.

        True means the base is on the far side of the mouth and the caller may
        retry the original goal from there. False falls straight back into the
        ordinary retry ladder: this is an extra route, not a replacement for
        the retry budget.
        """
        radius = float(
            self.get_parameter('pickup_fallback_completion_radius_m').value
        )
        points = ' -> '.join(f'({x:.2f}, {y:.2f})' for x, y, _ in route)
        self.get_logger().warning(
            f'{label}: repeated failure at the narrow mouth; taking the '
            f'verified outer route via {points} once, then re-joining the goal.'
        )
        reached = self._navigate_once(
            f'{label} outer route',
            route[-1],
            completion_radius_m=radius,
            chain=route[:-1] or None,
        )
        if not reached:
            self.get_logger().warning(
                f'{label}: the outer route did not complete; returning to the '
                'ordinary retry policy.'
            )
            return False
        self._stop_base()
        return (self._nav_budget_remaining() > 0.0
                and not getattr(self, '_cancel_requested', False)
                and not FixedJointPickPlace._navigation_motion_fault(self))

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
        backed_off = False
        last_row = None
        # Whether the last frame that produced a row was also centred. A blob
        # that vanished while it was still off to one side never sat on the
        # camera axis, so the row it left behind says nothing about the standoff
        # and must not be handed to the creep (see the "left the frame" branch).
        last_centred = False
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
                        # 2026-11-17: if the blob disappeared only after its
                        # bottom row had reached the acceptance band, the cube
                        # has simply slid under the level camera (the fifth cube
                        # leaves the image at about 0.22 m). Searching cannot
                        # find it again - hand the measured row to the creep
                        # instead of burning the remaining 70 s of the timeout.
                        # Only a CENTRED frame can be handed over: the row is
                        # the standoff along the camera axis, and the creep
                        # drives straight down that axis. A blob that was still
                        # off to one side when it vanished leaves a row that
                        # describes a cube somewhere else, and creep would
                        # grasp air. Such a loss falls through to the re-search
                        # below, which is what it was doing before this branch.
                        if (
                            last_row is not None
                            and last_centred
                            and last_row >= approach_row - row_tolerance
                        ):
                            self.get_logger().warning(
                                'Visual blob left the frame at row '
                                f'{last_row:.0f} (>= '
                                f'{approach_row - row_tolerance:.0f}); the cube '
                                'is under the camera, handing over to creep.'
                            )
                            stable_frames_used = max(1, stable)
                            return self._creep_to_grasp(last_row)
                    elif now - lost_since >= research_after:
                        # Stop waiting out the timeout: go back and look again.
                        self.get_logger().warning(
                            'Visual blob lost for '
                            f'{now - lost_since:.1f} s; restarting the search.'
                        )
                        # 2026-11-17: rotating on the spot cannot help when the
                        # cube is too close to be inside the frame at all (a
                        # 5 cm cube under a level camera leaves the image below
                        # ~0.22 m). Open the standoff once, then look again;
                        # without this the run spins for the whole 75 s window
                        # (run_logs/legprobe_rule1_20261117-184656).
                        handoff_cube_name = str(
                            self.get_parameter('cube_name').value
                        )
                        if not backed_off and handoff_cube_name:
                            near = self._cached_planar_distance_to_cube(
                                handoff_cube_name
                            )
                            coast_min = float(self.get_parameter(
                                'visual_handoff_coast_min_distance_m').value)
                            if near is not None and near < coast_min:
                                backed_off = True
                                self.get_logger().warning(
                                    f'{handoff_cube_name}: only {near:.3f} m away '
                                    'while searching; opening the standoff before '
                                    'looking again.'
                                )
                                self._retreat_to_handoff_standoff(
                                    handoff_cube_name, near
                                )
                        lost_since = None
                        search_started = time.monotonic()
                        break
                    continue
                lost_since = None
                centre_x, width, area, image_width, bottom_row = detection
                last_row = float(bottom_row)
                centre_error = (centre_x - image_width / 2.0) / (image_width / 2.0)
                last_centred = abs(centre_error) <= centre_tolerance
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

    def _verify_placement(self, *, count: bool = False) -> bool:
        """Confirm the released cube actually rests inside the target zone.

        `count` marks the authoritative check for this delivery: the early one,
        taken right after the release. The post-retreat confirmation reuses the
        same measurement but must NOT advance the slot rotation - measured
        2026-11-16 (lanes_gui1, x=1 red -> A) both calls incremented, so two
        cubes left the counter at 4 and the third delivery was aimed at slot 1,
        on top of the cube already there; it slid 0.283 m off and failed
        VERIFY_PLACE with a delivery that had actually beeped fine.
        """
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
            if not count:
                self.get_logger().info(
                    f'{message}, inside limits. (slot rotation unchanged: the '
                    f'early check already counted this cube)')
                return True
            # Only a cube that actually landed inside counts towards the slot
            # rotation, so a failed delivery does not consume a slot. Counted
            # exactly once per delivery - see the `count` note above.
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
        """Freeze the world around the joint mutation, then run the call.

        IFRA's attacher creates/detaches joints inside a ROS service callback,
        i.e. on a thread that is not the physics thread. Gazebo Classic does not
        serialise that: the ODE wrapper is stepped at the same time, and this
        project sees the result as `gzserver exit code -11` about half a second
        after DETACH, always during VERIFY_PLACE, always the same fault:
          segfault at 158 ... in libgazebo_ode.so.11.10.2[...+7f4ae]  (klog)
        i.e. a null dereference at a fixed struct offset inside libgazebo_ode.
        The same signature appears in runs from 2026-10-08 (final3/sim.log:885)
        and 2026-10-13 (full_round_repro/sim.log:2948) as well as in
        run_logs/competition_yieldfix2/launch.log:1416 and
        run_logs/competition_yieldfix5/launch.log:2297, so it predates the
        navigation work. Pausing the world for the ~15 ms the service needs puts
        the mutation between two steps instead of inside one.
        """
        frozen = self._freeze_physics()
        self._link_service_frozen = frozen
        try:
            return self._call_link_service_unfrozen(label, client, request)
        finally:
            self._link_service_frozen = False
            if frozen and self._thaw_physics() is False:
                raise RuntimeError('GAZEBO_UNPAUSE_STATE_UNKNOWN')

    def _link_service_freeze_enabled(self) -> bool:
        try:
            return bool(self.get_parameter(
                'freeze_physics_around_link_services').value)
        except Exception:
            return False

    def _physics_client(self, name: str):
        cache = getattr(self, '_physics_clients', None)
        if cache is None:
            cache = self._physics_clients = {}
        if name in cache:
            return cache[name]
        client = None
        try:
            client = self.create_client(Empty, name)
            if not client.wait_for_service(timeout_sec=0.5):
                self.destroy_client(client)
                client = None
        except Exception:
            client = None
        if client is None and name not in cache:
            self.get_logger().warning(
                f'{name} is unavailable; will retry discovery when needed.')
        # A missing service can appear later during Gazebo startup or recovery.
        # Never permanently cache a failed discovery.
        if client is not None:
            cache[name] = client
        return client

    def _freeze_physics(self) -> bool:
        """Pause Gazebo; return True when the world is actually paused."""
        if not self._link_service_freeze_enabled():
            return False
        client = self._physics_client('/pause_physics')
        if client is None:
            raise RuntimeError('GAZEBO_PAUSE_STATE_UNKNOWN: service unavailable')
        timeout = float(self.get_parameter('physics_freeze_timeout_sec').value)
        try:
            started = time.monotonic()
            future = client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                # ROS service requests cannot be cancelled. If this pause takes
                # effect late, undo it after its response, not merely now (an
                # early unpause can otherwise race ahead of the late pause).
                def undo_late_pause(done):
                    try:
                        if done.result() is not None:
                            resume = self._physics_client('/unpause_physics')
                            if resume is not None:
                                resume.call_async(Empty.Request())
                            else:
                                self.get_logger().error('Late Gazebo pause could not be undone: unpause service unavailable.')
                    except Exception as exc:
                        self.get_logger().error(f'Late Gazebo pause recovery failed: {exc}')
                future.add_done_callback(undo_late_pause)
                self._thaw_physics()
                raise RuntimeError('GAZEBO_PAUSE_TIMEOUT')
            ok = future.result() is not None
            if not ok:
                raise RuntimeError('GAZEBO_PAUSE_FAILED')
        except Exception as exc:
            self.get_logger().error(f'pause_physics failed; refusing joint mutation: {exc}')
            raise RuntimeError('GAZEBO_PAUSE_STATE_UNKNOWN') from exc
        if ok and not getattr(self, '_freeze_announced', False):
            self._freeze_announced = True
            self.get_logger().info(
                f'Gazebo paused for joint services '
                f'({time.monotonic() - started:.3f} s); the ATTACH/DETACH '
                'mutation now runs between two physics steps.')
        return ok

    def _thaw_physics(self) -> bool:
        timeout = float(self.get_parameter('physics_freeze_timeout_sec').value)
        for attempt in range(1, 4):
            try:
                client = self._physics_client('/unpause_physics')
                if client is not None:
                    future = client.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
                    if future.done() and future.result() is not None:
                        return True
                self.get_logger().warning(
                    f'unpause_physics did not respond (attempt {attempt}/3).')
            except Exception as exc:
                self.get_logger().warning(
                    f'unpause_physics failed (attempt {attempt}/3): {exc}')
        self.get_logger().error(
            'GAZEBO_UNPAUSE_FAILED: physics may still be paused; '
            'stop the task and inspect Gazebo before restarting.')
        return False

    def _link_service_timeout(self) -> float:
        """How long to wait for ATTACH/DETACH.

        While the world is frozen for the call the wait is capped by
        `physics_freeze_timeout_sec`: a wedged service must not leave the whole
        simulation paused (the controllers and health checks would time out).
        """
        timeout = float(self.get_parameter('service_timeout_sec').value)
        if getattr(self, '_link_service_frozen', False):
            try:
                timeout = min(timeout, float(
                    self.get_parameter('physics_freeze_timeout_sec').value))
            except Exception:
                pass
        return timeout

    def _call_link_service_unfrozen(self, label: str, client, request) -> bool:
        self.get_logger().info(
            f'{label}: {request.model1_name}::{request.link1_name} <-> '
            f'{request.model2_name}::{request.link2_name}'
        )
        timeout = self._link_service_timeout()
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
            if (not getattr(self, '_link_service_frozen', False)
                    and self._confirm_detached()):
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

    def _navigation_motion_fault(self) -> str | None:
        if getattr(self, '_navigation_control_fault', False):
            return 'NAV_STATE_UNKNOWN'
        if getattr(self, '_geometry_fault', False):
            return 'GEOMETRY_STATE_UNKNOWN'
        return None

    def run_once(self) -> bool:
        if fault := FixedJointPickPlace._navigation_motion_fault(self):
            self._publish_executor_error(fault)
            return False
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
            # The plate check runs before the arm leaves transport posture: if
            # the base is not actually on its dock the arm is still folded, so
            # a refusal costs nothing but the leg itself.
            ('PLACING', 'check the target plate', self._verify_place_target_plate),
            (
                'PLACING', 'pre-place',
                lambda: self._move_arm_placing('Move to pre-place', 'pre_place_joints'),
            ),
            (
                'PLACING', 'place',
                lambda: self._move_arm_placing('Lower to place', 'place_joints'),
            ),
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
            # This is the authoritative check: it owns the slot rotation.
            ('VERIFY_PLACE', 'verify placement',
             lambda: self._verify_placement(count=True)),
            (
                'VERIFY_PLACE', 'retreat',
                lambda: self._move_arm('Retreat from place', 'pre_place_joints', placing=True),
            ),
            ('VERIFY_PLACE', 'return home', lambda: self._move_arm('Return home', 'home_joints', placing=True)),
            # Final confirmation in the resting position: a cube that the
            # retreat later nudged out of the plate still counts as missed.
            # Same measurement, but it must not advance the slot rotation.
            ('VERIFY_PLACE', 'verify placement after retreat',
             lambda: self._verify_placement(count=False)),
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
            if self._cancel_requested:
                self._stop_base()
                if not unsafe_state:
                    self._publish_executor_error('CANCELLED_BY_SUPERVISOR')
                return False
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

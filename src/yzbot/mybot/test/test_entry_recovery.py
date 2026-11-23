"""P0 regression tests for the narrow-entrance (C zone) recovery.

2026-11-21: red_cube_1 -> C failed 0/5 after 174 s. The base stopped 0.30 m
from the wall tip of the 0.89 m mouth, i.e. inside the inscribed band of both
the 0.45 m planned radius and the 0.35 m local one, so NavFn rejected the start
cell, the planner-radius relax could not change that, the controller refused
the path, and the 0.70 m blind shove moved the base 0.03 m.

Each block below pins one thing a review found missing:

* the entry retreat delegates to the shared guarded retreat with the same
  absolute deadline and requires that the base leave the planner band;
* plan probing is separated from driving: a plan-only behaviour tree asks the
  global planner for a route, because a plain NavigateToPose is not
  side-effect free - both shipped trees clear the costmaps, back up 0.5 m and
  spin 0.8 rad on failure;
* driving uses a recovery-free tree, so nothing but this code retreats, waits or
  gives up between attempts;
* staging, probing and every attempt share ONE deadline, and an unknown pose is
  never accepted as "already staged";
* the configuration is pinned too: the waiting points must clear the loaded
  envelope on the map the competition actually loads (room_from_world.pgm, not
  map.pgm), must really differ, and the scored profile must not carry the
  switch yet.

Run with:
    source /opt/ros/humble/setup.bash; source install/setup.bash
    python3 -m unittest discover -s src/yzbot/mybot/test -p 'test_entry_recovery.py'
"""
import ast
import importlib.util
import math
import re
import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'fixed_joint_pick_place.py'
CONFIG = ROOT / 'config' / 'fixed_joint_pick_place.yaml'
OVERLAY = ROOT / 'config' / 'profiles' / 'entry_recovery_test.yaml'
SCORED_PROFILE = ROOT / 'config' / 'profiles' / 'executor_fast.yaml'
# The map src/yzbot/mybot/launch/competition_bringup.launch.py loads.
MAP = ROOT.parent / 'bot_navigation' / 'maps' / 'room_from_world.pgm'
MAP_META = ROOT.parent / 'bot_navigation' / 'maps' / 'room_from_world.yaml'

if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location(
    'pick_place_entry_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_params(path=CONFIG, node='fixed_joint_pick_place'):
    with path.open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)[node]['ros__parameters']


def read_pgm(path):
    """Return (width, height, pixels[bytes]) of a binary (P5) PGM."""
    data = path.read_bytes()
    index = 0

    def token():
        nonlocal index
        while True:
            while index < len(data) and data[index:index + 1].isspace():
                index += 1
            if data[index:index + 1] == b'#':
                while index < len(data) and data[index:index + 1] != b'\n':
                    index += 1
                continue
            break
        start = index
        while index < len(data) and not data[index:index + 1].isspace():
            index += 1
        return data[start:index]

    magic = token()
    if magic != b'P5':
        raise AssertionError(f'{path} is not a binary PGM: {magic!r}')
    width = int(token())
    height = int(token())
    token()  # maxval, unused: a trinary map compares occupied cells to zero
    index += 1  # the single whitespace byte after maxval
    return width, height, data[index:index + width * height]


_CELLS = {}


def occupied_cells():
    """World coordinates of every occupied cell on the competition map."""
    if 'cells' not in _CELLS:
        with MAP_META.open(encoding='utf-8') as handle:
            meta = yaml.safe_load(handle)
        width, height, pixels = read_pgm(MAP)
        resolution = float(meta['resolution'])
        origin_x, origin_y = (float(v) for v in meta['origin'][:2])
        # PGM row 0 is the TOP row: y = origin_y + (height - 1 - row) * res.
        _CELLS['cells'] = [
            (origin_x + (i % width + 0.5) * resolution,
             origin_y + (height - 1 - i // width + 0.5) * resolution)
            for i, value in enumerate(pixels) if value == 0
        ]
    return _CELLS['cells']


def static_clearance(x, y):
    return min(math.hypot(ox - x, oy - y) for ox, oy in occupied_cells())


class EntryRecoveryPoseTest(unittest.TestCase):
    def test_off_sentinels_are_none(self):
        self.assertIsNone(module.entry_recovery_pose(None))
        self.assertIsNone(module.entry_recovery_pose([]))
        self.assertIsNone(module.entry_recovery_pose([0.0]))
        self.assertIsNone(module.entry_recovery_pose([1.0]))
        self.assertIsNone(module.entry_recovery_pose([float('nan'), 1.0]))
        self.assertIsNone(module.entry_recovery_pose(['x', 1.0]))

    def test_two_values_gain_a_zero_yaw(self):
        self.assertEqual(
            module.entry_recovery_pose([-4.3, -6.8]), [-4.3, -6.8, 0.0])

    def test_three_values_pass_through(self):
        self.assertEqual(
            module.entry_recovery_pose([-4.3, -6.8, 1.57]), [-4.3, -6.8, 1.57])

    def test_describe_pose_keeps_the_heading_on_the_record(self):
        text = module.describe_pose((-4.19, -6.74, math.pi / 2))
        self.assertIn('(-4.19, -6.74)', text)
        self.assertIn('+90 deg', text)
        self.assertEqual(module.describe_pose(None), 'pose unknown')


class TravelClearanceTest(unittest.TestCase):
    """The retreat guard's only sensor input, tested without a live node."""

    STEP = 2 * math.pi / 5

    def clearance(self, ranges, direction):
        return module.travel_clearance(
            ranges, -math.pi, self.STEP, 0.05, 12.0, direction)

    def test_forward_uses_the_front_half_plane(self):
        # Beams land on -180, -108, -36, +36, +108 deg; +36 is the 0.4 m one.
        self.assertAlmostEqual(
            self.clearance([9.0, 9.0, 9.0, 0.4, 9.0], 1.0), 0.4)

    def test_reverse_uses_the_rear_half_plane(self):
        # -108 deg carries 0.8 m; a forward check must not see it.
        self.assertAlmostEqual(
            self.clearance([9.0, 0.8, 9.0, 9.0, 9.0], -1.0), 0.8)
        self.assertAlmostEqual(
            self.clearance([9.0, 0.8, 9.0, 9.0, 9.0], 1.0), 9.0)

    def test_out_of_range_returns_are_ignored(self):
        # nan and inf are skipped, 99.0 exceeds range_max, 0.01 is below
        # range_min; only the 3.0 beam at +36 deg is usable.
        ranges = [float('nan'), 0.01, 99.0, 3.0, float('inf')]
        self.assertAlmostEqual(self.clearance(ranges, 1.0), 3.0)

    def test_no_usable_return_is_none_not_zero(self):
        ranges = [float('nan'), 0.01, 99.0, float('inf'), float('nan')]
        self.assertIsNone(self.clearance(ranges, 1.0))
        self.assertIsNone(self.clearance(ranges, -1.0))


class EntryRecoveryClassificationTest(unittest.TestCase):
    def classify(self, banded=False, failure=None, moved=0.0, elapsed=0.0,
                 abort=5.0):
        return module.classify_entrance_failure(
            start_in_band=banded,
            failure=failure,
            moved_m=moved,
            elapsed_sec=elapsed,
            plan_abort_sec=abort,
        )

    def test_band_wins_over_every_other_signal(self):
        # The 2026-11-21 stop sat 0.30 m from the wall tip: this is the case
        # that actually ended the round, and backing up is the only fix.
        self.assertEqual(self.classify(True, 'no_motion', 0.0, 0.2), 'band')
        self.assertEqual(
            self.classify(True, 'wedge_refusals', 0.0, 20.0), 'band')

    def test_controller_refusal_is_its_own_class(self):
        self.assertEqual(
            self.classify(False, 'wedge_refusals', 0.0, 0.4), 'controller')

    def test_fast_failure_without_motion_is_not_a_planner_block(self):
        # Deliberately NOT 'no_path': a behaviour tree that failed to load
        # fails just as fast, so only the plan-only probe may decide.
        self.assertEqual(self.classify(False, None, 0.0, 0.01), 'refused')
        self.assertEqual(self.classify(False, 'no_motion', 0.02, 1.5), 'refused')

    def test_long_failure_without_motion_is_a_stall(self):
        self.assertEqual(
            self.classify(False, 'no_progress', 0.0, 30.0), 'stalled')

    def test_motion_makes_it_progress_unless_the_controller_refused(self):
        self.assertEqual(
            self.classify(False, 'no_progress', 0.60, 30.0), 'progress')
        self.assertEqual(
            self.classify(False, 'wedge_refusals', 0.60, 3.0), 'controller')


class EntryRecoveryConfigTest(unittest.TestCase):
    def setUp(self):
        self.params = load_params()
        self.reach = float(self.params['entry_recovery_reach_m'])

    def test_zone_c_is_the_enabled_zone(self):
        self.assertTrue(self.params['entry_recovery_enabled'])
        self.assertEqual(
            self.params['entry_recovery_zone'].strip().upper(), 'C')

    def test_the_round_failed_pose_is_separated_from_the_waiting_point(self):
        # The failed round re-started C from (-4.19, -6.74). The first version
        # of this feature staged 0.125 m from that spot with a 0.45 m arrival
        # radius, i.e. it replayed the same approach. Require the verified
        # arrival band (1.5x reach) to leave real room around the failed pose.
        pose = module.entry_recovery_pose(self.params['entry_recovery_wait_pose'])
        failed = (-4.19, -6.74)
        separation = math.hypot(pose[0] - failed[0], pose[1] - failed[1])
        self.assertGreater(
            separation - 1.5 * self.reach, 0.20,
            f'the waiting point is only {separation:.2f} m from the pose that '
            f'already failed, and a {self.reach:.2f} m arrival radius leaves '
            'no margin against replaying the same approach',
        )

    def test_both_staging_points_clear_the_loaded_envelope(self):
        radius = float(self.params['loaded_robot_radius_m'])
        for key in ('entry_recovery_wait_pose', 'entry_recovery_alt_pose'):
            pose = module.entry_recovery_pose(self.params[key])
            self.assertIsNotNone(pose, f'{key} is not a usable pose')
            clearance = static_clearance(pose[0], pose[1])
            self.assertGreaterEqual(
                clearance, radius + 0.25,
                f'{key} ({pose[0]:.2f}, {pose[1]:.2f}) clears only '
                f'{clearance:.2f} m on room_from_world.pgm',
            )

    def test_the_two_staging_points_really_differ(self):
        first = module.entry_recovery_pose(
            self.params['entry_recovery_wait_pose'])
        second = module.entry_recovery_pose(
            self.params['entry_recovery_alt_pose'])
        self.assertGreater(
            math.hypot(first[0] - second[0], first[1] - second[1]), 0.40)

    def test_bounds_are_sane(self):
        self.assertGreater(self.params['entry_recovery_wait_budget_sec'], 0.0)
        self.assertGreaterEqual(self.params['entry_recovery_max_attempts'], 2)
        self.assertGreater(
            self.params['entry_recovery_attempt_timeout_sec'],
            self.params['entry_recovery_plan_abort_sec'],
        )
        self.assertGreater(self.params['entry_recovery_retreat_m'], 0.0)
        self.assertGreater(
            self.params['entry_recovery_stop_clearance_m'],
            float(self.params['loaded_robot_radius_m']),
            'the retreat must stop before the loaded envelope touches anything',
        )

    def test_every_yaml_key_is_declared_in_the_executor(self):
        source = SCRIPT.read_text(encoding='utf-8')
        declared = {
            line.split("'")[1]
            for line in source.splitlines()
            if "declare_parameter('entry_recovery_" in line
        }
        configured = {
            key for key in self.params if key.startswith('entry_recovery_')}
        self.assertEqual(declared, configured)

    # The overlay exists to be identical to the base config except where a unit
    # run must differ; each exception is listed here with its reason.
    OVERLAY_EXCEPTIONS = {'entry_recovery_force_first'}

    def test_the_overlay_repeats_the_base_values(self):
        overlay = load_params(OVERLAY)
        shared = {k for k in overlay if k.startswith('entry_recovery_')}
        self.assertTrue(shared)
        for key in shared - self.OVERLAY_EXCEPTIONS:
            self.assertEqual(
                overlay[key], self.params[key],
                f'{key} differs between the base config and the opt-in '
                'overlay, so a unit run would not exercise what is documented',
            )

    def test_the_force_switch_is_test_only(self):
        # It must be off in the base config, on only in the opt-in overlay, and
        # absent from the scored profile, so a scored round can never skip the
        # ordinary first attempt.
        self.assertIs(self.params['entry_recovery_force_first'], False)
        self.assertIs(load_params(OVERLAY)['entry_recovery_force_first'], True)
        self.assertNotIn(
            'entry_recovery_force_first', load_params(SCORED_PROFILE))

    def test_the_scored_profile_does_not_carry_the_switch(self):
        # competition_bringup.launch.py loads config/profiles/executor_fast.yaml
        # INSTEAD OF the base config, so putting the keys there is the only way
        # a scored round could run them. Absent until the report justifies it.
        scored = load_params(SCORED_PROFILE)
        self.assertFalse(
            [key for key in scored if key.startswith('entry_recovery_')])


class EntryRecoverySourceTest(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPT.read_text(encoding='utf-8')
        self.tree = ast.parse(self.source)

    def method_source(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(self.source, node)
        raise AssertionError(f'{name} not found')

    def test_navigate_to_hands_the_zone_leg_to_the_recovery(self):
        body = self.method_source('_navigate_to')
        self.assertIn('self._entry_recovery_applies(label)', body)
        self.assertIn('self._entry_recovery(', body)

    def test_recovery_never_shrinks_the_footprint_or_clears_costmaps(self):
        for name in ('_entry_recovery', '_entry_retreat', '_entry_stage'):
            body = self.method_source(name)
            for forbidden in (
                '_enter_dead_zone_escape', '_relax_global_footprint',
                '_clear_costmaps', '_restore_nominal_geometry(',
            ):
                self.assertNotIn(
                    forbidden, body,
                    f'{name} must not call {forbidden}: the bounded exit '
                    'neither shrinks the loaded envelope nor re-clears maps',
                )

    def test_entry_orchestration_does_not_bypass_band_leaving_retreat(self):
        for name in ('_entry_recovery', '_entry_leave_band', '_entry_stage'):
            self.assertNotIn('self._back_off(', self.method_source(name))
        self.assertIn('self._entry_retreat(', self.method_source('_entry_leave_band'))

    def test_probe_and_drive_use_dedicated_behaviour_trees(self):
        self.assertIn(
            'nav_to_pose_entry_probe.xml',
            self.method_source('_entry_probe_tree'))
        self.assertIn(
            'nav_to_pose_entry_guard.xml',
            self.method_source('_entry_drive_tree'))
        for name in ('_entry_stage', '_entry_recovery'):
            self.assertIn(
                'self._entry_drive_tree()', self.method_source(name),
                f'{name} must drive with the recovery-free tree',
            )

    def test_one_deadline_covers_every_stage(self):
        body = self.method_source('_entry_recovery')
        self.assertEqual(body.count('deadline = time.monotonic()'), 1)
        self.assertIn('self._entry_stage(', body)
        self.assertIn('self._entry_probe(', body)

    def test_force_switch_only_skips_the_leg_the_recovery_owns(self):
        """The 2026-11-21 forced round skipped `red_cube_1 visual handoff`.

        `forced_entry` was computed from the attempt number and the parameter
        alone, so it also suppressed the ordinary first attempt of every other
        leg. Applicability is part of the condition, not a separate log line.
        """
        body = self.method_source('_navigate_to')
        match = re.search(r'forced_entry = \((.*?)\n\s*\)', body, re.S)
        self.assertIsNotNone(match, 'forced_entry assignment not found')
        self.assertIn('_entry_recovery_applies(label)', match.group(1))

    def test_probe_polls_the_future_instead_of_blocking(self):
        """`spin_until_future_complete` never delivered the probe result.

        In the forced round bt_navigator reported `Goal succeeded` 150 ms after
        accepting the probe, yet the node scored two 5 s timeouts and failed
        closed on a working planner answer. The probe must poll the future with
        the same spin_once mechanism the rest of the node uses.
        """
        body = self.method_source('_entry_probe')
        self.assertIn('future.done()', body)
        self.assertIn('rclpy.spin_once(self', body)
        code = '\n'.join(
            line for line in body.splitlines()
            if not line.lstrip().startswith('#'))
        self.assertNotIn('spin_until_future_complete', code)


class EntryRecoveryBehaviourTreeTest(unittest.TestCase):
    """The trees are part of the safety argument, so they are asserted."""

    FORBIDDEN = ('ClearEntireCostmap', 'BackUp', 'Spin', 'RecoveryNode')

    def parse(self, name):
        path = ROOT / 'config' / name
        self.assertTrue(path.is_file(), f'{name} is missing')
        text = path.read_text(encoding='utf-8')
        return text, ElementTree.fromstring(text)

    def test_probe_tree_only_plans(self):
        text, root = self.parse('nav_to_pose_entry_probe.xml')
        tags = [node.tag for node in root.iter() if node.tag != 'root']
        self.assertEqual(tags, ['BehaviorTree', 'ComputePathToPose'])
        for forbidden in self.FORBIDDEN + ('FollowPath',):
            self.assertNotIn(f'<{forbidden}', text)

    def test_drive_tree_has_no_recovery_actions(self):
        text, root = self.parse('nav_to_pose_entry_guard.xml')
        for forbidden in self.FORBIDDEN:
            self.assertNotIn(
                f'<{forbidden}', text,
                f"{forbidden} in the drive tree would move or clear the base "
                "behind the recovery's back")
        tags = [node.tag for node in root.iter() if node.tag != 'root']
        self.assertIn('FollowPath', tags)
        self.assertIn('ComputePathToPose', tags)


class EntryRecoveryRetreatTest(unittest.TestCase):
    """Pin the adapter to the shared guarded, deadline-bound retreat."""

    def test_requests_band_exit_with_original_deadline_and_selected_direction(self):
        for direction in (None, -1.0, 1.0):
            with self.subTest(direction=direction):
                stop = Mock(return_value=True)
                node = SimpleNamespace(_back_off=stop)
                ok = module.FixedJointPickPlace._entry_retreat(
                    node, 'warehouse C', direction, 0.6, 0.10, 100.05)
                self.assertTrue(ok)
                stop.assert_called_once_with(
                    'ENTRY_RECOVERY warehouse C', 0.6, 0.10, direction,
                    leave_band=True, until=100.05)

    def test_refusal_from_the_shared_guard_is_not_treated_as_success(self):
        retreat = Mock(return_value=False)
        node = SimpleNamespace(_back_off=retreat)
        self.assertFalse(module.FixedJointPickPlace._entry_retreat(
            node, 'warehouse C', -1.0, 0.20, 0.08, 100.25))
        retreat.assert_called_once_with(
            'ENTRY_RECOVERY warehouse C', 0.20, 0.08, -1.0,
            leave_band=True, until=100.25)


class EntryRecoveryStageTest(unittest.TestCase):
    """Drive the real `_entry_stage` arrival verification."""

    POSE = [-4.60, -6.90, 0.0]

    def make_node(self, *, base, answers=(True,)):
        node = SimpleNamespace(
            base=base,
            log=[],
            nav_calls=[],
            answers=list(answers),
        )
        node.get_logger = lambda: SimpleNamespace(
            info=node.log.append, warning=node.log.append,
            error=node.log.append)
        node._cached_base_pose = lambda: node.base
        node._entry_drive_tree = lambda: '/share/mybot/config/' \
            'nav_to_pose_entry_guard.xml'

        def navigate_once(label, values, visual=None, completion=None,
                          chain=None, **kwargs):
            node.nav_calls.append({
                'label': label, 'values': list(values),
                'completion': kwargs.get('completion_radius_m', completion),
                'tree': kwargs.get('behavior_tree'),
                'max_duration': kwargs.get('max_duration_sec'),
            })
            answer = node.answers.pop(0) if node.answers else True
            if not answer:
                return False
            # Land exactly on the staging point unless a test says otherwise.
            node.base = (values[0], values[1], values[2])
            return True

        node._navigate_once = navigate_once
        return node

    def run_stage(self, node, *, reach=0.15, yaw_tol=0.35, budget=30.0):
        with patch.object(module.time, 'monotonic', lambda: 1000.0):
            return module.FixedJointPickPlace._entry_stage(
                node, 'warehouse C', list(self.POSE), reach, yaw_tol,
                budget, 1000.0 + budget)

    def test_an_unknown_pose_is_never_read_as_already_staged(self):
        node = self.make_node(base=None)
        self.assertFalse(self.run_stage(node))
        self.assertEqual(node.nav_calls, [])
        self.assertTrue(any('pose unknown' in line for line in node.log))

    def test_a_base_inside_the_radius_is_already_staged(self):
        node = self.make_node(base=(-4.55, -6.85, 0.0))
        self.assertTrue(self.run_stage(node))
        self.assertEqual(node.nav_calls, [])
        self.assertFalse(any('off the configured heading' in line
                             for line in node.log))

    def test_a_heading_error_only_warns_and_keeps_the_actual_pose(self):
        node = self.make_node(base=(-4.55, -6.85, math.pi))
        self.assertTrue(self.run_stage(node))
        self.assertTrue(any('off the configured heading' in line
                            for line in node.log))

    def test_staging_drives_with_the_guard_tree_and_the_tight_radius(self):
        node = self.make_node(base=(-5.50, -6.00, 0.0))
        self.assertTrue(self.run_stage(node))
        self.assertEqual(len(node.nav_calls), 1)
        call = node.nav_calls[0]
        self.assertEqual(call['completion'], 0.15)
        self.assertTrue(call['tree'].endswith('nav_to_pose_entry_guard.xml'))
        self.assertEqual(call['max_duration'], 30.0)

    def test_landing_far_off_the_point_fails_closed(self):
        node = self.make_node(base=(-5.50, -6.00, 0.0))
        node._navigate_once = lambda *a, **k: setattr(
            node, 'base', (-4.60, -6.40, 0.0)) or True
        self.assertFalse(self.run_stage(node))
        self.assertTrue(any('over the verified radius' in line
                            for line in node.log))

    def test_a_failed_staging_drive_fails_closed(self):
        node = self.make_node(base=(-5.50, -6.00, 0.0), answers=(False,))
        self.assertFalse(self.run_stage(node))
        self.assertTrue(any('could not reach the staging point' in line
                            for line in node.log))

    def test_no_budget_left_stops_before_driving(self):
        node = self.make_node(base=(-5.50, -6.00, 0.0))
        with patch.object(module.time, 'monotonic', lambda: 1000.0):
            ok = module.FixedJointPickPlace._entry_stage(
                node, 'warehouse C', list(self.POSE), 0.15, 0.35,
                30.0, 1000.5)
        self.assertFalse(ok)
        self.assertEqual(node.nav_calls, [])
        self.assertTrue(any('no budget left' in line for line in node.log))


class EntryProbeTest(unittest.TestCase):
    """Run the real plan-only probe against a scripted NavigateToPose handle."""

    def make_node(self, *, accepted=True, status=None, spins_needed=0):
        node = SimpleNamespace(spins=0, log=[], stops=0, cancelled=0)
        node.get_parameter = lambda name: SimpleNamespace(
            value=2.0 if name == 'service_timeout_sec' else None)
        node.get_logger = lambda: SimpleNamespace(
            info=node.log.append, warning=node.log.append,
            error=node.log.append)
        node._stop_base = lambda: setattr(node, 'stops', node.stops + 1)
        node._entry_probe_tree = lambda: '/share/mybot/config/' \
            'nav_to_pose_entry_probe.xml'
        node._nav_goal = lambda values, chain=None, behavior_tree=None: (
            SimpleNamespace(), SimpleNamespace())
        node._send_goal_tolerant = (
            lambda label, client, goal, timeout, grace: handle)

        future = SimpleNamespace(
            done=lambda: spins_needed is not None and node.spins >= spins_needed,
            result=lambda: SimpleNamespace(status=status),
        )

        def cancel():
            node.cancelled += 1

        handle = SimpleNamespace(
            accepted=accepted,
            get_result_async=lambda: future,
            cancel_goal_async=cancel,
        )
        return node

    def run_probe(self, node, timeout=5.0):
        clock = [1000.0]

        def monotonic():
            clock[0] += 0.40
            return clock[0]

        def spin_once(*_a, **_k):
            node.spins += 1

        with patch.object(module.rclpy, 'ok', lambda: True), \
                patch.object(module.rclpy, 'spin_once', spin_once), \
                patch.object(module.time, 'monotonic', monotonic):
            return module.FixedJointPickPlace._entry_probe(
                node, 'warehouse C', [0.016, -2.702, -1.5708], timeout)

    def test_a_planner_answer_means_a_path(self):
        node = self.make_node(
            status=module.GoalStatus.STATUS_SUCCEEDED, spins_needed=2)
        self.assertEqual(self.run_probe(node), 'path')
        self.assertGreaterEqual(node.spins, 2)
        self.assertFalse(any('timed out' in line for line in node.log))

    def test_an_aborted_plan_means_no_path(self):
        node = self.make_node(
            status=module.GoalStatus.STATUS_ABORTED, spins_needed=1)
        self.assertEqual(self.run_probe(node), 'no_path')

    def test_a_probe_that_never_answers_is_unknown(self):
        node = self.make_node(status=None, spins_needed=None)
        self.assertEqual(self.run_probe(node), 'unknown')
        self.assertEqual(node.cancelled, 1)
        self.assertGreater(node.stops, 0)
        self.assertTrue(any('timed out' in line for line in node.log))

    def test_a_rejected_probe_is_unknown_not_no_path(self):
        node = self.make_node(accepted=False, spins_needed=1)
        self.assertEqual(self.run_probe(node), 'unknown')
        self.assertTrue(any('not accepted' in line for line in node.log))


class EntryRecoveryLoopTest(unittest.TestCase):
    """Drive the real `_entry_recovery` against a scripted Nav2 client."""

    PARAMS = {
        'entry_recovery_enabled': True,
        'entry_recovery_zone': 'C',
        'entry_recovery_wait_pose': [-4.60, -6.90, 0.0],
        'entry_recovery_alt_pose': [-5.10, -6.55, 0.0],
        'entry_recovery_wait_budget_sec': 45.0,
        'entry_recovery_max_attempts': 3,
        'entry_recovery_attempt_timeout_sec': 30.0,
        'entry_recovery_plan_abort_sec': 5.0,
        'entry_recovery_reach_m': 0.20,
        'entry_recovery_yaw_tol_rad': 0.35,
        'entry_recovery_retreat_m': 0.6,
        'entry_recovery_retreat_speed_mps': 0.10,
        'entry_recovery_stop_clearance_m': 0.40,
        'entry_recovery_scan_max_age_sec': 0.6,
    }

    def make_node(self, *, probes=('path',), drives=(('ok', 0.0),),
                  leaves=(True,), stages=(True,), banded=False,
                  base=(-4.60, -6.90, 0.0)):
        node = SimpleNamespace(
            _entry_recovery_running=False,
            _last_attempt_failure=None,
            _yield_arc_goal=None,
            _yield_decision=SimpleNamespace(holds=False),
            base=base,
            clock=1000.0,
            log=[],
            stage_calls=[],
            drive_calls=[],
            probe_calls=[],
            leave_calls=[],
            probe_answers=list(probes),
            drive_answers=list(drives),
            leave_answers=list(leaves),
            stage_answers=list(stages),
            banded=banded,
        )
        node.get_parameter = lambda name: SimpleNamespace(
            value=self.PARAMS[name])
        node.get_logger = lambda: SimpleNamespace(
            info=node.log.append, warning=node.log.append,
            error=node.log.append)
        node._cached_base_pose = lambda: node.base
        node._start_cell_inside_global_band = lambda: node.banded
        node._stop_base = lambda: None
        node._entry_probe_tree = lambda: '/share/mybot/config/' \
            'nav_to_pose_entry_probe.xml'
        node._entry_drive_tree = lambda: '/share/mybot/config/' \
            'nav_to_pose_entry_guard.xml'

        def leave(label, deadline):
            node.leave_calls.append(label)
            answer = node.leave_answers.pop(0) if node.leave_answers else False
            if answer:
                node.banded = False
            return answer

        def stage(label, pose, reach, yaw_tol, timeout, deadline):
            node.stage_calls.append(list(pose))
            # Reaching the staging point is the expected case; a test that
            # wants a staging failure passes stages=(False,).
            answer = node.stage_answers.pop(0) if node.stage_answers else True
            if answer:
                node.base = (pose[0], pose[1], pose[2])
            return answer

        def probe(label, values, timeout):
            node.probe_calls.append(label)
            if not node.probe_answers:
                return 'path'
            return (node.probe_answers.pop(0) if len(node.probe_answers) > 1
                    else node.probe_answers[0])

        def navigate_once(label, values, visual=None, completion=None,
                          chain=None, **kwargs):
            node.drive_calls.append((label, kwargs.get('behavior_tree')))
            outcome, moved = (node.drive_answers.pop(0)
                              if node.drive_answers else ('ok', 0.0))
            if outcome == 'ok':
                return True
            node._last_attempt_failure = {
                'controller': 'wedge_refusals',
                'band': 'no_progress',
            }.get(outcome, 'no_progress')
            if outcome == 'band':
                node.banded = True
            x, y, yaw = node.base
            node.base = (x + moved, y, yaw)
            return False

        node._entry_leave_band = leave
        node._entry_stage = stage
        node._entry_probe = probe
        node._navigate_once = navigate_once
        for name in ('_entry_stage_poses', '_entry_wait_pose',
                     '_entry_recovery_applies', '_entry_travelled'):
            setattr(node, name, (lambda name: lambda *a, **k: getattr(
                module.FixedJointPickPlace, name)(node, *a, **k))(name))
        return node

    def run_recovery(self, node, values=(0.016, -2.702, -1.570796)):
        def monotonic():
            node.clock += 0.5
            return node.clock

        def sleep(seconds):
            node.clock += max(0.0, float(seconds))

        with patch.object(module.time, 'monotonic', monotonic), \
                patch.object(module.time, 'sleep', sleep):
            return module.FixedJointPickPlace._entry_recovery(
                node, 'warehouse C', list(values), None, None, None)

    def test_a_clear_entrance_drives_once_with_the_guard_tree(self):
        node = self.make_node()
        self.assertTrue(self.run_recovery(node))
        self.assertEqual(len(node.drive_calls), 1)
        self.assertEqual(node.drive_calls[0][1], node._entry_drive_tree())
        self.assertFalse(node._entry_recovery_running)

    def test_a_static_block_from_both_points_fails_closed(self):
        node = self.make_node(probes=('no_path', 'no_path'))
        self.assertFalse(self.run_recovery(node))
        self.assertEqual(node.drive_calls, [])
        self.assertEqual(len(node.stage_calls), 2)
        self.assertTrue(any('static block' in line for line in node.log))

    def test_an_unknown_probe_never_becomes_a_goahead(self):
        node = self.make_node(probes=('unknown', 'unknown', 'unknown'))
        self.assertFalse(self.run_recovery(node))
        self.assertEqual(node.drive_calls, [])
        self.assertTrue(any('could not be evaluated' in line
                            for line in node.log))

    def test_a_no_path_answer_moves_to_the_other_staging_point(self):
        node = self.make_node(probes=('no_path', 'path'))
        self.assertTrue(self.run_recovery(node))
        self.assertEqual([p[:2] for p in node.stage_calls],
                         [[-4.60, -6.90], [-5.10, -6.55]])

    def test_a_band_after_an_attempt_leaves_it_again(self):
        node = self.make_node(
            probes=('path', 'path'), drives=(('band', 0.05), ('ok', 0.0)),
            leaves=(True, True))
        self.assertTrue(self.run_recovery(node))
        self.assertEqual(len(node.leave_calls), 2)

    def test_a_band_that_cannot_be_left_fails_closed(self):
        node = self.make_node(leaves=(False,))
        self.assertFalse(self.run_recovery(node))
        self.assertEqual(node.drive_calls, [])
        self.assertTrue(any('shrinking the loaded footprint' in line
                            for line in node.log))

    def test_controller_refusals_exhaust_the_attempts_and_fail_closed(self):
        # Two staging points exist, so the third attempt would have to start
        # from a pose that already failed; it is skipped rather than replayed.
        node = self.make_node(
            probes=('path',) * 3, drives=(('controller', 0.0),) * 3)
        self.assertFalse(self.run_recovery(node))
        self.assertEqual(len(node.drive_calls), 2)
        self.assertTrue(any('already failed' in line for line in node.log))
        self.assertTrue(any('entrance still blocked' in line
                            for line in node.log))

    def test_a_drive_that_keeps_failing_without_moving_fails_closed(self):
        node = self.make_node(
            probes=('path',) * 3, drives=(('refused', 0.0), ('refused', 0.0)))
        self.assertFalse(self.run_recovery(node))
        self.assertLessEqual(len(node.drive_calls), 2)
        self.assertTrue(any('loads' in line for line in node.log))

    def test_a_failed_staging_point_fails_closed(self):
        node = self.make_node(stages=(False,))
        self.assertFalse(self.run_recovery(node))
        self.assertEqual(node.drive_calls, [])
        self.assertEqual(node.probe_calls, [])

    def test_the_recovery_flag_is_cleared_when_the_tree_is_missing(self):
        node = self.make_node()

        def explode(*a, **k):
            raise RuntimeError('behaviour tree is not installed')

        node._navigate_once = explode
        self.assertFalse(self.run_recovery(node))
        self.assertFalse(node._entry_recovery_running)
        self.assertTrue(any('failing closed' in line for line in node.log))


class EntryRecoveryGateTest(unittest.TestCase):
    """`_entry_recovery_applies` decides which legs may use the recovery."""

    PARAMS = {
        'entry_recovery_enabled': True,
        'entry_recovery_zone': 'C',
        'entry_recovery_wait_pose': [-4.60, -6.90, 0.0],
        'entry_recovery_alt_pose': [-5.10, -6.55, 0.0],
    }

    def make_node(self, **overrides):
        params = dict(self.PARAMS)
        params.update(overrides)
        node = SimpleNamespace(_entry_recovery_running=False)
        node.get_parameter = lambda name: SimpleNamespace(value=params[name])
        for name in ('_entry_recovery_applies', '_entry_stage_poses',
                     '_entry_wait_pose'):
            setattr(node, name, (lambda name: lambda *a, **k: getattr(
                module.FixedJointPickPlace, name)(node, *a, **k))(name))
        return node

    def applies(self, label, **overrides):
        return module.FixedJointPickPlace._entry_recovery_applies(
            self.make_node(**overrides), label)

    def test_the_configured_dock_leg_is_covered(self):
        self.assertTrue(self.applies('warehouse C'))
        self.assertTrue(self.applies('  Warehouse c  '))

    def test_other_zones_and_legs_keep_the_old_ladder(self):
        for label in ('warehouse A', 'warehouse B',
                      'red_cube_1 to C continuous detour',
                      'red_cube_1 visual handoff'):
            self.assertFalse(self.applies(label), label)

    def test_the_switch_really_gates_it(self):
        self.assertFalse(
            self.applies('warehouse C', entry_recovery_enabled=False))

    def test_a_missing_waiting_point_keeps_the_old_ladder(self):
        self.assertFalse(
            self.applies('warehouse C', entry_recovery_wait_pose=[0.0]))

    def test_an_alternate_too_close_to_the_primary_is_dropped(self):
        node = self.make_node(entry_recovery_alt_pose=[-4.65, -6.95, 0.0])
        self.assertEqual(
            module.FixedJointPickPlace._entry_stage_poses(node),
            [[-4.6, -6.9, 0.0]])

    def test_a_running_recovery_is_not_reentered(self):
        node = self.make_node()
        node._entry_recovery_running = True
        self.assertFalse(
            module.FixedJointPickPlace._entry_recovery_applies(
                node, 'warehouse C'))


if __name__ == '__main__':
    unittest.main()

"""Static-map guard for the blue_cube_5 outer pickup route.

The rule 3 round of 2026-11-22 lost 426 s (62 % of the round) on one pickup:
a mover parked in the 1.10 m mouth at (-4.3, -1.1) blocked the local costmap,
the base inched at the wall tip, and two 45 s progress watchdogs expired in a
row before the retry ladder backed off. The executor now carries one verified
outer route for that pickup and drives it once after a failed attempt.

A map check cannot certify the Nav2 trajectory or the live costmap, and the
last leg back to the pickup pose is still planned by Nav2 (so its straight
chord is expected to cross geometry). What this file guards is the part the
route is responsible for: every configured point and every straight chord
between them stays clear of the static walls, the route stays away from the
mouth it exists to avoid, and the scored profile actually carries it.
"""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

BOT = Path(__file__).resolve().parents[1]
MAP = BOT.parent / 'bot_navigation' / 'maps'
PARAMS = BOT / 'config' / 'fixed_joint_pick_place.yaml'
PROFILE = BOT / 'config' / 'profiles' / 'executor_fast.yaml'
SCRIPT = BOT / 'scripts' / 'fixed_joint_pick_place.py'
CUBE = 'blue_cube_5'
POINTS = [f'{CUBE}_pickup_fallback_via_{i}' for i in (1, 2, 3)]
MOUTH = (-4.28, -1.22)  # tightest point of the direct route, 0.55 m clearance


def load_executor():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location('pickup_fallback_executor', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_occupied_boxes():
    metadata = yaml.safe_load((MAP / 'room_from_world.yaml').read_text())
    with (MAP / metadata['image']).open('rb') as f:
        assert f.readline().strip() == b'P5'
        header = f.readline()
        while header.startswith(b'#'):
            header = f.readline()
        width, height = map(int, header.split())
        assert int(f.readline()) == 255
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    # Treat unknown cells as unsafe too, not as free floor. The map origin
    # is the lower-left cell corner; image row zero is the north edge.
    grey = pixels.astype(float) / 255.0
    occupancy = grey if metadata['negate'] else 1.0 - grey
    rows, cols = np.nonzero(occupancy >= float(metadata['free_thresh']))
    resolution = float(metadata['resolution'])
    ox, oy = metadata['origin'][:2]
    xs = ox + (cols + 0.5) * resolution
    ys = oy + (height - rows - 0.5) * resolution
    return xs, ys, resolution


def executor_params(path):
    return yaml.safe_load(path.read_text())['fixed_joint_pick_place']['ros__parameters']


class PickupFallbackRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xs, cls.ys, cls.resolution = load_occupied_boxes()
        cls.params = executor_params(PARAMS)
        cls.profile = executor_params(PROFILE)

    @classmethod
    def clearance(cls, x, y):
        """Euclidean distance to nearest occupied *cell rectangle*, not centre."""
        half = cls.resolution / 2
        dx = np.maximum(np.abs(cls.xs - x) - half, 0.0)
        dy = np.maximum(np.abs(cls.ys - y) - half, 0.0)
        return float(np.hypot(dx, dy).min())

    @classmethod
    def chord_lower_bound(cls, start, end, spacing=0.005):
        length = math.dist(start[:2], end[:2])
        segments = max(1, math.ceil(length / spacing))
        sampled = min(
            cls.clearance(start[0] + (end[0] - start[0]) * i / segments,
                          start[1] + (end[1] - start[1]) * i / segments)
            for i in range(segments + 1)
        )
        # Distance to a fixed obstacle set is 1-Lipschitz. Every point on the
        # chord is <= half a sample interval from one evaluated point.
        return sampled - length / (2 * segments)

    def test_route_points_and_chords_clear_the_walls(self):
        empty_radius = float(self.profile['empty_robot_radius_m'])
        plan_radius = empty_radius + float(self.profile['global_radius_margin_m'])
        radius = float(self.profile.get('pickup_fallback_completion_radius_m', 0.35))
        route = [self.profile[name] for name in POINTS]
        self.assertEqual(len(route), 3)
        for name, point in zip(POINTS, route):
            self.assertEqual(len(point), 3, f'{name} must be x, y, yaw')
            self.assertGreaterEqual(
                self.clearance(*point[:2]), 0.60,
                f'{name} must keep an empty-robot lane past the walls',
            )
        # The last point is where the base stops before re-planning towards the
        # pickup pose, so its acceptance disk must stay off the walls too.
        self.assertGreaterEqual(
            self.clearance(*route[-1][:2]), empty_radius + radius + 0.05
        )
        # Check the nominal starting pose and straight chords as a static
        # reference, not as proof of the live path from a stalled pose.
        waypoints = [self.profile['warehouse_b_pose']] + route
        for start, end in zip(waypoints, waypoints[1:]):
            self.assertGreaterEqual(
                self.chord_lower_bound(start, end), plan_radius + 0.03,
                f'chord {start[:2]} -> {end[:2]} has insufficient wall clearance',
            )
        # The route exists to avoid the mouth; it must not pass near it.
        for name, point in zip(POINTS, route):
            self.assertGreater(
                math.dist(point[:2], MOUTH), 4.0,
                f'{name} is still inside the mouth it was meant to bypass',
            )

    def test_competition_profile_carries_the_route(self):
        # competition_bringup.launch.py loads the profile INSTEAD of
        # config/fixed_joint_pick_place.yaml, so a route only in the base YAML
        # would never run in a scored round.
        self.assertIs(self.profile.get('pickup_fallback_enabled'), True)
        self.assertIs(self.params.get('pickup_fallback_enabled'), True)
        for name in POINTS:
            self.assertEqual(self.profile[name], self.params[name], name)

    def test_executor_declares_the_same_route_and_wires_it(self):
        tree = ast.parse(SCRIPT.read_text())
        documented = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == 'pickup_fallback_defaults'
                for target in node.targets
            ):
                for cube, values in zip(node.value.keys, node.value.values):
                    documented[ast.literal_eval(cube)] = values
        self.assertIn(CUBE, documented)
        points = documented[CUBE].elts
        self.assertEqual(len(points), len(POINTS))
        for point, name in zip(points, POINTS):
            # Yaw may be an expression (math.pi); x and y must be literals that
            # agree with both YAMLs, otherwise the standalone default silently
            # differs from the scored profile.
            self.assertEqual(
                [ast.literal_eval(value) for value in point.elts[:2]],
                self.profile[name][:2], name,
            )
        # The route is useless if the retry ladder never asks for it.
        wiring = [
            node for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef) and fn.name == '_navigate_to'
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_drive_pickup_fallback'
        ]
        self.assertEqual(len(wiring), 1, 'exactly one call site in _navigate_to')


class PickupRecoveryRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_executor()
        cls.executor = cls.module.FixedJointPickPlace

    def test_mover_during_outer_route_does_not_trigger_manual_escape(self):
        for event in ('hold', 'arc'):
            with self.subTest(event=event):
                route = [[-10.6, -3.0, math.pi], [-12.95, -4.45, -0.2]]
                label = 'blue_cube_5 visual handoff'
                params = {'navigation_retry_count': 1, 'settle_sec': 0.0,
                          'navigation_timeout_sec': 30.0,
                          'pickup_fallback_completion_radius_m': 0.35}
                node = SimpleNamespace(
                    _yield_config=SimpleNamespace(enabled=False),
                    _yield_decision=SimpleNamespace(holds=False),
                    _yield_arc_goal=None,
                    _log_cube_positions=Mock(),
                    _pending_yield_arc=Mock(return_value=None),
                    _entry_recovery_applies=Mock(return_value=False),
                    _pickup_fallback_route=Mock(return_value=route),
                    _pickup_fallback_needed=Mock(return_value=True),
                    _relax_global_footprint=Mock(return_value=False),
                    _back_off=Mock(),
                    _clear_costmaps=Mock(return_value=True),
                    _restore_nominal_geometry=Mock(),
                    _clear_speed_limit=Mock(),
                    get_parameter=lambda name: SimpleNamespace(value=params[name]),
                    get_logger=Mock(),
                )

                def navigate(stage, *args, **kwargs):
                    if stage.endswith(' outer route'):
                        node._yield_decision.holds = event == 'hold'
                        node._yield_arc_goal = (1.0, 2.0) if event == 'arc' else None
                    return False

                node._navigate_once = Mock(side_effect=navigate)
                node._drive_pickup_fallback = self.executor._drive_pickup_fallback.__get__(node)
                self.assertFalse(self.executor._navigate_to(node, label, (-2.8, -11.6, 0.0)))
                labels = [call.args[0] for call in node._navigate_once.call_args_list]
                self.assertEqual(labels[:2], [label, f'{label} outer route'])
                node._clear_costmaps.assert_not_called()
                node._back_off.assert_not_called()

    def test_jam_escape_stops_if_scan_is_stale_or_clearance_closes(self):
        for readings, expected_publishes in (([None], 0), ([0.31], 0), ([0.65, 0.31], 1)):
            with self.subTest(readings=readings):
                node = SimpleNamespace(
                    _fine_cmd_pub=Mock(), _stop_base=Mock(),
                    _yield_config=SimpleNamespace(enabled=False),
                    _entry_yield_pending=Mock(return_value=False),
                    _entry_scan_clearance=Mock(side_effect=readings),
                    _recovery_pose=Mock(return_value=(0.0, 0.0, 0.0)),
                    _recovery_corridor_clear=Mock(return_value=True),
                    _nav_budget_remaining=lambda limit=math.inf: min(7.0, limit),
                    get_parameter=lambda name: SimpleNamespace(value={
                        'dead_zone_escape_enabled': True,
                        'entry_recovery_stop_clearance_m': 0.40,
                    }[name]),
                    get_logger=Mock(),
                )
                with patch.object(self.module.rclpy, 'ok', return_value=True), \
                     patch.object(self.module.rclpy, 'spin_once'):
                    self.executor._back_off(node, 'test', direction=-1.0)
                self.assertEqual(node._fine_cmd_pub.publish.call_count, expected_publishes)
                node._stop_base.assert_called()

    def test_relax_rechecks_start_cell_on_new_costmap(self):
        for new_cost, expected in ((80, True), (100, False), (None, False)):
            with self.subTest(new_cost=new_cost):
                node = SimpleNamespace(
                    _global_costmap=SimpleNamespace(cost=99, stamp=100),
                    _relaxed_global=False, _nominal_geometry=(0.35, 0.60, 'loaded'),
                    _recovery_envelope_radius=Mock(return_value=0.43),
                    _local_recovery_pose_clear=Mock(return_value=True),
                    _entry_yield_pending=Mock(return_value=False),
                    _nav_budget_remaining=lambda limit: limit,
                    _update_costmap_geometry=Mock(return_value={'global': {'robot_radius': 0.45}}),
                    _grid_stamp=lambda grid: grid.stamp,
                    get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100)),
                    get_parameter=lambda name: SimpleNamespace(value={
                        'stuck_relax_global_radius': True, 'global_radius_margin_m': 0.10,
                    }[name]),
                    get_logger=Mock(),
                )
                node._global_recovery_cost = lambda: node._global_costmap.cost
                node._restore_nominal_geometry = self.executor._restore_nominal_geometry.__get__(node)
                clock = [0.0]

                def spin(*args, **kwargs):
                    clock[0] += 0.5
                    if new_cost is not None:
                        node._global_costmap = SimpleNamespace(cost=new_cost, stamp=101)

                with patch.object(self.module.rclpy, 'ok', return_value=True), \
                     patch.object(self.module.rclpy, 'spin_once', side_effect=spin), \
                     patch.object(self.module.time, 'monotonic', side_effect=lambda: clock[0]):
                    self.assertEqual(self.executor._relax_global_footprint(node, 'test'), expected)
                self.assertEqual(node._relaxed_global, expected)
                self.assertEqual(node._update_costmap_geometry.call_count, 1 if expected else 2)


if __name__ == '__main__':
    unittest.main()

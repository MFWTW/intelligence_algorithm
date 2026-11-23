"""Exercise the production speed-limit policy without ROS or Gazebo."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import obstacle_yield

SCRIPT = SCRIPTS / 'fixed_joint_pick_place.py'
TREE = ast.parse(SCRIPT.read_text())
EXECUTOR = next(node for node in TREE.body if isinstance(node, ast.ClassDef)
                and node.name == 'FixedJointPickPlace')
METHODS = {'_slow_zone_limit', '_apply_slow_zones', '_publish_speed_limit', '_tick_yield'}
METHOD_TREE = ast.Module(body=[node for node in EXECUTOR.body
                               if isinstance(node, ast.FunctionDef) and node.name in METHODS],
                         type_ignores=[])
STOP = 0.001
NAMESPACE = {
    'math': math,
    'obstacle_yield': obstacle_yield,
    'time': SimpleNamespace(monotonic=lambda: 10.0),
    'YIELD_STOP_SPEED_LIMIT': STOP,
    'SpeedLimit': lambda: SimpleNamespace(header=SimpleNamespace()),
}
exec(compile(METHOD_TREE, str(SCRIPT), 'exec'), NAMESPACE)


class TestNarrowSpeedZone(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.position = (4.0, 0.0)
        self.params = {
            'slow_zone_brake_ahead_m': 0.8,
            'slow_zone_approach_width_m': 0.6,
            'slow_zone_approach_cruise_mps': 1.5,
            'yield_update_period_sec': 0.0,
        }
        self.node = SimpleNamespace(
            _slow_zones=[(0.0, 0.0, 1.0, 0.9)], _active_speed_limit=0.0,
            _speed_limit_pub=SimpleNamespace(publish=self.sent.append),
            _cached_base_position=lambda: self.position,
            get_parameter=lambda name: SimpleNamespace(value=self.params[name]),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: None)),
            get_logger=lambda: SimpleNamespace(warning=lambda *args: None),
        )
        for name in METHODS:
            setattr(self.node, name, NAMESPACE[name].__get__(self.node))

    def update(self, *positions):
        for x in positions:
            self.position = (x, 0.0)
            self.node._apply_slow_zones()
        return [msg.speed_limit for msg in self.sent]

    def test_disabled_by_default_and_invalid_opt_in_retains_old_threshold(self):
        for name in ('slow_zone_approach_width_m', 'slow_zone_approach_cruise_mps'):
            declaration = next(call for call in ast.walk(EXECUTOR)
                               if isinstance(call, ast.Call)
                               and isinstance(call.func, ast.Attribute)
                               and call.func.attr == 'declare_parameter' and call.args
                               and isinstance(call.args[0], ast.Constant)
                               and call.args[0].value == name)
            self.assertEqual(declaration.args[1].value, 0.0)
        for width, cruise in [(0.0, 1.5), (0.6, 0.0), (0.9, 1.5), (0.6, 1.6),
                              (math.nan, 1.5), (0.6, math.inf)]:
            with self.subTest(width=width, cruise=cruise):
                self.params.update(slow_zone_approach_width_m=width,
                                   slow_zone_approach_cruise_mps=cruise)
                self.sent.clear()
                self.assertEqual(self.update(2.2, 1.801, 1.8, 0.0, 1.801), [0.9, 0.0])

    def test_approach_keeps_legacy_prebrake_and_debounces_both_boundaries(self):
        self.assertEqual(self.update(2.51, 2.39, 2.41, 2.39, 1.81, 1.8, 1.81,
                                     1.8, 1.0, 0.0, 1.89, 1.91, 1.89, 2.49,
                                     2.51, 2.49, 2.8),
                         [1.2, 0.9, 1.2, 0.0])
        self.assertTrue(all(msg.percentage is False for msg in self.sent))

    def test_skipped_shoulder_and_pose_loss_do_not_delay_or_release_full_cap(self):
        self.assertEqual(self.update(4.0, 1.79), [0.9])
        for lost in (None, (math.nan, 0.0)):
            self.position = lost
            self.node._apply_slow_zones()
        self.assertEqual(self.update(1.81, 2.6), [0.9, 0.0])
        self.node._slow_zones = []
        self.assertEqual(self.node._slow_zone_limit(), 0.0)

    def test_overlapping_zones_take_the_lowest_and_never_raise_low_cruise(self):
        self.node._slow_zones.append((0.4, 0.0, 1.0, 0.7))
        self.assertEqual(self.update(2.6, 2.1), [1.1, 0.7])
        self.params['slow_zone_approach_cruise_mps'] = 0.7
        self.assertEqual(self.update(2.6), [1.1, 0.7, 0.0])

    def test_yield_stop_and_lower_cap_win_without_becoming_zone_history(self):
        self.position = (0.0, 0.0)
        self.node.__dict__.update(
            _yield_suppress=False, _yield_last_update=0.0,
            _yield_latched=False, _yield_wait_since=None,
            _yield_config=obstacle_yield.YieldConfig(),
            _update_yield_episode=lambda *args: None,
        )
        for action, speed, latched in [(obstacle_yield.ACTION_CAP, 0.4, False),
                                       (obstacle_yield.ACTION_STOP, 0.0, True),
                                       (obstacle_yield.ACTION_CAP, 1.1, False)]:
            decision = obstacle_yield.YieldDecision(action, 'test', speed_mps=speed)
            self.node._yield_step = lambda _: (decision, latched, None)
            self.node._tick_yield()
            self.position = None  # keep the 0.9 zone cap but never a yield hold
        self.assertEqual([msg.speed_limit for msg in self.sent], [0.4, STOP, 0.9])


if __name__ == '__main__':
    unittest.main()

"""Placement duration regression tests without importing ROS or starting a node."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/fixed_joint_pick_place.py'
TREE = ast.parse(SCRIPT.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'FixedJointPickPlace')
METHODS = {'_place_arm_motion_sec', '_move_arm', '_move_arm_placing', '_move_gripper', 'run_once'}
module = ast.Module(body=[ast.ClassDef(name='Subject', bases=[], keywords=[],
    body=[n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name in METHODS],
    decorator_list=[])], type_ignores=[])
namespace = {'math': math, 'ARM_JOINTS': [f'joint{i}' for i in range(1, 7)],
             'GRIPPER_JOINTS': ['finger_joint1']}
exec(compile(ast.fix_missing_locations(module), str(SCRIPT), 'exec'), namespace)


class Fake(namespace['Subject']):
    def __init__(self, duration=0.0, yaw=0.4, enabled=True):
        self.params = {'place_arm_motion_sec': duration, 'arm_motion_sec': 3.0,
                       'gripper_motion_sec': 1.0, 'place_yaw_compensation': enabled,
                       'place_yaw_comp_limit_rad': 0.3, 'cube_name': 'cube',
                       'gripper_open': 0.0, 'gripper_closed': 0.048}
        self._dock_yaw_error = yaw
        self._arm_client, self._gripper_client = object(), object()
        self._cancel_requested = False
        self.calls = []
        self.events = []

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def get_logger(self):
        return SimpleNamespace(info=lambda *args: None, warning=lambda *args: None,
                               error=lambda *args: None)

    def _double_list(self, name, count):
        return [0.1] * count

    def _execute_trajectory(self, label, client, joints, target, seconds):
        self.calls.append((label, client, list(target), seconds))
        return True

    def _wait_for_dependencies(self): return True
    def _warehouse_goal(self): return 'A', []
    def _publish_executor_state(self, state): self.events.append(state)
    def _navigate_to_pickup(self): return True
    def _visual_align(self): return True
    def _attach(self): return True
    def _verify_attachment(self): return True
    def _navigate_to_warehouse(self): return True
    def _detach(self): return True
    def _settle_after_release(self): return True
    def _verify_placement(self, count):
        self.events.append(('verify', count))
        return True


class PlaceDurationTests(unittest.TestCase):
    def test_declared_default(self):
        declarations = [n for n in ast.walk(CLASS) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'declare_parameter' and n.args
                        and isinstance(n.args[0], ast.Constant)
                        and n.args[0].value == 'place_arm_motion_sec']
        self.assertEqual(len(declarations), 1)
        self.assertEqual(ast.literal_eval(declarations[0].args[1]), 0.0)

    def test_default_and_invalid_values_fall_back(self):
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf, None, 'bad', []):
            with self.subTest(value=value):
                node = Fake(value)
                self.assertTrue(node.run_once())
                self.assertTrue(all(c[3] == 3.0 for c in node.calls
                                    if c[1] is node._arm_client))

    def test_only_four_arm_motions_and_compensation(self):
        node = Fake(0.5)
        self.assertTrue(node.run_once())
        arm = [c for c in node.calls if c[1] is node._arm_client]
        expected = {'Move to pre-place', 'Lower to place', 'Retreat from place', 'Return home'}
        self.assertEqual({c[0] for c in arm if c[3] == 0.5}, expected)
        self.assertEqual(sum(c[3] == 0.5 for c in arm), 4)
        self.assertTrue(all(c[3] == 3.0 for c in arm if c[0] not in expected))
        self.assertTrue(all(c[3] == 1.0 for c in node.calls if c[1] is node._gripper_client))
        for label, _, target, _ in arm:
            self.assertAlmostEqual(target[0], 0.4 if label in
                                   {'Move to pre-place', 'Lower to place'} else 0.1)
        self.assertEqual([e for e in node.events if isinstance(e, tuple)],
                         [('verify', True), ('verify', False)])
        self.assertIn('VERIFY_PLACE', node.events)

    def test_compensation_disabled_and_negative_clamp(self):
        for yaw, enabled, expected in [(-0.8, True, -0.2), (0.4, False, 0.1),
                                       (0.2, True, 0.3)]:
            with self.subTest(yaw=yaw, enabled=enabled):
                node = Fake(0.5, yaw, enabled)
                node._move_arm_placing('test', 'pre_place_joints')
                self.assertAlmostEqual(node.calls[0][2][0], expected)

    def test_other_exceptions_propagate(self):
        node = Fake(0.5)
        def fail(*args):
            raise RuntimeError('trajectory failure')
        node._execute_trajectory = fail
        with self.assertRaisesRegex(RuntimeError, 'trajectory failure'):
            node._move_arm_placing('test', 'place_joints')
        node.params['place_arm_motion_sec'] = 0.0
        node.params['arm_motion_sec'] = 'invalid'
        with self.assertRaises(ValueError):
            node._place_arm_motion_sec()


if __name__ == '__main__':
    unittest.main()

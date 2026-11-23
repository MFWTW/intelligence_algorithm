"""Pure and CLI-level regressions; no Gazebo or control interfaces required."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_final_placement.py'
spec = importlib.util.spec_from_file_location('verify_final_placement', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

PARAMS = '''fixed_joint_pick_place:
  ros__parameters:
    cube_link: link
    zone_a_centre: [1.0, 2.0]
    zone_b_centre: [0.0, 0.0]
    zone_half_length_m: 0.5
    zone_half_width_m: 0.25
    placement_tolerance_m: 0.1
'''


def pose(x, y):
    return types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y))


class FinalPlacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'params.yaml'
        self.path.write_text(PARAMS)
        self.items = v.parse_items(['one=A', 'two=B'], 2)
        self.config = v.load_config(self.path, self.items)

    def test_inclusive_limits_and_outside(self):
        # Use exact binary values to test the inclusive boundary.
        config = ('link', .5, .25, {'a': [0., 0.], 'b': [0., 0.]})
        code, rows = v.evaluate(self.items, config, ['one::link', 'two::link'],
                                [pose(.5, -.25), pose(.500001, 0)])
        self.assertEqual(code, 1)
        self.assertEqual([r['status'] for r in rows], ['inside', 'outside'])
        self.assertEqual(v.evaluate(self.items, config, ['one::link', 'two::link'],
                                    [pose(.5, -.25), pose(0, 0)])[0], 0)

    def test_missing_duplicate_mismatch_and_nonfinite(self):
        for names, poses in [(['one::link', 'two::link'], [pose(1, 2)]),
                             (['one::link', 'one::link', 'two::link'], [pose(1, 2)] * 3),
                             (['one::link', 'other'], [pose(1, 2), pose(0, 0)]),
                             (['one::link', 'two::link'], [pose(math.nan, 2), pose(0, 0)]),
                             (['one::link', 'two::link'], [pose(1, 2), pose(math.inf, 0)])]:
            with self.subTest(names=names, poses=poses):
                code, rows = v.evaluate(self.items, self.config, names, poses)
                self.assertEqual(code, 2)
                self.assertIn('unknown', [r['status'] for r in rows])
        code, rows = v.evaluate(self.items, self.config, ['one::link', 'two::link'],
                                [pose(10, 2), pose(math.nan, 0)])
        self.assertEqual(code, 2)  # unknown dominates outside
        self.assertEqual(rows[0]['status'], 'outside')

    def test_input_validation(self):
        for items, count in [(['one=A', 'one=B'], 2), (['one=A'], 2),
                             ([], 0), (['one::link=A'], 1), (['one='], 1)]:
            with self.subTest(items=items), self.assertRaises(v.InputError):
                v.parse_items(items, count)
        for text in [PARAMS.replace('cube_link: link', 'cube_link: link\n    cube_link: other'),
                     PARAMS.replace('zone_a_centre: [1.0, 2.0]', 'zone_a_centre: [.nan, 2.0]'),
                     PARAMS.replace('placement_tolerance_m: 0.1', 'placement_tolerance_m: 0.6'),
                     PARAMS.replace('zone_b_centre: [0.0, 0.0]', 'zone_b_centre: [0.0]'),
                     PARAMS.replace('cube_link: link', 'cube_link: null')]:
            self.path.write_text(text)
            with self.subTest(text=text), self.assertRaises(v.InputError):
                v.load_config(self.path, self.items)

    def invoke(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = v.main(argv)
        return code, json.loads(stream.getvalue())

    def test_cli_one_frame_and_timeout(self):
        args = ['--params', str(self.path), '--expected-count', '2',
                '--item', 'one=A', '--item', 'two=B']
        frame = types.SimpleNamespace(name=['one::link', 'two::link'],
                                      pose=[pose(1, 2), pose(0, 0)])
        with mock.patch.object(v, 'receive_frame', return_value=frame) as receive:
            code, out = self.invoke(args)
        self.assertEqual(code, 0)
        self.assertEqual(out['status'], 'pass')
        self.assertEqual(len(out['items']), 2)
        receive.assert_called_once_with(5.)
        with mock.patch.object(v, 'receive_frame', side_effect=v.InputError('timed out')):
            code, out = self.invoke(args)
        self.assertEqual(code, 2)
        self.assertEqual([r['status'] for r in out['items']], ['unknown', 'unknown'])
        self.assertIn('timed out', out['error'])
        self.assertEqual(self.invoke(['--invalid'])[0], 2)
        self.assertEqual(self.invoke(args + ['--timeout', 'nan'])[0], 2)

    def test_ros_wait_timeout_destroys_node(self):
        # Exercise the real wait loop with a fake ROS node and monotonic clock.
        try:
            import rclpy
            import gazebo_msgs.msg
        except ImportError:
            self.skipTest('ROS Python dependencies are unavailable')
        with mock.patch('rclpy.init'), mock.patch('rclpy.shutdown') as shutdown, \
             mock.patch('rclpy.ok', return_value=True), \
             mock.patch('rclpy.node.Node') as node, \
             mock.patch('rclpy.spin_once') as spin, \
             mock.patch.object(v.time, 'monotonic', side_effect=[10, 10, 16]):
            with self.assertRaisesRegex(v.InputError, 'timed out'):
                v.receive_frame(5)
            spin.assert_called_once()
            node.return_value.destroy_node.assert_called_once()
            shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()

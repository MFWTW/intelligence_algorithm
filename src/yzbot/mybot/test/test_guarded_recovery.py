"""Navigation recovery checks without starting Nav2 or loading ROS messages."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / 'scripts' / 'fixed_joint_pick_place.py'
METHODS = {
    '_update_costmap_geometry', '_restore_nominal_geometry', '_grid_sweep_clear',
    '_recovery_corridor_clear', '_back_off', '_relax_global_footprint', '_fresh_recovery_grid',
}


class GuardedRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.node = NS(
            _geometry_fault=False, _relaxed_global=False,
            _nav_budget_remaining=lambda limit=math.inf: min(10 - self.now, limit),
            _yield_config=NS(enabled=False), _entry_yield_pending=Mock(return_value=False),
            _fine_cmd_pub=Mock(), _stop_base=Mock(), get_logger=Mock(),
            _recovery_envelope_radius=lambda: 0.43,
        )
        params = {'service_timeout_sec': 2.0, 'dead_zone_escape_enabled': True,
                  'entry_recovery_stop_clearance_m': 0.24,
                  'stuck_relax_global_radius': True, 'global_radius_margin_m': 0.10}
        self.node.get_parameter = lambda name: NS(value=params[name])
        cls = next(n for n in ast.parse(EXECUTOR.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'FixedJointPickPlace')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
        env = {'time': NS(monotonic=lambda: self.now), 'math': math,
               'rclpy': NS(ok=lambda: True, spin_once=self.spin),
               'Twist': lambda: NS(linear=NS(x=0.0))}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(EXECUTOR), 'exec'), env)
        for name in METHODS:
            setattr(self.node, name, env[name] if name == '_grid_sweep_clear'
                    else env[name].__get__(self.node))

    def spin(self, node, timeout_sec):
        self.now += timeout_sec

    def test_second_node_failure_rolls_back_both(self):
        n = self.node
        n._read_map_geometry = Mock(side_effect=[{'r': 0.42}, {'r': 0.32}])
        n._write_map_geometry = Mock(side_effect=[True, False, True, True])
        self.assertIsNone(n._update_costmap_geometry({'global': {'r': 0.45}, 'local': {'r': 0.35}}))
        self.assertEqual([c.args[:2] for c in n._write_map_geometry.call_args_list], [
            ('global', {'r': 0.45}), ('local', {'r': 0.35}),
            ('local', {'r': 0.32}), ('global', {'r': 0.42})])
        self.assertFalse(n._geometry_fault)

    def test_failed_rollback_latches_and_blocks_next_update(self):
        n = self.node
        n._read_map_geometry = Mock(return_value={'r': 0.42})
        n._write_map_geometry = Mock(side_effect=[False, False])
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'GEOMETRY_STATE_UNKNOWN'):
                n._update_costmap_geometry({'global': {'r': 0.45}})
        self.assertEqual(n._write_map_geometry.call_count, 2)

    def test_failed_restore_retains_state_and_latches(self):
        n = self.node
        n._relaxed_global = True
        n._global_radius_before_relax = 0.45
        n._update_costmap_geometry = Mock(return_value=None)
        with self.assertRaisesRegex(RuntimeError, 'GEOMETRY_STATE_UNKNOWN'):
            n._restore_nominal_geometry('wall')
        n._update_costmap_geometry.assert_called_once_with(
            {'global': {'robot_radius': 0.45}}, cleanup=True)
        self.assertTrue(n._relaxed_global)
        self.assertTrue(n._geometry_fault)

    def grid(self):
        return NS(header=NS(frame_id='odom'), data=[0] * 1600,
                  info=NS(width=40, height=40, resolution=0.05,
                          origin=NS(position=NS(x=-1.0, y=-1.0),
                                    orientation=NS(x=0, y=0, z=0, w=1))))

    def test_inscribed_cell_is_not_physical_obstacle_but_lethal_and_unknown_are(self):
        grid = self.grid()
        idx = 20 * 40 + 9  # x=-.525; outside current disk but in reverse sweep.
        for cost, safe in [(99, True), (100, False), (-1, False)]:
            grid.data[idx] = cost
            self.assertEqual(self.node._grid_sweep_clear(grid, (0, 0, 0), 0.43, -0.15), safe)
            self.assertTrue(self.node._grid_sweep_clear(grid, (0, 0, 0), 0.43, +0.15))

    def test_missing_local_grid_or_outside_known_coverage_blocks_corridor(self):
        n = self.node
        n._fresh_recovery_grid = Mock(return_value=None)
        n._recovery_pose = Mock(return_value=(0.60, 0, 0))
        self.assertFalse(n._recovery_corridor_clear(-1, 0.10))
        n._fresh_recovery_grid.return_value = self.grid()
        self.assertFalse(n._recovery_corridor_clear(-1, 0.10))

    def test_wall_stale_local_grid_is_rejected_even_while_sim_paused(self):
        n = self.node
        n._local_costmap = self.grid()
        n._local_costmap_received_at = 0.0
        n._recovery_stamp_fresh = Mock(return_value=True)
        self.assertIsNotNone(n._fresh_recovery_grid('local'))
        self.now = 5.0
        self.assertIsNone(n._fresh_recovery_grid('local'))

    def test_global_relax_refuses_lethal_unknown_or_local_collision(self):
        n = self.node
        n._nominal_geometry = (0.35, 0.60, 'loaded')
        n._global_recovery_cost = Mock()
        n._local_recovery_pose_clear = Mock()
        n._update_costmap_geometry = Mock()
        for cost, clear in [(100, True), (None, True), (0, True), (99, False)]:
            n._global_recovery_cost.return_value = cost
            n._local_recovery_pose_clear.return_value = clear
            self.assertFalse(n._relax_global_footprint('wall'))
        n._update_costmap_geometry.assert_not_called()

    def test_global_relax_waits_for_new_grid_then_restores_only_global_radius(self):
        n = self.node
        n._nominal_geometry = (0.35, 0.60, 'loaded')
        n._global_costmap = self.grid()
        n._global_recovery_cost = Mock(side_effect=[99, 98])
        n._local_recovery_pose_clear = Mock(return_value=True)
        n._update_costmap_geometry = Mock(return_value={'global': {'robot_radius': 0.45}})
        n._grid_stamp = lambda _: 101 if self.now >= 0.05 else 99
        n.get_clock = lambda: NS(now=lambda: NS(nanoseconds=100))
        self.assertTrue(n._relax_global_footprint('wall'))
        self.assertGreaterEqual(self.now, 0.05)
        self.assertTrue(n._relaxed_global)
        self.assertFalse(n._relax_global_footprint('wall'))
        desired = n._update_costmap_geometry.call_args.args[0]
        self.assertEqual(desired, {'global': {'robot_radius': 0.43}})
        n._restore_nominal_geometry('wall')
        self.assertAlmostEqual(n._update_costmap_geometry.call_args.args[0]['global']['robot_radius'],
                               0.45)
        self.assertFalse(n._relaxed_global)

    def test_retreat_never_sends_motion_without_clear_sweep(self):
        n = self.node
        n._recovery_pose = Mock(return_value=(0.0, 0.0, 0.0))
        n._recovery_corridor_clear = Mock(return_value=False)
        n._entry_scan_clearance = Mock(return_value=2.0)
        self.assertFalse(n._back_off('wall'))
        n._fine_cmd_pub.publish.assert_not_called()
        n._stop_base.assert_called_once()

    def test_retreat_has_short_no_progress_limit(self):
        n = self.node
        n._recovery_pose = Mock(return_value=(0.0, 0.0, 0.0))
        n._entry_scan_clearance = Mock(return_value=2.0)
        n._recovery_corridor_clear = Mock(return_value=True)
        self.assertFalse(n._back_off('wall'))
        self.assertGreater(n._fine_cmd_pub.publish.call_count, 0)
        self.assertLessEqual(self.now, 3.05)
        n._stop_base.assert_called_once()

    def test_guarded_trees_have_no_independent_recovery(self):
        for name in ('nav_to_pose_entry_guard.xml', 'nav_through_poses_guarded.xml'):
            with self.subTest(tree=name):
                tags = {node.tag for node in ET.parse(ROOT / 'config' / name).iter()}
                self.assertIn('FollowPath', tags)
                self.assertTrue({'ComputePathToPose', 'ComputePathThroughPoses'} & tags)
                self.assertFalse({'Spin', 'BackUp', 'ClearEntireCostmap', 'RecoveryNode'} & tags)


if __name__ == '__main__':
    unittest.main()

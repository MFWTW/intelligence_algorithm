"""Opt-in retained-goal yield: gate and launch fail-closed invariants."""

import importlib.util
import math
from pathlib import Path
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from xml.etree import ElementTree

from geometry_msgs.msg import Twist

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts'
sys.path.insert(0, str(SRC))
from yield_velocity_gate import VelocityGateState
from yield_nav_profile import guarded_profile, guarded_executor_profile


def load(path):
    spec = importlib.util.spec_from_file_location('guarded_nav_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VelocityGateTests(unittest.TestCase):
    def test_both_axes_are_zero_during_hold_and_after_release_until_fresh_input(self):
        gate = VelocityGateState(stale_sec=0.3, release_debounce_sec=0.15)
        command = Twist()
        command.linear.x = 0.8
        command.angular.z = 1.5
        gate.input(command, 100.0)
        self.assertEqual((gate.output(100.1).linear.x,
                          gate.output(100.1).angular.z), (0.8, 1.5))
        gate.hold(True, 100.2)
        gate.input(command, 100.2)
        self.assertEqual((gate.output(100.2).linear.x,
                          gate.output(100.2).angular.z), (0.0, 0.0))
        gate.hold(False, 100.25)
        self.assertTrue(gate.held, 'a single release must not unlatch')
        gate.hold(False, 100.30)
        self.assertTrue(gate.held, 'a 50 ms flap must not unlatch')
        gate.hold(False, 100.45)
        self.assertFalse(gate.held)
        self.assertEqual((gate.output(100.46).linear.x,
                          gate.output(100.46).angular.z), (0.0, 0.0))
        gate.input(command, 100.5)
        self.assertEqual((gate.output(100.55).linear.x,
                          gate.output(100.55).angular.z), (0.8, 1.5))
        self.assertEqual((gate.output(100.9).linear.x,
                          gate.output(100.9).angular.z), (0.0, 0.0))

    def test_a_hold_engages_on_the_first_message(self):
        gate = VelocityGateState()
        command = Twist()
        command.linear.x = 1.0
        gate.input(command, 1.0)
        gate.hold(True, 1.01)
        self.assertTrue(gate.held)
        self.assertEqual(gate.output(1.02).linear.x, 0.0)

    def test_nonfinite_command_never_reaches_smoother(self):
        gate = VelocityGateState()
        bad = Twist()
        bad.angular.z = math.nan
        gate.input(bad, 1.0)
        self.assertIsNone(gate.last_input)
        self.assertEqual(gate.output(1.1).angular.z, 0.0)


class RetainedGoalTests(unittest.TestCase):
    def test_a_stale_hold_is_cleared_instead_of_failing_the_leg(self):
        from test_navigation_control import module as executor
        calls = []
        node = SimpleNamespace(
            _retain_nav_goal_on_yield=True,
            _yield_gate_status_time=time.monotonic() - 9.0,
            _yield_gate_held=True,
            _command_yield_gate=lambda held, timeout=0.5: calls.append(
                (held, timeout)) or True,
        )
        self.assertTrue(executor.FixedJointPickPlace._ensure_yield_gate_clear(node))
        self.assertEqual(calls, [(False, 1.5)])

    def test_a_fresh_released_gate_needs_no_round_trip(self):
        from test_navigation_control import module as executor
        node = SimpleNamespace(
            _retain_nav_goal_on_yield=True,
            _yield_gate_status_time=time.monotonic(),
            _yield_gate_held=False,
            _command_yield_gate=lambda *a, **k: self.fail('published on a clear gate'),
        )
        self.assertTrue(executor.FixedJointPickPlace._ensure_yield_gate_clear(node))

    def test_a_dead_gate_still_fails_closed(self):
        from test_navigation_control import module as executor
        node = SimpleNamespace(
            _retain_nav_goal_on_yield=True,
            _yield_gate_status_time=0.0,
            _yield_gate_held=True,
            _command_yield_gate=lambda *a, **k: False,
        )
        self.assertFalse(executor.FixedJointPickPlace._ensure_yield_gate_clear(node))
        plain = SimpleNamespace(_retain_nav_goal_on_yield=False)
        self.assertTrue(executor.FixedJointPickPlace._ensure_yield_gate_clear(plain))

    def test_clear_hold_preserves_the_existing_action_handle(self):
        from test_navigation_control import module as executor
        gate_calls = []
        publishes = []
        state = SimpleNamespace(holds=True)
        node = SimpleNamespace(
            _command_yield_gate=lambda value, timeout=0.5: gate_calls.append(value) or True,
            _yield_decision=state,
            _yield_arc_goal=None,
            _yield_pause_active=False,
            _yield_hold_pub=SimpleNamespace(publish=publishes.append),
            _cached_planar_distance_to_point=lambda *args: 2.0,
            _tick_yield=lambda *args: setattr(state, 'holds', False),
            _release_nav_control=lambda *args: self.fail('normal hold cancelled goal'),
            get_logger=lambda: SimpleNamespace(warning=lambda *a: None),
        )
        future = SimpleNamespace(done=lambda: False)
        with mock.patch.object(executor.rclpy, 'ok', return_value=True), \
             mock.patch.object(executor.rclpy, 'spin_once'):
            self.assertTrue(executor.FixedJointPickPlace._hold_until_clear_retained(
                node, 'detour', 1.0, 2.0, time.monotonic() + 2.0,
                object(), future, 5.0))
        self.assertEqual(gate_calls, [True, False])
        self.assertTrue(publishes and publishes[0].data)
        self.assertFalse(node._yield_pause_active)

    def test_arc_cancels_before_releasing_dual_axis_gate(self):
        from test_navigation_control import module as executor
        sequence = []
        state = SimpleNamespace(holds=True)
        node = SimpleNamespace(
            _command_yield_gate=lambda value, timeout=0.5: sequence.append(('gate', value)) or True,
            _yield_decision=state, _yield_arc_goal=(1.0, 2.0),
            _yield_pause_active=False,
            _yield_hold_pub=SimpleNamespace(publish=lambda *args: None),
            _cached_planar_distance_to_point=lambda *args: 2.0,
            _tick_yield=lambda *args: setattr(state, 'holds', False),
            _release_nav_control=lambda *args: sequence.append(('cancel', None)),
            get_logger=lambda: SimpleNamespace(warning=lambda *a: None),
        )
        with mock.patch.object(executor.rclpy, 'ok', return_value=True), \
             mock.patch.object(executor.rclpy, 'spin_once'):
            self.assertFalse(executor.FixedJointPickPlace._hold_until_clear_retained(
                node, 'detour', 1.0, 2.0, time.monotonic() + 2.0,
                object(), SimpleNamespace(done=lambda: False), 5.0))
        self.assertEqual(sequence, [('gate', True), ('cancel', None),
                                    ('gate', False)])


class GuardedLaunchTests(unittest.TestCase):
    def test_stock_nav_routes_behavior_and_smoother_through_gate(self):
        gate_source = (SRC / 'yield_velocity_gate.py').read_text(encoding='utf-8')
        self.assertIn("'input_topic', '/cmd_vel_nav'", gate_source)
        self.assertIn("'output_topic', '/cmd_vel_nav_guarded'", gate_source)
        self.assertIn("'release_debounce_sec'", gate_source)
        # Both stages latch on the same hold, and the actuator stage is the only
        # writer of /cmd_vel, so nothing can drive past a hold.
        self.assertIn("self.create_publisher(Twist, '/cmd_vel', 10)", gate_source)
        self.assertIn("'/cmd_vel_smoothed_raw'", gate_source)
        self.assertIn('self.drive_state.hold(state.data, now)', gate_source)
        module = load(ROOT.parent / 'bot_navigation/launch/navigation_yield_pause.launch.py')
        seen = []
        make = module.guarded_nodes(lambda **kwargs: kwargs, seen)
        behavior = make(package='nav2_behaviors', executable='behavior_server',
                        remappings=[('/tf', 'tf')])
        smoother = make(package='nav2_velocity_smoother',
                        executable='velocity_smoother',
                        remappings=[('cmd_vel', 'cmd_vel_nav'),
                                    ('cmd_vel_smoothed', 'cmd_vel')])
        self.assertEqual(sorted(seen), ['behavior', 'smoother'])
        self.assertIn(('cmd_vel', 'cmd_vel_nav'), behavior['remappings'])
        self.assertIn(('cmd_vel', 'cmd_vel_nav_guarded'), smoother['remappings'])
        # The smoother's own output must pass the drive-side gate too: sending
        # it straight to /cmd_vel left the base sliding ~0.6 s per hold because
        # only the deceleration limit was acting.
        self.assertIn(('cmd_vel_smoothed', 'cmd_vel_smoothed_raw'),
                      smoother['remappings'])
        self.assertNotIn(('cmd_vel_smoothed', 'cmd_vel'), smoother['remappings'])
        self.assertNotIn(('cmd_vel', 'cmd_vel_nav'), smoother['remappings'])
        with self.assertRaisesRegex(RuntimeError, 'Unexpected smoother input'):
            make(package='nav2_velocity_smoother', executable='velocity_smoother',
                 remappings=[])
        with self.assertRaisesRegex(RuntimeError, 'Unexpected smoother output'):
            make(package='nav2_velocity_smoother', executable='velocity_smoother',
                 remappings=[('cmd_vel', 'cmd_vel_nav')])

    def test_bt_plugin_exports_nav2_registration_symbol(self):
        import ctypes
        from ament_index_python.packages import get_package_prefix
        library = (Path(get_package_prefix('mybot')) / 'lib'
                   / 'libmybot_yield_pause_bt.so')
        self.assertTrue(library.is_file())
        plugin = ctypes.CDLL(str(library))
        self.assertTrue(hasattr(plugin, 'BT_RegisterNodesFromPlugin'))

    def test_both_action_trees_halt_the_entire_recovery_subtree(self):
        for filename in ('nav_to_pose_yield_pause.xml',
                         'nav_through_poses_yield_pause.xml'):
            with self.subTest(filename=filename):
                root = ElementTree.parse(ROOT / 'config' / filename).getroot()
                tree = root.find('BehaviorTree')
                self.assertEqual(len(tree), 1)
                self.assertEqual(tree[0].tag, 'YieldPause')
                self.assertEqual(tree[0][0].tag, 'RecoveryNode')
                self.assertIsNotNone(tree[0].find('.//FollowPath'))
                self.assertIsNotNone(tree[0].find('.//BackUp'))

    def test_guarded_drive_has_a_bounded_two_axis_command_deadman(self):
        source = (ROOT / 'src/safe_diff_drive.cpp').read_text(encoding='utf-8')
        self.assertIn('age > command_timeout_sec_', source)
        self.assertIn('target_x_ = 0.0;', source)
        self.assertIn('target_rot_ = 0.0;', source)
        self.assertIn('last_command_time_ = model_->GetWorld()->SimTime();', source)
        self.assertIn('command_timeout_sec_ = _sdf->Get<double>("command_timeout", 0.5)',
                      source)
        # The stock Gazebo plugin is untouched; the guarded one is opt-in only.
        launch = (ROOT / 'launch/gazebo_world2.launch.py').read_text()
        self.assertIn("if safe_drive:", launch)
        self.assertIn('libmybot_safe_diff_drive.so', launch)

    def test_opt_in_profiles_do_not_modify_production_yaml(self):
        import yaml
        nav = ROOT / 'config/profiles/nav_rpp15.yaml'
        executor = ROOT / 'config/profiles/executor_fast.yaml'
        raw_nav = nav.read_bytes()
        raw_executor = executor.read_bytes()
        nav_params = yaml.safe_load(Path(guarded_profile(nav)).read_text())
        exe_params = yaml.safe_load(Path(guarded_executor_profile(executor)).read_text())
        plugins = nav_params['bt_navigator']['ros__parameters']['plugin_lib_names']
        self.assertEqual(plugins.count('mybot_yield_pause_bt'), 1)
        self.assertIn('nav2_follow_path_action_bt_node', plugins)
        self.assertTrue(exe_params['fixed_joint_pick_place']['ros__parameters']
                        ['retain_nav_goal_on_yield'])
        self.assertEqual(nav.read_bytes(), raw_nav)
        self.assertEqual(executor.read_bytes(), raw_executor)
        self.assertNotIn('retain_nav_goal_on_yield',
                         yaml.safe_load(raw_executor)['fixed_joint_pick_place']
                         ['ros__parameters'])


if __name__ == '__main__':
    unittest.main()

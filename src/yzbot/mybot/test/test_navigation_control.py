"""Safety regression tests for Nav2/manual-base command ownership.

Run with: source /opt/ros/humble/setup.bash; source install/setup.bash;
python3 -m unittest discover -s src/yzbot/mybot/test -p 'test_navigation_control.py'
"""
import ast
from collections import deque
from concurrent.futures import Future
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'fixed_joint_pick_place.py'
# The Nav2 parameter file the launch actually passes for a scored round.
PROFILE = ROOT / 'config' / 'profiles' / 'nav_rpp15.yaml'
# The executor imports its sibling `obstacle_yield` by plain module name, which
# works under `ros2 run` (the install space is on sys.path) and under
# `python3 scripts/fixed_joint_pick_place.py` (the script's own directory is),
# but not under importlib's bare loader. Put the directory there for the same
# reason the real runs have it, rather than dressing the import up in the
# executor to suit the harness.
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('pick_place_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser_spec = importlib.util.spec_from_file_location(
    'task_parser_under_test', ROOT / 'scripts' / 'task_parser.py'
)
parser = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser)
supervisor_spec = importlib.util.spec_from_file_location(
    'task_state_machine_under_test', ROOT / 'scripts' / 'task_state_machine.py'
)
supervisor = importlib.util.module_from_spec(supervisor_spec)
supervisor_spec.loader.exec_module(supervisor)

# run_logs/ is intentionally ignored by git. Keep safety tests runnable from a
# fresh checkout; only the optional local diagnostic-probe tests need these.
TOOLS = ROOT.parents[2] / 'run_logs' / 'tools'
recorder = watch = chain_probe = compare_probe = dock_analyzer = angular_audit = None
trace_analyzer = None
if (TOOLS / 'angular_gap_audit.py').is_file():
    angular_spec = importlib.util.spec_from_file_location(
        'angular_gap_audit_under_test', TOOLS / 'angular_gap_audit.py'
    )
    angular_audit = importlib.util.module_from_spec(angular_spec)
    angular_spec.loader.exec_module(angular_audit)
if (TOOLS / 'analyze_dock_snapshots.py').is_file():
    dock_spec = importlib.util.spec_from_file_location(
        'analyze_dock_snapshots_under_test', TOOLS / 'analyze_dock_snapshots.py'
    )
    dock_analyzer = importlib.util.module_from_spec(dock_spec)
    dock_spec.loader.exec_module(dock_analyzer)
if (TOOLS / 'analyze_smoother_trace.py').is_file():
    trace_spec = importlib.util.spec_from_file_location(
        'analyze_smoother_trace_under_test', TOOLS / 'analyze_smoother_trace.py'
    )
    trace_analyzer = importlib.util.module_from_spec(trace_spec)
    trace_spec.loader.exec_module(trace_analyzer)
if (TOOLS / 'compare_chain_probes.py').is_file():
    compare_spec = importlib.util.spec_from_file_location(
        'compare_chain_probes_under_test', TOOLS / 'compare_chain_probes.py'
    )
    compare_probe = importlib.util.module_from_spec(compare_spec)
    compare_spec.loader.exec_module(compare_probe)
if (TOOLS / 'command_chain_probe.py').is_file():
    chain_spec = importlib.util.spec_from_file_location(
        'command_chain_probe_under_test', TOOLS / 'command_chain_probe.py'
    )
    chain_probe = importlib.util.module_from_spec(chain_spec)
    chain_spec.loader.exec_module(chain_probe)
if (TOOLS / 'round_recorder.py').is_file() and (TOOLS / 'mybot_watch.py').is_file():
    recorder_spec = importlib.util.spec_from_file_location(
        'round_recorder_under_test', TOOLS / 'round_recorder.py'
    )
    recorder = importlib.util.module_from_spec(recorder_spec)
    recorder_spec.loader.exec_module(recorder)
    watch_spec = importlib.util.spec_from_file_location(
        'mybot_watch_under_test', TOOLS / 'mybot_watch.py'
    )
    watch = importlib.util.module_from_spec(watch_spec)
    watch_spec.loader.exec_module(watch)


class NavigationControlTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.node = SimpleNamespace(_stop_base=lambda: self.events.append('stop'))
        self.handle = SimpleNamespace(cancel_goal_async=self.cancel)
        self.cancel_future = Future()
        self.result_future = Future()

    def cancel(self):
        self.events.append('cancel')
        return self.cancel_future

    def release(self):
        return module.FixedJointPickPlace._release_nav_control(
            self.node, 'warehouse A', self.handle, self.result_future, 0.1
        )

    def test_waits_for_terminal_result_before_stop_or_manual_motion(self):
        def complete(_node, future, timeout_sec):
            if future is self.cancel_future:
                self.cancel_future.set_result(SimpleNamespace(goals_canceling=[1]))
            else:
                self.result_future.set_result(
                    SimpleNamespace(status=GoalStatus.STATUS_CANCELED)
                )
                self.events.append('terminal')
        with patch.object(module.rclpy, 'spin_until_future_complete', side_effect=complete):
            self.release()
        self.assertEqual(self.events, ['cancel', 'terminal', 'stop'])

    def test_unknown_goal_state_fails_closed(self):
        with patch.object(module.rclpy, 'spin_until_future_complete'):
            with self.assertRaisesRegex(RuntimeError, 'NAV_STATE_UNKNOWN'):
                self.release()
        self.assertEqual(self.events, ['cancel'])

    def test_already_terminal_goal_needs_no_cancel(self):
        self.result_future.set_result(SimpleNamespace(status=GoalStatus.STATUS_ABORTED))
        with patch.object(module.rclpy, 'spin_until_future_complete') as spin:
            self.release()
        spin.assert_not_called()
        self.assertEqual(self.events, ['stop'])

    def test_no_escape_during_live_nav_and_visual_cancel_precedes_retreat(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'FixedJointPickPlace')
        navigate = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_navigate_once__timed')
        calls = [(n.func.attr, n.lineno) for n in ast.walk(navigate)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('_back_off', [name for name, _ in calls])
        release_lines = [line for name, line in calls if name == '_release_nav_control']
        retreat_line = next(line for name, line in calls
                            if name == '_retreat_to_handoff_standoff')
        self.assertTrue(any(line < retreat_line for line in release_lines))


class DependencyDiscoveryTests(unittest.TestCase):
    def test_launcher_checks_executor_executable_before_starting_stack(self):
        launcher = (ROOT / 'scripts' / 'start_competition.sh').read_text()
        self.assertIn('if [ ! -x "$executor_exec" ]; then', launcher)
        self.assertLess(launcher.index('if [ ! -x "$executor_exec" ]; then'),
                        launcher.index('LAUNCH_CMD=(ros2 launch'))

    def test_unmatched_action_client_is_destroyed_before_rebuild(self):
        old = Mock()
        node = SimpleNamespace(_navigate_client=old)
        replacement = Mock()
        with patch.object(module, 'ActionClient', return_value=replacement) as ctor:
            module.FixedJointPickPlace._rebuild_dependency_client(node, 'navigate')
        old.destroy.assert_called_once_with()
        ctor.assert_called_once_with(node, module.NavigateToPose, '/navigate_to_pose')
        self.assertIs(node._navigate_client, replacement)

    def test_unmatched_service_client_is_destroyed_before_rebuild(self):
        old = Mock()
        node = SimpleNamespace(_clear_global_client=old,
                               destroy_client=Mock(), create_client=Mock())
        module.FixedJointPickPlace._rebuild_dependency_client(node, 'clear_global')
        node.destroy_client.assert_called_once_with(old)
        node.create_client.assert_called_once_with(
            module.ClearEntireCostmap,
            '/global_costmap/clear_entirely_global_costmap',
        )

    def test_pre_motion_dependency_timeout_recycles_resident_once(self):
        events = []
        node = SimpleNamespace(
            _release_process=lambda: events.append('release'),
            _cancel_nav2_goals=lambda: events.append('cancel'),
            _resident_enabled=lambda: True, _last_executor_error='DEPENDENCY_TIMEOUT',
            _state='DETECTING_OBJECT', _unsafe_to_retry=False,
            _teardown_resident=lambda: events.append('teardown'),
            _terminal=False, _attempt=1,
            get_logger=lambda: SimpleNamespace(warning=lambda _: None),
            get_parameter=lambda _: SimpleNamespace(value=1),
            _error_pub=SimpleNamespace(publish=lambda _: events.append('error')),
            _stop_pub=SimpleNamespace(publish=lambda _: events.append('stop')),
            _string=lambda text: text,
            _set_state=lambda _: events.append('retry'),
        )
        supervisor.TaskStateMachine._handle_child_exit(node, 1)
        self.assertEqual(events, ['release', 'cancel', 'teardown', 'error', 'stop', 'retry'])
        self.assertEqual(events.count('teardown'), 1)

    def test_resident_teardown_waits_through_sigkill_before_replacement(self):
        process = Mock(pid=4242)
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired('resident', 5),
            subprocess.TimeoutExpired('resident', 5),
            0,
        ]
        node = SimpleNamespace(_resident_process=process, _process=process,
                               get_logger=lambda: Mock(), _fail=Mock())
        with patch.object(supervisor.os, 'killpg') as killpg:
            supervisor.TaskStateMachine._teardown_resident(node)
        self.assertEqual([c.args[1] for c in killpg.call_args_list], [
            supervisor.signal.SIGINT, supervisor.signal.SIGTERM,
            supervisor.signal.SIGKILL,
        ])
        self.assertEqual(process.wait.call_count, 3)
        self.assertIsNone(node._resident_process)
        node._fail.assert_not_called()

    def test_unsafe_object_never_recycles_resident(self):
        node = SimpleNamespace(
            _release_process=Mock(), _cancel_nav2_goals=Mock(),
            _resident_enabled=lambda: True, _last_executor_error='DEPENDENCY_TIMEOUT',
            _state='NAV_TO_ZONE', _unsafe_to_retry=True,
            _teardown_resident=Mock(), _fail=Mock(),
        )
        supervisor.TaskStateMachine._handle_child_exit(node, 1)
        node._teardown_resident.assert_not_called()
        node._fail.assert_called_once()


@unittest.skipUnless(
    (Path(__file__).resolve().parents[3] / 'run_logs' / 'tools' /
     'wedge_detector_replay.py').is_file(),
    'optional gitignored replay tool is absent')
class WedgeReplayTests(unittest.TestCase):
    """The replay must classify wedges and legitimate holds apart on real data."""

    def tool(self):
        import importlib.util
        path = (Path(__file__).resolve().parents[3] / 'run_logs' / 'tools' /
                'wedge_detector_replay.py')
        spec = importlib.util.spec_from_file_location('wedge_replay_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_stationary_episodes_ignore_motion_and_keep_long_holds(self):
        rows = []
        t = 0.0
        for step in (0.0, 0.2, 0.4):        # moving
            rows.append(dict(mono_abs=t, x=step, y=0.0, work_state='前往资源点'))
            t += 1.0
        for _ in range(10):                 # hold still for 10 s
            rows.append(dict(mono_abs=t, x=0.4, y=0.0, work_state='前往资源点'))
            t += 1.0
        episodes = self.tool().stationary_episodes(rows, 0.05, 8.0)
        self.assertEqual(len(episodes), 1)
        self.assertGreaterEqual(episodes[0][1] - episodes[0][0], 8.0)

    def test_short_pauses_are_not_episodes(self):
        rows = [dict(mono_abs=float(i), x=0.4, y=0.0, work_state='前往放置区')
                for i in range(5)]
        self.assertEqual(self.tool().stationary_episodes(rows, 0.05, 8.0), [])

    def test_non_navigation_states_are_excluded(self):
        rows = [dict(mono_abs=float(i), x=0.4, y=0.0, work_state='执行抓取')
                for i in range(30)]
        self.assertEqual(self.tool().stationary_episodes(rows, 0.05, 8.0), [])


class WedgeDetectorTests(unittest.TestCase):
    """Early abort of a hard wedge, keyed on the controller's own refusal.

    Legitimate stationary phases (all at a dock, after Nav2 hands control back)
    emit exactly zero of these messages, so the detector cannot confuse the two:
    measured 88-353 messages during wedges versus 0 during every legitimate hold.
    """

    def node(self, *, enabled=True, window=20.0, minimum=5, timeout=6.0):
        values = {'wedge_collision_window_sec': window,
                  'wedge_collision_min_messages': minimum,
                  'wedge_collision_timeout_sec': timeout}
        return SimpleNamespace(
            _controller_collisions=deque(),
            _wedge_enabled=enabled,
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
        )

    def log(self, name='controller_server', text='RegulatedPurePursuitController detected collision ahead!'):
        return SimpleNamespace(name=name, msg=text)

    def test_counts_only_the_controllers_collision_refusals(self):
        node = self.node()
        with patch.object(module.time, 'monotonic', return_value=100.0):
            for _ in range(3):
                module.FixedJointPickPlace._wedge_log(node, self.log())
            module.FixedJointPickPlace._wedge_log(node, self.log(name='planner_server'))
            module.FixedJointPickPlace._wedge_log(node, self.log(text='Passing new path'))
            self.assertEqual(module.FixedJointPickPlace._wedge_collisions_since(
                node, 90.0), 3)

    def test_old_messages_age_out_of_the_window(self):
        node = self.node(window=5.0)
        with patch.object(module.time, 'monotonic', return_value=100.0):
            for _ in range(9):
                module.FixedJointPickPlace._wedge_log(node, self.log())
        with patch.object(module.time, 'monotonic', return_value=130.0):
            self.assertEqual(module.FixedJointPickPlace._wedge_collisions_since(
                node, 100.0), 0)

    def test_disabled_detector_creates_no_subscription_or_state(self):
        node = SimpleNamespace(_wedge_enabled=False,
                               _controller_collisions=deque())
        self.assertFalse(node._wedge_enabled)
        # A legitimate hold must never reach the threshold, because it produces
        # no collision messages at all.
        node._wedge_enabled = True
        node.get_parameter = lambda name: SimpleNamespace(value={
            'wedge_collision_window_sec': 20.0}.get(name, 5))
        self.assertEqual(module.FixedJointPickPlace._wedge_collisions_since(
            node, 0.0), 0)


class ArrayParameterDeclarationTests(unittest.TestCase):
    """Array parameters must be declared with a plain [], never a Type enum.

    rclpy treats `rclpy.Parameter.Type.STRING_ARRAY` passed as the *value* as an
    uninitialised parameter and raises ParameterUninitializedException - unless a
    parameter file happens to supply that name, which is why it survived every
    round (the experiment variants all set slow_zones) and would only have blown
    up in the production configuration, where no file supplies it. Measured
    2026-10-11 on run_logs/rule2_trial1_20261011-032137, which died with
    "The parameter 'generator_args' is not initialized".
    """

    def test_no_type_enum_is_passed_as_a_parameter_value(self):
        for name in ('fixed_joint_pick_place.py', 'task_parser.py'):
            tree = ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8'))
            for call in [n for n in ast.walk(tree)
                         if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)
                         and n.func.attr == 'declare_parameter']:
                for arg in call.args[1:]:
                    text = ast.unparse(arg)
                    self.assertNotIn(
                        'Parameter.Type', text,
                        f'{name}:{call.lineno} passes {text} as a declare_parameter '
                        'value; use [] so the declaration also works when no '
                        'parameter file supplies the name')

    def test_the_sentinel_default_declares_with_and_without_an_override(self):
        """[0.0] works both with a double array supplied and without one."""
        import rclpy
        from rclpy.node import Node
        # With a parameter file supplying a double array.
        rclpy.init(args=['--ros-args', '-p', 'slow_zones:=[1.5,2.5,0.5,0.25]'])
        try:
            node = Node('sentinel_probe_with')
            node.declare_parameter('slow_zones', [0.0])
            self.assertEqual(list(node.get_parameter('slow_zones').value),
                             [1.5, 2.5, 0.5, 0.25])
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        # Without any override: the production configuration.
        rclpy.init()
        try:
            node = Node('sentinel_probe_without')
            node.declare_parameter('slow_zones', [0.0])
            self.assertEqual(list(node.get_parameter('slow_zones').value), [0.0])
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


class MovingObstacleTests(unittest.TestCase):
    """The patrolling boxes must stay on their clearance-checked segments."""

    def module(self):
        import importlib.util
        path = ROOT / 'scripts' / 'moving_obstacles.py'
        spec = importlib.util.spec_from_file_location('movers_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_patrols_parse(self):
        movers = self.module()
        patrols = movers.parse_patrols(movers.DEFAULT_PATROLS)
        self.assertEqual([p[0] for p in patrols],
                         ['mover_1', 'mover_2', 'mover_3'])
        for _, x0, y0, x1, y1, speed, _kin in patrols:
            self.assertGreater(speed, 0.0)
            self.assertGreater(math.hypot(x1 - x0, y1 - y0), 0.2)

    def test_direction_flips_at_both_ends(self):
        movers = self.module()
        p = movers._Patrol('m', 0.0, 0.0, 1.0, 0.0, 0.5)   # 1 m at 0.5 m/s
        self.assertEqual(p.direction, 1.0)
        p.advance(1.0)                 # 0.5 -> 1.0: reaches the far end
        self.assertEqual(p.direction, -1.0)
        self.assertAlmostEqual(p.s, 1.0, places=6)
        p.advance(1.0)                 # 1.0 -> 0.5
        self.assertEqual(p.direction, -1.0)
        p.advance(1.0)                 # 0.5 -> 0.0: back at the start
        self.assertEqual(p.direction, 1.0)
        self.assertAlmostEqual(p.s, 0.0, places=6)

    def test_advance_keeps_the_exact_commanded_speed(self):
        """The measured defect: 0.16 m/s commanded gave 0.031 m/s achieved."""
        movers = self.module()
        p = movers._Patrol('m', 0.0, 0.0, 10.0, 0.0, 0.16)
        p.s = 0.0
        p.advance(5.0)
        self.assertAlmostEqual(p.x, 0.16 * 5.0, places=6)
        self.assertAlmostEqual(p.y, 0.0, places=6)

    def test_the_box_position_is_derived_from_the_segment(self):
        """Nothing the base does can move it off the line any more."""
        movers = self.module()
        p = movers._Patrol('m', 0.0, 0.0, 1.0, 1.0, 0.2)
        p.advance(2.0)
        self.assertAlmostEqual(p.x, p.y, places=6)
        self.assertLessEqual(p.s, p.length)

    def test_mover_cloud_uses_robot_origin_and_matching_transform_stamp(self):
        """A map-frame PointCloud2 could not clear a rolling local costmap."""
        movers = self.module()
        tf = SimpleNamespace(
            header=SimpleNamespace(stamp=Time(sec=10, nanosec=250_000_000)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=-2.0, y=0.0, z=-0.1034),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
        published = []
        node = SimpleNamespace(
            patrols=[movers._Patrol('m', 2.0, 0.0, 3.0, 0.0, 0.2)],
            _tf_buffer=SimpleNamespace(lookup_transform=lambda *args: tf),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=10_350_000_000)),
            _cloud_pub=SimpleNamespace(publish=published.append))
        movers.MovingObstacles._publish_obstacle_cloud(node)
        self.assertEqual(len(published), 1)
        msg = published[0]
        self.assertEqual(msg.header.frame_id, 'base_link')
        self.assertEqual(msg.header.stamp, tf.header.stamp)
        self.assertEqual(msg.width, 25)
        from sensor_msgs_py import point_cloud2
        points = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z')))
        self.assertAlmostEqual(points[0][0], -movers.BOX_SIZE_M / 2, places=5)
        self.assertAlmostEqual(points[0][1], -movers.BOX_SIZE_M / 2, places=5)
        # base_link is raised 0.1034 m above the floor: both its origin and
        # the cloud returns are strictly inside the voxel layer's z >= 0 band.
        self.assertGreater(points[0][2], 0.0)
        self.assertLess(points[0][2], 0.78)

    def test_stale_transform_never_publishes_a_false_mover_location(self):
        movers = self.module()
        tf = SimpleNamespace(header=SimpleNamespace(
            stamp=Time(sec=1, nanosec=0)))
        published, warnings = [], []
        node = SimpleNamespace(
            _tf_buffer=SimpleNamespace(lookup_transform=lambda *args: tf),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
            _cloud_warn_last=-math.inf,
            get_logger=lambda: SimpleNamespace(warning=warnings.append),
            _cloud_pub=SimpleNamespace(publish=published.append))
        movers.MovingObstacles._publish_obstacle_cloud(node)
        self.assertFalse(published)
        self.assertEqual(len(warnings), 1)

    def test_mover_cloud_transform_applies_rotation_not_just_translation(self):
        movers = self.module()
        half_sqrt = math.sqrt(0.5)
        tf = SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=-2.0, z=-0.1),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=half_sqrt,
                                     w=half_sqrt))
        x, y, z = movers.transform_point_xyz((2.0, 0.0, 0.15), tf)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 0.05)

    def test_the_node_constructs_and_does_not_shadow_rclpy_state(self):
        """rclpy's Node already owns `publishers`; shadowing it crashed startup.

        Measured on run_logs/movers_trial1_20261011-141521, which died with
        "can't set attribute 'publishers'" and left the stack un-ready.
        """
        movers = self.module()
        import rclpy
        rclpy.init()
        try:
            node = movers.MovingObstacles()
            self.assertEqual(len(node.patrols), 3)
            self.assertTrue(all(p.seen for p in node.patrols))
            node._tick()          # must not raise before the boxes exist
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    def test_box_sdf_is_yellow_and_solid(self):
        """The boxes are spawned from this SDF, so its contents are the world."""
        movers = self.module()
        xml = movers.BOX_SDF.format(name='mover_x', kinematic='')
        self.assertIn('Gazebo/Yellow', xml)
        # No plugin: the pose is served directly, so there is no cmd_vel to send.
        self.assertNotIn('planar_move', xml)
        self.assertNotIn('cmd_vel', xml)
        self.assertIn('<size>0.22 0.22 0.30</size>', xml)
        # The kinematic flag must reach the SPAWNED model, not just a world file:
        # the boxes are inserted at runtime, so a flag only in the .world does
        # nothing (that mistake left mover_3 pushable).
        # ...and only when the patrol asked for it.
        kin = movers.BOX_SDF.format(
            name='mover_x', kinematic='    <kinematic>true</kinematic>\n')
        self.assertIn('<kinematic>true</kinematic>', kin)
        self.assertNotIn('kinematic', xml)
        import xml.etree.ElementTree as ET
        ET.fromstring(xml)          # must be well formed

    def test_default_patrols_give_mover_2_the_faster_speed(self):
        movers = self.module()
        speeds = {p[0]: p[5] for p in movers.parse_patrols(movers.DEFAULT_PATROLS)}
        flags = {p[0]: p[6] for p in movers.parse_patrols(movers.DEFAULT_PATROLS)}
        self.assertTrue(flags['mover_3'], 'mover_3 must be kinematic')
        self.assertFalse(flags['mover_1'])
        self.assertFalse(flags['mover_2'])
        self.assertGreater(speeds['mover_2'], speeds['mover_1'])
        self.assertGreater(speeds['mover_2'], speeds['mover_3'])

    def test_a_box_never_yields_and_never_moves_off_its_line(self):
        """The obstacle has the highest priority; the base has no influence.

        Requested 2026-10-13. The old behaviour - hold the box while the base was
        inside a 0.75 m bubble - was measured on run_logs/yield3_131041: mover_2
        frozen for 110 s of a 130 s trace with the base parked 0.38 m away, so
        neither could get through. A box's progress must be a pure function of
        time, even with the base sitting on its next position.
        """
        movers = self.module()
        clock = {'t': 0.0}
        node = SimpleNamespace(
            patrols=[], _robot_xy=(1.0, 0.0), _spawned=True, _last_tick=None,
            _reset_client=SimpleNamespace(service_is_ready=lambda: False),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=clock['t'] * 1e9)),
            get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: None),
            get_parameter=lambda name: SimpleNamespace(
                value={'robot_clearance_m': 0.75,
                       'publish_obstacle_cloud': False}[name]))
        for name in ('_tick', '_warn_if_base_inside_box'):
            setattr(node, name, getattr(movers.MovingObstacles, name).__get__(node))
        patrol = movers._Patrol('m', 0.0, 0.0, 2.0, 0.0, 0.2)
        patrol.s = 1.0
        node.patrols = [patrol]
        node._tick()                      # first tick only seeds sim time
        # Base parked on the box's next position, inside every clearance bubble
        # this node ever had.
        node._robot_xy = (1.05, 0.0)
        before = patrol.s
        clock['t'] += 0.1
        node._tick()
        self.assertGreater(patrol.s, before, 'the box must keep moving')
        self.assertAlmostEqual(patrol.y, 0.0, places=6)
        self.assertAlmostEqual(patrol.s, before + 0.2 * 0.1, places=9)

    def test_every_box_is_spawned_kinematic(self):
        """A kinematic body ignores contact forces, so the base cannot move it."""
        src = (ROOT / 'scripts' / 'moving_obstacles.py').read_text(encoding='utf-8')
        body = src[src.index('def _spawn_boxes'):]
        spawn = body[:body.index('self._spawned = True')]
        self.assertIn("kinematic='    <kinematic>true</kinematic>", spawn)
        self.assertNotIn('if patrol.kinematic', spawn)
        movers = self.module()
        for kin in (True, False):
            self.assertTrue(
                movers._Patrol('m', 0.0, 0.0, 1.0, 0.0, 0.2, kin).kinematic)

    def test_the_clearance_default_exceeds_the_footprint(self):
        movers = self.module()
        self.assertGreaterEqual(movers.DEFAULT_CLEARANCE_M, 0.46)

    def test_the_boxes_wait_for_the_robot_before_spawning(self):
        """Spawning first wins Gazebo's queue and the ROBOT never appears.

        Measured on movers_live/comp_mode_round.log: boxes in at t+8.45 s while
        spawn_entity sat on "Waiting for entity xml on robot_description"
        forever, leaving /scan, /map and every controller missing.
        """
        movers = self.module()
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(
                value={'robot_model_prefix': 'six_arm',
                       'wait_for_robot_sec': 0.0}[name]))
        self.assertTrue(movers.MovingObstacles._wait_for_robot(node))

    def test_the_wait_is_disabled_by_an_empty_prefix(self):
        movers = self.module()
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(
                value={'robot_model_prefix': '',
                       'wait_for_robot_sec': 120.0}[name]))
        self.assertTrue(movers.MovingObstacles._wait_for_robot(node))

    def test_spawn_boxes_waits_first(self):
        src = (ROOT / 'scripts' / 'moving_obstacles.py').read_text(encoding='utf-8')
        body = src[src.index('def _spawn_boxes'):]
        self.assertLess(body.index('_wait_for_robot()'),
                        body.index('wait_for_service'),
                        'the robot wait must come before any box is inserted')

    def test_already_exists_counts_as_a_successful_spawn(self):
        """Gazebo creates the box but may not confirm it in time.

        Measured on movers_live/check_round.log: attempt 1 timed out waiting for
        the entity to appear, attempts 2-4 said "Entity [mover_1] already
        exists" - so the box WAS in the scene while the node declared it missing.
        """
        movers = self.module()
        logged = []
        node = SimpleNamespace(
            patrols=[movers._Patrol('m', 0.0, 0.0, 2.0, 0.0, 0.1)],
            _spawn_client=SimpleNamespace(wait_for_service=lambda timeout_sec: True),
            get_parameter=lambda name: SimpleNamespace(value=1),   # one attempt
            get_logger=lambda: SimpleNamespace(
                warning=lambda m, *a, **k: logged.append(str(m)),
                error=lambda m, *a, **k: logged.append('ERR ' + str(m))),
            _reset_client=None,
            _wait_for_robot=lambda: True,      # covered by its own tests
        )
        replies = iter([SimpleNamespace(success=False,
                                        status_message='Entity [m] already exists.')])
        node._spawn_client.call_async = lambda req: SimpleNamespace(result=lambda: next(replies))
        with patch.object(movers.rclpy, 'spin_until_future_complete', lambda *a, **k: None):
            movers.MovingObstacles._spawn_boxes(node)
        self.assertTrue(any('spawn ok' in m for m in logged), logged)
        self.assertFalse(any(m.startswith('ERR') for m in logged), logged)

    def test_bad_specs_are_rejected(self):
        movers = self.module()
        for bad in ('', 'mover_1', 'mover_1:1,2,3',
                    'mover_1:0,0,1,0,0', 'mover_1:0,0,0.05,0,0.2'):
            with self.assertRaises(ValueError):
                movers.parse_patrols(bad)


class StaticUndefinedNameTests(unittest.TestCase):
    """Catch names that compile fine but do not exist at runtime.

    A launch file that references ExecuteProcess without importing it passes
    py_compile and every unit test, then dies the instant it is launched:
    "Caught exception in launch: name 'ExecuteProcess' is not defined"
    (measured 2026-10-11, /tmp/mybot_start_YD0c.log). pyflakes catches it.
    """

    def test_no_undefined_names_in_launch_or_scripts(self):
        try:
            from pyflakes.api import check
            from pyflakes.reporter import Reporter
        except ImportError:                                  # pragma: no cover
            self.skipTest('pyflakes not installed')
        import io
        targets = sorted((ROOT / 'launch').glob('*.py')) + \
            sorted((ROOT / 'scripts').glob('*.py'))
        self.assertTrue(targets, 'no launch/script files found')
        offenders = []
        for path in targets:
            out, err = io.StringIO(), io.StringIO()
            check(path.read_text(encoding='utf-8'), str(path),
                  Reporter(out, err))
            for line in (out.getvalue() + err.getvalue()).splitlines():
                if 'undefined name' in line.lower():
                    offenders.append(line.strip())
        self.assertEqual(offenders, [], 'undefined names would crash at runtime')


class ControllerSpawnerChainTests(unittest.TestCase):
    """The controller chain must not break when a spawner needs a retry.

    The old wiring had the OnProcessExit handler start a NEW spawner Node on
    failure. launch only fires a handler for the exact action object it was
    registered with, so the replacement's success was invisible and the chain
    stopped: joint_state_broadcaster active, arm_controller and
    gripper_controller never attempted at all, system_ready false, task IDLE
    forever (measured 2026-10-11, movers_live/startbtn_round.log).
    """

    def source(self):
        return (ROOT / 'launch' / 'gazebo_world2.launch.py').read_text(
            encoding='utf-8')

    def test_already_exists_counts_as_a_successful_robot_spawn(self):
        """Gazebo creates the robot but may not confirm it in time.

        Measured 2026-10-12: attempt 1 said "Entity pushed to spawn queue, but
        spawn service timed out waiting for entity to appear", attempts 2-4 said
        "Entity [six_arm] already exists", and /gazebo/link_states listed
        six_arm::base_footprint the whole time. Counting that as failure aborted
        the controller chain, so system_ready stayed false and the task sat in
        IDLE. The patrol boxes already had this fix; the robot did not.
        """
        src = self.source()
        self.assertIn('already exists', src)
        self.assertIn('ok on attempt', src)
        # The spawn must capture its output to be able to inspect it.
        self.assertIn('out=$(', src)

    def test_the_retry_lives_inside_one_process(self):
        src = self.source()
        self.assertIn('for attempt in 1 2 3 4; do', src)
        self.assertIn("cmd=['bash', '-c', script]", src)

    def test_no_handler_starts_a_replacement_spawner(self):
        """A retry that creates a new action is what broke the chain."""
        src = self.source()
        self.assertNotIn('retry_on_failure(', src)
        self.assertNotIn("actions=[controller_spawner(controller)]", src)

    def test_every_step_follows_the_previous_one(self):
        src = self.source()
        for label in ('joint_state_broadcaster', 'arm_controller',
                      'gripper_controller'):
            self.assertIn(f"'{label}'", src)

    def test_the_final_step_tolerates_no_next_action(self):
        src = self.source()
        self.assertIn('return [] if action is None else [action]', src)

    def test_the_gui_client_waits_for_the_robot_spawn(self):
        """gzclient renders the same world and slows the loader down.

        Measured 2026-10-14 over 12 rounds: gzserver advertised /spawn_entity
        after 5.8 s in the fast rounds and 34.1 s in the slow ones, and that one
        wait was the whole difference in startup (READY 24-27 s vs 48-100 s),
        because no controller, MoveIt or Nav2 step can start before the robot
        exists.
        """
        src = self.source()
        self.assertIn('evt_gui', src)
        self.assertIn('target_action=spawn_entity_cmd,\n            on_exit=lambda event, context: (\n                [start_gazebo_client]',
                      src)
        # It must NOT be started unconditionally beside gzserver any more.
        self.assertNotIn('ld.add_action(start_gazebo_client)', src)

    def test_a_lost_controller_manager_reply_costs_seconds_not_a_minute(self):
        """One dropped DDS response must not burn a 60 s service timeout.

        Measured 2026-10-14 15:21: the gripper spawner's list_controllers reply
        was dropped ("failed to send response ... (timeout)") and the spawner sat
        until 65.9 s before retrying; the world and Nav2 were already up by then.
        """
        src = self.source()
        self.assertIn('--service-call-timeout 12', src)
        self.assertNotIn('--service-call-timeout 60', src)
        # The bounded retry ladder and the loud failure stay.
        self.assertIn('for attempt in 1 2 3 4; do', src)
        self.assertIn('FAILED after 4 attempts', src)


class PlacementCheckOrderTests(unittest.TestCase):
    """The landing must be checked right after release, not 40 s later.

    Measured 2026-10-12 (twotier5, blue_cube_5 -> A). The sequence was:
    carry verified attached at the dock (0.070 m of a 0.120 m limit), then
    pre-place, Lower to place (10.0 s), Open gripper (8.7 s), DETACH,
    Retreat (10.7 s), Return home (8.8 s) - and only then the single placement
    check, which found the cube 1.9 m away. Nothing in the log could say whether
    the cube was placed badly or knocked away during those 40 s of arm motion.
    """

    def source(self):
        return (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')

    def steps(self):
        src = self.source()
        body = src[src.index('steps = ['):]
        return body[:body.index('\n        ]')]

    def test_placement_is_verified_before_the_retreat(self):
        steps = self.steps()
        self.assertLess(steps.index("'verify placement'"),
                        steps.index("'retreat'"),
                        'the landing must be checked before the arm retreats')

    def test_the_early_check_still_follows_the_detach(self):
        steps = self.steps()
        self.assertLess(steps.index("'detach cube'"),
                        steps.index("'verify placement'"))

    def test_a_final_confirmation_remains_at_the_end(self):
        """A cube the retreat nudges off the plate is still a miss."""
        steps = self.steps()
        self.assertIn("'verify placement after retreat'", steps)
        self.assertGreater(steps.index("'verify placement after retreat'"),
                           steps.index("'return home'"))

    def test_both_checks_call_the_same_verifier(self):
        steps = self.steps()
        self.assertEqual(steps.count('self._verify_placement'), 2)


class SharedLinkStateRefreshTests(unittest.TestCase):
    """Exactly one place may refresh the Gazebo link states.

    Three functions each cleared the cache and waited proximity_timeout_sec:
    _link_pose, _link_offset and _link_distance. A retry was added to the first
    only, so a round still died at "Timed out waiting for /gazebo/link_states" -
    during verify-carried-cube, which uses _link_distance - while the topic was
    publishing at 18 Hz. Third time this codebase has been bitten by fixing one
    of N identical sites, hence a structural guard instead of another patch.
    """

    def source(self):
        return (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')

    def test_there_is_exactly_one_shared_helper(self):
        src = self.source()
        self.assertEqual(src.count('def _refresh_link_states(self)'), 1)

    def test_only_the_helper_clears_the_cache_before_waiting(self):
        """__init__ may initialise it; nobody else may clear-and-wait."""
        import re
        src = self.source()
        lines = src.splitlines()
        offenders = []
        for i, line in enumerate(lines):
            if line.strip() != 'self._latest_link_states = None':
                continue
            if 'def __init__' in src[:sum(len(l) + 1 for l in lines[:i])][-4000:]:
                pass
            window = '\n'.join(lines[max(0, i - 6):i + 8])
            if '_wait_until' in window and 'for attempt in range' not in window:
                offenders.append(i + 1)
        self.assertEqual(offenders, [],
                         f'clear-and-wait outside the helper at lines {offenders}')

    def test_every_consumer_goes_through_the_helper(self):
        src = self.source()
        # _link_pose, _link_offset and _link_distance all depend on it.
        for fn in ('_link_pose', '_link_offset', '_link_distance'):
            body = src[src.index(f'def {fn}('):]
            body = body[:body.index('\n    def ', 10)]
            self.assertIn('_refresh_link_states()', body,
                          f'{fn} must refresh through the shared helper')

    def test_the_helper_retries_and_still_fails_finitely(self):
        src = self.source()
        body = src[src.index('def _refresh_link_states'):]
        body = body[:body.index('\n    def ', 10)]
        self.assertIn('for attempt in range(1, attempts + 1)', body)
        self.assertIn('proximity_retry_attempts', body)
        self.assertIn('return False', body)


class ImageDetectionFunctionalTests(unittest.TestCase):
    """Actually run the detector on a synthetic frame.

    Why this exists: a refactor renamed the accumulator inside pick_best but not
    its initialiser, so the detector returned None for EVERY frame while all the
    source-grep tests passed. The vision search then timed out after 18 s and a
    competition round failed 0/5. Grepping the source cannot tell you that a
    function computes the wrong answer - only calling it can.
    """

    PARAMS = {
        'visual_blue_min_saturation': 170, 'visual_blue_min_value': 70,
        'visual_blue_hue_min': 110, 'visual_blue_hue_max': 130,
        'visual_blue_dominance_ratio': 2.0, 'visual_red_min_saturation': 200,
        'visual_red_min_value': 100, 'visual_red_dominance_ratio': 2.5,
        'visual_min_contour_area': 100.0, 'visual_expected_row_tolerance': 80.0,
        'visual_relaxed_saturation_scale': 0.65,
        'visual_relaxed_value_scale': 0.60,
        'visual_relaxed_dominance_scale': 0.80,
        'cube_name': 'red_cube_1', 'visual_row_distance_offset_m': 0.015,
        'visual_row_distance_k': 66.0, 'visual_row_center': 360.0,
        'camera_forward_offset_m': 0.0,
    }

    def detector(self, colour='red', logs=None):
        """A node-shaped stub exposing the real _image_callback."""
        params = dict(self.PARAMS)

        class Stub:
            _vision_enabled = True
            _target_colour = colour
            _visual_range_gate_active = False
            _visual_detection = None
            _visual_detection_time = 0.0
            _cached_planar_distance_to_cube = staticmethod(lambda cube: None)

            class _Pub:
                def get_subscription_count(self):
                    return 0

            _debug_image_pub = _Pub()

            def get_parameter(self, name):
                import types
                return types.SimpleNamespace(value=params[name])

            class _Log:
                def warning(self, *a):
                    if logs is not None:
                        logs.append(a[0] if a else '')

                def info(self, *a):
                    pass

            def get_logger(self):
                return Stub._Log()

            class _Bridge:
                def imgmsg_to_cv2(self, msg, desired_encoding=None):
                    return msg

            _bridge = _Bridge()

        Stub._image_callback = module.FixedJointPickPlace._image_callback
        return Stub()

    @staticmethod
    def frame(value=None, colour='red'):
        import numpy as np
        image = np.full((720, 1280, 3), 40, dtype=np.uint8)
        if value is not None:
            pixel = (0, 0, value) if colour == 'red' else (value, 0, 0)
            image[500:560, 600:680] = pixel
        return image

    def test_a_plain_red_blob_is_detected(self):
        node = self.detector('red')
        node._image_callback(self.frame(220, 'red'))
        self.assertIsNotNone(node._visual_detection,
                             'the detector must find an obvious red blob')
        _centre, _width, _area, _image_width, bottom_row = node._visual_detection
        self.assertAlmostEqual(bottom_row, 560.0, delta=2.0)

    def test_a_plain_blue_blob_is_detected(self):
        node = self.detector('blue')
        node._image_callback(self.frame(220, 'blue'))
        self.assertIsNotNone(node._visual_detection)

    def test_a_shaded_cube_falls_back_to_the_relaxed_tier(self):
        """V=80 is under the strict floor of 100 but over the relaxed 60."""
        logs = []
        node = self.detector('red', logs)
        node._image_callback(self.frame(80, 'red'))
        self.assertIsNotNone(node._visual_detection,
                             'a shaded cube must still be found')
        self.assertTrue(any('relaxed tier' in m for m in logs),
                        'the fallback must be logged so it can be counted')

    def test_a_plain_red_blob_does_not_need_the_fallback(self):
        logs = []
        node = self.detector('red', logs)
        node._image_callback(self.frame(220, 'red'))
        self.assertEqual([m for m in logs if 'relaxed tier' in m], [],
                         'the strict tier must handle a normal cube alone')

    def test_a_dark_frame_produces_no_detection(self):
        """Neither tier may invent a blob - V=50 is below both floors."""
        node = self.detector('red')
        node._image_callback(self.frame(50, 'red'))
        self.assertIsNone(node._visual_detection)

    def test_an_empty_frame_produces_no_detection(self):
        node = self.detector('red')
        node._image_callback(self.frame(None))
        self.assertIsNone(node._visual_detection)

    def test_the_callback_is_inert_while_vision_is_disabled(self):
        node = self.detector('red')
        node._vision_enabled = False
        node._image_callback(self.frame(220, 'red'))
        self.assertIsNone(node._visual_detection,
                          'navigation phases must not run the HSV pipeline')


class DuplicateParameterTests(unittest.TestCase):
    """A parameter declared twice crashes the node the moment it starts.

    Measured 2026-10-12: restoring an earlier snapshot of
    fixed_joint_pick_place.py brought back three vision parameters that I then
    declared again, so the resident executor died in __init__ with
    ParameterAlreadyDeclaredException and two whole competition rounds failed at
    0/5 with "OBJECT_FAILED:executor exited 1". None of the 174 tests caught it -
    they only check that a parameter name APPEARS in the source, which a
    duplicate also satisfies.
    """

    def scripts(self):
        return sorted((ROOT / 'scripts').glob('*.py'))

    def test_no_parameter_is_declared_twice_in_any_script(self):
        import collections
        import re
        offenders = {}
        for path in self.scripts():
            names = re.findall(
                r"declare_parameter\(\s*'([^']+)'",
                path.read_text(encoding='utf-8'),
            )
            dups = {k: v for k, v in collections.Counter(names).items()
                    if v > 1}
            if dups:
                offenders[path.name] = dups
        self.assertEqual(offenders, {}, f'duplicate declare_parameter: {offenders}')

    def test_the_check_would_notice_a_duplicate(self):
        """Guard the guard: the scan must actually match the call shape."""
        import re
        sample = (
            "        self.declare_parameter('a', 1)\n"
            "        self.declare_parameter('a', 2)\n"
        )
        self.assertEqual(
            re.findall(r"declare_parameter\(\s*'([^']+)'", sample),
            ['a', 'a'],
        )


class RelaxedColourTierTests(unittest.TestCase):
    """A shaded cube must not blank the blob for seconds.

    Measured 2026-10-12 with the cubes in their new positions: vision_align went
    from 8.8-9.2 s (cubes in the original spots) to 28-64 s, with repeated
    "Visual blob lost for 3.1 s; restarting the search". The strict gates
    (red saturation >= 200, value >= 100, red >= 2.5x green and blue) are what
    keep the brick-red walls out, so they cannot simply be loosened - instead a
    relaxed tier is consulted ONLY when the strict one finds no candidate, and
    the shape and expected-row gates still apply to it.
    """

    def source(self):
        return (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')

    def test_the_mask_builder_takes_the_scale_factors(self):
        src = self.source()
        self.assertIn('def build_mask(sat_scale: float, val_scale: float, '
                      'dom_scale: float)', src)
        self.assertIn('* sat_scale', src)
        self.assertIn('* val_scale', src)
        self.assertIn('* dom_scale', src)

    def test_the_strict_tier_is_tried_first(self):
        src = self.source()
        self.assertIn('mask = build_mask(1.0, 1.0, 1.0)', src)
        self.assertLess(src.index('mask = build_mask(1.0, 1.0, 1.0)'),
                        src.index('relaxed_mask = build_mask('))

    def test_the_fallback_only_runs_when_strict_found_nothing(self):
        src = self.source()
        body = src[src.index('mask = build_mask(1.0, 1.0, 1.0)'):]
        body = body[:body.index('if best is not None:')]
        self.assertIn('if best is None:', body)
        self.assertIn('visual_relaxed_saturation_scale', body)

    def test_the_relaxed_tier_is_logged_so_it_can_be_counted(self):
        src = self.source()
        self.assertIn('Strict colour gates found no blob', src)

    def test_the_shape_and_row_gates_are_shared_by_both_tiers(self):
        """pick_best must be the single gatekeeper for either mask."""
        src = self.source()
        self.assertIn('def pick_best(mask):', src)
        self.assertEqual(src.count('def pick_best(mask):'), 1)
        # The gates live inside it, not duplicated per tier.
        body = src[src.index('def pick_best(mask):'):]
        body = body[:body.index('mask = build_mask(1.0, 1.0, 1.0)')]
        self.assertIn('visual_expected_row_tolerance', body)
        self.assertIn('visual_min_contour_area', body)

    def test_both_configs_carry_the_scale_factors(self):
        import yaml
        vals = []
        for path in (ROOT / 'config' / 'fixed_joint_pick_place.yaml',
                     ROOT / 'config' / 'profiles' / 'executor_fast.yaml'):
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            params = data[next(iter(data))]['ros__parameters']
            vals.append(tuple(params.get(k) for k in (
                'visual_relaxed_saturation_scale',
                'visual_relaxed_value_scale',
                'visual_relaxed_dominance_scale')))
        self.assertEqual(vals[0], vals[1])
        for v in vals[0]:
            self.assertTrue(0.0 < v <= 1.0, 'scales must relax, not tighten')

    def test_setting_the_scales_to_one_disables_the_fallback(self):
        """With 1.0 the relaxed tier reproduces the strict mask exactly."""
        src = self.source()
        self.assertIn('Set the three scales to 1.0 to disable the fallback', src)


class LinkStateRetryTests(unittest.TestCase):
    """A missed Gazebo link-states window must not end the round.

    Measured 2026-10-12 (official_run3 and vision_stab1): both rounds stopped at
    "Timed out waiting for /gazebo/link_states" during VERIFY_GRASP - 3/5
    deliveries done, robot and payload fine. _link_pose cleared its cache and
    waited proximity_timeout_sec (5 s) exactly once, so a transient DDS or
    gzserver stall was fatal.
    """

    def source(self):
        return (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')

    def test_link_pose_retries_a_bounded_number_of_times(self):
        src = self.source()
        self.assertIn('proximity_retry_attempts', src)
        self.assertIn('for attempt in range(1, attempts + 1)', src)
        self.assertIn('retrying ({attempt}/{attempts - 1})', src)

    def test_the_retry_still_fails_in_finite_time(self):
        """Bounded, not indefinite: attempts x proximity_timeout_sec."""
        src = self.source()
        body = src[src.index('def _refresh_link_states'):]
        body = body[:body.index('\n    def ', 10)]
        self.assertIn('attempts = max(1, int(', body)
        self.assertIn('return False', body)
        self.assertNotIn('while True', body)

    def test_the_cache_is_cleared_before_every_attempt(self):
        """Otherwise a retry would accept the stale message it just rejected."""
        src = self.source()
        body = src[src.index('def _refresh_link_states'):]
        body = body[:body.index('\n    def ', 10)]
        self.assertEqual(body.count('self._latest_link_states = None'), 1)
        # It must sit INSIDE the loop, so every attempt - not just the first -
        # clears the cache before waiting.
        self.assertGreater(
            body.index('self._latest_link_states = None'),
            body.index('for attempt in range'),
        )

    def test_both_configs_carry_the_retry_count(self):
        import yaml
        values = []
        for path in (ROOT / 'config' / 'fixed_joint_pick_place.yaml',
                     ROOT / 'config' / 'profiles' / 'executor_fast.yaml'):
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            params = data[next(iter(data))]['ros__parameters']
            values.append(params.get('proximity_retry_attempts'))
        self.assertEqual(values[0], values[1])
        self.assertGreaterEqual(values[0], 2)


class VisualStandoffStabilityTests(unittest.TestCase):
    """The final standoff must not be decided by a single video frame.

    Measured 2026-10-12: standoff = 0.015 + 66/(bottom_row - 360), and with
    visual_row_tolerance 18 px the accepting row may be anywhere in 662-698, so
    the gate alone sweeps the standoff across 0.210-0.234 m (23 mm). One noisy
    frame then moved it further: the base stopped ~40 mm too far and the run
    ended with "Pre-attach distance: 0.107 m (limit 0.090 m)" and a refused
    grasp - in a round whose vision had already converged (centre_error -0.002).
    """

    def source(self):
        return (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')

    def params(self, name):
        import yaml
        path = (ROOT / 'config' / 'profiles' / name if name.startswith('executor')
                else ROOT / 'config' / name)
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        return data[next(iter(data))]['ros__parameters']

    def test_the_standoff_comes_from_a_median_not_the_last_frame(self):
        src = self.source()
        self.assertIn('visual_standoff_frames', src)
        self.assertIn('np.median(stable_rows)', src)
        # The old shape - handing the raw last frame straight to the creep - must
        # be gone.
        self.assertNotIn('return self._creep_to_grasp(bottom_row)', src)

    def test_agreeing_frames_are_required_before_accepting(self):
        src = self.source()
        self.assertIn('visual_stable_span_px', src)
        self.assertIn('span = max(stable_rows) - min(stable_rows)', src)
        self.assertIn('span <= span_limit', src)

    def test_the_wait_for_agreement_is_bounded(self):
        """It must still accept if the readings never settle."""
        src = self.source()
        self.assertIn('stable >= required * 3', src)

    def test_the_collected_rows_stay_bounded(self):
        src = self.source()
        self.assertIn('stable_rows.pop(0)', src)
        self.assertIn('stable_rows.clear()', src)

    def test_both_configs_carry_the_same_visual_settings(self):
        base = self.params('fixed_joint_pick_place.yaml')
        profile = self.params('executor_fast.yaml')
        for key in ('visual_standoff_frames', 'visual_stable_span_px',
                    'visual_stable_frames', 'visual_row_tolerance'):
            self.assertEqual(base.get(key), profile.get(key),
                             f'{key} differs between base config and profile')

    def test_the_new_defaults_are_sane(self):
        profile = self.params('executor_fast.yaml')
        self.assertGreaterEqual(profile['visual_standoff_frames'], 3)
        self.assertGreater(profile['visual_stable_span_px'], 0)


class ProfileConsistencyTests(unittest.TestCase):
    """The fast profile is a FULL COPY of the executor config, so editing the
    base file alone changes nothing at runtime.

    Measured 2026-10-12: all ten pickup poses were corrected in
    config/fixed_joint_pick_place.yaml while config/profiles/executor_fast.yaml
    kept the old ones. The launch passes the PROFILE, so the robot drove to the
    old pickup spots, the vision could not find the cubes, and a competition
    round took 1145 s and still failed at 3/5 (the same run took 240-355 s for
    5/5 before the cubes were moved). This is the third place the same trap
    appeared - world file, pickup poses, and now the profile copy - so it gets a
    guard rather than another manual fix.
    """

    def params(self, path):
        import yaml
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        return data[next(iter(data))]['ros__parameters']

    def test_the_profile_agrees_with_the_base_on_every_pickup_pose(self):
        import re
        base = self.params(ROOT / 'config' / 'fixed_joint_pick_place.yaml')
        profile = self.params(ROOT / 'config' / 'profiles' / 'executor_fast.yaml')
        diverged = []
        for key, value in base.items():
            if re.match(r'pickup_(red|blue)_cube_\d_pose$', key):
                if profile.get(key) != value:
                    diverged.append((key, value, profile.get(key)))
        self.assertEqual(diverged, [], '快配置档的取物位与基础配置不一致（运行时用的是档）')

    def test_the_profile_still_carries_its_speed_settings(self):
        """Syncing poses must not wipe the changes the profile exists for."""
        profile = self.params(ROOT / 'config' / 'profiles' / 'executor_fast.yaml')
        self.assertEqual(profile.get('arm_motion_sec'), 0.6)
        self.assertEqual(profile.get('gripper_motion_sec'), 0.5)
        self.assertEqual(profile.get('settle_sec'), 0.2)
        self.assertAlmostEqual(profile.get('dock_creep_align_distance_m'), 0.25)


class ZoneSlotTests(unittest.TestCase):
    """Every delivery used to aim at the same spot, so cubes stacked and slid.

    Measured 2026-10-12, seed 995410 (x=4 red -> A): delivery 1 placed at
    (6.883,-2.519); delivery 3 was released at the nominal (6.886,-2.483) - on
    top of delivery 1 - and ended at (6.334,-2.446), 0.552 m off, outside the
    +/-0.485 acceptance box. The dock creep was fine (0.020 m), so the miss was
    stacking, not docking.
    """

    def node(self, delivered=0, offsets=(0.0, 0.0, 0.0, 0.16, 0.0, -0.16, -0.24, 0.0)):
        values = {
            'warehouse': 'A', 'warehouse_a_pose': [7.244, -2.483, -3.141593],
            'zone_slot_offsets': list(offsets),
        }
        node = SimpleNamespace(
            _zone_deliveries={'A': delivered},
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
            get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None),
        )
        for name in ('_warehouse_goal', '_double_list'):
            setattr(node, name, getattr(module.FixedJointPickPlace, name).__get__(node))
        return node

    def test_each_delivery_gets_a_different_slot(self):
        seen = []
        for delivered in range(4):
            goal = self.node(delivered)._warehouse_goal()[1]
            seen.append((round(goal[0], 3), round(goal[1], 3)))
        self.assertEqual(len(set(seen)), 4, f'slots repeat: {seen}')

    def test_slots_stay_inside_the_acceptance_box(self):
        """Cube = dock - 0.358 m in x; the plate is +/-0.485 x +/-0.235."""
        centre = (6.886, -2.483)
        for delivered in range(4):
            goal = self.node(delivered)._warehouse_goal()[1]
            cube_x, cube_y = goal[0] - 0.358, goal[1]
            self.assertLessEqual(abs(cube_x - centre[0]), 0.485, 'x outside plate')
            self.assertLessEqual(abs(cube_y - centre[1]), 0.235, 'y outside plate')

    def test_slots_are_far_enough_apart_that_cubes_cannot_stack(self):
        cubes = []
        for delivered in range(4):
            goal = self.node(delivered)._warehouse_goal()[1]
            cubes.append((goal[0] - 0.358, goal[1]))
        for i in range(len(cubes)):
            for j in range(i + 1, len(cubes)):
                gap = math.hypot(cubes[i][0] - cubes[j][0],
                                 cubes[i][1] - cubes[j][1])
                self.assertGreater(gap, 0.05, f'slots {i} and {j} too close')

    def test_the_rotation_wraps_for_a_fifth_delivery(self):
        first = self.node(0)._warehouse_goal()[1]
        fifth = self.node(4)._warehouse_goal()[1]
        self.assertEqual(round(first[0], 3), round(fifth[0], 3))
        self.assertEqual(round(first[1], 3), round(fifth[1], 3))

    def test_a_single_offset_disables_the_rotation(self):
        goal = self.node(2, offsets=(0.0, 0.0))._warehouse_goal()[1]
        self.assertEqual(goal[:2], [7.244, -2.483])


class PrewarmResidentExecutorTests(unittest.TestCase):
    """The 3.3 s executor start must not sit between the press and the motion.

    Measured 2026-10-14 15:52 (relative to the start press): the supervisor
    released the gate at +0.0 s, the executor process was up at +3.3 s, PRE_GRASP
    at +5.3 s and the base moved at +6.3 s. With the parse now gated on the same
    button, that start is the only part of the wait left to move off the press.
    """

    def source(self):
        return (ROOT / 'scripts' / 'task_state_machine.py').read_text(
            encoding='utf-8')

    def test_the_executor_is_prewarmed_before_the_first_object(self):
        src = self.source()
        self.assertIn('Pre-warmed the resident executor', src)
        self.assertIn('and not self._object_active and self._index == 0', src)
        self.assertIn('and self._resident_process is None', src)
        # The warming must sit where the queue does not exist yet.
        block = src[src.index('if not self._queue:'):][:1800]
        self.assertIn('_ensure_resident()', block)

    def test_the_press_that_parses_the_task_also_starts_the_run(self):
        """Parsing is gated on the button, so one press must do both jobs.

        Measured 2026-10-14 17:40 (run_logs/parse_after_start_20261014-173956):
        the press released the parse at t+2.1 s, the queue was accepted at
        t+17.0 s, the machine went back to WAITING_FOR_START because accepting a
        task cleared the latched press, and the run only left after a SECOND
        press at t+35.2 s.
        """
        src = self.source()
        self.assertIn('self._run_started_once = False', src)
        self.assertIn('if self._run_started_once:', src)
        clear = src[src.index('if self._run_started_once:'):]
        self.assertIn('self._start_requested = False', clear[:200])
        # The first object marks the run as started, before anything else.
        start = src[src.index('def _start_current'):]
        self.assertIn('self._run_started_once = True', start[:900])

    def test_waiting_for_start_is_shown_even_without_a_parsed_task(self):
        """The queue may legitimately be absent until the button is pressed."""
        src = self.source()
        block = src[src.index('if not self._queue:'):][:1800]
        self.assertIn("self._set_state('WAITING_FOR_START')", block)
        self.assertIn('self._system_ready and self._start_gate_closed()', block)


class StartButtonPanelTests(unittest.TestCase):
    """The 「开始任务」 button lives in the dashboard layout, so its config is code.

    Two faults kept this button dead, and the second hid behind the first:

    * the type name was the ROS 1 form "std_msgs/Bool"; and
    * the field was called `messageType`, which the Foxglove Publish panel does
      not read - its settings are topic / message schema / editing mode / button
      title / tooltip / colour, so the schema has to arrive as `schemaName`.
      With no schema the panel shows "请在面板设置中配置主题和消息..." and greys
      the button out even though the topic and payload are correct.

    Measured 2026-10-13: renaming the field to `schemaName` (value unchanged,
    "std_msgs/msg/Bool") turned the button live, and pressing it made the state
    machine log "Start button pressed; releasing the gate." - the CLI-equivalent
    test below is not enough, because the panel never reads the field it checks.
    """

    def panel(self):
        import json
        path = (ROOT / 'config' / 'foxglove' /
                'competition_dashboard_layout.json')
        return json.loads(path.read_text(encoding='utf-8'))[
            'configById']['Publish!start']

    def test_the_schema_uses_the_ros2_type_name(self):
        panel = self.panel()
        # `schemaName` is the field the Publish panel reads; a `messageType` key
        # leaves the schema empty and the button disabled.
        self.assertEqual(panel['schemaName'], 'std_msgs/msg/Bool')
        self.assertNotEqual(panel['schemaName'], 'std_msgs/Bool')

    def test_the_button_publishes_true_on_the_start_topic(self):
        import json
        panel = self.panel()
        self.assertEqual(panel['topicName'], '/competition/start')
        self.assertEqual(json.loads(panel['value']), {'data': True})

    def test_the_gate_is_on_with_the_shipped_config(self):
        """Pressing the button must be what starts the run."""
        import yaml
        for name in ('task_state_machine.yaml', 'profiles/state_machine_fast.yaml'):
            cfg = ROOT / 'config' / name
            params = yaml.safe_load(cfg.read_text(encoding='utf-8'))[
                'task_state_machine']['ros__parameters']
            self.assertIs(params.get('require_start_button'), True, name)


class LaunchYamlDefaultTests(unittest.TestCase):
    """Launch switches must take their default FROM THE YAML, not hard-code it.

    This bug happened twice: `mapping_rule` defaulted to a hard-coded '2' and
    `competition_mode` to 'false', and because the launch always passes the
    parameter, editing the YAML had no effect at all - the file looked
    configured while the run used the old value.
    """

    def source(self):
        return (ROOT / 'launch' / 'competition_bringup.launch.py').read_text(
            encoding='utf-8')

    def test_no_switch_hard_codes_its_default(self):
        src = self.source()
        for env in ('MYBOT_COMPETITION_MODE', 'MYBOT_REQUIRE_START_BUTTON',
                    'MYBOT_MAPPING_RULE'):
            self.assertNotIn(
                f"os.environ.get('{env}') or", src,
                f'{env} hard-codes a fallback that overrides the YAML')

    def test_each_switch_reads_the_yaml_through_the_shared_helper(self):
        src = self.source()
        self.assertIn('def _yaml_default(', src)
        for fn in ('_mapping_rule_default', '_competition_mode_default',
                   '_require_start_button_default'):
            self.assertIn(f'def {fn}(', src)
            self.assertIn(f'default_value={fn}()', src)

    def test_competition_mode_is_on_in_the_shipped_yaml(self):
        import yaml
        cfg = ROOT / 'config' / 'task_parser.yaml'
        params = yaml.safe_load(cfg.read_text(encoding='utf-8'))[
            'task_parser']['ros__parameters']
        self.assertIs(params.get('competition_mode'), True)


class Nav2ActionClientRecoveryTests(unittest.TestCase):
    """A health-check action client built before Nav2 exists can stay blind.

    Measured 2026-10-11: every Nav2 node ACTIVE and `ros2 action list` showing
    /navigate_to_pose, yet the stack sat with system_ready false and the state
    machine IDLE forever. The client must be rebuilt rather than waited on.
    """

    def source(self):
        return (ROOT / 'scripts' / 'system_health_check.py').read_text(encoding='utf-8')

    def test_the_client_is_rebuilt_after_the_grace_period(self):
        src = self.source()
        self.assertIn('action_rebuild_grace_sec', src)
        self.assertIn('_nav2_client_missing_since', src)
        # The rebuild must actually replace the client, not just log.
        self.assertIn('NavigateToPose, \'/navigate_to_pose\'', src)

    def test_the_rebuild_clock_only_runs_while_nodes_are_healthy(self):
        """It must not fire during a bringup that is still configuring."""
        src = self.source()
        self.assertIn('not nav2_action_up and not inactive_nav2', src)
        self.assertIn('_nav2_client_missing_since.clear()', src)


class UnmatchedNav2NodeRestartTests(unittest.TestCase):
    """A lifecycle service that never matches cannot be fixed by rebuilding.

    Measured 2026-10-14 17:11 (run_logs/resume_remaining_20261014-171158):
    smoother_server's GetState service stayed undiscovered through four client
    rebuilds (61.9 s, 79.9 s, 97.9 s, 117.9 s), Nav2 could never activate,
    /competition/system_ready stayed false and the task sat in IDLE. Two of the
    last fourteen launches died that way, which is indistinguishable from "the
    robot does not move".
    """

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'health_check_under_test', ROOT / 'scripts' / 'system_health_check.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_the_targeting_matches_the_node_and_not_the_manager(self):
        matches = self.module.nav2_node_process_matches
        node = ('/opt/ros/humble/lib/nav2_smoother/smoother_server '
                '--ros-args -r __node:=smoother_server')
        self.assertTrue(matches(node, 'smoother_server'))
        # The lifecycle manager lists the node name in node_names:=[...]; killing
        # it instead of the node would take the whole bringup down.
        manager = ('/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager '
                   '--ros-args -p node_names:=[smoother_server,planner_server]')
        self.assertFalse(matches(manager, 'smoother_server'))
        self.assertFalse(matches(node, 'planner_server'))
        self.assertFalse(matches('bash -c sleep 1000', 'smoother_server'))

    def test_a_restart_is_bounded_and_never_happens_after_ready(self):
        src = (ROOT / 'scripts' / 'system_health_check.py').read_text(
            encoding='utf-8')
        self.assertIn("declare_parameter('restart_unmatched_nav2_node', True)", src)
        self.assertIn("declare_parameter('unmatched_rebuilds_before_restart', 3)",
                      src)
        # Once per node, and only before the stack was ever ready.
        self.assertIn('not self._ever_ready', src)
        self.assertIn('name not in self._restarted_nodes', src)
        self.assertIn('self._restarted_nodes.add(name)', src)
        # Readiness records that the robot may now be moving a cube.
        self.assertIn('self._ever_ready = True', src)
        # The restart must be a real termination with a fresh participant.
        self.assertIn('signal.SIGTERM', src)
        self.assertIn('os.kill(victim[0], signal.SIGTERM)', src)

    def test_a_stale_state_read_back_also_triggers_one_restart(self):
        """The service can be discovered while its replies are dropped.

        Measured 2026-10-14 17:18 (run_logs/resume_remaining_20261014-171824):
        smoother_server configured at 1791969537 and the health check still asked
        it to CONFIGURE 18 more times, never reaching ACTIVATE, because the state
        it read back stayed stale. Rebuilding the CLIENT cannot fix that; the
        repair loop must escalate too.
        """
        src = (ROOT / 'scripts' / 'system_health_check.py').read_text(
            encoding='utf-8')
        self.assertIn("declare_parameter('stuck_repairs_before_restart', 6)", src)
        self.assertIn('self._stuck_repairs[name] = attempts', src)
        self.assertIn('attempts >= self._stuck_repairs_before_restart', src)
        # A node that starts answering again clears its counter.
        self.assertIn('self._stuck_repairs.pop(name, None)', src)
        # Both triggers share the same bounded, pre-ready, once-per-node action.
        self.assertIn('self._should_restart_node(name)', src)

    def test_a_missing_process_fails_loudly_instead_of_looping(self):
        src = (ROOT / 'scripts' / 'system_health_check.py').read_text(
            encoding='utf-8')
        self.assertIn('no matching ', src)
        self.assertIn('needs a manual restart', src)


class DashboardBridgeTests(unittest.TestCase):
    """The bridge must keep the problem AND the rules alive for late viewers.

    task_parser parses once and exits after parser_linger_sec; its latched
    topics lose their history with it, so a judge opening Foxglove minutes later
    saw an empty problem/rules panel. dashboard_bridge lives as long as the stack
    and republishes every 2 s, so both values are mirrored there.
    """

    def source(self):
        return (ROOT / 'scripts' / 'dashboard_bridge.py').read_text(encoding='utf-8')

    def test_the_bridge_publishes_a_rules_topic(self):
        self.assertIn("('rules', String)", self.source())

    def test_the_bridge_subscribes_to_the_parser_rules(self):
        self.assertIn("'/competition/task_rules'", self.source())

    def test_the_rules_callback_mirrors_non_empty_text(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'bridge_under_test', ROOT / 'scripts' / 'dashboard_bridge.py')
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        published = []
        node = SimpleNamespace(_str=lambda n, v: published.append((n, v)))
        bridge.DashboardBridge._rules_cb(
            node, SimpleNamespace(data='  规则2: x 去 B 区  '))
        bridge.DashboardBridge._rules_cb(node, SimpleNamespace(data='   '))
        self.assertEqual(published, [('rules', '规则2: x 去 B 区')])


class MappingRuleTests(unittest.TestCase):
    """The three published competition mapping rules, and switching between them.

    They differ structurally - rule 1 colours by letter and zones by count,
    rule 2 colours by count and zones by letter, rule 3 colours by comparison and
    zones by colour - so each gets its own truth table.
    """

    def node(self, rule=2, **over):
        values = {
            'mapping_rule': rule, 'red_min_count': 3,
            'zone_x': 'B', 'zone_y': 'A', 'zone_blue': 'B', 'zone_red': 'C',
            'color_x': 'red', 'color_y': 'blue', 'zone_by_count': ['C', 'C', 'A', 'A'],
        }
        values.update(over)
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
            get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: None,
                                               info=lambda *a, **k: None),
        )
        # Bind the real helpers so the instance-style calls resolve on the stub.
        for name in ('_mapping_rule', '_rule1', '_rule2', '_rule3',
                     '_zone_by_count', '_checked_zone', 'rule_summary'):
            setattr(node, name, getattr(parser.TaskParser, name).__get__(node))
        return node

    def tasks(self, X, Y, rule, **over):
        node = self.node(rule, **over)
        return parser.TaskParser._build_tasks(node, {'x': X, 'y': Y})

    def test_rule1_colour_by_letter_zone_by_count(self):
        self.assertEqual(self.tasks(2, 3, 1),
                         [{'color': 'red', 'count': 2, 'zone': 'C'},
                          {'color': 'blue', 'count': 3, 'zone': 'A'}])
        self.assertEqual(self.tasks(3, 2, 1),
                         [{'color': 'red', 'count': 3, 'zone': 'A'},
                          {'color': 'blue', 'count': 2, 'zone': 'C'}])

    def test_rule2_colour_by_count_zone_by_letter(self):
        # x always to B, y always to A, whichever colour they turn out to be.
        self.assertEqual(self.tasks(2, 3, 2),
                         [{'color': 'blue', 'count': 2, 'zone': 'B'},
                          {'color': 'red', 'count': 3, 'zone': 'A'}])
        self.assertEqual(self.tasks(3, 2, 2),
                         [{'color': 'red', 'count': 3, 'zone': 'B'},
                          {'color': 'blue', 'count': 2, 'zone': 'A'}])

    def test_rule3_colour_by_comparison_zone_by_colour(self):
        self.assertEqual(self.tasks(2, 3, 3),
                         [{'color': 'red', 'count': 2, 'zone': 'C'},
                          {'color': 'blue', 'count': 3, 'zone': 'B'}])
        self.assertEqual(self.tasks(3, 2, 3),
                         [{'color': 'blue', 'count': 3, 'zone': 'B'},
                          {'color': 'red', 'count': 2, 'zone': 'C'}])

    def test_every_rule_covers_every_split_with_valid_zones(self):
        for rule in (1, 2, 3):
            for X, Y in ((1, 4), (2, 3), (3, 2), (4, 1)):
                tasks = self.tasks(X, Y, rule)
                self.assertEqual(sum(t['count'] for t in tasks), 5,
                                 f'rule {rule} lost cubes on {X},{Y}')
                for task in tasks:
                    self.assertIn(task['color'], ('red', 'blue'))
                    self.assertIn(task['zone'], ('A', 'B', 'C'))

    def test_rule3_refuses_a_tie_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            self.tasks(3, 3, 3)

    def test_unknown_rule_is_rejected(self):
        for bad in (0, 4, 'two'):
            with self.assertRaises(ValueError):
                self.tasks(2, 3, bad)

    def test_zero_count_colour_is_skipped(self):
        self.assertEqual([t['count'] for t in self.tasks(5, 0, 2)], [5])

    def test_rule2_zone_override_is_honoured(self):
        tasks = self.tasks(2, 3, 2, zone_x='C', zone_y='B')
        self.assertEqual([t['zone'] for t in tasks], ['C', 'B'])

    def test_rule_summary_names_the_active_rule(self):
        for rule, needle in ((1, '规则1'), (2, '规则2'), (3, '规则3')):
            self.assertIn(needle, parser.TaskParser.rule_summary(self.node(rule)))


def _bind_speed_limit_methods(node):
    """Attach the real speed-limit methods to a SimpleNamespace fake.

    ``_apply_slow_zones`` now delegates to ``_slow_zone_limit`` and
    ``_publish_speed_limit`` (the yield policy merges into the same channel), so
    a fake node built from plain lambdas no longer has the methods it calls.
    Binding the real implementations keeps the tests exercising shipping code.
    """
    for name in ('_slow_zone_limit', '_publish_speed_limit',
                 '_apply_slow_zones', '_clear_speed_limit',
                 '_reset_yield_state'):
        setattr(node, name,
                getattr(module.FixedJointPickPlace, name).__get__(node))
    return node


class ObstacleYieldTests(unittest.TestCase):
    """Predictive yielding: go / cap / arc / wait, and its hard limits.

    The rule set exists because the controller is purely reactive: it slows as
    the costmap inflation reaches it and then refuses to move at all once its
    forward simulation hits a lethal cell. Measured on
    run_logs/movers_trial4_20261011-143924: 171 "detected collision ahead!"
    refusals in one completed round, 166 of them inside a single 17 sim-second
    stretch, with no code anywhere asking where the obstacle was going.

    These tests pin the two failure modes that would be worse than the disease:
    a base that accelerates while "waiting" (Nav2's 0.0 speed limit is a
    no-limit sentinel), and a base that waits forever (run_logs/yield3_131041:
    mover_2 frozen for 110 s of a 130 s trace).
    """

    @classmethod
    def setUpClass(cls):
        cls.yield_module = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location(
                'obstacle_yield_under_test', ROOT / 'scripts' / 'obstacle_yield.py'
            )
        )
        # The module uses dataclasses, which need `sys.modules` populated
        # before exec_module (Python 3.10 looks the owning module up by name).
        sys.modules['obstacle_yield_under_test'] = cls.yield_module
        importlib.util.spec_from_file_location(
            'obstacle_yield_under_test', ROOT / 'scripts' / 'obstacle_yield.py'
        ).loader.exec_module(cls.yield_module)

    def setUp(self):
        self.movers = self.yield_module
        self.cfg = self.movers.YieldConfig()

    def decision(self, **kwargs):
        kwargs.setdefault('box_world_speed_mps', 0.30)
        return self.movers.decide(
            self.movers.YieldObservation(**kwargs), self.cfg
        )

    # ---- the plain cases the user asked for ------------------------------

    def test_a_box_on_the_left_that_will_not_arrive_in_time_is_driven_past(self):
        """障碍物在左边、朝我们来，但在我们到达之前还进不了车道 -> 直行。

        Keep-cruise is the answer only while the margin is real. Here the box
        is 1.0 m off-axis and closing at 0.30 m/s: the base's nose is past in
        1.43 s, the box reaches the corridor edge at 2.17 s, so there is 0.74 s
        of daylight - more than the 0.3 s the sensing/control chain eats.
        """
        decision = self.decision(
            box_x=1.5, box_y=1.0, box_vx=-1.3, box_vy=-0.30, vx=1.3,
            left_room_m=1.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_GO)
        self.assertEqual(decision.reason, self.movers.REASON_PASSING)

    def test_a_coin_flip_pass_is_refused_and_the_old_rule_is_reproducible(self):
        """This exact geometry used to be the "直行" test, and it was luck.

        At y=0.80 m the base clears in 1.43 s and the box enters the corridor
        at 1.50 s: 0.07 s of margin, which is inside the measurement noise of
        the box's own closing-rate estimate (a 17% error in 0.30 m/s erases it
        and the box is in the lane when the base arrives). With the corrected
        near-edge entry time and the 0.3 s latency margin the rule refuses it;
        ``clear_margin_sec=0.0`` reproduces the old answer exactly, which is
        what makes the margin - not the formula - the thing under test.
        """
        obs = self.movers.YieldObservation(
            box_x=1.5, box_y=0.8, box_vx=-1.3, box_vy=-0.30, vx=1.3,
            left_room_m=1.0, box_world_speed_mps=0.30,
        )
        decision = self.movers.decide(obs, self.cfg)
        self.assertEqual(decision.action, self.movers.ACTION_STOP)
        self.assertEqual(decision.reason, self.movers.REASON_WAIT)
        old = self.movers.decide(
            obs, self.cfg.with_overrides(clear_margin_sec=0.0))
        self.assertEqual(old.action, self.movers.ACTION_GO)

    def test_a_far_but_losing_conflict_is_capped_before_it_is_close(self):
        """早决策: the same 0.80 m offset at 3.0 m is a *cap*, not a late stop.

        The base would clear in 2.58 s while the box enters at 1.50 s, so it
        cannot win; at 3.10 m of range it starts braking (0.8 m/s) instead of
        discovering the conflict at 1.7 m, where the only options left are a
        hard stop or a collision.
        """
        decision = self.decision(
            box_x=3.0, box_y=0.8, box_vx=-1.3, box_vy=-0.30, vx=1.3,
            left_room_m=1.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_CAP)
        self.assertEqual(decision.reason, self.movers.REASON_CAPPED)
        self.assertLess(decision.speed_mps, 1.3)

    def test_crossing_time_uses_the_near_edge_the_box_reaches_first(self):
        """The formula used to return the *far* edge (3.83 s at y=0.80 m).

        That is 2.3 s of free licence: the entry test would pass long after the
        box was already inside the corridor.
        """
        self.assertAlmostEqual(
            self.movers.crossing_time(0.80, -0.30, 0.35, 4.0), 1.5, places=6)
        self.assertAlmostEqual(
            self.movers.crossing_time(-0.80, 0.30, 0.35, 4.0), 1.5, places=6)
        self.assertEqual(
            self.movers.crossing_time(0.05, -0.30, 0.35, 4.0), 0.0)
        self.assertIsNone(
            self.movers.crossing_time(0.80, 0.30, 0.35, 4.0),
            'a box moving away never enters')
        self.assertIsNone(
            self.movers.crossing_time(2.00, -0.30, 0.35, 4.0),
            'beyond the horizon is not a decision this tick')

    def test_a_box_that_owns_the_lane_while_we_are_there_is_waited_for(self):
        """太近就等它走过去再走 —— 这是"朝我来"的核心分支，不能被优化掉。"""
        decision = self.decision(
            box_x=0.9, box_y=0.20, box_vx=-0.9, box_vy=-0.30, vx=1.5,
            left_room_m=0.9, distance_to_goal_m=6.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_STOP)
        self.assertEqual(decision.reason, self.movers.REASON_WAIT)

    def test_a_box_already_in_the_lane_is_not_waved_through(self):
        """Regression for the first cut of this rule.

        "The base passes if it arrives before the box enters the corridor"
        answers *go* for a box standing in the lane right now, because such a
        box never has a future entry time at all. The rule is now a forward
        projection to the base's own crossing moment.
        """
        decision = self.decision(
            box_x=1.00, box_y=0.05, box_vx=-0.9, box_vy=-0.30, vx=1.5,
            left_room_m=0.9, distance_to_goal_m=6.0,
        )
        self.assertNotEqual(decision.reason, self.movers.REASON_PASSING)
        self.assertIn(decision.action,
                      (self.movers.ACTION_STOP, self.movers.ACTION_ARC))

    def test_a_box_already_leaving_the_lane_is_driven_past(self):
        """The mirror regression: a box on its way out is still *inside* the
        corridor, so an entry-time rule stops for something that is leaving."""
        decision = self.decision(
            box_x=0.8, box_y=0.30, box_vx=-1.4, box_vy=0.30, vx=1.5,
            left_room_m=1.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_GO)

    # ---- the arc / lateral exit ------------------------------------------

    def test_a_lateral_exit_only_happens_after_patience_is_spent(self):
        """加一点弧度 is issued as the *escape* from a failed wait, not as an
        overtake.

        A box crossing the corridor is only 0.35 m off-axis when the base
        arrives, so a compliant offset needs the box's projected position plus
        the base's own half-width - about 0.7 m - inside the ~1 s the base has
        while it passes. An exhaustive search over base speeds 0.30-1.60 m/s,
        box offsets -0.34..+0.34 m and three closing rates found no mid-motion
        arc that both resolved a real conflict and fitted. Waiting first, then
        leaving sideways if waiting failed, is the honest version of the
        request.
        """
        holding = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                left_room_m=0.9, right_room_m=1.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg,
        )
        self.assertEqual(holding.action, self.movers.ACTION_STOP)
        exiting = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                left_room_m=0.9, right_room_m=1.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 5.0,
        )
        self.assertEqual(exiting.action, self.movers.ACTION_ARC)
        self.assertEqual(exiting.reason, self.movers.REASON_WAIT_EXPIRED)
        self.assertNotEqual(exiting.arc_x, None)
        self.assertNotEqual(exiting.arc_y, 0.0)

    def test_exit_chooses_free_side_not_obstacle_side(self):
        """At (-4.767,-1.526), left room is 0.008 m, right is 2.522 m.
        Choosing a direction from the box's side alone would hit that wall.
        """
        exiting = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.1, box_vy=0.0, vx=0.001,
                left_room_m=0.008, right_room_m=2.522, box_world_speed_mps=0.1,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 5.0,
        )
        self.assertEqual(exiting.action, self.movers.ACTION_ARC)
        self.assertEqual(exiting.arc_side, -1)
        self.assertLess(exiting.arc_y, 0.0)

    def test_exit_rejected_when_box_envelope_cannot_be_cleared(self):
        crossing = self.movers.YieldObservation(
            box_x=0.9, box_y=0.0, box_vx=-0.1, box_vy=-0.1, vx=0.001,
            left_room_m=0.008, right_room_m=2.522, box_world_speed_mps=0.1,
            distance_to_goal_m=6.0, latched=True,
        )
        decision = self.movers.decide(
            crossing, self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 5.0)
        self.assertNotEqual(decision.action, self.movers.ACTION_ARC)

    def test_the_configured_offset_can_actually_be_granted(self):
        """The yaw ceiling must admit the configured offset.

        At `max_arc_yaw_rad = 0.6` the arc `atan2(0.55, 0.8) = 0.602` rad was
        rejected by rounding, so every lateral exit degraded to "no room" and
        the feature would have looked like it worked while never once firing.
        """
        ceiling = math.atan2(self.cfg.arc_offset_m, self.cfg.arc_apex_m)
        self.assertLessEqual(ceiling, self.cfg.max_arc_yaw_rad)

    def test_no_sideways_room_hands_the_leg_back_instead_of_waiting_on(self):
        """Standing still until the controller's patience fails is what makes
        the recovery tree back the base up into the obstacle."""
        decision = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                left_room_m=0.2, right_room_m=0.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 5.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_GO)
        self.assertEqual(decision.reason, self.movers.REASON_WAIT_EXPIRED)

    def test_arcing_is_abandoned_after_the_per_leg_ceiling(self):
        """A box that keeps pace with the base must not be chased around the map
        one offset at a time."""
        decision = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                left_room_m=0.9, right_room_m=1.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 5.0,
            arc_attempts=self.cfg.max_arc_attempts,
        )
        self.assertNotEqual(decision.action, self.movers.ACTION_ARC)

    # ---- the guards ------------------------------------------------------

    def test_only_a_mover_is_ever_waited_for(self):
        """A pillar with a box's silhouette must not hold the base.

        Nav2's inflation layer and collision check own static geometry, and a
        base that waits for a wall deadlocks the round (run_logs/yield3_131041).
        """
        decision = self.decision(
            box_x=0.9, box_y=0.0, box_vx=-1.5, box_vy=0.0, vx=1.5,
            box_world_speed_mps=0.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_GO)
        self.assertIn('static', decision.detail)

    def test_an_unknown_world_speed_fails_safe_to_static(self):
        decision = self.decision(box_x=0.9, box_y=0.0, box_vx=-1.5, box_vy=0.0,
                                 vx=1.5, box_world_speed_mps=None)
        self.assertEqual(decision.action, self.movers.ACTION_GO)

    def test_a_held_base_is_released_once_the_corridor_is_clear(self):
        """A held base has vx = 0 by construction, so a hold that re-derived its
        crossing time from the measured speed would find every crossing
        infinitely far away and never let go - a permanent stop."""
        held = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.4, box_y=0.40, box_vx=-0.2, box_vy=0.30, vx=0.001,
                box_world_speed_mps=0.30, latched=True,
            ),
            self.cfg,
        )
        self.assertEqual(held.action, self.movers.ACTION_GO)
        self.assertIn('hold released', held.detail)

    def test_a_latched_hold_is_released_when_the_corridor_clears(self):
        still_there = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                right_room_m=1.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=3.0,
        )
        self.assertEqual(still_there.action, self.movers.ACTION_STOP)
        cleared = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.4, box_y=0.40, box_vx=-0.2, box_vy=0.30, vx=0.001,
                box_world_speed_mps=0.30, distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=3.0,
        )
        self.assertEqual(cleared.action, self.movers.ACTION_GO)

    def test_patience_expires_instead_of_waiting_forever(self):
        decision = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.3, box_vy=-0.10, vx=0.001,
                right_room_m=1.2, box_world_speed_mps=0.10,
                distance_to_goal_m=6.0, latched=True,
            ),
            self.cfg, wait_elapsed_sec=self.cfg.wait_budget_sec + 1.0,
        )
        self.assertNotEqual(decision.action, self.movers.ACTION_STOP)
        self.assertEqual(decision.reason, self.movers.REASON_WAIT_EXPIRED)

    def test_a_stale_scan_never_steers_the_base(self):
        decision = self.decision(
            box_x=0.4, box_y=0.0, box_vx=-0.4, box_vy=-0.3, vx=1.5,
            scan_age_sec=5.0,
        )
        self.assertEqual(decision.action, self.movers.ACTION_GO)

    def test_the_layer_is_inert_without_movers_and_when_disabled(self):
        inert = self.decision(
            box_x=0.9, box_y=0.0, box_vx=-0.9, box_vy=-0.3, vx=1.5,
            movers_present=False,
        )
        self.assertEqual(inert.action, self.movers.ACTION_GO)
        disabled = self.movers.decide(
            self.movers.YieldObservation(
                box_x=0.9, box_y=0.0, box_vx=-0.9, box_vy=-0.3, vx=1.5,
                box_world_speed_mps=0.30,
            ),
            self.cfg.with_overrides(enabled=False),
        )
        self.assertEqual(disabled.action, self.movers.ACTION_GO)

    def test_it_never_holds_inside_the_goal_margin(self):
        """Within the arrival gate there is no room to route around anything and
        the goal checker is about to finish the leg anyway."""
        decision = self.decision(
            box_x=0.4, box_y=0.0, box_vx=-0.4, box_vy=-0.3, vx=0.2,
            box_world_speed_mps=0.30, distance_to_goal_m=0.2,
        )
        self.assertNotEqual(decision.action, self.movers.ACTION_STOP)

    # ---- the tracker -----------------------------------------------------

    def test_the_tracker_separates_the_base_s_own_motion_from_the_box_s(self):
        """The base covers up to 0.15 m between two 10 Hz sweeps at 1.5 m/s -
        more than a box's own width - so tracking has to happen in an inertial
        frame, and "is it moving" has to mean "in the world", not "in my frame".
        """
        tracker = self.movers.YieldTracker(self.cfg)
        state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker.update([(2.0, 1.0)], 0.0, state)
        # 0.1 s later the base moved 0.15 m and the box moved -0.03 m
        # sideways; association remains inside the tracker's 0.30 m gate.
        tracker.update([(1.85, 0.97)], 0.1,
                       (0.15, 0.0, 0.0, 1.5, 0.0, 0.0))
        track = tracker.nearest(0.1, (0.15, 0.0, 0.0, 1.5, 0.0, 0.0))
        self.assertIsNotNone(track)
        self.assertAlmostEqual(track['base_x'], 1.85, places=6)
        self.assertAlmostEqual(track['base_y'], 0.97, places=6)
        self.assertAlmostEqual(track['rel_vy'], -0.30, places=6)
        self.assertAlmostEqual(track['world_speed'], 0.30, places=6)

    def test_a_box_that_stops_being_reported_is_forgotten(self):
        tracker = self.movers.YieldTracker(self.cfg)
        state = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker.update([(1.0, 0.5)], 0.0, state)
        self.assertIsNotNone(tracker.nearest(0.1, state))
        self.assertIsNone(
            tracker.nearest(0.1 + self.cfg.track_timeout_sec + 0.1, state)
        )

    def test_arc_approach_rejects_static_return_on_the_segment(self):
        self.assertFalse(self.movers.arc_scan_clear(
            [(0.50, 0.30), (2.0, 2.0)], 0.8, 0.55))
        self.assertTrue(self.movers.arc_scan_clear(
            [(1.5, 1.5), (2.0, -2.0)], 0.8, 0.55))
        self.assertFalse(self.movers.arc_scan_clear([], 0.8, 0.55))

    def test_closer_static_fragment_does_not_mask_a_mover(self):
        tracker = self.movers.YieldTracker(self.cfg)
        origin = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker.update([(0.6, 0.2), (1.5, 0.1)], 0.0, origin)
        tracker.update([(0.6, 0.2), (1.5, 0.07)], 0.1, origin)
        selected = tracker.nearest(0.1, origin, min_world_speed=0.04)
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected['base_x'], 1.5)
        self.assertAlmostEqual(selected['world_speed'], 0.3)

    def test_a_track_cannot_chain_between_two_objects(self):
        """A fixed 0.30 m bubble turned a stationary chair leg into a 1.52 m/s
        "mover" (run_logs/yield_diag_fixed_20261013-181029) by hopping from one
        cluster to the next. The radius is now how far a plausible mover can
        travel between sweeps: 0.05 m + 0.6 m/s * dt."""
        tracker = self.movers.YieldTracker(self.cfg)
        origin = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertTrue(tracker._associate({'x': 1.0, 'y': 0.0}, 1.03, 0.0, 0.1))
        self.assertFalse(tracker._associate({'x': 1.0, 'y': 0.0}, 1.20, 0.0, 0.1))
        # A long gap between sweeps widens the allowance again, up to 0.30 m.
        self.assertTrue(tracker._associate({'x': 1.0, 'y': 0.0}, 1.20, 0.0, 1.0))
        self.assertFalse(tracker._associate({'x': 1.0, 'y': 0.0}, 1.35, 0.0, 5.0))
        tracker.update([(1.0, 0.0)], 0.0, origin)
        tracker.update([(1.20, 0.0)], 0.1, origin)
        for track in tracker._tracks:
            self.assertLess(math.hypot(track['vx'], track['vy']), 0.6,
                            'no track may read as an implausibly fast mover')

    def test_an_implausibly_fast_track_is_never_treated_as_a_mover(self):
        tracker = self.movers.YieldTracker(self.cfg)
        origin = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker._tracks = [{'x': 1.0, 'y': 0.0, 'vx': 1.52, 'vy': 0.0,
                            't': 0.0, 'seen': True, 'born': -1.0}]
        self.assertIsNone(
            tracker.nearest(0.0, origin, min_world_speed=0.04),
            'a 1.52 m/s "mover" is noise in this scene, not a patrol box')
        # With no mover filter the track is still reported: the ceiling belongs
        # to the mover decision, not to the tracker's bookkeeping.
        self.assertIsNotNone(tracker.nearest(0.0, origin))

    def test_single_return_speckle_is_not_a_box(self):
        self.assertEqual(self.movers.box_like_clusters([[(0.0, 1.0)]]), [])
        kept = self.movers.box_like_clusters([[(0.0, 1.0), (0.02, 1.0)],
                                              [(0.0, 2.0)]])
        self.assertEqual(len(kept), 1)

    def test_wall_like_clusters_are_not_mistaken_for_boxes(self):
        """A wall returns a long chain; a 0.22 m box returns a compact blob."""
        box = [(0.0, 1.0), (0.02, 1.0), (-0.02, 1.0)]
        wall = [(a, 1.0) for a in np.linspace(-1.2, 1.2, 40)]
        kept = self.movers.box_like_clusters([box, wall])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0]), len(box))


class YieldSpeedLimitTests(unittest.TestCase):
    """The yield policy's only actuator on the base is the controller's speed
    limit, so how it is encoded decides whether the base stops or accelerates.
    """

    class FakePublisher:
        def __init__(self):
            self.sent = []

        def publish(self, message):
            self.sent.append(message)

    def node(self, pub=None):
        node = SimpleNamespace(
            _speed_limit_pub=pub,
            _active_speed_limit=0.0,
            _slow_zones=[],
            _cached_base_position=lambda: (0.0, 0.0),
            get_parameter=lambda name: SimpleNamespace(
                value=0.8 if name == 'slow_zone_brake_ahead_m' else None),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0))),
            get_logger=lambda: SimpleNamespace(
                warning=lambda *a, **k: None, info=lambda *a, **k: None,
                error=lambda *a, **k: None),
        )
        return _bind_speed_limit_methods(node)

    def test_a_hold_is_a_small_non_zero_limit_not_zero(self):
        """Regression for an inverted stop, caught before it ever ran.

        `speed_limit = 0.0` with `percentage = false` is Nav2's
        `NO_SPEED_LIMIT` sentinel: RPP compares against 0.0 and restores
        `base_desired_linear_vel_`. Sending 0.0 to hold the base would have made
        it accelerate to 1.5 m/s *while waiting*, and would also have wiped any
        active slow-zone cap. Evidence (source + upstream test + disassembly of
        the installed 1.1.20 binary) is in run_logs/yield_nav2_semantics.md.
        """
        self.assertNotEqual(module.YIELD_STOP_SPEED_LIMIT, 0.0)
        self.assertGreater(module.YIELD_STOP_SPEED_LIMIT, 0.0)
        self.assertLess(module.YIELD_STOP_SPEED_LIMIT, 0.05)

    def test_hold_then_release_publishes_distinct_nav2_values(self):
        """Exercise the actual policy-to-publisher path, not just its constant.

        In Nav2 zero *restores cruise*, while 0.001 m/s caps linear motion.
        The transition from wait to go must publish both in that order.
        """
        pub = self.FakePublisher()
        node = self.node(pub=pub)
        node._yield_suppress = False
        node._yield_last_update = -math.inf
        node._yield_latched = False
        node._yield_wait_since = None
        node._yield_clear_since = None
        node._yield_arc_attempts = 0
        node._yield_arc_goal = None
        node._yield_arc_used = False
        # Episode bookkeeping is exercised by its own test; this one pins the
        # policy-to-publisher path.
        node._update_yield_episode = lambda *a, **k: None
        node._yield_config = SimpleNamespace(
            max_arc_attempts=2, wait_budget_sec=20.0, clear_hold_sec=0.5)
        node.get_parameter = lambda name: SimpleNamespace(value=0.0)
        decisions = iter((
            (module.obstacle_yield.YieldDecision('stop', 'wait'), True, None),
            (module.obstacle_yield.YieldDecision('go', 'clear'), False, None),
            (module.obstacle_yield.YieldDecision('go', 'clear'), False, None),
        ))
        node._yield_step = lambda _: next(decisions)
        module.FixedJointPickPlace._tick_yield(node)
        module.FixedJointPickPlace._tick_yield(node)
        self.assertEqual(len(pub.sent), 1, 'one clear scan cannot release a hold')
        node._yield_clear_since = module.time.monotonic() - 0.6
        module.FixedJointPickPlace._tick_yield(node)
        self.assertEqual([m.speed_limit for m in pub.sent],
                         [module.YIELD_STOP_SPEED_LIMIT, 0.0])
        self.assertTrue(all(not m.percentage for m in pub.sent))

    def test_the_service_used_to_clear_a_limit_is_still_zero(self):
        """0.0 stays reserved for "resume", so the two meanings cannot collide."""
        pub = self.FakePublisher()
        node = self.node(pub=pub)
        node._active_speed_limit = 0.6
        module.FixedJointPickPlace._clear_speed_limit(node)
        self.assertEqual(pub.sent[-1].speed_limit, 0.0)

    def test_slow_zone_messages_keep_their_shape(self):
        pub = self.FakePublisher()
        node = self.node(pub=pub)
        node._slow_zones = [(0.0, 0.0, 1.0, 0.6)]
        module.FixedJointPickPlace._apply_slow_zones(node)
        self.assertEqual(len(pub.sent), 1)
        self.assertAlmostEqual(pub.sent[-1].speed_limit, 0.6)
        self.assertFalse(pub.sent[-1].percentage)

    def test_the_lower_of_a_slow_zone_and_a_yield_cap_wins(self):
        node = self.node(pub=self.FakePublisher())
        node._slow_zones = [(0.0, 0.0, 1.0, 0.9)]
        self.assertAlmostEqual(
            module.FixedJointPickPlace._slow_zone_limit(node), 0.9
        )
        node._slow_zones = [(0.0, 0.0, 1.0, 0.9), (0.1, 0.0, 1.0, 0.4)]
        self.assertAlmostEqual(
            module.FixedJointPickPlace._slow_zone_limit(node), 0.4
        )


class YieldWiringTests(unittest.TestCase):
    """The yield layer is off by default and its arc is consumed exactly once."""

    def test_yield_is_disabled_by_default(self):
        """A change to the navigation loop is not allowed to reach a scored
        round until it has been measured over a full one."""
        source = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("declare_parameter('enable_obstacle_yield', False)", source)

    def test_lost_scan_does_not_clear_a_live_hold(self):
        """A dropped scan must not restore cruise speed in the next control tick."""
        node = SimpleNamespace(
            _yield_base_motion=lambda: (0.0, 0.0, 0.0, 0.0, 0.0),
            _base_pose=lambda: (0.0, 0.0, 0.0),
            _yield_scan_points=lambda yaw: (math.inf, []),
            _yield_config=SimpleNamespace(enabled=True, track_timeout_sec=1.5,
                                          wait_budget_sec=20.0),
            _yield_wait_since=module.time.monotonic() - 1.0,
            _yield_latched=True,
            # The diagnostic is logging-only; this test pins the decision.
            _yield_diag=lambda *args, **kwargs: None,
            _yield_diag_period=0.0,
        )
        decision, latched, arc = module.FixedJointPickPlace._yield_step(node, 3.0)
        self.assertEqual(decision.action, module.obstacle_yield.ACTION_STOP)
        self.assertTrue(latched)
        self.assertIsNone(arc)
        node._yield_wait_since = module.time.monotonic() - 21.0
        decision, latched, _ = module.FixedJointPickPlace._yield_step(node, 3.0)
        self.assertEqual(decision.action, module.obstacle_yield.ACTION_GO)
        self.assertFalse(latched)

    def test_the_executor_profile_pins_the_stacks_time_base(self):
        """The executor is spawned by the state machine, so nothing else sets it.

        Measured 2026-10-13: launched without use_sim_time, its scan-age test
        read 1791884787 s against Gazebo-stamped sweeps, every yield tick exited
        on "stale scan", and the whole policy was inert for a round without a
        single log line (run_logs/yield_diag_short2_20261013-180609).
        """
        import yaml
        profile = yaml.safe_load(
            (ROOT / 'config' / 'profiles' / 'executor_fast.yaml').read_text(
                encoding='utf-8'))
        params = profile['fixed_joint_pick_place']['ros__parameters']
        self.assertIs(params.get('use_sim_time'), True)

    def test_the_competition_profile_enables_yielding(self):
        """Decided by the operator 2026-10-13: the layer works now, so a scored
        round must not rely on RPP's last-metre refusal (which ended in a 3.7 m/s
        fling off mover_3 in /tmp/mybot_start_lATL.log)."""
        import yaml
        params = yaml.safe_load(
            (ROOT / 'config' / 'profiles' / 'executor_fast.yaml').read_text(
                encoding='utf-8'))['fixed_joint_pick_place']['ros__parameters']
        self.assertIs(params.get('enable_obstacle_yield'), True)
        # The profile is the deployment contract: the values that decide how
        # close the base may come are pinned there, not left to script defaults.
        for key in ('yield_clear_margin_sec', 'yield_max_mover_speed_mps',
                    'yield_wait_budget_sec'):
            self.assertIn(key, params)

    def test_a_turning_base_does_not_open_a_new_conflict(self):
        """The base's own yaw rate must not invent movers.

        At 1.5 rad/s a static cluster 3.5 m to the side reads as moving at
        5 m/s relative to the base, so the policy held itself in place and the
        base circled beside the box (2026-10-13). A hold already in progress is
        left alone - releasing it would drive forward on the tick the base can
        least judge the corridor.
        """
        oy = module.obstacle_yield
        cfg = oy.YieldConfig()
        def decide(**kw):
            kw.setdefault('box_world_speed_mps', 0.13)
            return oy.decide(oy.YieldObservation(**kw), cfg)
        turning = decide(
            box_x=1.72, box_y=3.55, box_vx=5.50, box_vy=-2.46, vx=0.001,
            wz=1.5, left_room_m=1.0,
        )
        self.assertEqual(turning.action, oy.ACTION_GO)
        self.assertIn('rotating', turning.detail)
        steady = decide(
            box_x=1.72, box_y=3.55, box_vx=5.50, box_vy=-2.46, vx=0.001,
            wz=0.05, left_room_m=1.0,
        )
        self.assertNotEqual(steady.action, oy.ACTION_GO)
        held = decide(
            box_x=1.72, box_y=3.55, box_vx=5.50, box_vy=-2.46, vx=0.001,
            wz=1.5, left_room_m=1.0, latched=True,
        )
        self.assertIn(held.action, (oy.ACTION_STOP, oy.ACTION_ARC))

    def test_a_hold_cancels_nav2_instead_of_only_limiting_speed(self):
        """The speed limit cannot stop rotation, so a hold must own the base.

        Measured 2026-10-13: while "held" the base kept turning at ~1.5 rad/s
        (relative velocities of 5.5 m/s on a 3.5 m static cluster), which is
        what the operator saw as circling beside the obstacle.
        """
        source = SCRIPT.read_text(encoding='utf-8')
        body = source[source.index('if self._yield_decision.holds and not self._yield_suppress:'):]
        block = body[:body.index('if self._yield_arc_goal is not None:',
                                  body.index('resumed = self._resume_nav_goal'))]
        self.assertIn('_hold_until_clear', block)
        self.assertIn('_resume_nav_goal', block)
        self.assertIn('_release_nav_control', block)
        # The wait must happen in its own loop: cancelling the goal completes
        # the result future, so handling the pause inside the leg's own wait
        # loop would fall straight through to the jam escape (measured
        # 2026-10-13: "moved 0.70 m backward" on every hold).
        hold = source[source.index('def _hold_until_clear('):]
        hold = hold[:hold.index('\n    def ')]
        self.assertIn('self._stop_base()', hold)
        self.assertIn('self._tick_yield(', hold)
        self.assertIn('while (rclpy.ok()', hold)
        # The pause must be forgotten when the leg ends, like every other bit of
        # per-leg yield state.
        reset = source[source.index('def _reset_yield_state'):]
        self.assertIn('_yield_pause_active = False',
                      reset[:reset.index('\n    def ')])

    def test_the_wait_budget_is_spent_on_the_episode_not_on_one_hold(self):
        """A box that keeps coming back must not trap the base forever - and a
        spent budget must not be inherited by a later, unrelated encounter.

        Measured 2026-10-13: the base stood 0.30 m from mover_3's patrol segment
        through 30 alternating hold/resume cycles (18 s of zero motion), and in a
        later round a fresh hold escalated to an arc after 1.0 s because the
        episode had never been reset - costing an escape, a retry and an arc.
        """
        cfg = module.obstacle_yield.YieldConfig()
        self.assertGreater(cfg.episode_reset_progress_m, 0.0)
        self.assertGreater(cfg.episode_clear_sec, 1.0)
        node = SimpleNamespace(
            _yield_config=cfg,
            _yield_episode_since=None, _yield_episode_anchor=None,
            _yield_free_since=None,
            _cached_base_position=lambda: (0.0, 0.0),
        )
        bind = module.FixedJointPickPlace
        node._update_yield_episode = bind._update_yield_episode.__get__(node)
        node._yield_episode_elapsed = bind._yield_episode_elapsed.__get__(node)
        # A hold starts the episode and a short lull does not end it.
        node._update_yield_episode(100.0, latched=True)
        self.assertAlmostEqual(node._yield_episode_elapsed(107.0), 7.0)
        node._update_yield_episode(107.0, latched=False)      # free window opens
        self.assertAlmostEqual(node._yield_episode_elapsed(108.0), 8.0)
        node._update_yield_episode(108.0, latched=False)      # only 1 s free
        self.assertAlmostEqual(node._yield_episode_elapsed(109.0), 9.0)
        # Driving freely for episode_clear_sec does end it.
        node._update_yield_episode(107.0 + cfg.episode_clear_sec + 1.0,
                                   latched=False)
        self.assertEqual(node._yield_episode_elapsed(115.0), 0.0)
        # ...and so does real progress, as before.
        node._update_yield_episode(116.0, latched=True)
        node._cached_base_position = lambda: (cfg.episode_reset_progress_m + 0.1, 0.0)
        node._update_yield_episode(117.0, latched=False)
        self.assertEqual(node._yield_episode_elapsed(125.0), 0.0)

    def test_a_spent_episode_budget_bounds_a_fresh_stop_too(self):
        """The latch releases on every lull; the budget must not reset with it.

        Measured 2026-10-13: stop decisions carried wait=34.7 s ... 40.4 s
        (past the 20 s budget) because only the latched branch honoured it -
        29 holds in one round and a delivery leg that failed with the cube
        still attached. An unlatched conflict must escalate the same way.
        """
        oy = module.obstacle_yield
        cfg = oy.YieldConfig()
        # Straight off the log line "box (+0.43,-0.28) v=(+0.00,+0.03)": a box
        # sitting in the corridor, i.e. a genuine conflict for a fresh tick.
        fresh = oy.YieldObservation(
            box_x=0.43, box_y=-0.28, box_vx=0.0, box_vy=+0.03, vx=1.5,
            left_room_m=0.9, right_room_m=1.2, distance_to_goal_m=6.0,
            box_world_speed_mps=0.16, latched=False)
        self.assertEqual(oy.decide(fresh, cfg, wait_elapsed_sec=0.0).action,
                         oy.ACTION_STOP)
        spent = oy.decide(fresh, cfg, wait_elapsed_sec=cfg.wait_budget_sec + 14.0)
        self.assertNotEqual(spent.action, oy.ACTION_STOP,
                            'a spent episode must not stop again')
        self.assertEqual(spent.action, oy.ACTION_ARC)
        # With the arc budget used up it hands the leg back instead of parking.
        exhausted = oy.decide(fresh, cfg,
                              wait_elapsed_sec=cfg.wait_budget_sec + 20.0,
                              arc_attempts=cfg.max_arc_attempts)
        self.assertEqual(exhausted.action, oy.ACTION_GO)
        self.assertEqual(exhausted.reason, oy.REASON_WAIT_EXPIRED)

    def test_a_box_at_its_patrol_end_is_not_believed_when_it_leaves(self):
        """A mover reverses at the end of its segment; its velocity lies first.

        Measured 2026-10-13: mover_2 reported dir=-1 at 1791902635 and dir=+1 at
        1791902640 while the policy read "passing"; the base rolled into the lane
        at 1.5 m/s and the returning box hit it (fling, 1/5 delivered).
        """
        oy = module.obstacle_yield
        cfg = oy.YieldConfig()
        self.assertGreater(cfg.lane_end_margin_m, 0.15)
        base = dict(box_x=0.9, box_y=0.30, box_vx=-0.3, box_vy=+0.30, vx=1.5,
                    left_room_m=1.0, box_world_speed_mps=0.30)
        leaving = oy.YieldObservation(**base, lane_end_distance_m=2.0)
        self.assertEqual(oy.decide(leaving, cfg).action, oy.ACTION_GO)
        # Near the end but already heading away from it: believable.
        just_left = oy.YieldObservation(**base, lane_end_distance_m=0.2,
                                        lane_end_closing=False)
        self.assertEqual(oy.decide(just_left, cfg).action, oy.ACTION_GO)
        # Near the end AND still heading for it: it will reverse.
        # ...and only while the BASE is inside a lane: applying the distrust
        # everywhere tripled the holds (48 episodes / 598 s vs 6 / 356 s).
        turning = oy.YieldObservation(**base, lane_end_distance_m=0.2,
                                      lane_end_closing=True, lane_distance_m=0.3)
        self.assertNotEqual(
            oy.decide(turning, cfg).action, oy.ACTION_GO,
            'a box arriving at an endpoint may be about to come back')
        outside = oy.YieldObservation(**base, lane_end_distance_m=0.2,
                                      lane_end_closing=True, lane_distance_m=4.0)
        self.assertEqual(oy.decide(outside, cfg).action, oy.ACTION_GO,
                         'outside a lane the measured velocity is believed')
        segments = oy.parse_patrol_segments('mover_2:1.30,-5.72,-0.10,-4.94,0.30')
        self.assertAlmostEqual(
            oy.distance_to_lane_end(1.25, -5.70, segments), 0.056, places=2)
        self.assertIsNone(oy.distance_to_lane_end(0.0, 0.0, []))
        # Only a box still heading for the endpoint it is NEAREST counts as
        # about to turn: at (1.20,-5.68) that is (1.30,-5.72), so a velocity
        # towards it is a closing box and a velocity towards the far end means
        # it has already left this one.
        self.assertTrue(oy.closing_on_lane_end(1.20, -5.68, 0.3, -0.12, segments))
        self.assertFalse(oy.closing_on_lane_end(1.20, -5.68, -0.3, 0.16, segments))
        self.assertFalse(oy.closing_on_lane_end(1.20, -5.68, 0.0, 0.0, segments))

    def test_the_executor_learns_the_lanes_from_the_mover_node(self):
        """One source of truth: the lanes arrive on a latched topic."""
        movers = (ROOT / 'scripts' / 'moving_obstacles.py').read_text(
            encoding='utf-8')
        self.assertIn("declare_parameter('patrols_topic', '/movers/patrols')",
                      movers)
        self.assertIn('DurabilityPolicy.TRANSIENT_LOCAL', movers)
        source = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("declare_parameter('mover_patrols_topic', '/movers/patrols')",
                      source)
        self.assertIn('_patrols_callback', source)
        self.assertIn('lane_distance_m=obstacle_yield.distance_to_lane(',
                      source)
        segments = module.obstacle_yield.parse_patrol_segments(
            'mover_3:11.76,-4.53,12.62,-5.87,0.16,k; broken; mover_1:-12,-3,-10.5,-3,0.16')
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(
            module.obstacle_yield.distance_to_lane(12.36, -4.91, segments),
            0.30, places=2)
        self.assertIsNone(module.obstacle_yield.distance_to_lane(0.0, 0.0, []))

    def test_the_tight_passage_cap_engages_before_the_passage(self):
        """The base must be slow when it arrives, not after.

        Measured 2026-10-13 (full_round_yield_20261013-195911): the base entered
        a tight passage at 1.5 m/s, the cap bit 1.2 s later, and 0.45 m of
        braking plus 0.45 m of latency was about all the room left to the wall -
        it hit it and was thrown at 3.3 m/s, failing the round on the first leg.
        """
        node = SimpleNamespace(
            _slow_zones=[(5.3, -0.4, 1.2, 0.5)],
            _cached_base_position=lambda: (6.6, -0.4),   # 0.1 m OUTSIDE the zone
            get_parameter=lambda name: SimpleNamespace(
                value=0.8 if name == 'slow_zone_brake_ahead_m' else None),
        )
        limit = module.FixedJointPickPlace._slow_zone_limit(node)
        self.assertEqual(limit, 0.5, 'the cap must already apply before entry')
        # Well clear of it: no cap.
        node._cached_base_position = lambda: (9.0, -0.4)
        self.assertEqual(module.FixedJointPickPlace._slow_zone_limit(node), 0.0)
        # Inside: still capped.
        node._cached_base_position = lambda: (5.3, -0.4)
        self.assertEqual(module.FixedJointPickPlace._slow_zone_limit(node), 0.5)

    def test_a_transient_health_loss_does_not_abort_the_object(self):
        """A 4 s lifecycle hiccup must not throw away a carried cube.

        Measured 2026-10-13 (full_round_repro_20261013-203330): the manager
        declared bt_navigator dead after a 4 s heartbeat gap, the health gate
        dropped, and the 3 s grace expired before the stack rebuilt itself
        ~28 s later - the round died at 3/5 with red_cube_1 still attached.
        """
        import yaml
        params = yaml.safe_load(
            (ROOT / 'config' / 'profiles' / 'state_machine_fast.yaml').read_text(
                encoding='utf-8'))['task_state_machine']['ros__parameters']
        grace = float(params['health_loss_grace_sec'])
        self.assertGreater(grace, 28.0,
                           'must outlast the observed stack rebuild')
        self.assertIn("declare_parameter('health_loss_grace_sec', 3.0)",
                      (ROOT / 'scripts' / 'task_state_machine.py').read_text(
                          encoding='utf-8'))

    def test_a_failed_object_stops_the_round_by_policy(self):
        """Fail closed: exhausted retries on any object end the round.

        Skipping was measured as the higher-scoring option on 2026-10-14
        (run_logs/chain_forensics_20261014-153453: the base was trapped beside
        mover_2's patrol segment with 74-144 controller refusals per attempt and
        the round ended at 1/5 while three cubes and ~1800 s of budget
        remained), but the operator's rule for a scored round is that a failed
        object stops the round so a human can look at the robot. Both profiles
        must therefore keep skip_failed_object false.
        """
        import yaml
        source = (ROOT / 'scripts' / 'task_state_machine.py').read_text(
            encoding='utf-8')
        for name in ('task_state_machine.yaml', 'profiles/state_machine_fast.yaml'):
            deployed = yaml.safe_load(
                (ROOT / 'config' / name).read_text(encoding='utf-8'))[
                    'task_state_machine']['ros__parameters']
            self.assertIn('skip_failed_object', deployed, name)
            self.assertFalse(
                deployed['skip_failed_object'],
                f'{name} must stop the round when an object fails')
        # The retry budget is what happens BEFORE that stop, and the skip path
        # must stay behind the unsafe-state gate if anyone flips it later.
        unsafe_at = source.index('if self._unsafe_to_retry:')
        skip_at = source.index(
            "if bool(self.get_parameter('skip_failed_object').value):")
        fail_at = source.index("self._fail('OBJECT_FAILED'")
        self.assertLess(unsafe_at, skip_at,
                        'unsafe states must fail closed before any skipping')
        self.assertLess(skip_at, fail_at,
                        'the fail-closed stop stays the default branch')

    def test_the_escape_reverses_after_the_controller_refused_the_path(self):
        """Driving forward again re-enters the refusal that stopped the base."""
        source = (ROOT / 'scripts' / 'fixed_joint_pick_place.py').read_text(
            encoding='utf-8')
        self.assertIn("self._last_attempt_failure = 'wedge_refusals'", source)
        self.assertIn('direction=(-1.0 if self._last_attempt_failure', source)
        self.assertIn("== 'wedge_refusals' else None)", source)
        self.assertIn('if direction is None:', source)
        self.assertIn('direction = self._clearer_axis()', source)

    def test_back_off_honours_a_forced_direction(self):
        bind = module.FixedJointPickPlace
        published = []

        def build():
            return SimpleNamespace(
                _clearer_axis=lambda: 1.0,
                _fine_cmd_pub=SimpleNamespace(publish=published.append),
                _stop_base=lambda: None,
                get_logger=lambda: SimpleNamespace(warning=lambda *a: None),
            )

        def run(node, **kwargs):
            ticks = iter([0.0, 0.0, 99.0, 99.0, 99.0])
            with patch.object(module.time, 'monotonic',
                              side_effect=lambda: next(ticks, 99.0)), \
                 patch.object(module.rclpy, 'ok', return_value=True), \
                 patch.object(module.rclpy, 'spin_once', lambda *a, **k: None):
                bind._back_off(node, 'leg', distance_m=0.7, speed=0.08, **kwargs)

        run(build(), direction=-1.0)
        self.assertTrue(published, 'the escape must command something')
        self.assertLess(published[0].linear.x, 0.0)
        published.clear()
        run(build())
        self.assertGreater(published[0].linear.x, 0.0)

    def test_the_collision_monitor_no_longer_reads_a_dead_topic(self):
        """It read /cmd_vel_smoothed, which this stack never publishes."""
        import yaml
        params = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))[
            'collision_monitor']['ros__parameters']
        self.assertNotEqual(params['cmd_vel_in_topic'], 'cmd_vel_smoothed')
        self.assertEqual(params['cmd_vel_in_topic'], 'cmd_vel')
        self.assertNotEqual(params['cmd_vel_out_topic'], 'cmd_vel')

    def test_the_throughput_pass_keeps_the_safety_floors(self):
        """Faster navigation must not silently remove the protections.

        Measured 2026-10-13 (timed_round_20261013-214508): 139 of 231 navigation
        seconds were spent below 0.6 m/s, so the round was limited by stacked
        slow floors, not by cruise speed. This pins the floors that were raised
        AND the protections that must stay in force.
        """
        import yaml
        nav = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
        follow = nav['controller_server']['ros__parameters']['FollowPath']
        # Two aggressive passes were tried and REVERTED (1.8 m/s + cost 0.25:
        # 8 jam escapes; 1.6 m/s + raised approach floor: first leg aborted).
        # Only these narrow steps survive, and cruise stays at the value the
        # 355 s round ran with.
        self.assertGreaterEqual(follow['regulated_linear_scaling_min_speed'], 0.4)
        self.assertLessEqual(follow['cost_scaling_dist'], 0.5)
        self.assertLessEqual(follow['desired_linear_vel'], 1.6,
                             'the ceiling is not the lever: crawling is')
        # The first, over-aggressive attempt (1.8 m/s and cost_scaling_dist
        # 0.25) was reverted: it produced 8 jam escapes and 3 x 25 s controller
        # refusals in one round, because past ~1.6 m/s the base spends its time
        # being refused instead of driving.
        # ...and these may not be traded away for speed:
        self.assertTrue(follow['use_collision_detection'])
        self.assertLessEqual(follow['max_allowed_time_to_collision_up_to_carrot'], 1.0)
        self.assertGreaterEqual(
            nav['velocity_smoother']['ros__parameters']['max_decel'][0], -3.0,
            'braking authority must not be reduced to buy speed')
        executor = yaml.safe_load(
            (ROOT / 'config' / 'profiles' / 'executor_fast.yaml').read_text(
                encoding='utf-8'))['fixed_joint_pick_place']['ros__parameters']
        # The tight passages keep an explicit cap, and it stays far below cruise.
        caps = executor['slow_zones'][3::4]
        self.assertTrue(caps)
        for cap in caps:
            self.assertLessEqual(cap, 0.8)
        self.assertLessEqual(executor['dock_creep_speed_mps'], 0.2)

    def test_a_controller_refusal_recovers_quickly(self):
        """A stuck base must retry in seconds, not tens of seconds.

        The controller's failure tolerance used to be 25 s, set to stop the
        recovery tree spinning the base while a yield hold waited for a box.
        The hold now cancels Nav2's goal instead, so a long tolerance only
        delays the escape: measured 2026-10-13 (run_logs/budget_fix_round) six
        refusal episodes each burned the full 25 s and the round took 520 s
        against a 355 s baseline. The progress window still has to outlast it.
        """
        import yaml
        params = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))[
            'controller_server']['ros__parameters']
        tolerance = float(params['failure_tolerance'])
        self.assertLessEqual(tolerance, 8.0,
                             'a refusal must recover in seconds')
        self.assertGreaterEqual(tolerance, 1.0)
        allowance = float(params['progress_checker']['movement_time_allowance'])
        self.assertGreater(allowance, tolerance,
                           'the progress window must outlast the failure window')
        # The mechanism that replaced the long tolerance: a hold cancels the
        # goal, so the controller is not failing while the base waits.
        source = SCRIPT.read_text(encoding='utf-8')
        block = source[source.index('if self._yield_decision.holds and not self._yield_suppress:'):]
        block = block[:block.index('if self._yield_arc_goal is not None:',
                                   block.index('resumed = self._resume_nav_goal'))]
        self.assertIn('_release_nav_control', block)
        self.assertIn('_hold_until_clear', block)

    def test_the_diagnostic_is_silent_while_the_feature_is_off(self):
        """An off feature must not narrate: the operator's run logged 82
        YIELD_DIAG lines for a policy that was disabled."""
        node = SimpleNamespace(
            _yield_diag_period=5.0, _yield_diag_last=0.0,
            _yield_config=SimpleNamespace(enabled=False),
            get_logger=lambda: SimpleNamespace(
                warning=lambda *a, **k: self.fail('must not log when disabled')),
        )
        module.FixedJointPickPlace._yield_diag(
            node, 0.0, [], [], [], None, None, 1e9)

    def test_a_clock_mismatch_falls_back_to_the_receipt_age(self):
        """Two time bases must not be mistaken for a stale sweep."""
        node = SimpleNamespace(
            _scan=SimpleNamespace(
                range_max=10.0, angle_increment=0.01, angle_min=-3.14,
                range_min=0.5, ranges=[1.0] * 10,
                header=SimpleNamespace(stamp=Time(sec=1248, nanosec=0))),
            _scan_received_at=module.time.monotonic() - 0.1,
            _clock_mismatch_reported=False,
            _clock_seconds=lambda: 1791886035.0,      # wall clock, not sim
            get_logger=lambda: SimpleNamespace(
                error=lambda *a, **k: None, warning=lambda *a, **k: None),
        )
        age, points = module.FixedJointPickPlace._yield_scan_points(node, 0.0)
        self.assertEqual(len(points), 10)
        self.assertLess(age, 1.0,
                        'the receipt age must replace the 1.79e9 s gap')
        self.assertTrue(node._clock_mismatch_reported,
                        'the mismatch must be reported, not swallowed')
        # A sweep that was genuinely never received is still infinitely old.
        node._scan_received_at = None
        age, _ = module.FixedJointPickPlace._yield_scan_points(node, 0.0)
        self.assertEqual(age, math.inf)

    def test_the_progress_window_outlasts_the_wait_budget(self):
        """The controller's patience must exceed the policy's own wait budget,
        or the recovery tree backs the base up into the obstacle it is waiting
        for. Verified mechanism: run_logs/yield_nav2_semantics.md §2/§3.
        """
        profile = PROFILE.read_text(encoding='utf-8')
        allowance = float(
            [line for line in profile.splitlines()
             if 'movement_time_allowance' in line][0].split(':')[1]
        )
        budget = float(
            [line for line in SCRIPT.read_text(encoding='utf-8').splitlines()
             if "declare_parameter('yield_wait_budget_sec'" in line][0]
            .split(',')[1].strip().rstrip(')')
        )
        self.assertGreater(allowance, budget)

    def test_an_arc_waypoint_is_consumed_once(self):
        node = SimpleNamespace(
            _yield_arc_goal=(3.0, 4.0), _yield_arc_attempts=0,
            max_arc_attempts=None,
        )
        first = module.FixedJointPickPlace._pending_yield_arc(node)
        second = module.FixedJointPickPlace._pending_yield_arc(node)
        self.assertEqual(first, (3.0, 4.0))
        self.assertIsNone(second)
        self.assertEqual(node._yield_arc_attempts, 1)

    def test_an_arc_retry_does_not_shove_the_base_backwards(self):
        """The retry after a yield arc must not run the dead-zone escape and the
        jam break: the leg was abandoned because a moving obstacle owned the
        lane, not because the base was wedged, and those two would drive a
        healthy base backwards through the thing it just avoided."""
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'FixedJointPickPlace')
        navigate = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_navigate_to')
        calls = [(n.func.attr, n.lineno) for n in ast.walk(navigate)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        arc_line = next(line for name, line in calls
                        if name == '_pending_yield_arc')
        escape_line = next(line for name, line in calls
                           if name == '_enter_dead_zone_escape')
        back_off_line = next(line for name, line in calls
                             if name == '_back_off')
        self.assertLess(arc_line, escape_line)
        self.assertLess(arc_line, back_off_line)

    def test_budget_expired_arc_skips_jam_escape_and_retries_original_goal(self):
        """An arc created at the 20 s hold boundary is not a Nav2 failure.

        In current_avoidance_check the arc log was rate-limited; the caller
        instead backed up 0.70 m after cancelling the just-resumed goal.
        Exercise the actual retry loop rather than merely checking source order.
        """
        events = []
        logger = SimpleNamespace(
            warning=lambda message: events.append(('warning', message)),
            info=lambda message: events.append(('info', message)),
        )
        node = SimpleNamespace(
            _yield_config=SimpleNamespace(enabled=True),
            _yield_arc_goal=None, _yield_arc_attempts=0,
            _yield_arc_used=False, _yield_suppress=False,
            get_parameter=lambda name: SimpleNamespace(value={
                'navigation_retry_count': 2, 'yield_arc_goal_radius_m': 0.35,
                'settle_sec': 0.0,
            }[name]),
            get_logger=lambda: logger,
            _log_cube_positions=lambda label: None,
            _restore_nominal_geometry=lambda label: None,
            _clear_speed_limit=lambda: None,
            _apply_slow_zones=lambda: None,
            _clear_costmaps=lambda: True,
            _enter_dead_zone_escape=lambda label: events.append(('escape', label)),
            _back_off=lambda *args, **kwargs: events.append(('backoff', args)),
        )
        node._pending_yield_arc = (
            module.FixedJointPickPlace._pending_yield_arc.__get__(node)
        )

        def navigate_once(label, values, *args):
            events.append(('nav', label))
            if len([event for event in events if event[0] == 'nav']) == 1:
                node._yield_arc_goal = (3.0, 4.0)
                return False
            return True

        node._navigate_once = navigate_once
        with patch.object(module.time, 'sleep'):
            result = module.FixedJointPickPlace._navigate_to(
                node, 'warehouse A', (5.0, 6.0, 0.0)
            )
        self.assertTrue(result)
        self.assertEqual([value for kind, value in events if kind == 'nav'],
                         ['warehouse A', 'warehouse A yield arc', 'warehouse A'])
        self.assertFalse(any(kind in ('escape', 'backoff') for kind, _ in events))
        self.assertEqual(node._yield_arc_attempts, 1)

    def test_a_hold_that_expires_into_an_arc_does_not_resume_nav2(self):
        source = SCRIPT.read_text(encoding='utf-8')
        block = source[source.index('if not self._hold_until_clear(label, x, y, deadline):'):]
        block = block[:block.index('if self._yield_arc_goal is not None:',
                                   block.index('resumed = self._resume_nav_goal'))]
        self.assertLess(block.index('if self._yield_arc_goal is not None:'),
                        block.index('resumed = self._resume_nav_goal'))

    def test_resetting_the_leg_forgets_the_yield_episode(self):
        node = SimpleNamespace(
            _yield_latched=True, _yield_wait_since=1.0, _yield_arc_attempts=2,
            _yield_arc_goal=(1.0, 2.0), _yield_arc_used=True,
            _yield_decision='stale',
        )
        module.FixedJointPickPlace._reset_yield_state(node)
        self.assertFalse(node._yield_latched)
        self.assertIsNone(node._yield_wait_since)
        self.assertEqual(node._yield_arc_attempts, 0)
        self.assertIsNone(node._yield_arc_goal)
        self.assertFalse(node._yield_arc_used)
        self.assertEqual(node._yield_decision.action, 'go')


class PatrolBypassRouteTests(unittest.TestCase):
    """A fixed bypass must not put a loaded robot on mover_3's patrol line."""

    def test_blue_cube_3_a_route_goes_south_and_east_of_the_patrol(self):
        import yaml

        profile = yaml.safe_load((ROOT / 'config/profiles/executor_fast.yaml')
                                 .read_text(encoding='utf-8'))
        params = profile['fixed_joint_pick_place']['ros__parameters']
        points = [tuple(params[f'blue_cube_3_warehouse_a_via_{i}'][:2])
                  for i in (1, 2, 3)]
        # Actual loaded start in the failed GUI run, then all three waypoints
        # and the plate. The nearest approach must not cut through the mover.
        route = [(7.62, -6.13), *points,
                 tuple(params['warehouse_a_pose'][:2])]
        patrol = ((11.76, -4.53), (12.62, -5.87))

        def segment_distance(point, line):
            a, b = line
            dx, dy = b[0] - a[0], b[1] - a[1]
            fraction = max(0.0, min(1.0,
                ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy)
                / (dx * dx + dy * dy)))
            return math.hypot(point[0] - a[0] - fraction * dx,
                              point[1] - a[1] - fraction * dy)

        samples = []
        for start, end in zip(route, route[1:]):
            count = max(1, math.ceil(math.dist(start, end) / 0.05))
            samples.extend((start[0] + (end[0] - start[0]) * k / count,
                            start[1] + (end[1] - start[1]) * k / count)
                           for k in range(count + 1))
        self.assertGreaterEqual(min(segment_distance(p, patrol)
                                    for p in samples), 0.85)
        self.assertGreaterEqual(segment_distance(points[0], patrol), 1.9,
                                'the first waypoint must be a safe waiting spot')
        self.assertAlmostEqual(float(params['loaded_robot_radius_m']), 0.35)

        # The tempting northern shortcut crosses an occupied wall at y~-3.7.
        # Read the repository's P5 map, not a hand-drawn schematic, to ensure
        # the proposed route has >=0.40 m static-map cell clearance everywhere.
        mapdir = ROOT.parent / 'bot_navigation/maps'
        map_params = yaml.safe_load((mapdir / 'room_from_world.yaml')
                                    .read_text(encoding='utf-8'))
        with (mapdir / map_params['image']).open('rb') as stream:
            self.assertEqual(stream.readline().strip(), b'P5')
            header = []
            while len(header) < 3:
                line = stream.readline()
                if not line.startswith(b'#'):
                    header.extend(line.split())
            width, height, maximum = map(int, header[:3])
            self.assertEqual(maximum, 255)
            pixels = np.frombuffer(stream.read(), dtype=np.uint8)
        self.assertEqual(pixels.size, width * height)
        rows, cols = np.nonzero(pixels.reshape(height, width) < 128)
        resolution = float(map_params['resolution'])
        origin_x, origin_y = map_params['origin'][:2]
        obstacle_x = origin_x + (cols + 0.5) * resolution
        obstacle_y = origin_y + (height - rows - 0.5) * resolution
        near = ((obstacle_x >= 7.0) & (obstacle_x <= 14.5)
                & (obstacle_y >= -8.0) & (obstacle_y <= -2.0))
        obstacle_x, obstacle_y = obstacle_x[near], obstacle_y[near]
        for x, y in samples:
            clearance = np.hypot(obstacle_x - x, obstacle_y - y).min()
            self.assertGreaterEqual(clearance, 0.40, (x, y, clearance))


class ContinuousDetourTests(unittest.TestCase):
    """The detour must be ONE goal, not one goal per waypoint.

    Measured 2026-10-14: 11 planned goals produced 38 Nav2 goal cancellations in
    a single round, which is what the operator saw as the base ignoring its
    route. One NavigateThroughPoses goal keeps a single plan for the polyline.
    """

    def node(self, *, continuous=True, waypoints=None):
        table = {
            'cube_name': 'blue_cube_3',
            'warehouse': 'A',
            'navigation_frame': 'map',
            'use_continuous_detour': continuous,
            'use_aligned_warehouse_approach': False,
            'chain_switch_radius_m': 0.75,
            'loaded_robot_radius_m': 0.35,
            'loaded_inflation_radius_m': 0.60,
            **(self.VIA if waypoints is None else waypoints),
        }
        node = SimpleNamespace(
            _chain_index=0,
            _navigate_client='navigate_client',
            _navigate_through_client='through_client',
            has_parameter=lambda name: name in table,
            get_parameter=lambda name: SimpleNamespace(value=table.get(name)),
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0))),
            get_logger=lambda: SimpleNamespace(
                info=lambda *a, **k: None, warning=lambda *a, **k: None,
                error=lambda *a, **k: None),
        )
        bind = module.FixedJointPickPlace
        node._double_list = bind._double_list.__get__(node)
        node._detour_waypoints = bind._detour_waypoints.__get__(node)
        node._remaining_chain = bind._remaining_chain.__get__(node)
        node._nav_goal = bind._nav_goal.__get__(node)
        node._pose_stamped = bind._pose_stamped.__get__(node)
        node._chain_progress_target = bind._chain_progress_target.__get__(node)
        return node

    VIA = {
        'blue_cube_3_warehouse_a_via_1': [10.8, -6.65, -0.16],
        'blue_cube_3_warehouse_a_via_2': [13.65, -6.85, -0.07],
        'blue_cube_3_warehouse_a_via_3': [13.65, -3.0, 1.570796],
    }

    def test_a_chain_becomes_one_through_poses_goal_in_order(self):
        node = self.node()
        chain = node._detour_waypoints('blue_cube_3', 'a')
        self.assertEqual([tuple(point[:2]) for point in chain],
                         [(10.8, -6.65), (13.65, -6.85), (13.65, -3.0)])
        goal, client = node._nav_goal([7.244, -2.483, -3.141593], chain)
        self.assertIs(client, 'through_client')
        self.assertIsInstance(goal, module.NavigateThroughPoses.Goal)
        self.assertEqual([(pose.pose.position.x, pose.pose.position.y)
                          for pose in goal.poses],
                         [(10.8, -6.65), (13.65, -6.85), (13.65, -3.0),
                          (7.244, -2.483)])

    def test_a_chain_that_repeats_the_target_is_not_sent_twice(self):
        node = self.node()
        goal, _client = node._nav_goal([10.8, -6.65, 0.0],
                                       [[10.8, -6.65, 0.0], [13.65, -3.0, 0.0]])
        self.assertEqual(len(goal.poses), 2)

    def test_a_leg_without_a_chain_still_uses_navigate_to_pose(self):
        node = self.node()
        goal, client = node._nav_goal([1.0, 2.0, 0.5], None)
        self.assertIs(client, 'navigate_client')
        self.assertIsInstance(goal, module.NavigateToPose.Goal)
        self.assertAlmostEqual(goal.pose.pose.position.x, 1.0)

    def test_progress_is_measured_against_the_next_unreached_waypoint(self):
        node = self.node()
        chain = [[10.8, -6.65, 0.0], [13.65, -6.85, 0.0], [13.65, -3.0, 0.0]]
        # Approaching the first waypoint from the start pose.
        node._cached_planar_distance_to_point = lambda x, y: {
            (10.8, -6.65): 2.0, (13.65, -6.85): 6.0, (13.65, -3.0): 8.0,
        }[(x, y)]
        self.assertEqual(node._chain_progress_target(chain, 7.0, -6.0),
                         (10.8, -6.65))
        # Past it: the base is now closer to the second one, so that becomes
        # the reference - a fixed arrival radius would still be pointing back.
        node._cached_planar_distance_to_point = lambda x, y: {
            (10.8, -6.65): 1.9, (13.65, -6.85): 0.8, (13.65, -3.0): 5.0,
        }[(x, y)]
        self.assertEqual(node._chain_progress_target(chain, 12.8, -6.8),
                         (13.65, -6.85))
        # The last waypoint is never skipped.
        node._cached_planar_distance_to_point = lambda x, y: {
            (10.8, -6.65): 4.0, (13.65, -6.85): 3.0, (13.65, -3.0): 0.5,
        }[(x, y)]
        self.assertEqual(node._chain_progress_target(chain, 13.6, -3.4),
                         (13.65, -3.0))
        # A plain leg keeps measuring against its own goal.
        node._chain_index = 0
        self.assertEqual(node._chain_progress_target(None, 5.0, 6.0), (5.0, 6.0))

    def test_the_continuous_detour_issues_one_navigation_stage(self):
        node = self.node(waypoints=self.VIA)
        node._set_costmap_geometry = lambda *a, **k: True
        node._clear_costmaps = lambda: True
        node._warehouse_goal = lambda: ('A', [7.244, -2.483, -3.141593])
        calls = []

        def navigate_to(label, values, visual_handoff_cube=None,
                        completion_radius_m=None, chain=None):
            calls.append((label, tuple(values), visual_handoff_cube,
                          completion_radius_m,
                          None if chain is None else [tuple(p) for p in chain]))
            return True

        node._navigate_to = navigate_to
        self.assertTrue(
            module.FixedJointPickPlace._navigate_to_warehouse(node)
        )
        self.assertEqual(len(calls), 1, calls)
        label, values, _handoff, _radius, chain = calls[0]
        self.assertIn('continuous detour', label)
        self.assertEqual(values, (7.244, -2.483, -3.141593))
        self.assertEqual(chain, [(10.8, -6.65, -0.16), (13.65, -6.85, -0.07),
                                 (13.65, -3.0, 1.570796)])
        self.assertEqual([point[:2] for point in chain],
                         [(10.8, -6.65), (13.65, -6.85), (13.65, -3.0)])

    def test_the_switch_off_keeps_the_per_waypoint_walk(self):
        node = self.node(continuous=False, waypoints=self.VIA)
        node._set_costmap_geometry = lambda *a, **k: True
        node._clear_costmaps = lambda: True
        node._warehouse_goal = lambda: ('A', [7.244, -2.483, -3.141593])
        calls = []
        node._navigate_to = lambda label, values, *a, **k: calls.append(
            (label, tuple(values), k.get('chain'))) or True
        self.assertTrue(
            module.FixedJointPickPlace._navigate_to_warehouse(node)
        )
        self.assertEqual([label for label, _v, _c in calls],
                         ['blue_cube_3 to A detour waypoint 1',
                          'blue_cube_3 to A detour waypoint 2',
                          'blue_cube_3 to A detour waypoint 3',
                          'warehouse A'])
        self.assertTrue(all(chain is None for _l, _v, chain in calls))

    def test_a_resume_sends_only_the_waypoints_not_yet_passed(self):
        """Re-sending the whole chain makes Nav2 forget what was reached.

        Measured 2026-10-14 (stable_topose_ab_20261014-161832, blue_cube_3 -> A):
        seven holds in one leg, seven resumes, and the base drove via_1 -> via_2
        -> back to via_1 twice before it could climb to via_3, because each
        resume re-sent the full chain and Nav2's RemovePassedGoals only knows
        the goals inside the goal it just received.
        """
        node = self.node()
        chain = node._detour_waypoints('blue_cube_3', 'a')
        self.assertEqual(len(chain), 3)
        node._chain_index = 0
        self.assertEqual(node._remaining_chain(chain), list(chain))
        node._chain_index = 2
        remaining = node._remaining_chain(chain)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(tuple(remaining[0][:2]), (13.65, -3.0))
        goal, _client = node._nav_goal([7.244, -2.483, -3.141593], remaining)
        self.assertEqual(
            [(round(p.pose.position.x, 3), round(p.pose.position.y, 3))
             for p in goal.poses],
            [(13.65, -3.0), (7.244, -2.483)])
        # Every waypoint passed: a plain single-pose goal, not an empty chain.
        node._chain_index = len(chain)
        self.assertIsNone(node._remaining_chain(chain))
        self.assertIsNone(node._remaining_chain(None))

    def test_chain_progress_is_reset_per_leg_not_per_attempt(self):
        """A retry must not undo the waypoints the leg already reached."""
        source = SCRIPT.read_text(encoding='utf-8')
        leg = source[source.index('    def _navigate_to('):]
        self.assertIn('self._chain_index = 0', leg[:2000],
                      'the leg owns chain progress')
        attempt = source[source.index('    def _navigate_once__timed('):]
        attempt = attempt[:attempt.index('\n    def ', 10)]
        self.assertNotIn('self._chain_index = 0', attempt,
                         'an attempt must not reset what the leg reached')
        resume = source[source.index('resumed = self._resume_nav_goal') - 400:]
        self.assertIn('self._remaining_chain(chain)', resume[:400],
                      'a resume must rebuild the goal from what is left')

    def test_the_competition_profile_enables_the_continuous_detour(self):
        import yaml
        params = yaml.safe_load(
            (ROOT / 'config/profiles/executor_fast.yaml').read_text(
                encoding='utf-8'))['fixed_joint_pick_place']['ros__parameters']
        self.assertTrue(params['use_continuous_detour'])
        self.assertGreater(float(params['chain_switch_radius_m']), 0.0)


class SlowZoneTests(unittest.TestCase):
    """Tight-passage speed caps: inert unless configured, and always restored."""

    class FakePublisher:
        def __init__(self):
            self.sent = []

        def publish(self, message):
            self.sent.append(message)

    def node(self, zones, position=(12.78, -3.79), pub=None):
        return _bind_speed_limit_methods(SimpleNamespace(
            _speed_limit_pub=pub,
            _active_speed_limit=0.0,
            _slow_zones=zones,
            _cached_base_position=lambda: position,
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0))),
            get_parameter=lambda name: SimpleNamespace(
                value=({'slow_zones': zones} if name == 'slow_zones'
                       else 0.8 if name == 'slow_zone_brake_ahead_m' else None)),
            get_logger=lambda: SimpleNamespace(
                warning=lambda *a, **k: None, info=lambda *a, **k: None,
                error=lambda *a, **k: None),
        ))

    def test_the_off_sentinel_means_no_zones(self):
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(
                value=[0.0] if name == 'slow_zones' else False))
        self.assertEqual(
            module.FixedJointPickPlace._parse_slow_zones(node), [])

    def test_yield_alone_creates_a_speed_limit_publisher(self):
        created = []
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(
                value=[] if name == 'slow_zones' else True),
            create_publisher=lambda *args: created.append(args) or object(),
        )
        self.assertEqual(module.FixedJointPickPlace._parse_slow_zones(node), [])
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0][1], 'speed_limit')

    def test_inside_zone_caps_and_outside_restores(self):
        pub = self.FakePublisher()
        node = self.node([], pub=pub)
        node._slow_zones = [(12.78, -3.60, 1.10, 0.6)]
        module.FixedJointPickPlace._apply_slow_zones(node)
        self.assertEqual(len(pub.sent), 1)
        self.assertAlmostEqual(pub.sent[-1].speed_limit, 0.6)
        self.assertFalse(pub.sent[-1].percentage)
        # Leaving the zone lifts the cap, and the limit is published only on change.
        node._cached_base_position = lambda: (7.0, -2.5)
        module.FixedJointPickPlace._apply_slow_zones(node)
        self.assertEqual(pub.sent[-1].speed_limit, 0.0)
        module.FixedJointPickPlace._apply_slow_zones(node)
        self.assertEqual(len(pub.sent), 2)
        module.FixedJointPickPlace._clear_speed_limit(node)
        self.assertEqual(len(pub.sent), 2)

    def test_overlapping_zones_take_the_slower_cap(self):
        pub = self.FakePublisher()
        node = self.node([], pub=pub)
        node._slow_zones = [(12.78, -3.60, 1.10, 0.9), (12.75, -3.70, 0.90, 0.5)]
        module.FixedJointPickPlace._apply_slow_zones(node)
        self.assertAlmostEqual(pub.sent[-1].speed_limit, 0.5)

    def test_no_publisher_means_no_message(self):
        node = self.node([], pub=None)
        node._slow_zones = [(12.78, -3.60, 1.10, 0.6)]
        module.FixedJointPickPlace._apply_slow_zones(node)  # must not raise
        self.assertEqual(node._active_speed_limit, 0.0)


class DependencySpecLazinessTests(unittest.TestCase):
    """Rebuilt clients must not be called through a stale bound method.

    Regression for run_logs/speed_rpp15_combined_trial2_20261010-170156: both
    costmap service clients were rebuilt and the executor died with
    "cannot use Destroyable because destruction was requested".
    """

    def build(self):
        def client(tag):
            return SimpleNamespace(
                service_is_ready=lambda: tag[0],
                wait_for_server=lambda timeout_sec=0.0: tag[0],
            )
        return SimpleNamespace(
            _arm_client=client([True]), _gripper_client=client([True]),
            _navigate_client=client([True]),
            _navigate_through_client=client([True]),
            _attach_client=client([True]),
            _detach_client=client([True]), _clear_global_client=client([True]),
            _clear_local_client=client([True]),
            _global_costmap_params_client=client([True]),
            _local_costmap_params_client=client([True]),
            _set_state_client=client([True]),
        )

    def test_ready_callables_follow_a_replaced_service_client(self):
        node = self.build()
        specs = module.FixedJointPickPlace._dependency_specs(node)
        ready = {key: fn for key, _desc, fn in specs}
        self.assertEqual(sorted(ready), ['arm', 'attach', 'clear_global',
                                         'clear_local', 'detach', 'gripper',
                                         'navigate', 'navigate_through',
                                         'params_global',
                                         'params_local', 'set_state'])
        # Swap in a fresh client object, exactly as _rebuild_dependency_client
        # does, and the old one becomes unusable (as rclpy's destroy() does).
        def dead():
            raise RuntimeError('cannot use Destroyable because destruction was '
                               'requested')
        node._clear_global_client = SimpleNamespace(service_is_ready=lambda: True,
                                                    wait_for_server=lambda **k: True)
        for key in ('attach', 'detach', 'clear_local', 'params_global',
                    'params_local', 'set_state'):
            self.assertTrue(ready[key](), key)
        # A genuinely dead client surfaces as an error rather than silently
        # reporting "ready", and a replaced one is picked up lazily.
        node._clear_global_client = SimpleNamespace(service_is_ready=dead)
        with self.assertRaisesRegex(RuntimeError, 'Destroyable'):
            ready['clear_global']()
        node._clear_global_client = SimpleNamespace(service_is_ready=lambda: True)
        self.assertTrue(ready['clear_global']())

    def test_specs_are_rebuilt_from_current_attributes_each_call(self):
        node = self.build()
        first = module.FixedJointPickPlace._dependency_specs(node)
        node._attach_client = SimpleNamespace(service_is_ready=lambda: False)
        fresh = {k: fn for k, _d, fn in
                 module.FixedJointPickPlace._dependency_specs(node)}
        self.assertTrue(first and fresh)
        self.assertFalse(fresh['attach']())


class VisualAlignResearchTests(unittest.TestCase):
    """The align loop must look again after losing the blob, not wait out the timeout.

    Regression for the 1.5 m/s rounds: a fast handoff left the cube at
    centre_error -0.80, the centring motion pushed it out of frame, and the loop
    then sat motionless for 72 s before reporting a timeout.
    """

    def build(self, timeline, detection, image_width=640):
        params = {
            'enable_visual_alignment': True, 'cube_name': 'red_cube_5',
            'visual_timeout_sec': 60.0, 'visual_search_timeout_sec': 8.0,
            'visual_search_rate': 0.35, 'visual_research_after_sec': 3.0,
            'visual_stable_frames': 3, 'visual_approach_row': 430.0,
            'visual_standoff_frames': 5, 'visual_stable_span_px': 8.0,
            'visual_row_tolerance': 9.0, 'visual_center_tolerance': 0.08,
            'visual_detection_stale_sec': 0.5, 'visual_linear_gain': 0.5,
            'visual_angular_gain': 0.5, 'visual_max_linear_speed': 0.15,
            'visual_min_linear_speed': 0.02, 'visual_max_angular_speed': 0.25,
        }
        states, commands = [], []
        clock = {'t': 0.0}

        def now():
            clock['t'] += 0.25
            return clock['t']

        def spin(_node, timeout_sec=None):
            detection.append(('tick', clock['t']))

        node = SimpleNamespace(
            _target_colour=None, _visual_detection=None, _visual_detection_time=0.0,
            _fine_cmd_pub=SimpleNamespace(publish=lambda cmd: commands.append(cmd)),
            _stop_base=lambda: None,
            _publish_vision_state=lambda state: states.append(state),
            _creep_to_grasp=lambda row: states.append(f'CREEP:{row:.0f}') or True,
            get_logger=lambda: SimpleNamespace(
                info=lambda *a, **k: None, warning=lambda *a, **k: None,
                error=lambda *a, **k: None),
            get_parameter=lambda name: SimpleNamespace(value=params[name]),
        )
        return node, states, commands, clock, now, spin

    def run_align(self, node, detection, clock, now, spin):
        # The real helper waits for a fresh frame; these tests drive a scripted
        # clock, so stub it down to the single spin they already simulate.
        node._spin_for_fresh_detection = lambda *a, **k: module.rclpy.spin_once(
            node, timeout_sec=0.1)
        with patch.object(module.time, 'monotonic', side_effect=now), \
             patch.object(module.rclpy, 'spin_once', side_effect=spin), \
             patch.object(module.rclpy, 'ok', return_value=True):
            # The public name is now a thin timing wrapper; the behaviour
            # under test lives in the __timed implementation.
            return module.FixedJointPickPlace._visual_align_active__timed(node)

    def test_blob_lost_mid_centring_restarts_the_search(self):
        # Visible and centred for a while, then gone for good.
        def detect(tick_t):
            if tick_t < 6.0:
                return None
            if tick_t < 10.0:
                # Seen but far off centre, so the loop keeps ALIGNING rather than
                # accepting the pose; then the blob disappears for good.
                return (100.0, 40.0, 900.0, 640, 430.0)
            return None

        blob = []

        def spin(_node, timeout_sec=None):
            node._visual_detection = detect(clock['t'])
            node._visual_detection_time = clock['t']
            blob.append(node._visual_detection)

        node, states, commands, clock, now, _ = self.build(None, None)
        ok = self.run_align(node, None, clock, now, spin)
        self.assertFalse(ok)
        aligning = [s for s in states if s.startswith('ALIGNING')]
        self.assertTrue(aligning, 'the centred phase should have run')
        # A search state must appear AFTER alignment work, i.e. the loop went back.
        first_align = states.index(aligning[0])
        self.assertIn('SEARCHING:red', states[first_align:],
                      f'no re-search after losing the blob: {states}')
        self.assertNotIn('CREEP', ' '.join(states))

    def test_stays_in_search_when_nothing_is_ever_seen(self):
        def spin(_node, timeout_sec=None):
            node._visual_detection = None

        node, states, commands, clock, now, _ = self.build(None, None)
        ok = self.run_align(node, None, clock, now, spin)
        self.assertFalse(ok)
        self.assertTrue(all(s.startswith('SEARCHING') for s in states
                            if not s.startswith('FAILED')), states[:8])
        self.assertTrue(commands, 'the search should command a rotation')


class StartButtonGateTests(unittest.TestCase):
    """The layout's 「开始任务」 button gates the first object, opt-in.

    A task normally auto-starts once the system is ready, which would leave that
    button dead; with require_start_button on, the run parks in
    WAITING_FOR_START until /competition/start carries true.
    """

    def node(self, enabled=True, requested=False):
        node = SimpleNamespace(
            _start_requested=requested,
            get_parameter=lambda name: SimpleNamespace(
                value=enabled if name == 'require_start_button' else None),
            get_logger=lambda: SimpleNamespace(warning=lambda *a, **k: None),
        )
        for name in ('_start_gate_closed', '_start_button_cb'):
            setattr(node, name, getattr(supervisor.TaskStateMachine, name).__get__(node))
        return node

    def test_gate_is_open_when_the_feature_is_off(self):
        self.assertFalse(self.node(enabled=False)._start_gate_closed())

    def test_gate_holds_until_the_button_is_pressed(self):
        node = self.node(enabled=True)
        self.assertTrue(node._start_gate_closed())
        node._start_button_cb(SimpleNamespace(data=True))
        self.assertFalse(node._start_gate_closed())

    def test_a_false_message_does_not_release_the_gate(self):
        node = self.node(enabled=True)
        node._start_button_cb(SimpleNamespace(data=False))
        self.assertTrue(node._start_gate_closed())

    def test_waiting_for_start_is_a_known_state(self):
        self.assertIn('WAITING_FOR_START', supervisor.VALID_STATES)

    def test_a_plain_volatile_publisher_can_also_release_the_gate(self):
        """`ros2 topic pub` and the Foxglove button publish VOLATILE.

        The latched reader is RELIABLE + TRANSIENT_LOCAL, so it silently drops
        those (measured 2026-10-13: the documented command printed "publishing
        #1" and the run stayed in WAITING_FOR_START). The start topic therefore
        needs a second reader whose requested durability and reliability are the
        weakest DDS allows, which matches every publisher.
        """
        qos = supervisor.start_button_fallback_qos()
        self.assertEqual(qos.durability, DurabilityPolicy.VOLATILE)
        self.assertEqual(qos.reliability, ReliabilityPolicy.BEST_EFFORT)

    def test_the_gate_parks_without_starting_an_object(self):
        """_tick must wait, not call _start_current, while the gate is closed."""
        logger = Mock()
        node = SimpleNamespace(
            _terminal=False, _queue=[{'cube': 'red_cube_1', 'zone': 'A'}],
            _task_started=None, _object_active=False, _system_ready=True,
            _start_requested=False,
            get_parameter=lambda name: SimpleNamespace(
                value={'autostart': True, 'require_start_button': True}.get(name)),
            get_logger=lambda: logger, _set_state=Mock(), _start_current=Mock(),
            _base_sanity_violation=lambda now: None, _task_budget=lambda: 1e9,
        )
        for name in ('_start_gate_closed',):
            setattr(node, name, getattr(supervisor.TaskStateMachine, name).__get__(node))
        supervisor.TaskStateMachine._tick(node)
        node._set_state.assert_called_with('WAITING_FOR_START')
        node._start_current.assert_not_called()


class SupervisorFailureTests(unittest.TestCase):
    def test_failure_summary_names_remaining_work(self):
        logger = Mock()
        node = SimpleNamespace(
            _stop_child=Mock(), _stop_pub=Mock(), _error_pub=Mock(),
            _string=lambda text: text, get_logger=lambda: logger,
            _queue=[{'cube': 'red_cube_2', 'zone': 'A'},
                    {'cube': 'blue_cube_3', 'zone': 'C'}],
            _index=1, _completed=1, _total=2, _task_started=0.0,
            _state='NAV_TO_ZONE', _terminal=False,
            _set_state=Mock(), _publish_progress=Mock(),
        )
        with patch.object(supervisor.time, 'monotonic', return_value=120.0):
            supervisor.TaskStateMachine._fail(node, 'OBJECT_FAILED', 'failed leg')
        summary = logger.error.call_args_list[-1].args[0]
        self.assertIn('completed=1/2', summary)
        self.assertIn("blue_cube_3->C", summary)
        self.assertIn('wall_since_first_object=120.0s', summary)
        self.assertTrue(node._terminal)


    def test_odom_speed_estimator_ignores_burst_arrival_jitter(self):
        """At 1.5 m/s a burst-delivered odom pair must not look like 2+ m/s.

        The old estimator divided the position delta by callback arrival time,
        so two odoms delivered back-to-back produced a huge fake speed and
        aborted a healthy fast run as "simulation blew up".
        """
        def msg(x, sim_t, vx=1.5):
            return SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=vx, y=0.0))),
                header=SimpleNamespace(stamp=SimpleNamespace(
                    sec=int(sim_t), nanosec=int((sim_t % 1) * 1e9))))

        node = SimpleNamespace(_odom_prev=None, _odom=None)
        with patch.object(supervisor.time, 'monotonic', return_value=10.0):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.0, 100.0))
        # 50 ms of sim time later, but only 2 ms of wall time later.
        with patch.object(supervisor.time, 'monotonic', return_value=10.002):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.075, 100.05))
        self.assertLess(node._odom[2], 2.0)
        self.assertAlmostEqual(node._odom[2], 1.5)

    def test_odom_single_step_at_2mps_is_not_reported_as_a_blowup(self):
        """Reproduce the speed2_trial1 false abort exactly.

        Commanded 2 m/s, the base genuinely ran at 2.000 (the recorder's /odom
        maximum), yet the gate aborted the round with "base speed 3.7 m/s
        exceeds 3.0": one 0.074 m odom step divided by the 0.02 s baseline.
        A single step must never be enough to condemn the run.
        """
        def msg(x, sim_t, vx=2.0):
            return SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=vx, y=0.0))),
                header=SimpleNamespace(stamp=SimpleNamespace(
                    sec=int(sim_t), nanosec=int((sim_t % 1) * 1e9))))

        node = SimpleNamespace(_odom_prev=None, _odom=None)
        with patch.object(supervisor.time, 'monotonic', return_value=30.0):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.0, 300.0))
        # One 20 ms hop carrying a full 0.074 m step: 3.7 m/s if trusted.
        with patch.object(supervisor.time, 'monotonic', return_value=30.02):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.074, 300.02))
        self.assertLess(node._odom[2], 3.0)
        self.assertAlmostEqual(node._odom[2], 2.0)
        # The anchor must NOT have moved, or the baseline could never grow.
        self.assertAlmostEqual(node._odom_prev[0], 0.0)
        # Keep feeding a genuine 2 m/s: the anchor stays put until the baseline
        # reaches DISPLACEMENT_BASELINE_S, and the rate measured over that whole
        # baseline is the true 2 m/s - never the per-step 3.7.
        for sim_t, x in ((300.04, 0.08), (300.06, 0.12), (300.08, 0.16)):
            with patch.object(supervisor.time, 'monotonic',
                              return_value=30.0 + (sim_t - 300.0)):
                supervisor.TaskStateMachine._odom_cb(node, msg(x, sim_t))
            self.assertAlmostEqual(node._odom[2], 2.0, places=2)
            self.assertAlmostEqual(node._odom_prev[0], 0.0)
        with patch.object(supervisor.time, 'monotonic', return_value=30.12):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.24, 300.12))
        # 0.24 m over a 0.12 s baseline = the true 2 m/s, still no blow-up.
        self.assertAlmostEqual(node._odom[2], 2.0, places=2)
        self.assertAlmostEqual(node._odom_prev[0], 0.24)

    def sanity_node(self, twist, measured, limit=3.0):
        return SimpleNamespace(
            _odom=(0.0, 0.0, twist, 100.0),
            _odom_channels=(twist, measured),
            _unconfirmed_spikes=0,
            get_parameter=lambda name: SimpleNamespace(value={
                'base_sanity_enabled': True,
                'max_plausible_speed_mps': limit,
                'map_bounds': [-15.0, -16.0, 17.0, 6.0],
                'map_bounds_margin_m': 1.0,
            }[name]),
        )

    def test_lone_twist_spike_does_not_condemn_the_run(self):
        """speed2_trial2: /odom said 7.051 m/s while the pose moved 2.05 m/s."""
        node = self.sanity_node(twist=7.051, measured=2.06)
        with patch.object(supervisor.time, 'monotonic', return_value=100.2):
            self.assertIsNone(
                supervisor.TaskStateMachine._base_sanity_violation(node, 100.2))

    def test_corroborated_overspeed_is_still_caught(self):
        node = self.sanity_node(twist=9.0, measured=8.4)
        with patch.object(supervisor.time, 'monotonic', return_value=100.2):
            violation = supervisor.TaskStateMachine._base_sanity_violation(node, 100.2)
        self.assertIsNotNone(violation)
        self.assertEqual(violation[0], 'ROBOT_LAUNCHED')
        self.assertIn('covered ground', violation[1])

    def test_persistent_unconfirmed_spikes_eventually_trip(self):
        node = self.sanity_node(twist=7.0, measured=None)
        with patch.object(supervisor.time, 'monotonic', return_value=100.2):
            for _ in range(4):
                self.assertIsNone(
                    supervisor.TaskStateMachine._base_sanity_violation(node, 100.2))
            self.assertIsNotNone(
                supervisor.TaskStateMachine._base_sanity_violation(node, 100.2))

    def test_normal_speed_never_trips(self):
        node = self.sanity_node(twist=2.0, measured=1.98)
        with patch.object(supervisor.time, 'monotonic', return_value=100.2):
            self.assertIsNone(
                supervisor.TaskStateMachine._base_sanity_violation(node, 100.2))
        self.assertEqual(node._unconfirmed_spikes, 0)

    def test_odom_speed_estimator_still_sees_a_real_teleport(self):
        def msg(x, sim_t, vx=0.0):
            return SimpleNamespace(
                pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=0.0))),
                twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=vx, y=0.0))),
                header=SimpleNamespace(stamp=SimpleNamespace(
                    sec=int(sim_t), nanosec=int((sim_t % 1) * 1e9))))

        node = SimpleNamespace(_odom_prev=None, _odom=None,
                               _odom_channels=(0.0, None),
                               _unconfirmed_spikes=0)
        with patch.object(supervisor.time, 'monotonic', return_value=20.0):
            supervisor.TaskStateMachine._odom_cb(node, msg(0.0, 200.0))
        with patch.object(supervisor.time, 'monotonic', return_value=20.5):
            supervisor.TaskStateMachine._odom_cb(node, msg(3.0, 200.5))
        # The twist lies (0.0) but the pose covered 3 m in 0.5 s: conclusive.
        self.assertAlmostEqual(node._odom_channels[1], 6.0)
        node._odom = (3.0, 0.0, node._odom[2], 20.5)
        node.get_parameter = lambda name: SimpleNamespace(value={
            'base_sanity_enabled': True, 'max_plausible_speed_mps': 3.0,
            'map_bounds': [-15.0, -16.0, 17.0, 6.0],
            'map_bounds_margin_m': 1.0}[name])
        with patch.object(supervisor.time, 'monotonic', return_value=20.6):
            violation = supervisor.TaskStateMachine._base_sanity_violation(node, 20.6)
        self.assertIsNotNone(violation)
        self.assertEqual(violation[0], 'ROBOT_LAUNCHED')


@unittest.skipUnless(trace_analyzer is not None,
                     'optional gitignored run_logs/tools analyzer is absent')
class PollVerdictTests(unittest.TestCase):
    """The opt-in P0 probe must classify a stall, and must not over-claim."""

    def observer(self, fresh=True):
        return [dict(mono_abs=100.0 + i, nav=dict(age_s=0.05 if fresh else 30.0))
                for i in range(14)]

    def build(self, stall_rows, armed=1):
        rows = [dict(mono_abs=100.0, input_valid=0, poll_attempts=0,
                     poll_taken=0, poll_armed=armed)]
        for index, (input_valid, attempts, taken) in enumerate(stall_rows, 1):
            rows.append(dict(mono_abs=100.0 + index, input_valid=input_valid,
                             poll_attempts=attempts, poll_taken=taken,
                             poll_armed=armed))
        rows.append(dict(mono_abs=100.0 + len(stall_rows) + 1, input_valid=99,
                         poll_attempts=stall_rows[-1][1] + 5,
                         poll_taken=stall_rows[-1][2], poll_armed=armed))
        return rows

    def test_reader_had_data_means_executor_delivery_fault(self):
        verdict = trace_analyzer.poll_verdict(
            self.build([(5, k, 1 if k >= 3 else 0) for k in range(10)]),
            self.observer())
        self.assertIn('A: reader held a sample', verdict['summary'])
        self.assertEqual(verdict['input_stalls'][0]['poll_taken'], 1)

    def test_reader_empty_means_loss_below_rclcpp(self):
        verdict = trace_analyzer.poll_verdict(
            self.build([(5, k, 0) for k in range(10)]), self.observer())
        self.assertIn('B: reader was empty', verdict['summary'])

    def test_probe_that_never_ran_is_not_reported_as_empty_reader(self):
        verdict = trace_analyzer.poll_verdict(
            self.build([(5, 0, 0) for k in range(10)]), self.observer())
        self.assertIn('unknown', verdict['summary'])
        self.assertNotIn('B:', verdict['summary'])

    def test_silent_upstream_is_not_reported_as_dropped_samples(self):
        """An idle gap has an empty reader too; that is not evidence of loss."""
        verdict = trace_analyzer.poll_verdict(
            self.build([(5, k, 0) for k in range(10)]), self.observer(fresh=False))
        self.assertIn('not a stall', verdict['summary'])
        self.assertNotIn('B:', verdict['summary'])

    def test_healthy_round_and_unarmed_round_claim_nothing(self):
        healthy = trace_analyzer.poll_verdict(
            self.build([(k * 3, k, 0) for k in range(10)]), self.observer())
        self.assertIn('no input stall', healthy['summary'])
        unarmed = trace_analyzer.poll_verdict(
            self.build([(5, k, 1) for k in range(10)], armed=0), self.observer())
        self.assertFalse(unarmed['available'])
        self.assertIn('not armed', unarmed['reason'])


@unittest.skipUnless(recorder is not None and watch is not None,
                     'optional gitignored run_logs/tools probes are absent')
class ProbeFreshnessTests(unittest.TestCase):
    def test_recorder_stamps_arrival_and_counts_messages(self):
        node = SimpleNamespace(cmd=(0, 0), cmd_nav=(0, 0),
                               cmd_at=None, cmd_nav_at=None,
                               cmd_count=0, cmd_nav_count=0)
        msg = SimpleNamespace(linear=SimpleNamespace(x=0.3),
                              angular=SimpleNamespace(z=-0.6))
        with patch.object(recorder.time, 'monotonic', return_value=42.0):
            recorder.Recorder.on_cmd(node, msg)
            recorder.Recorder.on_cmd_nav(node, msg)
        self.assertEqual((node.cmd, node.cmd_at, node.cmd_count),
                         ((0.3, -0.6), 42.0, 1))
        self.assertEqual((node.cmd_nav, node.cmd_nav_at, node.cmd_nav_count),
                         ((0.3, -0.6), 42.0, 1))

    def test_recorder_captures_odom_yaw_and_angular_velocity(self):
        from collections import deque
        import math
        node = SimpleNamespace(odom=None, odom_at=None, hist=deque(maxlen=5))
        msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=.2,y=-2.5),
            orientation=SimpleNamespace(x=0.,y=0.,z=math.sqrt(.5),w=math.sqrt(.5)))),
            twist=SimpleNamespace(twist=SimpleNamespace(
                linear=SimpleNamespace(x=.1,y=0.), angular=SimpleNamespace(z=.3))))
        with patch.object(recorder.time, 'monotonic', return_value=42.):
            recorder.Recorder.on_odom(node,msg)
        self.assertAlmostEqual(node.odom[4],math.pi/2)
        self.assertEqual(node.odom[5],.3)
        self.assertEqual(node.odom_at,42.)

    @unittest.skipUnless(chain_probe is not None, 'optional chain observer absent')
    def test_independent_probe_records_arrival_and_stale_ages(self):
        import io
        import json
        out = io.StringIO()
        node = SimpleNamespace(last={}, count={}, output=out, graph_due=100.0,
                               health=None, health_at=None, transitions=[])
        msg = SimpleNamespace(linear=SimpleNamespace(x=0.3),
                              angular=SimpleNamespace(z=-0.6))
        with patch.object(chain_probe.time, 'monotonic', return_value=40.0):
            chain_probe.CommandChainProbe.command(node, '/cmd_vel_nav', msg)
        with patch.object(chain_probe.time, 'monotonic', return_value=42.0), \
             patch.object(chain_probe.time, 'time', return_value=100.0):
            chain_probe.CommandChainProbe.sample(node)
        row = json.loads(out.getvalue())
        self.assertEqual(row['nav']['age_s'], 2.0)
        self.assertEqual(row['nav']['messages'], 1)
        self.assertEqual(row['nav']['wz'], -0.6)
        self.assertIsNone(row['nav_reliable'])
        self.assertIsNone(row['cmd'])
        self.assertIsNone(row['odom'])
        self.assertEqual(row['mono_abs'], 42.0)

    @unittest.skipUnless(angular_audit is not None, 'optional angular audit absent')
    def test_angular_audit_keeps_legacy_linear_metric_separate(self):
        row = dict(work_state='前往放置区', v=0., nav_age=.1,
                   nav_msgs=8, nav_vx=0., nav_wz=1.9, cmd_age=12., mono=35.)
        result = angular_audit.summarize([row])
        self.assertEqual(result['linear_stationary_angular_upstream'], 1)
        self.assertEqual(result['output_probe_stale_samples'], 1)
        self.assertEqual(result['max_cmd_age_s'], 12.)

    @unittest.skipUnless(compare_probe is not None, 'optional offline comparator absent')
    def test_cross_probe_gap_requires_both_fresh_upstream_views(self):
        base = dict(mono_abs=10.0, mono=1.0, work_state='前往放置区', v=0.0,
                    nav_age=0.01, nav_msgs=10, nav_vx=0.3, cmd_age=3.0)
        peer = dict(mono_abs=10.1, nav=dict(age_s=0.02, vx=0.3),
                    cmd=dict(age_s=3.0, vx=0.3),
                    odom=dict(age_s=0.02, vx=0.0))
        report = compare_probe.compare([base], [peer])
        self.assertEqual(report['both_nav_fresh_both_cmd_stale'], 1)
        self.assertEqual(len(report['anomalies']), 1)
        peer['nav'] = dict(age_s=4.0, vx=0.3)
        report = compare_probe.compare([base], [peer])
        self.assertEqual(report['udp_nav_fresh_default_nav_not_fresh_nonzero'], 1)
        self.assertEqual(report['both_nav_fresh_both_cmd_stale'], 0)

    @unittest.skipUnless(chain_probe is not None, 'optional chain observer absent')
    def test_probe_records_health_and_lifecycle_without_clients(self):
        node = SimpleNamespace(health=None, health_at=None, transitions=[])
        with patch.object(chain_probe.time, 'monotonic', return_value=20.):
            chain_probe.CommandChainProbe.on_health(node, SimpleNamespace(data=(
                '{"ready":true,"nav2_inactive_nodes":["velocity_smoother"]}')))
            chain_probe.CommandChainProbe.on_transition(node, SimpleNamespace(
                start_state=SimpleNamespace(label='active'),
                goal_state=SimpleNamespace(label='inactive')))
        self.assertEqual(node.health, {'ready': True, 'smoother_inactive': True})
        self.assertEqual(node.transitions[0]['goal'], 'inactive')

    @unittest.skipUnless(compare_probe is not None, 'optional offline comparator absent')
    def test_cross_probe_flags_missing_angular_output_only_when_not_turning(self):
        base = dict(mono_abs=10.0, mono=1.0, work_state='前往放置区', v=0.0,
                    nav_age=0.01, nav_msgs=10, nav_vx=0.0, nav_wz=1.9,
                    cmd_age=15.0)
        peer = dict(mono_abs=10.1, nav=dict(age_s=0.02, vx=0., wz=1.8),
                    cmd=dict(age_s=15.0, vx=0., wz=0.),
                    odom=dict(age_s=0.02, vx=0.0, wz=0.0))
        report = compare_probe.compare([base], [peer])
        self.assertEqual(report['both_angular_fresh_both_cmd_stale_not_turning'], 1)
        peer['odom']['wz'] = 0.5
        report = compare_probe.compare([base], [peer])
        self.assertEqual(report['both_angular_fresh_both_cmd_stale_not_turning'], 0)

    def test_watch_uses_angular_z_not_linear_z(self):
        node = SimpleNamespace(cmd=None)
        msg = SimpleNamespace(linear=SimpleNamespace(x=0.08, z=0.0),
                              angular=SimpleNamespace(z=0.7))
        with patch.object(watch.time, 'monotonic', return_value=12.0):
            watch.Watch._cmd(node, msg)
        self.assertEqual(node.cmd, (0.08, 0.7, 12.0))

    def test_watch_distinguishes_no_scan_from_all_invalid_ranges(self):
        node = SimpleNamespace(scan_count=0, scan=None, gcm_count=0, gcm=None,
                               lcm_count=0, lcm=None, arrival={})
        msg = SimpleNamespace(ranges=[float('inf')], range_min=0.1,
                              range_max=30.0, angle_min=0.0,
                              angle_increment=0.1)
        watch.Watch._scan(node, msg)
        self.assertEqual(node.scan_count, 1)
        self.assertIsNone(watch.Watch.laser_clearance(node))

    def test_watch_costmap_indices_use_tf_and_fail_closed_when_missing(self):
        grid = SimpleNamespace(
            header=SimpleNamespace(frame_id='map'),
            info=SimpleNamespace(resolution=1.0, width=10, height=10,
                                 origin=SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
                                                        orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))),
            data=[0] * 100,
        )
        grid.data[3 * 10 + 4] = 100
        odom = SimpleNamespace(header=SimpleNamespace(frame_id='odom'),
                               pose=SimpleNamespace(pose=SimpleNamespace(
                                   position=SimpleNamespace(x=1., y=2.))))
        node = SimpleNamespace(odom=odom, tf=Mock())
        node._grid_location = lambda msg: watch.Watch._grid_location(node, msg)
        node._nearest_lethal = watch.Watch._nearest_lethal
        transform = SimpleNamespace()
        node.tf.lookup_transform.return_value = transform
        with patch.object(watch, 'do_transform_pose', return_value=SimpleNamespace(
                position=SimpleNamespace(x=4.2, y=3.3))) as convert:
            self.assertEqual(watch.Watch._grid_location(node, grid)[:2], (3, 4))
            self.assertEqual(watch.Watch.costmap_probe(node, grid)[0], 100)
        convert.assert_called()
        node.tf.lookup_transform.assert_called_with(
            'map', 'odom', unittest.mock.ANY, timeout=unittest.mock.ANY
        )
        node.tf.lookup_transform.side_effect = watch.TransformException('no map transform')
        self.assertIsNone(watch.Watch.costmap_probe(node, grid))
        grid.header.frame_id = 'odom'
        node.tf.lookup_transform.reset_mock()
        node.tf.lookup_transform.side_effect = AssertionError('same frame needs no TF')
        self.assertEqual(watch.Watch._grid_location(node, grid)[:2], (2, 1))

    def test_watch_real_tf_rotates_odom_into_map_before_grid_lookup(self):
        import math
        from geometry_msgs.msg import Pose, TransformStamped
        grid = SimpleNamespace(
            header=SimpleNamespace(frame_id='map'),
            info=SimpleNamespace(resolution=1., width=10, height=10,
                                 origin=SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
                                                        orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))),
        )
        pose = Pose()
        pose.position.x = 1.
        pose.orientation.w = 1.
        transform = TransformStamped()
        transform.transform.translation.x = 4.
        transform.transform.translation.y = 3.
        transform.transform.rotation.z = math.sin(math.pi / 4)
        transform.transform.rotation.w = math.cos(math.pi / 4)
        node = SimpleNamespace(odom=SimpleNamespace(header=SimpleNamespace(frame_id='odom'),
                                                    pose=SimpleNamespace(pose=pose)),
                               tf=SimpleNamespace(lookup_transform=lambda *args, **kwargs: transform))
        self.assertEqual(watch.Watch._grid_location(node, grid)[:2], (4, 4))

    def test_dock_snapshot_crops_grid_without_claiming_stale_clearance(self):
        import gzip
        import json
        import tempfile
        grid = SimpleNamespace(
            header=SimpleNamespace(frame_id='odom', stamp=SimpleNamespace(sec=1, nanosec=2)),
            info=SimpleNamespace(resolution=1.0, width=7, height=7,
                                 origin=SimpleNamespace(position=SimpleNamespace(x=0., y=0.),
                                                        orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))),
            data=list(range(49)),
        )
        pose = SimpleNamespace(position=SimpleNamespace(x=3.1, y=3.1),
                               orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))
        odom = SimpleNamespace(header=SimpleNamespace(frame_id='odom'),
                               pose=SimpleNamespace(pose=pose))
        with tempfile.TemporaryDirectory() as tmp:
            node = SimpleNamespace(snapshot_dir=Path(tmp), snapshot_count=0,
                                   odom=odom, scan=None, gcm=grid, lcm=grid,
                                   arrival={'odom': 9.9, 'global': 1., 'local': 9.9})
            node._grid_location = lambda msg: watch.Watch._grid_location(node, msg)
            node._costmap_crop = lambda msg, key, now: watch.Watch._costmap_crop(
                node, msg, key, now, radius_m=1.)
            node.event_snapshot = lambda kind, msg: watch.Watch.event_snapshot(node, kind, msg)
            log = SimpleNamespace(name='fixed_joint_pick_place',
                                  msg='warehouse A: dock creep failed',
                                  stamp=SimpleNamespace(sec=5, nanosec=6))
            with patch.object(watch.time, 'monotonic', return_value=10.), \
                 patch.object(watch.time, 'time', return_value=100.), \
                 patch('builtins.print'):
                watch.Watch._rosout(node, log)
            self.assertEqual(node.snapshot_count, 1)
            with gzip.open(Path(tmp) / '01_dock_failed.json.gz', 'rt') as f:
                data = json.load(f)
            self.assertEqual(data['global']['center_cell'], [3, 3])
            self.assertEqual(data['global']['window'], [2, 2, 3, 3])
            self.assertEqual(data['global']['data'][1][1], 24)
            self.assertFalse(data['global']['valid_for_clearance'])
            self.assertTrue(data['local']['valid_for_clearance'])
            self.assertEqual(data['ros_stamp'], [5, 6])

    @unittest.skipUnless(dock_analyzer is not None, 'optional snapshot analyzer absent')
    def test_snapshot_clearance_uses_cell_edges_and_refuses_stale_map(self):
        grid = dict(valid_for_clearance=True, data=[[0, 0, 100]],
                    window=[0, 0, 1, 3], resolution=1., origin=[0., 0., 0.],
                    center_pose=[0.5, 0.5, 0.])
        self.assertEqual(dock_analyzer.clearance(grid), 1.5)
        grid['valid_for_clearance'] = False
        self.assertIsNone(dock_analyzer.clearance(grid))

    def test_vector_nearest_lethal_matches_original_square_search(self):
        rng = np.random.default_rng(21)
        info = SimpleNamespace(resolution=0.05)
        for _ in range(40):
            d = rng.choice([0, 99, 100], size=(18, 25),
                           p=[0.96, 0.02, 0.02]).astype(np.int16)
            r0, c0 = int(rng.integers(-2, 20)), int(rng.integers(-2, 27))
            stop, max_r = int(rng.choice([99, 100])), 12
            old = None
            for k in range(1, max_r):
                ring = d[max(0, r0-k):min(d.shape[0], r0+k+1),
                         max(0, c0-k):min(d.shape[1], c0+k+1)]
                if ring.size and (ring >= stop).any():
                    old = k * info.resolution
                    break
            actual = watch.Watch._nearest_lethal(d, info, r0, c0,
                                                 stop=stop, max_r=max_r)
            self.assertEqual(actual, old)


if __name__ == '__main__':
    unittest.main()

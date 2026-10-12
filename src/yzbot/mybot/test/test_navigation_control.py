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
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'fixed_joint_pick_place.py'
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

    def test_a_box_yields_while_the_base_is_close(self):
        """Driving a served pose into the base flings it; the box must wait.

        Measured on movers_live/final_round.log: ROBOT_LAUNCHED aborted the round
        with "base covered ground at 4.4 m/s (reported twist 1.3 m/s)".
        """
        movers = self.module()
        patrol = movers._Patrol('m', 0.0, 0.0, 2.0, 0.0, 0.2)
        patrol.s = 1.0
        node = SimpleNamespace(patrols=[patrol], _robot_xy=(1.2, 0.0))
        for name in ('_robot_too_close',):
            setattr(node, name, getattr(movers.MovingObstacles, name).__get__(node))
        self.assertTrue(movers.MovingObstacles._robot_too_close(node, patrol, 0.55))
        node._robot_xy = (5.0, 5.0)
        self.assertFalse(movers.MovingObstacles._robot_too_close(node, patrol, 0.55))
        # No odom yet, or the feature disabled: never block on it.
        node._robot_xy = None
        self.assertFalse(movers.MovingObstacles._robot_too_close(node, patrol, 0.55))

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
            check(str(path), Reporter(out, err))
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


class StartButtonPanelTests(unittest.TestCase):
    """The 「开始任务」 button lives in the dashboard layout, so its config is code.

    It shipped with messageType "std_msgs/Bool" (the ROS 1 form). Foxglove on a
    ROS 2 bridge cannot resolve that schema, so the panel showed
    "请在面板设置中配置主题和消息..." and the button stayed greyed out even
    though topic and payload were correct.
    """

    def panel(self):
        import json
        path = (ROOT / 'config' / 'foxglove' /
                'competition_dashboard_layout.json')
        return json.loads(path.read_text(encoding='utf-8'))[
            'configById']['Publish!start']

    def test_the_schema_uses_the_ros2_type_name(self):
        panel = self.panel()
        self.assertEqual(panel['messageType'], 'std_msgs/msg/Bool')
        self.assertNotEqual(panel['messageType'], 'std_msgs/Bool')

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


class SlowZoneTests(unittest.TestCase):
    """Tight-passage speed caps: inert unless configured, and always restored."""

    class FakePublisher:
        def __init__(self):
            self.sent = []

        def publish(self, message):
            self.sent.append(message)

    def node(self, zones, position=(12.78, -3.79), pub=None):
        return SimpleNamespace(
            _speed_limit_pub=pub,
            _active_speed_limit=0.0,
            _slow_zones=zones,
            _cached_base_position=lambda: position,
            get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1, nanosec=0))),
            get_parameter=lambda name: SimpleNamespace(
                value=({'slow_zones': zones} if name == 'slow_zones' else None)),
            get_logger=lambda: SimpleNamespace(
                warning=lambda *a, **k: None, info=lambda *a, **k: None,
                error=lambda *a, **k: None),
        )

    def test_the_off_sentinel_means_no_zones(self):
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=[0.0]))
        self.assertEqual(
            module.FixedJointPickPlace._parse_slow_zones(node), [])

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
            _navigate_client=client([True]), _attach_client=client([True]),
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
                                         'navigate', 'params_global',
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

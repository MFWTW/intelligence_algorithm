"""Wall-clock navigation budget regressions; no running ROS graph is needed."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import unittest
from unittest.mock import Mock


EXECUTOR = Path(__file__).resolve().parents[1] / 'scripts' / 'fixed_joint_pick_place.py'
METHODS = {
    '_send_goal_tolerant', '_release_nav_control', '_resume_nav_goal',
    '_navigate_once__timed', '_creep_to_dock__timed', '_entry_probe', '_entry_stage',
}
STATUS = SimpleNamespace(STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def wait(self, duration):
        assert duration >= 0.0
        self.waits.append(duration)
        self.now += duration


class Future:
    def __init__(self, clock, value, ready_at=0.0):
        self.clock = clock
        self.value = value
        self.ready_at = ready_at

    def done(self):
        return self.clock.now >= self.ready_at - 1e-9

    def result(self):
        if not self.done():
            raise AssertionError('result read before future completion')
        return self.value


def result(status, details=None):
    return SimpleNamespace(status=status, result=details)


class NavDeadlineTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ros = SimpleNamespace(
            ok=lambda: True,
            spin_once=lambda _node, timeout_sec: self.clock.wait(timeout_sec),
            spin_until_future_complete=lambda _node, future, timeout_sec: self.clock.wait(
                min(timeout_sec, max(0.0, future.ready_at - self.clock.now))
            ),
        )
        parsed = ast.parse(EXECUTOR.read_text(encoding='utf-8'), filename=str(EXECUTOR))
        cls = next(node for node in parsed.body
                   if isinstance(node, ast.ClassDef) and node.name == 'FixedJointPickPlace')
        methods = [node for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name in METHODS]
        self.assertEqual({node.name for node in methods}, METHODS)
        env = {
            'rclpy': self.ros, 'math': math, 'Sequence': Sequence,
            'ActionClient': object, 'GoalStatus': STATUS,
            'Twist': lambda: SimpleNamespace(
                linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0)),
            'time': SimpleNamespace(monotonic=lambda: self.clock.now, sleep=self.clock.wait),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(EXECUTOR), 'exec'), env)
        self.node = SimpleNamespace(_nav_leg_deadline=10.0, _last_attempt_failure=None,
                                    _cancel_requested=False, _navigation_control_fault=False)
        for name in METHODS:
            setattr(self.node, name, env[name].__get__(self.node))
        params = {'action_retry_count': 3, 'service_timeout_sec': 5.0,
                  'navigation_timeout_sec': 8.0, 'navigation_accept_radius_m': 0.25,
                  'dock_creep_attempts': 2, 'dock_creep_timeout_sec': 4.0}
        self.params = params
        self.node.get_parameter = lambda name: SimpleNamespace(value=self.params.get(name, 0.0))
        self.node.get_logger = Mock()
        self.node._stop_base = Mock()
        self.node._ensure_yield_gate_clear = lambda: True
        self.node._remaining_chain = lambda chain: chain
        self.node._tick_yield = Mock()
        self.node._yield_suppress = False
        self.node._yield_arc_goal = None
        self.node._yield_decision = SimpleNamespace(holds=False)
        self.node._cached_planar_distance_to_point = lambda *_args: 0.1
        self.node._cached_goal_error = lambda *_args: None
        self.node._log_carried_cube_state = Mock()
        self.node._nav_goal = lambda *_args, **_kwargs: (object(), self.client)
        self.node._entry_probe_tree = lambda: 'probe.xml'

    def goal(self, *, response_at=0.0, result_at=math.inf, cancel_at=0.1,
             wrapped=None):
        future = Future(self.clock, wrapped or result(STATUS.STATUS_SUCCEEDED), result_at)

        def cancel():
            if cancel_at is not None:
                future.ready_at = self.clock.now + cancel_at
                future.value = result(STATUS.STATUS_CANCELED)
            return Future(self.clock, None, math.inf)  # missing cancel ACK

        handle = SimpleNamespace(accepted=True, get_result_async=lambda: future,
                                 cancel_goal_async=Mock(side_effect=cancel))
        self.client = SimpleNamespace(
            send_goal_async=Mock(return_value=Future(self.clock, handle, response_at)))
        return handle

    def test_expired_resume_does_not_send(self):
        self.goal()
        self.node._nav_leg_deadline = 0.0
        self.assertIsNone(self.node._resume_nav_goal('leg', object(), 5.0, self.client))
        self.client.send_goal_async.assert_not_called()
        self.assertEqual(self.node._last_attempt_failure, 'deadline')

    def test_shorter_attempt_budget_includes_goal_response_and_late_cancel(self):
        handle = self.goal(response_at=0.35)
        reached = self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0],
                                                  max_duration_sec=0.3)
        self.assertFalse(reached)
        self.assertEqual(self.node._last_attempt_failure, 'deadline')
        self.assertFalse(self.node._navigation_control_fault)
        self.assertAlmostEqual(self.clock.waits[0], 0.3)
        self.assertLessEqual(self.clock.now, 0.8)
        self.client.send_goal_async.assert_called_once()
        handle.cancel_goal_async.assert_called_once()

    def test_non_navigation_goal_keeps_tolerant_retry(self):
        self.node._nav_leg_deadline = None
        rejected = SimpleNamespace(accepted=False)
        self.client = SimpleNamespace(send_goal_async=Mock(side_effect=[
            Future(self.clock, None, math.inf), Future(self.clock, rejected),
        ]))
        answer = self.node._send_goal_tolerant('arm', self.client, object(), 0.1, 0.2)
        self.assertIs(answer, rejected)
        self.assertEqual(self.client.send_goal_async.call_count, 2)

    def test_missing_goal_ack_fails_closed_without_resend(self):
        self.goal(response_at=math.inf)
        self.node._nav_leg_deadline = 0.2
        with self.assertRaisesRegex(RuntimeError, 'NAV_STATE_UNKNOWN'):
            self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0])
        self.assertEqual(self.node._last_attempt_failure, 'nav_control_not_released')
        self.assertTrue(self.node._navigation_control_fault)
        self.client.send_goal_async.assert_called_once()
        self.assertAlmostEqual(self.clock.now, 0.7)

    def test_unconfirmed_clearance_handoff_cannot_succeed(self):
        handle = self.goal(cancel_at=None)
        self.node._nav_leg_deadline = 0.2
        with self.assertRaisesRegex(RuntimeError, 'cancellation not confirmed'):
            self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0], completion_radius_m=0.25)
        handle.cancel_goal_async.assert_called_once()
        self.node._stop_base.assert_not_called()
        self.assertEqual(self.node._last_attempt_failure, 'nav_control_not_released')
        self.assertAlmostEqual(self.clock.now, 0.7)

    def test_clearance_terminal_result_can_arrive_without_cancel_ack(self):
        handle = self.goal()
        reached = self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0],
                                                  completion_radius_m=0.25)
        self.assertTrue(reached)
        handle.cancel_goal_async.assert_called_once()
        self.node._stop_base.assert_called()

    def test_expired_pose_does_not_override_the_deadline(self):
        self.goal(result_at=0.2)
        self.node._nav_leg_deadline = 0.2
        self.assertFalse(self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0]))
        self.assertEqual(self.node._last_attempt_failure, 'deadline')

    def test_only_structured_failure_is_classified(self):
        class PlannerResult:
            NO_VALID_PATH = 208
            error_code = 208

        self.goal(wrapped=result(STATUS.STATUS_ABORTED, SimpleNamespace(error_code=208)),
                  result_at=0.0)
        self.assertFalse(self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0]))
        self.assertIsNone(self.node._last_attempt_failure)
        self.goal(wrapped=result(STATUS.STATUS_ABORTED, PlannerResult()), result_at=0.0)
        self.assertFalse(self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0]))
        self.assertEqual(self.node._last_attempt_failure, 'planner_no_path')

    def test_chain_waypoint_change_resets_stall_baseline(self):
        handle = self.goal(result_at=0.51)
        self.params.update(navigation_stall_timeout_sec=0.22, navigation_stall_progress_m=0.2)
        self.node._entry_recovery_applies = lambda _label: False
        self.node._chain_index = 0

        def progress_target(_chain, x, y):
            if self.clock.now >= 0.3:
                self.node._chain_index = 1
                return (x, y)
            return (0.0, 0.0)

        self.node._chain_progress_target = progress_target
        self.node._cached_planar_distance_to_point = lambda x, _y: 0.1 if x == 0.0 else 1.0
        self.assertTrue(self.node._navigate_once__timed(
            'leg', [1.0, 2.0, 0.0], chain=[[0.0, 0.0, 0.0]]))
        handle.cancel_goal_async.assert_not_called()
        self.assertIsNone(self.node._last_attempt_failure)

    def test_dock_settling_uses_leg_budget_before_publishing(self):
        self.node._nav_leg_deadline = 0.08
        self.node._base_pose = lambda: (0.0, 0.0, 0.0)
        self.node._fine_cmd_pub = Mock()
        self.assertFalse(self.node._creep_to_dock__timed('dock', 0.1, 0.0, 0.0))
        self.node._fine_cmd_pub.publish.assert_not_called()
        self.assertEqual(self.node._last_attempt_failure, 'deadline')
        self.assertAlmostEqual(self.clock.now, 0.08)

    def test_plan_only_probe_counts_response_against_short_deadline(self):
        self.goal(response_at=0.15, result_at=0.3)
        self.assertEqual(self.node._entry_probe('entry', [1.0, 2.0, 0.0], 0.2), 'unknown')
        self.client.send_goal_async.assert_called_once()
        self.assertLessEqual(self.clock.now, 0.7)

    def test_unavailable_result_subscription_cancels_and_marks_ownership_unknown(self):
        handle = self.goal()
        handle.get_result_async = Mock(side_effect=RuntimeError('transport unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'NAV_STATE_UNKNOWN'):
            self.node._navigate_once__timed('leg', [1.0, 2.0, 0.0])
        handle.cancel_goal_async.assert_called_once()
        self.node._stop_base.assert_not_called()
        self.assertEqual(self.node._last_attempt_failure, 'nav_control_not_released')

    def test_probe_rejects_nonterminal_result_message(self):
        self.goal(result_at=0.0, wrapped=result(status=2))
        with self.assertRaisesRegex(RuntimeError, 'NAV_STATE_UNKNOWN'):
            self.node._entry_probe('entry', [1.0, 2.0, 0.0], 0.2)
        self.assertEqual(self.node._last_attempt_failure, 'nav_control_not_released')

    def test_expired_stage_does_not_accept_the_cached_pose(self):
        self.node._cached_base_pose = Mock(return_value=(1.0, 2.0, 0.0))
        self.node._nav_leg_deadline = 0.0
        self.assertFalse(self.node._entry_stage('entry', (1.0, 2.0, 0.0), 0.1, 0.1, 2.0, 3.0))
        self.node._cached_base_pose.assert_not_called()


if __name__ == '__main__':
    unittest.main()

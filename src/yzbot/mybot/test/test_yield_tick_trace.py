"""Offline trace tests: load the selected production code without ROS."""
import ast
import json
import math
import pathlib
import queue
import sys
import time
import unittest
from types import SimpleNamespace

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import obstacle_yield

# Only extract the isolated functions and methods under test. No ROS or simulator
# dependencies, and no duplicate implementation of the logic being exercised.
tree = ast.parse((SCRIPTS / 'fixed_joint_pick_place.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == 'FixedJointPickPlace')
names = {'_yield_step', '_yield_trace_capture', '_yield_trace_emit',
         '_tick_yield', '_yield_episode_elapsed', '_update_yield_episode',
         '_recent_yield_path', '_fresh_yield_path_cloud', '_yield_cloud_near',
         '_yield_close_path_clearance', '_yield_path_clearance_at_cloud',
         '_mover_cloud_box'}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
trace_functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in {'_yield_trace_json', '_YieldTraceWriter'}]
namespace = {'obstacle_yield': obstacle_yield, 'math': math, 'time': time,
             'json': json, 'queue': queue, 'YIELD_STOP_SPEED_LIMIT': 0.001,
             'CLOCK_MISMATCH_SEC': 1.0e6,
             'Time': SimpleNamespace(from_msg=lambda stamp: stamp),
             'TransformException': LookupError}
import threading
from dataclasses import fields, is_dataclass
namespace.update(threading=threading, fields=fields, is_dataclass=is_dataclass)
fragment = ast.Module(body=trace_functions + [ast.ClassDef(name='Node', bases=[],
    keywords=[], body=methods, decorator_list=[])], type_ignores=[])
exec(compile(ast.fix_missing_locations(fragment), '<production>', 'exec'), namespace)
Node = namespace['Node']


class TraceTests(unittest.TestCase):
    def node(self, writer):
        node = Node()
        node._yield_trace_writer = writer
        node._yield_config = obstacle_yield.YieldConfig()
        node._yield_base_motion = lambda: (0.0, 0.0, 0.0, 0.0, 0.0)
        node._base_pose = lambda: (0.0, 0.0, 0.0)
        node._yield_scan_points = lambda yaw: (0.1, [])
        node._yield_tracker = SimpleNamespace(update=lambda *a, **k: None,
                                               nearest=lambda *a, **k: None)
        node._yield_room = lambda *a: (None, None)
        node._carrying_now = lambda: False
        node._patrol_segments = []
        node._clock_seconds = lambda: 10.0
        node._yield_diag = lambda *a, **kw: None
        node._yield_log_decision = lambda *a: None
        node.get_clock = lambda: SimpleNamespace(clock_type='ROS_TIME')
        node.get_logger = lambda: SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None,
            error=lambda *a, **k: None)
        node.get_parameter = lambda name: SimpleNamespace(value=0.0)
        node._cached_base_position = lambda: (0.0, 0.0)
        node._slow_zone_limit = lambda: 0.0
        node.outputs = []
        node._publish_speed_limit = lambda *a: node.outputs.append(a)
        node._yield_suppress = False
        node._yield_pause_active = False
        node._yield_last_update = 0.0
        node._yield_decision = obstacle_yield.YieldDecision(
            obstacle_yield.ACTION_GO, obstacle_yield.REASON_CLEAR)
        node._yield_latched = False
        node._yield_wait_since = node._yield_clear_since = None
        node._yield_episode_since = node._yield_free_since = None
        node._yield_episode_anchor = None
        node._yield_release_from = node._yield_release_since = None
        node._yield_arc_attempts = 0
        node._yield_arc_used = False
        node._yield_arc_goal = None
        node._yield_arc_target = None
        node._yield_arc_abort = None
        node._yield_arc_watch_speed = None
        node._yield_watch_last = 0.0
        return node

    def close_node(self):
        node = self.node(None)
        node._yield_config = obstacle_yield.YieldConfig(
            simple_radius_m=1.8, simple_radius_commit_m=1.0, hold_release_margin_m=0.12)
        params = {'yield_path_release_enabled': False, 'empty_robot_radius_m': 0.32,
                  'yield_mover_cloud_timeout_sec': 0.35}
        node.get_parameter = lambda name: SimpleNamespace(value=params.get(name, 0.0))
        node._yield_path_goal = (3.0, 0.0)
        node._yield_path_sample = (time.monotonic(), 'map', [(0.0, 0.0), (3.0, 0.0)])
        node._yield_release_path = None
        node._clock_seconds = lambda: 10.0
        transform = SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        node._recovery_tf = SimpleNamespace(
            lookup_transform=lambda *args: SimpleNamespace(transform=transform))
        return node

    def set_cloud(self, node, y, *, x=0.4, stamp_sec=10, stamp_nsec=0, wall_age=0.0):
        node._mover_release_cloud = (time.monotonic() - wall_age,
                                     SimpleNamespace(sec=stamp_sec, nanosec=stamp_nsec),
                                     [(x, y)])

    def test_active_nav_blocks_oblique_mover_within_its_motion_envelope(self):
        node = self.close_node()
        # Outside the old cone, but less than 1 m from the origin. A static
        # 0.70 m clearance would pass; plausible motion consumes 0.30 m.
        self.set_cloud(node, 0.70)
        decision = node._yield_step(3.0)[0]
        self.assertTrue(decision.holds)
        self.assertIn('predicted near-path clearance', decision.detail)
        self.set_cloud(node, 0.80)
        self.assertFalse(node._yield_step(3.0)[0].holds)
        self.set_cloud(node, 0.80, stamp_sec=9, stamp_nsec=700_000_000)
        self.assertTrue(node._yield_step(3.0)[0].holds)

    def test_close_path_missing_stale_or_wrong_goal_and_recovery_are_not_clear(self):
        node = self.close_node()
        self.set_cloud(node, 0.80)
        self.assertTrue(node._yield_step(None)[0].holds)  # no Nav2 path for recovery
        good = node._yield_path_sample
        for sample in [None, (good[0] - 3.0, *good[1:]),
                       (*good[:2], [(0, 0), (5, 0)])]:
            with self.subTest(sample=sample):
                node._yield_path_sample = sample
                self.assertTrue(node._yield_step(3.0)[0].holds)
        node._yield_path_sample = good
        self.set_cloud(node, 0.80, wall_age=0.5)
        self.assertTrue(node._yield_step(3.0)[0].holds)
        self.set_cloud(node, 0.80)

        def no_transform(*args):
            raise LookupError

        node._recovery_tf.lookup_transform = no_transform
        self.assertTrue(node._yield_step(3.0)[0].holds)

    def test_path_hold_cannot_release_on_obstacle_data_loss(self):
        node = self.close_node()
        self.set_cloud(node, 0.60)
        self.assertTrue(node._yield_step(3.0)[0].holds)
        node._yield_pause_active = True
        node._yield_release_path = node._yield_path_sample[1:]
        node._yield_path_sample = None  # frozen route only while paused
        node._mover_release_cloud = None
        self.assertTrue(node._yield_step(3.0)[0].holds)
        self.set_cloud(node, 0.82)
        self.assertFalse(node._yield_step(3.0)[0].holds)

    def test_unconfirmed_lidar_track_neither_holds_nor_checks_path(self):
        node = self.close_node()
        node._yield_path_sample = None
        node._yield_tracker.nearest = lambda *a, **k: {
            'base_x': 0.6, 'base_y': 0.0, 'world_speed': 0.5}
        node._yield_close_path_clearance = lambda *a: self.fail('unconfirmed mover checked')
        for x, wall_age, stamp_sec in [(3.0, 0.0, 10), (0.6, 0.5, 10), (0.6, 0.0, 9)]:
            with self.subTest(x=x, wall_age=wall_age, stamp_sec=stamp_sec):
                self.set_cloud(node, 0.0, x=x, wall_age=wall_age, stamp_sec=stamp_sec)
                decision = node._yield_step(3.0)[0]
                self.assertEqual(decision.action, obstacle_yield.ACTION_GO)
                self.assertFalse(node._yield_close_path_hold)
        node._mover_release_cloud = None
        self.assertEqual(node._yield_step(3.0)[0].action, obstacle_yield.ACTION_GO)

    def test_confirmed_track_keeps_outer_cone_hold(self):
        node = self.close_node()
        node._yield_tracker.nearest = lambda *a, **k: {
            'base_x': 1.4, 'base_y': 0.0, 'world_speed': 0.2,
            'x': 1.4, 'y': 0.0, 'rel_vx': -0.2, 'rel_vy': 0.0, 'vx': -0.2, 'vy': 0.0}
        self.set_cloud(node, 0.0, x=1.4)
        self.assertTrue(node._yield_step(3.0)[0].holds)

    def test_other_hold_keeps_opt_in_release_check(self):
        node = self.close_node()
        params = {'yield_path_release_enabled': True, 'empty_robot_radius_m': 0.32}
        node.get_parameter = lambda name: SimpleNamespace(value=params.get(name, 0.0))
        node._yield_path_release_clearance = lambda _: self.fail('checked outside actual hold')
        self.assertFalse(node._yield_step(3.0)[0].holds)
        node._yield_pause_active = True
        for clearance, holds in [(None, True), (0.30, True), (0.44, True), (0.45, False)]:
            with self.subTest(clearance=clearance):
                node._yield_path_release_clearance = lambda _, c=clearance: c
                self.assertEqual(node._yield_step(3.0)[0].holds, holds)
        params['yield_path_release_enabled'] = False
        node._yield_path_release_clearance = lambda _: self.fail('checked with opt-in disabled')
        self.assertFalse(node._yield_step(3.0)[0].holds)

    def test_default_parameter_is_false(self):
        declarations = [n for n in ast.walk(cls) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'declare_parameter'
                        and n.args and isinstance(n.args[0], ast.Constant)
                        and n.args[0].value == 'yield_tick_trace']
        self.assertEqual(len(declarations), 1)
        self.assertIs(declarations[0].args[1].value, False)

    def test_strict_json_and_missing(self):
        obs = obstacle_yield.YieldObservation(box_x=math.nan, box_y=math.inf,
                                              box_vx=-math.inf, box_vy=1)
        data = json.loads(namespace['_yield_trace_json']({'obs': obs, 'absent': None}),
                          parse_constant=lambda s: self.fail(s))
        self.assertIsNone(data['obs']['box_x'])
        self.assertIsNone(data['obs']['box_y'])
        self.assertIsNone(data['obs']['box_vx'])
        self.assertIsNone(data['absent'])
        self.assertEqual(set(data['obs']), set(obs.__dataclass_fields__))

    def test_enabled_and_disabled_same_decision_and_output(self):
        records = []
        writer = SimpleNamespace(submit=records.append)
        off, on = self.node(None), self.node(writer)
        off._tick_yield(3.0)
        on._tick_yield(3.0)
        self.assertEqual(off._yield_decision, on._yield_decision)
        self.assertEqual(off.outputs, on.outputs)
        self.assertEqual(off._yield_latched, on._yield_latched)
        self.assertFalse(hasattr(off, '_yield_trace_sample'))
        self.assertEqual(len(records), 1)
        record = json.loads(namespace['_yield_trace_json'](records[0]))
        self.assertIsNone(record['observation'])
        self.assertIsNone(record['track'])
        self.assertEqual(record['scan']['point_count'], 0)
        self.assertEqual(record['final_decision']['action'], 'go')
        self.assertEqual(record['clock']['tick_domain'], 'python_monotonic_seconds')
        self.assertIn('hold_release_margin_m', record['config_thresholds'])

    def test_early_no_scan_explicit_missing(self):
        records = []
        node = self.node(SimpleNamespace(submit=records.append))
        node._yield_scan_points = lambda yaw: (None, None)
        node._tick_yield()
        record = json.loads(namespace['_yield_trace_json'](records[0]))
        self.assertIsNone(record['scan']['point_count'])
        self.assertIsNone(record['scan']['cluster_count'])
        self.assertTrue(record['observation_missing'])
        self.assertEqual(record['final_decision']['detail'], 'no fresh pose/scan')

    def test_tracked_observation_and_final_clear_hold(self):
        records = []
        node = self.node(SimpleNamespace(submit=records.append))
        node._yield_latched = True
        node._yield_wait_since = time.monotonic()
        node.get_parameter = lambda name: SimpleNamespace(
            value=0.35 if name == 'yield_mover_cloud_timeout_sec' else 0.0)
        self.set_cloud(node, 0.5, x=1.0)
        node._yield_tracker.nearest = lambda *a, **k: {
            'born': 10.0, 't': 10.1, 'x': 1.0, 'y': 0.5,
            'vx': 0.1, 'vy': 0.0, 'base_x': 1.0, 'base_y': 0.5,
            'rel_vx': 0.1, 'rel_vy': 0.0, 'world_speed': 0.1,
            'age': 0.0, 'life': 0.1}
        # Force a policy GO to verify the caller's clear-window STOP is what
        # gets recorded as final; the policy result remains separately visible.
        from unittest.mock import patch
        go = obstacle_yield.YieldDecision(obstacle_yield.ACTION_GO,
                                          obstacle_yield.REASON_CLEAR)
        with patch.object(obstacle_yield, 'decide', return_value=go):
            node._tick_yield(3.0)
        record = json.loads(namespace['_yield_trace_json'](records[0]))
        self.assertEqual(set(record['observation']),
                         set(obstacle_yield.YieldObservation.__dataclass_fields__))
        self.assertEqual(record['track']['born'], 10.0)
        self.assertIsNone(record['track']['stable_id'])
        self.assertEqual(record['policy_decision']['action'], 'go')
        self.assertEqual(record['final_decision']['action'], 'stop')
        self.assertEqual(record['final_decision']['detail'], 'confirming clear corridor')
        self.assertTrue(record['before']['latched'])
        self.assertTrue(record['after']['latched'])

    def test_same_phase_crossing_limits_and_dropout(self):
        """Replay one crossing through the real executor tick, without physics."""
        from dataclasses import replace
        from unittest.mock import patch

        def replay(enabled, dropout):
            node = self.node(None)
            node._yield_config = replace(node._yield_config,
                                         approach_trial_enabled=enabled)
            clock = SimpleNamespace(t=100.0)
            node._mover_cloud_box = lambda cloud: (3.85, 0.60) if clock.t < 101.0 else None
            node._fresh_yield_path_cloud = lambda now: (
                now, None, [(0.95, -0.20 + 0.20 * (clock.t - 101.0))])
            # Prescribed approach speed until the first stop tick, stationary after.
            speed = lambda: 1.0 if clock.t <= 101.0 else 0.0
            node._yield_base_motion = lambda: (speed(), 0.0, 0.0, 0.0, 0.0)

            def nearest(*args, **kwargs):
                t = clock.t - 100.0
                if t < 1.0 or (dropout and 2.0 <= t < 3.0):
                    return None
                # The scripted crossing track moves at 0.20 m/s from -0.20 m
                # to +0.65 m. The robot pose is fixed: we test commands, not clearance.
                y = -0.20 + 0.20 * (t - 1.0)
                return {'x': 0.95, 'y': y, 'vx': 0.0, 'vy': 0.20,
                        'base_x': 0.95, 'base_y': y,
                        'rel_vx': -speed(), 'rel_vy': 0.20, 'world_speed': 0.20}

            node._yield_tracker.nearest = nearest
            with patch.dict(namespace, {'time': SimpleNamespace(monotonic=lambda: clock.t)}):
                for tick in range(76):
                    clock.t = 100.0 + tick / 10.0
                    node._tick_yield(6.0)
            return [speed for speed, _ in node.outputs]

        off = replay(False, False)
        on = replay(True, False)
        off_dropout = replay(False, True)
        on_dropout = replay(True, True)
        self.assertEqual(on[0], 0.8)
        self.assertNotEqual(off[0], 0.8)
        self.assertEqual(on[10:35], [0.001] * 25)
        self.assertEqual(on_dropout[10:35], on[10:35])
        self.assertIn(0.0, off_dropout[26:30])
        self.assertEqual(off[26:30], [0.001] * 4)
        resume = on.index(0.6)
        # Occupancy prediction releases the hold on the first tick whose whole
        # slow-resume path clears the box, which can be one tick before the
        # 0.35 m hysteresis band the old rule used.
        self.assertGreaterEqual(resume, 38)
        self.assertEqual(on[resume:resume + 12], [0.6] * 5 + [1.0] * 5 + [0.0] * 2)
        self.assertEqual(on_dropout, on)

    def test_approach_trial_missing_observation_budget_is_bounded(self):
        from dataclasses import replace
        from unittest.mock import patch
        node = self.node(None)
        node._yield_config = replace(node._yield_config, approach_trial_enabled=True)
        node._yield_latched = True
        node._yield_episode_since = node._yield_wait_since = 100.0
        with patch.dict(namespace, {'time': SimpleNamespace(monotonic=lambda: 121.0)}):
            decision, latched, _ = node._yield_step(6.0)
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)
        self.assertFalse(latched)

    def test_rejected_arc_records_final_stop(self):
        from unittest.mock import patch
        records = []
        node = self.node(SimpleNamespace(submit=records.append))
        arc = obstacle_yield.YieldDecision(obstacle_yield.ACTION_ARC,
                                          obstacle_yield.REASON_ARC,
                                          arc_x=1.0, arc_y=0.5)
        with patch.object(obstacle_yield, 'decide', return_value=arc), \
             patch.object(obstacle_yield, 'arc_scan_clear', return_value=False):
            node._tick_yield()
        record = json.loads(namespace['_yield_trace_json'](records[0]))
        self.assertEqual(record['policy_decision']['action'], 'arc')
        self.assertEqual(record['final_decision']['action'], 'stop')
        self.assertIn('blocked', record['final_decision']['detail'])
        self.assertEqual(record['after']['latched'], node._yield_latched)

    def test_rejected_arc_trace_does_not_change_control(self):
        from unittest.mock import patch
        records = []
        off = self.node(None)
        on = self.node(SimpleNamespace(submit=records.append))
        arc = obstacle_yield.YieldDecision(obstacle_yield.ACTION_ARC,
                                          obstacle_yield.REASON_ARC,
                                          arc_x=1.0, arc_y=0.5)
        with patch.object(obstacle_yield, 'decide', return_value=arc), \
             patch.object(obstacle_yield, 'arc_scan_clear', return_value=False):
            off._tick_yield()
            on._tick_yield()
        self.assertEqual(off._yield_decision, on._yield_decision)
        self.assertEqual(off.outputs, on.outputs)
        self.assertTrue(off._yield_latched)
        self.assertEqual(off._yield_latched, on._yield_latched)
        self.assertEqual(off._yield_arc_attempts, on._yield_arc_attempts)
        self.assertEqual(len(records), 1)

    def test_trace_failures_do_not_change_output(self):
        def fail(*args):
            raise RuntimeError('diagnostic failure')
        off = self.node(None)
        off._tick_yield(3.0)
        for mode in ('capture', 'emit', 'writer'):
            on = self.node(SimpleNamespace(submit=fail))
            if mode == 'capture':
                on._yield_trace_capture = fail
            elif mode == 'emit':
                on._yield_trace_emit = fail
            on._tick_yield(3.0)
            self.assertEqual(on._yield_decision, off._yield_decision)
            self.assertEqual(on.outputs, off.outputs)
            self.assertEqual(on._yield_latched, off._yield_latched)

    def test_writer_close_wakes_idle_worker_and_rejects_submissions(self):
        lines = []
        writer = namespace['_YieldTraceWriter'](SimpleNamespace(warning=lines.append))
        writer.close()
        self.assertFalse(writer._thread.is_alive())
        self.assertIsNone(writer.logger)
        writer.submit({'not_logged': True})
        self.assertEqual(lines, [])

    def test_close_full_queue_with_blocked_logger_is_bounded(self):
        entered = threading.Event()
        release = threading.Event()
        lines = []
        def blocked(line):
            entered.set()
            release.wait(1.0)
            lines.append(line)
        writer = namespace['_YieldTraceWriter'](SimpleNamespace(warning=blocked))
        try:
            writer.submit({'first': True})
            self.assertTrue(entered.wait(1.0))
            for i in range(64):
                writer.submit({'queued': i})
            self.assertTrue(writer.queue.full())
            start = time.monotonic()
            writer.close()
            self.assertLess(time.monotonic() - start, 0.3)
            writer.submit({'after_close': True})
        finally:
            release.set()
            writer._thread.join(1.0)
        self.assertFalse(writer._thread.is_alive())
        self.assertIsNone(writer.logger)
        self.assertEqual(len(lines), 1)

    def test_full_queue_drops_without_blocking(self):
        writer = object.__new__(namespace['_YieldTraceWriter'])
        writer.queue = queue.Queue(maxsize=1)
        writer._stopped = threading.Event()
        writer.dropped = 0
        writer.submit({})
        writer.submit({})
        self.assertEqual(writer.dropped, 1)
        self.assertEqual(writer.queue.qsize(), 1)

    def test_a_suppressed_tick_without_an_arc_only_applies_the_slow_zone(self):
        """Suppression exists to stop a SECOND lateral goal, not the danger check.

        With no arc under way there is nothing to watch, and the tick must stay
        as cheap as it was before the watch existed.
        """
        node = self.node(None)
        node._yield_suppress = True
        node._slow_zone_limit = lambda: 0.40
        node._yield_step = lambda d, watch=False: self.fail('no arc to watch')
        node._tick_yield(1.0)
        self.assertEqual(node.outputs, [(0.40, 'Slow zone')])

    def test_the_arc_watch_holds_the_arc_when_the_prediction_fails(self):
        """A failed watch stops the arc leg instead of driving it blind.

        Measured 2026-11-21 (r53 rule 2, warehouse B): the sidestep leg ran
        20.10 s and failed with status 5 because nothing re-checked the mover
        while the policy was suppressed.
        """
        node = self.node(None)
        node._yield_suppress = True
        node._yield_arc_target = (1.0, -0.5)
        node._yield_step = lambda d, watch=False: obstacle_yield.ArcWatch(
            False, 0.0, 'the sidestep no longer clears: arc ...')
        node._tick_yield(1.0)
        self.assertEqual(node._yield_arc_abort,
                         'the sidestep no longer clears: arc ...')
        self.assertEqual(node._yield_arc_watch_speed, 0.0)
        self.assertEqual(node.outputs, [(0.001, 'Yield arc hold')])

    def test_the_arc_watch_caps_the_arc_when_only_the_crawl_clears(self):
        node = self.node(None)
        node._yield_suppress = True
        node._yield_arc_target = (1.0, -0.5)
        node._yield_step = lambda d, watch=False: obstacle_yield.ArcWatch(
            True, 0.3, 'arc 0.80,-0.55 at 0.30 m/s: predicted clearance ...')
        node._tick_yield(1.0)
        self.assertIsNone(node._yield_arc_abort)
        self.assertEqual(node.outputs, [(0.3, 'Yield arc watch')])

    def test_the_arc_watch_leaves_a_clear_arc_to_the_controller(self):
        node = self.node(None)
        node._yield_suppress = True
        node._yield_arc_target = (1.0, -0.5)
        node._yield_step = lambda d, watch=False: obstacle_yield.ArcWatch(True)
        node._tick_yield(1.0)
        self.assertIsNone(node._yield_arc_abort)
        self.assertEqual(node.outputs, [(0.0, '')])


if __name__ == '__main__':
    unittest.main()

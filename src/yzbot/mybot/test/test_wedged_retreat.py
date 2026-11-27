"""One lidar-verified retreat for a loaded leg the costgrid calls healthy.

Measured 2026-11-25 (runs/2026-11-25_watch_rule3, blue_cube_2 -> warehouse B):
the leg started clear, the base drove 1 m into the inflated band beside the
west wall, GridBased then aborted for 18 s, the behaviour server answered
"Collision Ahead" to every backup and spin, and the local costmap carried
lethal cells 0.02 m from the base centre - so the leg died as status=6 and the
round ended 3/5 with the cube attached. `_wedged_retreat` covers exactly that
hole: when the configured ladder is spent, retreat 0.25 m against what the
LIDAR sees and give the leg one more pass.

What this file guards:
- the retreat is gated on the scan (no clearance, no shove),
- it fires at most once per leg, even through the leg-level re-entry,
- it is the LAST resort: never taken while a cancel was requested,
- a leg whose retry also fails still fails (it must not loop).
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

BOT = Path(__file__).resolve().parents[1]
SCRIPT = BOT / 'scripts' / 'fixed_joint_pick_place.py'
LABEL = 'blue_cube_2 visual handoff'


def load_executor():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location('wedged_retreat_executor', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WedgedRetreatSourceTest(unittest.TestCase):
    """The wiring is only safe if the guard order is what the comment says."""

    def setUp(self):
        self.source = SCRIPT.read_text(encoding='utf-8')
        self.tree = ast.parse(self.source)

    def method_source(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(self.source, node)
        raise AssertionError(f'{name} not found')

    def test_navigate_to_asks_for_the_retreat_only_after_the_ladder(self):
        body = self.method_source('_navigate_to')
        loop_end = body.rindex('unchanged_retry_used = True')
        tail = body[loop_end:]
        # The post-loop call stays as the fallback for every other break path
        # (jam recovery, entry recovery, pickup fallback).
        self.assertIn("getattr(self, '_wedged_retry', None)", tail)
        while_head = body.index('while attempt < attempts:')
        while_body = body[while_head:loop_end]
        # Inside the loop it may appear exactly once, and only as the last
        # resort of the spent-attempt branch: after the configured ladder
        # (relax / pickup fallback / back-off) had its turn, and before the leg
        # is written off as having no verified recovery. Measured 2026-11-26
        # 14:39 rule2: spending the remaining attempts on a goal that could
        # never be planned from an inscribed start cell, and retreating only
        # afterwards, cost the round.
        ladder = while_body.index('self._relax_global_footprint(label)')
        retreat = while_body.index("getattr(self, '_wedged_retry', None)")
        verdict = while_body.index('no verified recovery')
        self.assertLess(ladder, retreat)
        self.assertLess(retreat, verdict)
        self.assertEqual(while_body.count("'_wedged_retry'"), 1)
        self.assertNotIn('_wedged_retreat(', while_body)

    def test_cancel_and_already_used_both_suppress_the_retreat(self):
        helper = self.method_source('_wedged_retry')
        self.assertIn("getattr(self, '_wedged_retry_active', False)", helper)
        self.assertIn("getattr(self, '_cancel_requested', False)", helper)
        retreat = self.method_source('_wedged_retreat')
        self.assertIn("getattr(self, '_wedged_retreat_used', False)", retreat)

    def test_the_retry_is_bounded_by_the_used_flag(self):
        body = self.method_source('_navigate_to')
        self.assertIn("if not getattr(self, '_wedged_retry_active', False):", body)
        self.assertIn('self._wedged_retreat_used = False', body)
        helper = self.method_source('_wedged_retry')
        self.assertIn('self._wedged_retry_active = True', helper)
        # One re-entry, not a loop: the recursive call is the return value.
        self.assertIn('return self._navigate_to(', helper)

    def test_the_retreat_refuses_to_shove_a_base_the_scan_blocks(self):
        body = self.method_source('_wedged_retreat')
        self.assertIn('best_room < 0.35', body)
        self.assertLess(
            body.index('if best_room is None or best_room < 0.35:'),
            body.index('self._wedged_retreat_used = True'),
            'the used flag must be set only once the retreat is really taken',
        )


class WedgedRetreatRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_executor()
        cls.executor = cls.module.FixedJointPickPlace

    def node(self, front=0.90, rear=0.40, enabled=True):
        readings = {1.0: front, -1.0: rear}

        def cone(direction):
            return readings.get(direction)

        values = {
            'start_band_escape_enabled': enabled,
            # The shove sizing asks for these two; a stub that only serves the
            # switch would raise KeyError instead of exercising the retreat.
            'start_band_escape_attempts': 2,
            'start_band_escape_retreat_m': 0.25,
        }
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
            _entry_scan_clearance=Mock(side_effect=cone),
            _back_off=Mock(return_value=True),
            get_logger=Mock(),
        )
        # The retreat now shares its axis choice and shove sizing with the
        # pre-flight escape; bind them the way the node has them.
        node._band_escape_axis = self.executor._band_escape_axis.__get__(node)
        node._band_escape_shove = self.executor._band_escape_shove.__get__(node)
        return node

    def test_retreats_along_the_route_it_drove(self):
        # The rear wins even when the front looks roomier: reversing re-traces
        # ground this leg has already covered.
        node = self.node(front=0.90, rear=0.60)
        self.assertTrue(self.executor._wedged_retreat(node, LABEL))
        node._back_off.assert_called_once()
        args, kwargs = node._back_off.call_args
        self.assertIn('wedged start pose', args[0])
        self.assertEqual(args[1:], (0.25, 0.08, -1.0))
        self.assertIs(kwargs['leave_band'], True)
        self.assertIs(node._wedged_retreat_used, True)

    def test_a_blocked_rear_swerves_to_the_open_side(self):
        node = self.node(front=0.90, rear=0.20)
        self.assertTrue(self.executor._wedged_retreat(node, LABEL))
        args, _ = node._back_off.call_args
        self.assertEqual(args[1:], (0.25, 0.08, 1.0))

    def test_a_blocked_scan_is_refused_without_touching_the_base(self):
        node = self.node(front=0.31, rear=0.20)
        self.assertFalse(self.executor._wedged_retreat(node, LABEL))
        node._back_off.assert_not_called()
        # Not marked used: nothing was spent, so a later leg can still retreat.
        self.assertFalse(getattr(node, '_wedged_retreat_used', False))

    def test_the_switch_off_means_no_retreat(self):
        node = self.node(enabled=False)
        self.assertFalse(self.executor._wedged_retreat(node, LABEL))
        node._entry_scan_clearance.assert_not_called()

    def test_a_second_retreat_in_the_same_leg_is_refused(self):
        node = self.node()
        node._wedged_retreat_used = True
        self.assertFalse(self.executor._wedged_retreat(node, LABEL))
        node._back_off.assert_not_called()


class WedgedRetreatNavigationTests(unittest.TestCase):
    """`_navigate_to` must spend the retreat once and still fail when it must."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_executor()
        cls.executor = cls.module.FixedJointPickPlace

    def make_node(self, retreat_ok=True, navigate_ok=False):
        params = {
            'navigation_retry_count': 1,
            'settle_sec': 0.0,
            'navigation_timeout_sec': 30.0,
        }
        attempts = []

        def navigate_once(label, *args, **kwargs):
            attempts.append(label)
            return navigate_ok

        node = SimpleNamespace(
            _yield_config=SimpleNamespace(enabled=False),
            _yield_decision=SimpleNamespace(holds=False),
            _yield_arc_goal=None,
            _log_cube_positions=Mock(),
            _pending_yield_arc=Mock(return_value=None),
            _entry_recovery_applies=Mock(return_value=False),
            _relax_global_footprint=Mock(return_value=False),
            _pickup_fallback_route=Mock(return_value=None),
            _local_recovery_pose_clear=Mock(return_value=True),
            _global_recovery_cost=Mock(return_value=50),
            _restore_nominal_geometry=Mock(),
            _clear_speed_limit=Mock(),
            _wedged_retreat=Mock(),
            get_parameter=lambda name: SimpleNamespace(value=params[name]),
            get_logger=Mock(),
        )
        def wedged_retreat(label):
            # The real method sets the flag itself; the stub must too, or the
            # re-entry would retreat for ever.
            node._wedged_retreat_used = retreat_ok
            return retreat_ok

        node._wedged_retreat.side_effect = wedged_retreat
        node._navigate_once = Mock(side_effect=navigate_once)
        node._wedged_retry = self.executor._wedged_retry.__get__(node)
        # The leg-level re-entry calls itself through the instance, as it does
        # on the real node.
        node._navigate_to = self.executor._navigate_to.__get__(node)
        node.attempts = attempts
        return node

    def test_a_wedged_leg_retreats_once_and_re_enters_the_leg(self):
        node = self.make_node(retreat_ok=True, navigate_ok=False)
        self.assertFalse(self.executor._navigate_to(node, LABEL, (-2.8, -11.6, 0.0)))
        node._wedged_retreat.assert_called_once()
        # Two attempts before the retreat, two more after the re-entry.
        self.assertEqual(len(node.attempts), 4, node.attempts)
        self.assertTrue(all(label == LABEL for label in node.attempts))

    def test_a_refused_retreat_ends_the_leg_at_once(self):
        node = self.make_node(retreat_ok=False, navigate_ok=False)
        self.assertFalse(self.executor._navigate_to(node, LABEL, (-2.8, -11.6, 0.0)))
        node._wedged_retreat.assert_called_once()
        self.assertEqual(len(node.attempts), 2, node.attempts)

    def test_a_healthy_leg_never_retreats(self):
        node = self.make_node(retreat_ok=True, navigate_ok=True)
        self.assertTrue(self.executor._navigate_to(node, LABEL, (-2.8, -11.6, 0.0)))
        node._wedged_retreat.assert_not_called()
        self.assertEqual(len(node.attempts), 1)

    def test_the_retreat_does_not_fire_when_the_leg_was_cancelled(self):
        node = self.make_node(retreat_ok=True, navigate_ok=False)
        node._cancel_requested = True
        self.assertFalse(self.executor._navigate_to(node, LABEL, (-2.8, -11.6, 0.0)))
        node._wedged_retreat.assert_not_called()


class BandEscapeShoveTests(unittest.TestCase):
    """The pre-flight shove follows the scan, not a fixed reverse direction.

    Measured 2026-11-25 (runs/2026-11-25_1918_servo_m3): the base stood 0.32 m
    from the wall with 0.51 m on its free side. The old code always reversed,
    demanded 0.60 m of scan clearance, and so refused the only shove that could
    have freed the leg; both legs then died with "no verified recovery".
    """

    @classmethod
    def setUpClass(cls):
        cls.module = load_executor()
        cls.executor = cls.module.FixedJointPickPlace

    def node(self, front=0.90, rear=0.40, attempts=2, retreat=0.25,
             band=True, global_cost=50):
        values = {
            'start_band_escape_enabled': True,
            'start_band_escape_attempts': attempts,
            'start_band_escape_retreat_m': retreat,
        }
        readings = {1.0: front, -1.0: rear}
        node = SimpleNamespace(
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
            _local_recovery_pose_clear=Mock(return_value=not band),
            _global_recovery_cost=Mock(return_value=global_cost),
            _entry_scan_clearance=Mock(side_effect=lambda d: readings.get(d)),
            _back_off=Mock(return_value=True),
            get_logger=Mock(),
        )
        node._band_escape_axis = self.executor._band_escape_axis.__get__(node)
        node._band_escape_shove = self.executor._band_escape_shove.__get__(node)
        return node

    def test_the_shove_keeps_a_quarter_metre_of_the_room_it_verified(self):
        node = self.node()
        for room, expected in ((0.90, (0.25, 0.47)), (0.51, (0.25, 0.47)),
                               (0.40, (0.15, 0.37))):
            distance, floor = self.executor._band_escape_shove(node, room)
            self.assertAlmostEqual(distance, expected[0], msg=room)
            self.assertAlmostEqual(floor, expected[1], msg=room)

    def test_a_band_escape_retreats_on_the_wider_side(self):
        node = self.node(front=0.51, rear=0.60)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_called_once()
        args, kwargs = node._back_off.call_args
        self.assertIn('leaving the band', args[0])
        self.assertEqual(args[1:], (0.25, 0.08, -1.0))
        self.assertIs(kwargs['leave_band'], True)
        self.assertAlmostEqual(kwargs['clearance_floor'], 0.47)

    def test_a_retreat_reverses_along_the_route_it_drove(self):
        # Reversing re-traces ground the base just covered; going forward
        # commits the body to space only one scan has looked at.
        node = self.node(front=2.00, rear=0.55)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        args, _ = node._back_off.call_args
        self.assertEqual(args[1:], (0.25, 0.08, -1.0))

    def test_a_blocked_rear_still_uses_the_open_side(self):
        node = self.node(front=0.55, rear=0.20)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        args, _ = node._back_off.call_args
        self.assertEqual(args[1:], (0.25, 0.08, 1.0))

    def test_the_scan_floor_now_admits_a_051_m_side(self):
        # The regression this guards: a 0.51 m side used to be refused by the
        # 0.60 m gate the generic retreat uses.
        node = self.node(front=0.51, rear=0.30)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_called_once()

    def test_the_pre_flight_shove_is_capped_per_leg(self):
        node = self.node(attempts=1)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_called_once()

    def test_a_clear_start_pose_is_left_alone(self):
        node = self.node(band=False)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_not_called()

    def test_a_lethal_global_start_cell_shoves_even_when_the_local_grid_is_clean(self):
        # Measured 2026-11-25 (runs/2026-11-25_2022_l2_m2): the base stopped
        # 0.21 m from a static wall, the laser saw open floor, and 58 GridBased
        # requests aborted because the start cell was lethal in the static layer.
        # The local grid alone must not be able to veto the shove.
        node = self.node(band=False, global_cost=99)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_called_once()

    def test_an_unpublished_global_grid_is_not_a_veto(self):
        node = self.node(band=False, global_cost=None)
        self.executor._ensure_start_pose_outside_band(node, LABEL)
        node._back_off.assert_not_called()


class BandStallGuardTests(unittest.TestCase):
    """A leg that parks itself inside the planner's band must not wait it out.

    Measured 2026-11-26 14:39 (runs/2026-11-26_1439_ann3_m2, rule 2): the base
    carried 1.31 m/s through a 27 deg kink, drifted 0.35 m inside and stopped in
    a cell whose global cost is 99. NavFn refused to plan from it for 12.6 s and
    the loaded leg had no retry left, so the round died 0/5. The stall monitor
    now fails such an attempt after `navigation_band_stall_grace_sec` instead of
    watching the full stall timeout.
    """

    def setUp(self):
        self.source = SCRIPT.read_text(encoding='utf-8')

    def test_the_band_guard_runs_before_the_long_stall_timeout(self):
        band = self.source.find("self._last_attempt_failure = 'planner_band'")
        self.assertGreater(band, 0, 'the band guard is not wired in')
        self.assertLess(
            band,
            self.source.find('no progress for {stall_timeout:.0f} s'),
            'the band guard must fire before the generic stall timeout',
        )

    def test_the_band_guard_asks_the_global_grid(self):
        start = self.source.find("self._last_attempt_failure = 'planner_band'")
        window = self.source[max(0, start - 700):start + 200]
        self.assertIn('_start_cell_inside_global_band()', window)
        self.assertIn('band_grace', window)

    def test_both_configs_carry_the_grace(self):
        for name in ('config/fixed_joint_pick_place.yaml',
                     'config/profiles/executor_fast.yaml'):
            text = (BOT / name).read_text(encoding='utf-8')
            self.assertIn('navigation_band_stall_grace_sec: 3.0', text, name)


if __name__ == '__main__':
    unittest.main()

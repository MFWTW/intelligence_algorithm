"""The aligning base must not park itself on the planner's inscribed band.

Measured 2026-11-26 (run 2026-11-26_1539_A9_r1_2, blue_cube_5): the base
stopped 0.25 m from a shelf at global cost 80, the visual centring turn swung
it 79 deg at 1.4 rad/s, and while rotating the skid-steer chassis slid 0.18 m
towards the shelf - odom and the Gazebo pose agree on the slide, so it is real
motion, not a pose estimate. The cell under the wheel crossed 80 -> 99, and the
warehouse leg that followed had no legal start pose: NavFn refused on the first
tick, backup and spin were refused by the collision monitor, the guarded
retreat was cut short by its own motion tolerance, and the round ended 4/5.

The drift is not geometric - a sweep-radius prediction cannot see it - so the
guard watches the cell the base actually stands on, once per vision frame, and
steps back when that cell enters the band.

What this file guards:
- the watch reads the *global* cost of the base cell, not a guess,
- it retreats only when the cell is >= 99 and the retreat is bounded,
- it gives up after the configured attempts instead of shuffling forever,
- it runs inside the alignment loop, before the servo command is published.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

BOT = Path(__file__).resolve().parents[1]
SCRIPT = BOT / 'scripts' / 'fixed_joint_pick_place.py'


def load_executor():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location('align_band_guard_executor', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Param:
    def __init__(self, value):
        self.value = value


class _Node(SimpleNamespace):
    def get_parameter(self, name):
        return _Param(self.params[name])

    def get_logger(self):
        return self.logger

    def _global_recovery_cost(self):
        return self.cost

    def _recovery_pose(self, frame):
        return (0.0, 0.0, 0.0) if self.pose_ok else None

    def _entry_scan_clearance(self, direction):
        return self.room[direction]

    def _global_cell_cost(self, x, y):
        # Cost falls off with distance along the x axis, so the test can ask for
        # "legal only past 0.4 m forward" and watch which step it picks.
        if x >= self.legal_forward:
            return 0
        if x <= -self.legal_reverse:
            return 0
        return 99

    def _back_off(self, *a, **k):
        self.shoves.append((a, k))
        return self.retreat_ok


class AlignBandGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_executor()

    def make_node(self, cost, enabled=True, attempts=2, retreat_ok=True,
                  room=None, legal_forward=9.0, legal_reverse=9.0,
                  pose_ok=True):
        room = {-1.0: 2.0, 1.0: 2.0} if room is None else dict(room)
        logger = SimpleNamespace(warning=lambda msg: None, info=lambda msg: None)
        node = _Node(
            params={
                'align_band_guard': enabled,
                'align_band_retreat_m': 0.28,
                'align_band_retreat_max_m': 0.60,
                'align_band_retreat_speed_mps': 0.08,
                'align_band_guard_attempts': attempts,
            },
            cost=cost,
            logger=logger,
            retreat_ok=retreat_ok,
            shoves=[],
            _align_band_retreats=0,
            room=room,
            legal_forward=legal_forward,
            legal_reverse=legal_reverse,
            pose_ok=pose_ok,
        )
        node.guard = self.module.FixedJointPickPlace._guard_alignment_band
        node._plan_band_escape = (
            self.module.FixedJointPickPlace._plan_band_escape.__get__(node))
        return node

    def test_a_cell_in_the_band_triggers_one_retreat(self):
        node = self.make_node(cost=99, legal_reverse=0.20)
        self.assertTrue(node.guard.__get__(node)())
        (args, kwargs), = node.shoves
        self.assertEqual(args[0], 'alignment band')
        self.assertAlmostEqual(args[2], 0.08)
        self.assertEqual(args[3], -1.0)   # reverse, back the way it came
        self.assertEqual(node._align_band_retreats, 1)
        self.assertAlmostEqual(args[1], 0.28)

    def test_it_switches_axis_when_the_reverse_stays_banned(self):
        """Measured: reversing stayed on 99 for 0.6 m, forward reached 0 at 0.44."""
        node = self.make_node(cost=99, legal_forward=0.42, legal_reverse=9.0)
        self.assertTrue(node.guard.__get__(node)())
        (args, _kwargs), = node.shoves
        self.assertEqual(args[3], 1.0)
        self.assertGreaterEqual(args[1], 0.42)

    def test_it_never_shoves_further_than_the_scan_allows(self):
        node = self.make_node(cost=99, room={-1.0: 0.30, 1.0: 0.30},
                              legal_forward=0.25, legal_reverse=0.25)
        self.assertTrue(node.guard.__get__(node)())
        (args, _kwargs), = node.shoves
        self.assertLessEqual(args[1], 0.35)

    def test_a_legal_cell_is_left_alone(self):
        node = self.make_node(cost=80)
        self.assertFalse(node.guard.__get__(node)())
        self.assertEqual(node.shoves, [])
        self.assertEqual(node._align_band_retreats, 0)

    def test_no_pose_yet_means_no_shove(self):
        node = self.make_node(cost=99, pose_ok=False)
        self.assertFalse(node.guard.__get__(node)())
        self.assertEqual(node.shoves, [])

    def test_an_unknown_cell_is_not_read_as_free(self):
        """None means "no map/sweep": it must never trigger a blind shove."""
        node = self.make_node(cost=None)
        self.assertFalse(node.guard.__get__(node)())
        self.assertEqual(node.shoves, [])

    def test_it_stops_after_the_configured_attempts(self):
        node = self.make_node(cost=100, attempts=1)
        guard = node.guard.__get__(node)
        self.assertTrue(guard())
        self.assertFalse(guard())
        self.assertEqual(len(node.shoves), 1)

    def test_the_guard_can_be_switched_off(self):
        node = self.make_node(cost=100, enabled=False)
        self.assertFalse(node.guard.__get__(node)())
        self.assertEqual(node.shoves, [])

    def test_a_refused_retreat_still_counts_as_an_attempt(self):
        """Otherwise a blocked retreat would be retried every frame for 4 s."""
        node = self.make_node(cost=100, retreat_ok=False)
        self.assertFalse(node.guard.__get__(node)())
        self.assertEqual(node._align_band_retreats, 1)


class AlignBandGuardWiringTests(unittest.TestCase):
    """A guard that exists but never runs is the failure mode this catches."""

    def setUp(self):
        self.source = SCRIPT.read_text()

    def test_the_watch_runs_once_per_frame_inside_the_align_loop(self):
        align = self.source.split('def _visual_align_active__timed(', 1)[1]
        align = align.split('\ndef ', 1)[0]
        watch = align.index("getattr(self, '_guard_alignment_band', None)")
        command = align.index('self._fine_cmd_pub.publish(command)')
        self.assertLess(watch, command)
        # The previous frame's command must not be able to carry the base in.
        self.assertIn('if band_guard is not None and band_guard():\n                    continue', align)

    def test_the_counter_is_reset_for_each_grasp(self):
        start = self.source.index('self._visual_range_gate_active = True')
        self.assertIn('self._align_band_retreats = 0', self.source[start:start + 200])

    def test_the_parameters_carry_the_measured_defaults(self):
        for line in (
            "declare_parameter('align_band_guard', True)",
            "declare_parameter('align_band_retreat_m', 0.28)",
            "declare_parameter('align_band_retreat_max_m', 0.60)",
            "declare_parameter('align_band_retreat_speed_mps', 0.08)",
            "declare_parameter('align_band_guard_attempts', 2)",
        ):
            self.assertIn(line, self.source)


if __name__ == '__main__':
    unittest.main()

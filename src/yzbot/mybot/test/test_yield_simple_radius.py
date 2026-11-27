"""The 2026-11-25 operator rule: one circle around the base, nothing else.

A moving box inside ``simple_radius_m`` means "wait"; anything outside it means
"drive on and ignore it". These tests pin the three properties that matter:

* the decision depends on range only - not on the box's heading, its relative
  velocity, the lane it patrols, or which side it is on;
* a box that is standing still inside the circle still holds the base (the old
  band policy released there, because a static box never enters the swept path);
* the circle can be switched off, in which case the old policy decides again.
"""
import math
import pathlib
import sys
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import obstacle_yield


def observation(x, y, *, vx=0.0, vy=0.0, latched=False, scan_age=0.0):
    return obstacle_yield.YieldObservation(
        box_x=x, box_y=y, box_vx=vx, box_vy=vy,
        box_world_speed_mps=math.hypot(vx, vy),
        scan_age_sec=scan_age, latched=latched,
        lane_distance_m=0.0,
    )


class SimpleRadiusTests(unittest.TestCase):
    def setUp(self):
        self.cfg = obstacle_yield.YieldConfig(simple_radius_m=1.5)

    def decide(self, obs, **kwargs):
        return obstacle_yield.decide(obs, self.cfg, **kwargs)

    def test_box_outside_the_circle_is_ignored(self):
        decision = self.decide(observation(2.4, 0.1, vx=-0.2))
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)
        self.assertFalse(decision.holds)
        self.assertIn('outside the 1.50 m circle', decision.detail)

    def test_box_inside_the_circle_stops_the_base(self):
        decision = self.decide(observation(1.2, 0.4, vx=-0.2))
        self.assertEqual(decision.action, obstacle_yield.ACTION_STOP)
        self.assertTrue(decision.holds)
        self.assertEqual(decision.reason, obstacle_yield.REASON_WAIT)
        self.assertIn('inside the 1.50 m circle', decision.detail)

    def test_the_radius_is_measured_from_the_base_centre(self):
        # Straight ahead on the x axis: 1.45 m away is inside, 1.55 m outside.
        self.assertTrue(self.decide(observation(1.45, 0.0)).holds)
        self.assertFalse(self.decide(observation(1.55, 0.0)).holds)

    def test_a_box_beside_the_base_does_not_stop_it(self):
        # The operator's report, 2026-11-25: a patrol box grazing the rim of the
        # half circle (nearly 90 deg to the side) stopped the base while it was
        # driving past. A box 0.70 m ahead and 1.20 m to the side (59.7 deg off
        # the nose) cannot be driven into; it must not cost a stop.
        decision = self.decide(observation(0.70, 1.20))
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)
        self.assertIn('outside the 100 deg cone', decision.detail)
        self.assertIn('bearing +60 deg', decision.detail)

    def test_the_cone_edge_is_included(self):
        # +/-50 deg (the operator's number, 2026-11-25): 49.9 deg off the nose is
        # still "in the way", 50.7 deg is not.
        self.assertTrue(self.decide(observation(0.90, 1.07)).holds)
        self.assertFalse(self.decide(observation(0.90, 1.10)).holds)

    def test_the_cone_can_be_widened_back_to_the_front_half(self):
        cfg = obstacle_yield.YieldConfig(
            simple_radius_m=1.5, simple_radius_fov_deg=180.0)
        self.assertTrue(obstacle_yield.decide(observation(0.70, 1.20), cfg).holds)

    def test_a_still_box_inside_the_circle_still_holds(self):
        # The failure this replaces: a box that had stopped 0.40 m away was
        # released by the band policy because it could never enter the swept
        # path, and the base drove into it (2026-11-25 rule1: base flung at
        # 3.2 m/s once the box overlapped).
        decision = self.decide(observation(0.4, 0.0, vx=0.0, vy=0.0))
        self.assertTrue(decision.holds)

    def test_a_box_behind_the_base_does_not_stop_it(self):
        # The front half was added after measuring the full circle: over three
        # 5/5 rounds (runs/2026-11-25_1640_circle_m1..m3) 15/45/29 of the
        # 43/107/118 stops were for boxes behind the base, and object time rose
        # to 305/391/369 s from 299/315/264 s. A box behind cannot be driven
        # into, so it must not cost a stop.
        decision = self.decide(observation(-1.0, 0.2))
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)
        self.assertIn('behind the base', decision.detail)

    def test_the_full_circle_can_be_asked_for_explicitly(self):
        cfg = obstacle_yield.YieldConfig(
            simple_radius_m=1.5, simple_radius_front_only=False)
        decision = obstacle_yield.decide(observation(-1.0, 0.0), cfg)
        self.assertTrue(decision.holds)

    def test_a_stale_scan_never_freezes_the_base_inside_the_circle(self):
        cfg = obstacle_yield.YieldConfig(simple_radius_m=1.5)
        decision = obstacle_yield.decide(
            observation(0.8, 0.0, scan_age=cfg.track_timeout_sec + 0.5), cfg)
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)

    def test_zero_radius_restores_the_old_policy(self):
        cfg = obstacle_yield.YieldConfig(simple_radius_m=0.0)
        decision = obstacle_yield.decide(observation(1.2, 0.4, vx=-0.2), cfg)
        self.assertNotIn('circle', decision.detail)

    def test_the_competition_node_asks_for_the_circle(self):
        # The library default stays 0.0 so the band policy keeps its tests; the
        # node that actually drives the round must opt in, or the circle would
        # never be active in a race.
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / 'scripts' / 'fixed_joint_pick_place.py').read_text()
        self.assertIn("declare_parameter('yield_simple_radius_m', 1.8)", source)
        self.assertIn("self.get_parameter('yield_simple_radius_m')", source)

    def test_the_competition_node_asks_for_the_ninety_degree_cone(self):
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / 'scripts' / 'fixed_joint_pick_place.py').read_text()
        self.assertIn("declare_parameter('yield_simple_fov_deg', 100.0)", source)
        self.assertIn("self.get_parameter('yield_simple_fov_deg')", source)


class ClosePathRadiusTests(unittest.TestCase):
    """Check the path inside 1 m; keep the ordinary outer-cone stop rule."""

    def config(self, **kwargs):
        return obstacle_yield.YieldConfig(simple_radius_m=1.8, **kwargs)

    def test_a_box_in_the_annulus_stops_the_base(self):
        decision = obstacle_yield.decide(
            observation(1.4, 0.0, vx=-0.2), self.config(simple_radius_commit_m=1.0))
        self.assertEqual(decision.action, obstacle_yield.ACTION_STOP)
        self.assertTrue(decision.holds)
        self.assertIn('inside the 1.80 m circle', decision.detail)

    def test_a_box_inside_one_metre_requires_verified_clearance(self):
        cfg = self.config(simple_radius_commit_m=1.0)
        obs = observation(0.8, 0.0, vx=-0.2)
        self.assertTrue(obstacle_yield.decide(obs, cfg).holds)
        self.assertFalse(obstacle_yield.decide(obs, cfg, close_path_clear=True).holds)

    def test_exactly_one_metre_is_a_path_check_not_an_automatic_release(self):
        cfg = self.config(simple_radius_commit_m=1.0)
        self.assertTrue(obstacle_yield.decide(observation(1.0, 0.0), cfg).holds)
        self.assertFalse(obstacle_yield.decide(
            observation(1.0, 0.0), cfg, close_path_clear=True).holds)
        self.assertTrue(obstacle_yield.decide(
            observation(1.01, 0.0), cfg, close_path_clear=True).holds)

    def test_a_commit_radius_of_zero_keeps_the_circle_rule(self):
        decision = obstacle_yield.decide(
            observation(0.8, 0.0), self.config(simple_radius_commit_m=0.0))
        self.assertEqual(decision.action, obstacle_yield.ACTION_STOP)

    def test_close_oblique_or_rear_box_can_conflict_with_a_turn(self):
        cfg = self.config(simple_radius_commit_m=1.0)
        for xy in [(0.3, 0.85), (-0.6, 0.0)]:
            with self.subTest(xy=xy):
                self.assertTrue(obstacle_yield.decide(observation(*xy), cfg).holds)
                self.assertFalse(obstacle_yield.decide(
                    observation(*xy), cfg, close_path_clear=True).holds)

    def test_the_competition_node_asks_for_the_commit_radius(self):
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / 'scripts' / 'fixed_joint_pick_place.py').read_text()
        self.assertIn("declare_parameter('yield_simple_commit_m', 1.0)", source)
        self.assertIn("self.get_parameter('yield_simple_commit_m')", source)


if __name__ == '__main__':
    unittest.main()

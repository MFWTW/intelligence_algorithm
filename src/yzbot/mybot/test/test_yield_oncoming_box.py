"""A mover that is driving at the base is a stop, not a pass.

The predictor only trusts the box's velocity while the base is moving
(`min_speed_mps`), so a base that has already stopped reads an oncoming box as
"it will clear" and drives into it. These tests pin the rule that closes that
hole: inside `cap_range_m`, a box whose velocity has a closing component toward
the base holds it, whatever the predicted clearance says.

Operator report, 2026-11-26: "when the box is 1-1.8 m out and coming at us, I
want the base to stand still".
"""
import math
import pathlib
import sys
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import obstacle_yield


def observation(x, y, *, vx=0.0, vy=0.0, latched=False):
    return obstacle_yield.YieldObservation(
        box_x=x, box_y=y, box_vx=vx, box_vy=vy,
        box_world_speed_mps=math.hypot(vx, vy),
        scan_age_sec=0.0, latched=latched,
        lane_distance_m=0.0,
        vx=0.0, wz=0.0,
    )


class OncomingBoxTests(unittest.TestCase):
    def setUp(self):
        # The competition profile: no simple radius, the predictive layer decides.
        self.cfg = obstacle_yield.YieldConfig(simple_radius_m=0.0)
    def decide(self, obs, **kwargs):
        return obstacle_yield.decide(
            obs, self.cfg,
            wait_elapsed_sec=kwargs.pop('wait_elapsed_sec', 0.0))

    def test_closing_box_inside_the_band_holds(self):
        # 1.5 m out, 1.0 m to the side of the nose: the predictor calls it a
        # pass, the closing velocity says it is coming at us. Must hold.
        decision = self.decide(observation(1.4, 0.8, vx=-0.08, vy=0.0))
        self.assertTrue(decision.holds, decision.detail)
        self.assertEqual(decision.action, obstacle_yield.ACTION_STOP)
        self.assertIn('oncoming', decision.detail)

    def test_receding_box_inside_the_band_still_passes(self):
        decision = self.decide(observation(1.4, 0.8, vx=0.08, vy=0.0))
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)

    def test_closing_box_outside_the_band_is_not_held(self):
        decision = self.decide(observation(2.4, 1.0, vx=-0.08, vy=0.0))
        self.assertEqual(decision.action, obstacle_yield.ACTION_GO)

    def test_closing_box_holds_even_with_the_budget_spent(self):
        # The budget is what turns a hold into a lateral exit; it must not be
        # bypassed by this rule, or an oncoming box would park the base for
        # the rest of the leg.
        decision = self.decide(observation(1.4, 0.8, vx=-0.08),
                               wait_elapsed_sec=self.cfg.wait_budget_sec + 1.0)
        self.assertNotIn('oncoming', decision.detail or '')


if __name__ == '__main__':
    unittest.main()

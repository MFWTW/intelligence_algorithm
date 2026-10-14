#!/usr/bin/env python3
"""Pure predictive go/cap/hold/lateral-exit decisions for a tracked box.

A predicted crossing conflict is capped while distant, then held near the
base. After a finite wait, a lateral waypoint is a candidate only if both the
box envelope and measured room permit it; otherwise control returns to Nav2.
The executor separately rejects candidates with lidar returns along the
approach. None of these scan checks proves map/footprint clearance, and RPP's
speed limit constrains *linear* speed but not rotate-to-heading angular speed.
The feature is therefore OFF by default pending a controlled simulation test.

Movers are kinematic and never yield. A previous patrol trace had a mover
stationary for ~110 s while the base occupied its line. This policy never
waits indefinitely; its wall-clock wait budget is distinct from Nav2's
progress-checker time, which is configured in the active Nav2 profile.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

# Actions, in escalating order. A caller compares against these names, so keep
# them stable strings rather than an enum (the test fakes are plain namespaces).
ACTION_GO = 'go'
ACTION_CAP = 'cap'
ACTION_ARC = 'arc'
ACTION_STOP = 'stop'

# Why the action was chosen. Purely diagnostic: these strings land in the round
# log so a failed round can be attributed to a rule rather than to "it stopped".
REASON_CLEAR = 'clear'
REASON_PASSING = 'passing'
REASON_ALONGSIDE = 'alongside'
REASON_BEHIND = 'behind'
REASON_CAPPED = 'capped'
REASON_ARC = 'arc'
REASON_WAIT = 'wait'
REASON_WAIT_EXPIRED = 'wait-expired'
REASON_ARC_FAILED = 'arc-failed'


@dataclass(frozen=True)
class YieldConfig:
    """Tunables, all in SI units and all bounded by measurement elsewhere."""

    enabled: bool = True
    # The corridor the base must keep clear, as a half-width to each side of its
    # forward axis. Carried envelope is 0.32 m; 0.11 m is half the 0.22 m box,
    # so `half_width_m + obstacle_half_width_m` reproduces the 0.46 m clearance
    # the moving-obstacle node warns at, with a little margin on top.
    half_width_m: float = 0.24
    obstacle_half_width_m: float = 0.11
    # Chassis half-length from the lidar's axis to the front of the carried
    # envelope. The base is only *past* an obstacle once its nose is beyond it,
    # which is what makes the "直行" branch safe for an obstacle sitting in the
    # lane rather than a hypothetical crossing one.
    base_half_length_m: float = 0.25
    # The base's own half-width plus the carried envelope. A lateral offset that
    # does not clear this much past the obstacle is not a detour, it is a lane
    # change into it.
    base_half_width_m: float = 0.35
    # An obstacle this far behind the base cannot be hit by driving forward.
    behind_margin_m: float = 0.10
    # Never take an obstacle seriously before this range; the lidar sees 30 m
    # but a decision made at 4 m is noise. Measured closing geometry: the
    # closest a mover ever came to the base in a completed round was 0.58 m
    # while the base was doing 1.5 m/s.
    detection_range_m: float = 2.5
    # A tracked obstacle beyond this is somebody else's problem.
    track_range_m: float = 4.0
    # Predict at most this far ahead. The base covers 2.25 m at 1.5 m/s, but a
    # box that only clears the corridor in 6 s is still outside the base's
    # braking window: the decision can be revised every control tick.
    horizon_sec: float = 4.0
    # The base must be past the obstacle with this much time to spare - not
    # merely "past it at the instant the obstacle enters the lane". The value is
    # the measured control latency of this stack (10 Hz lidar -> 10 Hz costmap ->
    # 10 Hz controller -> 20 Hz smoother at RTF ~0.9), i.e. travel no prediction
    # can undo. Without it the rule called a 0.07 s clearance a pass; a 17%
    # error in the box's estimated closing rate (well inside tracker noise)
    # turns that same geometry into a collision. 0.0 restores the old rule and
    # is only for reproducing old logs.
    clear_margin_sec: float = 0.3
    # Prediction is only trusted while the base is actually driving. A base at a
    # standstill has no time-to-crossing, which would otherwise make every
    # obstacle look like it is about to arrive ("wait forever" bug).
    min_speed_mps: float = 0.05
    # Escalation bands, expressed as base-frame range to the obstacle.
    cap_range_m: float = 1.8
    stop_range_m: float = 1.2
    # Speed permitted while capping. Well under the measured 1.5 m/s profile and
    # above RPP's own 0.25 m/s cost-regulated floor, so the cap is what actually
    # sets the speed rather than being masked by it.
    cap_speed_mps: float = 0.8
    # Arc geometry: apex this far ahead of the base, offset this far to the side
    # that has room. Capped by max_arc_yaw_rad so the base is never asked to
    # veer more than a third of a radian off its heading.
    arc_apex_m: float = 0.8
    arc_offset_m: float = 0.55
    arc_offset_min_m: float = 0.25
    # Must stay above atan2(arc_offset_m, arc_apex_m) = atan2(0.55, 0.8) = 0.602
    # rad, or the configured offset can never be granted and the lateral exit
    # silently degrades to "no room" on every single call - which is exactly what
    # happened when this was 0.6 by accident of rounding.
    max_arc_yaw_rad: float = 0.65
    # How many times one leg may be rerouted around an obstacle before the
    # policy gives up and waits instead. A third arc means the obstacle is
    # following the base, not crossing it.
    max_arc_attempts: int = 2
    # Speed to assume when estimating how long the base needs to clear a point
    # while it is being held. Using the measured speed (0 when held) would make
    # every crossing look infinitely far away. Capped at the crawl ceiling
    # below, because the base resumes at a crawl after a yield, not at 1.5 m/s.
    yield_speed_mps: float = 0.6
    # After a stop, this is the speed the base is allowed to resume at. A held
    # base that restarts at 1.5 m/s would be back on top of the obstacle before
    # the next control tick; the cruise limit is restored by the caller once
    # the leg's yield state is clear.
    resume_speed_mps: float = 0.6
    # Reach distance for the arc waypoint, matching the executor's other
    # clearance waypoints.
    arc_goal_radius_m: float = 0.35
    # Cost model for "is arcing worth it". The arc costs this many simulated
    # seconds of extra travel (0 => arcing is always preferred when possible).
    cost_arc_sec: float = 0.8
    # Waiting is bounded three ways: the caller's own patience
    # (wait_budget_sec), the base's distance to the goal, and this hard ceiling.
    wait_goal_margin_m: float = 0.6
    wait_budget_sec: float = 20.0
    # An obstacle that is not moving *in the world* is static geometry wearing
    # a box's silhouette (a pillar, a dropped crate, a wall stub). Nav2's own
    # inflation and collision detection are the right tools for those, and
    # waiting for one to walk away is what killed the round before
    # (run_logs/yield3_131041: mover_2 frozen 110 s). The tracker estimates
    # world-frame speed by removing the base's own velocity, and only obstacles
    # above this threshold are ever waited for or arced around.
    min_mover_speed_mps: float = 0.04
    # ...and this is the other end of the same window. The 0.08 m/s patrols
    # are below it, while the measured 0.86-1.52 m/s false tracks from static
    # clusters remain excluded. Costmap/lidar checks still guard collisions.
    max_mover_speed_mps: float = 0.6
    # Association radius is *motion* plus this floor, not a fixed bubble: two
    # sweeps apart a real box moves at most max_mover_speed * dt, and anything
    # further away than that is a different object.
    association_floor_m: float = 0.05
    # Ignore tracks we have not heard from for this long.
    track_timeout_sec: float = 1.5
    # An empty corridor for this long releases the stop, even if the track that
    # caused it is still nominally alive somewhere in the scan.
    clear_hold_sec: float = 0.5
    # Parking inside a patrol lane is a trap: the box comes back every few
    # seconds, each individual hold is short, and the base can never finish
    # crossing - measured 2026-10-13, 0.30 m from mover_3's segment, 18 s of
    # zero motion across 30 hold/resume cycles. Any hold this close to a lane
    # centre line is refused: the base caps and keeps crossing instead.
    lane_hold_clearance_m: float = 0.5
    # ...and an episode that is already holding inside a lane leaves sideways
    # after this long rather than waiting out the full budget in the box's way.
    lane_escape_delay_sec: float = 1.0
    # How close to a patrol endpoint a box must be before its "I am leaving"
    # velocity stops being believable (it is about to turn around). Kept tight
    # AND combined with "is it still heading for that endpoint": the patrol
    # segments here are only 1.4-1.7 m long, so a generous margin covers most of
    # the line and turns every leaving box into a phantom threat - measured
    # 2026-10-13, a 0.6 m margin tripled the holds (28 in one round, 0 delivered).
    lane_end_margin_m: float = 0.3
    # One *episode* is one obstacle that stops the base from getting anywhere.
    # It ends when the base has actually travelled this far since the hold that
    # started it - not when a single hold releases. A box patrolling through the
    # spot the base parked in leaves and returns every few seconds, so a
    # per-hold budget restarts forever and the base is trapped (measured
    # 2026-10-13: 30 hold/resume cycles, 18 s frozen, no escape ever issued).
    episode_reset_progress_m: float = 2.0
    # ...and an episode also ends after this long of driving freely. Without it
    # a spent budget is inherited by every LATER encounter: measured 2026-10-13
    # (run_logs/visalign_probe_round) a hold escalated to an arc after 1.0 s
    # because the episode had never been reset, costing a 0.70 m jam escape, a
    # retry and an arc drive - five of those in one round, ~30 s each. A box
    # that keeps shuttling through the base's corridor never grants this much
    # freedom, so the trap the episode timer exists for is still bounded.
    episode_clear_sec: float = 3.0
    # World speed is estimated over this window, not frame to frame. A 0.22 m
    # box's centroid moves a few centimetres between 10 Hz sweeps and lidar
    # sampling noise is the same order, so an instantaneous difference doubles
    # the closing rate (measured 2026-10-13: 0.21-0.58 m/s for boxes that move
    # 0.16-0.30) and the policy stops for crossings that would have cleared.
    speed_window_sec: float = 0.5
    # Above this yaw rate the policy's lateral picture is invalid: a stationary
    # object at range r appears to move relative to the base at wz*r (1.5 rad/s
    # at 3.5 m is 5 m/s), so a turning base sees the whole room as incoming
    # movers and holds itself in place for them. Measured 2026-10-13: held at
    # ~1.5 rad/s the base circled beside a box while every static cluster read
    # as a fast mover. A turn is no reason to take on a new conflict; an
    # existing hold is left to its own budget.
    rotation_guard_rad_s: float = 0.4

    def with_overrides(self, **kwargs) -> 'YieldConfig':
        """Copy with only recognised field overrides (ignores None)."""
        known = {k: v for k, v in kwargs.items()
                 if v is not None and k in self.__dataclass_fields__}
        return replace(self, **known) if known else self


@dataclass(frozen=True)
class YieldObservation:
    """Everything the policy is allowed to look at, in the base frame.

    +X is forward, +Y is left. ``box_vx``/``box_vy`` are the obstacle's velocity
    *relative to the base* (the base's own motion already removed), so the
    extrapolation below is valid even though the base is driving.
    """

    box_x: float
    box_y: float
    box_vx: float
    box_vy: float
    # Base motion, for the time-to-crossing estimate and the standstill guard.
    vx: float = 0.0
    wz: float = 0.0
    # Signed room beside the *current* path, obstacle side only: how far the
    # base could move laterally before its carried envelope touches static
    # geometry. Supplied by the caller (it owns the costmap/scan); None means
    # "unknown", which is treated as no room.
    left_room_m: float | None = None
    right_room_m: float | None = None
    distance_to_goal_m: float | None = None
    # Age of the underlying scan. A stale scan must not steer the base.
    scan_age_sec: float = 0.0
    # Whether a moving box is expected in this environment at all. With no
    # movers configured the yield layer stays completely inert, so a static
    # round cannot regress because of it.
    movers_present: bool = True
    # The obstacle's speed in the world, with the base's own velocity removed.
    # This is what separates a mover from a wall; None means "unknown", which
    # is treated as static (fail safe: never wait for something that may be a
    # wall).
    box_world_speed_mps: float | None = None
    # Distance from the tracked box to the nearer END of its patrol segment, or
    # None when the lanes are unknown, plus whether it is still heading for that
    # end. A box about to arrive at an endpoint reverses on the next tick; one
    # that has just left it is telling the truth.
    lane_end_distance_m: float | None = None
    lane_end_closing: bool = False
    # Distance from the base to the nearest patrol centre line, or None when
    # the caller does not know the lanes (no obstacles configured). Stopping
    # inside one strands the base in a box's path, so the policy refuses it.
    lane_distance_m: float | None = None
    # True while the caller is already holding the base for this obstacle. The
    # policy must not re-decide from scratch in that state: a held base has
    # vx = 0 by construction, which would otherwise make every crossing look
    # infinitely far away and release the hold one tick after taking it.
    latched: bool = False

    def range_m(self) -> float:
        return math.hypot(self.box_x, self.box_y)


@dataclass(frozen=True)
class YieldDecision:
    action: str
    reason: str
    speed_mps: float = 0.0
    arc_side: int = 0            # +1 left, -1 right, 0 none
    arc_x: float | None = None   # base-frame apex the executor will convert
    arc_y: float | None = None
    wait_elapsed_sec: float = 0.0
    detail: str = ''

    @property
    def holds(self) -> bool:
        return self.action == ACTION_STOP


def parse_patrol_segments(text) -> list[tuple[float, float, float, float]]:
    """Patrol centre lines as ``[(x0, y0, x1, y1), ...]``.

    Accepts the same spec the moving-obstacle node takes
    (``name:x0,y0,x1,y1,speed[,k];...``) and skips anything malformed instead of
    raising: the executor must still drive if the batch is garbled, and an empty
    list simply means "lanes unknown", which leaves the old behaviour intact.
    """
    segments = []
    for chunk in str(text or '').split(';'):
        chunk = chunk.strip()
        if not chunk or ':' not in chunk:
            continue
        _, values = chunk.split(':', 1)
        parts = [v.strip() for v in values.split(',')]
        if parts and parts[-1].lower() in ('k', 'kinematic', 'true'):
            parts = parts[:-1]
        try:
            numbers = [float(v) for v in parts[:4]]
        except ValueError:
            continue
        if len(numbers) == 4:
            segments.append(tuple(numbers))
    return segments


def distance_to_lane_end(x: float, y: float, segments) -> float | None:
    """Distance from (x, y) to the nearest patrol endpoint, None if unknown.

    A box within ~0.6 m of an endpoint is about to reverse; its measured
    velocity still points away until it does.
    """
    if not segments:
        return None
    best = None
    for x0, y0, x1, y1 in segments:
        for ex, ey in ((x0, y0), (x1, y1)):
            distance = math.hypot(x - ex, y - ey)
            best = distance if best is None else min(best, distance)
    return best


def closing_on_lane_end(x: float, y: float, vx: float, vy: float,
                        segments) -> bool:
    """Is this box still travelling *towards* the end it is nearest to?

    Only then is its measured "I am leaving" velocity about to be contradicted
    by the reversal. None of the lanes known => False (nothing to distrust).
    """
    if not segments:
        return False
    best = None
    for x0, y0, x1, y1 in segments:
        for ex, ey in ((x0, y0), (x1, y1)):
            distance = math.hypot(x - ex, y - ey)
            if best is None or distance < best[0]:
                best = (distance, ex, ey)
    _, ex, ey = best
    towards = (ex - x) * vx + (ey - y) * vy
    return towards > 0.0 and math.hypot(vx, vy) > 1e-3


def distance_to_lane(x: float, y: float,
                     segments) -> float | None:
    """Shortest distance from (x, y) to any patrol centre line.

    ``None`` when there are no lanes to check (no movers in the scene).
    """
    if not segments:
        return None
    best = None
    for x0, y0, x1, y1 in segments:
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        if length2 <= 1e-9:
            distance = math.hypot(x - x0, y - y0)
        else:
            fraction = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length2))
            distance = math.hypot(x - (x0 + fraction * dx), y - (y0 + fraction * dy))
        best = distance if best is None else min(best, distance)
    return best


def crossing_time(y: float, vy: float, half_width: float,
                  horizon_sec: float) -> float | None:
    """When (seconds from now) the obstacle's y first enters [-half, +half].

    ``None`` means "not within the horizon" - including the harmless case of an
    obstacle already on its way out. A closing obstacle that is already inside
    the band returns 0.0.

    This is the *near* edge the obstacle reaches first, not the far one: for a
    box at y=+0.80 m closing at 0.30 m/s against a 0.35 m half-width, entry is
    at 1.50 s. The far-edge value (3.83 s) would postpone the entry test past
    the moment the box is already in the lane and quietly turn a coin-flip pass
    into a "safe" one.
    """
    if abs(y) <= half_width:
        return 0.0
    if abs(vy) < 1e-6:
        return None
    t = (math.copysign(half_width, y) - y) / vy
    if t < 0.0 or t > horizon_sec:
        return None
    return t


def lateral_side(y: float) -> int:
    """+1 when the obstacle is to the left of the base, -1 when to the right."""
    return 1 if y >= 0.0 else -1


@dataclass(frozen=True)
class _Conflict:
    """Whether the base and the obstacle want the same 0.7 m of floor."""

    conflict: bool
    clears_at: float          # when the base's nose is past the obstacle
    why: str


def detect_conflict(obs: 'YieldObservation', cfg: 'YieldConfig') -> _Conflict:
    """Decide whether this obstacle is actually in the base's way.

    The rule is one sentence: **project the obstacle to the moment the base's
    nose reaches it, and ask whether it is inside the base's corridor then.**

    Two simpler-sounding rules were tried first and both were wrong on cases
    that matter (they are kept here as the reason this is written the long way):

    *   "the base passes if it arrives before the obstacle enters the band"
        answers *go* for a box that is standing in the lane right now, because
        such a box is already inside the band and therefore has no future entry
        at all.
    *   "the base passes if it clears before the obstacle's entry time" answers
        *stop* for a box that is on its way out: it is still inside the band
        (enters_at = 0) while the base would in fact drive past behind it.

    The projection handles both, because it looks at where the obstacle will be
    *at the base's own crossing time*, not at when the obstacle crosses a line.

    The third refinement is the latency margin: for a box that is still outside
    the corridor, "the base's nose is past before the box arrives" is not enough
    - it must be past with ``clear_margin_sec`` to spare (the measured sensing
    and control latency of this stack). A box that is already inside the
    corridor keeps the projection rule, because an entry-time test would stop
    for a box that is on its way out.
    """
    # When does the base's nose get past this obstacle, with a margin for the
    # obstacle's own half-width? A base that is being held resumes at
    # ``yield_speed_mps``, not at 0, which is why the held case needs its own
    # speed rather than the measured one.
    needed = (max(0.0, obs.box_x) + cfg.base_half_length_m
              + cfg.obstacle_half_width_m)
    if obs.latched:
        speed = cfg.yield_speed_mps
    else:
        speed = max(obs.vx, cfg.min_speed_mps)
    clears_at = needed / speed

    half = cfg.half_width_m + cfg.obstacle_half_width_m

    # A box that is still *outside* the corridor is judged by when it will
    # enter it, and the base must be past with the latency margin to spare.
    # This is the "早决策" half of the rule: the answer is decided while the box
    # is still far, so the base either keeps cruise or brakes early - it never
    # leaves the decision to the last metre, where a 0.07 s clearance looks
    # like a pass and a 17% error in the box's closing rate makes it a crash.
    if abs(obs.box_y) > half:
        enters_at = crossing_time(obs.box_y, obs.box_vy, half,
                                  cfg.horizon_sec + cfg.clear_margin_sec)
        if enters_at is None:
            # No entry within the horizon: either it is leaving, or it is
            # parallel and outside the corridor, or it is too far away to
            # matter before the base is past.
            return _Conflict(
                False, clears_at,
                f'base clears in {clears_at:.2f} s; box never enters '
                f'+-{half:.2f} m within {cfg.horizon_sec:.1f} s',
            )
        margin = enters_at - clears_at
        if margin >= cfg.clear_margin_sec:
            return _Conflict(
                False, clears_at,
                f'base clears in {clears_at:.2f} s, box enters in '
                f'{enters_at:.2f} s (margin {margin:.2f} s >= '
                f'{cfg.clear_margin_sec:.2f} s)',
            )
        return _Conflict(
            True, clears_at,
            f'base clears in {clears_at:.2f} s but the box enters the corridor '
            f'in {enters_at:.2f} s (margin {margin:.2f} s < required '
            f'{cfg.clear_margin_sec:.2f} s)',
        )

    # Already inside the corridor: keep the projection rule. An entry-time rule
    # would answer "stop" for a box that is on its way out (enters_at = 0),
    # which is the deadlock the docstring above describes.
    projected_y = obs.box_y + obs.box_vy * clears_at
    if abs(projected_y) > half:
        return _Conflict(
            False, clears_at,
            f'base clears in {clears_at:.2f} s, obstacle will be at '
            f'y={projected_y:+.2f} m (outside +-{half:.2f} m)',
        )

    # Inside the corridor at the crossing moment. How long until it leaves?
    if abs(obs.box_vy) > 1e-6:
        # ...either it is on its way out (same sign as its position)...
        leaving_in = (-math.copysign(half, projected_y) - projected_y) / obs.box_vy
        t_clear = clears_at + max(0.0, leaving_in)
    else:
        # ...or it is not crossing at all. Only a mover gets this far, so this
        # is a mover running parallel to the base *inside* the corridor, which
        # is the one case where the honest answer is "not soon".
        t_clear = None
    return _Conflict(
        True, clears_at,
        f'base clears in {clears_at:.2f} s, obstacle will be at '
        f'y={projected_y:+.2f} m (inside +-{half:.2f} m), clear in '
        + ('never' if t_clear is None else f'{t_clear:.2f} s'),
    )


def decide(
    obs: YieldObservation | None,
    cfg: YieldConfig,
    *,
    wait_elapsed_sec: float = 0.0,
    arc_attempts: int = 0,
) -> YieldDecision:
    """Pick the yield action for the current control tick.

    Pure: no clock, no ROS, no side effects. ``wait_elapsed_sec`` is supplied
    by the caller so this stays testable.
    """
    if not cfg.enabled or obs is None or not obs.movers_present:
        return YieldDecision(ACTION_GO, REASON_CLEAR)
    if obs.scan_age_sec > cfg.track_timeout_sec:
        # Never steer on a stale scan: a dropped frame would otherwise freeze
        # the base on evidence that is no longer true.
        return YieldDecision(ACTION_GO, REASON_CLEAR,
                             detail=f'scan age {obs.scan_age_sec:.2f} s')

    rng = obs.range_m()
    if rng > cfg.track_range_m:
        return YieldDecision(ACTION_GO, REASON_CLEAR)

    # A base that is turning reports the whole room as incoming movers, because
    # the relative velocity it measures carries its own wz*r term. Do not open a
    # NEW conflict on that evidence: measure 2026-10-13, a held base turning at
    # ~1.5 rad/s kept finding fresh "movers" among static clusters and stayed
    # parked beside the box. A hold that is already protecting the base is left
    # to its own budget (releasing it here would drive forward on the very tick
    # the base is least able to judge what is in the corridor).
    rotating = abs(obs.wz) > cfg.rotation_guard_rad_s
    if rotating and not obs.latched:
        return YieldDecision(
            ACTION_GO, REASON_CLEAR,
            detail=f'base rotating at {obs.wz:+.2f} rad/s; lateral picture '
                   'is the base\'s own motion',
        )

    side = lateral_side(obs.box_y)

    # 0. The single most important guard: only a genuine mover may be waited
    #    for or arced around. A pillar, a dropped crate or a wall stub has a
    #    box's silhouette but no motion of its own, and Nav2's inflation layer
    #    and collision check are the correct (and already tuned) tools for it.
    #    Unknown world speed fails safe to "static" for the same reason: never
    #    wait for something that may be a wall. run_logs/yield3_131041 is the
    #    round that paid for this rule: a base that waits for the wrong thing
    #    deadlocks the round.
    if (
        obs.box_world_speed_mps is None
        or obs.box_world_speed_mps < cfg.min_mover_speed_mps
    ):
        return YieldDecision(
            ACTION_GO, REASON_CLEAR,
            detail='static obstacle: not a mover'
                   if obs.box_world_speed_mps is not None
                   else 'static obstacle: world speed unknown',
        )

    # 1. Already behind the base: driving forward cannot hit it.
    if obs.box_x < -cfg.behind_margin_m:
        return YieldDecision(ACTION_GO, REASON_BEHIND)

    # Is the base itself standing in a patrol box's lane? Only a caller that
    # knows the lanes can say (lane_distance_m is None without movers).
    in_lane = (obs.lane_distance_m is not None
               and obs.lane_distance_m < cfg.lane_hold_clearance_m)

    # A mover at the end of its segment is about to reverse, and its measured
    # velocity still points away until it does. Measured 2026-10-13: mover_2's
    # direction flipped from -1 to +1 between two samples while the policy read
    # "passing", the base rolled in at 1.5 m/s and the box came back at it
    # (fling). The distrust is limited to the case that actually happened - the
    # BASE inside the lane - because applying it everywhere tripled the holds
    # (48 episodes in one round, 598 s, vs 6 and 356 s).
    if (in_lane and obs.lane_end_distance_m is not None
            and obs.lane_end_distance_m < cfg.lane_end_margin_m
            and obs.lane_end_closing):
        obs = replace(obs, box_vx=0.0,
                      box_vy=0.0 if obs.box_vy * obs.box_y > 0 else obs.box_vy)

    # 2. The decision itself. This is the "直行 / 太近就等" split, and all of it
    #    lives in detect_conflict so it can be tested on its own.
    conflict = detect_conflict(obs, cfg)

    # 3. While the base is ALREADY being held for this obstacle, re-decide only
    #    on "is the corridor clear yet?". Re-running the full escalation here
    #    would flip the hold off on the next tick (a held base has vx = 0), and
    #    the velocity smoother turns that stutter into a lurch.
    if obs.latched:
        if not conflict.conflict:
            return YieldDecision(ACTION_GO, REASON_CLEAR,
                                 detail='hold released: ' + conflict.why)
        if in_lane and wait_elapsed_sec >= cfg.lane_escape_delay_sec:
            # Already holding inside a box's lane: waiting there is the trap.
            # Leave sideways now instead of spending the whole budget in its
            # way (measured: 18 s frozen in 30 alternating hold/resume cycles).
            return lateral_exit(obs, cfg, arc_attempts, wait_elapsed_sec,
                                conflict)
        if wait_elapsed_sec >= cfg.wait_budget_sec:
            # Patience is spent and the obstacle still owns the lane. Do NOT
            # simply keep waiting: the controller's own progress checker fails
            # the goal after `movement_time_allowance` and the recovery tree
            # then backs the base up and spins it - towards whatever it was
            # waiting for. Leave along a lateral waypoint instead.
            return lateral_exit(obs, cfg, arc_attempts, wait_elapsed_sec,
                                conflict)
        return YieldDecision(
            ACTION_STOP, REASON_WAIT, wait_elapsed_sec=wait_elapsed_sec,
            detail=f'holding: {conflict.why}',
        )

    if not conflict.conflict:
        # The base gets past it - the plain "直行" case. No action at all, not
        # even a cap: the obstacle clears itself, and every capped second costs
        # ~5.5 s of wall clock at this host's measured RTF of 0.18.
        return YieldDecision(ACTION_GO, REASON_PASSING, detail=conflict.why)

    # 4. A real conflict. Slow down while it is still some way off: this is the
    #    "加一点弧度" half of the request, and on this platform it is the only
    #    half that can be honoured. Steering round a box that is *crossing* the
    #    corridor needs ~0.7 m of offset (the box's projected position plus the
    #    base's own half width) inside the ~1 s the base has while it passes -
    #    at which point the base is already crawling, the box clears first, and
    #    there is no conflict left to solve. Slow down, let the costmap's own
    #    inflation layer bias the path away from it, and escalate to a stop.
    if rng > cfg.cap_range_m:
        return YieldDecision(
            ACTION_CAP, REASON_CAPPED,
            speed_mps=cfg.cap_speed_mps,
            detail=f'box {rng:.2f} m away; {conflict.why}',
        )

    # 4b. Do NOT cross just because the base is inside a lane. That rule was
    #     tried on 2026-10-13 and it is what caused the fling: the decision log
    #     reads "in a patrol lane (0.38 m); crossing rather than parking" at
    #     1791902641, mover_2's distance then collapsed 0.69 -> 0.24 m, and the
    #     base was thrown at 5.7 m/s (wz -72 rad/s). Standing still inside a
    #     lane is survivable - the box shuttles past - driving into it is not.
    #     The trap that motivated it is bounded instead by the episode budget
    #     and by the lane escape below, which only fire after a real wait.

    # 5. Stop and let it pass - bounded by the caller's wait budget, which is
    #    what turns "wait for it" into "wait for it, then do something else".
    if obs.distance_to_goal_m is not None and (
        obs.distance_to_goal_m <= cfg.wait_goal_margin_m
    ):
        # Stopping this close to the goal cannot be justified: within the goal
        # margin there is no room to route around anything, and Nav2's own goal
        # checker is about to finish the leg anyway.
        return YieldDecision(ACTION_GO, REASON_CLEAR,
                             detail='inside the goal margin')

    # The budget is spent on the EPISODE, so it binds even when this tick is a
    # *fresh* stop: the latch releases on every lull (a box stepping out of the
    # corridor for half a second), and without this check the policy simply
    # stopped again and again for the same box. Measured 2026-10-13: decisions
    # reading "wait=34.7 s ... 40.4 s" were still ACTION_STOP because the
    # latched branch owns the budget, 29 holds in one round, and the delivery
    # leg finally failed with the cube still attached.
    if wait_elapsed_sec >= cfg.wait_budget_sec:
        return lateral_exit(obs, cfg, arc_attempts, wait_elapsed_sec, conflict)

    room = obs.left_room_m if side > 0 else obs.right_room_m
    return YieldDecision(
        ACTION_STOP, REASON_WAIT, wait_elapsed_sec=wait_elapsed_sec,
        detail=f'range {rng:.2f} m, room '
               f'{room if room is None else round(room, 2)} m; {conflict.why}',
    )


def lateral_exit(obs, cfg, arc_attempts: int, wait_elapsed_sec: float,
             conflict) -> 'YieldDecision':
    """Lateral escape once waiting has failed.

    This is the only place an arc is issued, and the reason is geometry rather
    than taste. A lateral offset has to clear the obstacle's half-width plus the
    base's own half-width on the side the obstacle is leaving:

        offset >= |projected_y| + 0.11 + 0.35  ~= 0.7 m

    A box *crossing* the corridor is only 0.35 m off-axis when the base arrives,
    so a compliant arc needs ~0.7 m of free floor beside the path inside the
    roughly one second the base has while it passes - and if the base slows
    enough to make that arc possible, the box clears first and the conflict
    disappears. Exhaustive search over base speeds 0.30-1.60 m/s, box offsets
    -0.34..+0.34 m and three closing rates found ZERO mid-motion arcs that both
    resolved a real conflict and fitted the required offset.

    What is left is the case that matters: the base has already stood still for
    the whole wait budget and the obstacle is *still* in the lane (movers here
    are kinematic and never yield, so this really happens). Then a lateral exit
    is worth taking, and there is usually room for it - measured over a full
    round's 156.9 m of route, 99.5% of sampled poses have >= 0.45 m of clearance
    on at least one side, and the single 0.10 m stretch where neither side does
    is documented in run_logs/yield_arc_clearance.md.
    """
    if arc_attempts >= cfg.max_arc_attempts:
        return YieldDecision(
            ACTION_GO, REASON_WAIT_EXPIRED,
            wait_elapsed_sec=wait_elapsed_sec,
            detail=f'waited {wait_elapsed_sec:.1f} s and already used '
                   f'{arc_attempts} lateral exits; handing the leg back to Nav2',
        )
    projected_y = obs.box_y + obs.box_vy * max(conflict.clears_at, 0.0)
    # The old rule unconditionally followed the box to the side it was leaving.
    # Measured at (-4.767,-1.526), the left clearance is only 0.008 m while the
    # right clearance is 2.522 m: that rule would run into a wall. Evaluate BOTH
    # sides against the box's current and projected envelope and choose the side
    # with the most spare room. Unknown space is never treated as free.
    candidates = []
    for side, room in ((1, obs.left_room_m), (-1, obs.right_room_m)):
        if room is None or not math.isfinite(room):
            continue
        # Centre-to-centre separation must include the carried footprint and
        # half the box at both ends of the predicted crossing. If the box is on
        # this side, offset needs to move past it, not onto it.
        required = max(side * obs.box_y, side * projected_y, 0.0) + (
            cfg.base_half_width_m + cfg.obstacle_half_width_m
        )
        offset = min(cfg.arc_offset_m, room - cfg.obstacle_half_width_m)
        if (offset >= max(cfg.arc_offset_min_m, required)
                and math.atan2(offset, cfg.arc_apex_m) <= cfg.max_arc_yaw_rad):
            candidates.append((room - offset, side, room, offset))
    if candidates:
        _, arc_side, room, offset = max(candidates)
        return YieldDecision(
            ACTION_ARC, REASON_WAIT_EXPIRED, arc_side=arc_side,
            arc_x=cfg.arc_apex_m, arc_y=offset * arc_side,
            wait_elapsed_sec=wait_elapsed_sec,
            detail=f'waited {wait_elapsed_sec:.1f} s; candidate exit '
                   f'{offset:.2f} m {"left" if arc_side > 0 else "right"} '
                   f'with {room:.2f} m measured room',
        )
    return YieldDecision(
        ACTION_GO, REASON_WAIT_EXPIRED, wait_elapsed_sec=wait_elapsed_sec,
        detail=f'waited {wait_elapsed_sec:.1f} s; no side clears both the box '
               'and measured static room; returning control to Nav2',
    )


def arc_scan_clear(points, arc_x: float, arc_y: float,
                   radius_m: float = 0.40) -> bool:
    """Reject a lateral waypoint if a scan return intersects its approach.

    Points are base-frame (x, y) static returns with the tracked mover removed.
    This is an extra *rejection* gate, not a proof of free space: lidar cannot
    see behind an occluding box or replace Nav2's costmap/footprint checks.
    """
    length2 = arc_x * arc_x + arc_y * arc_y
    if length2 <= 0.01 or not points:
        return False
    for px, py in points:
        fraction = max(0.0, min(1.0, (px * arc_x + py * arc_y) / length2))
        # Ignore self-returns very close to the lidar origin.
        if fraction * math.sqrt(length2) < 0.20:
            continue
        if math.hypot(px - fraction * arc_x,
                      py - fraction * arc_y) < radius_m:
            return False
    return True


class YieldTracker:
    """Track scan clusters in the base frame and expose the nearest conflict.

    Deliberately dependency-free and free of ROS types: the executor hands it
    ``(x, y)`` base-frame points from its latest sweep and the current yaw rate,
    so the same object works in a unit test with a list of tuples.

    Velocity is estimated per track from the motion of its centroid after
    removing the contribution of the base's own rotation (``wz x r`` in the
    rotating base frame). That is the quantity the policy needs: motion
    *relative to the base*.
    """

    def __init__(self, cfg: YieldConfig):
        self.cfg = cfg
        self._tracks: list[dict] = []

    def _associate(self, track: dict, x: float, y: float, dt: float) -> bool:
        """Is this return the same obstacle the track was following?

        Association happens in the odom frame (see ``update``) and its radius is
        derived from *motion*, not from the base's travel: over ``dt`` a real box
        can only have moved ``max_mover_speed_mps * dt``, so anything further
        away is a different object. The old fixed 0.30 m bubble let one track
        chain from cluster to cluster across a cluttered sweep, which is how a
        stationary chair leg became a 1.52 m/s "mover" (2026-10-13).
        """
        allowed = min(0.30, self.cfg.association_floor_m
                      + self.cfg.max_mover_speed_mps * max(0.0, dt))
        return math.hypot(x - track['x'], y - track['y']) <= allowed

    def update(self, points, now: float, base_state) -> None:
        """Fold one sweep of base-frame points into the tracks.

        ``points`` is any iterable of ``(x, y)`` in the base frame; a box is
        usually one or two returns, not a dense cluster. ``base_state`` is
        ``(x, y, yaw, vx, vy, wz)`` in the odom frame.

        Tracks are kept in the *odom* frame, not the base frame, and that is
        the whole reason this class takes the base pose at all. The base covers
        up to 0.15 m between two 10 Hz sweeps at 1.5 m/s, which is the same
        order as the association radius: matching successive points in the
        moving frame would shred one box into a trail of one-hit tracks. In an
        inertial frame the box stays put and only its own motion moves it.
        ``vx``/``vy``/``wz`` are then removed when reporting, because the policy
        wants motion relative to the base.
        """
        bx, by, yaw, vx, vy, wz = base_state
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        for track in self._tracks:
            track['seen'] = False
        for px, py in points:
            rng = math.hypot(px, py)
            if not (0.12 <= rng <= self.cfg.track_range_m):
                continue
            # Base frame -> odom frame.
            x = bx + cos_yaw * px - sin_yaw * py
            y = by + sin_yaw * px + cos_yaw * py
            for track in self._tracks:
                if track['seen'] or not self._associate(
                        track, x, y, now - track['t']):
                    continue
                dt = now - track['t']
                if dt > 1e-3:
                    track['vx'] = (x - track['x']) / dt   # odom frame
                    track['vy'] = (y - track['y']) / dt
                track['x'], track['y'], track['t'] = x, y, now
                track['hist'].append((now, x, y))
                while (len(track['hist']) > 2
                       and now - track['hist'][0][0] > self.cfg.speed_window_sec):
                    track['hist'].popleft()
                # Average over the window instead of the last frame: centroid
                # noise is the same order as one frame's motion for a 0.22 m
                # box, and the noisy version doubled every closing rate.
                window = track['hist']
                if len(window) >= 2:
                    (t0, x0, y0) = window[0]
                    span = now - t0
                    if span > 1e-3:
                        track['vx'] = (x - x0) / span
                        track['vy'] = (y - y0) / span
                track['seen'] = True
                break
            else:
                self._tracks.append({
                    'x': x, 'y': y, 'vx': 0.0, 'vy': 0.0,
                    't': now, 'seen': True, 'born': now,
                    'hist': deque([(now, x, y)], maxlen=64),
                })
        self._tracks = [
            t for t in self._tracks
            if t['seen'] or (now - t['t']) <= self.cfg.track_timeout_sec
        ]

    def nearest(self, now: float, base_state,
                min_world_speed: float = 0.0) -> dict | None:
        """Closest live track, reported in the base frame at call time.

        When requested, exclude stationary returns in the world frame *before*
        choosing a track: otherwise the nearest wall fragment masks a moving
        box further along the path. The same request also excludes implausibly
        fast tracks, which are association noise rather than patrol boxes.
        """
        bx, by = base_state[:2]
        live = [t for t in self._tracks
                if (now - t['t']) <= self.cfg.track_timeout_sec
                and math.hypot(t['vx'], t['vy']) >= min_world_speed
                and (min_world_speed <= 0.0
                     or math.hypot(t['vx'], t['vy'])
                     <= self.cfg.max_mover_speed_mps)]
        if not live:
            return None
        track = min(live, key=lambda t: math.hypot(t['x'] - bx, t['y'] - by))
        bx, by, yaw, vx, vy, wz = base_state
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        track = dict(track)
        dx, dy = track['x'] - bx, track['y'] - by
        # Odom frame -> base frame at the moment of the call.
        track['base_x'] = cos_yaw * dx + sin_yaw * dy
        track['base_y'] = -sin_yaw * dx + cos_yaw * dy
        # Relative velocity in the base frame: the frame rotates at wz, so a
        # stationary point still appears to move at -wz x r, and the base's own
        # translation adds equally.
        track['rel_vx'] = (cos_yaw * track['vx'] + sin_yaw * track['vy']) - vx + wz * track['base_y']
        track['rel_vy'] = (-sin_yaw * track['vx'] + cos_yaw * track['vy']) - vy - wz * track['base_x']
        track['world_speed'] = math.hypot(track['vx'], track['vy'])
        track['age'] = now - track['t']
        track['life'] = now - track['born']
        return track


def cluster_points(points, gap: float = 0.25):
    """Group base-frame points into connected components by a range gap.

    Points are ``(angle, range)`` pairs from a scan. Adjacent returns whose 3D
    separation exceeds ``gap`` start a new cluster, which is how a box is
    separated from the wall behind it without any ROS machinery.
    """
    ordered = sorted(points, key=lambda p: p[0])
    clusters: list[list[tuple[float, float]]] = []
    for angle, rng in ordered:
        if clusters:
            last_angle, last_rng = clusters[-1][-1]
            span = math.hypot(
                rng * math.cos(angle) - last_rng * math.cos(last_angle),
                rng * math.sin(angle) - last_rng * math.sin(last_angle),
            )
            if span <= gap and (angle - last_angle) <= math.radians(45.0):
                clusters[-1].append((angle, rng))
                continue
        clusters.append([(angle, rng)])
    return clusters


def cluster_extent(cluster) -> float:
    """Largest pairwise distance inside a cluster, approximated from its ends."""
    if len(cluster) < 2:
        return 0.0
    (a0, r0), (a1, r1) = cluster[0], cluster[-1]
    return math.hypot(
        r1 * math.cos(a1) - r0 * math.cos(a0),
        r1 * math.sin(a1) - r0 * math.sin(a0),
    )


def box_like_clusters(clusters, max_extent_m: float = 0.5,
                      max_points: int = 60, min_points: int = 2):
    """Keep only clusters small enough to be a 0.22 m box.

    Walls and door frames produce long chains of returns; a mover produces a
    compact blob. Without this the policy would try to overtake the building.
    A single return is speckle, not a box: at the tracking range a 0.22 m box
    still subtends several of the lidar's 1 deg beams, so requiring two returns
    costs nothing real and removes the noise tracks that chained into 1.5 m/s
    phantoms on 2026-10-13.
    """
    return [
        c for c in clusters
        if min_points <= len(c) <= max_points
        and cluster_extent(c) <= max_extent_m
    ]

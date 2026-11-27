#!/usr/bin/env python3
"""Offline replay of an "escape segment" out of the global inscribed band.

Problem this answers
--------------------
A Nav2 base can stop *inside* a cell that the global costmap marks INSCRIBED
(253 / published 99).  NavFn then refuses at the entry of the search
("failed to create plan") because the start cell itself is lethal for the
planner, even though the body is physically clear: the inscribed band is
`robot_radius` wide, and the base's own half width is smaller than that.

`_back_off()` / `_band_escape_axis()` in
src/yzbot/mybot/scripts/fixed_joint_pick_place.py try to solve this by driving
front/back along the *heading*, capped at 0.25 m, and gated on a lidar
clearance that saturates at `range_min = 0.5 m`.  Both are structural
failures: the axis is unrelated to the wall, and the gate can never be met
closer than 0.57 m.

This tool re-derives the segment from the frozen recording instead:

  * the global costmap from probe.jsonl (static + inflation, map frame),
  * the scan from probe.jsonl, usable only beyond `range_min`,
  * the pose from odom or amcl,

and searches headings x lengths for a short straight segment that

  1. never crosses a LETHAL (100) cell,
  2. ends on a cell the planner will accept as a start (cost < 99),
  3. keeps a capsule of `half_width` clear of every usable scan point.

The start cell is amnestied: being inside it is the precondition, not a
disqualifier.  Only the *end* has to be legal -- NavFn takes over from there.

Usage
-----
  python3 tools/escape_segment.py runs/2026-11-26_1702_lookA_m2 --t 184.0
  python3 tools/escape_segment.py runs/2026-11-26_1702_lookA_m2 --sweep 184 220
  python3 tools/escape_segment.py runs/... --t 184.0 --pose amcl --plot out.png
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Published OccupancyGrid scale used by collision_probe: 100 lethal (254
# internally), 99 inscribed (253), 1..98 inflation, 0 free, -1 unknown.
LETHAL = 100
INSCRIBED = 99
# Anything a planner will accept as a start cell.
LEGAL_MAX = 98


# --------------------------------------------------------------------- loading
def load_streams(path: Path, topics: tuple[str, ...]):
    out = {t: [] for t in topics}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("topic") in out:
                out[rec["topic"]].append(rec)
    for t in out:
        out[t].sort(key=lambda r: r["t"])
    return out


@dataclass
class Grid:
    data: np.ndarray   # int16, row 0 = lowest y
    res: float
    ox: float
    oy: float

    @property
    def h(self) -> int:
        return self.data.shape[0]

    @property
    def w(self) -> int:
        return self.data.shape[1]

    def cell(self, x: float, y: float) -> int:
        i = int((x - self.ox) / self.res)
        j = int((y - self.oy) / self.res)
        if 0 <= i < self.w and 0 <= j < self.h:
            return int(self.data[j, i])
        return -1

    def world(self, i: int, j: int) -> tuple[float, float]:
        return self.ox + (i + 0.5) * self.res, self.oy + (j + 0.5) * self.res


def decode_grid(d: dict) -> Grid:
    raw = zlib.decompress(base64.b64decode(d["data"]))
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
    arr[arr > 127] -= 256
    return Grid(arr.reshape(d["h"], d["w"]), d["res"], d["ox"], d["oy"])


def scan_points(d: dict, near: float = 1.6) -> np.ndarray:
    """Scan hits in base frame, restricted to the range the lidar can trust.

    `range_min` saturates: a wall 0.38 m away is reported as exactly
    `range_min`, so hits at or below it carry no distance information and are
    dropped rather than being mistaken for a comfortable 0.5 m.
    """
    raw = zlib.decompress(base64.b64decode(d["ranges"]))
    r = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    ang = d["amin"] + d["ainc"] * np.arange(len(r))
    floor = d["rmin"] + 0.02
    ok = np.isfinite(r) & (r > floor) & (r < near)
    return np.c_[r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])]


def nearest(stream: list[dict], t: float) -> dict | None:
    if not stream:
        return None
    return min(stream, key=lambda r: abs(r["t"] - t))


def pose_at(run: Path, t: float, which: str) -> tuple[float, float, float, float]:
    """Return (x, y, yaw, t_actual) in the costmap frame."""
    if which == "amcl":
        stream = load_streams(run / "events.jsonl", ("amcl_pose",))["amcl_pose"]
        rec = nearest(stream, t)
        if rec is None:
            raise SystemExit("no amcl_pose in events.jsonl")
        d = rec["d"]
    else:
        stream = load_streams(run / "probe.jsonl", ("odom",))["odom"]
        rec = nearest(stream, t)
        d = rec["d"]
    return d["x"], d["y"], d.get("yaw", 0.0), rec["t"]


# --------------------------------------------------------------------- geometry
def segment_clearance(pts: np.ndarray, length: float, ux: float, uy: float) -> float:
    """Smallest distance from `pts` to the segment (0,0)->(ux,uy)*length."""
    if len(pts) == 0:
        return math.inf
    t = np.clip(pts[:, 0] * ux + pts[:, 1] * uy, 0.0, length)
    dx = pts[:, 0] - t * ux
    dy = pts[:, 1] - t * uy
    return float(np.hypot(dx, dy).min())


def away_heading(grid: Grid, x: float, y: float) -> float:
    """Heading that increases distance-to-obstacle fastest (clearance gradient).

    This is the "perpendicular away from the wall" direction, as opposed to
    the front/back axis along the robot heading that `_band_escape_axis()`
    restricts itself to.
    """
    i = int((x - grid.ox) / grid.res)
    j = int((y - grid.oy) / grid.res)
    if not (0 <= i < grid.w and 0 <= j < grid.h):
        return 0.0
    # LETHAL, not INSCRIBED: inside the band every cell is blocked, so a
    # distance-to-band field is flat there and gives no direction.  The wall
    # itself still does.
    blocked = (grid.data >= LETHAL).astype(np.float32)
    crop = blocked[max(0, j - 30):j + 31, max(0, i - 30):i + 31]
    if crop.size == 0 or blocked[j, i]:
        return 0.0
    dist = _brute_distance(crop)
    cj, ci = min(j, 30), min(i, 30)
    gy, gx = np.gradient(dist)
    # np.gradient rows are y, cols are x -- our rows are +y (row 0 lowest y)
    dx, dy = gx[cj, ci], gy[cj, ci]
    n = math.hypot(dx, dy)
    return math.atan2(dy / n, dx / n) if n > 1e-9 else 0.0


def _brute_distance(mask_free: np.ndarray) -> np.ndarray:
    """Euclidean distance (cells) from each cell to the nearest True cell."""
    h, w = mask_free.shape
    yy, xx = np.mgrid[0:h, 0:w]
    obs = np.argwhere(mask_free)
    if len(obs) == 0:
        return np.zeros((h, w), np.float32)
    # grid of candidates -> chunk to keep memory sane
    flat = np.stack([yy.ravel(), xx.ravel()], axis=1)
    out = np.empty(len(flat), np.float32)
    for k in range(0, len(flat), 512):
        chunk = flat[k:k + 512]
        d = np.hypot(chunk[:, None, 0] - obs[None, :, 0],
                     chunk[:, None, 1] - obs[None, :, 1])
        out[k:k + 512] = d.min(axis=1)
    return out.reshape(h, w)


# --------------------------------------------------------------------- search
@dataclass
class Candidate:
    length: float
    heading: float
    end: tuple[float, float]
    end_cost: int
    illegal_len: float
    clearance: float
    score: float = 0.0
    samples: list[tuple[float, float]] = field(default_factory=list)


def search(grid: Grid, x: float, y: float, pts: np.ndarray, *,
           half_width: float = 0.35, max_len: float = 0.60,
           step: float = 0.05, n_heading: int = 72,
           prefer: float | None = None) -> list[Candidate]:
    """Every (heading, length) whose end the planner will accept as a start."""
    start = grid.cell(x, y)
    out: list[Candidate] = []
    for k in range(n_heading):
        th = 2.0 * math.pi * k / n_heading
        ux, uy = math.cos(th), math.sin(th)
        for length in np.arange(step, max_len + 1e-9, step):
            ex, ey = x + length * ux, y + length * uy
            ce = grid.cell(ex, ey)
            if ce < 0 or ce > LEGAL_MAX:
                continue
            ts = np.arange(0.02, length + 1e-9, 0.02)
            cells = [grid.cell(x + q * ux, y + q * uy) for q in ts]
            if LETHAL in cells:
                continue
            clear = segment_clearance(pts, length, ux, uy)
            if clear < half_width:
                continue
            illegal = sum(1 for c in cells if c >= INSCRIBED) * 0.02
            # shorter first, then the cleanest exit, then the straightest
            ang = abs((math.degrees(th) - math.degrees(prefer) + 180) % 360 - 180) \
                if prefer is not None else 0.0
            score = length + 0.004 * ce + 0.8 * illegal + 0.0004 * ang
            out.append(Candidate(
                length=float(length), heading=math.degrees(th),
                end=(ex, ey), end_cost=ce, illegal_len=illegal,
                clearance=clear, score=float(score),
                samples=[(x + q * ux, y + q * uy) for q in np.append(ts, length)]))
    out.sort(key=lambda c: c.score)
    return out


def report(grid: Grid, x: float, y: float, pts: np.ndarray, cands: list[Candidate],
           away: float, tag: str) -> None:
    here = grid.cell(x, y)
    print(f"\n=== {tag}: pose ({x:.2f},{y:.2f})  start cost={here}"
          f"{'  <-- planner refuses this start' if here >= INSCRIBED else ''}")
    print(f"    away-from-wall heading {math.degrees(away):+.0f} deg,"
          f" usable scan hits {len(pts)}")
    if not cands:
        print("    NO feasible segment within the search radius "
              "(body is inside lethal clearance -- nothing to plan)")
        return
    print(f"    {len(cands)} feasible (heading,length) pairs; best:")
    for c in cands[:5]:
        print(f"      L={c.length:.2f} m  hdg={c.heading:+5.0f} deg  -> "
              f"({c.end[0]:.2f},{c.end[1]:.2f})  end cost={c.end_cost:3d}"
              f"  scan margin={c.clearance:.2f} m  in-band travel={c.illegal_len:.2f} m")


# --------------------------------------------------------------------- plotting
def plot(grid: Grid, x: float, y: float, pts: np.ndarray, cands: list[Candidate],
         away: float, out: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = 1.6
    i0 = max(0, int((x - span - grid.ox) / grid.res))
    i1 = min(grid.w, int((x + span - grid.ox) / grid.res))
    j0 = max(0, int((y - span - grid.oy) / grid.res))
    j1 = min(grid.h, int((y + span - grid.oy) / grid.res))
    crop = grid.data[j0:j1, i0:i1]
    ext = (grid.ox + i0 * grid.res, grid.ox + i1 * grid.res,
           grid.oy + j0 * grid.res, grid.oy + j1 * grid.res)

    fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=150)
    ax.imshow(np.where(crop == LETHAL, 1.0, np.nan), origin="lower", extent=ext,
              cmap="Greys", vmin=0, vmax=1)
    ax.contour(np.linspace(ext[0], ext[1], crop.shape[1]),
               np.linspace(ext[2], ext[3], crop.shape[0]),
               (crop >= INSCRIBED).astype(float), levels=[0.5],
               colors=["#d62728"], linestyles="--", linewidths=1.0)
    if len(pts):
        ax.plot(x + pts[:, 0], y + pts[:, 1], ".", ms=2.5, color="#1f77b4",
                alpha=0.6, label="lidar hits (> range_min)")
    best = cands[:1]
    for c in cands[:6]:
        xs = [p[0] for p in c.samples]
        ys = [p[1] for p in c.samples]
        ax.plot(xs, ys, "-", lw=1.0, color="#2ca02c", alpha=0.35)
    for c in best:
        xs = [p[0] for p in c.samples]
        ys = [p[1] for p in c.samples]
        ax.plot(xs, ys, "-", lw=2.6, color="#2ca02c",
                label=f"escape {c.length:.2f} m @ {c.heading:+.0f} deg")
        ax.plot(*c.end, "o", ms=7, mfc="none", mec="#2ca02c", mew=2)
    ax.plot([x, x + 0.55 * math.cos(away)], [y, y + 0.55 * math.sin(away)],
            "-", color="#ff7f0e", lw=1.8, label="away-from-wall (clearance +)")
    ax.plot(x, y, "o", ms=8, color="#d62728", label="base")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="runs/<date>_<rule>_<mN>")
    ap.add_argument("--t", type=float, default=None, help="seconds into the run")
    ap.add_argument("--sweep", nargs=2, type=float, metavar=("T0", "T1"),
                    help="walk the whole interval and report feasibility per step")
    ap.add_argument("--pose", choices=("odom", "amcl"), default="odom")
    ap.add_argument("--half-width", type=float, default=0.35,
                    help="capsule half width kept clear of lidar hits (m)")
    ap.add_argument("--max-len", type=float, default=0.60)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    streams = load_streams(args.run / "probe.jsonl",
                           ("global_costmap", "scan", "odom"))

    def snap(t: float, pose: str = None):
        pose = pose or args.pose
        grid = decode_grid(nearest(streams["global_costmap"], t)["d"])
        sc = nearest(streams["scan"], t)
        pts = scan_points(sc["d"]) if sc else np.zeros((0, 2))
        x, y, _, _ = pose_at(args.run, t, pose)
        return grid, x, y, pts

    if args.sweep:
        t0, t1 = args.sweep
        print(" t     pose(x,y)        grid_cost  #feasible  shortest")
        for t in np.arange(t0, t1 + 1e-9, 1.0):
            grid, x, y, pts = snap(float(t))
            away = away_heading(grid, x, y)
            c = search(grid, x, y, pts, half_width=args.half_width,
                       max_len=args.max_len, prefer=away)
            short = f"L={c[0].length:.2f} m @ {c[0].heading:+.0f} deg" if c else "NONE"
            print(f"{t:6.1f}  ({x:5.2f},{y:6.2f})   {grid.cell(x, y):3d}      "
                  f"{len(c):3d}     {short}")
        return 0

    if args.t is None:
        raise SystemExit("pass --t or --sweep")

    grid, x, y, pts = snap(args.t)
    away = away_heading(grid, x, y)
    cands = search(grid, x, y, pts, half_width=args.half_width,
                   max_len=args.max_len, prefer=away)
    report(grid, x, y, pts, cands, away, f"{args.run.name} t={args.t:.1f}")
    if args.plot:
        plot(grid, x, y, pts, cands, away, args.plot,
             f"{args.run.name} t={args.t:.1f}s -- escape segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

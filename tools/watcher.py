#!/usr/bin/env python3
"""External stall supervisor for one competition round.

The executor decides on its own whether a navigation leg "failed", and it does
so partly on elapsed time. That is the wrong judge for a robot that is moving:
a slow but progressing leg must never be killed, while a base that sits still
for 15 s is a defect worth stopping the whole round for.

So this process watches from the outside and never touches the car code:

  * base motion   - /odom, 5 cm of displacement is what counts as moving
  * commands      - /cmd_vel, tells a jam (commanded, not moving) from a wait
                    (nothing commanded, the stack decided to stand still)
  * round state   - /competition/state, so the pre-start wait is not a stall
  * stack log     - tail of <run>/stack.log, to say WHY the base is standing

When the base has not moved for --stall-sec while the round is running it
writes a diagnosis bundle (position, command history, the log window, the
dominant log reasons) and, with --kill, stops the stack so the round ends
where the defect happened instead of drifting into a timeout.

    python3 tools/watcher.py --run-dir runs/<round> [--stall-sec 15] [--kill]

It needs a ROS environment (rclpy for /odom, /cmd_vel, /competition/state):

    source /opt/ros/humble/setup.bash && source install/setup.bash
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

STAMP_RE = re.compile(r'\[(\d{9,}\.\d+)\]')
# Log signatures, most specific first. The first family that matches the log
# window is what the report names as the reason the base stood still.
SIGNATURES = [
    ('traffic', ('YIELD', 'mover hold', 'holding still for a moving',
                 'mover hold clear', 'held for', 'corridor clear')),
    ('controller_refusal', ('collision ahead', 'collision refusals',
                            'Collision Ahead', 'collision_monitor',
                            'Robot to approach')),
    ('behaviour', ('Running backup', 'Running spin', 'backup failed',
                   'spin failed', 'behavior_server')),
    ('planner', ('failed to create plan', 'Planning algorithm',
                 'No valid control', 'planner_server')),
    ('band', ('inscribed band', 'stale odometry', 'guarded retreat',
              'start pose is inside', 'no verified recovery')),
    ('manipulation', ('TIMING arm:', 'TIMING vision', 'Move to grasp',
                      'Open gripper', 'Lower to place', 'Close gripper',
                      'Arm transport', 'vision_align', 'creep_to_grasp')),
]
ACTIVE_STATES = {
    'NAV_TO_OBJECT', 'NAV_TO_ZONE', 'DETECTING_OBJECT', 'PRE_GRASP',
    'GRASPING', 'VERIFY_GRASP', 'PLACING', 'VERIFY_PLACE', 'NEXT_OBJECT',
    'RETREATING', 'LIFTING',
}
# States where standing still is the point for the BASE: the arm is working at
# a dock. Docking, grasping and placing are single-spot operations by design,
# and counting them as stalls "confirmed" a stall at every warehouse and let
# the watcher kill a healthy round (2026-11-25 runs/2026-11-25_1400_rule3_wedge:
# 4 stalls, all at a dock, all verdict WAIT). A wedge during manipulation is not
# what this tool is for: it watches navigation.
SETTLE_STATES = {
    'DETECTING_OBJECT', 'PRE_GRASP', 'GRASPING', 'VERIFY_GRASP',
    'PLACING', 'VERIFY_PLACE', 'NEXT_OBJECT', 'LIFTING',
}
# States where standing still is the point: the round has not started, it is
# parsing the task, or it is already over.
IDLE_STATES = {
    'IDLE', 'WAITING_FOR_START', 'PARSING_TASK', 'VALIDATING_TASK',
    'COMPLETED', 'FAILED', 'ABORTED', '',
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--stall-sec', type=float, default=15.0,
                   help='no motion for this long counts as a stall')
    p.add_argument('--stall-dist', type=float, default=0.05,
                   help='displacement below this over the window is "still"')
    p.add_argument('--area-radius', type=float, default=0.50,
                   help='the base counts as stalled once it has stayed within '
                        'this radius for --stall-sec, which also catches a base '
                        'that oscillates in a dead end')
    p.add_argument('--area-window', type=float, default=90.0,
                   help='pose history used to measure the area hold')
    p.add_argument('--kill', action='store_true',
                   help='stop the stack when a stall is confirmed')
    p.add_argument('--kill-after', type=float, default=30.0,
                   help='also stop when a standstill reaches this long, even '
                        'if the stack never commanded motion')
    p.add_argument('--grace-sec', type=float, default=0.0,
                   help='ignore stalls during the first seconds of the round')
    p.add_argument('--max-stalls', type=int, default=0,
                   help='exit after this many recorded stalls (0 = never)')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


class Watcher:
    def __init__(self, args):
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        import rclpy

        self.rclpy = rclpy
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.out = self.run_dir / 'watcher'
        self.out.mkdir(parents=True, exist_ok=True)

        rclpy.init(args=['--ros-args', '--log-level', 'WARN'])
        self.node = rclpy.create_node('external_watcher')
        self.node.create_subscription(Odometry, '/odom', self._on_odom, 20)
        self.node.create_subscription(Twist, '/cmd_vel', self._on_cmd, 20)
        self.node.create_subscription(String, '/competition/state',
                                      self._on_state, 20)

        self.track = collections.deque()      # (t, x, y, vx) recent poses
        self.cmd = collections.deque()        # (t, linear, angular)
        self.state = 'WAITING_FOR_START'
        self.round_start = None
        self.stalls = []
        self.timeline = []
        self.last_row = 0.0
        self.stopped = False
        self.csv_path = self.out / 'timeline.csv'
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['t', 'state', 'x', 'y', 'speed', 'cmd_lin',
                           'cmd_ang', 'displacement', 'moving'])

    # ------------------------------------------------------------ callbacks
    def _on_odom(self, msg):
        now = time.monotonic()
        twist = msg.twist.twist
        self.track.append((now, msg.pose.pose.position.x,
                           msg.pose.pose.position.y, twist.linear.x))
        while self.track and now - self.track[0][0] > 90.0:
            self.track.popleft()

    def _on_cmd(self, msg):
        now = time.monotonic()
        self.cmd.append((now, msg.linear.x, msg.angular.z))
        while self.cmd and now - self.cmd[0][0] > 12.0:
            self.cmd.popleft()

    def _on_state(self, msg):
        value = (msg.data or '').strip()
        if not value or value == self.state:
            return
        self.state = value
        if self.round_start is None and value not in IDLE_STATES:
            self.round_start = time.monotonic()
            self._say(f'round started (state={value})')

    # --------------------------------------------------------------- helpers
    def _say(self, text):
        if not self.args.quiet:
            print(f'[{(time.strftime("%H:%M:%S"))}] {text}', flush=True)

    def _recent_cmds(self, seconds=12.0):
        now = time.monotonic()
        return [c for c in self.cmd if now - c[0] <= seconds]

    def _window(self):
        """(displacement, mean speed, mean commanded linear speed)."""
        if len(self.track) < 2:
            return 0.0, 0.0, 0.0
        first, last = self.track[0], self.track[-1]
        dist = math.hypot(last[1] - first[1], last[2] - first[2])
        speeds = [abs(t[3]) for t in self.track]
        cmds = self._recent_cmds()
        cmd_lin = sum(abs(c[1]) for c in cmds) / len(cmds) if cmds else 0.0
        return dist, sum(speeds) / len(speeds), cmd_lin

    def _area_hold(self, radius):
        """How long the base has stayed inside one `radius`-sized area.

        A base that oscillates in a dead end is not progressing either, so the
        judge is "did it stay in one small area", not "was the wheel still".
        The area is measured as the bounding box of the samples, newest last:
        the span reported is the longest recent run that still fits in a box of
        ``2 * radius``, i.e. the same size the old anchor rule admitted.

        The previous version re-seated its anchor only when a sample was farther
        than ``radius`` *from the anchor*, which a straight 7 m drive never
        triggered once the first sample had been adopted - it reported
        "held one area for 15 s" with a 7.1 m displacement (runs/
        2026-11-25_1640_circle_m2/watcher/stall_01.md). A bounding box cannot
        make that mistake: moving grows the box, so the span collapses.
        """
        if len(self.track) < 2:
            return 0.0, None
        now = time.monotonic()
        window = [s for s in self.track if now - s[0] <= self.args.area_window]
        if len(window) < 2:
            return 0.0, None

        def box_fits(samples):
            # 2 * radius, not radius: the old criterion was "every sample within
            # radius of one anchor", whose worst case is two samples 2 * radius
            # apart. Same sensitivity, no chained random walk.
            xs = [s[1] for s in samples]
            ys = [s[2] for s in samples]
            return (math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                    <= 2.0 * radius)

        start = len(window) - 1
        while start > 0 and box_fits(window[start - 1:]):
            start -= 1
        run = window[start:]
        if not box_fits(run):
            return 0.0, None
        anchor = run[0]
        return now - anchor[0], anchor

    def _log_lines(self):
        """Last lines of stack.log, newest last, with wall-clock stamps."""
        path = self.run_dir / 'stack.log'
        if not path.exists():
            return []
        try:
            with path.open('rb') as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 400_000))
                blob = handle.read().decode('utf-8', 'replace')
        except OSError:
            return []
        return blob.splitlines()

    def _classify(self, since_wall):
        lines = self._log_lines()
        recent = []
        for line in reversed(lines):
            match = STAMP_RE.search(line)
            if not match:
                continue
            if float(match.group(1)) < since_wall:
                break
            recent.append(line)
        recent.reverse()
        counts = collections.Counter()
        for name, needles in SIGNATURES:
            for line in recent:
                if any(n in line for n in needles):
                    counts[name] += 1
        return recent, counts

    def _goal(self):
        lines = self._log_lines()
        for line in reversed(lines):
            if 'Begin navigating from current location' in line:
                return line.split(']')[-1].strip()
        return ''

    # ------------------------------------------------------------------ dump
    def _dump(self, index, dist, speed, cmd_lin, recent, counts):
        now = time.monotonic()
        pose = self.track[-1] if self.track else (now, 0.0, 0.0, 0.0)
        dominant = counts.most_common(3)
        head = ', '.join(f'{k} x{v}' for k, v in dominant) or 'no matching log lines'
        body = [
            f'# stall {index}',
            '',
            f'- wall time: {time.strftime("%H:%M:%S")}',
            f'- state: `{self.state}`',
            f'- base pose: ({pose[1]:.2f}, {pose[2]:.2f})',
            f'- held one area for: {self.args.stall_sec:.1f} s '
            f'(radius {self.args.area_radius:.2f} m)',
            f'- displacement in window: {dist:.3f} m (threshold '
            f'{self.args.stall_dist:.2f} m)',
            f'- mean odom speed: {speed:.3f} m/s, mean commanded linear: '
            f'{cmd_lin:.3f} m/s',
            f'- verdict: ' + ('JAM (commanded, but no progress)'
                              if cmd_lin > 0.05 else
                              'WAIT (nothing commanded, the stack stood still)'),
            f'- dominant log reasons: {head}',
            f'- last goal: {self._goal()}',
            '',
            '## log window (oldest first)',
            '```',
        ]
        body.extend(line[-220:] for line in recent[-120:])
        body.append('```')
        path = self.out / f'stall_{index:02d}.md'
        path.write_text('\n'.join(body) + '\n', encoding='utf-8')
        self._say(f'STALL #{index}: {dist:.3f} m in {self.args.stall_sec:.0f} s '
                  f'at ({pose[1]:.2f},{pose[2]:.2f}) state={self.state} '
                  f'-> {head}')
        self._say(f'  report: {path}')
        return path

    def _kill_stack(self):
        pid_file = self.run_dir / 'stack.pid'
        if not pid_file.exists():
            self._say('kill requested but stack.pid is missing')
            return
        try:
            pgid = int(pid_file.read_text().split()[0])
        except (OSError, ValueError):
            self._say('kill requested but stack.pid is unreadable')
            return
        for sig, wait in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                self.stopped = True
                return
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                time.sleep(0.5)
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    self.stopped = True
                    return
        self.stopped = True
        self._say('stack did not exit after SIGTERM')

    # ------------------------------------------------------------------ loop
    def run(self):
        loop = self.rclpy.spin_once
        reported = False
        last_beat = 0.0
        while self.rclpy.ok() and not self.stopped:
            try:
                loop(self.node, timeout_sec=0.1)
            except Exception:  # noqa: BLE001 - context shut down under us
                break
            now = time.monotonic()
            if self.state in IDLE_STATES or self.round_start is None:
                reported = False
                continue
            if now - self.round_start < self.args.grace_sec:
                continue
            if self.state in SETTLE_STATES:
                # The base is expected to hold position while the arm works.
                # Drop the history so the hold restarts once navigation does.
                self.track.clear()
                reported = False
                continue
            dist, speed, cmd_lin = self._window()
            held, anchor = self._area_hold(self.args.area_radius)
            if now - self.last_row >= 0.5:
                pose = self.track[-1] if self.track else (0, 0, 0, 0)
                self.csv.writerow([f'{now - self.round_start:.1f}', self.state,
                                   f'{pose[1]:.3f}', f'{pose[2]:.3f}',
                                   f'{speed:.3f}', f'{cmd_lin:.3f}',
                                   f'{held:.1f}', f'{dist:.3f}',
                                   int(dist >= self.args.stall_dist)])
                self.csv_file.flush()
                self.timeline.append((now - self.round_start, dist,
                                      dist >= self.args.stall_dist))
                self.last_row = now
            if now - last_beat >= 5.0 and held >= 2.0:
                last_beat = now
                where = anchor or self.track[-1]
                self._say(f'in one spot {held:.0f} s around '
                          f'({where[1]:.2f},{where[2]:.2f}) '
                          f'cmd={cmd_lin:.2f} state={self.state}')
            if held < self.args.stall_sec:
                reported = False
                continue
            # Confirmed stall: report once per episode, but keep watching so a
            # hold that turns into a wedge is still caught.
            if not reported:
                since_wall = time.time() - (held + 1.0)
                recent, counts = self._classify(since_wall)
                self.stalls.append(self._dump(len(self.stalls) + 1, dist, speed,
                                              cmd_lin, recent, counts))
                reported = True
                if self.args.max_stalls and \
                        len(self.stalls) >= self.args.max_stalls:
                    break
            jam = cmd_lin > 0.05
            # A standstill with nothing commanded is a wait until it outlasts
            # --kill-after; a commanded standstill is a wedge right away.
            if self.args.kill and (jam or held >= self.args.kill_after):
                why = 'jammed (commanded, not moving)' if jam else \
                      f'in one spot for {held:.0f} s with nothing commanded'
                self._say(f'stopping the round: {why}')
                self._kill_stack()
                break
        self.finish()

    def finish(self):
        self.csv_file.close()
        summary = self.out / 'summary.md'
        lines = [f'# watcher summary — {self.run_dir.name}', '',
                 f'- samples: {len(self.timeline)}',
                 f'- stalls: {len(self.stalls)}',
                 f'- final state: `{self.state}`']
        if self.stalls:
            lines.append('')
            lines.append('| # | report |')
            lines.append('|---|---|')
            for i, path in enumerate(self.stalls, 1):
                lines.append(f'| {i} | {path.name} |')
        summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        self._say(f'watcher finished: {len(self.stalls)} stall(s) -> {summary}')
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:  # noqa: BLE001 - shutdown races are not our result
            pass


def main(argv=None):
    args = parse_args(argv)
    # ~/.ros is not writable in this sandbox; keep ROS artefacts in the run dir.
    run_dir = Path(args.run_dir).resolve()
    os.environ.setdefault('ROS_HOME', str(run_dir / 'roshome'))
    os.environ.setdefault('ROS_LOG_DIR', str(run_dir / 'roslog'))
    os.makedirs(os.environ['ROS_LOG_DIR'], exist_ok=True)
    watcher = Watcher(args)
    try:
        watcher.run()
    except KeyboardInterrupt:
        watcher.finish()
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Turn one recorded competition round into structured timing/route/obstacle data.

Inputs (written by tools/run_round.sh):
    <round>/stack.log        ROS log of the whole launch (timestamps included)
    <round>/events.jsonl     tools/round_recorder.py output

Output: <round>/analysis.json, and a short human summary on stdout.

All positions are reported in the map frame (metres, compass heading measured
from +x towards +y).  Timing has two independent sources:

  * TIMING lines   - the executor's own stopwatch per step (nav:/vision/arm:)
  * Executor state - wall clock spent in NAV_TO_OBJECT, DETECTING_OBJECT, ...
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

LINE_RE = re.compile(
    r'\[(?P<sev>[A-Z]+)\] \[(?P<ts>\d+\.\d+)\] \[(?P<node>[^\]]+)\]: (?P<msg>.*)')
TIMING_RE = re.compile(r'^TIMING (\S+(?: \S+)*?) ([-\d.]+)$')
RESIDENT_RE = re.compile(
    r'^Resident goal (.+?) -> (.+?) \(object_id=(\d+)\)\.')
STATE_RE = re.compile(r'^State -> ([A-Z_]+)$')
EXEC_STATE_RE = re.compile(r'^Executor state: ([A-Z_]+)$')
# The reason token carries a hyphen (``predict-stop``, ``predict-arc``,
# ``predict-slow``, ``swept-stop``, ``wait-expired``), so ``\w+`` used to drop
# every real YIELD line and keep only the YIELD_DIAG status lines.  Fixed for
# the 2026-11-21 run set, where rule 3 had 116 real decisions and the report
# showed only the 115 diagnostics.
YIELD_RE = re.compile(r'^YIELD (\w+) \(([\w-]+)\) box (.+?) wait=([-\d.]+)s: (.*)$')
YIELD_DIAG_RE = re.compile(r'^YIELD_DIAG .*decision=(\w+)/(\w+)')
SPEED_RE = re.compile(r'^(\w+): speed limit ([\d.]+) -> ([\d.]+) m/s(.*)$')

TERMINAL = {'COMPLETED', 'FAILED', 'ABORTED'}

PHASE_OF_STATE = {
    'NAV_TO_OBJECT': 'nav', 'NAV_TO_ZONE': 'nav', 'HOMING': 'nav',
    'DETECTING_OBJECT': 'vision', 'VISUAL_ALIGN': 'vision',
    'PRE_GRASP': 'pick', 'GRASPING': 'pick', 'VERIFY_GRASP': 'pick',
    'LIFTING': 'pick', 'PLACING': 'place', 'VERIFY_PLACE': 'place',
    'RETREAT': 'place',
}


# --------------------------------------------------------------------- map
def read_map(yaml_path):
    """Return (occupied mask, resolution, origin_x, origin_y, height)."""
    text = Path(yaml_path).read_text(encoding='utf-8')
    image = re.search(r'^image:\s*(\S+)', text, re.M).group(1)
    res = float(re.search(r'^resolution:\s*([\d.]+)', text, re.M).group(1))
    m = re.search(r'^origin:\s*\[\s*([-\d.]+),\s*([-\d.]+)', text, re.M)
    ox, oy = float(m.group(1)), float(m.group(2))
    data = (Path(yaml_path).parent / image).read_bytes()
    fields, index = [], 0
    while len(fields) < 4:
        while data[index:index + 1].isspace():
            index += 1
        if data[index:index + 1] == b'#':
            while data[index:index + 1] != b'\n':
                index += 1
            continue
        end = index
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[index:end])
        index = end
    index += 1
    width, height = int(fields[1]), int(fields[2])
    pixels = np.frombuffer(
        data[index:index + width * height], dtype=np.uint8).reshape(height, width)
    occupied = pixels < 89          # map_server trinary: p > 0.65
    return occupied, res, ox, oy, height


class MapClearance:
    def __init__(self, yaml_path):
        self.occupied, self.res, self.ox, self.oy, self.h = read_map(yaml_path)
        self.dist = distance_transform_edt(~self.occupied) * self.res

    def to_pixel(self, x, y):
        col = int(round((x - self.ox) / self.res))
        row = self.h - 1 - int(round((y - self.oy) / self.res))
        return col, row

    def clearance(self, x, y):
        col, row = self.to_pixel(x, y)
        col = min(max(col, 0), self.dist.shape[1] - 1)
        row = min(max(row, 0), self.dist.shape[0] - 1)
        return float(self.dist[row, col])

    def clearance_many(self, pts):
        return np.array([self.clearance(x, y) for x, y in pts])

    def is_occupied(self, x, y):
        col, row = self.to_pixel(x, y)
        if not (0 <= col < self.occupied.shape[1] and
                0 <= row < self.occupied.shape[0]):
            return True
        return bool(self.occupied[row, col])


# --------------------------------------------------------------------- log
def parse_log(path):
    entries = []
    for raw in open(path, encoding='utf-8', errors='replace'):
        m = LINE_RE.search(raw)
        if not m:
            continue
        entries.append({'ts': float(m['ts']), 'sev': m['sev'],
                        'node': m['node'], 'msg': m['msg'].strip()})
    return entries


def timing_bucket(label):
    """Map a TIMING label to a coarse bucket; nested timers return None."""
    if label == 'vision_align' or label == 'creep_to_grasp':
        return 'vision'
    if label.startswith('nav:'):
        return 'nav'
    if label.startswith('creep:'):
        return None            # runs inside the enclosing nav timer
    if label.startswith('arm:'):
        return 'manip'
    if label in ('svc:attach', 'svc:detach'):
        return 'manip'
    return 'misc'


def build_objects(entries, fallback_end):
    objs = []
    for e in entries:
        m = RESIDENT_RE.match(e['msg'])
        if m:
            objs.append({'index': len(objs) + 1, 'cube': m.group(1),
                         'zone': m.group(2), 'object_id': int(m.group(3)),
                         'start_ts': e['ts'], 'end_ts': None,
                         'timings': []})
            continue
        if not objs:
            continue
        t = TIMING_RE.match(e['msg'])
        if t:
            objs[-1]['timings'].append(
                {'label': t.group(1), 'sec': float(t.group(2)), 'ts': e['ts']})
    for i in range(len(objs) - 1):
        objs[i]['end_ts'] = objs[i + 1]['start_ts']
    if objs:
        objs[-1]['end_ts'] = fallback_end
    for o in objs:
        buckets = {}
        for t in o['timings']:
            b = timing_bucket(t['label'])
            if b is None:
                o.setdefault('nested', []).append(
                    {'label': t['label'], 'sec': t['sec']})
                continue
            buckets[b] = round(buckets.get(b, 0.0) + t['sec'], 3)
        o['buckets'] = buckets
        o['timing_total'] = round(sum(t['sec'] for t in o['timings']), 3)
        o['wall_s'] = round(o['end_ts'] - o['start_ts'], 3)
        o['nav_s'] = buckets.get('nav', 0.0)
        o['vision_s'] = buckets.get('vision', 0.0)
        o['manip_s'] = buckets.get('manip', 0.0)
        o['misc_s'] = buckets.get('misc', 0.0)
        o['nav_labels'] = [
            {'label': t['label'][4:], 'sec': t['sec'], 'ts': t['ts']}
            for t in o['timings'] if t['label'].startswith('nav:')]
    return objs


def build_states(entries):
    intervals = []
    cur = None
    for e in entries:
        m = STATE_RE.match(e['msg'])
        if not m:
            continue
        if cur:
            intervals.append({'state': cur[0], 't0': cur[1], 't1': e['ts'],
                              'dur': e['ts'] - cur[1]})
        cur = (m.group(1), e['ts'])
    return intervals


def build_exec_states(entries):
    out = []
    for e in entries:
        m = EXEC_STATE_RE.match(e['msg'])
        if m:
            out.append({'state': m.group(1), 'ts': e['ts']})
    return out


# ------------------------------------------------------------------ events
def load_events(path):
    by = {}
    for line in open(path, encoding='utf-8'):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        by.setdefault(rec['topic'], []).append((rec['w'], rec['d']))
    return by


def series(events, topic):
    return events.get(topic, [])


def odom_arrays(events):
    rows = series(events, 'odom')
    if not rows:
        return None
    t = np.array([r[0] for r in rows])
    x = np.array([r[1]['x'] for r in rows])
    y = np.array([r[1]['y'] for r in rows])
    yaw = np.array([r[1]['yaw'] for r in rows])
    vx = np.array([r[1]['vx'] for r in rows])
    wz = np.array([r[1]['wz'] for r in rows])
    return {'t': t, 'x': x, 'y': y, 'yaw': yaw, 'vx': vx, 'wz': wz}


def fit_rigid(src, dst):
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return r, cd - r @ cs


def nearest_index(times, query):
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = times[idx - 1]
    right = times[idx]
    return np.where(np.abs(query - left) <= np.abs(query - right),
                    idx - 1, idx)


def align_to_map(odom, events):
    """Return (odom->map rotation, translation, residual) using AMCL poses."""
    amcl = series(events, 'amcl_pose')
    if len(amcl) < 5:
        return np.eye(2), np.zeros(2), None
    ta = np.array([a[0] for a in amcl])
    pa = np.array([[a[1]['x'], a[1]['y']] for a in amcl])
    idx = nearest_index(odom['t'], ta)
    po = np.column_stack([odom['x'][idx], odom['y'][idx]])
    r, tvec = fit_rigid(po, pa)
    resid = np.linalg.norm((po @ r.T + tvec) - pa, axis=1)
    return r, tvec, float(np.max(resid))


def odom_to_world(events, odom):
    """Fit odom -> Gazebo world with the ground-truth base_link poses."""
    rows = series(events, 'link_states')
    if len(rows) < 5:
        return None
    tw = np.array([r[0] for r in rows])
    pw = np.array([[r[1]['x'], r[1]['y']] for r in rows])
    idx = nearest_index(odom['t'], tw)
    po = np.column_stack([odom['x'][idx], odom['y'][idx]])
    r, tvec = fit_rigid(po, pw)
    resid = np.linalg.norm(transform(po, r, tvec) - pw, axis=1)
    return {'r': r, 't': tvec, 'residual_m': float(np.max(resid)),
            'p95_m': float(np.percentile(resid, 95)), 'n': len(rows)}


def transform(pts, r, tvec):
    return np.asarray(pts) @ r.T + tvec


def yaw_from_r(r):
    return math.atan2(r[1, 0], r[0, 0])


# -------------------------------------------------------------------- legs
def motion_legs(odom, r, tvec, yaw_off, gap=1.2, min_dur=1.0):
    t = odom['t']
    x, y = odom['x'], odom['y']
    yaw = odom['yaw']
    d = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(t)
    speed = np.concatenate([[0.0], d / np.maximum(dt, 1e-3)])
    wz = np.abs(odom['wz'])
    moving = (speed > 0.05) | (wz > 0.15)

    legs = []
    start = None
    last_true = None
    for i, m in enumerate(moving):
        if m:
            if start is None:
                start = i
            last_true = i
        elif start is not None and last_true is not None and \
                t[i] - t[last_true] > gap:
            if t[last_true] - t[start] >= min_dur:
                legs.append((start, last_true))
            start = None
    if start is not None and t[last_true] - t[start] >= min_dur:
        legs.append((start, last_true))

    out = []
    for k, (i0, i1) in enumerate(legs):
        pts = np.column_stack([x[i0:i1 + 1], y[i0:i1 + 1]])
        pts_map = transform(pts, r, tvec)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        dist = float(seg.sum())
        dur = float(t[i1] - t[i0])
        y0 = yaw[i0] + yaw_off
        y1 = yaw[i1] + yaw_off
        chord = pts_map[-1] - pts_map[0]
        chord_len = float(np.linalg.norm(chord))
        if chord_len > 0.05:
            dev = np.abs(np.cross(chord / chord_len,
                                  pts_map - pts_map[0]))
            max_dev = float(dev.max())
        else:
            max_dev = 0.0
        out.append({
            'index': k + 1,
            't0': float(t[i0]), 't1': float(t[i1]), 'dur': round(dur, 2),
            'dist': round(dist, 2),
            'start': [round(float(pts_map[0][0]), 2),
                      round(float(pts_map[0][1]), 2),
                      round(math.degrees(y0), 1)],
            'end': [round(float(pts_map[-1][0]), 2),
                    round(float(pts_map[-1][1]), 2),
                    round(math.degrees(y1), 1)],
            'heading_start': round(math.degrees(y0) % 360, 1),
            'heading_end': round(math.degrees(y1) % 360, 1),
            'turn_deg': round(math.degrees(
                math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))), 1),
            'mean_speed': round(dist / dur, 2) if dur > 0 else 0.0,
            'max_speed': round(float(speed[i0:i1 + 1].max()), 2),
            'max_dev_m': round(max_dev, 2),
            'chord': round(chord_len, 2),
            '_pts': pts_map.tolist(),
        })
    return out, speed


def nav_phases(objects, odom, r, tvec, yaw_off, cmap):
    """Route of every `nav:*` stopwatch inside one object (map frame)."""
    t = odom['t']
    yaw = odom['yaw'] + yaw_off
    px, py = transform(np.column_stack([odom['x'], odom['y']]), r, tvec).T
    for o in objects:
        phases = []
        for timed in o['timings']:
            if not timed['label'].startswith('nav:'):
                continue
            t0, t1 = timed['ts'] - timed['sec'], timed['ts']
            i0 = max(0, int(np.searchsorted(t, t0)))
            i1 = min(len(t) - 1, int(np.searchsorted(t, t1)))
            if i1 - i0 < 3:
                continue
            sx, sy = px[i0:i1 + 1], py[i0:i1 + 1]
            step = np.hypot(np.diff(sx), np.diff(sy))
            dist = float(step.sum())
            cum = np.r_[0.0, np.cumsum(step)]
            marks = np.arange(0.0, max(dist - 0.3, 0.05), 2.0).tolist()
            marks.append(max(dist, 0.0))
            pts, seen = [], set()
            for target in marks:
                k = min(int(np.searchsorted(cum, target)), len(sx) - 1)
                if k in seen:
                    continue
                seen.add(k)
                pts.append([round(float(sx[k]), 2), round(float(sy[k]), 2),
                            round(float(math.degrees(yaw[i0 + k]) % 360), 1)])
            cl = cmap.clearance_many(np.column_stack([sx, sy]))
            kmin = int(np.argmin(cl))
            chord_cl = []
            p0 = np.array([sx[0], sy[0]])
            p1 = np.array([sx[-1], sy[-1]])
            for frac in np.linspace(0, 1, 30):
                q = p0 + frac * (p1 - p0)
                chord_cl.append(cmap.clearance(q[0], q[1]))
            phases.append({
                'label': timed['label'][4:], 'sec': round(timed['sec'], 3),
                't0': round(float(t[i0]), 3), 't1': round(float(t[i1]), 3),
                'dist': round(dist, 2),
                'start': [round(float(sx[0]), 2), round(float(sy[0]), 2),
                          round(float(math.degrees(yaw[i0]) % 360), 1)],
                'end': [round(float(sx[-1]), 2), round(float(sy[-1]), 2),
                        round(float(math.degrees(yaw[i1]) % 360), 1)],
                'min_clearance': round(float(cl[kmin]), 3),
                'min_clearance_at': [round(float(sx[kmin]), 2),
                                     round(float(sy[kmin]), 2)],
                'chord_min_clearance': round(float(min(chord_cl)), 3),
                'chord_len': round(float(np.hypot(*(p1 - p0))), 2),
                'waypoints': pts[:16],
                'turns': turn_profile(odom, r, yaw_off,
                                      {'t0': float(t[i0]), 't1': float(t[i1])}),
            })
        o['nav_paths'] = phases
    return objects


def vision_stats(objects, events):
    """Search / align split of the visual servo, per object."""
    rows = series(events, 'vision_alignment')
    if not rows:
        return objects
    for o in objects:
        t0, t1 = o['start_ts'] - 1.0, (o['end_ts'] or 1e18) + 1.0
        seq = [(t, d['data']) for t, d in rows if t0 <= t <= t1]
        if not seq:
            o['vision'] = None
            continue
        search = align = 0.0
        runs, cur, cur_t0 = [], None, None
        xs, bottoms = [], []
        for t, msg in seq:
            kind = msg.split(':')[0]
            if kind != cur:
                if cur is not None:
                    runs.append({'kind': cur, 'dur': t - cur_t0})
                cur, cur_t0 = kind, t
            if kind == 'ALIGNING':
                parts = msg.split(':')
                for part in parts[2:]:
                    if part.startswith('x='):
                        try:
                            xs.append(float(part[2:]))
                        except ValueError:
                            pass
                    elif part.startswith('bottom='):
                        try:
                            bottoms.append(float(part[7:]))
                        except ValueError:
                            pass
        if cur is not None:
            runs.append({'kind': cur, 'dur': seq[-1][0] - cur_t0 + 0.35})
        for r in runs:
            if r['kind'] == 'SEARCHING':
                search += r['dur']
            elif r['kind'] == 'ALIGNING':
                align += r['dur']
        o['vision'] = {
            'search_s': round(search, 2), 'align_s': round(align, 2),
            'reacquire': max(0, sum(1 for r in runs
                                    if r['kind'] == 'SEARCHING') - 1),
            'x_first': round(xs[0], 3) if xs else None,
            'x_last': round(xs[-1], 3) if xs else None,
            'bottom_last': bottoms[-1] if bottoms else None,
            'samples': len(seq),
        }
    return objects


def leg_clearance(legs, cmap):
    for leg in legs:
        pts = np.array(leg['_pts'])
        cl = cmap.clearance_many(pts)
        i = int(np.argmin(cl))
        leg['min_clearance'] = round(float(cl[i]), 3)
        leg['min_clearance_at'] = [round(float(pts[i][0]), 2),
                                   round(float(pts[i][1]), 2)]
        leg['min_clearance_frac'] = round(i / max(1, len(pts) - 1), 2)
        leg['mean_clearance'] = round(float(cl.mean()), 3)
        # Does the straight chord cut through an obstacle the route avoids?
        p0, p1 = np.array(leg['_pts'][0]), np.array(leg['_pts'][-1])
        chord_cl = []
        for s in np.linspace(0, 1, 25):
            q = p0 + s * (p1 - p0)
            chord_cl.append(cmap.clearance(q[0], q[1]))
        leg['chord_min_clearance'] = round(float(min(chord_cl)), 3)
    return legs


def turn_profile(odom, r, yaw_off, leg):
    """Split a leg into straight runs and turns for the route narrative."""
    t = odom['t']
    i = np.searchsorted(t, leg['t0'])
    j = np.searchsorted(t, leg['t1'])
    i = max(0, i - 1)
    j = min(len(t) - 1, j + 1)
    tt = t[i:j + 1]
    yaw = odom['yaw'][i:j + 1] + yaw_off
    x, y = odom['x'][i:j + 1], odom['y'][i:j + 1]
    if len(tt) < 6:
        return []
    rate = np.gradient(np.unwrap(yaw), tt)
    rate = np.convolve(rate, np.ones(5) / 5, mode='same')
    turning = np.abs(rate) > 0.35
    events = []
    k = 0
    n = len(tt)
    while k < n:
        if not turning[k]:
            k += 1
            continue
        k2 = k
        while k2 + 1 < n and (turning[k2 + 1] or k2 + 1 < k + 3):
            k2 += 1
        delta = math.degrees(math.atan2(
            math.sin(yaw[k2] - yaw[k]), math.cos(yaw[k2] - yaw[k])))
        d = float(np.hypot(x[k2] - x[k], y[k2] - y[k]))
        if abs(delta) > 12:
            events.append({'kind': 'turn', 't0': float(tt[k]),
                           't1': float(tt[k2]), 'delta_deg': round(delta, 1),
                           'dist': round(d, 2)})
        k = k2 + 1
    return events


# ---------------------------------------------------------------- obstacles
def parse_yields(entries):
    out = []
    for e in entries:
        m = YIELD_RE.match(e['msg'])
        if m:
            box = m.group(3)
            coords = re.search(r'\(([-+\d.]+),([-+\d.]+)\)', box)
            out.append({
                'ts': e['ts'], 'action': m.group(1), 'reason': m.group(2),
                'box': [float(coords.group(1)), float(coords.group(2))]
                if coords else None,
                'wait': float(m.group(4)), 'detail': m.group(5)})
            continue
        m = YIELD_DIAG_RE.match(e['msg'])
        if m:
            out.append({'ts': e['ts'], 'action': m.group(1),
                        'reason': m.group(2), 'box': None, 'wait': None,
                        'detail': 'diag'})
    return out


def world_to_map(events, odom, r, tvec):
    """Fit the Gazebo world frame onto the map frame via the robot's poses."""
    rows = series(events, 'model_states')
    tw, pw = [], []
    for w, d in rows:
        if 'robot' in d:
            tw.append(w)
            pw.append(d['robot'][:2])
    if len(tw) < 5:
        return np.eye(2), np.zeros(2), 0.0, None
    tw = np.array(tw)
    pw = np.array(pw)
    idx = nearest_index(odom['t'], tw)
    po = np.column_stack([odom['x'][idx], odom['y'][idx]])
    pm = transform(po, r, tvec)
    r2, t2 = fit_rigid(pw, pm)
    resid = float(np.max(np.linalg.norm(transform(pw, r2, t2) - pm, axis=1)))
    return r2, t2, yaw_from_r(r2), resid


def mover_tracks(events, r2, t2, yaw_off2):
    """Model poses (world frame) transformed into the map frame."""
    rows = series(events, 'model_states')
    tracks = {}
    for w, d in rows:
        for name, xyz in d.items():
            tracks.setdefault(name, []).append((w, xyz[0], xyz[1], xyz[2]))
    out = {}
    for name, items in tracks.items():
        arr = np.array(items)
        pts = transform(arr[:, 1:3], r2, t2)
        out[name] = {'t': arr[:, 0], 'x': pts[:, 0], 'y': pts[:, 1],
                     'yaw': arr[:, 3] + yaw_off2}
    return out


def encounters(movers, odom, r, tvec, yaw_off, threshold=1.6):
    out = []
    t = odom['t']
    px = odom['x'] * r[0, 0] + odom['y'] * r[1, 0] + tvec[0]
    py = odom['x'] * r[0, 1] + odom['y'] * r[1, 1] + tvec[1]
    yaw = odom['yaw'] + yaw_off
    for name, tr in movers.items():
        if not name.startswith('mover') or len(tr['t']) < 3:
            continue
        mx = np.interp(t, tr['t'], tr['x'])
        my = np.interp(t, tr['t'], tr['y'])
        mvx = np.interp(t, tr['t'], np.gradient(tr['x'], tr['t']))
        mvy = np.interp(t, tr['t'], np.gradient(tr['y'], tr['t']))
        d = np.hypot(px - mx, py - my)
        idx = np.flatnonzero(d < threshold)
        if len(idx) == 0:
            continue
        splits = np.flatnonzero(np.diff(t[idx]) > 2.0)
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, len(idx) - 1]
        for a, b in zip(starts, ends):
            k, k2 = int(idx[a]), int(idx[b])
            imin = int(np.argmin(d[k:k2 + 1])) + k
            bearing = math.degrees(math.atan2(
                my[imin] - py[imin], mx[imin] - px[imin]) - yaw[imin])
            bearing = (bearing + 180) % 360 - 180
            out.append({
                'mover': name, 't0': float(t[k]), 't1': float(t[k2]),
                'dur': round(float(t[k2] - t[k]), 2),
                'min_dist': round(float(d[imin]), 3),
                'min_t': float(t[imin]),
                'bearing_deg': round(bearing, 1),
                'robot_speed': round(float(odom['vx'][imin]), 2),
                'mover_speed': round(float(np.hypot(mvx[imin], mvy[imin])), 2),
            })
    out.sort(key=lambda e: e['t0'])
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round_dir')
    ap.add_argument('--map', default=(
        '/home/xu/intelligence_algorithm-main/src/yzbot/bot_navigation/'
        'maps/room_from_world.yaml'))
    args = ap.parse_args()

    rdir = Path(args.round_dir)
    entries = parse_log(rdir / 'stack.log')
    events = load_events(rdir / 'events.jsonl')

    terminal_ts = max(
        [e['ts'] for e in entries
         if STATE_RE.search(e['msg']) and
         STATE_RE.search(e['msg']).group(1) in TERMINAL] or [entries[-1]['ts']])
    state_seq = [STATE_RE.match(e['msg']).group(1)
                 for e in entries if STATE_RE.match(e['msg'])]

    cmap = MapClearance(args.map)
    odom = odom_arrays(events)
    if odom is None:
        raise SystemExit('no odom in events')

    r, tvec, resid = align_to_map(odom, events)
    ow = odom_to_world(events, odom)
    if ow is not None and ow['residual_m'] < 0.35:
        # Gazebo ground truth: the world frame is the map frame, so this is the
        # most accurate odom -> map transform available.
        r, tvec = ow['r'], ow['t']
        transform_source = 'gazebo_link_states'
    else:
        transform_source = 'amcl'
    yaw_off = yaw_from_r(r)

    objects = build_objects(entries, terminal_ts)
    objects = nav_phases(objects, odom, r, tvec, yaw_off, cmap)
    objects = vision_stats(objects, events)
    states = build_states(entries)
    exec_states = build_exec_states(entries)
    legs, speed = motion_legs(odom, r, tvec, yaw_off)
    leg_clearance(legs, cmap)

    # Attribute legs to the object and the executor state active at mid-leg.
    for leg in legs:
        tm = 0.5 * (leg['t0'] + leg['t1'])
        leg['object'] = None
        for o in objects:
            if o['start_ts'] <= tm < o['end_ts']:
                leg['object'] = o['index']
                break
        st = [x for x in states if x['t0'] <= tm < x['t1']]
        leg['state'] = st[0]['state'] if st else None
        leg['phase'] = PHASE_OF_STATE.get(leg['state'], 'other')
        leg['turns'] = turn_profile(odom, r, yaw_off, leg)
    for leg in legs:
        leg.pop('_pts', None)

    # Per-object wall-clock phase totals from the state machine.
    for o in objects:
        acc = {}
        for s in states:
            lo = max(s['t0'], o['start_ts'])
            hi = min(s['t1'], o['end_ts'])
            if hi > lo:
                ph = PHASE_OF_STATE.get(s['state'], 'other')
                acc[ph] = acc.get(ph, 0.0) + (hi - lo)
        o['state_phases'] = {k: round(v, 2) for k, v in acc.items()}

    yields = parse_yields(entries)
    for y in yields:
        y['object'] = next((o['index'] for o in objects
                            if o['start_ts'] <= y['ts'] < o['end_ts']), None)

    speeds = []
    for e in entries:
        m = SPEED_RE.match(e['msg'])
        if m:
            speeds.append({'ts': e['ts'], 'reason': m.group(1),
                           'from': float(m.group(2)), 'to': float(m.group(3)),
                           'where': m.group(4).strip()})

    r2, t2, yaw_off2, w_resid = world_to_map(events, odom, r, tvec)
    movers = mover_tracks(events, r2, t2, yaw_off2)
    encs = encounters(movers, odom, r, tvec, yaw_off)
    for enc in encs:
        enc['object'] = next((o['index'] for o in objects
                              if o['start_ts'] <= enc['min_t'] < o['end_ts']),
                             None)
        enc['yields'] = [
            {'action': y['action'], 'reason': y['reason'],
             'wait': y['wait'], 'detail': y['detail']}
            for y in yields if enc['t0'] - 2.0 <= y['ts'] <= enc['t1'] + 2.0]

    plans = [{'ts': w, 'n': d['n']} for w, d in series(events, 'plan')]
    for p in plans:
        p['object'] = next((o['index'] for o in objects
                            if o['start_ts'] <= p['ts'] < o['end_ts']), None)

    press_ts = next((e['ts'] for e in entries
                     if STATE_RE.match(e['msg'])
                     and STATE_RE.match(e['msg']).group(1) == 'PARSING_TASK'),
                    None)
    path_len = float(np.hypot(np.diff(odom['x']), np.diff(odom['y'])).sum())

    result = {
        'round': rdir.name,
        'map_fit': {'transform': transform_source,
                    'odom_to_map_r': np.round(r, 6).tolist(),
                    'odom_to_map_t': np.round(tvec, 4).tolist(),
                    'odom_world_residual_m': None if ow is None else
                    round(ow['residual_m'], 3),
                    'odom_world_p95_m': None if ow is None else
                    round(ow['p95_m'], 3),
                    'amcl_odom_residual_m': resid,
                    'amcl_yaw_offset_deg': round(math.degrees(yaw_off), 2),
                    'world_map_residual_m': w_resid,
                    'world_yaw_offset_deg': round(math.degrees(yaw_off2), 2)},
        'state_sequence': state_seq,
        'objects': objects,
        'legs': legs,
        'yields': yields,
        'encounters': encs,
        'speed_limits': speeds,
        'plans': plans,
        'exec_states': exec_states,
        'totals': {
            'press_ts': press_ts,
            'terminal_ts': terminal_ts,
            'wall_s': round(terminal_ts - press_ts, 1) if press_ts else None,
            'objects': len(objects),
            'path_length_m': round(path_len, 2),
            'legs': len(legs),
            'yields': len(yields),
            'yield_actions': {a: sum(1 for y in yields if y['action'] == a)
                              for a in sorted({y['action'] for y in yields})},
            'encounters': len(encs),
            'min_dynamic_dist': round(min((e['min_dist'] for e in encs),
                                          default=None), 3)
            if encs else None,
            'min_static_clearance': round(min(
                (l['min_clearance'] for l in legs), default=99.0), 3),
            'timing_lines': sum(len(o['timings']) for o in objects),
            'nav_s': round(sum(o['nav_s'] for o in objects), 2),
            'vision_s': round(sum(o['vision_s'] for o in objects), 2),
            'manip_s': round(sum(o['manip_s'] for o in objects), 2),
        },
    }
    (rdir / 'analysis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f"round {rdir.name}: {len(objects)} objects, wall "
          f"{result['totals']['wall_s']} s, path "
          f"{result['totals']['path_length_m']} m, legs {len(legs)}, "
          f"yields {len(yields)}, encounters {len(encs)}")
    for o in objects:
        print(f"  #{o['index']} {o['cube']}->{o['zone']} wall {o['wall_s']}s "
              f"nav {o['nav_s']}s vision {o['vision_s']}s "
              f"manip {o['manip_s']}s misc {o['misc_s']}s")
    for leg in legs:
        print(f"  leg {leg['index']:2d} {leg['dur']:5.1f}s {leg['dist']:5.2f}m "
              f"{leg['heading_start']:6.1f}->{leg['heading_end']:6.1f}deg "
              f"clear {leg['min_clearance']:.2f}m dev {leg['max_dev_m']:.2f}m "
              f"{leg['phase']:6s} obj{leg['object']}")
    for enc in encs:
        print(f"  encounter {enc['mover']} {enc['dur']:.1f}s dmin "
              f"{enc['min_dist']:.2f}m bearing {enc['bearing_deg']:+.0f}deg "
              f"yields {[y['action'] + '/' + y['reason'] for y in enc['yields']]}")


if __name__ == '__main__':
    main()

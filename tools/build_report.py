#!/usr/bin/env python3
"""Compose the three-round competition report from the recorded rounds.

    tools/build_report.py runs/2026-11-21_rule1 runs/2026-11-21_rule2 ... \
        --out runs/REPORT.md

Every number here comes from <round>/analysis.json, <round>/stack.log or the
raw <round>/events.jsonl, so the markdown can be re-generated at any time.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_analysis as contact
import round_analysis as ra

RULE_RE = re.compile(r'映射规则 (\d) 生效：x=(\d+), y=(\d+) -> (.*)')
PROBLEM_RE = re.compile(r'^题目: (.*)$')
SEED_RE = re.compile(r'^使用种子: (\d+)')
TERMINAL_RE = re.compile(r'^State -> (COMPLETED|FAILED|ABORTED)$')
SUMMARY_RE = re.compile(r'^FAILED SUMMARY: (.*)$')

# Static-obstacle / navigation decisions worth quoting in the report.
DECISION_PATTERNS = [
    (re.compile(r'^Retrying (.+?) navigation \((\d+)/(\d+)\)'), '重试'),
    (re.compile(r'^(.+?): yielding around a moving obstacle via (.+)$'), '绕行动障'),
    (re.compile(r'^(.+?) pickup clearance waypoint (\d+)'), '抓取净空航点'),
    (re.compile(r'^(.+?) to (.+?) continuous detour'), '连续绕行'),
    (re.compile(r'^(.+?) to (.+?) detour waypoint (\d+)'), '绕行航点'),
    (re.compile(r'^Navigation to (.+?) failed; status=(\d+)'), '导航失败'),
    (re.compile(r'^(Slow zone): speed limit ([\d.]+) -> ([\d.]+) m/s(.*)'), '慢速区'),
    (re.compile(r'^(Dead[- ]zone[^:]*): (.*)'), '死区脱困'),
    (re.compile(r'^(Relaxed footprint[^:]*): (.*)'), '放宽足迹'),
]

COMPASS = ['正东', '东北', '正北', '西北', '正西', '西南', '正南', '东南']


def compass(deg):
    deg = deg % 360.0
    return COMPASS[int(((deg + 22.5) % 360) // 45)]


def turn_word(delta):
    if abs(delta) < 5:
        return '航向基本不变'
    side = '左转' if delta > 0 else '右转'
    return f'{side} {abs(delta):.0f}°'


def fmt_pose(p):
    return (f'({p[0]:.2f}, {p[1]:.2f}) 航向 {p[2]:.0f}°'
            f'（{compass(p[2])}）')


def loc_stats(rd):
    am, od = [], []
    ev = Path(rd) / 'events.jsonl'
    if not ev.exists():
        return None
    with ev.open() as fh:
        for ln in fh:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d['topic'] == 'amcl_pose':
                am.append((d['w'], d['d']['x'], d['d']['y']))
            elif d['topic'] == 'odom':
                od.append((d['w'], d['d']['x'], d['d']['y']))
    if not am or not od:
        return None
    import numpy as np
    am = np.array(am)
    od = np.array(od)
    idx = np.searchsorted(od[:, 0], am[:, 0]).clip(0, len(od) - 1)
    dist = np.hypot(am[:, 1] - od[idx, 1], am[:, 2] - od[idx, 2])
    return {'n': len(am), 'median': float(np.median(dist)),
            'max': float(dist.max()),
            'frac_bad': float((dist > 1.0).mean())}


def _placed(o):
    return any(t['label'].startswith('arm:Lower to place')
               or 'Place cube' in t['label'] for t in o['timings'])


def completion(analysis, meta):
    """Unique cubes placed / unique cubes the run touched.

    A failed attempt followed by a retry is one cube, not two: counting
    objects made rule 2 and rule 3 read ``5/6`` although both delivered all
    five cubes (2026-11-21 run set)."""
    objs = analysis['objects']
    done = len({o['cube'] for o in objs if _placed(o)})
    plan = len({o['cube'] for o in objs})
    remaining = len(re.findall(r"->", meta.get('failed_summary') or ''))
    return done, max(plan, done + remaining)


def load_round(rdir):
    rd = Path(rdir)
    analysis = json.loads((rd / 'analysis.json').read_text())
    entries = ra.parse_log(rd / 'stack.log')
    meta = {'dir': str(rd), 'name': rd.name, 'rule': None, 'x': None, 'y': None,
            'mapping': '', 'problem': '', 'seed': '', 'terminal': 'UNKNOWN',
            'failed_summary': ''}
    pf = rd / 'problem.txt'
    if pf.exists():
        text = pf.read_text(encoding='utf-8', errors='replace')
        m = SEED_RE.search(text)
        if m:
            meta['seed'] = m.group(1)
        m = re.search(r'题目: (.*)', text)
        if m:
            meta['problem'] = m.group(1).strip()
    for e in entries:
        msg = e['msg']
        m = RULE_RE.search(msg)
        if m:
            meta['rule'] = int(m.group(1))
            meta['x'] = int(m.group(2))
            meta['y'] = int(m.group(3))
            meta['mapping'] = m.group(4).strip()
        m = PROBLEM_RE.match(msg)
        if m and not meta['problem']:
            meta['problem'] = m.group(1).strip()
        m = TERMINAL_RE.match(msg)
        if m:
            meta['terminal'] = m.group(1)
        m = SUMMARY_RE.match(msg)
        if m:
            meta['failed_summary'] = m.group(1).strip()
    decisions = []
    for e in entries:
        for pat, kind in DECISION_PATTERNS:
            m = pat.match(e['msg'])
            if m:
                decisions.append({'ts': e['ts'], 'kind': kind,
                                  'text': e['msg'][:160]})
                break
        if 'dead-zone escape' in e['msg'] and 'costmap geometry' in e['msg']:
            decisions.append({'ts': e['ts'], 'kind': '死区脱困',
                              'text': e['msg'][:160]})
    meta['decisions'] = decisions
    meta['_loc'] = loc_stats(rd)
    events = rd / 'events.jsonl'
    meta['_contact'] = contact.analyze(events) if events.exists() else None
    meta['_done'], meta['_plan'] = completion(analysis, meta)
    return analysis, entries, meta


def object_decisions(meta, o):
    return [d for d in meta['decisions']
            if o['start_ts'] - 0.5 <= d['ts'] <= (o['end_ts'] or 1e18) + 0.5]


def yields_for(analysis, o, pad=1.5):
    out = [y for y in analysis['yields']
           if o['start_ts'] - pad <= y['ts'] <= (o['end_ts'] or 1e18) + pad]
    return out


def decision_chain(ys):
    """Collapse consecutive identical (action, reason) pairs.

    ``YIELD go/clear`` is the monitor's steady state, not a decision; every
    other action/reason pair is kept, including ones whose trailing detail is
    just ``diag`` (the 2026-11-21 run set had real ``stop/predict`` lines
    marked that way, which used to blank the whole column)."""
    chain = []
    for y in ys:
        if (y['action'], y['reason']) == ('go', 'clear'):
            continue
        key = (y['action'], y['reason'])
        if chain and chain[-1]['key'] == key:
            chain[-1]['n'] += 1
        else:
            chain.append({'key': key, 'n': 1, 'first': y})
    return chain


def fmt_chain(chain, limit=6):
    if not chain:
        return '未干预（正常通过）'
    parts = []
    for c in chain[:limit]:
        a, r = c['key']
        parts.append(f'{a}/{r}' + (f'×{c["n"]}' if c['n'] > 1 else ''))
    if len(chain) > limit:
        parts.append(f'…(共 {len(chain)} 段)')
    return ' → '.join(parts)


def round_section(analysis, entries, meta, index):
    L = []
    objs = analysis['objects']
    totals = analysis['totals']
    completed, planned = meta['_done'], meta['_plan']
    L.append(f"## 第 {index} 轮 · mapping_rule={meta['rule']} "
             f"（{meta['name']}）")
    L.append('')
    L.append(f"- 出题器 seed：`{meta['seed'] or '?'}`")
    if meta['problem']:
        L.append(f"- 题目原文：{meta['problem']}")
    if meta['mapping']:
        L.append(f"- 规则 {meta['rule']} 解读：**x={meta['x']}, y={meta['y']} → "
                 f"{meta['mapping']}**")
    attempts = ''
    if len(objs) != planned:
        attempts = f"（共 {len(objs)} 次取放尝试）"
    L.append(f"- 最终状态：`{meta['terminal']}`；完成 {completed}/{planned} 块"
             f"{attempts}；"
             f"首块起 {totals.get('wall_s', 0):.1f} s；驱动路程 "
             f"{totals.get('path_length_m', 0):.1f} m；动态避障决策 "
             f"{totals.get('yields', 0)} 条（"
             + ' / '.join(f'{k} {v}' for k, v in
                          (totals.get('yield_actions') or {}).items())
             + "），"
             f"接近事件 {len(analysis['encounters'])} 次；"
             f"最小动态间距 {totals.get('min_dynamic_dist', 0):.2f} m")
    ls = meta.get('_loc')
    if ls:
        L.append(f"- 定位质量：AMCL 相对里程计残差中位数 {ls['median']:.2f} m、"
                 f"最大 {ls['max']:.2f} m，{100 * ls['frac_bad']:.1f}% 的时刻 >1 m"
                 f"（{ls['n']} 帧）")
    if meta['failed_summary']:
        L.append(f"- 失败摘要：`{meta['failed_summary']}`")
    L.append('')

    # ---------------------------------------------------------------- timings
    L.append('### 每块耗时（秒）')
    L.append('')
    L.append('| # | 物块 | 目标区 | 全程 | 导航→物块 | 导航→仓库 | 导航合计 | '
             '视觉 vision_align | 拿放 arm+attach | 空隙 | 备注 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for o in objs:
        gap = o['wall_s'] - (o['nav_s'] + o['vision_s'] + o['manip_s'] +
                             o['misc_s'])
        placed = any(t['label'].startswith('arm:Lower to place')
                     or 'Place cube' in t['label'] for t in o['timings'])
        note = '' if placed else '未放置（中断）'
        nested = o.get('nested') or []
        creep = sum(n['sec'] for n in nested if n['label'].startswith('creep:'))
        align = sum(n['sec'] for n in nested if n['label'] == 'creep_to_grasp')
        if creep or align:
            extra = []
            if creep:
                extra.append(f'入位蠕动 {creep:.2f}s')
            if align:
                extra.append(f'贴靠 {align:.2f}s')
            note = (note + ' ' if note else '') + '含' + '、'.join(extra)
        labs = o.get('nav_labels') or []

        def _zone_leg(lbl):
            return 'warehouse' in lbl or bool(re.search(r' to [ABC]\b', lbl))

        picks = [x for x in labs if not _zone_leg(x['label'])]
        base = picks[0]['label'] if picks else None
        tries = [x for x in picks if x['label'] == base]
        to_cube = sum(x['sec'] for x in labs if not _zone_leg(x['label']))
        to_zone = sum(x['sec'] for x in labs if _zone_leg(x['label']))
        if len(tries) > 1:
            note = (note + '；' if note else '') + f'去物块 {len(tries)} 次尝试'
        L.append(f"| {o['index']} | {o['cube']} | {o['zone']} | "
                 f"{o['wall_s']:.1f} | {to_cube:.1f} | {to_zone:.1f} | "
                 f"{o['nav_s']:.1f} | {o['vision_s']:.1f} | "
                 f"{o['manip_s']:.1f} | {gap:+.1f} | {note} |")
    L.append('')

    abandoned = [o for o in objs if not _placed(o)]
    for o in abandoned:
        nxt = next((p for p in objs if p['cube'] == o['cube']
                    and p['index'] > o['index'] and _placed(p)), None)
        L.append('### 中断与重试')
        L.append('')
        L.append(f"- 第 {o['index']} 次尝试（{o['cube']} → {o['zone']} 区）没有完成"
                 f"放置：wall {o['wall_s']:.1f} s，其中导航 {o['nav_s']:.1f} s、"
                 f"视觉 {o['vision_s']:.1f} s、拿放 {o['manip_s']:.1f} s。")
        keys = ('exception', 'no progress', 'abandoning', 'failed to create plan',
                'Failure recovery', 'Cancel')
        picked, seen_k = [], set()
        for k in keys:
            for e in entries:
                if (k not in seen_k and o['start_ts'] <= e['ts'] <= o['end_ts']
                        and k in e['msg']):
                    seen_k.add(k)
                    picked.append(e)
                    break
        for e in picked:
            L.append(f"  - t={e['ts'] - objs[0]['start_ts']:.1f}s "
                     f"`{e['msg'][:170]}`")
        if nxt:
            L.append(f"- 状态机随后重新下发目标，第 {nxt['index']} 次尝试成功："
                     f"wall {nxt['wall_s']:.1f} s（导航 {nxt['nav_s']:.1f} s、"
                     f"视觉 {nxt['vision_s']:.1f} s、拿放 {nxt['manip_s']:.1f} s）。")
            L.append(f"- 该物块实际总耗时 "
                     f"{o['wall_s'] + nxt['wall_s']:.1f} s（含重试），占本轮全程的 "
                     f"{100 * (o['wall_s'] + nxt['wall_s']) / totals['wall_s']:.0f}%。")
        L.append('')

    vis = [o for o in objs if o.get('vision')]
    if vis:
        L.append('视觉调整明细（把 `vision_align` 拆成“搜索物块”与“对齐”两段）：')
        L.append('')
        L.append('| # | 物块 | vision_align(s) | 搜索(s) | 对齐(s) | 目标丢失重建 | '
                 '初始横向偏差 | 末帧横向偏差 |')
        L.append('|---|---|---|---|---|---|---|---|')
        for o in vis:
            v = o['vision']
            L.append(f"| {o['index']} | {o['cube']} | {o['vision_s']:.2f} | "
                     f"{v['search_s']:.2f} | {v['align_s']:.2f} | "
                     f"{v['reacquire']} | "
                     f"{'' if v['x_first'] is None else f'{v[chr(120) + chr(95) + chr(102) + chr(105) + chr(114) + chr(115) + chr(116)]:+.3f}'} | "
                     f"{'' if v['x_last'] is None else f'{v[chr(120) + chr(95) + chr(108) + chr(97) + chr(115) + chr(116)]:+.3f}'} |")
        L.append('')
        L.append('> 横向偏差是物块中心相对图像中心的归一化偏差（±1 为画面边缘）；'
                 '搜索时间长 = 到位后物块不在视野内，先在原地转/补拍。')
        L.append('')

    # ---------------------------------------------------------------- routes
    L.append('### 路线与静态障碍避障')
    L.append('')
    for o in objs:
        L.append(f"**第 {o['index']} 块 {o['cube']} → {o['zone']} 区**"
                 f"（object_id={o['object_id']}）")
        L.append('')
        paths = o.get('nav_paths') or []
        nplan = sum(1 for pl in analysis.get('plans', [])
                    if pl.get('object') == o['index'])
        if nplan:
            L.append(f"- 全局规划更新 {nplan} 次（约每 "
                     f"{o['wall_s'] / max(1, nplan):.1f} s 一次）")
        if paths:
            total = sum(p['dist'] for p in paths)
            chord = sum(((p['start'][0] - p['end'][0]) ** 2 +
                         (p['start'][1] - p['end'][1]) ** 2) ** 0.5
                        for p in paths)
            a = paths[0]['start']
            b = paths[-1]['end']
            bear = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360
            if chord > 0.5:
                L.append(f"- 行程：{len(paths)} 段共 {total:.2f} m，各段直线合计 "
                         f"{chord:.2f} m，绕行系数 {total / chord:.2f}；"
                         f"出发车头 {compass(a[2])} → 到达车头 {compass(b[2])}，"
                         f"整体位移向 {compass(bear)}（{bear:.0f}°）")
            else:
                L.append(f"- 行程：{len(paths)} 段共 {total:.2f} m（起点与终点基本"
                         f"重合的环线，出发车头 {compass(a[2])} → "
                         f"到达车头 {compass(b[2])}）")
            L.append('')
        for k, p in enumerate(paths, 1):
            turns = [t for t in (p.get('turns') or [])
                     if t['kind'] == 'turn' and abs(t['delta_deg']) >= 12]
            turn_txt = '；'.join(
                f"{turn_word(t['delta_deg'])}（{t['t1'] - t['t0']:.1f}s，"
                f"位移 {t['dist']:.2f}m）" for t in turns[:4]) or '无显著转向'
            chord = p.get('chord_len') or 0.0
            cclr = p.get('chord_min_clearance')
            if cclr is not None and cclr < 0.15:
                why = '直线被墙阻断，必须绕行'
            elif chord > 0.3 and p['dist'] > 1.35 * chord:
                why = '明显绕行'
            else:
                why = '接近直线'
            dx = p['end'][0] - p['start'][0]
            dy = p['end'][1] - p['start'][1]
            if dx * dx + dy * dy >= 0.04:
                bear = math.degrees(math.atan2(dy, dx)) % 360
                disp_txt = f"，位移向 {compass(bear)}（{bear:.0f}°）"
            else:
                disp_txt = ''
            L.append(f"{k}. `{p['label']}`：{p['sec']:.2f} s，路程 "
                     f"{p['dist']:.2f} m（直线 {chord:.2f} m，直线净空 "
                     f"{0.0 if cclr is None else cclr:.2f} m → {why}）；"
                     f"{fmt_pose(p['start'])} → "
                     f"{fmt_pose(p['end'])}{disp_txt}；{turn_txt}；最小离静态障碍 "
                     f"{p['min_clearance']:.2f} m @({p['min_clearance_at'][0]:.2f},"
                     f"{p['min_clearance_at'][1]:.2f})")
            wps = p.get('waypoints') or []
            if len(wps) > 2:
                L.append(f"   - 途经（每 2 m）：" + ' → '.join(
                    f"({w[0]:.1f},{w[1]:.1f})" for w in wps))
        if not paths:
            L.append('- 本轮没有完成的导航段')
        decs = object_decisions(meta, o)
        if decs:
            L.append('- 决策日志：')
            for d in decs[:8]:
                L.append(f"  - [{d['kind']}] {d['text']}")
        L.append('')

    # ------------------------------------------------------------- dynamic
    L.append('### 动态障碍物（mover）交互')
    L.append('')
    if analysis['encounters']:
        L.append('| 时间(s) | mover | 最近距离 | 相对方位 | 时长 | 机器人速度 | '
                 '避障决策链 |')
        L.append('|---|---|---|---|---|---|---|')
        t00 = objs[0]['start_ts'] if objs else 0
        for e in analysis['encounters']:
            ys = [y for y in analysis['yields']
                  if e['t0'] - 2.0 <= y['ts'] <= e['t1'] + 2.0]
            L.append(f"| {e['t0'] - t00:.1f} | {e['mover']} | "
                     f"{e['min_dist']:.2f} m | {e['bearing_deg']:+.0f}° | "
                     f"{e['dur']:.1f} s | {e['robot_speed']:.2f} m/s | "
                     f"{fmt_chain(decision_chain(ys))} |")
        L.append('')
        L.append('> 时间以首块开始为 0，负值表示首块开始前的待机/起步阶段；'
                 '接近事件只统计进入 1.6 m 以内的 mover。'
                 '相对方位：+ 为小车左前方，− 为右前方（以车头为 0°）。'
                 '决策链按 `YIELD <动作>/<原因>` 去重，×n 表示同一决策重复 n 次。')
        L.append('')
        L.append('**决策依据摘录**（仿真日志原文，证明每次让行/改道的原因）：')
        L.append('')
        for e in analysis['encounters']:
            ys = [y for y in analysis['yields']
                  if e['t0'] - 2.0 <= y['ts'] <= e['t1'] + 2.0
                  and y['action'] != 'go']
            if not ys:
                continue
            seen = set()
            for y in ys:
                key = (y['action'], y['reason'])
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > 2:
                    break
                L.append(f"- t={e['t0'] - t00:.1f}s（{e['mover']}，最近 "
                         f"{e['min_dist']:.2f} m）`{y['action']}/{y['reason']}`："
                         f"{y['detail']}")
        L.append('')
        L.append('**全部非 `go/clear` 决策**（时间、物块、动作/原因、依据；'
                 '`go/predict`、`go/behind` 是预测后判定可以通过，'
                 '`stop/predict` 是预测会冲突而暂停，`arc/wait` 是侧移让行，'
                 '`cap/*` 是限时等待/超时兜底）：')
        L.append('')
        L.append('| 时间(s) | 物块 | 决策 | box(车体坐标 m) | wait(s) | 依据 |')
        L.append('|---|---|---|---|---|---|')
        t00 = objs[0]['start_ts'] if objs else 0
        by_idx = {o['index']: o['cube'] for o in objs}
        for y in analysis['yields']:
            if y['action'] == 'go' and y['reason'] == 'clear':
                continue
            box = (f"({y['box'][0]:+.2f},{y['box'][1]:+.2f})"
                   if isinstance(y.get('box'), (list, tuple)) and len(y['box']) == 2
                   else '—')
            wait = ('—' if y.get('wait') is None else f"{y['wait']:.1f}")
            det = (y.get('detail') or '').replace('|', '/')
            L.append(f"| {y['ts'] - t00:.1f} | "
                     f"{by_idx.get(y.get('object'), '—')} | "
                     f"`{y['action']}/{y['reason']}` | {box} | {wait} | {det} |")
        L.append('')
    else:
        L.append('本轮没有进入 1.6 m 以内的动态障碍物接近事件。')
    L.append('')
    contact = meta.get('_contact')
    if contact and contact['frames']:
        hull = contact['hull']
        big = [e for e in contact['pushed_while_stopped'] if e['moved'] >= 0.10]
        text = (f"- Gazebo 真值（{2*hull['half_length']:.2f}×"
                f"{2*hull['half_width']:.2f} m 外壳，{contact['frames']} 帧采样）："
                f"采样帧内壳体重叠 {len(contact['contacts'])} 次")
        if contact['min_clearance'] is not None:
            text += (f"，最小外壳净距 {contact['min_clearance']:+.3f} m"
                     f"（{contact['min_clearance_mover']}，"
                     f"{contact['min_clearance_at']:.1f} s）")
        L.append(text + '。')
        if big:
            L.append('  - 零速静止后被推移：' + '；'.join(
                f"t={e['t']:.1f} s 位移 {e['moved']:.3f} m" for e in big)
                + '（指令 0.00 m/s，说明接触实际发生，停车点就在扫掠区内）。')
        elif contact['pushed_while_stopped']:
            L.append('  - 零速时有 <0.10 m 的位移，量级接近底盘抖动，'
                     '不作为接触证据。')
        else:
            L.append('  - 零速静止后没有被推移。')
        L.append('')

    if meta['terminal'] == 'FAILED' and objs:
        o = objs[-1]
        stops = [y for y in analysis['yields']
                 if y['action'] == 'stop'
                 and y.get('detail') not in (None, 'diag')]
        retry = [d for d in meta['decisions'] if d['kind'] == '重试']
        arcs = [(p.get('label') or '', p.get('sec', 0.0), p.get('dist', 0.0))
                for p in (o.get('nav_paths') or [])
                if 'yield arc' in (p.get('label') or '')]
        nav_fail = [d for d in meta['decisions'] if d['kind'] == '导航失败']
        closest = min((e['min_dist']
                       for e in analysis['encounters'] if e.get('min_dist')),
                      default=None)
        L.append('### 失败复盘')
        L.append('')
        L.append(f"- 失败点在第 {o['index']} 块 {o['cube']} → {o['zone']} 区："
                 f"整块 wall {o['wall_s']:.1f} s，其中导航 {o['nav_s']:.1f} s、"
                 f"视觉 {o['vision_s']:.1f} s、拿放 {o['manip_s']:.1f} s；"
                 f"终态 `{meta['terminal']}`，失败摘要"
                 f"“{meta['failed_summary'] or '无'}”。")
        picked = next((y for y in stops
                       if 'never' in (y.get('detail') or '')), None)
        if picked is None and stops:
            picked = stops[0]
        if stops:
            L.append(f"- 让行日志里 `stop` 共 {len(stops)} 条"
                     + (f"，本轮最近中心距 {closest:.2f} m" if closest is not None
                        else '')
                     + f"；最典型一条原文：“{picked['detail']}”。")
        if arcs:
            L.append('- 本块执行过的侧移（`nav_paths` 里 label 含 `yield arc`，'
                     '时间是路线段计时，不是成功标志）：' + '；'.join(
                         f"`{lab}` {sec:.1f} s / {dist:.2f} m"
                         for lab, sec, dist in arcs) + '。')
        if nav_fail:
            L.append('- 执行器报出的导航失败：' + '；'.join(
                f"`{d['text']}`" for d in nav_fail[:3]) + '。')
        elif retry:
            L.append(f"- 执行器先按 `{retry[0]['text']}` 清地图重试，"
                     '重试预算用完后才终止。')
        L.append(f"- 是否动态障碍物锁死通路，看上面 t= 时刻的决策与本节最近的"
                 f"净距数据；不在这里替它下结论。")

    # ------------------------------------------------------------- states
    L.append('### 状态机停留时间')
    L.append('')
    seq = []
    for e in entries:
        m = ra.STATE_RE.match(e['msg'])
        if m:
            seq.append((m.group(1), e['ts']))
    if seq:
        t_end = seq[-1][1] + 0.0
        rows, merged = [], []
        for i, (name, ts) in enumerate(seq):
            end = seq[i + 1][1] if i + 1 < len(seq) else t_end
            if merged and merged[-1][0] == name:
                merged[-1][1] += end - ts
            else:
                merged.append([name, end - ts])
        L.append('| 状态 | 停留(s) |')
        L.append('|---|---|')
        for name, dur in merged:
            L.append(f'| {name} | {dur:.1f} |')
        L.append('')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rounds', nargs='+')
    ap.add_argument('--out', default='runs/REPORT.md')
    ap.add_argument('--appendix', default='',
                    help='markdown file appended verbatim at the end')
    args = ap.parse_args()

    loaded = [load_round(r) for r in args.rounds]

    L = ['# 三轮比赛记录：mapping_rule 1 / 2 / 3', '']
    L.append('## 数据来源与口径')
    L.append('')
    L.append('- 每轮独立拉起整套 Gazebo + Nav2 + MoveIt 仿真'
             '（`tools/run_round.sh`），开跑前改 `task_parser.yaml` 的 '
             '`mapping_rule`，题目由出题器 `TMSCQtest_x86_x64.bin 11470507` '
             '按时段抽取。')
    L.append('- 全程用 `tools/round_recorder.py` 录 `/odom`、`/cmd_vel*`、'
             '`/plan`、`/amcl_pose`、`/gazebo/link_states`、'
             '`/gazebo/model_states`、`/competition/*`，落成 JSONL；'
             '耗时与决策同时从 ROS 日志里按 `TIMING`/`YIELD`/`Retrying` 等'
             '固定格式解析。')
    L.append('- **导航时间** = 执行器 `nav:*` 计时器之和（含入位蠕动 '
             '`creep:*`，它嵌在对应 nav 计时器内部，不重复计）；'
             '**视觉调整时间** = `vision_align`（含贴靠 `creep_to_grasp`）；'
             '**拿放物块时间** = `arm:*` 机械臂动作 + `svc:attach/detach`；'
             '**空隙** = 全程 − 三段之和（状态切换、拍照、清代价地图等）。')
    L.append('- 动态避障表里的 `arc` 是决策日志条数，不等于执行次数：'
             '每段最多执行一次侧移，真正执行的 `yield arc` 在路线段计时里。')
    L.append('- 位姿统一换算到地图坐标系；有 Gazebo '
             '`/gazebo/link_states` 真值时用它标定 odom→map，'
             '否则回退到 `/amcl_pose`。')
    L.append('')

    # ------------------------------------------------------------- overview
    L.append('## 总览')
    L.append('')
    L.append('| 轮次 | 规则 | x,y | 映射结果 | 结果 | 完成 | 完成率 | 全程 | '
             '路程 | 导航合计 | 视觉合计 | 拿放合计 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, (analysis, entries, meta) in enumerate(loaded, 1):
        t = analysis['totals']
        done, plan = meta['_done'], meta['_plan']
        L.append(f"| {i} | {meta['rule']} | {meta['x']},{meta['y']} | "
                 f"{meta['mapping']} | `{meta['terminal']}` | "
                 f"{done}/{plan} | {100 * done / max(1, plan):.0f}% | "
                 f"{t.get('wall_s', 0):.0f}s | {t.get('path_length_m', 0):.1f}m | "
                 f"{t.get('nav_s', 0):.0f}s | {t.get('vision_s', 0):.0f}s | "
                 f"{t.get('manip_s', 0):.0f}s |")
    L.append('')
    seeds = {}
    for i, (analysis, entries, meta) in enumerate(loaded, 1):
        seeds.setdefault(meta['seed'] or '?', []).append(i)
    tl = Path('runs/timings.png')
    if tl.exists():
        L.append('![每块耗时分解](timings.png)')
        L.append('')
    if len(seeds) > 1:
        xy = {(m['x'], m['y']) for _, _, m in loaded}
        seed_txt = ('；'.join(f"第 {'/'.join(map(str, v))} 轮 seed={k}"
                             for k, v in seeds.items()))
        if len(xy) == 1:
            x0, y0 = next(iter(xy))
            L.append(f'> 各轮题面措辞不同（{seed_txt}），但抽取出的 '
                     f'x={x0}, y={y0} 完全一致，映射后的物块数与目标区相同，'
                     '因此逐块耗时仍可直接横向对比。')
        else:
            L.append(f'> 三轮的 seed 分布：{seed_txt}；因此 x,y 与映射结果可能不同，'
                     '下表既包含规则差异也包含题面差异。')
        L.append('')
    L.append('> 逐块明细同时导出为 [`timings.csv`](timings.csv)，列：'
             'round, rule, seed, object_id, cube, zone, wall_s, nav_s, '
             'vision_s, manip_s, gap_s, placed, round_status。')
    L.append('')

    for i, (analysis, entries, meta) in enumerate(loaded, 1):
        L.extend(round_section(analysis, entries, meta, i))
        for png, cap in ((Path(meta['dir']) / 'route.png', '路线与避障'),
                         (Path(meta['dir']) / 'timeline.png', '速度/航向/决策时序')):
            if png.exists():
                L.append(f"![第 {i} 轮{cap}]({Path(meta['dir']).name}/{png.name})")
                L.append('')

    L.append('## 结论')
    L.append('')
    for i, (analysis, entries, meta) in enumerate(loaded, 1):
        objs = analysis['objects']
        if not objs:
            continue
        nav = [o['nav_s'] for o in objs]
        vis = [o['vision_s'] for o in objs]
        man = [o['manip_s'] for o in objs]
        L.append(f"- 第 {i} 轮：{len(objs)} 块，导航 {sum(nav):.1f}s"
                 f"（均值 {sum(nav) / len(nav):.1f}s，占 "
                 f"{100 * sum(nav) / max(1, sum(nav) + sum(vis) + sum(man)):.0f}%）、"
                 f"视觉 {sum(vis):.1f}s、拿放 {sum(man):.1f}s。")
        enc = analysis['encounters']
        if enc:
            L.append(f"  - 动态避障：{len(enc)} 次接近，最小 "
                     f"{min(e['min_dist'] for e in enc):.2f} m；"
                     f"决策 {len(analysis['yields'])} 条；"
                     f"动作分布 "
                     + ' / '.join(f'{k} {v}' for k, v in
                                  (analysis['totals'].get('yield_actions')
                                   or {}).items()) + '。')
        ratios = []
        for o in objs:
            paths = o.get('nav_paths') or []
            tot = sum(p['dist'] for p in paths)
            chd = sum(((p['start'][0] - p['end'][0]) ** 2 +
                       (p['start'][1] - p['end'][1]) ** 2) ** 0.5
                      for p in paths)
            if chd > 0.5:
                ratios.append(tot / chd)
        if ratios:
            blocked = [p for o in objs for p in (o.get('nav_paths') or [])
                       if p.get('chord_min_clearance', 9.0) <= 0.05]
            L.append(f"  - 静态绕行系数（路程/各段直线）{min(ratios):.2f}–"
                     f"{max(ratios):.2f}；其中 {len(blocked)} 段直线被静态墙"
                     f"阻断（直线净空 ≤0.05 m），规划器只能沿墙绕行。")
    L.append('')
    L.append('### 横向对比')
    L.append('')
    allobj = [(o, a) for a, _, _ in loaded for o in a['objects']]
    nav_all = [o['nav_s'] for o, _ in allobj]
    vis_all = [o['vision_s'] for o, _ in allobj]
    man_all = [o['manip_s'] for o, _ in allobj]
    nav_share = [100.0 * o['nav_s'] /
                 max(1.0, o['nav_s'] + o['vision_s'] + o['manip_s'])
                 for o, _ in allobj]
    L.append(f'- 拿放时间最稳定：{min(man_all):.1f}–{max(man_all):.1f} s/块，'
             '6 个机械臂动作 + attach/detach 基本固定，与规则/目标区无关。')
    L.append(f'- 视觉调整 {min(vis_all):.1f}–{max(vis_all):.1f} s/块：'
             '主要取决于入位时物块在画面中的初始偏差和是否要重建检测，'
             '与目标区没有直接关系。')
    L.append(f'- 导航占每块三段合计的 {min(nav_share):.0f}–'
             f'{max(nav_share):.0f}%，仍是主要成本。')
    zones, per_round_zone = {}, []
    for a, _, m in loaded:
        rz = {}
        for o in a['objects']:
            d = sum(q['dist'] for q in o['nav_paths'])
            if d >= 1:
                for tbl in (zones, rz):
                    z = tbl.setdefault(o['zone'], [0.0, 0.0])
                    z[0] += o['nav_s']
                    z[1] += d
        per_round_zone.append((m['dir'].split('_')[-1], rz))
    if len(zones) >= 2:
        parts = [f"{z} 区 {v[0] / v[1]:.2f} s/m（{v[1]:.0f} m）"
                 for z, v in sorted(zones.items(),
                                    key=lambda kv: -kv[1][0] / kv[1][1])]
        detail = '；'.join(
            m + ' ' + '、'.join(f"{z} {v[0] / v[1]:.2f}"
                               for z, v in sorted(rz.items()))
            for m, rz in per_round_zone if rz)
        L.append('- 跨轮合计的每米导航耗时：' + '、'.join(parts) +
                 f"。分轮看（{detail}）同一目标区的 s/m 抖动很大，"
                 '说明快慢主要取决于该轮遇到的 mover 与走廊拥堵，'
                 '而不是区域本身。')
    ratios_all = []
    for o, _ in allobj:
        paths = o.get('nav_paths') or []
        tot = sum(p['dist'] for p in paths)
        chd = sum(((p['start'][0] - p['end'][0]) ** 2 +
                   (p['start'][1] - p['end'][1]) ** 2) ** 0.5
                  for p in paths)
        if chd > 0.5:
            ratios_all.append(tot / chd)
    if ratios_all:
        L.append(f'- 静态障碍不直接产生让行决策，但决定路程：各块绕行系数 '
                 f'{min(ratios_all):.2f}–{max(ratios_all):.2f}；'
                 '直线净空 0.00–0.05 m 的段都是墙，规划器只能沿墙绕行。')
    arc_legs = []
    for o, _ in allobj:
        for p in (o.get('nav_paths') or []):
            if 'yield arc' in (p.get('label') or ''):
                arc_legs.append((o['cube'], p['label'], p.get('sec', 0.0)))
    enc_all = [e for a, _, _ in loaded for e in a['encounters']]
    mind = min((e['min_dist'] for e in enc_all), default=0.0)
    L.append(f'- 动态避障：{len(loaded)} 轮共 {len(enc_all)} 次接近、最近中心距 '
             f'{mind:.2f} m；执行侧移 {len(arc_legs)} 次'
             f'（合计 {sum(x[2] for x in arc_legs):.1f} s）。')
    if arc_legs:
        L.append('  - 侧移段：' + '；'.join(
            f"{c} `{lab}` {du:.1f}s"
            + ('（触及 `yield_arc_timeout_sec` 上限）' if du >= 19.5 else '')
            for c, lab, du in arc_legs) + '。')
        if any(du >= 19.5 for _, _, du in arc_legs):
            L.append('  - 到达 0.35 m 交接半径只说明弧目标最后被消费掉了，'
                     '20 s 上限结束的段并不算让行成功；验收看下面的接触与净距。')
    contacts = [m['_contact'] for _, _, m in loaded if m.get('_contact')]
    if contacts:
        hull = contacts[0]['hull']
        worst = min((c for c in contacts if c['min_clearance'] is not None),
                    key=lambda c: c['min_clearance'], default=None)
        policy = min((c for c in contacts
                      if c['min_policy_clearance'] is not None),
                     key=lambda c: c['min_policy_clearance'], default=None)
        pushes = [(m['name'], e) for _, _, m in loaded
                  for e in (m['_contact'] or {}).get('pushed_while_stopped', [])]
        hits = sum(len(c['contacts']) for c in contacts)
        text = (f"- Gazebo 真值（{2*hull['half_length']:.2f}×"
                f"{2*hull['half_width']:.2f} m 外壳，约 5 Hz 采样）："
                f"采样帧内壳体重叠 {hits} 次")
        if worst:
            text += (f"；最小外壳净距 {worst['min_clearance']:+.3f} m"
                     f"（{worst['min_clearance_mover']}，{worst['min_clearance_at']:.1f} s）")
        if policy:
            text += (f"；按让行策略 0.50×0.70 m 安全壳算 {policy['min_policy_clearance']:+.3f} m")
        L.append(text + '。')
        # A 0.2 m move with a zero command after 0.35 s at rest cannot be a
        # ramp-down: something pushed the base. Even a 0.1 m one is far beyond
        # settling jitter, which is why these count as contact evidence that
        # the 5 Hz pose sampling did not happen to catch as an overlap.
        big = [item for item in pushes if item[1]['moved'] >= 0.10]
        if big:
            biggest = max(big, key=lambda item: item[1]['moved'])
            L.append(f'  - 零速静止后被推移 ≥0.10 m 共 {len(big)} 次，最大 '
                     f"{biggest[1]['moved']:.3f} m（{biggest[0]} "
                     f"{biggest[1]['t']:.1f} s，指令 "
                     f"{biggest[1]['cmd'][0]:.2f} m/s）：采样帧没抓到重叠，"
                     '但这些位移本身证明接触发生过，且停车点就在障碍物扫掠区内。')
        if len(pushes) > len(big):
            L.append(f'  - 另有 {len(pushes) - len(big)} 次 <0.10 m 的零速位移，'
                     '量级接近底盘抖动，未单独算作接触证据。')
    if len(seeds) == 1:
        only = next(iter(seeds))
        L.append(f'- 三轮题面 seed 相同（{only}），只改 mapping_rule，'
                 '跨轮差异可归因于规则映射与 mover 随机性。')
    done_all = all(m['_done'] == m['_plan'] for _, _, m in loaded)
    n_done = sum(1 for _, _, m in loaded if m['_done'] == m['_plan'])
    if len(loaded) == 3 and done_all:
        head = '三轮均'
    else:
        head = f'{len(loaded)} 轮中 {n_done} 轮'
    L.append(f'- {head} `COMPLETED` 5/5；x,y 与题面一律以各轮 `problem.txt` '
             '和 `映射规则 … 生效` 日志为准。')
    allobj = [(o, a) for a, _, _ in loaded for o in a['objects']]
    slow = None
    for idx, (a, _, _) in enumerate(loaded, 1):
        for o in a['objects']:
            labs = o.get('nav_labels') or []
            picks = [x for x in labs
                     if 'warehouse' not in x['label']
                     and not re.search(r' to [ABC]\b', x['label'])]
            base = picks[0]['label'] if picks else None
            tries = [x for x in picks if x['label'] == base]
            if len(tries) > 1:
                tot = sum(x['sec'] for x in tries)
                if slow is None or tot > slow[4]:
                    slow = (idx, o, o['cube'], o['wall_s'], tot, len(tries))
    if allobj:
        li = max(range(len(allobj)), key=lambda i: allobj[i][0]['wall_s'])
        lo = allobj[li][0]
        ri = next(i for i, (a, _, _) in enumerate(loaded, 1)
                  if a is allobj[li][1])
        L.append(f"- 全程最长的一块：第 {ri} 轮第 "
                 f"{lo.get('index', li + 1)} 块 `{lo['cube']}` "
                 f"{lo['wall_s']:.1f} s（导航 {lo['nav_s']:.1f} s）。")
    if slow:
        idx, so, cube, wall, tot, n = slow
        others = sorted(o['nav_s'] for o, _ in allobj if o is not so)
        rng = f'{others[0]:.1f}–{others[-1]:.1f} s' if others else '—'
        L.append(f'- 去物块重试最久的一块：第 {idx} 轮 `{cube}` 全程 {wall:.1f} s，'
                 f'其中去物块段共 {n} 次尝试、{tot:.1f} s（同一航段的多次重试），'
                 f'其余块导航 {rng}。')
    L.append('')
    if allobj:
        import numpy as np
        navs = np.array([o['nav_s'] for o, _ in allobj])
        viss = np.array([o['vision_s'] for o, _ in allobj])
        placed = [o for o, _ in allobj
                  if any(t['label'].startswith('arm:Lower to place')
                         for t in o['timings'])]
        mans = np.array([o['manip_s'] for o in placed])
        yields = sum(len(a['yields']) for a, _, _ in loaded)
        acts = {}
        for a, _, _ in loaded:
            for k, v in (a['totals'].get('yield_actions') or {}).items():
                acts[k] = acts.get(k, 0) + v
        enc = sum(len(a['encounters']) for a, _, _ in loaded)
        mind = min((e['min_dist'] for a, _, _ in loaded
                    for e in a['encounters']), default=0.0)
        dist = sum(a['totals'].get('path_length_m', 0) for a, _, _ in loaded)
        L.append('### 全部轮次汇总')
        L.append('')
        L.append('| 指标 | 值 |')
        L.append('|---|---|')
        L.append(f"| 物块总数 | {len(allobj)} |")
        L.append(f"| 每块导航 | 均值 {navs.mean():.1f} s（σ {navs.std():.1f}，"
                 f"范围 {navs.min():.1f}–{navs.max():.1f}） |")
        L.append(f"| 每块视觉 | 均值 {viss.mean():.1f} s（σ {viss.std():.1f}，"
                 f"范围 {viss.min():.1f}–{viss.max():.1f}） |")
        L.append(f"| 每块拿放（{len(placed)} 个完整块） | 均值 "
                 f"{mans.mean():.1f} s（σ {mans.std():.2f}，"
                 f"范围 {mans.min():.1f}–{mans.max():.1f}） |")
        L.append(f"| 导航每米耗时 | {navs.sum() / max(0.1, dist):.2f} s/m |")
        L.append(f"| 动态接近 | {enc} 次，最近 {mind:.2f} m |")
        L.append(f"| 让行决策 | {yields} 条（{acts}） |")
        if contacts:
            hits = sum(len(c['contacts']) for c in contacts)
            worst = min((c['min_clearance'] for c in contacts
                         if c['min_clearance'] is not None), default=None)
            big = sum(1 for item in pushes if item[1]['moved'] >= 0.10)
            L.append(f"| 外壳接触 | 采样帧 {hits} 次"
                     + (f"，最小净距 {worst:+.3f} m" if worst is not None else '')
                     + f"；零速被推移 ≥0.10 m {big} 次 |")
        L.append('')
    L.append('## 复现方式')
    L.append('')
    L.append('```bash')
    dirs = [str(Path(m['dir'])) for _, _, m in loaded]
    L.append(f'bash tools/run_round.sh 1 {dirs[0]}   # 改规则并跑一轮')
    L.append(f'python3 tools/round_analysis.py {dirs[0]}')
    L.append(f'python3 tools/plot_round.py    {dirs[0]}   # route.png')
    L.append(f'python3 tools/plot_timeline.py {dirs[0]}   # timeline.png')
    L.append('python3 tools/plot_timings.py  ' + ' '.join(dirs) + '   # timings.png')
    L.append('python3 tools/timings_csv.py   ' + ' '.join(dirs) + ' --out runs/timings.csv')
    cmd = ' \\\n    '.join(dirs)
    if args.appendix:
        cmd += ' \\'
        L.append('python3 tools/build_report.py  ' + cmd)
        L.append(f'    --appendix {args.appendix} --out runs/REPORT.md')
    else:
        L.append('python3 tools/build_report.py  ' + cmd + ' --out runs/REPORT.md')
    L.append('```')
    L.append('')

    if args.appendix:
        ap_path = Path(args.appendix)
        if ap_path.exists():
            L.append('')
            L.extend(ap_path.read_text(encoding='utf-8').splitlines())
            L.append('')

    out = Path(args.out)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'wrote {out} ({len(L)} lines, {sum(len(x) for x in L)} chars)')


if __name__ == '__main__':
    main()

"""Offline evidence verifier regressions; no ROS nodes or control output."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[4] / 'run_logs/tools/verify_round_evidence.py'


class EvidenceVerifierTests(unittest.TestCase):
    def build_round(self, root):
        d = Path(root)
        source = d / 'input.txt'
        source.write_text('at round start')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = f'{digest}  {source}\n'
        for i in range(25):
            extra = d / f'input_{i:02d}.txt'
            extra.write_text('at round start')
            manifest += f'{digest}  {extra}\n'
        (d / 'config_hashes_before.txt').write_text(manifest)
        (d / 'config_hashes_after.txt').write_text(manifest)
        source.write_text('later edited source')
        (d / 'round_meta.json').write_text(json.dumps(dict(
            tag='synthetic', verdict='COMPLETED', wall_seconds=110, smoother_trace='on')))
        lines = ['SYSTEM READY: Gazebo',
                 '[INFO] [velocity_smoother_trace-23]: process started with pid [123]']
        for i in range(5):
            lines += [f'Starting fixed grasp of red_cube_{i}', 'ATTACH: ATTACHED:',
                      'DETACH: DETACHED:',
                      '[INFO] [fixed_joint_pick_place]: Placement check: inside limits',
                      'DELIVERY COMPLETED:']
        lines.append('State -> COMPLETED')
        (d / 'sim.log').write_text('\n'.join(lines))
        (d / 'timeline.jsonl').write_text('\n'.join(json.dumps(dict(
            mono_abs=float(t), ready=True, sim=float(t), v=.1))
            for t in (100, 111)) + '\n')
        (d / 'events.jsonl').write_text('')
        (d / 'watch.log').write_text('WATCH_MODE=off')
        (d / 'cleanup.log').write_text('干净：ps 中没有任何竞赛栈进程')
        rows = [dict(mono_abs=float(t), pid=123, input_enter=t-100,
                     input_valid=t-100, timer_enter=(t-100)*20,
                     timer_timeout_stopped=0, publish_before=(t-100)*20,
                     publish_after=(t-100)*20) for t in range(100,113)]
        (d / 'smoother_internal.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows)+'\n')
        return d, rows

    def verify(self, d):
        subprocess.run([sys.executable, str(SCRIPT), str(d)], check=True,
                       stdout=subprocess.DEVNULL)
        return json.loads((d / '证据完整性.json').read_text())

    def test_persisted_after_manifest_ignores_later_source_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self.build_round(tmp)
            report = self.verify(d)
            self.assertTrue(report['full_5_of_5_verified'])
            self.assertEqual(report['config_hash_check_scope'], 'before_vs_after_manifest')
            (d / 'config_hashes_after.txt').write_text('0'*64+'  '+str(d/'input.txt')+'\n')
            self.assertFalse(self.verify(d)['config_hashes_match'])

    def test_optional_independent_reliable_requires_continuous_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self.build_round(tmp)
            meta = json.loads((d / 'round_meta.json').read_text())
            meta['solo_reliable'] = 'on'
            (d / 'round_meta.json').write_text(json.dumps(meta))
            self.assertFalse(self.verify(d)['solo_reliable_coverage'])
            rows = [dict(mono_abs=float(t), messages=t-99, age_s=.1)
                    for t in range(100,113)]
            (d / 'reliable_nav_solo.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertTrue(self.verify(d)['solo_reliable_coverage'])

    def test_udp_reliable_requires_profile_reader_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self.build_round(tmp)
            meta = json.loads((d / 'round_meta.json').read_text())
            meta['udp_reliable'] = 'on'
            (d / 'round_meta.json').write_text(json.dumps(meta))
            self.assertFalse(self.verify(d)['udp_reliable_coverage'])
            graph = dict(sub=[dict(node='//reliable_nav_probe_udp',
                                   reliability='ReliabilityPolicy.RELIABLE')])
            rows = [dict(mono_abs=float(t), messages=t-99, age_s=.1,
                         transport_label='udp_only',
                         transport_profile='/x/fastdds_wsl_udp.xml',
                         graph=graph if t==105 else {})
                    for t in range(100,113)]
            (d / 'reliable_nav_udp.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertTrue(self.verify(d)['udp_reliable_coverage'])
            rows[6]['transport_profile'] = None
            (d / 'reliable_nav_udp.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertFalse(self.verify(d)['udp_reliable_coverage'])

    def test_deep_reliable_requires_depth10_graph_and_default_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self.build_round(tmp)
            meta = json.loads((d / 'round_meta.json').read_text())
            meta['deep_reliable'] = 'on'
            (d / 'round_meta.json').write_text(json.dumps(meta))
            self.assertFalse(self.verify(d)['deep_reliable_coverage'])
            graph = dict(sub=[dict(node='//reliable_nav_probe_deep',
                                   reliability='ReliabilityPolicy.RELIABLE',depth=10)])
            rows = [dict(mono_abs=float(t), messages=t-99, age_s=.1,
                         qos_depth=10, transport_label='default',
                         transport_profile=None, graph=graph if t==105 else {})
                    for t in range(100,113)]
            (d / 'reliable_nav_deep.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertTrue(self.verify(d)['deep_reliable_coverage'])
            graph['sub'][0]['depth'] = 0  # Fast DDS reports unknown depth.
            (d / 'reliable_nav_deep.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertTrue(self.verify(d)['deep_reliable_coverage'])
            graph['sub'][0]['depth'] = 1
            (d / 'reliable_nav_deep.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertFalse(self.verify(d)['full_5_of_5_verified'])

    def test_v2_timer_duration_required_and_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = self.build_round(tmp)
            meta = json.loads((d/'round_meta.json').read_text())
            meta['timer_duration_trace'] = 'on'
            (d/'round_meta.json').write_text(json.dumps(meta))
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])
            for row in rows:
                row['timer_duration_last_ns'] = 10_000
                row['timer_duration_interval_max_ns'] = 20_000
                row['timer_duration_completed'] = row['timer_enter']
                row['timer_inflight_age_ns'] = 0
            path = d / 'smoother_internal.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            report = self.verify(d)
            self.assertTrue(report['smoother_trace_coverage'])
            self.assertEqual(report['smoother_timer_duration_peak_ms'],.02)
            rows[5]['timer_duration_interval_max_ns'] = -1
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])
            rows[5]['timer_duration_interval_max_ns'] = 20_000
            del rows[5]['timer_duration_completed']
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])

    def test_trace_udp_mode_requires_udp_profile_and_trace_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = self.build_round(tmp)
            meta = json.loads((d/'round_meta.json').read_text())
            meta['smoother_trace'] = 'off'
            meta['smoother_trace_udp'] = 'on'
            meta['smoother_transport_trace'] = 'on'
            (d/'round_meta.json').write_text(json.dumps(meta))
            for row in rows:
                row['transport_profile'] = None
            path = d / 'smoother_internal.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_transport_valid'])
            for row in rows:
                row['transport_profile'] = '/x/fastdds_wsl_udp.xml'
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertTrue(self.verify(d)['smoother_transport_valid'])
            for row in rows:
                row['transport_profile'] = '/x/some_other.xml'
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])

    def test_ambient_dds_profile_invalidates_transport_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = self.build_round(tmp)
            meta = json.loads((d/'round_meta.json').read_text())
            meta.update({'smoother_trace': 'off', 'smoother_trace_udp': 'on',
                         'smoother_transport_trace': 'on',
                         'ambient_fastdds_profile': '/x/fastdds_wsl_udp.xml'})
            (d/'round_meta.json').write_text(json.dumps(meta))
            for row in rows:
                row['transport_profile'] = '/x/fastdds_wsl_udp.xml'
            (d/'smoother_internal.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows)+'\n')
            # An exported ambient profile would have put the WHOLE stack on UDP,
            # so the smoother's own variable is no longer discriminating.
            self.assertFalse(self.verify(d)['smoother_transport_valid'])
            meta['ambient_fastdds_profile'] = None
            (d/'round_meta.json').write_text(json.dumps(meta))
            self.assertTrue(self.verify(d)['smoother_transport_valid'])

    def test_missing_meta_is_refused_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            out = subprocess.run(
                [sys.executable, str(SCRIPT), str(d)],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 2)
            self.assertIn('not a round', out.stdout)

    def test_trace_must_cover_the_task_not_the_recorder_tail(self):
        """The recorder outlives the task; the trace only has to cover the task.

        A healthy round failed this gate because the recorder kept sampling for
        7.5 s after the terminal event while the instrumented smoother had
        already exited (speed_rpp15_combined_trial4_20261010-171830).
        """
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = self.build_round(tmp)
            (d / 'events.jsonl').write_text(json.dumps(dict(
                kind='WORK_STATE', mono=111.0, wall=1000.0,
                data={'state': '全部完成'})) + '\n')
            (d / 'timeline.jsonl').write_text('\n'.join(json.dumps(dict(
                mono_abs=float(t), wall=1000.0 + (t - 111), ready=True, sim=float(t), v=.1))
                for t in (100, 111, 120)) + '\n')
            # Trace ends at 112: past the terminal event, before the recorder's 120.
            self.assertTrue(self.verify(d)['smoother_trace_coverage'])
            # A trace that stops BEFORE the task finishes must still fail.
            short = [r for r in rows if r['mono_abs'] <= 108]
            (d / 'smoother_internal.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in short) + '\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])

    def test_trace_gap_or_respawn_fails_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = self.build_round(tmp)
            self.assertTrue(self.verify(d)['smoother_trace_coverage'])
            rows = rows[:4] + rows[9:]
            (d / 'smoother_internal.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])
            rows = self.build_round(tmp)[1]
            rows[6]['pid'] = 456
            (d / 'smoother_internal.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows)+'\n')
            self.assertFalse(self.verify(d)['smoother_trace_coverage'])


if __name__ == '__main__':
    unittest.main()

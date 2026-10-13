"""Tests for the labeling-kit fixes after the v1 audit (protocol v2 draft). Synthetic fixtures only, no AV2 data.

- vehicle station columns come from the labeler's own answers, never from a box (audit major 1)
- the labeler-visible task file holds no coordinates, columns, guides or hashes (audit major 1)
- the self-test oracle seed is the same in every process (audit major 2)
- coordinator-recorded session heads, private scratch folders, locked labeler instructions (audit minors)

Run: .local/venv-labels/bin/python -I -m unittest discover -s tests/evaluation -p 'test_labels*.py'
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_labels_tools import KIT, KIT_DIR, PROTOCOL, RectOracle, blurred_rect, car_task, drive  # noqa: E402

aggregate, build_validation, common, imaging, labelkit = KIT.aggregate, KIT.build_validation, KIT.common, KIT.imaging, KIT.labelkit
lock_protocol, oracle_mod, selftest, session = KIT.lock_protocol, KIT.oracle, KIT.selftest, KIT.session

FRACTIONS = [0.25, 0.5, 0.75]
PROTOCOL_V2 = copy.deepcopy(PROTOCOL)
PROTOCOL_V2['version'] = 'test-v2'
PROTOCOL_V2['stations'] = {'insideFraction': 0.45, 'carColumns': 'labelerExtremes', 'carFractions': FRACTIONS, 'carMinSpanPx': 4}
PROTOCOL_V2['aggregation'] = {'labelers': 3, 'spreadMaxPx': {'line': 3.0, 'pixel': 2.0}, 'columnSpreadMaxPx': 3.0,
                              'requireReportedHeads': True}
PROTOCOL_V2['labelers'] = {'privateScratch': True}
PUBLIC_FIELDS = ['taskId', 'kind', 'target', 'overview', 'estimatedQuestions', 'answerDir']
TASK_FILES = {'taskFiles': {'public': PUBLIC_FIELDS, 'publicNumbers': ['estimatedQuestions']}}
OVERVIEW_P = {'overview': {'marginMin': 0.35, 'marginJitter': 0.5, 'targetColumns': 18, 'cellChoices': [8, 12, 16, 20, 24, 32, 40],
                           'ringJitterFraction': 0.08}}


def own_columns(result, box_right=None):
    f = result['features']
    left = f['L']['value']
    right = f['R']['value'] if f['R']['status'] == 'resolved' else box_right
    return [int(np.floor(left + q * (right - left) + 0.5)) for q in FRACTIONS]


class StationColumnTests(unittest.TestCase):
    def setUp(self):
        self.box = (60.3, 50.6, 140.7, 120.2)
        self.rgb, self.cov = blurred_rect(box=self.box)

    def test_v2_columns_follow_the_labelers_own_left_and_right(self):
        task = car_task()
        task.pop('stationColumns')
        result, _ = drive(task, PROTOCOL_V2, self.rgb, RectOracle(self.cov, task).answer)
        expected = own_columns(result)
        self.assertEqual(result['stationColumns'], expected)
        self.assertEqual([result['features'][f'S{k}']['fixed'] for k in (1, 2, 3)], expected)
        for k in (1, 2, 3):
            self.assertAlmostEqual(result['features'][f'S{k}']['value'], self.box[1], delta=0.15)

    def test_v2_ignores_any_columns_in_the_task(self):
        clean = car_task()
        clean.pop('stationColumns')
        planted = car_task(stations=(61, 100, 139))  # e.g. columns derived from a box: must have no effect
        a, _ = drive(clean, PROTOCOL_V2, self.rgb, RectOracle(self.cov, clean).answer)
        b, _ = drive(planted, PROTOCOL_V2, self.rgb, RectOracle(self.cov, planted).answer)
        self.assertEqual(json.dumps(a, default=float), json.dumps(b, default=float))

    def test_v2_unresolved_side_falls_back_to_the_labelers_own_box(self):
        task = car_task()
        task.pop('stationColumns')
        base = RectOracle(self.cov, task)

        def answer(q):
            if q['feature'] == 'R' and q['kind'] == 'yesno':
                return 'cannot_tell'
            return base.answer(q)
        result, _ = drive(task, PROTOCOL_V2, self.rgb, answer)
        self.assertEqual(result['features']['R']['status'], 'unresolved')
        x_last = result['box'][1]  # labeler's own grid box: (x_first, x_last, y_first, y_last)
        self.assertEqual(result['stationColumns'], own_columns(result, box_right=x_last + 0.5))

    def test_v1_protocol_still_uses_task_columns(self):
        task = car_task(stations=(80, 100, 120))
        result, _ = drive(task, PROTOCOL, self.rgb, RectOracle(self.cov, task).answer)
        self.assertEqual([result['features'][f'S{k}']['fixed'] for k in (1, 2, 3)], [80, 100, 120])
        self.assertNotIn('stationColumns', result)  # v1 result shape unchanged (66 v1 sessions replay identically)


def real_car_entry():
    """A LiDAR-referenced vehicle with an AV2 annotation box 15 px wider than the LiDAR silhouette (V06-like)."""
    return {'source': 'lidar', 'kind': 'car', 'reference': {'L': 1225.6, 'T': 704.2, 'R': 1519.9, 'B': 905.3},
            'case': {'annotationRectPx': [1210.4, 690.0, 1500.3, 912.0]}, 'projectedCentroidPx': [1372.7, 804.8]}


def assemble(task_id, P, entry, rng):
    region, cell, ring, columns = build_validation.car_task_fields(entry, P, rng, 2048, 1550)
    meta = {'region': region, 'cell': cell, 'columns': 18, 'rows': 12, 'columnLabels': ['A', 'R'], 'rowLabels': [1, 12],
            'ring': ring, 'path': f'x/overviews/{task_id}.png', 'sha256': 'cd' * 32}
    return {'taskId': task_id, 'kind': 'car', 'tier': 1,
            'image': {'path': f'x/images/{task_id}.jpg', 'sha256': 'ab' * 32, 'width': 2048, 'height': 1550},
            'overview': meta, 'target': {'noun': 'vehicle', 'hint': 'The target vehicle is the one under the cyan ring.'},
            'stationColumns': columns, 'guides': [], 'estimatedQuestions': 90, 'answerDir': f'x/sessions/<LABELER>/{task_id}/'}


class TaskFileTests(unittest.TestCase):
    def test_v1_task_columns_reveal_the_av2_box(self):
        """Documents the audited v1 leak: the box edges follow from the columns to within 0.6 px."""
        P = dict(OVERVIEW_P, stations={'carFractions': FRACTIONS, 'syntheticSamplingJitter': 0.06})
        task = assemble('V06', P, real_car_entry(), random.Random(1))
        c = task['stationColumns']
        width = 2 * (c[2] - c[0])
        box = real_car_entry()['case']['annotationRectPx']
        self.assertLessEqual(abs(c[0] - 0.25 * width - box[0]), 0.6)
        self.assertLessEqual(abs(c[0] + 0.75 * width - box[2]), 0.6)
        with tempfile.TemporaryDirectory() as tmp:  # the v2 check refuses such a task list
            P2 = dict(P, **{'taskFiles': {'public': PUBLIC_FIELDS + ['stationColumns'], 'publicNumbers': ['estimatedQuestions']}})
            with self.assertRaises(common.ProtocolError):
                build_validation.write_task_files(tmp, [task], P2, {'protocol': {'sha256': 'x'}})

    def test_v2_labeler_visible_task_list_has_no_box_derived_numbers(self):
        P = dict(OVERVIEW_P, stations={'carColumns': 'labelerExtremes', 'carFractions': FRACTIONS}, **TASK_FILES)
        entry = real_car_entry()
        tasks = [assemble('V01', P, entry, random.Random(1)), assemble('V02', P, entry, random.Random(2))]
        self.assertEqual(tasks[0]['stationColumns'], [])
        with tempfile.TemporaryDirectory() as tmp:
            instructions = {'path': 'x/labeler-instructions-v2.md', 'sha256': 'ef' * 32}
            header = {'protocol': {'file': 'protocol-v2.json', 'sha256': 'x'}, 'createdUtc': 'now', 'instructions': 'x/i.md'}
            build_validation.write_task_files(tmp, copy.deepcopy(tasks), P, header, instructions)
            public_path = Path(tmp) / 'validation-tasks.json'
            public = common.read_json(public_path)
            self.assertEqual(public['schema'], common.PUBLIC_TASKS_SCHEMA)
            for task in public['tasks']:
                self.assertLessEqual(set(task), set(PUBLIC_FIELDS))
                self.assertEqual(task['overview'], {'path': f"x/overviews/{task['taskId']}.png"})
                self.assertEqual(build_validation.public_numbers(task), [('estimatedQuestions', 90)])
            text = public_path.read_text()
            for value in entry['case']['annotationRectPx'] + list(entry['reference'].values()) + tasks[0]['overview']['region']:
                self.assertNotIn(str(int(value)), text)
            for secret in ('ab' * 32, 'cd' * 32, 'ef' * 32, '"stationColumns"', '"guides"', '"tier"', '"region"', '"ring"', '"image"'):
                self.assertNotIn(secret, text)
            merged = common.load_tasks(public_path)
            for original, task in zip(tasks, merged['tasks']):
                self.assertEqual(task, dict(original, instructions=instructions))

    def test_kit_internal_file_is_bound_to_the_public_list(self):
        P = dict(OVERVIEW_P, stations={'carColumns': 'labelerExtremes', 'carFractions': FRACTIONS}, **TASK_FILES)
        tasks = [assemble('V01', P, real_car_entry(), random.Random(1))]
        with tempfile.TemporaryDirectory() as tmp:
            build_validation.write_task_files(tmp, tasks, P, {'protocol': {'sha256': 'x'}})
            public_path = Path(tmp) / 'validation-tasks.json'
            doc = common.read_json(public_path)
            doc['tasks'][0]['target']['hint'] = 'edited after the build'
            common.write_json(public_path, doc)
            with self.assertRaises(common.ProtocolError):
                common.load_tasks(public_path)

    def test_merge_refuses_conflicting_fields(self):
        with self.assertRaises(common.ProtocolError):
            common.merge_task({'taskId': 'V01', 'overview': {'path': 'a.png'}}, {'taskId': 'V01', 'overview': {'path': 'b.png'}})
        merged = common.merge_task({'taskId': 'V01', 'overview': {'path': 'a.png'}},
                                   {'taskId': 'V01', 'overview': {'path': 'a.png', 'cell': 16}, 'stationColumns': [5]})
        self.assertEqual(merged, {'taskId': 'V01', 'overview': {'path': 'a.png', 'cell': 16}, 'stationColumns': [5]})


class SelftestSeedTests(unittest.TestCase):
    def test_oracle_seed_is_the_same_in_every_process(self):
        code = f"import sys; sys.path.insert(0, {str(KIT_DIR)!r}); import selftest; print(selftest.oracle_seed('V06', 'O2', 7))"
        runs = {subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, check=True).stdout.strip()
                for _ in range(3)}
        self.assertEqual(runs, {str(selftest.oracle_seed('V06', 'O2', 7))})
        self.assertEqual(selftest.oracle_seed('V06', 'O2', 7), int(common.sha256_bytes(b'V06|O2|7')[:8], 16))
        self.assertNotEqual(selftest.oracle_seed('V06', 'O2', 7), selftest.oracle_seed('V06', 'O2', 8))

    def test_noisy_oracle_draws_repeat_for_the_same_seed(self):
        task = car_task()

        def draws(seed):
            noisy = oracle_mod.Oracle(None, task, flip=0.1, seed=selftest.oracle_seed('T01', 'O1', seed))
            return [noisy.rng.random() for _ in range(50)]
        self.assertEqual(draws(3), draws(3))
        self.assertNotEqual(draws(3), draws(4))


class AggregationV2Tests(unittest.TestCase):
    SPREAD = {'line': 3.0, 'pixel': 2.0}

    def entries(self, columns, values=(50.0, 50.4, 50.9)):
        return {name: {'status': 'resolved', 'value': v, 'fixed': c} for name, c, v in zip('ABC', columns, values)}

    def test_own_station_columns_must_agree(self):
        ok = aggregate.combine('S1', self.entries((100, 101, 103)), self.SPREAD, 3.0)
        self.assertEqual(ok['status'], 'resolved')
        self.assertEqual((ok['column'], ok['columnSpread']), (101, 3))
        apart = aggregate.combine('S1', self.entries((100, 101, 104)), self.SPREAD, 3.0)
        self.assertEqual(apart['status'], 'unresolved')
        self.assertIn('station columns differ', apart['reason'])
        v1 = aggregate.combine('S1', self.entries((100, 101, 104)), self.SPREAD)  # v1: columns were identical by construction
        self.assertEqual(v1['status'], 'resolved')
        wheel = aggregate.combine('W1', self.entries((100, 110, 120)), self.SPREAD, 3.0)  # tire columns are cell picks
        self.assertEqual(wheel['status'], 'resolved')


class LabelkitV2Tests(unittest.TestCase):
    """A v2 task set (public + kit-internal files, locked instructions) driven through the CLI."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = self.base = Path(self.tmp.name)
        rgb, self.cov = blurred_rect()
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format='PNG')
        image = base / 'image.png'
        image.write_bytes(buffer.getvalue())
        protocol = base / 'protocol-v2.json'
        protocol.write_bytes(common.pretty_json(PROTOCOL_V2))
        common.write_json(base / 'protocol-v2.lock.json', {'sha256': common.sha256_file(protocol)})
        self.instructions = base / 'labeler-instructions-v2.md'
        self.instructions.write_text('instructions\n')
        task = car_task()
        task.pop('stationColumns')
        png, meta = imaging.render_overview(rgb, task['overview']['region'], task['overview']['cell'])
        (base / 'overview.png').write_bytes(png)
        meta.update(path=str(base / 'overview.png'), sha256=common.sha256_bytes(png))
        task.update(overview=meta, image={'path': str(image), 'sha256': common.sha256_file(image), 'width': 200, 'height': 160},
                    estimatedQuestions=90, answerDir='sessions/<LABELER>/T01/')
        self.task = task
        instructions = {'path': str(self.instructions), 'sha256': common.sha256_file(self.instructions)}
        build_validation.write_task_files(base, [task], TASK_FILES, {'protocol': {'sha256': common.sha256_file(protocol)}}, instructions)
        self.args = ['--labeler', 'L1', '--task', 'T01', '--tasks', str(base / 'validation-tasks.json'), '--protocol', str(protocol),
                     '--sessions', str(base / 'sessions')]
        self.protocol_path = protocol

    def run_cli(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = labelkit.main(self.args + list(extra))
        return code, out.getvalue()

    def pending(self):
        args = labelkit.argparse.Namespace(labeler='L1', task='T01', tasks=self.args[5], protocol=self.args[7], sessions=self.args[9])
        with contextlib.redirect_stdout(io.StringIO()):
            return labelkit.command_next(args, echo=False)

    def finish(self):
        oracle = RectOracle(self.cov, self.task)
        text = ''
        for _ in range(400):
            state = self.pending()
            if state is None:
                return text
            code, text = self.run_cli('answer', oracle.answer(state[0]))
            self.assertEqual(code, 0)
        self.fail('session did not finish')

    def test_session_prints_private_scratch_and_head_and_uses_own_columns(self):
        code, text = self.run_cli('next')
        self.assertEqual(code, 0)
        self.assertIn('SCRATCH (yours only, for this task):', text)
        self.assertTrue((self.base / 'scratch/L1/T01').is_dir())
        last = self.finish()
        records = labelkit.Store(self.base / 'sessions', 'L1', 'T01').records()
        self.assertIn(f"SESSION HEAD L1/T01: {records[-1]['hash']}", last)
        label = common.read_json(self.base / 'sessions/L1/T01/label.json')
        self.assertEqual(label['answerLogHead'], records[-1]['hash'])
        self.assertEqual(label['result']['stationColumns'], own_columns(label['result']))
        self.assertEqual(label['taskSha256'], common.sha256_bytes(common.canonical_json(common.load_tasks(self.args[5])['tasks'][0])))

    def test_aggregation_requires_the_reported_heads(self):
        self.run_cli('next')
        self.finish()
        head = common.read_json(self.base / 'sessions/L1/T01/label.json')['answerLogHead']
        phash = common.sha256_file(self.protocol_path)
        tasks_path, sessions = self.base / 'validation-tasks.json', self.base / 'sessions'
        with self.assertRaises(common.ProtocolError):
            aggregate.aggregate(tasks_path, sessions, ['L1'], PROTOCOL_V2, phash)
        with self.assertRaises(common.ProtocolError):
            aggregate.aggregate(tasks_path, sessions, ['L1'], PROTOCOL_V2, phash, {'L1/T01': '0' * 64})
        with self.assertRaises(common.ProtocolError):
            aggregate.aggregate(tasks_path, sessions, ['L1'], PROTOCOL_V2, phash, {})
        result, hashes, evidence = aggregate.aggregate(tasks_path, sessions, ['L1'], PROTOCOL_V2, phash, {'L1/T01': head})
        self.assertEqual(result['T01']['status'], 'complete')
        self.assertIn('L1/T01', hashes)
        self.assertEqual(evidence['answerLogSha256']['L1/T01'], common.sha256_file(sessions / 'L1/T01/answers.jsonl'))
        self.assertIsNone(evidence['transcriptScanSha256'])

    # --- audit minor (2026-10-12): aggregation must check the answer log itself, not only label.json ---

    def aggregate_l1(self, heads):
        phash = common.sha256_file(self.protocol_path)
        return aggregate.aggregate(self.base / 'validation-tasks.json', self.base / 'sessions', ['L1'], PROTOCOL_V2, phash, heads)

    def rewrite_label(self, label):
        folder = self.base / 'sessions/L1/T01'
        digest = common.write_json(folder / 'label.json', label)
        (folder / 'label.sha256').write_text(digest + '  label.json\n')

    def test_edited_label_value_with_unchanged_head_is_refused(self):
        # The auditor's demonstration: +5 px on one feature in label.json, label.sha256 rewritten, head unchanged.
        self.run_cli('next')
        self.finish()
        label = common.read_json(self.base / 'sessions/L1/T01/label.json')
        head = label['answerLogHead']
        self.aggregate_l1({'L1/T01': head})  # the untouched session is accepted
        feature = next(k for k, v in sorted(label['result']['features'].items()) if v.get('status') == 'resolved')
        label['result']['features'][feature]['value'] += 5.0
        self.rewrite_label(label)
        with self.assertRaisesRegex(common.ProtocolError, 'replaying the answer log does not give label.json'):
            self.aggregate_l1({'L1/T01': head})

    def test_consistent_whole_session_rewrite_is_caught_by_the_reported_head(self):
        self.run_cli('next')
        self.finish()
        folder = self.base / 'sessions/L1/T01'
        reported = common.read_json(folder / 'label.json')['answerLogHead']
        records = labelkit.Store(self.base / 'sessions', 'L1', 'T01').records()
        lines, prev = [], None
        for record in records:  # same answers, a changed clock field, chain rebuilt end to end
            body = {k: v for k, v in record.items() if k != 'hash'}
            body.update(prev=prev, utc='2026-01-01T00:00:00Z')
            body['hash'] = prev = common.sha256_bytes(common.canonical_json(body))
            lines.append(json.dumps(body, sort_keys=True))
        (folder / 'answers.jsonl').write_text('\n'.join(lines) + '\n')
        label = common.read_json(folder / 'label.json')
        label['answerLogHead'] = prev
        self.rewrite_label(label)
        self.assertNotEqual(prev, reported)
        self.aggregate_l1({'L1/T01': prev})  # internally consistent: chain, head and replay all agree
        with self.assertRaisesRegex(common.ProtocolError, 'differs from the reported head'):
            self.aggregate_l1({'L1/T01': reported})

    def test_partial_log_edit_and_missing_log_are_refused(self):
        self.run_cli('next')
        self.finish()
        head = common.read_json(self.base / 'sessions/L1/T01/label.json')['answerLogHead']
        log = self.base / 'sessions/L1/T01/answers.jsonl'
        original = log.read_text()
        lines = original.splitlines()
        record = json.loads(lines[3])
        record['answer'] = 'no' if record['answer'] != 'no' else 'yes'
        lines[3] = json.dumps(record, sort_keys=True)
        log.write_text('\n'.join(lines) + '\n')
        with self.assertRaisesRegex(common.ProtocolError, 'chain broken'):
            self.aggregate_l1({'L1/T01': head})
        log.unlink()
        with self.assertRaisesRegex(common.ProtocolError, 'no answer log'):
            self.aggregate_l1({'L1/T01': head})

    def test_selftest_sessions_pass_the_same_checks(self):
        task = common.load_tasks(self.base / 'validation-tasks.json')['tasks'][0]
        rgb = imaging.load_rgb(self.base / 'image.png')
        result, answers = drive(task, PROTOCOL_V2, rgb, RectOracle(self.cov, task).answer, labeler='L1')
        _, _, asked = session.run(task, PROTOCOL_V2, 'L1', rgb, answers)
        phash = common.sha256_file(self.protocol_path)
        head = selftest.write_label(self.base / 'sessions', 'L1', task, phash, result, answers, asked, 0)
        records = labelkit.Store(self.base / 'sessions', 'L1', 'T01').records()
        self.assertEqual((len(records), records[-1]['hash']), (len(answers), head))
        out, hashes, _ = self.aggregate_l1({'L1/T01': head})
        self.assertEqual(out['T01']['status'], 'complete')

    def test_changed_instructions_are_refused(self):
        self.run_cli('next')
        self.instructions.write_text('instructions, edited during labeling\n')
        code, text = self.run_cli('next')
        self.assertEqual(code, 2)
        self.assertIn('instructions changed', text)


class LockTests(unittest.TestCase):
    def write(self, tmp, **extra):
        path = Path(tmp) / 'protocol-v9.json'
        (Path(tmp) / 'protocol-v9.md').write_text('doc\n')
        (Path(tmp) / 'instr.md').write_text('instructions\n')
        params = {'document': 'protocol-v9.md', 'labelerInstructions': 'instr.md', 'lockStatement': 'Frozen after X; before Y.'}
        params.update(extra)
        path.write_bytes(common.pretty_json(params))
        return path

    def test_lock_refuses_open_items_and_missing_statement(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(common.ProtocolError):
                lock_protocol.build_record(self.write(tmp, pendingBeforeLock=['coverage redesign']))
            with self.assertRaises(common.ProtocolError):
                lock_protocol.build_record(self.write(tmp, lockStatement=None))

    def test_lock_records_statement_and_instruction_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = lock_protocol.build_record(self.write(tmp, pendingBeforeLock=[]))
            self.assertEqual(record['statement'], 'Frozen after X; before Y.')
            self.assertEqual(record['labelerInstructions'], {'file': 'instr.md', 'sha256': common.sha256_file(Path(tmp) / 'instr.md')})


if __name__ == '__main__':
    unittest.main()

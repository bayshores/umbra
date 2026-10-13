"""Tests for the second post-run audit fixes of the labeling kit (2026-10-13). Synthetic fixtures only, no AV2 data.

- answer isolation (protocol `isolation`): coordinator-only folder outside the repository, clean-tree check before any
  labeler command, recorded transcript scan before aggregation
- blind-repeat precision compares only stations measured at (nearly) the same column
- the synthetic gate's minimum count of exact features per family

Run: .local/venv-labels/bin/python -I -m unittest discover -s tests/evaluation -p 'test_labels*.py'
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_labels_tools import KIT, PROTOCOL, ROOT, blurred_rect, car_task  # noqa: E402

aggregate, common, imaging, isolation, labelkit, score = (KIT.aggregate, KIT.common, KIT.imaging, KIT.isolation, KIT.labelkit,
                                                          KIT.score)
ENV = 'PAO_LABELS_TEST_COORDINATOR'
V2_DRAFT = ROOT / '.local/evaluation/pao-labels/v2/protocol-v2.json'


def iso_protocol(**extra):
    block = {'coordinatorRootEnv': ENV, 'requireCleanTree': True, 'answerBearingGlobs': ['a/sealed/**', 'a/tool-selftest*/**', 'a/report.json'],
             'requireTranscriptScan': True, 'ownSessionPrefix': '.local/evaluation/pao-labels/v2/',
             'forbiddenToolInputPatterns': [r'sealed|answer[-_ ]?key', r'\.\./']}
    block.update(extra)
    return {'version': 'test-iso', 'isolation': block}


class CoordinatorFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outside = Path(self.tmp.name).resolve()

    def test_folder_must_be_set_and_outside_the_repository(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            with self.assertRaisesRegex(common.ProtocolError, 'Set PAO_LABELS_TEST_COORDINATOR'):
                isolation.coordinator_root(ENV)
        with mock.patch.dict(os.environ, {ENV: str(ROOT / '.local')}):
            with self.assertRaisesRegex(common.ProtocolError, 'outside the repository'):
                isolation.coordinator_root(ENV)
        with mock.patch.dict(os.environ, {ENV: str(self.outside)}):
            self.assertEqual(isolation.coordinator_root(ENV), self.outside)
            self.assertEqual(isolation.protocol_store(iso_protocol()), self.outside / 'pao-labels/test-iso')

    def test_sealed_folder_comes_from_the_lock(self):
        base = Path('/nonexistent/base')
        self.assertEqual(isolation.sealed_dir(base, {}), base / 'sealed')  # v1 sets
        spec = isolation.sealed_spec(iso_protocol())
        self.assertEqual(spec, {'location': 'coordinator', 'env': ENV, 'subdir': 'pao-labels/test-iso/sealed'})
        self.assertNotIn(str(self.outside), json.dumps(spec))  # the lock records the variable, never the path
        with mock.patch.dict(os.environ, {ENV: str(self.outside)}):
            self.assertEqual(isolation.sealed_dir(base, {'sealed': spec}), self.outside / 'pao-labels/test-iso/sealed')

    def test_answer_bearing_outputs_are_refused_inside_the_repository(self):
        with self.assertRaisesRegex(common.ProtocolError, 'outside the repository'):
            isolation.output_dir(iso_protocol(), ROOT / '.local/evaluation/pao-labels/v2/scores', 'the score')
        self.assertEqual(isolation.output_dir(iso_protocol(), self.outside / 'scores', 'the score'), self.outside / 'scores')
        inside = ROOT / '.local/evaluation/pao-labels/v2/scores'
        self.assertEqual(isolation.output_dir({'version': 'v1'}, inside, 'the score'), inside)  # no isolation block: unchanged


class CleanTreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        top = Path(self.tmp.name).resolve()
        self.repo, self.store = top / 'repo', top / 'coordinator'
        self.store.mkdir()
        for name in ('a/sealed/answer-key.json', 'a/tool-selftest/flip-0/.sessions-x/O1/label.json', 'a/report.json', 'a/keep.md'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'{name}\n')
        patches = [mock.patch.object(isolation, 'ROOT', self.repo), mock.patch.object(common, 'ROOT', self.repo),
                   mock.patch.object(isolation, 'LABELS_DIR', self.repo / 'a'), mock.patch.dict(os.environ, {ENV: str(self.store)})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_check_finds_hidden_answer_files_and_refuses(self):
        found = isolation.answer_bearing(iso_protocol())
        self.assertEqual(found, ['a/report.json', 'a/sealed/answer-key.json', 'a/tool-selftest/flip-0/.sessions-x/O1/label.json'])
        with self.assertRaisesRegex(common.ProtocolError, '3 answer-bearing file'):
            isolation.check_clean_tree(iso_protocol())
        self.assertEqual(isolation.check_clean_tree(iso_protocol(requireCleanTree=False)), [])

    def test_move_is_a_dry_run_unless_applied_and_keeps_hashes(self):
        dry = isolation.move(iso_protocol())
        self.assertFalse(dry['apply'])
        self.assertTrue((self.repo / 'a/report.json').exists())
        done = isolation.move(iso_protocol(), apply=True)
        self.assertEqual(len(done['files']), 3)
        self.assertEqual(isolation.check_clean_tree(iso_protocol()), [])
        moved = self.store / 'pao-labels/test-iso/moved-from-repo'
        for item in done['files']:
            self.assertEqual(common.sha256_file(moved / item['path']), item['sha256'])
        self.assertTrue((self.repo / 'a/keep.md').exists())
        self.assertFalse((self.repo / 'a/sealed').exists() or (self.repo / 'a/tool-selftest').exists())  # emptied folders removed
        records = list((self.repo / 'a').glob('isolation-moves-*.json'))
        self.assertEqual(len(records), 1)
        self.assertNotIn(str(self.store), records[0].read_text())  # the in-repository record names the variable only

    def test_labelkit_refuses_while_answers_are_in_the_repository(self):
        rgb, _ = blurred_rect()
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format='PNG')
        work = self.repo.parent / 'work'
        work.mkdir()
        image = work / 'image.png'
        image.write_bytes(buffer.getvalue())
        protocol_path = work / 'protocol-iso.json'
        protocol_path.write_bytes(common.pretty_json({**PROTOCOL, **iso_protocol()}))
        common.write_json(work / 'protocol-iso.lock.json', {'sha256': common.sha256_file(protocol_path)})
        task = car_task()
        png, meta = imaging.render_overview(rgb, task['overview']['region'], task['overview']['cell'])
        (work / 'overview.png').write_bytes(png)
        meta.update(path=str(work / 'overview.png'), sha256=common.sha256_bytes(png))
        task.update(overview=meta, image={'path': str(image), 'sha256': common.sha256_file(image), 'width': 200, 'height': 160})
        (work / 'tasks.json').write_bytes(common.pretty_json({'protocol': {'sha256': common.sha256_file(protocol_path)}, 'tasks': [task]}))
        args = ['--labeler', 'L1', '--task', 'T01', '--tasks', str(work / 'tasks.json'), '--protocol', str(protocol_path),
                '--sessions', str(work / 'sessions'), 'next']
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = labelkit.main(args)
        self.assertEqual(code, 2)
        self.assertIn('answer-bearing', out.getvalue())
        isolation.move(iso_protocol(), apply=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(labelkit.main(args), 0)


def claude_line(tool, **inputs):
    return json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': [
        {'type': 'text', 'text': 'never open sealed/ (instructions quoted in text are not tool calls)'},
        {'type': 'tool_use', 'id': 'x', 'name': tool, 'input': inputs}]}})


def codex_line(command):
    return json.dumps({'type': 'response_item', 'payload': {'type': 'function_call', 'name': 'shell',
                                                            'arguments': json.dumps({'command': ['bash', '-lc', command]})}})


REPO = '/home/u/sightline/'
OWN = REPO + '.local/evaluation/pao-labels/v2/sessions/L1/V01/questions/0001.png'
KIT_CMD = '.local/venv-labels/bin/python -I scripts/evaluation/labels/labelkit.py --labeler L1 --task V01 '
CLEAN = [claude_line('Bash', command=KIT_CMD + 'next'), claude_line('Read', file_path=OWN),
         claude_line('Write', file_path=REPO + '.local/evaluation/pao-labels/v2/scratch/L1/V01/zoom.py',
                     content="from __future__ import annotations\nfrom PIL import Image\nImage.open('" + OWN + "')\n"),
         claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/v2/validation/overviews/V01.png'),
         claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/v2/labeler-instructions-v2.md'),
         claude_line('Bash', command=KIT_CMD + 'answer cannot_tell'), codex_line(KIT_CMD + 'status')]
FORBIDDEN = {
    'v1 answer key': claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/sealed/answer-key.json'),
    'other labeler session': claude_line('Bash', command='cat .local/evaluation/pao-labels/v2/sessions/L2/V01/questions/0001.png'),
    'session wildcard': claude_line('Bash', command='ls .local/evaluation/pao-labels/v2/sessions/*/V01'),
    'other task': claude_line('Bash', command=KIT_CMD.replace('V01', 'V02') + 'next'),
    'kit-internal': claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/v2/kit-internal/tasks-internal.json'),
    'oracle rows': claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/tool-selftest-multiseed/runs/v1-flip0/seed-0/x.json'),
    'other lane': claude_line('Bash', command='cat .local/evaluation/pao-accuracy/metrics.json'),
    'shared data': codex_line('python -c "print(1)" .local/data/av2/train/x/annotations.feather'),
    'parent folder': claude_line('Bash', command='cd .local/evaluation/pao-labels/v2/scratch/L1/V01 && cat ../../../sealed/x'),
    'walk': claude_line('Write', file_path=REPO + '.local/evaluation/pao-labels/v2/scratch/L1/V01/w.py', content='import os\nos.walk(".")\n'),
    'app': claude_line('Read', file_path=REPO + 'src/scene.js'),
    'other kit tool': claude_line('Bash', command='python scripts/evaluation/labels/score.py'),
    'v1 task list': claude_line('Read', file_path=REPO + '.local/evaluation/pao-labels/validation-tasks.json'),
}


@unittest.skipUnless(V2_DRAFT.exists(), 'needs the local v2 draft protocol')
class TranscriptScanTests(unittest.TestCase):
    """Uses the real patterns of the v2 draft protocol."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.protocol = common.read_json(V2_DRAFT)
        self.rules = isolation.scan_rules(self.protocol)

    def write(self, name, lines):
        path = self.dir / name
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_own_work_has_no_hit_and_is_bound_to_its_session(self):
        result = isolation.scan_transcript(self.write('clean.jsonl', CLEAN), 'L1', 'V01', self.rules)
        self.assertEqual(result['hits'], [])
        self.assertEqual(result['toolCalls'], len(CLEAN))
        self.assertEqual(result['bindingCalls'], 3)

    def test_each_forbidden_access_is_a_hit(self):
        for label, line in FORBIDDEN.items():
            with self.subTest(label):
                result = isolation.scan_transcript(self.write(f'{len(label)}.jsonl', CLEAN + [line]), 'L1', 'V01', self.rules)
                self.assertTrue(result['hits'], label)
                self.assertTrue(all(h['line'] == len(CLEAN) + 1 for h in result['hits']), label)

    def test_aggregation_requires_a_clean_complete_scan(self):
        phash = 'P'
        tasks = self.dir / 'tasks.json'
        tasks.write_bytes(common.pretty_json({'protocol': {'sha256': phash}, 'tasks': []}))
        protocol = {**self.protocol, 'aggregation': {'labelers': 1, 'spreadMaxPx': {'line': 3.0, 'pixel': 2.0}}}
        with self.assertRaisesRegex(common.ProtocolError, 'requires --transcript-scan'):
            aggregate.aggregate(tasks, self.dir / 'sessions', ['L1'], protocol, phash)
        clean = self.write('clean.jsonl', CLEAN)
        dirty = self.write('dirty.jsonl', CLEAN + [FORBIDDEN['v1 answer key']])
        good = self.dir / 'scan-good.json'
        common.write_json(good, isolation.scan(protocol, phash, {'sessions': {'L1/V01': [str(clean)]}}))
        _, _, evidence = aggregate.aggregate(tasks, self.dir / 'sessions', ['L1'], protocol, phash, transcript_scan=good)
        self.assertEqual(evidence['transcriptScanSha256'], common.sha256_file(good))
        self.assertEqual(isolation.verify_scan(good, protocol, phash, ['L1/V01']), common.sha256_file(good))
        with self.assertRaisesRegex(common.ProtocolError, 'L1/V02: no scanned transcript'):
            isolation.verify_scan(good, protocol, phash, ['L1/V01', 'L1/V02'])
        bad = self.dir / 'scan-bad.json'
        common.write_json(bad, isolation.scan(protocol, phash, {'sessions': {'L1/V01': [str(dirty)]}}))
        with self.assertRaisesRegex(common.ProtocolError, 'forbidden tool call'):
            isolation.verify_scan(bad, protocol, phash, ['L1/V01'])
        looser = json.loads(json.dumps(protocol))
        looser['isolation']['forbiddenToolInputPatterns'] = looser['isolation']['forbiddenToolInputPatterns'][1:]
        with self.assertRaisesRegex(common.ProtocolError, 'other rules'):
            isolation.verify_scan(good, looser, phash, ['L1/V01'])
        unbound = self.write('unbound.jsonl', [claude_line('Read', file_path=OWN)])
        common.write_json(self.dir / 'scan-unbound.json', isolation.scan(protocol, phash, {'sessions': {'L1/V01': [str(unbound)]}}))
        with self.assertRaisesRegex(common.ProtocolError, 'no transcript runs this labeler and task'):
            isolation.verify_scan(self.dir / 'scan-unbound.json', protocol, phash, ['L1/V01'])

    def test_v2_draft_isolation_settings(self):
        iso = self.protocol['isolation']
        self.assertTrue(iso['requireCleanTree'] and iso['requireTranscriptScan'])
        self.assertTrue(all(not g.startswith('/') and '..' not in g for g in iso['answerBearingGlobs']))
        self.assertEqual(len(iso['forbiddenToolInputPatterns']), len(iso['patternNotes']))
        self.assertEqual(self.protocol['scoring']['syntheticGate']['minExactPerFamily'], 10)


class RepeatPrecisionTests(unittest.TestCase):
    def test_only_stations_at_matching_columns_are_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp)
            features = {'V01': {'S1': (100, 50.0), 'S2': (120, 60.0), 'L': (None, 10.0)},
                        'V02': {'S1': (102, 51.0), 'S2': (130, 70.0), 'L': (None, 10.5)}}
            for task_id, items in features.items():
                result = {'features': {k: {'status': 'resolved', 'value': v, 'fixed': c} for k, (c, v) in items.items()}}
                common.write_json(sessions / 'A' / task_id / 'label.json', {'result': result})
            key = {'V01': {}, 'V02': {'repeatOf': 'V01'}}
            agg = {'labelers': ['A'], 'tasks': {'V01': {'status': 'complete'}, 'V02': {'status': 'complete'}}}
            v2 = score.repeat_precision(key, agg, sessions, 3.0)['A']
            self.assertEqual((v2['n'], v2['medianAbs'], v2['maxAbs']), (2, 0.75, 1.0))
            self.assertEqual([m['feature'] for m in v2['columnMismatched']], ['S2'])
            v1 = score.repeat_precision(key, agg, sessions)['A']  # v1 stations shared columns: any difference is a mismatch
            self.assertEqual((v1['n'], [m['feature'] for m in v1['columnMismatched']]), (1, ['S1', 'S2']))


class ScoreRequiresRecordedChecksTests(unittest.TestCase):
    def test_score_refuses_a_manifest_without_heads_scan_or_answer_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tasks = base / 'validation-tasks.json'
            tasks.write_bytes(common.pretty_json({'protocol': {'sha256': 'P'}, 'tasks': []}))
            key = base / 'sealed/answer-key.json'
            common.write_json(key, {'tasks': {}})
            common.write_json(base / 'validation-lock.json', {'protocolSha256': 'P', 'validationTasksSha256': common.sha256_file(tasks),
                                                               'answerKeySha256': common.sha256_file(key), 'coverageSha256': {}})
            agg_dir = base / 'aggregate'
            agg_hash = common.write_json(agg_dir / 'labels-aggregate.json', {'labelers': [], 'tasks': {}})
            protocol = {**iso_protocol(), 'aggregation': {'requireReportedHeads': True},
                        'scoring': {'syntheticGate': GateMinimumTests.LIMITS, 'lidarCoarse': {'minConsistentShare': 0.8,
                                                                                              'registrationAllowancePx': 8.0}}}
            complete = {'protocolSha256': 'P', 'aggregateSha256': agg_hash, 'labelSha256': {}, 'reportedHeadsSha256': 'h',
                        'transcriptScanSha256': 's', 'answerLogSha256': {'L1/V01': 'x'}}
            for missing, message in (('reportedHeadsSha256', 'coordinator-recorded heads'), ('transcriptScanSha256', 'transcript scan'),
                                     ('answerLogSha256', 'answer-log hashes')):
                common.write_json(agg_dir / 'labels-manifest.json', {**complete, missing: None})
                with self.subTest(missing), self.assertRaisesRegex(common.ProtocolError, message):
                    score.score(protocol, 'P', base, agg_dir, base / 'sessions')


class GateMinimumTests(unittest.TestCase):
    LIMITS = {'carExtremeMedianAbsPx': 2.0, 'carExtremeMaxAbsPx': 4.0, 'carProfileMedianAbsPx': 2.0, 'carProfileMaxAbsPx': 4.0,
              'wheelContactMedianAbsPx': 2.0, 'wheelContactMaxAbsPx': 4.0, 'paintMedianAbsPx': 1.0, 'paintMaxAbsPx': 2.0,
              'minResolvedShare': 0.7}

    def gate(self, n, **extra):
        protocol = {'scoring': {'syntheticGate': {**self.LIMITS, **extra}, 'lidarCoarse': {'minConsistentShare': 0.8}}}
        fam = {'medianAbs': 0.1, 'maxAbs': 0.2, 'exactN': n}
        summary = {k: dict(fam) for k in ('extreme/synthetic', 'profile/synthetic', 'contact/synthetic', 'paint/synthetic')}
        rows = [{'consistent': True}]
        return score.gate_check(protocol, summary, [9, 10], rows)

    def test_too_few_exact_features_fail_the_family(self):
        self.assertTrue(self.gate(2)['pass'])  # v1 rule: no minimum
        low = self.gate(4, minExactPerFamily=10)
        self.assertFalse(low['pass'])
        self.assertIn('only 4 exact features', low['checks']['paint']['reason'])
        self.assertTrue(self.gate(10, minExactPerFamily=10)['pass'])


if __name__ == '__main__':
    unittest.main()

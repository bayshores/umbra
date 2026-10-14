"""M7 metric A (fail-only paint check) after the paint v4 held-out review of 2026-10-14. Synthetic fixtures only, no AV2 data.

Run: nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -B -m unittest discover -s tests/evaluation -p 'test_m7*.py'
"""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
M7_DIR = ROOT / 'scripts/evaluation/m7'


def load_score():
    """Import m7/score.py without leaking its generic module names (common, score) into other tests."""
    saved = {name: sys.modules.pop(name) for name in ('common', 'score') if name in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(M7_DIR))
    try:
        return importlib.import_module('score')
    finally:
        for name in ('common', 'score'):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


score = load_score()
RECORD_SHA = 'ab' * 32


def paint_dir(folder, a, b, status='resolved', reason=None, script_sha=None):
    near = {'status': status, **({'label': 'resolved'} if status == 'resolved' else {'reason': reason})}
    result = {'protocolSha256': score.PAINT_PROTOCOL_SHA, 'scriptSha256': script_sha or score.PAINT_CODE_SHA,
              'estimatorA': {'edgeSAtEval': a}, 'estimatorB': {'edgeSAtEval': b}, 'nearEdge': near, 'frame': {'evalL': 0.5}}
    data = json.dumps(result).encode()
    (folder / 'result.json').write_bytes(data)
    receipt = {'siteRecord': {'sha256': RECORD_SHA}, 'outcome': 'measured', 'outputs': {'result.json': hashlib.sha256(data).hexdigest()}}
    (folder / 'paint-receipt.json').write_text(json.dumps(receipt))
    return folder


class MetricA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fails_only_when_both_offsets_agree_and_paint_v4_resolves_the_edge(self):
        got = score.metric_a(paint_dir(self.dir, .35, .40), RECORD_SHA)
        self.assertEqual(got['outcome'], 'fail')
        self.assertIn('counts against the app', got['reason'])
        self.assertIn('about 0.20 m', got['reason'])
        self.assertEqual((got['evalL'], got['paintV4']['status']), (.5, 'resolved'))

    def test_a_would_be_fail_that_paint_v4_does_not_resolve_is_listed_not_scored(self):
        got = score.metric_a(paint_dir(self.dir, .35, .40, 'unresolved', 'Estimator A jackknife sd exceeds 0.075 m'), RECORD_SHA)
        self.assertEqual(got['outcome'], 'not scored')
        self.assertTrue(got['notConfirmedByPaintV4'])
        self.assertIn('jackknife', got['reason'])

    def test_disagreeing_estimators_are_named_as_such(self):
        for a, b in ((.35, .60), (-.35, .40)):  # opposite signs can never agree within 0.15 m
            got = score.metric_a(paint_dir(self.dir, a, b), RECORD_SHA)
            self.assertEqual(got['outcome'], 'not scored')
            self.assertTrue(got['estimatorsDisagree'])
            self.assertNotIn('unresolved', got)

    def test_small_or_missing_offsets_are_not_scored(self):
        for a, b in ((.19, .17), (.35, .25), (None, .40)):
            got = score.metric_a(paint_dir(self.dir, a, b), RECORD_SHA)
            self.assertEqual(got['outcome'], 'not scored', (a, b))
            self.assertFalse(got.get('notConfirmedByPaintV4') or got.get('estimatorsDisagree'))

    def test_a_result_not_made_by_paint_v4_is_refused(self):
        with self.assertRaises(SystemExit):
            score.metric_a(paint_dir(self.dir, .35, .40, script_sha='cd' * 32), RECORD_SHA)


class PaintHeldOutOut(unittest.TestCase):
    def test_out_cannot_escape_local_with_dot_dot(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(M7_DIR / 'paint_heldout.py'), '--site', 'nowhere',
                                 '--out', '.local/../scripts/m7-test-paint', '--rehearse'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--out must be a new folder under .local/', result.stderr)
        self.assertFalse((ROOT / 'scripts/m7-test-paint').exists())


if __name__ == '__main__':
    unittest.main()

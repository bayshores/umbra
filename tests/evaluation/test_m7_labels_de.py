"""M7 after the re-review of 2026-10-14: the D/E hook (labels addendum anchor, label manifest, D and E rules, E in H), the
summary's completeness checks, masked stop reasons and the chunked-listing download rule. Synthetic fixtures only, no AV2
data, no network, no git: the committed-anchor readers are replaced in each test.

Run: nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -B -m unittest discover -s tests/evaluation -p 'test_m7*.py'
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
common = score.common
SITE = {'logId': 'aaaaaaaa-0000-0000-0000-000000000000', 'crossingId': '1', 'role': 'held-out-blocked-test'}
GROUND = '11' * 32
ADDENDUM_SHA = '22' * 32
LABELS = '33' * 32
PINNED = {'file': 'scripts/evaluation/m7/score.py', 'sha256': hashlib.sha256((M7_DIR / 'score.py').read_bytes()).hexdigest()}
ADDENDUM = {'deCode': {'scorer': PINNED}}


def f_results(child, adult=()):
    """Metric F results with rows (setbackM, app state, F class)."""
    rows = lambda spec: [{'setbackM': s, 'state': st, 'class': c} for s, st, c in spec]  # noqa: E731
    return {'heights': {'child': {'rows': rows(child)}, 'adult': {'rows': rows(adult)}}, 'testedBox': {'setbackM': 0.1}}


def de_doc(d=None, e_rows=None, **over):
    doc = {'kind': score.DE_KIND, 'site': {'logId': SITE['logId'], 'crossingId': SITE['crossingId']}, 'groundSha256': GROUND,
           'labelsAddendumSha256': ADDENDUM_SHA, 'labels': {'sha256': LABELS}, 'scorer': dict(PINNED),
           'D': d or {'labelledVehicle': True, 'errorsPx1024': [{'feature': f, 'errorPx': 1.0} for f in score.D_FEATURES],
                      'unresolvedLabels': []},
           'E': {'labelledVehicle': True, 'rows': e_rows or [], 'assumption': 'roofline between stations'}}
    doc.update(over)
    return doc


def d_part(errors, unresolved=()):
    return {'labelledVehicle': True, 'errorsPx1024': [{'feature': f, 'errorPx': v} for f, v in errors], 'unresolvedLabels': list(unresolved)}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, doc):
        path = self.dir / name
        path.write_text(json.dumps(doc))
        return path


class LabelsDE(Tmp):
    def check(self, doc, labels_sha=None):
        return score.labels_de(self.write('de.json', doc), ADDENDUM, ADDENDUM_SHA, GROUND, SITE, labels_sha)

    def test_a_matching_file_is_accepted(self):
        self.assertEqual(self.check(de_doc(), LABELS)['labels']['sha256'], LABELS)

    def test_each_mismatch_is_refused(self):
        for change in ({'kind': 'other'}, {'site': {'logId': SITE['logId'], 'crossingId': '2'}}, {'groundSha256': '44' * 32},
                       {'labelsAddendumSha256': '55' * 32}, {'labels': {}}, {'labels': {'sha256': 'not-a-sha'}},
                       {'scorer': {'file': PINNED['file'], 'sha256': '66' * 32}}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.check(de_doc(**change))

    def test_labels_other_than_the_manifest_lists_are_refused(self):
        with self.assertRaises(SystemExit):
            self.check(de_doc(), '77' * 32)

    def test_a_pinned_scorer_that_changed_is_refused(self):
        with self.assertRaises(SystemExit):
            score.labels_de(self.write('de.json', de_doc(scorer={'file': PINNED['file'], 'sha256': '88' * 32})),
                            {'deCode': {'scorer': {'file': PINNED['file'], 'sha256': '88' * 32}}}, ADDENDUM_SHA, GROUND, SITE)


class MetricD(unittest.TestCase):
    def test_limits_are_inclusive(self):
        got = score.metric_d({'D': d_part([('L', 3.0), ('R', -3.0), ('S1', 6.0), ('S2', 1.0), ('S3', 0.0)])})
        self.assertEqual((got['outcome'], got['medianPx'], got['maxPx']), ('pass', 3.0, 6.0))

    def test_just_over_a_limit_fails(self):
        for errors in ([('L', 3.01), ('R', 3.01), ('S1', 3.01), ('S2', 0), ('S3', 0)],
                       [('L', 6.01), ('R', 0), ('S1', 0), ('S2', 0), ('S3', 0)]):
            with self.subTest(errors=errors):
                self.assertEqual(score.metric_d({'D': d_part(errors)})['outcome'], 'fail')

    def test_unresolved_labels_are_left_out_and_counted(self):
        got = score.metric_d({'D': d_part([('L', 1.0), ('R', 2.0)], ['S1', 'S2', 'S3'])})
        self.assertEqual((got['outcome'], got['unresolvedCount'], got['medianPx']), ('pass', 3, 1.5))
        got = score.metric_d({'D': d_part([], score.D_FEATURES)})
        self.assertEqual(got['outcome'], 'not scored')

    def test_a_malformed_d_is_refused(self):
        bad = [{'labelledVehicle': True, 'errorsPx1024': []},  # no unresolvedLabels
               d_part([('L', 1.0), ('R', 1.0)]),  # S1-S3 missing
               d_part([('L', 1.0), ('L', 1.0), ('R', 1.0), ('S1', 1.0), ('S2', 1.0)], ['S3']),  # L twice
               d_part([('L', float('nan')), ('R', 1.0), ('S1', 1.0), ('S2', 1.0), ('S3', 1.0)]),
               d_part([('L', True), ('R', 1.0), ('S1', 1.0), ('S2', 1.0), ('S3', 1.0)]),
               d_part([('X', 1.0), ('R', 1.0), ('S1', 1.0), ('S2', 1.0), ('S3', 1.0)])]
        for d in bad:
            with self.subTest(d=d), self.assertRaises(SystemExit):
                score.metric_d({'D': d})


class MetricE(unittest.TestCase):
    results = f_results([(0.0, 'blocked', 'supported'), (0.25, 'unknown', 'ambiguous')], [(0.0, 'unknown', 'ambiguous')])

    def rows(self, *states):
        keys = [('child', 0.0, 'blocked'), ('child', 0.25, 'unknown'), ('adult', 0.0, 'unknown')]
        return [{'height': h, 'setbackM': s, 'appState': a, 'e': e} for (h, s, a), e in zip(keys, states)]

    def test_resolved_not_blocked_where_the_app_says_blocked_is_a_false_blocked(self):
        got = score.metric_e(de_doc(e_rows=self.rows('not blocked', 'unresolved', 'unresolved')), self.results)
        self.assertEqual((got['outcome'], got['falseBlocked']), ('fail', [{'height': 'child', 'setbackM': 0.0}]))

    def test_no_false_blocked_passes_and_no_resolved_point_is_not_scored(self):
        self.assertEqual(score.metric_e(de_doc(e_rows=self.rows('blocked', 'not blocked', 'unresolved')), self.results)['outcome'], 'pass')
        self.assertEqual(score.metric_e(de_doc(e_rows=self.rows('unresolved', 'unresolved', 'unresolved')), self.results)['outcome'],
                         'not scored')

    def test_without_metric_f_rows_e_is_not_scored(self):
        got = score.metric_e(de_doc(e_rows=self.rows('blocked', 'blocked', 'blocked')), None)
        self.assertEqual(got['outcome'], 'not scored')
        self.assertIn('no metric F rows', got['reason'])

    def test_rows_that_do_not_match_metric_f_are_refused(self):
        rows = self.rows('blocked', 'blocked', 'blocked')
        cases = {'missing row': rows[:2], 'row twice': rows + rows[:1],
                 'other app state': [dict(rows[0], appState='unknown')] + rows[1:],
                 'extra row': rows + [{'height': 'child', 'setbackM': 0.5, 'appState': 'unknown', 'e': 'blocked'}],
                 'unknown state': [dict(rows[0], e='maybe')] + rows[1:]}
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                score.metric_e(de_doc(e_rows=bad), self.results)


class MetricHWithE(unittest.TestCase):
    def test_e_overrides_f_and_unresolved_e_leaves_f(self):
        results = f_results([(0.0, 'blocked', 'supported'), (0.25, 'blocked', 'ambiguous'), (0.5, 'unknown', 'supported'),
                             (0.75, 'blocked', 'unresolved')])
        rows = [{'height': 'child', 'setbackM': 0.0, 'e': 'not blocked'}, {'height': 'child', 'setbackM': 0.25, 'e': 'blocked'},
                {'height': 'child', 'setbackM': 0.5, 'e': 'unresolved'}, {'height': 'child', 'setbackM': 0.75, 'e': 'unresolved'}]
        got = score.metric_h(results, 'held-out-blocked-test', rows)
        child = got['child']
        # Reference: 0.0 removed by E, 0.25 added by E, 0.5 kept by F, 0.75 stays out (F unresolved, E unresolved).
        self.assertEqual((child['referenceBlocked'], child['reportedBlocked']), (2, 1))
        self.assertEqual((got['outcome'], got['share']), ('pass', .5))
        self.assertEqual(child['notReferenceListedSetbacksM'], {'unresolved': [0.75]})
        self.assertEqual(child['eUnresolvedSetbacksM'], [0.5, 0.75])


def site_score(name, e='not scored', d='not scored', mode='rehearsal (development site)', role='development', **extra):
    log, crossing = name.split('/')
    metrics = {m: {'outcome': 'not scored', 'reason': 'test'} for m in ('A', 'B', 'C', 'F', 'F0', 'H', 'I', 'J', 'K')}
    metrics['D'] = {'outcome': d, **({'errors': [{'feature': 'L', 'errorPx': 2.0}]} if d != 'not scored' else {})}
    metrics['E'] = {'outcome': e}
    doc = {'mode': mode, 'site': {'logId': log + '-0000', 'crossingId': crossing, 'role': role}, 'protocolLockSha256': None,
           'labelsAddendumSha256': None, 'build': {'outcome': 'built-shape'}, 'metrics': metrics,
           'inputs': {'failedSteps': []}, 'scorer': {'sha256': 'ab' * 32}}
    doc.update(extra)
    return doc


class Summary(Tmp):
    def summarize(self, *docs):
        return score.summarize(argparse.Namespace(scores=[self.write(f'{i}.json', d) for i, d in enumerate(docs)]))

    def test_false_blocked_bound_and_d_each_site(self):
        got = self.summarize(site_score('aaaaaaaa/1', e='pass', d='pass'), site_score('bbbbbbbb/2', e='fail', d='fail'),
                             site_score('cccccccc/3'))
        fb = got['statistics']['falseBlocked']
        self.assertEqual((fb['sites'], fb['falseBlockedSites']), (2, ['bbbbbbbb/2']))
        self.assertAlmostEqual(fb['upperBound95'], 0.95 ** 0.5, places=6)  # 1 error in 2 sites: 1 - p^2 = 0.05
        self.assertEqual((got['D']['scoredSites'], got['D']['eachSitePasses'], got['D']['failedSites']), (2, False, ['bbbbbbbb/2']))

    def test_mixed_scorer_or_addendum_is_refused(self):
        for extra in ({'scorer': {'sha256': 'cd' * 32}}, {'labelsAddendumSha256': ADDENDUM_SHA}):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.summarize(site_score('aaaaaaaa/1'), site_score('bbbbbbbb/2', **extra))


HELD = 'held-out'


class HeldOutSummary(Tmp):
    """Held-out completeness: paint record for every site; with an anchored addendum, D/E as the label manifest lists."""

    def run_summary(self, docs, manifest=None, addendum=ADDENDUM_SHA):
        split = [{'logId': d['site']['logId'], 'crossingId': d['site']['crossingId'], 'role': 'held-out-blocked-test'} for d in docs]
        with mock.patch.object(common, 'lost_sites', return_value=(set(), '00' * 32)), \
                mock.patch.object(common, 'split_sites', return_value=split), \
                mock.patch.object(common, 'labels_addendum', return_value=addendum), \
                mock.patch.object(common, 'labels_manifest', return_value=(manifest or {}, '99' * 32)):
            return score.summarize(argparse.Namespace(scores=[self.write(f'{i}.json', d) for i, d in enumerate(docs)]))

    def doc(self, name, labels=None, failed=(), paint=True):
        inputs = {'failedSteps': list(failed), **({'paint': {'file': 'p', 'sha256': '00' * 32}} if paint else {}),
                  **({'labelsDE': {'labelsSha256': labels}} if labels else {})}
        return site_score(name, mode=HELD, role='held-out-blocked-test', labelsAddendumSha256=ADDENDUM_SHA, inputs=inputs)

    def test_complete_inputs_pass(self):
        got = self.run_summary([self.doc('aaaaaaaa/1', LABELS), self.doc('bbbbbbbb/2'), self.doc('cccccccc/3', failed=['labels-de'])],
                               {'aaaaaaaa/1': LABELS, 'bbbbbbbb/2': None, 'cccccccc/3': LABELS})
        self.assertEqual(got['siteSet']['labelsManifest']['sha256'], '99' * 32)

    def test_a_missing_paint_record_is_refused(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_summary([self.doc('aaaaaaaa/1', paint=False)], {'aaaaaaaa/1': None})
        self.assertIn('paint record', str(stop.exception))

    def test_d_e_left_out_or_not_listed_is_refused(self):
        cases = {'labels listed, no D/E file': ([self.doc('aaaaaaaa/1')], {'aaaaaaaa/1': LABELS}),
                 'D/E file, no labels listed': ([self.doc('aaaaaaaa/1', LABELS)], {'aaaaaaaa/1': None}),
                 'other labels': ([self.doc('aaaaaaaa/1', '44' * 32)], {'aaaaaaaa/1': LABELS})}
        for name, (docs, manifest) in cases.items():
            with self.subTest(name), self.assertRaises(SystemExit) as stop:
                self.run_summary(docs, manifest)
            self.assertIn('label manifest', str(stop.exception))

    def test_scores_under_another_addendum_are_refused(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_summary([self.doc('aaaaaaaa/1')], addendum=None)
        self.assertIn('labels addendum', str(stop.exception))


class Anchors(Tmp):
    def test_held_out_needs_the_addendum_file_to_equal_its_anchor(self):
        path = self.write('addendum.json', {'x': 1})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(common.labels_addendum(False, path), digest)
        self.assertIsNone(common.labels_addendum(False, self.dir / 'none.json'))
        for anchored, file, ok in ((digest, path, True), (None, self.dir / 'none.json', True), (None, path, False),
                                   (digest, self.dir / 'none.json', False), ('ab' * 32, path, False), ('short', path, False)):
            with self.subTest(anchored=anchored, file=file.name), \
                    mock.patch.object(common, 'committed', return_value=({'labelsAddendumSha256': anchored}, '00' * 32)):
                if ok:
                    self.assertEqual(common.labels_addendum(True, file), anchored)
                else:
                    self.assertRaises(SystemExit, common.labels_addendum, True, file)

    def test_the_manifest_names_every_held_out_site_once(self):
        split = [{'logId': 'aaaaaaaa-x', 'crossingId': '1', 'role': 'held-out-blocked-test'},
                 {'logId': 'bbbbbbbb-x', 'crossingId': '2', 'role': 'held-out-control'},
                 {'logId': 'cccccccc-x', 'crossingId': '3', 'role': 'development'}]
        good = [{'logId': 'aaaaaaaa-x', 'crossingId': '1', 'labelsSha256': LABELS},
                {'logId': 'bbbbbbbb-x', 'crossingId': '2', 'labelsSha256': None, 'reason': 'vehicle not in the photo'}]
        bad = {'missing site': good[:1], 'no reason': [good[0], {**good[1], 'reason': ' '}], 'twice': good + good[:1],
               'bad sha': [{**good[0], 'labelsSha256': 'x'}, good[1]]}
        with mock.patch.object(common, 'split_sites', return_value=split):
            with mock.patch.object(common, 'committed', return_value=({'labelsAddendumSha256': ADDENDUM_SHA, 'sites': good}, '00' * 32)):
                self.assertEqual(common.labels_manifest(ADDENDUM_SHA)[0], {'aaaaaaaa/1': LABELS, 'bbbbbbbb/2': None})
                self.assertRaises(SystemExit, common.labels_manifest, '44' * 32)
            for name, sites in bad.items():
                with self.subTest(name), mock.patch.object(common, 'committed',
                                                           return_value=({'labelsAddendumSha256': ADDENDUM_SHA, 'sites': sites}, '00' * 32)):
                    self.assertRaises(SystemExit, common.labels_manifest, ADDENDUM_SHA)


class HeldOutSiteScore(Tmp):
    def test_a_site_built_under_another_addendum_is_refused(self):
        site = self.dir / 'site'
        site.mkdir()
        (site / 'site-stopped.json').write_text(json.dumps({'mode': HELD, 'site': SITE, 'outcome': 'stopped', 'labelsAddendumSha256': None,
                                                            'stopped': {'step': 'download', 'reason': 'test'}}))
        args = argparse.Namespace(site=str(site), failed_step=None, labels_de=None, baseline=None, metric_f_results=None,
                                  metric_k=None, paint=None)
        with mock.patch.object(common, 'labels_addendum', return_value=ADDENDUM_SHA), self.assertRaises(SystemExit):
            score.score_site(args)
        with mock.patch.object(common, 'labels_addendum', return_value=None):
            got = score.score_site(args)
        self.assertEqual((got['labelsAddendumSha256'], got['metrics']['D']['outcome']), (None, 'not scored'))


class StopsAndDownloads(Tmp):
    def test_stop_reasons_mask_the_banned_word(self):
        word = 'PA' + 'RK'  # built here so this file never holds the word itself
        self.assertEqual(common.SiteStop('x', f'a {word}ed car').reason, 'a p*rked car')

    def fake_curl(self, headers, body=b'<xml/>'):
        def run(argv, **_):
            out = Path(argv[argv.index('--output') + 1])
            out.write_bytes(body)
            Path(argv[argv.index('--dump-header') + 1]).write_text(headers)
            return subprocess.CompletedProcess(argv, 0)
        return run

    def test_a_chunked_listing_is_accepted_but_a_chunked_object_is_refused(self):
        chunked = 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nETag: "abc"\r\n'
        with mock.patch('subprocess.run', self.fake_curl(chunked)):
            _, receipt = common.fetch('https://example.invalid/list', self.dir / 'listing', 'l.xml', 1000)
            self.assertEqual(receipt['httpContentLength'], 'none (chunked)')
            with self.assertRaises(common.SiteStop) as stop:
                common.fetch('https://example.invalid/obj', self.dir / 'object', 'o.bin', 1000, {'bytes': 6, 'etag': 'abc'})
            self.assertIn('without Content-Length', stop.exception.reason)
            self.assertEqual(len(list((self.dir / 'object').iterdir())), common.RETRIES + 1)  # a new folder per attempt

    def test_an_object_with_matching_length_and_md5_is_accepted(self):
        body = b'<xml/>'
        md5 = hashlib.md5(body).hexdigest()
        headers = f'HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nETag: "{md5}"\r\n'
        with mock.patch('subprocess.run', self.fake_curl(headers, body)):
            _, receipt = common.fetch('https://example.invalid/obj', self.dir / 'object', 'o.bin', 1000, {'bytes': 6, 'etag': md5})
        self.assertEqual((receipt['bytes'], receipt['md5Check']), (6, 'single-part: MD5 == ETag'))


class OutCannotEscapeLocal(unittest.TestCase):
    def test_dot_dot_is_refused_by_every_driver(self):
        cases = {'run_heldout.py': ['--log', 'x', '--crossing', '1', '--rehearse', '--out', '.local/../scripts/m7-test-out'],
                 'evaluate_site.py': ['--site', 'nowhere', '--rehearse', '--out', '.local/../scripts/m7-test-out'],
                 'score.py': ['summary', '--scores', 'nowhere', '--out', '.local/../scripts/m7-test-out']}
        for tool, argv in cases.items():
            with self.subTest(tool):
                result = subprocess.run([sys.executable, '-I', '-B', str(M7_DIR / tool), *argv], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('--out must be', result.stderr)
                self.assertFalse((ROOT / 'scripts/m7-test-out').exists())


if __name__ == '__main__':
    unittest.main()

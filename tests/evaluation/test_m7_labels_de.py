"""M7 after the re-review of 2026-10-14: the D/E hook (labels addendum anchor, label manifest, D and E rules, E in H), the
summary's completeness checks, masked stop reasons and the chunked-listing download rule; after the second re-review
the written-once anchors, the anchors each site score records, one run per step (markers and the evaluate receipt) and
the coordinator-only folder kept out of written text. Synthetic fixtures only, no AV2 data, no network, no git: the
committed-anchor readers, preflight and git are replaced in each test.

Run: nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -B -m unittest discover -s tests/evaluation -p 'test_m7*.py'
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import io
import json
import os
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


def load_driver(name):
    """Import an m7 driver bound to this file's common and score modules (so their patches apply), without leaking names."""
    saved = {n: sys.modules.pop(n) for n in ('common', 'score', name) if n in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(M7_DIR))
    sys.modules.update(common=common, score=score)
    try:
        return importlib.import_module(name)
    finally:
        for n in ('common', 'score', name):
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
SITE = {'logId': 'aaaaaaaa-0000-0000-0000-000000000000', 'crossingId': '1', 'role': 'held-out-blocked-test'}
GROUND = '11' * 32
ADDENDUM_SHA = '22' * 32
LABELS = '33' * 32
LOST = {'file': common.LOST_SITES_REL, 'sha256': '00' * 32}  # the lost-site list every held-out test record names
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
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(common, 'de_code', return_value=dict(PINNED))  # the D/E code folder: see DECode
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, doc, labels_sha=None):
        return score.labels_de(self.write('de.json', doc), ADDENDUM, ADDENDUM_SHA, GROUND, SITE, labels_sha)

    def test_a_matching_file_is_accepted(self):
        self.assertEqual(self.check(de_doc(), LABELS)['labels']['sha256'], LABELS)

    def test_each_mismatch_is_refused(self):
        for change in ({'kind': 'other'}, {'site': {'logId': SITE['logId'], 'crossingId': '2'}}, {'groundSha256': '44' * 32},
                       {'labelsAddendumSha256': '55' * 32}, {'labels': {}}, {'labels': {'sha256': 'not-a-sha'}},
                       {'labels': {'sha256': LABELS, 'path': '/somewhere/labels.json'}},
                       {'scorer': {'file': PINNED['file'], 'sha256': '66' * 32}},
                       {'site': {'logId': SITE['logId'], 'crossingId': 1}}, {'site': {**de_doc()['site'], 'note': 'x'}},
                       {'extra': 'x'}, {'D': []}, {'E': None}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.check(de_doc(**change))

    def test_a_missing_key_is_refused(self):
        for key in sorted(score.DE_KEYS):
            doc = de_doc()
            del doc[key]
            with self.subTest(key), self.assertRaises(SystemExit):
                self.check(doc)

    def test_labels_other_than_the_manifest_lists_are_refused(self):
        with self.assertRaises(SystemExit):
            self.check(de_doc(), '77' * 32)

    def test_d_e_code_that_differs_from_the_addendum_stops_before_the_file_is_read(self):
        with mock.patch.object(common, 'de_code', side_effect=SystemExit('the D/E code differs')), self.assertRaises(SystemExit):
            self.check(de_doc(), LABELS)


class DECode(Tmp):
    """common.de_code: the D/E code folder holds exactly the files the labels addendum pins, unchanged, with the scorer."""

    def setUp(self):
        super().setUp()
        self.code = self.dir / 'de-code'
        self.code.mkdir()
        patcher = mock.patch.object(common, 'DE_CODE', self.code)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ('scorer.py', 'tasks.py'):
            (self.code / name).write_text(f'# {name}\n')
        self.files = {common.rel(p): common.sha_file(p) for p in sorted(self.code.iterdir())}
        self.scorer = {'file': common.rel(self.code / 'scorer.py'), 'sha256': self.files[common.rel(self.code / 'scorer.py')]}

    def check(self, files=None, scorer=None):
        return common.de_code({'deCode': {'files': self.files if files is None else files, 'scorer': scorer or self.scorer}})

    def refused(self, why, **over):
        with self.assertRaises(SystemExit) as stop:
            self.check(**over)
        self.assertIn(why, str(stop.exception))

    def test_the_pinned_folder_is_accepted(self):
        self.assertEqual(self.check(), self.scorer)

    def test_an_unpinned_file_is_refused(self):
        (self.code / 'extra.py').write_text('x = 1\n')
        self.refused('differ from the pinned ones')

    def test_a_changed_file_is_refused(self):
        (self.code / 'tasks.py').write_text('# changed\n')
        self.refused('differs from its sha256')

    def test_a_link_is_refused(self):
        (self.dir / 'outside.py').write_text('x = 1\n')
        (self.code / 'outside.py').symlink_to(self.dir / 'outside.py')
        self.refused('a link in the D/E code folder')

    def test_a_scorer_that_is_not_pinned_is_refused(self):
        for scorer in ({'file': PINNED['file'], 'sha256': PINNED['sha256']}, {**self.scorer, 'sha256': '88' * 32}):
            with self.subTest(scorer=scorer['file']):
                self.refused('the scorer is not one of the pinned D/E files', scorer=scorer)

    def test_no_pins_or_no_folder_is_refused(self):
        self.refused('differ from the pinned ones', files={})
        with mock.patch.object(common, 'DE_CODE', self.dir / 'none'):  # the pinned files exist, but not in the D/E folder
            self.refused('differ from the pinned ones')


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

    def test_missing_rows_or_assumption_is_refused_not_a_crash(self):
        rows = self.rows('blocked', 'blocked', 'blocked')
        for e in ({'labelledVehicle': True, 'assumption': 'x'}, {'labelledVehicle': True, 'rows': rows},
                  {'labelledVehicle': True, 'rows': {'a': 1}, 'assumption': 'x'}, {'labelledVehicle': True, 'rows': [1], 'assumption': 'x'}):
            with self.subTest(e=e), self.assertRaises(SystemExit):
                score.metric_e(de_doc(E=e), self.results)


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
        self.assertIsNone(got['labels'])  # rehearsal: no label manifest

    def test_mixed_scorer_or_addendum_is_refused(self):
        for extra in ({'scorer': {'sha256': 'cd' * 32}}, {'labelsAddendumSha256': ADDENDUM_SHA}):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.summarize(site_score('aaaaaaaa/1'), site_score('bbbbbbbb/2', **extra))

    def test_a_paint_stop_after_paint_v4_wrote_its_result_is_flagged_and_listed(self):
        paint = self.dir / 'paint'
        paint.mkdir()
        self.write('paint/paint-stopped.json', {'siteRecord': {'sha256': '00' * 32}, 'outcome': 'not-scored',
                                                'stopped': {'step': 'interrupted', 'reason': 'test'}})
        got = score.metric_a(paint, '00' * 32)
        self.assertEqual(got['outcome'], 'not scored')
        self.assertNotIn('stoppedAfterMeasuring', got)
        self.write('paint/result.json', {})
        flagged = score.metric_a(paint, '00' * 32)
        self.assertTrue(flagged['stoppedAfterMeasuring'])
        stopped = site_score('bbbbbbbb/2')
        stopped['metrics']['A'] = flagged
        self.assertEqual(self.summarize(site_score('aaaaaaaa/1'), stopped)['A']['stoppedAfterMeasuringSites'], ['bbbbbbbb/2'])


HELD = 'held-out'


class HeldOutSummary(Tmp):
    """Held-out completeness: paint record for every site; with an anchored addendum, D/E as the label manifest lists."""

    def run_summary(self, docs, manifest=None, addendum=ADDENDUM_SHA, lost=(set(), '00' * 32), runs=None):
        split = [{'logId': d['site']['logId'], 'crossingId': d['site']['crossingId'], 'role': 'held-out-blocked-test'} for d in docs]
        with mock.patch.object(common, 'preflight', return_value={}), \
                mock.patch.object(common, 'lost_sites', return_value=lost), \
                mock.patch.object(common, 'split_sites', return_value=split), \
                mock.patch.object(common, 'labels_addendum', return_value=addendum), \
                mock.patch.object(common, 'labels_manifest', return_value=(manifest or {}, '99' * 32)), \
                mock.patch.object(score, 'one_run', side_effect=runs or (lambda path, doc: [])):
            return score.summarize(argparse.Namespace(scores=[self.write(f'{i}.json', d) for i, d in enumerate(docs)]))

    def doc(self, name, labels=None, failed=(), paint=True, anchors=None):
        inputs = {'failedSteps': list(failed), **({'paint': {'file': 'p', 'sha256': '00' * 32}} if paint else {}),
                  **({'labelsDE': {'labelsSha256': labels}} if labels else {})}
        return site_score(name, mode=HELD, role='held-out-blocked-test', labelsAddendumSha256=ADDENDUM_SHA, inputs=inputs,
                          scorer={'sha256': PINNED['sha256']},
                          anchors=anchors or {'lostSiteList': '00' * 32, 'labelsManifest': '99' * 32})

    def test_complete_inputs_pass(self):
        got = self.run_summary([self.doc('aaaaaaaa/1', LABELS), self.doc('bbbbbbbb/2'), self.doc('cccccccc/3', failed=['labels-de'])],
                               {'aaaaaaaa/1': LABELS, 'bbbbbbbb/2': None, 'cccccccc/3': LABELS})
        self.assertEqual(got['siteSet']['labelsManifest']['sha256'], '99' * 32)

    def test_labelled_sites_without_d_or_e_are_listed(self):
        scored = self.doc('aaaaaaaa/1', LABELS)
        scored['metrics']['D'] = {'outcome': 'pass', 'errors': [{'feature': 'L', 'errorPx': 1.0}]}
        scored['metrics']['E'] = {'outcome': 'pass'}
        stopped = self.doc('bbbbbbbb/2')  # its evaluation stopped: no D/E file although the manifest lists labels
        stopped['inputs']['evaluationStopped'] = {'file': 'e', 'sha256': '00' * 32}
        stopped['metrics']['D'] = stopped['metrics']['E'] = {'outcome': 'not scored', 'reason': 'the evaluation stopped: x'}
        got = self.run_summary([scored, stopped, self.doc('cccccccc/3')], {'aaaaaaaa/1': LABELS, 'bbbbbbbb/2': LABELS, 'cccccccc/3': None})
        self.assertEqual(got['labels']['sitesWithLabels'], ['aaaaaaaa/1', 'bbbbbbbb/2'])
        self.assertEqual(got['labels']['dOrENotScored'], {'bbbbbbbb/2': {'D': 'the evaluation stopped: x', 'E': 'the evaluation stopped: x'}})

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

    def test_a_manifest_or_lost_list_changed_after_scoring_is_refused(self):
        # The reviewer's probe: a D fail removed by editing the manifest, a miss removed by adding it to the lost list.
        for name, anchors in (('manifest', {'lostSiteList': '00' * 32, 'labelsManifest': '98' * 32}),
                              ('lost list', {'lostSiteList': '01' * 32, 'labelsManifest': '99' * 32}), ('none recorded', None)):
            docs = [self.doc('aaaaaaaa/1')]
            docs[0]['anchors'] = anchors
            with self.subTest(name), self.assertRaises(SystemExit) as stop:
                self.run_summary(docs, {'aaaaaaaa/1': None})
            self.assertIn('lost-site list or label manifest', str(stop.exception))

    def test_a_score_from_another_score_py_is_refused(self):
        docs = [self.doc('aaaaaaaa/1')]
        docs[0]['scorer'] = {'sha256': 'cd' * 32}
        with self.assertRaises(SystemExit) as stop:
            self.run_summary(docs, {'aaaaaaaa/1': None})
        self.assertIn('locked one', str(stop.exception))

    def test_a_score_that_is_not_the_recorded_run_is_refused(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_summary([self.doc('aaaaaaaa/1')], {'aaaaaaaa/1': None}, runs=lambda path, doc: ['failed steps differ'])
        self.assertIn('one recorded run', str(stop.exception))


class OneRun(Tmp):
    """Held-out run markers and the evaluate receipt: one run of each step per site, failed steps from the receipt."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(common, 'RUN_MARKERS', self.dir / 'markers')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log, self.crossing = SITE['logId'], SITE['crossingId']
        for folder in ('site', 'paint', 'eval'):
            (self.dir / folder).mkdir()
        record = self.write('site/build-receipt.json', {'site': SITE})
        paint = self.write('paint/paint-receipt.json', {'x': 1})
        self.inputs = {'siteRecord': {'file': common.rel(record), 'sha256': common.sha_file(record)},
                       'paint': {'file': common.rel(paint), 'sha256': common.sha_file(paint)}, 'failedSteps': ['metric-k']}
        self.score = self.write('eval/score.json', {'site': SITE, 'inputs': self.inputs})
        driver = M7_DIR / 'evaluate_site.py'
        self.receipt = {'kind': 'pao-m7-site-evaluation', 'site': SITE, 'siteRecord': self.inputs['siteRecord'],
                        'steps': [{'step': 'baseline', 'exit': 0}, {'step': 'metric-k', 'exit': 1}, {'step': 'score', 'exit': 0}],
                        'outputs': {common.rel(self.score): common.sha_file(self.score)},
                        'driver': {'sha256': common.sha_file(driver)}}
        for step, folder in (('build', 'site'), ('paint', 'paint'), ('evaluate', 'eval')):
            common.run_marker(step, self.log, self.crossing, self.dir / folder)

    def check(self, receipt=None, inputs=None):
        self.write('eval/evaluate-receipt.json', receipt or self.receipt)
        return score.one_run(self.score, {'site': SITE, 'inputs': inputs or self.inputs})

    def test_the_recorded_run_passes(self):
        self.assertEqual(self.check(), [])

    def test_a_second_run_of_a_step_is_refused(self):
        for step in ('build', 'paint', 'evaluate'):
            with self.subTest(step), self.assertRaises(SystemExit) as stop:
                common.run_marker(step, self.log, self.crossing, self.dir / 'other')
            self.assertIn('never rerun', str(stop.exception))

    def test_a_failed_step_claim_not_in_the_receipt_is_refused(self):
        self.assertIn('failed steps differ from the evaluate receipt', self.check(inputs={**self.inputs, 'failedSteps': ['labels-de']}))

    def test_runs_other_than_the_marked_ones_are_refused(self):
        (self.dir / 'paint2').mkdir()
        other = self.write('paint2/paint-receipt.json', {'x': 1})
        got = self.check(inputs={**self.inputs, 'paint': {'file': common.rel(other), 'sha256': common.sha_file(other)}})
        self.assertIn('not the paint run the paint marker names', got)
        self.assertIn('evaluate receipt does not name this score', self.check({**self.receipt, 'outputs': {}}))


    def test_a_score_whose_score_step_failed_is_refused(self):
        receipt = {**self.receipt, 'steps': [{'step': 'metric-k', 'exit': 1}, {'step': 'score', 'exit': 2}]}
        self.assertIn('the score step of the evaluate receipt failed', self.check(receipt))

    def stopped(self, receipt=None, name='stopped-score.json', failed=(), **stop_over):
        receipt_path = self.dir / 'eval/evaluate-receipt.json'
        if receipt:
            self.write('eval/evaluate-receipt.json', receipt)
        named = {'file': common.rel(receipt_path), 'sha256': common.sha_file(receipt_path)} if receipt else None
        stop = self.write('eval/evaluate-stopped.json', {
            'kind': score.STOPPED_KIND, 'site': SITE, 'siteRecord': self.inputs['siteRecord'], 'evaluateReceipt': named,
            'driver': {'sha256': common.sha_file(M7_DIR / 'evaluate_site.py')}, **stop_over})
        inputs = {**self.inputs, 'failedSteps': list(failed), 'evaluationStopped': {'file': common.rel(stop), 'sha256': common.sha_file(stop)}}
        return score.one_run(self.write(f'eval/{name}', {'site': SITE, 'inputs': inputs}), {'site': SITE, 'inputs': inputs})

    def test_a_stopped_evaluation_is_accepted_without_a_receipt_or_with_a_failed_score_step(self):
        self.assertEqual(self.stopped(), [])
        self.assertEqual(self.stopped({**self.receipt, 'steps': [{'step': 'score', 'exit': 2}]}), [])

    def test_a_stopped_score_that_does_not_fit_its_run_is_refused(self):
        failed_score = {**self.receipt, 'steps': [{'step': 'score', 'exit': 2}]}
        cases = {'the marked evaluation was scored: it is not a stopped site': {'receipt': self.receipt},
                 'a stopped site lists failed steps': {'failed': ['metric-k']},
                 'a stopped score is not stopped-score.json': {'name': 'other-score.json'},
                 'evaluate-stopped.json for another site record': {'siteRecord': {'file': 'x', 'sha256': '00' * 32}},
                 'evaluate-stopped.json made by another evaluate_site.py': {'driver': {'sha256': '00' * 32}},
                 'evaluate-stopped.json names another evaluate receipt': {'receipt': failed_score, 'evaluateReceipt': None}}
        for why, kw in cases.items():
            (self.dir / 'eval/evaluate-receipt.json').unlink(missing_ok=True)
            with self.subTest(why):
                self.assertIn(why, self.stopped(**kw))


class CommittedOnce(Tmp):
    def test_an_anchor_with_a_second_commit_is_refused(self):
        path = self.write('anchor.json', {'lostSites': []})
        for count, ok in (('1\n', True), ('2\n', False)):
            git = lambda *args, count=count: count if args[0] == 'rev-list' else ''  # noqa: E731
            with self.subTest(count=count), mock.patch.object(common, 'git', side_effect=git), \
                    mock.patch.object(common.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)):
                if ok:
                    self.assertEqual(common.committed(str(path), 'x')[0], {'lostSites': []})
                else:
                    with self.assertRaises(SystemExit) as stop:
                        common.committed(str(path), 'x')
                    self.assertIn('written once', str(stop.exception))


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


class LostSites(unittest.TestCase):
    def lost(self, entries, stamp='2026-10-20T10:00:00-07:00'):
        with mock.patch.object(common, 'committed', return_value=({'lostSites': entries}, '00' * 32)), \
                mock.patch.object(common, 'git', return_value=stamp + '\n'):
            return common.lost_sites()

    def test_entries_need_evidence_and_a_date_no_later_than_the_commit_day(self):
        good = {'logId': 'aaaaaaaa-x', 'crossingId': '1', 'evidence': 'the log is no longer listed in the bucket', 'date': '2026-10-20'}
        self.assertEqual(self.lost([]), (set(), '00' * 32))
        self.assertEqual(self.lost([good])[0], {'aaaaaaaa/1'})
        self.assertEqual(self.lost([{**good, 'date': '2026-10-21'}], '2026-10-20T20:00:00-07:00')[0], {'aaaaaaaa/1'})  # 03:00 UTC on the 21st
        bad = {'after the commit day': [{**good, 'date': '2026-10-21'}], 'no evidence': [{**good, 'evidence': ' '}],
               'no date': [{k: v for k, v in good.items() if k != 'date'}], 'not a day': [{**good, 'date': '2026-02-30'}],
               'other format': [{**good, 'date': '20261020'}], 'twice': [good, good]}
        for name, entries in bad.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                self.lost(entries)


class EndedRun(Tmp):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(common, 'RUN_MARKERS', self.dir / 'markers')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log, self.crossing = SITE['logId'], SITE['crossingId']

    def ended(self):
        with self.assertRaises(SystemExit) as stop:
            common.ended_run('paint', self.log, self.crossing)
        return str(stop.exception)

    def test_only_a_marked_run_whose_process_is_gone_is_returned(self):
        self.assertIn('nothing to record', self.ended())
        common.run_marker('paint', self.log, self.crossing, self.dir / 'paint')
        self.assertIn('may still be running', self.ended())  # this test's own process
        with mock.patch.object(common.os, 'kill', side_effect=PermissionError):
            self.assertIn('may still be running', self.ended())
        with mock.patch.object(common.os, 'kill', side_effect=ProcessLookupError):
            self.assertEqual(common.ended_run('paint', self.log, self.crossing), (self.dir / 'paint').resolve())

    def test_a_process_id_reused_after_the_marker_counts_as_ended(self):
        started = common.started_utc(os.getpid())
        self.assertTrue(started is not None and started <= datetime.now(timezone.utc))
        common.run_marker('paint', self.log, self.crossing, self.dir / 'paint')
        later = datetime.now(timezone.utc) + timedelta(seconds=5)
        with mock.patch.object(common, 'started_utc', return_value=later):
            self.assertEqual(common.ended_run('paint', self.log, self.crossing), (self.dir / 'paint').resolve())
        with mock.patch.object(common, 'started_utc', return_value=None):  # ps cannot tell: still treated as running
            self.assertIn('may still be running', self.ended())

    def test_a_marker_naming_no_process_is_refused(self):
        path = common.marker_path('paint', self.log, self.crossing)
        path.parent.mkdir(parents=True)
        for pid in (None, 0, -5, '12'):
            path.write_text(json.dumps({'folder': 'x', 'pid': pid}))
            with self.subTest(pid=pid), mock.patch.object(common.os, 'kill', side_effect=ProcessLookupError):
                self.assertIn('names no process', self.ended())


class FetchOrigin(unittest.TestCase):
    def test_a_failed_fetch_is_tried_again(self):
        done = lambda code: subprocess.CompletedProcess([], code)  # noqa: E731
        with mock.patch.object(common.subprocess, 'run', side_effect=[done(1), done(0)]) as run, \
                mock.patch.object(common.time, 'sleep') as sleep:
            common.fetch_origin()
        self.assertEqual((run.call_count, sleep.call_count), (2, 1))
        with mock.patch.object(common.subprocess, 'run', return_value=done(128)) as run, \
                mock.patch.object(common.time, 'sleep') as sleep, self.assertRaises(SystemExit):
            common.fetch_origin()
        self.assertEqual((run.call_count, sleep.call_count), (common.FETCH_TRIES, common.FETCH_TRIES - 1))

    def test_a_hung_fetch_is_stopped_and_counts_as_a_failed_try(self):
        with mock.patch.object(common.subprocess, 'run', side_effect=subprocess.TimeoutExpired('git', 1)) as run, \
                mock.patch.object(common.time, 'sleep'), self.assertRaises(SystemExit):
            common.fetch_origin()
        self.assertEqual((run.call_count, run.call_args.kwargs['timeout']), (common.FETCH_TRIES, common.FETCH_SECONDS))

    def test_the_score_step_waits_for_the_network(self):
        with mock.patch.object(common, 'fetch_origin', side_effect=[SystemExit('down'), SystemExit('down'), None]) as fetch, \
                mock.patch.object(common.time, 'sleep') as sleep:
            self.assertTrue(common.wait_for_origin())
        self.assertEqual((fetch.call_count, sleep.call_count), (3, 2))
        with mock.patch.object(common, 'fetch_origin', side_effect=SystemExit('down')) as fetch, \
                mock.patch.object(common.time, 'sleep'), mock.patch.object(common.time, 'time', side_effect=[0, 0, 30, 61]):
            self.assertFalse(common.wait_for_origin(limit=60, pause=30))
        self.assertEqual(fetch.call_count, 3)


class HeldOutSiteScore(Tmp):
    def test_a_site_built_under_another_addendum_is_refused(self):
        site = self.dir / 'site'
        site.mkdir()
        (site / 'site-stopped.json').write_text(json.dumps({'mode': HELD, 'site': SITE, 'outcome': 'stopped', 'labelsAddendumSha256': None,
                                                            'lostSiteList': LOST, 'stopped': {'step': 'download', 'reason': 'test'}}))
        args = argparse.Namespace(site=str(site), failed_step=None, labels_de=None, baseline=None, metric_f_results=None,
                                  metric_k=None, paint=None, evaluation_stopped=None)
        with mock.patch.object(common, 'preflight', return_value={}), \
                mock.patch.object(common, 'lost_sites', return_value=(set(), '00' * 32)):
            with mock.patch.object(common, 'labels_addendum', return_value=ADDENDUM_SHA), self.assertRaises(SystemExit):
                score.score_site(args)
            with mock.patch.object(common, 'labels_addendum', return_value=None):
                got = score.score_site(args)
            self.assertEqual((got['labelsAddendumSha256'], got['metrics']['D']['outcome']), (None, 'not scored'))
            self.assertEqual(got['anchors'], {'lostSiteList': '00' * 32, 'labelsManifest': None})
        with mock.patch.object(common, 'preflight', return_value={}), mock.patch.object(common, 'labels_addendum', return_value=None), \
                mock.patch.object(common, 'lost_sites', return_value=({'aaaaaaaa/1'}, '00' * 32)), self.assertRaises(SystemExit) as stop:
            score.score_site(args)
        self.assertIn('lost-site list', str(stop.exception))
        with mock.patch.object(common, 'preflight', return_value={}), mock.patch.object(common, 'labels_addendum', return_value=None), \
                mock.patch.object(common, 'lost_sites', return_value=(set(), '11' * 32)), self.assertRaises(SystemExit) as stop:
            score.score_site(args)
        self.assertIn('built under another lost-site list', str(stop.exception))


LOCK = 'ab' * 32
NAME = f"{SITE['logId'][:8]}/{SITE['crossingId']}"


class StoppedSiteScore(Tmp):
    """score.py site --evaluation-stopped: handled like a site with no feasible frame, A from paint, nothing else used."""

    def setUp(self):
        super().setUp()
        self.site = self.dir / 'site'
        (self.site / 'grounded').mkdir(parents=True)
        self.ground = self.write('site/grounded/ground.json', {'g': 1})
        self.addendum = self.write('addendum.json', {'deCode': {}})

    def make(self, mode=HELD, addendum=None):
        record = self.write('site/build-receipt.json', {'mode': mode, 'site': SITE, 'outcome': 'built-shape', 'protocolLockSha256': LOCK,
                                                        'labelsAddendumSha256': addendum, 'lostSiteList': LOST,
                                                        'outputs': {common.rel(self.ground): common.sha_file(self.ground)}})
        self.stop = self.write('stop.json', {'kind': score.STOPPED_KIND, 'site': SITE, 'reason': 'the score step failed',
                                             'siteRecord': {'file': common.rel(record), 'sha256': common.sha_file(record)}})

    def score(self, addendum=None, **over):
        args = argparse.Namespace(**{**dict(site=str(self.site), failed_step=None, labels_de=None, baseline=None, metric_f_results=None,
                                            metric_k=None, paint=None, evaluation_stopped=str(self.stop)), **over})
        with mock.patch.object(common, 'preflight', return_value={}), mock.patch.object(common, 'labels_addendum', return_value=addendum), \
                mock.patch.object(common, 'lost_sites', return_value=(set(), '00' * 32)), \
                mock.patch.object(common, 'labels_manifest', return_value=({NAME: LABELS}, '99' * 32)), \
                mock.patch.object(score, 'LABELS_ADDENDUM', self.addendum):
            return score.score_site(args)

    def test_a_stopped_evaluation_scores_like_no_feasible_frame(self):
        for addendum, de_reason in ((None, score.NOT_SCORED_DE), (ADDENDUM_SHA, 'the evaluation stopped: the score step failed')):
            self.make(addendum=addendum)
            with self.subTest(addendum=addendum):
                got = self.score(addendum)
                metrics = got['metrics']
                self.assertEqual({k: v['outcome'] for k, v in metrics.items()},
                                 {'A': 'not scored', 'B': 'fail', 'C': 'not scored', 'D': 'not scored', 'E': 'not scored', 'F': 'not scored',
                                  'F0': 'not scored', 'H': 'fail', 'I': 'not scored', 'J': 'not scored', 'K': 'not scored'})
                self.assertIn('the evaluation stopped: the score step failed', metrics['H']['reason'])
                self.assertEqual(metrics['D']['reason'], de_reason)
                self.assertEqual(got['inputs']['evaluationStopped']['sha256'], common.sha_file(self.stop))

    def test_a_paint_record_under_another_lost_site_list_is_refused(self):
        self.make()
        paint = self.dir / 'paint'
        paint.mkdir()
        doc = {'siteRecord': {'sha256': common.sha_file(self.site / 'build-receipt.json')}, 'outcome': 'not-scored',
               'stopped': {'step': 'interrupted', 'reason': 'test'}}
        self.write('paint/paint-stopped.json', {**doc, 'lostSiteList': LOST})
        self.assertEqual(self.score(paint=str(paint))['metrics']['A']['outcome'], 'not scored')
        (paint / 'paint-stopped.json').write_text(json.dumps({**doc, 'lostSiteList': {**LOST, 'sha256': '11' * 32}}))
        with self.assertRaises(SystemExit) as stop:
            self.score(paint=str(paint))
        self.assertIn('another lost-site list', str(stop.exception))

    def test_other_inputs_rehearsal_or_another_site_record_are_refused(self):
        self.make()
        for over in ({'failed_step': ['baseline']}, {'baseline': 'x'}, {'metric_f_results': 'x'}, {'metric_k': 'x'}, {'labels_de': 'x'}):
            with self.subTest(over=over), self.assertRaises(SystemExit):
                self.score(**over)
        self.make(mode='rehearsal (development site)')
        self.assertRaises(SystemExit, self.score)
        self.make()
        self.write('stop.json', {**json.loads(self.stop.read_bytes()), 'siteRecord': {'file': 'x', 'sha256': '00' * 32}})
        self.assertRaises(SystemExit, self.score)


class HeldOutDriver(Tmp):
    """A held-out site with its build and paint markers; every check that needs the lock, git or anchors is replaced."""

    def setUp(self):
        super().setUp()
        self.dir = self.dir.resolve()
        outside = tempfile.TemporaryDirectory()  # the coordinator-only folder lies outside the (test) repository
        self.addCleanup(outside.cleanup)
        self.coord = Path(outside.name).resolve() / 'coordinator'
        self.coord.mkdir()
        self.labels = self.coord / 'labels.json'
        self.labels.write_text(json.dumps({'labels': 1}))
        self.addendum = self.write('addendum.json', {'deCode': {}})
        self.site, self.paint, self.out = self.dir / 'site', self.dir / 'paint', self.dir / '.local' / 'eval'
        patches = [mock.patch.object(common, 'RUN_MARKERS', self.dir / 'markers'),
                   mock.patch.object(common, 'guard', return_value=({'role': SITE['role']}, {})),
                   mock.patch.object(common, 'labels_addendum', return_value=ADDENDUM_SHA),
                   mock.patch.object(common, 'labels_manifest', return_value=({NAME: common.sha_file(self.labels)}, '99' * 32)),
                   mock.patch.object(common, 'lost_sites', return_value=(set(), '00' * 32)),
                   mock.patch.object(common, 'de_code', return_value={'file': 'de-scorer.py', 'sha256': '00' * 32}),
                   mock.patch.object(common, 'wait_for_origin', return_value=True),  # no network in tests
                   mock.patch.object(score, 'LABELS_ADDENDUM', self.addendum),
                   mock.patch.dict('os.environ', {common.COORDINATOR_ENV: str(self.coord)})]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        common.run_marker('build', SITE['logId'], SITE['crossingId'], self.site)

    def built(self):
        self.site.mkdir()
        self.record = self.write('site/build-receipt.json', {'mode': HELD, 'site': SITE, 'outcome': 'built-shape', 'protocolLockSha256': LOCK,
                                                             'labelsAddendumSha256': ADDENDUM_SHA, 'outputs': {}})

    def painted(self):
        common.run_marker('paint', SITE['logId'], SITE['crossingId'], self.paint)
        self.paint.mkdir()
        self.write('paint/paint-stopped.json', {'siteRecord': {'sha256': common.sha_file(self.record)}, 'outcome': 'not-scored',
                                                'stopped': {'step': 'interrupted', 'reason': 'test'}})

    def main(self, driver, argv, root=None):
        """Run driver.main(argv); returns (exit code, stderr and any SystemExit message)."""
        err = io.StringIO()
        with mock.patch.object(sys, 'argv', [driver.__name__, *map(str, argv)]), mock.patch.object(sys, 'stderr', err), \
                mock.patch.object(sys, 'stdout', io.StringIO()), mock.patch.object(driver, 'ROOT', root or common.ROOT), \
                self.assertRaises(SystemExit) as stop:
            driver.main()
        code = stop.exception.code
        return code, err.getvalue() + (code if isinstance(code, str) else '')

    def written(self):
        """Every file under the test folder except the coordinator-only folder and the markers."""
        return [p for p in self.dir.rglob('*') if p.is_file() and not p.is_relative_to(self.coord)]


class EvaluateHeldOut(HeldOutDriver):
    def setUp(self):
        super().setUp()
        self.ev = load_driver('evaluate_site')
        self.built()
        self.painted()

    def evaluate(self, site=None, paint=True, labels=True):
        argv = ['--site', site or self.site, '--out', self.out, '--protocol-lock-sha256', LOCK,
                *(['--paint', self.paint if paint is True else paint] if paint else []),
                *(['--labels', self.labels if labels is True else labels] if labels else [])]
        return self.main(self.ev, argv, root=self.dir)

    def refused(self, why, **kw):
        code, err = self.evaluate(**kw)
        self.assertEqual(code, 2, err)
        self.assertIn(why, err)
        self.assertFalse(common.marker_path('evaluate', SITE['logId'], SITE['crossingId']).exists())
        self.assertFalse(self.out.exists())

    def test_wrong_build_or_paint_folders_are_refused_before_the_marker(self):
        other = self.dir / 'other'
        other.mkdir()
        (other / 'build-receipt.json').write_bytes(self.record.read_bytes())
        self.refused('--site must be the folder the build marker names', site=other)
        self.refused('--paint must be the folder the paint marker names', paint=None)
        self.refused('--paint must be the folder the paint marker names', paint=other)

    def test_a_paint_record_from_another_site_record_or_none_is_refused(self):
        stopped = self.paint / 'paint-stopped.json'
        stopped.write_text(json.dumps({'siteRecord': {'sha256': '00' * 32}}))
        self.refused('made from another site record')
        stopped.unlink()
        self.refused('no single paint record')

    def test_labels_outside_the_coordinator_only_folder_are_refused(self):
        self.refused('must lie inside the coordinator-only folder', labels=self.write('labels.json', {'labels': 1}))  # same sha256

    def test_d_e_output_naming_the_labels_folder_is_a_failed_step(self):
        calls = []

        def run(argv, **_):
            calls.append(argv)
            if '--labels' in argv:  # the D/E scorer: its output names the labels folder
                Path(argv[argv.index('--out') + 1]).write_text(json.dumps({'labels': str(self.labels)}))
                return subprocess.CompletedProcess(argv, 0, '', '')
            return subprocess.CompletedProcess(argv, 1, '', f'cannot open {self.labels}')
        with mock.patch.object(self.ev.subprocess, 'run', side_effect=run):
            code, err = self.evaluate()
        self.assertEqual(code, 3, err)
        receipt = json.loads((self.out / 'evaluate-receipt.json').read_bytes())
        steps = {s['step']: s for s in receipt['steps']}
        self.assertEqual(steps['labels-de']['refused'], 'the D/E output names the coordinator-only folder or the labels folder')
        self.assertIn('labels-de', receipt['notScoredBecauseAStepFailed'])
        self.assertFalse((self.out / 'labels-de.json').exists())
        self.assertIn(('--failed-step', 'labels-de'), list(zip(calls[-1], calls[-1][1:])))
        self.assertEqual(steps['baseline']['error'], 'cannot open <coordinator-only folder>/labels.json')
        for path in self.written():
            self.assertNotIn(str(self.coord), path.read_text(), path)

    def test_a_d_e_output_score_py_refuses_fails_only_its_step(self):
        (self.site / 'grounded').mkdir()
        (self.site / 'grounded/ground.json').write_text('{}')
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            self.assertNotIn(common.COORDINATOR_ENV, kw['env'])  # no step is told where the coordinator-only folder is
            if '--labels' in argv:
                Path(argv[argv.index('--out') + 1]).write_text(json.dumps(de_doc(kind='other')))
            return subprocess.CompletedProcess(argv, 0, '', '')
        with mock.patch.object(self.ev.subprocess, 'run', side_effect=run):
            code, err = self.evaluate()
        self.assertEqual(code, 0, err)
        receipt = json.loads((self.out / 'evaluate-receipt.json').read_bytes())
        steps = {s['step']: s for s in receipt['steps']}
        self.assertTrue(steps['labels-de']['refused'].startswith('score.py refuses the D/E output: D/E file'), steps['labels-de'])
        self.assertNotIn('pao-m7-de-', steps['labels-de']['refused'])  # the temporary folder is masked too
        self.assertEqual((receipt['notScoredBecauseAStepFailed'], receipt['originWait']['gitFetchWorks']), (['labels-de'], True))
        self.assertFalse((self.out / 'labels-de.json').exists())
        self.assertIn(('--failed-step', 'labels-de'), list(zip(calls[-1], calls[-1][1:])))

    def test_a_coordinator_or_temporary_folder_inside_the_repository_is_refused(self):
        inside = self.dir / 'coordinator'
        inside.mkdir()
        labels = inside / 'labels.json'
        labels.write_bytes(self.labels.read_bytes())
        with mock.patch.dict('os.environ', {common.COORDINATOR_ENV: str(inside)}):
            self.refused('must lie outside the repository', labels=labels)
        with mock.patch.object(self.ev.tempfile, 'gettempdir', return_value=str(self.dir / 'tmp')):
            self.refused('the temporary folder (TMPDIR) lies inside the repository')

    def stopped(self):
        calls = []

        def run(argv, **_):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, '{}', '')
        with mock.patch.object(common.os, 'kill', side_effect=ProcessLookupError), mock.patch.object(self.ev.subprocess, 'run', side_effect=run):
            code, err = self.main(self.ev, ['--site', self.site, '--record-stopped', '--protocol-lock-sha256', LOCK], root=self.dir)
        return code, err, calls

    def test_a_marked_evaluation_that_ended_without_a_score_is_recorded_once(self):
        code, err, _ = self.stopped()
        self.assertIn('never ran', err)
        common.run_marker('evaluate', SITE['logId'], SITE['crossingId'], self.out)
        code, err, calls = self.stopped()
        self.assertEqual(code, 0, err)
        stop = json.loads((self.out / 'evaluate-stopped.json').read_bytes())
        self.assertEqual((stop['kind'], stop['evaluateReceipt'], stop['reason']),
                         (score.STOPPED_KIND, None, 'the marked evaluation ended without a receipt'))
        argv = calls[-1]
        self.assertEqual(argv[argv.index('--evaluation-stopped') + 1], str(self.out / 'evaluate-stopped.json'))
        self.assertEqual(argv[argv.index('--paint') + 1], str(self.paint))
        self.assertEqual(self.stopped()[0], 0)  # the record is reused while no stopped score exists
        (self.out / 'stopped-score.json').write_text('{}')
        self.assertIn('already scored', self.stopped()[1])

    def test_only_a_failed_or_missing_score_step_is_recorded(self):
        common.run_marker('evaluate', SITE['logId'], SITE['crossingId'], self.out)
        self.out.mkdir(parents=True)
        receipt = self.write('.local/eval/evaluate-receipt.json', {'steps': [{'step': 'score', 'exit': 0}]})
        self.assertIn('was scored', self.stopped()[1])
        receipt.write_text(json.dumps({'steps': [{'step': 'score', 'exit': 1, 'error': 'boom'}]}))
        code, err, _ = self.stopped()
        stop = json.loads((self.out / 'evaluate-stopped.json').read_bytes())
        self.assertEqual((code, stop['reason'], stop['evaluateReceipt']['sha256']), (0, 'the score step failed: boom', common.sha_file(receipt)))

    def test_a_running_evaluation_is_not_recorded(self):
        common.run_marker('evaluate', SITE['logId'], SITE['crossingId'], self.out)
        code, err = self.main(self.ev, ['--site', self.site, '--record-stopped', '--protocol-lock-sha256', LOCK], root=self.dir)
        self.assertIn('may still be running', err)
        self.assertFalse(self.out.exists())


class RecordInterrupted(HeldOutDriver):
    """paint_heldout.py and run_heldout.py --record-interrupted: a marked run that ended without its record."""

    def recorded(self, driver, argv):
        with mock.patch.object(common.os, 'kill', side_effect=ProcessLookupError):
            return self.main(driver, [*argv, '--record-interrupted', '--protocol-lock-sha256', LOCK])

    def test_a_paint_run_that_ended_without_a_record(self):
        paint = load_driver('paint_heldout')
        self.built()
        self.assertIn('never ran', self.recorded(paint, ['--site', self.site])[1])
        common.run_marker('paint', SITE['logId'], SITE['crossingId'], self.paint)
        code, err = self.recorded(paint, ['--site', self.site])
        doc = json.loads((self.paint / 'paint-stopped.json').read_bytes())
        self.assertEqual((code, doc['outcome'], doc['stopped']['step']), (3, 'not-scored', 'interrupted'))
        self.assertEqual((doc['siteRecord']['sha256'], doc['lostSiteList']), (common.sha_file(self.record), LOST))
        self.assertIn('has its paint record', self.recorded(paint, ['--site', self.site])[1])

    def test_a_build_that_ended_without_a_record(self):
        build = load_driver('run_heldout')
        argv = ['--log', SITE['logId'], '--crossing', SITE['crossingId']]
        code, err = self.recorded(build, argv)
        doc = json.loads((self.site / 'site-stopped.json').read_bytes())
        self.assertEqual((code, doc['mode'], doc['outcome'], doc['stopped']['step']), (3, HELD, 'stopped', 'interrupted'))
        self.assertEqual((doc['labelsAddendumSha256'], doc['lostSiteList']), (ADDENDUM_SHA, LOST))
        self.assertIn('has its site record', self.recorded(build, argv)[1])
        with mock.patch.object(common, 'lost_sites', return_value=({NAME}, '00' * 32)):
            self.assertIn('lost-site list', self.recorded(build, argv)[1])


class StopsAndDownloads(Tmp):
    def test_stop_reasons_mask_the_banned_word(self):
        word = 'PA' + 'RK'  # built here so this file never holds the word itself
        self.assertEqual(common.SiteStop('x', f'a {word}ed car').reason, 'a p*rked car')

    def test_the_coordinator_only_folder_is_never_written(self):
        folder = str(self.dir / 'coordinator')
        with mock.patch.dict('os.environ', {common.COORDINATOR_ENV: folder}):
            self.assertEqual(common.masked(f'open {folder}/labels.json and /x/y/z.json', ['/x/y']),
                             'open <coordinator-only folder>/labels.json and <coordinator-only folder>/z.json')
            self.assertNotIn(folder, common.SiteStop('x', f'{folder}/a').reason)
            with self.assertRaises(ValueError):
                common.write_new(self.dir / 'out.json', {'error': f'{folder}/labels.json'})
        self.assertFalse((self.dir / 'out.json').exists())

    def test_a_write_is_whole_or_absent(self):
        target = self.dir / 'r.json'
        with mock.patch.object(common, 'durable', side_effect=OSError('disk')), self.assertRaises(OSError):
            common.write_new(target, {'a': 1})
        self.assertEqual(list(self.dir.iterdir()), [])  # neither the file nor its temporary copy
        common.write_new(target, {'a': 1})
        self.assertEqual((json.loads(target.read_bytes()), target.stat().st_mode & 0o777), ({'a': 1}, 0o444))
        with self.assertRaises(FileExistsError):
            common.write_new(target, {'a': 2})
        self.assertEqual([p.name for p in self.dir.iterdir()], ['r.json'])

    def test_other_spellings_of_the_folder_are_caught(self):
        folder = self.dir / 'Coordinator-\u00e9'
        with mock.patch.dict('os.environ', {common.COORDINATOR_ENV: str(folder)}):
            plain = json.dumps({'x': f'{folder}/labels.json'}, ensure_ascii=False)
            for text in (json.dumps({'x': f'{folder}/labels.json'}), plain.replace('/', '\\/'), plain.upper()):
                with self.subTest(text=text):
                    self.assertTrue(common.names_folder(text.encode()))
                    with self.assertRaises(ValueError):
                        common.write_new(self.dir / 'out.json', text.encode())
            self.assertFalse(common.names_folder(b'{"x": "/somewhere/else"}'))
            self.assertEqual(common.masked(f'open {str(folder).upper()}/a'), 'open <coordinator-only folder>/a')
            env = common.child_env(A='1')
            self.assertEqual((common.COORDINATOR_ENV in env, env['A']), (False, '1'))
        self.assertTrue(common.names_folder(b'{"x": "some/labels/a.json"}', [common.ROOT / 'some' / 'labels']))  # relative to the repository
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_the_resolved_coordinator_folder_is_masked_and_refused_too(self):
        real = self.dir / 'real'
        real.mkdir()
        (self.dir / 'link').symlink_to(real)
        with mock.patch.dict('os.environ', {common.COORDINATOR_ENV: str(self.dir / 'link')}):
            self.assertEqual(common.masked(f'{real.resolve()}/labels.json'), '<coordinator-only folder>/labels.json')
            with self.assertRaises(ValueError):
                common.write_new(self.dir / 'out.json', {'error': f'{real.resolve()}/labels.json'})
        self.assertFalse((self.dir / 'out.json').exists())

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

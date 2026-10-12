"""Placement suggestion preparation on the real local DrivingStereo scene.

Run: .local/venv-preparation/bin/python -I -B -m unittest discover -s tests -p 'test_suggest_placements.py' -v
Every run works on a temporary copy of the scene; the shipped files are only read.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/suggest-placements.py'
SCENE_ID = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
SCENE = ROOT / 'public/scenes' / SCENE_ID
EPOCH = '1791676800'
# Iteration-1 crossing, measured world point of pixel [572,242] (independently verified, verify-iter1.json).
ITERATION_ONE_CROSSING = [3.517, -0.031, -29.409]
UNKNOWN_KEYS = {'status', 'reason'}
KEYS = {
    'top': {'schemaVersion', 'sceneId', 'generatedUtc', 'generator', 'inputs', 'crossing', 'roadAxis', 'curb',
            'vehicle', 'person', 'detection', 'limitations'},
    'generator': {'script', 'sha256', 'python', 'numpy', 'pillow'},
    'inputs': {'scene.json', 'recorded.jpg', 'depth.bin'},
    'crossing': {'pixel', 'status', 'method', 'alongRoadUncertaintyM', 'distanceFromCameraM', 'barCount'},
    'roadAxis': {'status', 'source', 'barAgreementDeg'},
    'curb': {'status', 'rmsM', 'stepHeightM'},
    'vehicle': {'pixel', 'status', 'label', 'dimensions', 'setbackM', 'reason'},
    'person': {'pixel', 'status', 'label', 'heightM', 'reason'},
    'detection': {'status', 'model', 'package', 'license', 'weightsSha256', 'detectionsSha256', 'curbLaneParked'},
}
KEYS['crossCheck'] = KEYS['detection']
# Labelled fixture model block (the block scripts/cloud/consume_nvdino_result.py carries); no real NV-DINO output.
NVDINO_MODEL = {'name': 'NV-Grounding-DINO', 'license': 'NVIDIA Open Model License',
                'package': 'NGC nvidia/tao/grounding_dino grounding_dino_swin_tiny_commercial_trainable_v1.1, run once '
                           'with NVIDIA TAO Toolkit 7.2.0',
                'weightsSha256': '8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6'}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def copy_scene(parent, detections=True):
    target = Path(parent) / SCENE_ID
    target.mkdir()
    for name in ('scene.json', 'recorded.jpg', 'depth.bin') + (('detections.json',) if detections else ()):
        shutil.copyfile(SCENE / name, target / name)
    return target


def run(scene_dir, *args, epoch=EPOCH):
    env = {**os.environ, 'SOURCE_DATE_EPOCH': epoch}
    return subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), str(scene_dir), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=120)


def suggest(scene_dir, *args):
    result = run(scene_dir, *args)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads((scene_dir / 'suggestions.json').read_text())


def edit_depth(scene_dir, change):
    """Remove measured depth (raw 0 = no measurement) and re-declare the buffer hash, as preparation would."""
    meta = json.loads((scene_dir / 'scene.json').read_text())
    c = meta['camera']
    raw = np.fromfile(scene_dir / 'depth.bin', dtype='<u2').reshape(c['height'], c['width'])
    change(raw)
    (scene_dir / 'depth.bin').write_bytes(raw.astype('<u2').tobytes())
    meta['artifactHashes']['depth'] = sha256_bytes((scene_dir / 'depth.bin').read_bytes())
    (scene_dir / 'scene.json').write_text(json.dumps(meta))


def measured_point(scene_dir, pixel):
    """src/geometry.mjs measuredPoint for a depth scene; None where there is no measured depth."""
    meta = json.loads((scene_dir / 'scene.json').read_text())
    c, t = meta['camera'], meta['transform']
    u, v = pixel
    raw = int(np.fromfile(scene_dir / 'depth.bin', dtype='<u2')[v * c['width'] + u])
    if not raw:
        return None, meta
    d = raw / meta['processing']['depthDivisor']
    p = np.array([(u - c['cx']) * d / c['fx'], (v - c['cy']) * d / c['fy'], d]) - np.array(t['cameraOriginGround'])
    return np.array(t['worldFromCamera']) @ p, meta


@unittest.skipUnless((SCENE / 'depth.bin').is_file(), 'prepared DrivingStereo scene is not present locally')
class SuggestPlacements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.scene = copy_scene(cls.tmp.name)
        cls.doc = suggest(cls.scene, '--detections', cls.scene / 'detections.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scratch(self, detections=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return copy_scene(tmp.name, detections)

    def assert_keys(self, doc):
        self.assertEqual(set(doc), KEYS['top'] | ({'crossCheck'} if 'crossCheck' in doc else set()))
        if 'crossCheck' in doc:
            self.assertEqual(set(doc['crossCheck']), KEYS['crossCheck'])
            self.assertEqual(doc['crossCheck']['status'], 'ran')
        self.assertEqual(doc['schemaVersion'], 1)
        self.assertEqual(set(doc['generator']), KEYS['generator'])
        self.assertEqual(set(doc['inputs']), KEYS['inputs'])
        for key in ('crossing', 'roadAxis', 'curb', 'vehicle', 'person'):
            item = doc[key]
            self.assertEqual(set(item), UNKNOWN_KEYS if item['status'] == 'unknown' else KEYS[key], key)
        expected = {'status'} if doc['detection']['status'] == 'not-run' else KEYS['detection']
        self.assertEqual(set(doc['detection']), expected)
        self.assertTrue(doc['limitations'] and all(isinstance(s, str) and s for s in doc['limitations']))

    def test_contract_shape_and_labels(self):
        doc = self.doc
        self.assert_keys(doc)
        self.assertEqual(doc['sceneId'], SCENE_ID)
        self.assertEqual(doc['generatedUtc'], '2026-10-11T00:00:00Z')
        self.assertEqual(doc['crossing']['status'], 'detected')
        self.assertEqual(doc['crossing']['alongRoadUncertaintyM'], 0.5)
        self.assertEqual(doc['vehicle']['status'], 'declared-default')
        self.assertEqual((doc['vehicle']['label'], doc['vehicle']['dimensions'], doc['vehicle']['setbackM']),
                         ('SUV', {'width': 1.9, 'height': 1.8, 'length': 4.9}, 6.096))
        self.assertIn('No parked vehicle detected', doc['vehicle']['reason'])
        self.assertEqual((doc['person']['status'], doc['person']['label'], doc['person']['heightM']),
                         ('declared-default', 'child', 1.1))
        self.assertEqual((doc['detection']['status'], doc['detection']['curbLaneParked']), ('ran', 0))
        self.assertEqual(doc['detection']['detectionsSha256'], sha256_bytes((self.scene / 'detections.json').read_bytes()))
        for name in KEYS['inputs']:
            self.assertEqual(doc['inputs'][name], sha256_bytes((SCENE / name).read_bytes()))

    def test_pixels_pass_road_point(self):
        points = {}
        for key in ('crossing', 'vehicle', 'person'):
            pixel = self.doc[key]['pixel']
            self.assertTrue(all(isinstance(n, int) for n in pixel), key)
            point, meta = measured_point(self.scene, pixel)
            self.assertIsNotNone(point, f'{key} has no measured depth')
            self.assertLessEqual(abs(point[1]), 2 * meta['processing']['roadToleranceMeters'], key)
            points[key] = point
        # placementFromPixels: separated front on the camera side of the crossing.
        delta = (points['vehicle'] - points['crossing'])[[0, 2]]
        setback = float(np.hypot(*delta))
        self.assertGreater(setback, meta['processing']['roadToleranceMeters'])
        self.assertAlmostEqual(setback, 6.096, delta=0.5)
        axis = np.array([delta[0], 0, delta[1]]) / setback
        self.assertGreater(float((np.array(meta['camera']['eye']) - points['crossing']) @ axis), 0)

    def test_crossing_within_half_metre_of_iteration_one(self):
        point, _ = measured_point(self.scene, self.doc['crossing']['pixel'])
        self.assertLessEqual(float(np.hypot(point[0] - ITERATION_ONE_CROSSING[0], point[2] - ITERATION_ONE_CROSSING[2])), 0.5)
        self.assertAlmostEqual(self.doc['crossing']['distanceFromCameraM'], 29.35, delta=0.5)

    def test_deterministic(self):
        again = self.scratch()
        suggest(again, '--detections', again / 'detections.json')
        self.assertEqual((again / 'suggestions.json').read_bytes(), (self.scene / 'suggestions.json').read_bytes())

    def test_shipped_file_reproduces_and_matches_catalog(self):
        shipped = (SCENE / 'suggestions.json').read_bytes()
        stamp = json.loads(shipped)['generatedUtc']
        copy = self.scratch()
        epoch = str(int(np.datetime64(stamp.rstrip('Z'), 's').astype('int64')))
        result = run(copy, '--detections', copy / 'detections.json', epoch=epoch)
        self.assertEqual(result.returncode, 0, result.stderr)
        again = (copy / 'suggestions.json').read_bytes()
        if again != shipped:
            # The only allowed difference: the shipped file predates the current script (generator.sha256), until
            # the next preparation run regenerates it. Every computed field must still reproduce.
            new, old = json.loads(again), json.loads(shipped)
            self.assertNotEqual(old['generator']['sha256'], sha256_bytes(SCRIPT.read_bytes()))
            self.assertEqual(new['generator']['sha256'], sha256_bytes(SCRIPT.read_bytes()))
            new['generator']['sha256'] = old['generator']['sha256']
            self.assertEqual(json.dumps(new, indent=2) + '\n', shipped.decode())
        catalog = json.loads((SCENE.parent / 'index.json').read_text())
        entry = next(e for e in catalog if e['id'] == SCENE_ID)
        self.assertEqual(entry['suggestions'], {'file': 'suggestions.json', 'sha256': sha256_bytes(shipped)})
        self.assertEqual(entry['label'], '2018-10-19-09-40-40-716')

    def test_refuses_overwrite(self):
        copy = self.scratch()
        suggest(copy)
        before = (copy / 'suggestions.json').read_bytes()
        result = run(copy, '--detections', copy / 'detections.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--replace', result.stderr)
        self.assertEqual((copy / 'suggestions.json').read_bytes(), before)
        self.assertEqual(run(copy, '--detections', copy / 'detections.json', '--replace').returncode, 0)
        self.assertNotEqual((copy / 'suggestions.json').read_bytes(), before)

    def test_without_detector_vehicle_is_still_a_labelled_default(self):
        doc = suggest(self.scratch(detections=False))
        self.assert_keys(doc)
        self.assertEqual(doc['detection'], {'status': 'not-run'})
        self.assertEqual(doc['vehicle']['status'], 'declared-default')
        self.assertIn('not run', doc['vehicle']['reason'])
        self.assertEqual(doc['vehicle']['pixel'], self.doc['vehicle']['pixel'])

    def test_all_unknown_when_depth_is_removed(self):
        copy = self.scratch(detections=False)
        edit_depth(copy, lambda raw: raw.fill(0))
        doc = suggest(copy)
        self.assert_keys(doc)
        for key in ('crossing', 'roadAxis', 'curb', 'vehicle', 'person'):
            self.assertEqual(doc[key]['status'], 'unknown', key)
            self.assertNotIn('pixel', doc[key])

    def test_crossing_unknown_when_edge_rows_lose_depth(self):
        copy = self.scratch(detections=False)
        edit_depth(copy, lambda raw: raw.__setitem__(slice(236, 251), 0))
        doc = suggest(copy)
        self.assert_keys(doc)
        self.assertEqual(doc['curb']['status'], 'detected')
        for key in ('crossing', 'vehicle', 'person'):
            self.assertEqual(doc[key]['status'], 'unknown', key)
        self.assertIn('near edge', doc['crossing']['reason'])

    def test_rejects_detections_from_other_inputs(self):
        copy = self.scratch()
        det = json.loads((copy / 'detections.json').read_text())
        det['inputs']['depth.bin'] = '0' * 64
        (copy / 'detections.json').write_text(json.dumps(det))
        result = run(copy, '--detections', copy / 'detections.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((copy / 'suggestions.json').exists())

    def test_rejects_an_image_other_than_the_declared_source(self):
        copy = self.scratch(detections=False)
        meta = json.loads((copy / 'scene.json').read_text())
        meta['sourceHashes']['left'] = '0' * 64
        (copy / 'scene.json').write_text(json.dumps(meta))
        result = run(copy)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sourceHashes.left', result.stderr)
        self.assertFalse((copy / 'suggestions.json').exists())

    def with_candidate(self, lower, body):
        """Labelled synthetic test fixture: one extra curb-lane candidate added to the real detections."""
        copy = self.scratch()
        det = json.loads((copy / 'detections.json').read_text())
        det['detections'].append({'id': len(det['detections']), 'class': 'car', 'cocoCategoryId': 3, 'score': 0.9,
                                  'box': [600.0, 230.0, 700.0, 280.0], 'lowerMedianWorld': lower,
                                  'bodyForwardRangeM': body, 'curbLaneCandidate': {'eligible': True, 'reasons': []}})
        (copy / 'detections.json').write_text(json.dumps(det))
        return copy, suggest(copy, '--detections', copy / 'detections.json')

    def test_detected_parked_vehicle_sets_the_front(self):
        copy, doc = self.with_candidate([3.4, 0.0, -14.0], [14.5, 19.0])
        self.assert_keys(doc)
        vehicle = doc['vehicle']
        # Only the front is detected; the label and size stay the declared default.
        self.assertEqual((vehicle['status'], vehicle['label'], vehicle['dimensions'], doc['detection']['curbLaneParked']),
                         ('detected', 'SUV', {'width': 1.9, 'height': 1.8, 'length': 4.9}, 1))
        self.assertIn('Parked car detected', vehicle['reason'])
        point, meta = measured_point(copy, vehicle['pixel'])
        self.assertLessEqual(abs(point[1]), 2 * meta['processing']['roadToleranceMeters'])
        self.assertAlmostEqual(point[2], -19.0, delta=0.5)
        self.assertAlmostEqual(vehicle['setbackM'], 29.4 - 19.0, delta=0.6)

    def cross_pair(self, primary_extra=(), check_extra=()):
        """Labelled fixtures: the shipped RF-DETR detections relabelled as the preferred detector, plus the
        shipped file itself as the cross-check; optional extra boxes are appended to either side."""
        copy = self.scratch()
        primary = json.loads((copy / 'detections.json').read_text())
        primary['model'] = dict(NVDINO_MODEL)
        primary.pop('run')
        check = json.loads((copy / 'detections.json').read_text())
        for doc, extra in ((primary, primary_extra), (check, check_extra)):
            for box in extra:
                doc['detections'].append({'id': len(doc['detections']), **box})
        (copy / 'primary.json').write_text(json.dumps(primary))
        (copy / 'check.json').write_text(json.dumps(check))
        return copy

    # Labelled synthetic boxes. Score 0.3: inside the curb lane but below the 0.5 candidate cut-off, so curbLaneParked
    # does not count it. LANE_CAR: an eligible curb-lane candidate between the camera and the crossing (counted).
    # FAR_CAR: eligible, but beyond the crossing (not counted).
    LANE_BOX = {'class': 'car', 'score': 0.3, 'box': [600.0, 230.0, 700.0, 280.0], 'lowerMedianWorld': [3.4, 0.0, -14.0],
                'bodyForwardRangeM': [14.5, 19.0], 'curbLaneCandidate': {'eligible': False, 'reasons': ['score 0.3 below 0.5']}}
    LANE_CAR = {'class': 'car', 'score': 0.9, 'box': [600.0, 230.0, 700.0, 280.0], 'lowerMedianWorld': [3.4, 0.0, -14.0],
                'bodyForwardRangeM': [14.5, 19.0], 'curbLaneCandidate': {'eligible': True, 'reasons': []}}
    NEXT_CAR = {**LANE_CAR, 'box': [500.0, 220.0, 560.0, 250.0], 'lowerMedianWorld': [3.0, 0.0, -20.0], 'bodyForwardRangeM': [20.0, 24.5]}
    FAR_CAR = {**LANE_CAR, 'lowerMedianWorld': [3.4, 0.0, -40.0], 'bodyForwardRangeM': [40.0, 44.0]}

    def test_cross_check_records_second_detector_with_the_same_fields(self):
        copy = self.cross_pair()
        doc = suggest(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'check.json')
        self.assert_keys(doc)
        self.assertEqual(list(doc)[-2:], ['crossCheck', 'limitations'])
        self.assertEqual(doc['detection']['model'], 'NV-Grounding-DINO')
        self.assertEqual(doc['detection']['detectionsSha256'], sha256_bytes((copy / 'primary.json').read_bytes()))
        self.assertEqual(doc['crossCheck'], {'status': 'ran', 'model': 'RF-DETR Medium', 'package': 'rfdetr 1.11.2',
                                             'license': 'Apache-2.0', 'weightsSha256': doc['crossCheck']['weightsSha256'],
                                             'detectionsSha256': sha256_bytes((copy / 'check.json').read_bytes()),
                                             'curbLaneParked': 0})
        self.assertEqual(doc['detection']['curbLaneParked'], 0)
        for key in ('crossing', 'roadAxis', 'curb', 'vehicle', 'person'):
            self.assertEqual(doc[key], self.doc[key], key)
        self.assertTrue(any('cross-checked against RF-DETR Medium' in s for s in doc['limitations']))

    def test_cross_check_agreement_uses_the_curb_lane_parked_rule(self):
        """Agreement means equal curbLaneParked counts, both computed by the same rule; both counts are recorded."""
        agree = (((self.LANE_BOX,), (), 0), ((), (self.LANE_BOX,), 0),       # below the candidate score: not counted
                 ((self.FAR_CAR,), (), 0),                                    # beyond the crossing: not counted
                 ((self.LANE_CAR,), (self.LANE_CAR,), 1))                     # counted by both
        for primary_extra, check_extra, count in agree:
            copy = self.cross_pair(primary_extra, check_extra)
            doc = suggest(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'check.json')
            self.assertEqual((doc['detection']['curbLaneParked'], doc['crossCheck']['curbLaneParked']), (count, count))
            self.assertEqual(doc['vehicle']['status'], 'detected' if count else 'declared-default')
        disagree = (((self.LANE_CAR,), ()), ((), (self.LANE_CAR,)), ((self.LANE_CAR, self.NEXT_CAR), (self.LANE_CAR,)))
        for primary_extra, check_extra in disagree:
            copy = self.cross_pair(primary_extra, check_extra)
            result = run(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'check.json')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('disagree', result.stderr)
            self.assertIn('curbLaneParked', result.stderr)
            self.assertIn('Stop and ask', result.stderr)
            self.assertFalse((copy / 'suggestions.json').exists())

    def test_cross_check_refusals(self):
        copy = self.cross_pair()
        same = run(copy, '--detections', copy / 'check.json', '--cross-check', copy / 'check.json')
        alone = run(copy, '--cross-check', copy / 'check.json')
        check = json.loads((copy / 'check.json').read_text())
        check['testOnly'] = True
        (copy / 'test-only.json').write_text(json.dumps(check))
        test_only = run(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'test-only.json')
        check['testOnly'] = False
        rule = json.loads(json.dumps(check))
        rule['rules']['candidateMinScore'] = 0.25
        (copy / 'rule.json').write_text(json.dumps(rule))
        other_rule = run(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'rule.json')
        self.assertIn('different curb-lane rules', other_rule.stderr)
        check['inputs']['recorded.jpg'] = '0' * 64
        (copy / 'other.json').write_text(json.dumps(check))
        other = run(copy, '--detections', copy / 'primary.json', '--cross-check', copy / 'other.json')
        for name, result in (('same', same), ('alone', alone), ('test-only', test_only), ('rule', other_rule), ('other', other)):
            self.assertNotEqual(result.returncode, 0, name)
        self.assertFalse((copy / 'suggestions.json').exists())

    def test_vehicle_beyond_crossing_is_ignored_and_one_reaching_it_is_unknown(self):
        _, beyond = self.with_candidate([3.4, 0.0, -40.0], [40.0, 44.0])
        self.assertEqual((beyond['vehicle']['status'], beyond['detection']['curbLaneParked']), ('declared-default', 0))
        _, reaching = self.with_candidate([3.4, 0.0, -25.0], [25.0, 29.2])
        self.assertEqual(reaching['vehicle']['status'], 'unknown')
        self.assertNotIn('pixel', reaching['vehicle'])


if __name__ == '__main__':
    unittest.main()

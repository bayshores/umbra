"""Isolated analytic fixtures; never real-scene/model/visibility evidence."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/evaluation'))
import matched


class MatchedTests(unittest.TestCase):
    def setUp(self):
        private = ROOT/'.local/worker-output/evaluation'
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.shape = [2, 3]
        def pin(name, data, **fields):
            (self.folder/name).write_bytes(data)
            return dict(path=name, bytes=len(data), sha256=matched.digest(data), **fields)
        self.pin = pin
        buffer = io.BytesIO()
        Image.fromarray(np.array([[2560, 5120, 0], [7680, 0, 10240]], dtype=np.uint16)).save(buffer, format='PNG')
        reference = pin('reference.png', buffer.getvalue(), shape=self.shape,
                        encoding='uint16png', divisor=256, quantity='axialDepthMeters')
        calibration = pin('calibration.txt', b'Isolated analytic test placeholder, not actual calibration')
        sources = [pin('left.bin', b'analytic left', role='left')]
        processing = dict(originalDimensions=self.shape, inputDimensions=self.shape, outputDimensions=self.shape,
            crop=None, pad=None, resize=None, coordinateConvention='OpenCV-left-rectified-x-right-y-down-z-forward',
            depthConvention='axialZ', normalization='analytic test, no inference')
        self.observation = dict(id='analytic-fixture', siteGroup='test-only', split='development',
            physicalSite=None, sources=sources, calibration=dict(artifact=calibration),
            admission=dict(status='unresolved', reasons=['analytic fixture only']),
            reference=dict(raster=reference, processing=processing, independent=False, limits=['analytic fixture'],
                method='analytic fixture only', timing='no real measurements', uncertainty='not applicable'))
        self.manifest = dict(schemaVersion=1, observations=[self.observation])
        self.prediction = dict(schemaVersion=1, observationId='analytic-fixture', sources=sources,
            model=dict(provider='analytic-fixture', id='no-model-run', version='test-only'), processing=processing,
            prediction=pin('prediction.bin', np.array([[11, 25, 9], [np.nan, 0, -5]], dtype='<f4').tobytes(),
                           shape=self.shape, encoding='float32le', quantity='axialDepthMeters'),
            validity=pin('validity.bin', np.ones(self.shape, dtype='u1').tobytes(), shape=self.shape, encoding='uint8'),
            execution=dict(status='succeeded', codeHashes={'fixture': 'analytic'}, environment={'fixture': True}))

    def evaluate(self):
        observations, prediction = self.folder/'observations.json', self.folder/'prediction.json'
        observations.write_text(json.dumps(self.manifest))
        prediction.write_text(json.dumps(self.prediction))
        return matched.evaluate(observations, prediction)[0]

    def test_full_reference_denominator_and_nonfinite(self):
        result = self.evaluate()
        counts = result['counts']
        self.assertEqual(counts['validReference'], 4)
        self.assertEqual(counts['matchedPrediction'], 2)
        self.assertEqual(counts['invalidPredictionOnReference'], 2)
        self.assertEqual(counts['predictionOutsideReference'], 1)
        metrics = result['metrics']
        self.assertEqual(metrics['absoluteErrorMeters']['mean'], 3)
        self.assertEqual(metrics['predictionCoverageOnFullReference'], .5)
        diagnostic = metrics['diagnostics']['absErrorGt2m']
        self.assertEqual((diagnostic['invalidOrBad'], diagnostic['fullReferenceDenominator']), (3, 4))
        self.assertEqual(diagnostic['withinToleranceCoverage'], .25)
        self.assertIsNone(metrics['visibility'])
        self.assertIsNone(result['thresholdLock'])
        self.assertEqual(sum(v['counts']['validReference'] for v in metrics['byReferenceRangeMeters'].values()), 4)

    def test_all_invalid_never_improved_accuracy(self):
        (self.folder/'validity.bin').write_bytes(b'\x00'*6)
        self.prediction['validity']['sha256'] = matched.digest(b'\x00'*6)
        result = self.evaluate()
        self.assertEqual(result['counts']['invalidPredictionOnReference'], 4)
        self.assertIsNone(result['metrics']['absoluteErrorMeters'])
        self.assertEqual(result['metrics']['diagnostics']['absErrorGt2m']['invalidOrBadRate'], 1)
        self.assertEqual(result['metrics']['predictionCoverageOnFullReference'], 0)

    def test_hash_mismatch_rejected(self):
        (self.folder/'prediction.bin').write_bytes(b'x'*24)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.evaluate()

    def test_shape_mismatch_rejected(self):
        self.prediction['validity']['shape'] = [3, 2]
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            self.evaluate()

    def test_encoding_mismatch_rejected(self):
        self.prediction['prediction']['encoding'] = 'float32be'
        with self.assertRaisesRegex(ValueError, 'float32le'):
            self.evaluate()

    def test_undeclared_external_rejected(self):
        self.observation['calibration']['artifact']['path'] = str(self.folder/'calibration.txt')
        with self.assertRaisesRegex(ValueError, 'readOnlyExternal'):
            self.evaluate()
        self.observation['calibration']['artifact']['readOnlyExternal'] = True
        self.evaluate()

    def test_no_implicit_crop_or_radial_conversion(self):
        self.prediction['processing'] = copy.deepcopy(self.prediction['processing'])
        self.prediction['processing']['crop'] = [0, 0, 3, 2]
        with self.assertRaisesRegex(ValueError, 'resampling'):
            self.evaluate()
        self.prediction['processing']['crop'] = None
        self.prediction['processing']['depthConvention'] = 'radialRange'
        with self.assertRaisesRegex(ValueError, 'axial-Z'):
            self.evaluate()

    def test_disparity_requires_pinned_conversion_adapter(self):
        self.prediction['prediction']['quantity'] = 'disparityPixels'
        with self.assertRaisesRegex(ValueError, 'Convert disparity explicitly'):
            self.evaluate()

    def test_actual_receipt_required(self):
        self.prediction['execution']['status'] = 'planned'
        with self.assertRaisesRegex(ValueError, 'actual execution'):
            self.evaluate()

    def test_model_revision_supported_without_version(self):
        del self.prediction['model']['version']
        self.prediction['model']['revision'] = 'analytic-test-only'
        self.evaluate()

    def test_prediction_input_mismatch_rejected(self):
        self.prediction['sources'] = [self.pin('other.bin', b'other image', role='left')]
        with self.assertRaisesRegex(ValueError, 'input hashes/roles'):
            self.evaluate()

    def test_other_observation_source_bytes_verified_before_split_audit(self):
        other = copy.deepcopy(self.observation)
        other.update(id='different-frame', siteGroup='different-group')
        other['sources'] = [self.pin('other-source.bin', b'original', role='left')]
        self.manifest['observations'].append(other)
        (self.folder/'other-source.bin').write_bytes(b'changed!')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.evaluate()

    def test_mask_values_rejected(self):
        invalid = b'\xff'*6
        (self.folder/'validity.bin').write_bytes(invalid)
        self.prediction['validity']['sha256'] = matched.digest(invalid)
        with self.assertRaisesRegex(ValueError, 'only 0 and 1'):
            self.evaluate()

    def test_split_group_leakage(self):
        other = copy.deepcopy(self.observation)
        other.update(id='another-frame', split='heldoutCandidate', developmentExposed=False)
        audit = matched.split_audit([self.observation, other])
        self.assertTrue(any('siteGroup' in failure for failure in audit['failures']))
        self.assertTrue(any('imageHash' in failure for failure in audit['failures']))
        self.assertTrue(any('physical site unresolved' in limit for limit in audit['independenceLimits']))

    def test_physical_site_leakage_even_with_different_groups(self):
        self.observation['physicalSite'] = {'id': 'analytic-site', 'evidence': 'test-only'}
        other = copy.deepcopy(self.observation)
        other.update(id='other', siteGroup='different-sequence', split='heldoutCandidate', developmentExposed=False)
        other['sources'] = [self.pin('other.bin', b'other', role='left')]
        audit = matched.split_audit([self.observation, other])
        self.assertTrue(any('physicalSite' in failure for failure in audit['failures']))

    def test_unknown_group_not_holdout_eligibility(self):
        self.observation.update(siteGroup=None, split='heldoutCandidate', developmentExposed=False)
        audit = matched.split_audit([self.observation])
        self.assertEqual(audit['groupCount'], 0)
        self.assertTrue(any('unknown siteGroup' in limit for limit in audit['independenceLimits']))
        self.assertTrue(any('model/pretraining' in limit for limit in audit['independenceLimits']))

    def test_holdout_execution_refused(self):
        self.observation.update(split='heldoutCandidate', developmentExposed=False)
        with self.assertRaisesRegex(ValueError, 'threshold lock'):
            self.evaluate()

    def test_non_development_refused_before_any_artifact_read(self):
        other = copy.deepcopy(self.observation)
        other.update(id='another-development', siteGroup='another-group',
                     sources=[self.pin('other.bin', b'other analytic source', role='left')])
        self.manifest['observations'].insert(0, other)
        for split in ('heldoutCandidate', 'unassigned'):
            self.observation.update(split=split, developmentExposed=False, modelExposure='verifiedAbsent')
            with self.subTest(split=split), mock.patch.object(matched, 'artifact', side_effect=AssertionError('artifact read')) as read, \
                    self.assertRaisesRegex(ValueError, 'threshold lock'):
                self.evaluate()
            read.assert_not_called()

    def test_duplicate_ids_and_json_keys(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate observation'):
            matched.split_audit([self.observation, self.observation])
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            matched.parse_json('{"x":1,"x":2}')

    def test_unresolved_not_correct_unknown_and_false_clear(self):
        cases = [dict(id='a', reference='Blocked', prediction='Clear', independent=True,
                      referenceEvidence=['analytic-test-only'], independenceGroup='g1'),
                 dict(id='b', reference='Clear', prediction='Unknown', independent=True,
                      referenceEvidence=['analytic-test-only'], independenceGroup='g1'),
                 dict(id='c', reference='unresolved', prediction='Unknown'),
                 dict(id='d', reference='Clear', prediction='Clear', independent=False)]
        scores = matched.visibility_scores(cases)
        self.assertEqual(scores['resolvableCases'], 2)
        self.assertEqual(scores['unresolvedCases'], 2)
        self.assertEqual(scores['independentGroups'], 1)
        self.assertEqual(scores['falseClearCount'], 1)
        self.assertEqual(scores['blockedReferenceDenominator'], 1)
        self.assertEqual(scores['correctUsefulCoverage'], 0)
        self.assertEqual(scores['riskAmongDecisions'], 1)
        self.assertIsNone(scores['falseClearConfidenceBound'])
        absent = matched.visibility_scores([dict(id='u', reference='unresolved', prediction='Unknown')])
        self.assertIsNone(absent['falseClearCount'])
        self.assertIsNone(absent['correctUsefulCoverage'])

    def test_empty_reference_no_metric(self):
        counts, metrics, _ = matched.depth_scores(np.ones((1, 2)), np.zeros((1, 2)), np.ones((1, 2), dtype=bool))
        self.assertEqual(counts['validReference'], 0)
        self.assertIsNone(metrics['absoluteErrorMeters'])
        self.assertIsNone(metrics['predictionCoverageOnFullReference'])

    def test_visibility_manifest_hash_binding_and_artifact_checks(self):
        self.evaluate()
        evidence = self.pin('analytic-label-evidence.txt', b'analytic fixture, not physical labels')
        cases = dict(schemaVersion=1, observationId=self.observation['id'],
            predictionManifestHash=matched.digest((self.folder/'prediction.json').read_bytes()),
            predictionEvidence=self.pin('analytic-prediction-evidence.txt', b'analytic fixture, no model run'),
            cases=[dict(id='fixture-label', reference='Blocked', prediction='Unknown', independent=True,
                independenceGroup='analytic-fixture', condition='analytic', referenceEvidence=[evidence])])
        path = self.folder/'visibility.json'
        path.write_text(json.dumps(cases))
        result, _ = matched.evaluate(self.folder/'observations.json', self.folder/'prediction.json', path)
        scores = result['metrics']['visibility']
        self.assertEqual(scores['correctUsefulCoverage'], 0)
        self.assertEqual(scores['unknownOnResolvable'], 1)
        self.assertEqual(scores['byCondition']['analytic']['blockedReferenceDenominator'], 1)
        self.assertIsNone(scores['riskAmongDecisions'])
        cases['predictionManifestHash'] = 'wrong'
        path.write_text(json.dumps(cases))
        with self.assertRaisesRegex(ValueError, 'must bind'):
            matched.evaluate(self.folder/'observations.json', self.folder/'prediction.json', path)


if __name__ == '__main__':
    unittest.main()

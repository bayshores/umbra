"""Analytic receipt/NPY regression fixtures, never street or sponsor evidence.

Run with python -B -m unittest discover -s tests/evaluation -p test_replay_saved.py -v.
Tiny fixtures and rejection receipts are preserved under the exclusive private
replay-regressions directory. No original inference, terms or cloud dispatch.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
import uuid

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'scripts/evaluation'))
import replay_saved

PRIVATE = ROOT/'.local/worker-output/evaluation/replay-regressions'
CALIBRATION = '''# ANALYTIC TEST ONLY, not sensor calibration or metric truth
S_rect_101: 8 4
S_rect_103: 8 4
P_rect_101: 100 0 4 0 0 100 2 0 0 0 1 0
P_rect_103: 100 0 2 -240 0 100 2 0 0 0 1 0
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inventory(*directories):
    return {str(path): sha(path) for directory in directories
            for path in sorted(directory.rglob('*')) if path.is_file()}


class ReplaySavedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_root = PRIVATE/('analytic-'+uuid.uuid4().hex)
        cls.run_root.mkdir(parents=True)
        write_json(cls.run_root/'scope.json', dict(evidenceClass='isolated analytic regression fixtures only',
            baseCommit='a702f1fa25ff8b5192301a056ae6a30d6d1d0768',
            codePins={path.name: sha(path) for path in (ROOT/'scripts/evaluation/replay_saved.py', ROOT/'scripts/evaluation/matched.py')},
            testCodeHash=sha(Path(__file__)), newInferenceRuns=0, cloudRequests=0,
            performanceMeasured=False, av2TermsAndPayloads='declined; no AV2 access'))

    @classmethod
    def tearDownClass(cls):
        print(f'Analytic replay fixture evidence: {cls.run_root}')

    def setUp(self):
        self.folder = self.run_root/self._testMethodName
        self.source = self.folder/'source'
        self.baseline_root = self.folder/'original-receipt'
        self.source.mkdir(parents=True)
        self.baseline_root.mkdir()
        self.output = self.folder/'output'
        self.corpus_path = self.source/'corpus.json'
        self.baseline_path = self.baseline_root/'report.json'
        script = self.baseline_root/'evaluate-stereo-baseline.py'
        script.write_text('# ANALYTIC ORIGINAL RECEIPT FIXTURE. No inference executed.\n')
        self.corpus = dict(schemaVersion=1, corpusVersion='development-v2',
            evidenceClass='analytic fixture; never an acquired development corpus', records=[])
        self.baseline = dict(evidenceClass='analytic original-receipt fixture; no model execution',
            script_sha256=sha(script), run_artifact_hashes={script.name: sha(script)},
            environment=dict(opencv='analytic-fixture-no-opencv-run', numpy=np.__version__),
            parameters=dict(minDisparity=0), frames=[])
        # Hand-solvable d = 1 + 120/Z; source-full calibration above scales 0.5
        # to the 4x2 raster, so the explicit scaled numerator is 120, offset 1.
        data = [
            ('analytic-a', [[5, 7, -1, 1], [4, 11, 0, 13]],
             [[28, 18, 50, 60], [42, 0, 70, 12]], [2., 2., -2., -2.], True),
            ('analytic-b', [[11, 3, -1, -1], [0, -1, 1, -1]],
             [[10, 20, 0, 0], [0, 0, 0, 0]], [2., 40.], False),
        ]
        self.golden = {}
        for identity, disparity_values, reference_values, signed_values, accepted in data:
            source = self.source/identity
            saved = self.baseline_root/identity
            source.mkdir()
            saved.mkdir()
            for role, value in [('left', 40), ('right', 60)]:
                Image.fromarray(np.full((2, 4), value, dtype='u1')).save(source/(role+'.jpg'), format='JPEG')
            reference = np.array(reference_values, dtype=np.uint16)
            Image.fromarray(reference*256).save(source/'depth.png', format='PNG')
            (source/'calibration.txt').write_text(CALIBRATION)
            (source/'LICENSE').write_text('ANALYTIC FIXTURE ONLY. No dataset terms accepted.\n')
            names = dict(left='left.jpg', right='right.jpg', depth='depth.png',
                         calibration='calibration.txt', license='LICENSE')
            raw_sources = {role: dict(path=str((source/name).relative_to(self.source)),
                bytes=(source/name).stat().st_size, sha256=sha(source/name)) for role, name in names.items()}
            self.corpus['records'].append(dict(id=identity, leakageGroup='analytic-one-sequence',
                evaluationUse='development-only; no final holdout eligibility', verifiedCalifornia=False,
                rawSources=raw_sources, preparationAccepted=accepted))
            for name in ('left.jpg', 'right.jpg', 'depth.png'):
                (saved/name).write_bytes((source/name).read_bytes())
            (saved/'scaled_full-calibration.txt').write_bytes((source/'calibration.txt').read_bytes())
            disparity = np.array(disparity_values, dtype=np.float32)
            raw_disparity = (disparity*16).astype(np.int16)
            support = (disparity > -1) & (disparity-1 > 0)
            depth = np.full(disparity.shape, np.nan, dtype=np.float32)
            depth[support] = 120/(disparity[support]-1)
            np.save(saved/'disparity-px.npy', disparity)
            np.save(saved/'disparity-int16.npy', raw_disparity)
            np.save(saved/'scaled_full-depth-m.npy', depth)
            calibration = dict(source_sha256=raw_sources['calibration']['sha256'],
                source_rectified_size=[8., 4.], scale=[.5, .5, 1.],
                disparity_numerator_px_m=120., principal_point_offset_px=1.,
                left_projection=[[50., 0., 2., 0.], [0., 50., 1., 0.], [0., 0., 1., 0.]],
                right_projection=[[50., 0., 1., -120.], [0., 50., 1., 0.], [0., 0., 1., 0.]])
            signed = np.array(signed_values, dtype=np.float64)
            self.golden[identity] = dict(signed=signed, support=support, depth=depth,
                referenceCount=int((reference > 0).sum()), matchedCount=len(signed_values))
            self.baseline['frames'].append(dict(stem=identity, width=4, height=2,
                input_hashes={role: dict(sha256=raw_sources[role]['sha256']) for role in ('left', 'right', 'depth')},
                output_hashes={path.name: sha(path) for path in sorted(saved.iterdir())},
                conventions=dict(scaled_full=dict(calibration=calibration, depth_m=dict(
                    reference_points=int((reference > 0).sum()), valid_predictions=len(signed_values),
                    absolute_error=dict(mean=float(np.abs(signed).mean())))))))
        self.save_receipts()

    def save_receipts(self):
        write_json(self.corpus_path, self.corpus)
        write_json(self.baseline_path, self.baseline)

    def repin_saved(self, name, identity='analytic-a'):
        self.baseline['frames'][0 if identity == 'analytic-a' else 1]['output_hashes'][name] = sha(self.baseline_root/identity/name)
        self.save_receipts()

    def run_replay(self, *, baseline_hash=None, corpus_hash=None):
        before = inventory(self.source, self.baseline_root)
        write_json(self.folder/'source-inventory-before.json', before)
        stdout = io.StringIO()
        outcome = dict(evidenceClass='analytic fixture regression only',
            corpusHash=sha(self.corpus_path), baselineHash=sha(self.baseline_path),
            outputReportExists=False, newInferenceRuns=0, cloudRequests=0)
        try:
            with contextlib.redirect_stdout(stdout):
                result = replay_saved.replay(self.source, self.corpus_path, corpus_hash or sha(self.corpus_path),
                    self.baseline_path, baseline_hash or sha(self.baseline_path), self.output)
            outcome['status'] = 'accepted analytic fixture'
            return result
        except Exception as error:
            outcome.update(status='rejected', errorType=type(error).__name__, error=str(error))
            raise
        finally:
            after = inventory(self.source, self.baseline_root)
            outcome['sourceInventoryPreserved'] = before == after
            outcome['outputReportExists'] = (self.output/'report.json').exists()
            write_json(self.folder/'source-inventory-after.json', after)
            write_json(self.folder/'receipt.json', outcome)
            (self.folder/'stdout.txt').write_text(stdout.getvalue())
            self.assertEqual(after, before, 'Replay changed original analytic source or receipt bytes')

    def assert_rejected(self, message, **kwargs):
        with self.assertRaisesRegex(ValueError, message):
            self.run_replay(**kwargs)
        self.assertFalse((self.output/'report.json').exists(), 'Rejected replay published a success report')

    def test_success_pools_errors_and_full_reference_denominators(self):
        report = self.run_replay()
        self.assertEqual(report['counts']['validReference'], 9)
        self.assertEqual(report['counts']['matchedPrediction'], 6)
        self.assertEqual(report['counts']['invalidPredictionOnReference'], 3)
        self.assertEqual(report['counts']['preparationAccepted'], 1)
        self.assertEqual(report['counts']['preparationRejected'], 1)
        self.assertEqual(report['metrics']['predictionCoverageOnFullReference'], 6/9)
        self.assertAlmostEqual(report['metrics']['absoluteErrorMeters']['mean'], 50/6)
        self.assertEqual(report['metrics']['invalidOrAbsErrorGt2m'], dict(count=4, denominator=9, rate=4/9))
        self.assertEqual(report['metrics']['positiveErrorGt10m'], 1)
        self.assertEqual(report['metrics']['positiveErrorGt100m'], 0)
        pooled = np.fromfile(self.output/'pooled-signed-errors-float64le.bin', dtype='<f8')
        np.testing.assert_array_equal(pooled, [2, 2, -2, -2, 2, 40])
        self.assertIsNone(report['thresholdLock'])
        self.assertIsNone(report['counts']['falseClearCount'])
        self.assertIsNone(report['metrics']['usefulPhysicalCoverage'])
        for record in report['results']:
            result_path = self.output/record['path']
            self.assertEqual(sha(result_path), record['sha256'])
            result = json.loads(result_path.read_text())
            golden = self.golden[result['observationId']]
            np.testing.assert_array_equal(np.fromfile(result_path.parent/'signed-errors-float64le.bin', dtype='<f8'), golden['signed'])
            self.assertEqual(result['counts']['validReference'], golden['referenceCount'])
            self.assertEqual(result['counts']['matchedPrediction'], golden['matchedCount'])
            self.assertEqual(sum(s['counts']['validReference'] for s in result['metrics']['byReferenceRangeMeters'].values()), golden['referenceCount'])
            self.assertFalse(result['referenceAdequacy']['independentMetricReference'])

    def test_success_preserves_saved_float32_and_exact_validity(self):
        self.run_replay()
        for identity, golden in self.golden.items():
            folder = self.output/identity
            self.assertEqual((folder/'depth-float32le.bin').read_bytes(), golden['depth'].astype('<f4').tobytes())
            self.assertEqual((folder/'validity-uint8.bin').read_bytes(), golden['support'].astype('u1').tobytes())
            prediction = json.loads((folder/'prediction.json').read_text())
            self.assertEqual(prediction['execution']['usage'], dict(cloudRequests=0, newInferenceRuns=0))
            self.assertEqual(prediction['execution']['originalReceipt']['sha256'], sha(self.baseline_path))

    def test_wrong_original_report_hash_rejected_before_output(self):
        self.assert_rejected('Replay input hash mismatch', baseline_hash='0'*64)
        self.assertFalse(self.output.exists())

    def test_wrong_corpus_hash_rejected_before_output(self):
        self.assert_rejected('Replay input hash mismatch', corpus_hash='0'*64)
        self.assertFalse(self.output.exists())

    def test_original_script_hash_mismatch_rejected_before_output(self):
        (self.baseline_root/'evaluate-stereo-baseline.py').write_text('# Changed analytic source\n')
        self.assert_rejected('Replay input hash mismatch')
        self.assertFalse(self.output.exists())

    def test_changed_saved_npy_without_repin_rejected(self):
        path = self.baseline_root/'analytic-a/disparity-px.npy'
        values = np.load(path, allow_pickle=False)
        values[0, 0] += .25
        np.save(path, values)
        self.assert_rejected('Replay input hash mismatch')

    def test_repinned_float_disparity_does_not_match_int16_div16(self):
        path = self.baseline_root/'analytic-a/disparity-px.npy'
        values = np.load(path, allow_pickle=False)
        values[0, 0] += .25
        np.save(path, values)
        self.repin_saved('disparity-px.npy')
        self.assert_rejected('fixed-point binding mismatch')

    def test_repinned_int16_disparity_does_not_match_float(self):
        path = self.baseline_root/'analytic-a/disparity-int16.npy'
        values = np.load(path, allow_pickle=False)
        values[0, 0] += 1
        np.save(path, values)
        self.repin_saved('disparity-int16.npy')
        self.assert_rejected('fixed-point binding mismatch')

    def test_declared_numerator_does_not_produce_saved_depth(self):
        self.baseline['frames'][0]['conventions']['scaled_full']['calibration']['disparity_numerator_px_m'] = 121.
        self.save_receipts()
        self.assert_rejected('does not match recorded explicit disparity conversion')

    def test_declared_offset_changes_support_and_is_rejected(self):
        self.baseline['frames'][0]['conventions']['scaled_full']['calibration']['principal_point_offset_px'] = 0.
        self.save_receipts()
        self.assert_rejected('invalidity disagrees with actual matcher support')

    def test_repinned_depth_values_disagree_with_conversion(self):
        path = self.baseline_root/'analytic-a/scaled_full-depth-m.npy'
        values = np.load(path, allow_pickle=False)
        values[0, 0] += 1
        np.save(path, values)
        self.repin_saved('scaled_full-depth-m.npy')
        self.assert_rejected('does not match recorded explicit disparity conversion')

    def test_original_reference_denominator_mismatch_rejected(self):
        self.baseline['frames'][0]['conventions']['scaled_full']['depth_m']['reference_points'] = 6
        self.save_receipts()
        self.assert_rejected('Full reference/matched denominators disagree')

    def test_original_valid_denominator_mismatch_rejected(self):
        self.baseline['frames'][0]['conventions']['scaled_full']['depth_m']['valid_predictions'] = 3
        self.save_receipts()
        self.assert_rejected('Full reference/matched denominators disagree')

    def test_original_mae_mismatch_rejected(self):
        self.baseline['frames'][0]['conventions']['scaled_full']['depth_m']['absolute_error']['mean'] = 99.
        self.save_receipts()
        self.assert_rejected('MAE disagrees with original')


if __name__ == '__main__':
    unittest.main()

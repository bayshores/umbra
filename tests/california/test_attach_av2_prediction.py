"""Analytic unit fixtures only. No successful sponsor execution is fabricated."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('attach_av2_prediction', ROOT / 'scripts/california/attach_av2_prediction.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
ARTIFACTS = Path(os.environ.get('SIGHTLINE_TEST_ARTIFACT_ROOT',
                               ROOT / '.local/evaluation/native-nvidia-attachment-2026-10-05/producer'))


def calibration(offset=0):
    left = [[10., 0, 3., 0], [0, 20., 2., 0], [0, 0, 1., 0]]
    right = [[10., 0, 3. - offset, -10.], [0, 20., 2., 0], [0, 0, 1., 0]]
    return dict(decoded_source_size=[800, 400], source_rectified_size=[800, 400], source_to_decoded_scale=[1., 1.],
                crop_size=[736, 320], original=dict(left_projection=left, right_projection=right),
                transformed=dict(left_projection=left, right_projection=right), disparity_numerator_px_m=10.,
                principal_point_offset_px=offset, provisional=True)


class NativePredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = ARTIFACTS / ('analytic-unit-' + str(uuid4()))
        self.directory.mkdir(parents=True)
        (self.directory / 'TEST-ONLY.txt').write_text('Analytic unit fixtures only. No dataset, inference or sponsor execution evidence.\n')
        self.camera = dict(fx=10., fy=20., cx=3., cy=2., height=320, width=736, near=1., far=20.)
        self.disparity = np.full(tool.SHAPE, np.nan, '<f4')
        self.disparity[3, 8] = 2.  # Selected Z5.
        self.disparity[4, 9] = 1.  # Selected Z10.
        self.disparity[2, 0] = 1.  # Right input center -1, rejected independently.
        self.disparity[1, 3] = 1e-5  # Full raw Z1,000,000 tail, outside far.
        self.disparity[0, 200] = 100.  # Positive Z.1 below near, right input supported.
        self.disparity[5, 10] = -1.
        self.disparity[5, 11] = 0.
        self.disparity[5, 12] = np.inf
        y, x = np.indices(tool.SHAPE)
        self.rgb = np.stack((x % 256, y % 256, (x + y) % 256), -1).astype('u1')
        self.rotation = np.array([[0., 0, 1.], [0, -1., 0], [1., 0, 0]])

    def convert(self, cal=None, validity=None, camera=None, rotation=None, rgb=None):
        mask = (np.isfinite(self.disparity) & (self.disparity > 0)).astype('u1')
        return tool.points(self.disparity, mask if validity is None else validity, calibration() if cal is None else cal,
                           self.camera if camera is None else camera, self.rotation if rotation is None else rotation,
                           self.rgb if rgb is None else rgb)

    def test_exact_integer_centers_row_major_world_positions_and_color(self):
        depth, positions, colors, report = self.convert()
        expected_camera = np.array([[2.5, .25, 5.], [6., 1., 10.]])
        np.testing.assert_array_equal(positions, (expected_camera @ self.rotation.T).astype('<f4'))
        srgb = self.rgb[[3, 4], [8, 9]].astype(float) / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4).astype('<f4')
        np.testing.assert_array_equal(colors, linear)
        self.assertEqual(positions.dtype, np.dtype('<f4')); self.assertEqual(colors.dtype, np.dtype('<f4'))
        self.assertEqual(report['displayedPixels'], 2)
        self.assertEqual(depth[3, 8], 5.); self.assertEqual(depth[4, 9], 10.)

    def test_full_raw_tails_and_diagnostic_denominators_survive_display_clipping(self):
        depth, _, _, report = self.convert()
        self.assertEqual(report['totalPixels'], 235520)
        self.assertEqual(report['rawValidityPixels'], 5); self.assertEqual(report['numericDepthPixels'], 5)
        self.assertEqual(report['rawDepthMaximumMeters'], 1_000_000.)
        self.assertEqual(report['rawDepthMinimumMeters'], float(np.float32(.1)))
        self.assertEqual(report['rightDomainRejectedPixels'], 1)
        self.assertEqual(report['outsideDeclaredCameraClipPixels'], 2)
        self.assertEqual(report['withinDeclaredCameraClipPixels'], 3)
        self.assertEqual(report['belowNearPixels'], 1); self.assertEqual(report['aboveFarPixels'], 1)
        self.assertEqual(report['nonpositiveFiniteDisparityPixels'], 2)
        self.assertTrue(np.isnan(depth[5, 10:13]).all())
        # The full-size Float32 raster retains all unsupported NaNs and the large finite tail.
        saved = np.frombuffer(depth.tobytes(), '<f4').reshape(tool.SHAPE)
        np.testing.assert_array_equal(saved.view('u4'), depth.view('u4'))
        self.assertEqual(saved[1, 3], 1_000_000.)

    def test_nonzero_offset_uses_denominator_and_preserves_raw_validity(self):
        self.disparity[:] = np.nan
        self.disparity[0, 3:6] = [1., 2., 3.]
        depth, positions, _, report = self.convert(calibration(2.))
        self.assertTrue(np.isnan(depth[0, 3:5]).all()); self.assertEqual(depth[0, 5], 10.)
        self.assertEqual(len(positions), 1)
        self.assertEqual(report['rawValidityPixels'], 3); self.assertEqual(report['numericDepthPixels'], 1)
        self.assertEqual(report['nonpositiveDenominatorPixels'], 2)

    def test_float32_overflow_and_underflow_are_unsupported_not_hidden_raw_filters(self):
        self.disparity[1, 3] = np.nextafter(np.float32(0), np.float32(1))
        depth, _, _, report = self.convert()
        self.assertTrue(np.isnan(depth[1, 3])); self.assertEqual(report['unrepresentableDepthPixels'], 1)
        tiny = calibration(); tiny['transformed']['right_projection'][0][3] = -1e-300
        tiny['disparity_numerator_px_m'] = 1e-300
        with self.assertRaisesRegex(ValueError, 'No bounded native display points'): self.convert(tiny)

    def test_exact_inclusive_clips_and_right_input_boundaries(self):
        self.disparity[:] = np.nan
        self.disparity[0, 10] = 10.  # right center 0 and near Z1 are included.
        self.disparity[0, 735] = .5  # far Z20 is included.
        _, positions, _, report = self.convert()
        self.assertEqual(len(positions), 2); self.assertEqual(report['displayedPixels'], 2)
        self.assertEqual(report['outsideDeclaredCameraClipPixels'], 0)

    def test_tail_and_right_rejection_counts_overlap_without_changing_raw_depth(self):
        self.disparity[1, 0] = 1e-5
        depth, _, _, report = self.convert()
        self.assertEqual(report['numericDepthPixels'], 6)
        self.assertEqual(report['outsideDeclaredCameraClipPixels'], 3)
        self.assertEqual(report['rightDomainRejectedPixels'], 2)
        self.assertEqual(report['outsideClipAndRightDomainPixels'], 1)
        self.assertEqual(report['numericRejectedFromDisplayPixels'], 4)
        self.assertEqual(depth[1, 0], 1_000_000.)

    def test_masks_encodings_clips_empty_and_nonfinite_output_fail(self):
        validity = (np.isfinite(self.disparity) & (self.disparity > 0)).astype('u1'); validity[3, 8] = 0
        for kwargs in [dict(validity=validity), dict(camera={**self.camera, 'near': True}),
                       dict(camera={**self.camera, 'far': float('inf')}),
                       dict(rotation=np.full((3, 3), np.inf)), dict(rgb=self.rgb.astype('f4'))]:
            with self.subTest(kwargs=list(kwargs)):
                with self.assertRaises(ValueError): self.convert(**kwargs)
        self.disparity = self.disparity.astype('f8')
        with self.assertRaises(ValueError): self.convert()
        self.disparity = np.full(tool.SHAPE, np.nan, '<f4')
        with self.assertRaisesRegex(ValueError, 'No bounded native display points'): self.convert()

    def fixture_binding(self):
        prepared_path = self.directory / 'manifest.json'
        sources = {}
        for role in ('left', 'right', 'lidar', 'poses', 'intrinsics', 'extrinsics', 'map', 'groundRaster', 'groundTransform', 'license'):
            data = ('test-only-' + role).encode(); path = self.directory / role; path.write_bytes(data)
            sources[role] = dict(path=role, bytes=len(data), sha256=tool.common.digest(data))
        identity = np.eye(4).tolist()
        pose = deepcopy(identity); pose[0][3], pose[1][3], pose[2][3] = 20., -30., 4.
        cal = calibration(); crop = dict(x=0, y=0, width=736, height=320, source_to_crop_matrix=np.eye(3).tolist())
        proc = dict(R1=np.eye(3).tolist(), R2=np.eye(3).tolist(), P1=cal['original']['left_projection'],
                    P2=cal['original']['right_projection'], rightCameraFromLeftCamera=identity, additionalDistortion=[0.] * 5,
                    alpha=0, rectifiedSize=[800, 400], KLeft=np.eye(3).tolist(), KRight=np.eye(3).tolist(),
                    egoFromCameraLeft=identity, egoFromCameraRight=identity, cityFromEgoCamera=pose,
                    sourceHashes={role: pin['sha256'] for role, pin in sources.items()})
        prepared = dict(sourceClass='Argoverse2Sensor', dataset='Argoverse2Sensor', logId='test-only-log',
                        cameraTimestampNs='100', lidarTimestampNs='90', crossingId='8', sources=sources, processing=proc,
                        crop=crop, outputs={'references.npz': {'sha256': 'a' * 64},
                                            'calibration.json': {'sha256': tool.common.digest(tool.common.canonical(cal))}})
        world = [[1., 0, 0], [0, 0, 1.], [0, -1., 0]]
        rotation = np.array(world)
        scene = dict(sourceClass='Argoverse2Sensor', units='meters', camera={**self.camera, 'eye': [0, 0, 0],
                            'forward': rotation[:, 2].tolist(), 'up': (-rotation[:, 1]).tolist()},
                     sourceHashes=proc['sourceHashes'], transform=dict(worldFromCity=world, worldFromCamera=world,
                         cameraOrigin=[0, 0, 0], originRole='leftCamera', cityOrigin=[20., -30., 4.]),
                     measurements=dict(kind='indexedSparse', groundPolicy='unvalidated', provenance=dict(
                         referencesSha256='a' * 64, originalSources=sources,
                         **{key: prepared[key] for key in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId')})))
        producer = dict(model={'id': 'TEST-ONLY'}, cacheKey='b' * 64, prediction={'sha256': 'c' * 64},
                        validity={'sha256': 'd' * 64}, observationId='av2:test-only-log:100:8')
        proof = dict(status='PASS_exact_saved_prediction_input_equivalence_no_inference', model=producer['model'],
                     oldCacheKey=producer['cacheKey'], historicalOutputSha256='c' * 64, historicalValiditySha256='d' * 64,
                     newCalibrationSha256=prepared['outputs']['calibration.json']['sha256'])
        rect = {key: proc[key] for key in ('R1', 'R2', 'P1', 'P2', 'rightCameraFromLeftCamera', 'additionalDistortion', 'alpha')}
        rect.update(outputSize=proc['rectifiedSize'], rawK={'left': proc['KLeft'], 'right': proc['KRight']},
                    egoFromCamera={'left': identity, 'right': identity})
        old_crop = dict(crop=crop, processing=dict(logId='test-only-log', cameraTimestampNs='100', lidarTimestampNs='90',
                         rectification=rect, sourcePins={role: pin for role, pin in sources.items()
                                                      if role not in ('groundRaster', 'groundTransform', 'license')}))
        return scene, prepared, cal, old_crop, deepcopy(cal), proof, producer, prepared_path

    def test_current_sources_calibration_pose_and_proof_bind_separately(self):
        args = self.fixture_binding(); capture = tool.Capture()
        tool.bind_native(*args, capture)
        self.assertEqual(len(capture.files), 10)
        for mutate in [lambda a: a[0]['camera'].__setitem__('cx', 3.5),
                       lambda a: a[0]['transform'].__setitem__('cameraOrigin', [0, 1, 0]),
                       lambda a: a[0]['transform'].__setitem__('cityOrigin', [0, 0, 0]),
                       lambda a: a[0]['measurements']['provenance'].__setitem__('referencesSha256', 'e' * 64),
                       lambda a: a[0]['camera'].__setitem__('forward', [1, 0, 0]),
                       lambda a: a[1]['processing'].__setitem__('R1', [[1., 0, 0], [0, -1., 0], [0, 0, -1.]]),
                       lambda a: a[5].__setitem__('historicalOutputSha256', 'e' * 64),
                       lambda a: a[6].__setitem__('observationId', 'av2:other:100:8')]:
            changed = deepcopy(args); mutate(changed)
            with self.assertRaises(ValueError): tool.bind_native(*changed, tool.Capture())
        (self.directory / 'left').write_bytes(b'changed-source')
        with self.assertRaises(ValueError): tool.bind_native(*args, tool.Capture())

    def test_direct_input_uses_original_alias_and_same_measured_guards(self):
        args = self.fixture_binding()
        scene, prepared, cal, _, _, _, producer, path = args
        scene['id'] = producer['observationId'] = 'test-only-current-alias'
        direct = [scene, prepared, cal, producer, path, path]
        capture = tool.Capture()
        tool.bind_direct(*direct, capture)
        self.assertEqual(len(capture.files), 10)
        for mutate in [lambda a: a[0].__setitem__('id', 'other'),
                       lambda a: a[3].__setitem__('observationId', 'other'),
                       lambda a: a.__setitem__(5, self.directory / 'different-manifest'),
                       lambda a: a[0]['camera'].__setitem__('fx', 999),
                       lambda a: a[0]['transform'].__setitem__('cityOrigin', [0, 0, 0]),
                       lambda a: a[1]['processing']['sourceHashes'].__setitem__('left', 'f' * 64),
                       lambda a: a[0]['measurements'].__setitem__('groundPolicy', 'validated'),
                       lambda a: a[0]['measurements']['provenance'].__setitem__('cameraTimestampNs', '101')]:
            changed = deepcopy(direct); mutate(changed)
            with self.assertRaises(ValueError): tool.bind_direct(*changed, tool.Capture())

    def test_input_binding_modes_are_explicit_and_exclusive_before_reads(self):
        missing = self.directory / 'missing'
        args = [missing, 'a' * 64] * 4 + [None, None, missing, self.directory / 'never-published']
        with self.assertRaisesRegex(ValueError, 'Exclusive input binding'): tool.attach(*args)
        for mode in (None, 1, 'directNative'):
            with self.assertRaisesRegex(ValueError, 'Explicit binding mode'): tool.attach(*args, direct_native=mode)
        for path, digest in ((missing, 'a' * 64), (missing, None), (None, 'a' * 64)):
            changed = args.copy(); changed[8:10] = [path, digest]
            with self.assertRaisesRegex(ValueError, 'Exclusive input binding'): tool.attach(*changed, direct_native=True)

    def test_capture_sha_size_ceiling_and_prepublication_drift(self):
        path = self.directory / 'input'; path.write_bytes(b'original')
        capture = tool.Capture(); self.assertEqual(capture.read(path, tool.common.digest(b'original'), 8), b'original')
        with self.assertRaises(ValueError): capture.read(path, 'a' * 64)
        with self.assertRaises(ValueError): capture.read(path, size=True)
        capture.unchanged(); path.write_bytes(b'changed')
        with self.assertRaises(ValueError): capture.unchanged()
        with patch.object(tool, 'MAX_TOTAL', 2):
            with self.assertRaises(ValueError): tool.Capture().read(path)
        with patch.object(tool, 'MAX_FILE', 2):
            with self.assertRaises(ValueError): tool.Capture().read(path)

    def test_hash_only_inspection_capture_keeps_hash_limits_and_drift_checks(self):
        path = self.directory / 'inspection.json'
        data = tool.common.canonical({'analyticFixture': True, 'sessionCreated': False})
        path.write_bytes(data)
        # Historical review inspection descriptors declare path/hash, but no bytes.
        inspection = dict(path=path.name, sha256=tool.common.digest(data))
        capture = tool.Capture()
        self.assertEqual(capture.read(self.directory / inspection['path'], inspection['sha256']), data)
        capture.unchanged()
        for invalid in ('', 'a' * 63, 'A' * 64, 'z' * 64, True, 'a' * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError): tool.Capture().read(path, invalid)
        with patch.object(tool, 'MAX_FILE', len(data) - 1):
            with self.assertRaisesRegex(ValueError, 'Artifact exceeds byte ceiling'):
                tool.Capture().read(path, inspection['sha256'])
        with patch.object(tool, 'MAX_TOTAL', len(data) - 1):
            with self.assertRaisesRegex(ValueError, 'Cumulative input byte ceiling'):
                tool.Capture().read(path, inspection['sha256'])
        # Other artifact roles still require their explicitly declared byte count.
        with self.assertRaises(KeyError): tool.Capture().artifact(self.directory, inspection)
        path.write_bytes(b'changed inspection after capture')
        with self.assertRaises(ValueError): capture.unchanged()

    def test_attachment_accepts_hash_only_inspection_before_blocked_cache_checkpoint(self):
        # Stop at the cache guard: this fixture creates no execution record or prediction scene.
        def write(name, value):
            data = value if isinstance(value, bytes) else tool.common.canonical(value)
            (self.directory / name).write_bytes(data)
            return dict(path=name, bytes=len(data), sha256=tool.common.digest(data))
        inspection = write('inspection.json', {'analyticFixture': True, 'sessionCreated': False})
        review = write('review.json', {'inspection': {key: inspection[key] for key in ('path', 'sha256')}})
        crop = write('crop.json', {'outputs': {name: write(name, b'test-only-unexecuted-input')
                                              for name in ('left.png', 'right.png', 'calibration.json')}})
        cache_key = 'b' * 64
        plan = write('plan.json', dict(model=tool.nvidia_stereo.MODEL, cacheKey=cache_key, review=review, sources={}))
        producer = write('prediction.json', dict(model=tool.nvidia_stereo.MODEL, cacheKey=cache_key,
            sources={'cropManifest': crop}, prediction=write('raw-disparity', b'test-only-no-execution'),
            validity=write('raw-validity', b'test-only-no-execution')))
        proof = write('proof.json', dict(oldPlanSha256=plan['sha256'], oldProducerSha256=producer['sha256']))
        scene = write('scene.json', {'analyticFixture': True})
        prepared = write('prepared.json', {'analyticFixture': True})
        args = []
        for record in (scene, prepared, producer, plan, proof):
            args.extend([self.directory / record['path'], record['sha256']])
        args.extend([self.directory / 'model-never-read', self.directory / 'output-never-published'])
        with patch.object(tool.nvidia_stereo, 'cached', side_effect=ValueError('analytic blocked cache checkpoint')) as cache:
            with self.assertRaisesRegex(ValueError, 'analytic blocked cache checkpoint'): tool.attach(*args)
            cache.assert_called_once()
        self.assertFalse(args[-1].exists())

    def test_required_source_sha_and_exclusive_destination_before_cache_reader(self):
        missing = self.directory / 'missing'; output = self.directory / 'new-output'
        args = [missing, 'a' * 64] * 5 + [missing, output]
        with patch.object(tool.nvidia_stereo, 'cached', side_effect=AssertionError('No cache/model execution in analytic boundary test')):
            for value in (None, '', 'A' * 64, 'a' * 63, True):
                changed = args.copy(); changed[1] = value
                with self.assertRaisesRegex(ValueError, 'Expected source SHA256 required'): tool.attach(*changed)
            output.mkdir(); (output / 'sentinel').write_bytes(b'retain')
            with self.assertRaisesRegex(ValueError, 'Existing destination retained'): tool.attach(*args)
            self.assertEqual((output / 'sentinel').read_bytes(), b'retain')
            link = self.directory / 'dangling'; link.symlink_to(missing); args[-1] = link
            with self.assertRaisesRegex(ValueError, 'Existing destination retained'): tool.attach(*args)

    def test_caller_digest_and_exclusive_publication_primitive(self):
        # Exercises the exact existing primitive used by the adapter with test-only bytes.
        output = self.directory / 'test-only-published'
        scene = tool.common.canonical({'analyticFixture': True, 'predictedTriangleCount': 0})
        entry = tool.common.canonical({'metadataSha256': tool.common.digest(scene)})
        files = {'scene.json': scene, 'catalog-entry.json': entry, 'estimatedtriangles.bin': b''}
        tool.common.publish(output, files)
        for name, data in files.items(): self.assertEqual((output / name).read_bytes(), data)
        self.assertEqual(tool.common.parse(entry)['metadataSha256'], tool.common.digest((output / 'scene.json').read_bytes()))
        with self.assertRaises(ValueError): tool.common.publish(output, {'scene.json': b'overwrite'})
        self.assertEqual((output / 'scene.json').read_bytes(), scene)


if __name__ == '__main__': unittest.main()

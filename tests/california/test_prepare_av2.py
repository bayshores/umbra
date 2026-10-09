"""Generated analytic native-schema fixtures only; no real data or sponsor run."""
from io import BytesIO
from decimal import localcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.feather as feather

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_av2', ROOT / 'scripts/california/prepare_av2.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
import nvidia_stereo


class PublisherReportedTimingTests(unittest.TestCase):
    def test_exact_large_timestamps_signed_ages_and_asymmetric_cohorts(self):
        camera = 2 ** 53 + 101
        offsets = np.array([11, 0, 5, 2], dtype=np.int64)
        left = np.array([True, True, False, False]); right = np.array([True, False, True, False])
        with localcontext() as context:
            context.prec = 3
            result = tool.publisher_reported_timing(camera, camera - 5, offsets, left, right)
        self.assertEqual(result['cameraTimestampNs'], str(camera))
        self.assertEqual(result['sweepStartTimestampNs'], str(camera - 5))
        self.assertEqual(result['sweepStartMinusCameraNs'], '-5')
        self.assertEqual(result['cohorts'], {
            'allReturns': {'count': 4, 'minNs': '-5', 'maxNs': '6', 'medianNs': '-1.5', 'p95Ns': '5.1',
                           'beforeCameraCount': 2, 'atCameraCount': 1, 'afterCameraCount': 1},
            'leftEligible': {'count': 2, 'minNs': '-5', 'maxNs': '6', 'medianNs': '0.5', 'p95Ns': '5.45',
                             'beforeCameraCount': 1, 'atCameraCount': 0, 'afterCameraCount': 1},
            'rightEligible': {'count': 2, 'minNs': '0', 'maxNs': '6', 'medianNs': '3', 'p95Ns': '5.7',
                              'beforeCameraCount': 0, 'atCameraCount': 1, 'afterCameraCount': 1},
            'bothEligible': {'count': 1, 'minNs': '6', 'maxNs': '6', 'medianNs': '6', 'p95Ns': '6',
                             'beforeCameraCount': 0, 'atCameraCount': 0, 'afterCameraCount': 1}})
        np.testing.assert_array_equal(offsets, [11, 0, 5, 2])
        np.testing.assert_array_equal(left, [True, True, False, False])
        np.testing.assert_array_equal(right, [True, False, True, False])

    def test_empty_and_odd_cohorts(self):
        empty = {'count': 0, 'minNs': None, 'maxNs': None, 'medianNs': None, 'p95Ns': None,
                 'beforeCameraCount': 0, 'atCameraCount': 0, 'afterCameraCount': 0}
        for size in (0, 3):
            offsets = np.array([0, 1, 2][:size], dtype=np.uint64)
            left = np.ones(size, dtype=bool); right = np.zeros(size, dtype=bool)
            result = tool.publisher_reported_timing(0, 0, offsets, left, right)['cohorts']
            self.assertEqual(result['rightEligible'], empty)
            self.assertEqual(result['bothEligible'], empty)
            if size:
                self.assertEqual(result['allReturns']['medianNs'], '1')
                self.assertEqual(result['allReturns']['p95Ns'], '1.9')
            else:
                self.assertTrue(all(cohort == empty for cohort in result.values()))

    def test_exact_int64_boundaries_and_wide_quantile_span(self):
        maximum = np.iinfo(np.int64).max
        offsets = np.array([0, maximum], dtype=np.uint64); mask = np.ones(2, dtype=bool)
        result = tool.publisher_reported_timing(0, 0, offsets, mask, mask)['cohorts']['allReturns']
        self.assertEqual(result['maxNs'], '9223372036854775807')
        self.assertEqual(result['medianNs'], '4611686018427387903.5')
        self.assertEqual(result['p95Ns'], '8762203435012037016.65')
        result = tool.publisher_reported_timing(2 ** 63, 0, offsets, mask, mask)['cohorts']['allReturns']
        self.assertEqual(result['minNs'], '-9223372036854775808')
        self.assertEqual(result['maxNs'], '-1')
        self.assertEqual(result['medianNs'], '-4611686018427387904.5')
        self.assertEqual(result['p95Ns'], '-461168601842738791.35')
        large = 10 ** 30
        self.assertEqual(tool.publisher_reported_timing(large, large + 1, np.array([0]), mask[:1], mask[:1])['cohorts']['allReturns']['minNs'], '1')

    def test_rejects_malformed_timestamps_offsets_and_masks(self):
        offsets = np.array([0, 1], dtype=np.int64); mask = np.ones(2, dtype=bool)
        for invalid in (True, np.int64(1), -1, 1., '1', None):
            for camera, sweep in ((invalid, 0), (0, invalid)):
                with self.subTest(camera=camera, sweep=sweep), self.assertRaises(ValueError):
                    tool.publisher_reported_timing(camera, sweep, offsets, mask, mask)
        for invalid in ([0, 1], np.array(0), offsets.reshape(2, 1), np.array([False, True]),
                        np.array([0., 1.]), np.array(['0', '1']), np.array([0, None], dtype=object),
                        np.array([-1, 0]), np.array([2 ** 63], dtype=np.uint64),
                        np.array([2 ** 64 - 1], dtype=np.uint64)):
            with self.subTest(offsets=invalid), self.assertRaises(ValueError):
                tool.publisher_reported_timing(0, 0, invalid, mask, mask)
        for invalid in ([True, True], np.array([1, 1]), np.ones(1, dtype=bool), mask.reshape(2, 1)):
            for left, right in ((invalid, mask), (mask, invalid)):
                with self.subTest(mask=invalid), self.assertRaises(ValueError):
                    tool.publisher_reported_timing(0, 0, offsets, left, right)

    def test_rejects_overflow_before_age_arithmetic(self):
        mask = np.ones(1, dtype=bool)
        for camera, sweep, offsets in [(0, 1, np.array([2 ** 63 - 1], dtype=np.uint64)),
                                       (2 ** 63 + 1, 0, np.array([0], dtype=np.int64)),
                                       (0, 0, np.array([2 ** 63], dtype=np.uint64)),
                                       (0, 2 ** 63, np.array([0], dtype=np.int64))]:
            with self.subTest(camera=camera, sweep=sweep, offsets=offsets), self.assertRaises(ValueError):
                tool.publisher_reported_timing(camera, sweep, offsets, mask, mask)


class AV2PreparationTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.local/worker-output/test-fixtures'; scratch.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=scratch); self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name); self.output = self.directory / 'output'
        self.log = '12345678-1234-1234-1234-123456789abc'; self.camera = '100000000000000000'; self.sweep = '100000000040000000'
        raw = self.directory / self.log
        self.paths = {'left': raw / 'sensors/cameras/stereo_front_left' / (self.camera + '.jpg'),
                      'right': raw / 'sensors/cameras/stereo_front_right' / (self.camera + '.jpg'),
                      'lidar': raw / 'sensors/lidar' / (self.sweep + '.feather'), 'poses': raw / 'city_SE3_egovehicle.feather',
                      'intrinsics': raw / 'calibration/intrinsics.feather', 'extrinsics': raw / 'calibration/egovehicle_SE3_sensor.feather',
                      'map': raw / 'map' / ('log_map_archive_' + self.log + '____PAO_city_1.json'),
                      'groundRaster': raw / 'map' / (self.log + '_ground_height_surface____PAO.npy'),
                      'groundTransform': raw / 'map' / (self.log + '___img_Sim2_city.json'),
                      'license': self.directory / 'LICENSE-AND-ATTRIBUTION.md'}
        for path in self.paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.indices((400, 800)); rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=2).astype(np.uint8)
        for side in ('left', 'right'): Image.fromarray(rgb).save(self.paths[side], format='JPEG')
        def pose(timestamp, tx): return {'timestamp_ns': timestamp, 'qx': 0., 'qy': 0., 'qz': 0., 'qw': 1., 'tx_m': tx, 'ty_m': 0., 'tz_m': 0.}
        self.pose_rows = [pose(int(self.camera), 0.), pose(int(self.sweep), 1.)]
        self.intr_rows = [{'sensor_name': 'stereo_front_' + side, 'fx_px': 100., 'fy_px': 100., 'cx_px': 399.5, 'cy_px': 199.5, 'width_px': 800, 'height_px': 400} for side in ('left', 'right')]
        self.ext_rows = [{**pose(0, tx), 'sensor_name': name} for name, tx in [('stereo_front_left', 0.), ('stereo_front_right', .5), ('lidar', 999.)]]
        for role, rows in [('poses', self.pose_rows), ('intrinsics', self.intr_rows), ('extrinsics', self.ext_rows)]: self.write_table(role, rows)
        self.xyz = np.array([[-1., 0, 10], [0, 0, 10], [-6.1, -5., 10], [100., 0, 10], [-1., 0, -10], [np.nan, 0, 10]])
        self.offsets = np.arange(len(self.xyz), dtype=np.int64)
        self.write_table('lidar', [{'x': p[0], 'y': p[1], 'z': p[2], 'offset_ns': int(offset)} for p, offset in zip(self.xyz, self.offsets)])
        corners = [dict(x=x, y=y, z=10.) for x, y in [(-2., -1.), (2., -1.), (2., 1.), (-2., 1.)]]
        self.vector_map = {'pedestrian_crossings': {'1': {'id': 1, 'edge1': corners[:2], 'edge2': list(reversed(corners[2:]))}}}
        self.paths['map'].write_bytes(tool.canonical(self.vector_map))
        self.raster = np.arange(24 * 24, dtype=np.float16).reshape(24, 24); self.raster[10, 12] = np.nan
        np.save(self.paths['groundRaster'], self.raster)
        self.sim = {'R': [1., 0., 0., 1.], 't': [5., 5.], 's': 2.}
        self.paths['groundTransform'].write_bytes(tool.canonical(self.sim))
        self.paths['license'].write_bytes(b'Analytic test fixture only. No publisher or sponsor execution evidence.\n')
        self.plan = {'schemaVersion': 1, 'sourceClass': 'Argoverse2Sensor', 'analyticFixture': True,
                     'logId': self.log, 'cameraTimestampNs': self.camera, 'lidarTimestampNs': self.sweep,
                     'crossingId': '1', 'rectifiedSize': [800, 400], 'crop': [20, 30, 736, 320], 'sources': {}}
        self.plan_path = self.directory / 'plan.json'; self.refresh()

    def write_table(self, role, rows):
        feather.write_feather(pa.Table.from_pylist(rows), self.paths[role])

    def refresh(self, roles=None):
        for role in roles or self.paths:
            data = self.paths[role].read_bytes()
            self.plan['sources'][role] = {'path': str(self.paths[role].relative_to(self.directory)), 'bytes': len(data), 'sha256': tool.digest(data)}
        self.plan_path.write_bytes(tool.canonical(self.plan))

    def rejected(self):
        before = {role: path.read_bytes() for role, path in self.paths.items()}
        with self.assertRaises((ValueError, KeyError)):
            tool.prepare(self.plan_path, self.output)
        self.assertFalse(self.output.exists())
        for role, data in before.items(): self.assertEqual(self.paths[role].read_bytes(), data)

    def test_native_transforms_all_rows_ground_and_crop_reader(self):
        result = tool.prepare(self.plan_path, self.output)
        self.assertEqual(result['sourceClass'], 'Argoverse2Sensor')
        _, manifest, _, cal = nvidia_stereo.read_crop(self.output / 'manifest.json')  # No inference call.
        self.assertEqual(manifest['processing']['sourceRows'], len(self.xyz))
        self.assertEqual((self.output / 'LICENSE-AND-ATTRIBUTION.md').read_bytes(), self.paths['license'].read_bytes())
        with np.load(self.output / 'references.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sourceIndex'], np.arange(len(self.xyz)))
            np.testing.assert_array_equal(saved['egoXYZ'], self.xyz)
            expected_city = self.xyz + [1., 0., 0.]
            expected_city[5] = np.nan  # Invalid vector is retained; matrix arithmetic propagates unknowns.
            np.testing.assert_allclose(saved['cityXYZ'], expected_city, equal_nan=True)
            np.testing.assert_array_equal(saved['offsetNs'], self.offsets)
            timing = manifest['processing']['publisherReportedTiming']
            self.assertEqual(timing['cameraTimestampNs'], self.camera)
            self.assertEqual(timing['sweepStartTimestampNs'], self.sweep)
            self.assertEqual(timing['sweepStartMinusCameraNs'], '40000000')
            self.assertEqual(manifest['processing']['lidarMinusCameraNs'], '40000000')
            for name in ('allReturns', 'leftEligible', 'rightEligible', 'bothEligible'):
                mask = np.ones(len(self.xyz), dtype=bool) if name == 'allReturns' else saved[name]
                ages = sorted(int(offset) + 40000000 for offset in saved['offsetNs'][mask])
                stats = timing['cohorts'][name]
                self.assertEqual(stats['count'], len(ages))
                self.assertEqual(stats['minNs'], str(ages[0]))
                self.assertEqual(stats['maxNs'], str(ages[-1]))
                self.assertEqual(stats['beforeCameraCount'], 0)
                self.assertEqual(stats['atCameraCount'], 0)
                self.assertEqual(stats['afterCameraCount'], len(ages))
            self.assertEqual(timing['cohorts']['allReturns']['medianNs'], '40000002.5')
            self.assertEqual(timing['cohorts']['allReturns']['p95Ns'], '40000004.75')
            for name in ('leftEligible', 'rightEligible', 'bothEligible'):
                self.assertEqual(timing['cohorts'][name]['medianNs'], '40000001')
                self.assertEqual(timing['cohorts'][name]['p95Ns'], '40000001.9')
            self.assertTrue(saved['bothEligible'][:2].all())
            self.assertFalse(saved['bothEligible'][3:].any())
            self.assertTrue(saved['groundBoundaryAlias'][2])
            self.assertTrue(np.isnan(saved['nominalGroundHeights'][2]))
            self.assertFalse(saved['groundSupport'][2])
            self.assertTrue(np.isnan(saved['nominalGroundHeights'][1]))  # In-domain source NaN remains missing.
            self.assertTrue(saved['insideFootprint'][0] and saved['insideFootprint'][1])
            d = saved['leftCropUV'][0, 0] - saved['rightCropUV'][0, 0]
            self.assertAlmostEqual(d, cal['disparity_numerator_px_m'] / saved['leftRectifiedXYZ'][0, 2])
        self.assertEqual(len(result['processing']['nominalTargetQueries']), 5)
        self.assertEqual(result['processing']['leftCameraCityOrigin'], [0., 0., 0.])
        self.assertFalse(result['runtimeAdopted'])
        self.assertFalse((self.output / 'depth.png').exists())

    def test_native_invalid_return_offsets_rejected_without_publication(self):
        for value in (True, -1, 1., '1', None, 2 ** 63 - 1):
            rows = [{'x': p[0], 'y': p[1], 'z': p[2], 'offset_ns': value} for p in self.xyz]
            self.write_table('lidar', rows); self.refresh(['lidar']); self.rejected()
        rows = [{'x': p[0], 'y': p[1], 'z': p[2], 'offset_ns': 2 ** 63} for p in self.xyz]
        feather.write_feather(pa.Table.from_pylist(rows, schema=pa.schema([
            ('x', pa.float64()), ('y', pa.float64()), ('z', pa.float64()), ('offset_ns', pa.uint64())])), self.paths['lidar'])
        self.refresh(['lidar']); self.rejected()

    def test_native_original_timestamp_strings_retained(self):
        self.camera = '0' + self.camera; self.sweep = '0' + self.sweep
        for role, timestamp, extension in [('left', self.camera, '.jpg'), ('right', self.camera, '.jpg'), ('lidar', self.sweep, '.feather')]:
            original = self.paths[role]; self.paths[role] = original.with_name(timestamp + extension)
            original.rename(self.paths[role])
        self.plan['cameraTimestampNs'] = self.camera; self.plan['lidarTimestampNs'] = self.sweep
        self.refresh()
        result = tool.prepare(self.plan_path, self.output)
        _, manifest, _, _ = nvidia_stereo.read_crop(self.output / 'manifest.json')
        timing = manifest['processing']['publisherReportedTiming']
        self.assertEqual(timing['cameraTimestampNs'], self.camera)
        self.assertEqual(timing['sweepStartTimestampNs'], self.sweep)
        self.assertEqual(timing['sweepStartMinusCameraNs'], '40000000')
        self.assertEqual(result['processing']['lidarMinusCameraNs'], '40000000')

    def test_crop_uv_projection_domain_preserves_source_and_outside_rows(self):
        # Analytic identity rotations: native camera XYZ has sweep translation +1m X.
        self.xyz = np.vstack((self.xyz, [-1., 0, 0], [-1., 0, np.nan]))
        self.offsets = np.arange(len(self.xyz), dtype=np.int64)
        self.write_table('lidar', [{'x': p[0], 'y': p[1], 'z': p[2], 'offset_ns': int(offset)} for p, offset in zip(self.xyz, self.offsets)])
        self.refresh(['lidar'])
        tool.prepare(self.plan_path, self.output)
        with np.load(self.output / 'references.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sourceIndex'], np.arange(8))
            np.testing.assert_array_equal(saved['egoXYZ'], self.xyz)
            np.testing.assert_array_equal(saved['offsetNs'], self.offsets)
            for side, camera_x in [('left', 0.), ('right', .5)]:
                # Explicit pinhole expectation independent of producer projection code.
                expected_uv = np.column_stack((100 * (self.xyz[:4, 0] + 1 - camera_x) / 10 + 379.5,
                                               100 * self.xyz[:4, 1] / 10 + 169.5))
                np.testing.assert_allclose(saved[side + 'CropUV'][:4], expected_uv, atol=1e-12, rtol=0)
                self.assertTrue(np.isnan(saved[side + 'CropUV'][4:]).all())
                self.assertGreater(saved[side + 'CropUV'][3, 0], 735)  # Positive outside-crop UV retained.
                np.testing.assert_array_equal(saved[side + 'Eligible'], [True, True, True, False, False, False, False, False])
                np.testing.assert_allclose(saved[side + 'RectifiedXYZ'][4], [-camera_x, 0, -10], atol=1e-12)
                np.testing.assert_allclose(saved[side + 'RectifiedXYZ'][6], [-camera_x, 0, 0], atol=1e-12)
            np.testing.assert_array_equal(saved['bothEligible'], [True, True, True, False, False, False, False, False])
            # Negative/zero camera depth changes neither city XY footprint nor nominal ground.
            self.assertTrue(saved['insideFootprint'][[4, 6]].all())
            self.assertTrue(saved['groundSupport'][[4, 6]].all())
            np.testing.assert_array_equal(saved['nominalGroundHeights'][[4, 6]], [250., 250.])

    def test_native_identity_hash_and_duplicate_source_rejections(self):
        original = json.loads(self.plan_path.read_text())
        for change in [('cameraTimestampNs', '2'), ('lidarTimestampNs', '3'), ('cameraTimestampNs', float(self.camera)), ('logId', 'not-uuid')]:
            self.plan = json.loads(json.dumps(original)); self.plan[change[0]] = change[1]; self.plan_path.write_bytes(tool.canonical(self.plan)); self.rejected()
        self.plan = original; self.refresh()
        self.paths['left'].write_bytes(self.paths['left'].read_bytes() + b'changed'); self.rejected()
        self.refresh()
        self.plan['sources']['right'] = self.plan['sources']['left'].copy(); self.plan_path.write_bytes(tool.canonical(self.plan)); self.rejected()

    def test_read_bounds_and_duplicate_nonfinite_plan_json(self):
        original = self.plan_path.read_bytes()
        for raw in [original.replace(b'"schemaVersion":1', b'"schemaVersion":1,"schemaVersion":1'), original.replace(b'"schemaVersion":1', b'"schemaVersion":NaN')]:
            self.plan_path.write_bytes(raw); self.rejected()
        self.plan_path.write_bytes(original)
        self.plan['sources']['left']['bytes'] = tool.MAX_FILE + 1; self.plan_path.write_bytes(tool.canonical(self.plan)); self.rejected()
        for item in self.plan['sources'].values(): item['bytes'] = 8_000_000
        self.plan_path.write_bytes(tool.canonical(self.plan)); self.rejected()

    def test_exact_pose_and_sensor_mutations(self):
        for rows in [self.pose_rows[:1], self.pose_rows + [self.pose_rows[0]], [{**self.pose_rows[0], 'qw': 2.}, self.pose_rows[1]], [{**self.pose_rows[0], 'tx_m': np.nan}, self.pose_rows[1]]]:
            self.write_table('poses', rows); self.refresh(['poses']); self.rejected()
        self.write_table('poses', self.pose_rows); self.refresh(['poses'])
        for rows in [self.intr_rows[:1], self.intr_rows + [self.intr_rows[0]], [{**self.intr_rows[0], 'fx_px': np.inf}, self.intr_rows[1]], [{**self.intr_rows[0], 'width_px': 799}, self.intr_rows[1]]]:
            self.write_table('intrinsics', rows); self.refresh(['intrinsics']); self.rejected()

    def test_crop_support_rejects_unsupported_pixels(self):
        for crop in [[0, 0, 736, 320], [20, 30, 735, 320], [20., 30, 736, 320], [100, 30, 736, 320]]:
            self.plan['crop'] = crop; self.plan_path.write_bytes(tool.canonical(self.plan)); self.rejected()

    def test_partial_crossing_retains_all_corners_and_source_rows(self):
        self.vector_map['pedestrian_crossings']['1']['edge1'][0]['x'] = -100.
        self.paths['map'].write_bytes(tool.canonical(self.vector_map)); self.refresh(['map'])
        result = tool.prepare(self.plan_path, self.output)
        nvidia_stereo.read_crop(self.output / 'manifest.json')  # Source-supported RGB input, no inference.
        for side, expected_u in [('left', -620.5), ('right', -625.5)]:
            corners = result['processing']['mappedCrossing'][side]
            self.assertEqual(len(corners['rectifiedXYZ']), 4)
            self.assertEqual(corners['rectifiedXYZ'][0][2], 10.)
            self.assertAlmostEqual(corners['cropUV'][0][0], expected_u)
            self.assertEqual(corners['cropSupported'], [False, True, True, True])
        with np.load(self.output / 'references.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sourceIndex'], np.arange(len(self.xyz)))
            np.testing.assert_array_equal(saved['egoXYZ'], self.xyz)
        self.assertFalse(result['runtimeAdopted'])

    def test_behind_and_zero_depth_corners_stay_unsupported(self):
        edge = self.vector_map['pedestrian_crossings']['1']['edge1']
        edge[0]['z'] = -10.; edge[1]['z'] = 0.
        self.paths['map'].write_bytes(tool.canonical(self.vector_map)); self.refresh(['map'])
        result = tool.prepare(self.plan_path, self.output)
        for side in ('left', 'right'):
            corners = result['processing']['mappedCrossing'][side]
            self.assertEqual([p[2] for p in corners['rectifiedXYZ']], [-10., 0., 10., 10.])
            self.assertEqual(corners['cropSupported'], [False, False, True, True])
            self.assertEqual(corners['cropUV'][1], [None, None])
        self.assertFalse(result['runtimeAdopted'])

    def test_grid_ceiling_rejects_before_decoder_or_map_allocation(self):
        for dimensions in [[32767, 400], [5000, 5000]]:
            self.plan['rectifiedSize'] = dimensions; self.plan_path.write_bytes(tool.canonical(self.plan))
            with patch.object(tool.cv2, 'stereoRectify', side_effect=AssertionError('must not allocate')), patch.object(tool.Image, 'open', side_effect=AssertionError('must not decode')):
                self.rejected()
        self.plan['rectifiedSize'] = [800, 400]; self.refresh()
        self.intr_rows[0]['width_px'] = 32767
        self.write_table('intrinsics', self.intr_rows); self.refresh(['intrinsics'])
        with patch.object(tool.Image, 'open', side_effect=AssertionError('must not decode')):
            self.rejected()

    def test_ground_boundary_zero_nan_sim2_and_envelope(self):
        points = np.array([[-.1, 0, 0], [0, 0, 0], [1.9, 0, 0], [2., 0, 0], [np.nan, 0, 0]])
        raster = np.array([[0., np.nan]], dtype=np.float16); sim = {'R': [1., 0., 0., 1.], 't': [0., 0.], 's': 1.}
        sampled = tool.ground_sample(points, raster, sim)
        np.testing.assert_array_equal(sampled['groundBoundaryAlias'], [True, False, False, False, False])
        np.testing.assert_array_equal(sampled['groundSupport'], [False, True, False, False, False])
        self.assertEqual(sampled['nominalGroundHeights'][1], 0.)
        self.assertTrue(np.isnan(sampled['nominalGroundHeights'][0]))
        moved = tool.ground_sample(np.array([[0., 0., 0.]]), np.arange(16, dtype=np.float16).reshape(4, 4), {'R': [1., 0., 0., 1.], 't': [1., 1.], 's': 2.})
        self.assertEqual(moved['nominalGroundHeights'][0], 10.)  # Translation occurs before scale.
        for value in [np.ones((2, 2), np.float32), np.array([[np.inf]], np.float16), np.ones((2, 2, 2), np.float16)]:
            stream = BytesIO(); np.save(stream, value)
            with self.assertRaises(ValueError): tool.ground_raster(stream.getvalue())
        for bad in [{'R': [1., 0., 0., -1.], 't': [0., 0.], 's': 1.}, {'R': [1., 0., 0., 1.], 't': [0., 0.], 's': 0.}]:
            with self.assertRaises(ValueError): tool.ground_sample(points, raster, bad)

    def test_ground_raster_accepts_log_sized_surface_without_filling_nan(self):
        raster = np.full((1159, 986), np.nan, dtype=np.float16)
        raster[0, 0] = -22.5
        raster[-1, -1] = 0.
        stream = BytesIO(); np.save(stream, raster)
        decoded = tool.ground_raster(stream.getvalue())
        self.assertEqual(decoded.dtype, np.dtype('float16'))
        np.testing.assert_array_equal(decoded, raster)

    def test_ground_raster_rejects_over_byte_budget_before_parsing(self):
        stream = BytesIO(); np.save(stream, self.raster)
        data = stream.getvalue()
        with patch.object(tool, 'MAX_FILE', len(data) - 1), \
             patch.object(np.lib.format, 'read_magic', side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(ValueError, 'byte ceiling'):
                tool.ground_raster(data)

    def test_ground_raster_rejects_forged_large_shape_before_allocation(self):
        stream = BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '<f2', 'fortran_order': False, 'shape': (tool.MAX_FILE // 2 + 1, 1)})
        with patch.object(np, 'load', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(ValueError, 'shape ceiling'):
                tool.ground_raster(stream.getvalue())

    def test_native_flat_sim2_rotation_rejects_malformed_and_nested_values(self):
        points = np.array([[1., 0., 0.]])
        raster = np.arange(16, dtype=np.float16).reshape(4, 4)
        rotated = tool.ground_sample(points, raster, {'R': [0., -1., 1., 0.], 't': [0., 0.], 's': 1.})
        self.assertEqual(rotated['nominalGroundHeights'][0], 4.)
        for rotation in [[[1., 0.], [0., 1.]], [], [1., 0., 1.], [1., 0., 0., 1., 0.],
                         [np.nan, 0., 0., 1.], [np.inf, 0., 0., 1.], [True, 0., 0., 1.],
                         ['1', 0., 0., 1.], [2., 0., 0., 1.], [1., 0., 0., -1.]]:
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                tool.ground_sample(points, raster, {'R': rotation, 't': [0., 0.], 's': 1.})
        self.paths['groundTransform'].write_bytes(tool.canonical({'R': [[1., 0.], [0., 1.]], 't': [5., 5.], 's': 2.}))
        self.refresh(['groundTransform']); self.rejected()

    def test_footprint_euclidean_boundary_long_short_edges_and_orientations(self):
        corners = np.array([[0., 0., 0.], [100., 0., 0.], [100., .1, 0.], [0., .1, 0.]])
        points = np.array([[50., .05, 0.], [50., -.5e-9, 0.], [50., -2e-9, 0.],
                           [100. + .5e-9, .05, 0.], [100. + 5e-9, .05, 0.],
                           [-.5e-9, -.5e-9, 0.], [-.8e-9, -.8e-9, 0.],
                           [-10., -.5e-9, 0.], [110., .1 + .5e-9, 0.],
                           [0., 0., 0.], [np.nan, 0., 0.], [0., np.inf, 0.], [50., .05, np.nan]])
        # Direct rectangle geometry: edge distance is its perpendicular offset;
        # outside a vertex, distance is hypot(dx,dy), not either infinite line.
        expected = [True, True, False, True, False, True, False, False, False, True, False, False, True]
        before = points.copy()
        for polygon in [corners, corners[::-1]]:
            np.testing.assert_array_equal(tool.footprint_membership(points, polygon), expected)
        np.testing.assert_array_equal(points, before)

    def test_reviewer_scale_counterexample_with_analytic_coordinates(self):
        # Synthetic analogue of the reviewer's ~8.3 m edge; no map coordinates.
        polygon = np.array([[0., 0., 0.], [8.3, 0., 0.], [8.3, 1., 0.], [0., 1., 0.]])
        point = np.array([[4.15, -.5e-9, 0.]])
        for corners in [polygon, polygon[::-1]]:
            np.testing.assert_array_equal(tool.footprint_membership(point, corners), [True])

    def test_native_map_unused_lane_nan_preserved_without_affecting_references(self):
        baseline = tool.prepare(self.plan_path, self.output)
        self.vector_map['lane_segments'] = {'analyticUnusedLane': {'left_lane_boundary': [{'x': 1., 'y': 2., 'z': np.nan}]}}
        raw = json.dumps(self.vector_map, allow_nan=True).encode()
        self.paths['map'].write_bytes(raw); self.refresh(['map'])
        decoded, seen = tool.decode_native_map(raw)
        self.assertTrue(seen)
        self.assertTrue(np.isnan(decoded['lane_segments']['analyticUnusedLane']['left_lane_boundary'][0]['z']))
        self.assertEqual(self.paths['map'].read_bytes(), raw)
        second = self.directory / 'with-unknown-lane'
        result = tool.prepare(self.plan_path, second)
        self.assertTrue(result['processing']['mapSourceUnknownSeen'])
        self.assertIn('sourceUnknown', result['processing']['mapUnknownPolicy'])
        self.assertFalse(baseline['processing']['mapSourceUnknownSeen'])
        with np.load(self.output / 'references.npz') as before, np.load(second / 'references.npz') as after:
            self.assertEqual(before.files, after.files)
            for name in before.files:
                np.testing.assert_array_equal(before[name], after[name])

    def test_native_map_required_corner_nan_rejected(self):
        self.vector_map['pedestrian_crossings']['1']['edge1'][0]['z'] = np.nan
        self.paths['map'].write_bytes(json.dumps(self.vector_map, allow_nan=True).encode())
        self.refresh(['map']); self.rejected()

    def test_native_map_duplicate_and_infinite_values_rejected(self):
        for value in [b'{"unused":NaN,"unused":1}', b'{"unused":{"z":Infinity}}',
                      b'{"unused":{"z":-Infinity}}', b'{"unused":{"z":1e999}}',
                      b'{"unused":{"z":-1e999}}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.decode_native_map(value)
        self.paths['map'].write_bytes(b'{"unused":{"z":Infinity}}')
        self.refresh(['map']); self.rejected()

    def test_single_capture_and_existing_output_unchanged(self):
        reads = {}; original_snapshot = tool.snapshot
        def capture_once(path, *args, **kwargs):
            path = Path(path).resolve(); reads[path] = reads.get(path, 0) + 1
            data = original_snapshot(path, *args, **kwargs)
            if path == self.paths['left']:
                path.write_bytes(b'replaced after capture')
            return data
        with patch.object(tool, 'snapshot', side_effect=capture_once):
            manifest = tool.prepare(self.plan_path, self.output)
        self.assertTrue(all(count == 1 for count in reads.values()))
        self.assertEqual(manifest['sources']['left']['sha256'], self.plan['sources']['left']['sha256'])
        before = (self.output / 'manifest.json').read_bytes()
        with self.assertRaises(ValueError): tool.prepare(self.plan_path, self.output)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

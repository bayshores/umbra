"""Analytic independent-support, robust-fit and leakage checks for measured ground."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('measured_ground', Path(__file__).resolve().parents[2]/'scripts/california/measured_ground.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
SCORING = 'f'*64  # the scoring sweep's sha256; distinct from every support hash below


class MeasuredGround(unittest.TestCase):
    def plane(self):
        x, z = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
        p = np.column_stack([x.ravel(), -2+.03*x.ravel()-.02*z.ravel(), z.ravel()])
        return p, np.arange(len(p)) % 4

    def test_sloped_plane_fit_is_exact(self):
        p, ids = self.plane()
        self.assertAlmostEqual(tool.fit_plane(p, ids, [.13, -.17]), -2+.03*.13+.02*.17, places=12)

    def test_robust_fit_rejects_height_outliers(self):
        p, ids = self.plane(); p[::9, 1] += .2
        self.assertAlmostEqual(tool.fit_plane(p, ids, [0, 0]), -2, places=9)

    def test_last_robust_rejection_is_refitted(self):
        rng = np.random.default_rng(0); xz = rng.uniform(-.7, .7, (180, 2))
        y = -2+.03*xz[:, 0]-.02*xz[:, 1]+rng.normal(0, .008, len(xz)); y[:55] += rng.uniform(.03, .14, 55)
        p = np.column_stack([xz[:, 0], y, xz[:, 1]])
        self.assertAlmostEqual(tool.fit_plane(p, np.arange(len(p)) % 4, [0, 0]), -1.9990075953968334, places=12)

    def test_repeated_returns_from_one_sweep_are_insufficient(self):
        p, ids = self.plane()
        self.assertIsNone(tool.fit_plane(p, ids*0, [0, 0]))

    def test_nearest_measurement_and_return_count_are_required(self):
        p, ids = self.plane()
        self.assertIsNone(tool.fit_plane(p, ids, [3, 0]))
        self.assertIsNone(tool.fit_plane(p[:5], ids[:5], [0, 0]))

    def test_steep_surface_is_refused(self):
        p, ids = self.plane(); p[:, 1] = -2+p[:, 0]
        self.assertIsNone(tool.fit_plane(p, ids, [0, 0]))

    def test_empty_and_wide_unmeasured_gap_stay_holes(self):
        p, ids = self.plane(); p = np.concatenate([p+[-3, 0, 0], p+[3, 0, 0]]); ids = np.tile(ids, 2)
        areas = [{'outline': [[-5, -2], [5, -2], [5, 2], [-5, 2]]}]
        params = {**tool.PARAMETERS, 'windowHalfWidthMeters': 5}
        surface = tool.fit_surface(p, ids, areas, params)
        self.assertIsNone(surface['heights'][10*21+10])
        self.assertIsNotNone(surface['heights'][10*21+4])
        empty = tool.fit_surface([], [], areas, params)
        self.assertIsNone(empty['heightBounds']); self.assertTrue(all(h is None for h in empty['heights']))

    def test_partition_excludes_frame_neighbours_and_separates_calibration(self):
        frame = 1_000_000_000
        records = [{'timestampNs': str(frame+dt), 'sha256': str(i)*64} for i, dt in enumerate((-800_000_000, -500_000_000, -250_000_000, 0, 250_000_000, 500_000_000, 800_000_000, 1_100_000_000))]
        p = tool.partition_sweeps(records, frame, None, SCORING)
        self.assertEqual(len(p['excluded']), 3)
        fit = {r['timestampNs'] for r in p['fit']}; calibration = {r['timestampNs'] for r in p['calibration']}
        self.assertFalse(fit & calibration); self.assertNotIn(str(frame), fit | calibration)
        self.assertEqual(len(fit), 3); self.assertEqual(len(calibration), 2)

    def test_frame_sweep_timestamp_is_refused_as_support(self):
        with self.assertRaisesRegex(ValueError, 'scoring frame sweep is refused'):
            tool.require_support_timestamp('1000000000', '1000000000')
        with self.assertRaisesRegex(ValueError, 'temporal exclusion'):
            tool.require_support_timestamp('1250000000', '1000000000')
        tool.require_support_timestamp('1250000001', '1000000000')

    def test_maximum_time_window_precedes_the_calibration_partition(self):
        records = [{'timestampNs': str(t), 'sha256': str(i)*64} for i, t in enumerate((100_000_000, 500_000_000, 1_000_000_000, 1_500_000_000, 1_900_000_000, 2_100_000_000))]
        p = tool.partition_sweeps(records, 1_000_000_000, 1., SCORING)
        self.assertEqual([r['timestampNs'] for r in p['calibration']], ['100000000'])
        self.assertEqual(len(p['fit']), 3)
        self.assertTrue(any(r['reason'] == 'outside maximum support time window' for r in p['excluded']))

    def test_partition_refuses_numeric_aliases_scoring_bytes_and_repeated_bytes(self):
        records = [{'timestampNs': '1000000000', 'sha256': 'a'*64}, {'timestampNs': '01000000000', 'sha256': 'b'*64}]
        with self.assertRaisesRegex(ValueError, 'Duplicate support timestamps'):
            tool.partition_sweeps(records, 0, None, SCORING)
        records[1]['timestampNs'] = '2000000000'
        with self.assertRaisesRegex(ValueError, 'scoring frame sweep bytes'):
            tool.partition_sweeps(records, 0, None, 'a'*64)
        records[1]['sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'Duplicate support/calibration sweep bytes'):
            tool.partition_sweeps(records, 0, None, SCORING)
        records[1]['sha256'] = 'b'*64
        records[0]['timestampNs'] = '01000000000'
        self.assertEqual(tool.partition_sweeps(records, 0, None, SCORING)['calibration'][0]['timestampNs'], '1000000000')

    def test_partition_requires_a_valid_scoring_sweep_identity(self):
        records = [{'timestampNs': '1000000000', 'sha256': 'a'*64}]
        for bad in (None, '', 'A'*64, 'a'*63, 'g'*64, 64):
            with self.assertRaisesRegex(ValueError, 'valid scoring-sweep sha256'):
                tool.partition_sweeps(records, 0, None, bad)

    def test_support_receipt_binds_listing_frame_path_size_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); log = 'example'; ts = '1000000000'; key = f'datasets/av2/sensor/train/{log}/sensors/lidar/{ts}.feather'
            listing = {'prefix': key.rsplit('/', 1)[0]+'/', 'objects': [{'key': key, 'size': 10, 'etag': 'c'*32}]}
            raw = json.dumps(listing).encode(); (folder/'listing.json').write_bytes(raw)
            row = dict(timestampNs=ts, key=key, file=f'sweeps/{ts}.feather', bytes=10, contentLength=10, status=200,
                       sizeMatches=True, md5MatchesEtag=True, md5='c'*32, etag='c'*32, sha256='a'*64)
            receipt = dict(schema='support-sweeps-download-receipt', log=log, split='train', frameSweepTimestampNs='0',
                           rule={'excludesFrameSweep': True}, listingSha256=tool.digest(raw), sweeps=[row])
            path = folder/'receipt.json'; tool.write_json(path, receipt)
            self.assertEqual(tool.receipt_sweeps(path, log, '0', {})[0][0]['sha256'], 'a'*64)
            for field, value in [('file', '../bad.feather'), ('bytes', 9), ('md5', 'b'*32), ('key', key.replace(log, 'other'))]:
                bad = json.loads(json.dumps(receipt)); bad['sweeps'][0][field] = value; tool.write_json(path, bad)
                with self.assertRaises(ValueError): tool.receipt_sweeps(path, log, '0', {})
            tool.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'frame mismatch'): tool.receipt_sweeps(path, log, '1', {})
            (folder/'listing.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'): tool.receipt_sweeps(path, log, '0', {})

    def test_calibration_counts_the_closed_runtime_range_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); folder = out/'grounded'; folder.mkdir()
            scene = {'camera': dict(cx=2, cy=2, fx=50, fy=50, width=8, height=10),
                     'transform': {'worldFromCamera': [[1, 0, 0], [0, -1, 0], [0, 0, -1]]}, 'measurements': {}}
            surface = dict(originXZ=[-40, -40], columns=161, rows=81, cellMeters=.5, heights=[-.6]*(161*81),
                           heightBounds=[-.6, -.6], parameters={'bandRangeRule': 'linear'}, sourceSuggestionsSha256='a'*64)
            ground = dict(schemaVersion=4, maxRangeMeters=100, placementRangeLimitMeters=30, frame={'cityOrigin': [0, 0, 0]},
                          cuboids={'available': False}, drivableAreas=[{'outline': [[-40, -40], [40, -40], [40, 1], [-40, 1]]}],
                          measuredGround=surface, alongRayUncertainty={'minimumCount': 100}, license={}, protocol={'sha256': 'b'*64})
            suggestions = {role: {'status': 'unknown'} for role in ('crossing', 'vehicle', 'person')}
            suggestions.update(tolerances={}, inputs={})
            for name, value in (('scene', scene), ('ground', ground), ('suggestions', suggestions), ('catalog-entry', {})):
                tool.write_json(folder/f'{name}.json', value)
            tool.write_json(out/'calibration-points.json', [[0, -.6, -29.9]]*100)
            result = subprocess.run(['node', '--input-type=module', '-', str(out)], input=tool.FINALIZE_JS,
                                    cwd=tool.ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            errors = tool.read_json(out/'calibration-errors.json')
            self.assertEqual((errors['eligible'], errors['placed'], errors['unknown']), (100, 100, {}))
            self.assertTrue(all(r == 30 for r, _ in errors['rangeErrors']))
            self.assertEqual(len(tool.read_json(folder/'ground.json')['alongRayUncertainty']['knots']), 61)

    def test_annotation_removal_is_at_the_sweeps_ego_pose(self):
        box = dict(qw=1., qx=0., qy=0., qz=0., tx_m=5., ty_m=0., tz_m=0., length_m=2., width_m=2., height_m=2.)
        np.testing.assert_array_equal(tool.outside_cuboids(np.array([[5., 0., 0.], [4., 0., 0.], [3., 0., 0.]]), [box]), [False, False, True])

    def test_mapped_margin_is_explicit_and_bounded(self):
        areas = [{'outline': [[-1, -1], [1, -1], [1, 1], [-1, 1]]}]
        np.testing.assert_array_equal(tool.inside_polygons([[0, 0], [1.4, 0], [1.6, 0]], areas, .5), [True, True, False])


if __name__ == '__main__':
    unittest.main()

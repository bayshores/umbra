"""Analytic independent-support, robust-fit and leakage checks for measured ground."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('measured_ground', Path(__file__).resolve().parents[2]/'scripts/california/measured_ground.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


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
        records = [{'timestampNs': str(frame+dt), 'sha256': 'a'*64} for dt in (-800_000_000, -500_000_000, -250_000_000, 0, 250_000_000, 500_000_000, 800_000_000, 1_100_000_000)]
        p = tool.partition_sweeps(records, frame)
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
        records = [{'timestampNs': str(t), 'sha256': 'a'*64} for t in (100_000_000, 500_000_000, 1_000_000_000, 1_500_000_000, 1_900_000_000, 2_100_000_000)]
        p = tool.partition_sweeps(records, 1_000_000_000, 1.)
        self.assertEqual([r['timestampNs'] for r in p['calibration']], ['100000000'])
        self.assertEqual(len(p['fit']), 3)
        self.assertTrue(any(r['reason'] == 'outside maximum support time window' for r in p['excluded']))

    def test_annotation_removal_is_at_the_sweeps_ego_pose(self):
        box = dict(qw=1., qx=0., qy=0., qz=0., tx_m=5., ty_m=0., tz_m=0., length_m=2., width_m=2., height_m=2.)
        np.testing.assert_array_equal(tool.outside_cuboids(np.array([[5., 0., 0.], [4., 0., 0.], [3., 0., 0.]]), [box]), [False, False, True])

    def test_mapped_margin_is_explicit_and_bounded(self):
        areas = [{'outline': [[-1, -1], [1, -1], [1, 1], [-1, 1]]}]
        np.testing.assert_array_equal(tool.inside_polygons([[0, 0], [1.4, 0], [1.6, 0]], areas, .5), [True, True, False])


if __name__ == '__main__':
    unittest.main()

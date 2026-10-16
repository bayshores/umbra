"""Synthetic point-cloud tests for the development-only LiDAR paint-edge measurement (M3).

Every fixture has a known painted edge. Ring-shaped returns imitate stacked sweeps from an approaching
ego; intensities carry per-laser gain, noise and outliers. No real AV2 data is read.
"""
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/california/measure_paint_edge.py'
spec = importlib.util.spec_from_file_location('measure_paint_edge', SCRIPT)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)

# Same values as protocol v5 (.local/evaluation/pao-paint/protocol-v5.json) except bootstrapReps (1000 there; the
# strip bootstrap is a diagnostic only since v4), reduced for test speed; tests never read .local.
# v5 differs from v4 only in estimatorA's laser tail window keys and the uncertainty method keys (see PARAMS_V4).
PARAMS = {
    'sweepSelection': {'count': 15, 'lidarSensor': 'up_lidar', 'edgeAheadRangeM': [5.0, 30.0], 'maxHeadingToEdgeNormalDeg': 30.0},
    'groundBand': {'belowM': -0.10, 'aboveM': 0.10},
    'corridor': {'laneTypes': ['VEHICLE', 'BIKE', 'BUS'], 'approachLaneMaxAngleDeg': 45.0, 'trimM': 0.25, 'minWidthM': 1.0},
    'windows': {'searchS': [-3.0, 3.0], 'roadReferenceS': [-7.0, -3.0]},
    'estimatorA': {'stripWidthM': 0.15, 'stripOffsetM': 0.0, 'minStripWidthM': 0.075, 'minReferenceReturns': 20,
                   'roadMedianFloor': 1.0, 'ratio': 3.0, 'laserReferenceMinReturns': 20, 'laserTailQuantile': 0.90,
                   'onsetLookaheadReturns': 3, 'onsetMinPaintInLookahead': 2, 'confirmWindowM': 0.5, 'confirmMinPaintReturns': 3,
                   'confirmMinPaintFraction': 0.5, 'continuityWindowM': 1.0, 'maxIntervalM': 0.5, 'minStrips': 6,
                   'laserTailRegion': 'stripLateralWindow', 'laserTailLateralMarginM': 0.5, 'laserTailMaxLateralMarginM': 1.0,
                   'laserTailMarginStepM': 0.25},
    'estimatorB': {'stripWidthM': 0.15, 'stripOffsetM': 0.075, 'minStripWidthM': 0.075, 'minLaserReferenceReturns': 30,
                   'gainFloor': 1.0, 'segmentLengthM': 0.05, 'minReturnsPerSegment': 1, 'minSegments': 20,
                   'minSegmentsEachSide': 6, 'alpha': 1e-4, 'minAuc': 0.70, 'minStrips': 5},
    'agreementToleranceM': 0.15,
    'maxEstimatorSdM': 0.075,
    'evaluationPoint': 'centerOfLongestCoveredRun',
    'uncertainty': {'bootstrapReps': 200, 'minLeaveOneOutMeasuredFraction': 0.9, 'seed': 20261012,
                    'maxSlopeDeg': 10.0, 'slopeStep': 0.002, 'method': 'maxOfBoth', 'sweepGroups': 5,
                    'sweepGroupLayout': 'interleaved'},
    'coverage': {'stripWidthM': 0.25, 'halfWindowM': 0.5, 'minReturns': 3},
    'shadow': {'obstacleHeightM': [0.3, 3.0], 'skipNearLidarM': 1.0, 'stopBeforeTargetM': 0.5, 'corridorHalfWidthM': 0.25,
               'minObstacleReturns': 3, 'minObstructedFraction': 0.5, 'egoBoxM': [-1.5, 4.5, 1.2]},
}
PARAMS_V4 = copy.deepcopy(PARAMS)
for _key in ('laserTailRegion', 'laserTailLateralMarginM', 'laserTailMaxLateralMarginM', 'laserTailMarginStepM'):
    del PARAMS_V4['estimatorA'][_key]
for _key in ('method', 'sweepGroups', 'sweepGroupLayout'):
    del PARAMS_V4['uncertainty'][_key]
CORRIDOR = (-1.75, 4.75)
EVAL_L = (CORRIDOR[0] + CORRIDOR[1]) / 2
# Median ground-ring radii (m) of the AV2 roof LiDAR pair, from the x/y/z geometry of one development sweep
# (8c52d911 315969946859994000; intensity not read). Irregular spacing, as on the real sensor.
RINGS = [4.36, 6.98, 7.19, 9.73, 10.22, 12.06, 14.48, 14.61, 16.83, 18.95, 20.01, 20.91, 23.36, 24.75, 26.11, 26.52, 30.02, 33.0]


def ring_cloud(paint, seed=1, distances=None, road=4.0, paint_level=20.0, outliers=0.0, worn=0.0, drop=None, sampler=None):
    """Stacked ground returns in crossing coordinates (S along travel from the mapped edge, L right).

    `paint(S, L)` marks painted ground. Each sweep's sensor sits at S=-d, L=0; each ring is one laser with
    its own gain. `drop(S, L)` removes returns (a LiDAR shadow). `sampler(painted, ring_index, rng)` replaces the
    gain/noise intensity model (for intensity distributions measured on development crossings); a sampler with
    attribute `lateral = True` is called as sampler(painted, ring_index, rng, S, L) so the road can vary across
    the corridor (lane line, curbside band). Other samplers draw exactly the same random numbers as before."""
    rng = np.random.default_rng(seed)
    distances = np.linspace(30.0, 10.0, 15) if distances is None else distances
    gains = rng.uniform(0.8, 1.25, len(RINGS))
    rows = []
    for k, d in enumerate(distances):
        pitch, roll = rng.uniform(-0.03, 0.03, 2)  # ring radius varies with ego attitude and road slope
        for j, rho in enumerate(RINGS):
            theta = np.arange(-math.pi / 2, math.pi / 2, math.radians(0.2))
            radius = rho * (1 + pitch + roll * np.sin(theta))
            S = -d + radius * np.cos(theta); L = radius * np.sin(theta)
            keep = (S > -7.6) & (S < 7.6) & (L > CORRIDOR[0] - 0.6) & (L < CORRIDOR[1] + 0.6)
            S, L = S[keep] + rng.normal(0, 0.01, keep.sum()), L[keep]
            painted = paint(S, L)
            if worn:
                painted &= rng.random(len(S)) >= worn
            if sampler is None:
                level = np.where(painted, paint_level, road)
                I = np.clip(np.round(rng.normal(level, 0.25 * level) * gains[j]), 0, 255)
            elif getattr(sampler, 'lateral', False):
                I = np.asarray(sampler(painted, j, rng, S, L), float)
            else:
                I = np.asarray(sampler(painted, j, rng), float)
            if outliers:
                spike = rng.random(len(S)) < outliers
                I[spike] = rng.integers(40, 120, spike.sum())
            rows.append(np.column_stack([S, L, I, np.full(len(S), j), np.full(len(S), k)]))
    pts = np.vstack(rows)
    if drop is not None:
        pts = pts[~drop(pts[:, 0], pts[:, 1])]
    return pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3].astype(int), pts[:, 4].astype(int)


def dark_road_sampler(painted, ring, rng):
    """Development-crossing-8c-like intensities (v4 dev intensity models): dark road (median 2, raw integers) except
    far rings (radius >= 24 m) whose road reads 6-8 and which the road reference window never sees; paint median 13.
    v3 estimator A took such road returns for paint (cutoff 3 x 2 = 6)."""
    n = len(painted)
    if RINGS[ring] >= 24.0:
        road = rng.choice([6, 7, 8], n, p=[0.2, 0.4, 0.4])
    else:
        road = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.22, 0.55, 0.15, 0.04, 0.02, 0.02])
    paint = np.clip(np.round(rng.normal(13.0, 4.0, n)), 0, 255)
    return np.where(painted, paint, road)


def lateral_road_sampler(seed, edge, full=False, corridor=CORRIDOR, band_share=None, line=True, band_on=True):
    """Laterally varying road (v4 review of 8c: a lane line at L 2.0-2.5 m and a bright curbside band at L 3.5-5.5 m
    inside the road reference window). Its own parameters are drawn from `seed`:
      dark lane road as in dark_road_sampler (median 2, far rings 6-8);
      a solid lane line 0.15-0.5 m wide, intensity ~20 x laser gain (25% noise), at a random L in the lane part;
      a curbside band at the right corridor end, 25-45% of the corridor width (or `band_share`), median 5-11 x gain;
      dim paint: per laser, round(band upper tail) + 1, 2 or 3 units, with 25% noise, where the band upper tail is
      the band's 0.90 quantile, about median x gain x (1 + 1.2816 x 0.25).
    full=False: line and band only in the road reference window (S < -3 m), the mechanism the review found;
    full=True: the band runs along the whole road and the line stops 0.5-1.5 m before the paint edge."""
    prm = np.random.default_rng(seed)
    gains = prm.uniform(0.8, 1.25, len(RINGS))
    width = corridor[1] - corridor[0]
    share = prm.uniform(0.25, 0.45) if band_share is None else band_share
    band_lo = corridor[1] - share * width
    band_median = prm.uniform(5.0, 11.0)
    line_w = prm.uniform(0.15, 0.5)
    line_c = prm.uniform(corridor[0] + 0.5 + line_w / 2, max(corridor[0] + 0.5 + line_w / 2, band_lo - 0.5 - line_w / 2))
    line_end = edge - prm.uniform(0.5, 1.5) if full else -3.0
    band_end = 99.0 if full else -3.0
    band_tail = band_median * gains * (1 + 1.2816 * 0.25)
    paint_level = np.round(band_tail) + prm.integers(1, 4, len(RINGS))

    def sampler(painted, ring, rng, S, L):
        I = dark_road_sampler(np.zeros(len(S), bool), ring, rng).astype(float)
        on_band = (L >= band_lo) & (S < band_end) if band_on else np.zeros(len(S), bool)
        on_line = (np.abs(L - line_c) <= line_w / 2) & (S < line_end) if line else np.zeros(len(S), bool)
        I[on_band] = np.round(rng.normal(band_median, 0.25 * band_median, on_band.sum()) * gains[ring])
        I[on_line] = np.round(rng.normal(20.0, 5.0, on_line.sum()) * gains[ring])
        level = paint_level[ring]
        I[painted] = np.round(rng.normal(level, 0.25 * level, painted.sum()))
        return np.clip(I, 0, 255)
    sampler.lateral = True
    sampler.gains = gains; sampler.band_lo = float(band_lo); sampler.band_median = float(band_median)
    sampler.description = {'bandL': [float(band_lo), corridor[1]], 'bandMedian': float(band_median), 'lineL': [float(line_c - line_w / 2), float(line_c + line_w / 2)],
                           'lineEndS': float(line_end), 'bandEndS': float(band_end), 'paintLevelMedian': float(np.median(paint_level))}
    return sampler


def band(edge, width=3.0, slope=0.0):
    return lambda S, L: (S >= edge + slope * L) & (S < edge + slope * L + width)


def frame_stub():
    t = np.array([1.0, 0.0]); r = np.array([t[1], -t[0]])
    a, b = np.array([0.0, 10.0]), np.array([0.0, -10.0])
    return {'t': t, 'r': r, 'origin': np.zeros(2), 'near': (a, b, tool.oriented_normal(a, b, t)),
            'corridor': CORRIDOR, 'evalL': EVAL_L}


def measure(points, params=PARAMS):
    S, L, I, laser, sweep = points
    return tool.measure_edge(S, L, I, laser, sweep, frame_stub(), params)


class StatisticsTests(unittest.TestCase):
    def test_pettitt_matches_naive_double_sum_with_ties(self):
        rng = np.random.default_rng(3)
        x = rng.integers(0, 6, 60).astype(float)
        t, k = tool.pettitt_increasing(x)
        for split, value in zip(t, k):
            naive = sum(np.sign(x[j] - x[i]) for i in range(split) for j in range(split, len(x)))
            self.assertAlmostEqual(value, naive, places=9)

    def test_change_point_locates_step_and_effect_size(self):
        x = np.r_[np.full(50, 1.0), np.full(50, 5.0)]
        c = tool.change_point(x, 10)
        self.assertEqual((c['first'], c['last']), (50, 50))
        self.assertAlmostEqual(c['auc'], 1.0)
        self.assertLess(c['p'], 1e-6)
        self.assertTrue(c['increase'])
        self.assertIsNone(tool.change_point(x[:15], 10))
        self.assertFalse(tool.change_point(x[::-1], 10)['increase'])

    def test_l1_step_keeps_bright_road_value_before_imbalanced_step(self):
        # many road values, one slightly bright road value last, then few paint values: the rank statistic
        # alone would move the split early; the L1 step keeps the boundary at the true change.
        x = np.r_[np.tile([1.0, 0.8, 1.1, 0.9], 20), 1.25, np.full(12, 5.5)]
        c = tool.change_point(x, 6)
        self.assertEqual(c['first'], 81)

    def test_theil_sen_exact_and_robust(self):
        x = np.arange(6.0); y = 0.3 + 0.05 * (x - 2.0)
        value, slope = tool.theil_sen(x, y, 2.0)
        self.assertAlmostEqual(value, 0.3); self.assertAlmostEqual(slope, 0.05)
        y[5] = 9.0
        value, slope = tool.theil_sen(x, y, 2.0)
        self.assertAlmostEqual(value, 0.3, places=6); self.assertAlmostEqual(slope, 0.05, places=6)
        with self.assertRaises(ValueError):
            tool.theil_sen([1.0, 1.0], [0.0, 1.0], 0.0)

    def test_interval_line_uses_intersection_of_strip_intervals(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        lo = np.array([0.20, 0.10, 0.30, 0.25]); hi = np.array([0.50, 0.35, 0.60, 0.40])
        fit = tool.interval_line(x, lo, hi, 1.5, np.array([0.0]))
        self.assertAlmostEqual(fit['violationM'], 0.0)
        self.assertEqual(fit['valueRange'], [0.30, 0.35])
        self.assertAlmostEqual(fit['value'], 0.325)
        fit = tool.interval_line(x, lo + 0.1 * x, hi + 0.1 * x, 0.0, tool.slope_grid({'maxSlopeDeg': 10.0, 'slopeStep': 0.002}))
        self.assertLessEqual(fit['slopeRange'][0], 0.1); self.assertGreaterEqual(fit['slopeRange'][1], 0.1)
        self.assertAlmostEqual(fit['slope'], 0.1, delta=0.03)
        outlier = tool.interval_line(np.r_[x, 4.0], np.r_[lo, -2.0], np.r_[hi, -1.9], 1.5, np.array([0.0]))
        self.assertAlmostEqual(outlier['value'], 0.30, delta=0.051)

    def test_partition_offsets_and_minimum_width(self):
        self.assertEqual(tool.partition(0.0, 1.2, 0.5, 0.0, 0.25), [(0.0, 0.5), (0.5, 1.0)])
        self.assertEqual(tool.partition(0.0, 2.3, 1.0, 0.25, 0.5), [(0.25, 1.25), (1.25, 2.25)])

    @staticmethod
    def loo(sd_a=0.02, sd_b=0.02, frac_a=1.0, frac_b=1.0, agreeing=15, sd_mean=0.02):
        return {'runs': 15, 'agreeing': agreeing, 'A': {'measuredFraction': frac_a, 'jackknifeSdM': sd_a},
                'B': {'measuredFraction': frac_b, 'jackknifeSdM': sd_b}, 'mean': {'bothMeasured': 15, 'jackknifeSdM': sd_mean}}

    def test_combine_tolerance_and_precision_rules(self):
        a = {'status': 'measured', 'edgeSAtEval': 0.30}
        b = {'status': 'measured', 'edgeSAtEval': 0.16}
        out = tool.combine(a, b, self.loo(), PARAMS)
        self.assertEqual(out['status'], 'resolved'); self.assertAlmostEqual(out['paintEdgeS'], 0.23)
        self.assertEqual(out['label'], 'resolved')
        self.assertAlmostEqual(out['uncertainty']['combinedM'], math.hypot(0.02, 0.07))
        b['edgeSAtEval'] = 0.149
        out = tool.combine(a, b, self.loo(), PARAMS)
        self.assertEqual(out['status'], 'unresolved'); self.assertNotIn('paintEdgeS', out)
        b['edgeSAtEval'] = 0.25
        for loo in (self.loo(sd_b=0.2), self.loo(sd_a=None), self.loo(frac_a=13 / 15)):
            out = tool.combine(a, b, loo, PARAMS)
            self.assertEqual(out['status'], 'unresolved'); self.assertIn('jackknife', out['reason'])
        self.assertEqual(tool.combine(a, b, self.loo(frac_a=14 / 15), PARAMS)['status'], 'resolved')
        self.assertEqual(tool.combine({'status': 'unresolved'}, b, None, PARAMS)['status'], 'unresolved')

    def test_marginal_label_when_a_leave_one_out_run_disagrees_or_mean_is_imprecise(self):
        a = {'status': 'measured', 'edgeSAtEval': 0.30}; b = {'status': 'measured', 'edgeSAtEval': 0.25}
        self.assertEqual(tool.combine(a, b, self.loo(agreeing=14), PARAMS)['label'], 'marginally resolved')
        self.assertEqual(tool.combine(a, b, self.loo(sd_mean=0.08), PARAMS)['label'], 'marginally resolved')

    def test_stated_plus_minus_is_two_combined_sigma_and_labelled(self):
        a = {'status': 'measured', 'edgeSAtEval': 0.30}; b = {'status': 'measured', 'edgeSAtEval': 0.25}
        u = tool.combine(a, b, self.loo(), PARAMS)['uncertainty']
        self.assertAlmostEqual(u['statedPlusMinusM'], 2 * u['combinedM'])
        self.assertIn('2 x combinedM', u['statedPlusMinusLabel']); self.assertIn('not calibrated', u['statedPlusMinusLabel'])

    def test_sweep_groups_and_precision_methods(self):
        self.assertEqual(tool.sweep_groups(range(15), 5, 'interleaved'),
                         [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]])
        self.assertEqual(tool.sweep_groups(range(15), 5, 'consecutive')[0], [0, 1, 2])
        with self.assertRaises(ValueError):
            tool.sweep_groups(range(15), 5, 'random')
        loo = self.loo(sd_a=0.02, sd_b=0.05, frac_a=1.0, frac_b=14 / 15, agreeing=15, sd_mean=0.03)
        grp = {'runs': 5, 'agreeing': 4, 'A': {'measuredFraction': 0.8, 'jackknifeSdM': 0.04},
               'B': {'measuredFraction': 1.0, 'jackknifeSdM': 0.01}, 'mean': {'bothMeasured': 5, 'jackknifeSdM': 0.02}}
        self.assertIs(tool.precision(loo, None, 'leaveOneSweepOut'), loo)
        self.assertIs(tool.precision(loo, grp, 'sweepGroups'), grp)
        both = tool.precision(loo, grp, 'maxOfBoth')
        self.assertEqual((both['runs'], both['agreeing']), (20, 19))
        self.assertEqual((both['A']['jackknifeSdM'], both['A']['measuredFraction']), (0.04, 0.8))
        self.assertEqual((both['B']['jackknifeSdM'], both['B']['measuredFraction']), (0.05, 14 / 15))
        self.assertEqual(both['mean']['jackknifeSdM'], 0.03)
        with self.assertRaises(ValueError):
            tool.precision(loo, None, 'sweepGroups')

    def test_jackknife_sd_formula(self):
        v = [0.1, 0.2, 0.4, None]
        x = np.array([0.1, 0.2, 0.4])
        self.assertAlmostEqual(tool.jackknife_sd(v), math.sqrt(2 / 3 * np.sum((x - x.mean()) ** 2)))
        self.assertIsNone(tool.jackknife_sd([0.3, None]))


class EstimatorTests(unittest.TestCase):
    def assert_resolved_near(self, result, expected, tol=0.08):
        a, b, final = result
        self.assertEqual(a['status'], 'measured', a.get('reason'))
        self.assertEqual(b['status'], 'measured', b.get('reason'))
        self.assertEqual(final['status'], 'resolved', final.get('reason'))
        self.assertAlmostEqual(a['edgeSAtEval'], expected, delta=tol)
        self.assertAlmostEqual(b['edgeSAtEval'], expected, delta=tol)
        self.assertAlmostEqual(final['paintEdgeS'], expected, delta=tol)
        self.assertGreater(final['uncertainty']['combinedM'], 0)

    def assert_near_or_unresolved_by_jackknife(self, result, expected, tol=0.10):
        """Both estimators measure near the truth on all sweeps; the edge is either resolved near it or left
        unresolved by the leave-out precision gate (never confidently wrong)."""
        a, b, final = result
        self.assertAlmostEqual(a['edgeSAtEval'], expected, delta=tol)
        self.assertAlmostEqual(b['edgeSAtEval'], expected, delta=tol)
        if final['status'] == 'resolved':
            self.assertAlmostEqual(final['paintEdgeS'], expected, delta=tol)
        else:
            self.assertIn('jackknife', final['reason'])

    def test_known_edge_beyond_map(self):
        self.assert_resolved_near(measure(ring_cloud(band(0.40))), 0.40)

    def test_known_edge_before_map(self):
        # v5 (max of the leave-one-sweep-out and 5 interleaved sweep-group jackknifes): B does not measure when
        # sweeps {3, 8, 13} are left out, so B measures in 4 of 5 group runs (< 90%) and this cloud is unresolved;
        # under v4 (leave-one-sweep-out only) it resolves.
        result = measure(ring_cloud(band(-1.20)))
        self.assert_near_or_unresolved_by_jackknife(result, -1.20, tol=0.08)
        self.assertEqual(result[2]['status'], 'unresolved')
        self.assertEqual(result[2]['sweepGroupJackknife']['B']['measuredFraction'], 0.8)
        self.assert_resolved_near(measure(ring_cloud(band(-1.20)), PARAMS_V4), -1.20)

    def test_skewed_edge_recovers_slope(self):
        a, b, final = measure(ring_cloud(band(0.20, slope=0.05)))
        self.assertAlmostEqual(final['paintEdgeS'], 0.20 + 0.05 * EVAL_L, delta=0.08)
        self.assertAlmostEqual(a['slope'], 0.05, delta=0.03)
        self.assertAlmostEqual(b['slope'], 0.05, delta=0.03)

    def test_continental_bars_with_gaps(self):
        # v5 notes. Tail window: with a fixed +/- 0.5 m window (no widening) the far rings (24-26 m) have fewer than 20
        # reference returns per strip and are dropped; in this cloud they hold the returns nearest the edge, so A's
        # leave-one-sweep-out sd rose to 0.16 m. The per-laser widening to at most +/- 1.0 m keeps them (A -0.296 m,
        # leave-one-sweep-out sd 0.049 m). Uncertainty: leaving out sweeps {2, 7, 12} moves A and B to about -0.40 m, so
        # the sweep-group jackknife sds are 0.106 m (A) and 0.088 m (B) > 0.075 m and v5 leaves the cloud unresolved.
        bars = lambda S, L: band(-0.30)(S, L) & (np.mod(L - CORRIDOR[0], 1.2) < 0.6)
        result = measure(ring_cloud(bars, seed=4))
        self.assert_near_or_unresolved_by_jackknife(result, -0.30, tol=0.10)
        self.assertGreater(result[0]['stripStatus'].get('noPaint', 0), 0)
        self.assertLessEqual(result[2]['leaveOneSweepOut']['A']['jackknifeSdM'], 0.075)
        self.assert_resolved_near(measure(ring_cloud(bars, seed=4), PARAMS_V4), -0.30, tol=0.10)

    def test_narrow_bars_like_a_continental_crosswalk(self):
        bars = lambda S, L: band(0.15)(S, L) & (np.mod(L - CORRIDOR[0], 0.75) < 0.3)
        a, b, final = measure(ring_cloud(bars, seed=14))
        self.assertEqual(final['status'], 'resolved', final.get('reason'))
        self.assertAlmostEqual(final['paintEdgeS'], 0.15, delta=0.10)

    def test_ladder_border_line_then_bars(self):
        # B is less stable on this pattern (its leave-one-out jackknife sd often exceeds the precision rule), so the
        # crossing may be unresolved; A measures the border line and nothing resolves far from it.
        ladder = lambda S, L: band(0.0, width=0.4)(S, L) | (band(0.6, width=2.4)(S, L) & (np.mod(L - CORRIDOR[0], 1.2) < 0.6))
        for seed in (15, 25):
            a, b, final = measure(ring_cloud(ladder, seed=seed))
            self.assertAlmostEqual(a['edgeSAtEval'], 0.0, delta=0.08)
            if final['status'] == 'resolved':
                self.assertAlmostEqual(final['paintEdgeS'], 0.0, delta=0.10)
            else:
                self.assertNotIn('paintEdgeS', final)

    def test_noise_and_bright_outliers(self):
        self.assert_resolved_near(measure(ring_cloud(band(0.55), seed=5, outliers=0.02)), 0.55, tol=0.10)

    def test_faded_paint_near_ratio_is_never_confidently_wrong(self):
        for seed in (6, 16, 26):
            a, b, final = measure(ring_cloud(band(0.40), seed=seed, paint_level=9.0))
            if final['status'] == 'resolved':
                self.assertAlmostEqual(final['paintEdgeS'], 0.40, delta=0.15)
            else:
                self.assertNotIn('paintEdgeS', final)

    def test_patchy_worn_paint_still_measured_or_unresolved_never_biased(self):
        a, b, final = measure(ring_cloud(band(0.40), seed=7, worn=0.35))
        if final['status'] == 'resolved':
            self.assertAlmostEqual(final['paintEdgeS'], 0.40, delta=0.15)

    def test_no_paint_is_unresolved(self):
        a, b, final = measure(ring_cloud(lambda S, L: np.zeros(len(S), bool), seed=8))
        self.assertEqual(final['status'], 'unresolved')
        self.assertEqual(a['edgeStrips'], 0)

    def test_limit_line_before_crosswalk_makes_estimators_disagree(self):
        paint = lambda S, L: band(-1.5, width=0.3)(S, L) | band(0.0)(S, L)
        a, b, final = measure(ring_cloud(paint, seed=9))
        self.assertTrue(any(r['status'] == 'edge' and r['interval'][1] < -1.0 for r in a['strips']))
        self.assertAlmostEqual(b['edgeSAtEval'], 0.0, delta=0.1)
        self.assertEqual(final['status'], 'unresolved')

    def test_shadowed_strips_drop_out_but_edge_survives(self):
        shadow = lambda S, L: (L > 2.75) & (L < 4.75) & (S > -2.0)
        S, L, I, laser, k = ring_cloud(band(0.40), seed=10, drop=shadow)
        a, b, final = tool.measure_edge(S, L, I, laser, k, frame_stub(), PARAMS)
        self.assertEqual(final['status'], 'resolved', final.get('reason'))
        self.assertAlmostEqual(final['paintEdgeS'], 0.40, delta=0.10)
        hidden = [r['status'] for r in a['strips'] if 3.0 <= (r['L'][0] + r['L'][1]) / 2 <= 4.5]
        # nothing was observed at the edge there: its own status, not 'noPaint'
        self.assertTrue(hidden and all(s == 'noEdgeReturns' for s in hidden), hidden)

    def test_coverage_marks_occluded_strips_shadowed(self):
        shadow = lambda S, L: (L > 2.75) & (L < 4.75) & (S > -2.0)
        S, L, I, laser, k = ring_cloud(band(0.40), seed=11, distances=np.linspace(24, 10, 4), drop=shadow)
        frame = frame_stub()
        lidars = [np.array([-d, 0.0, 1.9]) for d in np.linspace(24, 10, 4)]
        # a box obstacle at S in [-3, -2], L in [2.75, 4.75] (right is -y in city): returns on its faces
        box = np.array([[s, -l] for s in np.linspace(-3, -2, 11) for l in np.linspace(2.75, 4.75, 41)])
        stack = {'lidars': lidars, 'elevated': [box] * 4}
        cov = tool.coverage(S, L, frame, (0.40, 0.0), PARAMS, stack)
        self.assertGreater(cov['shadowedStrips'], 0)
        self.assertTrue(all(lo >= 2.5 for lo, _ in cov['shadowedL']))
        self.assertLess(cov['coveredFraction'], 1.0)
        unblocked = tool.coverage(S, L, frame, (0.40, 0.0), PARAMS, {'lidars': lidars, 'elevated': [np.zeros((0, 2))] * 4})
        self.assertEqual(unblocked['shadowedStrips'], 0)


class DarkRoadTests(unittest.TestCase):
    """The v3 failure mode: on a dark road (median 2) A's 3x cutoff is 6, which far-ring road returns pass."""

    def test_laser_tails_come_from_the_reference_window_and_need_enough_returns(self):
        S = np.r_[np.full(30, -5.0), np.full(30, -5.0), np.full(5, -5.0), np.full(10, 0.5)]
        I = np.r_[np.arange(30.0), np.full(30, 2.0), np.full(5, 9.0), np.full(10, 50.0)]
        laser = np.r_[np.zeros(30, int), np.ones(30, int), np.full(5, 2), np.zeros(10, int)]
        tails = tool.laser_tails(S, I, laser, PARAMS, PARAMS['windows'])
        self.assertAlmostEqual(tails[0], np.quantile(np.arange(30.0), 0.90))
        self.assertAlmostEqual(tails[1], 2.0)
        self.assertNotIn(2, tails)  # 5 reference returns < laserReferenceMinReturns

    def test_first_onset_needs_consecutive_paint(self):
        A = PARAMS['estimatorA']
        ss = np.round(np.arange(-1.0, 1.0, 0.05), 3)
        paint = ss >= 0.0
        paint[np.searchsorted(ss, -0.2)] = True  # one bright road return 0.2 m early, then road again
        j = tool.first_onset(ss, paint, A)
        self.assertAlmostEqual(ss[j], 0.0)
        # v3 accepted it: >= 3 paint returns making >= 50% of the next 0.5 m
        w = (ss >= -0.2) & (ss < 0.3)
        self.assertTrue(paint[w].sum() >= 3 and paint[w].mean() >= 0.5)
        worn = ss >= 0.0
        worn[np.searchsorted(ss, 0.05)] = False  # 2 of 3 consecutive returns are enough
        self.assertAlmostEqual(ss[tool.first_onset(ss, worn, A)], 0.0)

    def test_dark_road_with_bright_unreferenced_far_rings_is_not_biased_early(self):
        for edge, seed in ((0.0, 31), (0.40, 32), (-0.30, 33)):
            S, L, I, laser, k = ring_cloud(band(edge), seed=seed, sampler=dark_road_sampler)
            far = [j for j, rho in enumerate(RINGS) if rho >= 24.0]
            pre = np.isin(laser, far) & (S < edge) & (S >= -3)
            self.assertGreater(np.mean(I[pre] >= 6), 0.9)  # the v3 cutoff (3 x 2) passes them
            a = tool.estimator_a(S, L, I, laser, CORRIDOR, EVAL_L, PARAMS, PARAMS['windows'], strip_bootstrap=False)
            self.assertEqual(a['status'], 'measured', a.get('reason'))
            self.assertAlmostEqual(a['edgeSAtEval'], edge, delta=0.06)
            self.assertGreater(a['returnsWithoutLaserReference'], 0)


class LateralRoadTests(unittest.TestCase):
    """Protocol v5: estimator A's laser tails come from the strip's own lateral neighbourhood (strip +/- 0.5 m in the
    road reference window, widened per laser in 0.25 m steps to at most +/- 1.0 m when the laser has fewer than 20
    returns there), so a bright lane line or curbside band elsewhere cannot raise them (v4 review, 8c)."""

    def test_tail_window_widens_the_strip_and_clips_to_the_corridor(self):
        m = PARAMS['estimatorA']['laserTailLateralMarginM']
        self.assertEqual(tool.tail_window(0.0, 0.15, CORRIDOR, m), (-0.5, 0.65))
        self.assertEqual(tool.tail_window(-1.75, -1.6, CORRIDOR, m), (-1.75, -1.1))
        self.assertEqual(tool.tail_window(4.6, 4.75, CORRIDOR, m), (4.1, 4.75))

    def test_strip_tails_ignore_bright_returns_outside_the_window_and_drop_sparse_lasers(self):
        corridor = (0.0, 3.0)
        ss = np.round(np.arange(-1.0, 1.0, 0.02), 3)
        S = np.r_[np.full(40, -5.0), np.full(40, -5.0), np.full(10, -5.0), ss, np.full(5, 0.5)]
        L = np.r_[np.full(40, 0.05), np.full(40, 2.5), np.full(10, 0.05), np.full(len(ss), 0.05), np.full(5, 0.05)]
        I = np.r_[np.full(40, 2.0), np.full(40, 30.0), np.full(10, 2.0), np.where(ss >= 0.0, 12.0, 2.0), np.full(5, 50.0)]
        laser = np.r_[np.zeros(40, int), np.zeros(40, int), np.ones(10, int), np.zeros(len(ss), int), np.ones(5, int)]
        v5 = tool.estimator_a(S, L, I, laser, corridor, 0.075, PARAMS, PARAMS['windows'], strip_bootstrap=False)
        row = v5['strips'][0]
        self.assertEqual(row['tailWindowL'], [0.0, 0.65])
        self.assertEqual(row['lasersWithTail'], 1)                 # laser 1 has 10 < 20 reference returns even at +/- 1.0 m
        self.assertEqual(row['lasersWidened'], 0)
        self.assertEqual(row['returnsWithoutLaserReference'], 5)    # its bright search-window returns are not used
        self.assertEqual(row['status'], 'edge')
        self.assertLessEqual(row['interval'][0], 0.0); self.assertGreaterEqual(row['interval'][1], 0.0)
        # version 4: laser 0's corridor-wide 0.90 quantile is 30 (the bright line at L 2.5 m), so the 12-unit paint fails
        v4 = tool.estimator_a(S, L, I, laser, corridor, 0.075, PARAMS_V4, PARAMS['windows'], strip_bootstrap=False)
        self.assertEqual(v4['strips'][0]['status'], 'noPaint')
        bad = copy.deepcopy(PARAMS); bad['estimatorA']['laserTailRegion'] = 'coveredRun'
        with self.assertRaises(ValueError):
            tool.estimator_a(S, L, I, laser, corridor, 0.075, bad, PARAMS['windows'], strip_bootstrap=False)

    def test_sparse_laser_widens_its_own_window_in_steps_up_to_the_cap(self):
        corridor = (0.0, 3.0)
        A = PARAMS['estimatorA']
        self.assertEqual(tool.tail_margins(A), [0.5, 0.75, 1.0])
        self.assertEqual(tool.tail_margins(PARAMS_V4['estimatorA'] | {'laserTailLateralMarginM': 0.5}), [0.5])
        # laser 0: 30 returns at L 0.1 (inside +/- 0.5 m of strip [0, 0.15)); laser 1: 12 at L 0.6 and 12 at L 0.85
        # (24 within +/- 0.75 m, 12 within +/- 0.5 m); laser 2: 25 at L 1.2 (outside +/- 1.0 m); a bright line at L 2.5.
        S = np.full(30 + 24 + 25 + 40, -5.0)
        L = np.r_[np.full(30, 0.1), np.full(12, 0.6), np.full(12, 0.85), np.full(25, 1.2), np.full(40, 2.5)]
        I = np.r_[np.arange(30) % 5, np.full(24, 3.0), np.full(25, 9.0), np.full(40, 30.0)]
        laser = np.r_[np.zeros(30, int), np.ones(24, int), np.full(25, 2), np.zeros(40, int)]
        in_ref = np.ones(len(S), bool)
        tails, used = tool.strip_laser_tails(S, L, I, laser, in_ref, 0.0, 0.15, corridor, tool.tail_margins(A), PARAMS, PARAMS['windows'])
        self.assertEqual(used, {0: 0.5, 1: 0.75})
        self.assertEqual(tails[0], float(np.quantile(np.arange(30) % 5, 0.9)))   # the bright line 2.4 m away is not used
        self.assertEqual(tails[1], 3.0)
        self.assertNotIn(2, tails)

    def test_bright_curbside_band_in_reference_window_moves_a_by_at_most_3cm(self):
        # 8c-like: dark lane road; a bright curbside band over the right 40% of the corridor only in S -7..-3; that
        # side shadowed near the edge (8c: stationary vehicle); dim paint 1-3 units above the band's upper tail.
        # The same cloud is measured with and without the band (intensities replaced with a separate generator).
        # Seeds 5 and 6 were picked from 12 tried (0-11) because the version-4 corridor-wide tails move A by more than
        # 0.03 m there (0.043, 0.056 m); the version-5 shift was <= 0.0283 m in all 12 (fixed +/- 0.5 m window and the
        # adopted widening to +/- 1.0 m alike).
        for seed in (5, 6):
            smp = lateral_road_sampler(seed, 0.0, band_share=0.4, line=False, band_on=False)
            edge = float(np.random.default_rng(seed).uniform(-0.8, 0.8))
            shadow = lambda S, L: (L >= smp.band_lo) & (S > -2.0)
            S, L, I, laser, k = ring_cloud(band(edge), seed=seed, sampler=smp, drop=shadow)
            on = (L >= smp.band_lo) & (S >= -7.0) & (S < -3.0)
            rng = np.random.default_rng(seed + 1000)
            J = I.copy()
            J[on] = np.clip(np.round(rng.normal(smp.band_median, 0.25 * smp.band_median, on.sum()) * smp.gains[laser[on]]), 0, 255)
            shift = {}
            for name, params in (('v5', PARAMS), ('v4', PARAMS_V4)):
                a0 = tool.estimator_a(S, L, I, laser, CORRIDOR, EVAL_L, params, params['windows'], strip_bootstrap=False)
                a1 = tool.estimator_a(S, L, J, laser, CORRIDOR, EVAL_L, params, params['windows'], strip_bootstrap=False)
                self.assertEqual(a1['status'], 'measured', a1.get('reason'))
                shift[name] = a1['edgeSAtEval'] - a0['edgeSAtEval']
                if name == 'v5':  # strips whose widest tail window (+/- 1.0 m) misses the band are untouched
                    far = [i for i, r in enumerate(a0['strips']) if r['L'][1] + 1.0 <= smp.band_lo]
                    self.assertGreater(len(far), 15)
                    for i in far:
                        self.assertEqual(a0['strips'][i], a1['strips'][i])
            self.assertLessEqual(abs(shift['v5']), 0.03, shift)
            self.assertGreater(abs(shift['v4']), 0.03, shift)

    def test_lateral_road_sampler_matches_its_description(self):
        smp = lateral_road_sampler(3, 0.2, full=True)
        S, L, I, laser, k = ring_cloud(band(0.2), seed=3, sampler=smp)
        d = smp.description
        road = (S < 0.2)
        in_band = road & (L >= d['bandL'][0]); lane = road & (L < d['lineL'][0] - 0.3)
        self.assertGreater(np.median(I[in_band]), 3 * np.median(I[lane]) - 1)
        line = (L > d['lineL'][0] + 0.02) & (L < d['lineL'][1] - 0.02) & (S < d['lineEndS'])
        self.assertGreater(np.median(I[line]), 14)
        self.assertLess(np.median(I[(L > d['lineL'][0]) & (L < d['lineL'][1]) & (S > d['lineEndS'] + 0.05) & (S < 0.2)]), 4)
        self.assertTrue(-1.5 <= d['lineEndS'] - 0.2 <= -0.5)


class VariantDiagnosticsTests(unittest.TestCase):
    def test_estimator_a_variants_reproduce_the_named_rules(self):
        S, L, I, laser, k = ring_cloud(band(0.1), seed=21)
        frame = frame_stub()
        out = tool.estimator_a_variants(S, L, I, laser, frame, PARAMS)
        v4 = tool.estimator_a(S, L, I, laser, CORRIDOR, EVAL_L, PARAMS_V4, PARAMS['windows'], strip_bootstrap=False)
        self.assertEqual(out['corridorWideTailsV4']['edgeSAtEval'], v4['edgeSAtEval'])
        self.assertEqual(out['corridorWideTailsV4']['laserTailRegion'], 'corridor')
        fixed = copy.deepcopy(PARAMS)
        for key in ('laserTailMaxLateralMarginM', 'laserTailMarginStepM'):
            del fixed['estimatorA'][key]
        f = tool.estimator_a(S, L, I, laser, CORRIDOR, EVAL_L, fixed, PARAMS['windows'], strip_bootstrap=False)
        self.assertEqual(out['fixedWindowNoWidening']['edgeSAtEval'], f['edgeSAtEval'])
        self.assertEqual(set(out), {'note', 'corridorWideTailsV4', 'noRunRule', 'laserTailAtMinimum', 'fixedWindowNoWidening'})


class EvaluationPointTests(unittest.TestCase):
    def test_eval_point_avoids_a_lidar_shadow(self):
        shadow = lambda S, L: (L > 2.75) & (L < 4.75) & (S > -2.0)
        S, L, I, laser, k = ring_cloud(band(0.40), seed=17, drop=shadow)
        point = tool.observed_eval_point(S, L, frame_stub(), PARAMS)
        lo, hi = point['coveredRunL']
        self.assertAlmostEqual(lo, CORRIDOR[0]); self.assertLessEqual(hi, 3.0)
        self.assertAlmostEqual(point['evalL'], (lo + hi) / 2)

    def test_eval_point_is_corridor_center_when_fully_covered_or_empty(self):
        S, L, I, laser, k = ring_cloud(band(0.40), seed=18)
        self.assertAlmostEqual(tool.observed_eval_point(S, L, frame_stub(), PARAMS)['evalL'], EVAL_L, places=6)
        empty = tool.observed_eval_point(np.zeros(0), np.zeros(0), frame_stub(), PARAMS)
        self.assertIsNone(empty['coveredRunL']); self.assertAlmostEqual(empty['evalL'], EVAL_L)

    def test_value_at_corridor_center_is_reported_with_its_own_sd(self):
        frame = dict(frame_stub()); frame['corridorCenterL'] = EVAL_L; frame['evalL'] = 0.0
        S, L, I, laser, k = ring_cloud(band(0.30, slope=0.03), seed=19)
        a, b, final = tool.measure_edge(S, L, I, laser, k, frame, PARAMS)
        self.assertAlmostEqual(a['edgeSAtEval'], 0.30, delta=0.08)
        self.assertAlmostEqual(a['atCorridorCenter']['edgeS'], 0.30 + 0.03 * EVAL_L, delta=0.08)
        self.assertGreater(a['atCorridorCenter']['stripBootstrapSd'], 0)


class GeometryTests(unittest.TestCase):
    def test_point_at_roundtrip_on_skewed_edge(self):
        t = tool.unit([1.0, 0.1]); r = np.array([t[1], -t[0]])
        a, b = np.array([2.0, 8.0]), np.array([1.0, -9.0])
        n = tool.oriented_normal(a, b, t)
        origin = a + 0.4 * (b - a)
        frame = {'t': t, 'r': r, 'origin': origin, 'near': (a, b, n)}
        for lat, s in ((0.0, 0.0), (3.0, 0.5), (-4.0, -1.2)):
            p = tool.point_at(frame, lat, s)
            self.assertAlmostEqual(float((p - origin) @ r), lat, places=9)
            self.assertAlmostEqual(float(tool.along_line(p[None], a, n, t)[0]), s, places=9)

    def test_ground_heights_match_preparation_rule(self):
        try:
            sys.path.insert(0, str(ROOT / 'scripts/california'))
            import prepare_av2
        except Exception as error:  # pragma: no cover - optional heavy dependency
            self.skipTest(f'prepare_av2 unavailable: {error}')
        rng = np.random.default_rng(12)
        raster = rng.normal(0, 1, (40, 60)).astype(np.float16)
        sim = {'R': [1.0, 0.0, 0.0, 1.0], 't': [10.0, 5.0], 's': 3.3333333333333335}
        xy = rng.uniform(-15, 15, (500, 2))
        ours = tool.ground_heights(xy, raster, sim)
        theirs = prepare_av2.ground_sample(np.column_stack([xy, np.zeros(len(xy))]), raster, sim)['nominalGroundHeights']
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
        np.testing.assert_array_equal(ours[~np.isnan(ours)], theirs[~np.isnan(theirs)])


# ------------------------------------------------------------------ end-to-end synthetic log (select + measure CLI)
def write_feather(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(pa.table(columns), str(path), compression='uncompressed')


def synthetic_log(root, edge=0.40, xs=None):
    """Ego drives +x along y=0 toward a crossing whose mapped near edge is x=0; paint starts at x=edge."""
    log = '00000000-0000-0000-0000-00000000c0de'
    xs = np.arange(-30.0, -4.0, 1.0) if xs is None else xs
    stamps = [315000000000000000 + i * 100_000_000 for i in range(len(xs))]
    n = len(xs)
    write_feather(root / 'poses.feather', {'timestamp_ns': pa.array(stamps, pa.int64()), 'qw': [1.0] * n, 'qx': [0.0] * n,
                                           'qy': [0.0] * n, 'qz': [0.0] * n, 'tx_m': list(xs), 'ty_m': [0.0] * n, 'tz_m': [0.0] * n})
    write_feather(root / 'extrinsics.feather', {'sensor_name': ['up_lidar', 'down_lidar'], 'qw': [1.0, 0.0], 'qx': [0.0, 1.0],
                                                'qy': [0.0, 0.0], 'qz': [0.0, 0.0], 'tx_m': [1.35, 1.35], 'ty_m': [0.0, 0.0],
                                                'tz_m': [1.64, 1.53]})
    def pt(x, y):
        return {'x': x, 'y': y, 'z': 0.0}
    vector_map = {
        'pedestrian_crossings': {'77': {'id': 77, 'edge1': [pt(0.0, 10.0), pt(0.0, -10.0)], 'edge2': [pt(3.5, -10.0), pt(3.5, 10.0)]}},
        'lane_segments': {
            '1': {'id': 1, 'lane_type': 'VEHICLE', 'is_intersection': False,
                  'left_lane_boundary': [pt(-40, 2.0), pt(20, 2.0)], 'right_lane_boundary': [pt(-40, -2.0), pt(20, -2.0)]},
            '2': {'id': 2, 'lane_type': 'VEHICLE', 'is_intersection': False,
                  'left_lane_boundary': [pt(20, 2.0), pt(-40, 2.0)], 'right_lane_boundary': [pt(20, 6.0), pt(-40, 6.0)]}},
        'drivable_areas': {'9': {'id': 9, 'area_boundary': [pt(-40, -5.0), pt(20, -5.0), pt(20, 9.0), pt(-40, 9.0)]}},
    }
    (root / 'map.json').write_text(json.dumps(vector_map))
    raster = np.zeros((40, 100), np.float16); buf = io.BytesIO(); np.save(buf, raster)
    (root / 'ground.npy').write_bytes(buf.getvalue())
    (root / 'sim2.json').write_text(json.dumps({'R': [1.0, 0.0, 0.0, 1.0], 't': [50.0, 20.0], 's': 1.0}))
    keys = ''.join(f'<Contents><Key>datasets/av2/sensor/train/{log}/sensors/lidar/{ts}.feather</Key></Contents>' for ts in stamps)
    (root / 'listing.xml').write_text(f'<ListBucketResult><IsTruncated>false</IsTruncated>{keys}</ListBucketResult>')
    rng = np.random.default_rng(13)
    gains = rng.uniform(0.85, 1.2, len(RINGS))
    receipts = []
    for ts, x0 in zip(stamps, xs):
        rows = []
        for j, rho in enumerate(RINGS):
            theta = np.arange(-math.pi, math.pi, math.radians(0.2))
            ex = 1.35 + rho * np.cos(theta); ey = rho * np.sin(theta)
            cx, cy = x0 + ex, ey
            keep = (cx > -9) & (cx < 12) & (np.abs(cy) < 8)
            ex, ey, cx, cy = ex[keep], ey[keep], cx[keep], cy[keep]
            painted = (cx >= edge) & (cx < 3.5)
            level = np.where(painted, 20.0, 4.0)
            inten = np.clip(np.round(rng.normal(level, 0.25 * level) * gains[j]), 0, 255)
            # shadow behind an obstacle right of the lane: no ground returns at y in [-4.6,-3.0] beyond x=-1
            visible = ~((cy < -3.0) & (cy > -4.6) & (cx > -1.0))
            rows.append(np.column_stack([ex[visible], ey[visible], rng.normal(0, 0.02, visible.sum()), inten[visible], np.full(visible.sum(), j)]))
        obstacle = np.array([[bx - x0, by, h, 10.0, 99] for bx in np.linspace(-3, -1.5, 8) for by in np.linspace(-4.6, -3.0, 9) for h in (0.5, 1.0)])
        rows.append(obstacle)
        cloud = np.vstack(rows)
        path = root / 'sweeps' / f'{ts}.feather'
        write_feather(path, {'x': pa.array(cloud[:, 0].astype(np.float16)), 'y': pa.array(cloud[:, 1].astype(np.float16)),
                             'z': pa.array(cloud[:, 2].astype(np.float16)), 'intensity': pa.array(cloud[:, 3].astype(np.uint8)),
                             'laser_number': pa.array(cloud[:, 4].astype(np.uint8)), 'offset_ns': pa.array(np.zeros(len(cloud), np.int32))})
        receipts.append({'timestampNs': str(ts), 'path': f'sweeps/{ts}.feather', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'receipts.json').write_text(json.dumps({'files': receipts}))
    sources = {}
    for role, name in (('poses', 'poses.feather'), ('extrinsics', 'extrinsics.feather'), ('map', 'map.json'), ('groundRaster', 'ground.npy'),
                       ('groundTransform', 'sim2.json'), ('lidarListing', 'listing.xml')):
        sources[role] = {'path': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()}
    (root / 'config.json').write_text(json.dumps({'schemaVersion': 1, 'logId': log, 'crossingId': '77', 'split': 'synthetic', 'sources': sources}))
    protocol = {'protocol': 'pao-paint-edge', 'version': 'test', 'parameters': PARAMS, 'limits': ['synthetic']}
    (root / 'protocol.json').write_text(json.dumps(protocol))
    return hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, command, out, extra=()):
        r = self.root
        return [command, '--protocol', str(r / 'protocol.json'), '--protocol-sha256', self.sha, '--config', str(r / 'config.json'),
                '--output', str(r / out), *extra]

    def test_select_then_measure_end_to_end(self):
        # Ego exactly 1 m apart: ring hits fall on a lattice and estimator A drops the far rings that have no road
        # reference, so several strips have > 0.5 m between their last road and first paint return. Under v3 this
        # fixture resolved (+0.44 m for a 0.40 m edge); under v4 it must be resolved near 0.40 m or unresolved.
        self.sha = synthetic_log(self.root, edge=0.40)
        tool.main(self.args('select', 'selection.json'))
        sel = json.loads((self.root / 'selection.json').read_text())
        self.assertTrue(sel['ok']); self.assertEqual(len(sel['chosen']), 15); self.assertFalse(sel['readLidarContent'])
        self.assertEqual(sel['nearEdge'], 'edge1')
        ahead = [c['edgeAheadM'] for c in sel['chosen']]
        self.assertTrue(all(5.0 <= d <= 30.0 for d in ahead)); self.assertEqual(len({c['timestampNs'] for c in sel['chosen']}), 15)
        tool.main(self.args('measure', 'result.json', ('--sweeps', str(self.root / 'receipts.json'))))
        out = json.loads((self.root / 'result.json').read_text())
        self.assertEqual(out['frame']['corridorL'], [-1.75, 4.75])
        self.assertEqual(out['frame']['corridorCenterL'], 1.5)
        run = out['frame']['evalRule']['coveredRunL']
        self.assertAlmostEqual(run[0], -1.75); self.assertLessEqual(run[1], 3.0 + 1e-9)
        self.assertGreater(out['estimatorA']['stripStatus'].get('edgeNotLocalized', 0), 0)
        self.assertGreater(out['estimatorA']['stripStatus'].get('noEdgeReturns', 0), 0)
        loo = out['nearEdge']['leaveOneSweepOut']
        self.assertEqual(loo['runs'], 15); self.assertEqual(len(loo['rows']), 15)
        self.assertEqual(set(out['diagnostics']['halves']), {'evenSweeps', 'oddSweeps', 'farHalf', 'nearHalf'})
        if out['nearEdge']['status'] == 'resolved':
            self.assertAlmostEqual(out['offsetToMappedEdgeM'], 0.40, delta=0.10)
            self.assertIn(out['label'], ('resolved', 'marginally resolved'))
        else:
            self.assertIsNone(out['offsetToMappedEdgeM']); self.assertIsNone(out['label'])
        self.assertGreater(out['coverage']['shadowedStrips'], 0)
        self.assertTrue(all(3.0 <= lo and hi <= 4.75 for lo, hi in out['coverage']['shadowedL']))
        self.assertGreater(out['returns']['searchWindow'], 1000)
        with self.assertRaises(ValueError):  # outputs are never overwritten
            tool.main(self.args('measure', 'result.json', ('--sweeps', str(self.root / 'receipts.json'))))

    def test_wrong_protocol_hash_or_tampered_sweep_is_refused(self):
        self.sha = synthetic_log(self.root)
        good = self.sha; self.sha = '0' * 64
        with self.assertRaises(ValueError):
            tool.main(self.args('select', 'selection.json'))
        self.sha = good
        tool.main(self.args('select', 'selection.json'))
        chosen = json.loads((self.root / 'selection.json').read_text())['chosen'][0]['timestampNs']
        victim = self.root / 'sweeps' / f'{chosen}.feather'
        victim.write_bytes(victim.read_bytes() + b'x')
        with self.assertRaises(ValueError):
            tool.main(self.args('measure', 'result.json', ('--sweeps', str(self.root / 'receipts.json'))))

    def test_too_few_candidate_sweeps_fails_selection(self):
        self.sha = synthetic_log(self.root, xs=np.arange(-12.0, -4.0, 1.0))
        tool.main(self.args('select', 'selection.json'))
        sel = json.loads((self.root / 'selection.json').read_text())
        self.assertFalse(sel['ok']); self.assertIn('candidate', sel['reason'])


if __name__ == '__main__':
    unittest.main()

"""Metric F car-top scoring on synthetic data only (no LiDAR, no scene files): a box-shaped car with a low hood."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/evaluation'))
import metric_f_car_top as mf

# The protocol thresholds (metric-f-v4.1), repeated here so the test needs no local files.
TH = {'binLengthNominalM': .25, 'lateralBinNominalM': .25, 'minHeightAboveBoxBottomM': .25, 'maxHeightAboveBoxTopM': .30,
      'minReturnsMeasured': 5, 'marginM': .05, 'crossedBinMinLengthM': 1e-6, 'overhangReachM': 1.0, 'overhangBinM': .25,
      'overhangSummaryReachM': .5, 'otherObjectBufferBeyondEndsM': .30}
L, W, H = 4.0, 1.8, 1.5
HOOD, ROOF, TRUNK = .9, 1.4, 1.0


def car_points(per_bin=10, sparse_bin=None, sparse_count=3):
    """Returns over the footprint: 16 bins of 0.25 m from the front; bins 0-3 hood (top 0.9 m), 4-11 roof (1.4 m),
    12-15 trunk (1.0 m). Each bin's highest return is exactly its top."""
    pts = []
    for b in range(16):
        top = HOOD if b < 4 else ROOF if b < 12 else TRUNK
        count = sparse_count if b == sparse_bin else per_bin
        xc = L / 2 - (b + .5) * .25
        for i in range(count):
            z = top if i == count - 1 else .3 + (top - .3) * i / max(count - 1, 1)
            pts.append([xc + (i % 3 - 1) * .05, -.8 + 1.6 * i / max(count - 1, 1), z])
    return np.array(pts)


def profile(points, **kw):
    return mf.profile_from_points(points, np.arange(len(points)) % 4, (L, W, H), TH, **kw)


def across_hood(z, x=1.6):
    """A level line across the car at x (inside hood bin 1: 1.50-1.75 m), from the right side to the left side."""
    return {'setbackM': 1.0, 'state': 'blocked', 'gapM': -.2, 'eyeCar': [x, -5., z], 'targetCar': [x, 5., z]}


class ProfileTests(unittest.TestCase):
    def test_bins_tops_and_measured(self):
        p = profile(car_points())
        bins = p['alongBins']['bins']
        self.assertEqual(p['alongBins']['n'], 16)
        self.assertAlmostEqual(p['alongBins']['widthM'], .25)
        self.assertEqual([b['topZ'] for b in bins[:4]], [HOOD] * 4)
        self.assertEqual(bins[5]['topZ'], ROOF)
        self.assertEqual(bins[15]['topZ'], TRUNK)
        self.assertTrue(all(b['measured'] and b['count'] == 10 for b in bins))
        self.assertAlmostEqual(bins[0]['xFront'], 2.0); self.assertAlmostEqual(bins[15]['xRear'], -2.0)

    def test_low_returns_high_returns_and_other_objects_are_not_used(self):
        pts = np.vstack([car_points(), [[1.6, 0., .2], [1.6, 0., H + .4], [1.6, .1, 1.2]]])
        other = np.zeros(len(pts), bool); other[-1] = True
        p = profile(pts, other_inside=other)
        b = p['alongBins']['bins'][1]
        self.assertEqual(b['topZ'], HOOD)
        self.assertEqual((b['count'], b['aboveTopLimit'], b['removedOtherObject']), (10, 1, 1))
        # Sensitivity profile: nothing removed, no upper limit (the low return still never counts).
        s = profile(pts, other_inside=other, limit_top=False, remove_other=False)['alongBins']['bins'][1]
        self.assertEqual((s['count'], s['topZ']), (12, H + .4))

    def test_overhang_beyond_the_ends(self):
        pts = np.vstack([car_points(), [[2.1, 0., .6]] * 6, [[2.6, 0., .6]], [[-2.2, 0., .7]] * 2, [[2.1, 1.2, .6]]])
        p = profile(pts)
        front, rear = p['overhang']['front'], p['overhang']['rear']
        self.assertEqual(front['returnsWithinSummaryReach'], 6)
        self.assertEqual([b['count'] for b in front['bins']], [6, 0, 1, 0])
        self.assertEqual(rear['returnsWithinSummaryReach'], 2)
        self.assertEqual(p['alongBins']['bins'][0]['count'], 10, 'overhang returns never enter the footprint bins')


class ClassTests(unittest.TestCase):
    def setUp(self):
        self.p = profile(car_points())

    def test_line_clearing_the_low_hood_by_006_is_likely_false(self):
        r = mf.score_row(across_hood(HOOD + .06), self.p, (L, W, H), TH)
        self.assertEqual(r['class'], 'likelyFalse')
        self.assertEqual(r['crossedBins'], 1)
        self.assertAlmostEqual(r['minClearanceM'], .06, places=9)
        self.assertFalse(r['measuredContact'])

    def test_line_clearing_the_low_hood_by_004_is_ambiguous(self):
        r = mf.score_row(across_hood(HOOD + .04), self.p, (L, W, H), TH)
        self.assertEqual(r['class'], 'ambiguous')
        r = mf.score_row(across_hood(HOOD - .04), self.p, (L, W, H), TH)
        self.assertEqual(r['class'], 'ambiguous')
        self.assertTrue(r['measuredContact'])

    def test_line_below_the_hood_is_supported(self):
        self.assertEqual(mf.score_row(across_hood(HOOD - .06), self.p, (L, W, H), TH)['class'], 'supported')

    def test_unmeasured_bin_is_unresolved_never_likely_false(self):
        sparse = profile(car_points(sparse_bin=1))
        self.assertFalse(sparse['alongBins']['bins'][1]['measured'])
        r = mf.score_row(across_hood(HOOD + .3), sparse, (L, W, H), TH)
        self.assertEqual(r['class'], 'unresolved')
        self.assertEqual(r['unmeasuredCrossedBins'], 1)
        self.assertIsNone(r['minClearanceM'])
        # A measured bin the line clearly passes under still makes the row supported.
        long_line = {'setbackM': 1., 'state': 'blocked', 'gapM': -.3, 'eyeCar': [-10., 0., 1.2], 'targetCar': [5., 0., 1.2]}
        self.assertEqual(mf.score_row(long_line, sparse, (L, W, H), TH)['class'], 'supported')

    def test_line_along_the_car_over_hood_roof_and_trunk(self):
        # Descending from behind (z 1.6 at the rear face) to the front (z 0.97 at the front face): it passes under the
        # roof in the middle, so the measured car top reaches above it.
        e, t = [-2., 0., 1.6], [2., 0., .97]
        row = {'setbackM': 1., 'state': 'blocked', 'gapM': -.3, 'eyeCar': e, 'targetCar': t}
        r = mf.score_row(row, self.p, (L, W, H), TH)
        self.assertEqual(r['crossedBins'], 16)
        self.assertEqual(r['class'], 'supported')
        # In 2D cells (7 lateral cells of 0.257 m) the line crosses only the centre column, which holds 2 returns per
        # bin here: unmeasured, so the secondary class is unresolved.
        self.assertEqual(r['class2D'], 'unresolved')

    def test_line_missing_the_footprint(self):
        row = {'setbackM': 1., 'state': 'unknown', 'gapM': .5, 'eyeCar': [-10., 3., 1.2], 'targetCar': [10., 3., 1.2]}
        self.assertEqual(mf.score_row(row, self.p, (L, W, H), TH)['class'], 'noFootprintCrossing')

    def test_classify_precedence(self):
        m = lambda c: {'measured': True, 'clearance': c}
        u = {'measured': False, 'clearance': None}
        self.assertEqual(mf.classify([m(-.06), u], .05), 'supported')
        self.assertEqual(mf.classify([m(.2), u], .05), 'unresolved')
        self.assertEqual(mf.classify([m(.05), m(.3)], .05), 'likelyFalse')
        self.assertEqual(mf.classify([m(.049), m(.3)], .05), 'ambiguous')
        self.assertEqual(mf.classify([m(-.05)], .05), 'ambiguous')
        self.assertEqual(mf.classify([], .05), 'noFootprintCrossing')


class StopAndFrameTests(unittest.TestCase):
    def test_measured_stop_blocking(self):
        rows = [{'setbackM': s, 'state': st, 'gapM': g, 'measuredContact': c, 'unmeasuredCrossedBins': u}
                for s, st, g, c, u in [(1, 'blocked', -.3, True, 0), (2, 'blocked', -.3, True, 0), (3, 'overlap', -.3, False, 0),
                                       (4, 'blocked', -.2, False, 0), (5, 'blocked', -.15, False, 1), (6, 'unknown', .2, False, 0)]]
        s = mf.stop_blocking(rows, {'stopsBlockingAt': 6, 'notBlockedFrom': 6})
        self.assertEqual((s['measuredFirstContactM'], s['measuredLastContactM'], s['measuredStopBlockingM']), (1, 2, 4))
        self.assertEqual((s['appFirstPlainBlockedM'], s['appLastPlainBlockedM'], s['measuredMinusAppStopM']), (1, 5, -2))
        self.assertEqual(s['rowsAfterMeasuredStopTouchingBoxWithUnmeasuredBin'], 1)
        # Protocol v2: the last plain Blocked row and stopsBlockingAt, each minus the measured stop (positive = app later).
        self.assertEqual((s['appLastPlainBlockedMinusMeasuredStopM'], s['appStopsBlockingAtMinusMeasuredStopM']), (1, 2))

    def test_upright_frame_matches_the_app_box(self):
        a = math.radians(30); c, s = math.cos(a), math.sin(a)
        # worldFromBox columns: heading, left, up (world Y up), with a small pitch the app drops.
        p = math.radians(.5)
        m = [[c * math.cos(p), s, 0.], [math.sin(p), 0., 1.], [s * math.cos(p), -c, 0.]]
        f = mf.upright_frame({'worldFromBox': m, 'center': [1., 2., 3.], 'length': 4., 'width': 2., 'height': 1.5})
        np.testing.assert_allclose(f['heading'], [c, 0, s], atol=1e-12)
        np.testing.assert_allclose(f['left'], [s, 0, -c], atol=1e-12)
        point = np.array([1., 2., 3.]) + 1. * f['heading'] + .5 * f['left'] + np.array([0., .2, 0.])
        np.testing.assert_allclose(mf.car_local([point], f)[0], [1., .5, .2 + .75], atol=1e-12)
        # The app's box-local frame (assessSightline): axis toward the camera = -heading, side = [axis.z, 0, -axis.x].
        axis = -f['heading']; side = np.array([axis[2], 0., -axis[0]])
        d = point - f['center']
        self.assertAlmostEqual(float(d @ side), -.5, places=12)
        self.assertAlmostEqual(float(d @ axis), -1., places=12)

    def test_outputs_refuse_the_word(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                mf.write_new(Path(d)/'a.json', {'note': 'a ' + 'pa' + 'rked car'})
            mf.write_new(Path(d)/'b.json', {'note': 'a stationary car'})
            with self.assertRaises(FileExistsError):
                mf.write_new(Path(d)/'b.json', {'note': 'again'})
            self.assertEqual(json.loads((Path(d)/'b.json').read_text())['note'], 'a stationary car')


TH2 = {**TH, 'freeRayBandM': .05, 'nearGroundFloorM': -.5}   # protocol v2 additions (report only)


def narrow_car(half_width=.45, per_bin=10):
    """The car_points shape, but the real car is only 0.9 m wide inside the 1.8 m box: the box's side strips are empty."""
    pts = []
    for b in range(16):
        top = HOOD if b < 4 else ROOF if b < 12 else TRUNK; xc = L / 2 - (b + .5) * .25
        for i in range(per_bin):
            z = top if i == per_bin - 1 else .3 + (top - .3) * i / (per_bin - 1)
            pts.append([xc, -half_width + 2 * half_width * i / (per_bin - 1), z])
    return np.array(pts)


class FreeSpaceTests(unittest.TestCase):
    """Protocol v2 evidence: rays that pass through a cell and hit something beyond it, and near-ground returns."""

    def test_rays_through_an_empty_side_strip(self):
        # Lateral cells of 1.8 / 7 = 0.257 m: cell 0 is y 0.643-0.9 (left edge strip), cell 1 is y 0.386-0.643.
        S = np.array([[1.6, 8., 1.9]] * 8)
        P = np.array([[1.6, .5, 1.]] * 5       # hit the real side in cell 1: pass through cell 0 at about 1.02-1.05 m
                     + [[1.6, .8, 1.]]         # ends inside cell 0: not a free ray there
                     + [[1.6, 0., .1]]         # dips below 0.25 m inside cell 0: not counted
                     + [[1.6, -8., 1.95]])     # passes cell 0 at about 1.92 m, above H + 0.30 = 1.8 m: not counted
        ev = mf.cell_evidence(np.zeros((0, 3)), None, (S, P), (L, W, H), TH2)
        strip, inner = ev['cells'][1 * 7 + 0], ev['cells'][1 * 7 + 1]
        self.assertEqual(strip['freeRays'], 5)
        self.assertEqual(strip['freeRayTopHistogram'][15], 5, 'highest in-cell height 1.048 m is in the 1.00-1.05 band')
        self.assertEqual(inner['freeRays'], 0, 'the rays end in cell 1')
        self.assertEqual(mf.free_rays_below(strip, ev, 1.2, TH2['marginM']), 5)
        self.assertEqual(mf.free_rays_below(strip, ev, 1.08, TH2['marginM']), 0, 'a band reaching above line - margin never counts')

    def test_near_ground_returns(self):
        pts = np.array([[1.6, .8, .05]] * 3 + [[1.6, .8, -.6], [1.6, .8, .1], [1.6, .8, .3]])
        other = np.zeros(len(pts), bool); other[4] = True
        ev = mf.cell_evidence(pts, other, (np.zeros((0, 3)), np.zeros((0, 3))), (L, W, H), TH2)
        self.assertEqual(ev['cells'][1 * 7 + 0]['belowMinHeight'], 3)

    def test_clip_many_matches_clip_segment(self):
        rng = np.random.default_rng(3)
        S, P = rng.uniform(-4, 4, (200, 3)), rng.uniform(-4, 4, (200, 3))
        a0, a1 = mf.clip_many(S, P, (-2., -.9), (2., .9))
        for k in range(200):
            r = mf.clip_segment(S[k], P[k], (-2., -.9), (2., .9))
            if r is None:
                self.assertFalse(a0[k] <= a1[k] and a1[k] - a0[k] > 1e-12)
            else:
                self.assertAlmostEqual(a0[k], r[0], places=12); self.assertAlmostEqual(a1[k], r[1], places=12)

    def test_line_entering_through_the_side_face_over_an_empty_strip(self):
        # The c780 case: a full-width slice is measured (its top is low), but the line crosses only the box's front-left
        # strip, where the real car is not. Primary class: likely false; 2D class: unresolved (the crossed cells hold no
        # returns); the free rays show the strip was seen through below the line.
        pts = narrow_car(); dims = (L, W, H)
        p = mf.profile_from_points(pts, np.arange(len(pts)) % 4, dims, TH)
        S = np.array([[1.9, 8., 1.9]] * 6 + [[1.62, 8., 1.9]] * 6)
        P = np.array([[1.9, .4, 1.]] * 6 + [[1.62, .4, 1.]] * 6)
        p['cellEvidence'] = mf.cell_evidence(pts, None, (S, P), dims, TH2)
        # y = 5 - 6s, x = -8 + 14s: enters the left face (y 0.9) at x 1.567, leaves the front face (x 2) at y 0.714.
        row = {'setbackM': 1., 'state': 'blocked', 'gapM': -.2, 'eyeCar': [-8., 5., 1.2], 'targetCar': [6., -1., 1.2]}
        r = mf.score_row(row, p, dims, TH2)
        self.assertEqual([b['bin'] for b in r['bins']], [0, 1])
        self.assertEqual(r['class'], 'likelyFalse')
        self.assertAlmostEqual(r['minClearanceM'], 1.2 - HOOD, places=9)
        self.assertEqual(r['class2D'], 'unresolved')
        self.assertEqual([(c['bin'], c['lateral'], c['count']) for c in r['cells2D']], [(0, 0, 0), (1, 0, 0)])
        self.assertEqual([c['freeRaysBelowLine'] for c in r['cells2D']], [6, 6])
        self.assertEqual((r['footprint']['entry']['face'], r['footprint']['exit']['face']), ('left', 'front'))
        self.assertAlmostEqual(r['footprint']['entry']['x'], 1.5667, places=4)
        self.assertAlmostEqual(r['footprint']['exit']['y'], .7143, places=4)

    def test_same_classes_check(self):
        row = {'state': 'blocked', 'class': 'likelyFalse', 'class2D': 'unresolved', 'classSensitivity': 'likelyFalse',
               'setbackM': 1., 'minClearanceM': .1}
        heights = {'child': {'rows': [row]}}
        self.assertTrue(mf.check_same_classes(heights, {'heights': {'child': {'rows': [dict(row)]}}})['same'])
        with self.assertRaises(AssertionError):
            mf.check_same_classes(heights, {'heights': {'child': {'rows': [{**row, 'class': 'ambiguous'}]}}})
        with self.assertRaises(AssertionError):
            mf.check_same_classes(heights, {'heights': {'child': {'rows': [{**row, 'minClearanceM': .11}]}}})


# ------------------------------------------------------------------ metric F shape-v1 (protocol v3)
def shape_row(setback, state, reason, box_state, box_reason=None, gap=-.3, hit=None, z=.95):
    """A synthetic shape-v1 row across hood bin 1 (top 0.9 m): line height z, the app's shape result (state, reason) and
    the box result."""
    row = across_hood(z); row.update({'setbackM': setback, 'state': state, 'gapM': gap,
                                      'box': {'state': box_state, 'reason': box_reason, 'gapM': gap},
                                      'shape': None if hit is None else {'status': 'measured', 'solidCutM': .2, 'hitXM': hit}})
    if reason:
        row['reason'] = reason
    return row


class ShapeV1Tests(unittest.TestCase):
    def test_allowed_change_table(self):
        self.assertIsNone(mf.allowed_change(shape_row(1, 'blocked', None, 'blocked')))
        self.assertIsNone(mf.allowed_change(shape_row(1, 'unknown', 'height-not-proven', 'blocked')))
        self.assertIsNone(mf.allowed_change(shape_row(1, 'unknown', 'too-close-to-call', 'unknown', 'too-close-to-call')))
        self.assertRegex(mf.allowed_change(shape_row(1, 'blocked', None, 'unknown', 'too-close-to-call')), 'became blocked')
        self.assertRegex(mf.allowed_change(shape_row(1, 'unknown', 'too-close-to-call', 'blocked')), 'box Blocked became unknown too-close-to-call')
        row = shape_row(1, 'blocked', None, 'blocked'); row['box']['gapM'] = -.31
        self.assertEqual(mf.allowed_change(row), 'gapM changed')

    def test_report_counts_new_blocked_lost_front_and_unresolved(self):
        p = profile(car_points(sparse_bin=1))
        rows = [shape_row(1.0, 'blocked', None, 'blocked', hit=.4, z=.95),                    # still Blocked, front half
                shape_row(1.25, 'unknown', 'height-not-proven', 'blocked', hit=None, z=.8),     # lost, F-supported (below the hood)
                shape_row(1.5, 'blocked', None, 'unknown', None, gap=.2, hit=-.3, z=.8),         # new Blocked (not allowed)
                shape_row(1.75, 'unknown', None, 'unknown', None, gap=.5)]
        rows[3]['eyeCar'] = [1.6, -5., 3.]; rows[3]['targetCar'] = [1.6, 5., 3.]
        # Row 0 crosses the sparse (unmeasured) hood bin 1: unresolved. Use the measured bin 2 for the other rows.
        for r in rows[1:]:
            r['eyeCar'][0] = r['targetCar'][0] = 1.4
        scored = [mf.score_row(r, p, (L, W, H), TH) for r in rows]
        self.assertEqual([r['class'] for r in scored[:3]], ['unresolved', 'supported', 'supported'])
        rep = mf.shape_v1_report(scored)
        self.assertEqual(rep['counts']['boxBlocked'], 2); self.assertEqual(rep['counts']['shapeBlocked'], 2)
        self.assertEqual(rep['counts']['heightNotProven'], 1); self.assertEqual(rep['counts']['boxBlockedToHeightNotProven'], 1)
        self.assertEqual(rep['noNewBlocked'], {'count': 1, 'setbacksM': [1.5], 'pass': False})
        self.assertFalse(rep['allowedChanges']['pass']); self.assertEqual(rep['allowedChanges']['violations'][0]['setbackM'], 1.5)
        self.assertEqual(rep['lostSupported']['count'], 1); self.assertEqual(rep['lostSupported']['setbacksM'], [1.25])
        self.assertEqual(rep['lostSupported']['boxBlockedSupported'], 1)
        self.assertEqual(rep['frontHalfFlag']['count'], 1); self.assertEqual(rep['frontHalfFlag']['rows'][0]['hitXM'], .4)
        self.assertEqual(rep['unresolvedBlocked'], {'count': 1, 'setbacksM': [1.0], 'note': 'never counted as passes'})
        self.assertEqual([f['box'] for f in rep['flipTable']], ['blocked', 'blocked', 'unknown', 'unknown'])
        self.assertEqual(rep['flipTable'][1]['shape'], 'unknown height-not-proven')
        with self.assertRaises(AssertionError):
            mf.shape_v1_report([mf.score_row(across_hood(.95), p, (L, W, H), TH)])

    def test_forced_unresolved_and_too_few_sweeps(self):
        p = profile(car_points())
        r = mf.score_row(shape_row(1, 'blocked', None, 'blocked', z=.95), p, (L, W, H), TH, forced='fewer than 2')
        self.assertEqual((r['class'], r['class2D'], r['forcedUnresolved']), ('unresolved', 'unresolved', 'fewer than 2'))
        self.assertTrue(mf.too_few_scoring_sweeps(1)); self.assertTrue(mf.too_few_scoring_sweeps(0)); self.assertFalse(mf.too_few_scoring_sweeps(2))

    def test_laser_check_drops_a_scoring_sweep_and_stops_v2(self):
        self.assertTrue(mf.laser_check(1, .1, .15, scoring=True))
        self.assertFalse(mf.laser_check(1, .2, .15, scoring=True))
        with self.assertRaises(AssertionError):
            mf.laser_check(1, .2, .15, scoring=False)
        with self.assertRaises(AssertionError):
            mf.laser_check(1, .15, .15, scoring=False)

    def test_listed_sweeps_must_be_present_with_their_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'sweeps'; d.mkdir()
            (d / '100.feather').write_bytes(b'one'); (d / '200.feather').write_bytes(b'two'); (d / '300.feather').write_bytes(b'not listed')
            one, two = mf.sha(d / '100.feather'), mf.sha(d / '200.feather')
            sweeps, skipped = mf.gather_listed_sweeps([d], [], [(200, two), (100, one)], {100, 200, 300}, 'log')
            self.assertEqual([s['timestampNs'] for s in sweeps], [100, 200]); self.assertEqual(skipped, [])
            with self.assertRaisesRegex(AssertionError, 'missing'):
                mf.gather_listed_sweeps([d], [], [(400, one)], {400}, 'log')
            with self.assertRaisesRegex(AssertionError, 'listed SHA-256'):
                mf.gather_listed_sweeps([d], [], [(100, two)], {100}, 'log')
            with self.assertRaisesRegex(AssertionError, 'annotation row'):
                mf.gather_listed_sweeps([d], [], [(100, one)], {200}, 'log')
            receipt = Path(tmp) / 'receipt.json'
            receipt.write_text(json.dumps({'sweeps': [{'key': 'datasets/av2/sensor/train/log/sensors/lidar/100.feather', 'sha256': two, 'log': 'log'}]}))
            with self.assertRaisesRegex(AssertionError, 'receipt'):
                mf.gather_listed_sweeps([d], [receipt], [(100, one)], {100}, 'log')
            ground = Path(tmp) / 'ground.json'
            ground.write_text(json.dumps({'carShape': {'track': 't', 'sweeps': {'scoring': [{'timestampNs': '100', 'sha256': one}]}}}))
            listed, shape, _ = mf.scoring_list(ground)
            self.assertEqual((listed, shape['track']), ([(100, one)], 't'))
            ground.write_text(json.dumps({'carShape': None}))
            with self.assertRaisesRegex(AssertionError, 'no carShape'):
                mf.scoring_list(ground)

    def test_box_reproduction_and_app_join(self):
        rows = [{'setbackM': 1.0, 'box': {'state': 'blocked', 'reason': None, 'gapM': -.3}}, {'setbackM': 1.25, 'box': {'state': 'unknown', 'reason': 'too-close-to-call', 'gapM': -.05}}]
        v2 = {'kind': 'metric-f app rows', 'heights': {'child': {'rows': [{'setbackM': 1.0, 'state': 'blocked', 'gapM': -.3},
                                                                          {'setbackM': 1.25, 'state': 'unknown', 'reason': 'too-close-to-call', 'gapM': -.05}]}}}
        rep = mf.box_reproduction({'child': {'rows': rows}}, v2)
        self.assertTrue(rep['pass']); self.assertEqual(rep['heights']['child']['rowsWithAnotherBoxStateOrReason'], [])
        v2['heights']['child']['rows'][1]['gapM'] = -.05 + 1e-8
        self.assertFalse(mf.box_reproduction({'child': {'rows': rows}}, v2)['pass'])
        heights = {'child': {'rows': [dict(r, state='unknown', reason='height-not-proven', **{'class': 'supported'}) for r in rows]}}
        app = {'heights': {'child': {'rows': [dict(r, state='blocked', **{'class': 'supported'}) for r in rows]}}}
        mf.join_app_results(heights, app)
        self.assertEqual(heights['child']['rows'][0]['appShape'], {'state': 'blocked', 'reason': None, 'class': 'supported'})
        app['heights']['child']['rows'][0]['box'] = {'state': 'unknown', 'reason': None, 'gapM': -.3}
        with self.assertRaises(AssertionError):
            mf.join_app_results(heights, app)


if __name__ == '__main__':
    unittest.main()

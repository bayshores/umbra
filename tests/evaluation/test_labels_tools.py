"""Unit tests for the evaluation-only labeling kit (scripts/evaluation/labels). Synthetic fixtures only.

Run: .local/venv-labels/bin/python -I -m unittest discover -s tests/evaluation -p 'test_labels*.py'
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = ROOT / 'scripts/evaluation/labels'
KIT_MODULES = ('common', 'av2io', 'imaging', 'snap', 'isolation', 'session', 'labelkit', 'synth', 'lidar_ref', 'truth', 'oracle', 'aggregate',
               'score', 'select_cases', 'build_validation', 'fetch_dev', 'selftest', 'lock_protocol')


def load_kit():
    """Import the kit without leaking its generic module names (common, session, ...) into other tests."""
    saved = {name: sys.modules.pop(name) for name in KIT_MODULES if name in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(KIT_DIR))
    try:
        import importlib
        modules = {name: importlib.import_module(name) for name in
                   ('common', 'imaging', 'snap', 'isolation', 'session', 'labelkit', 'synth', 'lidar_ref', 'aggregate', 'score',
                    'build_validation', 'selftest', 'lock_protocol', 'oracle')}
    finally:
        for name in KIT_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
    return types.SimpleNamespace(**modules)


KIT = load_kit()
aggregate, common, imaging, lidar_ref, score, session, snap, synth = (KIT.aggregate, KIT.common, KIT.imaging, KIT.lidar_ref, KIT.score,
                                                                      KIT.session, KIT.snap, KIT.synth)

PROTOCOL = {
    'version': 'test-v1',
    'render': {'contextMargin': 0.25, 'lineZoom': 7, 'lineWindowPerp': 33, 'lineWindowAlong': 81, 'pixelZoom': 8, 'pixelWindow': 49,
               'pickMarginPx': 10, 'pickMinZoom': 3, 'pickMaxPanelPx': 560},
    'search': {'bracketMargin': 2, 'maxExpansions': 2, 'candidateMinSpan': 6, 'candidateOutsidePixels': 2,
               'candidateInsidePixels': {'top': 2, 'contact': 2, 'paint': 1}, 'repeatJitter': 6, 'maxRestarts': 1, 'pixelExpandStep': 8},
    'snap': {'sigma': 0.7, 'window': 1.0, 'minStrength': 2.0, 'maxCandidates': 4},
    'stations': {'insideFraction': 0.45},
    'wheels': {'maxCount': 3},
    'aggregation': {'labelers': 3, 'spreadMaxPx': {'line': 3.0, 'pixel': 2.0}},
}


def blurred_rect(width=200, height=160, box=(60.3, 50.6, 140.7, 120.2), inside=30.0, outside=180.0, s=8):
    """Grey image with a dark rectangle whose edges sit at sub-pixel positions (pixel-centre convention)."""
    xs = (np.arange(width * s) + 0.5) / s - 0.5
    ys = (np.arange(height * s) + 0.5) / s - 0.5
    gx, gy = np.meshgrid(xs, ys)
    inside_mask = (gx >= box[0]) & (gx <= box[2]) & (gy >= box[1]) & (gy <= box[3])
    coverage = inside_mask.reshape(height, s, width, s).mean((1, 3))
    lum = outside + (inside - outside) * coverage
    rgb = np.repeat(np.clip(lum, 0, 255).astype(np.uint8)[..., None], 3, -1)
    return rgb, coverage


def car_task(box_cells=None, stations=(80, 100, 120)):
    return {'taskId': 'T01', 'kind': 'car', 'target': {'noun': 'vehicle', 'hint': ''},
            'overview': {'region': [0, 0, 200, 160], 'cell': 16, 'columns': 13, 'rows': 10, 'columnLabels': ['A', 'M'], 'rowLabels': [1, 10]},
            'stationColumns': list(stations)}


class RectOracle:
    """Answers from a coverage map with the 50 % rule (pixel judged 'on target' when >= half covered)."""

    def __init__(self, coverage, task, wheels=0):
        self.cov, self.task, self.wheels = coverage, task, wheels
        ys, xs = np.nonzero(coverage >= 0.5)
        self.pts = np.column_stack([xs, ys])

    def cell(self, i, j):
        cell = self.task['overview']['cell']
        return f'{imaging.column_label(int(i // cell))}{int(j // cell) + 1}'

    def answer(self, q):
        p = self.pts
        if q['kind'] == 'box':
            return f'{self.cell(p[:, 0].min(), p[:, 1].min())}:{self.cell(p[:, 0].max(), p[:, 1].max())}'
        if q['kind'] == 'cell':
            k = {'L': np.argmin(p[:, 0]), 'R': np.argmax(p[:, 0]), 'T': np.argmin(p[:, 1]), 'B': np.argmax(p[:, 1])}[q['feature']]
            return self.cell(*p[k])
        if q['kind'] == 'count':
            return str(self.wheels)
        if q['kind'] == 'pick':
            return '0'
        render = q['render']
        if render['marker']['type'] == 'pixel':
            i, j = render['marker']['at']
            return 'yes' if self.cov[j, i] >= 0.5 else 'no'
        pos, f = render['position'], q['feature']
        hit = {'L': (p[:, 0] <= pos - 1).any(), 'R': (p[:, 0] >= pos).any(), 'T': (p[:, 1] <= pos - 1).any(), 'B': (p[:, 1] >= pos).any()}[f]
        return 'yes' if hit else 'no'


def drive(task, protocol, rgb, answer_fn, labeler='X'):
    ctx = session.Context(task, protocol, labeler, rgb)
    gen = session.GENERATORS[task['kind']](ctx)
    answers = []
    try:
        q = next(gen)
        while True:
            a = session.normalize_answer(q, answer_fn(q))
            answers.append(a)
            q = gen.send(a)
    except StopIteration as stop:
        return stop.value, answers


class CommonTests(unittest.TestCase):
    def test_held_out_and_reserve_logs_are_refused(self):
        for prefix in common.FORBIDDEN_PREFIXES:
            with self.assertRaises(common.ProtocolError):
                common.assert_development(f'datasets/av2/sensor/train/{prefix}-xxxx/sensors/lidar/1.feather')
            with self.assertRaises(common.ProtocolError):
                common.full_log_id(prefix)
        self.assertTrue(common.full_log_id('8c52d911').startswith('8c52d911-'))

    def test_protocol_lock_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol-v9.json'
            path.write_bytes(common.pretty_json({'a': 1}))
            with self.assertRaises(common.ProtocolError):
                common.load_protocol(path)
            common.write_json(Path(tmp) / 'protocol-v9.lock.json', {'sha256': common.sha256_file(path)})
            self.assertEqual(common.load_protocol(path)[0], {'a': 1})
            path.write_bytes(common.pretty_json({'a': 2}))
            with self.assertRaises(common.ProtocolError):
                common.load_protocol(path)


class ImagingTests(unittest.TestCase):
    def test_column_labels_round_trip(self):
        for i in (0, 1, 25, 26, 27, 51, 700):
            self.assertEqual(imaging.parse_column_label(imaging.column_label(i)), i)
        self.assertEqual(imaging.column_label(26), 'AA')

    def test_cells_and_boxes(self):
        meta = {'region': [100, 50, 64, 48], 'cell': 16, 'columns': 4, 'rows': 3, 'columnLabels': ['A', 'D'], 'rowLabels': [1, 3]}
        self.assertEqual(imaging.cell_bounds(meta, 'b2'), (116, 131, 66, 81))
        self.assertEqual(imaging.parse_box(meta, 'A1:D3'), (100, 163, 50, 97))
        for bad in ('E1', 'A4', 'A0', '1A', 'B2:A1', 'A1-B2'):
            with self.assertRaises(ValueError):
                imaging.parse_box(meta, bad) if ':' in bad or '-' in bad else imaging.cell_bounds(meta, bad)

    def test_render_question_and_overview_are_png(self):
        rgb, _ = blurred_rect()
        png = imaging.render_question(rgb, [0, 0, 200, 160], [40, 40, 33, 81], 7, {'type': 'vline', 'at': 59.5})
        self.assertEqual(Image.open(io.BytesIO(png)).format, 'PNG')
        png = imaging.render_question(rgb, [0, 0, 200, 160], [60, 30, 49, 49], 8, {'type': 'pixel', 'at': (80, 50)},
                                      candidates=[(1, 'y', 50.6, 80)])
        self.assertEqual(Image.open(io.BytesIO(png)).format, 'PNG')
        png, meta = imaging.render_overview(rgb, [0, 0, 200, 160], 16, ring=(100, 85))
        self.assertEqual((meta['columns'], meta['rows']), (13, 10))
        self.assertEqual(Image.open(io.BytesIO(png)).format, 'PNG')

    def test_crop_padded_fills_outside(self):
        rgb = np.full((4, 4, 3), 9, np.uint8)
        crop = imaging.crop_padded(rgb, -2, -2, 4, 4)
        self.assertEqual(int(crop[0, 0, 0]), 0)
        self.assertEqual(int(crop[3, 3, 0]), 9)


class SnapTests(unittest.TestCase):
    def test_snap_finds_subpixel_edge(self):
        rgb, _ = blurred_rect(box=(60.3, 50.6, 140.7, 120.2))
        for axis, fixed, edge in (('x', 85, 60.3), ('x', 85, 140.7), ('y', 100, 50.6), ('y', 100, 120.2)):
            found = snap.snap(rgb, axis, fixed, round(edge), window=1.0)
            self.assertIsNotNone(found)
            self.assertAlmostEqual(found[0], edge, delta=0.12)

    def test_candidates_respect_sign(self):
        rgb, _ = blurred_rect(box=(60.3, 50.6, 140.7, 120.2), inside=30, outside=180)
        down = snap.candidates(rgb, 'y', 100, 40, 130, sign=-1)   # bright above, dark below: top edge of a dark box
        up = snap.candidates(rgb, 'y', 100, 40, 130, sign=1)
        self.assertEqual(len(down), 1)
        self.assertAlmostEqual(down[0][0], 50.6, delta=0.15)
        self.assertEqual(len(up), 1)
        self.assertAlmostEqual(up[0][0], 120.2, delta=0.15)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.box = (60.3, 50.6, 140.7, 120.2)
        self.rgb, self.cov = blurred_rect(box=self.box)
        self.task = car_task()

    def test_perfect_labeler_recovers_edges(self):
        result, answers = drive(self.task, PROTOCOL, self.rgb, RectOracle(self.cov, self.task).answer)
        f = result['features']
        L, T, R, B = self.box
        # 50 % rule: extremes within half a pixel
        self.assertLessEqual(abs(f['L']['value'] - L), 0.5)
        self.assertLessEqual(abs(f['R']['value'] - R), 0.5)
        self.assertLessEqual(abs(f['T']['value'] - T), 0.5)
        self.assertLessEqual(abs(f['B']['value'] - B), 0.5)
        for k in ('S1', 'S2', 'S3'):
            self.assertEqual(f[k]['status'], 'resolved')
            self.assertAlmostEqual(f[k]['value'], T, delta=0.15)  # snapped to the sub-pixel gradient peak
        self.assertTrue(all(isinstance(a, str) for a in answers))

    def test_replay_is_deterministic(self):
        result, answers = drive(self.task, PROTOCOL, self.rgb, RectOracle(self.cov, self.task).answer)
        pending, replay, asked = session.run(self.task, PROTOCOL, 'X', self.rgb, answers)
        self.assertIsNone(pending)
        self.assertEqual(json.dumps(replay, default=float), json.dumps(result, default=float))
        pending, _, asked = session.run(self.task, PROTOCOL, 'X', self.rgb, answers[:5])
        self.assertEqual(len(asked), 6)
        self.assertIsNotNone(pending)

    def test_cannot_tell_leaves_feature_unresolved(self):
        oracle = RectOracle(self.cov, self.task)

        def answer(q):
            if q['feature'] == 'R' and q['kind'] == 'yesno':
                return 'cannot_tell'
            return oracle.answer(q)
        result, _ = drive(self.task, PROTOCOL, self.rgb, answer)
        self.assertEqual(result['features']['R']['status'], 'unresolved')
        self.assertEqual(result['features']['L']['status'], 'resolved')

    def test_contradicting_repeat_is_caught(self):
        oracle = RectOracle(self.cov, self.task)
        state = {'flipped': 0}

        def answer(q):
            a = oracle.answer(q)
            if q['feature'] == 'L' and q['render'].get('purpose') == 'bisect' and state['flipped'] == 0:
                state['flipped'] = 1
                return 'no' if a == 'yes' else 'yes'
            return a
        result, _ = drive(self.task, PROTOCOL, self.rgb, answer)
        L = result['features']['L']
        self.assertGreaterEqual(L['restarts'], 1)
        if L['status'] == 'resolved':
            self.assertLessEqual(abs(L['value'] - self.box[0]), 0.5)

    def test_answers_are_validated(self):
        q = session.Question(kind='yesno', options=None, render={})
        self.assertEqual(session.normalize_answer(q, ' YES '), 'yes')
        self.assertEqual(session.normalize_answer(q, "can't tell"), 'cannot_tell')
        for bad in ('maybe', '123', '0.5', ''):
            with self.assertRaises(ValueError):
                session.normalize_answer(q, bad)
        pick = session.Question(kind='pick', options=3, render={})
        self.assertEqual(session.normalize_answer(pick, '2'), '2')
        for bad in ('4', '-1', '1.5', 'x'):
            with self.assertRaises(ValueError):
                session.normalize_answer(pick, bad)


class AggregationTests(unittest.TestCase):
    def test_median_agreement_and_disagreement(self):
        ok = aggregate.combine('S1', {'A': {'status': 'resolved', 'value': 10.0, 'fixed': 5}, 'B': {'status': 'resolved', 'value': 10.6, 'fixed': 5},
                                      'C': {'status': 'resolved', 'value': 11.5, 'fixed': 5}}, {'line': 3.0, 'pixel': 2.0})
        self.assertEqual(ok['status'], 'resolved')
        self.assertAlmostEqual(ok['value'], 10.6)
        wide = aggregate.combine('S1', {'A': {'status': 'resolved', 'value': 10.0}, 'B': {'status': 'resolved', 'value': 10.6},
                                        'C': {'status': 'resolved', 'value': 12.5}}, {'line': 3.0, 'pixel': 2.0})
        self.assertEqual(wide['status'], 'unresolved')
        line = aggregate.combine('L', {'A': {'status': 'resolved', 'value': 10.0}, 'B': {'status': 'resolved', 'value': 10.6},
                                       'C': {'status': 'resolved', 'value': 12.5}}, {'line': 3.0, 'pixel': 2.0})
        self.assertEqual(line['status'], 'resolved')
        missing = aggregate.combine('L', {'A': {'status': 'resolved', 'value': 10.0}, 'B': {'status': 'unresolved'}, 'C': None},
                                    {'line': 3.0, 'pixel': 2.0})
        self.assertEqual(missing['status'], 'unresolved')


class SynthAndLidarTests(unittest.TestCase):
    def test_silhouette_truth_matches_polygon_extremes(self):
        polys = [np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float), np.array([[3, -4], [7, -4], [8, 0], [2, 0]], float)]
        truth, profile = synth.silhouette_truth(polys, [5, 1])
        self.assertEqual((truth['L'], truth['R'], truth['T'], truth['B']), (0, 10, -4, 5))
        self.assertEqual(profile[5], (-4.0, 5.0))
        self.assertEqual(profile[1], (0.0, 5.0))

    def test_fit_plane_recovers_plane(self):
        rng = np.random.default_rng(0)
        xy = rng.uniform(-5, 5, (500, 2))
        z = 0.02 * xy[:, 0] - 0.01 * xy[:, 1] - 3.0 + rng.normal(0, 0.005, 500)
        coef, _ = synth.fit_plane(np.column_stack([xy, z]))
        np.testing.assert_allclose(coef, [0.02, -0.01, -3.0], atol=0.003)

    def test_step_edge_finds_paint_onset(self):
        rng = np.random.default_rng(1)
        s = np.sort(rng.uniform(-1.0, 2.0, 400))
        paint = s >= 0.25
        params = {'roadBeforeM': 0.6, 'minBarRunM': 1.0, 'minBefore': 3, 'minAfter': 5, 'roadMaxFraction': 0.2, 'runPaintShare': 0.5}
        edge, gap = lidar_ref.step_edge(s, paint, params)
        self.assertAlmostEqual(edge, 0.25, delta=gap)
        short = (s >= 0.25) & (s < 0.6)  # a 0.35 m triangle-like run is rejected
        self.assertIsNone(lidar_ref.step_edge(s, short, params))

    def test_persistence_drops_single_sweep_voxels(self):
        static = np.array([[0.05, 0.05, 0.05]] * 4)
        moving = np.array([[5.05, 0.05, 0.05]])
        xyz = np.vstack([static, moving])
        keep = lidar_ref.persistent(xyz, np.array([0, 1, 2, 3, 0]), 0.2, 2)
        self.assertEqual(keep.tolist(), [True, True, True, True, False])


class ScoreTests(unittest.TestCase):
    def test_lidar_inner_bound_consistency(self):
        coarse = {'registrationAllowancePx': 8.0}
        entry = {'spacingPx': {'L': 2.0}}
        inside = score.evaluate(105.0, 100.0, 'inner', 'L', entry, coarse)   # label 5 px inside the LiDAR bound
        self.assertTrue(inside['consistent'])
        far_in = score.evaluate(110.0, 100.0, 'inner', 'L', entry, coarse)
        self.assertFalse(far_in['consistent'])
        outside = score.evaluate(91.0, 100.0, 'inner', 'L', entry, coarse)  # 9 px outward <= spacing 2 + allowance 8
        self.assertTrue(outside['consistent'])
        one_sided = score.evaluate(150.0, 100.0, 'onesided', 'B', entry, coarse)
        self.assertTrue(one_sided['consistent'])
        exact = score.evaluate(10.4, 10.0, 'exact', 'S1', entry, coarse)
        self.assertAlmostEqual(exact['error'], 0.4)
        self.assertNotIn('consistent', exact)


if __name__ == '__main__':
    unittest.main()

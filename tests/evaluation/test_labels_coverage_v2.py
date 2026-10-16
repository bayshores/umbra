"""Tests for the protocol v2 coverage design (.local/evaluation/pao-labels/v2/coverage-design.md). Synthetic fixtures only.

- protocol v1 still gives byte-identical sessions, oracle sessions, pick renders and synthetic renders (fixed hashes,
  computed with the code before this change, in the labels venv)
- search.ambiguity: range results, their limits, repeat and bracket handling, cannot_tell in pick questions
- roofline stations on the overview grid, ticks on pixel boundaries
- v2 synthetic vehicles keep the exact-silhouette property; materials and decals never change coverage; box refused
- honest-oracle band, rangeHalfWidthPx in aggregation and scoring, case-selection exclusions, self-test helpers

Run: .local/venv-labels/bin/python -I -m unittest discover -s tests/evaluation -p 'test_labels*.py'
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_labels_tools import KIT, KIT_DIR, KIT_MODULES, PROTOCOL, RectOracle, blurred_rect, car_task, drive  # noqa: E402

aggregate, build_validation, common, imaging, oracle_mod = KIT.aggregate, KIT.build_validation, KIT.common, KIT.imaging, KIT.oracle
score, selftest, session, synth = KIT.score, KIT.selftest, KIT.session, KIT.synth
Truth = selftest.Truth


def load_select_cases():
    """Import select_cases.py the way test_labels_tools loads the kit (no leaked generic module names)."""
    saved = {name: sys.modules.pop(name) for name in KIT_MODULES if name in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(KIT_DIR))
    try:
        return importlib.import_module('select_cases')
    finally:
        for name in KIT_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


select_cases = load_select_cases()

FRACTIONS = [0.25, 0.5, 0.75]
PROTOCOL_COV = copy.deepcopy(PROTOCOL)
PROTOCOL_COV['version'] = 'test-v2-coverage'
PROTOCOL_COV['search']['ambiguity'] = {'maxCannotTell': 4, 'maxInteriorPx': 3}
PROTOCOL_COV['render']['tickOnPixelBoundary'] = True
PROTOCOL_COV['stations'] = {'insideFraction': 0.45, 'carColumns': 'labelerExtremesOnGrid', 'carFractions': FRACTIONS, 'carMinSpanPx': 4}
PROTOCOL_COV['aggregation'] = {'labelers': 3, 'spreadMaxPx': {'line': 3.0, 'pixel': 2.0}, 'columnSpreadMaxPx': 0}


# ----- protocol v1 fingerprints -------------------------------------------------------------------------

V1_HASHES = {
    'carSession': '7e8922c4ec5314f5195e32311ff99f23b6869e84fd607665c322432ec0a69aae',
    'paintSession': '2c728d0e1ed1a770927c79fe8e3d7c5b9c2d4ae945387f69fe0b166a958b3d02',
    'oracle-0.0-0': '392d412386cbcf3167bec54b23d2fe2ffc1a8cf7fe4fb7255797f4f92066c8e0',
    'oracle-0.1-3': '0e445ac21ec9c995872535678a3bdc4553d3edf8d6a2880b9dd351fdaafe7042',
    'pickRenderY': 'bac7a7b88c516f0721ae1272f258784d585ce0fe8777c2f185ca280410fe1717',
    'pickRenderX': '6ffc9755913098020755e71ce05c9f9b31157c9d5d31e036ab40f7d02534f329',
    'synth-box': '90571a562eab963a5f092136e762fbc1b2d807a709c219dff421a9a84071b362',
    'synth-sedan': '593ffc081087b133c2ec147d738fc3c0ce73825db1e6e933b71d34695a7f7b95',
    'synth-suv': '2d2d37b7727f203427054ffb61c5d1394d62ffc91a5677772224a159af190f21',
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def png_array(png):
    return np.asarray(Image.open(io.BytesIO(png)))


def paint_fixture():
    width, height, s = 200, 160, 8
    xs = (np.arange(width * s) + 0.5) / s - 0.5
    ys = (np.arange(height * s) + 0.5) / s - 0.5
    gx, gy = np.meshgrid(xs, ys)
    near = 90.3 + 0.02 * (gx - 100)
    mask = (gy <= near) & (gy >= near - 9.0) & (gx >= 40) & (gx <= 170)
    cov = mask.reshape(height, s, width, s).mean((1, 3))
    rgb = np.repeat(np.clip(60 + 150 * cov, 0, 255).astype(np.uint8)[..., None], 3, -1)
    task = {'taskId': 'P01', 'kind': 'paint', 'target': {'noun': 'crosswalk paint', 'hint': ''},
            'overview': {'region': [0, 0, 200, 160], 'cell': 16, 'columns': 13, 'rows': 10, 'columnLabels': ['A', 'M'], 'rowLabels': [1, 10]},
            'stationColumns': [60, 100, 140]}

    def answer(q):
        if q['kind'] == 'box':
            return 'C5:K7'
        if q['kind'] == 'pick':
            target = 90.3 + 0.02 * (q['render']['marker']['at'] - 100)
            best = min(q['render']['candidates'], key=lambda c: abs(c[2] - target))
            return str(best[0]) if abs(best[2] - target) <= 1.5 else '0'
        if q['render'].get('purpose') == 'visible':
            return 'yes'
        i, j = q['render']['marker']['at']
        return 'yes' if cov[j, i] >= 0.5 else 'no'
    return task, rgb, answer


def rect_truth(box=(60.3, 50.6, 140.7, 120.2)):
    rgb, cov = blurred_rect(box=box)
    x0, y0, x1, y1 = box
    entry = {'source': 'synthetic', 'kind': 'car', 'carPolygons': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
             '_coverage': cov, 'coverageOrigin': [0, 0]}
    return rgb, cov, entry


def test_camera():
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    R = np.array([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]])
    T4 = np.eye(4)
    T4[:3, :3] = R
    T4[:3, 3] = -R @ np.array([0, 0, 1.6])
    return synth.Camera(K, T4, 640, 480)


class ProtocolV1UnchangedTests(unittest.TestCase):
    """A v1 protocol (no search.ambiguity, carColumns 'task', v1 synthetic models) must reproduce v1 exactly."""

    def test_sessions_and_oracle_sessions(self):
        rgb, cov = blurred_rect()
        task = car_task()
        self.assertEqual(digest(drive(task, PROTOCOL, rgb, RectOracle(cov, task).answer)), V1_HASHES['carSession'])
        ptask, prgb, panswer = paint_fixture()
        self.assertEqual(digest(drive(ptask, PROTOCOL, prgb, panswer)), V1_HASHES['paintSession'])
        srgb, _, entry = rect_truth()
        for flip, seed in ((0.0, 0), (0.1, 3)):
            o = oracle_mod.Oracle(Truth(entry), task, flip=flip, seed=seed)
            self.assertEqual(digest(drive(task, PROTOCOL, srgb, o.answer)), V1_HASHES[f'oracle-{flip}-{seed}'])

    def test_pick_renders(self):
        rgb, _ = blurred_rect()
        png = imaging.render_question(rgb, [0, 0, 200, 160], [40, 30, 49, 81], 6, {'type': 'column', 'at': 80},
                                      candidates=[(1, 'y', 50.3, 80), (2, 'y', 57.8, 80)])
        self.assertEqual(array_digest(png_array(png)), V1_HASHES['pickRenderY'])
        png = imaging.render_question(rgb, [0, 0, 200, 160], [30, 40, 81, 49], 6, {'type': 'row', 'at': 80},
                                      candidates=[(1, 'x', 60.2, 80), (2, 'x', 66.9, 80)])
        self.assertEqual(array_digest(png_array(png)), V1_HASHES['pickRenderX'])

    def test_v1_synthetic_renders(self):
        cam = test_camera()
        mats = {'body': {'albedo': 0.3, 'ambient': 0.55, 'diffuse': 0.55, 'sky': 0.02},
                'cabin': {'albedo': 0.035, 'ambient': 0.5, 'diffuse': 0.3, 'sky': 0.06},
                'tire': {'albedo': 0.03, 'ambient': 0.6, 'diffuse': 0.3}}
        for model in ('box', 'sedan', 'suv'):
            pose = synth.plane_frame(np.zeros(3), np.array([12.0, 0.8]), np.radians(35))
            parts = [(n, synth.apply(pose, v), f, m) for n, v, f, m in synth.vehicle(model, 4.6, 1.85, 1.5)]
            image, coverage, region, polygons = synth.render_solids(np.full((480, 640, 3), 90, np.uint8), cam, parts, mats, 4,
                                                                    seed=7, exposure=1.2)
            foot = synth.apply(pose, np.array([[x, y, 0.005] for x in (-2.4, 2.4) for y in (-1, 1)]))
            shadowed = synth.shadow(np.full((480, 640, 3), 90.0), cam, foot, 0.5)
            self.assertEqual(array_digest(image, coverage, np.asarray(region), *polygons, shadowed), V1_HASHES['synth-' + model], model)


# ----- scripted range search ----------------------------------------------------------------------------

class FakeContext:
    def __init__(self, protocol):
        self.p, self.rng, self.counter = protocol, random.Random(0), 0

    def question(self, kind, feature, text, options=None, render=None):
        self.counter += 1
        return session.Question(id=self.counter, kind=kind, feature=feature, text=text, options=options, render=render or {})


def spec_for(a, b, qa=True, family='line', candidates=None, candidate_first=False):
    spec = {'family': family, 'axis': 'y', 'fixed': 40, 'a': a, 'b': b, 'qa': qa, 'expandStep': 4, 'context': [0, 0, 64, 64],
            'window': lambda pos, jitter: [0, 0, 9, 9], 'zoom': 6, 'marker': lambda pos: {'type': 'hline', 'at': pos - 0.5},
            'text': lambda pos: 'question'}
    if candidates is not None:
        spec.update(candidates=lambda lo, hi: [(c, 5.0) for c in candidates], candidateInside=2, candidateFirst=candidate_first,
                    pick_text=lambda n: 'pick')
    return spec


def scripted(edge, unsure=(), qa=True, override=None):
    """Answer qa at positions <= edge and (not qa) beyond; cannot_tell at `unsure`. override(q) may return an answer first."""
    def answer(q):
        if override:
            got = override(q)
            if got is not None:
                return got
        if q['kind'] == 'pick':
            return '0'
        pos = q['render']['position']
        if pos in unsure:
            return 'cannot_tell'
        return 'yes' if (pos <= edge) == qa else 'no'
    return answer


def run_search(protocol, spec, answer):
    gen = session.search(FakeContext(protocol), 'B', spec)
    asked = []
    try:
        q = next(gen)
        while True:
            asked.append(q)
            q = gen.send(session.normalize_answer(q, answer(q)))
    except StopIteration as stop:
        return stop.value, asked


class RangeSearchTests(unittest.TestCase):
    def test_ordinary_result_asks_and_estimates_exactly_as_v1(self):
        for family in ('line', 'pixel'):
            for qa in (True, False):
                v1, asked_v1 = run_search(PROTOCOL, spec_for(0, 16, qa, family), scripted(7, qa=qa))
                v2, asked_v2 = run_search(PROTOCOL_COV, spec_for(0, 16, qa, family), scripted(7, qa=qa))
                self.assertEqual([q['render']['position'] for q in asked_v1], [q['render']['position'] for q in asked_v2])
                self.assertEqual((v2['status'], v2['bracket'], v2['estimate']), (v1['status'], v1['bracket'], v1['estimate']))
                self.assertEqual((v2['rangeHalfWidthPx'], v2['ambiguous'], v2['cannotTell']), (0.0, [], 0))
                self.assertNotIn('rangeHalfWidthPx', v1)

    def test_one_ambiguous_position_gives_a_range(self):
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(7, unsure={8}))
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual((result['bracket'], result['ambiguous'], result['rangeHalfWidthPx']), ([7, 9], [8], 0.5))
        self.assertEqual(result['estimate'], 8.0)  # line, qa yes: (a + 0.5) + n / 2
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, False), scripted(7, unsure={8}, qa=False))
        self.assertEqual(result['estimate'], 7.0)  # line, qa no: (a - 0.5) + n / 2
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True, 'pixel'), scripted(7, unsure={8}))
        self.assertEqual(result['estimate'], 8.0)  # pixel: (a + b) / 2
        v1, _ = run_search(PROTOCOL, spec_for(0, 16, True), scripted(7, unsure={8}))
        self.assertEqual((v1['status'], v1['reason']), ('unresolved', 'cannot_tell during bisection'))

    def test_three_ambiguous_positions_are_the_widest_range(self):
        result, asked = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(6, unsure={7, 8, 9}))
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual((result['bracket'], result['ambiguous'], result['rangeHalfWidthPx']), ([6, 10], [7, 8, 9], 1.5))
        self.assertEqual((result['estimate'], result['cannotTell']), (8.0, 3))
        bisect = [q['render']['position'] for q in asked if q['render']['purpose'] == 'bisect']
        self.assertEqual(bisect[:3], [8, 7, 9])  # nearest the midpoint, ties to the lower position

    def test_four_ambiguous_positions_are_too_wide(self):
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(5, unsure={6, 7, 8, 9}))
        self.assertEqual((result['status'], result['reason']), ('unresolved', 'ambiguous range too wide'))
        self.assertEqual((result['bracket'], result['rangeHalfWidthPx'], result['cannotTell']), ([5, 10], 2.0, 4))

    def test_too_many_cannot_tell_answers(self):
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(2, unsure={3, 6, 7, 8, 9}))
        self.assertEqual((result['status'], result['reason']), ('unresolved', 'too many cannot_tell'))
        self.assertEqual((result['bracket'], result['rangeHalfWidthPx'], result['cannotTell']), ([2, 4], 0.5, 5))

    def test_cannot_tell_at_a_repeat_is_a_contradiction(self):
        state = {'repeats': 0}

        def first_repeat_unsure(q):
            if q['render']['purpose'] == 'repeat':
                state['repeats'] += 1
                return 'cannot_tell' if state['repeats'] == 1 else None
            return None
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(7, unsure={8}, override=first_repeat_unsure))
        self.assertEqual((result['status'], result['restarts'], result['bracket']), ('resolved', 1, [7, 9]))
        self.assertEqual(result['cannotTell'], 3)  # counted across the restart; the ambiguous set itself starts empty again
        self.assertEqual(result['ambiguous'], [8])
        always = scripted(7, unsure={8}, override=lambda q: 'cannot_tell' if q['render']['purpose'] == 'repeat' else None)
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), always)
        self.assertEqual((result['status'], result['reason']), ('unresolved', 'repeat answer contradicted bisection twice'))

    def test_cannot_tell_at_a_bracket_end_expands_the_bracket(self):
        state = {'asked': 0}

        def first_bracket_unsure(q):
            if q['render']['purpose'] == 'bracket-a':
                state['asked'] += 1
                return 'cannot_tell' if state['asked'] == 1 else None
            return None
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True), scripted(7, override=first_bracket_unsure))
        self.assertEqual((result['status'], result['expansions'], result['cannotTell']), ('resolved', 1, 1))
        self.assertEqual((result['estimate'], result['rangeHalfWidthPx']), (7.5, 0.0))
        state['asked'] = 0
        v1, _ = run_search(PROTOCOL, spec_for(0, 16, True), scripted(7, override=first_bracket_unsure))
        self.assertEqual((v1['status'], v1['reason']), ('unresolved', 'cannot_tell at bracket a'))


class PickCannotTellTests(unittest.TestCase):
    def test_pick_accepts_cannot_tell_only_under_ambiguity(self):
        unsure_pick = scripted(7, override=lambda q: 'cannot_tell' if q['kind'] == 'pick' else None)
        result, asked = run_search(PROTOCOL_COV, spec_for(0, 16, True, 'pixel', candidates=[5.4, 7.6]), unsure_pick)
        pick = next(q for q in asked if q['kind'] == 'pick')
        self.assertTrue(pick['render']['allowCannotTell'])
        self.assertTrue(pick['render']['tickOnPixelBoundary'])
        self.assertEqual((result['candidate']['picked'], result['candidate']['cannotTell']), (0, True))
        self.assertEqual((result['status'], result['estimate'], result['cannotTell']), ('resolved', 7.5, 1))  # acts like 0
        _, asked_v1 = run_search(PROTOCOL, spec_for(0, 16, True, 'pixel', candidates=[5.4, 7.6]), scripted(7))
        pick_v1 = next(q for q in asked_v1 if q['kind'] == 'pick')
        self.assertNotIn('allowCannotTell', pick_v1['render'])
        self.assertNotIn('tickOnPixelBoundary', pick_v1['render'])
        with self.assertRaises(ValueError):
            session.normalize_answer(pick_v1, 'cannot_tell')
        self.assertEqual(session.normalize_answer(pick, "can't tell"), 'cannot_tell')

    def test_paint_pick_cannot_tell_leaves_the_station_unresolved(self):
        unsure_pick = scripted(7, override=lambda q: 'cannot_tell' if q['kind'] == 'pick' else None)
        result, _ = run_search(PROTOCOL_COV, spec_for(0, 16, True, 'pixel', candidates=[7.6], candidate_first=True), unsure_pick)
        self.assertEqual((result['status'], result['reason']), ('unresolved', 'cannot_tell at candidate pick'))

    def test_pick_text_mentions_cannot_tell_only_under_ambiguity(self):
        rgb, cov = blurred_rect()
        for protocol, expected in ((PROTOCOL_COV, True), (PROTOCOL, False)):
            task = car_task()
            if protocol is PROTOCOL_COV:
                task.pop('stationColumns')
            oracle, picks = RectOracle(cov, task), []

            def answer(q):
                if q['kind'] == 'pick':
                    picks.append(q)
                return oracle.answer(q)
            drive(task, protocol, rgb, answer)
            self.assertTrue(picks)
            self.assertEqual({q['text'].endswith(' Answer cannot_tell if you cannot tell.') for q in picks}, {expected})


# ----- stations, ticks -------------------------------------------------------------------------------------

class GridStationTests(unittest.TestCase):
    def columns(self, left, right, region_x0=0, count=13):
        task = car_task()
        task['overview'] = dict(task['overview'], region=[region_x0, 0, 200, 160], columns=count)
        ctx = session.Context(task, PROTOCOL_COV, 'X', np.zeros((160, 200, 3), np.uint8))
        return session.station_columns(ctx, {'L': {'value': left}, 'R': {'value': right}}, (48, 159, 48, 127))

    def test_targets_move_to_the_nearest_grid_cell_centre(self):
        self.assertEqual(self.columns(60.5, 140.5), [88, 104, 120])
        self.assertEqual(self.columns(60.7, 140.2), [88, 104, 120])  # a pixel apart: the same columns
        self.assertEqual(self.columns(60.5, 140.5, region_x0=5), [77, 93, 125])
        self.assertEqual(self.columns(190.5, 199.5, count=12), [184, 184, 184])  # clamped to the last cell
        self.assertEqual(self.columns(60.5, 63.5), [])  # span under carMinSpanPx

    def test_stations_on_the_same_grid_column_are_kept_once(self):
        rgb, cov = blurred_rect(box=(60.3, 50.6, 70.7, 120.2))
        task = car_task()
        task.pop('stationColumns')
        result, _ = drive(task, PROTOCOL_COV, rgb, RectOracle(cov, task).answer)
        self.assertEqual(result['stationColumns'], [56, 72, 72])
        self.assertEqual(result['features']['S3'], {'status': 'unresolved', 'reason': 'same grid column', 'fixed': 72})
        self.assertNotEqual(result['features']['S2'].get('reason'), 'same grid column')

    def test_grid_columns_must_be_identical_in_aggregation(self):
        entries = {k: {'status': 'resolved', 'value': 50.0 + d, 'fixed': c} for k, c, d in (('A', 88, 0), ('B', 88, 0.4), ('C', 104, 0.6))}
        record = aggregate.combine('S1', entries, {'line': 3.0, 'pixel': 2.0}, 0)
        self.assertEqual((record['status'], record['columnSpread']), ('unresolved', 16))
        entries['C']['fixed'] = 88
        self.assertEqual(aggregate.combine('S1', entries, {'line': 3.0, 'pixel': 2.0}, 0)['status'], 'resolved')


class TickTests(unittest.TestCase):
    def render(self, coordinate, flag):
        rgb, _ = blurred_rect()
        return png_array(imaging.render_question(rgb, [0, 0, 200, 160], [40, 30, 49, 81], 6, {'type': 'column', 'at': 80},
                                                 candidates=[(1, 'y', coordinate, 80)], tick_on_pixel_boundary=flag))

    def test_ticks_sit_on_the_nearest_pixel_boundary_below(self):
        boundary = self.render(57.5, False)
        np.testing.assert_array_equal(self.render(57.3, True), boundary)
        np.testing.assert_array_equal(self.render(57.9, True), boundary)
        self.assertFalse(np.array_equal(self.render(57.3, False), boundary))


# ----- synthetic v2 vehicles ---------------------------------------------------------------------------------

MATERIALS_V2 = {'body': {'albedo': 0.3, 'ambient': 0.55, 'diffuse': 0.55, 'sky': 0.02},
                'glass': {'albedo': 0.035, 'ambient': 0.5, 'diffuse': 0.3, 'sky': 0.06, 'gradient': 0.04},
                'tire': {'albedo': 0.03, 'ambient': 0.6, 'diffuse': 0.3}, 'rim': {'albedo': 0.3, 'ambient': 0.6, 'diffuse': 0.4, 'sky': 0.03},
                'trim': {'albedo': 0.05, 'ambient': 0.6, 'diffuse': 0.4, 'sky': 0.02}, 'arch': {'albedo': 0.02, 'ambient': 0.6, 'diffuse': 0.2},
                'headlight': {'albedo': 0.6, 'ambient': 0.7, 'diffuse': 0.3, 'sky': 0.05},
                'taillight': {'albedo': 0.15, 'ambient': 0.7, 'diffuse': 0.3}}
MODELS_V2 = (('sedan', (4.7, 1.83, 1.45), 35), ('hatchback', (4.2, 1.78, 1.5), 80), ('suv', (4.8, 1.95, 1.78), -30),
             ('van', (5.0, 1.95, 1.98), 150))


class SyntheticV2Tests(unittest.TestCase):
    S = 4

    def render(self, kind, size, yaw, materials, decals=True, seed=5):
        pose = synth.plane_frame(np.zeros(3), np.array([11.0, 0.5]), np.radians(yaw))
        parts, local_decals, _wheels = synth.vehicle_v2(kind, *size)
        parts = [(n, synth.apply(pose, v), f, m) for n, v, f, m in parts]
        dc = {n: [(fi, synth.apply(pose, v), m) for fi, v, m in items] for n, items in local_decals.items()} if decals else None
        return synth.render_solids(np.full((480, 640, 3), 100.0), test_camera(), parts, materials, self.S, seed=seed, exposure=3.0,
                                   decals=dc)

    def test_coverage_is_exactly_the_union_of_the_part_polygons(self):
        s = self.S
        for kind, size, yaw in MODELS_V2:
            _, cov, region, polygons = self.render(kind, size, yaw, MATERIALS_V2)
            x0, y0, w, h = region
            gx, gy = np.meshgrid(x0 - 0.5 + (np.arange(w * s) + 0.5) / s, y0 - 0.5 + (np.arange(h * s) + 0.5) / s)
            hit = np.zeros(gx.shape, bool)
            for p in polygons:
                hit |= synth.inside_convex(p, gx, gy)
            np.testing.assert_array_equal(cov, hit.reshape(h, s, w, s).mean((1, 3)), err_msg=kind)
            self.assertGreater(len(polygons), 10, kind)

    def test_materials_and_decals_never_change_coverage(self):
        plain = {k: {'albedo': 0.2, 'ambient': 0.5, 'diffuse': 0.5} for k in MATERIALS_V2}
        for kind, size, yaw in MODELS_V2:
            image, cov, region, polygons = self.render(kind, size, yaw, MATERIALS_V2)
            image2, cov2, region2, polygons2 = self.render(kind, size, yaw, plain, decals=False, seed=9)
            np.testing.assert_array_equal(cov, cov2, err_msg=kind)
            self.assertEqual(region, region2)
            self.assertEqual(len(polygons), len(polygons2))
            for p, q in zip(polygons, polygons2):
                np.testing.assert_array_equal(p, q)
            self.assertFalse(np.array_equal(image, image2), kind)  # the look does change

    def test_v2_refuses_box_and_impossible_heights(self):
        with self.assertRaises(ValueError):
            synth.vehicle_v2('box', 4.6, 1.85, 1.5)
        with self.assertRaises(ValueError):
            synth.vehicle_v2('suv', 4.8, 1.95, 1.2)  # no room for a cabin above the belt line
        S = {'vehicleModels': 'v2', 'cases': [{'id': 'SV01', 'kind': 'car', 'model': 'sedan'}, {'id': 'SQ01', 'kind': 'paint'}]}
        build_validation.check_synthetic_models(S)
        S['cases'].append({'id': 'SV02', 'kind': 'car', 'model': 'box'})
        with self.assertRaises(common.ProtocolError):
            build_validation.check_synthetic_models(S)
        build_validation.check_synthetic_models({'cases': S['cases']})  # v1 keeps box

    def test_contact_footprints_sit_on_the_ground_under_the_wheels(self):
        _parts, _decals, wheels = synth.vehicle_v2('sedan', 4.7, 1.83, 1.45)
        body, tires = synth.contact_footprints(4.7, 1.83, wheels, (0.15, -0.1))
        self.assertEqual(len(tires), len(wheels))
        self.assertTrue(np.all(np.abs(body[:, 2] - 0.005) < 1e-9))
        self.assertTrue(all(np.all(np.abs(t[:, 2] - 0.004) < 1e-9) for t in tires))


# ----- oracle band, aggregation, scoring --------------------------------------------------------------------

class OracleBandTests(unittest.TestCase):
    def setUp(self):
        self.rgb, self.cov, entry = rect_truth(box=(60.0, 50.6, 140.7, 120.2))
        self.truth = Truth(entry)
        self.task = car_task()

    def yesno(self, marker, feature='L', position=0):
        return {'kind': 'yesno', 'feature': feature, 'render': {'marker': marker, 'position': position, 'purpose': 'bisect'}}

    def test_band_answers_cannot_tell_only_inside_the_band(self):
        band = oracle_mod.Oracle(self.truth, self.task, ambiguous_band=(0.3, 0.7))
        plain = oracle_mod.Oracle(self.truth, self.task)
        half = self.yesno({'type': 'pixel', 'at': (60, 80)})  # coverage 0.5
        self.assertEqual((band.answer(half), plain.answer(half)), ('cannot_tell', 'yes'))
        self.assertEqual(band.answer(self.yesno({'type': 'pixel', 'at': (61, 80)})), 'yes')
        self.assertEqual(band.answer(self.yesno({'type': 'pixel', 'at': (59, 80)})), 'no')
        line = {pos: self.yesno({'type': 'vline', 'at': pos - 0.5}, 'L', pos) for pos in (60, 61, 62)}
        self.assertEqual([band.answer(line[p]) for p in (60, 61, 62)], ['no', 'cannot_tell', 'yes'])
        self.assertEqual(band.answer({'kind': 'pick', 'feature': 'S1', 'render': {'marker': {'at': 100}, 'candidates': [(1, 'y', 50.6, 100)]}}), '1')

    def test_band_oracle_with_range_search_recovers_the_edge(self):
        band = oracle_mod.Oracle(self.truth, self.task, ambiguous_band=(0.3, 0.7))
        result, _ = drive(self.task, PROTOCOL_COV, self.rgb, band.answer)
        left = result['features']['L']
        self.assertEqual((left['status'], left['bracket'], left['rangeHalfWidthPx'], left['value']), ('resolved', [60, 62], 0.5, 60.0))
        v1, _ = drive(self.task, PROTOCOL, self.rgb, oracle_mod.Oracle(self.truth, self.task, ambiguous_band=(0.3, 0.7)).answer)
        self.assertEqual(v1['features']['L']['reason'], 'cannot_tell during bisection')


class RangeAggregationAndScoreTests(unittest.TestCase):
    SPREAD = {'line': 3.0, 'pixel': 2.0}

    def test_aggregate_keeps_the_widest_stated_range(self):
        entries = {k: {'status': 'resolved', 'value': v, 'rangeHalfWidthPx': h} for k, v, h in (('A', 50.0, 0.0), ('B', 50.5, 1.5), ('C', 50.2, 0.5))}
        record = aggregate.combine('B', entries, self.SPREAD)
        self.assertEqual((record['status'], record['value'], record['rangeHalfWidthPx']), ('resolved', 50.2, 1.5))
        for e in entries.values():
            e.pop('rangeHalfWidthPx')
        self.assertNotIn('rangeHalfWidthPx', aggregate.combine('B', entries, self.SPREAD))

    def test_score_counts_range_results(self):
        rows = [{'error': 0.2, 'referenceKind': 'exact', 'rangeHalfWidthPx': h} for h in (0.0, 0.5, 1.5)]
        self.assertEqual(score.stats(rows)['rangeN'], 2)
        self.assertNotIn('rangeN', score.stats([{'error': 0.2, 'referenceKind': 'exact'}]))


# ----- case selection, self-test helpers ----------------------------------------------------------------------

class ExclusionTests(unittest.TestCase):
    def test_frames_and_tracks_are_excluded(self):
        excluded = select_cases.frame_excluded({'excludeFrames': ['315970633399927215', 100]})
        self.assertTrue(excluded(315970633399927215) and excluded('100'))
        self.assertFalse(excluded(150))
        near = select_cases.frame_excluded({'excludeFrames': [1_000_000_000], 'excludeFrameWindowS': 0.5})
        self.assertTrue(near(1_400_000_000) and near(600_000_000))
        self.assertFalse(near(1_600_000_000))
        self.assertFalse(select_cases.frame_excluded({})(100))
        self.assertTrue(select_cases.track_excluded({'excludeTracks': ['b4645064']}, 'b4645064-0000-aaaa'))
        self.assertFalse(select_cases.track_excluded({}, 'b4645064-0000-aaaa'))

    def test_app_case_is_skipped_when_its_frame_or_track_is_excluded(self):
        app = select_cases.APP_CASE
        self.assertTrue(select_cases.app_case_allowed({}))
        self.assertFalse(select_cases.app_case_allowed({'excludeTracks': [app['track']]}))
        self.assertFalse(select_cases.app_case_allowed({'excludeFrames': [str(app['frame'])]}))


class SelftestHelperTests(unittest.TestCase):
    def test_band_and_variant_options(self):
        self.assertIsNone(selftest.parse_band(''))
        self.assertEqual(selftest.parse_band('0.3,0.7'), (0.3, 0.7))
        with self.assertRaises(common.ProtocolError):
            selftest.parse_band('0.7,0.3')
        earlier = selftest.protocol_variant(PROTOCOL_COV, 'earlier-draft')
        self.assertNotIn('ambiguity', earlier['search'])
        self.assertNotIn('tickOnPixelBoundary', earlier['render'])
        self.assertEqual((earlier['stations']['carColumns'], earlier['aggregation']['columnSpreadMaxPx']), ('labelerExtremes', 3.0))
        self.assertIn('ambiguity', PROTOCOL_COV['search'])  # the protocol itself is not changed
        with self.assertRaises(common.ProtocolError):
            selftest.protocol_variant(PROTOCOL_COV, 'other')

    def test_oracle_sessions_are_checked_without_a_transcript_scan_only(self):
        protocol = dict(copy.deepcopy(PROTOCOL_COV), isolation={'requireTranscriptScan': True, 'coordinatorRootEnv': 'X'})
        checked = selftest.oracle_protocol(protocol)
        self.assertFalse(checked['isolation']['requireTranscriptScan'])
        self.assertTrue(protocol['isolation']['requireTranscriptScan'])
        checked['isolation']['requireTranscriptScan'] = True
        self.assertEqual(checked, protocol)  # nothing else differs
        self.assertIs(selftest.oracle_protocol(PROTOCOL_COV), PROTOCOL_COV)


if __name__ == '__main__':
    unittest.main()

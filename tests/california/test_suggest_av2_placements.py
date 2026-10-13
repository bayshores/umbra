"""Synthetic analytic fixtures for AV2 map-first suggestions (schema v3, protocol v3), plus real-data checks skipped when files are absent."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/california/suggest_av2_placements.py'
spec = importlib.util.spec_from_file_location('suggest_av2_placements', SCRIPT)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
spec = importlib.util.spec_from_file_location('ground_fixture', ROOT / 'tests/california/test_prepare_av2_ground.py')
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
EPOCH = '1791768480'
PLACED = ('crossing', 'vehicle', 'person')
PENDING = {'basis': 'mapped near edge', 'status': 'pending-measurement', 'label': 'paint offset pending measurement', 'plusMinusM': None, 'evidence': None}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol_v32(**parameters):
    """The fixture protocol v3 with the protocol v3.2 crossingApplicability section (parameters may be overridden to test refusal)."""
    protocol = copy.deepcopy(fixture.PROTOCOL_V3); protocol['protocolRevision'] = '3.2'
    protocol['crossingApplicability'] = {'parameters': {**tool.APPLICABILITY_PARAMETERS, **parameters}, 'refusalCodes': list(tool.REFUSAL_CODES)}
    return protocol


def crossing_edges(edge1, edge2):
    """edit_map: replace crossing 77's edges with world (x, z) end points on the flat mapped ground."""
    def edit(native_map):
        crossing = native_map['pedestrian_crossings']['77']
        for name, ends in (('edge1', edge1), ('edge2', edge2)):
            crossing[name] = [fixture.point(x, fixture.GROUND_Y, z) for x, z in ends]
    return edit


class SuggestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def grounded(self, name='a', annotations=True, protocol=None, **options):
        args = fixture.write_fixture(self.root / name, protocol=fixture.PROTOCOL_V3 if protocol is None else protocol,
                                     returns=fixture.V3_RETURNS, eligible=fixture.V3_ELIGIBLE, **options)
        if not annotations:
            args.update(annotations_path=None, annotations_sha256=None)
        fixture.tool.build(**args)
        out = args['output']
        return dict(grounded=out, metadata_sha256=sha((out / 'scene.json').read_bytes()), buffers_dir=args['scene_path'].parent,
                    protocol_path=args['protocol_path'], protocol_sha256=args['protocol_sha256'])

    def document(self, paths):
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': EPOCH}):
            return tool.document(*tool.load(**paths))

    def context(self, terrain=None, **options):
        name = f'context{len(list(self.root.iterdir()))}'
        scene, ground, base_terrain, positions, eligible, protocol, _ = tool.load(**self.grounded(name, **options))
        return tool.Context(scene, ground, base_terrain if terrain is None else terrain(base_terrain, ground), positions, eligible, protocol)

    def test_annotated_stationary_vehicle_sets_crossing_and_hides_points(self):
        doc = self.document(self.grounded())
        v, c, p = doc['vehicle'], doc['crossing'], doc['person']
        band = doc['tolerances']['alongRayHeightBandMeters']
        self.assertEqual((doc['schemaVersion'], doc['sourceClass'], doc['generatedUtc']), (3, 'mappedGround', '2026-10-12T01:28:00Z'))
        self.assertEqual(doc['tolerances'], {'roadToleranceMeters': .08, 'supportHeightAgreementMeters': .08, 'supportRadiusMeters': .5,
                                             'maxRangeMeters': 60.0, 'placementRangeLimitMeters': 30.0, 'alongRayHeightBandMeters': .094})
        self.assertEqual((v['status'], v['label'], v['reason']), ('annotated', 'stationary vehicle', tool.STATIONARY_REASON))
        self.assertEqual(v['dimensions'], {'width': 1.8, 'height': 1.5, 'length': 4.0}); self.assertEqual(v['track'], fixture.TRACK)
        self.assertEqual(v['pose'], {'kind': 'averaged', 'annotations': 50}); self.assertEqual(v['appModel'], tool.APP_MODEL)
        # The averaged box sits 0.00392 m toward world +X of the sweep box (fixture jitter), heading +Z, front at Z = 19.5.
        self.assertAlmostEqual(v['frontGapToCrossingM'], .5, places=9); self.assertAlmostEqual(v['headingToRoadAxisDeg'], 0, places=6)
        np.testing.assert_allclose(np.asarray(v['mappedPoint'])[[0, 2]], [-4 + .00392, 19.5], atol=1e-9)
        self.assertEqual((c['status'], c['nearEdge'], c['crossingError']), ('mapped', 'edge1', PENDING))
        self.assertNotIn('paintOffset', c)
        np.testing.assert_allclose(np.asarray(c['mappedPoint'])[[0, 2]], [-4 + .00392, 20], atol=1e-9)
        np.testing.assert_allclose(np.asarray(p['mappedPoint'])[[0, 2]], [-5.7, 20], atol=1e-9)
        self.assertEqual((p['status'], p['label'], p['heightM']), ('declared-default', 'child', 1.1))
        # Analytic: each camera ray to these ground points passes through the car box (X -4.9..-3.1, Z 15.5..19.5, Y -1.6..-0.1).
        self.assertEqual([doc[k]['hidden'] for k in PLACED], [True, True, True])
        self.assertEqual([doc[k]['supportClass'] for k in PLACED], ['map-only'] * 3)
        self.assertEqual({doc[k]['supportLabel'] for k in PLACED}, {'mapped, not observed'})
        for k in PLACED:
            o = doc[k]; g = np.asarray(o['groundPoint'])
            self.assertAlmostEqual(g[1], fixture.RASTER_Y, places=9)  # float16 raster
            # Fractional evaluation reproduces the mapped point; the integer anchor alone would not.
            self.assertLess(o['pixelOffsetFromMappedM'], 1e-9); np.testing.assert_allclose(g, o['mappedPoint'], atol=1e-9)
            self.assertEqual(o['pixel'], [math.floor(o['pixelUV'][0] + .5), math.floor(o['pixelUV'][1] + .5)])
            np.testing.assert_allclose(tool.project(tool.load(**self.grounded(f'p{k}'))[0], o['mappedPoint']), o['pixelUV'], atol=1e-9)
            self.assertTrue(0 < o['integerPixelCheck']['residualM'] < .35)
            # alongRayUncertaintyM = band / sin(depression) of the pixelUV ray.
            self.assertAlmostEqual(o['alongRayUncertaintyM'], band * np.linalg.norm(g) / -g[1], places=9)
            self.assertAlmostEqual(o['horizontalRangeM'], math.hypot(g[0], g[2]), places=12)
            self.assertNotIn('pixelSearch', o); self.assertNotIn('projectedUV', o)
        r = v['frontResidualToAnnotatedM']
        self.assertAlmostEqual(r['horizontal'], v['pixelOffsetFromMappedM'], places=12); self.assertLess(r['horizontal'], 1e-9)
        a = doc['roadAxis']; np.testing.assert_allclose(a['directionOfTravel'], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(a['skewDeg'], 0, places=6); self.assertTrue(a['cameraInLane'])
        self.assertAlmostEqual(doc['curb']['nearEdgeLateralM'], 6.0, places=9); self.assertEqual(doc['curb']['label'], 'curb (map edge proxy)')
        self.assertEqual(doc['license']['id'], 'CC-BY-NC-SA-4.0'); self.assertEqual(doc['license']['attribution'], '(c) 2021 Argo AI, LLC')

    @staticmethod
    def raised_band(terrain, ground):
        """Raster rows 169-171 (world Z 18.4-19.3) 5 cm higher: rays to targets just behind the band hit its top instead."""
        w = terrain.window.copy(); w[169 - terrain.row0:172 - terrain.row0, :] += np.float32(.05)
        return tool.Terrain(w, terrain.column0, terrain.row0, terrain.shape, ground['terrain']['sim2'], ground['frame']['cityOrigin'])

    def test_fractional_position_is_exact_and_never_hides_raster_occlusion(self):
        ctx = self.context(terrain=self.raised_band)
        for z in (8.0, 12.3, 17.4, 18.0, 20.0, 24.7):  # flat cells: the pixelUV ray reaches the mapped point exactly
            placed = ctx.place(np.array([1.0, z]), {'status': 'mapped'})
            self.assertEqual(placed['status'], 'mapped'); self.assertLess(placed['pixelOffsetFromMappedM'], 1e-9)
            self.assertGreater(placed['integerPixelCheck']['residualM'], placed['pixelOffsetFromMappedM'])
        # Just behind the raised band the exact ray meets the band first: the residual is recorded, not searched away.
        hidden = ctx.place(np.array([1.0, 19.4]), {'status': 'mapped'})
        self.assertEqual(hidden['status'], 'mapped'); self.assertGreater(hidden['pixelOffsetFromMappedM'], .1)
        self.assertEqual(hidden['hitKind'], 'top'); self.assertGreater(hidden['groundPoint'][1], fixture.RASTER_Y + .04)  # on the band top
        # A pixelUV whose ray reaches no drivable ground keeps the object Unknown with the runtime reason (no fallback pixel).
        off = ctx.place(np.array([6.1, 20.0]), {'status': 'mapped'})
        self.assertEqual((off['status'], off['supportClass']), ('unknown', 'unknown')); self.assertIn('outside the mapped drivable area', off['reason'])
        self.assertEqual(off['pixel'], [math.floor(off['pixelUV'][0] + .5), math.floor(off['pixelUV'][1] + .5)])
        outside = ctx.place(np.array([40.0, 20.0]), {'status': 'mapped'})
        self.assertEqual(outside['reason'], 'Mapped point is outside the recorded image.')

    def test_placement_beyond_30_m_is_refused(self):
        ctx = self.context()
        def at(x, z):
            return ctx.evaluate_pixel(tool.project(ctx.scene, [x, fixture.RASTER_Y, z]))
        near, far = at(.5, 29.9), at(.5, 30.1)
        self.assertEqual(near['supportClass'], 'map-only'); self.assertLessEqual(near['horizontalRangeM'], 30.0)
        self.assertEqual(far, {'supportClass': 'unknown', 'reason': 'Too far to place (over 30 m).'})
        placed = ctx.place(np.array([.5, 31.0]), {'status': 'mapped'})
        self.assertEqual((placed['status'], placed['reason']), ('unknown', 'Too far to place (over 30 m).'))
        # The limit comes from the ground layer; a larger one (still <= maxRangeMeters) would place the same point.
        ctx.limit = 60.0
        self.assertEqual(at(.5, 30.1)['supportClass'], 'map-only')

    def test_support_classes_follow_the_runtime_pixel_rule(self):
        ctx = self.context()
        def at(x, z):
            return ctx.evaluate_pixel(tool.project(ctx.scene, [x, fixture.RASTER_Y, z]))
        near = at(1.0, 10.0)
        self.assertEqual(near['supportClass'], 'observed-nearby'); self.assertFalse(near['hidden'])
        # All returns are tested: one agreeing and one 0.6 m above the map lie within 0.5 m.
        self.assertEqual({k: near['supportEvidence'][k] for k in ('cameraVisibleWithinRadius', 'agreeingWithinRadius', 'aboveToleranceWithinRadius')},
                         {'cameraVisibleWithinRadius': 2, 'agreeingWithinRadius': 1, 'aboveToleranceWithinRadius': 1})
        self.assertEqual(at(1.0, 13.0)['supportClass'], 'map-only')
        off = at(8.0, 20.0); self.assertEqual(off['supportClass'], 'unknown'); self.assertIn('outside the mapped drivable area', off['reason'])
        nan = at(3.0, 29.0)
        self.assertEqual(nan['supportClass'], 'map-only')
        behind = at(-4.0, 21.0)  # an agreeing return exists here but the camera sees the car, so it is not support
        self.assertEqual((behind['supportClass'], behind['hidden']), ('map-only', True))
        # Integer pixel centres and fractional positions go through the same rule.
        uv = tool.project(ctx.scene, [1.0, fixture.RASTER_Y, 10.0]); pixel = [math.floor(uv[0] + .5), math.floor(uv[1] + .5)]
        self.assertEqual(ctx.evaluate_pixel(pixel)['supportClass'], 'observed-nearby')

    def test_without_annotations_vehicle_is_declared_default_and_hidden_unknown(self):
        doc = self.document(self.grounded(annotations=False))
        v = doc['vehicle']
        self.assertEqual((v['status'], v['label'], v['setbackM']), ('declared-default', 'SUV', 6.096))
        self.assertIn('occupancy unknown', v['reason']); self.assertEqual(v['annotatedVehicleRejected'], ['Annotations not supplied.'])
        self.assertAlmostEqual(v['lateralFromCameraM'], 6 - .15 - .95, places=9)
        np.testing.assert_allclose(np.asarray(v['mappedPoint'])[[0, 2]], [-4.9, 20 - 6.096], atol=1e-9)
        np.testing.assert_allclose(np.asarray(doc['crossing']['mappedPoint'])[[0, 2]], [-4.9, 20], atol=1e-9)
        self.assertEqual([doc[k]['hidden'] for k in PLACED], [None, None, None])
        self.assertNotIn('topHiddenDiagnostic', doc['person'])

    def test_rejected_track_falls_back_and_occupied_default_is_unknown(self):
        paths = self.grounded(moving=True); doc = self.document(paths)
        self.assertEqual(doc['vehicle']['status'], 'declared-default')
        rejected = doc['vehicle']['annotatedVehicleRejected']
        self.assertTrue(any('not stationary' in f for f in rejected)); self.assertTrue(any('No averaged annotated pose' in f for f in rejected))
        scene, ground, terrain, positions, eligible, protocol, inputs = tool.load(**paths)
        blocked = copy.deepcopy(ground); car = copy.deepcopy(blocked['cuboids']['items'][0])
        car.update(track='cccccccc-0000-4000-8000-000000000003', center=[-4.9, -.85, 11.5]); blocked['cuboids']['items'].append(car)
        result = tool.suggest(scene, blocked, terrain, positions, eligible, protocol)['vehicle']
        self.assertEqual((result['status'], result['supportClass']), ('unknown', 'unknown')); self.assertIn('occupies the declared default footprint', result['reason'])

    def test_removing_lidar_support_never_invents_observation(self):
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**self.grounded())
        bare = copy.deepcopy(ground); bare['support']['rows'] = []
        objects = tool.suggest(scene, bare, terrain, positions, np.zeros_like(eligible), protocol)
        self.assertTrue(all(objects[k]['supportClass'] in ('map-only', 'unknown') for k in PLACED))
        no_terrain = tool.Terrain(np.full((terrain.window.shape), np.nan, np.float32), terrain.column0, terrain.row0, terrain.shape,
                                  ground['terrain']['sim2'], ground['frame']['cityOrigin'])
        objects = tool.suggest(scene, bare, no_terrain, positions, np.zeros_like(eligible), protocol)
        self.assertEqual([objects[k]['status'] for k in PLACED], ['unknown'] * 3)

    def test_split_drivable_polygons_do_not_create_a_curb(self):
        whole = self.document(self.grounded('a', annotations=False)); split = self.document(self.grounded('b', annotations=False, split=True))
        self.assertAlmostEqual(whole['vehicle']['lateralFromCameraM'], split['vehicle']['lateralFromCameraM'], places=12)
        self.assertEqual(whole['person']['pixelUV'], split['person']['pixelUV'])

    def test_deterministic_and_never_names_a_reason_for_stopping(self):
        paths = self.grounded()
        first, second = tool.canonical(self.document(paths)), tool.canonical(self.document(paths))
        self.assertEqual(first, second)
        self.assertEqual(first.lower().count(b'park'), 0)  # no form of the banned word, not even negated
        self.assertEqual(json.loads(first)['vehicle']['reason'], 'stationary vehicle (AV2 annotation); reason for stopping unknown')
        with mock.patch.object(tool, 'STATIONARY_REASON', 'stationary vehicle; Parked'), self.assertRaisesRegex(ValueError, 'banned'):
            self.document(paths)

    def test_crossing_error_slot(self):
        def error(entry, crossing='77'):
            return tool.crossing_error({'crossingError': {'perCrossing': {} if entry is None else {'77': entry}}}, crossing)
        self.assertEqual(error({'status': 'pending-measurement'}), PENDING)
        self.assertEqual(error(None), {**PENDING, 'status': 'unmeasured', 'label': 'paint offset unmeasured'})
        evidence = {'file': '../pao-paint/result.json', 'sha256': 'a' * 64}
        measured = error({'status': 'measured', 'plusMinusM': .231, 'evidence': evidence})
        self.assertEqual(measured, {'basis': 'mapped near edge', 'status': 'measured', 'label': '±0.24 m', 'plusMinusM': .24, 'evidence': evidence})
        self.assertEqual(error({'status': 'measured', 'plusMinusM': .3, 'evidence': evidence})['label'], '±0.30 m')  # exact cm not bumped
        self.assertEqual(error({'status': 'unresolved', 'evidence': evidence}),
                         {'basis': 'mapped near edge', 'status': 'unresolved', 'label': 'paint offset unresolved', 'plusMinusM': None, 'evidence': evidence})
        for bad in ({'status': 'measured', 'plusMinusM': -.1, 'evidence': evidence}, {'status': 'measured', 'plusMinusM': .2},
                    {'status': 'measured', 'plusMinusM': float('nan'), 'evidence': evidence}, {'status': 'unresolved'}, {'status': 'guessed'}):
            with self.assertRaises(ValueError):
                error(bad)
        # The crossing position never depends on the crossing error: a v3.1-style measured entry changes only crossingError.
        paths = self.grounded('m'); base = self.document(paths)
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**paths)
        filled = copy.deepcopy(protocol); filled['crossingError']['perCrossing']['77'] = {'status': 'measured', 'plusMinusM': .2, 'evidence': evidence}
        objects = tool.suggest(scene, ground, terrain, positions, eligible, filled)
        self.assertEqual(objects['crossing']['crossingError']['label'], '±0.20 m')
        self.assertEqual({k: v for k, v in objects['crossing'].items() if k != 'crossingError'},
                         {k: v for k, v in base['crossing'].items() if k != 'crossingError'})

    def test_measured_crossing_error_is_offset_plus_two_combined_spreads(self):
        # protocol v3.2: plusMinusM = |offsetM| + 2 x combinedM (the far end of the development-grade range), rounded up.
        evidence = {'file': '../../pao-paint/results/x.json', 'sha256': 'b' * 64}
        def error(offset, combined, value=None):
            value = abs(offset) + 2 * combined if value is None else value
            return tool.crossing_error({'crossingError': {'perCrossing': {'77': {'status': 'measured', 'plusMinusM': value, 'offsetM': offset,
                                                                                    'combinedM': combined, 'evidence': evidence}}}}, '77')
        self.assertEqual(error(0.04613553806463408, 0.06440719002038603)['label'], '±0.18 m')   # 8c52d911 / 2658849
        self.assertEqual(error(0.1801640186836544, 0.03893304951645912)['label'], '±0.26 m')    # c780d53a / 2069452
        self.assertEqual(error(-0.1, 0.05)['plusMinusM'], .2)                                   # the sign never shrinks the band
        for bad in (dict(value=0.04613553806463408), dict(combined=-0.01), dict(offset=float('nan'))):
            args = {'offset': 0.04613553806463408, 'combined': 0.06440719002038603, **bad}
            with self.assertRaises(ValueError):
                error(**args)
        # In a v3.2 document the measured entry changes only crossingError and adds its limitation line.
        protocol = protocol_v32(); entry = {'status': 'measured', 'offsetM': 0.1801640186836544, 'combinedM': 0.03893304951645912,
                                            'evidence': evidence}
        protocol['crossingError']['perCrossing']['77'] = {**entry, 'plusMinusM': 0.1801640186836544 + 2 * 0.03893304951645912}
        measured = self.document(self.grounded('v32m', protocol=protocol)); pending = self.document(self.grounded('v32p', protocol=protocol_v32()))
        self.assertEqual(measured['crossing']['crossingError']['label'], '±0.26 m')
        self.assertEqual({k: v for k, v in measured['crossing'].items() if k != 'crossingError'},
                         {k: v for k, v in pending['crossing'].items() if k != 'crossingError'})
        self.assertIn(tool.MEASURED_ERROR_LIMITATION, measured['limitations']); self.assertNotIn(tool.MEASURED_ERROR_LIMITATION, pending['limitations'])

    def test_v3_2_guard_leaves_an_applicable_crossing_unchanged(self):
        # The fixture crossing is square to the lane and ahead: v3.2 places exactly what v3 places. Only the road-axis reason
        # (skew is now a limit) and one limitation line differ.
        for annotations in (True, False):
            v32 = self.document(self.grounded(f'v32-{annotations}', protocol=protocol_v32(), annotations=annotations))
            v3 = self.document(self.grounded(f'v3-{annotations}', annotations=annotations))
            for key in ('crossing', 'vehicle', 'person', 'curb'):
                self.assertEqual(v32[key], v3[key], key)
            self.assertEqual({**v32['roadAxis'], 'reason': None}, {**v3['roadAxis'], 'reason': None})
            self.assertEqual((v32['roadAxis']['reason'], v3['roadAxis']['reason']), (tool.ROAD_AXIS_REASON_V32, tool.ROAD_AXIS_REASON))
            self.assertEqual(v32['limitations'], v3['limitations'][:-1] + [tool.APPLICABILITY_LIMITATION] + v3['limitations'][-1:])
            self.assertEqual(v32['crossing']['status'], 'mapped')

    def test_v3_2_refuses_a_crossing_the_model_does_not_fit(self):
        cases = {  # name: (edge1, edge2, refusal codes, text in the reason)
            'skewed 50 deg': ([(6, 20 + 6 * math.tan(math.radians(50))), (-6, 20 - 6 * math.tan(math.radians(50)))],
                              [(6, 24 + 6 * math.tan(math.radians(50))), (-6, 24 - 6 * math.tan(math.radians(50)))],
                              ['crossing-skew-over-limit'], 'is 50.0 degrees from perpendicular to the lane (limit 45)'),
            # 453e-like: a crossing almost along the lane, beside it; the lane chord meets its line far behind the camera.
            'along the lane': ([(8, 10), (8.5, 30)], [(10, 10), (10.5, 30)],
                               ['crossing-skew-over-limit', 'crossing-point-off-edge-segment', 'crossing-point-behind-camera'],
                               'm behind the camera (along the road axis)'),
            'meets behind': ([(24, 10), (36, 20)], [(24, 14), (36, 24)], ['crossing-point-off-edge-segment', 'crossing-point-behind-camera'],
                             'The ego lane chord meets the near-edge line 10.00 m behind the camera'),
            'short edge': ([(-3, 20), (-6, 20)], [(-3, 24), (-6, 24)], ['crossing-point-off-edge-segment'],
                           'The ego lane chord meets the near-edge line 3.00 m beyond the end of the mapped edge (3.00 m long).'),
            'edge behind': ([(6, -2), (-6, -2)], [(6, 24), (-6, 24)], ['crossing-edge-not-ahead', 'crossing-point-behind-camera'],
                            'Crossing edge1 midpoint is -2.00 m along the road axis, not ahead of the camera.'),
            # The lane chord meets the edge, but the stationary vehicle's centre line (x = -4) misses it by 1 m.
            'vehicle line': ([(6, 20), (-3, 20)], [(6, 24), (-3, 24)], ['crossing-point-off-edge-segment'],
                             'The current vehicle centre line meets the near-edge line 1.00 m beyond the end of the mapped edge (9.00 m long).'),
        }
        for name, (edge1, edge2, codes, text) in cases.items():
            with self.subTest(name):
                tag = name.replace(' ', '-')
                doc = self.document(self.grounded(tag, protocol=protocol_v32(), edit_map=crossing_edges(edge1, edge2)))
                c = doc['crossing']
                self.assertEqual(set(c), {'status', 'supportClass', 'nearEdge', 'reason', 'crossingError'})
                self.assertEqual((c['status'], c['supportClass'], c['nearEdge'], c['crossingError']), ('unknown', 'unknown', 'edge1', PENDING))
                self.assertTrue(c['reason'].startswith(tool.REFUSAL_PREFIX + ' to crossing 77: '), c['reason'])
                self.assertTrue(c['reason'].endswith(f'Refusal: {", ".join(codes)}.'), c['reason'])
                self.assertIn(text, c['reason'])
                for key in ('vehicle', 'person', 'curb'):  # nothing anchored on a refused crossing is placed
                    self.assertEqual(doc[key]['status'], 'unknown', key); self.assertNotIn('pixelUV', doc[key])
                    self.assertIn('crossing model does not apply here', doc[key]['reason'])
                self.assertEqual((doc['person']['label'], doc['curb']['label']), ('child', 'curb (map edge proxy)'))
                axis = doc['roadAxis']
                self.assertEqual((axis['status'], axis['nearEdge'], axis['reason']), ('mapped', 'edge1', tool.ROAD_AXIS_REASON_V32))
                self.assertTrue(math.isfinite(axis['skewDeg']))
                self.assertIn(tool.APPLICABILITY_LIMITATION, doc['limitations'])
                self.assertNotIn(b'park', tool.canonical(doc).lower())
        # The same maps under protocol v3 (no guard): an edge behind still aborts the run, the others are not refused.
        with self.assertRaisesRegex(ValueError, 'not ahead of the camera'):
            self.document(self.grounded('v3-edge-behind', edit_map=crossing_edges(*cases['edge behind'][:2])))
        old = self.document(self.grounded('v3-skewed', edit_map=crossing_edges(*cases['skewed 50 deg'][:2])))
        self.assertNotIn(tool.REFUSAL_PREFIX, tool.canonical(old).decode())
        self.assertAlmostEqual(old['roadAxis']['skewDeg'], 50, places=9)

    def test_v3_2_protocol_must_carry_the_exact_applicability_section(self):
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**self.grounded('v32', protocol=protocol_v32()))
        missing = {k: v for k, v in protocol.items() if k != 'crossingApplicability'}
        with self.assertRaisesRegex(ValueError, 'needs its crossingApplicability'):
            tool.suggest(scene, ground, terrain, positions, eligible, missing)
        for changed in (protocol_v32(maxSkewDeg=60.0)['crossingApplicability'],
                        {**protocol['crossingApplicability'], 'refusalCodes': list(reversed(tool.REFUSAL_CODES))}):
            with self.assertRaisesRegex(ValueError, 'exactly the parameters and refusal codes'):
                tool.suggest(scene, ground, terrain, positions, eligible, {**protocol, 'crossingApplicability': changed})

    def test_along_ray_limitation_names_the_band_method(self):
        fitted = tool.along_ray_limitation({'method': 'fitThisDevelopmentScene'})
        self.assertEqual(fitted, 'The along-ray uncertainty uses a height band fitted on this development scene\'s LiDAR ground returns; it '
                                 'describes the mapped-ground height error along the pixel ray, not the crossing position or a physical accuracy bound.')
        protocol = copy.deepcopy(fixture.PROTOCOL_V3)
        protocol['alongRayUncertainty']['heightBand'].update(method='fixed', heightBandMeters=.075, fittedOn='fixture development scenes')
        doc = self.document(self.grounded('fixed', protocol=protocol))
        self.assertEqual(doc['tolerances']['alongRayHeightBandMeters'], .075)
        self.assertFalse(any('fitted on this development scene' in t for t in doc['limitations']))
        self.assertIn('The along-ray uncertainty uses a fixed height band fitted on development scenes (fixture development scenes); this '
                      'scene\'s LiDAR ground returns only test it (heldOutResiduals); it describes the mapped-ground height error along the '
                      'pixel ray, not the crossing position or a physical accuracy bound.', doc['limitations'])
        self.assertIn(fitted, self.document(self.grounded('fitted'))['limitations'])
        with self.assertRaisesRegex(ValueError, 'Unknown height band method'):
            tool.along_ray_limitation({'method': 'guessed'})

    def test_protocol_v1_v2_and_schema_1_ground_are_refused(self):
        v2_paths = self.grounded('v2', protocol=fixture.PROTOCOL)  # a v2 protocol builds a schema-1 ground layer
        with self.assertRaisesRegex(ValueError, 'Protocol v3 pixelSelection'):
            tool.document(*tool.load(**v2_paths))
        scene, ground, terrain, positions, eligible, _, _ = tool.load(**v2_paths)
        with self.assertRaisesRegex(ValueError, 'schemaVersion 2'):
            tool.Context(scene, ground, terrain, positions, eligible, fixture.PROTOCOL_V3)
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**self.grounded('v3'))
        for change in ({'protocolVersion': 2}, {'pixelSelection': {'rule': 'nearestHitWindow', 'windowPx': 3}}):
            with self.assertRaisesRegex(ValueError, 'Protocol v3'):
                tool.suggest(scene, ground, terrain, positions, eligible, {**protocol, **change})
        with self.assertRaisesRegex(ValueError, 'Protocol v3 ground layer'):
            tool.suggest(scene, ground, terrain, positions, eligible, {**protocol, 'placementRange': {'limitMeters': 25.0}})

    def test_protocol_v3_1_records_the_declared_replacement_front(self):
        # protocol v3.1 currentVehicle.declaredReplacement: a declared vehicle in place of the stationary one keeps its
        # front point (mappedPoint). The file says so in appModel and limitations; protocol v3 output is unchanged.
        protocol = copy.deepcopy(fixture.PROTOCOL_V3); protocol['currentVehicle']['declaredReplacement'] = {'rule': 'fixture'}
        doc = self.document(self.grounded('v31', protocol=protocol)); old = self.document(self.grounded('v3'))
        self.assertEqual(doc['vehicle']['appModel'], tool.APP_MODEL_REPLACEMENT); self.assertEqual(old['vehicle']['appModel'], tool.APP_MODEL)
        self.assertIn(tool.REPLACEMENT_LIMITATION, doc['limitations']); self.assertNotIn(tool.REPLACEMENT_LIMITATION, old['limitations'])
        self.assertEqual(doc['limitations'][-1], old['limitations'][-1])
        for key in ('crossing', 'person', 'roadAxis', 'curb'):
            self.assertEqual(doc[key], old[key], key)
        self.assertEqual({k: v for k, v in doc['vehicle'].items() if k != 'appModel'}, {k: v for k, v in old['vehicle'].items() if k != 'appModel'})
        # A declared default vehicle (no annotated vehicle to replace) gets no replacement limitation.
        default = self.document(self.grounded('v31-default', protocol=protocol, annotations=False))
        self.assertEqual(default['vehicle']['status'], 'declared-default'); self.assertNotIn(tool.REPLACEMENT_LIMITATION, default['limitations'])
        self.assertNotIn(b'park', tool.canonical(doc).lower())

    def test_vehicle_yaw_is_signed_toward_the_right(self):
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**self.grounded())
        turned = copy.deepcopy(ground); car = next(c for c in turned['cuboids']['items'] if c['track'] == fixture.TRACK)
        a = math.radians(2); s, c = math.sin(a), math.cos(a)
        car['worldFromBox'] = [[-s, c, 0.], [0., 0., 1.], [c, s, 0.]]  # forward turned 2 degrees toward world -X (right of +Z travel)
        v = tool.suggest(scene, turned, terrain, positions, eligible, protocol)['vehicle']
        self.assertEqual(v['status'], 'annotated')
        self.assertAlmostEqual(v['yawToRoadAxisDeg'], 2, places=9); self.assertAlmostEqual(v['headingToRoadAxisDeg'], 2, places=6)
        car['worldFromBox'] = [[s, c, 0.], [0., 0., 1.], [c, -s, 0.]]
        self.assertAlmostEqual(tool.suggest(scene, turned, terrain, positions, eligible, protocol)['vehicle']['yawToRoadAxisDeg'], -2, places=9)

    def test_cli_writes_exclusively(self):
        paths = self.grounded(); out = paths['grounded']
        command = [sys.executable, '-I', '-B', str(SCRIPT), '--grounded', str(out), '--metadata-sha256', paths['metadata_sha256'],
                   '--buffers', str(paths['buffers_dir']), '--protocol', str(paths['protocol_path']), '--protocol-sha256', paths['protocol_sha256']]
        env = dict(os.environ, SOURCE_DATE_EPOCH=EPOCH)
        done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        data = (out / 'suggestions.json').read_bytes(); self.assertEqual(done.stdout.strip(), sha(data))
        entry = json.loads((out / 'catalog-entry.json').read_bytes())
        self.assertEqual(entry, {'id': 'fixture-scene', 'label': 'fixture-scene', 'metadataSha256': paths['metadata_sha256'],
                                 'ground': {'file': 'ground.json', 'sha256': sha((out / 'ground.json').read_bytes())},
                                 'suggestions': {'file': 'suggestions.json', 'sha256': sha(data)}})
        again = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(again.returncode, 1); self.assertIn('Existing suggestions.json retained', again.stderr)
        self.assertEqual((out / 'suggestions.json').read_bytes(), data)
        other = self.grounded('b'); bad_command = [str(other['grounded']) if part == str(out) else part for part in command]
        bad = subprocess.run(bad_command[:-1] + ['0' * 64], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(bad.returncode, 1); self.assertIn('SHA256 mismatch', bad.stderr)
        self.assertFalse((other['grounded'] / 'suggestions.json').exists()); self.assertFalse((other['grounded'] / 'catalog-entry.json').exists())


REAL = ROOT / '.local/evaluation/av2-map-first-2026-10-11'
V2 = REAL / 'grounded-v2'
V3 = REAL / 'grounded-v3'
V31 = REAL / 'grounded-v3.1'


def real_paths(folder, freeze, protocol):
    return dict(grounded=REAL / folder, metadata_sha256=sha((REAL / folder / 'scene.json').read_bytes()), buffers_dir=REAL / 'assembled',
                protocol_path=REAL / protocol, protocol_sha256=json.loads((REAL / freeze).read_bytes())['sha256'])


def epoch_of(doc):
    from datetime import datetime
    return str(int(datetime.strptime(doc['generatedUtc'], '%Y-%m-%dT%H:%M:%S%z').timestamp()))


@unittest.skipUnless((V31 / 'suggestions.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c v3.1 files absent')
class RealV31SuggestionTests(unittest.TestCase):
    """grounded-v3.1 (protocol v3.1) is current; it reproduces byte for byte from the current script."""

    def test_real_v3_1_suggestions_reproduce_and_match_expected_values(self):
        stored = (V31 / 'suggestions.json').read_bytes(); doc = json.loads(stored)
        paths = real_paths('grounded-v3.1', 'protocol-v3.1-freeze.json', 'protocol-v3.1.json')
        self.assertEqual(sha((REAL / 'protocol-v3.1.json').read_bytes()), paths['protocol_sha256'])
        self.assertEqual(paths['protocol_sha256'], '6a7d3369e75dcb9eb5afbd46a66f94760245fe3b0e34733e0f60a106a12f2d92')
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': epoch_of(doc)}):
            rebuilt = json.loads(tool.canonical(tool.document(*tool.load(**paths))))
        # The NaN-height outline fix (shared ground script, stored 9dd444a9...) and the protocol v3.2 guard (this script, stored
        # 918e2bf7...) changed only the two generator hashes: a protocol without crossingApplicability runs unchanged.
        stored_hashes = ('918e2bf7135dd5483d08c0e1b6ee9c53a6c9ca7247ef10bded1b15efcfdb0ef9',
                         '9dd444a9d9b2914b1b2920115353a88ab0a578bb105b103b4c66724ffdd6f84d')
        self.assertEqual((doc['generator']['sha256'], doc['generator']['sharedSha256']), stored_hashes)
        self.assertEqual((rebuilt['generator']['sha256'], rebuilt['generator']['sharedSha256']),
                         (sha(SCRIPT.read_bytes()), sha((ROOT / doc['generator']['sharedScript']).read_bytes())))
        rebuilt['generator']['sha256'], rebuilt['generator']['sharedSha256'] = stored_hashes
        self.assertEqual(tool.canonical(rebuilt), stored)
        self.assertEqual((doc['schemaVersion'], doc['generatedUtc']), (3, '2026-10-12T11:51:45Z'))
        self.assertEqual(doc['tolerances'], {'roadToleranceMeters': .08, 'supportHeightAgreementMeters': .08, 'supportRadiusMeters': .5,
                                             'maxRangeMeters': 60.0, 'placementRangeLimitMeters': 30.0, 'alongRayHeightBandMeters': .075})
        v, c, p = doc['vehicle'], doc['crossing'], doc['person']
        self.assertEqual((v['status'], v['track'][:8], v['label'], v['reason'], v['pose']),
                         ('annotated', '396f15b5', 'stationary vehicle', 'stationary vehicle (AV2 annotation); reason for stopping unknown',
                          {'kind': 'averaged', 'annotations': 114}))
        self.assertEqual(v['appModel'], tool.APP_MODEL_REPLACEMENT); self.assertIn(tool.REPLACEMENT_LIMITATION, doc['limitations'])
        self.assertEqual([doc[k]['pixel'] for k in PLACED], [[371, 189], [374, 191], [484, 185]])
        for k, uv in zip(PLACED, ([371.14534464547864, 188.52832143204816], [374.45809937658737, 190.84864102416427], [484.3272234524834, 185.2703140345646])):
            np.testing.assert_allclose(doc[k]['pixelUV'], uv, atol=1e-9)
            self.assertEqual(doc[k]['pixel'], [math.floor(doc[k]['pixelUV'][0] + .5), math.floor(doc[k]['pixelUV'][1] + .5)])
        # Metric C (internal consistency, not accuracy): crossing and person reproduce their mapped points (target < 0.001 m;
        # v2: 0.144 and 0.210 m). The vehicle front's pixel ray lies in a raster shadow: it meets a cell 7.8 cm higher 1.03 m
        # sooner, so the stored hit misses the target (<= 0.05 m); protocol v3.1 uses the averaged front point instead.
        self.assertLess(c['pixelOffsetFromMappedM'], 1e-9); self.assertLess(p['pixelOffsetFromMappedM'], 1e-9)
        self.assertAlmostEqual(v['pixelOffsetFromMappedM'], 1.0285, places=4)
        self.assertAlmostEqual(v['groundPoint'][1] - v['mappedPoint'][1], .078125, places=9)
        for k, residual in zip(PLACED, (.9764, 1.0658, .3061)):
            self.assertAlmostEqual(doc[k]['integerPixelCheck']['residualM'], residual, places=4)
        for k, along in zip(PLACED, (1.0259, .9902, 1.0888)):
            self.assertAlmostEqual(doc[k]['alongRayUncertaintyM'], along, places=4)
        self.assertEqual([doc[k]['supportClass'] for k in PLACED], ['map-only'] * 3)
        self.assertEqual([doc[k]['hidden'] for k in PLACED], [True, True, False]); self.assertFalse(p['topHiddenDiagnostic'])
        self.assertTrue(all(doc[k]['horizontalRangeM'] <= 30 for k in PLACED))
        self.assertEqual(c['crossingError'], PENDING); self.assertAlmostEqual(c['forwardM'], 17.4435, places=4)
        self.assertAlmostEqual(v['frontGapToCrossingM'], .4199, places=4); self.assertAlmostEqual(v['yawToRoadAxisDeg'], 1.4121, places=4)
        self.assertAlmostEqual(v['pixelGroundGapAlongAxisM'], 1.4313, places=4); self.assertEqual(v['dimensions'], {'width': 1.88, 'height': 1.49, 'length': 4.03})
        # Same objects as grounded-v3: only the protocol, generator, scene and ground pins, the time, appModel and one
        # limitation differ (protocol v3.1 changes no stored geometry).
        old = json.loads((V3 / 'suggestions.json').read_bytes())
        for key in ('crossing', 'person', 'roadAxis', 'curb', 'tolerances', 'license', 'sceneId'):
            self.assertEqual(doc[key], old[key], key)
        self.assertEqual({k: x for k, x in v.items() if k != 'appModel'}, {k: x for k, x in old['vehicle'].items() if k != 'appModel'})
        entry = json.loads((V31 / 'catalog-entry.json').read_bytes())
        self.assertEqual(entry['suggestions']['sha256'], sha(stored)); self.assertEqual(entry['metadataSha256'], paths['metadata_sha256'])
        command = [sys.executable, '-I', '-B', str(SCRIPT), '--grounded', str(V31), '--metadata-sha256', paths['metadata_sha256'],
                   '--buffers', str(REAL / 'assembled'), '--protocol', str(REAL / 'protocol-v3.1.json'), '--protocol-sha256', paths['protocol_sha256']]
        refused = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(refused.returncode, 1); self.assertIn('Existing suggestions.json retained', refused.stderr)
        self.assertEqual((V31 / 'suggestions.json').read_bytes(), stored)

    def test_no_form_of_the_banned_word_in_v3_1_outputs(self):
        for path in (*sorted(V31.iterdir()), REAL / 'protocol-v3.1.json', REAL / 'protocol-v3.1-freeze.json', REAL / 'runtime-test-vectors-v3.1.json'):
            self.assertNotIn(b'park', path.read_bytes().lower(), path.name)


V32 = REAL / 'grounded-v3.2'
GUARD = ROOT / '.local/evaluation/pao-accuracy/scene-builder-v4.1/453e5558-20499018-guard-check'
GUARD_ASSEMBLED = ROOT / '.local/evaluation/av2-453e-2026-10-12/assembled'
GUARD_PROTOCOL = 'protocol-v3.2-453e5558-20499018-guard-check'


@unittest.skipUnless((V32 / 'suggestions.json').exists() and (V31 / 'suggestions.json').exists() and (REAL / 'assembled/scene.json').exists(),
                     'real 8c v3.2 files absent')
class RealV32SuggestionTests(unittest.TestCase):
    """grounded-v3.2 (protocol v3.2, development 8c): same geometry as grounded-v3.1; only the crossing error (paint v4
    measurement), the pins, the generator hashes, the road-axis reason and two limitation lines differ."""

    def test_real_v3_2_suggestions_reproduce_from_the_current_script(self):
        stored = (V32 / 'suggestions.json').read_bytes(); doc = json.loads(stored)
        paths = real_paths('grounded-v3.2', 'protocol-v3.2-freeze.json', 'protocol-v3.2.json')
        self.assertEqual(paths['protocol_sha256'], '1cef13e625ba8d0c0c00203c0bda6e256536173c4607ac79807e3e40dd173cf7')
        self.assertEqual(sha((REAL / 'protocol-v3.2.json').read_bytes()), paths['protocol_sha256'])
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': epoch_of(doc)}):
            rebuilt = tool.canonical(tool.document(*tool.load(**paths)))
        self.assertEqual(rebuilt, stored)
        self.assertEqual(json.loads((V32 / 'catalog-entry.json').read_bytes())['suggestions']['sha256'], sha(stored))

    def test_v3_2_differs_from_v3_1_only_in_the_allowed_fields(self):
        old = json.loads((V31 / 'suggestions.json').read_bytes()); new = json.loads((V32 / 'suggestions.json').read_bytes())
        error = new['crossing']['crossingError']
        self.assertEqual({k: x for k, x in error.items() if k != 'evidence'},
                         {'basis': 'mapped near edge', 'status': 'measured', 'label': '±0.18 m', 'plusMinusM': .18})
        evidence = ROOT / error['evidence']['file']
        if evidence.exists():
            self.assertEqual(sha(evidence.read_bytes()), error['evidence']['sha256'])
        self.assertEqual({k: x for k, x in new['crossing'].items() if k != 'crossingError'}, {k: x for k, x in old['crossing'].items() if k != 'crossingError'})
        self.assertEqual(old['crossing']['crossingError'], PENDING)
        for key in ('vehicle', 'person', 'curb', 'tolerances', 'license', 'sceneId', 'generatedUtc', 'schemaVersion'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual({k: x for k, x in new['roadAxis'].items() if k != 'reason'}, {k: x for k, x in old['roadAxis'].items() if k != 'reason'})
        self.assertIn('refuses the crossing model (protocol v3.2)', new['roadAxis']['reason'])
        self.assertEqual([x for x in new['limitations'] if x in old['limitations']], old['limitations'])
        self.assertEqual(len(new['limitations']), len(old['limitations']) + 2)
        changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        self.assertEqual(changed, {'crossing', 'generator', 'inputs', 'limitations', 'protocolSha256', 'roadAxis'})
        self.assertEqual({k for k in new['inputs'] if new['inputs'][k] != old['inputs'][k]}, {'scene.json', 'ground.json', 'protocol.json'})
        # Same terrain bytes: the NaN map-height fix changes no stored height on 8c.
        self.assertEqual((V32 / 'terrain.bin').read_bytes(), (V31 / 'terrain.bin').read_bytes())

    def test_no_form_of_the_banned_word_in_v3_2_outputs(self):
        for path in (*sorted(V32.iterdir()), REAL / 'protocol-v3.2.json', REAL / 'protocol-v3.2-freeze.json'):
            self.assertNotIn(b'park', path.read_bytes().lower(), path.name)


@unittest.skipUnless((GUARD / 'grounded/suggestions.json').exists() and (GUARD_ASSEMBLED / 'scene.json').exists(), 'real 453e guard-check files absent')
class Real453eGuardCheckTests(unittest.TestCase):
    """453e5558 / crossing 20499018 (development): the mapped crossing runs along the road, so protocol v3.2 refuses the
    crossing model and every object anchored on it stays Unknown with the named refusal."""

    def test_guard_check_refuses_and_reproduces(self):
        stored = (GUARD / 'grounded/suggestions.json').read_bytes(); doc = json.loads(stored)
        freeze = json.loads((GUARD / f'{GUARD_PROTOCOL}-freeze.json').read_bytes())
        self.assertEqual(sha((GUARD / f'{GUARD_PROTOCOL}.json').read_bytes()), freeze['sha256'])
        paths = dict(grounded=GUARD / 'grounded', metadata_sha256=sha((GUARD / 'grounded/scene.json').read_bytes()), buffers_dir=GUARD_ASSEMBLED,
                     protocol_path=GUARD / f'{GUARD_PROTOCOL}.json', protocol_sha256=freeze['sha256'])
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': epoch_of(doc)}):
            self.assertEqual(tool.canonical(tool.document(*tool.load(**paths))), stored)
        crossing = doc['crossing']
        self.assertEqual((crossing['status'], crossing['supportClass']), ('unknown', 'unknown'))
        self.assertTrue(crossing['reason'].startswith('Crossing model does not apply to crossing 20499018: '))
        self.assertTrue(crossing['reason'].endswith('Refusal: crossing-skew-over-limit, crossing-point-off-edge-segment.'))
        self.assertNotIn('behind the camera', crossing['reason'])
        for key in ('vehicle', 'person', 'curb'):
            self.assertEqual(doc[key]['status'], 'unknown', key)
            self.assertIn('crossing model does not apply here', doc[key]['reason'], key)
        self.assertEqual((doc['roadAxis']['laneId'], doc['roadAxis']['nearEdge'], round(doc['roadAxis']['skewDeg'], 3)), ('326068517', 'edge1', 89.996))
        self.assertNotIn(b'park', stored.lower())


@unittest.skipUnless((V3 / 'suggestions.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c v3 files absent')
class RealV3HistoricalSuggestionTests(unittest.TestCase):

    """grounded-v3 (protocol v3) is history: pinned by hash and written by the previous version of the script (sha256 5f2d9c7a...).
    The current script on protocol v3 reproduces its content except the two generator hashes."""

    def test_v3_is_pinned_and_differs_from_the_current_script_only_in_its_generator(self):
        stored = (V3 / 'suggestions.json').read_bytes(); doc = json.loads(stored)
        self.assertEqual(sha(stored), '56aafdc7d166538861e36b173c1197137879a9ceab82b6630e5cd64ffe31b73c')
        self.assertEqual(doc['generator']['sha256'], '5f2d9c7a30aa736e51d39b9a816abdd479885ceb519ad30e08c6422e32e704a5')
        paths = real_paths('grounded-v3', 'protocol-v3-freeze.json', 'protocol-v3.json')
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': epoch_of(doc)}):
            now = tool.document(*tool.load(**paths))
        self.assertEqual({k: x for k, x in now['generator'].items() if not k.endswith('ha256')}, {k: x for k, x in doc['generator'].items() if not k.endswith('ha256')})
        now['generator'] = doc['generator']
        self.assertEqual(tool.canonical(now), stored)

    def test_no_form_of_the_banned_word_in_v3_outputs(self):
        for path in (*sorted(V3.iterdir()), REAL / 'protocol-v3.json', REAL / 'protocol-v3-freeze.json'):
            self.assertNotIn(b'park', path.read_bytes().lower(), path.name)


@unittest.skipUnless((V2 / 'suggestions.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c v2 files absent')
class RealHistoricalSuggestionTests(unittest.TestCase):
    """grounded-v2 (protocol v2) and grounded/ (protocol v1) are history: pinned by hash, written by earlier versions of the
    script (v2: sha256 955d6b1e..., commit ae1c442) and refused by the current one."""

    def test_v2_is_pinned_and_refused(self):
        stored = (V2 / 'suggestions.json').read_bytes(); doc = json.loads(stored)
        self.assertEqual(sha(stored), '94b294f3b7f7abd1e9fc58994e8e0a72f277b59c5e6490e3596157d05533d248')
        self.assertEqual(doc['generator']['sha256'], '955d6b1ed7c81da05a6a96f0d3b34bb8c787724979a043e7a71494eadc96ccf1')
        self.assertEqual([doc[k]['pixel'] for k in PLACED], [[373, 188], [377, 188], [486, 186]])
        for k, expected in zip(PLACED, (.1437, .1767, .2096)):
            self.assertAlmostEqual(doc[k]['pixelOffsetFromMappedM'], expected, places=4)
        paths = real_paths('grounded-v2', 'protocol-v2-freeze.json', 'protocol-v2.json')
        with self.assertRaisesRegex(ValueError, 'Protocol v3 pixelSelection'):
            tool.document(*tool.load(**paths))

    def test_no_form_of_the_banned_word_in_v2_outputs(self):
        for path in (V2 / 'suggestions.json', V2 / 'ground.json', V2 / 'scene.json', V2 / 'catalog-entry.json',
                     REAL / 'protocol-v2.json', REAL / 'runtime-test-vectors-v2.json'):
            if path.exists():
                self.assertNotIn(b'park', path.read_bytes().lower(), path.name)

    @unittest.skipUnless((REAL / 'grounded/suggestions.json').exists(), 'historical v1 output absent')
    def test_v1_protocol_is_refused(self):
        paths = real_paths('grounded', 'protocol-freeze.json', 'protocol.json')
        with self.assertRaisesRegex(ValueError, 'pixelSelection'):
            tool.document(*tool.load(**paths))


if __name__ == '__main__':
    unittest.main()

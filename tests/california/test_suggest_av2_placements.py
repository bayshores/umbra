"""Synthetic analytic fixtures for AV2 map-first suggestions (schema v2), plus a real-data check skipped when files are absent."""
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


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SuggestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def grounded(self, name='a', annotations=True, **options):
        args = fixture.write_fixture(self.root / name, **options)
        if not annotations:
            args.update(annotations_path=None, annotations_sha256=None)
        fixture.tool.build(**args)
        out = args['output']
        return dict(grounded=out, metadata_sha256=sha((out / 'scene.json').read_bytes()), buffers_dir=args['scene_path'].parent,
                    protocol_path=args['protocol_path'], protocol_sha256=args['protocol_sha256'])

    def document(self, paths):
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': EPOCH}):
            return tool.document(*tool.load(**paths))

    def test_annotated_stationary_vehicle_sets_crossing_and_hides_points(self):
        doc = self.document(self.grounded())
        v, c, p = doc['vehicle'], doc['crossing'], doc['person']
        self.assertEqual((doc['schemaVersion'], doc['sourceClass'], doc['generatedUtc']), (2, 'mappedGround', '2026-10-12T01:28:00Z'))
        self.assertEqual((v['status'], v['label'], v['reason']), ('annotated', 'stationary vehicle', tool.STATIONARY_REASON))
        self.assertEqual(v['dimensions'], {'width': 1.8, 'height': 1.5, 'length': 4.0}); self.assertEqual(v['track'], fixture.TRACK)
        self.assertAlmostEqual(v['frontGapToCrossingM'], .5, places=9); self.assertAlmostEqual(v['headingToRoadAxisDeg'], 0, places=6)
        np.testing.assert_allclose(np.asarray(v['mappedPoint'])[[0, 2]], [-4, 19.5], atol=1e-9)
        self.assertEqual((c['status'], c['nearEdge'], c['paintOffset']), ('mapped', 'edge1', {'status': 'unmeasured', 'label': 'paint offset unmeasured'}))
        np.testing.assert_allclose(np.asarray(c['mappedPoint'])[[0, 2]], [-4, 20], atol=1e-9)
        np.testing.assert_allclose(np.asarray(p['mappedPoint'])[[0, 2]], [-5.7, 20], atol=1e-9)
        self.assertEqual((p['status'], p['label'], p['heightM']), ('declared-default', 'child', 1.1))
        # Analytic: each camera ray to these ground points passes through the car box (X -4.9..-3.1, Z 15.5..19.5, Y -1.6..-0.1).
        self.assertEqual([doc[k]['hidden'] for k in PLACED], [True, True, True])
        self.assertEqual([doc[k]['supportClass'] for k in PLACED], ['map-only'] * 3)
        self.assertEqual({doc[k]['supportLabel'] for k in PLACED}, {'mapped, not observed'})
        for k in PLACED:
            g = np.asarray(doc[k]['groundPoint']); self.assertAlmostEqual(g[1], float(np.float16(8.4)) - 10, places=9)  # float16 raster
            self.assertLess(doc[k]['pixelOffsetFromMappedM'], .35)  # flat ground: only pixel rounding moves the point
            self.assertGreater(doc[k]['alongRayUncertaintyM'], 0)
            search = doc[k]['pixelSearch']
            self.assertEqual(doc[k]['pixelOffsetFromMappedM'], search['residualM']); self.assertLessEqual(search['residualM'], search['roundedResidualM'])
            self.assertEqual(doc[k]['pixel'], [r + o for r, o in zip(search['roundedPixel'], search['offsetPx'])])
            self.assertEqual((search['windowPx'], search['candidatesInCrop'], search['candidatesPlaced']), (3, 49, 49))
        self.assertEqual(v['appModel'], tool.APP_MODEL); self.assertAlmostEqual(v['yawToRoadAxisDeg'], 0, places=6)
        r = v['frontResidualToAnnotatedM']
        self.assertEqual(r['horizontal'], v['pixelOffsetFromMappedM']); self.assertAlmostEqual(math.hypot(r['alongRoadAxis'], r['lateralRight']), r['horizontal'], places=12)
        a = doc['roadAxis']; np.testing.assert_allclose(a['directionOfTravel'], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(a['skewDeg'], 0, places=6); self.assertTrue(a['cameraInLane'])
        self.assertAlmostEqual(doc['curb']['nearEdgeLateralM'], 6.0, places=9); self.assertEqual(doc['curb']['label'], 'curb (map edge proxy)')
        self.assertEqual(doc['license']['id'], 'CC-BY-NC-SA-4.0'); self.assertEqual(doc['license']['attribution'], '(c) 2021 Argo AI, LLC')

    def test_support_classes_follow_the_runtime_pixel_rule(self):
        loaded = tool.load(**self.grounded()); ctx = tool.Context(*loaded[:6])
        def at(x, z):
            uv = tool.project(loaded[0], [x, fixture.GROUND_Y, z]); return ctx.evaluate_pixel([int(np.floor(uv[0] + .5)), int(np.floor(uv[1] + .5))])
        near = at(1.0, 10.0)
        self.assertEqual(near['supportClass'], 'observed-nearby'); self.assertFalse(near['hidden'])
        # All returns are tested: one agreeing and one 0.6 m above the map lie within 0.5 m.
        self.assertEqual({k: near['supportEvidence'][k] for k in ('cameraVisibleWithinRadius', 'agreeingWithinRadius', 'aboveToleranceWithinRadius')},
                         {'cameraVisibleWithinRadius': 2, 'agreeingWithinRadius': 1, 'aboveToleranceWithinRadius': 1})
        self.assertEqual(at(1.0, 13.0)['supportClass'], 'map-only')
        off = at(8.0, 20.0); self.assertEqual(off['supportClass'], 'unknown'); self.assertIn('outside the mapped drivable area', off['reason'])
        nan = at(3.0, 40.0); self.assertEqual(nan['supportClass'], 'unknown'); self.assertIn('without mapped ground height', nan['reason'])
        behind = at(-4.0, 21.0)  # an agreeing return exists here but the camera sees the car, so it is not support
        self.assertEqual((behind['supportClass'], behind['hidden']), ('map-only', True))

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
        self.assertTrue(any('not stationary' in f for f in doc['vehicle']['annotatedVehicleRejected']))
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
        bare['support']['rows'] = []
        objects = tool.suggest(scene, bare, no_terrain, positions, np.zeros_like(eligible), protocol)
        self.assertEqual([objects[k]['status'] for k in PLACED], ['unknown'] * 3)

    def test_split_drivable_polygons_do_not_create_a_curb(self):
        whole = self.document(self.grounded('a', annotations=False)); split = self.document(self.grounded('b', annotations=False, split=True))
        self.assertAlmostEqual(whole['vehicle']['lateralFromCameraM'], split['vehicle']['lateralFromCameraM'], places=12)
        self.assertEqual(whole['person']['pixel'], split['person']['pixel'])

    def test_deterministic_and_never_names_a_reason_for_stopping(self):
        paths = self.grounded()
        first, second = tool.canonical(self.document(paths)), tool.canonical(self.document(paths))
        self.assertEqual(first, second)
        self.assertEqual(first.lower().count(b'park'), 0)  # no form of the banned word, not even negated
        self.assertEqual(json.loads(first)['vehicle']['reason'], 'stationary vehicle (AV2 annotation); reason for stopping unknown')
        with mock.patch.object(tool, 'STATIONARY_REASON', 'stationary vehicle; Parked'), self.assertRaisesRegex(ValueError, 'banned'):
            self.document(paths)

    # ------------------------------------------------------------ protocol v2 pixel search
    def context(self, terrain=None, window=None, **options):
        name = f'context{len(list(self.root.iterdir()))}'
        scene, ground, base_terrain, positions, eligible, protocol, _ = tool.load(**self.grounded(name, **options))
        if window is not None:
            protocol = copy.deepcopy(protocol); protocol['pixelSelection']['windowPx'] = window
        return tool.Context(scene, ground, base_terrain if terrain is None else terrain(base_terrain, ground), positions, eligible, protocol)

    def brute_force(self, ctx, xz, uv, window):
        """Independent re-statement of the rule: evaluate every window pixel with the full runtime rule."""
        rounded = [math.floor(uv[0] + .5), math.floor(uv[1] + .5)]; best = None
        for dv in range(-window, window + 1):
            for du in range(-window, window + 1):
                pixel = [rounded[0] + du, rounded[1] + dv]
                if not ctx.in_crop(pixel):
                    continue
                result = ctx.evaluate_pixel(pixel)
                if result['supportClass'] == 'unknown':
                    continue
                g = np.asarray(result['groundPoint']); key = (float(np.hypot(*(g[[0, 2]] - xz))), math.hypot(pixel[0] - uv[0], pixel[1] - uv[1]), pixel[1], pixel[0])
                best = min(best, (key, pixel)) if best else (key, pixel)
        return best

    @staticmethod
    def raised_band(terrain, ground):
        """Raster rows 169-171 (world Z 18.4-19.3) 5 cm higher: rays to targets just behind the band hit its top instead."""
        w = terrain.window.copy(); w[169 - terrain.row0:172 - terrain.row0, :] += np.float32(.05)
        return tool.Terrain(w, terrain.column0, terrain.row0, terrain.shape, ground['terrain']['sim2'], ground['frame']['cityOrigin'])

    def test_pixel_search_is_never_worse_than_rounding_and_matches_brute_force(self):
        ctx = self.context(terrain=self.raised_band)
        for z in (17.4, 18.1, 19.4, 19.6, 20.0, 20.2):
            xz = np.array([1.0, z]); uv = tool.project(ctx.scene, [1.0, ctx.height(xz), z])
            pixel, record = ctx.search_pixel(xz, uv)
            self.assertLessEqual(record['residualM'], record['roundedResidualM'])
            self.assertLessEqual(max(map(abs, record['offsetPx'])), 3)
            key, expected = self.brute_force(ctx, xz, uv, 3)
            self.assertEqual(pixel, expected); self.assertAlmostEqual(record['residualM'], key[0], places=12)
        # Just behind the band plain rounding lands on the band top about 0.66 m short; a pixel two rows up gets within 0.12 m.
        xz = np.array([1.0, 19.4]); pixel, record = ctx.search_pixel(xz, tool.project(ctx.scene, [1.0, ctx.height(xz), 19.4]))
        self.assertGreater(record['roundedResidualM'], .6); self.assertLess(record['residualM'], .15); self.assertEqual(record['offsetPx'], [0, -2])

    def test_pixel_search_is_deterministic_and_breaks_ties_by_projection_then_v_then_u(self):
        ctx = self.context(terrain=self.raised_band)
        xz = np.array([1.0, 20.0]); uv = tool.project(ctx.scene, [1.0, ctx.height(xz), 20.0])
        self.assertEqual(ctx.search_pixel(xz, uv), ctx.search_pixel(xz, uv))
        flat = self.context()
        flat.ground_hit = lambda pixel: (None, {'point': [5., -1.6, 5.]}, None)  # every pixel reaches the same point: all residuals tie
        self.assertEqual(flat.search_pixel([1., 1.], [100.2, 50.7])[0], [100, 51])  # nearest the projection
        self.assertEqual(flat.search_pixel([1., 1.], [100.5, 50.5])[0], [100, 50])  # four-way tie: smaller v, then smaller u

    def test_pixel_search_window_bounds_the_choice(self):
        ctx = self.context()
        # 0.1 m outside the drivable edge (X = 6): rounding is Unknown, a pixel inside the window reaches drivable ground.
        near = ctx.place(np.array([6.1, 20.0]), {'status': 'mapped'})
        self.assertEqual(near['status'], 'mapped'); s = near['pixelSearch']
        self.assertIsNone(s['roundedResidualM']); self.assertIn('outside the mapped drivable area', s['roundedReason'])
        self.assertTrue(0 < max(map(abs, s['offsetPx'])) <= 3); self.assertLess(s['residualM'], .2)
        self.assertLess(near['groundPoint'][0], 6.0)
        # 1 m outside: drivable ground is about 40 px away, beyond the window, so the object stays Unknown.
        far = ctx.place(np.array([7.0, 20.0]), {'status': 'mapped'})
        self.assertEqual((far['status'], far['supportClass'], far['pixelSearch']['candidatesPlaced']), ('unknown', 'unknown', 0))
        self.assertIn('No pixel within 3 px', far['reason'])
        inside = tool.project(ctx.scene, [5.9, fixture.GROUND_Y, 20.0]); inside = [math.floor(inside[0] + .5), math.floor(inside[1] + .5)]
        self.assertNotEqual(ctx.evaluate_pixel(inside)['supportClass'], 'unknown')
        self.assertGreater(max(abs(a - b) for a, b in zip(inside, far['pixel'])), 3)
        # Window 0 is exactly plain rounding (the protocol v1 pixel).
        zero = self.context(window=0)
        for z in (12.0, 17.0, 20.0):
            placed = zero.place(np.array([1.0, z]), {'status': 'mapped'})
            self.assertEqual(placed['pixel'], placed['pixelSearch']['roundedPixel']); self.assertEqual(placed['pixelSearch']['offsetPx'], [0, 0])

    def test_protocol_without_v2_pixel_rule_is_refused(self):
        scene, ground, terrain, positions, eligible, protocol, _ = tool.load(**self.grounded())
        for rule in ('Desired ground point ... pixel = [floor(u+0.5), floor(v+0.5)]', {'rule': 'nearestHitWindow', 'windowPx': 3.0},
                     {'rule': 'nearestHitWindow', 'windowPx': 11}, {'rule': 'other', 'windowPx': 3}):
            bad = dict(protocol, pixelSelection=rule)
            with self.assertRaisesRegex(ValueError, 'pixelSelection'):
                tool.suggest(scene, ground, terrain, positions, eligible, bad)

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


def real_paths(folder, freeze, protocol):
    return dict(grounded=REAL / folder, metadata_sha256=sha((REAL / folder / 'scene.json').read_bytes()), buffers_dir=REAL / 'assembled',
                protocol_path=REAL / protocol, protocol_sha256=json.loads((REAL / freeze).read_bytes())['sha256'])


@unittest.skipUnless((V2 / 'suggestions.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c v2 files absent')
class RealSuggestionTests(unittest.TestCase):
    """grounded-v2 (protocol v2) is current; grounded/ (protocol v1) is historical and only cross-checked."""

    def test_real_v2_suggestions_reproduce_and_match_expected_values(self):
        stored = (V2 / 'suggestions.json').read_bytes(); doc = json.loads(stored)
        paths = real_paths('grounded-v2', 'protocol-v2-freeze.json', 'protocol-v2.json')
        self.assertEqual(sha((REAL / 'protocol-v2.json').read_bytes()), paths['protocol_sha256'])
        from datetime import datetime
        epoch = str(int(datetime.strptime(doc['generatedUtc'], '%Y-%m-%dT%H:%M:%S%z').timestamp()))
        with mock.patch.dict(os.environ, {'SOURCE_DATE_EPOCH': epoch}):
            self.assertEqual(tool.canonical(tool.document(*tool.load(**paths))), stored)
        v = doc['vehicle']
        self.assertEqual((v['status'], v['track'][:8], v['label'], v['reason']),
                         ('annotated', '396f15b5', 'stationary vehicle', 'stationary vehicle (AV2 annotation); reason for stopping unknown'))
        self.assertEqual([doc[k]['pixel'] for k in PLACED], [[373, 188], [377, 188], [486, 186]])
        self.assertEqual([doc[k]['pixelSearch']['roundedPixel'] for k in PLACED], [[373, 189], [377, 191], [484, 185]])
        self.assertEqual([doc[k]['pixelSearch']['offsetPx'] for k in PLACED], [[0, -1], [0, -3], [2, 1]])
        for k, expected in zip(PLACED, (.1437, .1767, .2096)):
            search = doc[k]['pixelSearch']
            self.assertAlmostEqual(search['residualM'], expected, places=4); self.assertLessEqual(search['residualM'], search['roundedResidualM'])
            self.assertEqual(search['residualM'], doc[k]['pixelOffsetFromMappedM']); self.assertEqual(search['candidatesPlaced'], 49)
        self.assertEqual([doc[k]['supportClass'] for k in PLACED], ['map-only'] * 3)
        self.assertEqual([doc[k]['hidden'] for k in PLACED], [True, True, False]); self.assertFalse(doc['person']['topHiddenDiagnostic'])
        self.assertAlmostEqual(v['frontGapToCrossingM'], .5456, places=3); self.assertAlmostEqual(doc['roadAxis']['skewDeg'], 3.338, places=2)
        self.assertAlmostEqual(v['yawToRoadAxisDeg'], 1.407, places=3); self.assertAlmostEqual(v['headingToRoadAxisDeg'], abs(v['yawToRoadAxisDeg']), places=9)
        r = v['frontResidualToAnnotatedM']
        self.assertEqual(r['horizontal'], v['pixelOffsetFromMappedM']); self.assertAlmostEqual(r['alongRoadAxis'], -.1726, places=3)
        self.assertAlmostEqual(r['lateralRight'], -.0378, places=3); self.assertAlmostEqual(v['pixelGroundGapAlongAxisM'], .8591, places=3)
        entry = json.loads((V2 / 'catalog-entry.json').read_bytes())
        self.assertEqual(entry['suggestions']['sha256'], sha(stored)); self.assertEqual(entry['metadataSha256'], paths['metadata_sha256'])
        command = [sys.executable, '-I', '-B', str(SCRIPT), '--grounded', str(V2), '--metadata-sha256', paths['metadata_sha256'],
                   '--buffers', str(REAL / 'assembled'), '--protocol', str(REAL / 'protocol-v2.json'), '--protocol-sha256', paths['protocol_sha256']]
        refused = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(refused.returncode, 1); self.assertIn('Existing suggestions.json retained', refused.stderr)
        self.assertEqual((V2 / 'suggestions.json').read_bytes(), stored)

    def test_no_form_of_the_banned_word_in_v2_outputs(self):
        for path in (V2 / 'suggestions.json', V2 / 'ground.json', V2 / 'scene.json', V2 / 'catalog-entry.json',
                     REAL / 'protocol-v2.json', REAL / 'runtime-test-vectors-v2.json'):
            if path.exists():
                self.assertNotIn(b'park', path.read_bytes().lower(), path.name)

    @unittest.skipUnless((REAL / 'grounded/suggestions.json').exists(), 'historical v1 output absent')
    def test_v2_rounding_pixel_is_the_historical_v1_pixel_and_never_worse(self):
        v1 = json.loads((REAL / 'grounded/suggestions.json').read_bytes()); v2 = json.loads((V2 / 'suggestions.json').read_bytes())
        for k in PLACED:
            search = v2[k]['pixelSearch']
            self.assertEqual(search['roundedPixel'], v1[k]['pixel'])
            self.assertAlmostEqual(search['roundedResidualM'], v1[k]['pixelOffsetFromMappedM'], places=12)
            self.assertLess(search['residualM'], v1[k]['pixelOffsetFromMappedM'])

    @unittest.skipUnless((REAL / 'grounded/suggestions.json').exists(), 'historical v1 output absent')
    def test_v1_protocol_is_refused(self):
        paths = real_paths('grounded', 'protocol-freeze.json', 'protocol.json')
        with self.assertRaisesRegex(ValueError, 'pixelSelection'):
            tool.document(*tool.load(**paths))


if __name__ == '__main__':
    unittest.main()

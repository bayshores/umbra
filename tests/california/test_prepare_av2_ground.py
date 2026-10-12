"""Synthetic analytic fixtures for the AV2 mapped-ground layer, plus a real-data check skipped when files are absent."""
import copy
import hashlib
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_av2_ground', ROOT / 'scripts/california/prepare_av2_ground.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
prepare_av2 = tool.prepare_av2

ORIGIN = np.array([100., 200., 10.])          # left camera city position
SIM2 = {'R': [1.0, 0.0, 0.0, 1.0], 't': [-30.0, -130.0], 's': 1 / 0.3}
GROUND_Y = -1.6                                 # flat mapped ground 1.6 m below the camera
NAN_BLOCK = ((2.0, 4.0), (30.0, 35.0))          # world X, Z ranges with no raster value
PITCH = math.radians(5)
TRACK = 'aaaaaaaa-0000-4000-8000-000000000001'
SWEEP = 1_000_000_000
LOG = '8c52d911-fe34-3424-9864-d3fdfac38064'
PROTOCOL = {
    'scene': {'sceneId': 'fixture-scene', 'logId': LOG, 'cameraTimestampNs': str(SWEEP + 40_000_000), 'lidarTimestampNs': str(SWEEP),
              'annotationTimestampNs': str(SWEEP), 'crossingId': '77'},
    'tolerances': {'roadToleranceMeters': 0.08, 'supportHeightAgreementMeters': 0.08, 'supportRadiusMeters': 0.5,
                   'maxRangeMeters': 60.0, 'boundaryEpsilonMeters': 0.01},
    'groundRaster': {'expectedCellMeters': 0.3, 'windowHalfWidthMeters': 61.0},
    'mapElements': {'roadAxis': 'Ego lane segment 43186931 (fixed): analytic fixture.'},
    'currentVehicle': {'trackPrefix': 'aaaaaaaa'},
    'pixelSelection': {'rule': 'nearestHitWindow', 'windowPx': 3},
    'declaredDefaults': {'vehicle': {'label': 'SUV', 'dimensions': {'width': 1.9, 'height': 1.8, 'length': 4.9}, 'setbackMeters': 6.096,
                                     'curbGapMeters': 0.15, 'reason': 'declared 20 ft default, not a vehicle in the image'},
                         'person': {'label': 'child', 'heightMeters': 1.1, 'curbOffsetMeters': 0.30,
                                    'reason': 'declared default at the curb end of the crossing near edge'}}}
# World positions of returns: agreeing visible, above ground, agreeing but behind the car, not eligible, unavailable, agreeing far.
RETURNS = np.array([[1.0, GROUND_Y, 10.0], [1.2, GROUND_Y + .6, 10.0], [-4.0, GROUND_Y, 21.0],
                    [-1.0, GROUND_Y, 12.0], [np.nan, np.nan, np.nan], [0.0, GROUND_Y + .05, 30.0]])
ELIGIBLE = np.array([1, 1, 1, 0, 0, 1], np.uint8)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def city(world):
    w = np.asarray(world, float); return np.column_stack([w[:, 0] + ORIGIN[0], -w[:, 2] + ORIGIN[1], w[:, 1] + ORIGIN[2]])


def point(x, y, z):
    c = city([[x, y, z]])[0]; return {'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2])}


def raster():
    r = np.full((500, 500), np.float16(GROUND_Y + ORIGIN[2]), np.float16)
    (x0, x1), (z0, z1) = NAN_BLOCK
    corners = city([[x0, 0, z0], [x1, 0, z1]])
    i0, i1 = sorted((corners[:, 0] + SIM2['t'][0]) * SIM2['s']); j0, j1 = sorted((corners[:, 1] + SIM2['t'][1]) * SIM2['s'])
    r[int(j0):int(j1) + 1, int(i0):int(i1) + 1] = np.nan
    return r


def camera_scene():
    f = np.array([0, -math.sin(PITCH), math.cos(PITCH)]); up = np.array([0, math.cos(PITCH), math.sin(PITCH)]); right = np.cross(f, up)
    w = np.column_stack([right, -up, f])
    return {'camera': {'fx': 860., 'fy': 860., 'cx': 368., 'cy': 160., 'width': 736, 'height': 320},
            'transform': {'worldFromCamera': w.tolist(), 'cityOrigin': ORIGIN.tolist(), 'worldFromCity': tool.WORLD_FROM_CITY.tolist()}}


def vector_map(split=False):
    def area(i, z0, z1):
        return {'id': i, 'area_boundary': [point(x, 0, z) for x, z in ((-6, z0), (6, z0), (6, z1), (-6, z1))]}
    areas = {'1': area(1, -5, 18), '2': area(2, 18, 60)} if split else {'1': area(1, -5, 60)}
    return {'drivable_areas': areas,
            'pedestrian_crossings': {'77': {'id': 77, 'edge1': [point(6, GROUND_Y, 20), point(-6, GROUND_Y, 20)],
                                            'edge2': [point(6, GROUND_Y, 24), point(-6, GROUND_Y, 24)]}},
            'lane_segments': {'43186931': {'id': 43186931, 'lane_type': 'VEHICLE', 'left_lane_mark_type': 'DOUBLE_SOLID_YELLOW',
                                           'right_lane_mark_type': 'SOLID_WHITE', 'is_intersection': False,
                                           'left_lane_boundary': [point(1.75, GROUND_Y, z) for z in (-5, 20, 50)],
                                           'right_lane_boundary': [point(-1.75, GROUND_Y, z) for z in (-5, 50)]}}}


def annotation_table(moving=False):
    """Stationary car centred at world (-4, -0.85, 17.5) heading +Z (front at Z=19.5, 0.5 m before edge1), plus a pedestrian."""
    rows = []; half = math.sqrt(.5)
    for k in range(50):
        ts = SWEEP - (49 - k) * 100_000_000
        jitter = 3.0 * (k - 49) / 49 if moving else 0.004 * ((49 - k) % 3)
        rows.append(dict(timestamp_ns=ts, track_uuid=TRACK, category='REGULAR_VEHICLE', length_m=4.0, width_m=1.8, height_m=1.5,
                         qw=half, qx=0., qy=0., qz=-half, tx_m=-4.0 + jitter, ty_m=-17.5, tz_m=.75, num_interior_pts=300))
        rows.append(dict(timestamp_ns=ts, track_uuid='bbbbbbbb-0000-4000-8000-000000000002', category='PEDESTRIAN', length_m=.6, width_m=.6,
                         height_m=1.7, qw=1., qx=0., qy=0., qz=0., tx_m=3.0, ty_m=-40.0 + .1 * k, tz_m=.85, num_interior_pts=20))
    return pa.Table.from_pylist(rows)


def pose_table():
    stamps = sorted({SWEEP - k * 100_000_000 for k in range(50)} | {SWEEP + 40_000_000})
    return pa.Table.from_pylist([dict(timestamp_ns=t, qx=0., qy=0., qz=0., qw=1., tx_m=ORIGIN[0], ty_m=ORIGIN[1], tz_m=ORIGIN[2] + GROUND_Y)
                                 for t in stamps])


def write_fixture(root, split=False, moving=False):
    """Analytic stand-ins for an assembled scene, its preparation and AV2 sources. Returns build() keyword arguments."""
    root = Path(root); prepared = root / 'prepared'; assembled = root / 'assembled'; sources = root / 'sources'
    for d in (prepared, assembled, sources):
        d.mkdir(parents=True)
    files = {'map.json': json.dumps(vector_map(split)).encode(), 'sim2.json': json.dumps(SIM2).encode()}
    stream = BytesIO(); np.save(stream, raster()); files['ground.npy'] = stream.getvalue()
    for name, table in (('poses.feather', pose_table()), ('annotations.feather', annotation_table(moving))):
        feather.write_feather(table, sources / name); files[name] = (sources / name).read_bytes()
    for name in ('map.json', 'sim2.json', 'ground.npy'):
        (sources / name).write_bytes(files[name])
    heights = prepare_av2.ground_sample(city(RETURNS), raster(), SIM2)['nominalGroundHeights']
    stream = BytesIO(); np.savez_compressed(stream, nominalGroundHeights=heights)
    references = stream.getvalue(); (prepared / 'references.npz').write_bytes(references)
    manifest = {'logId': LOG, 'crossingId': '77', 'outputs': {'references.npz': {'sha256': sha(references), 'bytes': len(references)}},
                'sources': {role: {'path': '../sources/' + name, 'sha256': sha(files[name]), 'bytes': len(files[name])}
                            for role, name in (('map', 'map.json'), ('groundRaster', 'ground.npy'), ('groundTransform', 'sim2.json'),
                                               ('poses', 'poses.feather'))}}
    manifest_raw = json.dumps(manifest).encode(); (prepared / 'manifest.json').write_bytes(manifest_raw)
    positions = RETURNS.astype('<f8').tobytes(); eligible = ELIGIBLE.tobytes()
    (assembled / 'referencepositions.bin').write_bytes(positions); (assembled / 'referenceeligible.bin').write_bytes(eligible)
    scene = {'id': 'fixture-scene', 'sourceClass': 'Argoverse2Sensor', **camera_scene(),
             'recordedImage': {'file': 'recorded.png', 'sha256': 'c' * 64},
             'measurements': {'kind': 'indexedSparse', 'groundPolicy': 'unvalidated', 'count': len(RETURNS),
                              'buffers': {'referencePositions': 'referencepositions.bin', 'referenceEligible': 'referenceeligible.bin'},
                              'artifactHashes': {'referencePositions': sha(positions), 'referenceEligible': sha(eligible)},
                              'provenance': {'preparedManifestSha256': sha(manifest_raw), 'referencesSha256': sha(references),
                                             'originalSourceRecordBase': '/analytic/prepared', **{k: PROTOCOL['scene'][k] for k in
                                                                                         ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId')}}},
             'processing': {'groundPolicy': 'unvalidated; nominal terrain is not observed ground'},
             'limitations': ['Native grounding/movement is unvalidated; nominal terrain is not promoted to observed road.', 'Other.']}
    scene_raw = json.dumps(scene).encode(); (assembled / 'scene.json').write_bytes(scene_raw)
    protocol_raw = json.dumps(PROTOCOL).encode(); (root / 'protocol.json').write_bytes(protocol_raw)
    return dict(scene_path=assembled / 'scene.json', scene_sha256=sha(scene_raw), prepared_path=prepared / 'manifest.json',
                prepared_sha256=sha(manifest_raw), protocol_path=root / 'protocol.json', protocol_sha256=sha(protocol_raw),
                annotations_path=sources / 'annotations.feather', annotations_sha256=sha(files['annotations.feather']),
                output=root / 'grounded')


def flat_terrain(height_city=GROUND_Y + ORIGIN[2], window=None):
    w = np.full((500, 500), height_city, np.float32) if window is None else window
    return tool.Terrain(w, 0, 0, w.shape, SIM2, ORIGIN)


class GroundGeometryTests(unittest.TestCase):
    def test_terrain_sampling_matches_prepare_av2(self):
        rng = np.random.default_rng(3)
        for sim in (SIM2, {'R': [math.cos(.5), -math.sin(.5), math.sin(.5), math.cos(.5)], 't': [83.0, -157.5], 's': 1 / 0.3}):
            full = raster().astype(np.float16); full[::7, ::5] = np.nan
            window = full[40:400, 60:450].astype(np.float32)
            terrain = tool.Terrain(window, 60, 40, full.shape, sim, ORIGIN)
            world = np.column_stack([rng.uniform(-100, 100, 4000), np.zeros(4000), rng.uniform(-100, 100, 4000)])
            heights, idx = terrain.sample(world[:, [0, 2]])
            expected = prepare_av2.ground_sample(city(world), full, sim)
            in_window = idx[:, 0] >= 0
            self.assertGreater(in_window.sum(), 300); self.assertGreater((~in_window).sum(), 300)
            np.testing.assert_array_equal(heights[in_window], expected['nominalGroundHeights'][in_window])
            self.assertTrue(np.isnan(heights[~in_window]).all())

    def test_flat_ray_hits_cell_top_exactly(self):
        d = tool.ray_direction(camera_scene(), [368, 200]); hit = tool.ray_hit(d, flat_terrain(), 60)
        self.assertTrue(hit['hit']); self.assertEqual(hit['kind'], 'top')
        ground = float(np.float32(GROUND_Y + ORIGIN[2])) - ORIGIN[2]  # float32 window value
        self.assertAlmostEqual(hit['point'][1], ground, places=12); self.assertAlmostEqual(hit['t'], ground / d[1], places=9)

    def test_raised_cell_gives_step_hit_and_matches_brute_force(self):
        w = np.full((500, 500), GROUND_Y + ORIGIN[2], np.float32)
        w[169:172, :] += .05  # rows 169-171 cover world Z 18.4-19.3: 5 cm higher, entered below their top
        terrain = flat_terrain(window=w); d = tool.ray_direction(camera_scene(), [368, 158])
        hit = tool.ray_hit(d, terrain, 60)
        self.assertTrue(hit['hit']); self.assertEqual(hit['kind'], 'step')
        self.assertGreater(hit['point'][1], GROUND_Y)  # on the vertical face of the raised cells
        t = np.arange(1e-4, 60, 1e-4); pts = np.outer(t, d); h = terrain.sample(pts[:, [0, 2]])[0] - ORIGIN[2]
        first = t[np.flatnonzero(pts[:, 1] <= h)[0]]
        self.assertLess(abs(first - hit['t']), 2e-4)

    def test_nan_cell_upward_ray_and_range_stay_unknown(self):
        terrain = tool.Terrain(raster().astype(np.float32), 0, 0, (500, 500), SIM2, ORIGIN)
        target = np.array([3.0, GROUND_Y, 40.0]); d = target / np.linalg.norm(target)
        hit = tool.ray_hit(d, terrain, 60)
        self.assertFalse(hit['hit']); self.assertIn('without mapped ground height', hit['reason'])
        self.assertFalse(tool.ray_hit(np.array([0, .1, 1]) / math.hypot(.1, 1), terrain, 60)['hit'])
        far = np.array([0, -1.6, 80.]); self.assertIn('No mapped ground within 60', tool.ray_hit(far / np.linalg.norm(far), flat_terrain(), 60)['reason'])

    def test_polygons_and_curb_scan_ignore_shared_internal_edges(self):
        whole = [np.array([[-6, -5], [6, -5], [6, 60], [-6, 60]], float)]
        split = [np.array([[-6, -5], [6, -5], [6, 18], [-6, 18]], float), np.array([[-6, 18], [6, 18], [6, 60], [-6, 60]], float)]
        np.testing.assert_array_equal(tool.inside_polygons([[0, 0], [7, 0], [0, 61]], whole), [True, False, False])
        for polys in (whole, split):  # scanning along +Z from Z=0 crosses the shared edge at 18 but exits only at 60
            self.assertAlmostEqual(tool.curb_scan([0., 0.], [0., 1.], polys, .01, limit=100), 60.)
            self.assertAlmostEqual(tool.curb_scan([0., 10.], [-1., 0.], polys, .01), 6.)
        self.assertIsNone(tool.curb_scan([9., 0.], [1., 0.], whole, .01))

    def test_segment_box_hidden_rule(self):
        box = {'center': [0., -1., 10.], 'worldFromBox': [[0, 1, 0], [0, 0, 1], [1, 0, 0]], 'length': 4., 'width': 2., 'height': 1.5}
        hits = tool.segment_hits_boxes([[0., -1.6, 20.], [5., -1.6, 20.], [0., -1.6, 10.5], [0., -1.6, 7.]], [box])
        np.testing.assert_array_equal(hits, [True, False, True, False])  # behind, beside, ending inside, in front

    def test_support_rows_test_all_returns_and_respect_cuboids(self):
        terrain = tool.Terrain(raster().astype(np.float32), 0, 0, (500, 500), SIM2, ORIGIN)
        box = {'center': [-4., -.85, 17.5], 'worldFromBox': [[0, 1, 0], [0, 0, 1], [1, 0, 0]], 'length': 4., 'width': 1.8, 'height': 1.5}
        rows, stats = tool.support_rows(RETURNS, ELIGIBLE.astype(bool), terrain, [box], .08)
        self.assertEqual(rows.tolist(), [0, 5]); self.assertEqual(stats['eligibleBehindCuboid'], 1); self.assertEqual(stats['eligibleAgreeing'], 3)
        rows, stats = tool.support_rows(RETURNS, ELIGIBLE.astype(bool), terrain, None, .08)
        self.assertEqual(rows.tolist(), [0, 2, 5]); self.assertFalse(stats['cuboidOcclusionApplied'])
        rows, _ = tool.support_rows(RETURNS, np.zeros(len(RETURNS), bool), terrain, None, .08)
        self.assertEqual(rows.tolist(), [])


class GroundBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_build_writes_pinned_layer_and_refuses_overwrite(self):
        args = write_fixture(self.root / 'a'); result = tool.build(**args)
        out = args['output']; ground_raw = (out / 'ground.json').read_bytes(); ground = json.loads(ground_raw)
        scene = json.loads((out / 'scene.json').read_bytes()); terrain_raw = (out / 'terrain.bin').read_bytes()
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['ground.json', 'scene.json', 'terrain.bin'])
        self.assertEqual(scene['measurements']['groundPolicy'], 'mappedGround')
        self.assertEqual(scene['measurements']['ground'], {'file': 'ground.json', 'bytes': len(ground_raw), 'sha256': sha(ground_raw)})
        self.assertEqual(ground['terrain']['sha256'], sha(terrain_raw)); self.assertEqual(result['groundSha256'], sha(ground_raw))
        self.assertEqual(ground['terrain']['bytes'], ground['terrain']['rows'] * ground['terrain']['columns'] * 4)
        self.assertEqual(ground['support']['rows'], [0, 5]); self.assertEqual(ground['support']['preparedLookupMismatches'], 0)
        self.assertEqual(ground['roadToleranceMeters'], .08); self.assertNotIn('roadToleranceMeters', scene['processing'])
        car = next(c for c in ground['cuboids']['items'] if c['track'] == TRACK)
        np.testing.assert_allclose(car['center'], [-4., -.85, 17.5], atol=1e-9)
        np.testing.assert_allclose(np.asarray(car['worldFromBox'])[:, 0], [0, 0, 1], atol=1e-9)
        self.assertLess(car['motion']['maxDisplacementFromFirstM'], .01); self.assertAlmostEqual(car['motion']['spanSeconds'], 4.9)
        for text in (ground_raw.decode(), (out / 'scene.json').read_text()):
            self.assertNotIn(str(self.root), text); self.assertNotIn('"/', text)
        self.assertEqual(scene['measurements']['provenance']['originalSourceRecordBase'], '../prepared')
        with self.assertRaisesRegex(ValueError, 'Existing output retained'):
            tool.build(**args)

    def test_build_is_deterministic(self):
        first = write_fixture(self.root / 'a'); second = write_fixture(self.root / 'b')
        tool.build(**first); tool.build(**second)
        for name in ('ground.json', 'terrain.bin', 'scene.json'):
            self.assertEqual((first['output'] / name).read_bytes(), (second['output'] / name).read_bytes())

    def test_without_annotations_hidden_is_not_evaluated(self):
        args = write_fixture(self.root / 'a'); args.update(annotations_path=None, annotations_sha256=None); tool.build(**args)
        ground = json.loads((args['output'] / 'ground.json').read_bytes())
        self.assertFalse(ground['cuboids']['available']); self.assertEqual(ground['cuboids']['items'], [])
        self.assertIsNone(ground['sources']['annotations']); self.assertEqual(ground['support']['rows'], [0, 2, 5])
        self.assertTrue(any('hidden flags are not evaluated' in t for t in ground['limitations']))

    def test_rejects_changed_inputs(self):
        args = write_fixture(self.root / 'a')
        for key in ('scene_sha256', 'prepared_sha256', 'protocol_sha256', 'annotations_sha256'):
            bad = dict(args, **{key: '0' * 64})
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                tool.build(**bad)
        raw = Path(args['scene_path']).read_bytes(); scene = json.loads(raw); scene['measurements']['groundPolicy'] = 'mappedGround'
        changed = json.dumps(scene).encode(); Path(args['scene_path']).write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'unvalidated ground required'):
            tool.build(**dict(args, scene_sha256=sha(changed)))
        self.assertFalse(args['output'].exists())


REAL = ROOT / '.local/evaluation/av2-map-first-2026-10-11'


@unittest.skipUnless((REAL / 'grounded-v2/ground.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c files absent')
class RealGroundTests(unittest.TestCase):
    """grounded-v2 (protocol v2, current) and grounded/ (protocol v1, historical) come from the same unchanged ground script."""

    def test_real_layers_are_pinned_and_reproducible(self):
        for folder, freeze in (('grounded-v2', 'protocol-v2-freeze.json'), ('grounded', 'protocol-freeze.json')):
            if (REAL / folder / 'ground.json').exists():
                with self.subTest(folder=folder):
                    self.check_layer(REAL / folder, freeze)

    def test_v2_layer_differs_from_v1_only_in_the_protocol(self):
        if not (REAL / 'grounded/ground.json').exists():
            self.skipTest('historical v1 layer absent')
        v1 = json.loads((REAL / 'grounded/ground.json').read_bytes()); v2 = json.loads((REAL / 'grounded-v2/ground.json').read_bytes())
        self.assertEqual({k for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}, {'protocol'})
        self.assertEqual(v2['protocol']['path'], '../protocol-v2.json')
        self.assertEqual((REAL / 'grounded/terrain.bin').read_bytes(), (REAL / 'grounded-v2/terrain.bin').read_bytes())
        for name in ('ground.json', 'scene.json'):
            self.assertNotIn(b'park', (REAL / 'grounded-v2' / name).read_bytes().lower())

    def check_layer(self, grounded, freeze):
        ground_raw = (grounded / 'ground.json').read_bytes(); ground = json.loads(ground_raw)
        scene = json.loads((grounded / 'scene.json').read_bytes()); assembled = json.loads((REAL / 'assembled/scene.json').read_bytes())
        self.assertEqual(scene['measurements']['ground']['sha256'], sha(ground_raw))
        self.assertEqual(ground['baseSceneSha256'], sha((REAL / 'assembled/scene.json').read_bytes()))
        self.assertEqual(ground['protocol']['sha256'], json.loads((REAL / freeze).read_bytes())['sha256'])
        terrain_raw = (grounded / 'terrain.bin').read_bytes(); t = ground['terrain']; self.assertEqual(sha(terrain_raw), t['sha256'])
        window = np.frombuffer(terrain_raw, '<f4').reshape(t['rows'], t['columns'])
        source = (REAL / 'prepared' / json.loads((REAL / 'prepared/manifest.json').read_bytes())['sources']['groundRaster']['path']).resolve()
        full = np.load(source)
        np.testing.assert_array_equal(window, full[t['row0']:t['row0'] + t['rows'], t['column0']:t['column0'] + t['columns']].astype(np.float32))
        changed = {k for k in scene if scene[k] != assembled[k]}
        self.assertEqual(changed, {'measurements', 'processing', 'limitations'})
        self.assertEqual({k for k in scene['measurements'] if scene['measurements'][k] != assembled['measurements'].get(k)},
                         {'groundPolicy', 'ground', 'provenance'})
        m = assembled['measurements']; count = m['count']
        positions = np.frombuffer((REAL / 'assembled' / m['buffers']['referencePositions']).read_bytes(), '<f8').reshape(count, 3)
        eligible = np.frombuffer((REAL / 'assembled' / m['buffers']['referenceEligible']).read_bytes(), 'u1').astype(bool)
        terrain = tool.Terrain(window, t['column0'], t['row0'], t['rasterShape'], t['sim2'], ground['frame']['cityOrigin'])
        rows, stats = tool.support_rows(positions, eligible, terrain, ground['cuboids']['items'], ground['supportHeightAgreementMeters'])
        self.assertEqual(rows.tolist(), ground['support']['rows']); self.assertEqual(ground['support']['preparedLookupMismatches'], 0)
        car = [c for c in ground['cuboids']['items'] if c['track'].startswith('396f15b5')]
        self.assertEqual(len(car), 1); self.assertAlmostEqual(car[0]['motion']['maxDisplacementFromFirstM'], .506, places=3)
        self.assertNotIn('"/', ground_raw.decode()); self.assertNotIn('"/', (grounded / 'scene.json').read_text())


if __name__ == '__main__':
    unittest.main()

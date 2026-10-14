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
# Protocol v3 fixture: the v2 fields plus exactProjection, the averaged pose, the fitted band, the 30 m limit and the
# crossing-error slot (the same keys as .local/evaluation/av2-map-first-2026-10-11/protocol-v3.json).
PROTOCOL_V3 = {**PROTOCOL, 'protocolVersion': 3, 'pixelSelection': {'rule': 'exactProjection'},
               'placementRange': {'limitMeters': 30.0},
               'currentVehicle': {'trackPrefix': 'aaaaaaaa', 'pose': {'rule': 'averageAllAnnotations'}},
               'alongRayUncertainty': {'rule': 'heightBandMeters / sin(depression)',
                                       'heightBand': {'method': 'fitThisDevelopmentScene', 'fitSet': 'fixture', 'groundGateMeters': 0.25,
                                                      'statistic': 'nearest-rank q95 of |dz|, rounded up to 1 mm', 'minimumCount': 50},
                                       'coverageReport': {'bandsMeters': [[0, 10], [10, 12], [12, 25], [25, 30]], 'definition': 'fixture'}},
               'crossingError': {'perCrossing': {'77': {'status': 'pending-measurement'}}}}
RASTER_Y = float(np.float16(GROUND_Y + ORIGIN[2])) - ORIGIN[2]  # the float16 raster's world height (-1.6015625)
# 100 extra ground returns for the v3 band fit: x = -1.5 (clear of the car, the NaN block and every point the suggestion
# tests evaluate), z 6.5-29.5, dz = -0.100 ... +0.098 m from the raster (|dz| nearest-rank q95 = 0.094 m), plus five that
# the fit set must exclude: outside the drivable area, beyond 30 m, above the ground gate, behind the car, not camera-eligible.
GRID_DZ = (np.arange(100) - 50) * .002
V3_RETURNS = np.vstack([RETURNS, np.column_stack([np.full(100, -1.5), RASTER_Y + GRID_DZ, np.linspace(6.5, 29.5, 100)]),
                        [[7.0, RASTER_Y, 20.0], [0.0, RASTER_Y, 40.0], [2.5, RASTER_Y + .3, 20.0], [-4.0, RASTER_Y, 21.5], [2.5, RASTER_Y, 15.0]]])
V3_ELIGIBLE = np.concatenate([ELIGIBLE, np.ones(104, np.uint8), [0]]).astype(np.uint8)


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


def uv_of(world):
    """Floating crop projection of world points with the fixture camera (NaN behind the camera or for NaN rows)."""
    s = camera_scene(); c = s['camera']; q = np.asarray(world, float) @ np.asarray(s['transform']['worldFromCamera'], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = np.column_stack([c['fx'] * q[:, 0] / q[:, 2] + c['cx'], c['fy'] * q[:, 1] / q[:, 2] + c['cy']])
    uv[~(q[:, 2] > 0)] = np.nan
    return uv


def write_fixture(root, split=False, moving=False, protocol=None, returns=None, eligible=None, annotations=None, edit_map=None, poses=None):
    """Analytic stand-ins for an assembled scene, its preparation and AV2 sources. Returns build() keyword arguments.
    protocol/returns/eligible/annotations/poses replace PROTOCOL, RETURNS, ELIGIBLE, the annotation table and the pose table (v3 tests);
    edit_map(vector_map) changes the native map in place before it is written (NaN heights are written as the NaN token)."""
    protocol = PROTOCOL if protocol is None else protocol
    returns = RETURNS if returns is None else np.asarray(returns, float)
    eligible = ELIGIBLE if eligible is None else np.asarray(eligible, np.uint8)
    root = Path(root); prepared = root / 'prepared'; assembled = root / 'assembled'; sources = root / 'sources'
    for d in (prepared, assembled, sources):
        d.mkdir(parents=True)
    native_map = vector_map(split)
    if edit_map is not None:
        edit_map(native_map)
    files = {'map.json': json.dumps(native_map).encode(), 'sim2.json': json.dumps(SIM2).encode()}
    stream = BytesIO(); np.save(stream, raster()); files['ground.npy'] = stream.getvalue()
    for name, table in (('poses.feather', pose_table() if poses is None else poses),
                        ('annotations.feather', annotation_table(moving) if annotations is None else annotations)):
        feather.write_feather(table, sources / name); files[name] = (sources / name).read_bytes()
    for name in ('map.json', 'sim2.json', 'ground.npy'):
        (sources / name).write_bytes(files[name])
    heights = prepare_av2.ground_sample(city(returns), raster(), SIM2)['nominalGroundHeights']
    stream = BytesIO(); np.savez_compressed(stream, nominalGroundHeights=heights)
    references = stream.getvalue(); (prepared / 'references.npz').write_bytes(references)
    manifest = {'logId': LOG, 'crossingId': '77', 'outputs': {'references.npz': {'sha256': sha(references), 'bytes': len(references)}},
                'sources': {role: {'path': '../sources/' + name, 'sha256': sha(files[name]), 'bytes': len(files[name])}
                            for role, name in (('map', 'map.json'), ('groundRaster', 'ground.npy'), ('groundTransform', 'sim2.json'),
                                               ('poses', 'poses.feather'))}}
    manifest_raw = json.dumps(manifest).encode(); (prepared / 'manifest.json').write_bytes(manifest_raw)
    positions = returns.astype('<f8').tobytes(); uv = uv_of(returns).astype('<f8').tobytes(); eligible = eligible.tobytes()
    (assembled / 'referencepositions.bin').write_bytes(positions); (assembled / 'referenceeligible.bin').write_bytes(eligible)
    (assembled / 'referenceuv.bin').write_bytes(uv)
    scene = {'id': 'fixture-scene', 'sourceClass': 'Argoverse2Sensor', **camera_scene(),
             'recordedImage': {'file': 'recorded.png', 'sha256': 'c' * 64},
             'measurements': {'kind': 'indexedSparse', 'groundPolicy': 'unvalidated', 'count': len(returns),
                              'buffers': {'referencePositions': 'referencepositions.bin', 'referenceEligible': 'referenceeligible.bin',
                                          'referenceUV': 'referenceuv.bin'},
                              'artifactHashes': {'referencePositions': sha(positions), 'referenceEligible': sha(eligible), 'referenceUV': sha(uv)},
                              'provenance': {'preparedManifestSha256': sha(manifest_raw), 'referencesSha256': sha(references),
                                             'originalSourceRecordBase': '/analytic/prepared', **{k: protocol['scene'][k] for k in
                                                                                         ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId')}}},
             'processing': {'groundPolicy': 'unvalidated; nominal terrain is not observed ground'},
             'limitations': ['Native grounding/movement is unvalidated; nominal terrain is not promoted to observed road.', 'Other.']}
    scene_raw = json.dumps(scene).encode(); (assembled / 'scene.json').write_bytes(scene_raw)
    protocol_raw = json.dumps(protocol).encode(); (root / 'protocol.json').write_bytes(protocol_raw)
    return dict(scene_path=assembled / 'scene.json', scene_sha256=sha(scene_raw), prepared_path=prepared / 'manifest.json',
                prepared_sha256=sha(manifest_raw), protocol_path=root / 'protocol.json', protocol_sha256=sha(protocol_raw),
                annotations_path=sources / 'annotations.feather', annotations_sha256=sha(files['annotations.feather']),
                output=root / 'grounded')


# Protocol v3.4 fixture: protocol v3 plus protocolRevision '3.4' and the carShape section pinning the frozen shape-v2
# protocol (the same keys build() checks in .local/evaluation/av2-map-first-2026-10-11/protocol-v3.4.json).
SHAPE = tool.car_shape
PROTOCOL_V34 = {**PROTOCOL_V3, 'protocolRevision': '3.4',
                'carShape': {'shapeProtocol': {'path': '../pao-accuracy/shape-v2/protocol.json', 'sha256': SHAPE.SHAPE_PROTOCOL_SHA256},
                             'groundSchemaVersion': 3}}
STEP_NS = 100_000_000
CAR_CELLS = [(i, j) for i in range(0, 40, 4) for j in range(0, 18, 4)]  # 10 x 5 cells of the 40 x 18 grid with car-top returns
spec = importlib.util.spec_from_file_location('car_shape_fixture', ROOT / 'tests/california/test_car_shape.py')
shape_fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(shape_fixture)


def approach_back(ts):
    """How far (m, city +y) the ego is behind its SWEEP position: 1.5 m per 100 ms before SWEEP, 0 from SWEEP on."""
    return 1.5 * max(0, (SWEEP - ts) // STEP_NS)


def approach_tables():
    """The fixture with an ego that drives 1.5 m per sweep along city -y and reaches ORIGIN at SWEEP. Every annotation's
    ego-frame ty follows the ego, so every object keeps its city position (and every v3 field its value)."""
    poses = pose_table().to_pylist(); rows = annotation_table().to_pylist()
    for p in poses:
        p['ty_m'] += approach_back(p['timestamp_ns'])
    for r in rows:
        r['ty_m'] -= approach_back(r['timestamp_ns'])
    return pa.Table.from_pylist(rows), pa.Table.from_pylist(poses)


def car_top_sweeps(ground, poses, height=1.25):
    """One AV2-like sweep per pose with a non-negative timestamp (AV2 timestamps are positive; this fixture's earliest
    annotation times are not), ego frame, float16: returns at `height` above the averaged box bottom, one at the centre of
    each CAR_CELLS cell of that box."""
    box = next(c for c in ground['cuboids']['items'] if 'poseAverage' in c); o = ground['frame']['cityOrigin']
    frame = SHAPE.upright_frame(box); n, m = SHAPE.grid_size(box['length'], box['width'])
    x = np.array([[frame['L'] / 2 - (i + .5) * frame['L'] / n] for i, _ in CAR_CELLS])
    y = np.array([[frame['W'] / 2 - (j + .5) * frame['W'] / m] for _, j in CAR_CELLS])
    scene = frame['center'] + x * frame['heading'] + y * frame['left'] + (height - frame['H'] / 2) * np.array([0., 1., 0.])
    city_points = np.column_stack([scene[:, 0] + o[0], o[1] - scene[:, 2], scene[:, 1] + o[2]])
    return {p['timestamp_ns']: shape_fixture.sweep_bytes(city_points - [p['tx_m'], p['ty_m'], p['tz_m']])  # identity ego rotation
            for p in poses.to_pylist() if p['timestamp_ns'] >= 0}


def car_shape_layer(root, protocol=PROTOCOL_V34, approach=True):
    """The protocol v3.4 operator flow on the fixture: ground.json with --no-car-shape (pre/), the sweep plan from it
    (metadata only, saved listing), the downloads from a fake public bucket, then ground.json with --car-shape-dir
    (grounded/). Returns (build kwargs, pre ground.json, plan_dir, plan sha256, build result)."""
    root = Path(root)
    annotations, poses = approach_tables() if approach else (annotation_table(), pose_table())
    args = write_fixture(root, protocol=protocol, returns=V3_RETURNS, eligible=V3_ELIGIBLE, annotations=annotations, poses=poses)
    tool.build(**{**args, 'output': root / 'pre'}, no_car_shape=True)
    ground_raw = (root / 'pre/ground.json').read_bytes(); ground = json.loads(ground_raw)
    log_dir = root / 'train' / LOG; log_dir.mkdir(parents=True)  # AV2 layout: the plan reads the log's own files
    ann_raw = (root / 'sources/annotations.feather').read_bytes(); poses_raw = (root / 'sources/poses.feather').read_bytes()
    (log_dir / 'annotations.feather').write_bytes(ann_raw); (log_dir / 'city_SE3_egovehicle.feather').write_bytes(poses_raw)
    files = car_top_sweeps(ground, poses)
    listing = shape_fixture.listing_xml([(ts, len(d), shape_fixture.md5(d)) for ts, d in sorted(files.items())])
    (root / 'listing.xml').write_bytes(listing)
    plan_dir, plan_sha = SHAPE.make_plan(LOG, root / 'pre/ground.json', sha(ground_raw), log_dir / 'annotations.feather', sha(ann_raw),
                                         log_dir / 'city_SE3_egovehicle.feather', sha(poses_raw), root / 'listing.xml', sha(listing),
                                         out_root=root / 'data', fetch=shape_fixture.no_network, sleep=shape_fixture.no_network)
    bucket = shape_fixture.Bucket({SHAPE.BUCKET_URL + SHAPE.LIDAR_PREFIX.format(log=LOG) + f'{ts}.feather': d for ts, d in files.items()})
    SHAPE.download_sweeps(plan_dir, plan_sha, fetch=bucket, sleep=lambda _s: None)
    result = tool.build(**args, car_shape_dir=plan_dir, car_shape_plan_sha256=plan_sha)
    return args, ground, plan_dir, plan_sha, result


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

    def test_nan_height_drivable_outlines_keep_their_horizontal_position(self):
        # Publisher maps can carry NaN heights in drivable outlines (453e5558's map: 321 vertices in 9 of 14 outlines).
        # Heights are not used for outlines, so the layer must equal the finite-height layer (no fill, no crash).
        def nan_heights(native):
            native['drivable_areas']['1']['area_boundary'][2]['z'] = float('nan')           # the outline under the crossing
            native['drivable_areas']['3'] = {'id': 3, 'area_boundary': [dict(point(x, 0, z), z=float('nan'))  # off to the side
                                                                         for x, z in ((20, 0), (30, 0), (30, 10), (20, 10))]}
        def finite_heights(native):
            native['drivable_areas']['3'] = {'id': 3, 'area_boundary': [point(x, 0, z) for x, z in ((20, 0), (30, 0), (30, 10), (20, 10))]}
        nan_args = write_fixture(self.root / 'nan', edit_map=nan_heights); finite_args = write_fixture(self.root / 'finite', edit_map=finite_heights)
        self.assertIn(b'NaN', Path(nan_args['scene_path']).parents[1].joinpath('sources/map.json').read_bytes())
        tool.build(**nan_args); tool.build(**finite_args)
        nan_layer = json.loads((nan_args['output'] / 'ground.json').read_bytes())
        finite_layer = json.loads((finite_args['output'] / 'ground.json').read_bytes())
        self.assertEqual([a['outline'] for a in nan_layer['drivableAreas']],
                         [[[-6., -5.], [6., -5.], [6., 60.], [-6., 60.]], [[20., 0.], [30., 0.], [30., 10.], [20., 10.]]])
        for key in ('drivableAreas', 'crossing', 'egoLane', 'support', 'cuboids', 'terrain'):
            self.assertEqual(nan_layer[key], finite_layer[key], key)
        self.assertEqual((nan_args['output'] / 'terrain.bin').read_bytes(), (finite_args['output'] / 'terrain.bin').read_bytes())

    def test_world_xz_equals_the_matrix_conversion_for_finite_heights(self):
        rng = np.random.default_rng(7); city_xyz = ORIGIN + rng.uniform(-80, 80, (200, 3))
        np.testing.assert_array_equal(tool._world_xz(city_xyz, ORIGIN), tool._world(city_xyz, ORIGIN)[:, [0, 2]])
        city_xyz[::3, 2] = np.nan
        self.assertTrue(np.isfinite(tool._world_xz(city_xyz, ORIGIN)).all())
        self.assertFalse(np.isfinite(tool._world(city_xyz, ORIGIN)[:, [0, 2]]).all())  # the defect the helper avoids

    def test_nan_ego_lane_height_is_refused_not_filled(self):
        def nan_lane(native):
            native['lane_segments']['43186931']['left_lane_boundary'][1]['z'] = float('nan')
        args = write_fixture(self.root / 'a', edit_map=nan_lane)
        with self.assertRaisesRegex(ValueError, r'Ego lane left_lane_boundary has a missing \(NaN\) publisher height'):
            tool.build(**args)
        self.assertFalse(args['output'].exists())


class ProtocolV3GroundTests(unittest.TestCase):
    """Protocol v3: averaged stationary pose, fitted along-ray height band, 30 m placement limit (ground.json schemaVersion 2)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def rows(yaws_deg, centres, dims=None, start=0):
        """Synthetic annotation rows of one track with identity ego poses (ego frame = city frame)."""
        out = []
        for k, (yaw, c) in enumerate(zip(yaws_deg, centres)):
            half = math.radians(yaw) / 2; d = dims[k] if dims else (4.0, 1.8, 1.5)
            out.append(dict(timestamp_ns=start + k * 100_000_000, track_uuid=TRACK, category='REGULAR_VEHICLE', length_m=d[0], width_m=d[1],
                            height_m=d[2], qw=math.cos(half), qx=0., qy=0., qz=math.sin(half), tx_m=c[0], ty_m=c[1], tz_m=c[2], num_interior_pts=9))
        return out, {r['timestamp_ns']: np.eye(4) for r in out}

    def test_circular_mean_wraps_and_refuses_cancelling_headings(self):
        self.assertAlmostEqual(abs(math.degrees(tool.circular_mean(np.radians([179., -179., 180.])))), 180., places=9)
        self.assertAlmostEqual(math.degrees(tool.circular_mean(np.radians([-1., 1., 3.]))), 1., places=6)
        self.assertAlmostEqual(math.degrees(tool.wrap_angle(math.radians(359.))), -1., places=9)
        with self.assertRaisesRegex(ValueError, 'cancel'):
            tool.circular_mean(np.radians([0., 180.]))

    def test_averaged_pose_math(self):
        # Headings straddle +/-180 degrees: an arithmetic mean would give about 0; the circular mean gives 180.
        yaws = [178., -178., 179., -179., 180.]
        centres = [(10.0, 5.0, 1.0), (10.2, 5.1, 1.1), (9.8, 4.9, .9), (10.1, 5.0, 1.0), (9.9, 5.0, 1.0)]
        dims = [(4.0, 1.8, 1.5), (4.2, 1.9, 1.4), (4.1, 1.85, 1.6), (4.1, 1.8, 1.5), (3.9, 1.8, 1.5)]
        rows, poses = self.rows(yaws, centres, dims)
        centre, yaw, median, record = tool.average_track_pose(rows, poses, rows[2]['timestamp_ns'])
        np.testing.assert_allclose(centre, np.mean(centres, 0), atol=1e-12)
        self.assertAlmostEqual(abs(math.degrees(yaw)), 180., places=9)
        np.testing.assert_allclose(median, np.median(dims, 0), atol=0)
        # Offsets: heading -X (yaw 180), so along = -(x - mean x) and left (+lateral) = -(y - mean y).
        mean = np.mean(centres, 0); along = -(np.array(centres)[:, 0] - mean[0]); lateral = -(np.array(centres)[:, 1] - mean[1])
        spread = record['centreSpreadM']
        self.assertAlmostEqual(spread['alongStd'], float(np.std(along)), places=12); self.assertAlmostEqual(spread['lateralStd'], float(np.std(lateral)), places=12)
        self.assertAlmostEqual(spread['verticalStd'], float(np.std(np.array(centres)[:, 2])), places=12)
        self.assertAlmostEqual(spread['maxHorizontalFromMean'], float(np.hypot(along, lateral).max()), places=12)
        self.assertAlmostEqual(record['yawSpreadDeg']['maxAbs'], 2.0, places=9)
        sweep = record['sweepRow']
        self.assertAlmostEqual(sweep['alongM'], along[2], places=12); self.assertAlmostEqual(sweep['lateralM'], lateral[2], places=12)
        self.assertAlmostEqual(sweep['yawDeg'], -1.0, places=9)
        self.assertEqual((record['rows'], record['rule']), (5, 'averageAllAnnotations')); self.assertAlmostEqual(record['spanSeconds'], .4)
        self.assertEqual(record['dimensionRangeM']['length'], [3.9, 4.2])
        # Input order does not matter, and the upright box points along the averaged yaw.
        again = tool.average_track_pose(list(reversed(rows)), poses, rows[2]['timestamp_ns'])
        np.testing.assert_array_equal(again[0], centre); self.assertEqual(again[1:], (yaw, median, record))
        m = np.asarray(tool.upright_world_from_box(yaw))
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12); self.assertAlmostEqual(np.linalg.det(m), 1., places=12)
        np.testing.assert_allclose(m[:, 0], [-1, 0, 0], atol=1e-9); np.testing.assert_allclose(m[:, 2], [0, 1, 0], atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'exactly one row'):
            tool.average_track_pose(rows, poses, 123)
        missing = dict(poses); missing.pop(rows[0]['timestamp_ns'])
        with self.assertRaisesRegex(ValueError, 'pose missing'):
            tool.average_track_pose(rows, missing, rows[2]['timestamp_ns'])

    def test_fit_height_band_is_order_free_and_rounds_up(self):
        rng = np.random.default_rng(7); dz = rng.normal(-.008, .03, 2000)
        band, stats = tool.fit_height_band(dz, 200)
        q = float(np.percentile(np.abs(dz), 95, method='inverted_cdf'))  # independent nearest-rank definition
        self.assertEqual(stats['q95AbsM'], q); self.assertEqual(band, math.ceil(q * 1000) / 1000); self.assertGreaterEqual(band, q)
        self.assertAlmostEqual(stats['medianM'], float(np.median(dz)), places=15); self.assertEqual(stats['count'], 2000)
        self.assertAlmostEqual(stats['sigmaRobustM'], 1.4826 * float(np.median(np.abs(dz - np.median(dz)))), places=15)
        self.assertGreaterEqual(stats['withinBandFraction'], .95)
        for order in (rng.permutation(len(dz)), np.argsort(dz), np.argsort(-np.abs(dz))):
            self.assertEqual(tool.fit_height_band(dz[order], 200), (band, stats))  # bit-identical in any order
        self.assertEqual(tool.fit_height_band([.074] * 300, 200)[0], .074)  # an exact millimetre is not bumped up
        with self.assertRaisesRegex(ValueError, 'at least 200'):
            tool.fit_height_band(dz[:199], 200)
        with self.assertRaisesRegex(ValueError, 'at least'):
            tool.fit_height_band([np.nan] * 300, 200)

    def test_band_fit_refuses_held_out_and_reserve_logs(self):
        for prefix in ('88f47a10', '3153b5b3', 'd33f667d', 'fd5c6932', '25e5c600', 'e1d68dde', 'a91d4c7b', '4a60c567', 'b6c4361a'):
            protocol = copy.deepcopy(PROTOCOL_V3); protocol['scene']['logId'] = prefix + '-0000-0000-0000-000000000000'
            with self.assertRaisesRegex(ValueError, 'Held-out'):
                tool.along_ray_uncertainty(None, None, None, None, None, None, None, protocol, 60.)

    def build_v3(self, name='a', **options):
        args = write_fixture(self.root / name, protocol=PROTOCOL_V3, returns=V3_RETURNS, eligible=V3_ELIGIBLE, **options)
        result = tool.build(**args)
        return args, result, json.loads((args['output'] / 'ground.json').read_bytes())

    def test_v3_layer_averages_the_current_vehicle_fits_the_band_and_limits_range(self):
        args, result, ground = self.build_v3()
        self.assertEqual((ground['schemaVersion'], ground['placementRangeLimitMeters'], ground['roadToleranceMeters']), (2, 30.0, .08))
        # Averaged pose: the fixture rows jitter in ego x (world X) by 0.004 * ((49 - k) % 3), mean 0.00392 m; the sweep row has 0.
        car = next(c for c in ground['cuboids']['items'] if c['track'] == TRACK); other = next(c for c in ground['cuboids']['items'] if c['track'] != TRACK)
        np.testing.assert_allclose(car['center'], [-4. + .00392, -.85, 17.5], atol=1e-9)
        np.testing.assert_allclose(np.asarray(car['worldFromBox']), [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-12)
        pa = car['poseAverage']; self.assertEqual((pa['rows'], pa['sweepRow']['timestampNs']), (50, str(SWEEP)))
        self.assertAlmostEqual(pa['sweepRow']['lateralM'], -.00392, places=9); self.assertAlmostEqual(pa['sweepRow']['alongM'], 0, places=9)
        self.assertNotIn('poseAverage', other); self.assertEqual(result['averagedTracks'], [TRACK])
        # Band: the fit set is the 100 grid returns plus fixture rows 0 and 5 (dz 0.0015625 and 0.0515625).
        a = ground['alongRayUncertainty']; fit = a['fit']
        expected = np.concatenate([GRID_DZ, [GROUND_Y - RASTER_Y, GROUND_Y + .05 - RASTER_Y]])
        q = float(np.percentile(np.abs(expected), 95, method='inverted_cdf'))
        self.assertEqual((fit['count'], fit['removedByGroundGate']), (102, 2)); self.assertAlmostEqual(fit['q95AbsM'], q, places=12)
        self.assertEqual(a['heightBandMeters'], math.ceil(q * 1000 - 1e-9) / 1000); self.assertEqual(result['heightBandMeters'], a['heightBandMeters'])
        self.assertEqual(a['method'], 'fitThisDevelopmentScene')
        bands = [b['band'] for b in a['coverage']['fractionalClick']]; self.assertEqual(bands, ['0-10 m', '10-12 m', '12-25 m', '25-30 m'])
        # Overshooting rays near 30 m are refused by the limit, never placed beyond it.
        self.assertIn('Too far to place (over 30 m).', a['coverage']['fractionalClick'][-1]['unknown'])
        self.assertEqual(set(a['coverage']['splitHalf']), {'fitEvenTestOdd', 'fitOddTestEven'})
        self.assertTrue(any('average of all its annotations' in t for t in ground['limitations']))

    def test_v3_click_errors_match_the_flat_raster_analytically(self):
        args, _, ground = self.build_v3()
        scene = json.loads(Path(args['scene_path']).read_bytes()); t = ground['terrain']
        window = np.frombuffer((args['output'] / 'terrain.bin').read_bytes(), '<f4').reshape(t['rows'], t['columns'])
        terrain = tool.Terrain(window, t['column0'], t['row0'], t['rasterShape'], t['sim2'], ground['frame']['cityOrigin'])
        polygons = [np.asarray(x['outline']) for x in ground['drivableAreas']]
        rows = np.arange(6, 106); positions = V3_RETURNS; uv = uv_of(V3_RETURNS)
        errors = tool.click_errors(scene, terrain, polygons, positions, uv, rows, 60., 30.)['fractional']
        placed = np.isfinite(errors['error']); self.assertGreater(placed.sum(), 90)
        p = positions[rows]; dep = np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2]))
        # The ray through a return dz above (below) the flat raster lands |dz| / tan(depression) beyond (before) it.
        np.testing.assert_allclose(errors['error'][placed], np.abs(GRID_DZ[placed]) / np.tan(dep[placed]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(errors['sinDepression'][placed], np.sin(dep[placed]), atol=1e-12)
        band = ground['alongRayUncertainty']['heightBandMeters']
        table = tool.coverage_table(errors, np.hypot(p[:, 0], p[:, 2]), band, [(0, 30)])[0]
        self.assertEqual(table['covered'], int((errors['error'][placed] <= band / errors['sinDepression'][placed]).sum()))

    def test_v3_build_is_deterministic_and_v2_protocol_keeps_schema_1(self):
        first, _, _ = self.build_v3('a'); second, _, _ = self.build_v3('b')
        for name in ('ground.json', 'terrain.bin', 'scene.json'):
            self.assertEqual((first['output'] / name).read_bytes(), (second['output'] / name).read_bytes())
        old = write_fixture(self.root / 'c', returns=V3_RETURNS, eligible=V3_ELIGIBLE); tool.build(**old)
        ground = json.loads((old['output'] / 'ground.json').read_bytes())
        self.assertEqual(ground['schemaVersion'], 1)
        self.assertFalse({'placementRangeLimitMeters', 'alongRayUncertainty'} & set(ground))
        self.assertFalse(any('poseAverage' in c for c in ground['cuboids']['items']))

    def test_v3_fixed_band_records_held_out_residuals_as_a_test_not_a_fit(self):
        # Protocol v3.1 held-out mode: the band is the fixed development value; this scene's residuals are only a test of it.
        # The default fixture has 2 fit-set returns, far below minimumCount (50), which applies to fits only.
        protocol = copy.deepcopy(PROTOCOL_V3); band = protocol['alongRayUncertainty']['heightBand']
        band.update(method='fixed', heightBandMeters=.075, fittedOn='fixture development scenes')
        args = write_fixture(self.root / 'fixed', protocol=protocol); tool.build(**args)
        ground = json.loads((args['output'] / 'ground.json').read_bytes()); a = ground['alongRayUncertainty']
        self.assertEqual((a['heightBandMeters'], a['method'], a['fittedOn']), (.075, 'fixed', 'fixture development scenes'))
        self.assertNotIn('fit', a); self.assertNotIn('splitHalf', a['coverage'])
        held = a['heldOutResiduals']; dz = np.array([GROUND_Y - RASTER_Y, GROUND_Y + .05 - RASTER_Y])
        self.assertEqual(held['count'], 2); self.assertAlmostEqual(held['maxAbsM'], float(np.abs(dz).max()), places=12)
        self.assertEqual(held['withinBandFraction'], 1.0); self.assertTrue(held['scope'].startswith('test:'))
        self.assertEqual({'fractionalClick', 'integerClick', 'definition'}, set(a['coverage']))
        self.assertTrue(any('fixed height band' in t and 'fixture development scenes' in t for t in ground['limitations']))
        self.assertFalse(any('fitted on this development scene' in t for t in ground['limitations']))
        # An empty test set is recorded as empty, never as a failed fit.
        self.assertEqual(tool.residual_stats([], .075)['count'], 0)
        with self.assertRaisesRegex(ValueError, 'fixed height band needs'):
            broken = copy.deepcopy(protocol); del broken['alongRayUncertainty']['heightBand']['fittedOn']
            tool.build(**write_fixture(self.root / 'broken', protocol=broken))

    def test_v3_moving_track_is_not_averaged(self):
        _, result, ground = self.build_v3(moving=True)
        self.assertEqual(result['averagedTracks'], []); self.assertFalse(any('poseAverage' in c for c in ground['cuboids']['items']))


class ProtocolV34GroundTests(unittest.TestCase):
    """Protocol v3.4: ground.json schemaVersion 3 with carShape (shape-v2), every other field exactly as under v3.2.
    Protocol v3.3 (shape-v1) is history: refused here, it runs only from its committed files (8ec75f8)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_v3_4_without_a_car_shape_differs_from_v3_2_only_in_schema_car_shape_protocol_and_code(self):
        v32 = write_fixture(self.root / 'v32', protocol={**PROTOCOL_V3, 'protocolRevision': '3.2'}, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        v34 = write_fixture(self.root / 'v34', protocol=PROTOCOL_V34, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        old_result = tool.build(**v32); new_result = tool.build(**v34, no_car_shape=True)
        old = json.loads((v32['output'] / 'ground.json').read_bytes()); new = json.loads((v34['output'] / 'ground.json').read_bytes())
        self.assertEqual({k for k in old.keys() | new.keys() if k not in old or k not in new or old[k] != new[k]},
                         {'schemaVersion', 'carShape', 'protocol', 'processingCodeHashes'})
        self.assertEqual((old['schemaVersion'], new['schemaVersion'], new['carShape']), (2, 3, None)); self.assertNotIn('carShape', old)
        self.assertEqual(set(new['processingCodeHashes']) - set(old['processingCodeHashes']), {'california/car_shape.py'})
        self.assertEqual({k: new['processingCodeHashes'][k] for k in old['processingCodeHashes']}, old['processingCodeHashes'])
        self.assertEqual(new['processingCodeHashes']['california/car_shape.py'], sha((ROOT / 'scripts/california/car_shape.py').read_bytes()))
        self.assertEqual((v32['output'] / 'terrain.bin').read_bytes(), (v34['output'] / 'terrain.bin').read_bytes())
        self.assertEqual(new_result['carShape'], None); self.assertNotIn('carShape', old_result)
        self.assertEqual({k: v for k, v in new_result.items() if k != 'carShape'}.keys(), old_result.keys())

    def test_v3_3_is_refused_as_history(self):
        old = write_fixture(self.root / 'v33', protocol={**PROTOCOL_V34, 'protocolRevision': '3.3'}, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        with self.assertRaisesRegex(ValueError, 'Protocol v3.3 \\(shape-v1\\) is history'):
            tool.build(**old, no_car_shape=True)
        self.assertFalse(old['output'].exists())

    def test_v3_4_options_are_checked_before_anything_is_written(self):
        args = write_fixture(self.root / 'a', protocol=PROTOCOL_V34, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        plain = write_fixture(self.root / 'b', protocol=PROTOCOL_V3, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        cases = [(args, {}, 'exactly one of'), (args, dict(no_car_shape=True, car_shape_dir=self.root, car_shape_plan_sha256='a' * 64), 'exactly one of'),
                 (args, dict(car_shape_dir=self.root), 'go together'), (plain, dict(no_car_shape=True), 'only under protocol v3.4'),
                 (plain, dict(car_shape_dir=self.root, car_shape_plan_sha256='a' * 64), 'only under protocol v3.4')]
        for kwargs, options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                tool.build(**kwargs, **options)
        for name, protocol, message in (
                ('pin', {**PROTOCOL_V34, 'carShape': {**PROTOCOL_V34['carShape'], 'shapeProtocol': {'sha256': 'f' * 64}}}, 'frozen shape-v2'),
                ('schema', {**PROTOCOL_V34, 'carShape': {**PROTOCOL_V34['carShape'], 'groundSchemaVersion': 2}}, 'schemaVersion 3'),
                ('section', {**PROTOCOL_V3, 'carShape': PROTOCOL_V34['carShape']}, 'Only protocol v3.4 carries')):
            bad = write_fixture(self.root / name, protocol=protocol, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                tool.build(**bad, no_car_shape=True)
        for kwargs in (args, plain):
            self.assertFalse(kwargs['output'].exists())

    def test_v3_4_measured_car_shape_end_to_end(self):
        args, pre, plan_dir, plan_sha, result = car_shape_layer(self.root / 'a')
        ground_raw = (args['output'] / 'ground.json').read_bytes(); ground = json.loads(ground_raw); shape = ground['carShape']
        # Every field but carShape equals the --no-car-shape layer of the same inputs.
        self.assertEqual({k for k in pre.keys() | ground.keys() if pre.get(k) != ground.get(k)}, {'carShape'})
        self.assertEqual((self.root / 'a/pre/terrain.bin').read_bytes(), (args['output'] / 'terrain.bin').read_bytes())
        self.assertEqual((ground['schemaVersion'], shape['status'], shape['refusal'], shape['track']), (3, 'measured', None, TRACK))
        # Listed sweeps: SWEEP - n * 100 ms for n = 0..10 (and SWEEP + 40 ms, which has no car row). Candidates: n = 0..8, the
        # ego 0..12 m behind SWEEP (box centre 29.77 m away at n = 8, 31.26 m at n = 9); 1.5 m steps (at least 0.25 m) keep all 9; split 5 / 4.
        w = shape['sweeps']
        self.assertEqual((w['candidates'], w['thinned'], w['kept'], len(w['shape']), len(w['scoring']), w['downloadDropped']), (9, 9, 9, 5, 4, []))
        self.assertEqual([int(s['timestampNs']) for s in w['shape']], [SWEEP - n * STEP_NS for n in (8, 6, 4, 2, 0)])
        self.assertEqual((w['planSha256'], shape['protocol']['sha256'], shape['protocol']['placementProtocolVersion']), (plan_sha, args['protocol_sha256'], '3.4'))
        self.assertEqual((shape['grid']['along'], shape['grid']['lateral']), (40, 18))
        tops = {(i // 18, i % 18): v for i, v in enumerate(shape['solidTopM']) if v is not None}
        self.assertEqual(sorted(tops), CAR_CELLS)
        for value in tops.values():  # 1.25 m returns (float16 in the sweep files); the floor may give 1.249
            self.assertTrue(1.249 <= value <= 1.25, value)
        self.assertEqual({shape['counts']['returnSweeps'][i * 18 + j] for i, j in CAR_CELLS}, {5})
        self.assertEqual(result['carShape'], {'status': 'measured', 'track': TRACK, 'refusal': None, 'solidCells': 50, 'cells': 720,
                                              'shapeSweeps': 5, 'planSha256': plan_sha})
        self.assertEqual(ground['processingCodeHashes']['california/car_shape.py'], sha((ROOT / 'scripts/california/car_shape.py').read_bytes()))
        self.assertNotIn(b'park', ground_raw.lower()); self.assertNotIn(b'park', (plan_dir / SHAPE.PLAN_NAME).read_bytes().lower())
        self.assertNotIn(b'park', (plan_dir / SHAPE.RECEIPT_NAME).read_bytes().lower())
        # The plan belongs to these annotations and poses: the static fixture (same averaged box) is refused, no output.
        static = write_fixture(self.root / 'static', protocol=PROTOCOL_V34, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        with self.assertRaisesRegex(ValueError, 'other annotations or poses'):
            tool.build(**static, car_shape_dir=plan_dir, car_shape_plan_sha256=plan_sha)
        self.assertFalse(static['output'].exists())
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            tool.build(**{**args, 'output': self.root / 'a/again'}, car_shape_dir=plan_dir, car_shape_plan_sha256='0' * 64)

    def test_v3_4_standing_ego_gives_too_few_sweeps(self):
        args, _, _, _, result = car_shape_layer(self.root / 'a', approach=False)
        shape = json.loads((args['output'] / 'ground.json').read_bytes())['carShape']
        self.assertEqual((shape['status'], shape['refusal']['code'], shape['sweeps']['candidates'], shape['sweeps']['thinned']),
                         ('refused', 'too-few-sweeps', 11, 1))
        self.assertTrue(all(v is None for v in shape['solidTopM']))
        self.assertEqual((result['carShape']['status'], result['carShape']['refusal'], result['carShape']['solidCells']), ('refused', 'too-few-sweeps', 0))

    def test_v3_4_moving_track_has_no_car_shape(self):
        args = write_fixture(self.root / 'a', moving=True, protocol=PROTOCOL_V34, returns=V3_RETURNS, eligible=V3_ELIGIBLE)
        with self.assertRaisesRegex(ValueError, 'does not apply'):
            tool.build(**args, car_shape_dir=self.root, car_shape_plan_sha256='a' * 64)
        self.assertFalse(args['output'].exists())
        result = tool.build(**args, no_car_shape=True)
        self.assertEqual((result['averagedTracks'], result['carShape']), ([], None))
        ground_raw = (args['output'] / 'ground.json').read_bytes(); log_dir = self.root / 'train' / LOG; log_dir.mkdir(parents=True)
        for source, name in (('annotations.feather', 'annotations.feather'), ('poses.feather', 'city_SE3_egovehicle.feather')):
            (log_dir / name).write_bytes((self.root / 'a/sources' / source).read_bytes())
        with self.assertRaisesRegex(ValueError, 'does not apply'):
            SHAPE.make_plan(LOG, args['output'] / 'ground.json', sha(ground_raw), log_dir / 'annotations.feather',
                            sha((log_dir / 'annotations.feather').read_bytes()), log_dir / 'city_SE3_egovehicle.feather',
                            sha((log_dir / 'city_SE3_egovehicle.feather').read_bytes()), out_root=self.root / 'data',
                            fetch=shape_fixture.no_network, sleep=shape_fixture.no_network)
        self.assertFalse((self.root / 'data').exists())


REAL = ROOT / '.local/evaluation/av2-map-first-2026-10-11'


@unittest.skipUnless((REAL / 'grounded-v2/ground.json').exists() and (REAL / 'assembled/scene.json').exists(), 'real 8c files absent')
class RealGroundTests(unittest.TestCase):
    """grounded-v3.4 (protocol v3.4, shape-v2 carShape) is the app's current 8c folder (checked by ProtocolV34GroundTests);
    grounded-v3.3 (protocol v3.3, shape-v1) was the app's folder from October 13, 2026 until shape-v2; the shape-v2 app
    refuses it. grounded-v3.2 (the app's folder until October 13; the app refuses its schema 2 ground), grounded-v3.1
    (rebuilds byte for byte apart from this script's own hash), grounded-v3, grounded-v2 and grounded/ are pinned history."""

    def test_real_layers_are_pinned_and_reproducible(self):
        for folder, freeze in (('grounded-v3.2', 'protocol-v3.2-freeze.json'), ('grounded-v3.1', 'protocol-v3.1-freeze.json'),
                               ('grounded-v3', 'protocol-v3-freeze.json'),
                               ('grounded-v2', 'protocol-v2-freeze.json'), ('grounded', 'protocol-freeze.json')):
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

    @unittest.skipUnless((REAL / 'grounded-v3.1/ground.json').exists() and (REAL / 'grounded-v3/ground.json').exists(), 'v3.1 layer absent')
    def test_v3_1_layer_differs_from_v3_only_in_its_protocol_and_code_record(self):
        v3 = json.loads((REAL / 'grounded-v3/ground.json').read_bytes()); v31 = json.loads((REAL / 'grounded-v3.1/ground.json').read_bytes())
        self.assertEqual({k for k in v3.keys() | v31.keys() if v3.get(k) != v31.get(k)}, {'protocol', 'processingCodeHashes'})
        self.assertEqual((v31['protocol']['path'], v31['protocol']['sha256']), ('../protocol-v3.1.json', '6a7d3369e75dcb9eb5afbd46a66f94760245fe3b0e34733e0f60a106a12f2d92'))
        self.assertEqual((REAL / 'grounded-v3/terrain.bin').read_bytes(), (REAL / 'grounded-v3.1/terrain.bin').read_bytes())

    @unittest.skipUnless((REAL / 'grounded-v3.2/ground.json').exists() and (REAL / 'grounded-v3.1/ground.json').exists(), 'v3.2 layer absent')
    def test_v3_2_layer_differs_from_v3_1_only_in_its_protocol_and_code_record(self):
        # Protocol v3.2's NaN map-height fix changes no stored value on 8c: same terrain bytes, same support rows and cuboids.
        v31 = json.loads((REAL / 'grounded-v3.1/ground.json').read_bytes()); v32 = json.loads((REAL / 'grounded-v3.2/ground.json').read_bytes())
        self.assertEqual({k for k in v31.keys() | v32.keys() if v31.get(k) != v32.get(k)}, {'protocol', 'processingCodeHashes'})
        self.assertEqual((v32['protocol']['path'], v32['protocol']['sha256']), ('../protocol-v3.2.json', '1cef13e625ba8d0c0c00203c0bda6e256536173c4607ac79807e3e40dd173cf7'))
        old, new = v31['processingCodeHashes'], v32['processingCodeHashes']
        self.assertEqual({k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}, {'california/prepare_av2_ground.py'})
        self.assertEqual((REAL / 'grounded-v3.1/terrain.bin').read_bytes(), (REAL / 'grounded-v3.2/terrain.bin').read_bytes())
        scenes = [json.loads((REAL / folder / 'scene.json').read_bytes()) for folder in ('grounded-v3.1', 'grounded-v3.2')]
        for scene in scenes:
            del scene['measurements']['ground']['sha256']
        self.assertEqual(scenes[0], scenes[1])

    @unittest.skipUnless((REAL / 'grounded-v3.1/ground.json').exists(), 'v3.1 layer absent')
    def test_v3_layer_differs_from_v2_only_where_protocol_v3_says_and_v3_1_rebuilds_byte_for_byte(self):
        v2 = json.loads((REAL / 'grounded-v2/ground.json').read_bytes()); v3_raw = (REAL / 'grounded-v3/ground.json').read_bytes(); v3 = json.loads(v3_raw)
        self.assertEqual({k for k in v2.keys() | v3.keys() if v2.get(k) != v3.get(k)},
                         {'schemaVersion', 'protocol', 'cuboids', 'support', 'alongRayUncertainty', 'placementRangeLimitMeters', 'limitations',
                          'processingCodeHashes'})
        self.assertEqual((v3['schemaVersion'], v3['placementRangeLimitMeters'], v3['protocol']['path']), (2, 30.0, '../protocol-v3.json'))
        self.assertEqual((REAL / 'grounded-v2/terrain.bin').read_bytes(), (REAL / 'grounded-v3/terrain.bin').read_bytes())
        old = {c['track']: c for c in v2['cuboids']['items']}; new = {c['track']: c for c in v3['cuboids']['items']}
        self.assertEqual(old.keys(), new.keys()); changed = [t for t in new if new[t] != old[t]]
        self.assertEqual([t[:8] for t in changed], ['396f15b5'])
        car = new[changed[0]]; pa = car['poseAverage']
        self.assertEqual((pa['rows'], pa['firstTimestampNs'], pa['lastTimestampNs']), (114, '315969943360434000', '315969954660010000'))
        self.assertAlmostEqual(pa['sweepRow']['horizontalM'], .1320, places=4); self.assertAlmostEqual(pa['sweepRow']['alongM'], -.1269, places=4)
        self.assertAlmostEqual(pa['centreSpreadM']['alongStd'], .1044, places=4); self.assertAlmostEqual(pa['centreSpreadM']['lateralStd'], .0495, places=4)
        self.assertAlmostEqual(pa['yawSpreadDeg']['std'], .0109, places=4)
        self.assertEqual((car['length'], car['width'], car['height']), (old[changed[0]]['length'], old[changed[0]]['width'], old[changed[0]]['height']))
        self.assertEqual(car['motion'], old[changed[0]]['motion'])
        a = v3['alongRayUncertainty']; fit = a['fit']
        self.assertEqual((a['heightBandMeters'], a['method'], fit['count'], fit['removedByGroundGate']), (.075, 'fitThisDevelopmentScene', 2426, 65))
        self.assertAlmostEqual(fit['medianM'], -.0081, places=4); self.assertAlmostEqual(fit['sigmaRobustM'], .0296, places=4)
        self.assertAlmostEqual(fit['q95AbsM'], .0745, places=4)
        self.assertEqual([b['coverage'] >= .95 for b in a['coverage']['fractionalClick'] + a['coverage']['integerClick']], [True] * 8)
        self.assertEqual(v3['support']['count'], 3055)
        # Same inputs, same bytes: rebuild into a temporary folder with the recorded command's inputs.
        manifest = REAL / 'prepared/manifest.json'; ann = ROOT / '.local/data/av2/train/8c52d911-fe34-3424-9864-d3fdfac38064/annotations.feather'
        if not ann.exists():
            self.skipTest('8c annotations absent')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'grounded'
            tool.build(REAL / 'assembled/scene.json', sha((REAL / 'assembled/scene.json').read_bytes()), manifest, sha(manifest.read_bytes()),
                       REAL / 'protocol-v3.1.json', json.loads((REAL / 'protocol-v3.1-freeze.json').read_bytes())['sha256'], ann, sha(ann.read_bytes()), out)
            self.assertEqual((out / 'terrain.bin').read_bytes(), (REAL / 'grounded-v3.1/terrain.bin').read_bytes())
            v3 = json.loads((REAL / 'grounded-v3.1/ground.json').read_bytes())
            # Source records are relative to the output folder, so compare them as the files they resolve to.
            def resolved(value, base):
                if isinstance(value, dict):
                    return {k: str((base / v).resolve()) if k == 'path' else resolved(v, base) for k, v in value.items()}
                return [resolved(v, base) for v in value] if isinstance(value, list) else value
            rebuilt = json.loads((out / 'ground.json').read_bytes())
            # The NaN-height outline fix changed only this script's own hash (9dd444a9... recorded in grounded-v3.1); every
            # other value must rebuild exactly.
            ground_code = 'california/prepare_av2_ground.py'
            self.assertEqual(rebuilt['processingCodeHashes'][ground_code], sha((ROOT / 'scripts' / ground_code).read_bytes()))
            self.assertEqual(v3['processingCodeHashes'][ground_code], '9dd444a9d9b2914b1b2920115353a88ab0a578bb105b103b4c66724ffdd6f84d')
            for layer in (rebuilt, v3):
                del layer['processingCodeHashes'][ground_code]
            self.assertEqual(resolved(rebuilt, out), resolved(v3, REAL / 'grounded-v3.1'))
            # scene.json pins ground.json's bytes and hash, which differ only by those relative path strings.
            scenes = [json.loads((base / 'scene.json').read_bytes()) for base in (out, REAL / 'grounded-v3.1')]
            for scene in scenes:
                del scene['measurements']['ground']
            self.assertEqual(scenes[0], scenes[1])

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

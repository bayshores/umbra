"""Synthetic analytic fixtures for the rule-based AV2 scene builder (M5b), plus a real-data check skipped when absent.

The synthetic log: the ego drives along city +x at 10 m/s (20 Hz stereo, 10 Hz LiDAR) toward a mapped crossing whose
near edge is the line x = 40 across a 12 m road (curb at y = -6 on the right). A stationary car waits at the right
curb with its front 1 m before the crossing. Nothing here is real data.
"""
import argparse
import copy
import hashlib
from io import BytesIO
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('build_av2_scene', ROOT / 'scripts/california/build_av2_scene.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)

LOG = '0b0b0b0b-1111-4222-8333-444444444444'
HELD = '0c0c0c0c-1111-4222-8333-444444444444'
CROSSING = '77'
T0 = 1_000_000_000_000_000
CAMERAS = [T0 + k * 50_000_000 for k in range(81)]          # ego x = 0.5 k
SWEEPS = [T0 + 10_000_000 + j * 100_000_000 for j in range(40)]
CAR = 'aaaaaaaa-0000-4000-8000-000000000001'
CAM_R = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])        # camera (x right, y down, z forward) -> ego (x fwd, y left, z up)
PLACEMENT = {'protocolVersion': 3, 'pixelSelection': {'rule': 'exactProjection'},
             'tolerances': {'boundaryEpsilonMeters': 0.01, 'maxRangeMeters': 60.0}, 'placementRange': {'limitMeters': 30.0},
             'currentVehicle': {'trackPrefix': 'unused', 'pose': {'rule': 'averageAllAnnotations'}},
             'declaredDefaults': {'vehicle': {'label': 'SUV', 'dimensions': {'width': 1.9, 'height': 1.8, 'length': 4.9},
                                              'setbackMeters': 6.096, 'curbGapMeters': 0.15, 'reason': 'declared 20 ft default'},
                                  'person': {'label': 'child', 'heightMeters': 1.1, 'curbOffsetMeters': 0.30, 'reason': 'declared default'}}}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ego_x(t):
    return (t - T0) / 1e8


def pt(x, y, z=0.):
    return {'x': float(x), 'y': float(y), 'z': float(z)}


def listing_xml(log, folder, ext, entries):
    """entries: [(timestamp, bytes, md5)] -> public S3 ListBucketResult."""
    contents = ''.join(f'<Contents><Key>datasets/av2/sensor/train/{log}/{folder}/{t}.{ext}</Key><Size>{size}</Size>'
                       f'<ETag>&quot;{etag}&quot;</ETag></Contents>' for t, size, etag in entries)
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f'<KeyCount>{len(entries)}</KeyCount><IsTruncated>false</IsTruncated>{contents}</ListBucketResult>').encode()


def default_evidence(car, car_y, moving):
    """split.json per-site vehicle rows for the synthetic log, written by hand the way make_split.py rounds them (0.1 m):
    the stationary car's greedy cluster (40 sweeps at x = 37.0; moving: seed sweep 10 covers sweeps 0-29, centre x 37.435)
    matches the selection rule only on the right side."""
    if not car or car_y > -1.5:
        return []
    return [{'cat': 'REGULAR_VEHICLE', 'nSweeps': 30 if moving else 40, 'cityXY': [37.4 if moving else 37.0, car_y], 'matchesSelectionRule': True}]


class Fixture:
    """Writes one synthetic log; options change one thing at a time."""

    def __init__(self, root, fx=1000., parallel=False, car=True, car_y=-3.5, moving=False, on_disk=(), role='development',
                 evidence='default', expected_blocked='default'):
        self.root = Path(root); self.log_dir = self.root / 'train' / LOG; self.fx = fx
        for sub in ('map', 'calibration', 'sensors/cameras/stereo_front_left', 'sensors/cameras/stereo_front_right', 'sensors/lidar'):
            (self.log_dir / sub).mkdir(parents=True, exist_ok=True)
        stamps = sorted(set(CAMERAS) | set(SWEEPS))
        feather.write_feather(pa.Table.from_pylist([dict(timestamp_ns=t, qx=0., qy=0., qz=0., qw=1., tx_m=ego_x(t), ty_m=0., tz_m=0.)
                                                    for t in stamps]), self.log_dir / 'city_SE3_egovehicle.feather')
        q = Rotation.from_matrix(CAM_R).as_quat()
        feather.write_feather(pa.Table.from_pylist([dict(sensor_name=f'stereo_front_{side}', qx=q[0], qy=q[1], qz=q[2], qw=q[3],
                                                         tx_m=1.5, ty_m=y, tz_m=1.5) for side, y in (('left', .25), ('right', -.25))]),
                              self.log_dir / 'calibration/egovehicle_SE3_sensor.feather')
        feather.write_feather(pa.Table.from_pylist([dict(sensor_name=f'stereo_front_{side}', fx_px=fx, fy_px=fx, cx_px=799.5, cy_px=399.5,
                                                         width_px=1600, height_px=800) for side in ('left', 'right')]),
                              self.log_dir / 'calibration/intrinsics.feather')
        if parallel:  # crossing runs along the road on the right (like 453e5558 / 20499018)
            edges = {'edge1': [pt(30, -3), pt(45, -3)], 'edge2': [pt(30, -6), pt(45, -6)]}
        else:
            edges = {'edge1': [pt(40, 6), pt(40, -6)], 'edge2': [pt(43, 6), pt(43, -6)]}
        vector_map = {'pedestrian_crossings': {CROSSING: {'id': int(CROSSING), **edges}},
                      'lane_segments': {'500': {'id': 500, 'lane_type': 'VEHICLE', 'is_intersection': False,
                                                'left_lane_boundary': [pt(-20, 1.75), pt(80, 1.75)],
                                                'right_lane_boundary': [pt(-20, -1.75), pt(80, -1.75)]},
                                        '501': {'id': 501, 'lane_type': 'VEHICLE', 'is_intersection': False,  # oncoming lane
                                                'left_lane_boundary': [pt(80, 1.75), pt(-20, 1.75)],
                                                'right_lane_boundary': [pt(80, 5.25), pt(-20, 5.25)]}},
                      'drivable_areas': {'9': {'id': 9, 'area_boundary': [pt(-30, -6), pt(100, -6), pt(100, 6), pt(-30, 6)]}}}
        self.map_path = self.log_dir / 'map' / f'log_map_archive_{LOG}____PAO_city_1.json'
        self.map_path.write_text(json.dumps(vector_map))
        np.save(self.log_dir / 'map' / f'{LOG}_ground_height_surface____PAO.npy', np.zeros((210, 450), np.float16))
        (self.log_dir / 'map' / f'{LOG}___img_Sim2_city.json').write_text(json.dumps({'R': [1.0, 0.0, 0.0, 1.0], 't': [30.0, 30.0], 's': 1 / 0.3}))
        rows = []
        for j, t in enumerate(SWEEPS):
            x = ego_x(t)
            if car:
                drift = 0.03 * j if moving else 0.002 * (j % 3)
                rows.append(dict(timestamp_ns=t, track_uuid=CAR, category='REGULAR_VEHICLE', length_m=4.0, width_m=1.8, height_m=1.5,
                                 qw=1., qx=0., qy=0., qz=0., tx_m=37.0 + drift - x, ty_m=car_y, tz_m=.75, num_interior_pts=200))
            rows.append(dict(timestamp_ns=t, track_uuid='bbbbbbbb-0000-4000-8000-000000000002', category='PEDESTRIAN', length_m=.6,
                             width_m=.6, height_m=1.7, qw=1., qx=0., qy=0., qz=0., tx_m=60. - x, ty_m=5.5, tz_m=.85, num_interior_pts=20))
            rows.append(dict(timestamp_ns=t, track_uuid='cccccccc-0000-4000-8000-000000000003', category='REGULAR_VEHICLE', length_m=4.5,
                             width_m=1.9, height_m=1.6, qw=1., qx=0., qy=0., qz=0., tx_m=10. + .8 * j - x, ty_m=3.5, tz_m=.8,
                             num_interior_pts=150))  # oncoming-side traffic, moving
        feather.write_feather(pa.Table.from_pylist(rows), self.log_dir / 'annotations.feather')
        self.license = self.root / 'LICENSE-AND-ATTRIBUTION.md'; self.license.write_text('Synthetic fixture only.\n')
        self.write_sensors(on_disk)
        rows = default_evidence(car, car_y, moving) if evidence == 'default' else evidence
        blocked = (bool(rows) if rows is not None else None) if expected_blocked == 'default' else expected_blocked
        split = {'sites': [{'role': role, 'logId': LOG, 'crossingId': CROSSING, 'expectedBlocked': blocked,
                            'stationaryAnnotatedVehiclesInViewWithin12m': rows},
                           {'role': 'held-out-blocked-test', 'logId': HELD, 'crossingId': '5', 'expectedBlocked': True,
                            'stationaryAnnotatedVehiclesInViewWithin12m': []}]}
        self.split = self.root / 'split.json'; self.split.write_text(json.dumps(split))
        self.lock = self.root / 'split.lock.json'; self.lock.write_text(json.dumps({'sha256': sha(self.split.read_bytes())}))
        self.placement = self.root / 'placement-protocol.json'; self.placement.write_text(json.dumps(PLACEMENT))
        self.protocol = self.root / 'builder-protocol.json'; self.write_protocol()

    def write_protocol(self, **changes):
        """The frozen builder protocol fields the builder checks: parameters, version, runtime code hashes and pins."""
        doc = {'builderVersion': tool.BUILDER_VERSION, 'parameters': tool.PARAMETERS, 'code': tool.runtime_code_hashes(),
               'split': {'sha256': sha(self.split.read_bytes())}, 'placementProtocol': {'sha256': sha(self.placement.read_bytes())}}
        doc.update(changes)
        self.protocol.write_text(json.dumps(doc))

    def write_lock(self, **changes):
        """A protocol-lock.json beside split.json naming this split, builder protocol and code; returns its sha256."""
        doc = {'split': {'sha256': sha(self.split.read_bytes())},
               'sceneBuilder': {'protocolSha256': sha(self.protocol.read_bytes()), 'code': tool.runtime_code_hashes()}}
        doc.update(changes)
        path = self.split.parent / 'protocol-lock.json'; path.write_text(json.dumps(doc))
        return sha(path.read_bytes())

    def write_sensors(self, on_disk, sensor_root=None):
        """Images and sweeps for on_disk frames under the log directory (or under sensor_root, publisher layout)."""
        base = Path(sensor_root) if sensor_root is not None else self.log_dir
        yy, xx = np.indices((800, 1600)); rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=2).astype(np.uint8)
        records = {'left': {}, 'right': {}, 'lidar': {}}
        for t in on_disk:
            for side in ('left', 'right'):
                path = base / f'sensors/cameras/stereo_front_{side}/{t}.jpg'; path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(path, format='JPEG')
                records[side][t] = path.read_bytes()
            sweep = min(SWEEPS, key=lambda s: (abs(s - t), s))
            path = base / f'sensors/lidar/{sweep}.feather'; path.parent.mkdir(parents=True, exist_ok=True)
            feather.write_feather(pa.Table.from_pylist([dict(x=10. + k, y=-1., z=-1.5, offset_ns=k) for k in range(5)]), path)
            records['lidar'][sweep] = path.read_bytes()
        self.listings = {}
        for side, folder, ext, stamps in (('left', 'sensors/cameras/stereo_front_left', 'jpg', CAMERAS),
                                          ('right', 'sensors/cameras/stereo_front_right', 'jpg', CAMERAS),
                                          ('lidar', 'sensors/lidar', 'feather', SWEEPS)):
            entries = [(t, len(records[side][t]), hashlib.md5(records[side][t]).hexdigest()) if t in records[side] else (t, 1000, 'f' * 32)
                       for t in stamps]
            self.listings[side] = self.root / f'{side}-listing.xml'; self.listings[side].write_bytes(listing_xml(LOG, folder, ext, entries))

    def args(self, output='out', log=LOG, crossing=CROSSING, **extra):
        values = dict(protocol=str(self.protocol), protocol_sha256=sha(self.protocol.read_bytes()), placement_protocol=str(self.placement),
                      placement_protocol_sha256=sha(self.placement.read_bytes()), split=str(self.split), split_lock=str(self.lock),
                      log_id=log, crossing_id=crossing, log_dir=str(self.root / 'train' / log), listing_left=str(self.listings['left']),
                      listing_right=str(self.listings['right']), listing_lidar=str(self.listings['lidar']), license=str(self.license),
                      output=str(self.root / output), map_dir=None, sensor_dir=None, diagnostic_camera_timestamp=None,
                      held_out_run=False, protocol_lock_sha256=None)
        values.update(extra)
        return argparse.Namespace(**values)


def frame_at(x):
    return T0 + int(round(x * 2)) * 50_000_000


# ---------------------------------------------------------------- literal transcriptions of the frozen selection scripts
def frames_py_reference(left, lidar, po, e1, e2):
    """selection-evidence/scripts/frames.py inner loop (unchanged arithmetic); returns (window, best, lidar)."""
    lt = np.array(left); lid = np.array(lidar)
    q = po.reindex(lt, method=None)
    xy = q[['tx_m', 'ty_m']].values; yw = np.arctan2(2 * (q.qw * q.qz + q.qx * q.qy), 1 - 2 * (q.qy ** 2 + q.qz ** 2)).values
    win = []
    for k, t in enumerate(lt):
        fs = []; bs = []
        for e in (e1, e2):
            m = e.mean(0) - xy[k]; f = m[0] * math.cos(yw[k]) + m[1] * math.sin(yw[k]); lat = -m[0] * math.sin(yw[k]) + m[1] * math.cos(yw[k])
            fs.append(f); bs.append(abs(math.degrees(math.atan2(lat, f))))
        near = min(fs) if min(fs) > 0 else None
        if near is None:
            continue
        j = int(np.argmin(fs))
        if 8 <= near <= 25 and bs[j] < 31:
            win.append((t, round(near, 2)))
    win2 = [w for w in win if abs(lid[np.argmin(np.abs(lid - w[0]))] - w[0]) <= 50_000_000] or win
    tb, nb = min(win2, key=lambda x: abs(x[1] - 12))
    return win, (int(tb), float(nb)), int(lid[np.argmin(np.abs(lid - tb))])


def stationary_py_reference(an, po):
    """selection-evidence/scripts/stationary.py clustering (unchanged arithmetic)."""
    veh = {'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER', 'SCHOOL_BUS', 'ARTICULATED_BUS',
           'MESSAGE_BOARD_TRAILER', 'RAILED_VEHICLE'}
    an = an[an.category.isin(veh)].copy()
    q = po.loc[an.timestamp_ns.values]
    qw, qx, qy, qz = [q[c].values for c in ('qw', 'qx', 'qy', 'qz')]
    r00 = 1 - 2 * (qy * qy + qz * qz); r01 = 2 * (qx * qy - qz * qw); r10 = 2 * (qx * qy + qz * qw); r11 = 1 - 2 * (qx * qx + qz * qz)
    an['cx'] = r00 * an.tx_m.values + r01 * an.ty_m.values + q.tx_m.values
    an['cy'] = r10 * an.tx_m.values + r11 * an.ty_m.values + q.ty_m.values
    pts = an[['cx', 'cy']].values; ts = an.timestamp_ns.values; used = np.zeros(len(pts), bool); clusters = []
    for i in np.argsort(ts):
        if used[i]:
            continue
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]); m = (d < 0.6) & ~used
        uts = np.unique(ts[m])
        if len(uts) >= 30:
            used |= m
            clusters.append({'c': pts[m].mean(0), 'nSweeps': int(len(uts)), 't0': int(uts[0]), 't1': int(uts[-1])})
    return clusters


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_builder(self, fixture, **extra):
        selection = tool.build(fixture.args(**extra))
        out = Path(fixture.args(**extra).output)
        return selection, out

    def test_twelve_metre_frame_chosen_when_everything_fits_and_plan_is_consumable(self):
        target = frame_at(28.0)  # near edge 12.0 m ahead of the ego origin
        fx = Fixture(self.root, on_disk=[target])
        selection, out = self.run_builder(fx)
        self.assertEqual(selection['status'], 'plan-written')
        chosen = selection['chosen']
        self.assertEqual((chosen['cameraTimestampNs'], chosen['rankInWindow'], chosen['nearEdgeM']), (str(target), 1, 12.0))
        self.assertEqual(chosen['lidarTimestampNs'], str(min(SWEEPS, key=lambda s: (abs(s - target), s))))
        self.assertEqual(chosen['egoLaneId'], '500')
        self.assertEqual((chosen['currentVehicle']['status'], chosen['currentVehicle']['track'], chosen['currentVehicle']['trackPrefix']),
                         ('annotated', CAR, CAR[:8]))
        self.assertAlmostEqual(chosen['currentVehicle']['frontGapToCrossingM'], 1.0, delta=0.005)  # averaged pose (2 mm jitter)
        self.assertEqual((chosen['currentVehicle']['pose'], chosen['currentVehicle']['poseAverageRows']), ('averageAllAnnotations', 40))
        frame = next(e for e in selection['evaluatedFrames'] if e['feasible'])
        origin = np.array(frame['diagnostics']['originCity'])
        np.testing.assert_allclose(origin, [28.0 + 1.5, .25, 1.5], atol=1e-9)
        # Anchors are the suggestion script's desired points: crossing on the car centre line, car front, person 0.3 m from the curb.
        for name, city_xy in (('crossing', (40., -3.5)), ('vehicleFront', (39., -3.5)), ('person', (40., -5.7))):
            world = frame['anchors'][name]['world']
            np.testing.assert_allclose([world[0] + origin[0], -world[2] + origin[1], world[1] + origin[2]], [*city_xy, 0.], atol=5e-3)
        self.assertEqual(selection['protocolSceneValues']['scene']['crop'], chosen['crop'])
        self.assertEqual(selection['protocolSceneValues']['currentVehicleTrackPrefix'], CAR[:8])
        plan = json.loads((out / 'plan.json').read_bytes())
        self.assertEqual(list(plan), ['schemaVersion', 'sourceClass', 'logId', 'crossingId', 'cameraTimestampNs', 'lidarTimestampNs',
                                      'rectifiedSize', 'crop', 'sources'])
        self.assertEqual((plan['rectifiedSize'], plan['crop'][2:]), ([800, 400], [736, 320]))
        self.assertEqual(set(plan['sources']), tool.prepare_av2.ROLES)
        for spec in plan['sources'].values():
            self.assertEqual(sha(Path(spec['path']).read_bytes()), spec['sha256'])
        manifest = tool.prepare_av2.prepare(out / 'plan.json', self.root / 'prepared')  # the unchanged pipeline consumes it
        self.assertEqual([manifest['crop'][k] for k in ('x', 'y', 'width', 'height')], plan['crop'])
        self.assertEqual(selection['frameWindow']['frozenFramesRuleChoice']['cameraTimestampNs'], str(target))
        for path in out.iterdir():
            self.assertNotIn(b'park', path.read_bytes().lower())

    def test_narrow_view_moves_to_the_nearest_feasible_frame_farther_back(self):
        fx = Fixture(self.root, fx=2000.)
        selection, out = self.run_builder(fx)
        self.assertEqual(selection['status'], 'sources-missing')
        chosen = selection['chosen']; frames = selection['evaluatedFrames']
        self.assertGreater(chosen['rankInWindow'], 1); self.assertGreater(chosen['nearEdgeM'], 12.0)
        for e in frames[:chosen['rankInWindow'] - 1]:
            self.assertFalse(e['feasible']); self.assertTrue(any('Outside the source-supported crop' in r for r in e['reasons']))
        self.assertTrue(frames[chosen['rankInWindow'] - 1]['feasible'])
        self.assertEqual(selection['frameWindow']['frozenFramesRuleChoice']['nearEdgeM'], 12.0)
        self.assertEqual({m['role'] for m in selection['missingSources']}, {'left', 'right', 'lidar'})
        self.assertTrue(all(m['listing']['key'].endswith(('.jpg', '.feather')) for m in selection['missingSources']))
        self.assertFalse((out / 'plan.json').exists())

    def test_parallel_crossing_has_no_feasible_frame(self):
        selection, out = self.run_builder(Fixture(self.root, parallel=True))
        self.assertEqual(selection['status'], 'no-feasible-frame'); self.assertIsNone(selection['chosen'])
        reasons = {r for e in selection['evaluatedFrames'] for r in e['reasons']}
        self.assertTrue(all('parallel' in r or 'does not cross the ego lane' in r for r in reasons), reasons)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['selection.json'])

    def test_a_selection_rule_vehicle_failing_the_protocol_is_not_swapped_for_a_default(self):
        selection, _ = self.run_builder(Fixture(self.root, moving=True))  # frozen-rule cluster, but 1.17 m > 1.0 m displacement
        self.assertEqual(selection['splitRule']['selectionRuleTracks'], [CAR])
        self.assertEqual(selection['status'], 'no-feasible-frame')
        failures = selection['evaluatedFrames'][0]['vehicleChecks']['failures'][CAR]
        self.assertTrue(any('not stationary' in f for f in failures), failures)

    def test_without_a_selection_rule_vehicle_the_declared_default_is_used(self):
        for name, options in (('none', dict(car=False)), ('left', dict(car_y=3.5))):
            with self.subTest(name):
                fx = Fixture(self.root / name, **options)
                selection, _ = self.run_builder(fx)
                self.assertEqual(selection['splitRule']['selectionRuleTracks'], [])
                self.assertEqual(selection['splitCrossCheck']['status'], 'agrees')
                self.assertIsNotNone(selection['chosen'], selection['status'])
                self.assertEqual(selection['chosen']['currentVehicle']['status'], 'declared-default')
                self.assertIsNone(selection['protocolSceneValues']['currentVehicleTrackPrefix'])

    def test_refusals(self):
        fx = Fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'Held-out'):  # refused before any held-out file is opened (directory does not exist)
            tool.build(fx.args(log=HELD, crossing='5'))
        self.assertFalse((self.root / 'train' / HELD).exists())
        bad = copy.deepcopy(tool.PARAMETERS); bad['frame']['nearEdgeTargetM'] = 15.0
        fx.write_protocol(parameters=bad)
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            tool.build(fx.args(output='bad-protocol'))
        fx.write_protocol()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):  # the pinned builder protocol hash
            tool.build(fx.args(output='bad-hash', protocol_sha256='0' * 64))
        original = fx.split.read_text(); fx.split.write_text(original + ' ')
        with self.assertRaisesRegex(ValueError, 'split.lock'):
            tool.build(fx.args(output='bad-split'))
        fx.lock.write_text(json.dumps({'sha256': sha(fx.split.read_bytes())}))
        with self.assertRaisesRegex(ValueError, 'frozen against another split'):  # split and its lock agree, the protocol pins the old one
            tool.build(fx.args(output='other-split'))
        fx.split.write_text(original); fx.lock.write_text(json.dumps({'sha256': sha(fx.split.read_bytes())}))
        with self.assertRaisesRegex(ValueError, 'not in the frozen split'):
            tool.build(fx.args(output='bad-site', crossing='78'))
        (self.root / 'exists').mkdir()
        with self.assertRaisesRegex(ValueError, 'Existing output'):
            tool.build(fx.args(output='exists'))
        with self.assertRaisesRegex(ValueError, 'not in the frozen frame window'):
            tool.build(fx.args(output='bad-diagnostic', diagnostic_camera_timestamp=frame_at(1.0)))

    def test_frozen_code_hashes_are_enforced(self):
        fx = Fixture(self.root)
        code = tool.runtime_code_hashes()
        self.assertEqual(set(code), set(tool.RUNTIME_CODE))
        self.assertEqual(code['scripts/california/build_av2_scene.py'], sha(Path(tool.__file__).read_bytes()))
        for name in ('scripts/california/build_av2_scene.py', 'scripts/california/prepare_av2_ground.py', 'scripts/sponsors/common.py'):
            with self.subTest(name):
                fx.write_protocol(code={**code, name: '0' * 64})
                with self.assertRaisesRegex(ValueError, 'Frozen code hash mismatch.*' + re.escape(name)):
                    tool.build(fx.args(output='changed-' + Path(name).stem))
        fx.write_protocol(code={k: v for k, v in code.items() if k != 'scripts/remap_support.py'})
        with self.assertRaisesRegex(ValueError, 'every runtime code file'):
            tool.build(fx.args(output='missing-code'))
        fx.write_protocol(code=None)
        with self.assertRaisesRegex(ValueError, 'every runtime code file'):
            tool.build(fx.args(output='no-code'))
        self.assertFalse(any((self.root / n).exists() for n in ('changed-build_av2_scene', 'missing-code', 'no-code')))

    def test_placement_protocol_must_be_the_pinned_one(self):
        fx = Fixture(self.root)
        other = self.root / 'other-placement.json'; other.write_text(json.dumps({**PLACEMENT, 'note': 'edited'}))
        with self.assertRaisesRegex(ValueError, 'pinned in the builder protocol'):
            tool.build(fx.args(placement_protocol=str(other), placement_protocol_sha256=sha(other.read_bytes())))

    def test_held_out_mode_requires_a_valid_protocol_lock(self):
        fx = Fixture(self.root, role='held-out-blocked-test')  # the synthetic log itself is held out here
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            tool.build(fx.args(output='no-flag'))
        with self.assertRaisesRegex(ValueError, 'protocol-lock-sha256'):  # the flag alone is never enough
            tool.build(fx.args(output='flag-only', held_out_run=True))
        with self.assertRaisesRegex(ValueError, 'protocol-lock.json beside split.json.*does not exist'):
            tool.build(fx.args(output='no-lock', held_out_run=True, protocol_lock_sha256='a' * 64))
        good = fx.write_lock()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            tool.build(fx.args(output='lock-hash', held_out_run=True, protocol_lock_sha256='a' * 64))
        for name, change, message in (('split', {'split': {'sha256': 'b' * 64}}, 'split.json sha256'),
                                      ('protocol', {'sceneBuilder': {'protocolSha256': 'c' * 64, 'code': tool.runtime_code_hashes()}},
                                       'builder protocol sha256'),
                                      ('code', {'sceneBuilder': {'protocolSha256': sha(fx.protocol.read_bytes()),
                                                                 'code': {**tool.runtime_code_hashes(), 'scripts/remap_support.py': 'd' * 64}}},
                                       'runtime code hashes')):
            with self.subTest(name):
                lock_sha = fx.write_lock(**change)
                with self.assertRaisesRegex(ValueError, message):
                    tool.build(fx.args(output='lock-' + name, held_out_run=True, protocol_lock_sha256=lock_sha))
        good = fx.write_lock()
        selection, _ = self.run_builder(fx, output='locked', held_out_run=True, protocol_lock_sha256=good)
        self.assertEqual(selection['heldOutLock']['sha256'], good)
        self.assertEqual(Path(selection['heldOutLock']['path']), (self.root / 'protocol-lock.json').resolve())
        self.assertEqual(selection['site']['role'], 'held-out-blocked-test')
        # A development site with the flag still needs the lock.
        dev = Fixture(self.root / 'dev')
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            tool.build(dev.args(held_out_run=True, protocol_lock_sha256='a' * 64))

    def test_split_evidence_disagreement_stops_the_site(self):
        cases = (('evidence-lists-no-vehicle', dict(evidence=[], expected_blocked=False), 'differ from split.json'),
                 ('evidence-centre-elsewhere', dict(evidence=[{'cat': 'REGULAR_VEHICLE', 'nSweeps': 40, 'cityXY': [37.1, -3.5],
                                                               'matchesSelectionRule': True}]), 'differ from split.json'),
                 ('evidence-other-sweep-count', dict(evidence=[{'cat': 'REGULAR_VEHICLE', 'nSweeps': 39, 'cityXY': [37.0, -3.5],
                                                                'matchesSelectionRule': True}]), 'differ from split.json'),
                 ('expected-blocked-without-vehicle', dict(car=False, evidence=[], expected_blocked=True), 'expectedBlocked is True'),
                 ('held-out-without-evidence', dict(role='held-out-control', evidence=None, expected_blocked=False), 'no per-site vehicle evidence'))
        for name, options, message in cases:
            with self.subTest(name):
                fx = Fixture(self.root / name, **options)
                extra = {}
                if options.get('role'):
                    extra = dict(held_out_run=True, protocol_lock_sha256=fx.write_lock())
                selection, out = self.run_builder(fx, **extra)
                self.assertEqual(selection['status'], 'split-rule-mismatch')
                self.assertEqual(selection['splitCrossCheck']['status'], 'mismatch')
                self.assertIn(message, selection['splitCrossCheck']['reason'])
                self.assertEqual(selection['evaluatedFrames'], []); self.assertIsNone(selection['chosen'])
                self.assertEqual(sorted(p.name for p in out.iterdir()), ['selection.json'])
        # Rounding inside the 0.05 m evidence tolerance agrees; a development site may lack evidence (recorded).
        fx = Fixture(self.root / 'rounded', evidence=[{'cat': 'REGULAR_VEHICLE', 'nSweeps': 40, 'cityXY': [37.05, -3.45], 'matchesSelectionRule': True}])
        selection, _ = self.run_builder(fx)
        self.assertEqual(selection['splitCrossCheck']['status'], 'agrees'); self.assertEqual(len(selection['splitCrossCheck']['pairs']), 1)
        fx = Fixture(self.root / 'no-evidence-dev', evidence=None, expected_blocked=None)
        selection, _ = self.run_builder(fx)
        self.assertEqual(selection['splitCrossCheck']['status'], 'no-evidence-development-site')
        self.assertNotEqual(selection['status'], 'split-rule-mismatch')

    def test_sensor_files_may_come_from_a_separate_download_directory(self):
        target = frame_at(28.0); fx = Fixture(self.root)
        downloads = self.root / 'downloads'; fx.write_sensors([target], sensor_root=downloads)
        selection, out = self.run_builder(fx, output='missing')
        self.assertEqual(selection['status'], 'sources-missing')
        self.assertTrue(all(len(m['searchedPaths']) == 1 for m in selection['missingSources']))
        selection, out = self.run_builder(fx, output='found', sensor_dir=[str(downloads)])
        self.assertEqual(selection['status'], 'plan-written')
        plan = json.loads((out / 'plan.json').read_bytes())
        for role in ('left', 'right', 'lidar'):
            self.assertTrue(Path(plan['sources'][role]['path']).is_relative_to(downloads.resolve()), role)
        self.assertTrue(Path(plan['sources']['poses']['path']).is_relative_to(fx.log_dir.resolve()))
        tool.prepare_av2.prepare(out / 'plan.json', self.root / 'prepared')  # the unchanged pipeline consumes it
        # A different copy of the same file in the log directory is refused, never silently preferred.
        copy_path = fx.log_dir / f'sensors/cameras/stereo_front_left/{target}.jpg'
        copy_path.write_bytes((downloads / f'sensors/cameras/stereo_front_left/{target}.jpg').read_bytes()[:-1] + b'\0')
        with self.assertRaisesRegex(ValueError, 'Conflicting left sensor files'):
            tool.build(fx.args(output='conflict', sensor_dir=[str(downloads)]))

    def test_dependency_versions_come_from_the_loaded_modules(self):
        selection, _ = self.run_builder(Fixture(self.root))
        deps = selection['builder']['dependencies']
        self.assertEqual(deps, {'numpy': np.__version__, 'opencv': tool.prepare_av2.cv2.__version__, 'pyarrow': pa.__version__})
        self.assertFalse(hasattr(tool, 'version'))  # importlib.metadata.version is not used (no PackageNotFoundError)
        self.assertEqual(selection['builder']['runtimeCode'], tool.runtime_code_hashes())

    def test_on_disk_file_must_match_the_public_listing(self):
        target = frame_at(28.0); fx = Fixture(self.root, on_disk=[target])
        path = fx.log_dir / f'sensors/cameras/stereo_front_left/{target}.jpg'; path.write_bytes(path.read_bytes() + b'\0')
        with self.assertRaisesRegex(ValueError, 'differs from the public listing'):
            tool.build(fx.args())

    def test_diagnostic_mode_is_labelled_and_writes_a_diagnostic_plan(self):
        target = frame_at(20.0); fx = Fixture(self.root, on_disk=[target])
        selection, out = self.run_builder(fx, diagnostic_camera_timestamp=target)
        self.assertEqual((selection['mode'], selection['status'], selection['chosen']['cameraTimestampNs']),
                         ('diagnostic-forced-frame', 'plan-written', str(target)))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['diagnostic-plan.json', 'selection.json'])
        self.assertEqual(len(selection['evaluatedFrames']), 1); self.assertGreater(selection['chosen']['rankInWindow'], 1)

    def test_frame_window_matches_the_frozen_frames_script(self):
        fx = Fixture(self.root)
        poses = tool.load_poses((fx.log_dir / 'city_SE3_egovehicle.feather').read_bytes())
        edges = tool.crossing_geometry(json.loads(fx.map_path.read_text()), CROSSING)
        left = tool.load_listing(fx.listings['left'].read_bytes()); lidar = tool.load_listing(fx.listings['lidar'].read_bytes())
        window, excluded = tool.frame_window(poses, edges, left, left, lidar, tool.PARAMETERS['frame'])
        po = feather.read_table(fx.log_dir / 'city_SE3_egovehicle.feather').to_pandas().set_index('timestamp_ns')
        win, (tb, nb), li = frames_py_reference(sorted(left), sorted(lidar), po, edges['edge1'][:, :2], edges['edge2'][:, :2])
        self.assertEqual(sorted((w['cameraTimestampNs'], w['nearEdgeM']) for w in window), sorted((int(t), float(n)) for t, n in win))
        self.assertEqual((window[0]['cameraTimestampNs'], window[0]['nearEdgeM'], window[0]['lidarTimestampNs']), (tb, nb, li))
        self.assertEqual(excluded, {'noExactPose': 0, 'noRightImage': 0, 'lidarTooFar': 0})

    def test_stationary_clusters_match_the_frozen_stationary_script(self):
        fx = Fixture(self.root, moving=True)  # 0.03 m/sweep drift splits the car into several greedy clusters
        poses = tool.load_poses((fx.log_dir / 'city_SE3_egovehicle.feather').read_bytes())
        raw = (fx.log_dir / 'annotations.feather').read_bytes()
        mine = tool.stationary_clusters(tool.load_annotations(raw), poses, tool.PARAMETERS['vehicle'])
        an = feather.read_table(BytesIO(raw)).to_pandas()
        po = feather.read_table(fx.log_dir / 'city_SE3_egovehicle.feather').to_pandas().set_index('timestamp_ns')
        ref = stationary_py_reference(an, po)
        self.assertEqual(len(mine), len(ref))
        for a, b in zip(mine, ref):
            np.testing.assert_allclose(a['centre'], b['c'], atol=1e-12)
            self.assertEqual((a['nSweeps'], a['t0'], a['t1']), (b['nSweeps'], b['t0'], b['t1']))

    def test_lane_choice_prefers_non_intersection_aligned_lane(self):
        straight = {'id': '7', 'laneType': 'VEHICLE', 'isIntersection': False,
                    'left': np.array([[1.75, 0, -10], [1.75, 0, 50]]), 'right': np.array([[-1.75, 0, -10], [-1.75, 0, 50]])}
        junction = {**straight, 'id': '3', 'isIntersection': True}
        oncoming = {'id': '1', 'laneType': 'VEHICLE', 'isIntersection': False, 'left': straight['right'][::-1], 'right': straight['left'][::-1]}
        bike = {**straight, 'id': '2', 'laneType': 'BIKE'}
        turn = {'id': '5', 'laneType': 'VEHICLE', 'isIntersection': False,  # straight, then turning: chord about 41 deg off the heading
                'left': np.array([[1.75, 0, -10], [1.75, 0, 2], [30., 0, 22]]), 'right': np.array([[-1.75, 0, -10], [-1.75, 0, 2], [26.5, 0, 22]])}
        lanes = tool.choose_lane([oncoming, bike, junction, straight, turn], np.array([0., 1.]), tool.PARAMETERS['lane']['maxHeadingDeg'])
        self.assertEqual([lane['id'] for lane in lanes], ['7', '3'])
        self.assertEqual([lane['id'] for lane in tool.choose_lane([turn, junction], np.array([0., 1.]), 90.)], ['5', '3'])  # v1 behaviour


class RealDataTests(unittest.TestCase):
    """8c52d911 / 2658849 at the hand-picked frame: the builder's geometry equals the pipeline's (skipped when data is absent)."""
    BASE = ROOT / '.local'
    FROZEN = BASE / f'evaluation/pao-accuracy/scene-builder/builder-protocol-v{tool.BUILDER_VERSION}.json'

    def test_frozen_protocol_matches_this_builder(self):
        freeze = self.FROZEN.with_name(self.FROZEN.stem + '-freeze.json')
        if not (self.FROZEN.exists() and freeze.exists()):
            self.skipTest('frozen builder protocol absent')
        raw = self.FROZEN.read_bytes(); protocol = json.loads(raw)
        self.assertEqual(sha(raw), json.loads(freeze.read_bytes())['sha256'])
        self.assertEqual((protocol['builderVersion'], protocol['parameters']), (tool.BUILDER_VERSION, tool.PARAMETERS))
        # The builder file itself must be the frozen one (shared modules are enforced at run time by check_frozen_protocol).
        self.assertEqual(protocol['code']['scripts/california/build_av2_scene.py'], sha(Path(tool.__file__).read_bytes()))
        self.assertEqual(set(protocol['code']), set(tool.RUNTIME_CODE))

    def test_hand_frame_anchors_match_the_pipeline(self):
        ev = self.BASE / 'evaluation'; log = '8c52d911-fe34-3424-9864-d3fdfac38064'
        paths = {'protocol': self.FROZEN,
                 'placement': ev / 'av2-map-first-2026-10-11/protocol-v3.1.json',
                 'split': ev / 'pao-accuracy/split.json', 'lock': ev / 'pao-accuracy/split.lock.json',
                 'log': self.BASE / 'data/av2/train' / log, 'inv': ev / 'av2-second-observation-2026-10-05/inventories',
                 'ground': ev / 'california-8c-ground-preparation-2026-10-08/sources/map',
                 'scene': ev / 'av2-map-first-2026-10-11/grounded-v2/scene.json',
                 'suggestions': ev / 'av2-map-first-2026-10-11/grounded-v2/suggestions.json',
                 'license': self.BASE / 'data/av2/LICENSE-AND-ATTRIBUTION.md'}
        if not all(p.exists() for p in paths.values()):
            self.skipTest('local 8c development data or frozen builder protocol absent')
        with tempfile.TemporaryDirectory() as temp:
            # Geometry check, not a freeze check: a copy of the frozen protocol with the current runtime code hashes, so a
            # later edit of a shared module does not hide this comparison (the freeze is enforced at run time).
            protocol = json.loads(paths['protocol'].read_bytes()); protocol['code'] = tool.runtime_code_hashes()
            copy_path = Path(temp) / 'builder-protocol.json'; copy_path.write_text(json.dumps(protocol))
            args = argparse.Namespace(protocol=str(copy_path), protocol_sha256=sha(copy_path.read_bytes()),
                                      placement_protocol=str(paths['placement']), placement_protocol_sha256=sha(paths['placement'].read_bytes()),
                                      split=str(paths['split']), split_lock=str(paths['lock']), log_id=log, crossing_id='2658849',
                                      log_dir=str(paths['log']), listing_left=str(paths['inv'] / 'stereo_front_left.xml'),
                                      listing_right=str(paths['inv'] / 'stereo_front_right.xml'), listing_lidar=str(paths['inv'] / 'lidar.xml'),
                                      license=str(paths['license']), output=str(Path(temp) / 'out'), map_dir=[str(paths['ground'])], sensor_dir=None,
                                      diagnostic_camera_timestamp=315969946899927216, held_out_run=False, protocol_lock_sha256=None)
            selection = tool.build(args)
        self.assertEqual(selection['splitCrossCheck']['status'], 'agrees')
        frame = selection['evaluatedFrames'][0]
        self.assertTrue(frame['feasible'], frame['reasons']); self.assertEqual(frame['egoLaneId'], '43186931')
        self.assertEqual(frame['currentVehicle']['trackPrefix'], '396f15b5'); self.assertEqual(frame['lidarTimestampNs'], '315969946859994000')
        scene = json.loads(paths['scene'].read_bytes())
        np.testing.assert_allclose(frame['diagnostics']['originCity'], scene['transform']['cityOrigin'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(frame['cropCamera']['transform']['worldFromCamera'], scene['transform']['worldFromCamera'], rtol=0, atol=1e-12)
        self.assertAlmostEqual(frame['cropCamera']['camera']['fx'], scene['camera']['fx'], places=9)
        x0, y0 = frame['crop'][:2]
        self.assertAlmostEqual(frame['cropCamera']['camera']['cx'] + x0, scene['camera']['cx'] + 287, places=9)
        self.assertAlmostEqual(frame['cropCamera']['camera']['cy'] + y0, scene['camera']['cy'] + 265, places=9)
        # The person anchor does not depend on the vehicle pose: it equals the grounded-v2 suggestion's mapped point exactly.
        suggestions = json.loads(paths['suggestions'].read_bytes())
        np.testing.assert_allclose(frame['anchors']['person']['world'], suggestions['person']['mappedPoint'], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(frame['anchors']['person']['rectifiedUV']) - [287, 265], suggestions['person']['projectedUV'],
                                   rtol=0, atol=1e-6)
        # The vehicle front and crossing point follow the protocol v3 averaged pose: recompute it independently here.
        ann = feather.read_table(paths['log'] / 'annotations.feather').to_pandas(); ann = ann[ann.track_uuid.str.startswith('396f15b5')]
        po = feather.read_table(paths['log'] / 'city_SE3_egovehicle.feather').to_pandas().set_index('timestamp_ns')
        centres, yaws = [], []
        for row in ann.itertuples():
            p = po.loc[row.timestamp_ns]; rot = Rotation.from_quat([p.qx, p.qy, p.qz, p.qw]).as_matrix()
            centres.append(rot @ [row.tx_m, row.ty_m, row.tz_m] + [p.tx_m, p.ty_m, p.tz_m])
            h = (rot @ Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix())[:, 0]; yaws.append(math.atan2(h[1], h[0]))
        centre = np.mean(centres, 0); yaw = math.atan2(np.sin(yaws).sum(), np.cos(yaws).sum()); heading = np.array([math.cos(yaw), math.sin(yaw)])
        front = centre[:2] + float(np.median(ann.length_m)) / 2 * heading
        edge = json.loads(next((paths['log'] / 'map').glob('log_map_archive_*.json')).read_bytes())['pedestrian_crossings']['2658849']['edge1']
        e0, e1 = np.array([edge[0]['x'], edge[0]['y']]), np.array([edge[-1]['x'], edge[-1]['y']])
        t = np.linalg.solve(np.column_stack([heading, e0 - e1]), e0 - centre[:2])[0]; cross = centre[:2] + t * heading
        origin = np.asarray(scene['transform']['cityOrigin'])
        for name, city_xy in (('vehicleFront', front), ('crossing', cross)):
            world = frame['anchors'][name]['world']
            np.testing.assert_allclose([world[0] + origin[0], -world[2] + origin[1]], city_xy, rtol=0, atol=1e-6)
        self.assertEqual(frame['currentVehicle']['poseAverageRows'], len(ann))


if __name__ == '__main__':
    unittest.main()

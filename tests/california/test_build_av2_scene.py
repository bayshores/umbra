"""Synthetic analytic fixtures for the rule-based AV2 scene builder (M5b), plus real-data checks skipped when absent.

The synthetic log: the ego drives along city +x at 10 m/s (20 Hz stereo, 10 Hz LiDAR) toward a mapped crossing whose
near edge is the line x = 40 across a 12 m road (curb at y = -6 on the right). A stationary car waits at the right
curb with its front 1 m before the crossing. Nothing here is real data.

BuilderTests run builder protocol version 3 (the v3 rules, still selectable); V4RuleTests cover the version 4 candidate
rules (candidate-frame vehicle eligibility, Unknown when a real vehicle occupies the default spot, frame target variants);
V41RuleTests cover the version 4.1 rule 1 filter (clusters outside the drivable area and moving tracks are ignored);
V3ReproducibilityTests compare the version 3 path with the frozen builder at commit cb2f497.
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
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

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
                 evidence='default', expected_blocked='default', version=3, variants=None, car_x=37.0, drift=None, pedestrian_at=None):
        self.root = Path(root); self.log_dir = self.root / 'train' / LOG; self.fx = fx
        self.version = version; self.variants = variants or tool.V4_VARIANTS
        self.variant = None if version == 3 else sorted(self.variants)[0]
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
                shift = drift * j if drift is not None else (0.03 * j if moving else 0.002 * (j % 3))
                rows.append(dict(timestamp_ns=t, track_uuid=CAR, category='REGULAR_VEHICLE', length_m=4.0, width_m=1.8, height_m=1.5,
                                 qw=1., qx=0., qy=0., qz=0., tx_m=car_x + shift - x, ty_m=car_y, tz_m=.75, num_interior_pts=200))
            if pedestrian_at is not None:
                rows.append(dict(timestamp_ns=t, track_uuid='dddddddd-0000-4000-8000-000000000004', category='PEDESTRIAN', length_m=.6,
                                 width_m=.6, height_m=1.7, qw=1., qx=0., qy=0., qz=0., tx_m=pedestrian_at[0] - x, ty_m=pedestrian_at[1],
                                 tz_m=.85, num_interior_pts=20))
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
        """The frozen builder protocol fields the builder checks: parameters, version, runtime code hashes and pins.
        Version 3: parameters; version 4 and 4.1: parametersByVariant (one entry per frame-target variant)."""
        if self.version == 3:
            doc = {'builderVersion': 3, 'parameters': tool.PARAMETERS_V3, 'code': tool.runtime_code_hashes(3)}
        else:
            make = tool.v41_parameters if self.version == 4.1 else tool.v4_parameters
            doc = {'builderVersion': self.version, 'parametersByVariant': {n: make(t) for n, t in self.variants.items()},
                   'code': tool.runtime_code_hashes(self.version)}
        doc.update({'split': {'sha256': sha(self.split.read_bytes())}, 'placementProtocol': {'sha256': sha(self.placement.read_bytes())}})
        doc.update(changes)
        self.protocol.write_text(json.dumps(doc))

    def write_lock(self, path=None, **changes):
        """A protocol-lock.json naming this split, builder protocol and code (v4: also variant and approval), beside
        split.json unless path is given; returns its sha256."""
        doc = {'split': {'sha256': sha(self.split.read_bytes())},
               'sceneBuilder': {'protocolSha256': sha(self.protocol.read_bytes()), 'code': tool.runtime_code_hashes(self.version)}}
        if self.version >= 4:
            doc['sceneBuilder']['variant'] = self.variant; doc['approval'] = {'by': 'test approver', 'utc': '2026-10-20T00:00:00Z'}
        doc.update(changes)
        path = Path(path) if path is not None else self.split.parent / 'protocol-lock.json'; path.write_text(json.dumps(doc))
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
                      held_out_run=False, protocol_lock_sha256=None, protocol_variant=self.variant)
        values.update(extra)
        return argparse.Namespace(**values)

    def cli(self, **extra):
        """The same arguments as command-line flags (for running another builder file in a subprocess)."""
        args = vars(self.args(**extra)); flags = []
        for key, value in args.items():
            if value is None or value is False:
                continue
            flag = '--' + key.replace('_', '-')
            if value is True:
                flags.append(flag)
            elif isinstance(value, list):
                for item in value:
                    flags += [flag, str(item)]
            else:
                flags += [flag, str(value)]
        return flags


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
        bad = copy.deepcopy(tool.PARAMETERS_V3); bad['frame']['nearEdgeTargetM'] = 15.0
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
        code = tool.runtime_code_hashes(3)
        self.assertEqual(set(code), set(tool.RUNTIME_CODE_V3))
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
                                      ('protocol', {'sceneBuilder': {'protocolSha256': 'c' * 64, 'code': tool.runtime_code_hashes(3)}},
                                       'builder protocol sha256'),
                                      ('code', {'sceneBuilder': {'protocolSha256': sha(fx.protocol.read_bytes()),
                                                                 'code': {**tool.runtime_code_hashes(3), 'scripts/remap_support.py': 'd' * 64}}},
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
        self.assertEqual(selection['builder']['runtimeCode'], tool.runtime_code_hashes(3))
        self.assertEqual(selection['builder']['version'], 3)

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
        window, excluded = tool.frame_window(poses, edges, left, left, lidar, tool.PARAMETERS_V3['frame'])
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


FROZEN_V3_COMMIT = 'cb2f497'


def frozen_v3_tree(dest):
    """The runtime code files of builder v3 as committed at cb2f497 (git show), in the repository layout under dest;
    None when git or the commit is unavailable."""
    for name in tool.RUNTIME_CODE_V3 + ('scripts/rectified_calibration.py',):
        try:
            data = subprocess.run(['git', '-C', str(ROOT), 'show', f'{FROZEN_V3_COMMIT}:{name}'], check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        path = Path(dest) / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return Path(dest)


def committed_hashes(commit, names):
    """sha256 of each file as committed at commit (git show); None when git or the commit is unavailable."""
    out = {}
    for name in names:
        try:
            out[name] = sha(subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'], check=True, capture_output=True).stdout)
        except (OSError, subprocess.CalledProcessError):
            return None
    return out


def without_builder_identity(selection_bytes, changed=()):
    """selection.json minus the fields that name the builder file or the protocol file itself (its own sha256, the
    runtime code hash of build_av2_scene.py, the builder protocol path/sha256) and the recorded hashes of the shared
    files named in changed (files edited after the freeze, e.g. prepare_av2_ground.py in v4.1): everything else must be
    identical."""
    doc = json.loads(selection_bytes)
    doc['builder'].pop('sha256')
    for name in {'scripts/california/build_av2_scene.py', *changed}:
        doc['builder']['runtimeCode'].pop(name)
    doc['inputs'].pop('builderProtocol')
    return doc


def changed_since(recorded):
    """The runtime files whose current sha256 differs from the recorded one (a dict of path -> sha256)."""
    return tuple(name for name, digest in sorted(recorded.items()) if sha((ROOT / name).read_bytes()) != digest)


class V4RuleTests(unittest.TestCase):
    """Builder protocol version 4 candidate rules on the synthetic log (development data only)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_builder(self, fixture, **extra):
        return tool.build(fixture.args(**extra)), Path(fixture.args(**extra).output)

    def test_v4_12_keeps_the_v3_choice_where_no_rule_differs(self):
        target = frame_at(28.0)
        v3 = Fixture(self.root / 'v3', on_disk=[target]); v4 = Fixture(self.root / 'v4', on_disk=[target], version=4)
        s3, o3 = self.run_builder(v3); s4, o4 = self.run_builder(v4, protocol_variant='v4-12')
        self.assertEqual((s3['status'], s4['status']), ('plan-written', 'plan-written'))
        self.assertEqual((s4['builder']['version'], s4['builder']['variant']), (4, 'v4-12'))
        for key in ('cameraTimestampNs', 'lidarTimestampNs', 'nearEdgeM', 'rankInWindow', 'egoLaneId', 'crop', 'currentVehicle'):
            self.assertEqual(s3['chosen'][key], s4['chosen'][key], key)
        self.assertEqual(s4['chosen']['vehicleRuleAtFrame'], {'referencePoseTracks': [CAR], 'frameTracks': [CAR], 'qualifyingTracks': [CAR]})
        self.assertAlmostEqual(s4['chosen']['captureDistanceM'], 40.0 - 29.5, places=6)  # crossing anchor x 40, camera x 28 + 1.5
        plan3 = json.loads((o3 / 'plan.json').read_bytes()); plan4 = json.loads((o4 / 'plan.json').read_bytes())
        for plan in (plan3, plan4):
            plan['sources']['license']['path'] = 'same'; [plan['sources'][r].update(path='same') for r in plan['sources']]
        self.assertEqual(plan3, plan4)  # same frame, crop and source bytes (paths differ only by the fixture folder)
        self.assertEqual(s4['frameWindow']['framesPy12mChoice']['cameraTimestampNs'], str(target))
        self.assertEqual(s4['firstInTargetOrder']['outcome'], 'feasible')

    def test_rule1_vehicle_in_view_only_at_the_candidate_frame_qualifies(self):
        # Car centre (29, -5): at the reference pose (ego x 21.5, crossing centroid 20 m ahead) it is 7.5 m ahead at a
        # 33.7 deg bearing, outside the split rule's 31 deg view; from ego x < 20.68 it is inside. It overlaps the
        # declared default footprint (x 29.0-33.9, y -5.85 to -3.95).
        options = dict(car_x=29.0, car_y=-5.0, evidence=[], expected_blocked=False)
        s3, _ = self.run_builder(Fixture(self.root / 'v3', **options))
        self.assertEqual((s3['status'], s3['splitRule']['selectionRuleTracks']), ('no-feasible-frame', []))
        self.assertTrue(all(e['reasons'] == ['Annotated REGULAR_VEHICLE occupies the declared default footprint.'] for e in s3['evaluatedFrames']))
        fx = Fixture(self.root / 'v4', version=4, **options)
        s12, _ = self.run_builder(fx, protocol_variant='v4-12', output='v4-12')
        self.assertEqual(s12['status'], 'sources-missing')  # chosen; only the synthetic images are not on disk
        chosen = s12['chosen']; ego = ego_x(int(chosen['cameraTimestampNs'])) + 0.0
        self.assertEqual((chosen['currentVehicle']['status'], chosen['currentVehicle']['track']), ('annotated', CAR))
        self.assertEqual(chosen['vehicleRuleAtFrame'], {'referencePoseTracks': [], 'frameTracks': [CAR], 'qualifyingTracks': [CAR]})
        self.assertLess(math.degrees(math.atan2(5.0, 29.0 - ego)), 31.0)
        self.assertEqual(chosen['nearEdgeM'], 19.5)  # the frame nearest 12 m where the car is in view (ego x 20.5)
        before = s12['evaluatedFrames'][:chosen['rankInWindow'] - 1]
        self.assertTrue(before and all(e['outcome'] == tool.OCCUPIED for e in before))
        self.assertEqual(s12['framesBeforeChosen']['outcomes'], {tool.OCCUPIED: len(before)})
        occupant = before[0]['defaultSpotOccupants'][0]
        self.assertEqual((occupant['track'], occupant['nearCrossingStationaryCluster'], occupant['matchesAtReferencePose'],
                          occupant['matchesAtFrame']), (CAR, True, False, False))
        self.assertTrue(any('outside the split rule' in w for w in occupant['whyNotQualifying']))
        # v4-18.5: capture distance 38.5 - ego x; the vehicle front (31, -5) must also be >= 12 m from the camera, which
        # first holds at ego x 18.5 (12.19 m); x 20 (capture 18.5) and the frames between fail that minimum.
        s18, _ = self.run_builder(fx, protocol_variant='v4-18.5', output='v4-18.5')
        chosen = s18['chosen']
        self.assertEqual(chosen['cameraTimestampNs'], str(frame_at(18.5)))
        self.assertAlmostEqual(chosen['captureDistanceM'], 20.0, places=6)
        self.assertTrue(all(d >= 12.0 for d in chosen['anchorDistancesM'].values()))
        ahead = s18['evaluatedFrames'][:chosen['rankInWindow'] - 1]
        self.assertEqual(ahead[0]['cameraTimestampNs'], str(frame_at(20.0)))
        self.assertTrue(any('closer than the 12 m minimum' in r for r in ahead[0]['reasons']), ahead[0]['reasons'])
        for e in ahead:  # each frame ahead is too close (car in view) or Unknown (car out of view at ego x >= 20.68)
            in_view = ego_x(int(e['cameraTimestampNs'])) < 20.68
            self.assertEqual(e['outcome'], 'infeasible' if in_view else tool.OCCUPIED, e['reasons'])
            if in_view:
                self.assertTrue(any('closer than the 12 m minimum' in r for r in e['reasons']), e['reasons'])

    def test_rule2_a_real_vehicle_in_the_default_spot_makes_the_site_unknown(self):
        # A slowly creeping car (0.05 m per sweep, 2 m in all) forms no stationary cluster, so it never qualifies, and it
        # overlaps the declared default footprint at every sweep.
        options = dict(car_x=31.0, car_y=-5.0, drift=0.05, evidence=[], expected_blocked=False)
        s3, _ = self.run_builder(Fixture(self.root / 'v3', **options))
        self.assertEqual(s3['status'], 'no-feasible-frame')
        for variant in ('v4-12', 'v4-18.5'):
            with self.subTest(variant):
                selection, out = self.run_builder(Fixture(self.root / variant, version=4, **options), protocol_variant=variant)
                self.assertEqual(selection['status'], tool.OCCUPIED)
                self.assertEqual((selection['result']['state'], selection['result']['reason']), ('Unknown', 'a real vehicle occupies the default spot'))
                self.assertEqual(selection['result']['framesWithThisOutcome'], len(selection['evaluatedFrames']))
                self.assertEqual(selection['result']['reportedFrame']['rankInTargetOrder'], 1)
                occupant = selection['result']['occupants'][0]
                self.assertEqual((occupant['track'], occupant['inStationaryCluster']), (CAR, False))
                self.assertEqual(occupant['whyNotQualifying'], ['not a stationary annotated vehicle cluster within 12 m of the crossing (split rule)'])
                self.assertIsNone(selection['chosen']); self.assertEqual(sorted(p.name for p in out.iterdir()), ['selection.json'])
                text = (out / 'selection.json').read_bytes()
                self.assertNotIn(b'park', text.lower()); self.assertNotIn(b'"Clear"', text)
        # Only vehicles: a pedestrian in the default spot keeps the v3 reason (frame not feasible, site not Unknown).
        selection, _ = self.run_builder(Fixture(self.root / 'ped', version=4, car=False, pedestrian_at=(31.0, -5.0), evidence=[]),
                                        protocol_variant='v4-12')
        self.assertEqual(selection['status'], 'no-feasible-frame')
        self.assertIn('Annotated PEDESTRIAN occupies the declared default footprint.', selection['evaluatedFrames'][0]['reasons'])

    def test_rule3_frame_target_is_a_protocol_parameter(self):
        variants = {**tool.V4_VARIANTS, 'capture-12-min-12': {'measure': 'captureDistanceAlongLane', 'targetM': 12.0, 'roundingDecimals': 2,
                                                              'minAnchorDistanceM': 12.0}}
        fx = Fixture(self.root, version=4, variants=variants)
        selection, _ = self.run_builder(fx, protocol_variant='v4-18.5', output='18.5')
        chosen = selection['chosen']  # crossing anchor (40, -3.5): capture distance = 40 - (ego x + 1.5) = 18.5 at ego x 20
        self.assertEqual((chosen['cameraTimestampNs'], chosen['rankInWindow']), (str(frame_at(20.0)), 1))
        self.assertAlmostEqual(chosen['captureDistanceM'], 18.5, places=6)
        self.assertEqual(selection['frameWindow']['target'], tool.V4_VARIANTS['v4-18.5'])
        keys = [abs(round(e['captureDistanceM'], 2) - 18.5) for e in selection['evaluatedFrames'] if 'captureDistanceM' in e]
        self.assertEqual(keys, sorted(keys))
        # Same log, capture 12 m with every anchor >= 12 m: ego x 26.5 puts the vehicle front (39, -3.5) 11.62 m away.
        selection, _ = self.run_builder(fx, protocol_variant='capture-12-min-12', output='12')
        self.assertEqual(selection['chosen']['cameraTimestampNs'], str(frame_at(26.0)))
        first = selection['evaluatedFrames'][0]
        self.assertEqual(first['cameraTimestampNs'], str(frame_at(26.5)))
        self.assertTrue(any('closer than the 12 m minimum' in r and 'vehicleFront 11.62 m' in r for r in first['reasons']), first['reasons'])

    def test_v4_protocol_is_checked(self):
        fx = Fixture(self.root, version=4)
        with self.assertRaisesRegex(ValueError, 'protocol-variant'):
            tool.build(fx.args(output='no-variant', protocol_variant=None))
        with self.assertRaisesRegex(ValueError, 'protocol-variant'):
            tool.build(fx.args(output='other-variant', protocol_variant='v4-15'))
        for name, target, message in (('outside-window', {**tool.V4_VARIANTS['v4-12'], 'targetM': 30.0}, 'inside the frame window'),
                                      ('unknown-measure', {**tool.V4_VARIANTS['v4-12'], 'measure': 'egoSpeed'}, 'measure'),
                                      ('extra-key', {**tool.V4_VARIANTS['v4-12'], 'note': 'x'}, 'exactly'),
                                      ('integer-target', {**tool.V4_VARIANTS['v4-12'], 'targetM': 12}, 'inside the frame window')):
            with self.subTest(name):
                params = tool.v4_parameters(tool.V4_VARIANTS['v4-12']); params['frame']['target'] = target
                fx.write_protocol(parametersByVariant={'v4-12': params})
                with self.assertRaisesRegex(ValueError, message):
                    tool.build(fx.args(output=name, protocol_variant='v4-12'))
        params = tool.v4_parameters(tool.V4_VARIANTS['v4-12']); params['crop']['width'] = 640
        fx.write_protocol(parametersByVariant={'v4-12': params})
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            tool.build(fx.args(output='crop', protocol_variant='v4-12'))
        fx.write_protocol(code=tool.runtime_code_hashes(3))  # v4 must also hash rectified_calibration.py
        with self.assertRaisesRegex(ValueError, 'every runtime code file'):
            tool.build(fx.args(output='v3-code'))
        fx.write_protocol(parameters=tool.PARAMETERS_V3)
        with self.assertRaisesRegex(ValueError, 'parametersByVariant'):
            tool.build(fx.args(output='flat'))
        v3 = Fixture(self.root / 'v3')
        with self.assertRaisesRegex(ValueError, 'no variants'):
            tool.build(v3.args(protocol_variant='v4-12'))
        self.assertIn('scripts/rectified_calibration.py', tool.runtime_code_hashes(4))
        self.assertFalse(any((self.root / n).exists() for n in ('no-variant', 'crop', 'v3-code', 'flat')))

    def test_v4_imported_modules_must_be_the_hashed_files(self):
        fx = Fixture(self.root, version=4)
        shadow = types.ModuleType('remap_support'); shadow.__file__ = str(self.root / 'scripts/california/remap_support.py')
        with mock.patch.dict(sys.modules, {'remap_support': shadow}):
            with self.assertRaisesRegex(ValueError, 'Module remap_support was loaded from'):
                tool.build(fx.args(output='shadow'))
        shadow = types.ModuleType('rectified_calibration'); shadow.__file__ = str(self.root / 'rectified_calibration.py')
        with mock.patch.dict(sys.modules, {'rectified_calibration': shadow}):
            with self.assertRaisesRegex(ValueError, 'Module rectified_calibration was loaded from'):
                tool.build(fx.args(output='shadow2'))
        self.assertFalse((self.root / 'shadow').exists())

    def test_v4_held_out_lock_has_one_path_a_variant_and_an_approval(self):
        fx = Fixture(self.root, version=4, role='held-out-blocked-test')
        fixed = self.root / 'locks' / 'protocol-lock.json'; fixed.parent.mkdir()
        with mock.patch.object(tool, 'LOCK_PATH_V4', fixed):
            beside = fx.write_lock()  # beside split.json, as v3 accepts: v4 reads only the fixed path
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                tool.build(fx.args(output='beside', held_out_run=True, protocol_lock_sha256=beside))
            for name, change, message in (('variant', {'sceneBuilder': {'protocolSha256': sha(fx.protocol.read_bytes()), 'code': tool.runtime_code_hashes(4),
                                                                        'variant': 'v4-18.5'}}, 'variant'),
                                          ('approval', {'approval': {'by': '', 'utc': '2026-10-20T00:00:00Z'}}, 'approval'),
                                          ('v3-code', {'sceneBuilder': {'protocolSha256': sha(fx.protocol.read_bytes()), 'code': tool.runtime_code_hashes(3),
                                                                        'variant': 'v4-12'}}, 'runtime code hashes')):
                with self.subTest(name):
                    lock_sha = fx.write_lock(path=fixed, **change)
                    with self.assertRaisesRegex(ValueError, message):
                        tool.build(fx.args(output='lock-' + name, held_out_run=True, protocol_lock_sha256=lock_sha))
            good = fx.write_lock(path=fixed)
            selection, _ = self.run_builder(fx, output='locked', held_out_run=True, protocol_lock_sha256=good)
        self.assertEqual((selection['heldOutLock']['path'], selection['heldOutLock']['sha256']), (str(fixed), good))
        self.assertEqual(tool.LOCK_PATH_V4, ROOT / '.local/evaluation/pao-accuracy/protocol-lock.json')


class V41RuleTests(unittest.TestCase):
    """Builder protocol version 4.1: rule 1 ignores a near-crossing stationary cluster whose centre is outside the mapped
    drivable area (y below -6 here) and any of its tracks that moves more than 1.0 m from its first annotated position.
    Both cases are forced to the window frame at ego x 16 (near edge 24 m), where the car is inside the split rule's view
    but not at the reference pose (ego x 21.5), so only rule 1 can make it qualify."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, version, name, **options):
        fx = Fixture(self.root / f'{name}-{version}', version=version, evidence=[], expected_blocked=False, **options)
        selection = tool.build(fx.args(output='out', diagnostic_camera_timestamp=frame_at(16.0)))
        self.assertEqual((selection['mode'], len(selection['evaluatedFrames'])), ('diagnostic-forced-frame', 1))
        self.assertEqual(selection['splitRule']['selectionRuleTracks'], [])
        return selection, selection['evaluatedFrames'][0]

    def test_v4_1_ignores_a_moving_track(self):
        # 0.034 m per sweep: one stationary cluster from sweeps 0-29 (centre x 28.49), but the track moves 1.326 m in all.
        options = dict(car_x=28.0, car_y=-5.0, drift=0.034)
        s4, f4 = self.frame(4, 'moving', **options)
        self.assertEqual(f4['vehicleRuleAtFrame']['frameTracks'], [CAR])
        self.assertEqual(f4['outcome'], 'infeasible')
        self.assertIn('No qualifying stationary vehicle passes the placement protocol checks at this sweep.', f4['reasons'])
        self.assertIn('Track is not stationary under the protocol (>= 3.0 s, <= 1.0 m).', f4['vehicleChecks']['failures'][CAR])
        s41, f41 = self.frame(4.1, 'moving', **options)
        self.assertEqual(s41['builder']['version'], 4.1)
        rule = f41['vehicleRuleAtFrame']
        self.assertEqual((rule['frameTracks'], rule['qualifyingTracks'], rule['ignoredFrameTracks']), ([], [], [CAR]))
        self.assertEqual(f41['outcome'], tool.OCCUPIED)  # the moving car still occupies the declared default spot
        self.assertEqual((s41['status'], s41['result']['state']), (tool.OCCUPIED, 'Unknown'))
        cluster = s41['frameRuleV41']['clusters'][0]
        self.assertEqual((cluster['tracks'], cluster['centreInsideDrivable'], cluster['ignoredTracks']), ([CAR], True, [CAR]))
        self.assertAlmostEqual(cluster['trackMotion'][CAR]['maxDisplacementFromFirstM'], 39 * 0.034, places=6)
        self.assertEqual(cluster['reasons'][CAR], ['track aaaaaaaa moves 1.33 m from its first annotated position (more than 1 m)'])
        occupant = f41['defaultSpotOccupants'][0]
        self.assertEqual(occupant['track'], CAR)
        self.assertIn('ignored by the builder v4.1 frame rule: track aaaaaaaa moves 1.33 m from its first annotated position (more than 1 m)',
                      occupant['whyNotQualifying'])
        self.assertTrue(any(line.startswith('Builder v4.1:') for line in s41['limitations']))
        self.assertEqual(s41['frameRuleV41']['parameters'], tool.FRAME_RULE_V41)

    def test_v4_1_ignores_a_cluster_outside_the_drivable_area(self):
        options = dict(car_x=29.0, car_y=-6.5)  # stationary, centre 0.5 m outside the drivable outline (y = -6)
        _s4, f4 = self.frame(4, 'outside', **options)
        self.assertEqual(f4['vehicleRuleAtFrame']['frameTracks'], [CAR])
        self.assertIn('Box centre is outside the mapped drivable area.', f4['vehicleChecks']['failures'][CAR])
        self.assertIn('No qualifying stationary vehicle passes the placement protocol checks at this sweep.', f4['reasons'])
        s41, f41 = self.frame(4.1, 'outside', **options)
        self.assertEqual(f41['vehicleRuleAtFrame']['ignoredFrameTracks'], [CAR])
        self.assertNotIn('No qualifying stationary vehicle passes the placement protocol checks at this sweep.', f41['reasons'])
        cluster = s41['frameRuleV41']['clusters'][0]
        self.assertFalse(cluster['centreInsideDrivable'])
        self.assertLessEqual(cluster['trackMotion'][CAR]['maxDisplacementFromFirstM'], 1.0)
        self.assertEqual(cluster['reasons'][CAR], ['cluster centre (city 29.00, -6.50) is outside the mapped drivable area'])
        self.assertEqual(f41['outcome'], tool.OCCUPIED)

    def test_v4_1_leaves_a_qualifying_stationary_car_unchanged(self):
        target = frame_at(28.0)
        s4 = tool.build(Fixture(self.root / 'v4', on_disk=[target], version=4).args())
        s41 = tool.build(Fixture(self.root / 'v41', on_disk=[target], version=4.1).args())
        self.assertEqual((s4['status'], s41['status']), ('plan-written', 'plan-written'))
        for key in ('cameraTimestampNs', 'lidarTimestampNs', 'nearEdgeM', 'rankInWindow', 'egoLaneId', 'crop', 'currentVehicle', 'captureDistanceM'):
            self.assertEqual(s4['chosen'][key], s41['chosen'][key], key)
        self.assertEqual(s41['chosen']['vehicleRuleAtFrame']['ignoredFrameTracks'], [])
        self.assertEqual(s41['frameRuleV41']['clusters'][0]['ignoredTracks'], [])

    def test_v4_1_protocol_is_checked(self):
        fx = Fixture(self.root, version=4.1)
        fx.write_protocol(parametersByVariant={n: tool.v4_parameters(t) for n, t in tool.V4_VARIANTS.items()})
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            tool.build(fx.args(output='v4-parameters'))
        params = tool.v41_parameters(tool.V4_VARIANTS['v4-12']); params['vehicle']['frameRuleIgnore']['trackMaxDisplacementFromFirstOverM'] = 1.5
        fx.write_protocol(parametersByVariant={'v4-12': params})
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            tool.build(fx.args(output='threshold'))
        v4 = Fixture(self.root / 'v4', version=4)
        v4.write_protocol(parametersByVariant={n: tool.v41_parameters(t) for n, t in tool.V4_VARIANTS.items()})
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            tool.build(v4.args(output='v41-in-v4'))
        self.assertEqual(tool.runtime_code_hashes(4.1), tool.runtime_code_hashes(4))
        self.assertFalse(any((self.root / n).exists() for n in ('v4-parameters', 'threshold')))


class V3ReproducibilityTests(unittest.TestCase):
    """A builder version 3 protocol still runs the v3 rules: the same selection and plan as the frozen builder file at
    commit cb2f497 (run as its own process), apart from the fields naming the builder file and protocol file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.tree = frozen_v3_tree(self.root / 'cb2f497')
        if self.tree is None:
            self.skipTest('git or commit cb2f497 unavailable')

    def tearDown(self):
        self.temp.cleanup()

    def run_frozen(self, fixture, protocol, output):
        """The cb2f497 builder file with a protocol naming its own (frozen v3) code hashes."""
        result = subprocess.run([sys.executable, '-I', '-B', str(self.tree / 'scripts/california/build_av2_scene.py'),
                                 *fixture.cli(output=output, protocol=str(protocol), protocol_sha256=sha(protocol.read_bytes()), protocol_variant=None)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return fixture.root / output

    def test_v3_protocol_reproduces_the_cb2f497_builder_on_synthetic_logs(self):
        frozen_code = {name: sha((self.tree / name).read_bytes()) for name in tool.RUNTIME_CODE_V3}
        cases = {'twelve-metre': dict(on_disk=[frame_at(28.0)]), 'narrow-view': dict(fx=2000.), 'parallel': dict(parallel=True),
                 'moving': dict(moving=True), 'left-car': dict(car_y=3.5), 'out-of-view-car': dict(car_x=29.0, car_y=-5.0, evidence=[]),
                 'creeping-car': dict(car_x=31.0, car_y=-5.0, drift=0.05, evidence=[])}
        for name, options in cases.items():
            with self.subTest(name):
                fx = Fixture(self.root / name, **options)
                frozen_protocol = fx.root / 'frozen-protocol.json'
                frozen_protocol.write_text(json.dumps({**json.loads(fx.protocol.read_bytes()), 'code': frozen_code}))
                old = self.run_frozen(fx, frozen_protocol, 'old')
                selection = tool.build(fx.args(output='new')); new = fx.root / 'new'
                self.assertEqual(selection['builder']['version'], 3)
                self.assertEqual(sorted(p.name for p in old.iterdir()), sorted(p.name for p in new.iterdir()))
                changed = changed_since(frozen_code)
                self.assertEqual(without_builder_identity((old / 'selection.json').read_bytes(), changed),
                                 without_builder_identity((new / 'selection.json').read_bytes(), changed))
                if (old / 'plan.json').exists():
                    self.assertEqual((old / 'plan.json').read_bytes(), (new / 'plan.json').read_bytes())


class RealDataTests(unittest.TestCase):
    """Development data on disk (8c52d911 / 2658849), skipped when absent: the frozen protocols match this builder, the
    v3 run is reproducible byte for byte, and the builder's geometry equals the pipeline's at the hand-picked frame."""
    BASE = ROOT / '.local'
    FROZEN = BASE / 'evaluation/pao-accuracy/scene-builder/builder-protocol-v3.json'
    FROZEN_V4 = BASE / 'evaluation/pao-accuracy/scene-builder-v4/builder-protocol-v4.json'
    FROZEN_V41 = BASE / 'evaluation/pao-accuracy/scene-builder-v4.1/builder-protocol-v4.1.json'
    V4_COMMIT = '772fdce'  # builder v4 runs only from these committed files (v4.1 changed the builder and shared scripts)
    V3_RUN = BASE / 'evaluation/pao-accuracy/scene-builder/v3/runs/8c52d911-2658849'
    PROTOCOL_V3_SHA = '9c22c1f613ebbbd292fe92ead485ffedafffcbfc2f039fd12ad136dc14172393'
    LOG = '8c52d911-fe34-3424-9864-d3fdfac38064'

    def test_frozen_protocols_match_this_builder(self):
        found = 0
        for path, version in ((self.FROZEN_V41, 4.1), (self.FROZEN_V4, 4), (self.FROZEN, 3)):
            freeze = path.with_name(path.stem + '-freeze.json')
            if not (path.exists() and freeze.exists()):
                continue
            found += 1
            raw = path.read_bytes(); protocol = json.loads(raw)
            self.assertEqual(sha(raw), json.loads(freeze.read_bytes())['sha256'])
            self.assertEqual(protocol['builderVersion'], version)
            self.assertEqual(set(protocol['code']), set(tool.RUNTIME_CODE_BY_VERSION[version]))
            if version == 4.1:  # this builder file and every shared module are the frozen v4.1 ones
                self.assertEqual(protocol['code'], tool.runtime_code_hashes(4.1))
                for name, params in protocol['parametersByVariant'].items():
                    self.assertEqual(params, tool.v41_parameters(params['frame']['target']), name)
                self.assertEqual({n: p['frame']['target'] for n, p in protocol['parametersByVariant'].items()}, tool.V4_VARIANTS)
            elif version == 4:  # v4 parameters are still selectable; its code block names the 772fdce files
                for name, params in protocol['parametersByVariant'].items():
                    self.assertEqual(params, tool.v4_parameters(params['frame']['target']), name)
                self.assertNotEqual(protocol['code']['scripts/california/build_av2_scene.py'], sha(Path(tool.__file__).read_bytes()))
                committed = committed_hashes(self.V4_COMMIT, tool.RUNTIME_CODE_BY_VERSION[4])
                if committed is not None:
                    self.assertEqual(protocol['code'], committed)
            else:  # v3 parameters are still selectable; its code block names the cb2f497 files (this file changed in v4)
                self.assertEqual(protocol['parameters'], tool.PARAMETERS_V3)
                self.assertNotEqual(protocol['code']['scripts/california/build_av2_scene.py'], sha(Path(tool.__file__).read_bytes()))
        if not found:
            self.skipTest('no frozen builder protocol on disk')

    def v3_args(self, out, protocol, protocol_sha):
        ev = self.BASE / 'evaluation'; inv = ev / 'av2-second-observation-2026-10-05/inventories'
        return argparse.Namespace(protocol=str(protocol), protocol_sha256=protocol_sha,
                                  placement_protocol=str(ev / 'av2-map-first-2026-10-11/protocol-v3.1.json'),
                                  placement_protocol_sha256='6a7d3369e75dcb9eb5afbd46a66f94760245fe3b0e34733e0f60a106a12f2d92',
                                  split=str(ev / 'pao-accuracy/split.json'), split_lock=str(ev / 'pao-accuracy/split.lock.json'),
                                  license=str(self.BASE / 'data/av2/LICENSE-AND-ATTRIBUTION.md'), log_id=self.LOG, crossing_id='2658849',
                                  log_dir=str(self.BASE / 'data/av2/train' / self.LOG), listing_left=str(inv / 'stereo_front_left.xml'),
                                  listing_right=str(inv / 'stereo_front_right.xml'), listing_lidar=str(inv / 'lidar.xml'),
                                  map_dir=[str(ev / 'california-8c-ground-preparation-2026-10-08/sources/map')],
                                  sensor_dir=[str(self.BASE / 'data/av2/builder-8c-2026-10-12')], output=str(out),
                                  diagnostic_camera_timestamp=None, held_out_run=False, protocol_lock_sha256=None, protocol_variant=None)

    def test_frozen_v3_run_is_byte_reproducible(self):
        """The frozen v3 protocol (pinned sha256) with the cb2f497 builder files reproduces the recorded 8c plan.json and
        selection.json byte for byte; this builder's v3 path gives the same plan bytes and the same selection apart from
        the fields naming the builder file and protocol file."""
        needed = [self.FROZEN, self.V3_RUN / 'plan.json', self.V3_RUN / 'selection.json', self.BASE / 'data/av2/train' / self.LOG,
                  self.BASE / 'data/av2/builder-8c-2026-10-12']
        if not all(p.exists() for p in needed):
            self.skipTest('local 8c development data or the recorded v3 run absent')
        with tempfile.TemporaryDirectory() as temp:
            tree = frozen_v3_tree(Path(temp) / 'cb2f497')
            if tree is None:
                self.skipTest('git or commit cb2f497 unavailable')
            args = self.v3_args(Path(temp) / 'old', self.FROZEN, self.PROTOCOL_V3_SHA)
            flags = [f for k, v in vars(args).items() if v not in (None, False) and k != 'protocol_variant'
                     for f in (['--' + k.replace('_', '-'), str(v)] if not isinstance(v, list) else [x for i in v for x in ('--' + k.replace('_', '-'), i)])]
            result = subprocess.run([sys.executable, '-I', '-B', str(tree / 'scripts/california/build_av2_scene.py'), *flags], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            old = Path(temp) / 'old'
            self.assertEqual((old / 'plan.json').read_bytes(), (self.V3_RUN / 'plan.json').read_bytes())
            self.assertEqual((old / 'selection.json').read_bytes(), (self.V3_RUN / 'selection.json').read_bytes())
            protocol = json.loads(self.FROZEN.read_bytes()); protocol['code'] = tool.runtime_code_hashes(3)
            copy_path = Path(temp) / 'builder-protocol-v3-current-code.json'; copy_path.write_text(json.dumps(protocol))
            tool.build(self.v3_args(Path(temp) / 'new', copy_path, sha(copy_path.read_bytes())))
            new = Path(temp) / 'new'
            self.assertEqual((new / 'plan.json').read_bytes(), (self.V3_RUN / 'plan.json').read_bytes())
            recorded = json.loads((self.V3_RUN / 'selection.json').read_bytes())['builder']['runtimeCode']
            changed = changed_since(recorded)
            self.assertEqual(without_builder_identity((new / 'selection.json').read_bytes(), changed),
                             without_builder_identity((self.V3_RUN / 'selection.json').read_bytes(), changed))

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
            protocol = json.loads(paths['protocol'].read_bytes()); protocol['code'] = tool.runtime_code_hashes(3)
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

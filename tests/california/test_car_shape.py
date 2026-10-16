"""Synthetic tests for shape-v2 (scripts/california/car_shape.py): the sweep rule, the download rule and its receipt, the
pose and registration, the return filters, the solid top per cell, the millimetre floor, the refusals, the stationary test
and the approach-half split addendum. Synthetic data only: nothing is downloaded (a fake bucket stands in for the public
Argoverse files) and no real log file is read, except the frozen protocol files for their pins when they exist."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


shape = load('car_shape', 'scripts/california/car_shape.py')
ground_tool = load('prepare_av2_ground_for_car_shape', 'scripts/california/prepare_av2_ground.py')

LOG = '8c52d911-fe34-3424-9864-d3fdfac38064'     # a development log id (the fixture reads no real file of it)
TRACK = 'aaaaaaaa-0000-4000-8000-000000000001'
OTHER = 'bbbbbbbb-0000-4000-8000-000000000002'
ORIGIN = np.array([100., 200., 0.])
CAR = (130., 205.)        # averaged box centre (city x, y); box bottom at city z 0, top at 1.5; heading city +x, left city +y
L, W, H = 4.0, 1.8, 1.5   # 40 x 18 cells of 0.1 m
CELLS = 40 * 18
T0, STEP = 1_000_000_000_000, 100_000_000
BOX = {'track': TRACK, 'center': [CAR[0] - ORIGIN[0], H / 2, -(CAR[1] - ORIGIN[1])],
       'worldFromBox': [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], 'length': L, 'width': W, 'height': H}
PLACEMENT_SHA = 'c' * 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def md5(data):
    return hashlib.md5(data).hexdigest()


def feather_bytes(table):
    sink = pa.BufferOutputStream(); feather.write_feather(table, sink)
    return sink.getvalue().to_pybytes()


def drive(xs, ys=None, yaws=None):
    """Ego poses (city frame) at T0 + k * 100 ms: origin (x, y, 0), heading yaw about the vertical."""
    return [{'ts': T0 + k * STEP, 'x': float(x), 'y': 200. if ys is None else float(ys[k]), 'yaw': 0. if yaws is None else float(yaws[k])}
            for k, x in enumerate(xs)]


def pose_bytes(poses):
    return feather_bytes(pa.Table.from_pylist([dict(timestamp_ns=p['ts'], qw=math.cos(p['yaw'] / 2), qx=0., qy=0., qz=math.sin(p['yaw'] / 2),
                                                    tx_m=p['x'], ty_m=p['y'], tz_m=0.) for p in poses]))


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def to_ego(pose, city_points):
    """City points -> the ego frame of this pose (inverse of city = R ego + t; heights stay exact)."""
    return (np.asarray(city_points, float).reshape(-1, 3) - np.array([pose['x'], pose['y'], 0.])) @ rotation(pose['yaw'])


def ann_row(pose, track, centre, dims=(L, W, H), yaw=0.):
    """One annotation row (ego frame) of a cuboid whose city centre is `centre` and city heading `yaw`."""
    c = to_ego(pose, [centre])[0]; rel = yaw - pose['yaw']
    return dict(timestamp_ns=pose['ts'], track_uuid=track, category='REGULAR_VEHICLE', length_m=dims[0], width_m=dims[1], height_m=dims[2],
                qw=math.cos(rel / 2), qx=0., qy=0., qz=math.sin(rel / 2), tx_m=float(c[0]), ty_m=float(c[1]), tz_m=float(c[2]), num_interior_pts=50)


def car_rows(poses, drift=0.0, shift=0.0):
    """The vehicle's rows at every pose time (box centre city (130, 205, 0.75)). drift moves it along city x over the drive
    (a moving vehicle); shift moves every row 0.3 m in its own ego frame (its own rows must never move the shape)."""
    rows = []
    for k, p in enumerate(poses):
        row = ann_row(p, TRACK, (CAR[0] + drift * k / max(1, len(poses) - 1), CAR[1], H / 2))
        row['tx_m'] += shift; rows.append(row)
    return rows


def ann_bytes(rows):
    return feather_bytes(pa.Table.from_pylist(rows))


def cell(i, j, z):
    """City point at the centre of cell (i, j) (i from the front face, j from the left side), z above the box bottom."""
    return [CAR[0] + L / 2 - (i + .5) * L / 40, CAR[1] + W / 2 - (j + .5) * W / 18, z]


def sweep_bytes(points):
    """An AV2-like sweep: x, y, z stored as float16 (plus an intensity column)."""
    p = np.asarray(points, float).reshape(-1, 3)
    return feather_bytes(pa.table({'x': p[:, 0].astype(np.float16), 'y': p[:, 1].astype(np.float16), 'z': p[:, 2].astype(np.float16),
                                   'intensity': np.zeros(len(p), np.uint8)}))


def roof_value(i, j, k):
    return 1.0 + .125 * ((i + j + k) % 3)


def roof(k):
    """Car points of sweep k (city frame): every third cell along and across, at 1.0, 1.125 or 1.25 m by cell and sweep."""
    return [cell(i, j, roof_value(i, j, k)) for i in range(0, 40, 3) for j in range(0, 18, 3)]


def reference_tops(ks):
    """Solid tops of the roof sweeps ks, computed by hand from the rule: the second-highest per-sweep value per cell."""
    tops = [None] * CELLS
    for i in range(0, 40, 3):
        for j in range(0, 18, 3):
            values = sorted((roof_value(i, j, k) for k in ks), reverse=True)
            tops[i * 18 + j] = values[1] if len(values) >= 2 else None
    return tops


def listing_xml(entries, log=LOG, truncated=False, count=None):
    prefix = shape.LIDAR_PREFIX.format(log=log)
    contents = ''.join(f'<Contents><Key>{prefix}{ts}.feather</Key><Size>{size}</Size><ETag>&quot;{etag}&quot;</ETag></Contents>'
                       for ts, size, etag in entries)
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>argoverse</Name>'
            f'<Prefix>{prefix}</Prefix><KeyCount>{len(entries) if count is None else count}</KeyCount>'
            f'<IsTruncated>{"true" if truncated else "false"}</IsTruncated>{contents}</ListBucketResult>').encode()


def listing_entries(stamps):
    prefix = shape.LIDAR_PREFIX.format(log=LOG)
    return {ts: {'key': f'{prefix}{ts}.feather', 'bytes': 1000, 'etag': '0' * 32} for ts in stamps}


def no_network(*_args):
    raise AssertionError('nothing may be fetched or waited for here')


class Bucket:
    """A fake public bucket. It serves the given files; `script` maps a URL to responses used first, in order (an
    exception is raised, a tuple is returned as (status, length, etag, body)). Every requested URL is recorded."""

    def __init__(self, files, script=None):
        self.files = dict(files); self.script = {k: list(v) for k, v in (script or {}).items()}; self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        queue = self.script.get(url)
        if queue:
            response = queue.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        data = self.files.get(url)
        return (200, len(data), None, data) if data is not None else (404, None, None, b'')


class Site:
    """A synthetic development log on disk: annotations and poses in <root>/train/<log>/, a ground.json naming the
    averaged box, a saved lidar listing and the sweep files the fake bucket serves (default: the roof points)."""

    def __init__(self, root, poses, rows=None, sweeps=None, etags=None):
        self.root = Path(root); self.poses = poses
        folder = self.root / 'train' / LOG; folder.mkdir(parents=True)
        self.ann_raw = ann_bytes(car_rows(poses) if rows is None else rows); self.pose_raw = pose_bytes(poses)
        (folder / 'annotations.feather').write_bytes(self.ann_raw); (folder / 'city_SE3_egovehicle.feather').write_bytes(self.pose_raw)
        self.box = {**copy.deepcopy(BOX), 'poseAverage': {'rows': len(poses)}}
        other = {'track': OTHER, 'center': [0., .5, 0.], 'worldFromBox': BOX['worldFromBox'], 'length': 1., 'width': 1., 'height': 1.}
        ground = {'sources': {'annotations': {'sha256': sha(self.ann_raw)}, 'poses': {'sha256': sha(self.pose_raw)}},
                  'cuboids': {'items': [self.box, other]}, 'frame': {'cityOrigin': ORIGIN.tolist()}}
        self.ground_raw = json.dumps(ground).encode(); (self.root / 'ground.json').write_bytes(self.ground_raw)
        self.files = sweeps if sweeps is not None else {p['ts']: sweep_bytes(to_ego(p, roof(k))) for k, p in enumerate(poses)}
        etags = etags or {}
        self.listing_raw = listing_xml([(ts, len(data), etags.get(ts, md5(data))) for ts, data in sorted(self.files.items())])
        (self.root / 'listing.xml').write_bytes(self.listing_raw)
        self.out_root = self.root / 'data'

    def plan_args(self, saved_listing=True):
        folder = self.root / 'train' / LOG
        args = dict(log=LOG, ground_path=self.root / 'ground.json', ground_sha256=sha(self.ground_raw),
                    annotations_path=folder / 'annotations.feather', annotations_sha256=sha(self.ann_raw),
                    poses_path=folder / 'city_SE3_egovehicle.feather', poses_sha256=sha(self.pose_raw), out_root=self.out_root)
        if saved_listing:
            args.update(listing_path=self.root / 'listing.xml', listing_sha256=sha(self.listing_raw))
        return args

    def url(self, ts):
        return shape.BUCKET_URL + shape.LIDAR_PREFIX.format(log=LOG) + f'{ts}.feather'

    def bucket(self, script=None, listing=False):
        files = {self.url(ts): data for ts, data in self.files.items()}
        if listing:
            files[shape.LISTING_URL.format(log=LOG)] = self.listing_raw
        return Bucket(files, script)

    def plan(self):
        return shape.make_plan(**self.plan_args(), fetch=no_network, sleep=no_network)

    def download(self, plan_dir, plan_sha, script=None):
        waits = []; bucket = self.bucket(script)
        receipt = shape.download_sweeps(plan_dir, plan_sha, fetch=bucket, sleep=waits.append)
        return receipt, bucket, waits

    def build(self, plan_dir, plan_sha):
        return shape.build_car_shape(plan_dir, plan_sha, self.box, ORIGIN, self.ann_raw, self.pose_raw, PLACEMENT_SHA)

    def measured(self):
        plan_dir, plan_sha = self.plan(); self.download(plan_dir, plan_sha)
        return plan_dir, plan_sha, self.build(plan_dir, plan_sha)

    def ground_with(self, block):
        """A minimal placement protocol v3.4 ground layer carrying this carShape (for the approach split)."""
        return json.dumps({'sceneId': 'fixture-scene', 'schemaVersion': 3, 'carShape': block, 'cuboids': {'items': [self.box]},
                           'frame': {'cityOrigin': ORIGIN.tolist()}}).encode()


def approach_drive():
    """40 sweeps 100 ms apart, the ego moving 0.5 m per sweep along city x from 115 to 134.5 (4.9 m past the box centre)."""
    return drive([115 + .5 * k for k in range(40)])


def shape_of(sweeps, poses, rows):
    """shape_from_sweeps on in-memory (timestamp, ego points) sweeps."""
    return shape.shape_from_sweeps(sweeps, BOX, ORIGIN, shape.load_annotations(ann_bytes(rows)), shape.load_poses(pose_bytes(poses)))


# ---------------------------------------------------------------------------------------------- sweep rule (6.1)
class SweepRuleTests(unittest.TestCase):
    def long_drive(self):
        """141 poses, the ego moving 0.5 m per sweep from city x 95 to 165 at y 200 (5 m from the box centre line); the
        vehicle has rows at every listed time (also outside the pose record) except at k = 70, 71."""
        poses = drive([95 + .5 * k for k in range(141)])
        outside = [{'ts': T0 + k * STEP, 'x': 95 + .5 * k, 'y': 200., 'yaw': 0.} for k in (*range(-5, 0), *range(141, 146))]
        rows = [r for r in car_rows(poses) if r['timestamp_ns'] not in (T0 + 70 * STEP, T0 + 71 * STEP)] + car_rows(outside)
        listing = listing_entries([T0 + k * STEP for k in range(-5, 146)])
        return poses, rows, listing

    def select(self, poses, rows, listing):
        return shape.select_sweeps(listing, shape.load_annotations(ann_bytes(rows)), TRACK, shape.load_poses(pose_bytes(poses)), CAR)

    def test_candidates_need_a_track_row_the_pose_record_and_30_m(self):
        poses, rows, listing = self.long_drive()
        result = self.select(poses, rows, listing)
        # hypot(x - 130, 5) <= 30 means |x - 130| <= 29.58: k = 11 (x 100.5) .. 129 (x 159.5); k 70 and 71 have no row;
        # the listed times before and after the pose record have rows but are not inside it.
        expected = [k for k in range(141) if k not in (70, 71) and math.hypot(95 + .5 * k - 130, 5) <= 30]
        self.assertEqual(expected, [k for k in range(11, 130) if k not in (70, 71)])
        self.assertEqual(result['candidates'], len(expected))
        self.assertTrue(all(s['egoDistanceM'] <= 30 for s in result['kept']))
        self.assertTrue(all(T0 <= s['timestampNs'] <= T0 + 140 * STEP for s in result['kept']))
        # Another track's rows never make a candidate.
        others = [{**r, 'track_uuid': OTHER} for r in rows]
        self.assertEqual(self.select(poses, others, listing)['candidates'], 0)

    def test_thinning_keeps_the_next_sweep_a_full_step_on(self):
        poses = drive([120.0, 120.09375, 120.21875, 120.25, 120.375, 120.53125, 120.53125, 120.78125])
        result = self.select(poses, car_rows(poses), listing_entries([p['ts'] for p in poses]))
        self.assertEqual((result['candidates'], result['thinned']), (8, 4))
        self.assertEqual([s['timestampNs'] for s in result['kept']], [T0, T0 + 3 * STEP, T0 + 5 * STEP, T0 + 7 * STEP])  # 0.25 m counts
        self.assertEqual([s['set'] for s in result['kept']], ['shape', 'scoring', 'shape', 'scoring'])
        # The step is horizontal: a sideways move counts like a forward one.
        poses = drive([120.0, 120.0, 120.0], ys=[200.0, 200.125, 200.25])
        self.assertEqual(self.select(poses, car_rows(poses), listing_entries([p['ts'] for p in poses]))['thinned'], 2)

    def test_cap_keeps_40_evenly_spaced_and_the_split_alternates(self):
        poses, rows, listing = self.long_drive()
        result = self.select(poses, rows, listing)
        thinned = [k for k in range(11, 130) if k not in (70, 71)]  # 117: every 0.5 m move clears the 0.25 m step
        self.assertEqual(result['thinned'], 117)
        expected = [thinned[round(k * 116 / 39)] for k in range(40)]
        self.assertEqual(len(set(expected)), 40); self.assertEqual((expected[0], expected[-1]), (11, 129))
        self.assertEqual([s['timestampNs'] for s in result['kept']], [T0 + k * STEP for k in expected])
        self.assertEqual([s['set'] for s in result['kept']], ['shape', 'scoring'] * 20)
        for s, k in zip(result['kept'], expected):
            self.assertEqual(s['egoDistanceM'], float(np.hypot(95 + .5 * k - 130, -5.)))

    def test_standing_ego_gives_one_sweep(self):
        poses = drive([120.0] * 40)
        result = self.select(poses, car_rows(poses), listing_entries([p['ts'] for p in poses]))
        self.assertEqual((result['candidates'], result['thinned'], len(result['kept'])), (40, 1, 1))

    def test_listing_parser_takes_complete_listings_only(self):
        entries = [(T0, 10, 'a' * 32), (T0 + STEP, 11, 'b' * 32 + '-2')]
        parsed = shape.parse_listing(listing_xml(entries), LOG)
        self.assertEqual(parsed[T0 + STEP], {'key': shape.LIDAR_PREFIX.format(log=LOG) + f'{T0 + STEP}.feather', 'bytes': 11, 'etag': 'b' * 32 + '-2'})
        for bad, message in ((listing_xml(entries, truncated=True), 'Truncated'), (listing_xml(entries, count=3), 'count'),
                             (listing_xml(entries, log='c780d53a-2d37-3cd8-9e89-530966aef53e'), 'another log'),
                             (listing_xml([(T0, 10, 'xyz')]), 'ETag'), (listing_xml(entries + entries[:1]), 'Duplicate')):
            with self.assertRaisesRegex(ValueError, message):
                shape.parse_listing(bad, LOG)


# ---------------------------------------------------------------------------------------------- plan, downloads, receipts (6.1, 6.3)
class PlanAndDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_is_written_and_hashed_before_any_download_and_is_deterministic(self):
        site = Site(self.root / 'a', approach_drive())
        plan_dir, plan_sha = site.plan()  # no_network: choosing reads metadata only
        raw = (plan_dir / shape.PLAN_NAME).read_bytes(); plan = json.loads(raw)
        self.assertEqual(sha(raw), plan_sha); self.assertEqual(plan_dir, site.out_root / f'shape-v2-{LOG}')
        self.assertFalse((plan_dir / shape.SWEEP_DIR).exists()); self.assertFalse((plan_dir / shape.RECEIPT_NAME).exists())
        self.assertEqual((plan_dir / shape.LISTING_NAME).read_bytes(), site.listing_raw)
        self.assertEqual((plan['candidates'], plan['thinned'], plan['kept']), (40, 40, 40))
        self.assertEqual(plan['shape'], [str(T0 + k * STEP) for k in range(0, 40, 2)])
        self.assertEqual(plan['scoring'], [str(T0 + k * STEP) for k in range(1, 40, 2)])
        self.assertEqual(plan['listing'], {'status': 'ok', 'attempts': 0, 'objects': 40})
        self.assertEqual(plan['rule'], shape.RULE); self.assertEqual(plan['shapeProtocol']['sha256'], shape.SHAPE_PROTOCOL_SHA256)
        self.assertNotIn(b'park', raw.lower()); self.assertNotIn(str(self.root), raw.decode())
        with self.assertRaisesRegex(ValueError, 'Existing sweep plan retained'):
            site.plan()
        other = Site(self.root / 'b', approach_drive()); _, other_sha = other.plan()
        self.assertEqual(other_sha, plan_sha)

    def test_plan_inputs_must_match_the_ground_layer_and_the_log(self):
        site = Site(self.root / 'a', approach_drive())
        args = site.plan_args(); args['annotations_path'] = self.root / 'elsewhere.feather'
        (self.root / 'elsewhere.feather').write_bytes(site.ann_raw)
        with self.assertRaisesRegex(ValueError, 'log directory'):
            shape.make_plan(**args, fetch=no_network, sleep=no_network)
        args = site.plan_args(); args['log'] = 'not-a-log'
        with self.assertRaisesRegex(ValueError, 'UUID'):
            shape.make_plan(**args, fetch=no_network, sleep=no_network)
        empty = Site(self.root / 'empty', approach_drive(), sweeps={})
        with self.assertRaisesRegex(ValueError, 'listing of this log is empty'):
            empty.plan()

    def test_listing_is_fetched_under_the_same_attempt_rule(self):
        site = Site(self.root / 'a', approach_drive())
        url = shape.LISTING_URL.format(log=LOG); waits = []
        bucket = site.bucket({url: [(503, None, None, b''), (200, None, None, listing_xml([], truncated=True))]}, listing=True)
        plan_dir, plan_sha = shape.make_plan(**site.plan_args(saved_listing=False), fetch=bucket, sleep=waits.append)
        plan = json.loads((plan_dir / shape.PLAN_NAME).read_bytes())
        self.assertEqual(plan['listing'], {'status': 'ok', 'attempts': 3, 'objects': 40}); self.assertEqual(waits, [5, 10])
        self.assertEqual(plan['inputs']['listing']['source'], 'fetched'); self.assertEqual(plan['kept'], 40)
        self.assertEqual((plan_dir / shape.LISTING_NAME).read_bytes(), site.listing_raw)

    def test_failed_listing_gives_zero_candidates_and_too_few_sweeps(self):
        site = Site(self.root / 'a', approach_drive())
        url = shape.LISTING_URL.format(log=LOG); waits = []
        bucket = Bucket({}, {url: [(503, None, None, b''), OSError('reset'), (500, None, None, b''), (200, None, None, b'<not xml')]})
        plan_dir, plan_sha = shape.make_plan(**site.plan_args(saved_listing=False), fetch=bucket, sleep=waits.append)
        plan = json.loads((plan_dir / shape.PLAN_NAME).read_bytes())
        self.assertEqual(plan['listing'], {'status': 'failed', 'attempts': 4, 'objects': 0}); self.assertEqual(waits, [5, 10, 20])
        self.assertEqual((plan['candidates'], plan['thinned'], plan['kept']), (0, 0, 0)); self.assertIsNone(plan['inputs']['listing']['file'])
        self.assertFalse((plan_dir / shape.LISTING_NAME).exists())
        receipt, bucket, _ = site.download(plan_dir, plan_sha)
        self.assertEqual((receipt['sweeps'], receipt['downloadDropped'], bucket.calls), ([], [], []))
        block = site.build(plan_dir, plan_sha)
        self.assertEqual((block['status'], block['refusal']['code']), ('refused', 'too-few-sweeps'))
        self.assertEqual(block['sweeps']['listing'], {'status': 'failed', 'attempts': 4})

    def test_download_retries_drops_and_never_replaces(self):
        site = Site(self.root / 'a', approach_drive())
        plan_dir, plan_sha = site.plan(); plan_raw = (plan_dir / shape.PLAN_NAME).read_bytes(); plan = json.loads(plan_raw)
        first, second = plan['sweeps'][0], plan['sweeps'][1]
        good = site.files[int(second['timestampNs'])]; flipped = good[:-1] + bytes([good[-1] ^ 1])
        script = {site.url(first['timestampNs']): [OSError('reset'), (500, None, None, b'')],
                  site.url(second['timestampNs']): [(200, len(good) + 1, None, good + b'x'), (200, len(good), None, flipped),
                                                    (404, None, None, b''), OSError('timeout')]}
        receipt, bucket, waits = site.download(plan_dir, plan_sha, script)
        self.assertEqual(waits, [5, 10, 5, 10, 20])
        self.assertEqual(receipt['downloadDropped'], [{'timestampNs': second['timestampNs'], 'set': 'scoring', 'reason': 'download-failed', 'attempts': 4}])
        self.assertEqual(len(receipt['sweeps']), 39)
        record = receipt['sweeps'][0]
        self.assertEqual((record['timestampNs'], record['attempts'], record['status'], record['source']), (first['timestampNs'], 3, 200, 'download'))
        self.assertEqual((record['sizeMatches'], record['md5MatchesEtag'], record['md5']), (True, True, first['etag']))
        self.assertTrue(set(record) >= {'attempts', 'bytes', 'contentLength', 'etag', 'key', 'md5', 'md5MatchesEtag', 'sha256', 'sizeMatches', 'status', 'url'})
        self.assertEqual(bucket.calls.count(site.url(second['timestampNs'])), 4)
        self.assertEqual(set(bucket.calls), {site.url(s['timestampNs']) for s in plan['sweeps']})  # nothing unplanned: not replaced
        folder = plan_dir / shape.SWEEP_DIR
        self.assertFalse((folder / f"{second['timestampNs']}.feather").exists())
        self.assertEqual([p.name for p in folder.iterdir() if p.name.startswith('.partial-')], [])
        self.assertEqual((plan_dir / shape.PLAN_NAME).read_bytes(), plan_raw)  # the split is never redone
        block = site.build(plan_dir, plan_sha)
        self.assertEqual(block['status'], 'measured'); self.assertEqual((len(block['sweeps']['shape']), len(block['sweeps']['scoring'])), (20, 19))
        self.assertEqual(block['sweeps']['downloadDropped'], receipt['downloadDropped']); self.assertEqual(block['sweeps']['kept'], 40)
        with self.assertRaisesRegex(ValueError, 'already receipted'):
            site.download(plan_dir, plan_sha)

    def test_multipart_etag_checks_the_size_only(self):
        poses = approach_drive()
        site = Site(self.root / 'a', poses, etags={poses[0]['ts']: 'f' * 32 + '-2'})
        data = site.files[poses[0]['ts']]
        entry = {'etag': 'f' * 32 + '-2', 'bytes': len(data)}
        self.assertEqual(shape.byte_check(entry, data), {'ok': True, 'sizeMatches': True, 'md5': md5(data), 'md5MatchesEtag': None})
        self.assertFalse(shape.byte_check(entry, data + b'x')['ok'])
        self.assertFalse(shape.byte_check({'etag': '0' * 32, 'bytes': len(data)}, data)['ok'])
        plan_dir, plan_sha = site.plan(); receipt, _, _ = site.download(plan_dir, plan_sha)
        record = receipt['sweeps'][0]
        self.assertEqual((record['md5MatchesEtag'], record['sizeMatches'], record['attempts']), (None, True, 1))
        self.assertTrue(all(r['md5MatchesEtag'] is True for r in receipt['sweeps'][1:]))
        self.assertEqual(site.build(plan_dir, plan_sha)['status'], 'measured')

    def test_a_file_already_on_disk_is_reused_only_when_it_passes(self):
        site = Site(self.root / 'a', approach_drive())
        plan_dir, plan_sha = site.plan(); plan = json.loads((plan_dir / shape.PLAN_NAME).read_bytes())
        folder = plan_dir / shape.SWEEP_DIR; folder.mkdir()
        good, bad = plan['sweeps'][0]['timestampNs'], plan['sweeps'][2]['timestampNs']
        (folder / f'{good}.feather').write_bytes(site.files[int(good)]); (folder / f'{bad}.feather').write_bytes(b'not the file')
        receipt, bucket, _ = site.download(plan_dir, plan_sha)
        records = {r['timestampNs']: r for r in receipt['sweeps']}
        self.assertEqual((records[good]['source'], records[good]['attempts'], records[good]['status']), ('disk', 0, None))
        self.assertEqual((records[bad]['source'], records[bad]['attempts']), ('download', 1))
        self.assertNotIn(site.url(good), bucket.calls); self.assertEqual((folder / f'{bad}.feather').read_bytes(), site.files[int(bad)])

    def test_integrity_failures_stop_the_build(self):
        site = Site(self.root / 'a', approach_drive())
        plan_dir, plan_sha, block = site.measured()
        self.assertEqual(block['status'], 'measured')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            site.build(plan_dir, 'd' * 64)
        sweep = plan_dir / shape.SWEEP_DIR / f"{block['sweeps']['shape'][3]['timestampNs']}.feather"
        original = sweep.read_bytes(); sweep.write_bytes(original + b'\0')
        with self.assertRaisesRegex(ValueError, 'changed on disk'):
            site.build(plan_dir, plan_sha)
        sweep.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            site.build(plan_dir, plan_sha)
        sweep.write_bytes(original); self.assertEqual(site.build(plan_dir, plan_sha), block)
        # A plan made for another box, other annotations or other poses is refused.
        with self.assertRaisesRegex(ValueError, 'another averaged box'):
            shape.build_car_shape(plan_dir, plan_sha, {**site.box, 'length': 4.1}, ORIGIN, site.ann_raw, site.pose_raw, PLACEMENT_SHA)
        with self.assertRaisesRegex(ValueError, 'other annotations or poses'):
            shape.build_car_shape(plan_dir, plan_sha, site.box, ORIGIN, site.ann_raw + b'\0', site.pose_raw, PLACEMENT_SHA)

    def test_build_is_byte_identical_and_names_no_forbidden_word(self):
        first = Site(self.root / 'a', approach_drive()).measured()[2]; second = Site(self.root / 'b', approach_drive()).measured()[2]
        self.assertEqual(shape.canonical(first), shape.canonical(second))
        self.assertNotIn(b'park', shape.canonical(first).lower())
        self.assertEqual(first['code'], {'path': 'scripts/california/car_shape.py', 'sha256': sha((ROOT / 'scripts/california/car_shape.py').read_bytes())})
        self.assertEqual(first['protocol'], {'placementProtocolVersion': '3.4', 'sha256': PLACEMENT_SHA, 'shapeProtocolSha256': shape.SHAPE_PROTOCOL_SHA256})
        self.assertEqual(first['solidTopM'], reference_tops(range(0, 40, 2)))


# ---------------------------------------------------------------------------------------------- the shape (5.3, 5.4, 6.2, 6.4)
class ShapeTests(unittest.TestCase):
    def test_turned_ego_poses_register_returns_into_the_right_cells(self):
        poses = drive([118., 121., 124.], ys=[199., 198.5, 201.], yaws=[.3, -.5, 1.2])
        heights = {(0, 0): [1.25, 1.125, 1.0], (39, 17): [1.0, 1.0, 1.0], (20, 9): [1.375, 1.5, .5]}
        sweeps = [(p['ts'], to_ego(p, [cell(i, j, z[k]) for (i, j), z in heights.items()])) for k, p in enumerate(poses)]
        result = shape_of(sweeps, poses, car_rows(poses))
        expected = [None] * CELLS
        expected[0] = 1.125; expected[39 * 18 + 17] = 1.0; expected[20 * 18 + 9] = 1.375
        self.assertEqual(result['solidTopM'], expected)
        counts = result['counts']
        self.assertEqual([(i, counts['returns'][i], counts['returnSweeps'][i]) for i in range(CELLS) if counts['returns'][i]],
                         [(0, 3, 3), (20 * 18 + 9, 3, 3), (39 * 18 + 17, 3, 3)])
        self.assertEqual(result['diagnostics']['frontFaceToFirstSolidCellM'], 0.0)
        # The vehicle's own per-sweep annotations never move or filter returns: shifting them 0.3 m changes nothing.
        self.assertEqual(shape_of(sweeps, poses, car_rows(poses, shift=.3)), result)

    def test_pose_is_interpolated_between_the_two_nearest_poses(self):
        poses = drive([118., 120.])
        record = shape.load_poses(pose_bytes(poses))
        q, t = shape.ego_poses_at([T0 + STEP // 4], record)
        np.testing.assert_allclose(t[0], [118.5, 200., 0.], atol=1e-12); np.testing.assert_allclose(q[0], [1., 0., 0., 0.], atol=1e-15)
        with self.assertRaisesRegex(ValueError, 'outside the pose record'):
            shape.ego_poses_at([T0 + 2 * STEP], record)

    def test_solid_top_examples(self):
        poses = drive([118., 120., 122.])
        returns = {  # cell: [(height, sweep index)] (the protocol's examples, plus one high return from one sweep)
            (5, 5): [(1.4, 0), (1.38, 0), (1.2, 1)], (6, 6): [(1.4, 0), (1.35, 1)], (7, 7): [(1.4, 0), (1.1, 0)],
            (8, 8): [(1.45, 0), (.9, 1), (.8, 2)]}
        sweeps = [(p['ts'], to_ego(p, [cell(i, j, z) for (i, j), values in returns.items() for z, s in values if s == k])) for k, p in enumerate(poses)]
        result = shape_of(sweeps, poses, car_rows(poses))
        tops = result['solidTopM']
        self.assertEqual((tops[5 * 18 + 5], tops[6 * 18 + 6], tops[7 * 18 + 7], tops[8 * 18 + 8]), (1.2, 1.35, None, .9))
        self.assertEqual(sum(v is not None for v in tops), 3)
        self.assertEqual([result['counts']['returns'][i * 18 + i] for i in (5, 6, 7, 8)], [3, 2, 2, 3])
        self.assertEqual([result['counts']['returnSweeps'][i * 18 + i] for i in (5, 6, 7, 8)], [2, 2, 1, 3])
        self.assertEqual(result['diagnostics']['frontFaceToFirstSolidCellM'], 5 * .1)

    def test_return_filters(self):
        poses = drive([120., 121.])
        other = ann_row(poses[0], OTHER, (129.05, 205.05, 1.0), dims=(.4, .4, .4))  # at sweep A only
        rows = car_rows(poses) + [other]
        points = [(129.05, 205.05, 1.0),     # inside the other cuboid (A): removed
                  (129.33, 205.05, 1.0),     # 0.08 m outside it, inside the 0.10 m growth (A): removed
                  (129.37, 205.05, 1.0),     # 0.12 m outside it: kept
                  (131.05, 205.05, .25),     # exactly the 0.25 m floor: kept
                  (131.05, 205.05, .234375),  # below the floor: ignored, counted
                  (131.05, 205.05, 1.5),     # exactly the box top: kept
                  (131.05, 205.05, 1.515625),  # above the box top: dropped, counted, never lowered
                  (132.3, 205.05, 1.0),      # 0.3 m beyond the front face: diagnostics only
                  (132.6, 205.05, 1.0),      # 0.6 m beyond: not counted
                  (127.55, 205.05, 1.0),     # 0.45 m beyond the rear face: diagnostics only
                  (132.3, 205.05, 1.6),      # beyond and above the top: not counted
                  (132.3, 205.05, .1),       # beyond and below the floor: not counted
                  (131.05, 206.0, 1.0)]      # beside the footprint: ignored
        sweeps = [(p['ts'], to_ego(p, points)) for p in poses]
        result = shape_of(sweeps, poses, rows)
        self.assertEqual(result['removed'], {'belowFloor': 2, 'otherObjects': 2, 'aboveBoxTop': 2})
        self.assertEqual(result['diagnostics']['returnsBeyondEnds'], {'front': 2, 'rear': 2})
        c = result['counts']; tops = result['solidTopM']
        self.assertEqual(sum(c['returns']), 3 + 5)
        self.assertEqual((c['returns'][9 * 18 + 8], c['returnSweeps'][9 * 18 + 8], tops[9 * 18 + 8]), (4, 2, 1.5))
        self.assertEqual((c['returns'][26 * 18 + 8], c['returnSweeps'][26 * 18 + 8], tops[26 * 18 + 8]), (3, 2, 1.0))
        self.assertEqual((c['returns'][29 * 18 + 8], c['returnSweeps'][29 * 18 + 8], tops[29 * 18 + 8]), (1, 1, None))
        # Without the other track's row every return in its cuboid counts.
        clear = shape_of(sweeps, poses, car_rows(poses))
        self.assertEqual(clear['removed']['otherObjects'], 0); self.assertEqual(clear['solidTopM'][29 * 18 + 8], 1.0)

    def test_float16_points_and_millimetre_floor(self):
        poses = approach_drive()[:31]
        values = {(5, 5): 1.2509765625, (6, 6): 1.2, (7, 7): 1.23456}  # float16: 1.2509765625, 1.2001953125, 1.234375
        files = {p['ts']: sweep_bytes(to_ego(p, [cell(i, j, z) for (i, j), z in values.items()])) for p in poses}
        site = Site(Path(self.temp.name), poses, sweeps=files)
        read = shape.read_sweep(files[poses[0]['ts']])
        self.assertEqual(read.dtype, np.float64); self.assertEqual(sorted(read[:, 2].tolist()), [1.2001953125, 1.234375, 1.2509765625])
        block = site.measured()[2]; tops = block['solidTopM']
        self.assertEqual((tops[5 * 18 + 5], tops[6 * 18 + 6], tops[7 * 18 + 7]), (1.25, 1.2, 1.234))  # floored, never rounded up
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            shape.read_sweep(sweep_bytes([[np.nan, 0., 1.]]))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_floor_mm_follows_the_formula_and_never_enlarges(self):
        mm = np.arange(250, 1501) / 1000.0
        for z in np.concatenate([mm, np.nextafter(mm, 0), np.nextafter(mm, 9), np.random.default_rng(5).uniform(.25, 1.5, 20000)]):
            z = float(z); value = shape.floor_mm(z); literal = math.floor(z * 1000) / 1000
            self.assertLessEqual(value, z)
            self.assertEqual(value, literal if literal <= z else (math.floor(z * 1000) - 1) / 1000)
        self.assertEqual((shape.floor_mm(1.2), shape.floor_mm(.3), shape.floor_mm(1.2509765625)), (1.2, .3, 1.25))
        self.assertEqual(shape.floor_mm(float(np.nextafter(.3, 0))), .299)

    def test_kept_returns_equal_metric_f_profile(self):
        """Cross-check with metric F (test-only import) on a random cloud with no other objects."""
        metric_f = load('metric_f_car_top_for_test', 'scripts/evaluation/metric_f_car_top.py')
        rng = np.random.default_rng(11); poses = drive([112., 119., 126.], ys=[198., 199.5, 197.], yaws=[.2, -.1, .05])
        city = [np.column_stack([CAR[0] + rng.uniform(-2.7, 2.7, 4000), CAR[1] + rng.uniform(-1.0, 1.0, 4000), rng.uniform(0., 1.7, 4000)])
                for _ in poses]
        result = shape_of([(p['ts'], to_ego(p, c)) for p, c in zip(poses, city)], poses, car_rows(poses))
        frame = metric_f.upright_frame(BOX)
        local = np.vstack([metric_f.car_local(metric_f.world(c, ORIGIN), frame) for c in city])
        th = {'minHeightAboveBoxBottomM': .25, 'maxHeightAboveBoxTopM': 0, 'binLengthNominalM': .1, 'lateralBinNominalM': .1,
              'minReturnsMeasured': 1, 'overhangReachM': .5, 'overhangBinM': .1, 'overhangSummaryReachM': .5}
        profile = metric_f.profile_from_points(local, np.repeat(np.arange(3), 4000), (L, W, H), th)
        self.assertEqual(sum(result['counts']['returns']), profile['counts']['kept'])
        self.assertEqual(result['counts']['returns'], [c['count'] for c in profile['cells']['cells']])
        self.assertEqual(result['removed']['aboveBoxTop'], profile['counts']['aboveTopLimit'])
        self.assertEqual(result['diagnostics']['returnsBeyondEnds'], {end: profile['overhang'][end]['returnsWithinSummaryReach'] for end in ('front', 'rear')})
        for top, c in zip(result['solidTopM'], profile['cells']['cells']):
            if top is not None:
                self.assertLessEqual(top, c['topZ'])
        self.assertGreater(sum(v is not None for v in result['solidTopM']), 600)


# ---------------------------------------------------------------------------------------------- refusals and the stationary test (6.5, 10.2)
class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def assert_refused(self, block, code):
        self.assertEqual((block['status'], block['refusal']), ('refused', {'code': code, 'reason': shape.REFUSALS[code]}))
        self.assertTrue(all(v is None for v in block['solidTopM'])); self.assertEqual(len(block['solidTopM']), CELLS)
        self.assertFalse(any(block['counts']['returns']) or any(block['counts']['returnSweeps']))

    def test_standing_ego_gives_too_few_sweeps(self):
        site = Site(self.root / 'a', drive([120.] * 40))
        plan_dir, plan_sha, block = site.measured()
        self.assert_refused(block, 'too-few-sweeps')
        self.assertEqual((block['sweeps']['kept'], len(block['sweeps']['shape']), len(block['sweeps']['scoring'])), (1, 1, 0))

    def test_a_dropped_shape_sweep_can_leave_too_few(self):
        site = Site(self.root / 'a', drive([120.] * 10 + [121.] * 10 + [122.] * 20))  # kept 3: shape 2, scoring 1
        plan_dir, plan_sha = site.plan(); plan = json.loads((plan_dir / shape.PLAN_NAME).read_bytes())
        self.assertEqual((plan['shape'], plan['scoring']), ([str(T0), str(T0 + 20 * STEP)], [str(T0 + 10 * STEP)]))
        site.download(plan_dir, plan_sha, {site.url(plan['shape'][1]): [(500, None, None, b'')] * 4})
        block = site.build(plan_dir, plan_sha)
        self.assert_refused(block, 'too-few-sweeps')
        self.assertEqual(block['sweeps']['downloadDropped'], [{'timestampNs': plan['shape'][1], 'set': 'shape', 'reason': 'download-failed', 'attempts': 4}])

    def test_stationary_test_matches_the_placement_protocol(self):
        poses = drive([115 + .5 * k for k in range(31)])  # 3.0 s
        rigid = shape.load_poses(pose_bytes(poses))['rigid']
        def check(rows):
            return shape.stationary(shape.load_annotations(ann_bytes(rows)), rigid, TRACK)
        passed = check(car_rows(poses, drift=1.0))  # exactly 3.0 s and exactly 1.0 m: both limits are inclusive
        self.assertEqual((passed['passed'], passed['rows'], passed['spanSeconds']), (True, 31, 3.0))
        self.assertAlmostEqual(passed['maxDisplacementFromFirstM'], 1.0, places=12)
        self.assertFalse(check(car_rows(poses, drift=1.0625))['passed'])
        self.assertFalse(check(car_rows(poses[:30]))['passed'])  # 2.9 s
        missing = car_rows(poses) + [ann_row({'ts': T0 + 31 * STEP + 1, 'x': 130., 'y': 200., 'yaw': 0.}, TRACK, (130., 205., .75))]
        self.assertEqual(check(missing)['passed'], False)  # a row without an exact ego pose fails
        self.assertEqual((shape.STATIONARY_SPAN_S, shape.STATIONARY_DISPLACEMENT_M),
                         (ground_tool.STATIONARY_SPAN_S, ground_tool.STATIONARY_DISPLACEMENT_M))

    def test_a_moving_vehicle_is_refused_before_counting_sweeps(self):
        poses = approach_drive()
        site = Site(self.root / 'a', poses, rows=car_rows(poses, drift=2.0))
        self.assert_refused(site.measured()[2], 'not-stationary')
        standing = drive([120.] * 40)
        site = Site(self.root / 'b', standing, rows=car_rows(standing, drift=2.0))  # also too few: not-stationary is checked first
        self.assert_refused(site.measured()[2], 'not-stationary')


# ---------------------------------------------------------------------------------------------- the carShape block (10.1)
class BlockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_block_fields_and_validator(self):
        block = Site(self.root / 'a', approach_drive()).measured()[2]
        self.assertEqual(sorted(block), ['box', 'code', 'counts', 'diagnostics', 'frame', 'grid', 'protocol', 'refusal', 'removed', 'schema',
                                         'solidTopM', 'status', 'sweeps', 'track'])
        self.assertEqual(block['grid'], {'along': 40, 'lateral': 18, 'cellAlongM': L / 40, 'cellLateralM': W / 18, 'maxCellM': .1, 'order': shape.GRID_ORDER})
        self.assertEqual(block['sweeps']['rule'], shape.RULE); self.assertEqual(block['sweeps']['candidates'], 40)
        self.assertEqual(shape.grid_size(4.03, 1.8825), (41, 19)); self.assertEqual(shape.grid_size(4.0, 1.8), (40, 18))
        shape.validate_block(block, BOX)
        index = next(i for i, v in enumerate(block['solidTopM']) if v is not None)
        def broken(change):
            value = copy.deepcopy(block); change(value)
            with self.assertRaises(ValueError):
                shape.validate_block(value, BOX)
        broken(lambda b: b['solidTopM'].pop())                                                      # wrong length
        broken(lambda b: b['counts']['returnSweeps'].__setitem__(index, 1))                         # a top without two sweeps
        broken(lambda b: b['solidTopM'].__setitem__(index, H + .001))                                # above H
        broken(lambda b: b['solidTopM'].__setitem__(index, .249))                                   # below 0.25
        broken(lambda b: b.__setitem__('track', OTHER))                                             # track mismatch
        broken(lambda b: b['sweeps']['scoring'].append(b['sweeps']['shape'][0]))                    # overlapping sets
        broken(lambda b: b['sweeps'].__setitem__('kept', 21))                                       # lists do not add up to kept
        broken(lambda b: b.__setitem__('schema', 'shape-v0'))
        broken(lambda b: b.update(status='refused', refusal={'code': 'too-few-sweeps', 'reason': shape.REFUSALS['too-few-sweeps']}))  # refused with tops
        with self.assertRaisesRegex(ValueError, 'either measured or refused'):
            shape.car_shape_block(BOX, block['sweeps'], PLACEMENT_SHA, refusal='made-up')

    def test_constants_match_the_placement_scripts_and_the_app(self):
        self.assertEqual(shape.HELD_OUT_LOG_PREFIXES, ground_tool.HELD_OUT_LOG_PREFIXES)
        geometry = (ROOT / 'src/geometry.mjs').read_text()
        match = re.search(r"export const CAR_SHAPE_REFUSALS=Object\.freeze\(\{'too-few-sweeps':'([^']*)','not-stationary':'([^']*)'\}\)", geometry)
        self.assertIsNotNone(match)
        self.assertEqual(shape.REFUSALS, {'too-few-sweeps': match.group(1), 'not-stationary': match.group(2)})
        source = (ROOT / 'scripts/california/car_shape.py').read_text()
        self.assertNotIn('scripts/evaluation', source.replace('imports nothing from scripts/evaluation/', ''))
        self.assertNotRegex(source, r'(?m)^\s*(import|from)\s+metric_f')
        # The banned word appears once: in the constant every writer checks its output against.
        self.assertEqual([line for line in source.splitlines() if 'park' in line.lower()], ["BANNED = b'park'"])

    @unittest.skipUnless((ROOT / shape.SHAPE_PROTOCOL_PATH).exists(), 'frozen shape-v2 protocol absent')
    def test_numbers_and_pins_equal_the_frozen_protocol(self):
        raw = (ROOT / shape.SHAPE_PROTOCOL_PATH).read_bytes(); delta = json.loads(raw)
        self.assertEqual(sha(raw), shape.SHAPE_PROTOCOL_SHA256)
        self.assertEqual(sha((ROOT / shape.SPLIT_ADDENDUM_PATH).read_bytes()), shape.SPLIT_ADDENDUM_SHA256)
        # shape-v2 is a list of changes to the frozen shape-v1 protocol: every number comes from the base except the step.
        base_raw = (ROOT / delta['base']['path']).read_bytes(); protocol = json.loads(base_raw)
        self.assertEqual(sha(base_raw), delta['base']['sha256'])
        step = delta['changes']['sweeps.thinning.minEgoStepM']
        s = protocol['sweeps']
        self.assertEqual(s['thinning']['minEgoStepM']['value'], step['from'])
        self.assertEqual((s['candidate']['maxEgoDistanceM']['value'], step['to'], s['cap']['maxKept']['value'],
                          s['minimum']['minShapeSweeps']['value'], s['downloads']['maxAttempts']['value'], s['downloads']['retryWaitsS']),
                         (shape.RULE['maxEgoDistanceM'], shape.RULE['minEgoStepM'], shape.RULE['maxKept'], shape.RULE['minShapeSweeps'],
                          shape.RULE['downloadAttempts'], shape.RULE['retryWaitsS']))
        r = protocol['returns']; rep = protocol['representation']
        self.assertEqual((r['minHeightM']['value'], r['otherObjectGrowM']['value'], rep['maxCellM']['value'], rep['ceilTolerance']['value']),
                         (shape.MIN_HEIGHT_M, shape.OTHER_GROW_M, shape.MAX_CELL_M, shape.CEIL_TOLERANCE))

    def test_held_out_logs_need_the_lock(self):
        held = '88f47a10-87b4-3ea8-a0c7-a07d825b647d'  # a real held-out id: split.json is read, no log file
        self.assertIsNone(shape.check_held_out(LOG, False, None))
        with self.assertRaisesRegex(ValueError, 'held-out-run'):
            shape.check_held_out(held, False, None)
        lock_path = self.root / 'protocol-lock.json'
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            shape.check_held_out(held, True, 'e' * 64, lock_path=lock_path)
        def lock(pipeline, approval=True):
            raw = json.dumps({'pipeline': pipeline, **({'approval': {'by': 'Sean', 'utc': '2026-10-20T00:00:00Z'}} if approval else {})}).encode()
            if lock_path.exists():
                lock_path.unlink()
            lock_path.write_bytes(raw); return sha(raw)
        pins = {'shapeProtocol': {'sha256': shape.SHAPE_PROTOCOL_SHA256}, 'shapeSplitAddendum': {'sha256': shape.SPLIT_ADDENDUM_SHA256}}
        digest = lock({'shapeProtocol': {'sha256': 'f' * 64}})
        with self.assertRaisesRegex(ValueError, 'shape-v2 protocol'):
            shape.check_held_out(held, True, digest, lock_path=lock_path)
        digest = lock({'shapeProtocol': pins['shapeProtocol']})
        self.assertEqual(shape.check_held_out(held, True, digest, lock_path=lock_path)['sha256'], digest)
        with self.assertRaisesRegex(ValueError, 'split addendum'):
            shape.check_held_out(held, True, digest, need_addendum=True, lock_path=lock_path)
        digest = lock(pins, approval=False)
        with self.assertRaisesRegex(ValueError, 'approval'):
            shape.check_held_out(held, True, digest, need_addendum=True, lock_path=lock_path)
        digest = lock(pins)
        self.assertIsNotNone(shape.check_held_out(held, True, digest, need_addendum=True, lock_path=lock_path))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            shape.check_held_out(held, True, 'f' * 64, lock_path=lock_path)


# ---------------------------------------------------------------------------------------------- approach-half split addendum (report only)
class ApproachSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_halves_order_ties_and_an_odd_count(self):
        entries = [{'timestampNs': ts, 'distanceM': d} for ts, d in ((5, 9.), (1, 3.), (4, 6.4), (2, 6.4), (3, 12.), (6, 1.), (7, 6.4))]
        halves = shape.approach_halves(entries)
        self.assertEqual((halves['K'], halves['h']), (7, 3))
        self.assertEqual([e['timestampNs'] for e in halves['far']], [3, 5, 2])   # ties by timestamp ascending
        self.assertEqual([e['timestampNs'] for e in halves['unused']], [4])
        self.assertEqual([e['timestampNs'] for e in halves['near']], [7, 1, 6])
        self.assertEqual(shape.approach_halves(entries[:3])['h'], 1)

    def test_far_half_shape_is_built_from_the_far_sweeps_only(self):
        site = Site(self.root / 'a', approach_drive())
        plan_dir, plan_sha, block = site.measured()
        ground_raw = site.ground_with(block)
        doc, data = shape.approach_split_document(ground_raw, plan_dir, site.ann_raw, site.pose_raw, 'ground.json')
        split = doc['approachSplit']
        # Kept: every k = 0 .. 39 (x 115 .. 134.5). Far: x 115 .. 124.5; near: x 125, then the tie 125.5 / 134.5 (6.73 m) by time, ...
        x_of = lambda e: 115 + .5 * ((int(e['timestampNs']) - T0) // STEP)  # noqa: E731
        self.assertEqual((split['K'], split['h'], split['unused']), (40, 20, []))
        self.assertEqual([x_of(e) for e in split['far']], [115. + .5 * k for k in range(20)])
        self.assertEqual([x_of(e) for e in split['near']], [125.] + [x for d in (4.5, 4., 3.5, 3., 2.5, 2., 1.5, 1., .5) for x in (130 - d, 130 + d)] + [130.])
        far_ks = list(range(20))
        self.assertEqual(doc['carShape']['solidTopM'], reference_tops(far_ks))
        self.assertEqual([e['timestampNs'] for e in doc['carShape']['sweeps']['shape']], [str(T0 + k * STEP) for k in far_ks])
        self.assertEqual(doc['carShape']['sweeps']['kept'], 40); self.assertEqual(doc['scene']['appCarShapeSha256'], sha(shape.canonical(block)))
        self.assertEqual(doc['addendum']['sha256'], shape.SPLIT_ADDENDUM_SHA256); self.assertNotIn(b'park', data.lower())
        self.assertEqual(shape.approach_split_document(ground_raw, plan_dir, site.ann_raw, site.pose_raw, 'ground.json')[1], data)
        # Every sweep used must still be the receipted file.
        sweep = plan_dir / shape.SWEEP_DIR / f"{split['far'][0]['timestampNs']}.feather"; sweep.write_bytes(sweep.read_bytes() + b'\0')
        with self.assertRaisesRegex(ValueError, 'changed on disk'):
            shape.approach_split_document(ground_raw, plan_dir, site.ann_raw, site.pose_raw, 'ground.json')

    def test_too_few_and_not_stationary_refuse_the_check(self):
        site = Site(self.root / 'a', drive([120.] * 10 + [121.] * 10 + [122.] * 20))  # kept 3: the app shape is measured, h = 1
        plan_dir, _, block = site.measured()
        self.assertEqual(block['status'], 'measured')
        doc, _ = shape.approach_split_document(site.ground_with(block), plan_dir, site.ann_raw, site.pose_raw)
        self.assertEqual((doc['approachSplit']['h'], doc['carShape']['status'], doc['carShape']['refusal']['code']), (1, 'refused', 'too-few-sweeps'))
        poses = approach_drive()
        moving = Site(self.root / 'b', poses, rows=car_rows(poses, drift=2.0))
        plan_dir, _, block = moving.measured()
        doc, _ = shape.approach_split_document(moving.ground_with(block), plan_dir, moving.ann_raw, moving.pose_raw)
        self.assertEqual(doc['carShape']['refusal']['code'], 'not-stationary')
        with self.assertRaisesRegex(ValueError, 'does not apply'):
            shape.approach_split_document(site.ground_with(None), plan_dir, site.ann_raw, site.pose_raw)


if __name__ == '__main__':
    unittest.main()

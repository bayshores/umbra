"""Metric K (LiDAR shadow) on any built AV2 scene: the frozen lidarShadow rule (protocol v3.1 evaluation.lidarShadow, carried
unchanged in every later scene protocol) on the recorded vehicle's tested box. Report only: no target, no row class, never
a reference source for another metric.

  nice -n 10 python3 -I -B scripts/evaluation/metric_k.py --grounded G --assembled A --prepared P --out OUT
      [--sweep-dir D [--sweep-dir D ...] [--receipt R ...]]

Every input is found through the scene's own records and checked against its recorded SHA-256: the scene protocol
(ground.json protocol; its lidarShadow block must equal the v3.1 one), the assembled reference positions, the prepared
references, and the scene sweep, ego poses and sensor calibration named in scene.json measurements.provenance, and the
annotations named in ground.json sources.

Scene sweep (the frozen rule, as in .local/evaluation/av2-map-first-2026-10-11/v3.1-analysis/metric_k.py): every finite
return of the scene sweep, ray origin at its own capture time, laser check (a failure stops the run). K1 see-through and
K2 unboxed for the tested box (the averaged pose, ground.json cuboid with poseAverage) and, as the protocol's diagnostic,
for the sweep-time box of the same track (prepare_av2_ground.cuboids_from_annotations without averaging).

Scoring sweeps (only when ground.json has a car shape): the same K1/K2 for the tested box on each sweep listed in
ground.json carShape.sweeps.scoring (the sweeps the shape result is scored with), loaded by metric_f_car_top's listed-sweep
rule (listed SHA-256, receipt, annotation row of the tested track). Other boxes of the K2 rule are that sweep's own
annotations. A sweep failing the laser check is dropped and listed (the metric F scoring rule). Counts are summed over
the kept sweeps and cut depths pooled.

A scene without a recorded-vehicle box (declared-default track) is written as not scored. Blocked rows with LiDAR
evidence against them are metric F's likely-false rows (listed in the F results); K classifies no row.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.feather as feather

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f'cannot load {path}'
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


ground_tool = load('prepare_av2_ground', ROOT / 'scripts/california/prepare_av2_ground.py')
mf = load('metric_f_car_top', HERE / 'metric_f_car_top.py')
# Canonical JSON SHA-256 of protocol-v3.1.json evaluation.lidarShadow (frozen 2026-10-12, protocol sha256 6a7d3369...).
LIDAR_SHADOW_SHA = '49cd2c1e96357ef4d66450026e7cc84e39b539d412c12092e578d270955f8478'
LASER_LIMIT_DEG = .15
RESULT_MARGIN_M = .10
SENSORS = ('up_lidar', 'down_lidar')
SIDES = {(0, 1): 'front', (0, -1): 'rear', (1, 1): 'left', (1, -1): 'right', (2, 1): 'top', (2, -1): 'bottom'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def checked(base, record, what):
    """A {path, sha256} record resolved against base; the file must match."""
    path = (Path(base) / record['path']).resolve()
    assert path.is_file(), f'{what} is missing: {path}'
    assert sha(path) == record['sha256'], f'{what} differs from its recorded SHA-256'
    return path


# ------------------------------------------------------------------ the frozen rule (copied from the v3.1 script)
def box_frame(box):
    m = np.asarray(box['worldFromBox'], float); c = np.asarray(box['center'], float)
    return m, c, np.array([box['length'], box['width'], box['height']]) / 2


def local(points, m, c):
    return (np.asarray(points, float) - c) @ m


def touches(a, b, half, shrink):
    """Closed segments a->b (box-local) against the box with half sizes half - shrink (inclusive slab test); shrink is a
    number or one value per segment."""
    h = np.asarray(half, float)[None, :] - np.broadcast_to(np.asarray(shrink, float), (len(a),))[:, None]
    ok = (h >= 0).all(1); enter = np.zeros(len(a)); leave = np.ones(len(a))
    for k in range(3):
        d = b[:, k] - a[:, k]; flat = np.abs(d) < 1e-12
        ok &= ~(flat & ((a[:, k] < -h[:, k]) | (a[:, k] > h[:, k])))
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = (-h[:, k] - a[:, k]) / d; hi = (h[:, k] - a[:, k]) / d
        enter = np.where(flat, enter, np.maximum(enter, np.minimum(lo, hi))); leave = np.where(flat, leave, np.minimum(leave, np.maximum(lo, hi)))
    return ok & (enter <= leave), enter, leave


def cut_depth(a, b, half, margin):
    """Largest equal shrink (>= margin) at which each segment still touches the box, by bisection to 1 mm."""
    lo = np.full(len(a), float(margin)); hi = np.full(len(a), float(half.min()))
    while (hi - lo).max(initial=0) >= 1e-3:
        mid = (lo + hi) / 2; t = touches(a, b, half, mid)[0]
        lo = np.where(t, mid, lo); hi = np.where(t, hi, mid)
    return lo


def side_of(points, half):
    """Box side a box-local point is nearest to (largest |coordinate| / half size)."""
    r = np.abs(points) / half; k = r.argmax(1); s = np.sign(points[np.arange(len(points)), k]).astype(int)
    return [SIDES[(int(a), int(b) if b else 1)] for a, b in zip(k, s)]


def stats(values):
    v = np.sort(np.asarray(values, float))
    if not len(v):
        return None
    q = lambda p: float(v[max(math.ceil(p * len(v)) - 1, 0)])  # noqa: E731
    return {'median': q(.5), 'p95': q(.95), 'max': float(v[-1])}


def evaluate(box, others, positions, origins, dz, sensor, spec):
    """The frozen K1/K2 rule. Returns (report, raw) where raw keeps the per-return values for pooling over sweeps."""
    margin, ground_m, near_m, buffer_m = spec['marginMeters'], spec['groundMeters'], spec['neighbourhoodMeters'], spec['otherObjectBufferMeters']
    m, c, half = box_frame(box)
    lp = local(positions, m, c); lo = local(origins, m, c)
    outside = (np.abs(lp) - half).max(1)
    inside = outside <= 0
    hit, enter, leave = touches(lo, lp, half, margin)
    k1 = (outside > margin) & hit
    idx = np.flatnonzero(k1)
    depth = cut_depth(lo[idx], lp[idx], half, margin) if len(idx) else np.zeros(0)
    mid = lo[idx] + ((enter[idx] + leave[idx]) / 2)[:, None] * (lp[idx] - lo[idx]) if len(idx) else np.zeros((0, 3))
    sides = side_of(mid, half) if len(idx) else []
    ground = np.isfinite(dz) & (np.abs(dz) <= ground_m)
    # K2: beside the box, above the ground, not above the box top + buffer, outside every other cuboid grown by the buffer.
    near = (np.abs(lp[:, 0]) <= half[0] + near_m) & (np.abs(lp[:, 1]) <= half[1] + near_m) & (lp[:, 2] <= half[2] + buffer_m)
    above = np.isfinite(dz) & (dz > ground_m)
    other = np.zeros(len(positions), bool)
    for o in others:
        om, oc, oh = box_frame(o); other |= (np.abs(local(positions, om, oc)) <= oh + buffer_m).all(1)
    k2 = (outside > margin) & near & above & ~other & ~k1
    j = np.flatnonzero(k2); excess = np.abs(lp[j]) - half; excess[:, 2] = lp[j, 2] - half[2]
    k2_side = []
    for row, e in zip(j, excess):
        a = int(e.argmax()); s = ['rear', 'front'][int(lp[row, 0] > 0)] if a == 0 else ['right', 'left'][int(lp[row, 1] > 0)] if a == 1 else 'top'
        k2_side.append((s, float(outside[row])))
    raw = {'inside': int(inside.sum()), 'insideBySensor': {s: int((inside & (sensor == s)).sum()) for s in SENSORS},
           'k1': int(k1.sum()), 'k1Ground': int((k1 & ground).sum()), 'k1BySensor': {s: int((k1 & (sensor == s)).sum()) for s in SENSORS},
           'depth': [float(d) for d in depth], 'sides': sides, 'k2Side': k2_side}
    return report(raw), raw


def report(raw):
    by_side = {}
    for s, d in zip(raw['sides'], raw['depth']):
        e = by_side.setdefault(s, {'count': 0, 'maxCutM': 0.}); e['count'] += 1; e['maxCutM'] = max(e['maxCutM'], d)
    k2 = {}
    for s, v in raw['k2Side']:
        k2.setdefault(s, []).append(v)
    return {
        'insideBox': {'count': raw['inside'], 'bySensor': raw['insideBySensor']},
        'seeThroughK1': {'count': raw['k1'], 'ground': raw['k1Ground'], 'aboveGround': raw['k1'] - raw['k1Ground'],
                         'bySensor': raw['k1BySensor'], 'cutDepthM': stats(raw['depth']), 'bySideOfDeepestPass': dict(sorted(by_side.items())),
                         'cutAtLeastResultMarginM': int(sum(d >= RESULT_MARGIN_M for d in raw['depth'])), 'resultMarginM': RESULT_MARGIN_M},
        'unboxedK2': {'count': len(raw['k2Side']),
                      'bySide': {s: {'count': len(v), 'maxProtrusionM': max(v), 'p95ProtrusionM': (stats(v) or {})['p95']} for s, v in sorted(k2.items())}},
    }


def pool(raws):
    out = {'inside': 0, 'insideBySensor': dict.fromkeys(SENSORS, 0), 'k1': 0, 'k1Ground': 0, 'k1BySensor': dict.fromkeys(SENSORS, 0),
           'depth': [], 'sides': [], 'k2Side': []}
    for r in raws:
        for k in ('inside', 'k1', 'k1Ground'):
            out[k] += r[k]
        for k in ('insideBySensor', 'k1BySensor'):
            for s in SENSORS:
                out[k][s] += r[k][s]
        for k in ('depth', 'sides', 'k2Side'):
            out[k] += r[k]
    return out


# ------------------------------------------------------------------ sweep geometry
def ray_origins(world_points, origin, ts, offset, laser, poses, calib):
    """Each return's own sensor origin at its capture time (world frame) and the laser check (largest per-laser elevation
    spread in the assigned sensor frame at capture time)."""
    qi, ti = mf.capture_poses(ts + offset, poses)
    city = np.column_stack([world_points[:, 0] + origin[0], -world_points[:, 2] + origin[1], world_points[:, 1] + origin[2]])
    conj = qi * np.array([1., -1., -1., -1.]); ego_at_capture = mf.rotate(conj, city - ti)
    sensor = np.where(laser < 32, 'up_lidar', 'down_lidar')
    origins = np.empty((len(world_points), 3)); spreads = {}
    for name in SENSORS:
        sel = sensor == name; r = calib.loc[name]; t_s = np.array([r.tx_m, r.ty_m, r.tz_m]); rot = mf.quat_to_matrix([r.qw, r.qx, r.qy, r.qz])
        origins[sel] = mf.world(mf.rotate(qi[sel], t_s) + ti[sel], origin)
        q = (ego_at_capture[sel] - t_s) @ rot; elevation = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
        for k in np.unique(laser[sel]):
            spreads[int(k)] = float(elevation[laser[sel] == k].std())
    return origins, sensor, spreads


def heights(terrain, positions, origins, origin):
    finite = np.isfinite(positions).all(1) & np.isfinite(origins).all(1); dz = np.full(len(positions), np.nan)
    dz[finite] = positions[finite, 1] - (terrain.sample(positions[finite][:, [0, 2]])[0] - origin[2])
    return finite, dz


def write(out_path, result):
    text = json.dumps(result, indent=1, sort_keys=True) + '\n'
    assert 'park' not in text.lower(), 'banned vehicle-status word'
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    print(hashlib.sha256(text.encode()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--grounded', required=True); parser.add_argument('--assembled', required=True)
    parser.add_argument('--prepared', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--sweep-dir', action='append', default=[]); parser.add_argument('--receipt', action='append', default=[])
    args = parser.parse_args(argv)
    g, a, p, out_path = Path(args.grounded), Path(args.assembled), Path(args.prepared), Path(args.out)
    if os.path.lexists(out_path):
        parser.error('--out exists (outputs are never overwritten)')
    ground = json.loads((g / 'ground.json').read_bytes()); scene = json.loads((g / 'scene.json').read_bytes())
    m = scene['measurements']
    assert m['ground']['sha256'] == sha(g / 'ground.json'), 'scene.json names another ground.json'
    protocol_path = checked(g, ground['protocol'], 'scene protocol')
    spec = json.loads(protocol_path.read_bytes())['evaluation']['lidarShadow']
    assert canonical_sha(spec) == LIDAR_SHADOW_SHA, 'the scene protocol lidarShadow block differs from protocol v3.1'
    assert checked(g, ground['sources']['assembledScene'], 'assembled scene.json').samefile(a / 'scene.json'), '--assembled is not this scene'
    refs_path = checked(g, ground['sources']['references'], 'prepared references')
    assert refs_path.samefile(p / 'references.npz'), '--prepared is not this scene'
    pv = m['provenance']; base = g / pv['originalSourceRecordBase']; src = pv['originalSources']
    sweep_path, poses_path, calib_path = (checked(base, src[k], k) for k in ('lidar', 'poses', 'extrinsics'))
    ann_path = checked(g, ground['sources']['annotations'], 'annotations')
    assert sha(poses_path) == ground['sources']['poses']['sha256'], 'scene and ground name different poses'
    log_id = pv['logId']; lidar_ts = int(pv['lidarTimestampNs'])
    count = m['count']; origin = np.asarray(scene['transform']['cityOrigin'], float)
    inputs = {'protocol': {'file': protocol_path.name, 'sha256': sha(protocol_path)}, 'ground.json': sha(g / 'ground.json'),
              'scene.json': sha(g / 'scene.json'), 'references.npz': sha(refs_path),
              'sweep': {'file': f'sensors/lidar/{lidar_ts}.feather', 'sha256': sha(sweep_path)},
              'city_SE3_egovehicle.feather': sha(poses_path), 'egovehicle_SE3_sensor.feather': sha(calib_path),
              'annotations.feather': sha(ann_path), 'script': sha(Path(__file__)),
              'prepare_av2_ground.py': sha(ROOT / 'scripts/california/prepare_av2_ground.py'), 'metric_f_car_top.py': sha(HERE / 'metric_f_car_top.py')}
    head = {'metric': 'K (LiDAR shadow), the frozen protocol v3.1 evaluation.lidarShadow rule carried in the scene protocol',
            'scene': scene['id'], 'logId': log_id,
            'label': 'report only, no target; partly independent (annotators drew the boxes from the same LiDAR); never a reference source',
            'definition': spec, 'definitionCanonicalSha256': LIDAR_SHADOW_SHA,
            'rows': "Blocked rows with LiDAR evidence against them are metric F's likely-false rows (F results); K classifies no row."}
    boxes = [c for c in ground['cuboids']['items'] if 'poseAverage' in c]
    assert len(boxes) <= 1, 'more than one averaged box'
    if not boxes:
        write(out_path, {**head, 'inputs': inputs, 'outcome': 'not scored', 'reason': 'no recorded-vehicle box (declared-default track)'})
        return
    box = boxes[0]; others = [c for c in ground['cuboids']['items'] if c['track'] != box['track']]

    # Scene sweep: the frozen rule on the publisher positions of the scene's reference buffers.
    pos_path = a / m['buffers']['referencePositions']; assert sha(pos_path) == m['artifactHashes']['referencePositions'], 'referencepositions.bin differs'
    inputs['referencepositions.bin'] = sha(pos_path)
    positions = np.frombuffer(pos_path.read_bytes(), '<f8').reshape(count, 3)
    refs = np.load(refs_path); source = refs['sourceIndex']
    sweep = feather.read_table(sweep_path).to_pandas()
    assert np.array_equal(refs['egoXYZ'], sweep[['x', 'y', 'z']].to_numpy(float)[source]), 'Prepared references disagree with the sweep'
    laser = sweep['laser_number'].to_numpy()[source]; offset = sweep['offset_ns'].to_numpy(np.int64)[source]
    calib = feather.read_table(calib_path).to_pandas().set_index('sensor_name')
    poses = feather.read_table(poses_path).to_pandas()
    origins, sensor, spreads = ray_origins(positions, origin, lidar_ts, offset, laser, poses, calib)
    assert max(spreads.values()) < LASER_LIMIT_DEG, f'Laser assignment check failed: largest elevation spread {max(spreads.values()):.3f} deg'
    t = ground['terrain']; window = np.frombuffer((g / 'terrain.bin').read_bytes(), '<f4').reshape(t['rows'], t['columns'])
    terrain = ground_tool.Terrain(window, t['column0'], t['row0'], t['rasterShape'], t['sim2'], ground['frame']['cityOrigin'])
    inputs['terrain.bin'] = sha(g / 'terrain.bin')
    keep, dz = heights(terrain, positions, origins, origin)
    ann_bytes, poses_bytes = ann_path.read_bytes(), poses_path.read_bytes()
    pose_table = ground_tool._poses(poses_bytes)
    at_sweep = ground_tool.cuboids_from_annotations(ann_bytes, pose_table, lidar_ts, origin, None)
    sweep_box = next((c for c in at_sweep if c['track'] == box['track']), None)
    up = sensor == 'up_lidar'
    travel = np.linalg.norm(np.diff(origins[up][np.argsort(offset[up])][[0, -1]], axis=0))
    scene_part = {
        'returns': {'total': count, 'finite': int(keep.sum()), 'bySensor': {s: int((keep & (sensor == s)).sum()) for s in SENSORS}},
        'laserAssignmentCheck': {'rule': 'laser 0-31 up_lidar, 32-63 down_lidar; elevation spread per laser in its sensor frame at capture time',
                                 'largestElevationSpreadDeg': max(spreads.values()), 'medianElevationSpreadDeg': float(np.median(list(spreads.values()))),
                                 'limitDeg': LASER_LIMIT_DEG},
        'rayOriginTravelDuringSweepM': float(travel),
        'testedBox': {'source': 'ground.json averaged pose (poseAverage)',
                      **evaluate(box, others, positions[keep], origins[keep], dz[keep], sensor[keep], spec)[0]},
        'diagnosticSweepBox': None if sweep_box is None else {
            'source': 'sweep-time annotation box of the same track (cuboids_from_annotations without averaging)',
            **evaluate(sweep_box, others, positions[keep], origins[keep], dz[keep], sensor[keep], spec)[0]},
    }

    # Scoring sweeps: the same rule for the tested box on each listed sweep.
    if ground.get('carShape') is None:
        scoring = {'outcome': 'not applicable', 'reason': 'no car shape in ground.json (box-only scene)'}
    else:
        if not args.sweep_dir:
            parser.error('ground.json has a car shape: give --sweep-dir (and --receipt) for its scoring sweeps')
        listed, _, _ = mf.scoring_list(g / 'ground.json')
        ann = feather.read_table(ann_path).to_pandas()
        timestamps = set(int(x) for x in ann[ann.track_uuid == box['track']].timestamp_ns)
        sweeps, _ = mf.gather_listed_sweeps(args.sweep_dir, args.receipt, listed, timestamps, log_id)
        raws, per_sweep, dropped = [], [], []
        for s in sweeps:
            ts = s['timestampNs']
            table = feather.read_table(s['path'], columns=['x', 'y', 'z', 'laser_number', 'offset_ns']).to_pandas()
            ego = table[['x', 'y', 'z']].to_numpy(np.float64)
            q0, t0 = mf.capture_poses([ts], poses)
            pts = mf.world(mf.rotate(np.repeat(q0, len(ego), 0), ego) + t0, origin)
            o, sen, sp = ray_origins(pts, origin, ts, table['offset_ns'].to_numpy(np.int64), table['laser_number'].to_numpy(), poses, calib)
            spread = max(sp.values())
            if spread >= LASER_LIMIT_DEG:
                dropped.append({'timestampNs': str(ts), 'sha256': s['sha256'], 'laserSpreadDeg': round(spread, 4), 'reason': 'laser check failed: dropped'})
                continue
            ok, d = heights(terrain, pts, o, origin)
            own = [c for c in ground_tool.cuboids_from_annotations(ann_bytes, pose_table, ts, origin, None) if c['track'] != box['track']]
            rep, raw = evaluate(box, own, pts[ok], o[ok], d[ok], sen[ok], spec)
            raws.append(raw)
            per_sweep.append({'timestampNs': str(ts), 'sha256': s['sha256'], 'receiptVerified': s['receiptVerified'], 'finiteReturns': int(ok.sum()),
                              'laserSpreadDeg': round(spread, 4), 'k1': rep['seeThroughK1']['count'], 'k1CutDepthM': rep['seeThroughK1']['cutDepthM'],
                              'k2': rep['unboxedK2']['count'], 'inside': rep['insideBox']['count']})
        scoring = {'outcome': 'reported' if raws else 'not scored', 'listed': len(listed), 'used': len(raws), 'dropped': dropped,
                   'rule': 'tested box only; each sweep with its own annotations as the other boxes; counts summed, cut depths pooled',
                   'testedBox': report(pool(raws)) if raws else None, 'perSweep': per_sweep}
        if not raws:
            scoring['reason'] = 'every listed scoring sweep failed the laser check'
    write(out_path, {**head, 'inputs': inputs, 'outcome': 'reported', 'track': box['track'], 'sceneSweep': scene_part, 'scoringSweeps': scoring})


if __name__ == '__main__':
    sys.exit(main())

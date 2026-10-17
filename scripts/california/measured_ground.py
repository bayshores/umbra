"""Build umbra's measured ground from receipted other sweeps of the same AV2 log.

The publisher raster remains byte-identical for evaluation reference selection. It is never
a runtime fallback. Calibration sweeps and the scored frame cannot contribute to the fitted
surface or support mask. The temporal guard is explicit. No downloads or optional dependencies.
"""
import argparse
from collections import defaultdict
from hashlib import md5, sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[2]
CODE_BYTES = Path(__file__).read_bytes()
CODE_SHA256 = sha256(CODE_BYTES).hexdigest()
EXCLUDE_NS = 250_000_000
PARAMETERS = dict(cellMeters=.5, fitRadiusMeters=1., nearestSupportMeters=.5,
                  minimumReturns=12, minimumSweeps=3, groundGateMeters=.25,
                  drivableMarginMeters=.5, residualFloorMeters=.025, madMultiplier=3.,
                  robustRefits=2, maximumSlope=.25, windowHalfWidthMeters=32., maximumSupportSeconds=1., bandRangeRule='bins')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def pinned(path, expected):
    data = Path(path).read_bytes()
    require(digest(data) == expected, f'Input checksum mismatch: {path}')
    return data


def table(data, columns):
    t = feather.read_table(BytesIO(data), use_threads=False)
    require(len(set(t.column_names)) == len(t.column_names) and set(columns) <= set(t.column_names), 'Missing/duplicate Feather columns')
    return t.select(columns)


def rotation(row):
    w, x, y, z = [row[k] for k in ('qw', 'qx', 'qy', 'qz')]
    require(all(math.isfinite(v) for v in (w, x, y, z)) and abs(w*w+x*x+y*y+z*z-1) <= 1e-6, 'Invalid pose quaternion')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def partition_sweeps(records, frame_timestamp, maximum_seconds, scoring_sha256):
    require(isinstance(scoring_sha256, str) and re.fullmatch(r'[0-9a-f]{64}', scoring_sha256), 'A valid scoring-sweep sha256 is required')
    frame = int(frame_timestamp)
    require(all(isinstance(r['timestampNs'], str) and re.fullmatch(r'[0-9]{1,32}', r['timestampNs']) for r in records), 'Invalid support timestamp')
    ordered = sorted([{**r, 'timestampNs': str(int(r['timestampNs']))} for r in records], key=lambda r: int(r['timestampNs']))
    require(len({r['timestampNs'] for r in ordered}) == len(ordered), 'Duplicate support timestamps')
    within_window = lambda r: maximum_seconds is None or abs(int(r['timestampNs'])-frame) <= maximum_seconds*1e9
    allowed = [r for r in ordered if abs(int(r['timestampNs']) - frame) > EXCLUDE_NS and within_window(r)]
    require(all(r['sha256'] != scoring_sha256 for r in allowed), 'The scoring frame sweep bytes are refused as support/calibration')
    require(len({r['sha256'] for r in allowed}) == len(allowed), 'Duplicate support/calibration sweep bytes')
    excluded = [{**r, 'reason': 'scoring frame' if int(r['timestampNs']) == frame else 'within temporal exclusion'}
                for r in ordered if abs(int(r['timestampNs']) - frame) <= EXCLUDE_NS]
    excluded += [{**r, 'reason': 'outside maximum support time window'} for r in ordered if not within_window(r)]
    return {'frameTimestampNs': str(frame), 'excludedNearNs': EXCLUDE_NS,
            'fit': [r for i, r in enumerate(allowed) if i % 4 != 0],
            'calibration': [r for i, r in enumerate(allowed) if i % 4 == 0], 'excluded': excluded}


def require_support_timestamp(timestamp, frame_timestamp):
    require(int(timestamp) != int(frame_timestamp), 'The scoring frame sweep is refused as support')
    require(abs(int(timestamp) - int(frame_timestamp)) > EXCLUDE_NS, 'Support sweep lies inside the temporal exclusion')


def receipt_sweeps(path, log, frame, ground):
    receipt = read_json(path)
    require(receipt['log'] == log, 'Receipt log mismatch')
    if receipt['schema'] == 'shape-v2-download-receipt':
        plan = json.loads(pinned(path.parent/'sweep-plan.json', receipt['planSha256']))
        require(plan['log'] == log, 'Sweep plan log mismatch')
        require(plan['inputs']['poses']['sha256'] == ground['sources']['poses']['sha256'] and
                plan['inputs']['annotations']['sha256'] == ground['sources']['annotations']['sha256'], 'Receipt poses/annotations differ from ground inputs')
        planned = {str(int(r['timestampNs'])): r for r in plan['sweeps']}
        for row in receipt['sweeps']:
            expected = planned.get(str(int(row['timestampNs'])))
            require(expected and row['bytes'] == expected['bytes'] and row['sizeMatches'] is True, 'Invalid shape sweep receipt record')
    else:
        require(receipt['schema'] == 'support-sweeps-download-receipt' and int(receipt['frameSweepTimestampNs']) == int(frame) and
                receipt['rule']['excludesFrameSweep'] is True, 'Support receipt schema/frame mismatch')
        listing = json.loads(pinned(path.parent/'listing.json', receipt['listingSha256']))
        prefix = f"datasets/av2/sensor/{receipt['split']}/{log}/sensors/lidar/"
        require(listing['prefix'] == prefix, 'Support listing log mismatch')
        objects = {r['key']: r for r in listing['objects']}
        for row in receipt['sweeps']:
            expected = objects.get(row['key'])
            require(row['key'] == prefix+f"{int(row['timestampNs'])}.feather" and expected and
                    row['bytes'] == row['contentLength'] == expected['size'] and row['status'] == 200 and
                    row['sizeMatches'] is True and row['md5MatchesEtag'] is True and
                    row['md5'] == row['etag'] == expected['etag'] and re.fullmatch(r'[0-9a-f]{32}', row['md5']), 'Invalid support sweep receipt record')
            require(int(row['timestampNs']) != int(frame), 'Support receipt contains the scoring sweep')
    for row in receipt['sweeps']:
        require(row['file'] == f"sweeps/{int(row['timestampNs'])}.feather" and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'Invalid sweep receipt path/hash')
    return [{k: r[k] for k in ('timestampNs', 'sha256', 'bytes')} for r in receipt['sweeps']], receipt


def inside_polygons(xz, areas, margin=0.):
    p = np.asarray(xz, float).reshape(-1, 2); inside = np.zeros(len(p), bool)
    for area in areas:
        poly = np.asarray(area['outline'], float); hit = np.zeros(len(p), bool)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            with np.errstate(divide='ignore', invalid='ignore'):
                hit ^= ((a[1] > p[:, 1]) != (b[1] > p[:, 1])) & (p[:, 0] < (b[0]-a[0])*(p[:, 1]-a[1])/(b[1]-a[1])+a[0])
            if margin:
                e = b-a; norm = float(e@e)
                t = np.clip((p-a)@e/norm, 0, 1) if norm else np.zeros(len(p))
                inside |= np.sum((p-a-t[:, None]*e)**2, axis=1) <= margin**2
        inside |= hit
    return inside


def raster_heights(ground, heights, xz):
    t = ground['terrain']; sim = t['sim2']; r = np.asarray(sim['R']).reshape(2, 2); o = ground['frame']['cityOrigin']
    city = np.column_stack([xz[:, 0]+o[0], -xz[:, 1]+o[1]])
    image = sim['s']*(city@r.T+sim['t']); idx = np.floor(image).astype(np.int64)-[t['column0'], t['row0']]
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < t['columns']) & (idx[:, 1] >= 0) & (idx[:, 1] < t['rows'])
    out = np.full(len(xz), np.nan); out[valid] = heights[idx[valid, 1], idx[valid, 0]]-o[2]
    return out


def outside_cuboids(ego, rows):
    keep = np.ones(len(ego), bool)
    for row in rows:
        c = np.array([row[k] for k in ('tx_m', 'ty_m', 'tz_m')]); half = np.array([row[k] for k in ('length_m', 'width_m', 'height_m')])/2
        local = (ego-c)@rotation(row)
        keep &= ~np.all(np.abs(local) <= half, axis=1)
    return keep


def camera_visible(points, boxes):
    keep = np.ones(len(points), bool)
    for box in boxes:
        m = np.asarray(box['worldFromBox']); c = np.asarray(box['center']); half = np.array([box['length'], box['width'], box['height']])/2
        a = -c@m; d = points@m; enter = np.zeros(len(points)); leave = np.ones(len(points)); ok = np.ones(len(points), bool)
        for k in range(3):
            flat = np.abs(d[:, k]) < 1e-9
            ok &= ~(flat & ((a[k] < -half[k]) | (a[k] > half[k])))
            with np.errstate(divide='ignore', invalid='ignore'):
                lo = (-half[k]-a[k])/d[:, k]; hi = (half[k]-a[k])/d[:, k]
            enter = np.where(flat, enter, np.maximum(enter, np.minimum(lo, hi)))
            leave = np.where(flat, leave, np.minimum(leave, np.maximum(lo, hi)))
        keep &= ~(ok & (enter <= leave))
    return keep


def fit_plane(points, sweep_ids, xz, params=PARAMETERS):
    """Robust, distance-weighted local plane; absence of measured support returns None."""
    p = np.asarray(points, float); ids = np.asarray(sweep_ids); delta = p[:, [0, 2]]-xz
    distance = np.sum(delta**2, axis=1); mask = distance < params['fitRadiusMeters']**2
    a = np.column_stack([np.ones(len(p)), delta]); weights = np.maximum(0, 1-distance/params['fitRadiusMeters']**2)**2
    for _ in range(params['robustRefits']+1):
        if mask.sum() < params['minimumReturns'] or len(np.unique(ids[mask])) < params['minimumSweeps']:
            return None
        if distance[mask].min() > params['nearestSupportMeters']**2:
            return None
        w = np.sqrt(weights[mask]); coefficients, _, rank, _ = np.linalg.lstsq(a[mask]*w[:, None], p[mask, 1]*w, rcond=None)
        if rank != 3:
            return None
        residual = p[:, 1]-a@coefficients; centre = np.median(residual[mask]); mad = np.median(np.abs(residual[mask]-centre))
        cutoff = max(params['residualFloorMeters'], params['madMultiplier']*1.4826*mad)
        revised = mask & (np.abs(residual-centre) <= cutoff)
        if np.array_equal(revised, mask):
            break
        mask = revised
    if mask.sum() < params['minimumReturns'] or len(np.unique(ids[mask])) < params['minimumSweeps'] or distance[mask].min() > params['nearestSupportMeters']**2:
        return None
    w = np.sqrt(weights[mask]); coefficients, _, rank, _ = np.linalg.lstsq(a[mask]*w[:, None], p[mask, 1]*w, rcond=None)
    if rank != 3:
        return None
    if np.hypot(coefficients[1], coefficients[2]) > params['maximumSlope']:
        return None
    return float(coefficients[0])


def fit_surface(points, sweep_ids, areas, params=PARAMETERS, tier0=None):
    p = np.asarray(points, float).reshape(-1, 3); ids = np.asarray(sweep_ids)
    half = params['windowHalfWidthMeters']; step = params['cellMeters']; count = int(math.ceil(2*half/step))+1
    origin = [-half, -half]; heights = np.full((count, count), np.nan)
    if tier0 is not None:
        require(tier0['originXZ'] == origin and tier0['cellMeters'] == step and tier0['columns'] == tier0['rows'] == count, 'Tier-0 grid differs')
        heights = np.array([np.nan if h is None else h for h in tier0['heights']]).reshape(count, count)
    nodes = np.array([(x, z) for z in np.arange(count)*step-half for x in np.arange(count)*step-half])
    candidates = inside_polygons(nodes, areas, params['drivableMarginMeters'])
    bucket_size = params['fitRadiusMeters']; buckets = defaultdict(list)
    for i, key in enumerate(np.floor(p[:, [0, 2]]/bucket_size).astype(int)):
        buckets[tuple(key)].append(i)
    buckets = {k: np.asarray(v, np.int64) for k, v in buckets.items()}
    for i in np.flatnonzero(candidates & ~np.isfinite(heights.ravel())):
        node = nodes[i]; key = np.floor(node/bucket_size).astype(int)
        parts = [buckets[(key[0]+dx, key[1]+dz)] for dx in (-1, 0, 1) for dz in (-1, 0, 1) if (key[0]+dx, key[1]+dz) in buckets]
        if not parts:
            continue
        indices = np.concatenate(parts); value = fit_plane(p[indices], ids[indices], node, params)
        if value is not None:
            heights.flat[i] = value
    values = heights[np.isfinite(heights)]
    surface = {'version': 'measured-ground-v1', 'originXZ': origin, 'cellMeters': step, 'columns': count, 'rows': count,
            'heights': [float(h) if np.isfinite(h) else None for h in heights.ravel()],
            'heightBounds': [float(values.min()), float(values.max())] if len(values) else None}
    if tier0 is not None:
        surface.update(version='measured-ground-v2', tiers=[None if h is None else 0 if tier0['heights'][i] is not None else 1
                                                        for i, h in enumerate(surface['heights'])])
    return surface


def partition_tiers(records, tier0, scoring_sha256):
    frame = tier0['frameTimestampNs']
    full = partition_sweeps(records, frame, None, scoring_sha256)
    require(all(r['sha256'] != scoring_sha256 and int(r['timestampNs']) != int(frame) for r in records), 'The full receipt must exclude scoring sweep bytes and timestamp')
    by_time = {str(int(r['timestampNs'])): r for r in records}
    for group in ('fit', 'calibration', 'excluded'):
        for row in tier0[group]:
            if int(row['timestampNs']) == int(frame):
                require(row['sha256'] == scoring_sha256, 'Tier-0 scoring identity differs')
                continue
            other = by_time.get(row['timestampNs'])
            require(other and all(other[k] == row[k] for k in ('sha256', 'bytes')), 'Full receipt is missing or changed a v1 sweep')
    near = {r['timestampNs'] for group in ('fit', 'calibration') for r in tier0[group]}
    extra = [r for r in records if str(int(r['timestampNs'])) not in near]
    other = partition_sweeps(extra, frame, None, scoring_sha256)
    tiers = [{k: tier0[k] for k in ('fit', 'calibration')}, {k: other[k] for k in ('fit', 'calibration')}]
    return {**full, 'fit': [r for t in tiers for r in t['fit']], 'calibration': [r for t in tiers for r in t['calibration']], 'tiers': tiers}


def surface_heights(surface, xz):
    p = np.asarray(xz, float).reshape(-1, 2); f = (p-surface['originXZ'])/surface['cellMeters']; cells = np.floor(f).astype(int)
    c, r = cells.T; valid = (c >= 0) & (r >= 0) & (c < surface['columns']-1) & (r < surface['rows']-1)
    out = np.full(len(p), np.nan); i = np.flatnonzero(valid); c, r = c[i], r[i]
    k = r*surface['columns']+c; upper = f[i, 1]-r > f[i, 0]-c
    heights = np.array([np.nan if h is None else h for h in surface['heights']])
    h, diagonal, side = heights[k], heights[k+surface['columns']+1], heights[k+np.where(upper, surface['columns'], 1)]
    dx = np.where(upper, diagonal-side, side-h)/surface['cellMeters']; dz = np.where(upper, side-h, diagonal-side)/surface['cellMeters']
    out[i] = h+dx*(p[i, 0]-surface['originXZ'][0]-c*surface['cellMeters'])+dz*(p[i, 1]-surface['originXZ'][1]-r*surface['cellMeters'])
    return out


def sweep_consistency(points, surface, ego_distance):
    p = np.asarray(points, float).reshape(-1, 3); heights = surface_heights(surface, p[:, [0, 2]])
    p = p[np.isfinite(heights)]; residual = p[:, 1]-heights[np.isfinite(heights)]
    result = {'overlapReturns': len(p), 'egoDistanceFromFrameMeters': float(ego_distance), 'robustHeightOffsetMeters': None,
              'planeOffsetAtCameraMeters': None, 'tiltXDegrees': None, 'tiltZDegrees': None}
    if len(p) < 12:
        return {**result, 'reason': 'Under 12 returns overlapping tier-0 triangles'}
    a = np.column_stack([np.ones(len(p)), p[:, [0, 2]]]); mask = np.ones(len(p), bool)
    for _ in range(3):
        coefficients, _, rank, _ = np.linalg.lstsq(a[mask], residual[mask], rcond=None)
        if rank != 3:
            return {**result, 'reason': 'Overlapping returns do not constrain a plane'}
        errors = residual-a@coefficients; centre = np.median(errors[mask]); mad = np.median(np.abs(errors[mask]-centre))
        revised = mask & (np.abs(errors-centre) <= max(.025, 3*1.4826*mad))
        if revised.sum() < 12 or np.array_equal(revised, mask):
            break
        mask = revised
    coefficients, _, rank, _ = np.linalg.lstsq(a[mask], residual[mask], rcond=None)
    if rank != 3:
        return {**result, 'reason': 'Retained overlapping returns do not constrain a plane'}
    return {**result, 'robustHeightOffsetMeters': float(np.median(residual[mask])), 'planeOffsetAtCameraMeters': float(coefficients[0]),
            'tiltXDegrees': math.degrees(math.atan(float(coefficients[1]))), 'tiltZDegrees': math.degrees(math.atan(float(coefficients[2]))),
            'retainedReturns': int(mask.sum()), 'reason': None}


def build(site, output, params=PARAMETERS, support_receipt=None):
    site = Path(site).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Output already exists; use a new development variant directory')
    require(params.get('bandRangeRule', 'bins') in ('bins', 'linear'), 'Unsupported band range rule')
    scene = read_json(site/'grounded/scene.json'); ground = read_json(site/'grounded/ground.json')
    provenance = scene['measurements']['provenance']; log = provenance['logId']; frame = provenance['lidarTimestampNs']
    require(log.startswith(('8c52d911-', 'c780d53a-', '88f47a10-', 'fd5c6932-', 'e1d68dde-')), 'This development build requires an explicitly allowed log')
    receipts = sorted((site/'shape').glob(f'shape-v2-{log}/receipt.json')) if (site/'shape').is_dir() else []
    if support_receipt is not None:
        receipts = [Path(support_receipt).resolve()]
    elif not receipts:
        extra = ROOT/'.local/evaluation/pao-accuracy/support-sweeps-2026-10-16'/f"{log.split('-')[0]}-{provenance['crossingId']}"/'receipt.json'
        if extra.is_file():
            receipts = [extra]
    require(len(receipts) <= 1, 'Ambiguous sweep receipt')
    records = []
    if receipts:
        records, receipt = receipt_sweeps(receipts[0], log, frame, ground)
    scoring_hash = ((provenance.get('originalSources') or {}).get('lidar') or {}).get('sha256')  # validated by partition_sweeps
    partition = partition_sweeps(records, frame, params['maximumSupportSeconds'], scoring_hash)
    # The partition is written before loading any sensor bytes or performing a fit.
    output.mkdir(parents=True); write_json(output/'partition.json', partition)
    (output/'measured-ground-source.py').write_bytes(CODE_BYTES)
    data_dir = site/'data'/log
    pose_columns = ['qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m']
    poses_data = pinned(data_dir/'city_SE3_egovehicle.feather', ground['sources']['poses']['sha256'])
    poses = {int(r['timestamp_ns']): r for r in table(poses_data, ['timestamp_ns']+pose_columns).to_pylist()}
    annotations_data = pinned(data_dir/'annotations.feather', ground['sources']['annotations']['sha256'])
    annotations = defaultdict(list)
    for row in table(annotations_data, ['timestamp_ns', 'length_m', 'width_m', 'height_m']+pose_columns).to_pylist():
        annotations[int(row['timestamp_ns'])].append(row)
    raster = np.frombuffer(pinned(site/'grounded/terrain.bin', ground['terrain']['sha256']), '<f4').reshape(ground['terrain']['rows'], ground['terrain']['columns'])
    fit_points, fit_ids, calibration, counts = [], [], [], []
    origin = np.asarray(ground['frame']['cityOrigin'])
    for group in ('fit', 'calibration'):
        for sweep_id, row in enumerate(partition[group]):
            ts = int(row['timestampNs']); require_support_timestamp(ts, frame)
            require(ts in poses and ts in annotations, 'Support sweep needs exact timestamp poses and annotations')
            raw = pinned(receipts[0].parent/'sweeps'/f'{ts}.feather', row['sha256'])
            require(len(raw) == row['bytes'], 'Sweep byte count mismatch')
            require(digest(raw) != scoring_hash, 'The scoring frame sweep bytes are refused as support/calibration')
            if receipt['schema'] == 'support-sweeps-download-receipt':
                source_row = next(r for r in receipt['sweeps'] if int(r['timestampNs']) == ts)
                require(md5(raw).hexdigest() == source_row['md5'], 'Support sweep MD5/ETag mismatch')
            sweep = table(raw, ['x', 'y', 'z']); ego = np.column_stack([np.asarray(sweep[k]) for k in ('x', 'y', 'z')])
            valid = np.isfinite(ego).all(1); ego = ego[valid]; outside = outside_cuboids(ego, annotations[ts]); ego = ego[outside]
            pose = poses[ts]; city = ego@rotation(pose).T+np.array([pose[k] for k in ('tx_m', 'ty_m', 'tz_m')])
            delta = city-origin; points = delta[:, [0, 2, 1]]*np.array([1, 1, -1]); xz = points[:, [0, 2]]
            in_window = (np.abs(xz) <= params['windowHalfWidthMeters']).all(1)
            points = points[in_window]; xz = xz[in_window]
            h = raster_heights(ground, raster, xz)
            selected = np.isfinite(h) & (np.abs(points[:, 1]-h) <= params['groundGateMeters']) & inside_polygons(xz, ground['drivableAreas'], params['drivableMarginMeters'])
            selected_points = points[selected]
            counts.append({'timestampNs': str(ts), 'set': group, 'finiteReturns': int(valid.sum()), 'removedInsideAnnotations': int((~outside).sum()), 'groundCandidates': len(selected_points)})
            if group == 'fit':
                fit_points.append(selected_points); fit_ids.append(np.full(len(selected_points), sweep_id, np.int32))
            else:
                keep = inside_polygons(selected_points[:, [0, 2]], ground['drivableAreas']) & camera_visible(selected_points, ground['cuboids']['items'])
                calibration.extend(selected_points[keep].tolist())
    all_points = np.concatenate(fit_points) if fit_points else np.empty((0, 3)); all_ids = np.concatenate(fit_ids) if fit_ids else np.empty(0, np.int32)
    surface = fit_surface(all_points, all_ids, ground['drivableAreas'], params)
    surface.update(partition=partition, parameters=params, fitReturns=len(all_points),
                   sourceGroundSha256=digest((site/'grounded/ground.json').read_bytes()),
                   sourceSuggestionsSha256=digest((site/'grounded/suggestions.json').read_bytes()),
                   receiptSha256=digest(receipts[0].read_bytes()) if receipts else None,
                   receiptPath=os.path.relpath(receipts[0], output) if receipts else None,
                   processingSha256=CODE_SHA256,
                   sourceSite=os.path.relpath(site, output))
    write_json(output/'support-counts.json', counts)
    write_json(output/'calibration-points.json', calibration)
    shutil.copytree(site/'grounded', output/'grounded'); shutil.copytree(site/'assembled', output/'assembled')
    scene['processing']['groundPolicy'] = 'mappedGround with measured-ground-v1: continuous independently measured triangles; missing support is Unknown; publisher raster retained only as the evaluation selection reference'
    write_json(output/'grounded/scene.json', scene)
    ground.update(schemaVersion=4, measuredGround=surface)
    ground['limitations'] = [
        'Runtime ground is a continuous surface fitted to independent support sweeps; unmeasured regions stay holes.' if s.startswith('Mapped ground is publisher terrain') else
        'Displayed uncertainty is calibrated from independent end-to-end integer-click errors, without clipping; scoring-frame returns are excluded.' if 'along-ray uncertainty' in s else s
        for s in ground['limitations']]
    ground['limitations'].append('Support, calibration and scoring sweeps are from the same log and remain correlated; physical calibration and terrain contact remain provisional.')
    ground['license']['changes'] = 'Mapped geometry and annotations converted to umbra scene coordinates; independent LiDAR sweeps registered into a continuous measured ground surface with explicit support holes and click-error calibration. No endorsement implied.'
    ground['support'] = {'rows': [], 'count': 0, 'rule': 'independent measured surface only; scoring-sweep support rows are not used'}
    ground['alongRayUncertainty'] = {'method': 'supportSweepHoldout', 'heightBandMeters': None, 'rule': 'independent end-to-end integer-click horizontal error by hit range',
                                   'quantile': .975, 'minimumCount': 100, 'bins': []}
    for source in ground['sources'].values():
        if source is not None:
            source['path'] = os.path.relpath((site/'grounded'/source['path']).resolve(), output/'grounded')
    ground['processingCodeHashes']['scripts/california/measured_ground.py'] = CODE_SHA256
    protocol = {'schema': 'measured-ground-v1-development', 'parameters': params, 'excludeWithinNs': EXCLUDE_NS,
                'calibration': {'split': 'every fourth eligible sweep, excluded from fit and support mask', 'quantile': .975, 'minimumCount': 100,
                                'rangeEdgesM': [0, 10, 12, 15, 18, 20, 22, 25, 30], 'overLimit': 'display without clipping',
                                'rangeRule': params.get('bandRangeRule', 'bins'), 'linearKnotStepM': .5, 'linearPoolHalfWidthM': 1.},
                'legacyPlacementProtocolSha256': ground['protocol']['sha256']}
    write_json(output/'grounded/measured-ground-protocol.json', protocol)
    raw_protocol = (output/'grounded/measured-ground-protocol.json').read_bytes()
    ground['protocol'] = {'path': 'measured-ground-protocol.json', 'bytes': len(raw_protocol), 'sha256': digest(raw_protocol), 'readOnlyExternal': True}
    write_json(output/'grounded/ground.json', ground)
    result = subprocess.run(['node', '--input-type=module', '-', str(output)], input=FINALIZE_JS, text=True, capture_output=True, cwd=ROOT)
    (output/'finalize-stdout.txt').write_text(result.stdout); (output/'finalize-stderr.txt').write_text(result.stderr)
    require(result.returncode == 0, f'JavaScript calibration/finalization failed: {result.stderr}')
    return {'site': scene['id'], 'fitSweeps': len(partition['fit']), 'calibrationSweeps': len(partition['calibration']),
            'fitReturns': len(all_points), 'supportedVertices': sum(h is not None for h in surface['heights']), 'output': str(output)}


def build_v2(site, output, tier0_dir, support_receipt):
    site = Path(site).resolve(); output = Path(output).resolve(); tier0_dir = Path(tier0_dir).resolve(); receipt_path = Path(support_receipt).resolve()
    require(not output.exists(), 'Output already exists; use a new development variant directory')
    scene = read_json(site/'grounded/scene.json'); ground = read_json(site/'grounded/ground.json')
    old = read_json(tier0_dir/'grounded/ground.json'); entry = read_json(tier0_dir/'grounded/catalog-entry.json')
    require(digest((tier0_dir/'grounded/ground.json').read_bytes()) == entry['ground']['sha256'], 'Tier-0 catalog ground hash differs')
    base = old['measuredGround']; params = dict(PARAMETERS)
    require(old['schemaVersion'] == 4 and base['version'] == 'measured-ground-v1' and base['parameters'] == params, 'Tier 0 must be the unchanged binned v1 layer')
    require(base['sourceGroundSha256'] == digest((site/'grounded/ground.json').read_bytes()) and
            base['sourceSuggestionsSha256'] == digest((site/'grounded/suggestions.json').read_bytes()), 'Tier-0 source inputs differ')
    provenance = scene['measurements']['provenance']; log = provenance['logId']; frame = provenance['lidarTimestampNs']
    require(log.startswith(('8c52d911-', 'c780d53a-', '88f47a10-', 'fd5c6932-', 'e1d68dde-')), 'This development build requires an explicitly allowed log')
    require(base['partition']['frameTimestampNs'] == frame and base['partition']['excludedNearNs'] == EXCLUDE_NS, 'Tier-0 frame/guard differs')
    records, receipt = receipt_sweeps(receipt_path, log, frame, ground)
    require(receipt['schema'] == 'support-sweeps-download-receipt', 'V2 requires the full-recording support receipt')
    scoring_hash = ((provenance.get('originalSources') or {}).get('lidar') or {}).get('sha256')
    partition = partition_tiers(records, base['partition'], scoring_hash)
    output.mkdir(parents=True); write_json(output/'partition.json', partition)
    (output/'measured-ground-source.py').write_bytes(CODE_BYTES)
    (output/'tier-0-ground.json').write_bytes((tier0_dir/'grounded/ground.json').read_bytes())
    (output/'tier-0-calibration-errors.json').write_bytes((tier0_dir/'calibration-errors.json').read_bytes())
    data_dir = site/'data'/log; pose_columns = ['qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m']
    poses = {int(r['timestamp_ns']): r for r in table(pinned(data_dir/'city_SE3_egovehicle.feather', ground['sources']['poses']['sha256']), ['timestamp_ns']+pose_columns).to_pylist()}
    annotations = defaultdict(list)
    for row in table(pinned(data_dir/'annotations.feather', ground['sources']['annotations']['sha256']), ['timestamp_ns', 'length_m', 'width_m', 'height_m']+pose_columns).to_pylist():
        annotations[int(row['timestamp_ns'])].append(row)
    raster = np.frombuffer(pinned(site/'grounded/terrain.bin', ground['terrain']['sha256']), '<f4').reshape(ground['terrain']['rows'], ground['terrain']['columns'])
    fit_points, fit_ids, counts, diagnostics = [], [], [], []
    calibration = [[], []]; origin = np.asarray(ground['frame']['cityOrigin'])
    require(int(frame) in poses, 'The frame pose is required')
    frame_pose = np.array([poses[int(frame)][k] for k in ('tx_m', 'ty_m', 'tz_m')])
    receipt_rows = {r['timestampNs']: r for r in receipt['sweeps']}
    for tier, groups in enumerate(partition['tiers']):
        for group in ('fit', 'calibration'):
            for row in groups[group]:
                ts = int(row['timestampNs']); require_support_timestamp(ts, frame)
                require(row['sha256'] != scoring_hash, 'The scoring frame sweep bytes are refused as support/calibration')
                require(ts in poses and ts in annotations, 'Support sweep needs exact timestamp poses and annotations')
                raw = pinned(receipt_path.parent/'sweeps'/f'{ts}.feather', row['sha256'])
                require(len(raw) == row['bytes'] and md5(raw).hexdigest() == receipt_rows[str(ts)]['md5'], 'Support sweep byte count or MD5/ETag mismatch')
                sweep = table(raw, ['x', 'y', 'z']); ego = np.column_stack([np.asarray(sweep[k]) for k in ('x', 'y', 'z')])
                valid = np.isfinite(ego).all(1); ego = ego[valid]; outside = outside_cuboids(ego, annotations[ts]); ego = ego[outside]
                pose = poses[ts]; city = ego@rotation(pose).T+np.array([pose[k] for k in ('tx_m', 'ty_m', 'tz_m')])
                points = (city-origin)[:, [0, 2, 1]]*np.array([1, 1, -1]); xz = points[:, [0, 2]]
                in_window = (np.abs(xz) <= params['windowHalfWidthMeters']).all(1); points = points[in_window]; xz = xz[in_window]
                h = raster_heights(ground, raster, xz)
                selected = np.isfinite(h) & (np.abs(points[:, 1]-h) <= params['groundGateMeters']) & inside_polygons(xz, ground['drivableAreas'], params['drivableMarginMeters'])
                selected_points = points[selected]
                counts.append({'timestampNs': str(ts), 'tier': tier, 'set': group, 'finiteReturns': int(valid.sum()),
                               'removedInsideAnnotations': int((~outside).sum()), 'groundCandidates': len(selected_points)})
                if tier == 1:
                    ego_distance = np.linalg.norm(np.array([pose[k] for k in ('tx_m', 'ty_m', 'tz_m')])-frame_pose)
                    diagnostics.append({'timestampNs': str(ts), 'sha256': row['sha256'], 'set': group,
                                        **sweep_consistency(selected_points, base, ego_distance)})
                if group == 'fit':
                    fit_ids.append(np.full(len(selected_points), len(fit_points), np.int32)); fit_points.append(selected_points)
                else:
                    keep = inside_polygons(selected_points[:, [0, 2]], ground['drivableAreas']) & camera_visible(selected_points, ground['cuboids']['items'])
                    calibration[tier].extend(selected_points[keep].tolist())
    require(calibration[0] == read_json(tier0_dir/'calibration-points.json'), 'Tier-0 calibration points changed')
    all_points = np.concatenate(fit_points) if fit_points else np.empty((0, 3)); all_ids = np.concatenate(fit_ids) if fit_ids else np.empty(0, np.int32)
    surface = fit_surface(all_points, all_ids, ground['drivableAreas'], params, tier0=base)
    surface.update(partition=partition, parameters={**params, 'maximumSupportSeconds': None}, fitReturns=len(all_points),
                   sourceGroundSha256=digest((site/'grounded/ground.json').read_bytes()), sourceSuggestionsSha256=digest((site/'grounded/suggestions.json').read_bytes()),
                   receiptSha256=digest(receipt_path.read_bytes()), receiptPath=os.path.relpath(receipt_path, output), processingSha256=CODE_SHA256,
                   sourceSite=os.path.relpath(site, output), tier0={'path': os.path.relpath(tier0_dir, output), 'groundSha256': digest((tier0_dir/'grounded/ground.json').read_bytes())})
    require(all(a is None or a == b for a, b in zip(base['heights'], surface['heights'])), 'Tier-0 height changed')
    write_json(output/'support-counts.json', counts); write_json(output/'sweep-consistency.json', diagnostics)
    write_json(output/'calibration-points.json', calibration)
    shutil.copytree(site/'grounded', output/'grounded'); shutil.copytree(site/'assembled', output/'assembled')
    scene['processing']['groundPolicy'] = 'mappedGround with measured-ground-v2: measured triangles from other sweeps of the same recording; scoring sweep excluded; missing support or calibration is Unknown'
    write_json(output/'grounded/scene.json', scene)
    ground.update(schemaVersion=4, measuredGround=surface)
    ground['limitations'] = [
        'Runtime ground uses other sweeps of the same recording, with the scoring sweep excluded; unmeasured regions stay holes.' if s.startswith('Mapped ground is publisher terrain') else
        'Displayed uncertainty uses held-back integer-click errors separately for near-time and other-time ground, without clipping; scoring returns are excluded.' if 'along-ray uncertainty' in s else s
        for s in ground['limitations']]
    ground['limitations'].append('Sweeps from the same recording remain correlated; physical calibration and terrain contact remain provisional.')
    ground['license']['changes'] = 'Mapped geometry and annotations converted to umbra scene coordinates; other same-recording LiDAR sweeps registered into measured triangles. Scoring sweep excluded; holes retained. No endorsement implied.'
    ground['support'] = {'rows': [], 'count': 0, 'rule': 'other sweeps of the same recording only; scoring-sweep support rows are not used'}
    ground['alongRayUncertainty'] = {'method': 'supportSweepHoldout', 'heightBandMeters': None, 'rule': 'held-back same-recording integer-click horizontal error by hit range and triangle tier',
                                   'quantile': .975, 'minimumCount': 100, 'bins': [], 'tiers': []}
    for source in ground['sources'].values():
        if source is not None:
            source['path'] = os.path.relpath((site/'grounded'/source['path']).resolve(), output/'grounded')
    ground['processingCodeHashes']['scripts/california/measured_ground.py'] = CODE_SHA256
    protocol = {'schema': 'measured-ground-v2-development', 'variant': 'V0', 'parameters': surface['parameters'], 'excludeWithinNs': EXCLUDE_NS,
                'tier0': 'exact v1 partition, heights and calibration bins', 'tier1': 'every other receipted sweep; sorted timestamps; every fourth held back; fill v1 holes only',
                'calibration': {'rangeEdgesM': [0, 10, 12, 15, 18, 20, 22, 25, 30], 'quantile': .975, 'minimumCount': 100, 'pool': 'expand symmetrically to neighbouring bins', 'roundUpMeters': .001},
                'legacyPlacementProtocolSha256': ground['protocol']['sha256']}
    write_json(output/'grounded/measured-ground-protocol.json', protocol)
    raw_protocol = (output/'grounded/measured-ground-protocol.json').read_bytes()
    ground['protocol'] = {'path': 'measured-ground-protocol.json', 'bytes': len(raw_protocol), 'sha256': digest(raw_protocol), 'readOnlyExternal': True}
    write_json(output/'grounded/ground.json', ground)
    result = subprocess.run(['node', '--input-type=module', '-', str(output)], input=finalize_v2_js(), text=True, capture_output=True, cwd=ROOT)
    (output/'finalize-stdout.txt').write_text(result.stdout); (output/'finalize-stderr.txt').write_text(result.stderr)
    require(result.returncode == 0, f'JavaScript calibration/finalization failed: {result.stderr}')
    return {'site': scene['id'], 'variant': 'V0', 'fitSweepsByTier': [len(t['fit']) for t in partition['tiers']],
            'calibrationSweepsByTier': [len(t['calibration']) for t in partition['tiers']], 'verticesByTier': [surface['tiers'].count(t) for t in (0, 1, None)], 'output': str(output)}


FINALIZE_JS = r'''
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {mappedGroundHeight,mappedGroundHit,measuredSurfaceHit} from './src/geometry.mjs';
const dir=process.argv[2],folder=path.join(dir,'grounded'),read=name=>JSON.parse(fs.readFileSync(path.join(folder,name))),
  sha=b=>createHash('sha256').update(b).digest('hex'),write=(name,value)=>{const b=JSON.stringify(value)+'\n';fs.writeFileSync(path.join(folder,name),b);return {file:name,bytes:Buffer.byteLength(b),sha256:sha(b)};};
const scene=read('scene.json'),g=read('ground.json'),suggestions=read('suggestions.json'),entry=read('catalog-entry.json');
scene.ground=g;scene.groundSupportXZ=new Float64Array();
const c=scene.camera,M=scene.transform.worldFromCamera,project=p=>{const q=[0,1,2].map(k=>p[0]*M[0][k]+p[1]*M[1][k]+p[2]*M[2][k]);return q[2]>0?[c.fx*q[0]/q[2]+c.cx,c.fy*q[1]/q[2]+c.cy]:null;},
  inCrop=uv=>uv&&uv[0]>=0&&uv[0]<=c.width-1&&uv[1]>=0&&uv[1]<=c.height-1,
  horizontal=(a,b)=>Math.hypot(a[0]-b[0],a[2]-b[2]),edges=[0,10,12,15,18,20,22,25,30],errors=edges.slice(1).map(()=>[]),rangeErrors=[],unknown={};
let eligible=0,placed=0;
for(const p of JSON.parse(fs.readFileSync(path.join(dir,'calibration-points.json')))) {
  const uv=project(p);if(!inCrop(uv))continue;
  const pixel=uv.map(n=>Math.floor(n+.5));if(!inCrop(pixel))continue;eligible++;
  try {const hit=measuredSurfaceHit(scene,pixel),range=Math.hypot(hit.point[0],hit.point[2]),index=edges.findIndex((n,i)=>i<edges.length-1&&range>=n&&(range<edges[i+1]||i===edges.length-2&&range===edges[i+1]));
    if(index>=0){const error=horizontal(hit.point,p);errors[index].push(error);rangeErrors.push([range,error]);placed++;}
  }catch(e){unknown[e.message]=(unknown[e.message]??0)+1;}
}
const calibrationBins=[];
for(let i=0;i<errors.length;i++) {
  let lo=i,hi=i,pool=[...errors[i]];
  while(pool.length<g.alongRayUncertainty.minimumCount&&(lo>0||hi<errors.length-1)) {
    if(lo>0)pool.push(...errors[--lo]);if(hi<errors.length-1)pool.push(...errors[++hi]);
  }
  if(pool.length<g.alongRayUncertainty.minimumCount)continue;
  pool.sort((a,b)=>a-b);const meters=Math.ceil(pool[Math.ceil(.975*pool.length)-1]*1000)/1000;
  calibrationBins.push({minRangeM:edges[i],maxRangeM:edges[i+1],meters,count:pool.length,ownCount:errors[i].length,poolMinRangeM:edges[lo],poolMaxRangeM:edges[hi+1]});
}
g.alongRayUncertainty.bins=calibrationBins;g.alongRayUncertainty.coverage={eligible,placed,unknown};
if(g.measuredGround.parameters.bandRangeRule==='linear') {
  const knots=[],minimum=g.alongRayUncertainty.minimumCount;
  if(rangeErrors.length>=minimum)for(let rangeM=0;rangeM<=30;rangeM+=.5) {
    const distances=rangeErrors.map(([r])=>Math.abs(r-rangeM)).sort((a,b)=>a-b),radius=Math.max(1,distances[minimum-1]),
      sample=rangeErrors.filter(([r])=>Math.abs(r-rangeM)<=radius),pool=sample.map(([,e])=>e).sort((a,b)=>a-b),
      meters=Math.ceil(pool[Math.ceil(.975*pool.length)-1]*1000)/1000;
    knots.push({rangeM,meters,count:sample.length,poolMinRangeM:Math.min(...sample.map(([r])=>r)),poolMaxRangeM:Math.max(...sample.map(([r])=>r))});
  }
  Object.assign(g.alongRayUncertainty,{rangeRule:'linear',knots,bins:[],rule:'independent end-to-end integer-click errors: .5 m knots, +/-1 m placed-range pools expanded to at least100, linear interpolation without clipping'});
}
fs.writeFileSync(path.join(dir,'calibration-errors.json'),JSON.stringify({edges,errors,rangeErrors,eligible,placed,unknown})+'\n');
const unknownElement=(e,reason)=>({status:'unknown',supportClass:'unknown',reason,label:e.label,...(e.crossingError&&{crossingError:e.crossingError}),...(e.mappedPoint&&{mappedPoint:e.mappedPoint}),...(e.pixelUV&&{pixelUV:e.pixelUV,pixel:e.pixel})});
for(const role of ['crossing','vehicle','person']) {
  const e=suggestions[role];if(e.status==='unknown')continue;
  const y=mappedGroundHeight(scene,e.mappedPoint[0],e.mappedPoint[2]);
  if(y===null){suggestions[role]=unknownElement(e,'No independently measured ground support at the source anchor.');continue;}
  e.mappedPoint[1]=y;const uv=project(e.mappedPoint);
  if(!inCrop(uv)){suggestions[role]=unknownElement(e,'Measured source anchor projects outside the image.');continue;}
  e.pixelUV=uv;e.pixel=uv.map(n=>Math.floor(n+.5));
  try {
    const hit=mappedGroundHit(scene,uv);Object.assign(e,{groundPoint:hit.point,supportClass:hit.supportClass,supportLabel:hit.supportLabel,hidden:hit.hidden,
      hitKind:hit.hitKind,terrainCityZ:hit.terrainCityZ,terrainCell:hit.terrainCell,alongRayUncertaintyM:hit.alongRayUncertaintyM,rangeM:hit.rangeM,
      horizontalRangeM:Math.hypot(hit.point[0],hit.point[2]),pixelOffsetFromMappedM:horizontal(hit.point,e.mappedPoint),
      supportEvidence:{independentFitSweeps:g.measuredGround.partition.fit.length}});
    try{const integer=mappedGroundHit(scene,e.pixel);e.integerPixelCheck={groundPoint:integer.point,residualM:horizontal(integer.point,e.mappedPoint)};}
    catch(error){e.integerPixelCheck={reason:error.message};}
  }catch(error){suggestions[role]=unknownElement(e,error.message);}
}
if(suggestions.vehicle.status!=='annotated')g.carShape=null;
suggestions.tolerances.alongRayHeightBandMeters=null;
suggestions.license=g.license;
suggestions.protocolSha256=g.protocol.sha256;suggestions.inputs['protocol.json']=g.protocol.sha256;
g.measuredGround.suggestionAdapter={horizontalAnchors:'unchanged from source suggestions; heights, projections and ray diagnostics refreshed',sourceSha256:g.measuredGround.sourceSuggestionsSha256};
const gd=write('ground.json',g);delete scene.ground;delete scene.groundSupportXZ;
scene.measurements.ground=gd;const sd=write('scene.json',scene);
suggestions.inputs['scene.json']=sd.sha256;suggestions.inputs['ground.json']=gd.sha256;
const vd=write('suggestions.json',suggestions);entry.metadataSha256=sd.sha256;entry.ground={file:gd.file,sha256:gd.sha256};entry.suggestions={file:vd.file,sha256:vd.sha256};
write('catalog-entry.json',entry);
console.log(JSON.stringify({calibration:g.alongRayUncertainty,groundBytes:gd.bytes,suggestionStatus:Object.fromEntries(['crossing','vehicle','person'].map(k=>[k,suggestions[k].status]))}));
'''


def finalize_v2_js():
    # Share the suggestion/hash adapter while leaving the v1 finalizer byte-for-byte unchanged.
    adapter = FINALIZE_JS[FINALIZE_JS.index('const unknownElement='):]
    adapter = adapter.replace('No independently measured ground support at the source anchor.', 'No measured ground support at the source anchor.')
    adapter = adapter.replace('independentFitSweeps:', 'sameLogFitSweeps:')
    adapter = adapter.replace('e.mappedPoint[1]=y;const uv=project(e.mappedPoint);', """const tier=measuredGroundTier(scene,e.mappedPoint[0],e.mappedPoint[2]);
  if(measuredGroundBand(scene,Math.hypot(e.mappedPoint[0],e.mappedPoint[2]),tier)===null){suggestions[role]=unknownElement(e,`No click-error calibration for measured ground tier ${tier} at the source anchor range.`);continue;}
  e.mappedPoint[1]=y;const uv=project(e.mappedPoint);""")
    return FINALIZE_V2_JS + adapter


FINALIZE_V2_JS = r'''
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mappedGroundHeight,mappedGroundHit,measuredSurfaceHit,measuredGroundBand,measuredGroundTier} from './src/geometry.mjs';
const dir=process.argv[2],folder=path.join(dir,'grounded'),read=name=>JSON.parse(fs.readFileSync(path.join(folder,name))),
  sha=b=>createHash('sha256').update(b).digest('hex'),write=(name,value)=>{const b=JSON.stringify(value)+'\n';fs.writeFileSync(path.join(folder,name),b);return {file:name,bytes:Buffer.byteLength(b),sha256:sha(b)};};
const scene=read('scene.json'),g=read('ground.json'),suggestions=read('suggestions.json'),entry=read('catalog-entry.json'),
  old=JSON.parse(fs.readFileSync(path.join(dir,'tier-0-ground.json'))),points=JSON.parse(fs.readFileSync(path.join(dir,'calibration-points.json')));
scene.ground=g;scene.groundSupportXZ=new Float64Array();
const c=scene.camera,M=scene.transform.worldFromCamera,project=p=>{const q=[0,1,2].map(k=>p[0]*M[0][k]+p[1]*M[1][k]+p[2]*M[2][k]);return q[2]>0?[c.fx*q[0]/q[2]+c.cx,c.fy*q[1]/q[2]+c.cy]:null;},
  inCrop=uv=>uv&&uv[0]>=0&&uv[0]<=c.width-1&&uv[1]>=0&&uv[1]<=c.height-1,
  horizontal=(a,b)=>Math.hypot(a[0]-b[0],a[2]-b[2]),edges=[0,10,12,15,18,20,22,25,30];
function calibrate(sample,ground,wantedTier=null) {
  scene.ground=ground;
  const errors=edges.slice(1).map(()=>[]),rangeErrors=[],unknown={};let eligible=0,placed=0,otherTierHits=0;
  for(const p of sample) {
    const uv=project(p);if(!inCrop(uv))continue;
    const pixel=uv.map(n=>Math.floor(n+.5));if(!inCrop(pixel))continue;eligible++;
    try {
      const hit=measuredSurfaceHit(scene,pixel);
      if(wantedTier!==null&&hit.tier!==wantedTier){eligible--;otherTierHits++;continue;}
      const range=Math.hypot(hit.point[0],hit.point[2]),index=edges.findIndex((n,i)=>i<edges.length-1&&range>=n&&(range<edges[i+1]||i===edges.length-2&&range===edges[i+1]));
      if(index>=0){const error=horizontal(hit.point,p);errors[index].push(error);rangeErrors.push([range,error]);placed++;}
    }catch(e){unknown[e.message]=(unknown[e.message]??0)+1;}
  }
  const result={edges,errors,rangeErrors,eligible,placed,unknown};
  return wantedTier===null?result:{...result,otherTierHits};
}
function bins(errors) {
  const result=[];
  for(let i=0;i<errors.length;i++) {
    let lo=i,hi=i,pool=[...errors[i]];
    while(pool.length<100&&(lo>0||hi<errors.length-1)) {
      if(lo>0)pool.push(...errors[--lo]);if(hi<errors.length-1)pool.push(...errors[++hi]);
    }
    if(pool.length<100)continue;
    pool.sort((a,b)=>a-b);const meters=Math.ceil(pool[Math.ceil(.975*pool.length)-1]*1000)/1000;
    result.push({minRangeM:edges[i],maxRangeM:edges[i+1],meters,count:pool.length,ownCount:errors[i].length,poolMinRangeM:edges[lo],poolMaxRangeM:edges[hi+1]});
  }
  return result;
}
const tier0=calibrate(points[0],old),tier0Bins=bins(tier0.errors);
assert.deepEqual(tier0,JSON.parse(fs.readFileSync(path.join(dir,'tier-0-calibration-errors.json'))),'Tier-0 calibration errors must equal v1');
assert.deepEqual(tier0Bins,old.alongRayUncertainty.bins,'Tier-0 calibration bins must equal v1');
const tier1=calibrate([...points[0],...points[1]],g,1),coverage=r=>({eligible:r.eligible,placed:r.placed,unknown:r.unknown});
g.alongRayUncertainty.tiers=[{bins:tier0Bins,coverage:coverage(tier0)},{bins:bins(tier1.errors),coverage:coverage(tier1)}];
scene.ground=g;
fs.writeFileSync(path.join(dir,'calibration-errors.json'),JSON.stringify({tier0,tier1})+'\n');
fs.writeFileSync(path.join(dir,'tier-0-parity.json'),JSON.stringify({heights:true,calibrationPoints:true,calibrationErrors:true,calibrationBins:true})+'\n');
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cell-meters', type=float, default=PARAMETERS['cellMeters'])
    parser.add_argument('--fit-radius-meters', type=float, default=PARAMETERS['fitRadiusMeters'])
    parser.add_argument('--support-receipt', type=Path)
    parser.add_argument('--band-range-rule', choices=('bins', 'linear'), default=PARAMETERS['bandRangeRule'])
    parser.add_argument('--version', choices=('measured-ground-v1', 'measured-ground-v2'), default='measured-ground-v1')
    parser.add_argument('--tier-0', type=Path, help='Exact v1 artifact to preserve when building v2')
    args = parser.parse_args(); params = {**PARAMETERS, 'cellMeters': args.cell_meters, 'fitRadiusMeters': args.fit_radius_meters, 'bandRangeRule': args.band_range_rule}
    require(.1 <= args.cell_meters <= 1 and .5 <= args.fit_radius_meters <= 2, 'Unsupported grid/radius')
    if args.version == 'measured-ground-v2':
        require(args.tier_0 is not None and args.support_receipt is not None and params == PARAMETERS, 'V2 requires the v1 artifact, full receipt and unchanged v1 fit parameters')
        result = build_v2(args.site, args.out, args.tier_0, args.support_receipt)
    else:
        require(args.tier_0 is None, '--tier-0 is only for v2')
        result = build(args.site, args.out, params, args.support_receipt)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

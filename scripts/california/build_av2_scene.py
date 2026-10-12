"""Rule-based AV2 scene builder (Palo Alto accuracy plan, M5b): no human choices.

For one frozen split site (AV2 log + mapped crossing from split.json) it picks, by fixed rules only:
  * the current vehicle   - stationary annotated vehicle on the approach (right) side, by the frozen split
                            selection rule (selection-evidence stationary.py + relgeom.py), then the
                            placement protocol's current-vehicle checks (suggest_av2_placements.annotated_vehicle);
  * the frame             - the frozen frames.py window (near edge 8-25 m ahead, bearing < 31 deg, nearest LiDAR
                            sweep within 50 ms), ordered by |near edge - 12 m|; the first frame whose lane,
                            vehicle and suggestion anchor points all fit a source-supported crop;
  * the ego lane          - the mapped VEHICLE lane segment that contains the left camera;
  * the crop              - 736x320, centred on the projected anchor points, clamped to the common stereo support
                            exactly as prepare_av2.py checks it;
and writes the exact plan file prepare_av2.py consumes (only when every source file is on disk), plus a
selection record with the scene values a placement protocol needs (scene block, ego lane, vehicle track).

Geometry is the pipeline's own: drivable outlines, lane chord, near edge, curb scan, annotated-vehicle checks
and the declared defaults come from prepare_av2_ground.py / suggest_av2_placements.py, and the rectification
and crop-support rule from prepare_av2.py. Only metadata (poses, calibration, map, ground raster, annotations,
public S3 listings) is needed to choose; images and LiDAR are only hashed into the plan. Nothing is downloaded.
No invented geometry: an infeasible site is reported as such, never filled in.

Freeze guards (builder version 3): the frozen builder protocol is read by a pinned sha256 (--protocol-sha256, a literal
in the run recipe); it must carry exactly PARAMETERS, the sha256 of every runtime code file (RUNTIME_CODE, checked on
disk), the split sha256 and the placement-protocol sha256. The reproduced split selection rule must agree with the
per-site evidence frozen in split.json. A held-out, control or reserve log runs only with --held-out-run AND a valid
protocol-lock.json beside split.json (the M6 lock: it must name the split sha256, this builder protocol sha256 and the
same code hashes) whose own sha256 is given as --protocol-lock-sha256; everything is checked before any of that log's
files is opened.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
from uuid import UUID
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(SCRIPTS / 'sponsors')); sys.path.insert(0, str(HERE))
from common import canonical, digest, parse, publish, require, snapshot  # noqa: E402
import prepare_av2  # noqa: E402
from remap_support import supported_rectangle  # noqa: E402
_spec = importlib.util.spec_from_file_location('suggest_av2_placements', HERE / 'suggest_av2_placements.py')
suggest_tool = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(suggest_tool)
ground_tool = suggest_tool.ground_tool  # the prepare_av2_ground module instance the suggestion script itself uses

# v2: ego-lane chord must lie within 20 deg of the ego heading (v1 let a turn lane win).
# v3: freeze guards (frozen code hashes, pinned split and placement protocol, split-evidence cross-check, lock-gated
#     held-out mode) and --sensor-dir; no selection rule changed.
BUILDER_VERSION = 3
MAX_FILE = 16_000_000
S3 = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
SIDES = ('left', 'right')
SENSOR_ROLES = ('left', 'right', 'lidar')
LOCK_NAME = 'protocol-lock.json'  # the M6 lock, beside split.json (.local/evaluation/pao-accuracy/protocol-lock.json)
# Every code file that runs when the builder runs (repository-relative). The frozen protocol must name each one's sha256.
RUNTIME_CODE = ('scripts/california/build_av2_scene.py', 'scripts/california/prepare_av2.py', 'scripts/california/prepare_av2_ground.py',
                'scripts/california/suggest_av2_placements.py', 'scripts/remap_support.py', 'scripts/sponsors/common.py',
                'scripts/prepare-stereo-crop.py')
# Every number the rules use. The frozen builder protocol must carry exactly this object (checked at run time),
# so the code and the frozen protocol cannot drift apart.
PARAMETERS = {
    'site': {'allowedRolesWithoutHeldOutFlag': ['development', 'development-optional']},
    'frame': {'nearEdgeTargetM': 12.0, 'windowMinM': 8.0, 'windowMaxM': 25.0, 'maxBearingDeg': 31.0,
              'nearRoundingDecimals': 2, 'lidarMaxOffsetNs': 50_000_000},
    'rectification': {'divisor': 2, 'alpha': 0},
    'crop': {'width': 736, 'height': 320, 'supportMarginPx': 1, 'anchorMarginPx': 3},
    'lane': {'maxHeadingDeg': 20.0},
    'vehicle': {'splitCategories': ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB',
                                    'VEHICULAR_TRAILER', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER',
                                    'RAILED_VEHICLE'],
                'clusterRadiusM': 0.6, 'clusterMinSweeps': 30, 'nearCrossingM': 12.0,
                'viewMinForwardM': 5.0, 'viewMaxForwardM': 40.0, 'viewHalfAngleDeg': 31.0, 'referenceCrossingForwardM': 20.0,
                'minForwardM': 2.0, 'betweenMarginM': 2.0, 'rightLateralM': -1.5, 'trackPrefixMinChars': 8},
    # split.json rounds cluster centres to 0.1 m (selection-evidence stationary.py), so a reproduced centre may differ
    # from its frozen record by up to half of that on each axis.
    'splitCheck': {'cityXYToleranceM': 0.05},
}


def _sha(path):
    return digest(Path(path).read_bytes())


def runtime_code_hashes():
    """sha256 of every runtime code file, by repository-relative path (what the frozen protocol must name)."""
    return {name: _sha(REPO / name) for name in RUNTIME_CODE}


# ------------------------------------------------------------------ freeze guards
def check_frozen_protocol(protocol, split_sha, placement_sha):
    """The frozen builder protocol (already read by its pinned sha256) must match this code exactly: parameters,
    builder version, every runtime code hash on disk, the split it was frozen against and the placement protocol."""
    require(protocol.get('parameters') == PARAMETERS and protocol.get('builderVersion') == BUILDER_VERSION,
            'Builder protocol parameters differ from this builder')
    code = protocol.get('code')
    require(isinstance(code, dict) and set(code) == set(RUNTIME_CODE), 'Builder protocol must name the sha256 of every runtime code file')
    changed = sorted(name for name, value in runtime_code_hashes().items() if code[name] != value)
    require(not changed, 'Frozen code hash mismatch (a code change needs a new builder protocol version): ' + ', '.join(changed))
    require((protocol.get('split') or {}).get('sha256') == split_sha, 'Builder protocol was frozen against another split.json')
    require((protocol.get('placementProtocol') or {}).get('sha256') == placement_sha,
            'Placement protocol sha256 differs from the one pinned in the builder protocol')


def check_protocol_lock(split_path, lock_sha256, split_sha, protocol_sha, code):
    """M7 only. protocol-lock.json beside split.json, read by its pinned sha256, must name this split, this builder
    protocol and the same runtime code hashes. Called before any held-out, control or reserve log file is opened."""
    path = Path(split_path).resolve().parent / LOCK_NAME
    require(isinstance(lock_sha256, str) and re.fullmatch('[0-9a-f]{64}', lock_sha256),
            f'Held-out runs need --protocol-lock-sha256 (the sha256 of {LOCK_NAME} recorded at the M6 lock)')
    require(path.is_file(), f'Held-out runs need {LOCK_NAME} beside split.json (written at the M6 lock); it does not exist')
    lock = parse(snapshot(path, lock_sha256))
    builder = lock.get('sceneBuilder') if isinstance(lock.get('sceneBuilder'), dict) else {}
    require((lock.get('split') or {}).get('sha256') == split_sha, f'{LOCK_NAME} does not name this split.json sha256')
    require(builder.get('protocolSha256') == protocol_sha, f'{LOCK_NAME} does not name this builder protocol sha256')
    require(builder.get('code') == code, f'{LOCK_NAME} does not name the same runtime code hashes')
    return {'path': str(path), 'sha256': lock_sha256}


def split_cross_check(site, rows, params):
    """The reproduced selection rule must agree with the per-site evidence frozen in split.json: the clusters that match
    the rule pair one-to-one (same category and sweep count, centre within the evidence rounding), and expectedBlocked
    agrees with whether any cluster matches. Returns a record with ok True/False."""
    evidence = site.get('stationaryAnnotatedVehiclesInViewWithin12m'); expected = site.get('expectedBlocked')
    mine = [r for r in rows if r['matchesSelectionRule']]
    out = {'evidenceRows': None if evidence is None else len(evidence), 'expectedBlocked': expected,
           'reproducedMatching': len(mine), 'pairs': [], 'unpairedReproduced': [], 'unpairedEvidence': [], 'ok': False}
    if evidence is None:
        out['reason'] = 'split.json has no per-site vehicle evidence for this site'
        return out
    theirs = [r for r in evidence if r.get('matchesSelectionRule') is True]; free = list(range(len(theirs)))
    tol = params['cityXYToleranceM'] + 1e-9
    for r in mine:
        hit = next((j for j in free if theirs[j].get('cat') == r['category'] and theirs[j].get('nSweeps') == r['nSweeps'] and
                    all(abs(a - b) <= tol for a, b in zip(r['cityXY'], theirs[j].get('cityXY', [math.inf, math.inf])))), None)
        if hit is None:
            out['unpairedReproduced'].append(r['cityXY'])
        else:
            free.remove(hit); out['pairs'].append({'reproducedCityXY': r['cityXY'], 'evidenceCityXY': theirs[hit]['cityXY'], 'tracks': r['tracks']})
    out['unpairedEvidence'] = [theirs[j].get('cityXY') for j in free]
    reasons = []
    if out['unpairedReproduced'] or out['unpairedEvidence']:
        reasons.append('the clusters matching the selection rule differ from split.json')
    if expected is not None and bool(expected) != bool(mine):
        reasons.append(f"split.json expectedBlocked is {expected} but {len(mine)} reproduced cluster(s) match the selection rule")
    out['ok'] = not reasons
    out['reason'] = '; '.join(reasons) or None
    return out


# ------------------------------------------------------------------ inputs
def load_listing(data):
    """Public S3 ListBucketResult -> {timestamp: {key, bytes, etag}}; complete (not truncated) listings only."""
    doc = ET.fromstring(data)
    require(doc.tag == '{%s}ListBucketResult' % S3['s3'], 'S3 ListBucketResult required')
    require(doc.find('s3:IsTruncated', S3).text == 'false', 'Truncated listing')
    result = {}
    for node in doc.findall('s3:Contents', S3):
        key = node.find('s3:Key', S3).text
        match = re.fullmatch(r'.*/([0-9]+)\.(jpg|feather)', key)
        require(match is not None, 'Unexpected listing key')
        ts = int(match.group(1))
        require(ts not in result, 'Duplicate listing timestamp')
        result[ts] = {'key': key, 'bytes': int(node.find('s3:Size', S3).text), 'etag': node.find('s3:ETag', S3).text.strip('"')}
    require(result and len(result) == int(doc.find('s3:KeyCount', S3).text), 'Listing count mismatch')
    return result


def load_poses(data):
    """Pose rows in file order (strictly increasing timestamps) plus rigid matrices by timestamp."""
    rows = prepare_av2.table(data, ['timestamp_ns', 'qx', 'qy', 'qz', 'qw', 'tx_m', 'ty_m', 'tz_m']).to_pylist()
    ts = np.array([r['timestamp_ns'] for r in rows], np.int64)
    require(len(ts) > 1 and bool(np.all(np.diff(ts) > 0)), 'Pose timestamps must be strictly increasing')
    rigid = {int(r['timestamp_ns']): prepare_av2.rigid(r) for r in rows}
    q = {k: np.array([r[k] for r in rows], float) for k in ('qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m')}
    # Same yaw expression as the frozen selection scripts (analyze.py, frames.py, relgeom.py).
    yaw = np.arctan2(2 * (q['qw'] * q['qz'] + q['qx'] * q['qy']), 1 - 2 * (q['qy'] ** 2 + q['qz'] ** 2))
    return {'ts': ts, 'xy': np.column_stack([q['tx_m'], q['ty_m']]), 'yaw': yaw, 'quat': q, 'rigid': rigid,
            'index': {int(t): i for i, t in enumerate(ts)}}


def load_annotations(data):
    table = prepare_av2.table(data, ['timestamp_ns', 'track_uuid', 'category', 'tx_m', 'ty_m', 'tz_m', 'length_m'])
    return {k: np.asarray(table.column(k).to_numpy(zero_copy_only=False)) for k in
            ('timestamp_ns', 'track_uuid', 'category', 'tx_m', 'ty_m', 'length_m')}


def crossing_geometry(vector_map, crossing_id):
    require(crossing_id in vector_map['pedestrian_crossings'], 'Crossing not in the log map')
    c = vector_map['pedestrian_crossings'][crossing_id]
    require(str(c['id']) == crossing_id, 'Crossing identity mismatch')
    edges = {name: ground_tool._xyz(c[name]) for name in ('edge1', 'edge2')}
    for e in edges.values():
        require(len(e) >= 2 and np.isfinite(e).all(), 'Finite crossing edges required')
    return edges


# ------------------------------------------------------------------ frozen split rule (selection-evidence scripts)
def crossing_view(poses, edges, params):
    """analyze.py: poses where the crossing centroid is 5-40 m ahead and within 31 deg; first/last timestamps."""
    centroid = np.nanmean(np.vstack([edges['edge1'], edges['edge2']])[:, :2], 0)
    rel = centroid - poses['xy']; yw = poses['yaw']
    f = rel[:, 0] * np.cos(yw) + rel[:, 1] * np.sin(yw); lat = -rel[:, 0] * np.sin(yw) + rel[:, 1] * np.cos(yw)
    vis = (f > params['viewMinForwardM']) & (f < params['viewMaxForwardM']) & \
        (np.abs(np.degrees(np.arctan2(lat, f))) < params['viewHalfAngleDeg'])
    if not vis.any():
        return None
    idx = np.flatnonzero(vis)
    return {'centroid': centroid, 't0': int(poses['ts'][idx[0]]), 't1': int(poses['ts'][idx[-1]]),
            'minForwardM': float(f[idx].min()), 'maxForwardM': float(f[idx].max())}


def stationary_clusters(ann, poses, params):
    """stationary.py: greedy grouping in time order of vehicle-category annotation centres within 0.6 m of a seed,
    kept when they cover at least 30 distinct sweeps. Returns clusters with their member row indices."""
    keep = np.isin(ann['category'], params['splitCategories'])
    rows = np.flatnonzero(keep); ts = ann['timestamp_ns'][rows]
    require(all(int(t) in poses['index'] for t in np.unique(ts)), 'Exact ego pose missing for an annotation timestamp')
    k = np.array([poses['index'][int(t)] for t in ts], np.int64); q = poses['quat']
    qw, qx, qy, qz = (q[c][k] for c in ('qw', 'qx', 'qy', 'qz'))
    r00 = 1 - 2 * (qy * qy + qz * qz); r01 = 2 * (qx * qy - qz * qw); r10 = 2 * (qx * qy + qz * qw); r11 = 1 - 2 * (qx * qx + qz * qz)
    tx, ty = ann['tx_m'][rows].astype(float), ann['ty_m'][rows].astype(float)
    pts = np.column_stack([r00 * tx + r01 * ty + q['tx_m'][k], r10 * tx + r11 * ty + q['ty_m'][k]])
    used = np.zeros(len(pts), bool); clusters = []
    for i in np.argsort(ts):
        if used[i]:
            continue
        m = (np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]) < params['clusterRadiusM']) & ~used
        uts = np.unique(ts[m])
        if len(uts) >= params['clusterMinSweeps']:
            used |= m; centre = pts[m].mean(0)
            clusters.append({'centre': centre, 'nSweeps': int(len(uts)), 't0': int(uts[0]), 't1': int(uts[-1]),
                             'spanS': (int(uts[-1]) - int(uts[0])) / 1e9, 'category': str(ann['category'][rows][m][0]),
                             'lengthM': float(np.median(ann['length_m'][rows][m])),
                             'spreadM': float(np.max(np.hypot(*(pts[m] - centre).T))),
                             'rows': rows[m], 'tracks': sorted({str(t) for t in ann['track_uuid'][rows][m]})})
    return clusters


def _segment_distance(p, a, b):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_distance(p, polygon):
    return min(_segment_distance(p, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def site_vehicles(poses, edges, clusters, params):
    """relgeom.py at the reference pose (crossing centroid closest to 20 m ahead inside the view window):
    a cluster near the crossing matches when it is ahead (> 2 m, within 31 deg), between the ego and the crossing
    and right of the ego (lateral < -1.5 m)."""
    view = crossing_view(poses, edges, params)
    if view is None:
        return None, [], 'Crossing never inside the stereo view window (5-40 m, 31 deg).'
    polygon = np.vstack([edges['edge1'][:, :2], edges['edge2'][::-1, :2]])
    ts, xy, yw = poses['ts'], poses['xy'], poses['yaw']
    rel = view['centroid'] - xy; fc = rel[:, 0] * np.cos(yw) + rel[:, 1] * np.sin(yw)
    idx = np.flatnonzero((ts >= view['t0']) & (ts <= view['t1'])); k = int(idx[np.argmin(np.abs(fc[idx] - params['referenceCrossingForwardM']))])
    reference = {'t': str(int(ts[k])), 'crossingFwdM': float(fc[k])}
    rows = []
    for cl in clusters:
        d = polygon_distance(cl['centre'], polygon)
        if not (d <= params['nearCrossingM'] and cl['t1'] >= view['t0'] and cl['t0'] <= view['t1']):
            continue
        p = cl['centre'] - xy[k]; f = p[0] * math.cos(yw[k]) + p[1] * math.sin(yw[k]); lat = -p[0] * math.sin(yw[k]) + p[1] * math.cos(yw[k])
        in_view = f > params['minForwardM'] and abs(math.degrees(math.atan2(lat, f))) < params['viewHalfAngleDeg']
        between = 0 < f < fc[k] + params['betweenMarginM']
        side = 'right' if lat < params['rightLateralM'] else ('left' if lat > -params['rightLateralM'] else 'ego-lane')
        rows.append({'cityXY': [float(cl['centre'][0]), float(cl['centre'][1])], 'category': cl['category'], 'lengthM': cl['lengthM'],
                     'stationaryS': cl['spanS'], 'nSweeps': cl['nSweeps'], 'posSpreadM': cl['spreadM'], 'distToCrossingM': d,
                     'tracks': cl['tracks'],
                     'atReferencePose': {'vehFwdM': f, 'vehLatM': lat, 'inStereoFov': bool(in_view),
                                         'betweenEgoAndCrossing': bool(between), 'side': side},
                     'matchesSelectionRule': bool(in_view and between and side == 'right')})
    rows.sort(key=lambda r: r['distToCrossingM'])
    view_out = {k2: (str(v) if k2 in ('t0', 't1') else (v.tolist() if isinstance(v, np.ndarray) else v)) for k2, v in view.items()}
    return {'view': view_out, 'referencePose': reference}, rows, None


# ------------------------------------------------------------------ frozen frame window (frames.py)
def frame_window(poses, edges, left, right, lidar, params):
    """frames.py: camera frames whose nearer crossing edge (edge-vertex mean, ego-origin forward) is 8-25 m ahead
    within 31 deg, with the nearest LiDAR sweep; ordered by |round(near, 2) - 12| then timestamp. Frames without an
    exact ego pose, without a right image or without a sweep within 50 ms are excluded (counted)."""
    means = [edges[n][:, :2].mean(0) for n in ('edge1', 'edge2')]
    lid = np.array(sorted(lidar), np.int64); window = []; excluded = {'noExactPose': 0, 'noRightImage': 0, 'lidarTooFar': 0}
    for t in sorted(left):
        if t not in poses['index']:
            excluded['noExactPose'] += 1; continue
        i = poses['index'][t]; xy, yw = poses['xy'][i], poses['yaw'][i]; fs = []; bs = []
        for mean in means:
            m = mean - xy; f = m[0] * math.cos(yw) + m[1] * math.sin(yw); lat = -m[0] * math.sin(yw) + m[1] * math.cos(yw)
            fs.append(f); bs.append(abs(math.degrees(math.atan2(lat, f))))
        if not min(fs) > 0:
            continue
        near = min(fs); j = int(np.argmin(fs))
        if not (params['windowMinM'] <= near <= params['windowMaxM'] and bs[j] < params['maxBearingDeg']):
            continue
        if t not in right:
            excluded['noRightImage'] += 1; continue
        sweep = int(lid[np.argmin(np.abs(lid - t))])
        if abs(sweep - t) > params['lidarMaxOffsetNs']:
            excluded['lidarTooFar'] += 1; continue
        rounded = float(round(np.float64(near), params['nearRoundingDecimals']))
        window.append({'cameraTimestampNs': t, 'lidarTimestampNs': sweep, 'lidarMinusCameraMs': (sweep - t) / 1e6,
                       'nearEdgeM': rounded, 'nearEdgeName': ('edge1', 'edge2')[j], 'nearEdgeBearingDeg': bs[j]})
    window.sort(key=lambda w: (abs(w['nearEdgeM'] - params['nearEdgeTargetM']), w['cameraTimestampNs']))
    return window, excluded


# ------------------------------------------------------------------ calibration and rectification (prepare_av2.py)
def rectification(intrinsics_data, extrinsics_data, divisor):
    """The exact prepare_av2.prepare() rectification and common-support rectangle for the rectified grid."""
    cv2 = prepare_av2.cv2; cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    tables = {}
    for name, data, columns in (('intrinsics', intrinsics_data, ['sensor_name', 'fx_px', 'fy_px', 'cx_px', 'cy_px', 'width_px', 'height_px']),
                                ('extrinsics', extrinsics_data, ['sensor_name', 'qx', 'qy', 'qz', 'qw', 'tx_m', 'ty_m', 'tz_m'])):
        rows = prepare_av2.table(data, columns).to_pylist(); tables[name] = {r['sensor_name']: r for r in rows}
        require(len(tables[name]) == len(rows) and all('stereo_front_' + s in tables[name] for s in SIDES), 'Stereo sensors missing')
    ks, extr, sizes = [], [], []
    for side in SIDES:
        row = tables['intrinsics']['stereo_front_' + side]
        ks.append(np.array([[row['fx_px'], 0, row['cx_px']], [0, row['fy_px'], row['cy_px']], [0, 0, 1.]], float))
        extr.append(prepare_av2.rigid(tables['extrinsics']['stereo_front_' + side])); sizes.append([int(row['width_px']), int(row['height_px'])])
    require(sizes[0] == sizes[1] and all(n % divisor == 0 for n in sizes[0]), 'Stereo sizes must match and divide evenly')
    rect_size = (sizes[0][0] // divisor, sizes[0][1] // divisor)
    relative = prepare_av2.inverse(extr[1]) @ extr[0]; zero_d = np.zeros(5)
    r1, r2, p1, p2, _q, _a, _b = cv2.stereoRectify(ks[0], zero_d, ks[1], zero_d, tuple(sizes[0]), relative[:3, :3], relative[:3, 3],
                                                     flags=cv2.CALIB_ZERO_DISPARITY, alpha=PARAMETERS['rectification']['alpha'], newImageSize=rect_size)
    masks = []
    for k, rotation, projection in zip(ks, (r1, r2), (p1, p2)):
        mx, my = cv2.initUndistortRectifyMap(k, zero_d, rotation, projection, rect_size, cv2.CV_32FC1)
        strict = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0) & (mx < sizes[0][0] - 1) & (my < sizes[0][1] - 1)
        ends = np.isfinite(mx) & np.isfinite(my) & (np.floor(mx) >= 0) & (np.floor(my) >= 0) & (np.ceil(mx) < sizes[0][0]) & (np.ceil(my) < sizes[0][1])
        masks.append(strict & ends)
    crop = PARAMETERS['crop']
    support = supported_rectangle(masks[0] & masks[1], (crop['width'], crop['height']), margin=crop['supportMarginPx'])
    return {'rectifiedSize': list(rect_size), 'R1': r1, 'P1': p1, 'egoFromLeft': extr[0], 'support': [int(v) for v in support]}


def project_city(points_city, city_from_camera, rect):
    """prepare_av2's mapped-corner projection: city -> left camera -> R1 -> P1 (rectified grid, before crop)."""
    xyz = prepare_av2.transform(np.asarray(points_city, float), prepare_av2.inverse(city_from_camera)) @ rect['R1'].T
    h = xyz @ rect['P1'][:, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = h[:, :2] / h[:, 2:3]
    return uv, xyz[:, 2]


def to_world(city, origin):
    """City -> scene world (X = cityX - x0, Y = cityZ - z0, Z = -(cityY - y0)), element-wise so a missing publisher
    height (NaN cityZ) never blanks the horizontal position (same values as prepare_av2_ground._world otherwise)."""
    c = np.asarray(city, float).reshape(-1, 3)
    return np.column_stack([c[:, 0] - origin[0], c[:, 2] - origin[2], -(c[:, 1] - origin[1])])


def to_city(world, origin):
    w = np.asarray(world, float).reshape(-1, 3)
    return np.column_stack([w[:, 0] + origin[0], -w[:, 2] + origin[1], w[:, 1] + origin[2]])


# ------------------------------------------------------------------ per-frame geometry (suggest_av2_placements rules)
class FrameGeometry:
    """The attributes suggest_av2_placements.Context reads for near_edge, curb_lateral_at, annotated_vehicle and
    default_vehicle, built for one frame (world origin = left camera at the camera timestamp)."""
    near_edge = suggest_tool.Context.near_edge
    curb_lateral_at = suggest_tool.Context.curb_lateral_at

    def __init__(self, lane_world, polygons, crossing_world, cuboids, placement_protocol, epsilon):
        left, right = np.asarray(lane_world['left'], float), np.asarray(lane_world['right'], float)
        self.m0 = (left[0, [0, 2]] + right[0, [0, 2]]) / 2
        self.a = suggest_tool.unit((left[-1, [0, 2]] + right[-1, [0, 2]]) / 2 - self.m0); self.r = np.array([-self.a[1], self.a[0]])
        self.polygons = polygons; self.cuboids = cuboids; self.protocol = placement_protocol
        self.ground = {'crossing': crossing_world, 'boundaryEpsilonMeters': epsilon}

    def with_track(self, track):
        clone = object.__new__(FrameGeometry); clone.__dict__.update(self.__dict__)
        clone.protocol = {**self.protocol, 'currentVehicle': {**self.protocol['currentVehicle'], 'trackPrefix': track}}
        return clone


def lane_polygon(lane_world):
    left, right = np.asarray(lane_world['left'], float), np.asarray(lane_world['right'], float)
    return np.vstack([left[:, [0, 2]], right[::-1][:, [0, 2]]])


def choose_lane(lanes_world, heading_xz, max_heading_deg):
    """Ego lane: VEHICLE lane segments whose outline contains the left camera (world origin) and whose chord lies
    within max_heading_deg of the ego heading; ordering: non-intersection first, smallest chord angle to the ego
    heading, smallest camera offset from the chord line, smallest id."""
    found = []
    for lane in lanes_world:
        if lane['laneType'] != 'VEHICLE' or not ground_tool.inside_polygons([0., 0.], [lane_polygon(lane)])[0]:
            continue
        left, right = np.asarray(lane['left'], float), np.asarray(lane['right'], float)
        m0 = (left[0, [0, 2]] + right[0, [0, 2]]) / 2; chord = (left[-1, [0, 2]] + right[-1, [0, 2]]) / 2 - m0
        n = float(np.linalg.norm(chord))
        if n < 1e-9:
            continue
        a = chord / n; cosine = float(a @ heading_xz)
        if cosine <= 0:
            continue
        angle = math.degrees(math.acos(min(1., cosine))); offset = abs(float(-m0 @ np.array([-a[1], a[0]])))
        if angle > max_heading_deg:
            continue
        found.append(((bool(lane['isIntersection']), angle, offset, int(lane['id'])), lane, angle, offset))
    found.sort(key=lambda item: item[0])
    return [{'id': item[1]['id'], 'isIntersection': item[1]['isIntersection'], 'angleToHeadingDeg': item[2], 'cameraOffsetM': item[3],
             'lane': item[1]} for item in found]


def front_face(box):
    m = np.asarray(box['worldFromBox'], float); c = np.asarray(box['center'], float)
    return [c + m @ np.array([box['length'] / 2, sw * box['width'] / 2, sh * box['height'] / 2]) for sw in (-1, 1) for sh in (-1, 1)]


def evaluate_frame(frame, ctx, site_tracks):
    """Lane, vehicle, anchors and crop for one window frame. Returns a record with feasible True/False and reasons."""
    params = PARAMETERS; out = {**frame, 'feasible': False, 'reasons': []}
    t, sweep = frame['cameraTimestampNs'], frame['lidarTimestampNs']
    poses, rect = ctx['poses'], ctx['rect']
    if sweep not in poses['rigid'] or t not in poses['rigid']:
        out['reasons'].append('Exact camera or sweep pose missing.'); return out
    city_from_camera = poses['rigid'][t] @ rect['egoFromLeft']; origin = city_from_camera[:3, 3]
    world = lambda pts: to_world(pts, origin)  # noqa: E731
    polygons = [world(a)[:, [0, 2]] for a in ctx['drivable']]
    crossing_world = {name: world(e).tolist() for name, e in ctx['edges'].items()}
    heading = poses['rigid'][t][:3, 0]; heading_xz = suggest_tool.unit([heading[0], -heading[1]])
    lanes = choose_lane([{**lane, 'left': world(lane['left']), 'right': world(lane['right'])} for lane in ctx['lanes']], heading_xz,
                        params['lane']['maxHeadingDeg'])
    out['laneCandidates'] = [{k: v for k, v in lane.items() if k != 'lane'} for lane in lanes]
    if not lanes:
        out['reasons'].append(f"Left camera is not inside a mapped VEHICLE lane whose chord is within {params['lane']['maxHeadingDeg']:g} deg "
                              'of the ego heading.'); return out
    lane = lanes[0]['lane']; out['egoLaneId'] = lane['id']
    try:  # every cuboid at the sweep; the current-vehicle candidate is averaged per track below (protocol v3 pose)
        cuboids = ground_tool.cuboids_from_annotations(ctx['annotationsData'], poses['rigid'], sweep, origin)
    except ValueError as error:
        out['reasons'].append(f'Annotations unusable at the sweep: {error}'); return out
    geo = FrameGeometry(lane, polygons, crossing_world, cuboids, ctx['placementProtocol'], ctx['epsilon'])
    try:
        edge_name, edge_forward, p0, eu = geo.near_edge()
    except ValueError as error:
        out['reasons'].append(f'Near edge undefined: {error}'); return out
    out['nearEdge'] = {'name': edge_name, 'forwardFromCameraM': float(edge_forward),
                       'skewDeg': 90 - math.degrees(math.acos(min(1., abs(float(eu @ geo.a)))))}
    start, _ = suggest_tool.intersect(geo.m0, geo.a, p0, eu)
    if start is None:
        out['reasons'].append('Near edge is parallel to the ego lane chord (no crossing point on the lane).'); return out
    s = ground_tool.curb_scan(start, eu, polygons, ctx['epsilon'])
    if s is None:  # same rule as v1; the message now names the geometric cause
        if not ground_tool.inside_polygons(start, polygons)[0]:
            out['reasons'].append(f"The near edge meets the ego lane chord outside the mapped drivable area ({float(np.linalg.norm(start)):.0f} m "
                                  f"from the camera; skew {out['nearEdge']['skewDeg']:.3f} deg): the crossing does not cross the ego lane ahead.")
        else:
            out['reasons'].append('No curb proxy on the near edge right of the ego lane.')
        return out
    curb_point = start + s * eu
    # Current vehicle: the site's selection-rule tracks that pass the placement protocol checks at this sweep.
    defaults = ctx['placementProtocol']['declaredDefaults']; vehicle = None; checks = {}
    present = sorted(track for track in site_tracks if any(c['track'] == track for c in cuboids))
    for track in present:
        try:
            averaged = ground_tool.cuboids_from_annotations(ctx['annotationsData'], poses['rigid'], sweep, origin, average_prefix=track)
        except ValueError as error:
            checks[track] = [f'Averaged pose unavailable: {error}']; continue
        candidate = geo.with_track(track); candidate.cuboids = averaged
        v, failures = suggest_tool.annotated_vehicle(candidate, p0, eu)
        checks[track] = failures
        if v is not None and (vehicle is None or (v['gap'], track) < (vehicle['gap'], vehicle['box']['track'])):
            vehicle = v; cuboids = averaged
    out['vehicleChecks'] = {'siteTracks': sorted(site_tracks), 'presentAtSweep': present, 'failures': checks}
    if site_tracks and vehicle is None:
        out['reasons'].append('No selection-rule vehicle passes the placement protocol checks at this sweep.'); return out
    if vehicle is not None:
        box = vehicle['box']; crossing_xz = vehicle['crossing']; front_xz = vehicle['front']; face = front_face(box)
        prefix = next(box['track'][:n] for n in range(params['vehicle']['trackPrefixMinChars'], len(box['track']) + 1)
                      if sum(c['track'].startswith(box['track'][:n]) for c in cuboids) == 1)
        out['currentVehicle'] = {'status': 'annotated', 'track': box['track'], 'trackPrefix': prefix, 'category': box['category'],
                                 'pose': 'averageAllAnnotations', 'poseAverageRows': box['poseAverage']['rows'],
                                 'sweepRowOffsetFromAverageM': box['poseAverage']['sweepRow']['horizontalM'],
                                 'frontGapToCrossingM': float(vehicle['gap']), 'yawToRoadAxisDeg': float(vehicle['yawToRoadAxisDeg']),
                                 'dimensions': {k: box[k] for k in ('length', 'width', 'height')}, 'motion': box['motion']}
    else:
        try:
            chosen, reason = suggest_tool.default_vehicle(geo, p0, eu, float(curb_point @ geo.r))
        except (TypeError, ValueError) as error:
            chosen, reason = None, f'Declared default vehicle undefined: {error}'
        if chosen is not None:
            dims = defaults['vehicle']['dimensions']; side = np.array([-geo.a[1], geo.a[0]])
            footprint = {'center': chosen['front'] - dims['length'] / 2 * geo.a, 'axes': [geo.a, side], 'half': [dims['length'] / 2, dims['width'] / 2]}
            for c in cuboids:
                h = suggest_tool.unit(np.asarray(c['worldFromBox'], float)[[0, 2], 0])
                other = {'center': np.asarray(c['center'], float)[[0, 2]], 'axes': [h, np.array([-h[1], h[0]])], 'half': [c['length'] / 2, c['width'] / 2]}
                if suggest_tool.rectangles_overlap(footprint, other):
                    chosen, reason = None, f"Annotated {c['category']} occupies the declared default footprint."
                    break
        if chosen is None:
            out['reasons'].append(reason or 'Declared default vehicle unavailable.'); return out
        crossing_xz = chosen['crossing']; front_xz = chosen['front']
        out['currentVehicle'] = {'status': 'declared-default', 'reason': 'No stationary vehicle matches the frozen split selection rule.'}
        face = None
    person_xz = curb_point - defaults['person']['curbOffsetMeters'] * eu
    anchors = {}
    for name, xz in (('crossing', crossing_xz), ('vehicleFront', front_xz), ('person', person_xz)):
        h = ctx['terrain'].sample([xz])[0][0]
        if not np.isfinite(h):
            out['reasons'].append(f'No mapped ground height at the {name} anchor.'); return out
        anchors[name] = np.array([xz[0], float(h) - origin[2], xz[1]])
        if math.hypot(xz[0], xz[1]) > ctx['rangeLimit']:
            out['reasons'].append(f"The {name} anchor is beyond the {ctx['rangeLimit']:g} m placement range."); return out
    extents = {'personTop': anchors['person'] + [0, defaults['person']['heightMeters'], 0]}
    if face is None:  # declared default: front face of the default box on the lane axis
        dims = defaults['vehicle']['dimensions']; f0 = anchors['vehicleFront']; side = np.array([geo.r[0], 0, geo.r[1]])
        face = [f0 + sw * dims['width'] / 2 * side + [0, sh * dims['height'], 0] for sw in (-1, 1) for sh in (0, 1)]
    for i, corner in enumerate(face):
        extents[f'vehicleFrontFace{i}'] = np.asarray(corner, float)
    names = list(anchors) + list(extents); points = np.vstack([anchors[n] for n in anchors] + [extents[n] for n in extents])
    uv, depth = project_city(to_city(points, origin), city_from_camera, rect)
    if not (np.isfinite(uv).all() and (depth > 0).all()):
        out['reasons'].append('An anchor or extent point is behind the camera.'); return out
    crop_p = params['crop']; sx, sy, sw_, sh_ = rect['support']; w, hgt = crop_p['width'], crop_p['height']
    centre = (uv.min(0) + uv.max(0)) / 2
    x0 = int(np.clip(math.floor(centre[0] - w / 2), sx, sx + sw_ - w)); y0 = int(np.clip(math.floor(centre[1] - hgt / 2), sy, sy + sh_ - hgt))
    local = uv - [x0, y0]; margin = crop_p['anchorMarginPx']
    inside = {}
    for i, n in enumerate(names):
        if n in anchors:  # the exact projection (protocol v3 pixelUV) with room for the integer anchor and a nudge
            inside[n] = bool(margin <= local[i, 0] <= w - 1 - margin and margin <= local[i, 1] <= hgt - 1 - margin)
        else:
            inside[n] = bool(0 <= local[i, 0] <= w - 1 and 0 <= local[i, 1] <= hgt - 1)
    out['crop'] = [x0, y0, w, hgt]
    out['anchors'] = {n: {'world': anchors[n].tolist(), 'rectifiedUV': uv[i].tolist(), 'cropUV': local[i].tolist(), 'inside': inside[n]}
                      for i, n in enumerate(names) if n in anchors}
    out['extents'] = {n: {'cropUV': local[i].tolist(), 'inside': inside[n]} for i, n in enumerate(names) if n in extents}
    missing = [n for n in names if not inside[n]]
    if missing:
        out['reasons'].append('Outside the source-supported crop: ' + ', '.join(missing)); return out
    # The app's runtime rule at each anchor's exact crop projection (protocol v3 pixelUV), with the crop camera built as
    # assemble_av2_scene.py builds it: an Unknown here would leave that suggestion unplaced.
    cropped = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]]) @ rect['P1']
    scene = {'camera': {'fx': float(cropped[0, 0]), 'fy': float(cropped[1, 1]), 'cx': float(cropped[0, 2]), 'cy': float(cropped[1, 2]),
                        'width': w, 'height': hgt},
             'transform': {'worldFromCamera': (ground_tool.WORLD_FROM_CITY @ city_from_camera[:3, :3] @ rect['R1'].T).tolist()}}
    runtime = {}
    for i, n in enumerate(names):
        if n not in anchors:
            continue
        check = ground_tool.project(scene, anchors[n])
        require(check is not None and np.allclose(check, local[i], rtol=0, atol=1e-6), 'Crop camera disagrees with the rectified projection')
        _d, hit, reason = ground_tool.pixel_ground_hit(scene, local[i].tolist(), ctx['terrain'], polygons, ctx['maxRange'], ctx['rangeLimit'])
        runtime[n] = {'placed': hit is not None, 'reason': reason, 'offsetFromAnchorM':
                      None if hit is None else float(math.hypot(hit['point'][0] - anchors[n][0], hit['point'][2] - anchors[n][2]))}
    out['runtimeCheck'] = runtime; out['cropCamera'] = scene
    unplaced = [n for n, r in runtime.items() if not r['placed']]
    if unplaced:
        out['reasons'].append('The runtime ray rule leaves anchors unplaced: ' + ', '.join(unplaced)); return out
    # Diagnostics only (never decide): how much of the crossing and of the whole vehicle box the crop shows.
    corners = np.vstack([ctx['edges']['edge1'], ctx['edges']['edge2'][::-1]])
    cuv, cz = project_city(corners, city_from_camera, rect)
    out['diagnostics'] = {'crossingCornersInCrop': int(sum(bool(z > 0 and 0 <= u - x0 <= w - 1 and 0 <= v - y0 <= hgt - 1) for (u, v), z in zip(cuv, cz))),
                          'originCity': origin.tolist()}
    if vehicle is not None:
        box = vehicle['box']; m = np.asarray(box['worldFromBox'], float); c = np.asarray(box['center'], float)
        allc = np.array([c + m @ np.array([sl * box['length'] / 2, sw * box['width'] / 2, sh * box['height'] / 2])
                         for sl in (-1, 1) for sw in (-1, 1) for sh in (-1, 1)])
        buv, bz = project_city(to_city(allc, origin), city_from_camera, rect)
        out['diagnostics']['vehicleBoxCornersInCrop'] = int(sum(bool(z > 0 and 0 <= u - x0 <= w - 1 and 0 <= v - y0 <= hgt - 1) for (u, v), z in zip(buv, bz)))
    out['feasible'] = True
    return out


# ------------------------------------------------------------------ sources and plan
def find_sources(log_dir, map_dirs, log, camera_ts, lidar_ts, license_path, listings, sensor_dirs=()):
    """The ten prepare_av2 source roles, found by publisher names. The three sensor files (left/right image, LiDAR
    sweep) are searched under the log directory, then under each --sensor-dir (same publisher layout); several copies
    must be identical. Missing sensor files are reported with their listing record; present ones must match the
    listing size and MD5 ETag."""
    log_dir = Path(log_dir).resolve()
    relative = {'left': Path('sensors/cameras/stereo_front_left') / f'{camera_ts}.jpg',
                'right': Path('sensors/cameras/stereo_front_right') / f'{camera_ts}.jpg',
                'lidar': Path('sensors/lidar') / f'{lidar_ts}.feather'}
    roots = [log_dir] + [Path(d).resolve() for d in sensor_dirs]
    fixed = {'poses': log_dir / 'city_SE3_egovehicle.feather',
             'intrinsics': log_dir / 'calibration/intrinsics.feather',
             'extrinsics': log_dir / 'calibration/egovehicle_SE3_sensor.feather',
             'license': Path(license_path).resolve()}
    candidates = {}
    for role in SENSOR_ROLES:
        found = [root / relative[role] for root in roots if (root / relative[role]).is_file()]
        require(len({_sha(p) for p in found}) <= 1, f'Conflicting {role} sensor files: ' + ', '.join(map(str, found)))
        fixed[role] = found[0] if found else log_dir / relative[role]
        candidates[role] = [str(root / relative[role]) for root in roots]
    maps = sorted((log_dir / 'map').glob(f'log_map_archive_{log}____*.json'))
    require(len(maps) == 1, 'Exactly one log map archive required')
    fixed['map'] = maps[0]
    fixed['groundRaster'], fixed['groundTransform'] = ground_files(log_dir, map_dirs, log)
    sources, missing = {}, []
    for role in ('left', 'right', 'lidar', 'poses', 'intrinsics', 'extrinsics', 'map', 'license', 'groundTransform', 'groundRaster'):
        path = fixed[role]
        listed = {'left': listings['left'].get(camera_ts), 'right': listings['right'].get(camera_ts),
                  'lidar': listings['lidar'].get(lidar_ts)}.get(role)
        if not path.is_file():
            require(role in SENSOR_ROLES, f'Required metadata source missing: {role}')
            missing.append({'role': role, 'expectedPath': str(path), 'searchedPaths': candidates[role], 'listing': listed}); continue
        data = snapshot(path, max_bytes=MAX_FILE)
        if listed is not None:
            require(len(data) == listed['bytes'], f'{role} size differs from the public listing')
            if '-' not in listed['etag']:  # single-part upload: the ETag is the MD5 of the object
                require(hashlib.md5(data).hexdigest() == listed['etag'], f'{role} MD5 differs from the listing ETag')
        sources[role] = {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}
    return sources, missing


def ground_files(log_dir, map_dirs, log):
    """Publisher ground raster and Sim2 by name, searched in <log>/map then each extra directory; identical copies only."""
    found = {'raster': [], 'sim2': []}
    for d in [Path(log_dir) / 'map'] + [Path(p).resolve() for p in map_dirs]:
        found['raster'] += sorted(d.glob(f'{log}_ground_height_surface____*.npy'))
        found['sim2'] += [d / f'{log}___img_Sim2_city.json'] if (d / f'{log}___img_Sim2_city.json').is_file() else []
    for kind, paths in found.items():
        require(paths, f'Ground {kind} not found')
        require(len({_sha(p) for p in paths}) == 1, f'Conflicting ground {kind} files')
    return found['raster'][0].resolve(), found['sim2'][0].resolve()


def plan_bytes(log, crossing, frame, rect_size, crop, sources):
    plan = {'schemaVersion': 1, 'sourceClass': 'Argoverse2Sensor', 'logId': log, 'crossingId': crossing,
            'cameraTimestampNs': str(frame['cameraTimestampNs']), 'lidarTimestampNs': str(frame['lidarTimestampNs']),
            'rectifiedSize': rect_size, 'crop': crop, 'sources': sources}
    return (json.dumps(plan, indent=2) + '\n').encode()


def site_entry(split, log, crossing, held_out_run, lock_check):
    """The split site. A log with any non-development role runs only when held_out_run is set AND lock_check() (the
    protocol-lock.json verification) passes; both happen before any file of the log is opened."""
    roles = {s['role'] for s in split['sites'] if s['logId'] == log}
    require(roles, 'Log is not in the frozen split')
    allowed = set(PARAMETERS['site']['allowedRolesWithoutHeldOutFlag'])
    require(held_out_run or roles <= allowed,
            'Held-out, control or reserve logs run only in M7 with --held-out-run and a valid protocol-lock.json')
    lock = lock_check() if held_out_run else None  # the flag alone is never enough
    sites = [s for s in split['sites'] if s['logId'] == log and s['crossingId'] == crossing]
    require(len(sites) == 1, 'Site (log + crossing) is not in the frozen split')
    return sites[0], lock


def build(args):
    output = Path(args.output).absolute()
    require(not os.path.lexists(output), 'Existing output retained')
    protocol_raw = snapshot(args.protocol, args.protocol_sha256); protocol = parse(protocol_raw)  # pinned literal sha256
    split_raw = snapshot(args.split, max_bytes=1_000_000); lock = parse(snapshot(args.split_lock))
    require(digest(split_raw) == lock['sha256'], 'split.json differs from split.lock.json')
    check_frozen_protocol(protocol, digest(split_raw), args.placement_protocol_sha256)
    split = parse(split_raw)
    log, crossing = args.log_id, args.crossing_id
    require(str(UUID(log)) == log and re.fullmatch('[0-9]+', crossing), 'Canonical log UUID and decimal crossing id required')
    site, lock_record = site_entry(split, log, crossing, args.held_out_run,  # before any log file is opened
                                   lambda: check_protocol_lock(args.split, args.protocol_lock_sha256, digest(split_raw),
                                                               args.protocol_sha256, protocol['code']))
    placement_raw = snapshot(args.placement_protocol, args.placement_protocol_sha256); placement = parse(placement_raw)
    require(placement.get('protocolVersion') == 3 and placement['currentVehicle']['pose']['rule'] == 'averageAllAnnotations'
            and placement['pixelSelection']['rule'] == 'exactProjection', 'Placement protocol v3 (averaged pose, exact projection) required')
    require('average_prefix' in inspect.signature(ground_tool.cuboids_from_annotations).parameters,
            'Shared ground code lacks the protocol v3 averaged pose')
    epsilon = placement['tolerances']['boundaryEpsilonMeters']; range_limit = placement['placementRange']['limitMeters']
    max_range = placement['tolerances']['maxRangeMeters']
    log_dir = Path(args.log_dir).resolve(); require(log_dir.name == log, 'Log directory name must be the log id')
    listings = {side: load_listing(snapshot(path, max_bytes=1_000_000)) for side, path in
                (('left', args.listing_left), ('right', args.listing_right), ('lidar', args.listing_lidar))}
    for side, prefix in (('left', 'sensors/cameras/stereo_front_left/'), ('right', 'sensors/cameras/stereo_front_right/'), ('lidar', 'sensors/lidar/')):
        require(all(f'/{log}/{prefix}' in v['key'] for v in listings[side].values()), f'{side} listing is for another log or sensor')
    poses_raw = snapshot(log_dir / 'city_SE3_egovehicle.feather', max_bytes=MAX_FILE); poses = load_poses(poses_raw)
    map_path = next(iter(sorted((log_dir / 'map').glob(f'log_map_archive_{log}____*.json'))), None); require(map_path is not None, 'Log map missing')
    map_raw = snapshot(map_path, max_bytes=MAX_FILE); vector_map, _unknown = prepare_av2.decode_native_map(map_raw)
    edges = crossing_geometry(vector_map, crossing)
    ann_raw = snapshot(log_dir / 'annotations.feather', max_bytes=MAX_FILE); ann = load_annotations(ann_raw)
    intr_raw = snapshot(log_dir / 'calibration/intrinsics.feather', max_bytes=MAX_FILE)
    extr_raw = snapshot(log_dir / 'calibration/egovehicle_SE3_sensor.feather', max_bytes=MAX_FILE)
    rect = rectification(intr_raw, extr_raw, PARAMETERS['rectification']['divisor'])
    # Ground raster for anchor heights (same sampling rule as the ground layer).
    raster_path, sim_path = ground_files(log_dir, args.map_dir or [], log)
    raster = prepare_av2.ground_raster(snapshot(raster_path, max_bytes=MAX_FILE)); sim2 = parse(snapshot(sim_path))
    drivable = []
    for key in sorted(vector_map['drivable_areas'], key=str):
        outline = ground_tool._xyz(vector_map['drivable_areas'][key]['area_boundary'])
        require(len(outline) >= 3 and np.isfinite(outline[:, :2]).all(), 'Drivable outline needs finite XY vertices')
        drivable.append(outline)
    lanes = []
    for key in sorted(vector_map['lane_segments'], key=str):
        seg = vector_map['lane_segments'][key]
        left, right = ground_tool._xyz(seg['left_lane_boundary']), ground_tool._xyz(seg['right_lane_boundary'])
        if len(left) < 2 or len(right) < 2 or not (np.isfinite(left[:, :2]).all() and np.isfinite(right[:, :2]).all()):
            continue
        lanes.append({'id': str(seg['id']), 'laneType': seg.get('lane_type'), 'isIntersection': bool(seg.get('is_intersection')),
                      'left': left, 'right': right})
    clusters = stationary_clusters(ann, poses, PARAMETERS['vehicle'])
    site_info, site_rows, site_reason = site_vehicles(poses, edges, clusters, PARAMETERS['vehicle'])
    site_tracks = sorted({t for r in site_rows if r['matchesSelectionRule'] for t in r['tracks']})
    # The reproduction must agree with split.json's frozen evidence; a disagreement stops the site (it is never
    # silently turned into a declared-default scene). Only a development site may lack evidence.
    cross_check = split_cross_check(site, site_rows, PARAMETERS['splitCheck'])
    if cross_check['ok']:
        cross_check['status'] = 'agrees'
    elif cross_check['evidenceRows'] is None and site['role'] in PARAMETERS['site']['allowedRolesWithoutHeldOutFlag']:
        cross_check['status'] = 'no-evidence-development-site'
    else:
        cross_check['status'] = 'mismatch'
    window, excluded = frame_window(poses, edges, listings['left'], listings['right'], listings['lidar'], PARAMETERS['frame'])
    if args.diagnostic_camera_timestamp is not None:
        forced = [w for w in window if w['cameraTimestampNs'] == args.diagnostic_camera_timestamp]
        require(forced, 'Diagnostic frame is not in the frozen frame window')
        order = forced
    else:
        order = window
    if cross_check['status'] == 'mismatch':
        order = []
    ctx = {'poses': poses, 'rect': rect, 'edges': edges, 'drivable': drivable, 'lanes': lanes, 'annotationsData': ann_raw,
           'placementProtocol': placement, 'epsilon': epsilon, 'rangeLimit': range_limit, 'maxRange': max_range, 'terrain': None}
    evaluated = []
    for frame in order:
        origin = (poses['rigid'][frame['cameraTimestampNs']] @ rect['egoFromLeft'])[:3, 3] if frame['cameraTimestampNs'] in poses['rigid'] else np.zeros(3)
        ctx['terrain'] = ground_tool.Terrain(raster, 0, 0, raster.shape, sim2, origin)
        evaluated.append(evaluate_frame(frame, ctx, site_tracks))
    chosen = next((e for e in evaluated if e['feasible']), None)
    files = {}
    selection = {
        'schemaVersion': 1, 'kind': 'av2SceneBuilderSelection',
        'mode': 'diagnostic-forced-frame' if args.diagnostic_camera_timestamp is not None else 'rule',
        'site': {'logId': log, 'crossingId': crossing, 'role': site['role']},
        'builder': {'script': 'scripts/california/build_av2_scene.py', 'sha256': _sha(__file__), 'version': BUILDER_VERSION,
                    'runtimeCode': runtime_code_hashes(), 'runtimeCodeMatchesFrozenProtocol': True,
                    # Module attributes, not package metadata: opencv-python-headless/contrib or a missing dist-info
                    # must not stop a finished run.
                    'python': sys.version.split()[0],
                    'dependencies': {'numpy': np.__version__, 'opencv': prepare_av2.cv2.__version__, 'pyarrow': pyarrow.__version__}},
        'heldOutLock': lock_record,
        'splitCrossCheck': cross_check,
        'inputs': {'builderProtocol': {'path': str(Path(args.protocol).resolve()), 'sha256': args.protocol_sha256},
                   'placementProtocol': {'path': str(Path(args.placement_protocol).resolve()), 'sha256': args.placement_protocol_sha256},
                   'split': {'path': str(Path(args.split).resolve()), 'sha256': digest(split_raw)},
                   'listings': {side: {'path': str(Path(p).resolve()), 'sha256': _sha(p), 'objects': len(listings[side])}
                                for side, p in (('left', args.listing_left), ('right', args.listing_right), ('lidar', args.listing_lidar))},
                   'poses': digest(poses_raw), 'map': digest(map_raw), 'annotations': digest(ann_raw),
                   'intrinsics': digest(intr_raw), 'extrinsics': digest(extr_raw), 'groundRaster': _sha(raster_path), 'groundTransform': _sha(sim_path)},
        'rectification': {'rectifiedSize': rect['rectifiedSize'], 'commonSupportRectangle': rect['support']},
        'splitRule': {'reason': site_reason, **(site_info or {}), 'stationaryClusters': len(clusters), 'vehiclesNearCrossing': site_rows,
                      'selectionRuleTracks': site_tracks},
        'frameWindow': {'candidates': len(window), 'excluded': excluded,
                        'frozenFramesRuleChoice': window[0] if window else None},
        'evaluatedFrames': evaluated,
        'status': None, 'chosen': None, 'missingSources': [], 'protocolSceneValues': None,
        'limitations': ['Choices use metadata only (poses, calibration, map, ground raster, annotations, public listings); images and '
                        'LiDAR are hashed into the plan, never inspected to choose.',
                        'The 12 m target is measured from the ego-vehicle origin to the mean of the nearer edge polyline (frozen '
                        'frames.py definition), not from the camera along the lane axis.',
                        'Anchors follow placement protocol v3: the current vehicle is the average of all its annotations; the '
                        'crossing point lies on its centre line; the person is 0.30 m inside the curb proxy on the near edge.',
                        'Stationary means only that the annotation stayed in place; no reason for stopping is stated or implied.']}
    if cross_check['status'] == 'mismatch':
        selection['status'] = 'split-rule-mismatch'
    elif chosen is None:
        selection['status'] = 'no-feasible-frame' if window else 'no-frame-in-window'
    else:
        sources, missing = find_sources(log_dir, args.map_dir or [], log, chosen['cameraTimestampNs'], chosen['lidarTimestampNs'],
                                        args.license, listings, args.sensor_dir or [])
        vehicle = chosen['currentVehicle']
        selection['chosen'] = {'cameraTimestampNs': str(chosen['cameraTimestampNs']), 'lidarTimestampNs': str(chosen['lidarTimestampNs']),
                               'nearEdgeM': chosen['nearEdgeM'],
                               'rankInWindow': [w['cameraTimestampNs'] for w in window].index(chosen['cameraTimestampNs']) + 1,
                               'egoLaneId': chosen['egoLaneId'], 'crop': chosen['crop'], 'currentVehicle': vehicle}
        selection['protocolSceneValues'] = {
            'scene': {'logId': log, 'cameraTimestampNs': str(chosen['cameraTimestampNs']), 'lidarTimestampNs': str(chosen['lidarTimestampNs']),
                      'annotationTimestampNs': str(chosen['lidarTimestampNs']), 'crossingId': crossing,
                      'sceneId': f'av2-pao-{log[:8]}-{crossing}', 'crop': chosen['crop']},
            'roadAxisLaneId': chosen['egoLaneId'],
            'currentVehicleTrackPrefix': vehicle.get('trackPrefix')}
        selection['missingSources'] = missing
        if missing:
            selection['status'] = 'sources-missing'
        else:
            selection['status'] = 'plan-written'
            name = 'diagnostic-plan.json' if args.diagnostic_camera_timestamp is not None else 'plan.json'
            files[name] = plan_bytes(log, crossing, chosen, rect['rectifiedSize'], chosen['crop'], sources)
            selection['plan'] = {'file': name, 'sha256': digest(files[name])}
    for record in evaluated + ([window[0]] if window else []):  # decimal-string timestamps, as in every AV2 plan
        for key in ('cameraTimestampNs', 'lidarTimestampNs'):
            record[key] = str(record[key])
    files['selection.json'] = (json.dumps(selection, indent=1, sort_keys=True, default=_jsonable) + '\n').encode()
    for data in files.values():
        require(b'park' not in data.lower(), 'Builder output must not use any form of the banned vehicle-status word')
    publish(output, files)
    return selection


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('protocol', 'protocol-sha256', 'placement-protocol', 'placement-protocol-sha256', 'split', 'split-lock',
                 'log-id', 'crossing-id', 'log-dir', 'listing-left', 'listing-right', 'listing-lidar', 'license', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--map-dir', action='append', help='Extra directory searched for the ground raster and Sim2')
    parser.add_argument('--sensor-dir', action='append',
                        help='Extra directory (publisher layout sensors/...) searched for the chosen images and LiDAR sweep')
    parser.add_argument('--diagnostic-camera-timestamp', type=int, help='Diagnostic only: evaluate the rules at this window frame')
    parser.add_argument('--held-out-run', action='store_true', help='M7 only: needs protocol-lock.json beside split.json')
    parser.add_argument('--protocol-lock-sha256', help='M7 only: sha256 of protocol-lock.json recorded at the M6 lock')
    args = parser.parse_args()
    try:
        selection = build(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'AV2 scene builder rejected: {error}\n')
    chosen = selection['chosen'] or {}
    print(canonical({'status': selection['status'], 'mode': selection['mode'], 'camera': chosen.get('cameraTimestampNs'),
                     'crop': chosen.get('crop'), 'lane': chosen.get('egoLaneId'),
                     'vehicle': (chosen.get('currentVehicle') or {}).get('track'), 'missing': len(selection['missingSources'])}).decode())


if __name__ == '__main__':
    main()

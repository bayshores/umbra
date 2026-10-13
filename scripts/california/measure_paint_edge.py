#!/usr/bin/env python3
"""Development-only LiDAR paint-edge measurement for one mapped AV2 pedestrian crossing (umbra M3).

Measures where the painted crosswalk starts along the approach direction from a fixed number of
stacked AV2 LiDAR sweeps, without image vision, and reports the signed offset from the HD-map near
edge. Two estimators with different detection rules must agree within the protocol tolerance;
otherwise the crossing is unresolved. Every parameter comes from a frozen protocol file (version 4:
estimator A only uses lasers with a road reference and needs paint brighter than that laser's road
upper tail and a run of consecutive paint returns; precision is a leave-one-sweep-out jackknife).

A and B share the returns, frame, ground band, windows and line fit, so their agreement checks the two
detection rules only; it is not independent confirmation, and systematic terms (pose/raster
registration, beam footprint, float16 storage) shift both alike and are not in the stated uncertainty.

Measurement only. Nothing here changes the app, a scene file or the mapped edge (decision 5 keeps the
mapped edge with a measured +/- label). Evaluation tool; never shipped. No downloads, no network.

  select   choose the protocol's fixed number of sweeps from poses, map and the public sweep listing
           (no LiDAR content is read, so the choice cannot depend on the result)
  measure  stack the selected sweeps (hash-verified) and run estimators A and B

Coordinates: S is the signed distance along the approach direction t from the mapped near-edge line
(positive = into the crossing, away from the approaching driver). L is the lateral coordinate along
r = t rotated 90 degrees clockwise (positive = right of travel), from the mean point where the ego
headings meet the mapped near edge. A paint offset is the paint-edge S; positive means the paint starts
beyond the mapped edge.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata

SCHEMA = 1
MAX_SOURCE = 32_000_000
AV2_LIDAR_KEY = re.compile(r'<Key>datasets/av2/sensor/(?:train|val|test)/([0-9a-f-]{36})/sensors/lidar/(\d{18})\.feather</Key>')


# ------------------------------------------------------------------ small helpers
def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f'Duplicate JSON key {key!r}')
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=pairs)


def read_verified(path, expected):
    data = Path(path).read_bytes()
    require(len(data) <= MAX_SOURCE, f'{path}: exceeds byte ceiling')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), f'{path}: invalid expected SHA-256')
    require(sha256(data) == expected, f'{path}: SHA-256 mismatch')
    return data


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    require(n > 0 and math.isfinite(n), 'Zero or nonfinite direction')
    return v / n


def rnd(x, digits=4):
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


# ------------------------------------------------------------------ robust statistics
def theil_sen(x, y, at):
    """Theil-Sen line through (x, y): (value at x=at, slope). Needs two distinct x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-12
    require(ok.any(), 'Theil-Sen needs two distinct x values')
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return float(np.median(y - slope * (x - at))), slope


def robust_sigma(values):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def pettitt_increasing(values):
    """One-sided Pettitt (1979) statistic for an increase in level after each split.

    Returns (t, k): for t = 1..n-1 the first t values precede the split and
    k_t = -U_t = sum_{i<=t<j} sgn(x_j - x_i), computed as t(n+1) - 2*sum(rank_1..t) with average ranks."""
    x = np.asarray(values, float); n = len(x)
    require(n >= 3, 'Pettitt needs at least three values')
    t = np.arange(1, n)
    return t, t * (n + 1) - 2 * np.cumsum(rankdata(x))[:-1]


def l1_step(values, min_side):
    """Least-absolute-deviation two-level step: every split t (first t values before it, at least min_side
    each side) minimizing sum|x - median(before)| + sum|x - median(after)|. Returns (t_first, t_last, cost)
    over the tied minimizers, or None."""
    x = np.asarray(values, float); n = len(x)
    if n < 2 * min_side or min_side < 1:
        return None
    splits = np.arange(min_side, n - min_side + 1)
    cost = np.array([np.abs(x[:t] - np.median(x[:t])).sum() + np.abs(x[t:] - np.median(x[t:])).sum() for t in splits])
    ties = splits[cost <= cost.min() + 1e-9]
    return int(ties.min()), int(ties.max()), float(cost.min())


def change_point(values, min_side):
    """Pettitt significance for an increase, located by the L1 two-level step.

    p ~= exp(-6K^2/(n^3+n^2)) with K the largest one-sided Pettitt statistic over allowed splits; the
    location is the L1 step; auc = P(after > before) + P(tie)/2 at that location (from the Pettitt statistic,
    k_t = n1*n2*(2*auc - 1)); `increase` compares the two medians."""
    x = np.asarray(values, float); n = len(x)
    step = l1_step(x, min_side)
    if step is None:
        return None
    t, k = pettitt_increasing(x)
    allowed = (t >= min_side) & (t <= n - min_side)
    K = float(k[allowed].max())
    p = float(min(1.0, math.exp(-6.0 * K * K / (n ** 3 + n ** 2)))) if K > 0 else 1.0
    first, last, cost = step
    n1, n2 = first, n - first
    auc = (float(k[first - 1]) / (n1 * n2) + 1.0) / 2.0
    increase = float(np.median(x[first:])) > float(np.median(x[:first]))
    return {'first': first, 'last': last, 'K': K, 'p': p, 'auc': auc, 'increase': bool(increase), 'n': n}


# ------------------------------------------------------------------ geometry
def partition(lo, hi, width, offset, min_width):
    """Strips [a, b) of `width` from lo+offset (a leading partial strip [lo, lo+offset) when offset > 0)."""
    edges = [lo] if offset > 0 else []
    a = lo + offset
    while a < hi - 1e-9:
        edges.append(a); a += width
    edges.append(hi)
    strips = [(float(a), float(min(b, hi))) for a, b in zip(edges[:-1], edges[1:])]
    return [s for s in strips if s[1] - s[0] >= min_width - 1e-9]


def segment_intersection(p1, p2, q1, q2):
    """Parameter u on p1->p2 where it meets segment q1->q2, else None."""
    d1 = p2 - p1; d2 = q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    w = q1 - p1
    u = (w[0] * d2[1] - w[1] * d2[0]) / den
    v = (w[0] * d1[1] - w[1] * d1[0]) / den
    return float(u) if 0 <= u <= 1 and 0 <= v <= 1 else None


def edge_points(crossing, key):
    pts = np.array([[float(q['x']), float(q['y'])] for q in crossing[key]], float)
    require(pts.shape == (2, 2) and np.isfinite(pts).all(), f'Crossing {key} needs two finite points')
    return pts


def along_line(points_xy, a, normal, t):
    """Signed distance along t from the line through a with unit normal `normal` (normal.t > 0)."""
    return ((np.asarray(points_xy, float) - a) @ normal) / float(normal @ t)


def point_at(frame, lateral, s):
    """City XY of the point with lateral coordinate `lateral` and along-travel distance `s` from the mapped near edge."""
    _, _, n = frame['near']; t, r = frame['t'], frame['r']
    on_edge = frame['origin'] + lateral * r - lateral * float(r @ n) / float(t @ n) * t
    return on_edge + s * t


def oriented_normal(a, b, t):
    u = unit(b - a); n = np.array([-u[1], u[0]])
    return n if n @ t > 0 else -n


# ------------------------------------------------------------------ inputs
def load_poses(data):
    table = feather.read_table(pa_buffer(data)).to_pydict()
    for key in ('timestamp_ns', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m'):
        require(key in table, f'Poses missing column {key}')
    index = {int(ts): i for i, ts in enumerate(table['timestamp_ns'])}
    require(len(index) == len(table['timestamp_ns']), 'Duplicate pose timestamps')
    return table, index


def pa_buffer(data):
    import pyarrow as pa
    return pa.BufferReader(data)


def pose_matrix(table, i):
    q = np.array([table[k][i] for k in ('qx', 'qy', 'qz', 'qw')], float)
    t = np.array([table[k][i] for k in ('tx_m', 'ty_m', 'tz_m')], float)
    require(np.isfinite(q).all() and np.isfinite(t).all() and abs(q @ q - 1) <= 1e-6, 'Invalid pose')
    m = np.eye(4); m[:3, :3] = Rotation.from_quat(q).as_matrix(); m[:3, 3] = t
    return m


def lidar_offset(extrinsics_data, sensor):
    table = feather.read_table(pa_buffer(extrinsics_data)).to_pydict()
    names = list(table['sensor_name'])
    require(names.count(sensor) == 1, f'Extrinsics need exactly one {sensor}')
    i = names.index(sensor)
    t = np.array([table[k][i] for k in ('tx_m', 'ty_m', 'tz_m')], float)
    require(np.isfinite(t).all(), 'Nonfinite LiDAR extrinsics')
    return t


def listing_timestamps(xml_bytes, log_id):
    text = xml_bytes.decode('utf-8')
    require('<IsTruncated>false</IsTruncated>' in text, 'Sweep listing must be complete (IsTruncated false)')
    found = AV2_LIDAR_KEY.findall(text)
    require(found and all(log == log_id for log, _ in found), 'Listing must contain only this log\'s sweeps')
    stamps = sorted({int(ts) for _, ts in found})
    require(len(stamps) == len(found), 'Duplicate sweep keys in listing')
    return stamps


def ground_raster(data):
    import io
    raster = np.load(io.BytesIO(data), allow_pickle=False)
    require(raster.ndim == 2 and raster.dtype == np.float16, 'Publisher float16 ground raster required')
    return raster


def ground_heights(xy, raster, sim2):
    """Same sampling rule as prepare_av2.ground_sample: index trunc(s*(R*xy + t)); NaN outside/aliased."""
    r = np.asarray(sim2['R'], float).reshape(2, 2); t = np.asarray(sim2['t'], float); s = float(sim2['s'])
    require(np.allclose(r.T @ r, np.eye(2), atol=1e-9) and abs(np.linalg.det(r) - 1) < 1e-9 and s > 0, 'Proper Sim2 required')
    with np.errstate(invalid='ignore', over='ignore'):
        img = s * (np.asarray(xy, float) @ r.T + t)
    ok = np.isfinite(img).all(1) & (np.abs(img) < 2 ** 62).all(1)
    idx = np.full(img.shape, -1, np.int64); idx[ok] = np.trunc(img[ok]).astype(np.int64)
    inside = ok & (idx[:, 0] >= 0) & (idx[:, 0] < raster.shape[1]) & (idx[:, 1] >= 0) & (idx[:, 1] < raster.shape[0])
    inside &= ~(img < 0).any(1)
    out = np.full(len(img), np.nan)
    out[inside] = raster[idx[inside, 1], idx[inside, 0]].astype(float)
    return out


def load_map(data):
    value = json.loads(data, parse_constant=lambda token: float('nan'))
    require(isinstance(value, dict) and 'pedestrian_crossings' in value, 'AV2 vector map required')
    return value


# ------------------------------------------------------------------ sweep selection (no LiDAR content)
def crossing_edges(vector_map, crossing_id):
    crossing = vector_map['pedestrian_crossings'].get(str(crossing_id))
    require(crossing is not None, f'Crossing {crossing_id} not in map')
    return {k: edge_points(crossing, k) for k in ('edge1', 'edge2')}


def select_sweeps(stamps, poses, pose_index, offset, edges, P):
    """Deterministic protocol sweep choice from poses + map only."""
    sel = P['sweepSelection']
    states = []
    for ts in stamps:
        require(ts in pose_index, f'No exact pose for sweep {ts}')
        m = pose_matrix(poses, pose_index[ts])
        states.append({'ts': ts, 'lidar': (m[:3, :3] @ offset + m[:3, 3])[:2], 'heading': unit(m[:2, 0])})
    corners = np.vstack(list(edges.values())); centroid = corners.mean(0)
    approach = [s for s in states if (centroid - s['lidar']) @ s['heading'] > 0]
    require(approach, 'Ego never approaches the crossing')
    closest = min(approach, key=lambda s: float(np.linalg.norm(centroid - s['lidar'])))
    near_key = min(edges, key=lambda k: float((edges[k].mean(0) - closest['lidar']) @ closest['heading']))
    far_key = [k for k in edges if k != near_key][0]
    a, b = edges[near_key]
    cos_max = math.cos(math.radians(sel['maxHeadingToEdgeNormalDeg']))
    lo, hi = sel['edgeAheadRangeM']
    candidates = []
    for s in states:
        n = oriented_normal(a, b, s['heading'])
        c = float(n @ s['heading'])
        if c < cos_max:
            continue
        d = float(((a - s['lidar']) @ n) / c)
        hit = s['lidar'] + d * s['heading']
        u = float((hit - a) @ (b - a) / ((b - a) @ (b - a)))
        if lo <= d <= hi and 0 <= u <= 1:
            candidates.append({**s, 'edgeAheadM': d, 'hit': hit})
    count = int(sel['count'])
    if len(candidates) < count:
        return {'ok': False, 'reason': f'Only {len(candidates)} candidate sweeps for {count}', 'nearEdge': near_key,
                'farEdge': far_key, 'candidates': len(candidates)}
    ds = np.array([c['edgeAheadM'] for c in candidates])
    targets = np.linspace(ds.min(), ds.max(), count)
    used, chosen = set(), []
    for target in targets:
        order = sorted(range(len(candidates)), key=lambda i: (abs(ds[i] - target), candidates[i]['ts']))
        i = next(i for i in order if i not in used)
        used.add(i); chosen.append(candidates[i])
    chosen.sort(key=lambda c: -c['edgeAheadM'])
    return {'ok': True, 'nearEdge': near_key, 'farEdge': far_key, 'candidates': len(candidates),
            'candidateEdgeAheadRangeM': [float(ds.min()), float(ds.max())], 'targetsM': targets.tolist(), 'chosen': chosen}


def build_frame(selection, edges, vector_map, P):
    """Travel direction, lateral origin and the approach-side corridor from map geometry only."""
    chosen = selection['chosen']
    t = unit(np.mean([c['heading'] for c in chosen], axis=0))
    r = np.array([t[1], -t[0]])
    a, b = edges[selection['nearEdge']]
    n = oriented_normal(a, b, t)
    origin = np.mean([c['hit'] for c in chosen], axis=0)
    origin = origin - (along_line(origin[None], a, n, t)[0]) * t  # snap onto the near-edge line along t
    fa, fb = edges[selection['farEdge']]
    fn = oriented_normal(fa, fb, t)
    lat = lambda p: float((np.asarray(p, float) - origin) @ r)
    C = P['corridor']
    cos_lane = math.cos(math.radians(C['approachLaneMaxAngleDeg']))
    lane_ls, lane_ids = [], []
    for lid, lane in vector_map.get('lane_segments', {}).items():
        if lane.get('lane_type') not in C['laneTypes']:
            continue
        for side in ('left_lane_boundary', 'right_lane_boundary'):
            pts = np.array([[float(q['x']), float(q['y'])] for q in lane[side]], float)
            for p1, p2 in zip(pts[:-1], pts[1:]):
                if not (np.isfinite(p1).all() and np.isfinite(p2).all()) or np.linalg.norm(p2 - p1) < 1e-9:
                    continue
                u = segment_intersection(a, b, p1, p2)
                if u is not None and unit(p2 - p1) @ t >= cos_lane:
                    lane_ls.append(lat(a + u * (b - a))); lane_ids.append(str(lid))
    require(lane_ls, 'No approach lane crosses the mapped near edge')
    curb_ls = []
    for area in vector_map.get('drivable_areas', {}).values():
        pts = np.array([[float(q['x']), float(q['y'])] for q in area['area_boundary']], float)
        for p1, p2 in zip(pts, np.roll(pts, -1, 0)):
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            u = segment_intersection(a, b, p1, p2)
            if u is not None:
                curb_ls.append(lat(a + u * (b - a)))
    left = min(lane_ls); lane_right = max(lane_ls)
    right_candidates = [x for x in curb_ls if x >= lane_right - 1e-6]
    right = min(right_candidates) if right_candidates else lane_right
    seg = sorted([lat(a), lat(b)])
    left, right = max(left, seg[0]), min(right, seg[1])
    lo, hi = left + C['trimM'], right - C['trimM']
    require(hi - lo >= C['minWidthM'], 'Approach corridor too narrow')
    frame = {'t': t, 'r': r, 'origin': origin, 'near': (a, b, n), 'far': (fa, fb, fn),
             'corridor': (float(lo), float(hi)), 'evalL': float((lo + hi) / 2),
             'corridorSources': {'approachLaneBoundaryL': sorted(float(x) for x in lane_ls),
                                 'approachLaneIds': sorted(set(lane_ids)), 'drivableBoundaryL': sorted(curb_ls),
                                 'rightBoundFromCurb': bool(right_candidates), 'untrimmed': [float(left), float(right)]}}
    # mapped crossing depth: along-travel distance from the mapped near edge to the mapped far edge
    frame['mappedFarEdgeSAtCorridorEnds'] = [float(-along_line(point_at(frame, x, 0.0)[None], fa, fn, t)[0]) for x in (lo, hi)]
    return frame


# ------------------------------------------------------------------ stacking
def stack_sweeps(sweeps, poses, pose_index, offset, raster, sim2, frame, P):
    """City points of each sweep in the crossing frame: ground returns and elevated returns (for shadows)."""
    G = P['groundBand']; W = P['windows']
    lo_c, hi_c = frame['corridor']
    t, r, origin = frame['t'], frame['r'], frame['origin']
    a, _, n = frame['near']; fa, _, fn = frame['far']
    ego = P['shadow']['egoBoxM']
    ground, elevated, lidars, counts = [], [], [], []
    span = (min(W['roadReferenceS'][0], W['searchS'][0]), max(W['roadReferenceS'][1], W['searchS'][1]))
    for k, (ts, table) in enumerate(sweeps):
        x = np.asarray(table['x'], float); y = np.asarray(table['y'], float); z = np.asarray(table['z'], float)
        inten = np.asarray(table['intensity'], float); laser = np.asarray(table['laser_number'], int)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        own = (x >= ego[0]) & (x <= ego[1]) & (np.abs(y) <= ego[2])
        m = pose_matrix(poses, pose_index[ts])
        city = np.column_stack([x, y, z]) @ m[:3, :3].T + m[:3, 3]
        lidar = (m[:3, :3] @ offset + m[:3, 3])
        lidars.append(lidar)
        xy = city[:, :2]
        S = along_line(xy, a, n, t); Sf = -along_line(xy, fa, fn, t)
        L = (xy - origin) @ r
        lat_ok = (L >= lo_c - 1.0) & (L <= hi_c + 1.0)
        roi = finite & ~own & lat_ok & (((S >= span[0] - 0.5) & (S <= span[1] + 0.5)) | ((Sf >= span[0] - 0.5) & (Sf <= span[1] + 0.5)))
        h = np.full(len(x), np.nan); h[roi] = ground_heights(xy[roi], raster, sim2)
        dz = city[:, 2] - h
        g = roi & np.isfinite(dz) & (dz >= G['belowM']) & (dz <= G['aboveM'])
        ground.append(np.column_stack([S[g], Sf[g], L[g], inten[g], laser[g], np.full(g.sum(), k)]))
        # elevated returns anywhere between this sweep's LiDAR and the corridor (shadow evidence only)
        ends = np.array([point_at(frame, lat, along) for lat in (lo_c, hi_c) for along in (span[0], span[1])] + [lidar[:2]])
        near_box = finite & ~own & (xy >= ends.min(0) - 2).all(1) & (xy <= ends.max(0) + 2).all(1)
        he = np.full(len(x), np.nan); he[near_box] = ground_heights(xy[near_box], raster, sim2)
        dze = city[:, 2] - he
        e = near_box & np.isfinite(dze) & (dze >= P['shadow']['obstacleHeightM'][0]) & (dze <= P['shadow']['obstacleHeightM'][1])
        elevated.append(xy[e])
        counts.append({'timestampNs': str(ts), 'rows': int(len(x)), 'groundInRegion': int(g.sum()), 'elevatedNearPath': int(e.sum())})
    pts = np.vstack(ground) if ground else np.zeros((0, 6))
    return {'S': pts[:, 0], 'Sf': pts[:, 1], 'L': pts[:, 2], 'I': pts[:, 3], 'laser': pts[:, 4].astype(int),
            'sweep': pts[:, 5].astype(int), 'elevated': elevated, 'lidars': lidars, 'perSweep': counts}


# ------------------------------------------------------------------ estimator A: intensity ratio + first sustained paint return
def laser_tails(S, I, laser, P, W):
    """Estimator A's per-laser road upper tail: the laserTailQuantile of each laser's raw corridor ground
    intensities in the road reference window (lasers with fewer than laserReferenceMinReturns are not used by A).

    A laser's ground ring has a fixed radius, so its range and incidence angle are the same before the crossing
    and in the reference window; a laser with a high gain (or one never seen in the reference window) therefore
    cannot pass road returns as paint."""
    A = P['estimatorA']; r_lo, r_hi = W['roadReferenceS']
    ref = (S >= r_lo) & (S < r_hi)
    tails = {}
    for l in (np.unique(laser[ref]) if len(laser) else []):
        v = I[ref & (laser == l)]
        if len(v) >= A['laserReferenceMinReturns']:
            tails[int(l)] = float(np.quantile(v, A['laserTailQuantile']))
    return tails


def first_onset(ss, paint, A):
    """First paint return such that (i) at least onsetMinPaintInLookahead of it and the next returns in S order
    (onsetLookaheadReturns in all) are paint, and (ii) >= confirmMinPaintReturns paint returns make
    >= confirmMinPaintFraction of the returns within confirmWindowM. Rule (i) stops a single bright road return
    from being accepted only because the confirmation window reaches into real paint (v3 failure mode on a dark
    road); needing 2 of 3 rather than 3 of 3 keeps worn paint from being pushed late."""
    m, k = int(A['onsetLookaheadReturns']), int(A['onsetMinPaintInLookahead'])
    n = len(paint)
    cs = np.r_[0, np.cumsum(paint)]
    ahead = cs[np.minimum(np.arange(n) + m, n)] - cs[:n]
    for j in np.nonzero(paint & (ahead >= k))[0]:
        kk = np.searchsorted(ss, ss[j] + A['confirmWindowM'], side='left')
        n_paint = int(paint[j:kk].sum()); n_all = kk - j
        if n_paint >= A['confirmMinPaintReturns'] and n_paint >= A['confirmMinPaintFraction'] * n_all:
            return int(j)
    return None


def estimator_a(S, L, I, laser, corridor, at, P, W, also_at=None, strip_bootstrap=True, tails=None):
    """Per narrow lateral strip, using only lasers with a road reference: a return is paint when its intensity is
    >= ratio x the strip's road median AND above its laser's road upper tail; the onset is the first paint return
    starting a run of consecutive paint returns that is also sustained; the strip interval runs from the last
    non-paint return before it."""
    A = P['estimatorA']; C = P['coverage']
    if tails is None:
        tails = laser_tails(S, I, laser, P, W)
    lookup = np.full(int(laser.max()) + 1 if len(laser) else 1, np.nan)
    for l, v in tails.items():
        if l < len(lookup):
            lookup[l] = v
    tail = lookup[laser] if len(laser) else np.zeros(0)
    use = np.isfinite(tail)
    strips = partition(corridor[0], corridor[1], A['stripWidthM'], A['stripOffsetM'], A['minStripWidthM'])
    w_lo, w_hi = W['searchS']; r_lo, r_hi = W['roadReferenceS']
    out = []
    for s0, s1 in strips:
        m = (L >= s0) & (L < s1)
        ref = m & use & (S >= r_lo) & (S < r_hi)
        w = m & use & (S >= w_lo) & (S < w_hi)
        row = {'L': [rnd(s0, 3), rnd(s1, 3)], 'referenceReturns': int(ref.sum()), 'windowReturns': int(w.sum()),
               'returnsWithoutLaserReference': int((m & ~use & (S >= w_lo) & (S < w_hi)).sum())}
        if ref.sum() < A['minReferenceReturns']:
            out.append({**row, 'status': 'noRoadReference'}); continue
        road = max(float(np.median(I[ref])), A['roadMedianFloor'])
        threshold = A['ratio'] * road
        order = np.argsort(S[w], kind='stable')
        ss = S[w][order]; paint = (I[w][order] >= threshold) & (I[w][order] > tail[w][order])
        row.update({'roadMedian': road, 'threshold': threshold, 'paintReturns': int(paint.sum())})
        onset = first_onset(ss, paint, A)
        if onset is None:
            near_edge = int((m & (np.abs(S) <= C['halfWindowM'])).sum())
            out.append({**row, 'status': 'noEdgeReturns' if near_edge < C['minReturns'] else 'noPaint',
                        'returnsNearMappedEdge': near_edge}); continue
        before = np.nonzero(~paint[:onset])[0]
        if len(before) == 0:
            out.append({**row, 'status': 'onsetAtWindowStart', 'onsetS': rnd(ss[onset])}); continue
        lo_s, hi_s = float(ss[before[-1]]), float(ss[onset])
        k = np.searchsorted(ss, hi_s + A['continuityWindowM'], side='left')
        row.update({'lastRoadReturnS': rnd(lo_s), 'onsetReturnS': rnd(hi_s), 'gapM': rnd(hi_s - lo_s),
                    'paintFractionAfterOnset': rnd(float(paint[onset:k].mean()), 3)})
        if hi_s - lo_s > A['maxIntervalM'] + 1e-9:
            out.append({**row, 'status': 'edgeNotLocalized'}); continue
        out.append({**row, 'status': 'edge', 'interval': (lo_s, hi_s), 'centerL': (s0 + s1) / 2})
    result = fit_strips(out, at, A['minStrips'], P['uncertainty'], also_at, strip_bootstrap)
    fractions = [r['paintFractionAfterOnset'] for r in out if r['status'] == 'edge']
    result['paintFractionAfterOnsetMedian'] = float(np.median(fractions)) if fractions else None
    result['lasersWithReference'] = len(tails)
    result['returnsWithoutLaserReference'] = int((~use).sum())
    return result


# ------------------------------------------------------------------ estimator B: per-laser gain + rank change point
def laser_gains(S, I, laser, P, W):
    B = P['estimatorB']; r_lo, r_hi = W['roadReferenceS']
    ref = (S >= r_lo) & (S < r_hi)
    gains = {}
    for l in np.unique(laser[ref]):
        v = I[ref & (laser == l)]
        if len(v) >= B['minLaserReferenceReturns']:
            gains[int(l)] = max(float(np.median(v)), B['gainFloor'])
    return gains


def ring_segments(S, z, sweep, laser, origin, width, min_returns):
    """One observation per (sweep, laser, along-road cell): median S and median normalized intensity.

    A ring crossing a strip is a short arc at nearly constant S; summarizing it by medians keeps single
    bright or dark returns from moving the change point."""
    cell = np.floor((S - origin) / width).astype(np.int64)
    keys = np.column_stack([sweep.astype(np.int64), laser.astype(np.int64), cell])
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    s_med, z_med = group_medians(inverse, S, len(uniq)), group_medians(inverse, z, len(uniq))
    ok = counts >= min_returns
    return s_med[ok], z_med[ok]


def group_medians(group, values, n_groups):
    """Median of `values` per group id 0..n_groups-1 (same result as np.median per group; vectorized)."""
    order = np.lexsort((values, group))
    v = np.asarray(values, float)[order]
    counts = np.bincount(group, minlength=n_groups)
    start = np.r_[0, np.cumsum(counts)[:-1]]
    lo = start + (counts - 1) // 2; hi = start + counts // 2
    return (v[lo] + v[hi]) / 2


def estimator_b(S, L, I, laser, sweep, corridor, at, P, W, gains=None, also_at=None, strip_bootstrap=True):
    B = P['estimatorB']
    if gains is None:
        gains = laser_gains(S, I, laser, P, W)
    g = np.array([gains.get(int(l), np.nan) for l in laser], float) if len(laser) else np.zeros(0)
    keep = np.isfinite(g)
    z = np.where(keep, I / np.where(keep, g, 1.0), np.nan)
    strips = partition(corridor[0], corridor[1], B['stripWidthM'], B['stripOffsetM'], B['minStripWidthM'])
    w_lo, w_hi = W['searchS']
    out = []
    for s0, s1 in strips:
        m = keep & (L >= s0) & (L < s1) & (S >= w_lo) & (S < w_hi)
        row = {'L': [rnd(s0, 3), rnd(s1, 3)], 'windowReturns': int(m.sum())}
        if m.sum() == 0:
            out.append({**row, 'segments': 0, 'status': 'tooFewSegments'}); continue
        seg_s, seg_z = ring_segments(S[m], z[m], sweep[m], laser[m], w_lo, B['segmentLengthM'], B['minReturnsPerSegment'])
        row['segments'] = int(len(seg_s))
        if len(seg_s) < B['minSegments']:
            out.append({**row, 'status': 'tooFewSegments'}); continue
        order = np.argsort(seg_s, kind='stable')
        ss = seg_s[order]; zz = seg_z[order]
        c = change_point(zz, B['minSegmentsEachSide'])
        if c is None:
            out.append({**row, 'status': 'tooFewSegments'}); continue
        row.update({'p': c['p'], 'auc': rnd(c['auc']), 'K': c['K']})
        if not (c['increase'] and c['p'] < B['alpha'] and c['auc'] >= B['minAuc']):
            out.append({**row, 'status': 'noChange'}); continue
        lo_s, hi_s = float(ss[c['first'] - 1]), float(ss[c['last']])
        row.update({'beforeSegmentS': rnd(lo_s), 'afterSegmentS': rnd(hi_s), 'gapM': rnd(hi_s - lo_s)})
        out.append({**row, 'status': 'edge', 'interval': (lo_s, hi_s), 'centerL': (s0 + s1) / 2})
    result = fit_strips(out, at, B['minStrips'], P['uncertainty'], also_at, strip_bootstrap)
    result['lasersWithGain'] = len(gains)
    result['returnsWithoutGain'] = int((~keep).sum())
    return result


# ------------------------------------------------------------------ line fit, agreement, coverage
def interval_line(x, lo, hi, at, slopes):
    """Line e(x) = value + slope*(x - at) with the least total violation of per-strip intervals [lo, hi].

    For each grid slope the L1 violation is convex piecewise linear in `value`, so its minimizers form an
    interval whose ends are interval endpoints. The estimate is the centroid of the optimal (value, slope)
    set (all minimizers within 1e-9 m); `valueRange` is that set's extent at x = at."""
    x = np.asarray(x, float) - at; lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    slopes = np.asarray(slopes, float)
    LO = lo[None, :] - slopes[:, None] * x[None, :]
    HI = hi[None, :] - slopes[:, None] * x[None, :]
    cand = np.concatenate([LO, HI], axis=1)
    cost = (np.maximum(0.0, LO[:, None, :] - cand[:, :, None]) + np.maximum(0.0, cand[:, :, None] - HI[:, None, :])).sum(2)
    per_slope = cost.min(1)
    best = float(per_slope.min())
    rows = np.nonzero(per_slope <= best + 1e-9)[0]
    a_lo = np.array([cand[i][cost[i] <= per_slope[i] + 1e-9].min() for i in rows])
    a_hi = np.array([cand[i][cost[i] <= per_slope[i] + 1e-9].max() for i in rows])
    width = a_hi - a_lo
    weight = width if width.sum() > 1e-12 else np.ones(len(rows))
    value = float(np.sum(weight * (a_lo + a_hi) / 2) / weight.sum())
    slope = float(np.sum(weight * slopes[rows]) / weight.sum())
    return {'value': value, 'slope': slope, 'violationM': best, 'valueRange': [float(a_lo.min()), float(a_hi.max())],
            'slopeRange': [float(slopes[rows].min()), float(slopes[rows].max())],
            'slopeAtGridLimit': bool(rows.min() == 0 or rows.max() == len(slopes) - 1)}


def slope_grid(U):
    limit = math.tan(math.radians(U['maxSlopeDeg']))
    count = int(round(2 * limit / U['slopeStep'])) + 1
    return np.linspace(-limit, limit, count)


def fit_strips(rows, at, min_strips, U, also_at=None, strip_bootstrap=True):
    edges = [r for r in rows if r['status'] == 'edge']
    status_counts = {}
    for r in rows:
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1
    def clean(r):
        out = {}
        for k, v in r.items():
            if k == 'interval':
                out[k] = [rnd(v[0]), rnd(v[1])]
            elif isinstance(v, float) and k not in ('p', 'K'):
                out[k] = rnd(v)
            else:
                out[k] = v
        return out
    result = {'strips': [clean(r) for r in rows], 'stripStatus': status_counts, 'edgeStrips': len(edges)}
    if len(edges) < min_strips:
        result.update({'status': 'unresolved', 'reason': f'{len(edges)} strips with an edge interval; protocol needs {min_strips}'})
        return result
    x = np.array([r['centerL'] for r in edges])
    lo = np.array([r['interval'][0] for r in edges]); hi = np.array([r['interval'][1] for r in edges])
    mid = (lo + hi) / 2
    value, slope = theil_sen(x, mid, at)
    resid = mid - (value + slope * (x - at))
    if not strip_bootstrap:
        result.update({'status': 'measured', 'edgeSAtEval': value, 'slope': slope})
        if also_at is not None:
            result['atCorridorCenter'] = {'L': float(also_at), 'edgeS': value + slope * (also_at - at)}
        return result
    rng = np.random.default_rng(U['seed'])
    boot, boot_also = [], []
    for _ in range(U['bootstrapReps']):
        k = rng.integers(0, len(x), len(x))
        if np.ptp(x[k]) <= 1e-12:
            continue
        bv, bs = theil_sen(x[k], mid[k], at)
        boot.append(bv)
        if also_at is not None:
            boot_also.append(bv + bs * (also_at - at))
    boot = np.asarray(boot)
    sd = float(np.std(boot, ddof=1)) if len(boot) > 1 else None
    l1 = interval_line(x, lo, hi, at, slope_grid(U))
    result.update({
        'status': 'measured', 'edgeSAtEval': value, 'slope': slope, 'angleToMappedEdgeDeg': math.degrees(math.atan(slope)),
        'stripMidpointS': {'min': float(mid.min()), 'median': float(np.median(mid)), 'max': float(mid.max())},
        'residualRobustSigmaM': robust_sigma(resid), 'residualMaxAbsM': float(np.max(np.abs(resid))),
        'stripsWithin0.15mOfLine': int(np.sum(np.abs(resid) <= 0.15)),
        'stripIntervalWidthM': {'median': float(np.median(hi - lo)), 'max': float(np.max(hi - lo))},
        'stripL': {'min': float(x.min()), 'max': float(x.max())},
        'stripBootstrap': None if sd is None else {'sd': sd, 'p2_5': float(np.percentile(boot, 2.5)), 'p97_5': float(np.percentile(boot, 97.5)), 'reps': int(len(boot)),
                                                   'note': 'Resamples lateral strips as if independent (under-dispersed; the v3 gate). Diagnostic only since v4.'},
        'uncertaintyM': None,
        'intervalL1Diagnostic': {'edgeSAtEval': l1['value'], 'slope': l1['slope'], 'violationM': l1['violationM'], 'valueRange': l1['valueRange']}})
    if also_at is not None and len(boot_also) > 1:
        result['atCorridorCenter'] = {'L': float(also_at), 'edgeS': value + slope * (also_at - at),
                                      'stripBootstrapSd': float(np.std(boot_also, ddof=1)), 'note': 'Extrapolation diagnostic, not a gate.'}
    return result


def quick_edge(S, L, I, laser, sweep, frame, P):
    """Both estimators without any bootstrap, and whether they agree within the tolerance."""
    W = P['windows']
    a = estimator_a(S, L, I, laser, frame['corridor'], frame['evalL'], P, W, strip_bootstrap=False)
    b = estimator_b(S, L, I, laser, sweep, frame['corridor'], frame['evalL'], P, W, strip_bootstrap=False)
    va, vb = a.get('edgeSAtEval'), b.get('edgeSAtEval')
    agree = va is not None and vb is not None and abs(va - vb) <= P['agreementToleranceM'] + 1e-12
    return {'A': va, 'B': vb, 'agree': bool(agree), 'mean': (va + vb) / 2 if agree else None}


def jackknife_sd(values):
    """Delete-1 jackknife sd sqrt((n-1)/n * sum (v_i - mean)^2) over the leave-one-out values, or None."""
    v = np.asarray([x for x in values if x is not None], float)
    n = len(v)
    return float(math.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2))) if n > 1 else None


def leave_one_sweep_out(S, L, I, laser, sweep, frame, P):
    """Leave-one-sweep-out runs of both estimators (each run re-derives road references, laser tails and gains
    from the remaining sweeps) and their delete-1 jackknife sds.

    Whole sweeps are left out because neighbouring strips are sampled by the same rings of the same sweeps, so
    their errors are shared (the v3 strip bootstrap treated strips as independent and was under-dispersed). A
    whole-sweep bootstrap with replacement was rejected before the v4 freeze: a resample keeps about 9-10 distinct
    sweeps of 15, which widens the along-road gap at the edge and changes the measurement design (synthetic
    prototype, v4-dev/prototype-uncertainty.json)."""
    ids = [int(x) for x in np.unique(sweep)]
    runs = []
    for k in ids:
        keep = sweep != k
        runs.append({'leftOut': k, **quick_edge(S[keep], L[keep], I[keep], laser[keep], sweep[keep], frame, P)})
    both = [r for r in runs if r['A'] is not None and r['B'] is not None]
    agreeing = [r for r in runs if r['agree']]
    measured = lambda key: sum(r[key] is not None for r in runs) / len(runs) if runs else 0.0
    vals = [r['mean'] for r in agreeing]
    return {'runs': len(runs),
            'A': {'measuredFraction': measured('A'), 'jackknifeSdM': jackknife_sd([r['A'] for r in runs])},
            'B': {'measuredFraction': measured('B'), 'jackknifeSdM': jackknife_sd([r['B'] for r in runs])},
            'mean': {'bothMeasured': len(both), 'jackknifeSdM': jackknife_sd([(r['A'] + r['B']) / 2 for r in both])},
            'agreeing': len(agreeing),
            'agreeingMeanRangeM': [float(min(vals)), float(max(vals))] if vals else None,
            'rows': runs}


def combine(a, b, loo, P):
    """Resolution rule: both estimators measure on all sweeps; in the leave-one-sweep-out runs each estimator
    measures in >= minLeaveOneOutMeasuredFraction of runs and has jackknife sd <= maxEstimatorSdM; |A - B| <=
    agreementToleranceM. A resolved edge is labelled 'marginally resolved' when any leave-one-out run does not
    measure both estimators in agreement, or the jackknife sd of (A+B)/2 exceeds maxEstimatorSdM."""
    tolerance, max_sd = P['agreementToleranceM'], P['maxEstimatorSdM']
    min_fraction = P['uncertainty']['minLeaveOneOutMeasuredFraction']
    if a['status'] != 'measured' or b['status'] != 'measured':
        return {'status': 'unresolved', 'reason': 'An estimator did not measure an edge',
                'estimatorA': a['status'], 'estimatorB': b['status']}
    loose = [name for name in ('A', 'B') if loo[name]['jackknifeSdM'] is None or loo[name]['jackknifeSdM'] > max_sd + 1e-12
             or loo[name]['measuredFraction'] < min_fraction - 1e-12]
    if loose:
        return {'status': 'unresolved', 'reason': f'Estimator {"/".join(loose)} leave-one-sweep-out jackknife sd exceeds {max_sd} m '
                                                  f'or it measures in fewer than {min_fraction:.0%} of leave-one-out runs',
                'estimatorA': a['status'], 'estimatorB': b['status']}
    diff = a['edgeSAtEval'] - b['edgeSAtEval']
    if abs(diff) > tolerance + 1e-12:
        return {'status': 'unresolved', 'reason': f'Estimators differ by {abs(diff):.3f} m > {tolerance} m',
                'differenceAminusB': diff}
    edge = (a['edgeSAtEval'] + b['edgeSAtEval']) / 2
    stat = loo['mean']['jackknifeSdM'] or 0.0
    method = abs(diff) / 2
    marginal = loo['agreeing'] < loo['runs'] or stat > max_sd + 1e-12
    return {'status': 'resolved', 'label': 'marginally resolved' if marginal else 'resolved',
            'paintEdgeS': edge, 'differenceAminusB': diff,
            'uncertainty': {'jackknifeSdOfMeanM': stat, 'estimatorHalfDifferenceM': method,
                            'combinedM': math.sqrt(stat ** 2 + method ** 2),
                            'calibration': 'Not calibrated: on synthetic known edges the 2-sigma coverage of combinedM was 0.87-1.00 '
                                           'per case (v4-dev/prototype-uncertainty.json; validation/synthetic-validation-v4.json).',
                            'note': 'Statistical only: leave-one-sweep-out jackknife sd of (A+B)/2 combined with |A-B|/2. '
                                    'A-B agreement is not independent confirmation (shared returns, frame, windows and line fit). '
                                    'Systematic terms (beam footprint, pose/raster registration, float16 storage, '
                                    'intensity-dependent range bias, worn or partial paint) are not included.'}}


def halves(S, L, I, laser, sweep, frame, P, sweep_order):
    """Even/odd and far/near halves of the selected sweeps (diagnostics, never gates). `sweep_order` lists
    sweep ids far to near."""
    half = len(sweep_order) // 2
    groups = {'evenSweeps': sweep_order[0::2], 'oddSweeps': sweep_order[1::2],
              'farHalf': sweep_order[:half + 1], 'nearHalf': sweep_order[half + 1:]}
    out = {}
    for name, group in groups.items():
        keep = np.isin(sweep, group)
        out[name] = {'sweeps': [int(x) for x in group], **quick_edge(S[keep], L[keep], I[keep], laser[keep], sweep[keep], frame, P)}
    return out


def coverage(S, L, frame, reference, P, stack):
    """Fine-strip ground coverage around the reference edge line and an obstruction test for gaps."""
    C = P['coverage']; Sh = P['shadow']
    lo, hi = frame['corridor']
    value, slope = reference
    strips = partition(lo, hi, C['stripWidthM'], 0.0, C['stripWidthM'] / 2)
    rows, covered = [], 0
    for s0, s1 in strips:
        cl = (s0 + s1) / 2
        ref_s = value + slope * (cl - frame['evalL'])
        m = (L >= s0) & (L < s1) & (np.abs(S - ref_s) <= C['halfWindowM'])
        n = int(m.sum())
        row = {'L': [rnd(s0, 3), rnd(s1, 3)], 'returns': n}
        if n >= C['minReturns']:
            covered += 1; row['state'] = 'covered'
        else:
            target = point_at(frame, cl, ref_s)
            blocked = 0
            for lidar, elev in zip(stack['lidars'], stack['elevated']):
                o = lidar[:2]; d = target - o; length = float(np.linalg.norm(d))
                if length < 1e-6 or len(elev) == 0:
                    continue
                u = (elev - o) @ d / (length * length)
                perp = np.abs((elev - o) @ np.array([-d[1], d[0]]) / length)
                hit = (u * length >= Sh['skipNearLidarM']) & (u * length <= length - Sh['stopBeforeTargetM']) & (perp <= Sh['corridorHalfWidthM'])
                if hit.sum() >= Sh['minObstacleReturns']:
                    blocked += 1
            row['sweepsObstructed'] = blocked
            row['state'] = 'shadowed' if blocked >= Sh['minObstructedFraction'] * len(stack['lidars']) else 'unobserved'
        rows.append(row)
    shadowed = [r for r in rows if r['state'] == 'shadowed']
    return {'stripWidthM': C['stripWidthM'], 'strips': len(rows), 'covered': covered,
            'coveredFraction': covered / len(rows) if rows else None,
            'shadowedStrips': len(shadowed), 'shadowedFraction': len(shadowed) / len(rows) if rows else None,
            'shadowedL': merge_ranges([r['L'] for r in shadowed]),
            'unobservedL': merge_ranges([r['L'] for r in rows if r['state'] == 'unobserved']),
            'rows': rows}


def observed_eval_point(S, L, frame, P):
    """Center of the longest contiguous run of corridor strips with ground returns at the mapped edge.

    Uses return counts only (no intensity), so the point cannot depend on paint. A LiDAR shadow (for example
    behind a stationary vehicle) then does not force the estimators to extrapolate into an unobserved region."""
    C = P['coverage']
    lo, hi = frame['corridor']
    strips = partition(lo, hi, C['stripWidthM'], 0.0, C['stripWidthM'] / 2)
    covered = [int(((L >= a) & (L < b) & (np.abs(S) <= C['halfWindowM'])).sum()) >= C['minReturns'] for a, b in strips]
    best, run = None, None
    for i, ok in enumerate(covered + [False]):
        if ok and run is None:
            run = i
        elif not ok and run is not None:
            if best is None or (i - run) > (best[1] - best[0]):
                best = (run, i)
            run = None
    if best is None:
        return {'evalL': float((lo + hi) / 2), 'coveredRunL': None, 'rule': 'no covered strip; corridor center'}
    a, b = strips[best[0]][0], strips[best[1] - 1][1]
    return {'evalL': float((a + b) / 2), 'coveredRunL': [float(a), float(b)], 'rule': 'center of longest covered run'}


def merge_ranges(ranges):
    out = []
    for a, b in ranges:
        if out and abs(out[-1][1] - a) < 1e-6:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def measure_edge(S, L, I, laser, sweep, frame, P):
    """Both estimators (with the strip-bootstrap diagnostic), the leave-one-sweep-out jackknife when both measure,
    and the resolution rule. Each estimator's uncertaintyM is its leave-one-sweep-out jackknife sd."""
    W = P['windows']
    also = frame.get('corridorCenterL')
    a = estimator_a(S, L, I, laser, frame['corridor'], frame['evalL'], P, W, also)
    b = estimator_b(S, L, I, laser, sweep, frame['corridor'], frame['evalL'], P, W, None, also)
    loo = None
    if a['status'] == 'measured' and b['status'] == 'measured':
        loo = leave_one_sweep_out(S, L, I, laser, sweep, frame, P)
        a['uncertaintyM'] = loo['A']['jackknifeSdM']; b['uncertaintyM'] = loo['B']['jackknifeSdM']
    final = combine(a, b, loo, P)
    if loo is not None:
        final['leaveOneSweepOut'] = loo
    return a, b, final


# ------------------------------------------------------------------ commands
def resolve(base, path):
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def load_inputs(config_path):
    config_bytes = Path(config_path).read_bytes(); config = strict_json(config_bytes)
    require(config.get('schemaVersion') == SCHEMA, 'Config schemaVersion 1 required')
    base = Path(config_path).resolve().parent
    src = config['sources']
    data = {k: read_verified(resolve(base, v['path']), v['sha256']) for k, v in src.items()}
    return config, sha256(config_bytes), base, data


def load_protocol(path, expected=None):
    data = Path(path).read_bytes()
    if expected:
        require(sha256(data) == expected, 'Protocol SHA-256 does not match the lock')
    protocol = strict_json(data)
    require(protocol.get('protocol') == 'pao-paint-edge', 'Not a paint-edge protocol')
    return protocol['parameters'], sha256(data), protocol


def run_selection(config, data, P):
    poses, index = load_poses(data['poses'])
    offset = lidar_offset(data['extrinsics'], P['sweepSelection']['lidarSensor'])
    vector_map = load_map(data['map'])
    edges = crossing_edges(vector_map, config['crossingId'])
    stamps = listing_timestamps(data['lidarListing'], config['logId'])
    return poses, index, offset, vector_map, edges, select_sweeps(stamps, poses, index, offset, edges, P)


def selection_record(selection):
    keep = {k: v for k, v in selection.items() if k != 'chosen'}
    if selection.get('ok'):
        keep['chosen'] = [{'timestampNs': str(c['ts']), 'edgeAheadM': rnd(c['edgeAheadM']),
                           'lidarCityXY': [rnd(v, 3) for v in c['lidar']],
                           'headingDeg': rnd(math.degrees(math.atan2(c['heading'][1], c['heading'][0])), 3)}
                          for c in selection['chosen']]
    return keep


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), f'{path} exists; outputs are never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def cmd_select(args):
    P, protocol_sha, _ = load_protocol(args.protocol, args.protocol_sha256)
    config, config_sha, _, data = load_inputs(args.config)
    *_, selection = run_selection(config, data, P)
    write_new(args.output, {'schemaVersion': SCHEMA, 'kind': 'paint-edge-sweep-selection', 'protocolSha256': protocol_sha,
                            'configSha256': config_sha, 'logId': config['logId'], 'crossingId': str(config['crossingId']),
                            'readLidarContent': False, **selection_record(selection)})


def cmd_measure(args):
    P, protocol_sha, protocol = load_protocol(args.protocol, args.protocol_sha256)
    config, config_sha, base, data = load_inputs(args.config)
    poses, index, offset, vector_map, edges, selection = run_selection(config, data, P)
    require(selection['ok'], selection.get('reason', 'Sweep selection failed'))
    receipts_path = Path(args.sweeps).resolve(); receipts = strict_json(receipts_path.read_bytes())
    by_ts = {str(r['timestampNs']): r for r in receipts['files']}
    sweeps, sweep_inputs = [], []
    for c in selection['chosen']:
        r = by_ts.get(str(c['ts']))
        require(r is not None, f'No verified download for sweep {c["ts"]}')
        raw = read_verified(resolve(receipts_path.parent, r['path']), r['sha256'])
        table = feather.read_table(pa_buffer(raw)).to_pydict()
        for key in ('x', 'y', 'z', 'intensity', 'laser_number'):
            require(key in table, f'Sweep missing {key}')
        sweeps.append((c['ts'], table)); sweep_inputs.append({'timestampNs': str(c['ts']), 'sha256': r['sha256'], 'bytes': len(raw)})
    raster = ground_raster(data['groundRaster']); sim2 = strict_json(data['groundTransform'])
    frame = build_frame(selection, edges, vector_map, P)
    stack = stack_sweeps(sweeps, poses, index, offset, raster, sim2, frame, P)
    lo, hi = frame['corridor']
    inside = (stack['L'] >= lo) & (stack['L'] < hi)
    S, Sf, L, I, laser, k = (stack[x][inside] for x in ('S', 'Sf', 'L', 'I', 'laser', 'sweep'))
    frame['corridorCenterL'] = frame['evalL']
    if P.get('evaluationPoint') == 'centerOfLongestCoveredRun':
        point = observed_eval_point(S, L, frame, P)
        frame['evalL'] = point['evalL']; frame['evalRule'] = point
    a, b, final = measure_edge(S, L, I, laser, k, frame, P)
    if final['status'] == 'resolved':
        reference = (final['paintEdgeS'], 0.5 * (a['slope'] + b['slope']))
    else:
        reference = (0.0, 0.0)
    cov = coverage(S, L, frame, reference, P, stack)
    # diagnostics with the same frozen parameters (never gates; no bootstrap)
    split = halves(S, L, I, laser, k, frame, P, list(range(len(sweeps))))
    fa_ = estimator_a(Sf, L, I, laser, frame['corridor'], frame['evalL'], P, P['windows'], strip_bootstrap=False)
    fb_ = estimator_b(Sf, L, I, laser, k, frame['corridor'], frame['evalL'], P, P['windows'], strip_bootstrap=False)
    fva, fvb = fa_.get('edgeSAtEval'), fb_.get('edgeSAtEval')
    fagree = fva is not None and fvb is not None and abs(fva - fvb) <= P['agreementToleranceM'] + 1e-12
    fq = {'agree': bool(fagree), 'mean': (fva + fvb) / 2 if fagree else None}
    W = P['windows']
    n_window = int(((S >= W['searchS'][0]) & (S < W['searchS'][1])).sum())
    n_ref = int(((S >= W['roadReferenceS'][0]) & (S < W['roadReferenceS'][1])).sum())
    result = {
        'schemaVersion': SCHEMA, 'kind': 'paint-edge-measurement', 'protocolSha256': protocol_sha,
        'protocolVersion': protocol.get('version'), 'configSha256': config_sha,
        'scriptSha256': sha256(Path(__file__).read_bytes()),
        'logId': config['logId'], 'crossingId': str(config['crossingId']), 'split': config.get('split'),
        'inputs': {'sources': {k: v['sha256'] for k, v in config['sources'].items()}, 'sweeps': sweep_inputs},
        'selection': selection_record(selection),
        'frame': {'travelDirectionCityXY': [rnd(v, 6) for v in frame['t']], 'lateralOriginCityXY': [rnd(v, 3) for v in frame['origin']],
                  'corridorL': [rnd(v, 3) for v in frame['corridor']], 'evalL': rnd(frame['evalL'], 3),
                  'corridorCenterL': rnd(frame['corridorCenterL'], 3), 'evalRule': frame.get('evalRule', 'corridor center'),
                  'mappedFarEdgeSAtCorridorEnds': [rnd(v, 3) for v in frame['mappedFarEdgeSAtCorridorEnds']],
                  **{k: v for k, v in frame['corridorSources'].items()}},
        'returns': {'groundInCorridorAllSweeps': int(len(S)), 'searchWindow': n_window, 'roadReference': n_ref,
                    'perSweep': stack['perSweep']},
        'estimatorA': a, 'estimatorB': b, 'nearEdge': final, 'label': final.get('label'),
        'offsetToMappedEdgeM': final.get('paintEdgeS') if final['status'] == 'resolved' else None,
        'coverage': cov,
        'diagnostics': {
            'halves': split,
            'farEdge': {'note': 'Same frozen parameters mirrored from beyond the mapped far edge; S measured back toward the '
                                'approaching driver from the mapped far edge (positive = paint ends before the mapped far edge). '
                                'Diagnostic only; not a gate; no bootstrap.',
                        'estimatorA': {kk: fa_.get(kk) for kk in ('status', 'edgeSAtEval', 'slope', 'edgeStrips', 'stripStatus', 'reason')},
                        'estimatorB': {kk: fb_.get(kk) for kk in ('status', 'edgeSAtEval', 'slope', 'edgeStrips', 'stripStatus', 'reason')},
                        'agreement': fq}},
        'limits': protocol.get('limits', []),
    }
    write_new(args.output, json.loads(json.dumps(result, allow_nan=False, default=_json_default)))


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('select', 'measure'):
        p = sub.add_parser(name)
        p.add_argument('--protocol', required=True)
        p.add_argument('--protocol-sha256', required=True, help='Locked SHA-256 of the protocol file')
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        if name == 'measure':
            p.add_argument('--sweeps', required=True, help='Download receipts JSON with per-sweep path and SHA-256')
    args = parser.parse_args(argv)
    (cmd_select if args.command == 'select' else cmd_measure)(args)


if __name__ == '__main__':
    main()

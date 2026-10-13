"""Metric F (false Blocked caused by the box shape): LiDAR car-top profile of the tested stationary vehicle, and scoring
of the app's setback-sweep rows against it. Report only; it changes no app result. Results are Blocked or Unknown; there
is no Clear.

  python3 -I -B metric_f_car_top.py profile --protocol P --protocol-lock K --ground G --suggestions S --scene NAME
      --annotations A --poses C --calibration E --sweep-dir D [--sweep-dir D ...] [--receipt R ...] --out OUT
  python3 -I -B metric_f_car_top.py score --protocol P --protocol-lock K --profile F --rows R --scene NAME --out OUT
      [--same-classes-as EARLIER_RESULTS]
  python3 -I -B metric_f_car_top.py summarize --protocol P --protocol-lock K --results F [F ...] --out OUT
  python3 -I -B metric_f_car_top.py receipt --protocol P --protocol-lock K --commands FILE --hash FILE [FILE ...] --out OUT

Every path comes from the arguments. Thresholds come from the locked protocol file (its sha256 must equal the lock).

Car-local frame (the tested box, upright like the app's box): x forward along the box heading (front face at +L/2),
y left, z height above the box bottom (world Y minus box centre Y plus H/2). The sweep loader (pose interpolation and
laser-to-sensor assignment check) is copied from the project's metric K script (LiDAR shadow, v3.1 analysis).

Protocol v2 (protocol-v2.json) adds report-only evidence and changes no class: per 2D cell, the near-ground returns and
the LiDAR rays that passed through the cell and hit something beyond it ("free rays"); per row, the crossed 2D cells with
that evidence and where the line enters and leaves the footprint; per sweep, where the ego was; and the app's last plain
Blocked row against the measured stop. A protocol without freeRayBandM (v1) gets none of these.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

SUBCOMMANDS = ('profile', 'score', 'summarize', 'receipt')


# ----------------------------------------------------------------------------------------------- shared helpers
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(protocol_path, lock_path):
    """The locked protocol: its sha256 must equal the lock file's, else stop."""
    raw = Path(protocol_path).read_bytes(); lock = json.loads(Path(lock_path).read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == lock['sha256'], 'protocol.json changed after its lock'
    return json.loads(raw), digest


def write_new(path, doc):
    """Write JSON to a new file only (never overwrite); the text must not contain the refused word."""
    text = json.dumps(doc, indent=1, sort_keys=True) + '\n'
    assert 'park' not in text.lower(), 'Output text contains a refused word'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    return hashlib.sha256(text.encode()).hexdigest()


def rnd(v, d=4):
    return None if v is None else round(float(v), d)


# ------------------------------------------------- copied from the metric K script (LiDAR shadow, v3.1 analysis)
def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotate(q, v):
    """Rotate vectors v by unit quaternions q = (w, x, y, z), row by row: v + 2w(u x v) + 2u x (u x v)."""
    w, u = q[:, :1], q[:, 1:]; v = np.broadcast_to(np.asarray(v, float), u.shape); uv = np.cross(u, v)
    return v + 2 * w * uv + 2 * np.cross(u, uv)


def capture_poses(times, poses):
    """Ego pose (city_SE3_egovehicle) at each time, interpolated linearly between the two nearest poses (rotation by
    normalized linear quaternion interpolation). Returns (quaternions w,x,y,z; translations)."""
    t = poses['timestamp_ns'].to_numpy(np.int64); order = np.argsort(t); t = t[order]
    q = poses[['qw', 'qx', 'qy', 'qz']].to_numpy(float)[order]; p = poses[['tx_m', 'ty_m', 'tz_m']].to_numpy(float)[order]
    times = np.asarray(times, np.int64)
    assert (times >= t[0]).all() and (times <= t[-1]).all(), 'Return time outside the pose record'
    i = np.clip(np.searchsorted(t, times, side='right') - 1, 0, len(t) - 2)
    a = ((times - t[i]) / (t[i + 1] - t[i]))[:, None]
    q0, q1 = q[i], q[i + 1]; q1 = np.where((q0 * q1).sum(1, keepdims=True) < 0, -q1, q1)
    qi = (1 - a) * q0 + a * q1; qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    return qi, (1 - a) * p[i] + a * p[i + 1]


def world(city, origin):
    return np.column_stack([city[:, 0] - origin[0], city[:, 2] - origin[2], -(city[:, 1] - origin[1])])


# ---------------------------------------------------------------------------------------------- car-local frame
def upright_frame(box):
    """The tested box as the app tests it: cuboid centre, horizontal heading from worldFromBox column 0 (pitch and roll
    dropped), unrounded length, width and height."""
    m = np.asarray(box['worldFromBox'], float); n = math.hypot(m[0][0], m[2][0])
    heading = np.array([m[0][0] / n, 0., m[2][0] / n])
    return {'center': np.asarray(box['center'], float), 'heading': heading, 'left': np.array([heading[2], 0., -heading[0]]),
            'L': float(box['length']), 'W': float(box['width']), 'H': float(box['height'])}


def car_local(points, frame):
    """World points to car-local [forward, left, height above the box bottom]."""
    d = np.asarray(points, float) - frame['center']
    return np.column_stack([d @ frame['heading'], d @ frame['left'], d[:, 1] + frame['H'] / 2])


# ----------------------------------------------------------------------------------------------------- profile
def along_bins(length, nominal):
    """n = round(L / nominal) equal bins numbered from the front face; returns (n, width, [(x_front, x_rear)])."""
    n = max(1, int(round(length / nominal))); w = length / n
    return n, w, [(length / 2 - b * w, length / 2 - (b + 1) * w if b < n - 1 else -length / 2) for b in range(n)]


def lateral_bins(width, nominal):
    """m = round(W / nominal) equal bins numbered from the left side; returns (m, width, [(y_left, y_right)])."""
    m = max(1, int(round(width / nominal))); w = width / m
    return m, w, [(width / 2 - j * w, width / 2 - (j + 1) * w if j < m - 1 else -width / 2) for j in range(m)]


def profile_from_points(local, sweep, dims, th, other_inside=None, other_grown=None, limit_top=True, remove_other=True):
    """Car-top profile from car-local returns. local: (N, 3) [x, y, z]; sweep: (N,) sweep index per return; dims: (L, W, H);
    th: protocol thresholds; other_inside / other_grown: returns inside another annotated cuboid of the same sweep
    (no growth / grown by the buffer). Returns along-bins (primary), 2D cells (secondary) and the end overhang."""
    local = np.asarray(local, float).reshape(-1, 3); sweep = np.asarray(sweep).reshape(-1)
    L, W, H = dims; x, y, z = local[:, 0], local[:, 1], local[:, 2]
    none = np.zeros(len(local), bool)
    other_inside = none if other_inside is None or not remove_other else np.asarray(other_inside, bool)
    other_grown = none if other_grown is None or not remove_other else np.asarray(other_grown, bool)
    lateral = np.abs(y) <= W / 2; high_enough = z >= th['minHeightAboveBoxBottomM']
    capped = z <= H + th['maxHeightAboveBoxTopM'] if limit_top else np.ones(len(local), bool)
    footprint = lateral & high_enough & (x >= -L / 2) & (x <= L / 2)
    removed = footprint & other_inside; above = footprint & ~capped & ~removed
    kept = footprint & capped & ~removed
    n, bw, edges = along_bins(L, th['binLengthNominalM']); m, cw, lat_edges = lateral_bins(W, th['lateralBinNominalM'])
    b = np.clip(np.floor((L / 2 - x) / bw).astype(int), 0, n - 1); j = np.clip(np.floor((W / 2 - y) / cw).astype(int), 0, m - 1)
    need = th['minReturnsMeasured']
    along = []
    for k in range(n):
        sel = kept & (b == k); count = int(sel.sum())
        along.append({'bin': k, 'xFront': edges[k][0], 'xRear': edges[k][1], 'count': count, 'sweeps': int(len(np.unique(sweep[sel]))),
                      'topZ': float(z[sel].max()) if count else None, 'measured': count >= need,
                      'removedOtherObject': int((removed & (b == k)).sum()), 'aboveTopLimit': int((above & (b == k)).sum())})
    cells = []
    for k in range(n):
        for jj in range(m):
            sel = kept & (b == k) & (j == jj); count = int(sel.sum())
            cells.append({'bin': k, 'lateral': jj, 'xFront': edges[k][0], 'xRear': edges[k][1], 'yLeft': lat_edges[jj][0], 'yRight': lat_edges[jj][1],
                          'count': count, 'topZ': float(z[sel].max()) if count else None, 'measured': count >= need})
    reach, step = th['overhangReachM'], th['overhangBinM']
    beyond_ok = lateral & high_enough & capped & ~other_grown
    overhang = {}
    for end, dist in (('front', x - L / 2), ('rear', -L / 2 - x)):
        sel_end = beyond_ok & (dist > 0) & (dist <= reach)
        steps = int(math.ceil(reach / step - 1e-9)); out = []
        for k in range(steps):
            sel = sel_end & (dist > k * step) & (dist <= (k + 1) * step); count = int(sel.sum())
            out.append({'fromM': k * step, 'toM': (k + 1) * step, 'count': count, 'sweeps': int(len(np.unique(sweep[sel]))),
                        'topZ': float(z[sel].max()) if count else None})
        within = sel_end & (dist <= th['overhangSummaryReachM'])
        overhang[end] = {'bins': out, 'returnsWithinSummaryReach': int(within.sum()), 'sweepsWithinSummaryReach': int(len(np.unique(sweep[within]))),
                         'removedOtherObjectGrown': int((lateral & high_enough & capped & other_grown & (dist > 0) & (dist <= reach)).sum())}
    return {'alongBins': {'n': n, 'widthM': bw, 'bins': along}, 'cells': {'lateralN': m, 'lateralWidthM': cw, 'cells': cells},
            'overhang': overhang, 'counts': {'footprint': int(footprint.sum()), 'kept': int(kept.sum()), 'removedOtherObject': int(removed.sum()),
                                             'aboveTopLimit': int(above.sum())}, 'limitTop': bool(limit_top), 'removeOther': bool(remove_other)}


# ------------------------------------------------------------------ protocol v2: free-space evidence (report only)
def clip_many(S, P, lo, hi):
    """clip_segment for many segments S -> P at once (x, y only): parameter ranges (a0, a1); a0 >= a1 where outside."""
    S, P = np.asarray(S, float).reshape(-1, 3), np.asarray(P, float).reshape(-1, 3)
    a0, a1 = np.zeros(len(S)), np.ones(len(S))
    for k in (0, 1):
        d = P[:, k] - S[:, k]; flat = np.abs(d) < 1e-15; out = (S[:, k] < lo[k]) | (S[:, k] > hi[k])
        with np.errstate(divide='ignore', invalid='ignore'):
            ta, tb = (lo[k] - S[:, k]) / d, (hi[k] - S[:, k]) / d
        a0 = np.maximum(a0, np.where(flat, np.where(out, np.inf, -np.inf), np.minimum(ta, tb)))
        a1 = np.minimum(a1, np.where(flat, np.where(out, -np.inf, np.inf), np.maximum(ta, tb)))
    return a0, a1


def free_ray_tops(S, P, x_rear, x_front, y_right, y_left, z_lo, z_hi, min_len):
    """Rays (sensor S -> return P, car-local) that pass through one column x_rear..x_front, y_right..y_left: more than
    min_len horizontally inside it, the return more than min_len beyond it, and every in-column height within z_lo..z_hi.
    Returns each such ray's highest in-column height."""
    S, P = np.asarray(S, float).reshape(-1, 3), np.asarray(P, float).reshape(-1, 3)
    a0, a1 = clip_many(S, P, (x_rear, y_right), (x_front, y_left))
    hlen = np.hypot(P[:, 0] - S[:, 0], P[:, 1] - S[:, 1])
    with np.errstate(invalid='ignore'):
        ok = ((a1 - a0) * hlen > min_len) & ((1 - a1) * hlen > min_len)
    z0, z1 = S[:, 2] + a0 * (P[:, 2] - S[:, 2]), S[:, 2] + a1 * (P[:, 2] - S[:, 2])
    with np.errstate(invalid='ignore'):
        ok &= (np.minimum(z0, z1) >= z_lo) & (np.maximum(z0, z1) <= z_hi)
    return np.maximum(z0, z1)[ok]


def cell_evidence(local, other_inside, rays, dims, th):
    """Per 2D cell (same cells as the secondary profile): near-ground returns (nearGroundFloorM <= z < minHeight, not
    inside another cuboid) and free rays (free_ray_tops, heights minHeight .. H + maxHeightAboveBoxTop) with a histogram
    of their highest in-cell height in freeRayBandM bands from minHeight. rays: (S, P) car-local arrays."""
    local = np.asarray(local, float).reshape(-1, 3); L, W, H = dims; x, y, z = local[:, 0], local[:, 1], local[:, 2]
    other = np.zeros(len(local), bool) if other_inside is None else np.asarray(other_inside, bool)
    n, bw, edges = along_bins(L, th['binLengthNominalM']); m, cw, lat = lateral_bins(W, th['lateralBinNominalM'])
    z_lo, z_hi, band, min_len = th['minHeightAboveBoxBottomM'], H + th['maxHeightAboveBoxTopM'], th['freeRayBandM'], th['crossedBinMinLengthM']
    nb = int(math.ceil((z_hi - z_lo) / band - 1e-9))
    ground = (np.abs(y) <= W / 2) & (x >= -L / 2) & (x <= L / 2) & (z >= th['nearGroundFloorM']) & (z < z_lo) & ~other
    b = np.clip(np.floor((L / 2 - x) / bw).astype(int), 0, n - 1); j = np.clip(np.floor((W / 2 - y) / cw).astype(int), 0, m - 1)
    S, P = (np.asarray(a, float).reshape(-1, 3) for a in rays)
    a0, a1 = clip_many(S, P, (-L / 2, -W / 2), (L / 2, W / 2))
    with np.errstate(invalid='ignore'):
        over = (a1 - a0) * np.hypot(P[:, 0] - S[:, 0], P[:, 1] - S[:, 1]) > min_len
    S, P = S[over], P[over]
    cells = []
    for k in range(n):
        for jj in range(m):
            tops = free_ray_tops(S, P, edges[k][1], edges[k][0], lat[jj][1], lat[jj][0], z_lo, z_hi, min_len)
            hist = np.bincount(np.clip(np.floor((tops - z_lo) / band).astype(int), 0, nb - 1), minlength=nb)
            cells.append({'bin': k, 'lateral': jj, 'belowMinHeight': int((ground & (b == k) & (j == jj)).sum()),
                          'freeRays': int(len(tops)), 'freeRayTopHistogram': hist.tolist()})
    return {'fromM': z_lo, 'toM': z_hi, 'bandM': band, 'raysOverFootprint': int(len(S)), 'cells': cells}


def free_rays_below(cell, evidence, line_z, margin):
    """Free rays of one cell whose highest in-cell height is at least margin below line_z (whole bands only, so it
    never counts a ray that may reach above line_z - margin)."""
    lo, band = evidence['fromM'], evidence['bandM']
    return int(sum(c for k, c in enumerate(cell['freeRayTopHistogram']) if lo + band * (k + 1) <= line_z - margin + 1e-12))


def footprint_entry_exit(e, t, dims):
    """Where the segment enters and leaves the footprint (car-local x, y, z and the face), or None."""
    L, W, _ = dims; r = clip_segment(e, t, (-L / 2, -W / 2), (L / 2, W / 2))
    if r is None:
        return None

    def at(s):
        p = [e[k] + s * (t[k] - e[k]) for k in range(3)]
        if s in (0., 1.):
            face = 'inside'
        else:
            gaps = {'front': abs(p[0] - L / 2), 'rear': abs(p[0] + L / 2), 'left': abs(p[1] - W / 2), 'right': abs(p[1] + W / 2)}
            face = min(gaps, key=gaps.get)
        return {'x': rnd(p[0]), 'y': rnd(p[1]), 'z': rnd(p[2]), 'face': face}
    return {'entry': at(r[0]), 'exit': at(r[1])}


# ------------------------------------------------------------------------------------------------------ scoring
def clip_segment(e, t, lo, hi, s0=0., s1=1.):
    """Parameter range [s0, s1] of segment e -> t (x, y only) inside lo <= (x, y) <= hi; None when outside."""
    for k in (0, 1):
        d = t[k] - e[k]
        if abs(d) < 1e-15:
            if e[k] < lo[k] or e[k] > hi[k]:
                return None
            continue
        a, b = (lo[k] - e[k]) / d, (hi[k] - e[k]) / d
        s0, s1 = max(s0, min(a, b)), min(s1, max(a, b))
        if s0 > s1:
            return None
    return s0, s1


def crossed(e, t, x_rear, x_front, y_right, y_left, min_len):
    """The segment's lowest height inside one footprint bin, or None when it runs at most min_len there horizontally."""
    r = clip_segment(e, t, (x_rear, y_right), (x_front, y_left))
    if r is None:
        return None
    s0, s1 = r
    if (s1 - s0) * math.hypot(t[0] - e[0], t[1] - e[1]) <= min_len:
        return None
    z = lambda s: e[2] + s * (t[2] - e[2])
    return min(z(s0), z(s1))


def classify(bins, margin):
    """bins: crossed bins [{measured, clearance}]. Precedence: supported, unresolved, likelyFalse or ambiguous."""
    if not bins:
        return 'noFootprintCrossing'
    measured = [b for b in bins if b['measured']]
    if any(b['clearance'] < -margin for b in measured):
        return 'supported'
    if len(measured) < len(bins):
        return 'unresolved'
    return 'likelyFalse' if min(b['clearance'] for b in measured) >= margin else 'ambiguous'


def line_against(e, t, profile, dims, th, cells=False):
    """Crossed bins (or 2D cells) of one sightline with their line height, top and clearance."""
    L, W, _ = dims; min_len = th['crossedBinMinLengthM']; out = []
    if not cells:
        for b in profile['alongBins']['bins']:
            z = crossed(e, t, b['xRear'], b['xFront'], -W / 2, W / 2, min_len)
            if z is not None:
                out.append({'bin': b['bin'], 'lineZ': z, 'topZ': b['topZ'], 'measured': b['measured'], 'count': b['count'],
                            'clearance': z - b['topZ'] if b['measured'] else None})
    else:
        for c in profile['cells']['cells']:
            z = crossed(e, t, c['xRear'], c['xFront'], c['yRight'], c['yLeft'], min_len)
            if z is not None:
                out.append({'bin': c['bin'], 'lateral': c['lateral'], 'lineZ': z, 'topZ': c['topZ'], 'measured': c['measured'],
                            'clearance': z - c['topZ'] if c['measured'] else None})
    return out


def score_row(row, profile, dims, th, sensitivity=None):
    e, t = row['eyeCar'], row['targetCar']; margin = th['marginM']
    bins = line_against(e, t, profile, dims, th); cells = line_against(e, t, profile, dims, th, cells=True)
    measured = [b for b in bins if b['measured']]
    low = min(measured, key=lambda b: b['clearance']) if measured else None
    out = {'setbackM': row['setbackM'], 'state': row['state'], 'gapM': row['gapM'], 'class': classify(bins, margin),
           'minClearanceM': low['clearance'] if low else None, 'minClearanceBin': low['bin'] if low else None,
           'crossedBins': len(bins), 'unmeasuredCrossedBins': len(bins) - len(measured),
           'measuredContact': any(b['clearance'] < 0 for b in measured),
           'bins': [{'bin': b['bin'], 'lineZ': rnd(b['lineZ']), 'topZ': rnd(b['topZ']), 'count': b['count'],
                     'clearance': rnd(b['clearance'])} for b in bins],
           'class2D': classify(cells, margin),
           'minClearance2DM': min((c['clearance'] for c in cells if c['measured']), default=None),
           'mappedGroundMinusBoxBottomM': row.get('mappedGroundMinusBoxBottomM')}
    if row.get('reason'):
        out['reason'] = row['reason']
    if sensitivity is not None:
        out['classSensitivity'] = classify(line_against(e, t, sensitivity, dims, th), margin)
    evidence = profile.get('cellEvidence')
    if evidence is not None:
        # Protocol v2, report only: the crossed 2D cells with their returns and free rays, and the footprint entry and exit.
        m = profile['cells']['lateralN']; by_cell = profile['cells']['cells']; ev = evidence['cells']
        out['cells2D'] = [{'bin': c['bin'], 'lateral': c['lateral'], 'count': by_cell[c['bin'] * m + c['lateral']]['count'],
                           'topZ': rnd(c['topZ']), 'lineZ': rnd(c['lineZ']), 'clearance': rnd(c['clearance']),
                           'belowMinHeight': ev[c['bin'] * m + c['lateral']]['belowMinHeight'],
                           'freeRays': ev[c['bin'] * m + c['lateral']]['freeRays'],
                           'freeRaysBelowLine': free_rays_below(ev[c['bin'] * m + c['lateral']], evidence, c['lineZ'], margin)} for c in cells]
        out['footprint'] = footprint_entry_exit(e, t, dims)
    return out


def plain_blocked(row):
    return row['state'] == 'blocked'


def stop_blocking(scored, app):
    """Measured stop-blocking setback (first non-overlap row after the last row with measured contact) against the app."""
    usable = [r for r in scored if r['state'] != 'overlap']
    contact = [i for i, r in enumerate(usable) if r['measuredContact']]
    blocked = [r['setbackM'] for r in usable if plain_blocked(r)]
    out = {'measuredFirstContactM': usable[contact[0]]['setbackM'] if contact else None,
           'measuredLastContactM': usable[contact[-1]]['setbackM'] if contact else None,
           'measuredStopBlockingM': usable[contact[-1] + 1]['setbackM'] if contact and contact[-1] + 1 < len(usable) else None,
           'appFirstPlainBlockedM': blocked[0] if blocked else None, 'appLastPlainBlockedM': blocked[-1] if blocked else None,
           'appStopsBlockingAtM': app.get('stopsBlockingAt'), 'appNotBlockedFromM': app.get('notBlockedFrom')}
    after = usable[contact[-1] + 1:] if contact else usable
    out['rowsAfterMeasuredStopTouchingBoxWithUnmeasuredBin'] = sum(1 for r in after if r['gapM'] <= 0 and r['unmeasuredCrossedBins'] > 0)
    a, mstop, last = out['appStopsBlockingAtM'], out['measuredStopBlockingM'], out['appLastPlainBlockedM']
    out['measuredMinusAppStopM'] = None if a is None or mstop is None else mstop - a
    # Protocol v2 (report only): positive = the app is later than the measured stop. stopsBlockingAt also counts the
    # too-close-to-call rows (Unknown), so the plain-Blocked comparison is the last plain Blocked row.
    out['appStopsBlockingAtMinusMeasuredStopM'] = None if a is None or mstop is None else a - mstop
    out['appLastPlainBlockedMinusMeasuredStopM'] = None if last is None or mstop is None else last - mstop
    return out


CLASSES = ('supported', 'likelyFalse', 'ambiguous', 'unresolved', 'noFootprintCrossing')


def tally(scored, key='class'):
    blocked = [r for r in scored if plain_blocked(r)]; close = [r for r in scored if r.get('reason') == 'too-close-to-call']
    out = {'rows': len(scored), 'plainBlockedRows': len(blocked), 'tooCloseToCallRows': len(close),
           'overlapRows': sum(1 for r in scored if r['state'] == 'overlap'),
           'plainBlocked': {c: sum(1 for r in blocked if r[key] == c) for c in CLASSES},
           'plainBlockedSetbacksM': {c: [rnd(r['setbackM']) for r in blocked if r[key] == c] for c in ('likelyFalse', 'ambiguous', 'unresolved')},
           'tooCloseToCall': {c: sum(1 for r in close if r[key] == c) for c in CLASSES}}
    return out


def score_scene(rows_doc, profile_doc, th):
    dims = (profile_doc['box']['L'], profile_doc['box']['W'], profile_doc['box']['H'])
    primary, sens = profile_doc['primary'], profile_doc['sensitivity']
    out = {}
    for name, h in rows_doc['heights'].items():
        if 'error' in h:
            out[name] = {'targetHeightM': h['targetHeightM'], 'error': h['error']}
            continue
        scored = [score_row(r, primary, dims, th, sens) for r in h['rows']]
        out[name] = {'targetHeightM': h['targetHeightM'], 'counts': tally(scored), 'counts2D': tally(scored, 'class2D'),
                     'countsSensitivity': tally(scored, 'classSensitivity'), 'stopBlocking': stop_blocking(scored, h),
                     'at15ft': h.get('at15ft'), 'at20ft': h.get('at20ft'), 'rows': scored}
    return out


def check_same_classes(heights, earlier):
    """Protocol v2 adds report-only fields; every row must keep the class it had in the earlier results (primary, 2D and
    sensitivity, same setback, same smallest clearance), else stop."""
    old = earlier['heights']; problems = []
    if set(old) != set(heights):
        problems.append(f'heights {sorted(heights)} vs {sorted(old)}')
    for name in sorted(set(old) & set(heights)):
        a, b = heights[name].get('rows', []), old[name].get('rows', [])
        if len(a) != len(b):
            problems.append(f'{name}: {len(a)} rows vs {len(b)}'); continue
        for i, (r, s) in enumerate(zip(a, b)):
            for k in ('state', 'class', 'class2D', 'classSensitivity'):
                if r.get(k) != s.get(k):
                    problems.append(f'{name} row {i}: {k} {r.get(k)} vs {s.get(k)}')
            if abs(r['setbackM'] - s['setbackM']) > 1e-12:
                problems.append(f'{name} row {i}: setback {r["setbackM"]} vs {s["setbackM"]}')
            c, d = r.get('minClearanceM'), s.get('minClearanceM')
            if (c is None) != (d is None) or (c is not None and abs(c - d) > 1e-12):
                problems.append(f'{name} row {i}: smallest clearance {c} vs {d}')
    assert not problems, 'Classes differ from the earlier results:\n' + '\n'.join(problems[:20])
    return {'file': None, 'rowsCompared': sum(len(h.get('rows', [])) for h in heights.values()), 'same': True}


def check_frames(rows_doc, profile_doc):
    """The Node rows' tested box must be the profile's box: same centre (1e-9 m), heading (1e-12) and sizes (1e-9 m)."""
    box, tb = profile_doc['box'], rows_doc['testedBox']
    assert np.allclose(tb['center'], box['center'], atol=1e-9, rtol=0), 'Tested box centre differs from the profile box'
    assert np.allclose(tb['heading'], box['heading'], atol=1e-12, rtol=0), 'Tested box heading differs from the profile box'
    v = tb['vehicle']
    assert max(abs(v['length'] - box['L']), abs(v['width'] - box['W']), abs(v['height'] - box['H'])) <= 1e-9, 'Tested box sizes differ'
    assert rows_doc['track'] == box['track'], 'Tested track differs'


# --------------------------------------------------------------------------------------------- sweep loading
def receipt_index(paths):
    """basename -> set of sha256 values from every receipt record that names a lidar sweep file."""
    index = {}

    def walk(node):
        if isinstance(node, dict):
            digest = node.get('sha256')
            if isinstance(digest, str) and len(digest) == 64:
                for k in ('path', 'key', 'url', 'file'):
                    v = node.get(k)
                    if isinstance(v, str) and v.endswith('.feather') and 'lidar' in v:
                        index.setdefault((v.rsplit('/', 1)[-1], node.get('log') or next((p for p in v.split('/') if len(p) == 36 and p.count('-') == 4), None)), set()).add(digest)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    for p in paths:
        walk(json.loads(Path(p).read_bytes()))
    return index


def gather_sweeps(sweep_dirs, receipts, timestamps, log_id):
    """Sweep files whose timestamp has an annotation row for the tested track; de-duplicated by timestamp."""
    index = receipt_index(receipts); found = {}
    for d in sweep_dirs:
        for f in sorted(Path(d).glob('*.feather')):
            if not f.stem.isdigit():
                continue
            found.setdefault(int(f.stem), []).append(f)
    sweeps, skipped = [], []
    for ts in sorted(found):
        files = found[ts]; digests = {sha(f) for f in files}
        assert len(digests) == 1, f'Copies of sweep {ts} differ'
        digest = digests.pop()
        named = set()
        for (name, log), values in index.items():
            if name == f'{ts}.feather' and log in (None, log_id):
                named |= values
        assert not named or digest in named, f'Sweep {ts} does not match its receipt'
        if ts not in timestamps:
            skipped.append({'timestampNs': str(ts), 'reason': 'no annotation row for the tested track at this timestamp'})
            continue
        sweeps.append({'timestampNs': ts, 'path': files[0], 'copies': len(files), 'sha256': digest, 'receiptVerified': bool(named)})
    return sweeps, skipped


def inside_cuboids(ego_points, rows, grow):
    """Ego-frame points inside any annotation cuboid of the same sweep, each grown by grow on every side."""
    inside = np.zeros(len(ego_points), bool)
    for r in rows.itertuples():
        rot = quat_to_matrix([r.qw, r.qx, r.qy, r.qz]); p = (ego_points - np.array([r.tx_m, r.ty_m, r.tz_m])) @ rot
        inside |= (np.abs(p) <= np.array([r.length_m, r.width_m, r.height_m]) / 2 + grow).all(1)
    return inside


def load_returns(args, frame, origin, track, th, rays=False):
    """rays (protocol v2): also return, for every return of a sweep whose ray crosses the footprint horizontally, its
    sensor origin at capture time and the return, both car-local."""
    import pyarrow.feather as feather
    ann = feather.read_table(args.annotations).to_pandas(); poses = feather.read_table(args.poses).to_pandas()
    calib = feather.read_table(args.calibration).to_pandas().set_index('sensor_name')
    mine = ann[ann.track_uuid == track]; timestamps = set(int(t) for t in mine.timestamp_ns)
    sweeps, skipped = gather_sweeps(args.sweep_dir, args.receipt or [], timestamps, args.log_id)
    assert sweeps, 'No sweep of the tested track is available'
    L, W, H = frame['L'], frame['W'], frame['H']; reach = th['overhangReachM']
    sensors = {}
    for name in ('up_lidar', 'down_lidar'):
        r = calib.loc[name]; sensors[name] = (np.array([r.tx_m, r.ty_m, r.tz_m]), quat_to_matrix([r.qw, r.qx, r.qy, r.qz]))
    local_all, sweep_all, inside_all, grown_all, per_sweep, largest_spread = [], [], [], [], [], 0.
    ray_s, ray_p = [], []
    for k, s in enumerate(sweeps):
        table = feather.read_table(s['path'], columns=['x', 'y', 'z', 'laser_number', 'offset_ns']).to_pandas()
        ego = table[['x', 'y', 'z']].to_numpy(np.float64); laser = table['laser_number'].to_numpy(); offset = table['offset_ns'].to_numpy(np.int64)
        ts = s['timestampNs']; q0, t0 = capture_poses([ts], poses)
        city = rotate(np.repeat(q0, len(ego), 0), ego) + t0
        # Laser assignment check (metric K): each laser's elevation spread in its assigned sensor frame at capture time.
        qi, ti = capture_poses(ts + offset, poses); conj = qi * np.array([1., -1., -1., -1.]); at_capture = rotate(conj, city - ti)
        sensor = np.where(laser < 32, 'up_lidar', 'down_lidar'); spreads = []
        for name, (t_s, rot) in sensors.items():
            sel = sensor == name; q = (at_capture[sel] - t_s) @ rot
            elevation = np.degrees(np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1])))
            spreads += [float(elevation[laser[sel] == n].std()) for n in np.unique(laser[sel])]
        spread = max(spreads); largest_spread = max(largest_spread, spread)
        assert spread < th['laserElevationSpreadLimitDeg'], f'Laser assignment check failed in sweep {ts}: {spread:.3f} deg'
        local = car_local(world(city, origin), frame)
        if rays:
            # Each return's own sensor origin at its capture time (up_lidar lasers 0-31, down_lidar 32-63).
            t_sensor = np.where((laser < 32)[:, None], sensors['up_lidar'][0], sensors['down_lidar'][0])
            origin_local = car_local(world(rotate(qi, t_sensor) + ti, origin), frame)
            a0, a1 = clip_many(origin_local, local, (-L / 2, -W / 2), (L / 2, W / 2))
            with np.errstate(invalid='ignore'):
                over = (a1 - a0) * np.hypot(local[:, 0] - origin_local[:, 0], local[:, 1] - origin_local[:, 1]) > th['crossedBinMinLengthM']
            ray_s.append(origin_local[over]); ray_p.append(local[over])
        near =(np.abs(local[:, 0]) <= L / 2 + reach + .5) & (np.abs(local[:, 1]) <= W / 2 + .5) & (local[:, 2] > -1) & (local[:, 2] < H + 2)
        others = ann[(ann.timestamp_ns == ts) & (ann.track_uuid != track)]
        inside = inside_cuboids(ego[near], others, 0.); grown = inside_cuboids(ego[near], others, th['otherObjectBufferBeyondEndsM'])
        loc = local[near]
        own = mine[mine.timestamp_ns == ts].iloc[0]
        centre = car_local(world(rotate(q0, np.array([[own.tx_m, own.ty_m, own.tz_m]])) + t0, origin), frame)[0]
        fp = (np.abs(loc[:, 1]) <= W / 2) & (np.abs(loc[:, 0]) <= L / 2) & (loc[:, 2] >= th['minHeightAboveBoxBottomM']) & (loc[:, 2] <= H + th['maxHeightAboveBoxTopM']) & ~inside
        xs = np.sort(loc[fp, 0])
        pct = lambda p: float(xs[max(math.ceil(p * len(xs)) - 1, 0)]) if len(xs) else None
        ego_at = car_local(world(t0, origin), frame)[0]
        per_sweep.append({'timestampNs': str(ts), 'sha256': s['sha256'], 'copies': s['copies'], 'receiptVerified': s['receiptVerified'],
                          'file': s['path'].name, 'keptFootprintReturns': int(fp.sum()), 'laserSpreadDeg': rnd(spread, 4),
                          'annotatedCentreOffsetM': {'along': rnd(centre[0]), 'lateral': rnd(centre[1]), 'vertical': rnd(centre[2] - H / 2)},
                          'keptXp05M': rnd(pct(.05)), 'keptXp95M': rnd(pct(.95)),
                          'egoHorizontalFromBoxCentreM': rnd(math.hypot(ego_at[0], ego_at[1]))})
        if rays:
            per_sweep[-1].update({'egoAlongM': rnd(ego_at[0]), 'egoLateralM': rnd(ego_at[1]), 'raysOverFootprint': int(len(ray_s[-1]))})
        local_all.append(loc); sweep_all.append(np.full(len(loc), k)); inside_all.append(inside); grown_all.append(grown)
    out = (np.concatenate(local_all), np.concatenate(sweep_all), np.concatenate(inside_all), np.concatenate(grown_all),
           per_sweep, skipped, largest_spread, {'annotations': sha(args.annotations), 'poses': sha(args.poses), 'calibration': sha(args.calibration)})
    return out + ((np.concatenate(ray_s), np.concatenate(ray_p)),) if rays else out


def cmd_profile(args):
    protocol, digest = load_protocol(args.protocol, args.protocol_lock); th = protocol['thresholds']
    ground = json.loads(Path(args.ground).read_bytes()); sugg = json.loads(Path(args.suggestions).read_bytes())
    assert sugg['vehicle']['status'] == 'annotated', 'Metric F applies to an annotated stationary vehicle only'
    track = sugg['vehicle']['track']; boxes = [c for c in ground['cuboids']['items'] if c['track'] == track]
    assert len(boxes) == 1, 'The tested track must appear once in ground.json'
    box = boxes[0]; frame = upright_frame(box); origin = np.asarray(ground['frame']['cityOrigin'], float)
    v2 = 'freeRayBandM' in th
    loaded = load_returns(args, frame, origin, track, th, rays=v2)
    local, sweep, inside, grown, per_sweep, skipped, spread, log_hashes = loaded[:8]
    dims = (frame['L'], frame['W'], frame['H'])
    primary = profile_from_points(local, sweep, dims, th, inside, grown)
    sensitivity = profile_from_points(local, sweep, dims, th, inside, grown, limit_top=False, remove_other=False)
    if v2:
        primary['cellEvidence'] = cell_evidence(local, inside, loaded[8], dims, th)
    doc = {'kind': 'metric-f LiDAR car-top profile', 'scene': args.scene, 'protocolSha256': digest,
           'frame': 'car-local: x forward along the tested box heading (front face +L/2), y left, z above the box bottom (box centre Y - H/2), metres',
           'box': {'track': track, 'category': box['category'], 'center': frame['center'].tolist(), 'heading': frame['heading'].tolist(),
                   'L': frame['L'], 'W': frame['W'], 'H': frame['H'], 'poseAverage': 'poseAverage' in box,
                   'cityCentre': [float(frame['center'][0] + origin[0]), float(-frame['center'][2] + origin[1]), float(frame['center'][1] + origin[2])],
                   'source': 'ground.json cuboid of suggestions.json vehicle.track, upright (pitch and roll dropped, as the app does)'},
           'inputs': {'ground.json': sha(args.ground), 'suggestions.json': sha(args.suggestions), **log_hashes, 'logId': args.log_id},
           'sweeps': {'used': len(per_sweep), 'skipped': skipped, 'perSweep': per_sweep, 'largestLaserSpreadDeg': spread,
                      'receiptVerified': sum(1 for s in per_sweep if s['receiptVerified'])},
           'primary': primary, 'sensitivity': sensitivity}
    print(write_new(args.out, doc))


def cmd_score(args):
    protocol, digest = load_protocol(args.protocol, args.protocol_lock); th = protocol['thresholds']
    rows_doc = json.loads(Path(args.rows).read_bytes()); profile_doc = json.loads(Path(args.profile).read_bytes())
    assert rows_doc['protocolSha256'] == digest and profile_doc['protocolSha256'] == digest, 'Inputs were made under another protocol'
    check_frames(rows_doc, profile_doc)
    heights = score_scene(rows_doc, profile_doc, th)
    prim = profile_doc['primary']
    doc = {'kind': 'metric-f results', 'scene': args.scene, 'protocolSha256': digest,
           'label': protocol['label'], 'heightFrame': protocol['definition']['heightFrame'],
           'inputs': {'rows': {'file': Path(args.rows).name, 'sha256': sha(args.rows)}, 'profile': {'file': Path(args.profile).name, 'sha256': sha(args.profile)}},
           'app': rows_doc['app'], 'testedBox': {**profile_doc['box'], 'setbackM': rows_doc['testedBox']['setbackM']},
           'profile': {'alongBins': [{k: (rnd(v) if isinstance(v, float) else v) for k, v in b.items()} for b in prim['alongBins']['bins']],
                       'binWidthM': prim['alongBins']['widthM'], 'counts': prim['counts'],
                       'overhang': prim['overhang'], 'sweepsUsed': profile_doc['sweeps']['used']},
           'heights': heights}
    if prim.get('cellEvidence') is not None:
        ev = prim['cellEvidence']; W = profile_doc['box']['W']; lat = [s.get('egoLateralM') for s in profile_doc['sweeps']['perSweep']]
        doc['profile']['cells2D'] = {'lateralN': prim['cells']['lateralN'], 'lateralWidthM': prim['cells']['lateralWidthM'],
                                     'freeRayHeightsM': [ev['fromM'], ev['toM']], 'freeRayBandM': ev['bandM'],
                                     'cells': [{'bin': c['bin'], 'lateral': c['lateral'], 'yLeft': rnd(c['yLeft']), 'yRight': rnd(c['yRight']),
                                                'count': c['count'], 'topZ': rnd(c['topZ']), 'belowMinHeight': e['belowMinHeight'], 'freeRays': e['freeRays']}
                                               for c, e in zip(prim['cells']['cells'], ev['cells'])]}
        doc['profile']['egoSide'] = {'sweeps': len(lat), 'leftOfBox': sum(1 for v in lat if v > W / 2), 'rightOfBox': sum(1 for v in lat if v < -W / 2),
                                     'withinBoxWidth': sum(1 for v in lat if -W / 2 <= v <= W / 2),
                                     'note': 'egoLateralM: car-local y of the ego origin at each sweep timestamp (left of the tested car is +)'}
    if rows_doc.get('baselineCheck'):
        doc['baselineCheck'] = rows_doc['baselineCheck']
    if args.same_classes_as:
        earlier = json.loads(Path(args.same_classes_as).read_bytes())
        doc['sameClassesAs'] = {**check_same_classes(heights, earlier), 'file': Path(args.same_classes_as).name,
                                'sha256': sha(args.same_classes_as), 'protocolSha256': earlier['protocolSha256']}
    print(write_new(args.out, doc))


def cmd_summarize(args):
    protocol, digest = load_protocol(args.protocol, args.protocol_lock)
    scored = set(protocol['target']['scoredScenes']); scenes = []
    for f in args.results:
        r = json.loads(Path(f).read_bytes()); assert r['protocolSha256'] == digest
        entry = {'scene': r['scene'], 'scored': r['scene'] in scored, 'file': Path(f).name, 'sha256': sha(f), 'app': r['app']['dir'],
                 'frontOverhangReturns': r['profile']['overhang']['front']['returnsWithinSummaryReach'],
                 'rearOverhangReturns': r['profile']['overhang']['rear']['returnsWithinSummaryReach'],
                 'sweepsUsed': r['profile']['sweepsUsed'], 'heights': {}}
        total = {'rows': 0, 'plainBlockedRows': 0, 'likelyFalse': 0, 'ambiguous': 0, 'unresolved': 0, 'supported': 0}
        v2 = 'egoSide' in r['profile']
        if v2:
            entry['egoSide'] = r['profile']['egoSide']; entry['likelyFalseRows'] = []
            total2d = {k: 0 for k in ('likelyFalse', 'ambiguous', 'unresolved', 'supported')}
        for name, h in r['heights'].items():
            if 'error' in h:
                entry['heights'][name] = {'error': h['error']}; continue
            c, s = h['counts'], h['stopBlocking']
            if v2:
                for k in total2d:
                    total2d[k] += h['counts2D']['plainBlocked'][k]
                for row in h['rows']:
                    if not (plain_blocked(row) and row['class'] == 'likelyFalse'):
                        continue
                    cells = row['cells2D']; unmeasured = [x for x in cells if x['count'] < protocol['thresholds']['minReturnsMeasured']]
                    entry['likelyFalseRows'].append({
                        'height': name, 'setbackM': rnd(row['setbackM']), 'cutIntoBoxM': rnd(-row['gapM']),
                        'minClearanceM': rnd(row['minClearanceM']), 'minClearanceBin': row['minClearanceBin'],
                        'crossedBins': [b['bin'] for b in row['bins']], 'lineZM': [min(b['lineZ'] for b in row['bins']), max(b['lineZ'] for b in row['bins'])],
                        'class2D': row['class2D'], 'minClearance2DM': rnd(row['minClearance2DM']),
                        'crossedCells': [[x['bin'], x['lateral']] for x in cells], 'crossedCellReturns': [x['count'] for x in cells],
                        'unmeasuredCrossedCells': len(unmeasured),
                        'unmeasuredCellsFreeRaysBelowLine': [x['freeRaysBelowLine'] for x in unmeasured],
                        'unmeasuredCellsBelowMinHeightReturns': [x['belowMinHeight'] for x in unmeasured],
                        'footprint': row['footprint']})
            entry['heights'][name] = {'targetHeightM': h['targetHeightM'], 'rows': c['rows'], 'plainBlockedRows': c['plainBlockedRows'],
                                      'tooCloseToCallRows': c['tooCloseToCallRows'], **{k: c['plainBlocked'][k] for k in ('likelyFalse', 'ambiguous', 'unresolved', 'supported')},
                                      'likelyFalseSetbacksM': c['plainBlockedSetbacksM']['likelyFalse'], 'ambiguousSetbacksM': c['plainBlockedSetbacksM']['ambiguous'],
                                      'unresolvedSetbacksM': c['plainBlockedSetbacksM']['unresolved'],
                                      'secondary2D': {k: h['counts2D']['plainBlocked'][k] for k in ('likelyFalse', 'ambiguous', 'unresolved', 'supported')},
                                      'sensitivityNoRemovalNoTopLimit': {k: h['countsSensitivity']['plainBlocked'][k] for k in ('likelyFalse', 'ambiguous', 'unresolved', 'supported')},
                                      'stopBlocking': {k: rnd(v) if isinstance(v, float) else v for k, v in s.items()}}
            total['rows'] += c['rows']; total['plainBlockedRows'] += c['plainBlockedRows']
            for k in ('likelyFalse', 'ambiguous', 'unresolved', 'supported'):
                total[k] += c['plainBlocked'][k]
        entry['total'] = total
        if v2:
            entry['total2D'] = total2d
            entry['likelyFalseAlso2DLikelyFalse'] = sum(1 for x in entry['likelyFalseRows'] if x['class2D'] == 'likelyFalse')
        entry['targetMet'] = (total['likelyFalse'] == 0) if entry['scored'] else None
        scenes.append(entry)
    doc = {'kind': 'metric-f summary', 'protocolSha256': digest, 'target': protocol['target']['rule'], 'label': protocol['label'],
           'heightFrame': protocol['definition']['heightFrame'], 'scenes': scenes,
           'targetMetAllScoredScenes': all(s['targetMet'] for s in scenes if s['scored'])}
    print(write_new(args.out, doc))


def cmd_receipt(args):
    _, digest = load_protocol(args.protocol, args.protocol_lock)
    import platform
    files = {}
    for f in args.hash:
        files[f] = sha(f)
    versions = {'python': platform.python_version(), 'numpy': np.__version__}
    try:
        import pyarrow
        versions['pyarrow'] = pyarrow.__version__
    except ImportError:
        pass
    doc = {'kind': 'metric-f run receipt', 'protocolSha256': digest, 'sha256': files, 'versions': versions,
           'commands': Path(args.commands).read_text().splitlines(), 'extra': json.loads(Path(args.extra).read_bytes()) if args.extra else None}
    print(write_new(args.out, doc))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = p.add_subparsers(dest='cmd', required=True)
    for name in SUBCOMMANDS:
        s = sub.add_parser(name); s.add_argument('--protocol', required=True); s.add_argument('--protocol-lock', required=True)
        s.add_argument('--out', required=True)
        if name == 'profile':
            for k in ('ground', 'suggestions', 'scene', 'annotations', 'poses', 'calibration', 'log-id'):
                s.add_argument('--' + k, required=True)
            s.add_argument('--sweep-dir', action='append', required=True); s.add_argument('--receipt', action='append')
        elif name == 'score':
            for k in ('profile', 'rows', 'scene'):
                s.add_argument('--' + k, required=True)
            s.add_argument('--same-classes-as', help='earlier results file whose row classes must be reproduced exactly')
        elif name == 'summarize':
            s.add_argument('--results', nargs='+', required=True)
        else:
            s.add_argument('--commands', required=True); s.add_argument('--hash', nargs='+', required=True); s.add_argument('--extra')
    args = p.parse_args(argv)
    {'profile': cmd_profile, 'score': cmd_score, 'summarize': cmd_summarize, 'receipt': cmd_receipt}[args.cmd](args)


if __name__ == '__main__':
    main()

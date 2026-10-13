"""Offline AV2 mapped-ground layer for map-first placement; mapped ground is not observed road.

Reads an assembled indexed AV2 scene, its preparation manifest, the frozen placement protocol and
(optionally) the AV2 annotations, and writes a NEW directory with:
  ground.json  drivable outlines, crossing edges, ego lane and annotated cuboids in scene coordinates,
               the terrain window description, tolerances and LiDAR support rows
  terrain.bin  float32 little-endian window of the publisher float16 ground raster, NaN = no value
  scene.json   the assembled scene metadata with measurements.groundPolicy 'mappedGround' and a
               hash-pinned measurements.ground descriptor (buffers stay byte-identical)
No hole filling, interpolation or invented geometry. Missing evidence stays Unknown; nothing here
establishes free space or Clear. The shared geometry helpers are also used by
suggest_av2_placements.py so the two scripts and the app apply one rule.

Protocol v3 (protocolVersion 3) adds, and only then: ground.json schemaVersion 2 with the current
stationary vehicle drawn at the average of all its annotations (poseAverage), an along-ray uncertainty
height band fitted on this development scene's camera-visible LiDAR ground returns (alongRayUncertainty)
and a 30 m placement range limit (placementRangeLimitMeters). Protocols v1/v2 produce the schema-1 layer
exactly as before.
"""
import argparse
from importlib.metadata import version
from io import BytesIO
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS / 'sponsors'))
sys.path.insert(0, str(HERE))
from common import canonical, digest, parse, publish, record, require, snapshot  # noqa: E402
import prepare_av2  # noqa: E402  (sampling rule, map decoding, Feather reading)

WORLD_FROM_CITY = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
MAX_SCENE = 128 * 1024
MAX_FILE = 16 * 1024 * 1024
EPS = 1e-9
LICENSE = {'id': 'CC-BY-NC-SA-4.0', 'attribution': '(c) 2021 Argo AI, LLC', 'source': 'Argoverse 2 Sensor Dataset',
           'changes': 'Mapped drivable areas, crossing edges, ego lane, a ground-raster window and annotated cuboids converted to '
                      'umbra scene coordinates; LiDAR support rows selected by a frozen protocol. No endorsement implied.'}
# Held-out tests, controls and reserve of the frozen Palo Alto split (.local/evaluation/pao-accuracy/split.json, sha256
# 47f64fe0...). Nothing is ever fitted on these logs; the protocol v3 band fit refuses them.
HELD_OUT_LOG_PREFIXES = ('88f47a10', '3153b5b3', 'd33f667d', 'fd5c6932', '25e5c600', 'e1d68dde', 'a91d4c7b', '4a60c567', 'b6c4361a')
# protocol currentVehicle.checks 'stationary' (same numbers as suggest_av2_placements.annotated_vehicle).
STATIONARY_SPAN_S, STATIONARY_DISPLACEMENT_M = 3.0, 1.0


def too_far(limit):
    return f'Too far to place (over {limit:g} m).'


# ------------------------------------------------------------------ shared geometry (one rule for scripts and app)
class Terrain:
    """Window of the publisher raster with prepare_av2.ground_sample's exact sampling rule, in world X/Z."""

    def __init__(self, window, column0, row0, shape, sim2, city_origin):
        self.window = np.asarray(window, np.float32)
        self.column0, self.row0 = int(column0), int(row0)
        self.shape = (int(shape[0]), int(shape[1]))
        r = sim2['R']
        require(isinstance(r, list) and len(r) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in r),
                'Publisher Sim2 R requires four finite flat numeric values')
        self.R = np.asarray(r, float).reshape(2, 2); self.t = np.asarray(sim2['t'], float); self.s = float(sim2['s'])
        require(self.t.shape == (2,) and np.isfinite(self.t).all() and math.isfinite(self.s) and self.s > 0 and
                np.allclose(self.R.T @ self.R, np.eye(2), rtol=0, atol=1e-9) and abs(np.linalg.det(self.R) - 1) < 1e-9,
                'Proper positive Sim2 required')
        self.origin = np.asarray(city_origin, float)

    def image(self, world_xz):
        xz = np.asarray(world_xz, float).reshape(-1, 2)
        city = np.column_stack([xz[:, 0] + self.origin[0], -xz[:, 1] + self.origin[1]])
        r, t = self.R, self.t
        with np.errstate(invalid='ignore', over='ignore'):
            # Same arithmetic as prepare_av2.ground_sample: s * (R*cityXY + t).
            return self.s * np.column_stack([r[0, 0] * city[:, 0] + r[0, 1] * city[:, 1] + t[0],
                                             r[1, 0] * city[:, 0] + r[1, 1] * city[:, 1] + t[1]])

    def sample(self, world_xz):
        """Return (cityZ heights with NaN for no value, global [column,row] indices or -1)."""
        image = self.image(world_xz)
        castable = np.isfinite(image).all(1) & (np.abs(image) < np.iinfo(np.int64).max).all(1)
        idx = np.full(image.shape, -1, np.int64); idx[castable] = np.trunc(image[castable]).astype(np.int64)
        inside = castable & (idx[:, 0] >= 0) & (idx[:, 0] < self.shape[1]) & (idx[:, 1] >= 0) & (idx[:, 1] < self.shape[0])
        domain = inside & ~(image < 0).any(1)
        local = idx - [self.column0, self.row0]
        rows, cols = self.window.shape
        win = domain & (local[:, 0] >= 0) & (local[:, 0] < cols) & (local[:, 1] >= 0) & (local[:, 1] < rows)
        heights = np.full(len(image), np.nan)
        heights[win] = self.window[local[win, 1], local[win, 0]]
        idx[~win] = -1
        return heights, idx


def half_ulp16(city_z):
    return float(np.spacing(np.abs(np.float16(city_z)))) / 2


def ray_direction(scene, pixel):
    c = scene['camera']; u, v = pixel
    d = np.asarray(scene['transform']['worldFromCamera'], float) @ np.array([(u - c['cx']) / c['fx'], (v - c['cy']) / c['fy'], 1.])
    return d / np.linalg.norm(d)


def project(scene, point):
    """World point -> floating crop [u, v] (None behind the camera)."""
    c = scene['camera']; q = np.asarray(scene['transform']['worldFromCamera'], float).T @ np.asarray(point, float)
    if q[2] <= 0:
        return None
    return [c['fx'] * q[0] / q[2] + c['cx'], c['fy'] * q[1] / q[2] + c['cy']]


def ray_hit(direction, terrain, max_range):
    """First hit of a unit ray from the world origin with the piecewise-constant raster (protocol pixelRay.firstHit)."""
    d = np.asarray(direction, float); oz = terrain.origin[2]
    if not d[1] < 0:
        return {'hit': False, 'reason': 'Pixel ray does not descend toward the ground.'}
    a = terrain.image([[0., 0.]])[0]
    b = terrain.s * (terrain.R @ np.array([d[0], -d[2]]))  # image(t*d) = a + t*b (affine)
    cuts = {0., float(max_range)}
    for k in (0, 1):
        if b[k] != 0:
            lo, hi = sorted((a[k], a[k] + max_range * b[k]))
            for n in range(math.floor(lo) + 1, math.ceil(hi)):
                t = (n - a[k]) / b[k]
                if 0 < t < max_range:
                    cuts.add(float(t))
    cuts = sorted(cuts)
    for ta, tb in zip(cuts[:-1], cuts[1:]):
        if tb <= ta:
            continue
        tm = (ta + tb) / 2
        h, idx = terrain.sample([[tm * d[0], tm * d[2]]])
        if not np.isfinite(h[0]):
            return {'hit': False, 'reason': 'Ray crosses a cell without mapped ground height before reaching the ground.', 't': ta}
        ground = float(h[0]) - oz
        if ta * d[1] <= ground:
            t, kind = ta, 'step'
        elif ground / d[1] <= tb:
            t, kind = ground / d[1], 'top'
        else:
            continue
        if t <= 0:
            return {'hit': False, 'reason': 'Camera origin is not above the mapped ground.'}
        return {'hit': True, 't': float(t), 'point': (t * d).tolist(), 'kind': kind, 'cellCityZ': float(h[0]),
                'cell': idx[0].tolist()}
    return {'hit': False, 'reason': f'No mapped ground within {max_range:g} m along the pixel ray.'}


def inside_polygons(xz, polygons):
    """Even-odd membership in any outline (world X/Z)."""
    p = np.asarray(xz, float).reshape(-1, 2); result = np.zeros(len(p), bool)
    for poly in polygons:
        poly = np.asarray(poly, float); x, y = poly[:, 0], poly[:, 1]; inside = np.zeros(len(p), bool); j = len(poly) - 1
        for i in range(len(poly)):
            with np.errstate(divide='ignore', invalid='ignore'):
                crosses = ((y[i] > p[:, 1]) != (y[j] > p[:, 1])) & \
                          (p[:, 0] < (x[j] - x[i]) * (p[:, 1] - y[i]) / (y[j] - y[i]) + x[i])
            inside ^= crosses; j = i
        result |= inside
    return result


def pixel_ground_hit(scene, uv, terrain, polygons, max_range, range_limit=None):
    """The runtime placement rule for a crop position [u, v] (integer pixel centre or fractional): its unit camera ray, the
    first mapped-ground hit, then (protocol v3) the horizontal range limit, then the drivable test.
    Returns (direction, hit or None, Unknown reason or None)."""
    d = ray_direction(scene, uv); hit = ray_hit(d, terrain, max_range)
    if not hit['hit']:
        return d, None, hit['reason']
    p = hit['point']
    if range_limit is not None and math.hypot(p[0], p[2]) > range_limit:
        return d, None, too_far(range_limit)
    if not inside_polygons([p[0], p[2]], polygons)[0]:
        return d, None, 'First mapped-ground hit is outside the mapped drivable area.'
    return d, hit, None


def segment_hits_boxes(ends, cuboids):
    """Closed segment world origin -> each end intersects any cuboid (inclusive slab test, app segmentIntersectsBox)."""
    ends = np.asarray(ends, float).reshape(-1, 3); hit = np.zeros(len(ends), bool)
    for box in cuboids:
        m = np.asarray(box['worldFromBox'], float); c = np.asarray(box['center'], float)
        half = np.array([box['length'], box['width'], box['height']]) / 2
        s = (-c) @ m; e = (ends - c) @ m  # box-local start (camera) and ends
        enter = np.zeros(len(ends)); leave = np.ones(len(ends)); ok = np.ones(len(ends), bool)
        for k in range(3):
            dk = e[:, k] - s[k]; flat = np.abs(dk) < EPS
            ok &= ~(flat & ((s[k] < -half[k]) | (s[k] > half[k])))
            with np.errstate(divide='ignore', invalid='ignore'):
                lo = (-half[k] - s[k]) / dk; hi = (half[k] - s[k]) / dk
            enter = np.where(flat, enter, np.maximum(enter, np.minimum(lo, hi)))
            leave = np.where(flat, leave, np.minimum(leave, np.maximum(lo, hi)))
        hit |= ok & (enter <= leave)
    return hit


def line_crossings(start, direction, polygons):
    """Parameters s >= 0 where start + s*direction meets any outline edge (world X/Z), ascending."""
    p = np.asarray(start, float); u = np.asarray(direction, float); found = []
    for poly in polygons:
        poly = np.asarray(poly, float)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            e = b - a; den = u[0] * e[1] - u[1] * e[0]
            if abs(den) < EPS:
                continue
            w = a - p; s = (w[0] * e[1] - w[1] * e[0]) / den; r = (w[0] * u[1] - w[1] * u[0]) / den
            if s >= 0 and -EPS <= r <= 1 + EPS:
                found.append(float(s))
    return sorted(found)


def curb_scan(start, direction, polygons, epsilon, limit=30.):
    """Distance to the first drivable-union exit moving along direction; None when unavailable."""
    p = np.asarray(start, float); u = np.asarray(direction, float) / np.linalg.norm(direction)
    if not inside_polygons(p, polygons)[0]:
        return None
    for s in line_crossings(p, u, polygons):
        if s > limit:
            break
        if not inside_polygons(p + (s + epsilon) * u, polygons)[0]:
            return s
    return None



def support_rows(positions, eligible, terrain, cuboids, tolerance):
    """Camera-visible returns agreeing with the mapped ground (protocol supportClasses); all rows tested, no pre-filter."""
    positions = np.asarray(positions, float); eligible = np.asarray(eligible, bool); count = len(positions)
    finite = np.isfinite(positions).all(1); dz = np.full(count, np.nan)
    if finite.any():
        dz[finite] = positions[finite, 1] - (terrain.sample(positions[finite][:, [0, 2]])[0] - terrain.origin[2])
    agree = np.isfinite(dz) & (np.abs(dz) <= tolerance)
    occluded = np.zeros(count, bool)
    if cuboids and (eligible & finite).any():
        occluded[eligible & finite] = segment_hits_boxes(positions[eligible & finite], cuboids)
    rows = np.flatnonzero(eligible & finite & ~occluded & agree)
    return rows, {'cameraEligibleRows': int(eligible.sum()), 'eligibleBehindCuboid': int(occluded.sum()),
                  'eligibleAgreeing': int((eligible & agree).sum()), 'eligibleWithoutTerrainValue': int((eligible & ~np.isfinite(dz)).sum()),
                  'cuboidOcclusionApplied': cuboids is not None}


# ------------------------------------------------------------------ protocol v3: averaged stationary pose
def circular_mean(angles):
    """Circular mean (radians) of headings: atan2(sum sin, sum cos); undefined when they cancel."""
    a = np.asarray(angles, float); s = float(np.sin(a).sum()); c = float(np.cos(a).sum())
    require(len(a) > 0 and math.hypot(s, c) > 1e-6 * len(a), 'Circular mean undefined: headings cancel')
    return math.atan2(s, c)


def wrap_angle(angle):
    """Angle difference wrapped to [-pi, pi)."""
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def upright_world_from_box(yaw):
    """worldFromBox of an upright box whose +x points along city yaw (box +z = city up)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (WORLD_FROM_CITY @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])).tolist()


def _std(values):
    return float(np.sqrt(np.mean(np.square(values - np.mean(values))))) if len(values) else 0.


def average_track_pose(history, poses, sweep_timestamp):
    """protocol v3 currentVehicle.pose 'averageAllAnnotations': every row of one stationary track, each mapped to the city
    frame with the exact ego pose at its own timestamp. Centre = arithmetic mean; yaw = circular mean of each row's
    horizontal heading (box +x); upright box (row pitch/roll not used); dimensions = medians. Rows are sorted by
    timestamp first, so the result does not depend on input order. Returns (city centre, yaw, [length, width, height],
    poseAverage record)."""
    rows = sorted(history, key=lambda r: r['timestamp_ns'])
    require(len(rows) > 0 and all(r['timestamp_ns'] in poses for r in rows), 'Exact pose missing for an annotation of the averaged track')
    centres, yaws, dims = [], [], []
    for r in rows:
        pose = poses[r['timestamp_ns']]
        centres.append(pose[:3, :3] @ np.array([r['tx_m'], r['ty_m'], r['tz_m']], float) + pose[:3, 3])
        h = (pose[:3, :3] @ _quat(r))[:, 0]
        require(math.hypot(h[0], h[1]) > 1e-6, 'Annotated heading is vertical')
        yaws.append(math.atan2(h[1], h[0]))
        dims.append([r['length_m'], r['width_m'], r['height_m']])
    centres = np.asarray(centres, float); dims = np.asarray(dims, float)
    require(np.isfinite(centres).all() and np.isfinite(dims).all() and (dims > 0).all(), 'Invalid annotation of the averaged track')
    centre = centres.mean(0); yaw = circular_mean(yaws); median = np.median(dims, 0)
    heading = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-heading[1], heading[0]])
    offset = centres - centre; along, lateral, vertical = offset[:, :2] @ heading, offset[:, :2] @ left, offset[:, 2]
    dyaw = np.degrees([wrap_angle(y - yaw) for y in yaws])
    sweep = [i for i, r in enumerate(rows) if r['timestamp_ns'] == sweep_timestamp]
    require(len(sweep) == 1, 'The averaged track needs exactly one row at the sweep timestamp')
    i = sweep[0]
    spread = {'alongStd': _std(along), 'lateralStd': _std(lateral), 'verticalStd': _std(vertical),
              'alongMin': float(along.min()), 'alongMax': float(along.max()), 'lateralMin': float(lateral.min()),
              'lateralMax': float(lateral.max()), 'verticalMin': float(vertical.min()), 'verticalMax': float(vertical.max()),
              'maxHorizontalFromMean': float(np.hypot(along, lateral).max())}
    record = {'rule': 'averageAllAnnotations', 'rows': len(rows), 'firstTimestampNs': str(rows[0]['timestamp_ns']),
              'lastTimestampNs': str(rows[-1]['timestamp_ns']), 'spanSeconds': (rows[-1]['timestamp_ns'] - rows[0]['timestamp_ns']) / 1e9,
              'centreSpreadM': spread, 'yawSpreadDeg': {'std': _std(dyaw), 'maxAbs': float(np.abs(dyaw).max())},
              'dimensionRangeM': {k: [float(dims[:, j].min()), float(dims[:, j].max())] for j, k in enumerate(('length', 'width', 'height'))},
              'sweepRow': {'timestampNs': str(sweep_timestamp), 'alongM': float(along[i]), 'lateralM': float(lateral[i]),
                           'verticalM': float(vertical[i]), 'horizontalM': float(math.hypot(along[i], lateral[i])), 'yawDeg': float(dyaw[i])},
              'convention': 'offsets are row minus average: along the averaged heading (+ = forward), lateral (+ = left of the heading), '
                            'vertical (+ = up); population standard deviations; yaw offsets wrapped to [-180, 180) degrees',
              'numInteriorPoints': 'the sweep row\'s count, not recomputed for the averaged box'}
    return centre, yaw, median.tolist(), record


# ------------------------------------------------------------------ protocol v3: fitted along-ray height band
def nearest_rank(sorted_values, p):
    """Nearest-rank percentile of an ascending array: element ceil(p n) - 1."""
    return float(sorted_values[max(math.ceil(p * len(sorted_values)) - 1, 0)])


def residual_stats(dz, band):
    """Descriptive statistics of height residuals dz and the share within +/- band (count 0 and nulls when empty). Computed
    from sorted values, so any input order gives identical numbers."""
    d = np.sort(np.asarray(dz, float))
    require(np.isfinite(d).all(), 'Height residuals must be finite')
    if not len(d):
        return {'count': 0, 'medianM': None, 'sigmaRobustM': None, 'stdM': None, 'meanM': None, 'q68AbsM': None, 'q95AbsM': None,
                'q99AbsM': None, 'maxAbsM': None, 'withinBandFraction': None}
    a = np.sort(np.abs(d)); median = float(np.median(d))
    return {'count': int(len(d)), 'medianM': median, 'sigmaRobustM': 1.4826 * float(np.median(np.sort(np.abs(d - median)))),
            'stdM': _std(d), 'meanM': float(np.mean(d)), 'q68AbsM': nearest_rank(a, .68), 'q95AbsM': nearest_rank(a, .95),
            'q99AbsM': nearest_rank(a, .99), 'maxAbsM': float(a[-1]), 'withinBandFraction': float((a <= band).mean())}


def fit_height_band(dz, minimum_count):
    """protocol v3 alongRayUncertainty.heightBand: heightBandMeters = nearest-rank 95th percentile of |dz| rounded UP to
    0.001 m, plus descriptive statistics (residual_stats)."""
    a = np.sort(np.abs(np.asarray(dz, float)))
    require(np.isfinite(a).all() and len(a) >= max(minimum_count, 1), f'Height band fit needs at least {minimum_count} finite returns ({len(a)})')
    band = math.ceil(nearest_rank(a, .95) * 1000 - 1e-9) / 1000
    return band, residual_stats(dz, band)


def height_residuals(positions, eligible, terrain, cuboids, polygons, gate, range_limit):
    """protocol v3 fit set: camera-visible (eligible, finite, not behind any cuboid), raster value at the return's own XY,
    inside the drivable union, horizontal range <= range_limit, |dz| <= gate. Returns (rows, dz of rows, counts)."""
    positions = np.asarray(positions, float); eligible = np.asarray(eligible, bool); count = len(positions)
    finite = np.isfinite(positions).all(1); dz = np.full(count, np.nan)
    dz[finite] = positions[finite, 1] - (terrain.sample(positions[finite][:, [0, 2]])[0] - terrain.origin[2])
    candidate = eligible & finite
    occluded = np.zeros(count, bool)
    if cuboids and candidate.any():
        occluded[candidate] = segment_hits_boxes(positions[candidate], cuboids)
    drivable = np.zeros(count, bool)
    drivable[finite] = inside_polygons(positions[finite][:, [0, 2]], polygons)
    within = np.zeros(count, bool); within[finite] = np.hypot(positions[finite, 0], positions[finite, 2]) <= range_limit
    base = candidate & ~occluded & np.isfinite(dz) & drivable & within
    rows = np.flatnonzero(base & (np.abs(np.nan_to_num(dz, nan=np.inf)) <= gate))
    return rows, dz[rows], {'cameraVisibleWithTerrain': int((candidate & ~occluded & np.isfinite(dz)).sum()),
                            'insideDrivableWithinRange': int(base.sum()), 'removedByGroundGate': int(base.sum() - len(rows))}


def click_errors(scene, terrain, polygons, positions, uv, rows, max_range, range_limit):
    """For each row: the runtime rule at its floating projection (fractional click) and at the nearest integer pixel
    (integer click). Returns {kind: {'error': horizontal distance hit-to-return (NaN when Unknown), 'sinDepression',
    'reason'}}."""
    camera = scene['camera']; out = {}
    for kind in ('fractional', 'integer'):
        error = np.full(len(rows), np.nan); sin_dep = np.full(len(rows), np.nan); reasons = []
        for k, row in enumerate(rows):
            u, v = float(uv[row, 0]), float(uv[row, 1])
            if kind == 'integer':
                u, v = math.floor(u + .5), math.floor(v + .5)
            if not (0 <= u <= camera['width'] - 1 and 0 <= v <= camera['height'] - 1):
                reasons.append('Click is outside the recorded image.'); continue
            d, hit, reason = pixel_ground_hit(scene, [u, v], terrain, polygons, max_range, range_limit)
            if hit is None:
                reasons.append(reason); continue
            p = hit['point']; error[k] = math.hypot(p[0] - positions[row, 0], p[2] - positions[row, 2]); sin_dep[k] = -d[1]
            reasons.append(None)
        out[kind] = {'error': error, 'sinDepression': sin_dep, 'reason': reasons}
    return out


def coverage_table(errors, ranges, band, bands, last_inclusive=True):
    """Per horizontal-range band: placed count, Unknown reasons, share of errors <= band / sin(depression), error and
    shown-uncertainty percentiles (nearest rank)."""
    table = []
    for j, (a, b) in enumerate(bands):
        top = ranges <= b if (last_inclusive and j == len(bands) - 1) else ranges < b
        inside = (ranges >= a) & top
        placed = inside & np.isfinite(errors['error'])
        unknown = {}
        for k in np.flatnonzero(inside & ~np.isfinite(errors['error'])):
            unknown[errors['reason'][k]] = unknown.get(errors['reason'][k], 0) + 1
        e = np.sort(errors['error'][placed]); shown = np.sort(band / errors['sinDepression'][placed])
        covered = int((errors['error'][placed] <= band / errors['sinDepression'][placed]).sum())
        table.append({'band': f'{a:g}-{b:g} m', 'candidates': int(inside.sum()), 'placed': int(placed.sum()),
                      'unknown': dict(sorted(unknown.items())), 'covered': covered,
                      'coverage': covered / int(placed.sum()) if placed.any() else None,
                      'errorM': {'median': nearest_rank(e, .5), 'p95': nearest_rank(e, .95)} if len(e) else None,
                      'shownM': {'median': nearest_rank(shown, .5), 'p95': nearest_rank(shown, .95), 'max': float(shown[-1])} if len(shown) else None})
    return table


def along_ray_uncertainty(scene, positions, eligible, uv, terrain, cuboids, polygons, protocol, max_range):
    """protocol v3 alongRayUncertainty. Development scene (method fitThisDevelopmentScene): fit the height band here and
    report its in-sample coverage and split-half stability. Held-out scene (method fixed, protocol v3.1): use the fixed
    development band unchanged; this scene's residuals and coverage are recorded as a test of that band (heldOutResiduals),
    never as a fit: no minimum count, no split-half refit."""
    spec = protocol['alongRayUncertainty']; fit = spec['heightBand']; report = spec['coverageReport']
    limit = protocol['placementRange']['limitMeters']
    fixed = fit['method'] == 'fixed'
    if fixed:
        band = fit['heightBandMeters']
        require(type(band) is float and 0 < band <= 1 and isinstance(fit.get('fittedOn'), str), 'A fixed height band needs a value and its development source')
    else:
        require(fit['method'] == 'fitThisDevelopmentScene', 'Unknown height band method')
        require(not protocol['scene']['logId'].startswith(HELD_OUT_LOG_PREFIXES), 'Held-out or reserve data must never be used to fit the height band')
    rows, dz, counts = height_residuals(positions, eligible, terrain, cuboids, polygons, fit['groundGateMeters'], limit)
    if fixed:
        stats = residual_stats(dz, band)
    else:
        band, stats = fit_height_band(dz, fit['minimumCount'])
    ranges = np.hypot(positions[rows, 0], positions[rows, 2])
    errors = click_errors(scene, terrain, polygons, positions, uv, rows, max_range, limit)
    bands = [tuple(b) for b in report['bandsMeters']]
    coverage = {'definition': report['definition'],
                'fractionalClick': coverage_table(errors['fractional'], ranges, band, bands),
                'integerClick': coverage_table(errors['integer'], ranges, band, bands)}
    record_ = {**stats, **counts, 'groundGateMeters': fit['groundGateMeters'], 'rangeLimitMeters': limit, 'selection': fit['fitSet']}
    if fixed:
        return {'rule': spec['rule'], 'heightBandMeters': band, 'method': 'fixed', 'fittedOn': fit['fittedOn'],
                'heldOutResiduals': {**record_, 'scope': 'test: the fixed development band checked on this scene\'s returns; nothing is fitted here'},
                'coverage': coverage}
    halves = {}
    for name, part, other in (('fitEvenTestOdd', rows % 2 == 0, rows % 2 == 1), ('fitOddTestEven', rows % 2 == 1, rows % 2 == 0)):
        half_band, _ = fit_height_band(dz[part], 1)
        tested = {'error': errors['fractional']['error'][other], 'sinDepression': errors['fractional']['sinDepression'][other],
                  'reason': [r for r, keep in zip(errors['fractional']['reason'], other) if keep]}
        table = coverage_table(tested, ranges[other], half_band, [(bands[0][0], bands[-1][1])])[0]
        halves[name] = {'heightBandMeters': half_band, 'testedPlaced': table['placed'], 'testedCoverage': table['coverage']}
    return {'rule': spec['rule'], 'heightBandMeters': band, 'method': fit['method'],
            'fit': {**record_, 'statistic': fit['statistic'],
                    'scope': 'in-sample: fitted and reported on this development scene only; held-out testing is separate'},
            'coverage': {**coverage, 'splitHalf': halves}}


# ------------------------------------------------------------------ inputs
def _no_absolute(value, where):
    if isinstance(value, dict):
        for k, v in value.items():
            _no_absolute(v, where)
    elif isinstance(value, list):
        for v in value:
            _no_absolute(v, where)
    elif isinstance(value, str):
        require(not value.startswith('/') and not re.match(r'^[A-Za-z]:\\', value), f'Absolute local path in {where}')


def _world(city, origin):
    return (np.asarray(city, float) - origin) @ WORLD_FROM_CITY.T


def _world_xz(city, origin):
    """Horizontal scene position only: world X = cityX - x0, world Z = -(cityY - y0). Element-wise, so a missing publisher
    height (NaN cityZ) cannot blank X/Z (in _world's matrix product 0 * NaN = NaN). Same values as _world(...)[:, [0, 2]]
    for finite heights; no height is read, filled or invented."""
    c = np.asarray(city, float).reshape(-1, 3)
    return np.column_stack([c[:, 0] - origin[0], -(c[:, 1] - origin[1])])


def _xyz(points):
    return np.array([[q['x'], q['y'], q['z']] for q in points], float)


def _quat(row):
    q = np.array([row['qx'], row['qy'], row['qz'], row['qw']], float)
    n = float(np.linalg.norm(q))
    require(np.isfinite(q).all() and abs(n - 1) <= 1e-6, 'Annotation quaternion is not unit length')
    q /= n; x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _poses(data):
    rows = prepare_av2.table(data, ['timestamp_ns', 'qx', 'qy', 'qz', 'qw', 'tx_m', 'ty_m', 'tz_m']).to_pylist()
    poses = {}
    for row in rows:
        require(type(row['timestamp_ns']) is int and row['timestamp_ns'] not in poses, 'Invalid/duplicate pose timestamp')
        poses[row['timestamp_ns']] = prepare_av2.rigid(row)
    return poses


def cuboids_from_annotations(data, poses, timestamp, origin, average_prefix=None):
    """Annotated cuboids at the sweep timestamp. average_prefix (protocol v3): the one track starting with it at the sweep,
    if it passes the stationary check, is replaced by the average of all its annotations (average_track_pose)."""
    rows = prepare_av2.table(data, ['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m', 'height_m',
                                    'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m', 'num_interior_pts']).to_pylist()
    require(timestamp in poses, 'Exact sweep pose missing for annotations')
    tracks = {}
    for row in rows:
        tracks.setdefault(row['track_uuid'], []).append(row)
    matching = [] if average_prefix is None else [r for r in rows if r['timestamp_ns'] == timestamp and r['track_uuid'].startswith(average_prefix)]
    items = []
    for row in rows:
        if row['timestamp_ns'] != timestamp:
            continue
        size = [row[k] for k in ('length_m', 'width_m', 'height_m')]
        require(all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in size), 'Invalid annotation size')
        pose = poses[timestamp]
        center_city = pose[:3, :3] @ np.array([row['tx_m'], row['ty_m'], row['tz_m']], float) + pose[:3, 3]
        rotation = WORLD_FROM_CITY @ pose[:3, :3] @ _quat(row)
        motion = None; history = sorted(tracks[row['track_uuid']], key=lambda r: r['timestamp_ns'])
        if all(r['timestamp_ns'] in poses for r in history):
            xy = np.array([(poses[r['timestamp_ns']][:3, :3] @ np.array([r['tx_m'], r['ty_m'], r['tz_m']], float) +
                            poses[r['timestamp_ns']][:3, 3])[:2] for r in history])
            motion = {'sweeps': len(history), 'spanSeconds': (history[-1]['timestamp_ns'] - history[0]['timestamp_ns']) / 1e9,
                      'maxDisplacementFromFirstM': float(np.linalg.norm(xy - xy[0], axis=1).max())}
        item = {'track': row['track_uuid'], 'category': row['category'], 'center': _world(center_city, origin).tolist(),
                'worldFromBox': rotation.tolist(), 'length': size[0], 'width': size[1], 'height': size[2],
                'numInteriorPoints': int(row['num_interior_pts']), 'motion': motion}
        # Only an unambiguous, stationary current vehicle is averaged; otherwise the suggestion step refuses it.
        if (len(matching) == 1 and matching[0] is row and motion is not None and motion['spanSeconds'] >= STATIONARY_SPAN_S and
                motion['maxDisplacementFromFirstM'] <= STATIONARY_DISPLACEMENT_M):
            centre, yaw, dims, record = average_track_pose(history, poses, timestamp)
            item.update(center=_world(centre, origin).tolist(), worldFromBox=upright_world_from_box(yaw),
                        length=dims[0], width=dims[1], height=dims[2], poseAverage=record)
        items.append(item)
    require(items, 'No annotation rows at the sweep timestamp')
    return sorted(items, key=lambda item: item['track'])


def build(scene_path, scene_sha256, prepared_path, prepared_sha256, protocol_path, protocol_sha256,
          annotations_path, annotations_sha256, output):
    output = Path(output).absolute()
    require(not os.path.lexists(output), 'Existing output retained')
    scene_path = Path(scene_path).resolve(); prepared_path = Path(prepared_path).resolve(); protocol_path = Path(protocol_path).resolve()
    scene_raw = snapshot(scene_path, scene_sha256, MAX_SCENE); scene = parse(scene_raw)
    manifest_raw = snapshot(prepared_path, prepared_sha256, MAX_SCENE); manifest = parse(manifest_raw)
    protocol_raw = snapshot(protocol_path, protocol_sha256); protocol = parse(protocol_raw)
    p_scene, tol = protocol['scene'], protocol['tolerances']
    protocol_version = protocol.get('protocolVersion', 1)
    require(protocol_version in (1, 2, 3), 'Unsupported protocol version')
    v3 = protocol_version == 3
    if v3:
        pose_rule = protocol['currentVehicle']['pose']['rule']; limit = protocol['placementRange']['limitMeters']
        require(pose_rule == 'averageAllAnnotations' and type(limit) is float and 0 < limit <= tol['maxRangeMeters'],
                'Protocol v3 needs the averaged pose rule and a placement range limit within maxRangeMeters')
    m = scene.get('measurements', {}); prov = m.get('provenance', {})
    require(m.get('kind') == 'indexedSparse' and m.get('groundPolicy') == 'unvalidated' and scene.get('sourceClass') == 'Argoverse2Sensor',
            'Assembled indexed AV2 scene with unvalidated ground required')
    require(scene['id'] == p_scene['sceneId'] and all(prov.get(k) == p_scene[k] for k in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId')) and
            manifest.get('logId') == p_scene['logId'] and manifest.get('crossingId') == p_scene['crossingId'], 'Scene identity disagrees with the protocol')
    require(prov.get('preparedManifestSha256') == prepared_sha256, 'Scene was assembled from a different preparation')
    require(p_scene['annotationTimestampNs'] == p_scene['lidarTimestampNs'], 'Annotations must be at the sweep timestamp')
    origin = np.asarray(scene['transform']['cityOrigin'], float)
    require(np.array_equal(np.asarray(scene['transform']['worldFromCity'], float), WORLD_FROM_CITY) and np.isfinite(origin).all(), 'World frame mismatch')
    base = prepared_path.parent; captured = {}; records = {}
    for role in ('map', 'groundRaster', 'groundTransform', 'poses'):
        spec = manifest['sources'][role]; path = (base / spec['path']).resolve()
        captured[role] = snapshot(path, spec['sha256'], MAX_FILE)
        require(len(captured[role]) == spec['bytes'], 'Source byte identity mismatch')
        records[role] = record(path, captured[role], output)
    out = manifest['outputs']['references.npz']; ref_path = base / 'references.npz'
    references_raw = snapshot(ref_path, out['sha256'], MAX_FILE)
    require(out['sha256'] == prov.get('referencesSha256'), 'References differ from the scene')
    records['preparedManifest'] = record(prepared_path, manifest_raw, output); records['references'] = record(ref_path, references_raw, output)
    records['assembledScene'] = record(scene_path, scene_raw, output)
    count = m['count']; buffers = {}
    for key, dtype, width in (('referencePositions', '<f8', 3), ('referenceEligible', 'u1', 1)):
        path = scene_path.parent / m['buffers'][key]
        data = snapshot(path, m['artifactHashes'][key], MAX_FILE)
        array = np.frombuffer(data, dtype=dtype)
        require(array.size == count * width, 'Scene buffer length mismatch')
        buffers[key] = array.reshape(count, width) if width > 1 else array.astype(bool)
    positions, eligible = buffers['referencePositions'], buffers['referenceEligible']

    vector_map, _unknown = prepare_av2.decode_native_map(captured['map'])
    areas = []
    for key in sorted(vector_map['drivable_areas'], key=str):
        outline = _xyz(vector_map['drivable_areas'][key]['area_boundary'])
        require(len(outline) >= 3 and np.isfinite(outline[:, :2]).all(), 'Drivable outline needs finite XY vertices')
        # Outlines are horizontal; publisher maps can carry NaN heights in them (453e5558: 321 vertices in 9 of 14 outlines).
        areas.append({'id': str(vector_map['drivable_areas'][key]['id']), 'outline': _world_xz(outline, origin).tolist()})
    crossing = vector_map['pedestrian_crossings'][p_scene['crossingId']]
    edges = {}
    for name in ('edge1', 'edge2'):
        e = _xyz(crossing[name]); require(len(e) >= 2 and np.isfinite(e).all(), 'Finite crossing edges required')
        edges[name] = _world(e, origin).tolist()
    lane_id = re.search(r'lane segment (\d+)', protocol['mapElements']['roadAxis']).group(1)
    lane = vector_map['lane_segments'][lane_id]
    bounds = {}
    for name in ('left_lane_boundary', 'right_lane_boundary'):
        # ground.json stores full XYZ lane vertices (the app requires finite 3-vectors), so a missing height is refused, not filled.
        b = _xyz(lane[name])
        require(len(b) >= 2 and np.isfinite(b[:, :2]).all(), 'Finite lane boundaries required')
        require(np.isfinite(b[:, 2]).all(), f'Ego lane {name} has a missing (NaN) publisher height; it is not filled')
        bounds[name] = _world(b, origin).tolist()

    raster = prepare_av2.ground_raster(captured['groundRaster']); sim2 = parse(captured['groundTransform'])
    terrain_full = Terrain(raster, 0, 0, raster.shape, sim2, origin)
    cell = 1 / terrain_full.s
    require(abs(cell - protocol['groundRaster']['expectedCellMeters']) < 1e-9, 'Unexpected raster cell size')
    half = protocol['groundRaster']['windowHalfWidthMeters']
    corners = terrain_full.image([[sx * half, sz * half] for sx in (-1, 1) for sz in (-1, 1)])
    c0 = max(0, math.floor(corners[:, 0].min())); c1 = min(raster.shape[1], math.ceil(corners[:, 0].max()) + 1)
    r0 = max(0, math.floor(corners[:, 1].min())); r1 = min(raster.shape[0], math.ceil(corners[:, 1].max()) + 1)
    require(c1 > c0 and r1 > r0, 'Terrain window outside the raster')
    window = raster[r0:r1, c0:c1].astype('<f4')
    terrain = Terrain(window, c0, r0, raster.shape, sim2, origin)
    terrain_bytes = window.tobytes()

    # Same sampling rule: world-based window lookup equals the prepared city-based samples inside the window.
    npz = np.load(BytesIO(references_raw), allow_pickle=False)
    finite = np.isfinite(positions).all(1)
    heights = np.full(count, np.nan); in_window = np.zeros(count, bool)
    sampled, sampled_idx = terrain.sample(positions[finite][:, [0, 2]])
    heights[finite] = sampled; in_window[finite] = sampled_idx[:, 0] >= 0
    prepared_h = npz['nominalGroundHeights']
    mismatch = int((in_window & ~((heights == prepared_h) | (np.isnan(heights) & np.isnan(prepared_h)))).sum())
    require(mismatch == 0, 'World-based terrain lookup disagrees with prepare_av2.ground_sample')

    timestamp = int(p_scene['annotationTimestampNs']); cuboids = []
    annotations_record = None
    if annotations_path is not None:
        annotations_path = Path(annotations_path).resolve()
        ann = snapshot(annotations_path, annotations_sha256, MAX_FILE)
        cuboids = cuboids_from_annotations(ann, _poses(captured['poses']), timestamp, origin,
                                           protocol['currentVehicle']['trackPrefix'] if v3 else None)
        annotations_record = record(annotations_path, ann, output)
    road_tol = tol['roadToleranceMeters']
    rows, stats = support_rows(positions, eligible, terrain, cuboids if annotations_path is not None else None,
                               tol['supportHeightAgreementMeters'])
    v3_fields = {}
    if v3:
        key = 'referenceUV'
        uv_raw = snapshot(scene_path.parent / m['buffers'][key], m['artifactHashes'][key], MAX_FILE)
        require(len(uv_raw) == count * 2 * 8, 'Scene buffer length mismatch')
        uv = np.frombuffer(uv_raw, '<f8').reshape(count, 2)
        polygons = [np.asarray(a['outline'], float) for a in areas]
        v3_fields = {'schemaVersion': 2, 'placementRangeLimitMeters': limit,
                     'alongRayUncertainty': along_ray_uncertainty(scene, positions, eligible, uv, terrain,
                                                                  cuboids if annotations_path is not None else None,
                                                                  polygons, protocol, tol['maxRangeMeters'])}
    ground = {
        'schemaVersion': 1, 'kind': 'av2MappedGround', 'sceneId': scene['id'], 'baseSceneSha256': scene_sha256,
        'protocol': record(protocol_path, protocol_raw, output),
        'sources': {**records, 'annotations': annotations_record},
        'frame': {'cityOrigin': origin.tolist(), 'worldFromCity': WORLD_FROM_CITY.tolist(),
                  'convention': 'World X=cityX-originX, Y=cityZ-originZ (up), Z=-(cityY-originY); left camera at the world origin'},
        'roadToleranceMeters': road_tol, 'supportHeightAgreementMeters': tol['supportHeightAgreementMeters'],
        'supportRadiusMeters': tol['supportRadiusMeters'], 'maxRangeMeters': tol['maxRangeMeters'],
        'boundaryEpsilonMeters': tol['boundaryEpsilonMeters'],
        'terrain': {'file': 'terrain.bin', 'bytes': len(terrain_bytes), 'sha256': digest(terrain_bytes), 'dtype': 'float32',
                    'byteOrder': 'little', 'layout': 'row-major: value(row, column) at float index row*columns+column',
                    'columns': int(window.shape[1]), 'rows': int(window.shape[0]), 'column0': c0, 'row0': r0,
                    'rasterShape': list(raster.shape), 'sim2': sim2, 'cellMeters': cell,
                    'values': 'publisher cityZ (float16 widened exactly to float32); NaN = no value',
                    'finiteCells': int(np.isfinite(window).sum()), 'nanCells': int(np.isnan(window).sum()),
                    'samplingRule': 'cityX=worldX+cityOrigin[0], cityY=-worldZ+cityOrigin[1]; imageX=s*((R[0]*cityX+R[1]*cityY)+t[0]), '
                                    'imageY=s*((R[2]*cityX+R[3]*cityY)+t[1]); column=trunc(imageX), row=trunc(imageY); '
                                    'no value when any image coordinate is negative, non-finite, outside rasterShape or outside the window; '
                                    'groundWorldY = value - cityOrigin[2]; no interpolation'},
        'drivableAreas': areas,
        'crossing': {'id': p_scene['crossingId'], 'edge1': edges['edge1'], 'edge2': edges['edge2']},
        'egoLane': {'id': lane_id, 'laneType': lane.get('lane_type'), 'leftMarkType': lane.get('left_lane_mark_type'),
                    'rightMarkType': lane.get('right_lane_mark_type'), 'leftBoundary': bounds['left_lane_boundary'],
                    'rightBoundary': bounds['right_lane_boundary']},
        'cuboids': {'available': annotations_path is not None, 'timestampNs': p_scene['annotationTimestampNs'],
                    'frame': 'worldFromBox columns are box +x forward (length), +y left (width), +z up (height); center is the box centre',
                    'items': cuboids},
        'support': {'rows': rows.tolist(), 'count': int(len(rows)),
                    'rule': 'row is leftEligible, finite, not behind an annotated cuboid from the camera, and |worldY-(terrain(row)-cityOrigin[2])| <= supportHeightAgreementMeters',
                    **stats, 'preparedLookupMismatches': mismatch},
        'processingCodeHashes': {str(path.relative_to(SCRIPTS)): digest(path.read_bytes()) for path in
                                 (Path(__file__).resolve(), HERE / 'prepare_av2.py', SCRIPTS / 'sponsors/common.py')},
        'dependencies': {name: version(name) for name in ('numpy', 'pyarrow')},
        'license': LICENSE,
        'limitations': ['Mapped ground is publisher terrain inside mapped drivable areas, not observed road; support classes describe nearby LiDAR agreement only.',
                        'The LiDAR agreement check is a consistency check: the publisher built the ground map from LiDAR too.',
                        'Annotated cuboids are at the sweep timestamp, about 40 ms before the image; dynamic objects may have moved.',
                        'Drivable-area edges are a curb proxy, not measured curbs.',
                        'Free space and Clear remain unestablished; results are Blocked or Unknown only.']}
    if annotations_path is None:
        ground['limitations'].append('Annotations were not supplied: hidden flags are not evaluated and no cuboid occlusion was applied to LiDAR support.')
    if v3:
        ground.update(v3_fields)
        ground['limitations'] += [
            'Protocol v3: the current stationary vehicle (cuboid with poseAverage) is the average of all its annotations in the log '
            '(upright, circular-mean heading, median dimensions); every other cuboid is at the sweep timestamp.',
            ('Protocol v3: the along-ray uncertainty uses a fixed height band fitted on development scenes '
             f'({v3_fields["alongRayUncertainty"]["fittedOn"]}); this scene\'s LiDAR ground returns only test it (heldOutResiduals).'
             if v3_fields['alongRayUncertainty']['method'] == 'fixed' else
             'Protocol v3: the along-ray uncertainty uses a height band fitted on this development scene\'s camera-visible LiDAR ground '
             'returns (in-sample; not a held-out test, and the LiDAR shares calibration with the camera).'),
            f'Protocol v3: a mapped-ground hit more than {limit:g} m from the camera (horizontal) is refused.']
    _no_absolute(ground, 'ground.json')
    ground_bytes = canonical(ground)
    grounded = parse(scene_raw)
    gm = grounded['measurements']
    gm['groundPolicy'] = 'mappedGround'
    gm['ground'] = {'file': 'ground.json', 'bytes': len(ground_bytes), 'sha256': digest(ground_bytes)}
    gm['provenance']['originalSourceRecordBase'] = os.path.relpath(base, scene_path.parent)
    grounded['processing']['groundPolicy'] = ('mappedGround: a pixel maps to the first hit of its camera ray with the publisher ground raster '
                                              'inside mapped drivable areas; mapped ground is not observed road; per-point support class')
    grounded['limitations'] = [text if not text.startswith('Native grounding') else
                               'Placement uses publisher-mapped ground inside mapped drivable areas (mapped, not observed unless a camera-visible '
                               'LiDAR return agrees nearby); contact and free space remain unvalidated.' for text in grounded['limitations']]
    _no_absolute(grounded, 'scene.json')
    scene_bytes = canonical(grounded)
    publish(output, {'terrain.bin': terrain_bytes, 'ground.json': ground_bytes, 'scene.json': scene_bytes})
    result = {'groundSha256': digest(ground_bytes), 'terrainSha256': digest(terrain_bytes), 'sceneSha256': digest(scene_bytes),
              'supportRows': int(len(rows)), 'cuboids': len(cuboids), 'window': [c0, r0, int(window.shape[1]), int(window.shape[0])]}
    if v3:
        result.update(heightBandMeters=v3_fields['alongRayUncertainty']['heightBandMeters'],
                      averagedTracks=[c['track'] for c in cuboids if 'poseAverage' in c])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('scene', 'scene-sha256', 'prepared', 'prepared-sha256', 'protocol', 'protocol-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--annotations'); parser.add_argument('--annotations-sha256')
    args = parser.parse_args()
    if (args.annotations is None) != (args.annotations_sha256 is None):
        parser.exit(1, 'AV2 ground layer rejected: --annotations and --annotations-sha256 go together\n')
    try:
        result = build(args.scene, args.scene_sha256, args.prepared, args.prepared_sha256, args.protocol, args.protocol_sha256,
                       args.annotations, args.annotations_sha256, args.output)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f'AV2 ground layer rejected: {error}\n')
    print(canonical(result).decode())


if __name__ == '__main__':
    main()

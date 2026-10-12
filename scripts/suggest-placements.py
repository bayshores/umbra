#!/usr/bin/env python3
"""Suggest crossing, vehicle-front and person pixels for one prepared measured scene.

Deterministic, numpy + Pillow only (scripts/requirements-preparation.txt); no models, downloads or network.

Run:
  .local/venv-preparation/bin/python -I scripts/suggest-placements.py public/scenes/<id> \
      [--detections public/scenes/<id>/detections.json] [--replace]

Writes <scene>/suggestions.json (refuses to overwrite unless --replace), records its sha256 in the sibling
index.json catalog entry when that catalog lists the scene, and prints the sha256 last. Set SOURCE_DATE_EPOCH
to reproduce a file byte for byte (it only sets generatedUtc).

Method (ported from the iteration-1 prototype, .local/evaluation/auto-placement-2026-10-11/geometry):
  1. Curb face: half-height point of the measured road-to-sidewalk depth step, fitted along the road.
  2. Crosswalk: periodic bright bars on a bird's-eye grid resampled on the depth-registered ground plane; bar
     near ends are mapped back to the image, and the near edge's 3D position comes from measured depth on that
     image row (median over the curb-side lane).
  3. Every suggested pixel has measured depth and passes Sightline's roadPoint rule (|world y| <= 2 x road
     tolerance) plus stricter gates. The plane only resamples image brightness; no depth is filled or invented.
Anything that cannot be established is {"status": "unknown", "reason": ...} with no pixel.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

SCRIPT = 'scripts/suggest-placements.py'
OUTPUT = 'suggestions.json'

CFG = {
    'bevResolutionMeters': 0.05,
    'bevXMeters': [-14.0, 10.0],
    'bevZMeters': [-60.0, -6.0],
    'lateralBackgroundWindowMeters': 2.0,
    'barPeriodRangeMeters': [0.7, 2.0],
    'minRowRoadMeters': 4.0,
    'bandRelativeThreshold': 0.5,
    'bandMinLengthMeters': 1.5,
    'orientationSearchDegrees': [-8.0, 8.0, 0.05],
    'barWidthRangeMeters': [0.2, 0.8],
    'curbWindowMeters': 2.0,
    'curbWindowStepMeters': 1.0,
    'curbMinStepMeters': 0.12,
    'fallbackCurbLaneWidthMeters': 2.5,     # only if no lane line is found
    'maxAxisDisagreementDegrees': 2.0,      # curb direction vs crosswalk bar direction
    # Measured depth near 29 m clusters ~1 m apart along the ray, so the near edge is good to about +/-0.5 m.
    'crossingAlongRoadUncertaintyMeters': 0.5,
    'selection': {
        'maxGroundDistanceMeters': 0.5,
        'maxAbsHeightMeters': 0.08,         # stricter than roadPoint's 0.16
        'minCurbClearanceMeters': 0.15,     # stay off the curb face / transition
        'maxPlaneDepthResidualMeters': 0.5,  # off-edge picks: image row agrees with the measured point
        'maxEdgeAlongDeviationMeters': 0.3,  # edge-row picks agree with the measured curb-lane edge median
    },
    # A parked object the detector missed would add measured points at body height in the curb lane.
    'laneCensus': {'minHeightMeters': 0.3, 'maxHeightMeters': 2.5, 'nearestForwardMeters': 5.0, 'maxPoints': 20},
}

# Declared defaults: written into suggestions.json as labelled data, never detections.
DECLARED = {
    'vehicle': {'label': 'SUV', 'dimensions': {'width': 1.9, 'height': 1.8, 'length': 4.9},
                'setbackMeters': 6.096,     # 20 ft from the crossing near edge toward the camera
                'curbGapMeters': 0.15},     # curb face to the vehicle side
    'person': {'label': 'child', 'heightMeters': 1.1,
               'curbOffsetMeters': 0.30},   # person ground point from the curb face, on the road side
}


def fail(message):
    raise SystemExit(f'suggest-placements: {message}')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unknown(reason):
    return {'status': 'unknown', 'reason': reason}


# ---------------------------------------------------------------- scene and measured depth
def load_scene(scene_dir):
    try:
        meta = json.loads((scene_dir / 'scene.json').read_text())
    except (OSError, ValueError) as exc:
        fail(f'cannot read scene.json: {exc}')
    if meta.get('buffers', {}).get('depth') != 'depth.bin' or 'measurements' in meta:
        fail('scene has no depth.bin buffer; suggestions need a depth-image scene')
    c = meta['camera']
    W, H = c['width'], c['height']
    depth_path, image_path = scene_dir / 'depth.bin', scene_dir / 'recorded.jpg'
    if sha256(depth_path) != meta['artifactHashes']['depth']:
        fail('depth.bin does not match scene.json artifactHashes.depth')
    if 'left' in meta.get('sourceHashes', {}) and sha256(image_path) != meta['sourceHashes']['left']:
        fail('recorded.jpg does not match scene.json sourceHashes.left')
    raw = np.fromfile(depth_path, dtype='<u2')
    if raw.size != W * H:
        fail('depth.bin size does not match camera width*height')
    rgb = Image.open(image_path).convert('RGB')
    if rgb.size != (W, H):
        fail('recorded.jpg size does not match camera')
    return {
        'meta': meta, 'W': W, 'H': H, 'fx': c['fx'], 'fy': c['fy'], 'cx': c['cx'], 'cy': c['cy'],
        'raw': raw.reshape(H, W), 'gray': np.asarray(rgb.convert('L'), dtype=np.float64),
        'R': np.array(meta['transform']['worldFromCamera'], dtype=np.float64),
        'O': np.array(meta['transform']['cameraOriginGround'], dtype=np.float64),
        'divisor': float(meta['processing']['depthDivisor']),
        'tol': float(meta['processing']['roadToleranceMeters']),
        'eye': np.array(c['eye'], dtype=np.float64),
        'inputs': {name: sha256(scene_dir / name) for name in ('scene.json', 'recorded.jpg', 'depth.bin')},
    }


def measured_world(sc):
    """Same math as src/geometry.mjs measuredPoint, for every pixel; plus the plane depth along each ray."""
    H, W = sc['H'], sc['W']
    v, u = np.mgrid[0:H, 0:W]
    d = sc['raw'] / sc['divisor']
    ray = np.stack([(u - sc['cx']) / sc['fx'], (v - sc['cy']) / sc['fy'], np.ones((H, W))], -1)
    world = (ray * d[..., None] - sc['O']) @ sc['R'].T
    r1 = sc['R'][1]
    denom = ray @ r1
    with np.errstate(divide='ignore', invalid='ignore'):
        plane_depth = np.where(denom < 0, (r1 @ sc['O']) / denom, np.nan)
    return world, d, plane_depth


def world_to_pixel(sc, P):
    p = P @ sc['R'] + sc['O']
    with np.errstate(divide='ignore', invalid='ignore'):
        u = sc['fx'] * p[..., 0] / p[..., 2] + sc['cx']
        v = sc['fy'] * p[..., 1] / p[..., 2] + sc['cy']
    return u, v, p[..., 2]


def ground(x, z):
    return np.array([x, 0.0, z])


def robust_line(t, y, max_res, iters=3):
    """Least-squares y = a + b t with iterative outlier rejection."""
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        if keep.sum() < 3:
            return None
        b, a = np.polyfit(t[keep], y[keep], 1)
        r = y - (a + b * t)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= max(max_res, 3 * mad)
        if (new == keep).all():
            break
        keep = new
    b, a = np.polyfit(t[keep], y[keep], 1)
    r = y - (a + b * t)
    return {'a': float(a), 'b': float(b), 'keep': keep, 'rms': float(np.sqrt(np.mean(r[keep] ** 2))), 'residuals': r}


# ---------------------------------------------------------------- curb (measured depth only)
def detect_curb(world, has_depth, cfg):
    x, y, z = world[has_depth].T
    sel = (x > 0) & (x < 10) & (y > -0.4) & (y < 0.8)
    x, y, z = x[sel], y[sel], z[sel]
    bw = 0.1
    nb = int(round(9.0 / bw))
    centers = np.arange(nb) * bw + bw / 2
    windows = []
    for zc in np.arange(-8.0, -46.0, -cfg['curbWindowStepMeters']):
        k = np.abs(z - zc) <= cfg['curbWindowMeters'] / 2
        if k.sum() < 50:
            continue
        xb, yb = x[k], y[k]
        idx = np.floor(xb / bw).astype(int)
        inb = idx < nb
        xb, yb, idx = xb[inb], yb[inb], idx[inb]
        med = np.full(nb, np.nan)
        cnt = np.bincount(idx, minlength=nb)
        for i in np.nonzero(cnt >= 3)[0]:
            med[i] = np.median(yb[idx == i])
        road_bins = med[(centers >= 0.5) & (centers < 2.5)]
        if np.isnan(road_bins).all():
            continue
        road = float(np.nanmedian(road_bins))
        found = None
        for i in range(nb):
            if centers[i] < 1.0 or np.isnan(med[i]):
                continue
            if med[i] - road >= 0.08:
                ahead = med[i:i + 6]
                ahead = ahead[~np.isnan(ahead)]
                if len(ahead) >= 3 and np.all(ahead - road >= 0.06):
                    found = i
                    break
        if found is None:
            continue
        side_bins = med[(centers >= centers[found] + 0.3) & (centers <= centers[found] + 1.0)]
        if np.isnan(side_bins).all():
            continue
        step = float(np.nanmedian(side_bins)) - road
        if step < cfg['curbMinStepMeters']:
            continue
        half = road + step / 2
        xc = None
        for j in range(max(found - 6, 1), min(found + 6, nb)):
            prev = j - 1
            while prev >= 0 and np.isnan(med[prev]):
                prev -= 1
            if prev < 0 or np.isnan(med[j]):
                continue
            if med[prev] < half <= med[j]:
                t = (half - med[prev]) / (med[j] - med[prev])
                xc = float(centers[prev] + t * (centers[j] - centers[prev]))
                break
        if xc is not None:
            windows.append({'z': float(zc), 'x': xc, 'step': step})
    if len(windows) < 4:
        return None
    zz = np.array([w['z'] for w in windows])
    xx = np.array([w['x'] for w in windows])
    fit = robust_line(zz, xx, 0.15)
    if fit is None:
        return None
    steps = [w['step'] for w, kp in zip(windows, fit['keep']) if kp]
    return {'a': fit['a'], 'b': fit['b'], 'rms': fit['rms'], 'inliers': int(fit['keep'].sum()),
            'windows': len(windows), 'stepMedian': float(np.median(steps))}


def curb_x(curb, z):
    return curb['a'] + curb['b'] * z


def curb_signed_distance(curb, x, z):
    """Perpendicular distance from the curb face line, positive toward the road (-x side)."""
    return (curb_x(curb, z) - x) / np.hypot(1.0, curb['b'])


def along_curb(curb, x, z):
    """Coordinate along the curb direction (increasing toward the camera)."""
    return (x * curb['b'] + z) / np.hypot(1.0, curb['b'])


def curb_point(curb, along, offset):
    """Ground point at perpendicular distance `offset` from the curb face (road side) and curb coordinate `along`."""
    a, b = curb['a'], curb['b']
    n = np.hypot(1.0, b)
    z0 = (along * n - a * b) / (1 + b * b)
    return np.array([a + b * z0 - offset / n, 0.0, z0 + offset * b / n])


# ---------------------------------------------------------------- bird's-eye grid and crosswalk
def bilinear(img, u, v):
    H, W = img.shape
    u0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    fu, fv = u - u0, v - v0
    return (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
            + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)


def build_bev(sc, cfg):
    res = cfg['bevResolutionMeters']
    xs = np.arange(cfg['bevXMeters'][0], cfg['bevXMeters'][1], res) + res / 2
    zs = np.arange(cfg['bevZMeters'][0], cfg['bevZMeters'][1], res) + res / 2
    X, Z = np.meshgrid(xs, zs)
    u, v, depth = world_to_pixel(sc, np.stack([X, np.zeros_like(X), Z], -1))
    ok = (depth > 0.5) & (u >= 0) & (u <= sc['W'] - 1) & (v >= 0) & (v <= sc['H'] - 1)
    val = np.where(ok, bilinear(sc['gray'], np.where(ok, u, 0), np.where(ok, v, 0)), 0.0)
    return {'xs': xs, 'zs': zs, 'X': X, 'Z': Z, 'val': val, 'ok': ok, 'res': res}


def lateral_contrast(val, ok, win):
    n = ok.shape[1]
    f = np.where(ok, val, 0.0)
    cs = np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(f, 1)], 1)
    cn = np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(ok, 1)], 1)
    half = win // 2
    idx = np.arange(n)
    lo, hi = np.clip(idx - half, 0, n), np.clip(idx + half + 1, 0, n)
    cnt = cn[:, hi] - cn[:, lo]
    mean = (cs[:, hi] - cs[:, lo]) / np.maximum(cnt, 1)
    okc = ok & (cnt >= half)
    return np.where(okc, val - mean, 0.0), okc


def runs(mask, max_gap=0):
    """Contiguous True runs as (start, end) inclusive, closing gaps of <= max_gap."""
    m = mask.copy()
    if max_gap:
        idx = np.nonzero(m)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= max_gap + 1:
                m[a:b] = True
    out, start = [], None
    for i, on in enumerate(m):
        if on and start is None:
            start = i
        if not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(m) - 1))
    return out


def detect_band(bev, C, okr, cfg):
    """Nearest band (to the camera) of rows whose lateral brightness repeats like zebra bars."""
    res = bev['res']
    F = np.fft.rfft(np.where(okr, C, 0.0), axis=1)
    freqs = np.fft.rfftfreq(C.shape[1], d=res)
    lo, hi = cfg['barPeriodRangeMeters']
    band = (freqs >= 1 / hi) & (freqs <= 1 / lo)
    n_ok = okr.sum(1)
    power = np.abs(F[:, band]) ** 2
    amp = np.sqrt(2 * power.sum(1) / (C.shape[1] * np.maximum(n_ok, 1)))
    amp[n_ok * res < cfg['minRowRoadMeters']] = 0.0
    k = 5
    amp_s = np.convolve(amp, np.ones(k) / k, mode='same')
    if not amp_s.max() > 0:
        return None
    thr = cfg['bandRelativeThreshold'] * amp_s.max()
    accepted = []
    for s, e in runs(amp_s >= thr, max_gap=int(0.25 / res)):
        if (e - s + 1) * res >= cfg['bandMinLengthMeters']:
            accepted.append({'zFar': float(bev['zs'][s] - res / 2), 'zNear': float(bev['zs'][e] + res / 2)})
    return max(accepted, key=lambda c: c['zNear']) if accepted else None


def rotate(X, Z, th):
    c, s = np.cos(th), np.sin(th)
    return X * c - Z * s, X * s + Z * c  # l (lateral), a (along; +a toward the camera)


def unrotate(l, a, th):
    c, s = np.cos(th), np.sin(th)
    return l * c + a * s, -l * s + a * c


def profile(coord, weights, lo, res, nbins):
    idx = np.floor((coord - lo) / res).astype(int)
    k = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[k], weights=weights[k], minlength=nbins)
    n = np.bincount(idx[k], minlength=nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def detect_bars(bev, C, okr, band, cfg):
    """Bar direction (rotation search), bar positions, each bar's near end, and the near-edge line."""
    res = bev['res']
    zf, zn = band['zFar'], band['zNear']
    rows = (bev['zs'] >= zf - 0.5) & (bev['zs'] <= zn + 0.5)
    X, Z, W8, M = bev['X'][rows], bev['Z'][rows], C[rows], okr[rows]
    X, Z, W8 = X[M], Z[M], W8[M]
    lmin, lmax = bev['xs'][0] - 6, bev['xs'][-1] + 6
    nbins = int((lmax - lmin) / res)
    t0, t1, ts = cfg['orientationSearchDegrees']
    thetas = np.deg2rad(np.arange(t0, t1 + ts / 2, ts))
    scores = []
    for th in thetas:
        l, _ = rotate(X, Z, th)
        P, n = profile(l, W8, lmin, res, nbins)
        good = n >= 0.5 * np.median(n[n > 0])
        scores.append(float(np.nanvar(P[good])))
    scores = np.array(scores)
    i = int(scores.argmax())
    th = thetas[i]
    if 0 < i < len(scores) - 1:
        y0, y1, y2 = scores[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            th = th + 0.5 * (y0 - y2) / den * (thetas[1] - thetas[0])
    l, a = rotate(X, Z, th)
    P, n = profile(l, W8, lmin, res, nbins)
    good = (n >= 0.5 * np.median(n[n > 0])) & ~np.isnan(P)
    Pg = np.where(good, P, np.nan)
    thr = (np.nanpercentile(Pg, 90) + np.nanpercentile(Pg, 10)) / 2
    centers_l = lmin + (np.arange(nbins) + 0.5) * res
    bars = []
    for s, e in runs(np.nan_to_num(Pg, nan=-1e9) > thr):
        w = (e - s + 1) * res
        if not (cfg['barWidthRangeMeters'][0] <= w <= cfg['barWidthRangeMeters'][1]):
            continue
        wts = Pg[s:e + 1] - thr
        bars.append({'l': float(np.sum(centers_l[s:e + 1] * wts) / wts.sum()), 'width': float(w)})
    # Keep the longest chain with consistent spacing (drops isolated lane lines / curb edges).
    if len(bars) >= 3:
        sp = np.diff([b['l'] for b in bars])
        med = float(np.median(sp))
        chain, best = [0], [0]
        for j in range(1, len(bars)):
            if sp[j - 1] <= 1.6 * med:
                chain.append(j)
            else:
                chain = [j]
            if len(chain) > len(best):
                best = list(chain)
        bars = [bars[j] for j in best]
    # Along-road extent of each bar (wider along-road window around the band).
    rows2 = (bev['zs'] >= zf - 3) & (bev['zs'] <= zn + 3)
    X2, Z2, W2, M2 = bev['X'][rows2], bev['Z'][rows2], C[rows2], okr[rows2]
    X2, Z2, W2 = X2[M2], Z2[M2], W2[M2]
    l2, a2 = rotate(X2, Z2, th)
    amin = float(a2.min())
    nab = int((a2.max() - amin) / res) + 1
    a_centers = amin + (np.arange(nab) + 0.5) * res
    band_a = (a_centers >= zf) & (a_centers <= zn)
    for b in bars:
        k = np.abs(l2 - b['l']) <= max(0.3 * b['width'], 0.06)
        Q, nq = profile(a2[k], W2[k], amin, res, nab)
        Q = np.where(nq >= 2, Q, np.nan)
        level = np.nanmedian(Q[band_a])
        thr_k = 0.5 * level
        above = np.nan_to_num(Q, nan=-1e9) > thr_k
        best = None
        mid = int(np.argmin(np.abs(a_centers - (zf + zn) / 2)))
        for s, e in runs(above, max_gap=3):
            if s <= mid <= e or best is None or (abs((s + e) / 2 - mid) < abs((best[0] + best[1]) / 2 - mid)):
                best = (s, e)
                if s <= mid <= e:
                    break
        if best is None or not (level > 0):
            continue
        s, e = best

        def cross(j, step):
            q0, q1 = Q[j], Q[j + step] if 0 <= j + step < nab else np.nan
            if np.isnan(q1) or q0 == q1:
                return a_centers[j] + step * res / 2
            return a_centers[j] + step * res * (q0 - thr_k) / (q0 - q1)
        b['aNear'] = float(cross(e, +1))
        b['measured'] = True
    ok_bars = [b for b in bars if b.get('measured')]
    edge = None
    if len(ok_bars) >= 3:
        L = np.array([b['l'] for b in ok_bars])
        A = np.array([b['aNear'] for b in ok_bars])
        fit = robust_line(L, A, 0.3)
        if fit is not None:
            for b, kp in zip(ok_bars, fit['keep']):
                b['edgeInlier'] = bool(kp)
            edge = {'alpha': fit['a'], 'beta': fit['b'], 'inliers': int(fit['keep'].sum())}
    return {'theta': float(th), 'bars': bars, 'edge': edge}


def edge_point(bars, l):
    """Plane-grid near-edge line at lateral coordinate l (used only to bound the lane-line search)."""
    e, th = bars['edge'], bars['theta']
    x, z = unrotate(l, e['alpha'] + e['beta'] * l, th)
    return np.array([x, 0.0, z])


def image_edge(sc, bars):
    """Bar near ends mapped back to the image (the grid is an invertible resampling of the photo)."""
    pts = []
    for b in bars['bars']:
        if b.get('measured') and b.get('edgeInlier'):
            x, z = unrotate(b['l'], b['aNear'], bars['theta'])
            u, v, _ = world_to_pixel(sc, ground(x, z))
            pts.append((float(u), float(v)))
    if len(pts) < 4:
        return None
    U, V = np.array(pts).T
    return {'coef': np.polyfit(U, V, 2), 'uMin': float(U.min()), 'bars': len(pts)}


def lane_line_distance(bev, C, okr, curb, bars):
    """Curb-side lane width: strongest bright lateral peak 1-5 m from the curb face before the crosswalk."""
    res = bev['res']
    X, Z = bev['X'], bev['Z']
    edge_z = edge_point(bars, 0.0)[2]
    rows = (Z > edge_z + 1.0) & (Z < -8.0) & okr
    s = curb_signed_distance(curb, X[rows], Z[rows])
    P, n = profile(s, C[rows], 0.0, res, int(5.0 / res))
    centers = (np.arange(len(P)) + 0.5) * res
    win = (centers >= 1.0) & (centers <= 5.0) & (n > 100)
    if not win.any():
        return None
    return float(centers[int(np.nanargmax(np.where(win, P, -np.inf)))])


def measured_edge(sc, world, img_edge, curb, lane_width, gated):
    """Gated measured pixels on the near-edge image row; their own depth gives the edge's along-road position."""
    H, W = sc['H'], sc['W']
    vv, uu = np.mgrid[0:H, 0:W]
    E = gated & (np.abs(vv - np.polyval(img_edge['coef'], uu)) <= 0.5) & (uu >= img_edge['uMin'] - 5)
    cd = curb_signed_distance(curb, world[..., 0], world[..., 2])
    k = E & (cd >= 0.0) & (cd <= lane_width)
    if k.sum() < 3:
        return E, None, int(k.sum())
    return E, float(np.median(along_curb(curb, world[..., 0][k], world[..., 2][k]))), int(k.sum())


def choose_pixel(world, d, plane_depth, cand, target, curb=None, along_ref=None):
    """Nearest gated measured road pixel to a ground target (x/z distance), or None beyond the limit."""
    g = CFG['selection']
    if along_ref is None:
        cand = cand & (np.nan_to_num(np.abs(d - plane_depth), nan=np.inf) <= g['maxPlaneDepthResidualMeters'])
    else:
        cand = cand & (np.abs(along_curb(curb, world[..., 0], world[..., 2]) - along_ref) <= g['maxEdgeAlongDeviationMeters'])
    dist = np.hypot(world[..., 0] - target[0], world[..., 2] - target[2])
    vs, us = np.nonzero(cand & (dist <= g['maxGroundDistanceMeters']))
    if not len(vs):
        return None
    i = int(np.argsort(dist[vs, us], kind='stable')[0])
    u, v = int(us[i]), int(vs[i])
    return {'pixel': [u, v], 'world': world[v, u].astype(float), 'offset': float(dist[v, u])}


# ---------------------------------------------------------------- detections (from scripts/detect-vehicles.py)
def load_detections(path, sc):
    """The detections document and the sha256 of its bytes (recorded in suggestions.json so the count is traceable)."""
    try:
        data = Path(path).read_bytes()
        doc = json.loads(data)
    except (OSError, ValueError) as exc:
        fail(f'cannot read detections: {exc}')
    model = doc.get('model') if isinstance(doc, dict) else None
    if (doc.get('schemaVersion') != 1 or doc.get('sceneId') != sc['meta']['id'] or not isinstance(model, dict)
            or not isinstance(doc.get('detections'), list)):
        fail('detections file is not schemaVersion 1 for this scene')
    for name in ('recorded.jpg', 'depth.bin'):
        if doc.get('inputs', {}).get(name) != sc['inputs'][name]:
            fail(f'detections were made from a different {name}')
    for key in ('name', 'package', 'license', 'weightsSha256'):
        if not isinstance(model.get(key), str) or not model[key]:
            fail(f'detections model.{key} missing')
    return doc, hashlib.sha256(data).hexdigest()


def vehicle_from_detections(doc, curb, along_edge, cross_world, gated, world, d, plane_depth, offset, tol):
    """Curb-lane parked vehicles between the camera and the crossing. Returns (count, suggestion or None)."""
    cands = [r for r in doc['detections'] if (r.get('curbLaneCandidate') or {}).get('eligible') is True
             and isinstance(r.get('lowerMedianWorld'), list) and len(r['lowerMedianWorld']) == 3]
    if along_edge is not None:
        cands = [r for r in cands if along_curb(curb, r['lowerMedianWorld'][0], r['lowerMedianWorld'][2]) > along_edge]
    if not cands or along_edge is None:
        return len(cands), None
    # The vehicle nearest the crossing governs the crossing view.
    near = min(cands, key=lambda r: along_curb(curb, r['lowerMedianWorld'][0], r['lowerMedianWorld'][2]))
    name = f"Parked {near['class']} detected (score {near['score']:.2f})"
    body = near.get('bodyForwardRangeM')
    if not body:
        return len(cands), unknown(f'{name}, but its front has too little measured depth.')
    # Crossing-side end: far (P90) forward extent of its measured body points.
    along_front = float(along_curb(curb, near['lowerMedianWorld'][0], -body[1]))
    if along_front <= along_edge + CFG['crossingAlongRoadUncertaintyMeters']:
        return len(cands), unknown(f'{name}, but it reaches the crossing.')
    pick = choose_pixel(world, d, plane_depth, gated, curb_point(curb, along_front, offset))
    if pick is None:
        return len(cands), unknown(f'{name}, but no measured road pixel lies at its front.')
    setback = float(np.hypot(*(pick['world'] - cross_world)[[0, 2]]))
    if setback <= 2 * tol:
        return len(cands), unknown(f'{name}, but its front is not separated from the crossing.')
    # Schema v1 detects only the front point; the label and dimensions stay the declared default vehicle.
    v = DECLARED['vehicle']
    return len(cands), {'pixel': pick['pixel'], 'status': 'detected', 'label': v['label'],
                        'dimensions': dict(v['dimensions']), 'setbackM': round(setback, 3),
                        'reason': f'{name}; front from measured depth; {v["label"]} dimensions declared.'}


def lane_census(world, has, curb, along_edge, lane_width):
    """Measured points at body height in the curb lane between the camera and the crossing."""
    c = CFG['laneCensus']
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    cd, al = curb_signed_distance(curb, x, z), along_curb(curb, x, z)
    lane = has & (cd >= CFG['selection']['minCurbClearanceMeters']) & (cd <= lane_width)
    lane &= (al >= along_edge) & (al <= -c['nearestForwardMeters'])
    return int((lane & (y >= c['minHeightMeters']) & (y <= c['maxHeightMeters'])).sum())


# ---------------------------------------------------------------- main
def generated_utc():
    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    try:
        t = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    except ValueError:
        fail('SOURCE_DATE_EPOCH must be an integer')
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def suggest(sc, detections, detections_sha256=None):
    cfg, g = CFG, CFG['selection']
    world, d, plane_depth = measured_world(sc)
    has = sc['raw'] > 0
    road_rule = has & (np.abs(world[..., 1]) <= 2 * sc['tol'])   # Sightline roadPoint rule

    curb = detect_curb(world, has, cfg)
    bev = build_bev(sc, cfg)
    road_bev = bev['ok'].copy()
    if curb:
        road_bev &= curb_signed_distance(curb, bev['X'], bev['Z']) >= 0.05
    C, okr = lateral_contrast(bev['val'], road_bev, int(round(cfg['lateralBackgroundWindowMeters'] / bev['res'])))
    band = detect_band(bev, C, okr, cfg)
    bars = detect_bars(bev, C, okr, band, cfg) if band else None
    img_edge = image_edge(sc, bars) if bars and bars['edge'] else None

    out = {}
    out['curb'] = (unknown('No persistent curb height step in measured depth.') if not curb else
                   {'status': 'detected', 'rmsM': round(curb['rms'], 3), 'stepHeightM': round(curb['stepMedian'], 3)})

    axis = None   # unit road direction toward the camera, set only when the bar cross-check agrees
    axis_curb = np.array([curb['b'], 0.0, 1.0]) / np.hypot(1.0, curb['b']) if curb else None
    if not curb:
        out['roadAxis'] = unknown('Curb not measured.')
    elif not bars:
        out['roadAxis'] = unknown('No crosswalk bars to cross-check the curb direction.')
    else:
        axis_bars = np.array([np.sin(bars['theta']), 0.0, np.cos(bars['theta'])])
        agree = float(np.degrees(np.arccos(np.clip(axis_curb @ axis_bars, -1, 1))))
        if agree > cfg['maxAxisDisagreementDegrees']:
            out['roadAxis'] = unknown(f'Curb and crosswalk bar directions disagree by {agree:.1f} deg.')
        else:
            axis = axis_curb
            out['roadAxis'] = {'status': 'detected', 'source': 'curb face fit', 'barAgreementDeg': round(agree, 2)}

    vdecl, pdecl = DECLARED['vehicle'], DECLARED['person']
    offset = vdecl['curbGapMeters'] + vdecl['dimensions']['width'] / 2   # declared vehicle centreline
    along_edge, cross, lane_width, edge_mask, gated = None, None, None, None, None
    if not curb:
        out['crossing'] = unknown('Curb not measured.')
    elif not band:
        out['crossing'] = unknown('No crosswalk stripe band found.')
    elif not img_edge:
        out['crossing'] = unknown('Crosswalk near edge not found.')
    else:
        lane_width = lane_line_distance(bev, C, okr, curb, bars) or cfg['fallbackCurbLaneWidthMeters']
        gated = road_rule & (np.abs(world[..., 1]) <= g['maxAbsHeightMeters'])
        gated &= curb_signed_distance(curb, world[..., 0], world[..., 2]) >= g['minCurbClearanceMeters']
        edge_mask, along_edge, _ = measured_edge(sc, world, img_edge, curb, lane_width, gated)
        if along_edge is None:
            out['crossing'] = unknown('Too little measured depth on the crosswalk near edge.')
        else:
            # Sightline centres the vehicle box on the crossing-to-front line, so the crossing point sits on the
            # declared vehicle centreline: declared curb gap + half the declared width from the curb face.
            cross = choose_pixel(world, d, plane_depth, edge_mask, curb_point(curb, along_edge, offset), curb, along_edge)
            if cross is None:
                out['crossing'] = unknown('No measured road pixel at the crosswalk near edge.')
            else:
                edge_curb = curb_point(curb, along_edge, 0.0)
                out['crossing'] = {
                    'pixel': cross['pixel'], 'status': 'detected',
                    'method': 'zebra-stripe detector on the depth-registered ground plane',
                    'alongRoadUncertaintyM': cfg['crossingAlongRoadUncertaintyMeters'],
                    'distanceFromCameraM': round(float((sc['eye'] - edge_curb) @ axis_curb), 2),
                    'barCount': img_edge['bars']}

    det_count = None
    if cross is None:
        out['vehicle'] = unknown('Crossing is unknown.')
    elif axis is None:
        out['vehicle'] = unknown('Road axis is unknown.')
    else:
        found = None
        if detections:
            det_count, found = vehicle_from_detections(detections, curb, along_edge, cross['world'], gated,
                                                       world, d, plane_depth, offset, sc['tol'])
        if found:
            out['vehicle'] = found
        else:
            census = lane_census(world, has, curb, along_edge, lane_width)
            target = np.array([cross['world'][0], 0.0, cross['world'][2]]) + axis * vdecl['setbackMeters']
            pick = choose_pixel(world, d, plane_depth, gated, target)
            if census > cfg['laneCensus']['maxPoints']:
                out['vehicle'] = unknown(f'{census} measured points stand in the curb lane but no parked vehicle was detected.')
            elif pick is None:
                out['vehicle'] = unknown('No measured road pixel at the 20 ft default front.')
            else:
                why = 'No parked vehicle detected in the curb lane' if detections else 'Vehicle detector not run'
                out['vehicle'] = {'pixel': pick['pixel'], 'status': 'declared-default', 'label': vdecl['label'],
                                  'dimensions': dict(vdecl['dimensions']), 'setbackM': vdecl['setbackMeters'],
                                  'reason': f'{why}; 20 ft default.'}
    if detections and det_count is None:
        det_count = len([r for r in detections['detections'] if (r.get('curbLaneCandidate') or {}).get('eligible') is True])

    if cross is None:
        out['person'] = unknown('Crossing is unknown.')
    else:
        pick = choose_pixel(world, d, plane_depth, edge_mask, curb_point(curb, along_edge, pdecl['curbOffsetMeters']),
                            curb, along_edge)
        out['person'] = unknown('No measured road pixel at the curb end of the crossing.') if pick is None else {
            'pixel': pick['pixel'], 'status': 'declared-default', 'label': pdecl['label'],
            'heightM': pdecl['heightMeters'], 'reason': 'Curb end of the crossing near edge.'}

    if detections:
        m = detections['model']
        out['detection'] = {'status': 'ran', 'model': m['name'], 'package': m['package'], 'license': m['license'],
                            'weightsSha256': m['weightsSha256'], 'detectionsSha256': detections_sha256,
                            'curbLaneParked': det_count}
    else:
        out['detection'] = {'status': 'not-run'}

    limitations = ['One frame; no independent reference. The hand test annotation is not ground truth.']
    if out['crossing']['status'] == 'detected':
        limitations.append('Crossing is the near edge of the crosswalk in the curb lane. Measured depth near it clusters '
                           'about 1 m apart; the nearer cluster is used, so it is uncertain by about 0.5 m along the road.')
        limitations.append('The near-edge angle is not measured reliably and is not used; the road axis follows the curb.')
    if out['curb']['status'] == 'detected':
        limitations.append('The curb face is the half-height point of a measured transition about 0.3 m wide; '
                           'the declared 0.15 m curb gap is measured from it.')
    if out['vehicle']['status'] == 'declared-default':
        limitations.append('No parked vehicle is measured here: the vehicle is a declared what-if with default SUV size.')
    if out['person']['status'] == 'declared-default':
        limitations.append('The person position and 1.1 m height are declared defaults, not detections.')
    limitations.append('Geometric visibility only; not parking safety, recognition or compliance.')

    return {'schemaVersion': 1, 'sceneId': sc['meta']['id'], 'generatedUtc': generated_utc(),
            'generator': {'script': SCRIPT, 'sha256': sha256(__file__), 'python': platform.python_version(),
                          'numpy': np.__version__, 'pillow': PIL.__version__},
            'inputs': sc['inputs'],
            'crossing': out['crossing'], 'roadAxis': out['roadAxis'], 'curb': out['curb'],
            'vehicle': out['vehicle'], 'person': out['person'], 'detection': out['detection'],
            'limitations': limitations}


def update_catalog(scene_dir, scene_id, digest):
    catalog_path = scene_dir.parent / 'index.json'
    if not catalog_path.is_file():
        return 'no catalog beside the scene'
    catalog = json.loads(catalog_path.read_text())
    entries = [e for e in catalog if isinstance(e, dict) and e.get('id') == scene_id] if isinstance(catalog, list) else []
    if len(entries) != 1:
        return 'scene not listed in the catalog'
    entries[0]['suggestions'] = {'file': OUTPUT, 'sha256': digest}
    tmp = catalog_path.with_name(catalog_path.name + '.tmp')
    tmp.write_text(json.dumps(catalog, indent=2) + '\n')
    os.replace(tmp, catalog_path)
    return f'catalog entry updated: {catalog_path}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scene', type=Path, help='prepared scene directory (scene.json, recorded.jpg, depth.bin)')
    ap.add_argument('--detections', type=Path, help='detections JSON from scripts/detect-vehicles.py')
    ap.add_argument('--replace', action='store_true', help='overwrite an existing suggestions.json')
    args = ap.parse_args()
    scene_dir = args.scene.resolve()
    target = scene_dir / OUTPUT
    if target.exists() and not args.replace:
        fail(f'{target} exists; pass --replace to overwrite')
    t = time.perf_counter()
    sc = load_scene(scene_dir)
    detections, detections_sha256 = load_detections(args.detections, sc) if args.detections else (None, None)
    doc = suggest(sc, detections, detections_sha256)
    elapsed = time.perf_counter() - t
    data = (json.dumps(doc, indent=2) + '\n').encode()
    tmp = target.with_name(OUTPUT + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, target)
    digest = hashlib.sha256(data).hexdigest()
    note = update_catalog(scene_dir, doc['sceneId'], digest)
    for key in ('crossing', 'roadAxis', 'curb', 'vehicle', 'person', 'detection'):
        item = doc[key]
        print(f"{key}: {item['status']}" + (f" {item['pixel']}" if 'pixel' in item else '')
              + (f" ({item['reason']})" if 'reason' in item else ''), file=sys.stderr)
    print(f'{elapsed:.2f} s; {note}; wrote {target}', file=sys.stderr)
    print(digest)


if __name__ == '__main__':
    main()

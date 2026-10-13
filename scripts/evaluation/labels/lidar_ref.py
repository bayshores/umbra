"""LiDAR known answers for DEVELOPMENT validation cases (car silhouettes and crosswalk-paint near edges).

Car: stack many sweeps in the city frame (the car is stationary), keep voxels seen in >= minSweeps sweeps
(drops passing objects), fit a local ground plane by RANSAC, keep points above the plane, and grow one
Euclidean cluster from points near the seed. The AV2 box is used only for the seed position and the
ROI radius class; the reference extent comes only from LiDAR points. Points are projected into the
target camera frame; extremes and per-column top/bottom profiles are robust order statistics (the
`robustRank`-th most extreme point). LiDAR samples the surface, so these are INNER bounds of the true
silhouette, with a sampling-gap estimate (median nearest-neighbour spacing of projected points near the
boundary). The lowest returns on tires are above the true contact, so bottom values are one-sided.

Paint: ground returns (|height| <= groundBandM from a RANSAC plane) in a crossing ROI, intensity >=
paintRatio x road median = paint. Coordinates: w along the mapped near edge, s along its normal into the
crossing (the map only defines this frame and the region). Per lateral bin, the near edge is the first
road->paint transition followed by a paint run >= minBarRunM (rejects short yield triangles). Each bar's
edge points get a robust line fit; its projected segment gives the reference row at any column.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from synth import fit_plane


def stack(log_obj, sweeps):
    xyz, intensity, sweep_index = [], [], []
    for k, t in enumerate(sweeps):
        s = log_obj.sweep_city(t)
        xyz.append(s['xyz'])
        intensity.append(s['intensity'])
        sweep_index.append(np.full(len(s['xyz']), k))
    return np.vstack(xyz), np.concatenate(intensity), np.concatenate(sweep_index)


def persistent(xyz, sweep_index, voxel, min_sweeps):
    keys = np.floor(xyz / voxel).astype(np.int64)
    packed = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    order = np.lexsort((sweep_index, packed))
    pk, sw = packed[order], sweep_index[order]
    first = np.r_[True, (pk[1:] != pk[:-1]) | (sw[1:] != sw[:-1])]
    unique_keys = pk[first]
    k_values, k_counts = np.unique(unique_keys, return_counts=True)
    count_of = dict(zip(k_values.tolist(), k_counts.tolist()))
    return np.array([count_of[k] >= min_sweeps for k in packed.tolist()])


def ground_plane(points, cell=1.0, band=0.15, threshold=0.05, seed=0):
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    low = np.zeros(len(points), bool)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for g in np.unique(inverse):
        idx = np.where(inverse == g)[0]
        floor = np.percentile(points[idx, 2], 5)
        low[idx[points[idx, 2] <= floor + band]] = True
    coef, _ = fit_plane(points[low], threshold=threshold, seed=seed)
    height = points[:, 2] - (points[:, :2] @ coef[:2] + coef[2])
    return coef, height


def components(points, radius):
    """Connected-component labels of the radius graph (Euclidean clustering)."""
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1]


def robust_extreme(values, rank, largest):
    ordered = np.sort(values)
    k = min(rank, len(ordered)) - 1
    return float(ordered[-1 - k] if largest else ordered[k])


def image_spacing(u, v, mask):
    if mask.sum() < 3:
        return None
    pts = np.column_stack([u[mask], v[mask]])
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


def car_reference(log_obj, case, sweeps, frame_sweep, params, return_points=False):
    p = params
    xyz, intensity, sweep_index = stack(log_obj, sweeps)
    seed = np.array(case['seedCity'])
    radius = p['roiRadiusM'].get(case['category'], p['roiRadiusM']['default'])
    roi = np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) <= radius + p['groundMarginM']
    xyz, sweep_index = xyz[roi], sweep_index[roi]
    keep = persistent(xyz, sweep_index, p['voxelM'], p['minSweeps'])
    dropped = int((~keep).sum())
    xyz, sweep_index = xyz[keep], sweep_index[keep]
    coef, height = ground_plane(xyz, threshold=p['groundRansacM'])
    inside = np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) <= radius
    max_h = p['maxHeightM'].get(case['category'], p['maxHeightM']['default'])
    body = inside & (height > p['minHeightM']) & (height < max_h)
    idx = np.where(body)[0]
    labels = components(xyz[idx], p['clusterRadiusM'])
    near_seed = np.linalg.norm(xyz[idx, :2] - seed[:2], axis=1) <= p['seedRadiusM']
    if not near_seed.any():
        return {'status': 'unresolved', 'reason': 'no LiDAR points near seed'}
    values, counts = np.unique(labels[near_seed], return_counts=True)
    chosen = values[np.argmax(counts)]
    cluster = idx[labels == chosen]
    # attach low returns (tires, bumper bottom) next to low cluster points
    low = np.where(inside & (height > p['lowMinHeightM']) & (height <= p['minHeightM']))[0]
    base = cluster[height[cluster] < p['lowAttachBelowM']]
    added = 0
    if len(low) and len(base):
        d, _ = cKDTree(xyz[base]).query(xyz[low])
        attach = low[d <= p['lowAttachM']]
        cluster = np.concatenate([cluster, attach])
        added = len(attach)
    pts = xyz[cluster]
    centered = pts[:, :2] - pts[:, :2].mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    extent = np.ptp(centered @ axes.T, axis=0)
    qa = {'points': int(len(cluster)), 'sweepsUsed': int(len(np.unique(sweep_index[cluster]))), 'persistenceDropped': dropped,
          'lowPointsAdded': int(added), 'extentM': [round(float(e), 3) for e in extent],
          'heightMaxM': round(float(height[cluster].max()), 3), 'groundPlane': [float(c) for c in coef]}
    if len(cluster) < p['minPoints']:
        return {'status': 'unresolved', 'reason': 'too few cluster points', 'qa': qa}
    size = case['annotatedSizeM']  # QA cap only (catches merges with neighbours); never a reference
    limit = [size[0] + p['extentSlackM'][0], size[1] + p['extentSlackM'][1]]
    if extent[0] > limit[0] or extent[1] > limit[1]:
        return {'status': 'unresolved', 'reason': f'cluster extent {extent.round(2).tolist()} exceeds {limit}', 'qa': qa}
    # per-sweep consistency (pose agreement): distance of each sweep's points to the other sweeps' points
    consistency = []
    for k in np.unique(sweep_index[cluster]):
        mine = cluster[sweep_index[cluster] == k]
        other = cluster[sweep_index[cluster] != k]
        if len(mine) >= 30 and len(other) >= 30:
            d, _ = cKDTree(xyz[other]).query(xyz[mine])
            consistency.append(float(np.median(d)))
    qa['perSweepMedianNnM'] = {'median': round(float(np.median(consistency)), 4) if consistency else None,
                               'max': round(float(np.max(consistency)), 4) if consistency else None}
    u, v, z = log_obj.project(pts, case['frame'])
    front = z > 0.5
    u, v, z = u[front], v[front], z[front]
    rank = p['robustRank']
    ref = {'L': robust_extreme(u, rank, False), 'R': robust_extreme(u, rank, True),
           'T': robust_extreme(v, rank, False), 'B': robust_extreme(v, rank, True)}
    band = p['spacingBandPx']
    spacing = {'L': image_spacing(u, v, u <= ref['L'] + band), 'R': image_spacing(u, v, u >= ref['R'] - band),
               'T': image_spacing(u, v, v <= ref['T'] + band), 'B': image_spacing(u, v, v >= ref['B'] - band)}
    columns = np.arange(int(np.ceil(ref['L'])), int(np.floor(ref['R'])) + 1)
    half = p['columnHalfWidthPx']
    top, bottom = {}, {}
    for c in columns:
        m = np.abs(u - c) <= half
        if m.sum() >= p['minColumnPoints']:
            top[int(c)] = robust_extreme(v[m], rank, False)
            bottom[int(c)] = robust_extreme(v[m], rank, True)
    # occlusion: non-cluster, above-ground returns of the frame sweep that project onto the car and are nearer
    others = np.setdiff1d(np.where(height > p['minHeightM'])[0], cluster)
    occ_all, occ_depth = occluder_projection(log_obj, case['frame'], frame_sweep, xyz[others], sweep_index[others], coef,
                                             {'occluderVoxelM': p['voxelM'], 'occluderMinHeightM': p['minHeightM']})
    fu, fv, fz = occ_all[:, 0], occ_all[:, 1], occ_depth
    car_depth = float(np.percentile(z, 5))
    occluders = (fz < car_depth - p['occluderDepthGapM']) & \
                (fu >= ref['L'] - band) & (fu <= ref['R'] + band) & (fv >= ref['T'] - band) & (fv <= ref['B'] + band)
    occ_uv = np.column_stack([fu[occluders], fv[occluders]])
    def occluded_near(point):
        if not len(occ_uv):
            return False
        return bool((np.linalg.norm(occ_uv - np.array(point), axis=1) <= p['occluderRadiusPx']).any())
    occlusion = {'occluderPoints': int(occluders.sum())}
    vmid = (ref['T'] + ref['B']) / 2
    umid = (ref['L'] + ref['R']) / 2
    occlusion['L'] = occluded_near((ref['L'], v[np.argmin(u)]))
    occlusion['R'] = occluded_near((ref['R'], v[np.argmax(u)]))
    occlusion['T'] = occluded_near((u[np.argmin(v)], ref['T']))
    occlusion['B'] = occluded_near((u[np.argmax(v)], ref['B']))
    if return_points:
        return {'u': u, 'v': v, 'z': z, 'reference': ref}
    return {'status': 'resolved', 'kind': 'car', 'reference': ref, 'spacingPx': spacing, 'topProfile': top, 'bottomProfile': bottom,
            'occlusion': occlusion, 'qa': qa, 'medianDepthM': round(float(np.median(z)), 3),
            'projectedCentroidPx': [round(umid, 1), round(vmid, 1)],
            'semantics': 'LiDAR inner bound: true silhouette extends outward by up to about spacingPx (plus beam width, calibration); '
                         'B and bottomProfile are one-sided (lowest returns sit above the tire contact).'}


def paint_reference(log_obj, case, sweeps, frame_sweep, params, crossing):
    p = params
    xyz, intensity, sweep_index = stack(log_obj, sweeps)
    centre = np.array(case['aheadCity'])
    roi = np.linalg.norm(xyz[:, :2] - centre[:2], axis=1) <= p['roiRadiusM']
    xyz, intensity = xyz[roi], intensity[roi]
    sweep_roi = sweep_index[roi]
    coef, height = ground_plane(xyz, threshold=p['groundRansacM'])
    ground = np.abs(height) <= p['groundBandM']
    near = np.array(crossing['near'])
    far = np.array(crossing['far'])
    e = (near[-1] - near[0])[:2]
    e /= np.linalg.norm(e)
    n = np.array([-e[1], e[0]])
    if ((far.mean(0) - near.mean(0))[:2] @ n) < 0:
        n = -n
    origin = near[0][:2]
    w = (xyz[:, :2] - origin) @ e
    s = (xyz[:, :2] - origin) @ n
    road_band = ground & (s >= p['roadBandM'][0]) & (s <= p['roadBandM'][1])
    road_median = float(np.median(intensity[road_band]))
    threshold = p['paintRatio'] * road_median
    paint_pt = intensity >= threshold
    sel = ground & (s >= p['searchM'][0]) & (s <= p['searchM'][1])
    edges = []
    for w0 in np.arange(w[sel].min(), w[sel].max(), p['lateralBinM']):
        m = sel & (w >= w0) & (w < w0 + p['lateralBinM'])
        if m.sum() < p['minBefore'] + p['minAfter']:
            continue
        found = step_edge(s[m], paint_pt[m], p)
        if found is not None:
            edges.append((float(w0 + p['lateralBinM'] / 2), found[0], found[1]))
    if not edges:
        return {'status': 'unresolved', 'reason': 'no paint edge found', 'roadMedianIntensity': road_median}
    edges = np.array([(a, b, np.nan if c is None else c) for a, b, c in edges])
    # group consecutive lateral bins into bars
    order = np.argsort(edges[:, 0])
    edges = edges[order]
    groups, current = [], [0]
    for i in range(1, len(edges)):
        if edges[i, 0] - edges[i - 1, 0] <= p['lateralBinM'] * 1.5 and abs(edges[i, 1] - edges[i - 1, 1]) <= p['maxStepM']:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    ground_tree, ground_z = cKDTree(xyz[ground, :2]), xyz[ground, 2]
    occ_uv, occ_z = occluder_projection(log_obj, case['frame'], frame_sweep, xyz[height > p['occluderMinHeightM']],
                                        sweep_roi[height > p['occluderMinHeightM']], coef, p)
    bars = []
    for g in groups:
        if len(g) < p['minBarBins']:
            continue
        ws, ss = edges[g, 0], edges[g, 1]
        if len(g) >= 3:
            beta, alpha = np.polyfit(ws, ss, 1)
        else:
            beta, alpha = 0.0, float(np.median(ss))
        resid = ss - (alpha + beta * ws)
        lo, hi = ws.min() + p['barTrimM'], ws.max() - p['barTrimM']
        if hi <= lo:
            continue
        ends = []
        for wq in (lo, hi):
            xy = origin + wq * e + (alpha + beta * wq) * n
            ends.append([xy[0], xy[1], local_height(ground_tree, ground_z, xy, coef, p)])
        ends = np.array(ends)
        u, v, z = log_obj.project(ends, case['frame'])
        mid = ends.mean(0)
        mu, mv, mz = log_obj.project(mid[None], case['frame'])
        occluded = False
        if len(occ_uv):
            for q in (0.0, 0.5, 1.0):  # both ends and the middle of the bar edge
                pu, pv = u[0] + q * (u[1] - u[0]), v[0] + q * (v[1] - v[0])
                near_px = np.linalg.norm(occ_uv - np.array([pu, pv]), axis=1) <= p['occluderRadiusPx']
                occluded |= bool((near_px & (occ_z < mz[0] - p['occluderDepthGapM'])).any())
        # along-s to image-row derivative at the bar centre
        step = log_obj.project((mid + np.r_[0.05 * n, 0])[None], case['frame'])[1][0] - mv[0]
        bars.append({'wRangeM': [round(float(lo), 3), round(float(hi), 3)], 'sEdgeM': round(float(alpha + beta * (lo + hi) / 2), 4),
                     'residualStdM': round(float(resid.std()), 4), 'bins': len(g), 'alongGapM': None if np.all(np.isnan(edges[g, 2])) else round(float(np.nanmedian(edges[g, 2])), 4),
                     'segmentPx': [[round(float(u[0]), 3), round(float(v[0]), 3)], [round(float(u[1]), 3), round(float(v[1]), 3)]],
                     'pxPerMetreAlong': round(float(step / 0.05), 3), 'occluded': occluded, 'depthM': round(float(mz[0]), 3)})
    if bars:
        consensus = float(np.median([b['sEdgeM'] for b in bars]))
        for b in bars:
            b['consensus'] = abs(b['sEdgeM'] - consensus) <= p['consensusM']
    else:
        consensus = None
    return {'status': 'resolved', 'kind': 'paint', 'roadMedianIntensity': road_median, 'threshold': threshold, 'bars': bars,
            'consensusSEdgeM': consensus,
            'groundPlane': [float(c) for c in coef], 'groundReturns': int(sel.sum()),
            'semantics': 'Near edge of crosswalk bars from stacked LiDAR intensity; the map defines only the coordinate frame and region.'}


def occluder_projection(log_obj, frame, frame_sweep, above_xyz, above_sweeps, coef, p):
    """Projected (uv, depth) of possible occluders: persistent stacked above-ground points plus the frame sweep's."""
    keep = persistent(above_xyz, above_sweeps, p['occluderVoxelM'], 2) if len(above_xyz) else np.zeros(0, bool)
    sweep = log_obj.sweep_city(frame_sweep)['xyz']
    sweep = sweep[(sweep[:, 2] - (sweep[:, :2] @ coef[:2] + coef[2])) > p['occluderMinHeightM']]
    points = np.vstack([above_xyz[keep], sweep])
    u, v, z = log_obj.project(points, frame)
    front = z > 0.5
    return np.column_stack([u[front], v[front]]), z[front]


def local_height(tree, z_values, xy, coef, p):
    """Median height of ground returns within localGroundRadiusM (crowned roads defeat one plane); plane fallback."""
    idx = tree.query_ball_point(xy, p['localGroundRadiusM'])
    if len(idx) >= p['localGroundMinPoints']:
        return float(np.median(z_values[idx]))
    return float(coef[0] * xy[0] + coef[1] * xy[1] + coef[2])


def step_edge(s_values, is_paint, p):
    """Best road->paint step along s for one lateral strip: (edge s, gap between the bracketing returns) or None.

    Score of a split = road before + paint after - paint before - road after, counted over windows of
    roadBeforeM before and minBarRunM after the split. Accepted when the best split has >= minBefore /
    minAfter returns, road share before >= 1 - roadMaxFraction and paint share after >= runPaintShare.
    """
    order = np.argsort(s_values)
    s_sorted, y = s_values[order], is_paint[order].astype(bool)
    best = None
    for i in range(1, len(s_sorted)):
        if not (y[i] and not y[i - 1]):
            continue
        split = 0.5 * (s_sorted[i - 1] + s_sorted[i])
        before = (s_sorted >= split - p['roadBeforeM']) & (s_sorted < split)
        after = (s_sorted >= split) & (s_sorted < split + p['minBarRunM'])
        nb, na = int(before.sum()), int(after.sum())
        if nb < p['minBefore'] or na < p['minAfter']:
            continue
        road_share = 1 - y[before].mean()
        paint_share = y[after].mean()
        if road_share < 1 - p['roadMaxFraction'] or paint_share < p['runPaintShare']:
            continue
        score = (~y[before]).sum() + y[after].sum() - y[before].sum() - (~y[after]).sum()
        if best is None or score > best[0]:
            best = (score, float(split), float(s_sorted[i] - s_sorted[i - 1]))
    return None if best is None else (best[1], best[2])


def paint_row_at(bars, column, trim_px=2.0):
    """Reference row of the bar near edge at a pixel-centre column, or None."""
    for bar in bars:
        (u0, v0), (u1, v1) = bar['segmentPx']
        lo, hi = min(u0, u1) + trim_px, max(u0, u1) - trim_px
        if lo <= column <= hi and not bar['occluded'] and bar.get('consensus', True):
            t = (column - u0) / (u1 - u0)
            return v0 + t * (v1 - v0), bar
    return None, None

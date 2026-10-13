"""Choose the real DEVELOPMENT known-answer cases and the LiDAR sweeps to fetch (rule-based, no human picks).

Rules (frozen in protocol-v1.json `caseSelection`):
- Car cases: stationary vehicle tracks (city RMS spread <= maxSpreadM, >= minAnnotations rows). A frame is
  eligible when the track's median annotated box projects fully inside the raw image with >= marginPx,
  the camera-to-box range is within [minRangeM, maxRangeM] and no nearer annotated object's projected box
  overlaps it. The frame closest to targetRangeM is used. Per log the `carsPerLog` tracks with the most
  LiDAR interior points are taken; the app's current vehicle (8c 396f15b5) is always included at the app
  frame. Annotated boxes are used only to pick cases and seed LiDAR clustering, never as a reference shape.
- Paint cases: the development crossing of each log, in the frame where the mapped near-edge midpoint is
  closest to targetRangeM and both near-edge ends project inside the image (8c uses the app frame).
  The mapped crossing is used only to choose the frame and region, never as a reference.
- Sweeps: every sweep whose ego origin is within sweepRadiusM of a case target, thinned to one sweep per
  sweepSpacingM of ego travel, plus the sweep nearest each case frame.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av2io import Log, default_roots  # noqa: E402
from common import LABELS_DIR, assert_development, full_log_id, load_protocol, read_json, write_json  # noqa: E402

CROSSINGS = {'8c52d911': '2658849', '453e5558': '20499018', 'c780d53a': '2069452'}
APP_CASE = {'log': '8c52d911', 'track': '396f15b5', 'frame': 315969946899927216}


def box_corners(row):
    half = np.array([row['length_m'], row['width_m'], row['height_m']]) / 2
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    return (signs * half) @ row['city_rotation'].T + row['city_center']


def frames_for(log_short):
    listing = read_json(LABELS_DIR / f'listings/{log_short}-stereo-left.json')
    return sorted(int(k['key'].rsplit('/', 1)[1][:-4]) for k in listing['keys'])


def sweeps_for(log_short):
    listing = read_json(LABELS_DIR / f'listings/{log_short}-lidar.json')
    return sorted(int(k['key'].rsplit('/', 1)[1][:-8]) for k in listing['keys'])


def map_file(log):
    for root in default_roots(log):
        found = sorted((root / 'map').glob('log_map_archive_*.json')) if (root / 'map').exists() else []
        if found:
            return found[0]
    raise FileNotFoundError(log)


def nearest_annotation_rows(log_obj, timestamp):
    times = sorted({int(r['timestamp_ns']) for r in log_obj.annotations})
    nearest = min(times, key=lambda t: abs(t - timestamp))
    return [r for r in log_obj.annotations if int(r['timestamp_ns']) == nearest]


def box_rect(log_obj, row, timestamp):
    u, v, z = log_obj.project(box_corners(row), timestamp)
    if (z <= 0.5).any():
        return None
    return float(u.min()), float(v.min()), float(u.max()), float(v.max()), float(np.median(z))


def occluded_by_annotation(log_obj, track, rect, timestamp):
    for other in nearest_annotation_rows(log_obj, timestamp):
        if other['track_uuid'] == track:
            continue
        r = box_rect(log_obj, other, timestamp)
        if r is None:
            continue
        overlap = not (r[2] < rect[0] or r[0] > rect[2] or r[3] < rect[1] or r[1] > rect[3])
        if overlap and r[4] < rect[4]:
            return other['track_uuid'][:8]
    return None


def car_cases(log_short, log_obj, rules):
    frames = [t for t in frames_for(log_short) if t in log_obj.poses]
    tracks = [s for s in log_obj.stationary_vehicles(rules['maxSpreadM'], rules['minAnnotations'])]
    candidates = []
    for s in tracks:
        row = s['rows'][len(s['rows']) // 2]
        best = None
        for t in frames:
            rect = box_rect(log_obj, row, t)
            if rect is None:
                continue
            rng = float(np.linalg.norm((log_obj.camera_center_city(t) - s['meanCenter'])[:2]))
            m = rules['marginPx']
            inside = rect[0] >= m and rect[1] >= m and rect[2] <= log_obj.width - m and rect[3] <= log_obj.height - m
            if not inside or not (rules['minRangeM'] <= rng <= rules['maxRangeM']):
                continue
            if best is None or abs(rng - rules['targetRangeM']) < abs(best[1] - rules['targetRangeM']):
                best = (t, rng, rect)
        if best is None:
            continue
        occluder = occluded_by_annotation(log_obj, s['track'], best[2], best[0])
        interior = max(int(r['num_interior_pts']) for r in s['rows'])
        candidates.append({'kind': 'car', 'log': log_obj.log, 'track': s['track'], 'category': s['category'],
                           'frame': best[0], 'rangeM': round(best[1], 3), 'annotationRectPx': [round(x, 1) for x in best[2][:4]],
                           'maxInteriorPoints': interior, 'citySpreadM': round(s['spreadM'], 3),
                           'annotatedSizeM': [round(float(np.median([r[k] for r in s['rows']])), 3) for k in ('length_m', 'width_m', 'height_m')],
                           'seedCity': [round(float(x), 3) for x in s['meanCenter']], 'annotationOccluder': occluder})
    eligible = [c for c in candidates if c['annotationOccluder'] is None]
    eligible.sort(key=lambda c: (-c['maxInteriorPoints'], c['track']))
    chosen = eligible[:rules['carsPerLog']]
    if log_short == APP_CASE['log']:
        app = [c for c in candidates if c['track'].startswith(APP_CASE['track'])]
        if app:
            app = dict(app[0])
            app['frame'] = APP_CASE['frame']
            rect = box_rect(log_obj, [s for s in tracks if s['track'] == app['track']][0]['rows'][0], APP_CASE['frame'])
            app['rangeM'] = round(float(np.linalg.norm((log_obj.camera_center_city(APP_CASE['frame']) - np.array(app['seedCity']))[:2])), 3)
            app['annotationRectPx'] = [round(x, 1) for x in rect[:4]]
            app['appCase'] = True
            chosen = [app] + [c for c in chosen if c['track'] != app['track']][:rules['carsPerLog'] - 1]
    return chosen, candidates


def visible_edge(log_obj, timestamp, edge, samples=200):
    """Samples of a mapped edge that project inside the image, with their XY range from the camera."""
    points = edge[0] + np.linspace(0, 1, samples)[:, None] * (edge[-1] - edge[0])
    u, v, z = log_obj.project(points, timestamp)
    visible = (z > 0.5) & (u >= 0) & (u < log_obj.width) & (v >= 0) & (v < log_obj.height)
    ranges = np.linalg.norm((points - log_obj.camera_center_city(timestamp))[:, :2], axis=1)
    length = float(np.linalg.norm((edge[-1] - edge[0])[:2])) * visible.mean()
    return points[visible], ranges[visible], length


def paint_cases(log_short, log_obj, rules):
    """One paint case per target range in rules['paintTargetRangesM'] (plus the app frame for the app log)."""
    out, frames = [], set()
    targets = [None] + list(rules['paintTargetRangesM']) if log_short == APP_CASE['log'] else list(rules['paintTargetRangesM'])
    for target in targets:
        case = paint_case(log_short, log_obj, rules, target)
        if case and case['frame'] not in frames:
            frames.add(case['frame'])
            out.append(case)
    return out


def paint_case(log_short, log_obj, rules, target_range):
    """Near edge = the mapped edge whose visible part is closer; range = median range of its visible samples.

    target_range None means the app frame (app log only)."""
    crossing_id = CROSSINGS[log_short]
    crossing = read_json(map_file(log_obj.log))['pedestrian_crossings'][crossing_id]
    edges = [np.array([[p['x'], p['y'], p['z']] for p in crossing[k]]) for k in ('edge1', 'edge2')]
    frames = [t for t in frames_for(log_short) if t in log_obj.poses]
    if target_range is None:
        frames, target_range = [APP_CASE['frame']], rules['targetRangeM']
    best = None
    for t in frames:
        views = [visible_edge(log_obj, t, e) for e in edges]
        options = [(float(np.median(r)), i, p) for i, (p, r, length) in enumerate(views) if length >= rules['minVisibleEdgeM']]
        if not options:
            continue
        rng, near_index, points = min(options, key=lambda item: item[0])
        if best is None or abs(rng - target_range) < abs(best[1] - target_range):
            best = (t, rng, near_index, points[len(points) // 2])
    if best is None:
        return None
    near = edges[best[2]]
    u, v, _ = log_obj.project(near, best[0])
    cu, cv, _ = log_obj.project(best[3][None], best[0])
    return {'kind': 'paint', 'log': log_obj.log, 'crossing': crossing_id, 'frame': best[0], 'rangeM': round(best[1], 3),
            'nearEdge': 'edge1' if best[2] == 0 else 'edge2', 'mappedNearEdgeCity': near.round(3).tolist(),
            'mappedNearEdgePx': np.column_stack([u, v]).round(1).tolist(),
            'visibleMidPointPx': [round(float(cu[0]), 1), round(float(cv[0]), 1)],
            'aheadCity': best[3].round(3).tolist(), 'appCase': best[0] == APP_CASE['frame'],
            'targetRangeM': target_range}


def sweep_plan(log_short, log_obj, cases, rules):
    sweeps = [t for t in sweeps_for(log_short) if t in log_obj.poses]
    targets = []
    for c in cases:
        if c['kind'] == 'car':
            targets.append(np.array(c['seedCity'][:2]))
        else:
            targets.append(np.array(c['aheadCity'])[:2])
    keep, last = [], None
    for t in sweeps:
        origin = log_obj.pose(t)[:3, 3]
        if not any(np.linalg.norm(origin[:2] - g) <= rules['sweepRadiusM'] for g in targets):
            continue
        if last is None or np.linalg.norm(origin[:2] - last) >= rules['sweepSpacingM']:
            keep.append(t)
            last = origin[:2]
    for c in cases:
        nearest = min(sweeps, key=lambda s: abs(s - c['frame']))
        c['frameSweep'] = nearest
        if nearest not in keep:
            keep.append(nearest)
    return sorted(keep)


def main(argv):
    """`select_cases.py [PROTOCOL]` uses a locked protocol (default v1) and writes cases/selection.json next to it;
    `select_cases.py --draft rules.json out.json` only plans downloads."""
    if argv[:1] == ['--draft']:
        rules, protocol_hash, target = read_json(argv[1]), None, Path(argv[2])
    else:
        path = Path(argv[0]) if argv else LABELS_DIR / 'protocol-v1.json'
        protocol, protocol_hash = load_protocol(path)
        rules, target = protocol['caseSelection'], path.parent / 'cases/selection.json'
    out = {'protocolSha256': protocol_hash, 'rules': rules, 'logs': {}}
    for log_short in ('8c52d911', '453e5558', 'c780d53a'):
        assert_development(log_short)
        log_obj = Log(log_short, default_roots(log_short))
        cars, candidates = car_cases(log_short, log_obj, rules)
        paints = paint_cases(log_short, log_obj, rules)
        cases = cars + paints
        plan = sweep_plan(log_short, log_obj, cases, rules)
        out['logs'][log_short] = {'log': full_log_id(log_short), 'cases': cases, 'carCandidates': len(candidates),
                                  'sweeps': [str(s) for s in plan]}
        print(log_short, 'cars', [(c['track'][:8], round(c['rangeM'], 1)) for c in cars], 'paint',
              [round(c['rangeM'], 1) for c in paints], 'sweeps', len(plan))
    write_json(target, json.loads(json.dumps(out, default=str)))


if __name__ == '__main__':
    main(sys.argv[1:])

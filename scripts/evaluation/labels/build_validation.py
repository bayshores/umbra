"""Build the DEVELOPMENT known-answer validation set for the labeling protocol.

  python -I build_validation.py --protocol P.json --base DIR [--selection cases/selection.json]   # locked protocol
  python -I build_validation.py --draft --protocol P.json --base OUTDIR --selection S.json        # unlocked trial build
  python -I build_validation.py --draft --synthetic-only --protocol P.json --base OUTDIR          # synthetic cases only,
        into a folder outside the repository (tool smoke tests; no selection file, no real cases, no coordinator folder)

Writes:
  validation/images/<task>.jpg, validation/overviews/<task>.png   what labelers see (raw or synthetic photo)
  validation-tasks.json                                          task list labelers read
  kit-internal/tasks-internal.json                               (protocol v2) coordinates, columns, guides, hashes
  sealed/answer-key.json, sealed/coverage/<task>.npz             known answers (labelers must not open). Under a
                                                                 protocol with `isolation` this folder is in the
                                                                 coordinator-only folder outside the repository
                                                                 (the lock records the variable name, not a path)
  validation-lock.json                                           hashes of all of the above (v2: and of the labeler
                                                                 instructions), written before labeling
Real cases use LiDAR inner-bound references (coarse, see protocol); synthetic cases are exact.
Protocol v2 synthetic vehicles (`synthetic.vehicleModels` "v2"): sedan, hatchback, SUV and van from synth.vehicle_v2
with per-part materials, decals and a contact shadow (`synthetic.materials`, `sunDirection`, `contactShadow`); the
v1 `box` model is refused. Without `vehicleModels` (protocol v1) the v1 renderer and materials are used unchanged.

Protocol v1 put every task field, including vehicle station columns at 25/50/75 % of the projected AV2 box,
into the one file labelers read (audit finding, protocol-v1.errata.md). Protocol v2 (`taskFiles`) keeps only
the fields named in `taskFiles.public` there (no number except the question estimate) and chooses vehicle
station columns inside the session from the labeler's own left/right answers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imaging  # noqa: E402
import isolation  # noqa: E402
import lidar_ref  # noqa: E402
import synth  # noqa: E402
from av2io import Log, apply, default_roots  # noqa: E402
from common import (INTERNAL_TASKS, INTERNAL_TASKS_SCHEMA, PUBLIC_TASKS_SCHEMA, ROOT, assert_development,  # noqa: E402
                    load_protocol, read_json, relative, require, sha256_bytes, sha256_file, write_bytes_atomic, write_json)
from select_cases import map_file  # noqa: E402

def log_sweeps(log_obj):
    return sorted({int(p.name[:-8]) for r in log_obj.roots if (r / 'sensors/lidar').exists() for p in (r / 'sensors/lidar').glob('*.feather')})


def nearest(values, target):
    return min(values, key=lambda x: abs(x - target))


# ----- synthetic ---------------------------------------------------------------------------------------

def ground_for(log_obj, frame, point_xy, sweeps, radius=6.0):
    """Plane through the lower 60 % of all stacked returns within `radius` of the placement point."""
    pts = []
    for s in sweeps:
        xyz = log_obj.sweep_city(s)['xyz']
        pts.append(xyz[np.linalg.norm(xyz[:, :2] - point_xy, axis=1) <= radius])
    pts = np.vstack(pts)
    require(len(pts) >= 200, f'Too few LiDAR returns near the synthetic placement ({len(pts)})')
    pts = pts[pts[:, 2] <= np.percentile(pts[:, 2], 60)]
    coef, _ = synth.fit_plane(pts)
    return coef


def camera_axes(log_obj, frame):
    pose = log_obj.city_from_camera(frame)
    f = pose[:3, 2][:2] / np.linalg.norm(pose[:3, 2][:2])
    return pose[:3, 3], f, np.array([-f[1], f[0]])


def road_luminance(rgb, camera, points_city):
    uv = camera.project_camera(camera.to_camera(points_city))
    lum = imaging.luminance(rgb)
    vals = [lum[int(round(v)), int(round(u))] for u, v in uv if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]]
    return float(np.median(vals))


def synthetic_case(spec, S):
    log_obj = Log(spec['log'], default_roots(spec['log']))
    frame = int(spec['frame'])
    rgb = imaging.load_rgb(log_obj.find(f'sensors/cameras/stereo_front_left/{frame}.jpg'))
    camera = synth.Camera(log_obj.K, log_obj.camera_from_city(frame), log_obj.width, log_obj.height)
    cam, f, left = camera_axes(log_obj, frame)
    point = cam[:2] + spec['forwardM'] * f + spec['lateralM'] * left
    coef = ground_for(log_obj, frame, point, log_sweeps(log_obj))
    yaw = np.arctan2(f[1], f[0]) + np.radians(spec['yawDeg'])
    pose = synth.plane_frame(coef, point, yaw)
    image = rgb.astype(float)
    entry = {'source': 'synthetic', 'kind': spec['kind'], 'spec': spec, 'groundPlane': [float(c) for c in coef]}
    if spec['kind'] == 'car' and S.get('vehicleModels') == 'v2':
        return synthetic_car_v2(spec, S, rgb, camera, pose, image, entry)
    if spec['kind'] == 'car':
        L, W, H = spec['sizeM']
        parts = [(n, apply(pose, v), fc, m) for n, v, fc, m in synth.vehicle(spec['model'], L, W, H)]
        road = road_luminance(rgb, camera, apply(pose, np.array([[x, y, 0] for x in (-L / 2 - 1.5, L / 2 + 1.5) for y in (-W, W)])))
        exposure = (road / 255.0) / S['roadAlbedo']
        a = spec['albedo']
        materials = {'body': {'albedo': a, 'ambient': 0.55, 'diffuse': 0.55, 'sky': 0.02},
                     'cabin': {'albedo': a if spec['model'] == 'box' else S['glassAlbedo'], 'ambient': 0.5, 'diffuse': 0.3, 'sky': 0.06},
                     'tire': {'albedo': S['tireAlbedo'], 'ambient': 0.6, 'diffuse': 0.3}}
        sx, sy = spec.get('shadowOffsetM', [0.3, -0.25])
        foot = apply(pose, np.array([[x + sx, y + sy, 0.005] for x in (-L / 2 - 0.1, L / 2 + 0.1) for y in (-W / 2 - 0.1, W / 2 + 0.1)]))
        image = synth.shadow(image, camera, foot, spec.get('shadowStrength', 0.5))
        image, coverage, region, polygons = synth.render_solids(image, camera, parts, materials, S['supersample'], seed=spec['seed'],
                                                                exposure=exposure)
        mask = np.zeros(rgb.shape[:2], bool)
        x0, y0, w, h = region
        mask[y0:y0 + h, x0:x0 + w] = coverage > 0
        entry.update({'carPolygons': [p.round(4).tolist() for p in polygons], 'coverageOrigin': [x0, y0], '_coverage': coverage,
                      'exposure': exposure, 'roadLuminance': road})
    else:
        quads = crosswalk_quads(spec, pose)
        road = road_luminance(rgb, camera, np.vstack(quads))
        image, coverage, region, polys = synth.paint(image, camera, quads, S['paintGain'] * road, S['supersample'],
                                                     wear=spec.get('wear', 0.25), seed=spec['seed'])
        mask = np.zeros(rgb.shape[:2], bool)
        x0, y0, w, h = region
        mask[y0:y0 + h, x0:x0 + w] = coverage > 0
        entry.update({'paintPolygons': [p.round(4).tolist() for p in polys], 'paintCoverageOrigin': [x0, y0], '_paintCoverage': coverage,
                      'roadLuminance': road, 'barCentersPx': bar_centres(polys, spec, S['paintStationSpacingPx'])})
    jpeg = synth.camera_finish(np.clip(image, 0, 255), mask, S['blurPx'], S['noise'], S['jpegQuality'], seed=spec['seed'])
    return jpeg, entry


def synthetic_car_v2(spec, S, rgb, camera, pose, image, entry):
    """Protocol v2 synthetic vehicle (coverage-design.md section 4). Truth = the convex parts only: the contact shadow
    is drawn on the background before the vehicle, and materials/decals only recolour pixels the parts cover."""
    L, W, H = spec['sizeM']
    local_parts, local_decals, wheels = synth.vehicle_v2(spec['model'], L, W, H)  # refuses `box`
    parts = [(n, apply(pose, v), fc, m) for n, v, fc, m in local_parts]
    decals = {n: [(fi, apply(pose, v), m) for fi, v, m in items] for n, items in local_decals.items()}
    road = road_luminance(rgb, camera, apply(pose, np.array([[x, y, 0] for x in (-L / 2 - 1.5, L / 2 + 1.5) for y in (-W, W)])))
    exposure = (road / 255.0) / S['roadAlbedo']
    materials = {name: dict(values) for name, values in S['materials'].items()}
    materials['body']['albedo'] = spec['albedo']
    cs = S['contactShadow']
    margin = int(np.ceil(3 * max(cs['bodyBlurPx'], cs['tireBlurPx']))) + 2
    body_foot, tire_feet = synth.contact_footprints(L, W, wheels, cs['offsetM'])
    image = synth.shadow(image, camera, apply(pose, body_foot), cs['bodyStrength'], cs['bodyBlurPx'], margin=margin)
    for foot in tire_feet:
        image = synth.shadow(image, camera, apply(pose, foot), cs['tireStrength'], cs['tireBlurPx'], margin=margin)
    image, coverage, region, polygons = synth.render_solids(image, camera, parts, materials, S['supersample'], seed=spec['seed'],
                                                            exposure=exposure, light=tuple(S['sunDirection']), decals=decals)
    mask = np.zeros(rgb.shape[:2], bool)
    x0, y0, w, h = region
    mask[y0:y0 + h, x0:x0 + w] = coverage > 0
    entry.update({'carPolygons': [p.round(4).tolist() for p in polygons], 'coverageOrigin': [x0, y0], '_coverage': coverage,
                  'exposure': exposure, 'roadLuminance': road, 'vehicleModels': 'v2'})
    jpeg = synth.camera_finish(np.clip(image, 0, 255), mask, S['blurPx'], S['noise'], S['jpegQuality'], seed=spec['seed'])
    return jpeg, entry


def check_synthetic_models(S):
    """Protocol v2 refuses the v1 `box` model (and any unknown model) before anything is built."""
    if S.get('vehicleModels') != 'v2':
        return
    for spec in S['cases']:
        if spec['kind'] == 'car':
            require(spec['model'] in synth.VEHICLE_V2, f"synthetic case {spec['id']}: model {spec['model']!r} is not a v2 model "
                                                       f"({', '.join(sorted(synth.VEHICLE_V2))}); box is v1 only")


def crosswalk_quads(spec, pose):
    """Paint quads in the crossing frame (x along travel from the near edge, y across the road), on the plane."""
    quads = []
    half = spec['acrossM'] / 2
    near = 0.0
    def quad(x0, x1, y0, y1):
        return apply(pose, np.array([[x0, y0, 0.002], [x1, y0, 0.002], [x1, y1, 0.002], [x0, y1, 0.002]]))
    if spec['pattern'] in ('continental', 'continental-border'):
        start = near
        if spec['pattern'] == 'continental-border':
            quads.append(quad(near, near + spec['borderM'], -half, half))
            start = near + spec['borderM'] + spec.get('borderGapM', 0.0)
        y = -half
        while y + spec['barM'] <= half + 1e-9:
            quads.append(quad(start, start + spec['barLengthM'], y, y + spec['barM']))
            y += spec['barM'] + spec['gapM']
    else:  # two transverse lines
        quads.append(quad(near, near + spec['lineM'], -half, half))
        quads.append(quad(near + spec['lengthM'] - spec['lineM'], near + spec['lengthM'], -half, half))
    return quads


def bar_centres(polys, spec, spacing=100.0):
    """Candidate station columns: centre of each narrow painted piece; long pieces sampled every `spacing` px."""
    out = []
    for p in polys:
        u0, u1 = float(p[:, 0].min()), float(p[:, 0].max())
        if u0 < 2 or u1 > 2045:
            continue
        if u1 - u0 <= 1.5 * spacing:
            out.append((u0 + u1) / 2)
        else:
            out += list(np.arange(u0 + spacing / 2, u1 - 10, spacing))
    return sorted(out)


# ----- real (LiDAR) ------------------------------------------------------------------------------------

def real_car(case, P):
    log_obj = Log(case['log'], default_roots(case['log']))
    sweeps = log_sweeps(log_obj)
    frame_sweep = nearest(sweeps, case['frame'])
    ref = lidar_ref.car_reference(log_obj, case, sweeps, frame_sweep, P['lidarCar'])
    entry = {'source': 'lidar', 'kind': 'car', 'case': case, 'sweeps': [str(s) for s in sweeps], 'frameSweep': str(frame_sweep)}
    entry.update(ref)
    if ref['status'] == 'resolved':
        entry['topProfile'] = {str(k): v for k, v in ref['topProfile'].items()}
        entry['bottomProfile'] = {str(k): v for k, v in ref['bottomProfile'].items()}
    image = log_obj.find(f'sensors/cameras/stereo_front_left/{case["frame"]}.jpg').read_bytes()
    return image, entry


def real_paint(case, P):
    log_obj = Log(case['log'], default_roots(case['log']))
    sweeps = log_sweeps(log_obj)
    frame_sweep = nearest(sweeps, case['frame'])
    crossing = read_json(map_file(log_obj.log))['pedestrian_crossings'][case['crossing']]
    far = 'edge2' if case['nearEdge'] == 'edge1' else 'edge1'
    geometry = {'near': [[p['x'], p['y'], p['z']] for p in crossing[case['nearEdge']]],
                'far': [[p['x'], p['y'], p['z']] for p in crossing[far]]}
    ref = lidar_ref.paint_reference(log_obj, case, sweeps, frame_sweep, P['lidarPaint'], geometry)
    entry = {'source': 'lidar', 'kind': 'paint', 'case': case, 'sweeps': [str(s) for s in sweeps], 'frameSweep': str(frame_sweep)}
    entry.update(ref)
    image = log_obj.find(f'sensors/cameras/stereo_front_left/{case["frame"]}.jpg').read_bytes()
    return image, entry


def real_tier(entry):
    """protocol realTiers: app car, app paint, 453e b4645064 and the c780 paint case nearest 13 m are tier 1."""
    case = entry['case']
    if case.get('appCase'):
        return 1
    if case['kind'] == 'car' and case['track'].startswith('b4645064'):
        return 1
    if case['kind'] == 'paint' and case['log'].startswith('c780d53a') and case.get('targetRangeM') == 13.0:
        return 1
    return 2


def usable_bars(entry, width=2048):
    bars = []
    for b in entry.get('bars', []):
        (u0, v0), (u1, v1) = b['segmentPx']
        if b.get('consensus') and not b['occluded'] and min(u0, u1) >= 2 and max(u0, u1) <= width - 3 and abs(u1 - u0) >= 4:
            bars.append(b)
    return bars


# ----- task assembly ------------------------------------------------------------------------------------

def overview_region(bbox, P, rng, width, height):
    o = P['overview']
    x0, y0, x1, y1 = bbox
    size = max(x1 - x0, y1 - y0)
    m = [size * (o['marginMin'] + rng.random() * o['marginJitter']) for _ in range(4)]
    rx0, ry0 = int(np.floor(x0 - m[0])), int(np.floor(y0 - m[1]))
    rx1, ry1 = int(np.ceil(x1 + m[2])), int(np.ceil(y1 + m[3]))
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(width - 1, rx1), min(height - 1, ry1)
    target = (rx1 - rx0) / o['targetColumns']
    cell = min(o['cellChoices'], key=lambda c: abs(c - target))
    return [rx0, ry0, rx1 - rx0 + 1, ry1 - ry0 + 1], int(cell)


def car_task_fields(entry, P, rng, width, height):
    """Overview region, cell size, ring and (protocol v1 only) station columns for a vehicle task.

    v1: columns at carFractions of the projected AV2 box (real) or the jittered true extent (synthetic); these
    reached the labeler-visible task list. v2 (stations.carColumns = 'labelerExtremes'): no columns here; the
    session derives them from the labeler's own answers. The AV2 box is never read.
    """
    task_columns = P['stations'].get('carColumns', 'task') == 'task'
    if entry['source'] == 'synthetic':
        allv = np.vstack([np.asarray(p) for p in entry['carPolygons']])
        bbox = [allv[:, 0].min(), allv[:, 1].min(), allv[:, 0].max(), allv[:, 1].max()]
        if task_columns:
            jit = P['stations']['syntheticSamplingJitter']
            bw = bbox[2] - bbox[0]
            sampling = [bbox[0] + rng.uniform(-jit, jit) * bw, bbox[2] + rng.uniform(-jit, jit) * bw]
        centre = [(bbox[0] + bbox[2]) / 2, (bbox[1] + 2 * bbox[3]) / 3]
    else:
        r = entry['reference']
        bbox = [r['L'], r['T'], r['R'], r['B']]
        if task_columns:
            a = entry['case']['annotationRectPx']
            sampling = [a[0], a[2]]
        centre = entry['projectedCentroidPx']
    jr = P['overview']['ringJitterFraction']
    ring = [centre[0] + rng.uniform(-jr, jr) * (bbox[2] - bbox[0]), centre[1] + rng.uniform(-jr, jr) * (bbox[3] - bbox[1])]
    columns = [int(round(sampling[0] + f * (sampling[1] - sampling[0]))) for f in P['stations']['carFractions']] if task_columns else []
    region, cell = overview_region(bbox, P, rng, width, height)
    return region, cell, ring, columns


def split_task(task, public_fields):
    """(labeler-visible, kit-only) parts of a task. The visible part keeps only `public_fields`, and of the overview
    only its path; image paths and hashes, overview geometry, ring, columns, guides and tier stay kit-only."""
    require('taskId' in public_fields, 'taskFiles.public must include taskId')
    public = {k: task[k] for k in public_fields if k in task and k != 'overview'}
    if 'overview' in public_fields:
        public['overview'] = {'path': task['overview']['path']}
    internal = {k: v for k, v in task.items() if k not in public or k in ('taskId', 'overview')}
    return public, internal


def public_numbers(value, path=''):
    """Every number in a labeler-visible structure, with its path (used by the build check and tests)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(path, value)]
    if isinstance(value, dict):
        return [item for k, v in value.items() for item in public_numbers(v, f'{path}.{k}' if path else k)]
    if isinstance(value, (list, tuple)):
        return [item for k, v in enumerate(value) for item in public_numbers(v, f'{path}[{k}]')]
    return []


def paint_task_fields(entry, P, rng, width, height):
    if entry['source'] == 'synthetic':
        centres = entry['barCentersPx']
        anchor = np.mean(centres)
        rows = [max(np.asarray(p)[:, 1]) for p in entry['paintPolygons']]
        vmin, vmax = min(rows) - 20, max(rows)
    else:
        bars = usable_bars(entry)
        centres = [np.mean([b['segmentPx'][0][0], b['segmentPx'][1][0]]) for b in bars]
        anchor = entry['case']['visibleMidPointPx'][0]
        rows = [np.mean([b['segmentPx'][0][1], b['segmentPx'][1][1]]) for b in bars]
        vmin, vmax = (min(rows) - 20, max(rows)) if rows else (0, 0)
    st = P['stations']
    order = sorted(range(len(centres)), key=lambda k: abs(centres[k] - anchor))
    chosen = []
    for k in order:
        c = int(round(centres[k]))
        if abs(c - anchor) <= st['paintMaxHalfSpanPx'] and all(abs(c - q) >= st['paintMinSeparationPx'] for q in chosen):
            chosen.append(c)
        if len(chosen) == st['paintMaxStations']:
            break
    chosen.sort()
    if not chosen:
        return None
    o = P['overview']['paint']
    span = max(chosen) - min(chosen) + 40
    mx = [span * rng.uniform(*o['marginXFraction']) for _ in range(2)]
    x0, x1 = int(min(chosen) - 20 - mx[0]), int(max(chosen) + 20 + mx[1])
    y0, y1 = int(vmin - rng.uniform(*o['marginTopPx'])), int(vmax + rng.uniform(*o['marginBottomPx']))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)
    cell = min(P['overview']['cellChoices'], key=lambda c: abs(c - (x1 - x0) / P['overview']['targetColumns']))
    region = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    return region, cell, chosen


def write_task_files(base, tasks, P, common_fields, instructions=None):
    """Write the task list(s). Returns (public sha256, kit-internal sha256 or None).

    Protocol v2 (`taskFiles`): validation-tasks.json gets only the public fields and is refused if it holds any
    number other than `taskFiles.publicNumbers`; everything else goes to kit-internal/tasks-internal.json."""
    base = Path(base)
    if not P.get('taskFiles'):
        return write_json(base / 'validation-tasks.json', {'schema': 'pao-labels-tasks-v1', **common_fields, 'tasks': tasks}), None
    public_fields = P['taskFiles']['public']
    allowed_numbers = set(P['taskFiles'].get('publicNumbers', []))
    publics, internals = [], {}
    for task in tasks:
        if instructions:
            task['instructions'] = instructions
        public, internal = split_task(task, public_fields)
        leaks = [path for path, _ in public_numbers(public) if path not in allowed_numbers]
        require(not leaks, f"task {task['taskId']}: labeler-visible numbers {leaks}")
        publics.append(public)
        internals[task['taskId']] = internal
    leaks = [path for path, _ in public_numbers(common_fields)]
    require(not leaks, f'labeler-visible numbers in the task list header {leaks}')
    tasks_hash = write_json(base / 'validation-tasks.json', {'schema': PUBLIC_TASKS_SCHEMA, **common_fields, 'tasks': publics})
    internal_hash = write_json(base / INTERNAL_TASKS, {'schema': INTERNAL_TASKS_SCHEMA, 'publicSha256': tasks_hash,
                                                       'note': 'Kit-only task fields. Labelers never open this folder.',
                                                       'tasks': internals})
    return tasks_hash, internal_hash


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--selection')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='draft smoke test: synthetic cases only, written to --base outside the repository')
    args = parser.parse_args(argv)
    draft, protocol_path, base = args.draft, Path(args.protocol), Path(args.base)
    synthetic_only = args.synthetic_only
    if synthetic_only:
        require(draft, '--synthetic-only is a draft smoke test: add --draft')
        require(not isolation.inside_repo(base), '--synthetic-only writes its answer key under --base: use a folder outside the repository')
        selection = {'logs': {}}
    else:
        selection_path = Path(args.selection) if args.selection else base / 'cases/selection.json'
        selection = read_json(selection_path)
    if draft:
        P = read_json(protocol_path)
        phash = 'DRAFT-' + sha256_file(protocol_path)
    else:
        P, phash = load_protocol(protocol_path)
        require(selection['protocolSha256'] == phash, f'{selection_path.name} was not made with the locked protocol')
        require(not (base / 'validation-lock.json').exists(), 'validation-lock.json exists: the validation set is frozen')
    instructions = None
    if P.get('labelerInstructions'):
        path = protocol_path.parent / P['labelerInstructions']
        require(path.exists(), f'missing labeler instructions {path.name}')
        instructions = {'path': relative(path), 'sha256': sha256_file(path)}
    check_synthetic_models(P['synthetic'])
    rng = random.Random(P['taskOrderSeed'])
    built = []
    for short, info in selection['logs'].items():
        assert_development(short)
        for case in info['cases']:
            if case['kind'] == 'car':
                image, entry = real_car(case, P)
            else:
                image, entry = real_paint(case, P)
            label = f"{short}-{case.get('track', case.get('crossing'))[:8]}-{case['frame']}"
            built.append((label, image, entry))
            print('real', label, entry['status'], entry.get('reason', ''), flush=True)
    for spec in P['synthetic']['cases']:
        image, entry = synthetic_case(spec, P['synthetic'])
        built.append((spec['id'], image, entry))
        print('synthetic', spec['id'], flush=True)
    rng.shuffle(built)
    tasks, key, coverage, excluded = [], {}, {}, []
    images_dir, overview_dir = base / 'validation/images', base / 'validation/overviews'
    for label, image, entry in built:
        if entry.get('status', 'resolved') != 'resolved':
            excluded.append({'case': label, 'reason': entry.get('reason')})
            continue
        if entry['source'] == 'lidar' and entry['kind'] == 'paint' and len(usable_bars(entry)) < P['stations']['paintMinUsableBars']:
            excluded.append({'case': label, 'reason': f"only {len(usable_bars(entry))} unoccluded consensus bars"})
            continue
        if entry['source'] == 'lidar' and entry['kind'] == 'car' and any(entry['occlusion'].get(k) for k in 'LRTB'):
            excluded.append({'case': label, 'reason': 'occluded extreme'})
            continue
        task_id = f"V{len(tasks) + 1:02d}"
        rgb = imaging.load_rgb(io.BytesIO(image))
        h, w = rgb.shape[:2]
        if entry['kind'] == 'car':
            region, cell, ring, columns = car_task_fields(entry, P, rng, w, h)
            guides = []
            target = {'noun': 'vehicle', 'hint': 'The target vehicle is the one under the cyan ring.'}
        else:
            fields = paint_task_fields(entry, P, rng, w, h)
            if fields is None:
                excluded.append({'case': label, 'reason': 'no stations'})
                continue
            region, cell, columns = fields
            ring, guides = None, columns
            target = {'noun': 'crosswalk paint', 'hint': ('The crosswalk is the painted pedestrian crossing across the road. Label the '
                                                          'crosswalk markings themselves (border line or bars), not stop lines, lane '
                                                          'lines, arrows or triangle (yield) markings.')}
        png, meta = imaging.render_overview(rgb, region, cell, P['overview']['maxDisplayWidth'], guides=guides, ring=ring)
        image_path, overview_path = images_dir / f'{task_id}.jpg', overview_dir / f'{task_id}.png'
        write_bytes_atomic(image_path, image)
        write_bytes_atomic(overview_path, png)
        meta.update({'path': relative(overview_path), 'sha256': sha256_bytes(png)})
        estimated = 90 if entry['kind'] == 'car' else (1 + len(columns) * 8)  # observed with a perfect oracle: cars 70-105
        tier = real_tier(entry) if entry['source'] == 'lidar' else entry['spec']['tier']
        task = {'taskId': task_id, 'kind': entry['kind'], 'tier': tier,
                'image': {'path': relative(image_path), 'sha256': sha256_bytes(image), 'width': w, 'height': h},
                'overview': meta, 'target': target, 'stationColumns': columns, 'guides': guides,
                'estimatedQuestions': estimated, 'answerDir': f"{relative(base / 'sessions')}/<LABELER>/{task_id}/"}
        tasks.append(task)
        stored = {k: v for k, v in entry.items() if not k.startswith('_')}
        stored['caseLabel'] = label
        key[task_id] = stored
        for name in ('_coverage', '_paintCoverage'):
            if name in entry:
                coverage.setdefault(task_id, {})[name[1:]] = entry[name]
    # blind repeats: same image (v1: and station columns), freshly jittered overview; scored for within-labeler precision.
    # v2 requires each repeat to run in a different fresh agent from its original (protocol labelers.isolation).
    for label in P['repeatCases']:
        originals = [t for t in tasks if key[t['taskId']]['caseLabel'].startswith(label)]
        if synthetic_only and not originals:
            continue  # smoke test: real repeat cases are not built
        require(len(originals) == 1, f'repeat case {label} matched {len(originals)} tasks')
        orig = originals[0]
        entry = key[orig['taskId']]
        image = (ROOT / orig['image']['path']).read_bytes()
        rgb = imaging.load_rgb(io.BytesIO(image))
        h, w = rgb.shape[:2]
        region, cell, ring, _columns = car_task_fields(entry, P, rng, w, h)
        task_id = f"V{len(tasks) + 1:02d}"
        png, meta = imaging.render_overview(rgb, region, cell, P['overview']['maxDisplayWidth'], ring=ring)
        image_path, overview_path = images_dir / f'{task_id}.jpg', overview_dir / f'{task_id}.png'
        write_bytes_atomic(image_path, image)
        write_bytes_atomic(overview_path, png)
        meta.update({'path': relative(overview_path), 'sha256': sha256_bytes(png)})
        task = dict(orig, taskId=task_id, overview=meta, tier=2 if entry['source'] == 'synthetic' else 1,
                    image={**orig['image'], 'path': relative(image_path)},
                    answerDir=f"{relative(base / 'sessions')}/<LABELER>/{task_id}/")
        tasks.append(task)
        key[task_id] = dict(entry, repeatOf=orig['taskId'])
        if orig['taskId'] in coverage:
            coverage[task_id] = coverage[orig['taskId']]
    created = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    common_fields = {'protocol': {'file': protocol_path.name, 'sha256': phash}, 'createdUtc': created,
                     'instructions': instructions['path'] if instructions else '.local/evaluation/pao-labels/labeler-instructions.md',
                     'cli': 'python -I scripts/evaluation/labels/labelkit.py --labeler <ID> --task <TASK> next|answer <A>|status',
                     'interpreter': '.local/venv-labels/bin/python'}
    tasks_hash, internal_hash = write_task_files(base, tasks, P, common_fields, instructions)
    sealed_spec = None if synthetic_only else isolation.sealed_spec(P)
    if sealed_spec:
        sealed = isolation.coordinator_root(sealed_spec['env']) / sealed_spec['subdir']
        require(draft or not (sealed / 'answer-key.json').exists(), 'a sealed answer key already exists for this protocol version')
    else:
        sealed = base / 'sealed'
    key_doc = {'schema': 'pao-labels-answer-key-v1', 'protocolSha256': phash, 'tasks': key, 'excluded': excluded}
    key_hash = write_json(sealed / 'answer-key.json', json.loads(json.dumps(key_doc, default=float)))
    cov_hashes = {}
    for task_id, arrays in coverage.items():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        cov_hashes[task_id] = write_bytes_atomic(sealed / f'coverage/{task_id}.npz', buffer.getvalue())
    lock = {'schema': 'pao-labels-validation-lock-v1', 'protocolSha256': phash, 'validationTasksSha256': tasks_hash,
            'answerKeySha256': key_hash, 'coverageSha256': cov_hashes,
            'images': {t['taskId']: t['image']['sha256'] for t in tasks}, 'overviews': {t['taskId']: t['overview']['sha256'] for t in tasks},
            'kitCodeSha256': {p.name: sha256_file(p) for p in sorted(HERE.glob('*.py'))},
            'excluded': excluded, 'frozenUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'statement': 'Written before any labeler session. Scoring refuses if any hash differs.'}
    if internal_hash:
        lock['tasksInternalSha256'] = internal_hash
    if sealed_spec:
        lock['sealed'] = sealed_spec
    if instructions:
        lock['labelerInstructions'] = instructions
    write_json(base / 'validation-lock.json', lock, overwrite=draft)
    print(json.dumps({'tasks': len(tasks), 'excluded': excluded, 'tasksSha256': tasks_hash, 'answerKeySha256': key_hash}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])

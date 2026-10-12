"""Map-first placement suggestions (schema v2) for one grounded AV2 scene; never Clear, never invented.

Usage:
  SOURCE_DATE_EPOCH=<int> python3 -I -B scripts/california/suggest_av2_placements.py \
    --grounded <dir with scene.json, ground.json, terrain.bin> --metadata-sha256 <grounded scene.json sha256> \
    --buffers <assembled scene dir with the reference buffers> --protocol <protocol.json> --protocol-sha256 <sha256>

Writes <grounded>/suggestions.json and <grounded>/catalog-entry.json; refuses to overwrite either. Every
object is a crop pixel plus the ground point its camera ray reaches on the mapped ground raster inside the
mapped drivable area (the same rule the app applies), with a support class (observed-nearby / map-only /
unknown) and a hidden flag (ray passes through an annotated cuboid). Missing evidence stays unknown.
The pixel follows the protocol v2 pixelSelection rule: within +/-windowPx of the rounded projection, the
pixel whose ground hit is horizontally nearest the mapped point; its residual and offset are recorded.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('prepare_av2_ground', HERE / 'prepare_av2_ground.py')
ground_tool = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ground_tool)
from common import canonical, digest, parse, require, snapshot  # noqa: E402  (path set by prepare_av2_ground)

Terrain, ray_hit, ray_direction, project = ground_tool.Terrain, ground_tool.ray_hit, ground_tool.ray_direction, ground_tool.project
inside_polygons, segment_hits_boxes, curb_scan = ground_tool.inside_polygons, ground_tool.segment_hits_boxes, ground_tool.curb_scan
EPS = 1e-9
VEHICLES = {'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER'}
STATIONARY_LABEL = 'stationary vehicle'
STATIONARY_REASON = 'stationary vehicle (AV2 annotation); reason for stopping unknown'
APP_MODEL = 'declared box: front at the pixel ground point, annotated dimensions, axis along the road axis'
# Outputs never use any form of the word 'park' for this scene (Sean, Oct 11; coordinator, Oct 12).
BANNED = b'park'
SUPPORT_LABEL = {'observed-nearby': 'observed nearby', 'map-only': 'mapped, not observed', 'unknown': 'unknown'}
PAINT_OFFSET = {'2069452': {'status': 'label', 'label': '±2 m', 'meters': 2.0}}
UNMEASURED = {'status': 'unmeasured', 'label': 'paint offset unmeasured'}


def generated_utc():
    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    try:
        t = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    except ValueError:
        raise ValueError('SOURCE_DATE_EPOCH must be an integer')
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def unit(v):
    v = np.asarray(v, float); n = float(np.linalg.norm(v)); require(n > EPS, 'Degenerate direction'); return v / n


def intersect(p, u, q, v):
    """Point and parameter t on p + t*u where it meets the line q + s*v (2D); None when parallel."""
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < EPS:
        return None, None
    w = np.asarray(q, float) - p; t = (w[0] * v[1] - w[1] * v[0]) / den
    return np.asarray(p, float) + t * np.asarray(u, float), float(t)


def rectangles_overlap(a, b):
    """Separating-axis test for two 2D rectangles {center, axes:[u,v], half:[hu,hv]}."""
    for axis in (*a['axes'], *b['axes']):
        def extent(r):
            c = float(np.dot(r['center'], axis)); e = sum(h * abs(float(np.dot(ax, axis))) for ax, h in zip(r['axes'], r['half']))
            return c - e, c + e
        (a0, a1), (b0, b1) = extent(a), extent(b)
        if a1 < b0 or b1 < a0:
            return False
    return True


class Context:
    def __init__(self, scene, ground, terrain, positions, eligible, protocol):
        self.scene, self.ground, self.terrain, self.protocol = scene, ground, terrain, protocol
        self.positions, self.eligible = positions, eligible
        self.polygons = [np.asarray(a['outline'], float) for a in ground['drivableAreas']]
        self.cuboids = ground['cuboids']['items'] if ground['cuboids']['available'] else None
        rows = np.asarray(ground['support']['rows'], np.int64)
        require(rows.ndim == 1 and (rows.size == 0 or (rows.min() >= 0 and rows.max() < len(positions))), 'Support rows out of range')
        require(bool(eligible[rows].all()) if rows.size else True, 'Support rows must be camera-eligible')
        self.support = positions[rows][:, [0, 2]] if rows.size else np.zeros((0, 2))
        self.oz = terrain.origin[2]
        lane = ground['egoLane']; left = np.asarray(lane['leftBoundary'], float); right = np.asarray(lane['rightBoundary'], float)
        self.m0 = (left[0, [0, 2]] + right[0, [0, 2]]) / 2
        self.a = unit((left[-1, [0, 2]] + right[-1, [0, 2]]) / 2 - self.m0); self.r = np.array([-self.a[1], self.a[0]])
        self.lane_polygon = np.vstack([left[:, [0, 2]], right[::-1][:, [0, 2]]])
        sel = protocol.get('pixelSelection')
        require(isinstance(sel, dict) and sel.get('rule') == 'nearestHitWindow' and type(sel.get('windowPx')) is int and
                0 <= sel['windowPx'] <= 10, 'Protocol v2 pixelSelection (nearestHitWindow) required; protocol v1 outputs are historical')
        self.window = sel['windowPx']

    def height(self, xz):
        h = self.terrain.sample([xz])[0][0]
        return None if not np.isfinite(h) else float(h) - self.oz

    def ground_hit(self, pixel):
        """Runtime ray rule only: (ray direction, first mapped-ground hit inside the drivable area or None, Unknown reason)."""
        d = ray_direction(self.scene, pixel); hit = ray_hit(d, self.terrain, self.ground['maxRangeMeters'])
        if not hit['hit']:
            return d, None, hit['reason']
        if not inside_polygons(np.asarray(hit['point'])[[0, 2]], self.polygons)[0]:
            return d, None, 'First mapped-ground hit is outside the mapped drivable area.'
        return d, hit, None

    def evaluate_pixel(self, pixel):
        """The runtime rule: pixel -> first mapped-ground hit inside the drivable area, hidden flag, support class."""
        g, tol = self.ground, self.protocol['tolerances']
        d, hit, reason = self.ground_hit(pixel)
        if hit is None:
            return {'supportClass': 'unknown', 'reason': reason}
        p = np.asarray(hit['point'])
        hidden = None if self.cuboids is None else bool(segment_hits_boxes([p], self.cuboids)[0])
        distances = np.linalg.norm(self.support - p[[0, 2]], axis=1) if len(self.support) else np.zeros(0)
        observed = bool((distances <= g['supportRadiusMeters']).any())
        uncertainty = (g['roadToleranceMeters'] + ground_tool.half_ulp16(hit['cellCityZ'])) / -d[1]
        # Diagnostics on ALL returns within the radius (no height pre-filter); the class uses only the pinned rows.
        finite = np.isfinite(self.positions).all(1)
        near = np.zeros(len(self.positions), bool)
        near[finite] = np.linalg.norm(self.positions[finite][:, [0, 2]] - p[[0, 2]], axis=1) <= g['supportRadiusMeters']
        idx = np.flatnonzero(near & self.eligible)
        visible = idx[~segment_hits_boxes(self.positions[idx], self.cuboids)] if self.cuboids is not None and len(idx) else idx
        heights = self.terrain.sample(self.positions[visible][:, [0, 2]])[0] if len(visible) else np.zeros(0)
        dz = self.positions[visible, 1] - (heights - self.oz) if len(visible) else np.zeros(0)
        agreeing = int((np.isfinite(dz) & (np.abs(dz) <= tol['supportHeightAgreementMeters'])).sum())
        require(agreeing == int((distances <= g['supportRadiusMeters']).sum()), 'Support rows disagree with recomputed agreement')
        return {'supportClass': 'observed-nearby' if observed else 'map-only', 'groundPoint': p.tolist(), 'hidden': hidden,
                'alongRayUncertaintyM': float(uncertainty), 'rangeM': hit['t'], 'hitKind': hit['kind'],
                'terrainCityZ': hit['cellCityZ'], 'terrainCell': hit['cell'],
                'supportEvidence': {'returnsWithinRadius': int(near.sum()), 'cameraEligibleWithinRadius': int(len(idx)),
                                    'cameraVisibleWithinRadius': int(len(visible)), 'agreeingWithinRadius': agreeing,
                                    'aboveToleranceWithinRadius': int((np.isfinite(dz) & (dz > tol['supportHeightAgreementMeters'])).sum()),
                                    'belowToleranceWithinRadius': int((np.isfinite(dz) & (dz < -tol['supportHeightAgreementMeters'])).sum()),
                                    'noTerrainWithinRadius': int((~np.isfinite(dz)).sum()),
                                    'nearestSupportM': float(distances.min()) if len(distances) else None}}

    def in_crop(self, pixel):
        c = self.scene['camera']
        return 0 <= pixel[0] <= c['width'] - 1 and 0 <= pixel[1] <= c['height'] - 1

    def search_pixel(self, xz, uv):
        """Protocol v2 pixelSelection: among in-crop pixels within +/-window (Chebyshev) of the rounded projection whose
        runtime ray reaches mapped drivable ground, the one whose hit is horizontally nearest the mapped point xz.
        Ties: nearest the floating projection uv, then smaller v, then smaller u. Returns (best or None, record)."""
        rounded = [math.floor(uv[0] + .5), math.floor(uv[1] + .5)]
        best = None; record = {'windowPx': self.window, 'roundedPixel': rounded, 'roundedResidualM': None,
                               'candidatesInCrop': 0, 'candidatesPlaced': 0}
        for dv in range(-self.window, self.window + 1):
            for du in range(-self.window, self.window + 1):
                pixel = [rounded[0] + du, rounded[1] + dv]
                if not self.in_crop(pixel):
                    continue
                record['candidatesInCrop'] += 1
                _d, hit, reason = self.ground_hit(pixel)
                if hit is None:
                    if du == dv == 0:
                        record['roundedReason'] = reason
                    continue
                record['candidatesPlaced'] += 1
                residual = math.hypot(hit['point'][0] - xz[0], hit['point'][2] - xz[1])
                if du == dv == 0:
                    record['roundedResidualM'] = residual
                key = (residual, math.hypot(pixel[0] - uv[0], pixel[1] - uv[1]), pixel[1], pixel[0])
                if best is None or key < best[0]:
                    best = (key, pixel, [du, dv])
        if best is not None:
            record.update(offsetPx=best[2], residualM=best[0][0])
        return (None if best is None else best[1]), record

    def place(self, xz, base):
        """Desired mapped ground point -> protocol v2 pixel (nearest ground hit within the window) -> its runtime result."""
        h = self.height(xz)
        if h is None:
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'reason': 'No mapped ground height at the mapped point.'}
        desired = [float(xz[0]), h, float(xz[1])]; uv = project(self.scene, desired)
        rounded = None if uv is None else [math.floor(uv[0] + .5), math.floor(uv[1] + .5)]
        if rounded is None or not self.in_crop(rounded):
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'mappedPoint': desired, 'reason': 'Mapped point is outside the recorded image.'}
        pixel, search = self.search_pixel(xz, uv)
        if pixel is None:
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'mappedPoint': desired, 'pixel': rounded, 'pixelSearch': search,
                    'reason': f'No pixel within {self.window} px reaches mapped drivable ground ({search.get("roundedReason", "")})'}
        result = self.evaluate_pixel(pixel)
        require(result['supportClass'] != 'unknown', 'Chosen pixel disagrees with the runtime rule')
        return {**base, **result, 'pixel': pixel, 'mappedPoint': desired, 'projectedUV': uv,
                'supportLabel': SUPPORT_LABEL[result['supportClass']], 'pixelSearch': search,
                'pixelOffsetFromMappedM': search['residualM']}

    # ------------------------------------------------------------ mapped elements
    def near_edge(self):
        best = None
        for name in ('edge1', 'edge2'):
            e = np.asarray(self.ground['crossing'][name], float)[:, [0, 2]]; p0, p1 = e[0], e[-1]
            forward = float(((p0 + p1) / 2) @ self.a)
            require(forward > 0, 'Crossing edge is not ahead of the camera')
            if best is None or forward < best[1]:
                u = unit(p1 - p0); u = u if u @ self.r > 0 else -u
                best = (name, forward, p0, u)
        return best

    def curb_lateral_at(self, forward):
        start = self.m0 + (forward - self.m0 @ self.a) * self.a
        s = curb_scan(start, self.r, self.polygons, self.ground['boundaryEpsilonMeters'])
        return None if s is None else float(start @ self.r + s)


def annotated_vehicle(ctx, edge_p0, edge_u):
    prefix = ctx.protocol['currentVehicle']['trackPrefix']; failures = []
    if ctx.cuboids is None:
        return None, ['Annotations not supplied.']
    found = [c for c in ctx.cuboids if c['track'].startswith(prefix)]
    if len(found) != 1:
        return None, [f'{len(found)} annotated tracks start with {prefix} at the sweep.']
    box = found[0]; m = np.asarray(box['worldFromBox'], float); center = np.asarray(box['center'], float)[[0, 2]]
    heading = unit(m[[0, 2], 0]); half = box['length'] / 2
    cross, t = intersect(center, heading, edge_p0, edge_u)
    motion = box['motion']; angle = math.degrees(math.acos(max(-1., min(1., float(heading @ ctx.a)))))
    yaw = math.degrees(math.atan2(float(heading @ ctx.r), float(heading @ ctx.a)))  # signed, + = toward the right side
    front = center + half * heading
    if box['category'] not in VEHICLES:
        failures.append(f"Category {box['category']} is not a vehicle.")
    if motion is None or motion['spanSeconds'] < 3.0 or motion['maxDisplacementFromFirstM'] > 1.0:
        failures.append('Track is not stationary under the protocol (>= 3.0 s, <= 1.0 m).')
    if not inside_polygons(center, ctx.polygons)[0]:
        failures.append('Box centre is outside the mapped drivable area.')
    if angle > 15:
        failures.append('Box heading is more than 15 degrees from the direction of travel.')
    if cross is None or not (0 < front @ ctx.a < cross @ ctx.a):
        failures.append('Box front is not between the camera and the near edge.')
    if failures:
        return None, failures
    return {'box': box, 'center': center, 'heading': heading, 'front': front, 'crossing': cross, 'gap': t - half,
            'headingToRoadAxisDeg': angle, 'yawToRoadAxisDeg': yaw}, []


def default_vehicle(ctx, edge_p0, edge_u, curb_at_edge):
    spec = ctx.protocol['declaredDefaults']['vehicle']; dims = spec['dimensions']
    width, length, gap, setback = dims['width'], dims['length'], spec['curbGapMeters'], spec['setbackMeters']
    lateral = curb_at_edge - gap - width / 2
    for _ in range(10):
        cross, _t = intersect(lateral * ctx.r, ctx.a, edge_p0, edge_u)
        front_f = float(cross @ ctx.a) - setback
        samples = np.linspace(front_f - length, front_f, int(round(length / 0.05)) + 1)
        curbs = [ctx.curb_lateral_at(f) for f in samples]
        if any(c is None for c in curbs):
            return None, 'Curb proxy unavailable along the default vehicle length.'
        new = min(curbs) - gap - width / 2
        done = abs(new - lateral) < 0.001; lateral = new
        if done:
            cross, _t = intersect(lateral * ctx.r, ctx.a, edge_p0, edge_u); front_f = float(cross @ ctx.a) - setback
            return {'lateral': lateral, 'front': front_f * ctx.a + lateral * ctx.r, 'crossing': cross,
                    'curbAlongLength': [min(curbs), max(curbs)]}, None
    return None, 'Default vehicle lateral did not converge.'


def suggest(scene, ground, terrain, positions, eligible, protocol):
    ctx = Context(scene, ground, terrain, positions, eligible, protocol)
    edge_name, edge_forward, p0, eu = ctx.near_edge()
    skew = 90 - math.degrees(math.acos(min(1., abs(float(eu @ ctx.a)))))
    start, _ = intersect(ctx.m0, ctx.a, p0, eu)
    s = curb_scan(start, eu, ctx.polygons, ground['boundaryEpsilonMeters'])
    curb_point = None if s is None else start + s * eu
    curb_at_edge = None if curb_point is None else float(curb_point @ ctx.r)
    vehicle, failures = annotated_vehicle(ctx, p0, eu)
    defaults = protocol['declaredDefaults']; result_vehicle = None; crossing_xz = None; default_reason = None
    if vehicle is not None:
        crossing_xz = vehicle['crossing']; box = vehicle['box']
        dims = {k: round(box[k], 2) for k in ('width', 'height', 'length')}
        base = {'status': 'annotated', 'label': STATIONARY_LABEL, 'reason': STATIONARY_REASON, 'dimensions': dims,
                'track': box['track'], 'category': box['category'], 'motion': box['motion'],
                'headingWorld': [float(vehicle['heading'][0]), 0., float(vehicle['heading'][1])],
                'headingToRoadAxisDeg': vehicle['headingToRoadAxisDeg'], 'yawToRoadAxisDeg': vehicle['yawToRoadAxisDeg'],
                'frontGapToCrossingM': vehicle['gap'], 'appModel': APP_MODEL}
        result_vehicle = ctx.place(vehicle['front'], base)
        if result_vehicle['status'] != 'unknown':  # app model front (pixel ground point) minus the annotated front, horizontal
            diff = np.asarray(result_vehicle['groundPoint'], float)[[0, 2]] - vehicle['front']
            result_vehicle['frontResidualToAnnotatedM'] = {'horizontal': float(math.hypot(diff[0], diff[1])),
                                                           'alongRoadAxis': float(diff @ ctx.a), 'lateralRight': float(diff @ ctx.r)}
    else:
        spec = defaults['vehicle']
        base = {'status': 'declared-default', 'label': spec['label'], 'dimensions': dict(spec['dimensions']),
                'setbackM': spec['setbackMeters'], 'reason': spec['reason'] + ('' if ctx.cuboids is not None else '; annotations not supplied, occupancy unknown'),
                'annotatedVehicleRejected': failures}
        chosen, default_reason = (None, 'No curb proxy at the near edge.') if curb_at_edge is None else default_vehicle(ctx, p0, eu, curb_at_edge)
        if chosen is not None and ctx.cuboids is not None:
            dims = spec['dimensions']; side = np.array([-ctx.a[1], ctx.a[0]])
            footprint = {'center': chosen['front'] - dims['length'] / 2 * ctx.a, 'axes': [ctx.a, side], 'half': [dims['length'] / 2, dims['width'] / 2]}
            for c in ctx.cuboids:
                h = unit(np.asarray(c['worldFromBox'], float)[[0, 2], 0])
                other = {'center': np.asarray(c['center'], float)[[0, 2]], 'axes': [h, np.array([-h[1], h[0]])], 'half': [c['length'] / 2, c['width'] / 2]}
                if rectangles_overlap(footprint, other):
                    chosen, default_reason = None, f"Annotated {c['category']} occupies the declared default footprint."
                    break
        if chosen is None:
            result_vehicle = {**base, 'status': 'unknown', 'supportClass': 'unknown', 'reason': default_reason}
        else:
            crossing_xz = chosen['crossing']
            result_vehicle = ctx.place(chosen['front'], {**base, 'curbLateralAlongLengthM': chosen['curbAlongLength'], 'lateralFromCameraM': chosen['lateral']})
    if crossing_xz is None:
        crossing_xz = start
    crossing = ctx.place(crossing_xz, {'status': 'mapped', 'label': 'crosswalk mapped', 'nearEdge': edge_name,
                                      'reason': f'Mapped near edge ({edge_name}) of crossing {ground["crossing"]["id"]} on the current vehicle centre line.',
                                      'paintOffset': PAINT_OFFSET.get(ground['crossing']['id'], UNMEASURED)})
    if crossing['status'] != 'unknown':
        crossing['forwardM'] = float(np.asarray(crossing['groundPoint'])[[0, 2]] @ ctx.a)
    pspec = defaults['person']
    if curb_point is None:
        person = {'status': 'unknown', 'label': pspec['label'], 'supportClass': 'unknown', 'reason': 'No curb proxy on the near edge.'}
    else:
        person = ctx.place(curb_point - pspec['curbOffsetMeters'] * eu, {'status': 'declared-default', 'label': pspec['label'],
                                                                         'heightM': pspec['heightMeters'], 'reason': pspec['reason']})
        if person['status'] != 'unknown' and ctx.cuboids is not None:  # diagnostic only: declared top of the person
            top = np.asarray(person['groundPoint']) + [0, pspec['heightMeters'], 0]
            person['topHiddenDiagnostic'] = bool(segment_hits_boxes([top], ctx.cuboids)[0])
    if 'groundPoint' in crossing and 'groundPoint' in result_vehicle:
        result_vehicle['pixelGroundGapAlongAxisM'] = float((np.asarray(crossing['groundPoint']) - result_vehicle['groundPoint'])[[0, 2]] @ ctx.a)
    curb = {'status': 'unknown', 'label': 'curb (map edge proxy)', 'reason': 'No drivable-area exit right of the ego lane on the near edge.'} \
        if curb_point is None else \
        {'status': 'mapped', 'label': 'curb (map edge proxy)', 'side': 'right', 'reason': 'Edge of the mapped drivable area right of the ego lane.',
         'nearEdgeXZ': [float(curb_point[0]), float(curb_point[1])], 'nearEdgeGroundY': ctx.height(curb_point), 'nearEdgeLateralM': curb_at_edge}
    road_axis = {'status': 'mapped', 'label': 'road axis (mapped ego lane)', 'laneId': ground['egoLane']['id'],
                 'directionOfTravel': [float(ctx.a[0]), 0., float(ctx.a[1])], 'skewDeg': skew, 'nearEdge': edge_name,
                 'cameraInLane': bool(inside_polygons([0., 0.], [ctx.lane_polygon])[0]),
                 'reason': 'Chord of the mapped ego lane; a declared vehicle moves along the reverse direction. Skew is reported, never a limit.'}
    return {'crossing': crossing, 'roadAxis': road_axis, 'curb': curb, 'vehicle': result_vehicle, 'person': person}


def load(grounded, metadata_sha256, buffers_dir, protocol_path, protocol_sha256):
    grounded = Path(grounded).resolve(); buffers_dir = Path(buffers_dir).resolve()
    scene_raw = snapshot(grounded / 'scene.json', metadata_sha256, 128 * 1024); scene = parse(scene_raw)
    m = scene['measurements']; descriptor = m.get('ground')
    require(m.get('groundPolicy') == 'mappedGround' and isinstance(descriptor, dict) and descriptor.get('file') == 'ground.json', 'Grounded scene required')
    ground_raw = snapshot(grounded / 'ground.json', descriptor['sha256'], 1024 * 1024); ground = parse(ground_raw)
    protocol_raw = snapshot(protocol_path, protocol_sha256); protocol = parse(protocol_raw)
    require(ground['protocol']['sha256'] == protocol_sha256 and ground['sceneId'] == scene['id'], 'Ground layer was built with a different protocol or scene')
    t = ground['terrain']; terrain_raw = snapshot(grounded / t['file'], t['sha256'], 16 * 1024 * 1024)
    require(len(terrain_raw) == t['bytes'] == t['rows'] * t['columns'] * 4, 'Terrain byte length mismatch')
    window = np.frombuffer(terrain_raw, '<f4').reshape(t['rows'], t['columns'])
    terrain = Terrain(window, t['column0'], t['row0'], t['rasterShape'], t['sim2'], ground['frame']['cityOrigin'])
    count = m['count']; raws = {}
    for key, dtype, width in (('referencePositions', '<f8', 3), ('referenceEligible', 'u1', 1)):
        raws[key] = snapshot(buffers_dir / m['buffers'][key], m['artifactHashes'][key], 16 * 1024 * 1024)
        require(len(raws[key]) == count * width * np.dtype(dtype).itemsize, 'Reference buffer length mismatch')
    positions = np.frombuffer(raws['referencePositions'], '<f8').reshape(count, 3)
    eligible = np.frombuffer(raws['referenceEligible'], 'u1').astype(bool)
    inputs = {'scene.json': digest(scene_raw), 'ground.json': digest(ground_raw), 'terrain.bin': digest(terrain_raw),
              'recorded.png': scene['recordedImage']['sha256'], 'referencepositions.bin': digest(raws['referencePositions']),
              'referenceeligible.bin': digest(raws['referenceEligible']), 'protocol.json': digest(protocol_raw)}
    return scene, ground, terrain, positions, eligible, protocol, inputs


def document(scene, ground, terrain, positions, eligible, protocol, inputs):
    objects = suggest(scene, ground, terrain, positions, eligible, protocol)
    doc = {'schemaVersion': 2, 'sceneId': scene['id'], 'sourceClass': 'mappedGround', 'generatedUtc': generated_utc(),
            'generator': {'script': 'scripts/california/suggest_av2_placements.py', 'sha256': digest(Path(__file__).read_bytes()),
                          'sharedScript': 'scripts/california/prepare_av2_ground.py', 'sharedSha256': digest((HERE / 'prepare_av2_ground.py').read_bytes()),
                          'python': platform.python_version(), 'numpy': np.__version__},
            'inputs': inputs, 'protocolSha256': inputs['protocol.json'], 'license': ground['license'],
            'tolerances': {k: ground[k] for k in ('roadToleranceMeters', 'supportHeightAgreementMeters', 'supportRadiusMeters', 'maxRangeMeters')},
            **objects,
            'limitations': ['One frame; no independent reference. Mapped ground is publisher terrain inside mapped drivable areas, not observed road.',
                            'The crossing is the mapped near edge on the current vehicle centre line; map-to-paint offset is shown as a label and never moves it.',
                            'A point labeled mapped, not observed has no camera-visible LiDAR return agreeing with the map within 0.5 m (for example in the LiDAR shadow).',
                            'A hidden point lies behind an annotated object in the image: the pixel shows that object, not the ground.',
                            'Annotated objects are from the LiDAR sweep about 40 ms before the image.',
                            'The person position and height are declared defaults, not detections.',
                            'Each pixel is the one within the protocol window whose ground hit is nearest the mapped point; '
                            'the remaining horizontal residual is recorded per object (pixelSearch.residualM).',
                            'The declared vehicle box starts at the vehicle pixel ground point and runs along the road axis with the '
                            'annotated dimensions; its residual to the annotated front and the annotated yaw difference are recorded.',
                            'Geometric visibility only, never Clear; not a safety, recognition or compliance judgment.']}
    require(BANNED not in canonical(doc).lower(), 'Suggestions must not use any form of the banned vehicle-status word')
    return doc


def write_exclusive(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('grounded', 'metadata-sha256', 'buffers', 'protocol', 'protocol-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    grounded = Path(args.grounded)
    try:
        for name in ('suggestions.json', 'catalog-entry.json'):
            require(not os.path.lexists(grounded / name), f'Existing {name} retained')
        loaded = load(args.grounded, args.metadata_sha256, args.buffers, args.protocol, args.protocol_sha256)
        data = canonical(document(*loaded))
        scene, ground = loaded[0], loaded[1]
        entry = {'id': scene['id'], 'label': scene['id'], 'metadataSha256': args.metadata_sha256,
                 'ground': {'file': 'ground.json', 'sha256': scene['measurements']['ground']['sha256']},
                 'suggestions': {'file': 'suggestions.json', 'sha256': digest(data)}}
        write_exclusive(grounded / 'suggestions.json', data)
        write_exclusive(grounded / 'catalog-entry.json', canonical(entry))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'AV2 suggestions rejected: {error}\n')
    print(digest(data))


if __name__ == '__main__':
    main()

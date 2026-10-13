"""Map-first placement suggestions (schema v3) for one grounded AV2 scene; never Clear, never invented.

Usage:
  SOURCE_DATE_EPOCH=<int> python3 -I -B scripts/california/suggest_av2_placements.py \
    --grounded <dir with scene.json, ground.json, terrain.bin> --metadata-sha256 <grounded scene.json sha256> \
    --buffers <assembled scene dir with the reference buffers> --protocol <protocol.json> --protocol-sha256 <sha256>

Writes <grounded>/suggestions.json and <grounded>/catalog-entry.json; refuses to overwrite either. Every
object is a crop pixel plus the ground point its camera ray reaches on the mapped ground raster inside the
mapped drivable area (the same rule the app applies), with a support class (observed-nearby / map-only /
unknown) and a hidden flag (ray passes through an annotated cuboid). Missing evidence stays unknown.
Protocol v3 only (exactProjection): each object stores the exact floating projection of its mapped point
(pixelUV, the position the app evaluates) and the integer pixel containing it (the display and nudge anchor).
Protocols v1/v2 are refused: their outputs are historical and were written by earlier versions of this script.
Protocol v3.2 (crossingApplicability) adds a guard: when the mapped-edge crossing model does not apply (near edge more than
45 degrees from perpendicular to the lane, an edge not ahead of the camera, or the centre line meeting the near-edge line off
the mapped edge or behind the camera), no crossing point is emitted; the crossing and everything anchored on it stay Unknown
with the named refusal in the reason. A protocol without that section runs unchanged (v3, v3.1).
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
pixel_ground_hit = ground_tool.pixel_ground_hit
inside_polygons, segment_hits_boxes, curb_scan = ground_tool.inside_polygons, ground_tool.segment_hits_boxes, ground_tool.curb_scan
EPS = 1e-9
VEHICLES = {'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER'}
STATIONARY_LABEL = 'stationary vehicle'
STATIONARY_REASON = 'stationary vehicle (AV2 annotation); reason for stopping unknown'
APP_MODEL = ('annotated pose: the averaged annotated box from ground.json (centre, horizontal heading, dimensions); '
             'the front pixel only names it')
# protocol v3.1 currentVehicle.declaredReplacement: a declared or resized vehicle that replaces the stationary vehicle while
# the front pixel still names it keeps the averaged front point (mappedPoint), never the front pixel's own ray hit.
APP_MODEL_REPLACEMENT = APP_MODEL + ('; a declared or resized vehicle in its place keeps this front point (mappedPoint) and runs '
                                     'along the lane axis from it')
REPLACEMENT_LIMITATION = ('A declared or resized vehicle that replaces the stationary vehicle while the front pixel still names it keeps the '
                          'stationary vehicle front (vehicle mappedPoint) on the lane axis; the front pixel ray hit '
                          '(pixelOffsetFromMappedM) is display only.')
# Outputs never use any form of the word 'park' for this scene (Sean, Oct 11; coordinator, Oct 12).
BANNED = b'park'
SUPPORT_LABEL = {'observed-nearby': 'observed nearby', 'map-only': 'mapped, not observed', 'unknown': 'unknown'}
CROSSING_ERROR_LABELS = {'unmeasured': 'paint offset unmeasured', 'pending-measurement': 'paint offset pending measurement',
                         'unresolved': 'paint offset unresolved'}
# protocol v3.2 crossingApplicability: the exact parameters and refusal codes this script implements. The protocol must carry
# exactly these (a different value needs a new script and protocol version, never a silent reinterpretation).
APPLICABILITY_PARAMETERS = {'maxSkewDeg': 45.0, 'minForwardMeters': 0.0, 'segmentToleranceMeters': 0.0}
REFUSAL_CODES = ('crossing-edge-not-ahead', 'crossing-skew-over-limit', 'crossing-parallel-to-lane',
                 'crossing-point-off-edge-segment', 'crossing-point-behind-camera')
REFUSAL_PREFIX = 'Crossing model does not apply'
ROAD_AXIS_REASON = 'Chord of the mapped ego lane; a declared vehicle moves along the reverse direction. Skew is reported, never a limit.'
ROAD_AXIS_REASON_V32 = ('Chord of the mapped ego lane; a declared vehicle moves along the reverse direction. A near edge more than '
                        '45 degrees from perpendicular to this axis refuses the crossing model (protocol v3.2).')
APPLICABILITY_LIMITATION = ('The mapped-edge crossing model applies only when both crossing edges are ahead of the camera, the near '
                            'edge is within 45 degrees of perpendicular to the lane, and the centre line meets the mapped near edge '
                            'segment ahead of the camera; otherwise the crossing, and the vehicle, person and curb anchored on it, '
                            'stay Unknown with the named refusal.')
MEASURED_ERROR_LIMITATION = ('A measured crossing error is the largest distance between the mapped near edge and the paint edge '
                             'within the measured range: |offset| + 2 x combined spread from the frozen LiDAR paint method, '
                             'rounded up to 0.01 m. It is development-grade, not calibrated, and excludes systematic errors.')


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
        require(protocol.get('protocolVersion') == 3 and isinstance(sel, dict) and sel.get('rule') == 'exactProjection',
                'Protocol v3 pixelSelection (exactProjection) required; protocol v1/v2 outputs are historical')
        band = ground.get('alongRayUncertainty', {}).get('heightBandMeters'); limit = ground.get('placementRangeLimitMeters')
        require(ground.get('schemaVersion') == 2 and type(band) is float and 0 < band <= 1 and type(limit) is float and
                limit == protocol['placementRange']['limitMeters'], 'Protocol v3 ground layer (schemaVersion 2) required')
        self.band, self.limit = band, limit

    def height(self, xz):
        h = self.terrain.sample([xz])[0][0]
        return None if not np.isfinite(h) else float(h) - self.oz

    def ground_hit(self, uv):
        """Runtime ray rule only: (ray direction, first mapped-ground hit within the range limit inside the drivable area
        or None, Unknown reason). uv is an integer pixel centre or a fractional crop position."""
        return pixel_ground_hit(self.scene, uv, self.terrain, self.polygons, self.ground['maxRangeMeters'], self.limit)

    def evaluate_pixel(self, uv):
        """The runtime rule: crop position -> first mapped-ground hit inside the drivable area, hidden flag, support class."""
        g, tol = self.ground, self.protocol['tolerances']
        d, hit, reason = self.ground_hit(uv)
        if hit is None:
            return {'supportClass': 'unknown', 'reason': reason}
        p = np.asarray(hit['point'])
        hidden = None if self.cuboids is None else bool(segment_hits_boxes([p], self.cuboids)[0])
        distances = np.linalg.norm(self.support - p[[0, 2]], axis=1) if len(self.support) else np.zeros(0)
        observed = bool((distances <= g['supportRadiusMeters']).any())
        uncertainty = self.band / -d[1]  # protocol v3 alongRayUncertainty (fitted height band, no half-ulp term)
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
                'alongRayUncertaintyM': float(uncertainty), 'rangeM': hit['t'], 'horizontalRangeM': float(math.hypot(p[0], p[2])),
                'hitKind': hit['kind'],
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

    def integer_check(self, pixel, xz):
        """Diagnostic only: the runtime rule at the integer anchor pixel and its horizontal residual to the mapped point."""
        _d, hit, reason = self.ground_hit(pixel)
        if hit is None:
            return {'reason': reason}
        return {'groundPoint': list(hit['point']), 'residualM': float(math.hypot(hit['point'][0] - xz[0], hit['point'][2] - xz[1]))}

    def place(self, xz, base):
        """Protocol v3 pixelSelection (exactProjection): desired mapped ground point -> its exact floating projection pixelUV
        (evaluated with the runtime rule, no rounding or search) and the integer pixel containing it (display/nudge anchor)."""
        h = self.height(xz)
        if h is None:
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'reason': 'No mapped ground height at the mapped point.'}
        desired = [float(xz[0]), h, float(xz[1])]; uv = project(self.scene, desired)
        if uv is None or not self.in_crop(uv):
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'mappedPoint': desired, 'reason': 'Mapped point is outside the recorded image.'}
        uv = [float(uv[0]), float(uv[1])]; pixel = [math.floor(uv[0] + .5), math.floor(uv[1] + .5)]
        result = self.evaluate_pixel(uv)
        if result['supportClass'] == 'unknown':  # no fallback pixel: the object stays Unknown with the runtime reason
            return {**base, 'status': 'unknown', 'supportClass': 'unknown', 'mappedPoint': desired, 'pixelUV': uv, 'pixel': pixel,
                    'reason': result['reason']}
        g = result['groundPoint']
        return {**base, **result, 'pixel': pixel, 'pixelUV': uv, 'mappedPoint': desired,
                'supportLabel': SUPPORT_LABEL[result['supportClass']],
                'pixelOffsetFromMappedM': float(math.hypot(g[0] - desired[0], g[2] - desired[2])),
                'integerPixelCheck': self.integer_check(pixel, xz)}

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

    def crossing_edges(self):
        """Both mapped crossing edges as (name, forward, p0, u, p1), with the same arithmetic as near_edge() but no
        requirement: protocol v3.2 refuses the crossing model instead of aborting the run."""
        edges = []
        for name in ('edge1', 'edge2'):
            e = np.asarray(self.ground['crossing'][name], float)[:, [0, 2]]; p0, p1 = e[0], e[-1]
            u = unit(p1 - p0); u = u if u @ self.r > 0 else -u
            edges.append((name, float(((p0 + p1) / 2) @ self.a), p0, u, p1))
        return edges

    def curb_lateral_at(self, forward):
        start = self.m0 + (forward - self.m0 @ self.a) * self.a
        s = curb_scan(start, self.r, self.polygons, self.ground['boundaryEpsilonMeters'])
        return None if s is None else float(start @ self.r + s)


def applicability_rules(protocol):
    """protocol v3.2 crossingApplicability parameters, or None for a protocol without that section (v3, v3.1 run unchanged)."""
    spec = protocol.get('crossingApplicability')
    if spec is None:
        require(protocol.get('protocolRevision') != '3.2', 'Protocol v3.2 needs its crossingApplicability section')
        return None
    require(isinstance(spec, dict) and spec.get('parameters') == APPLICABILITY_PARAMETERS and spec.get('refusalCodes') == list(REFUSAL_CODES),
            'crossingApplicability must carry exactly the parameters and refusal codes this script implements')
    return dict(APPLICABILITY_PARAMETERS)


def point_problems(ctx, point, p0, p1, rules, line):
    """protocol v3.2: where a centre line meets the near-edge line, the point must lie on the mapped edge segment (p0..p1,
    within segmentToleranceMeters) and ahead of the camera (forward along the road axis > minForwardMeters)."""
    seg = np.asarray(p1, float) - p0; length = float(np.linalg.norm(seg)); s = float((np.asarray(point, float) - p0) @ seg) / length
    tol = rules['segmentToleranceMeters']; problems = []
    if s < -tol or s > length + tol:
        outside = -s if s < 0 else s - length
        problems.append(('crossing-point-off-edge-segment',
                         f'The {line} meets the near-edge line {outside:.2f} m beyond the end of the mapped edge ({length:.2f} m long).'))
    forward = float(np.asarray(point, float) @ ctx.a)
    if forward <= rules['minForwardMeters']:
        problems.append(('crossing-point-behind-camera',
                         f'The {line} meets the near-edge line {-forward:.2f} m behind the camera (along the road axis).'))
    return problems


def refusal_reason(crossing_id, problems):
    codes = sorted({code for code, _ in problems}, key=REFUSAL_CODES.index)
    return f'{REFUSAL_PREFIX} to crossing {crossing_id}: ' + ' '.join(text for _, text in problems) + f' Refusal: {", ".join(codes)}.'


def crossing_error(protocol, crossing_id):
    """protocol v3 crossingError: the crossing stays the mapped near edge; the shown +/- is the measured mapped-to-paint
    offset, or a named status while it is unmeasured, pending measurement or unresolved."""
    entry = protocol['crossingError']['perCrossing'].get(crossing_id, {'status': 'unmeasured'})
    status = entry.get('status'); base = {'basis': 'mapped near edge', 'status': status}
    if status == 'measured':
        value, evidence = entry.get('plusMinusM'), entry.get('evidence')
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 5 and isinstance(evidence, dict) and
                set(evidence) == {'file', 'sha256'} and isinstance(evidence['sha256'], str) and len(evidence['sha256']) == 64,
                'A measured crossing error needs plusMinusM (0-5 m) and its evidence {file, sha256}')
        if 'offsetM' in entry or 'combinedM' in entry:  # protocol v3.2: plusMinusM = |offset| + 2 x combined spread
            offset, combined = entry.get('offsetM'), entry.get('combinedM')
            require(all(type(x) is float and math.isfinite(x) for x in (offset, combined)) and combined >= 0 and
                    abs(float(value) - (abs(offset) + 2 * combined)) <= 1e-12,
                    'A measured crossing error must equal |offsetM| + 2 x combinedM')
        shown = math.ceil(float(value) * 100 - 1e-9) / 100
        return {**base, 'label': f'±{shown:.2f} m', 'plusMinusM': shown, 'evidence': dict(evidence)}
    require(status in CROSSING_ERROR_LABELS, f'Unknown crossing error status {status!r}')
    evidence = entry.get('evidence')
    require(status != 'unresolved' or (isinstance(evidence, dict) and set(evidence) == {'file', 'sha256'}),
            'An unresolved crossing error needs its evidence {file, sha256}')
    return {**base, 'label': CROSSING_ERROR_LABELS[status], 'plusMinusM': None, 'evidence': None if evidence is None else dict(evidence)}


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
    if 'poseAverage' not in box:
        failures.append('No averaged annotated pose in the ground layer (protocol v3 currentVehicle.pose).')
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


def refused(ctx, ground, protocol, edge_name, skew, problems):
    """protocol v3.2: the crossing model does not apply. No crossing point is emitted; the crossing and the vehicle, person
    and curb anchored on it are Unknown with the named refusal; the road axis stays the mapped lane chord."""
    crossing_id = ground['crossing']['id']; reason = refusal_reason(crossing_id, problems)
    anchored = 'Not placed: {} is anchored on the crossing, and the crossing model does not apply here (see the crossing reason).'
    crossing = {'status': 'unknown', 'supportClass': 'unknown', 'nearEdge': edge_name, 'reason': reason,
                'crossingError': crossing_error(protocol, crossing_id)}
    vehicle = {'status': 'unknown', 'supportClass': 'unknown', 'reason': anchored.format('the vehicle selection')}
    person = {'status': 'unknown', 'label': protocol['declaredDefaults']['person']['label'], 'supportClass': 'unknown',
              'reason': anchored.format('the declared person')}
    curb = {'status': 'unknown', 'label': 'curb (map edge proxy)', 'reason': anchored.format('the curb proxy')}
    return {'crossing': crossing, 'roadAxis': road_axis(ctx, ground, skew, edge_name, ROAD_AXIS_REASON_V32), 'curb': curb,
            'vehicle': vehicle, 'person': person}


def road_axis(ctx, ground, skew, edge_name, reason):
    return {'status': 'mapped', 'label': 'road axis (mapped ego lane)', 'laneId': ground['egoLane']['id'],
            'directionOfTravel': [float(ctx.a[0]), 0., float(ctx.a[1])], 'skewDeg': skew, 'nearEdge': edge_name,
            'cameraInLane': bool(inside_polygons([0., 0.], [ctx.lane_polygon])[0]), 'reason': reason}


def suggest(scene, ground, terrain, positions, eligible, protocol):
    ctx = Context(scene, ground, terrain, positions, eligible, protocol)
    rules = applicability_rules(protocol); problems = []
    if rules is None:  # protocol v3 / v3.1: unchanged (an edge not ahead aborts the run)
        edge_name, edge_forward, p0, eu = ctx.near_edge()
    else:
        edges = ctx.crossing_edges()
        problems = [('crossing-edge-not-ahead', f'Crossing {name} midpoint is {forward:.2f} m along the road axis, not ahead of the camera.')
                    for name, forward, *_ in edges if forward <= rules['minForwardMeters']]
        edge_name, edge_forward, p0, eu = ctx.near_edge() if not problems else min(edges, key=lambda e: e[1])[:4]
        p1 = next(e[4] for e in edges if e[0] == edge_name)
    skew = 90 - math.degrees(math.acos(min(1., abs(float(eu @ ctx.a)))))
    start, _ = intersect(ctx.m0, ctx.a, p0, eu)
    if rules is not None:
        if skew > rules['maxSkewDeg']:
            problems.append(('crossing-skew-over-limit', f'The near edge ({edge_name}) is {skew:.1f} degrees from perpendicular to the '
                                                         f'lane (limit {rules["maxSkewDeg"]:g}).'))
        if start is None:
            problems.append(('crossing-parallel-to-lane', 'The ego lane chord never meets the near-edge line (parallel).'))
        else:
            problems += point_problems(ctx, start, p0, p1, rules, 'ego lane chord')
        if problems:
            return refused(ctx, ground, protocol, edge_name, skew, problems)
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
                'pose': {'kind': 'averaged', 'annotations': box['poseAverage']['rows']},
                'headingWorld': [float(vehicle['heading'][0]), 0., float(vehicle['heading'][1])],
                'headingToRoadAxisDeg': vehicle['headingToRoadAxisDeg'], 'yawToRoadAxisDeg': vehicle['yawToRoadAxisDeg'],
                'frontGapToCrossingM': vehicle['gap'],
                'appModel': APP_MODEL_REPLACEMENT if 'declaredReplacement' in ctx.protocol['currentVehicle'] else APP_MODEL}
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
    elif rules is not None:  # protocol v3.2: the vehicle centre line must also meet the mapped edge ahead of the camera
        line = 'current vehicle centre line' if vehicle is not None else 'declared vehicle centre line'
        problems = point_problems(ctx, crossing_xz, p0, p1, rules, line)
        if problems:
            return refused(ctx, ground, protocol, edge_name, skew, problems)
    crossing = ctx.place(crossing_xz, {'status': 'mapped', 'label': 'crosswalk mapped', 'nearEdge': edge_name,
                                      'reason': f'Mapped near edge ({edge_name}) of crossing {ground["crossing"]["id"]} on the current vehicle centre line.',
                                      'crossingError': crossing_error(protocol, ground['crossing']['id'])})
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
    axis = road_axis(ctx, ground, skew, edge_name, ROAD_AXIS_REASON if rules is None else ROAD_AXIS_REASON_V32)
    return {'crossing': crossing, 'roadAxis': axis, 'curb': curb, 'vehicle': result_vehicle, 'person': person}


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


def along_ray_limitation(spec):
    """The along-ray limitation names how the height band was obtained: fitted here (development) or fixed (fitted on development
    scenes and only tested on this scene's returns)."""
    tail = 'it describes the mapped-ground height error along the pixel ray, not the crossing position or a physical accuracy bound.'
    if spec['method'] == 'fixed':
        return (f'The along-ray uncertainty uses a fixed height band fitted on development scenes ({spec["fittedOn"]}); this scene\'s '
                f'LiDAR ground returns only test it (heldOutResiduals); ' + tail)
    require(spec['method'] == 'fitThisDevelopmentScene', 'Unknown height band method')
    return 'The along-ray uncertainty uses a height band fitted on this development scene\'s LiDAR ground returns; ' + tail


def document(scene, ground, terrain, positions, eligible, protocol, inputs):
    objects = suggest(scene, ground, terrain, positions, eligible, protocol)
    doc = {'schemaVersion': 3, 'sceneId': scene['id'], 'sourceClass': 'mappedGround', 'generatedUtc': generated_utc(),
            'generator': {'script': 'scripts/california/suggest_av2_placements.py', 'sha256': digest(Path(__file__).read_bytes()),
                          'sharedScript': 'scripts/california/prepare_av2_ground.py', 'sharedSha256': digest((HERE / 'prepare_av2_ground.py').read_bytes()),
                          'python': platform.python_version(), 'numpy': np.__version__},
            'inputs': inputs, 'protocolSha256': inputs['protocol.json'], 'license': ground['license'],
            'tolerances': {**{k: ground[k] for k in ('roadToleranceMeters', 'supportHeightAgreementMeters', 'supportRadiusMeters', 'maxRangeMeters',
                                                    'placementRangeLimitMeters')},
                           'alongRayHeightBandMeters': ground['alongRayUncertainty']['heightBandMeters']},
            **objects,
            'limitations': ['One frame; no independent reference. Mapped ground is publisher terrain inside mapped drivable areas, not observed road.',
                            'The crossing is the mapped near edge on the current vehicle centre line; its error is the map-to-paint offset '
                            '(crossingError), which never moves it.',
                            'A point labeled mapped, not observed has no camera-visible LiDAR return agreeing with the map within 0.5 m (for example in the LiDAR shadow).',
                            'A hidden point lies behind an annotated object in the image: the pixel shows that object, not the ground.',
                            'Annotated objects other than the current vehicle are from the LiDAR sweep about 40 ms before the image; the current '
                            'stationary vehicle is the average of all its annotations in the log.',
                            'The person position and height are declared defaults, not detections.',
                            'Each object is evaluated at the exact floating projection of its mapped point (pixelUV); the integer pixel only '
                            'anchors display and small edits. pixelOffsetFromMappedM is internal consistency, not accuracy.',
                            along_ray_limitation(ground['alongRayUncertainty']),
                            f'Placements more than {ground["placementRangeLimitMeters"]:g} m from the camera (horizontal) are refused.',
                            *([REPLACEMENT_LIMITATION] if 'declaredReplacement' in protocol['currentVehicle'] and
                              objects['vehicle']['status'] == 'annotated' else []),
                            *([APPLICABILITY_LIMITATION] if applicability_rules(protocol) is not None else []),
                            *([MEASURED_ERROR_LIMITATION] if objects['crossing']['crossingError']['status'] == 'measured' and
                              applicability_rules(protocol) is not None else []),
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

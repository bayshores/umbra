import { whatIfBox } from './geometry.mjs';

// Night-map layers for the top-down view, drawn only from the scene's real data: the publisher map in ground.json
// (drivable areas, the ego lane's painted boundaries, the crossing polygon) and its annotated objects, plus the lane ruler
// computed from the comparison itself. Nothing here is invented scenery: no buildings, sidewalks or street names.

const sub = (a, b) => a.map((v, i) => v - b[i]), add = (a, b) => a.map((v, i) => v + b[i]), mul = (a, k) => a.map(v => v * k);
const flat = v => { const n = Math.hypot(v[0], v[2]); return [v[0] / n, 0, v[2] / n]; };

// Painted line style per AV2 mark type: colour, double or single, dashed or solid. Unknown or unpainted marks are skipped.
export function markStyle(type) {
  const t = String(type ?? '').toUpperCase();
  if (!t || t === 'NONE' || t === 'UNKNOWN') return null;
  const color = t.includes('YELLOW') ? 'yellow' : t.includes('WHITE') ? 'white' : null;
  if (!color) return null;
  return { color, double: t.startsWith('DOUBLE'), dashed: t.includes('DASH') };
}

// Footprint corners [x, z] of an annotated cuboid (worldFromBox columns: length, width, height axes).
export function footprint(item) {
  const R = item.worldFromBox, [cx, , cz] = item.center;
  const along = [R[0][0], R[2][0]], across = [R[0][1], R[2][1]];
  return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([l, w]) => [
    cx + along[0] * item.length / 2 * l + across[0] * item.width / 2 * w,
    cz + along[1] * item.length / 2 * l + across[1] * item.width / 2 * w]);
}

const KIND = { REGULAR_VEHICLE: 'vehicle', LARGE_VEHICLE: 'vehicle', BOX_TRUCK: 'vehicle', TRUCK: 'vehicle', BUS: 'vehicle',
  PEDESTRIAN: 'person', BICYCLIST: 'cyclist', BICYCLE: 'cyclist', MOTORCYCLIST: 'cyclist', MOTORCYCLE: 'cyclist' };

// Map layers from a mapped-ground scene's ground.json, or null when the scene has no map (measured-depth scenes).
export function mapLayers(ground) {
  if (!ground?.drivableAreas || !ground?.egoLane || !ground?.crossing) return null;
  const { edge1, edge2 } = ground.crossing;
  return {
    areas: ground.drivableAreas.map(a => a.outline).filter(o => Array.isArray(o) && o.length >= 3),
    lanes: [
      { line: ground.egoLane.leftBoundary, style: markStyle(ground.egoLane.leftMarkType) },
      { line: ground.egoLane.rightBoundary, style: markStyle(ground.egoLane.rightMarkType) },
    ].filter(l => l.style && l.line?.length >= 2),
    crossing: [edge1[0], edge1[1], edge2[1], edge2[0]],
    objects: (ground.cuboids?.items ?? []).map(item => ({ kind: KIND[item.category] ?? 'other', corners: footprint(item),
      center: item.center, ground: item.center[1] - item.height / 2 })),
  };
}

// The what-if vehicle's front point at setback 0 (the box end nearest the crossing) and its lateral right side.
function rulerFrame(scenario) {
  const axis = flat(scenario.axis), box = whatIfBox(scenario.whatIf, 0), along = flat(box.axis);
  const ends = [add(box.center, mul(along, box.vehicle.length / 2)), add(box.center, mul(along, -box.vehicle.length / 2))];
  const dist = p => Math.hypot(p[0] - scenario.crossing[0], p[2] - scenario.crossing[2]);
  const front = dist(ends[0]) <= dist(ends[1]) ? ends[0] : ends[1];
  // The driver faces the crossing (against the setback axis); their right on the ground is (axis.z, 0, -axis.x).
  const right = [axis[2], 0, -axis[0]];
  const groundY = box.center[1] - box.vehicle.height / 2;
  return { axis, right, origin: [front[0], groundY, front[2]], offset: box.vehicle.width / 2 + .7 };
}

// Ruler dots every `step` metres from the crossing (0 m) to the driver, beside the what-if vehicle's line, each with the
// state the checked distances found at that setback ('blocked', 'unknown', 'overlap') or null where nothing was tested:
// outside the checked range (rows sorted by setback) or more than half a step from every checked row.
export function rulerDots(scenario, rows = null, step = .25) {
  const f = rulerFrame(scenario), end = scenario.limits.captureDistance, dots = [];
  const tested = rows?.length ? rows : null, lo = tested?.[0].setback, hi = tested?.at(-1).setback;
  for (let k = 0; k * step <= end + 1e-9; k++) {
    const s = k * step;
    const row = tested && s >= lo - 1e-9 && s <= hi + 1e-9
      && tested.reduce((best, r) => Math.abs(r.setback - s) < Math.abs(best.setback - s) ? r : best, tested[0]);
    const state = row && Math.abs(row.setback - s) <= step / 2 + 1e-9 ? row.state : null;
    dots.push({ setback: s, state, position: add(add(f.origin, mul(f.axis, s)), mul(f.right, f.offset)) });
  }
  return dots;
}

// Where a ruler label, the proposed-vehicle handle or the driver handle sits: on the ruler line, `extra` metres farther out.
export function rulerPoint(scenario, setback, extra = 0) {
  const f = rulerFrame(scenario);
  return add(add(f.origin, mul(f.axis, setback)), mul(f.right, f.offset + extra));
}

// Distance along the setback axis from the crossing to a ground point (the inverse of rulerPoint along the axis).
export function setbackAt(scenario, point) {
  const f = rulerFrame(scenario);
  const d = sub(point, f.origin);
  return d[0] * f.axis[0] + d[2] * f.axis[2];
}

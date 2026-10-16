import { mappedGroundHeight } from './geometry.mjs';

const measuredGrid = scene => scene.ground?.schemaVersion === 4 ? scene.ground.measuredGround : null;

export function mapSurface(scene) {
  const s = measuredGrid(scene);
  if (!s) return null;
  const positions = [], triangles = [], vertices = new Int32Array(s.heights.length).fill(-1);
  for (let row = 0; row < s.rows; row++) for (let column = 0; column < s.columns; column++) {
    const k = row * s.columns + column, h = s.heights[k];
    if (h === null) continue;
    vertices[k] = positions.length / 3;
    positions.push(s.originXZ[0] + column * s.cellMeters, h, s.originXZ[1] + row * s.cellMeters);
  }
  // Match geometry.mjs measuredTriangle: the shared diagonal is (column,row) -> (column+1,row+1).
  for (let row = 0; row < s.rows - 1; row++) for (let column = 0; column < s.columns - 1; column++) {
    const k = row * s.columns + column, a = vertices[k], d = vertices[k + s.columns + 1];
    if (a < 0 || d < 0) continue;
    const right = vertices[k + 1], above = vertices[k + s.columns];
    if (right >= 0) triangles.push(a, d, right);
    if (above >= 0) triangles.push(a, above, d);
  }
  return { positions: new Float32Array(positions), triangles: new Uint32Array(triangles) };
}

// Only the Map's draw index changes; the measured and reconstruction buffers remain shared and untouched.
export function mapObjectPointIndices(scene, positions) {
  const s = measuredGrid(scene), gate = s?.parameters?.groundGateMeters;
  if (!s || !Number.isFinite(gate) || gate < 0) return null;
  const indices = [];
  for (let i = 0; i < positions.length; i += 3) {
    const height = mappedGroundHeight(scene, positions[i], positions[i + 2]);
    if (height === null || Math.abs(positions[i + 1] - height) > gate) indices.push(i / 3);
  }
  return new Uint32Array(indices);
}

export function cameraViewEdges(scene) {
  const c = scene.camera, R = scene.transform.worldFromCamera;
  // The ground-ray helpers do not export a ray for image-edge columns (including width).
  const directions = [0, c.width].map(u => {
    const x = (u - c.cx) / c.fx, ray = R.map(row => row[0] * x + row[2]), n = Math.hypot(ray[0], ray[2]);
    return [ray[0] / n, 0, ray[2] / n];
  });
  let length = scene.ground?.maxRangeMeters;
  if (!(length > 0)) {
    length = 0;
    const bounds = scene.reconstruction?.bounds ?? scene.bounds;
    for (const x of [bounds.min[0], bounds.max[0]]) for (const z of [bounds.min[2], bounds.max[2]])
      length = Math.max(length, Math.hypot(x - c.eye[0], z - c.eye[2]));
  }
  return { origin: [...c.eye], directions, length };
}

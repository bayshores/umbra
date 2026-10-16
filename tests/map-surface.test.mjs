import assert from 'node:assert/strict';
import test from 'node:test';
import { mappedGroundHeight } from '../src/geometry.mjs';
import { mapSurface, mapObjectPointIndices, cameraViewEdges } from '../src/map-surface.mjs';

const near = (a, b, tolerance = 1e-6) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
function fixture(yaw = 0) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return {
    camera: { eye: [2, 1.5, 3], forward: [s, 0, -c], up: [0, 1, 0], width: 800, height: 400, fx: 600, fy: 650, cx: 230, cy: 170 },
    transform: { worldFromCamera: [[c, 0, s], [0, -1, 0], [s, 0, -c]] },
    bounds: { min: [-3, -2, -20], max: [8, 4, 2] },
    ground: { schemaVersion: 4, maxRangeMeters: 40, measuredGround: {
      columns: 3, rows: 3, originXZ: [-1.25, 2.5], cellMeters: .5,
      heights: [-1.5, -1.47, -1.4, -1.6, -1.41, -1.32, -1.55, -1.48, -1.37],
      parameters: { groundGateMeters: .125 },
    } },
  };
}
function meshHeight(mesh, x, z) {
  for (let i = 0; i < mesh.triangles.length; i += 3) {
    const [a, b, c] = Array.from(mesh.triangles.subarray(i, i + 3), k => mesh.positions.subarray(k * 3, k * 3 + 3));
    const dx = x - a[0], dz = z - a[2], bx = b[0] - a[0], bz = b[2] - a[2], cx = c[0] - a[0], cz = c[2] - a[2];
    const det = bx * cz - bz * cx, u = (dx * cz - dz * cx) / det, v = (bx * dz - bz * dx) / det;
    if (u >= -1e-10 && v >= -1e-10 && u + v <= 1 + 1e-10) return a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]);
  }
  return null;
}

test('map surface: mesh heights agree with mappedGroundHeight on both halves of nonplanar cells', () => {
  for (const [originXZ, cellMeters] of [[[-1.25, 2.5], .5], [[1.375, -3.25], .75]]) {
    const scene = fixture(), s = scene.ground.measuredGround;
    Object.assign(s, { originXZ, cellMeters });
    const before = structuredClone(scene), mesh = mapSurface(scene);
    assert.equal(mesh.triangles.length, (s.rows - 1) * (s.columns - 1) * 6);
    for (let row = 0; row < s.rows - 1; row++) for (let column = 0; column < s.columns - 1; column++) {
      for (const [u, v] of [[.2, .1], [.8, .1], [.1, .2], [.1, .8], [.5, .5], [.999, .998], [.998, .999]]) {
        const x = originXZ[0] + (column + u) * cellMeters, z = originXZ[1] + (row + v) * cellMeters;
        const expected = mappedGroundHeight(scene, x, z), actual = meshHeight(mesh, x, z);
        assert.notEqual(expected, null); assert.notEqual(actual, null);
        near(actual, expected);
      }
    }
    assert.deepEqual(scene, before);
  }
});

test('map surface: every null-vertex combination omits exactly the unsupported triangles', () => {
  for (let mask = 0; mask < 16; mask++) {
    const scene = fixture(), s = scene.ground.measuredGround;
    Object.assign(s, { columns: 2, rows: 2, originXZ: [0, 0], cellMeters: 1, heights: [-1, -1.2, -.8, -1.1].map((h, i) => mask & (1 << i) ? null : h) });
    const mesh = mapSurface(scene);
    let supported = 0;
    for (const [x, z] of [[.75, .25], [.25, .75]]) {
      const expected = mappedGroundHeight(scene, x, z), actual = meshHeight(mesh, x, z);
      if (expected === null) assert.equal(actual, null, `mask ${mask} at ${x},${z}`);
      else { supported++; assert.notEqual(actual, null); near(actual, expected); }
    }
    assert.equal(mesh.triangles.length, supported * 3, `mask ${mask}`);
    assert.ok(mesh.positions.every(Number.isFinite));
  }
});

test('map surface: an interior hole is never bridged and the mesh does not extend outside the grid', () => {
  const scene = fixture(), s = scene.ground.measuredGround;
  Object.assign(s, { columns: 4, rows: 4, originXZ: [0, 0], cellMeters: .5, heights: Array(16).fill(-1) });
  s.heights[5] = null;
  const mesh = mapSurface(scene);
  for (let z = .0625; z < 1.5; z += .125) for (let x = .03125; x < 1.5; x += .125) {
    const expected = mappedGroundHeight(scene, x, z);
    assert.equal(meshHeight(mesh, x, z), expected);
  }
  for (const [x, z] of [[-.01, .5], [.5, -.01], [1.51, .5], [.5, 1.51]]) assert.equal(meshHeight(mesh, x, z), null);
});

test('map surface: triangles face upward for quiet smooth slope shading', () => {
  const mesh = mapSurface(fixture());
  for (let i = 0; i < mesh.triangles.length; i += 3) {
    const [a, b, c] = Array.from(mesh.triangles.subarray(i, i + 3), k => mesh.positions.subarray(k * 3, k * 3 + 3));
    assert.ok((b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]) > 0);
  }
});

test('map dots: only returns within the configured surface-height gate are removed', () => {
  const scene = fixture(), s = scene.ground.measuredGround;
  Object.assign(s, { columns: 2, rows: 2, originXZ: [0, 0], cellMeters: 1, heights: [-1, null, -1, -1] });
  const positions = new Float32Array([
    .25, -1, .75, .25, -.875, .75, .25, -1.125, .75,
    .25, -.86, .75, .25, -1.14, .75,
    .75, -1, .25, 1.25, -1, .75, -.1, -1, .75, 1, -1, .75,
  ]);
  scene.positions = positions;
  scene.colors = new Float32Array(positions.length).fill(.5);
  const before = structuredClone(scene);
  assert.deepEqual([...mapObjectPointIndices(scene, positions)], [3, 4, 5, 6, 7, 8], 'above/below gate, missing triangle, outside and outer edge stay');
  assert.deepEqual(scene, before, 'drawing never changes source or calibration data');
  s.parameters.groundGateMeters = .25;
  assert.deepEqual([...mapObjectPointIndices(scene, positions)], [5, 6, 7, 8], 'reads the scene gate rather than assuming a constant');
});

test('map dots: height gate follows the sloping triangle, including its specified diagonal', () => {
  const scene = fixture(), s = scene.ground.measuredGround;
  Object.assign(s, { columns: 2, rows: 2, originXZ: [0, 0], cellMeters: 1, heights: [-1, -.5, -1.25, -.75] });
  const positions = new Float32Array([.75, -.6875, .25, .25, -1.0625, .75, .5, -.875, .5, .25, -.5, .75]);
  assert.deepEqual([...mapObjectPointIndices(scene, positions)], [3]);
});

test('map dots: a reconstruction gets its own display index without changes to either point buffer', () => {
  const scene = fixture(), s = scene.ground.measuredGround;
  s.heights.fill(-1);
  scene.positions = new Float32Array([-1, 2, 2.75]);
  scene.reconstruction = { positions: new Float32Array([-1, -1, 2.75, -1, 2, 2.75]), colors: new Float32Array(6).fill(.5) };
  const before = structuredClone(scene);
  assert.deepEqual([...mapObjectPointIndices(scene, scene.reconstruction.positions)], [1]);
  assert.deepEqual(scene, before);
});

test('map dots: missing or invalid gate metadata leaves all dots intact', () => {
  const scene = fixture(), points = new Float32Array([-1, -1.5, 2.5]);
  for (const gate of [undefined, null, NaN, -.1]) {
    scene.ground.measuredGround.parameters.groundGateMeters = gate;
    assert.equal(mapObjectPointIndices(scene, points), null);
  }
});

test('camera view: straight camera edges use columns zero and width at the principal row', () => {
  const scene = fixture(), before = structuredClone(scene), { camera: c } = scene, edges = cameraViewEdges(scene);
  near(Math.atan2(edges.directions[0][0], -edges.directions[0][2]), -Math.atan(c.cx / c.fx), 1e-12);
  near(Math.atan2(edges.directions[1][0], -edges.directions[1][2]), Math.atan((c.width - c.cx) / c.fx), 1e-12);
  assert.deepEqual(edges.origin, c.eye);
  assert.equal(edges.length, scene.ground.maxRangeMeters);
  assert.ok(edges.directions.every(d => d[1] === 0));
  assert.deepEqual(scene, before);
});

test('camera view: both field-of-view directions rotate with the camera in the scene frame', () => {
  for (const yaw of [Math.PI / 3, -Math.PI / 4, Math.PI * .95]) {
    const scene = fixture(yaw), c = scene.camera, edges = cameraViewEdges(scene);
    const angles = [yaw - Math.atan(c.cx / c.fx), yaw + Math.atan((c.width - c.cx) / c.fx)];
    edges.directions.forEach((d, i) => {
      near(d[0], Math.sin(angles[i]), 1e-12); near(d[2], -Math.cos(angles[i]), 1e-12);
      near(Math.hypot(...d), 1, 1e-12);
    });
  }
});

test('camera view: without a range limit, edges reach the far edge of the drawn data', () => {
  const scene = fixture();
  delete scene.ground;
  near(cameraViewEdges(scene).length, Math.hypot(8 - scene.camera.eye[0], -20 - scene.camera.eye[2]), 1e-12);
  scene.reconstruction = { bounds: { min: [-10, -2, -30], max: [20, 4, 2] } };
  near(cameraViewEdges(scene).length, Math.hypot(20 - scene.camera.eye[0], -30 - scene.camera.eye[2]), 1e-12);
});

test('map surface: scenes without measuredGround get camera edges and unchanged dots only', () => {
  for (const ground of [undefined, { schemaVersion: 3 }, { schemaVersion: 4 }]) {
    const scene = fixture();
    scene.ground = ground;
    scene.positions = new Float32Array([0, 0, -4, 1, 0, -8]);
    const before = structuredClone(scene);
    assert.equal(mapSurface(scene), null);
    assert.equal(mapObjectPointIndices(scene, scene.positions), null);
    assert.equal(cameraViewEdges(scene).directions.length, 2);
    assert.deepEqual(scene, before);
  }
});

import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { planCamera } from '../src/plan-camera.mjs';
import { createScenario, measuredPoint } from '../src/geometry.mjs';
import { scene, input } from './scene-input.mjs';

test('top-down framing stays fixed as vehicle and driver move', () => {
  const initial = createScenario(scene, input);
  const moved = createScenario(scene, {...input, proposedSetback:initial.limits.maximumSetback, driverDistance:initial.limits.minimumDriverDistance});
  assert.deepEqual(initial.target, moved.target, 'moving vehicle and camera never relocates the person');
  for (const aspect of [3.8, 1, .65]) {
    const a = planCamera(scene, initial, aspect), b = planCamera(scene, moved, aspect);
    assert.deepEqual(a.matrixWorld.elements, b.matrixWorld.elements);
    assert.deepEqual(a.projectionMatrix.elements, b.projectionMatrix.elements);
    for (const scenario of [initial, moved]) {
      const points = [scenario.target, scenario.eye, scenario.crossing];
      for (const {center} of [scenario.baseline, scenario.proposed]) {
        // Use the actual Three.js vehicle rotation, independently of camera framing.
        const rotation = new THREE.Matrix4().makeRotationY(Math.atan2(scenario.axis[0], scenario.axis[2]));
        for (const side of [-1, 1]) for (const end of [-1, 1]) {
          const corner = new THREE.Vector3(side * scenario.vehicle.width / 2, -scenario.vehicle.height / 2, end * scenario.vehicle.length / 2).applyMatrix4(rotation).add(new THREE.Vector3(...center));
          assert.ok(Math.abs(corner.y - measuredPoint(scene, input.pixels.parking)[1]) < 1e-9, 'vehicle retains the measured front-anchor height');
          points.push(corner.toArray());
        }
      }
      for (const point of points) {
        const p = new THREE.Vector3(...point).project(a);
        assert.ok(Math.abs(p.x) < .9 && Math.abs(p.y) < .9 && Math.abs(p.z) < 1, 'actor and driver remain in view with margin');
      }
    }
  }
});

import * as THREE from 'three';
import { whatIfBox } from './geometry.mjs';

// Frame the entire allowed movement once, so dragging never moves the map.
export function planCamera(scene, scenario, aspect) {
  const axis = scenario ? new THREE.Vector3(scenario.axis[0], 0, scenario.axis[2]).normalize() : new THREE.Vector3(0, 0, 1);
  const side = new THREE.Vector3(axis.z, 0, -axis.x);
  // Use the long side of the viewport for the approach, keeping actors readable.
  const right = aspect >= 1.4 ? axis.clone() : side.clone();
  const up = aspect >= 1.4 ? side.clone() : axis.clone().negate();
  const points = [];
  if (scenario) {
    points.push(scenario.crossing, scenario.target, scene.camera.eye);
    points.push(scene.camera.eye.map((v, i) => v + axis.getComponent(i) * (scenario.limits.minimumDriverDistance - scenario.limits.captureDistance)));
    // Frame the corners of the boxes actually drawn: the current vehicle and the what-if at both ends of its movement
    // (setback 0 to the maximum; it moves in a straight line, so every setback between is inside). geometry.mjs whatIfBox
    // places each one: a declared box on its own line (on mapped scenes through its own front, not the crossing), an
    // annotated pose with its own heading and size.
    for (const b of [scenario.baseline, whatIfBox(scenario.whatIf, 0), whatIfBox(scenario.whatIf, scenario.limits.maximumSetback)]) {
      const along = new THREE.Vector3(b.axis[0], 0, b.axis[2]).normalize(), across = new THREE.Vector3(along.z, 0, -along.x);
      for (const sign of [-1, 1]) for (const end of [-1, 1]) {
        points.push(b.center.map((v, i) => v + along.getComponent(i) * end * b.vehicle.length / 2 + across.getComponent(i) * sign * b.vehicle.width / 2));
      }
    }
  } else {
    for (const x of [scene.bounds.min[0], scene.bounds.max[0]]) for (const z of [scene.bounds.min[2], scene.bounds.max[2]]) points.push([x, 0, z]);
  }
  const horizontal = points.map(p => new THREE.Vector3(...p).dot(right));
  const vertical = points.map(p => new THREE.Vector3(...p).dot(up));
  const minX = Math.min(...horizontal), maxX = Math.max(...horizontal), minY = Math.min(...vertical), maxY = Math.max(...vertical);
  const halfHeight = Math.max((maxY - minY) / 2, (maxX - minX) / (2 * aspect)) * 1.2;
  if (!(halfHeight > 0)) throw new Error('Measured scene bounds cannot frame a top-down view.');
  const center = right.clone().multiplyScalar((minX + maxX) / 2).addScaledVector(up, (minY + maxY) / 2);
  const height = Math.max(scene.bounds.max[1] - scene.bounds.min[1], halfHeight * 2);
  const camera = new THREE.OrthographicCamera(-halfHeight * aspect, halfHeight * aspect, halfHeight, -halfHeight, 0.01, height * 8);
  camera.position.set(center.x, scene.bounds.max[1] + height * 2, center.z);
  camera.up.copy(up);
  camera.lookAt(center.x, 0, center.z);
  camera.updateMatrixWorld();
  return camera;
}

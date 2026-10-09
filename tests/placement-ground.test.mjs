import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {measuredPoint,placementFromPixels,placementPreview,createScenario} from '../src/geometry.mjs';
import {scene,input} from './scene-input.mjs';

function projectedPixel(point) {
  const c=scene.camera,camera=new THREE.PerspectiveCamera();
  camera.position.set(...c.eye);camera.up.set(...c.up).normalize();
  camera.lookAt(camera.position.clone().add(new THREE.Vector3(...c.forward).normalize()));
  camera.projectionMatrix.makePerspective(-c.cx*c.near/c.fx,(c.width-c.cx)*c.near/c.fx,c.cy*c.near/c.fy,-(c.height-c.cy)*c.near/c.fy,c.near,c.far);
  camera.updateMatrixWorld(true);
  const p=new THREE.Vector3(...point).project(camera);
  return [(p.x+1)*c.width/2,(1-p.y)*c.height/2];
}
function close(actual,expected) {
  assert.equal(actual.length,expected.length);
  actual.forEach((n,i)=>assert.ok(Math.abs(n-expected[i])<1e-8,`${n} differs from ${expected[i]}`));
}
test('admitted anchors retain measured height and recorded-image pixel registration',()=>{
  const placement=placementFromPixels(scene,input.pixels);
  for(const role of ['crossing','parking','target']) {
    const measured=measuredPoint(scene,input.pixels[role]);
    assert.notEqual(measured[1],0,'Actual regression input must expose vertical flattening');
    close(placement[role],measured);
    close(projectedPixel(placement[role]),input.pixels[role]);
  }
  assert.equal(placement.axis[1],0);
  close([Math.hypot(...placement.axis)],[1]);
});
test('vehicle front and target foot stay on their selected measurements through previews and movement',()=>{
  const scenario=createScenario(scene,input),preview=placementPreview(scene,input.pixels,input.vehicle,input.targetHeight);
  const parking=measuredPoint(scene,input.pixels.parking),ground=measuredPoint(scene,input.pixels.target);
  close(scenario.targetGround,ground);close(preview.targetGround,ground);
  close(scenario.target,ground.map((n,i)=>n+(i===1?input.targetHeight:0)));
  close(preview.target,scenario.target);
  const front=scenario.baseline.center.map((n,i)=>n-scenario.axis[i]*input.vehicle.length/2-(i===1?input.vehicle.height/2:0));
  close(front,parking);close(projectedPixel(front),input.pixels.parking);
  close(preview.vehicle.center,scenario.baseline.center);
  close([scenario.vehicleGroundHeight],[parking[1]]);
  const moved=createScenario(scene,{...input,proposedSetback:(scenario.baselineSetback+scenario.limits.maximumSetback)/2});
  close(moved.targetGround,ground);
  for(const key of ['baseline','proposed'])close([moved[key].center[1]-input.vehicle.height/2],[parking[1]]);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { measuredPoint, placementPreview, createScenario, segmentIntersectsBox, classifySightline } from '../src/geometry.mjs';
import { loadScene } from '../src/scene.mjs';
import { scene,input,fetchAsset } from './scene-input.mjs';
test('measured pixels round-trip through the fitted camera transform',()=>{
  const [u,v]=input.pixels.crossing,p=measuredPoint(scene,[u,v]),R=scene.transform.worldFromCamera;
  const camera=scene.transform.cameraOriginGround.map((o,i)=>o+R.reduce((s,row,j)=>s+row[i]*p[j],0));
  assert.ok(Math.abs(scene.camera.fx*camera[0]/camera[2]+scene.camera.cx-u)<1e-8);
  assert.ok(Math.abs(scene.camera.fy*camera[1]/camera[2]+scene.camera.cy-v)<1e-8);
});
test('zero depth is rejected rather than supplied by a nearby pixel',()=>{
  const i=scene.depth.findIndex(n=>n===0);
  assert.throws(()=>measuredPoint(scene,[i%scene.camera.width,Math.floor(i/scene.camera.width)]),/No measured depth/);
});
test('placement previews use declared measured actors without inventing incomplete inputs',()=>{
  assert.deepEqual(placementPreview(scene,{}, {width:'',height:'',length:''},''),{});
  const preview=placementPreview(scene,input.pixels,input.vehicle,input.targetHeight),scenario=createScenario(scene,input);
  assert.deepEqual(preview.vehicle.center,scenario.baseline.center);
  assert.deepEqual(preview.vehicle.axis,scenario.axis);
  assert.deepEqual(preview.target,scenario.target);
  assert.deepEqual(placementPreview(scene,input.pixels,{...input.vehicle,width:''},'').vehicle,undefined);
  assert.throws(()=>placementPreview(scene,input.pixels,{...input.vehicle,width:1e6},input.targetHeight),/scene extent/);
});
test('actual data forms a declared vehicle occlusion and a data-derived driver position',()=>{
  const r=createScenario(scene,input);
  assert.equal(r.currentVisibility.state,'blocked');assert.deepEqual(r.eye,scene.camera.eye);
  assert.equal(r.baselineSetback,input.proposedSetback);
});
test('browser serialization of derived range endpoints remains valid',()=>{
  const {limits}=createScenario(scene,input);
  for(const driverDistance of [Number(limits.minimumDriverDistance.toPrecision(15)),Number(limits.captureDistance.toPrecision(15))])assert.doesNotThrow(()=>createScenario(scene,{...input,driverDistance}));
  assert.doesNotThrow(()=>createScenario(scene,{...input,proposedSetback:Number(limits.maximumSetback.toPrecision(15))}));
});
test('missing, invalid and out-of-extent inputs never produce a default comparison',()=>{
  for(const bad of [null,{...input,proposedSetback:''},{...input,proposedSetback:1e6},{...input,driverDistance:1e6},{...input,vehicle:{...input.vehicle,width:NaN}}])assert.throws(()=>createScenario(scene,bad));
});
test('unsupported sightline outside the capture remains unknown',()=>{
  const r=classifySightline(scene,[0,2,1],[0,2,4],input.vehicle,[0,0,1],[100,2,100]);
  assert.equal(r.state,'unknown');assert.equal(r.supportedFraction,0);
});
test('depth coverage alone cannot establish Clear; missing samples reduce its diagnostic coverage',()=>{
  const [u,v]=input.pixels.crossing,c=scene.camera,depth=scene.depth[v*c.width+u]/scene.processing.depthDivisor;
  const toWorld=z=>{const p=[(u-c.cx)*z/c.fx,(v-c.cy)*z/c.fy,z].map((n,i)=>n-scene.transform.cameraOriginGround[i]);return scene.transform.worldFromCamera.map(row=>row.reduce((s,n,i)=>s+n*p[i],0));};
  const start=toWorld(depth*.1),end=toWorld(depth*.5),axis=[0,0,1],center=[100,2,100];
  const covered=classifySightline(scene,start,end,input.vehicle,axis,center);
  assert.equal(covered.state,'unknown');assert.equal(covered.supportedFraction,1);
  const missing=scene.depth.slice();missing[v*c.width+u]=0;
  const unsupported=classifySightline({...scene,depth:missing},start,end,input.vehicle,axis,center);
  assert.equal(unsupported.state,'unknown');assert.equal(unsupported.supportedFraction,0);
});
test('finite segment-box math includes grazing edges, excludes misses and rejects invalid geometry',()=>{
  assert.equal(segmentIntersectsBox([-2,1,0],[2,1,0],[-1,-1,-1],[1,1,1]),true);
  assert.equal(segmentIntersectsBox([-2,2,0],[2,2,0],[-1,-1,-1],[1,1,1]),false);
  assert.throws(()=>segmentIntersectsBox([0,0,0],[1,0,0],[1,0,0],[0,1,1]));
});
test('missing scene and altered buffer fail the loader',async()=>{
  await assert.rejects(loadScene(fetchAsset,'unavailable-scene'),/metadata unavailable/);
  await assert.rejects(loadScene(async path=>{
    const response=await fetchAsset(path);
    if(path.endsWith('depth.bin')){const b=new Uint8Array(await response.arrayBuffer());b[0]^=1;return new Response(b);}
    return response;
  },scene.id),/checksum mismatch/);
});

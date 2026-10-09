import assert from 'node:assert/strict';
import test from 'node:test';
import { loadScene } from '../src/scene.mjs';
import { scene,fetchAsset } from './scene-input.mjs';

function alteredMetadata(mutate,requests=[]) {
  return async path=>{
    requests.push(path);
    const response=await fetchAsset(path);
    if(path.endsWith('/scene.json')){
      const metadata=await response.json();
      mutate(metadata.transform.worldFromCamera);
      return new Response(JSON.stringify(metadata));
    }
    return response;
  };
}

test('actual measured scene loads with its original proper rotation',async()=>{
  const loaded=await loadScene(fetchAsset,scene.id);
  assert.deepEqual(loaded.transform.worldFromCamera,scene.transform.worldFromCamera);
  assert.equal(loaded.positions.length,scene.pointCount*3);
});

test('six-decimal serialized rotation loads without normalization',async()=>{
  const rounded=scene.transform.worldFromCamera.map(row=>row.map(n=>Number(n.toFixed(6))));
  const loaded=await loadScene(alteredMetadata(R=>R.splice(0,3,...rounded)),scene.id);
  assert.deepEqual(loaded.transform.worldFromCamera,rounded);
});

const malformedRotations={
  'doubled row':R=>{R[0]=R[0].map(n=>n*2);},
  'unit-length sheared row':R=>{
    const row=R[1].map((n,i)=>n+.1*R[0][i]),length=Math.hypot(...row);
    R[1]=row.map(n=>n/length);
  },
  'collinear rows':R=>{R[1]=R[0].slice();},
  'orthonormal reflection':R=>{R[2]=R[2].map(n=>-n);},
  'NaN entry':R=>{R[0][0]=NaN;},
  'infinite entry':R=>{R[0][0]=Infinity;},
};
for(const [name,mutate] of Object.entries(malformedRotations))test(`${name} is rejected before fetching geometry`,async()=>{
  const requests=[];
  await assert.rejects(loadScene(alteredMetadata(mutate,requests),scene.id),/Invalid measured calibration/);
  assert.deepEqual(requests,[`/scenes/${scene.id}/scene.json`]);
});

test('a proper rotation inconsistent with the unchanged renderer pose is rejected',async()=>{
  const requests=[];
  // World-axis rotation preserves all row norms and determinant while changing the actual camera pose.
  await assert.rejects(loadScene(alteredMetadata(R=>{
    const [x,y,z]=R.map(row=>row.slice());R.splice(0,3,z,y,x.map(n=>-n));
  },requests),scene.id),/camera pose disagrees/);
  assert.deepEqual(requests,[`/scenes/${scene.id}/scene.json`]);
});

test('a renderer-only camera direction change is rejected',async()=>{
  const requests=[];
  await assert.rejects(loadScene(async path=>{
    requests.push(path);const response=await fetchAsset(path);
    const metadata=await response.json();metadata.camera.forward=metadata.camera.forward.map(n=>-n);
    return new Response(JSON.stringify(metadata));
  },scene.id),/camera pose disagrees/);
  assert.deepEqual(requests,[`/scenes/${scene.id}/scene.json`]);
});

const malformedMetadata={
  'infinite depth divisor':s=>{s.processing.depthDivisor=Infinity;},
  'string depth divisor':s=>{s.processing.depthDivisor='256';},
  'infinite road tolerance':s=>{s.processing.roadToleranceMeters=Infinity;},
  'string road tolerance':s=>{s.processing.roadToleranceMeters='0.08';},
  'fractional image width':s=>{s.camera.width+=.5;},
  'inverted clipping planes':s=>{s.camera.near=s.camera.far;},
  'fractional point count':s=>{s.pointCount+=.5;},
  'negative triangle count':s=>{s.triangleCount=-1;},
  'infinite bounds':s=>{s.bounds.max[0]=Infinity;},
  'reversed bounds':s=>{s.bounds.min[0]=s.bounds.max[0]+1;},
  'zero extent':s=>{s.bounds.max=s.bounds.min.slice();},
};
for(const [name,mutate] of Object.entries(malformedMetadata))test(`${name} is rejected before geometry reads`,async()=>{
  const requests=[];
  await assert.rejects(loadScene(async path=>{
    requests.push(path);const response=await fetchAsset(path),metadata=await response.json();
    mutate(metadata);return new Response(JSON.stringify(metadata));
  },scene.id),/Invalid measured (calibration|geometry metadata)/);
  assert.deepEqual(requests,[`/scenes/${scene.id}/scene.json`]);
});

import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { loadScene,renderGeometry } from '../src/scene.mjs';
import { measuredPoint,placementFromPixels,createScenario } from '../src/geometry.mjs';
import { scene,fetchAsset,input } from './scene-input.mjs';

const sha=data=>createHash('sha256').update(Buffer.from(data)).digest('hex');
// Isolated analytic renderer fixture, never a model result or runtime asset.
const arrays={positions:new Float32Array([-.1,.2,-10,.1,.2,-10,0,.4,-10]),
  colors:new Float32Array([.1,.2,.3,.1,.2,.3,.1,.2,.3]),triangles:new Uint32Array([0,1,2])};
const pin='a'.repeat(64);
function descriptor() {
  return {evidence:'predicted',role:'rendererOnly',model:{provider:'nvidia',id:'analytic-test-only',revision:'fixture',sha256:pin},
    provenance:Object.fromEntries(['registeredPredictionSha256','producerManifestSha256','planSha256','observationsSha256','measuredSceneSha256'].map(k=>[k,pin])),
    pointCount:3,triangleCount:1,bounds:{min:[-.10000001,.20000000,-10],max:[.10000001,.40000001,-10]},
    displayDomain:{maxDepthMeters:scene.processing.maxDepthMeters,meshEdgeLimitMeters:scene.processing.meshEdgeLimitMeters},
    buffers:{positions:'estimatedpositions.bin',colors:'estimatedcolors.bin',triangles:'estimatedtriangles.bin'},
    artifactHashes:Object.fromEntries(Object.entries(arrays).map(([k,a])=>[k,sha(a.buffer)]))};
}
async function source({mutate=()=>{},mutateBuffer=()=>{},missing=null,tamper=null}={}) {
  const geometry=descriptor();mutate(geometry);
  const payloads=Object.fromEntries(Object.entries(arrays).map(([k,a])=>[k,a.slice().buffer]));
  mutateBuffer(payloads,geometry);
  return async path=>{
    if(path.endsWith('/scene.json')) {
      const response=await fetchAsset(path),metadata=await response.json();metadata.reconstruction=geometry;
      return new Response(JSON.stringify(metadata));
    }
    for(const key of Object.keys(payloads))if(path.endsWith('/'+descriptor().buffers[key])) {
      if(missing===key)return new Response(null,{status:404});
      const bytes=payloads[key];if(tamper===key)new Uint8Array(bytes)[0]^=1;
      return new Response(bytes);
    }
    return fetchAsset(path);
  };
}

test('estimated display retains exact measured buffers, placements, extent and classifications',async()=>{
  const loaded=await loadScene(await source(),scene.id);
  for(const key of ['depth','positions','colors','triangles'])assert.deepEqual(loaded[key],scene[key]);
  assert.deepEqual(loaded.bounds,scene.bounds);
  assert.notDeepEqual(renderGeometry(loaded).positions,loaded.positions);
  for(const pixel of Object.values(input.pixels))assert.deepEqual(measuredPoint(loaded,pixel),measuredPoint(scene,pixel));
  assert.deepEqual(placementFromPixels(loaded,input.pixels),placementFromPixels(scene,input.pixels));
  const base=createScenario(scene,input);
  for(const setback of [base.baselineSetback,base.limits.maximumSetback/2]) {
    const options={...input,proposedSetback:setback};
    assert.deepEqual(createScenario(loaded,options),createScenario(scene,options));
  }
});

for(const key of Object.keys(arrays)) {
  test(`declared estimate missing ${key} fails rather than substituting measured data`,async()=>{
    await assert.rejects(loadScene(await source({missing:key}),scene.id),/unavailable/);
  });
  test(`tampered estimated ${key} fails checksum`,async()=>{
    await assert.rejects(loadScene(await source({tamper:key}),scene.id),/checksum mismatch/);
  });
}
for(const [label,mutate] of Object.entries({
  'wrong evidence':g=>g.evidence='measured','query role':g=>g.role='visibility',
  'missing producer pin':g=>delete g.provenance.producerManifestSha256,
  'bad model pin':g=>g.model.sha256='wrong','negative points':g=>g.pointCount=-1,
  'reversed bounds':g=>g.bounds.min[0]=2,'range mismatch':g=>g.displayDomain.maxDepthMeters=500,
  'measured buffer alias':g=>g.buffers.positions=scene.buffers.positions,
  'duplicate estimated buffers':g=>g.buffers.colors=g.buffers.positions,
}))test(`invalid estimate ${label} rejects`,async()=>{
  await assert.rejects(loadScene(await source({mutate}),scene.id),/estimated reconstruction|separate measured buffers/i);
});

for(const [label,key,alter] of [
  ['nonfinite points','positions',a=>new Float32Array(a)[0]=Infinity],
  ['invalid color','colors',a=>new Float32Array(a)[0]=2],
  ['invalid index','triangles',a=>new Uint32Array(a)[0]=4],
  ['point outside declared bounds','positions',a=>new Float32Array(a)[0]=-5],
])test(`repinned estimated ${label} still rejects`,async()=>{
  const fetcher=await source({mutateBuffer:(p,g)=>{alter(p[key]);g.artifactHashes[key]=sha(p[key]);}});
  await assert.rejects(loadScene(fetcher,scene.id),/Invalid estimated reconstruction buffers/);
});

test('explicit null reconstruction fails loading',async()=>{
  const fetcher=async path=>{
    const response=await fetchAsset(path);
    if(!path.endsWith('/scene.json'))return response;
    const metadata=await response.json();metadata.reconstruction=null;return new Response(JSON.stringify(metadata));
  };
  await assert.rejects(loadScene(fetcher,scene.id),/Invalid estimated reconstruction metadata/);
});

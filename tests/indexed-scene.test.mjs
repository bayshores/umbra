import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {loadScene,renderGeometry} from '../src/scene.mjs';
import {measuredPoint,roadPoint,placementPreview,placementFromPixels,createScenario,classifySightline} from '../src/geometry.mjs';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
function fixture(mutate=()=>{},prediction=false) {
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAgAAAAGCAIAAABxZ0isAAAAD0lEQVR4nGNowAEYBlICABnRSAEtC4j6AAAAAElFTkSuQmCC','base64'); // Analytic8×6 RGB image only.
  const xyz=[[0,0,10],[.2,0,10],[.201,0,10],[.202,0,10],[-.3,0,10],[0,0,-10],[0,0,0],[NaN,NaN,NaN],[.000001,0,.0001]],
    uv=xyz.map(([x,y,z])=>z>0?[100*x/z+2,100*y/z+2]:[NaN,NaN]),
    eligible=uv.map(([u,v])=>Number(u>=0&&u<=7&&v>=0&&v<=5)),
    world=xyz.map(([x,y,z])=>[x,-y,-z]),draw=world.filter((_,i)=>eligible[i]),
    arrays={positions:new Float32Array(draw.flat()),colors:new Float32Array(draw.flat().map(()=>.5)),triangles:new Uint32Array(),
      sourceIndices:new Uint32Array(xyz.map((_,i)=>i)),referencePositions:new Float64Array(world.flat()),referenceCameraPositions:new Float64Array(xyz.flat()),
      referenceUV:new Float64Array(uv.flat()),referenceEligible:new Uint8Array(eligible)},
    scene={id:'indexed-fixture',units:'meters',pointCount:draw.length,triangleCount:0,
      camera:{eye:[0,0,0],forward:[0,0,-1],up:[0,1,0],fx:100,fy:100,cx:2,cy:2,width:8,height:6,near:.01,far:40},
      transform:{worldFromCamera:[[1,0,0],[0,-1,0],[0,0,-1]],worldFromCity:[[1,0,0],[0,0,1],[0,-1,0]],cityOrigin:[101,201,10],cameraOrigin:[0,0,0],originRole:'leftCamera'},
      bounds:{min:draw[0].map((_,k)=>Math.min(...draw.map(p=>Math.fround(p[k])))),max:draw[0].map((_,k)=>Math.max(...draw.map(p=>Math.fround(p[k]))))},processing:{},buffers:{},artifactHashes:{},
      measurements:{kind:'indexedSparse',evidence:'geometricallyProjected',groundPolicy:'unvalidated',binaryEncoding:'littleEndian',pixelCellPolicy:'nearestIntegerCenterRejectCollisions',count:xyz.length,
        provenance:{preparedManifestSha256:'a'.repeat(64),referencesSha256:'b'.repeat(64),logId:'81d2b40a-c579-3e9c-b520-bee26cda947d',cameraTimestampNs:'123',lidarTimestampNs:'124',crossingId:'5'},buffers:{},artifactHashes:{}}};
  scene.recordedImage={file:'recorded.png',mimeType:'image/png',bytes:png.length,sha256:hash(png)};
  if(prediction) { // Analytic renderer-only contract fixture, never a sponsor execution artifact.
    const model={provider:'nvidia',id:'analytic-test-only',revision:'analytic',sha256:'c'.repeat(64),bytes:1};
    scene.reconstruction={kind:'nativePoints',evidence:'predicted',role:'rendererOnly',model,pointCount:3,triangleCount:0,
      bounds:{min:[-1,0,-10],max:[1,1,-8]},buffers:{},artifactHashes:{},
      provenance:Object.fromEntries(['baseMeasuredSceneSha256','preparedManifestSha256','calibrationSha256','referencesSha256','inputEquivalenceResultSha256','producerManifestSha256','planSha256','rawPredictionSha256','rawValiditySha256'].map(k=>[k,k==='preparedManifestSha256'?'a'.repeat(64):'b'.repeat(64)])),
      reuse:{newInferenceRuns:0,originalCacheKey:'d'.repeat(64),originalRecordBase:'analytic-test-only',originalExecution:{status:'succeeded',provider:'local_onnxruntime_cpu',modelArtifact:{sha256:model.sha256,bytes:1},createdUtc:'2026-10-05T11:13:09Z',elapsedSeconds:1,usage:{cloudRequests:0,jobs:1,retries:0,threads:1}}},
      displayDomain:{policy:'numericDepth_rightInputBounds_existingCameraClips',nearMeters:.01,farMeters:40,rightPixelCenters:[0,7]}};
    arrays.estimatedPositions=new Float32Array([-1,0,-10,1,0,-10,0,1,-8]);arrays.estimatedColors=new Float32Array(9).fill(.5);arrays.estimatedTriangles=new Uint32Array();
  }
  mutate(scene,arrays);
  const bytes=new Map([['recorded.png',png]]);
  for(const [key,array] of Object.entries(arrays)) {
    const estimated=key.startsWith('estimated'),logicalKey=estimated?key[9].toLowerCase()+key.slice(10):key,
      target=estimated?scene.reconstruction:key.startsWith('reference')||key==='sourceIndices'?scene.measurements:scene,
      name=key.toLowerCase()+'.bin',buffer=Buffer.from(array.buffer,array.byteOffset,array.byteLength);
    target.buffers[logicalKey]=name;target.artifactHashes[logicalKey]=hash(buffer);bytes.set(name,buffer);
  }
  return {scene,bytes,fetch:async path=>{
    const name=path.split('/').at(-1);
    return name==='scene.json'?new Response(JSON.stringify(scene)):bytes.has(name)?new Response(bytes.get(name)):new Response('',{status:404});
  }};
}
const metadataHash=f=>hash(JSON.stringify(f.scene));
const load=f=>loadScene(f.fetch,f.scene.id,metadataHash(f));

test('indexed source rows retain exact XYZ, unavailable rows and collision ownership',async()=>{
  const s=await load(fixture());
  assert.equal(s.measurements.count,9);assert.equal(s.pointCount,5);assert.equal(s.triangleCount,0);
  assert.equal(s.depth,undefined);assert.equal(s.transform.cameraOriginGround,undefined);
  assert.deepEqual(Buffer.from(await s.recordedImageBlob.arrayBuffer()),fixture().bytes.get('recorded.png'));
  assert.deepEqual(measuredPoint(s,[2,2]),[0,-0,-10]); // Row zero is not a missing sentinel.
  assert.deepEqual(measuredPoint(s,[3,2]),[.000001,-0,-.0001]); // Render near clip is not eligibility.
  assert.throws(()=>measuredPoint(s,[4,2]),/Multiple/); // Three returns retain collision sentinel.
  assert.throws(()=>measuredPoint(s,[0,0]),/No projected/);
  assert.ok(Number.isNaN(s.measurements.referenceUV[10]));
  assert.equal(s.measurements.referenceUV[8],-1); // Positive outside-crop UV retained.
  assert.ok(Number.isNaN(s.measurements.referencePositions[21]));
  for(const call of [()=>roadPoint(s,[2,2],'person'),()=>placementFromPixels(s,{crossing:[2,2],parking:[3,2],target:[2,2]}),
    ()=>placementPreview(s,{target:[2,2]},null,'1.3'),()=>createScenario(s,{pixels:{crossing:[2,2],parking:[3,2],target:[2,2]}})])
    assert.throws(call,/Ground placement has no validated support/);
  assert.deepEqual(placementPreview(s,{},null,''),{});
  const vehicle={width:2,height:2,length:2},axis=[0,0,1];
  assert.equal(classifySightline(s,[-2,0,0],[2,0,0],vehicle,axis,[0,0,0]).state,'blocked');
  assert.deepEqual(classifySightline(s,[-2,0,0],[2,0,0],vehicle,axis,[0,0,10]),{state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null});
});

test('native metadata is pinned by its caller before exposure, independently of its declared buffer hashes',async()=>{
  const f=fixture(),pin=metadataHash(f);
  await assert.rejects(loadScene(f.fetch,f.scene.id),/caller-pinned/);
  await assert.rejects(loadScene(f.fetch,f.scene.id,'bad'),/expected scene metadata/);
  f.scene.camera.cx+=.5;
  await assert.rejects(loadScene(f.fetch,f.scene.id,pin),/metadata checksum mismatch/);
});

test('native recorded image rejects invalid ownership, path, type and resource declarations',async()=>{
  for(const mutate of [s=>{delete s.recordedImage;},s=>{s.recordedImage.file='../recorded.png';},s=>{s.recordedImage.file='recorded.jpg';},
    s=>{s.recordedImage.mimeType='image/jpeg';},s=>{s.recordedImage.bytes=0;},s=>{s.recordedImage.bytes=16*1024*1024+1;},s=>{s.recordedImage.sha256='x'.repeat(64);}])
    await assert.rejects(load(fixture(mutate)),/recorded image descriptor/);
});

test('native image bytes cannot drift, truncate, overrun or hide different decoded dimensions',async()=>{
  let f=fixture();f.bytes.delete('recorded.png');await assert.rejects(load(f),/image unavailable/);
  f=fixture();f.bytes.get('recorded.png')[40]^=1;await assert.rejects(load(f),/image checksum mismatch/);
  f=fixture();f.scene.recordedImage.bytes--;await assert.rejects(load(f),/exceeds declared/);
  f=fixture();f.scene.recordedImage.bytes++;await assert.rejects(load(f),/byte length mismatch/);
  for(const offset of [0,12,16,20]) {
    f=fixture();const bytes=f.bytes.get('recorded.png');bytes[offset]^=1;f.scene.recordedImage.sha256=hash(bytes);
    await assert.rejects(load(f),/PNG dimensions/);
  }
});

test('native predicted points have their own exact lengths and never own measurement queries or ground',async()=>{
  const measured=await load(fixture()),predicted=await load(fixture(()=>{},true));
  assert.equal(renderGeometry(predicted),predicted.reconstruction);
  assert.equal(predicted.pointCount,5);assert.equal(predicted.reconstruction.pointCount,3);
  assert.equal(predicted.reconstruction.positions.length,9);assert.equal(predicted.reconstruction.triangles.length,0);
  for(const key of ['positions','colors','triangles','pixelRows'])assert.deepEqual(predicted[key],measured[key]);
  for(const key of ['sourceIndices','referencePositions','referenceCameraPositions','referenceUV','referenceEligible'])assert.deepEqual(predicted.measurements[key],measured.measurements[key]);
  assert.deepEqual(measuredPoint(predicted,[2,2]),measuredPoint(measured,[2,2]));
  assert.throws(()=>roadPoint(predicted,[2,2],'person'),/no validated support/);
  for(const scene of [measured,predicted])assert.equal(classifySightline(scene,[-2,0,0],[2,0,0],{width:2,height:2,length:2},[0,0,1],[0,0,10]).state,'unknown');
});

test('native prediction records bounded CPU configuration without changing measured queries',async()=>{
  const original=await load(fixture(()=>{},true));
  for(const threads of [1,2,4]) {
    const f=fixture(scene=>{
      const execution=scene.reconstruction.reuse.originalExecution;
      execution.usage.threads=threads;
      execution.cpuConfiguration={intraOpThreads:threads,interOpThreads:1,provider:'CPUExecutionProvider'};
    },true);
    const loaded=await load(f);
    assert.equal(loaded.measurements.groundPolicy,'unvalidated');
    assert.deepEqual(loaded.measurements.referencePositions,original.measurements.referencePositions);
    assert.deepEqual(loaded.measurements.referenceUV,original.measurements.referenceUV);
    assert.equal(loaded.reconstruction.reuse.originalExecution.usage.threads,threads);
  }
});

test('native prediction rejects unsupported or incomplete CPU configuration',async()=>{
  const config={intraOpThreads:4,interOpThreads:1,provider:'CPUExecutionProvider'};
  for(const [threads,cpu] of [[2,undefined],[4,undefined],[0,undefined],[3,undefined],[8,undefined],[true,undefined],
    [4,null],[4,{...config,intraOpThreads:2}],[4,{...config,intraOpThreads:true}],
    [4,{...config,interOpThreads:2}],[4,{...config,interOpThreads:true}],
    [4,{...config,provider:'CUDAExecutionProvider'}],[4,{...config,extra:true}],[1,{...config,intraOpThreads:4}]]) {
    const f=fixture(scene=>{
      const execution=scene.reconstruction.reuse.originalExecution;
      execution.usage.threads=threads;
      if(cpu!==undefined)execution.cpuConfiguration=cpu;
    },true);
    await assert.rejects(load(f),/Invalid native reconstruction provenance/);
  }
});

const predictionCorruptions={
  'wrong native kind':e=>{e.kind='mesh';},'query ownership':e=>{e.role='query';},
  'new inference attribution':e=>{e.reuse.newInferenceRuns=1;},'cache identity missing':e=>{delete e.reuse.originalCacheKey;},
  'record base missing':e=>{delete e.reuse.originalRecordBase;},'changed model identity':e=>{e.reuse.originalExecution.modelArtifact.sha256='f'.repeat(64);},
  'invented execution':e=>{e.reuse.originalExecution.status='mock';},'invalid time':e=>{e.reuse.originalExecution.createdUtc='invalid';},
  'additional retries':e=>{e.reuse.originalExecution.usage.retries=1;},'cloud attribution':e=>{e.reuse.originalExecution.usage.cloudRequests=1;},
  'native reference relabeling':e=>{e.provenance.referencesSha256='e'.repeat(64);},'missing producer pin':e=>{delete e.provenance.producerManifestSha256;},
  'new clipping threshold':e=>{e.displayDomain.farMeters=30;},'invalid right domain':e=>{e.displayDomain.rightPixelCenters=[0,8];},
  'oversized point allocation':e=>{e.pointCount=1_000_001;},'invented continuity':e=>{e.triangleCount=1;},
  'unexpected buffer ownership':e=>{e.buffers.pixels='pixels.bin';},'measured reference alias':e=>{e.buffers.colors='referenceuv.bin';},
};
for(const [name,mutate] of Object.entries(predictionCorruptions))test('native prediction rejects '+name,async()=>{
  const f=fixture(()=>{},true);mutate(f.scene.reconstruction);await assert.rejects(load(f),/Invalid|separate measured/);
});

test('declared native prediction rejects missing, overflowing and truncated buffers instead of falling back',async()=>{
  for(const kind of ['missing','overflow','truncated','checksum']) {
    const f=fixture(()=>{},true),name=f.scene.reconstruction.buffers.positions;
    if(kind==='missing')f.bytes.delete(name);
    if(kind==='overflow')f.bytes.set(name,Buffer.concat([f.bytes.get(name),Buffer.from([0])]));
    if(kind==='truncated')f.bytes.set(name,f.bytes.get(name).subarray(0,32));
    if(kind==='checksum')f.bytes.get(name)[0]^=1;
    await assert.rejects(load(f),/unavailable|exceeds declared|byte length mismatch|checksum mismatch/);
  }
});

const corruptions={
  'wrong encoding':s=>{s.measurements.binaryEncoding='bigEndian';},
  'ground promotion':s=>{s.measurements.groundPolicy='road';},
  'dense depth fallback':s=>{s.buffers.depth='depth.bin';},
  'fabricated road tolerance':s=>{s.processing.roadToleranceMeters=.3;},
  'wrong city orientation':s=>{s.transform.worldFromCity[2][1]=1;},
  'incorrect camera origin':s=>{s.transform.cameraOrigin[0]=1;},
  'oversized references':s=>{s.measurements.count=1_000_001;},
  'oversized pixel index':s=>{s.camera.width=16_000_001;},
  'missing provenance':s=>{delete s.measurements.provenance;},
  'missing source timestamp':s=>{delete s.measurements.provenance.cameraTimestampNs;},
  'malformed log identity':s=>{s.measurements.provenance.logId='-'.repeat(36);},
  'numeric source timestamp':s=>{s.measurements.provenance.cameraTimestampNs=123;},
  'malformed native prediction attachment':s=>{s.reconstruction={};},
  'quotient overflow':(_,a)=>{a.referenceCameraPositions.set([1,0,1e-320],0);a.referencePositions.set([1,0,-1e-320],0);},
  'row reorder':(_,a)=>{a.sourceIndices[0]=1;},
  'nonbinary eligibility':(_,a)=>{a.referenceEligible[0]=2;},
  'behind-camera eligibility':(_,a)=>{a.referenceEligible[5]=1;},
  'unavailable UV fabricated':(_,a)=>{a.referenceUV[10]=2;},
  'positive outside UV hidden':(_,a)=>{a.referenceUV[8]=NaN;},
  'UV shift with refreshed hash':(_,a)=>{a.referenceUV[0]+=.01;},
  'camera/world identity mismatch':(_,a)=>{a.referenceCameraPositions[0]=.1;},
  'rendered row mismatch':(_,a)=>{a.positions[0]=.1;},
  'bounds exclude real points':s=>{s.bounds.min[2]=-5;},
  'source infinity':(_,a)=>{a.referencePositions[0]=Infinity;},
  'NaN row replaced by zero':(_,a)=>{a.referencePositions.fill(0,21,24);},
};
for(const [name,mutate] of Object.entries(corruptions))test(`native rejects ${name}`,async()=>{
  await assert.rejects(load(fixture(mutate)),name==='malformed native prediction attachment'?/Invalid estimated reconstruction metadata/:/Invalid indexed|Invalid measured|Indexed|Unavailable|Rendered/);
});
test('native streaming rejects byte overflow before constructing a typed buffer',async()=>{
  const f=fixture();f.bytes.set('referenceuv.bin',Buffer.alloc(f.bytes.get('referenceuv.bin').length+1));
  await assert.rejects(load(f),/exceeds declared byte length/);
});
test('finite invented UV cannot pass a quotient overflow with an infinite tolerance',async()=>{
  const f=fixture((s,a)=>{
    a.referenceCameraPositions.set([1,0,1e-320],0);a.referencePositions.set([1,0,-1e-320],0);a.positions.set([1,0,-0],0);
    s.bounds.min[0]=Math.fround(.000001);s.bounds.max[0]=1;s.bounds.max[2]=0;
  });
  await assert.rejects(load(f),/calibrated projection/);
});
test('scene metadata is bounded before JSON parsing and asset requests',async()=>{
  const f=fixture();let requests=0;
  await assert.rejects(loadScene(async()=>{requests++;return new Response(' '.repeat(128*1024+1));},f.scene.id),/exceeds declared byte length/);
  assert.equal(requests,1);
});
test('native rejects artifact aliases and checksum corruption',async()=>{
  const alias=fixture();alias.scene.measurements.buffers.referenceUV=alias.scene.buffers.positions;
  await assert.rejects(load(alias),/artifact ownership/);
  const corrupt=fixture();corrupt.bytes.get('positions.bin')[0]^=1;
  await assert.rejects(load(corrupt),/checksum mismatch/);
});

// Analytic contract mutations only; these are not sponsor execution evidence.
test('direct native binding is explicit and exclusive with historical equivalence',async()=>{
  const direct=scene=>{
    const p=scene.reconstruction.provenance;
    delete p.inputEquivalenceResultSha256;
    p.inputBindingKind='directNative';p.inputBindingResultSha256='e'.repeat(64);
  };
  const loaded=await load(fixture(direct,true));
  assert.equal(loaded.reconstruction.provenance.inputBindingKind,'directNative');
  assert.equal(loaded.measurements.groundPolicy,'unvalidated');
  for(const mutate of [p=>delete p.inputBindingResultSha256,p=>{p.inputBindingKind='unknown';},
    p=>{p.inputBindingResultSha256='e'.repeat(63);},p=>{p.inputEquivalenceResultSha256='a'.repeat(64);}]){
    await assert.rejects(load(fixture(scene=>{direct(scene);mutate(scene.reconstruction.provenance);},true)),/input binding/);
  }
  await assert.rejects(load(fixture(scene=>{scene.reconstruction.provenance.inputBindingResultSha256='e'.repeat(64);},true)),/input binding/);
  await assert.rejects(load(fixture(scene=>{delete scene.reconstruction.provenance.inputEquivalenceResultSha256;},true)),/input binding/);
});

#!/usr/bin/env node
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, readdir, writeFile, access} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import os from 'node:os';
import {performance} from 'node:perf_hooks';
import {loadScene} from '../src/scene.mjs';
import {classifySightline} from '../src/geometry.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const output='.local/evaluation/scene-scaling';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const identity=async path=>{const b=await readFile(`${root}/${path}`);return {bytes:b.length,sha256:hash(b)};};
const stats=raw=>{const s=[...raw].sort((a,b)=>a-b);return {count:s.length,p50:s[Math.ceil(s.length*.5)-1],p95:s[Math.ceil(s.length*.95)-1],min:s[0],max:s.at(-1)};};
const mem=()=>process.memoryUsage();
const sourcePaths=['scripts/benchmark-scene-scaling.mjs','scripts/prepare-scene.py','scripts/rectified_calibration.py','src/scene.mjs','src/geometry.mjs'];
const sources=async()=>Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,await identity(p)])));
const environment={node:process.version,platform:process.platform,architecture:process.arch,osRelease:os.release(),cpuModel:os.cpus()[0]?.model,logicalCpuCount:os.cpus().length,systemMemoryBytes:os.totalmem()};

if(process.argv[2]==='--child') {
  const id=process.argv[3],dir=`${root}/${output}/prepared/${id}`;
  assert.ok(global.gc,'Run child with --expose-gc');
  const fetchLocal=async url=>{
    assert.ok(url.startsWith(`/scenes/${id}/`));
    const name=url.slice(`/scenes/${id}/`.length);assert.match(name,/^[a-z]+\.(json|bin)$/);
    const b=await readFile(`${dir}/${name}`);
    return {ok:true,status:200,json:async()=>JSON.parse(b.toString()),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};
  };
  global.gc();const beforeLoad=mem();
  let t=performance.now(),scene=await loadScene(fetchLocal,id);
  const firstLoadMilliseconds=performance.now()-t,afterFirstLoad=mem();
  global.gc();const afterFirstLoadForcedGC=mem();
  const loadMilliseconds=[];
  for(let i=0;i<13;i++) {
    // Repeated same-ID Node loads retain the previous scene until resolution; browser lifecycle is unmeasured.
    t=performance.now();scene=await loadScene(fetchLocal,id);const elapsed=performance.now()-t;
    if(i>=3)loadMilliseconds.push(elapsed);
  }
  global.gc();const beforeQuery=mem();
  const start=scene.bounds.max.map(n=>n+100),end=start.map((n,i)=>n+(i===0?1:0));
  const center=start.map(n=>n+1000),vehicle={width:1,height:1,length:1},axis=[0,0,1];
  const expected={state:'unknown',reason:'Part of the sightline has no supported free-space evidence.',supportedFraction:0};
  const queryMilliseconds=[];
  for(let i=0;i<60;i++) {
    t=performance.now();const result=classifySightline(scene,start,end,vehicle,axis,center);const elapsed=performance.now()-t;
    assert.deepEqual(result,expected);if(i>=10)queryMilliseconds.push(elapsed);
  }
  const afterQuery=mem();global.gc();const afterQueryForcedGC=mem();
  // Hashing follows all timing/memory checkpoints; peak RSS still includes hashing and file reads.
  const assets=Object.fromEntries(await Promise.all((await readdir(dir)).sort().map(async name=>[name,await identity(`${output}/prepared/${id}/${name}`)])));
  console.log(JSON.stringify({id,status:'completed',pointCount:scene.pointCount,triangleCount:scene.triangleCount,depthPixelCount:scene.depth.length,supportedDepthPixels:scene.depth.reduce((n,d)=>n+Number(d>0),0),camera:{width:scene.camera.width,height:scene.camera.height},bounds:scene.bounds,preparedAssetBytes:Object.values(assets).reduce((n,a)=>n+a.bytes,0),geometryBufferBytes:scene.positions.byteLength+scene.colors.byteLength+scene.triangles.byteLength+scene.depth.byteLength,assets,sourceHashesDeclaredByScene:scene.sourceHashes,processingCodeHashesDeclaredByScene:scene.processingCodeHashes,firstLoadMilliseconds,load:{rawMilliseconds:loadMilliseconds,...stats(loadMilliseconds)},query:{input:{start,end,vehicle,axis,center},expectedOutput:expected,verifiedOutputs:60,rawMilliseconds:queryMilliseconds,...stats(queryMilliseconds)},memory:{beforeLoad,afterFirstLoad,afterFirstLoadForcedGC,beforeQuery,afterQuery,afterQueryForcedGC,peakResidentSetBytes:process.resourceUsage().maxRSS*1024}}));
} else {
  const begun=performance.now(),deadline=begun+300000;
  const baselinePath='.local/evaluation/stereo-baseline/five-frame/report.json';
  const baseline=JSON.parse(await readFile(`${root}/${baselinePath}`));
  const frames=baseline.frames.map(f=>f.stem);assert.equal(frames.length,5);assert.equal(new Set(frames).size,5);
  const sourceFiles=await sources();
  let priorPreparationReport=null;
  try {priorPreparationReport=JSON.parse(await readFile(`${root}/${output}/initial-report.json`));}catch(e){if(e.code!=='ENOENT')throw e;}
  if(priorPreparationReport)for(const p of ['scripts/prepare-scene.py','scripts/rectified_calibration.py'])assert.deepEqual(priorPreparationReport.sourceFiles[p],sourceFiles[p],'Preparation changed; prior preparation cannot be reused.');
  const protocol={schemaVersion:1,scope:'Five existing genuine DrivingStereo CPU-baseline development frames; measured size variation, not duplicated/scaled geometry or independent physical sites.',frames,selection:baseline.selection,sourceFiles,baselineReport:await identity(baselinePath),priorPreparationEvidence:priorPreparationReport?await identity(`${output}/initial-report.json`):null,preparation:{script:'scripts/prepare-scene.py',dataRoot:'.local/data/drivingstereo',roadTolerance:0.08,maxDepth:80,meshEdgeLimit:0.4,policy:'Unchanged preparer/helper and gates; preserve originals/public; rejection retained; existing task-prepared scenes and source-matching recorded rejections reused on replay; no rejection threshold relaxed.'},load:{first:1,warmup:3,samples:10,timing:'performance.now around real loadScene including local file reads, JSON parsing, buffer copies, SHA-256 and validation. No image decoding/rendering; OS file cache state uncontrolled.'},query:{warmup:10,samples:50,timing:'performance.now around production classifySightline only; output assertion after timer.',workload:'Full triangle-scan Unknown path: segment starts bounds.max + [100,100,100] and ends one meter farther in x; declared unit box is +1000 on each axis. No actor scenario or calibration invented.'},quantiles:'Nearest rank ceil(n*p)-1, milliseconds; timer overhead included.',order:'Baseline listed frame order, sequential preparation and one fresh child Node per accepted scene. Child performs first load, three warmups and ten measured same-ID Node loader calls, then ten warmup and fifty measured queries.',garbageCollection:'Fresh child with --expose-gc; forced GC before first load, after first load, after repeated loads and after queries; natural GC within timing loops. The previous loaded scene stays live until each same-ID loader call resolves; this is a Node retention pattern, not the browser app lifecycle.',limits:{childTimeoutMilliseconds:30000,preparationTimeoutMilliseconds:30000,totalElapsedMilliseconds:300000},memory:'process.memoryUsage checkpoints; absolute process bytes, not isolated retained geometry. Loader reads/copies and checksum work included. Parent raw-source/hash reads do not enter child memory but warm OS cache. Child asset hashes run after checkpoint/timing; child peak RSS includes hashes, all phases and Node. No browser/GPU/drag/model measurement.',failurePolicy:'Any nonzero, signaled or timed-out benchmark child makes report status failed and parent exit nonzero. Preparation gate rejections remain distinct.',interpretation:'No accuracy, release threshold, causal complexity fit, browser lifecycle or multi-location generalization claim.'};
  await mkdir(`${root}/${output}`,{recursive:true});
  const protocolPath=`${root}/${output}/protocol.json`;
  try {assert.deepEqual(JSON.parse(await readFile(protocolPath)),protocol,'Protocol/source changed; preserve this evidence and choose a new owned directory before rerun.');}
  catch(e){if(e.code==='ENOENT')await writeFile(protocolPath,JSON.stringify(protocol,null,2)+'\n');else throw e;}
  const report={schemaVersion:1,status:'failed',generatedAt:new Date().toISOString(),command:'node scripts/benchmark-scene-scaling.mjs',environment,protocol:await identity(`${output}/protocol.json`),sourceFiles,frames:[],errors:[],limitations:['Five correlated frames from one sunny non-California sequence; not held out or multiple locations.','Scene counts vary naturally; fixed query path covers a full triangle scan, not all user workload paths.','First-load measurement is fresh process, not cold disk/cache or fresh hardware. Preparation has one sample per scene, not preprocessing p50/p95.','Node CPU load/query and process memory only; browser/drag/GPU/inference remain unmeasured.']};
  try {
    for(const id of frames) {
      if(performance.now()>deadline)throw new Error('Five-minute elapsed budget exceeded');
      const sequence=id.split('_')[0];
      const paths={left:`.local/data/drivingstereo/extracted/left-image-half-size/${id}.jpg`,right:`.local/data/drivingstereo/extracted/right-image-half-size/${id}.jpg`,depth:`.local/data/drivingstereo/extracted/depth-map-half-size/${id}.png`,calibration:`.local/data/drivingstereo/calibration/${sequence}-full-reference.txt`,license:'.local/data/drivingstereo/LICENSE'};
      const rawSourceFiles=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([k,p])=>[k,{path:p,...await identity(p)}])));
      const entry={id,rawSourceFiles,preparation:null};report.frames.push(entry);
      const prior=priorPreparationReport?.frames.find(f=>f.id===id);
      if(prior)assert.deepEqual(prior.rawSourceFiles,rawSourceFiles,'Raw sources changed since preparation.');
      if(prior?.preparation.status==='rejected'){entry.preparation={...prior.preparation,reusedPriorRejection:true};continue;}
      let exists=true;try{await access(`${root}/${output}/prepared/${id}/scene.json`);}catch{exists=false;}
      if(!exists) {
        const args=['scripts/prepare-scene.py','--data-root','.local/data/drivingstereo','--frame',id,'--output',`${output}/prepared`,'--road-tolerance','0.08','--max-depth','80','--mesh-edge-limit','0.4'];
        const before=performance.now(),p=spawnSync('python3',args,{cwd:root,encoding:'utf8',timeout:Math.min(30000,deadline-performance.now()),maxBuffer:2*1024*1024});
        entry.preparation={command:['python3',...args],milliseconds:performance.now()-before,status:p.status===0?'accepted':'rejected',exitStatus:p.status,stdout:p.stdout,stderr:p.stderr,error:p.error?.message};
        if(p.status!==0)continue;
      } else entry.preparation=prior?{...prior.preparation,reusedPriorAcceptedPreparation:true}:{status:'reused-existing-task-assets',timing:'No new preparation timing on replay'};
      const before=performance.now(),child=spawnSync(process.execPath,['--expose-gc','scripts/benchmark-scene-scaling.mjs','--child',id],{cwd:root,encoding:'utf8',timeout:Math.min(30000,deadline-performance.now()),maxBuffer:4*1024*1024});
      entry.childWallMilliseconds=performance.now()-before;
      if(child.status!==0) {entry.benchmark={status:'failed',exitStatus:child.status,signal:child.signal,stdout:child.stdout,stderr:child.stderr,error:child.error?.message};report.errors.push({name:'BenchmarkChildFailure',id,exitStatus:child.status,signal:child.signal,message:child.error?.message??'Benchmark child did not exit successfully.'});continue;}
      entry.benchmark=JSON.parse(child.stdout);
      assert.equal(entry.benchmark.status,'completed','Benchmark child did not report completion');
      assert.deepEqual(entry.benchmark.sourceHashesDeclaredByScene,Object.fromEntries(Object.entries(rawSourceFiles).map(([k,v])=>[k,v.sha256])),'Prepared metadata differs from current raw source hashes');
      assert.equal(entry.benchmark.processingCodeHashesDeclaredByScene.preparation,sourceFiles['scripts/prepare-scene.py'].sha256);
      assert.equal(entry.benchmark.processingCodeHashesDeclaredByScene.calibration,sourceFiles['scripts/rectified_calibration.py'].sha256);
    }
    assert.deepEqual(await sources(),sourceFiles,'Source changed during measurement');
    report.acceptedScenes=report.frames.filter(f=>f.benchmark?.status==='completed').length;
    report.failedBenchmarkScenes=report.frames.filter(f=>f.benchmark?.status==='failed').length;
    report.preparationRejectedScenes=report.frames.filter(f=>f.preparation?.status==='rejected').length;
    report.status=report.failedBenchmarkScenes?'failed':report.acceptedScenes===5?'completed':report.acceptedScenes?'completed-with-rejections':'rejected-data';
    if(report.failedBenchmarkScenes)process.exitCode=1;
  } catch(e){report.errors.push({name:e.name,message:e.message});process.exitCode=1;}
  report.elapsedMilliseconds=performance.now()-begun;
  await writeFile(`${root}/${output}/report.json`,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({status:report.status,acceptedScenes:report.acceptedScenes,elapsedMilliseconds:report.elapsedMilliseconds,output:`${output}/report.json`,errors:report.errors}));
}

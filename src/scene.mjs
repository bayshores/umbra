import { validateSuggestions } from './suggestions.mjs';
import { prepareCarShape, CAR_SHAPE_REFUSALS } from './geometry.mjs';

const LOAD_TIMEOUT_MS=30_000;
const timeoutError=()=>new Error('Scene loading exceeded 30 seconds. Retry loading to try again.');
function checkActive(signal,deadline) {
  signal.throwIfAborted();
  if(performance.now()>=deadline)throw timeoutError();
}
function checkResponse(response,signal,deadline) {
  if(!response.ok||signal.aborted||performance.now()>=deadline)
    void response.body?.cancel(signal.reason).catch(()=>{});
  checkActive(signal,deadline);
  return response;
}
async function boundedBytes(response,limit,exact,signal,deadline) {
  checkResponse(response,signal,deadline);
  const reader=response.body?.getReader();
  if(!reader)throw new Error('Artifact stream unavailable.');
  // Cleanup must not await an untrusted underlying cancel promise.
  const cancel=reason=>{void reader.cancel(reason).catch(()=>{});};
  const abort=()=>cancel(signal.reason);
  signal.addEventListener('abort',abort,{once:true});
  const bytes=new Uint8Array(limit);let offset=0;
  try {
    for(;;){checkActive(signal,deadline);const {done,value}=await reader.read();checkActive(signal,deadline);if(done)break;
      if(offset+value.byteLength>limit)throw new Error('Artifact exceeds declared byte length.');
      bytes.set(value,offset);offset+=value.byteLength;}
    if(exact&&offset!==limit)throw new Error('Artifact byte length mismatch.');
    return exact?bytes.buffer:bytes.slice(0,offset).buffer;
  }catch(error){cancel(error);throw error;}finally{signal.removeEventListener('abort',abort);reader.releaseLock();}
}

function recordedCpuConfiguration(execution) {
  const threads=execution?.usage?.threads,cpu=execution?.cpuConfiguration;
  if(![1,2,4].includes(threads))return false;
  // Original pinned one-thread artifacts predate the explicit CPU configuration.
  if(cpu===undefined)return threads===1;
  return cpu!==null&&typeof cpu==='object'&&!Array.isArray(cpu)&&
    Object.keys(cpu).sort().join()==='interOpThreads,intraOpThreads,provider'&&
    cpu.intraOpThreads===threads&&cpu.interOpThreads===1&&cpu.provider==='CPUExecutionProvider';
}

function matchingBounds(positions,bounds) {
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<positions.length;i++) {
    const k=i%3;min[k]=Math.min(min[k],positions[i]);max[k]=Math.max(max[k],positions[i]);
  }
  // Legacy descriptors retain pre-Float32 extrema; preserve the existing rendering tolerance.
  return min.every((n,k)=>Math.abs(n-bounds.min[k])<=2e-5&&Math.abs(max[k]-bounds.max[k])<=2e-5);
}

// ground.json schemaVersion 3 (placement protocol v3.4): the schema 2 keys (protocol v3: the schema 1 keys plus the
// placement range limit and the fitted along-ray uncertainty band) plus carShape, the recorded stationary vehicle's
// LiDAR shape (shape-v2). Schema 1 and 2 layers, and schema 3 layers with a shape-v1 car shape (placement protocol v3.3),
// are history and are refused (they load only with the app source of their own commit).
const GROUND_KEYS=['alongRayUncertainty','baseSceneSha256','boundaryEpsilonMeters','carShape','crossing','cuboids','dependencies','drivableAreas','egoLane','frame','kind',
  'license','limitations','maxRangeMeters','placementRangeLimitMeters','processingCodeHashes','protocol','roadToleranceMeters','sceneId','schemaVersion','sources',
  'support','supportHeightAgreementMeters','supportRadiusMeters','terrain'];
// Frozen protocols whose layers have the current schema numbers but are superseded, by sha256. Protocol v3
// (.local/evaluation/av2-map-first-2026-10-11/protocol-v3.json) put a vehicle declared in place of the stationary one at
// the front pixel's raster-shadowed ray hit (false Blocked on AV2 8c); protocol v3.1 fixed it (declaredReplacement). Its
// files are internally consistent, so without this check the app would apply the v3.1 rule to v3 data.
const SUPERSEDED_PROTOCOLS=Object.freeze({c91066034d6b13ce06ab1ee14cd70a148a4531f93692ccac7e21b58ec2f796eb:'protocol v3'});
const SOURCE_KEYS='annotations,assembledScene,groundRaster,groundTransform,map,poses,preparedManifest,references';
const CUBOID_KEYS='category,center,height,length,motion,numInteriorPoints,track,width,worldFromBox';
const TERRAIN_KEYS='byteOrder,bytes,cellMeters,column0,columns,dtype,file,finiteCells,layout,nanCells,rasterShape,row0,rows,samplingRule,sha256,sim2,values';
// Structural checks on ground.json (schemaVersion 3) before any value is used; the bytes are already hash-pinned.
function validateGround(g,scene) {
  const need=(ok,detail)=>{if(!ok)throw new Error(`Invalid mapped ground: ${detail}.`);};
  const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v),text=v=>typeof v==='string'&&v.length>0&&v.length<=4096,
    hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v),finite=v=>typeof v==='number'&&Number.isFinite(v),
    finite3=p=>Array.isArray(p)&&p.length===3&&p.every(finite),count=v=>Number.isSafeInteger(v)&&v>=0,
    positiveInt=v=>Number.isSafeInteger(v)&&v>0,keys=(v,list)=>object(v)&&Object.keys(v).sort().join()===list;
  need(object(g)&&(g.schemaVersion===3||g.schemaVersion===4&&object(g.measuredGround)),
    'schemaVersion must be 3 (placement protocol v3.4 with the measured car shape; earlier ground layers are history)');
  const measured=g.schemaVersion===4;
  need(keys(g,(measured?[...GROUND_KEYS,'measuredGround'].sort():GROUND_KEYS).join()),'unexpected or missing top-level fields');
  need(g.kind==='av2MappedGround'&&g.sceneId===scene.id,'schema, kind or scene identity');
  need(hash(g.baseSceneSha256)&&keys(g.protocol,'bytes,path,readOnlyExternal,sha256')&&hash(g.protocol.sha256)&&text(g.protocol.path)&&count(g.protocol.bytes),'protocol record');
  need(!Object.hasOwn(SUPERSEDED_PROTOCOLS,g.protocol.sha256),`${SUPERSEDED_PROTOCOLS[g.protocol.sha256]} is superseded (protocol v3.1 or later is required)`);
  // Every source is a relative read-only record; only the annotations may be absent (null), and then no cuboids exist.
  const source=s=>keys(s,'bytes,path,readOnlyExternal,sha256')&&hash(s.sha256)&&text(s.path)&&!s.path.startsWith('/')&&count(s.bytes)&&s.readOnlyExternal===true;
  need(keys(g.sources,SOURCE_KEYS)&&Object.entries(g.sources).every(([k,s])=>k==='annotations'&&s===null||source(s))&&
    g.baseSceneSha256===g.sources.assembledScene.sha256&&(g.sources.annotations===null)===(g.cuboids?.available===false),'source records');
  need(object(g.processingCodeHashes)&&Object.values(g.processingCodeHashes).every(hash)&&object(g.dependencies)&&Object.values(g.dependencies).every(text),'processing provenance');
  need(keys(g.frame,'cityOrigin,convention,worldFromCity')&&finite3(g.frame.cityOrigin)&&g.frame.cityOrigin.every((n,i)=>n===scene.transform.cityOrigin[i])&&
    JSON.stringify(g.frame.worldFromCity)==='[[1,0,0],[0,0,1],[0,-1,0]]','frame must equal the scene city frame');
  need(['roadToleranceMeters','supportHeightAgreementMeters','supportRadiusMeters','boundaryEpsilonMeters'].every(k=>finite(g[k])&&g[k]>0&&g[k]<=5)&&
    finite(g.maxRangeMeters)&&g.maxRangeMeters>0&&g.maxRangeMeters<=1000,'tolerances and range');
  need(finite(g.placementRangeLimitMeters)&&g.placementRangeLimitMeters>0&&g.placementRangeLimitMeters<=g.maxRangeMeters,'placement range limit');
  // The app uses only heightBandMeters; the rest is provenance (a development fit, or a fixed band tested on this scene).
  const band=g.alongRayUncertainty;
  if(measured) {
    need(object(band)&&band.method==='supportSweepHoldout'&&band.heightBandMeters===null&&text(band.rule)&&band.quantile===.975&&
      positiveInt(band.minimumCount)&&Array.isArray(band.bins)&&band.bins.every((b,i)=>object(b)&&finite(b.minRangeM)&&b.minRangeM>=0&&
        finite(b.maxRangeM)&&b.maxRangeM>b.minRangeM&&finite(b.meters)&&b.meters>=0&&positiveInt(b.count)&&b.count>=band.minimumCount&&
        (i===0||b.minRangeM>=band.bins[i-1].maxRangeM)),'independent click-error calibration');
    need(band.rangeRule==='linear'?band.bins.length===0&&Array.isArray(band.knots)&&(band.knots.length===0||band.knots.length>=2)&&
      band.knots.every((k,i)=>object(k)&&finite(k.rangeM)&&k.rangeM>=0&&finite(k.meters)&&k.meters>=0&&positiveInt(k.count)&&k.count>=band.minimumCount&&
        (i===0||k.rangeM>band.knots[i-1].rangeM)):!Object.hasOwn(band,'rangeRule')&&!Object.hasOwn(band,'knots'),'calibrated range function');
  }else need(object(band)&&finite(band.heightBandMeters)&&band.heightBandMeters>0&&band.heightBandMeters<=1&&text(band.rule)&&object(band.coverage)&&
    (band.method==='fitThisDevelopmentScene'&&object(band.fit)&&!Object.hasOwn(band,'heldOutResiduals')||
     band.method==='fixed'&&text(band.fittedOn)&&object(band.heldOutResiduals)&&!Object.hasOwn(band,'fit')),'along-ray uncertainty band');
  const t=g.terrain,sim=t?.sim2,R=sim?.R;
  need(keys(t,TERRAIN_KEYS)&&t.file==='terrain.bin'&&t.dtype==='float32'&&t.byteOrder==='little'&&hash(t.sha256)&&positiveInt(t.columns)&&positiveInt(t.rows)&&
    t.bytes===t.rows*t.columns*4&&t.bytes<=16*1024*1024&&count(t.column0)&&count(t.row0)&&Array.isArray(t.rasterShape)&&t.rasterShape.length===2&&t.rasterShape.every(positiveInt)&&
    t.row0+t.rows<=t.rasterShape[0]&&t.column0+t.columns<=t.rasterShape[1]&&count(t.finiteCells)&&count(t.nanCells)&&t.finiteCells+t.nanCells===t.rows*t.columns&&
    finite(t.cellMeters)&&t.cellMeters>0,'terrain descriptor');
  need(keys(sim,'R,s,t')&&Array.isArray(R)&&R.length===4&&R.every(finite)&&Array.isArray(sim.t)&&sim.t.length===2&&sim.t.every(finite)&&finite(sim.s)&&sim.s>0&&
    Math.abs(R[0]*R[0]+R[2]*R[2]-1)<=1e-9&&Math.abs(R[1]*R[1]+R[3]*R[3]-1)<=1e-9&&Math.abs(R[0]*R[1]+R[2]*R[3])<=1e-9&&Math.abs(R[0]*R[3]-R[1]*R[2]-1)<=1e-9,'terrain Sim2 must be a proper similarity');
  need(Array.isArray(g.drivableAreas)&&g.drivableAreas.length>0&&g.drivableAreas.every(a=>keys(a,'id,outline')&&text(a.id)&&Array.isArray(a.outline)&&a.outline.length>=3&&
    a.outline.every(p=>Array.isArray(p)&&p.length===2&&p.every(finite))),'drivable areas');
  const edge=e=>Array.isArray(e)&&e.length>=2&&e.every(finite3);
  need(keys(g.crossing,'edge1,edge2,id')&&g.crossing.id===scene.measurements.provenance.crossingId&&edge(g.crossing.edge1)&&edge(g.crossing.edge2),'crossing');
  const lane=g.egoLane;
  need(keys(lane,'id,laneType,leftBoundary,leftMarkType,rightBoundary,rightMarkType')&&['id','laneType','leftMarkType','rightMarkType'].every(k=>text(lane[k]))&&
    edge(lane.leftBoundary)&&edge(lane.rightBoundary),'ego lane');
  const mid=(l,r)=>[(l[0]+r[0])/2,(l[2]+r[2])/2],a=mid(lane.leftBoundary[0],lane.rightBoundary[0]),b=mid(lane.leftBoundary.at(-1),lane.rightBoundary.at(-1));
  need(Math.hypot(b[0]-a[0],b[1]-a[1])>1e-6,'ego lane chord is degenerate');
  const c=g.cuboids;
  need(keys(c,'available,frame,items,timestampNs')&&typeof c.available==='boolean'&&text(c.frame)&&typeof c.timestampNs==='string'&&/^[0-9]{1,32}$/.test(c.timestampNs)&&
    Array.isArray(c.items)&&(c.available||c.items.length===0),'cuboids');
  for(const box of c.items) {
    // poseAverage: the current stationary vehicle's averaged annotations (protocol v3 currentVehicle.pose); provenance only.
    const pose=box?.poseAverage,averaged=object(box)&&Object.hasOwn(box,'poseAverage');
    const M=box?.worldFromBox,ok=keys(box,averaged?CUBOID_KEYS.replace('numInteriorPoints','numInteriorPoints,poseAverage'):CUBOID_KEYS)&&
      text(box.track)&&text(box.category)&&finite3(box.center)&&['length','width','height'].every(k=>finite(box[k])&&box[k]>0)&&
      count(box.numInteriorPoints)&&(box.motion===null||object(box.motion)&&Object.values(box.motion).every(finite))&&Array.isArray(M)&&M.length===3&&M.every(finite3)&&
      (!averaged||object(pose)&&pose.rule==='averageAllAnnotations'&&positiveInt(pose.rows)&&object(pose.centreSpreadM)&&
        ['alongStd','lateralStd'].every(k=>finite(pose.centreSpreadM[k])&&pose.centreSpreadM[k]>=0)&&object(pose.yawSpreadDeg)&&finite(pose.yawSpreadDeg.std)&&pose.yawSpreadDeg.std>=0);
    need(ok,'cuboid item');
    const det=M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
    need([0,1,2].every(i=>[0,1,2].every(j=>Math.abs(M[0][i]*M[0][j]+M[1][i]*M[1][j]+M[2][i]*M[2][j]-(i===j?1:0))<=1e-6))&&Math.abs(det-1)<=1e-6,'cuboid rotation must be proper orthonormal');
  }
  const s=g.support,m=scene.measurements;
  need(object(s)&&Array.isArray(s.rows)&&s.count===s.rows.length&&s.rows.every((r,i)=>Number.isSafeInteger(r)&&r>=0&&r<m.count&&(i===0||r>s.rows[i-1])&&m.referenceEligible[r]===1&&
    Number.isFinite(m.referencePositions[r*3])&&Number.isFinite(m.referencePositions[r*3+2])),'support rows must be ascending camera-eligible source rows');
  if(measured) {
    const surface=g.measuredGround,p=surface?.partition,stamp=v=>typeof v==='string'&&/^(0|[1-9][0-9]{0,31})$/.test(v);
    need(object(surface)&&surface.version==='measured-ground-v1'&&finite(surface.cellMeters)&&surface.cellMeters>0&&
      positiveInt(surface.columns)&&positiveInt(surface.rows)&&surface.columns>=2&&surface.rows>=2&&surface.columns*surface.rows<=262144&&
      Array.isArray(surface.originXZ)&&surface.originXZ.length===2&&surface.originXZ.every(finite)&&Array.isArray(surface.heights)&&
      surface.heights.length===surface.rows*surface.columns&&surface.heights.every(h=>h===null||finite(h)),'measured surface grid');
    const heights=surface.heights.filter(h=>h!==null);
    let low=Infinity,high=-Infinity;for(const height of heights){low=Math.min(low,height);high=Math.max(high,height);}
    need(heights.length?Array.isArray(surface.heightBounds)&&surface.heightBounds.length===2&&surface.heightBounds[0]===low&&
      surface.heightBounds[1]===high:surface.heightBounds===null,'measured height bounds');
    // The scoring sweep's identity must exist for the byte-level exclusion below to mean anything.
    const scoringHash=m.provenance.originalSources?.lidar?.sha256;
    need(hash(scoringHash),'measured support needs the scoring sweep sha256 (measurements.provenance.originalSources.lidar)');
    need(s.rows.length===0&&object(p)&&p.frameTimestampNs===m.provenance.lidarTimestampNs&&stamp(p.frameTimestampNs)&&
      p.excludedNearNs===250000000&&Array.isArray(p.fit)&&Array.isArray(p.calibration)&&[...p.fit,...p.calibration].every(v=>object(v)&&stamp(v.timestampNs)&&hash(v.sha256)&&
        v.sha256!==scoringHash&&
        (BigInt(v.timestampNs)-BigInt(p.frameTimestampNs)>250000000n||BigInt(p.frameTimestampNs)-BigInt(v.timestampNs)>250000000n))&&
      new Set([...p.fit,...p.calibration].map(v=>v.timestampNs)).size===p.fit.length+p.calibration.length&&
      new Set([...p.fit,...p.calibration].map(v=>v.sha256)).size===p.fit.length+p.calibration.length&&(!heights.length||p.fit.length>=3),
      'measured support must exclude the scoring sweep and its temporal neighbours, with disjoint calibration sweeps');
    const coverage=band.coverage;
    const calibrated=band.rangeRule==='linear'?band.knots:band.bins;
    need(!calibrated.length||(p.calibration.length>0&&object(coverage)&&count(coverage.eligible)&&positiveInt(coverage.placed)&&
      object(coverage.unknown)&&Object.values(coverage.unknown).every(count)&&coverage.placed+Object.values(coverage.unknown).reduce((a,b)=>a+b,0)===coverage.eligible&&
      calibrated.every(b=>b.count<=coverage.placed)),'bands require independent calibration sweeps and consistent coverage counts');
  }
  need(keys(g.license,'attribution,changes,id,source')&&g.license.id==='CC-BY-NC-SA-4.0'&&['attribution','changes','source'].every(k=>text(g.license[k])),'license');
  need(Array.isArray(g.limitations)&&g.limitations.every(text),'limitations');
  if(g.carShape!==null)validateCarShape(g.carShape,c.items,{need,object,text,hash,finite,count,positiveInt,keys});
}
// shape-v2 (.local/evaluation/pao-accuracy/shape-v2/protocol.json: the shape-v1 dataSchema and loaderValidation with
// the 0.25 m sweep step, schema 'shape-v2' and placement protocol v3.4): the recorded
// stationary vehicle's LiDAR shape. Any failure rejects the whole scene; there is no box fallback. Whether it must be
// present (exactly when suggestions name an annotated vehicle, with the same track) is checked after suggestions load.
const CAR_SHAPE_KEYS='box,code,counts,diagnostics,frame,grid,protocol,refusal,removed,schema,solidTopM,status,sweeps,track';
const CAR_SHAPE_SWEEP_KEYS='candidates,downloadDropped,kept,listing,planSha256,rule,scoring,shape,thinned';
const CAR_SHAPE_SCHEMA='shape-v2',CAR_SHAPE_PLACEMENT_PROTOCOL='3.4';
const CAR_SHAPE_RULE=Object.freeze({maxEgoDistanceM:30,minEgoStepM:.25,maxKept:40,split:'alternate',minShapeSweeps:2,downloadAttempts:4,retryWaitsS:[5,10,20]});
const CAR_SHAPE_GRID_ORDER='index = i*lateral + j; i from the front face, j from the left side';
function validateCarShape(s,cuboids,{need,object,text,hash,finite,count,positiveInt,keys}) {
  const fail=detail=>need(false,`car shape ${detail}`),check=(ok,detail)=>{if(!ok)fail(detail);};
  check(keys(s,CAR_SHAPE_KEYS),'has unexpected or missing fields');
  check(s.schema!=='shape-v1','is shape-v1 (placement protocol v3.3); earlier car shapes are history');
  check(s.schema===CAR_SHAPE_SCHEMA&&['measured','refused'].includes(s.status)&&text(s.track)&&text(s.frame),'schema, status, track or frame');
  const owners=cuboids.filter(box=>Object.hasOwn(box,'poseAverage')&&box.track===s.track);
  check(owners.length===1,'track must name exactly one averaged annotated cuboid');
  const [box]=owners,L=box.length,W=box.width,H=box.height;
  check(keys(s.box,'height,length,width')&&s.box.length===L&&s.box.width===W&&s.box.height===H,'box must equal the averaged cuboid sizes exactly');
  const n=Math.ceil(L/.1-1e-9),m=Math.ceil(W/.1-1e-9),N=n*m,g=s.grid;
  check(keys(g,'along,cellAlongM,cellLateralM,lateral,maxCellM,order')&&g.along===n&&g.lateral===m&&g.cellAlongM===L/n&&g.cellLateralM===W/m&&g.maxCellM===.1&&
    g.order===CAR_SHAPE_GRID_ORDER&&positiveInt(n)&&positiveInt(m),'grid must be recomputed from the box');
  const cellCounts=a=>Array.isArray(a)&&a.length===N&&a.every(count);
  check(keys(s.counts,'returnSweeps,returns')&&cellCounts(s.counts.returns)&&cellCounts(s.counts.returnSweeps)&&
    s.counts.returnSweeps.every((k,i)=>k<=s.counts.returns[i]),'counts');
  check(Array.isArray(s.solidTopM)&&s.solidTopM.length===N&&s.solidTopM.every((v,i)=>v===null||finite(v)&&v>=.25&&v<=H&&s.counts.returnSweeps[i]>=2),
    'solid tops must be null or in [0.25 m, box height] with returns from at least 2 sweeps');
  check(keys(s.removed,'aboveBoxTop,belowFloor,otherObjects')&&Object.values(s.removed).every(count),'removed counts');
  const w=s.sweeps,rule=w?.rule,decimal=v=>typeof v==='string'&&/^[0-9]{1,32}$/.test(v);
  check(keys(w,CAR_SHAPE_SWEEP_KEYS)&&keys(rule,Object.keys(CAR_SHAPE_RULE).sort().join())&&
    Object.entries(CAR_SHAPE_RULE).every(([k,v])=>JSON.stringify(rule[k])===JSON.stringify(v)),'sweep rule');
  check(hash(w.planSha256)&&keys(w.listing,'attempts,status')&&['ok','failed'].includes(w.listing.status)&&count(w.listing.attempts)&&
    [w.candidates,w.thinned,w.kept].every(count),'sweep plan record');
  // Each list is in increasing time order (compared as integers); no sweep is in two lists.
  const increasing=list=>list.every((x,i)=>i===0||BigInt(x.timestampNs)>BigInt(list[i-1].timestampNs));
  const used=list=>Array.isArray(list)&&list.every(x=>keys(x,'sha256,timestampNs')&&decimal(x.timestampNs)&&hash(x.sha256))&&increasing(list);
  check(used(w.shape)&&used(w.scoring),'shape and scoring sweeps must be increasing decimal timestamps with SHA-256 hashes');
  check(Array.isArray(w.downloadDropped)&&w.downloadDropped.every(x=>keys(x,'attempts,reason,set,timestampNs')&&decimal(x.timestampNs)&&
    ['shape','scoring'].includes(x.set)&&x.reason==='download-failed'&&x.attempts===4),'dropped downloads');
  const stamps=[...w.shape,...w.scoring,...w.downloadDropped].map(x=>BigInt(x.timestampNs));
  check(new Set(stamps).size===stamps.length,'sweep lists must be disjoint');
  check(w.shape.length+w.scoring.length+w.downloadDropped.length===w.kept,'sweep lists must add up to the kept count');
  if(s.status==='measured')check(s.refusal===null&&w.shape.length>=2,'measured needs no refusal and at least 2 shape sweeps');
  else check(keys(s.refusal,'code,reason')&&Object.hasOwn(CAR_SHAPE_REFUSALS,s.refusal.code)&&s.refusal.reason===CAR_SHAPE_REFUSALS[s.refusal.code]&&
    s.solidTopM.every(v=>v===null)&&s.counts.returns.every(k=>k===0)&&s.counts.returnSweeps.every(k=>k===0),'refused needs a known refusal, no solid tops and zero counts');
  const d=s.diagnostics;
  check(keys(d,'frontFaceToFirstSolidCellM,returnsBeyondEnds')&&(d.frontFaceToFirstSolidCellM===null||finite(d.frontFaceToFirstSolidCellM))&&
    keys(d.returnsBeyondEnds,'front,rear')&&count(d.returnsBeyondEnds.front)&&count(d.returnsBeyondEnds.rear),'diagnostics');
  check(keys(s.protocol,'placementProtocolVersion,sha256,shapeProtocolSha256')&&s.protocol.placementProtocolVersion===CAR_SHAPE_PLACEMENT_PROTOCOL&&hash(s.protocol.sha256)&&
    hash(s.protocol.shapeProtocolSha256),'protocol record');
  check(keys(s.code,'path,sha256')&&s.code.path==='scripts/california/car_shape.py'&&hash(s.code.sha256),'code record');
}
// The scene's descriptor {file,bytes,sha256} for ground.json and the catalog's caller pin {file,sha256} must agree.
function validGroundDescriptor(descriptor,pin) {
  const plain=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
  return plain(descriptor)&&Object.keys(descriptor).sort().join()==='bytes,file,sha256'&&descriptor.file==='ground.json'&&
    Number.isSafeInteger(descriptor.bytes)&&descriptor.bytes>0&&descriptor.bytes<=1024*1024&&typeof descriptor.sha256==='string'&&/^[a-f0-9]{64}$/.test(descriptor.sha256)&&
    plain(pin)&&Object.keys(pin).sort().join()==='file,sha256'&&pin.file===descriptor.file&&pin.sha256===descriptor.sha256;
}

// suggestions: the catalog entry's {file,sha256} descriptor, or undefined when the entry has none.
// ground: the catalog entry's {file,sha256} pin for a mapped-ground scene's ground.json, or undefined.
export async function loadScene(fetchResource,id,expectedMetadataSha256,callerSignal,suggestions,ground) {
  const controller=new AbortController(),signal=controller.signal,deadline=performance.now()+LOAD_TIMEOUT_MS;
  const abort=()=>controller.abort(callerSignal.reason);
  const interrupted=new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));
  const timer=setTimeout(()=>controller.abort(timeoutError()),LOAD_TIMEOUT_MS);
  if(callerSignal?.aborted)abort();else callerSignal?.addEventListener('abort',abort,{once:true});
  const fetchScoped=async path=>{
    checkActive(signal,deadline);
    const response=await fetchResource(path,{signal});
    // A custom transport may ignore abort; discard/cancel a response arriving after failure.
    return checkResponse(response,signal,deadline);
  };
  try {
    return await Promise.race([loadSceneData(fetchScoped,id,expectedMetadataSha256,signal,deadline,suggestions,ground),interrupted]);
  }catch(error){controller.abort(error);throw error;}
  finally{clearTimeout(timer);callerSignal?.removeEventListener('abort',abort);}
}

async function loadSceneData(fetchResource,id,expectedMetadataSha256,signal,deadline,suggestions,ground) {
  if(typeof id!=='string'||!/^[a-zA-Z0-9_-]+$/.test(id))throw new Error('Invalid scene identifier.');
  const hash=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s),
    digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
  if(expectedMetadataSha256!==undefined&&!hash(expectedMetadataSha256))throw new Error('Invalid expected scene metadata checksum.');
  const base=`/scenes/${id}/`,response=await fetchResource(base+'scene.json');
  if(!response.ok)throw new Error(`Scene metadata unavailable (${response.status}).`);
  const metadataBytes=await boundedBytes(response,128*1024,false,signal,deadline);
  if(expectedMetadataSha256!==undefined&&await digest(metadataBytes)!==expectedMetadataSha256)throw new Error('Scene metadata checksum mismatch.');
  const scene=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(metadataBytes)),finite3=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),
    positive=n=>Number.isFinite(n)&&n>0,integer=n=>Number.isSafeInteger(n)&&n>0;
  const indexed=Object.hasOwn(scene,'measurements'),measurement=scene.measurements,
    cameraOrigin=indexed?scene.transform?.cameraOrigin:scene.transform?.cameraOriginGround;
  if(indexed&&!hash(expectedMetadataSha256))throw new Error('Indexed scene requires a caller-pinned metadata checksum.');
  // 'unvalidated' stays read-only. 'mappedGround' needs a hash-pinned ground.json named by both the pinned scene and the catalog.
  const mapped=indexed&&measurement?.groundPolicy==='mappedGround';
  if(mapped?!validGroundDescriptor(measurement.ground,ground):ground!==undefined||(indexed&&Object.hasOwn(measurement??{},'ground')))
    throw new Error('Invalid mapped ground descriptor.');
  if(indexed&&(measurement?.kind!=='indexedSparse'||measurement.evidence!=='geometricallyProjected'||!['unvalidated','mappedGround'].includes(measurement.groundPolicy)||
    measurement.pixelCellPolicy!=='nearestIntegerCenterRejectCollisions'||measurement.binaryEncoding!=='littleEndian'||!integer(measurement.count)||measurement.count>1_000_000||scene.triangleCount!==0||
    scene.transform?.originRole!=='leftCamera'||!finite3(scene.transform.cityOrigin)||Object.hasOwn(scene.transform,'cameraOriginGround')||
    JSON.stringify(scene.transform.worldFromCity)!=='[[1,0,0],[0,0,1],[0,-1,0]]'||
    !Array.isArray(cameraOrigin)||cameraOrigin.some(n=>n!==0)||scene.camera?.eye?.some(n=>n!==0)||scene.pointCount>measurement.count||scene.camera.width*scene.camera.height>16_000_000||
    Object.hasOwn(scene.buffers??{},'depth')||Object.hasOwn(scene.processing??{},'depthDivisor')||Object.hasOwn(scene.processing??{},'roadToleranceMeters')))
    throw new Error('Invalid indexed measurement policy.');
  if(scene.id!==id||scene.units!=='meters'||!finite3(scene.camera?.eye)||!finite3(scene.camera?.forward)||!finite3(scene.camera?.up)||!finite3(cameraOrigin)||
    !Array.isArray(scene.transform?.worldFromCamera)||scene.transform.worldFromCamera.length!==3||!scene.transform.worldFromCamera.every(finite3)||
    !['fx','fy','near','far'].every(k=>positive(scene.camera[k]))||
    !['width','height'].every(k=>integer(scene.camera[k]))||!Number.isSafeInteger(scene.camera.width*scene.camera.height)||scene.camera.width*scene.camera.height>16_000_000||scene.camera.near>=scene.camera.far||
    !['cx','cy'].every(k=>Number.isFinite(scene.camera[k]))||(!indexed&&(!positive(scene.processing?.depthDivisor)||!positive(scene.processing?.roadToleranceMeters))))throw new Error('Invalid measured calibration.');
  if(!integer(scene.pointCount)||scene.pointCount>1_000_000||!Number.isSafeInteger(scene.triangleCount)||scene.triangleCount<0||scene.triangleCount>2_000_000||
    !finite3(scene.bounds?.min)||!finite3(scene.bounds?.max)||scene.bounds.max.some((n,i)=>n<scene.bounds.min[i])||
    !positive(Math.hypot(...scene.bounds.max.map((n,i)=>n-scene.bounds.min[i]))))throw new Error('Invalid measured geometry metadata.');
  // cameraPoint uses the transpose as the inverse; accept six-decimal serialization, never repair the rotation.
  const R=scene.transform.worldFromCamera,rotationTolerance=1e-5;
  const [a,b,c]=R,determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
  if(R.some((row,i)=>R.some((other,j)=>Math.abs(row.reduce((sum,n,k)=>sum+n*other[k],0)-(i===j?1:0))>rotationTolerance))||
    Math.abs(determinant-1)>rotationTolerance)throw new Error('Invalid measured calibration: expected a proper orthonormal rotation.');
  // Calculation and renderer must use the same camera pose (OpenCV x-right/y-down/z-forward).
  if(R.some((row,i)=>Math.abs(scene.camera.eye[i]+row.reduce((sum,n,j)=>sum+n*cameraOrigin[j],0))>rotationTolerance||
    Math.abs(scene.camera.forward[i]-row[2])>rotationTolerance||Math.abs(scene.camera.up[i]+row[1])>rotationTolerance))
    throw new Error('Invalid measured calibration: camera pose disagrees with its transform.');
  if(indexed) {
    const image=scene.recordedImage;
    if(image?.file!=='recorded.png'||image.mimeType!=='image/png'||!integer(image.bytes)||image.bytes>16*1024*1024||!hash(image.sha256))
      throw new Error('Invalid indexed recorded image descriptor.');
    const r=await fetchResource(base+image.file);if(!r.ok)throw new Error(`Recorded image unavailable (${r.status}).`);
    const bytes=await boundedBytes(r,image.bytes,true,signal,deadline);
    if(await digest(bytes)!==image.sha256)throw new Error('Recorded image checksum mismatch.');
    // Bind PNG's declared dimensions before browser decoding; the app also checks actual decoded size.
    const view=new DataView(bytes),signature=[137,80,78,71,13,10,26,10];
    if(bytes.byteLength<33||!signature.every((n,i)=>view.getUint8(i)===n)||view.getUint32(8)!==13||view.getUint32(12)!==0x49484452||
      view.getUint32(16)!==scene.camera.width||view.getUint32(20)!==scene.camera.height)
      throw new Error('Recorded PNG dimensions disagree with calibration.');
    scene.recordedImageBlob=new Blob([bytes],{type:image.mimeType});
  }
  const nativeTypes={sourceIndices:Uint32Array,referencePositions:Float64Array,referenceCameraPositions:Float64Array,referenceUV:Float64Array,referenceEligible:Uint8Array};
  if(indexed) {
    const expected=['positions','colors','triangles',...Object.keys(nativeTypes)],names=[...Object.values(scene.buffers??{}),...Object.values(measurement.buffers??{})],
      hash=n=>typeof n==='string'&&/^[a-f0-9]{64}$/.test(n);
    if(names.length!==expected.length||new Set(names).size!==names.length||Object.keys(scene.buffers).sort().join()!=='colors,positions,triangles'||
      Object.keys(measurement.buffers??{}).sort().join()!==Object.keys(nativeTypes).sort().join()||
      !hash(measurement.provenance?.preparedManifestSha256)||!hash(measurement.provenance?.referencesSha256)||
      typeof measurement.provenance?.logId!=='string'||!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(measurement.provenance.logId)||
      !['cameraTimestampNs','lidarTimestampNs','crossingId'].every(k=>typeof measurement.provenance?.[k]==='string'&&/^[0-9]{1,32}$/.test(measurement.provenance[k])))throw new Error('Invalid indexed artifact ownership.');
  }
  const byteLengths={positions:scene.pointCount*3*4,colors:scene.pointCount*3*4,triangles:scene.triangleCount*3*4,depth:scene.camera.width*scene.camera.height*2,
    sourceIndices:measurement?.count*4,referencePositions:measurement?.count*3*8,referenceCameraPositions:measurement?.count*3*8,referenceUV:measurement?.count*2*8,referenceEligible:measurement?.count};
  const loadBuffers=async(target,types)=>Promise.all(Object.entries(types).map(async([key,Type])=>{
    const name=target.buffers?.[key];
    if(!/^[a-z]+\.bin$/.test(name))throw new Error(`Invalid ${key} artifact path.`);
    const length=target===scene||target===measurement?byteLengths[key]:key==='triangles'?target.triangleCount*3*4:target.pointCount*3*4;
    const r=await fetchResource(base+name);if(!r.ok)throw new Error(`${key} unavailable (${r.status}).`);
    const buffer=await boundedBytes(r,length,true,signal,deadline);
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),n=>n.toString(16).padStart(2,'0')).join('');
    if(digest!==target.artifactHashes?.[key])throw new Error(`${key} checksum mismatch.`);
    target[key]=new Type(buffer);
  }));
  await loadBuffers(scene,indexed?{positions:Float32Array,colors:Float32Array,triangles:Uint32Array}:{positions:Float32Array,colors:Float32Array,triangles:Uint32Array,depth:Uint16Array});
  if(scene.positions.length!==scene.pointCount*3||scene.colors.length!==scene.positions.length||scene.triangles.length!==scene.triangleCount*3||(!indexed&&scene.depth.length!==scene.camera.width*scene.camera.height)||
    !scene.positions.every(Number.isFinite)||!scene.colors.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||!scene.triangles.every(n=>n<scene.pointCount)||
    (!indexed&&!matchingBounds(scene.positions,scene.bounds)))throw new Error('Invalid measured geometry buffers.');
  if(indexed) {
    await loadBuffers(measurement,nativeTypes);
    const {count,sourceIndices,referencePositions,referenceCameraPositions,referenceUV,referenceEligible}=measurement,c=scene.camera;
    if(sourceIndices.length!==count||referencePositions.length!==count*3||referenceCameraPositions.length!==count*3||referenceUV.length!==count*2||referenceEligible.length!==count||
      !referencePositions.every(n=>Number.isFinite(n)||Number.isNaN(n))||!referenceCameraPositions.every(n=>Number.isFinite(n)||Number.isNaN(n))||!sourceIndices.every((n,i)=>n===i)||!referenceEligible.every(n=>n===0||n===1))
      throw new Error('Invalid indexed measurement buffers.');
    // This maps pixels to source rows, never depth. -1 is absent; -2 retains ambiguous ownership in the full source buffers.
    scene.pixelRows=new Int32Array(c.width*c.height).fill(-1);
    let rendered=0,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<count;i++) {
      const p=referencePositions.subarray(i*3,i*3+3),
        xyz=referenceCameraPositions.subarray(i*3,i*3+3),
        uv=referenceUV.subarray(i*2,i*2+2),finite=xyz.every(Number.isFinite),forward=finite&&xyz[2]>0,
        eligible=forward&&uv.every(Number.isFinite)&&uv[0]>=0&&uv[0]<=c.width-1&&uv[1]>=0&&uv[1]<=c.height-1;
      if(finite?!p.every(Number.isFinite)||p.some((n,k)=>Math.abs(n-R[k].reduce((sum,value,j)=>sum+value*(xyz[j]-cameraOrigin[j]),0))>1e-9):!p.every(Number.isNaN))
        throw new Error('Indexed camera and world coordinates disagree.');
      if(!forward&&!uv.every(Number.isNaN))throw new Error('Unavailable indexed projection must retain NaN coordinates.');
      if(forward&&uv.some((n,k)=>{
        const projected=(k?c.fy*xyz[1]/xyz[2]+c.cy:c.fx*xyz[0]/xyz[2]+c.cx),
          tolerance=1e-9+256*Number.EPSILON*(Math.abs(n)+Math.abs(projected)+1);
        return !Number.isFinite(n)||!Number.isFinite(projected)||!Number.isFinite(tolerance)||Math.abs(n-projected)>tolerance;
      }))
        throw new Error('Indexed source coordinates disagree with calibrated projection.');
      if(referenceEligible[i]!==Number(eligible))throw new Error('Indexed eligibility disagrees with source projection.');
      if(!eligible)continue;
      if(p.some((n,k)=>scene.positions[rendered*3+k]!==Math.fround(n)||scene.positions[rendered*3+k]<scene.bounds.min[k]||scene.positions[rendered*3+k]>scene.bounds.max[k]))throw new Error('Rendered point lost indexed source identity or bounds.');
      for(let k=0;k<3;k++){const value=scene.positions[rendered*3+k];min[k]=Math.min(min[k],value);max[k]=Math.max(max[k],value);}
      rendered++;
      const pixel=Math.floor(uv[1]+.5)*c.width+Math.floor(uv[0]+.5);
      scene.pixelRows[pixel]=scene.pixelRows[pixel]===-1?i:-2;
    }
    if(rendered!==scene.pointCount||min.some((n,k)=>n!==scene.bounds.min[k])||max.some((n,k)=>n!==scene.bounds.max[k]))throw new Error('Indexed render population or bounds mismatch.');
    if(mapped) {
      // Any failure rejects the whole scene: no read-only fallback and no partial ground.
      const descriptor=measurement.ground,r=await fetchResource(base+descriptor.file);
      if(!r.ok)throw new Error(`Mapped ground unavailable (${r.status}).`);
      const bytes=await boundedBytes(r,descriptor.bytes,true,signal,deadline);
      if(await digest(bytes)!==descriptor.sha256)throw new Error('Mapped ground checksum mismatch.');
      let parsed;
      try{parsed=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));}catch{throw new Error('Invalid mapped ground: not UTF-8 JSON.');}
      validateGround(parsed,scene);
      const t=parsed.terrain,tr=await fetchResource(base+t.file);
      if(!tr.ok)throw new Error(`Mapped ground terrain unavailable (${tr.status}).`);
      const terrain=await boundedBytes(tr,t.bytes,true,signal,deadline);
      if(await digest(terrain)!==t.sha256)throw new Error('Mapped ground terrain checksum mismatch.');
      const heights=new Float32Array(terrain),finiteCells=heights.reduce((n,h)=>n+Number.isFinite(h),0);
      if(heights.some(h=>!Number.isFinite(h)&&!Number.isNaN(h))||finiteCells!==t.finiteCells)throw new Error('Invalid mapped ground terrain values.');
      scene.ground=parsed;scene.terrainHeights=heights;
      scene.groundSupportXZ=Float64Array.from(parsed.support.rows.flatMap(row=>[measurement.referencePositions[row*3],measurement.referencePositions[row*3+2]]));
    }
  }
  if(Object.hasOwn(scene,'reconstruction')) {
    const estimate=scene.reconstruction,hash=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);
    const provenanceFields=indexed?['baseMeasuredSceneSha256','preparedManifestSha256','calibrationSha256','referencesSha256','producerManifestSha256','planSha256','rawPredictionSha256','rawValiditySha256']:
      ['registeredPredictionSha256','producerManifestSha256','planSha256','observationsSha256','measuredSceneSha256'];
    if(!estimate||estimate.evidence!=='predicted'||estimate.role!=='rendererOnly'||
      (indexed&&estimate.kind!=='nativePoints')||
      estimate.model?.provider!=='nvidia'||typeof estimate.model?.id!=='string'||!estimate.model.id||
      typeof estimate.model?.revision!=='string'||!estimate.model.revision||!hash(estimate.model?.sha256)||
      !provenanceFields.every(k=>hash(estimate.provenance?.[k]))||
      !integer(estimate.pointCount)||estimate.pointCount>1_000_000||(indexed&&estimate.pointCount>scene.camera.width*scene.camera.height)||
      !Number.isSafeInteger(estimate.triangleCount)||estimate.triangleCount>2_000_000||(indexed?estimate.triangleCount!==0:estimate.triangleCount<=0)||!finite3(estimate.bounds?.min)||!finite3(estimate.bounds?.max)||
      estimate.bounds.max.some((n,i)=>n<estimate.bounds.min[i])||
      (!indexed&&(estimate.displayDomain?.maxDepthMeters!==scene.processing.maxDepthMeters||
      estimate.displayDomain?.meshEdgeLimitMeters!==scene.processing.meshEdgeLimitMeters)))
      throw new Error('Invalid estimated reconstruction metadata.');
    if(indexed) {
      const reuse=estimate.reuse,execution=reuse?.originalExecution,domain=estimate.displayDomain,provenance=estimate.provenance;
      const directBinding=provenance.inputBindingKind==='directNative';
      if(directBinding?(!hash(provenance.inputBindingResultSha256)||Object.hasOwn(provenance,'inputEquivalenceResultSha256')):
        (Object.hasOwn(provenance,'inputBindingKind')||Object.hasOwn(provenance,'inputBindingResultSha256')||!hash(provenance.inputEquivalenceResultSha256)))
        throw new Error('Invalid native reconstruction input binding.');
      if(estimate.provenance.preparedManifestSha256!==measurement.provenance.preparedManifestSha256||estimate.provenance.referencesSha256!==measurement.provenance.referencesSha256||
        reuse?.newInferenceRuns!==0||!hash(reuse.originalCacheKey)||typeof reuse.originalRecordBase!=='string'||!reuse.originalRecordBase||
        execution?.status!=='succeeded'||execution.provider!=='local_onnxruntime_cpu'||
        !integer(estimate.model.bytes)||execution.modelArtifact?.sha256!==estimate.model.sha256||execution.modelArtifact.bytes!==estimate.model.bytes||
        typeof execution.createdUtc!=='string'||!Number.isFinite(Date.parse(execution.createdUtc))||
        !Number.isFinite(execution.elapsedSeconds)||execution.elapsedSeconds<0||execution.elapsedSeconds>120||
        execution.usage?.cloudRequests!==0||execution.usage.jobs!==1||execution.usage.retries!==0||!recordedCpuConfiguration(execution)||
        domain?.policy!=='numericDepth_rightInputBounds_existingCameraClips'||domain.nearMeters!==scene.camera.near||domain.farMeters!==scene.camera.far||
        JSON.stringify(domain.rightPixelCenters)!==JSON.stringify([0,scene.camera.width-1]))
        throw new Error('Invalid native reconstruction provenance or rendering domain.');
      if(Object.hasOwn(reuse,'serviceExecution')) {
        // This is compiled service provenance, not authentication performed in the browser.
        // The backend collects Job control directly before publishing these pinned artifacts.
        const service=reuse.serviceExecution,artifacts=estimate.serviceArtifacts,
          same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),
          serviceHashes=['expectedJobSha256','runtimeContractSha256','bindingSha256','planSha256','producerSha256',
            'wrapperSha256','nativeManifestSha256','reviewSha256','jobControlReceiptSha256'],
          safeId=id=>typeof id==='string'&&/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(id),
          utc=value=>{
            if(typeof value!=='string')return null;
            const match=value.replace(/\+00:00$/,'Z').match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z$/);
            return match&&Number.isFinite(Date.parse(value))?match[1]+'.'+(match[2]??'').padEnd(9,'0'):null;
          };
        if(!directBinding||service?.schemaVersion!==1||service.claim!=='job_control_and_application_output_association'||
          service.provider!=='nebius_job'||!safeId(service.jobId)||!safeId(service.projectId)||
          typeof service.submissionId!=='string'||!/^[a-f0-9]{32}$/.test(service.submissionId)||
          typeof service.requestedImage!=='string'||!/^[a-z0-9][a-z0-9._:/-]*@sha256:[a-f0-9]{64}$/.test(service.requestedImage)||
          !serviceHashes.every(k=>hash(service[k]))||!hash(provenance.serviceExecutionSha256)||
          service.nativeManifestSha256!==provenance.preparedManifestSha256||service.planSha256!==provenance.planSha256||
          service.producerSha256!==provenance.producerManifestSha256||
          service.inputSha256?.cropManifest!==provenance.preparedManifestSha256||
          service.inputSha256?.['calibration.json']!==provenance.calibrationSha256||
          !['cropManifest','left.png','right.png','calibration.json'].every(k=>hash(service.inputSha256?.[k]))||
          !Array.isArray(service.inputTensorSha256)||service.inputTensorSha256.length!==2||!service.inputTensorSha256.every(hash)||
          !same(service.inputTensorSha256,execution.inputTensorSha256)||!execution.cpuConfiguration||
          !same(service.cpuConfiguration,execution.cpuConfiguration)||
          !same(service.environment,execution.environment)||
          typeof service.environment!=='object'||service.environment===null||
          Object.keys(service.environment).sort().join()!=='numpy,onnx,onnxruntime,pillow,python'||
          !Object.values(service.environment).every(v=>typeof v==='string'&&v.length>0&&v.length<=8192)||
          provenance.processingCodeHashes?.['cloud/run_stereo_job.py']!==service.wrapperSha256)
          throw new Error('Invalid native service provenance.');
        const times=service.timestamps,ordered=['jobCreatedAt','startedAt','cpuCreatedUtc','finishedAt','acquiredAt'].map(k=>utc(times?.[k]));
        if(ordered.some(t=>t===null)||ordered.some((t,i)=>i>0&&t<ordered[i-1])||times.cpuCreatedUtc!==execution.createdUtc)
          throw new Error('Invalid native service timestamps.');
        const names={context:'service-execution.json',jobControl:'job-control.json',outputBinding:'job-result.json'},records={};
        if(!artifacts||Object.keys(artifacts).sort().join()!=='context,jobControl,outputBinding')
          throw new Error('Invalid native service artifacts.');
        for(const [key,name] of Object.entries(names)) {
          const spec=artifacts[key];
          if(spec?.path!==name||!integer(spec.bytes)||spec.bytes>2*1024*1024||!hash(spec.sha256))
            throw new Error('Invalid native service artifact descriptor.');
          const response=await fetchResource(base+name);if(!response.ok)throw new Error('Native service artifact unavailable.');
          const bytes=await boundedBytes(response,spec.bytes,true,signal,deadline);
          if(await digest(bytes)!==spec.sha256)throw new Error('Native service artifact checksum mismatch.');
          records[key]=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));
        }
        const job=records.jobControl,binding=records.outputBinding;
        if(!job||typeof job!=='object'||Array.isArray(job)||!binding||typeof binding!=='object'||Array.isArray(binding))
          throw new Error('Invalid native service records.');
        if(artifacts.context.sha256!==provenance.serviceExecutionSha256||!same(records.context,service)||
          artifacts.jobControl.sha256!==service.jobControlReceiptSha256||artifacts.outputBinding.sha256!==service.bindingSha256||
          job.schemaVersion!==1||job.claim!=='job_control_only'||job.modelExecutionVerified!==false||job.outputsBound!==false||
          job.job?.metadata?.id!==service.jobId||job.job.metadata.parentId!==service.projectId||
          job.job.spec?.image!==service.requestedImage||!['COMPLETED',6].includes(job.job.status?.state)||
          job.source?.kind!=='nebius_job_get'||job.source.httpStatus!==200||job.source.request?.method!=='GET'||
          job.source.expectedSha256!==service.expectedJobSha256||
          job.source.request?.url!==`https://api.nebius.cloud/ai/v1/jobs/${service.jobId}`||
          job.job.metadata.createdAt!==times.jobCreatedAt||job.job.status.startedAt!==times.startedAt||
          job.job.status.finishedAt!==times.finishedAt||job.acquiredAt!==times.acquiredAt||
          binding.schemaVersion!==1||binding.claim!=='application_output_binding_only'||binding.submissionId!==service.submissionId||
          binding.planSha256!==service.planSha256||binding.producerSha256!==service.producerSha256||
          binding.wrapperSha256!==service.wrapperSha256||binding.modelSha256!==estimate.model.sha256||
          binding.cacheKey!==reuse.originalCacheKey||!same(binding.inputTensorSha256,service.inputTensorSha256)||
          binding.outputs?.['disparity.float32le']?.sha256!==provenance.rawPredictionSha256||
          binding.outputs?.['validity.uint8']?.sha256!==provenance.rawValiditySha256)
          throw new Error('Native service artifacts disagree with reconstruction provenance.');
      } else if(Object.hasOwn(estimate,'serviceArtifacts')||Object.hasOwn(provenance,'serviceExecutionSha256')) {
        throw new Error('Native service artifacts require explicit service provenance.');
      }
    }
    if(Object.keys(estimate.buffers??{}).sort().join()!=='colors,positions,triangles'||new Set(Object.values(estimate.buffers??{})).size!==3||
      Object.values(estimate.buffers??{}).some(name=>[...Object.values(scene.buffers),...Object.values(measurement?.buffers??{})].includes(name)))
      throw new Error('Estimated reconstruction must retain separate measured buffers.');
    await loadBuffers(estimate,{positions:Float32Array,colors:Float32Array,triangles:Uint32Array});
    if(estimate.positions.length!==estimate.pointCount*3||estimate.colors.length!==estimate.positions.length||
      estimate.triangles.length!==estimate.triangleCount*3||!estimate.positions.every(Number.isFinite)||
      !estimate.colors.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||!estimate.triangles.every(n=>n<estimate.pointCount)||
      !estimate.positions.every((n,i)=>n>=estimate.bounds.min[i%3]-2e-5&&n<=estimate.bounds.max[i%3]+2e-5)||!matchingBounds(estimate.positions,estimate.bounds))
      throw new Error('Invalid estimated reconstruction buffers.');
  }
  if(suggestions!==undefined) {
    // Prepared placement suggestions are a pinned scene asset: a bad, missing or stale file fails the load.
    if(suggestions===null||typeof suggestions!=='object'||Object.keys(suggestions).sort().join()!=='file,sha256'||
      typeof suggestions.file!=='string'||!/^[a-z][a-z0-9-]*\.json$/.test(suggestions.file)||!hash(suggestions.sha256))
      throw new Error('Invalid placement suggestions descriptor.');
    const r=await fetchResource(base+suggestions.file);if(!r.ok)throw new Error(`Placement suggestions unavailable (${r.status}).`);
    const bytes=await boundedBytes(r,64*1024,false,signal,deadline);
    if(await digest(bytes)!==suggestions.sha256)throw new Error('Placement suggestions checksum mismatch.');
    let raw;
    try{raw=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));}catch{throw new Error('Invalid placement suggestions: not UTF-8 JSON.');}
    validateSuggestions(raw,scene);
    const inputs=raw.inputs,metadataDigest=await digest(metadataBytes);
    if(mapped) {
      // Mapped suggestions name every pinned input they were computed from, including the frozen protocol.
      if(inputs['scene.json']!==metadataDigest||inputs['ground.json']!==measurement.ground.sha256||inputs['terrain.bin']!==scene.ground.terrain.sha256||
        inputs['recorded.png']!==scene.recordedImage.sha256||inputs['referencepositions.bin']!==measurement.artifactHashes.referencePositions||
        inputs['referenceeligible.bin']!==measurement.artifactHashes.referenceEligible||inputs['protocol.json']!==scene.ground.protocol.sha256)
        throw new Error('Placement suggestions were prepared from different scene inputs.');
    // recorded.jpg is a byte copy of the declared left source image (prepare-scene.py); this compares the file's
    // recorded input hash with scene.json's source hash, not the displayed image bytes themselves.
    }else if(inputs['scene.json']!==metadataDigest||inputs['depth.bin']!==scene.artifactHashes?.depth||
      (scene.sourceHashes?.left!==undefined&&inputs['recorded.jpg']!==scene.sourceHashes.left))
      throw new Error('Placement suggestions were prepared from different scene inputs.');
    scene.suggestions=raw;
  }
  if(mapped) {
    // car shape: the LiDAR shape exists exactly for the annotated stationary vehicle the suggestions name, by its track.
    const vehicle=scene.suggestions?.vehicle,shape=scene.ground.carShape;
    if((vehicle?.status==='annotated')!==(shape!==null))
      throw new Error('Invalid mapped ground: car shape must be present exactly when the suggestions name an annotated stationary vehicle.');
    if(shape!==null&&shape.track!==vehicle.track)throw new Error('Invalid mapped ground: car shape track differs from the suggested vehicle.');
    scene.carShape=prepareCarShape(shape);
  }
  checkActive(signal,deadline);
  return scene;
}

// Selecting a declared source is explicit; failed estimates never fall back during loading.
export function renderGeometry(scene) {
  return Object.hasOwn(scene,'reconstruction') ? scene.reconstruction : scene;
}

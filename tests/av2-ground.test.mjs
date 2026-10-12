import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {deflateSync,crc32} from 'node:zlib';
import {loadScene} from '../src/scene.mjs';
import {roadPoint,mappedGroundHit,mappedGroundHeight,placementFromPixels,placementPreview,createScenario,classifySightline,isMappedGround,roadTolerance,annotatedVehicle,comparisonStart,whatIfBox} from '../src/geometry.mjs';
import {planCamera} from '../src/plan-camera.mjs';
import * as THREE from 'three';
import {suggestedSetup,suggestionStatus,isStillSuggested,supportSummary} from '../src/suggestions.mjs';
import {scenarioFacts,supportFacts,runTool,setbackSweep,applyDeclaration,allowedNumbersFrom,checkAnswer,sweepPhrase,withheldNote} from '../src/scenario-assistant.mjs';
import {SYSTEM_PROMPTS,TOOLS} from '../worker/nemotron.js';
import {scene as legacyScene,input as legacyInput,fetchAsset} from './scene-input.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);
const close3=(a,b,tolerance,what)=>a.forEach((n,i)=>near(n,b[i],tolerance,what));
const r2=n=>Number(n.toFixed(2));

// ------------------------------------------------------------------ analytic fixture (never a runtime asset)
// Camera at the origin looking along -Z (fx = fy = 20, 8x10 pixels); flat mapped ground at world Y -1.4375 except a
// raised cell (step) and a no-value cell; one drivable rectangle |x| <= 3, -30 <= z <= 0; one annotated box at z -12.
function png(width,height) {
  const chunk=(type,data)=>{
    const body=Buffer.concat([Buffer.from(type),data]),size=Buffer.alloc(4),crc=Buffer.alloc(4);
    size.writeUInt32BE(data.length);crc.writeUInt32BE(crc32(body));return Buffer.concat([size,body,crc]);
  };
  const header=Buffer.alloc(13);header.writeUInt32BE(width,0);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(Buffer.alloc((width*3+1)*height))),chunk('IEND',Buffer.alloc(0))]);
}
const FLAT=8.5625,COLUMNS=25,ROWS=50;// cityZ; world Y = cityZ - 10
function analytic({scene:mutateScene=()=>{},ground:mutateGround=()=>{},terrain:mutateTerrain=()=>{}}={}) {
  const image=png(8,10),c={eye:[0,0,0],forward:[0,0,-1],up:[0,1,0],fx:20,fy:20,cx:2,cy:2,width:8,height:10,near:.01,far:100};
  const xyz=[[0,1.4375,10],[1,1.4375,20],[0,0,-10],[NaN,NaN,NaN]],
    uv=xyz.map(([x,y,z])=>z>0?[c.fx*x/z+c.cx,c.fy*y/z+c.cy]:[NaN,NaN]),
    eligible=uv.map(([u,v])=>Number(u>=0&&u<=7&&v>=0&&v<=9)),world=xyz.map(([x,y,z])=>[x,-y,-z]),draw=world.filter((_,i)=>eligible[i]),
    arrays={positions:new Float32Array(draw.flat()),colors:new Float32Array(draw.flat().map(()=>.5)),triangles:new Uint32Array(),
      sourceIndices:new Uint32Array(xyz.map((_,i)=>i)),referencePositions:new Float64Array(world.flat()),referenceCameraPositions:new Float64Array(xyz.flat()),
      referenceUV:new Float64Array(uv.flat()),referenceEligible:new Uint8Array(eligible)};
  const scene={id:'mapped-fixture',units:'meters',pointCount:draw.length,triangleCount:0,camera:c,
    transform:{worldFromCamera:[[1,0,0],[0,-1,0],[0,0,-1]],worldFromCity:[[1,0,0],[0,0,1],[0,-1,0]],cityOrigin:[101,201,10],cameraOrigin:[0,0,0],originRole:'leftCamera'},
    bounds:{min:draw[0].map((_,k)=>Math.min(...draw.map(p=>Math.fround(p[k])))),max:draw[0].map((_,k)=>Math.max(...draw.map(p=>Math.fround(p[k]))))},
    processing:{},buffers:{},artifactHashes:{},recordedImage:{file:'recorded.png',mimeType:'image/png',bytes:image.length,sha256:sha(image)},
    measurements:{kind:'indexedSparse',evidence:'geometricallyProjected',groundPolicy:'mappedGround',binaryEncoding:'littleEndian',pixelCellPolicy:'nearestIntegerCenterRejectCollisions',
      count:xyz.length,provenance:{preparedManifestSha256:'a'.repeat(64),referencesSha256:'b'.repeat(64),logId:'81d2b40a-c579-3e9c-b520-bee26cda947d',
        cameraTimestampNs:'123',lidarTimestampNs:'124',crossingId:'5'},buffers:{},artifactHashes:{}}};
  const heights=new Float32Array(COLUMNS*ROWS).fill(FLAT);
  heights[10*COLUMNS+10]=9;// raised cell: x in [0,1), z in (-6,-5], world Y -1
  heights[19*COLUMNS+8]=NaN;// no value: x in [-2,-1), z in (-15,-14]
  mutateTerrain(heights);
  const terrain=Buffer.from(heights.buffer),finiteCells=heights.reduce((n,h)=>n+Number.isFinite(h),0);
  const ground={schemaVersion:1,kind:'av2MappedGround',sceneId:scene.id,baseSceneSha256:'a'.repeat(64),
    protocol:{path:'../protocol-v2.json',sha256:'b'.repeat(64),bytes:1,readOnlyExternal:true},
    sources:Object.fromEntries(['preparedManifest','references','assembledScene','map','groundRaster','groundTransform','poses','annotations']
      .map((key,i)=>[key,{path:`../${key}.bin`,sha256:key==='assembledScene'?'a'.repeat(64):String(i).repeat(64),bytes:1,readOnlyExternal:true}])),
    frame:{cityOrigin:[101,201,10],worldFromCity:[[1,0,0],[0,0,1],[0,-1,0]],convention:'analytic'},
    roadToleranceMeters:.08,supportHeightAgreementMeters:.08,supportRadiusMeters:.5,maxRangeMeters:20,boundaryEpsilonMeters:.01,
    terrain:{file:'terrain.bin',bytes:terrain.length,sha256:sha(terrain),dtype:'float32',byteOrder:'little',layout:'row-major',columns:COLUMNS,rows:ROWS,column0:0,row0:0,
      rasterShape:[ROWS,COLUMNS],sim2:{R:[1,0,0,1],t:[-91,-196],s:1},cellMeters:1,finiteCells,nanCells:heights.length-finiteCells,values:'analytic',samplingRule:'analytic'},
    drivableAreas:[{id:'1',outline:[[-3,0],[3,0],[3,-30],[-3,-30]]}],
    crossing:{id:'5',edge1:[[-3,-1.4375,-14.375],[3,-1.4375,-14.375]],edge2:[[-3,-1.4375,-17],[3,-1.4375,-17]]},
    egoLane:{id:'7',laneType:'VEHICLE',leftMarkType:'SOLID_WHITE',rightMarkType:'SOLID_WHITE',leftBoundary:[[-1.5,-1.4375,0],[-1.5,-1.4375,-30]],rightBoundary:[[1.5,-1.4375,0],[1.5,-1.4375,-30]]},
    cuboids:{available:true,frame:'analytic',timestampNs:'124',items:[{category:'REGULAR_VEHICLE',center:[0,-.75,-12],height:1.5,length:1.5,motion:null,numInteriorPoints:1,
      track:'analytic',width:1.5,worldFromBox:[[1,0,0],[0,1,0],[0,0,1]]}]},
    support:{rows:[0],count:1,rule:'analytic',cameraEligibleRows:2,eligibleBehindCuboid:1,eligibleAgreeing:1,eligibleWithoutTerrainValue:0,cuboidOcclusionApplied:true,preparedLookupMismatches:0},
    processingCodeHashes:{'analytic.py':'d'.repeat(64)},dependencies:{numpy:'analytic'},
    license:{id:'CC-BY-NC-SA-4.0',attribution:'(c) analytic',source:'analytic',changes:'analytic'},limitations:['Analytic fixture only.']};
  mutateGround(ground);
  const bytes=new Map([['recorded.png',image],['terrain.bin',terrain]]);
  for(const [key,array] of Object.entries(arrays)) {
    const target=key.startsWith('reference')||key==='sourceIndices'?scene.measurements:scene,name=key.toLowerCase()+'.bin',buffer=Buffer.from(array.buffer,array.byteOffset,array.byteLength);
    target.buffers[key]=name;target.artifactHashes[key]=sha(buffer);bytes.set(name,buffer);
  }
  const groundBytes=Buffer.from(JSON.stringify(ground));bytes.set('ground.json',groundBytes);
  scene.measurements.ground={file:'ground.json',bytes:groundBytes.length,sha256:sha(groundBytes)};
  mutateScene(scene);
  const pin={file:'ground.json',sha256:sha(groundBytes)};
  return {scene,ground,bytes,pin,fetch:async path=>{
    const name=path.split('/').at(-1);
    return name==='scene.json'?new Response(JSON.stringify(scene)):bytes.has(name)?new Response(bytes.get(name)):new Response('',{status:404});
  }};
}
// The catalog pin defaults to the fixture's own; passing undefined explicitly means the catalog entry has none.
const loadAnalytic=(f,...pin)=>loadScene(f.fetch,f.scene.id,sha(JSON.stringify(f.scene)),undefined,undefined,pin.length?pin[0]:f.pin);

test('mapped ground: runtime rule returns the first drivable hit, support class, hidden flag and each Unknown reason',async()=>{
  const s=await loadAnalytic(analytic());
  assert.equal(isMappedGround(s),true);assert.equal(roadTolerance(s),.08);assert.equal(s.ground.license.id,'CC-BY-NC-SA-4.0');
  const observed=mappedGroundHit(s,[2,5]);
  close3(observed.point,[0,-1.4375,-28.75/3],1e-12);
  assert.deepEqual([observed.supportClass,observed.supportLabel,observed.hidden,observed.hitKind,observed.terrainCityZ,observed.terrainCell],
    ['observed-nearby','observed nearby',false,'top',FLAT,[10,14]]);
  // (0.08 m + half a float16 step at cityZ 8.5625) / sin(depression angle)
  near(observed.alongRayUncertaintyM,(.08+2**-8)/(.15/Math.hypot(.15,1)),1e-12);
  const hidden=mappedGroundHit(s,[2,4]);
  close3(hidden.point,[0,-1.4375,-14.375],1e-12);
  assert.deepEqual([hidden.supportClass,hidden.supportLabel,hidden.hidden],['map-only','mapped, not observed',true]);
  const step=mappedGroundHit(s,[2,7]);// enters the raised cell below its top
  close3(step.point,[0,-1.25,-5],1e-12);assert.equal(step.hitKind,'step');assert.equal(step.hidden,false);
  assert.deepEqual(roadPoint(s,[2,4],'crossing'),hidden.point);
  for(const [pixel,reason] of [[[2,1],'Pixel ray does not descend toward the ground.'],[[2,2],'Pixel ray does not descend toward the ground.'],
    [[2,3],'No mapped ground within 20 m along the pixel ray.'],[[0,4],'Ray crosses a cell without mapped ground height before reaching the ground.'],
    [[7,4],'First mapped-ground hit is outside the mapped drivable area.'],[[8,4],'Select a pixel inside the recorded image.'],[[2.5,4],'Select a pixel inside the recorded image.']])
    assert.throws(()=>mappedGroundHit(s,pixel),{message:reason},String(pixel));
  assert.equal(mappedGroundHeight(s,0,-10),-1.4375);assert.equal(mappedGroundHeight(s,-1.5,-14.5),null);
  // Without annotations the hidden flag is not evaluated (null), never false.
  const bare=await loadAnalytic(analytic({ground:g=>{g.cuboids.available=false;g.cuboids.items=[];g.sources.annotations=null;}}));
  assert.equal(mappedGroundHit(bare,[2,4]).hidden,null);
});

test('mapped ground: placement uses the mapped lane axis, the vehicle line and the drivable run; results stay Blocked or Unknown',async()=>{
  const s=await loadAnalytic(analytic()),pixels={crossing:[2,4],parking:[2,5],target:[3,5]};
  const p=placementFromPixels(s,pixels);
  assert.deepEqual(p.axis,[-0,0,1]);
  near(p.baselineSetback,14.375-28.75/3,1e-12);near(p.captureDistance,14.375,1e-12);
  near(p.availableSetback,14.375,1e-12,'run reaches the recorded camera');
  assert.deepEqual(p.support.crossing,{supportClass:'map-only',supportLabel:'mapped, not observed',hidden:true,alongRayUncertaintyM:mappedGroundHit(s,[2,4]).alongRayUncertaintyM});
  const input={pixels,vehicle:{width:1.9,height:1.5,length:4},targetHeight:1.1,proposedSetback:p.baselineSetback,driverDistance:p.captureDistance};
  const scenario=createScenario(s,input),front=p.parking;
  // Vehicle centre = front ground point + axis * length/2, Y = front Y + height/2; the person top is 1.1 m above its foot.
  close3(scenario.baseline.center,[front[0],front[1]+.75,front[2]+2],1e-12);
  close3(scenario.target,[p.target[0],p.target[1]+1.1,p.target[2]],1e-12);
  assert.equal(scenario.limits.maximumSetback,p.availableSetback-4);
  assert.equal(scenario.currentVisibility.state,'blocked');
  const low=createScenario(s,{...input,vehicle:{...input.vehicle,height:.5}});
  assert.deepEqual(low.currentVisibility,{state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null});
  assert.deepEqual(Object.keys(scenario.support),['crossing','parking','target']);
  close3(scenario.vehicleOrigin,[front[0],front[1],-14.375],1e-12);
  // Preview and comparison share the same vehicle line; the hidden target ground point is flagged for a hollow marker.
  const preview=placementPreview(s,pixels,input.vehicle,'1.1');
  close3(preview.vehicle.center,scenario.baseline.center,1e-12);assert.equal(preview.targetHidden,false);
  assert.equal(placementPreview(s,{target:[2,4]},null,'1').targetHidden,true);
  // Moving the proposed vehicle keeps its front height and line.
  const moved=createScenario(s,{...input,proposedSetback:8});
  close3(moved.proposed.center,[front[0],front[1]+.75,-14.375+8+2],1e-12);
  // A drivable area ending at z -3 ends the run there.
  const short=await loadAnalytic(analytic({ground:g=>{g.drivableAreas[0].outline=[[-3,-3],[3,-3],[3,-30],[-3,-30]];}}));
  near(placementFromPixels(short,pixels).availableSetback,11.375,1e-12);
  assert.throws(()=>createScenario(short,{...input,vehicle:{...input.vehicle,length:7}}),/beyond the captured road extent/);
  assert.throws(()=>placementFromPixels(s,{...pixels,parking:[2,4]}),/separated from the crossing/);
  // A front beyond the crossing asks for the camera side, in placement and in the preview.
  for(const run of [()=>placementFromPixels(s,{...pixels,crossing:[2,5],parking:[2,4]}),()=>placementPreview(s,{crossing:[2,5],parking:[2,4]},input.vehicle,'')])
    assert.throws(run,{message:'Place the current vehicle on the camera side of the crossing.'});
  assert.throws(()=>placementFromPixels(s,{...pixels,target:[7,4]}),/outside the mapped drivable area/);
  // Distance sweeps use the vehicle line with the same arithmetic as the comparison and never produce Clear.
  const sweep=setbackSweep(s,{pixels,driverDistance:p.captureDistance},{label:'van',width:1.9,height:1.5,length:4},1.1);
  assert.equal(sweep.range[0],.08);
  for(const row of sweep.rows) {
    assert.ok(['blocked','unknown','overlap'].includes(row.state));
    if(row.state!=='overlap')assert.equal(createScenario(s,{...input,proposedSetback:row.setback}).proposedVisibility.state,row.state);
  }
  // Every placed point's support reaches Nemotron's facts and the comparison's evidence key, suggested or not.
  const target=mappedGroundHit(s,[3,5]).supportClass;
  assert.deepEqual(scenarioFacts(s,scenario).support,{crossing:'map only',vehicle_front:'with nearby LiDAR returns',person_point:target==='map-only'?'map only':'with nearby LiDAR returns'});
  assert.deepEqual(supportFacts({crossing:null,parking:{supportClass:'map-only'}}),{vehicle_front:'map only'});
  const summary=supportSummary({crossing:{supportLabel:'mapped, not observed',hidden:true},parking:{supportLabel:'observed nearby',hidden:false},target:null});
  assert.deepEqual(summary,{text:'crossing: mapped, not observed · vehicle front: observed nearby',
    title:'crossing: mapped, not observed, pixel shows an annotated object, point behind it · vehicle front: observed nearby'});
  assert.deepEqual(supportSummary(undefined),{text:'',title:''});
  // Measured-depth scenarios carry no support facts.
  assert.equal(Object.hasOwn(scenarioFacts(legacyScene,{...scenario,support:undefined}),'support'),false);
  assert.equal(applyDeclaration(s,{crossing:[2,4]},null,{vehicle_label:'van',vehicle_width_m:2,vehicle_height_m:2.5,vehicle_length_m:5.5,target_label:'adult',target_height_m:1.5}).error,
    'Place the crossing, vehicle and person on the mapped ground first.');
});

test('mapped ground: the annotated stationary vehicle keeps its exact annotated pose until its front or size changes',async()=>{
  // Geometry-only fixture: the analytic cuboid moved beside the lane and turned 10 degrees, named by a minimal suggestion
  // record attached after loading (the loader's suggestion validation is covered by the AV2 8c tests).
  const yaw=10*Math.PI/180,sin=Math.sin(yaw),cos=Math.cos(yaw);
  const s=await loadAnalytic(analytic({ground:g=>{Object.assign(g.cuboids.items[0],{center:[2.2,-.6,-12],length:4,width:1.8,height:1.6,
    worldFromBox:[[sin,-cos,0],[0,0,1],[-cos,-sin,0]]});}}));
  s.suggestions={vehicle:{status:'annotated',track:'analytic',pixel:[2,5],dimensions:{width:1.8,height:1.6,length:4}}};
  const pixels={crossing:[2,4],parking:[2,5],target:[3,4]},fields={width:'1.8',height:'1.6',length:'4'};
  const pose=annotatedVehicle(s,pixels,fields);
  assert.deepEqual(pose.center,[2.2,-.6,-12]);assert.deepEqual(pose.dimensions,{width:1.8,height:1.6,length:4});
  close3(pose.axis,[-sin,0,cos],1e-15,'axis is the reversed horizontal box heading');
  const p=placementFromPixels(s,pixels),front=p.parking;
  // Setback of the annotated pose: lane axis (+z, toward the camera) distance from the crossing (z -14.375) to the centre
  // of the front face, centre z -12 minus half the length along the turned heading.
  const setback=14.375-12-2*cos,start=comparisonStart(s,pixels,fields);
  near(start.proposedSetback,setback,1e-12);assert.equal(start.driverDistance,p.captureDistance);
  assert.deepEqual(comparisonStart(s,pixels,fields,true),{proposedSetback:p.baselineSetback,driverDistance:p.captureDistance});
  const input={pixels,vehicle:fields,targetHeight:1.1,...start};
  const scenario=createScenario(s,input);
  assert.deepEqual({...scenario.baseline,axis:undefined},{center:[2.2,-.6,-12],axis:undefined,vehicle:{width:1.8,height:1.6,length:4},pose:'annotated'});
  // The what-if starts as the annotated box exactly, so both results are the same (this box lies beside the line: Unknown).
  assert.deepEqual(scenario.proposed,scenario.baseline);assert.equal(scenario.baselineSetback,start.proposedSetback);
  assert.equal(scenario.currentVisibility.state,'unknown');assert.deepEqual(scenario.proposedVisibility,scenario.currentVisibility);
  // Its own centre line (x 2.2 + 2 sin 10 deg) runs on mapped drivable ground to the camera; the turned box reaches
  // 4 cos 10 deg + 0.9 sin 10 deg (its rear corner) behind its front-face centre along the lane axis.
  near(scenario.limits.maximumSetback,14.375-4*cos-.9*sin,1e-12);
  const facts=scenarioFacts(s,scenario),annotated={pose:'annotated',turnedFromLane_deg:10,setback_m:.41,setback_ft:1.33,state:'unknown',targetInside:false,eyeInside:false};
  assert.deepEqual(facts.current,annotated);assert.deepEqual(facts.proposed,annotated);
  // Moving the what-if translates the annotated pose along the lane axis (same heading and size).
  const moved=createScenario(s,{...input,proposedSetback:setback+3});
  close3(moved.proposed.center,[2.2,-.6,-9],1e-12);assert.deepEqual(moved.proposed.axis,pose.axis);assert.deepEqual(moved.proposed.vehicle,pose.dimensions);
  assert.deepEqual(moved.baseline,scenario.baseline);assert.equal(scenarioFacts(s,moved).proposed.setback_m,r2(setback+3));
  // Turned 10 degrees, the nearer front corner sits 0.9 m x sin 10 deg nearer the crossing than the front-face centre.
  // The box's nearest point gets the declared front's rules: the slider and the distance check start where that corner
  // is the road tolerance (0.08 m) from the crossing, and a what-if whose corner would reach the crossing is refused.
  near(scenario.whatIf.cornerOffset,.9*sin,1e-12);near(scenario.limits.minimumSetback,.08+.9*sin,1e-12);
  const nearest=b=>Math.min(...[-1,1].flatMap(side=>[-1,1].map(end=>{
    const c=b.center.map((n,i)=>n+[b.axis[0],0,b.axis[2]][i]*end*b.vehicle.length/2+[b.axis[2],0,-b.axis[0]][i]*side*b.vehicle.width/2);
    return (c[0]-scenario.crossing[0])*scenario.axis[0]+(c[2]-scenario.crossing[2])*scenario.axis[2];})));
  near(nearest(whatIfBox(scenario.whatIf,scenario.limits.minimumSetback)),.08,1e-12);
  assert.throws(()=>createScenario(s,{...input,proposedSetback:.9*sin-1e-6}),{message:'Keep the proposed vehicle on the camera side of the crossing.'});
  assert.ok(nearest(createScenario(s,{...input,proposedSetback:.9*sin+1e-6}).proposed)>0);
  // The distance check sweeps the same translated pose; every row agrees with the comparison.
  const sweep=setbackSweep(s,{pixels,driverDistance:p.captureDistance},{label:'your vehicle',width:1.8,height:1.6,length:4},1.1);
  assert.deepEqual(sweep.range,[scenario.limits.minimumSetback,scenario.limits.maximumSetback]);
  for(const row of sweep.rows)assert.equal(createScenario(s,{...input,proposedSetback:row.setback}).proposedVisibility.state,row.state,`setback ${row.setback}`);
  // A turned current vehicle whose nearer corner is within the road tolerance of the crossing is refused, like a front.
  const close=await loadAnalytic(analytic({ground:g=>{Object.assign(g.cuboids.items[0],{center:[2.2,-.6,.15-14.375+2*cos],length:4,width:1.8,height:1.6,
    worldFromBox:[[sin,-cos,0],[0,0,1],[-cos,-sin,0]]});}}));
  close.suggestions=s.suggestions;
  assert.throws(()=>createScenario(close,{...input,proposedSetback:1}),{message:'Current vehicle front must be separated from the crossing.'});
  close3(placementPreview(s,pixels,fields,'1.1').vehicle.center,pose.center,0);
  // Moving the front pixel, changing a size or a declaration (even at the annotated sizes) makes both the current
  // vehicle and the what-if declared boxes on the lane axis at the front pixel.
  const lane={...input,proposedSetback:p.baselineSetback};
  for(const [name,next] of [['moved',{...lane,pixels:{...pixels,parking:[2,6]}}],['resized',{...lane,vehicle:{...fields,width:'1.9'}}],['declared',{...lane,vehicleDeclared:true}]]) {
    const declared=createScenario(s,next),q=placementFromPixels(s,next.pixels);
    if(name!=='declared')assert.equal(annotatedVehicle(s,next.pixels,next.vehicle),null,name);
    assert.equal(declared.baseline.pose,'declared',name);assert.equal(declared.proposed.pose,'declared',name);
    assert.deepEqual(declared.baseline.axis,declared.axis,name);assert.deepEqual(declared.proposed.axis,declared.axis,name);
    close3(declared.baseline.center,[q.parking[0],q.parking[1]+Number(next.vehicle.height)/2,q.parking[2]+2],1e-12,name);
    assert.equal(declared.baselineSetback,q.baselineSetback,name);
    assert.equal(declared.currentVisibility.state,'blocked',name);
    assert.equal(Object.hasOwn(scenarioFacts(s,declared).current,'pose'),false,name);
  }
  assert.equal(placementPreview(s,pixels,fields,'1.1',true).vehicle.annotated,undefined);
  assert.equal(setbackSweep(s,{pixels,driverDistance:p.captureDistance,vehicleDeclared:true},{label:'van',width:1.8,height:1.6,length:4},1.1).rows[0].state,'blocked');
  // A heading against the direction of travel, or turned more than 15 degrees from it, would make the camera-side end
  // the front face: refused explicitly, never measured from the wrong end and never a declared stand-in.
  for(const turn of [180,20]) {
    const t=turn*Math.PI/180,ts=Math.sin(t),tc=Math.cos(t);
    const turned=await loadAnalytic(analytic({ground:g=>{Object.assign(g.cuboids.items[0],{center:[2.2,-.6,-12],length:4,width:1.8,height:1.6,
      worldFromBox:[[ts,-tc,0],[0,0,1],[-tc,-ts,0]]});}}));
    turned.suggestions=s.suggestions;
    const facing={message:'The annotated vehicle does not face the direction of travel.'};
    assert.throws(()=>annotatedVehicle(turned,pixels,fields),facing,`${turn} deg`);
    assert.throws(()=>createScenario(turned,input),facing,`${turn} deg`);
  }
  // A suggestion naming a track the annotations do not hold is an error, never a declared stand-in.
  s.suggestions.vehicle.track='missing';
  assert.throws(()=>createScenario(s,input),/annotated vehicle is missing/);
});

test('mapped ground: descriptors, hashes and ground schema are pinned; tampering fails the load explicitly',async()=>{
  const f=analytic();
  await loadAnalytic(f);
  const descriptor=/Invalid mapped ground descriptor/;
  await assert.rejects(loadAnalytic(f,undefined),descriptor);
  await assert.rejects(loadAnalytic(f,{...f.pin,sha256:'f'.repeat(64)}),descriptor);
  await assert.rejects(loadAnalytic(f,{...f.pin,bytes:1}),descriptor);
  await assert.rejects(loadAnalytic(f,{file:'other.json',sha256:f.pin.sha256}),descriptor);
  for(const mutate of [s=>{s.measurements.ground.file='other.json';},s=>{s.measurements.ground.bytes=1024*1024+1;},s=>{s.measurements.ground.extra=1;},s=>{delete s.measurements.ground;}])
    await assert.rejects(loadAnalytic(analytic({scene:mutate})),descriptor);
  // Read-only scenes cannot carry or be handed a ground layer.
  const readOnly=analytic({scene:s=>{s.measurements.groundPolicy='unvalidated';delete s.measurements.ground;}});
  const unvalidated=await loadScene(readOnly.fetch,readOnly.scene.id,sha(JSON.stringify(readOnly.scene)));
  assert.equal(isMappedGround(unvalidated),false);assert.equal(unvalidated.ground,undefined);
  assert.throws(()=>roadPoint(unvalidated,[2,4],'crossing'),/no validated support/);
  await assert.rejects(loadAnalytic(readOnly,readOnly.pin),descriptor);
  await assert.rejects(loadAnalytic(analytic({scene:s=>{s.measurements.groundPolicy='unvalidated';}})),descriptor);
  await assert.rejects(loadScene(fetchAsset,legacyScene.id,undefined,undefined,undefined,{file:'ground.json',sha256:'a'.repeat(64)}),descriptor);
  // Served bytes that drift from the pins.
  let g=analytic();g.bytes.get('ground.json')[5]^=1;await assert.rejects(loadAnalytic(g),/Mapped ground checksum mismatch/);
  g=analytic();g.bytes.delete('ground.json');await assert.rejects(loadAnalytic(g),/Mapped ground unavailable \(404\)/);
  g=analytic();g.bytes.get('terrain.bin')[0]^=1;await assert.rejects(loadAnalytic(g),/terrain checksum mismatch/);
  g=analytic();g.bytes.set('terrain.bin',g.bytes.get('terrain.bin').subarray(4));await assert.rejects(loadAnalytic(g),/byte length mismatch/);
  g=analytic();g.bytes.delete('terrain.bin');await assert.rejects(loadAnalytic(g),/terrain unavailable \(404\)/);
  // Ground layers re-sealed with consistent hashes but invalid content.
  const invalid={
    'other scene':x=>{x.sceneId='other';},'city origin':x=>{x.frame.cityOrigin[0]+=1e-9;},'city orientation':x=>{x.frame.worldFromCity[2][1]=1;},
    'unexpected field':x=>{x.extra=1;},'missing field':x=>{delete x.egoLane;},'kind':x=>{x.kind='road';},'zero tolerance':x=>{x.roadToleranceMeters=0;},
    'tolerance in processing name':x=>{x.roadToleranceMeters='0.08';},'unbounded range':x=>{x.maxRangeMeters=1e9;},
    'support row not eligible':x=>{x.support.rows=[2];},'support rows unsorted':x=>{x.support.rows=[1,0];x.support.count=2;},'support count':x=>{x.support.count=2;},
    'support row out of range':x=>{x.support.rows=[4];},'cuboid rotation':x=>{x.cuboids.items[0].worldFromBox[0][0]=2;},
    'cuboid reflection':x=>{x.cuboids.items[0].worldFromBox[2][2]=-1;},'cuboid size':x=>{x.cuboids.items[0].length=0;},
    'unavailable annotations with items':x=>{x.cuboids.available=false;x.sources.annotations=null;},'license':x=>{x.license.id='CC-BY-4.0';},
    'crossing identity':x=>{x.crossing.id='6';},'drivable outline':x=>{x.drivableAreas[0].outline.length=2;},'no drivable area':x=>{x.drivableAreas=[];},
    'terrain file':x=>{x.terrain.file='../terrain.bin';},'terrain shape':x=>{x.terrain.rows=49;},'terrain window':x=>{x.terrain.row0=1;},
    'terrain sim2 scale in R':x=>{x.terrain.sim2.R=[2,0,0,2];},'terrain big endian':x=>{x.terrain.byteOrder='big';},
    'degenerate lane':x=>{x.egoLane.rightBoundary[1]=[1.5,-1.4375,0];x.egoLane.leftBoundary[1]=[-1.5,-1.4375,0];},
  };
  for(const [name,mutate] of Object.entries(invalid))await assert.rejects(loadAnalytic(analytic({ground:mutate})),/Invalid mapped ground: /,name);
  // Source records: the contract's exact eight, read-only and relative; only the annotations may be null (exactly when no
  // cuboids are available); the base scene hash is the assembled scene record's.
  const sources={'absolute source path':x=>{x.sources.map.path='/Users/private/map.json';},'null map source':x=>{x.sources.map=null;},
    'missing poses source':x=>{delete x.sources.poses;},'extra source':x=>{x.sources.extra=x.sources.map;},'writable source':x=>{x.sources.map.readOnlyExternal=false;},
    'annotations source missing while cuboids available':x=>{x.sources.annotations=null;},'annotations source without cuboids':x=>{x.cuboids.available=false;x.cuboids.items=[];},
    'base scene hash':x=>{x.baseSceneSha256='f'.repeat(64);}};
  for(const [name,mutate] of Object.entries(sources))await assert.rejects(loadAnalytic(analytic({ground:mutate})),{message:'Invalid mapped ground: source records.'},name);
  await assert.rejects(loadAnalytic(analytic({ground:x=>{x.terrain.finiteCells--;x.terrain.nanCells++;}})),/Invalid mapped ground terrain values/);
  await assert.rejects(loadAnalytic(analytic({terrain:h=>{h[0]=Infinity;}})),/Invalid mapped ground terrain values/);
  const raw=Buffer.from('{"schemaVersion":1');
  g=analytic({scene:s=>{s.measurements.ground={file:'ground.json',bytes:raw.length,sha256:sha(raw)};}});g.bytes.set('ground.json',raw);
  await assert.rejects(loadAnalytic(g,{file:'ground.json',sha256:sha(raw)}),/Invalid mapped ground: not UTF-8 JSON/);
});

test('Nemotron prompts and tool descriptions describe a generic declared vehicle',()=>{
  for(const prompt of Object.values(SYSTEM_PROMPTS)) {
    assert.doesNotMatch(prompt,/park/i);
    assert.match(prompt,/declared vehicle \(a box\)/);
    assert.ok(prompt.length<=1800,`${prompt.length} characters`);
  }
  assert.doesNotMatch(JSON.stringify(TOOLS),/park/i);
});

test('mapped ground: model-written answers and labels never call a vehicle parked; measured-depth scenes keep their wording',async()=>{
  const s=await loadAnalytic(analytic()),allowed=[.69,2.25];
  // Answers: withheld on a mapped-ground scene (the app's own line is shown instead), unchanged on a measured-depth scene.
  for(const text of ['The parked car at 0.69 m (2.25 ft) blocks the line.','Parking it 0.69 m back still blocks.','PARKED: blocked at 0.69 m.','A carpark van is blocked at 0.69 m.']) {
    const r=checkAnswer(text,allowed,{mapped:true});
    assert.deepEqual([r.ok,r.kind],[false,'wording'],text);assert.equal(r.text,undefined,text);
    assert.equal(withheldNote(r),'Nemotron\'s answer was withheld: used wording Sightline does not allow.');
    assert.equal(checkAnswer(text,allowed).ok,true,text);
  }
  assert.equal(checkAnswer('The stationary vehicle at 0.69 m (2.25 ft) blocks the line.',allowed,{mapped:true}).ok,true);
  // Labels: a declaration or distance check naming a vehicle or person with such a word is refused, so the model renames
  // it. The error never repeats the word (a tool error can reach the app's own line).
  const pixels={crossing:[2,4],parking:[2,5],target:[3,5]},w={pixels,vehicle:{width:'1.9',height:'1.5',length:'4'},targetHeight:'1.1',input:null,setupInput:null,run:{}};
  const declaration={vehicle_label:'van',vehicle_width_m:1.9,vehicle_height_m:1.5,vehicle_length_m:4,target_label:'adult',target_height_m:1.5};
  const refused=field=>`${field} must name only the type, like "SUV" or "child", with no status words.`;
  for(const [field,label] of [['vehicle_label','parked car'],['vehicle_label','Parked SUV'],['vehicle_label','parking truck'],['target_label','park ranger']]) {
    const r=runTool(s,'ask',w,'declare_scenario',{...declaration,[field]:label});
    assert.equal(r.error,refused(field),label);assert.doesNotMatch(r.error,/park/i);
  }
  const suv={label:'SUV',width_m:2,height_m:1.8,length_m:5};
  assert.equal(runTool(s,'check',w,'sweep_setbacks',{vehicles:[suv,{...suv,label:'parked SUV'}]}).error,refused('vehicles[1].label'));
  // A field name the model invents is never repeated on a mapped scene either.
  for(const [mode,tool,args] of [['ask','declare_scenario',{...declaration,parked_vehicle:true}],['check','sweep_setbacks',{vehicles:[suv],parked_vehicle:1}],
    ['check','sweep_setbacks',{vehicles:[{...suv,parking_side:'left'}]}]])assert.equal(runTool(s,mode,w,tool,args).error,'Unexpected field.',tool);
  const ok=runTool(s,'ask',w,'declare_scenario',declaration);
  assert.equal(ok.error,undefined);assert.equal(ok.parts[0].label,'van');
  // The measured-depth (DrivingStereo) scene keeps accepting these labels.
  const ds={pixels:legacyInput.pixels,vehicle:legacyInput.vehicle,targetHeight:legacyInput.targetHeight,input:legacyInput,setupInput:legacyInput,run:{}};
  assert.equal(runTool(legacyScene,'ask',ds,'declare_scenario',{...declaration,vehicle_label:'parked car'}).error,undefined);
  assert.equal(runTool(legacyScene,'check',ds,'sweep_setbacks',{vehicles:[{...suv,label:'parked SUV'}]}).error,undefined);
  assert.equal(runTool(legacyScene,'ask',ds,'declare_scenario',{...declaration,parked_vehicle:true}).error,'Unexpected field parked_vehicle.');
});

// ------------------------------------------------------------------ AV2 8c (protocol v2) local data
// The prepared AV2 scene is CC BY-NC-SA data kept out of public/ and dist/; it lives in ignored local evaluation folders.
const AV2=new URL('../.local/evaluation/av2-map-first-2026-10-11/',import.meta.url),ID='av2-pao-8c52d911-2658849';
const local=existsSync(new URL('grounded-v2/scene.json',AV2))&&existsSync(new URL('assembled/positions.bin',AV2));
const skip=local?false:'AV2 8c data is local-only and not present';
const GROUNDED=new Set(['scene.json','ground.json','terrain.bin','suggestions.json']);
const av2Bytes=name=>readFile(new URL((GROUNDED.has(name)?'grounded-v2/':'assembled/')+name,AV2));
// Serves the contract's directory: assembled buffers plus every grounded-v2 file, with optional replacements.
const av2Fetch=(replace={})=>async path=>{
  const name=path.split('/').at(-1);
  if(Object.hasOwn(replace,name))return replace[name]===null?new Response(null,{status:404}):new Response(replace[name]);
  if(!path.startsWith(`/scenes/${ID}/`)||['catalog-entry.json','LICENSE-AND-ATTRIBUTION.md'].includes(name))return new Response(null,{status:404});
  try{return new Response(await av2Bytes(name));}catch{return new Response(null,{status:404});}
};
const entry=local?JSON.parse(await readFile(new URL('grounded-v2/catalog-entry.json',AV2),'utf8')):null;
const loadAV2=(replace,overrides={})=>{const e={...entry,...overrides};return loadScene(av2Fetch(replace),ID,e.metadataSha256,undefined,e.suggestions,e.ground);};
// Contract section 8 (grounded-v2), asserted to 1e-6 m.
const EXPECTED={
  crossing:{pixel:[373,188],groundPoint:[7.653733590650668,-1.2992260826921616,16.150640359155755],hidden:true,alongRay:1.2111531809435245,range:17.919565014290743},
  parking:{pixel:[377,188],groundPoint:[7.21627686159278,-1.2367260826921618,15.40708808525625],hidden:true,alongRay:1.2112011139627483,range:17.058209355428126},
  target:{pixel:[486,186],groundPoint:[5.609009207266723,-1.2523510826921616,17.030180348951735],hidden:false,alongRay:1.2602869158469958,range:17.973770061935944},
};
const TRAVEL=[0.5833967978610233,0,0.8121872790468369];

test('AV2 8c: the contract pins load and the runtime rule reproduces every prepared vector',{skip},async()=>{
  assert.deepEqual(Object.keys(entry).sort(),['ground','id','label','metadataSha256','suggestions']);
  assert.equal(entry.metadataSha256,'c466441cb25ad84431a98a962dcb90b80dc12a26df9672d7f9f7fa45e14d825f');
  assert.deepEqual(entry.ground,{file:'ground.json',sha256:'6dc1a733c9ea37c89c50e58839b0ca41b6625848fec57134426c36bfe183acd7'});
  assert.deepEqual(entry.suggestions,{file:'suggestions.json',sha256:'94b294f3b7f7abd1e9fc58994e8e0a72f277b59c5e6490e3596157d05533d248'});
  const s=await loadAV2();
  assert.equal(s.measurements.groundPolicy,'mappedGround');assert.equal(s.ground.terrain.sha256,'2f7a2914ddfb89593a086514ad9ceab6c8f7eb25087e467cae4c4fc371b8f082');
  assert.equal(s.ground.support.count,3019);assert.equal(s.groundSupportXZ.length,2*3019);assert.equal(s.terrainHeights.length,409*408);
  assert.equal(s.ground.license.attribution,'(c) 2021 Argo AI, LLC');assert.equal(roadTolerance(s),.08);
  for(const [role,e] of Object.entries(EXPECTED)) {
    const hit=mappedGroundHit(s,e.pixel);
    close3(hit.point,e.groundPoint,1e-6,role);
    assert.deepEqual([hit.supportClass,hit.supportLabel,hit.hidden,hit.hitKind],['map-only','mapped, not observed',e.hidden,'top'],role);
    near(hit.alongRayUncertaintyM,e.alongRay,1e-6,role);near(hit.rangeM,e.range,1e-6,role);
  }
  const vectors=JSON.parse(await readFile(new URL('runtime-test-vectors-v2.json',AV2),'utf8'));
  assert.equal(vectors.groundSha256,entry.ground.sha256);assert.equal(vectors.vectors.length,16);
  for(const v of vectors.vectors) {
    if(v.supportClass==='unknown'){assert.throws(()=>mappedGroundHit(s,v.pixel),{message:v.reason},String(v.pixel));continue;}
    const hit=mappedGroundHit(s,v.pixel);
    close3(hit.point,v.groundPoint,1e-6,String(v.pixel));
    assert.deepEqual([hit.supportClass,hit.hidden,hit.hitKind,hit.terrainCityZ,hit.terrainCell],[v.supportClass,v.hidden,v.hitKind,v.terrainCityZ,v.terrainCell],String(v.pixel));
    near(hit.rangeM,v.rangeM,1e-6);near(hit.alongRayUncertaintyM,v.alongRayUncertaintyM,1e-6);
  }
});

test('AV2 8c: suggestions open the comparison with no clicks, in the annotated pose, labelled stationary and mapped, not observed',{skip},async()=>{
  const s=await loadAV2(),raw=s.suggestions,setup=suggestedSetup(s,raw);
  assert.equal(raw.schemaVersion,2);assert.equal(setup.error,'');
  assert.deepEqual(setup.pixels,{crossing:EXPECTED.crossing.pixel,parking:EXPECTED.parking.pixel,target:EXPECTED.target.pixel});
  assert.deepEqual(setup.vehicle,{width:'1.88',height:'1.49',length:'4.03'});assert.equal(setup.targetHeight,'1.1');
  assert.deepEqual(Object.fromEntries(Object.entries(setup.provenance).map(([k,p])=>[k,[p.status,p.supportLabel,p.hidden]])),
    {crossing:['mapped','mapped, not observed',true],parking:['annotated','mapped, not observed',true],target:['default','mapped, not observed',false],
      vehicle:['annotated',undefined,undefined],targetHeight:['default',undefined,undefined]});
  const scenario=createScenario(s,setup.input);
  close3(scenario.axis,TRAVEL.map(n=>-n),1e-12,'axis is the reversed ego-lane chord');
  // Setback of the annotated pose: lane-axis distance from the app crossing point to the centre of the box's front face.
  // It equals the prepared front-pixel gap plus the prepared front residual along the road axis (0.86 - 0.17 m).
  near(scenario.baselineSetback,.6865496210129542,1e-9);
  near(scenario.baselineSetback,raw.vehicle.pixelGroundGapAlongAxisM+raw.vehicle.frontResidualToAnnotatedM.alongRoadAxis,1e-9);
  assert.equal(setup.input.proposedSetback,scenario.baselineSetback);assert.equal(scenario.proposedSetback,scenario.baselineSetback);
  near(scenario.limits.captureDistance,raw.crossing.forwardM,1e-9);
  // The mapped drivable run along the box line reaches the recorded camera; the turned box's lane-axis extent behind its
  // front-face centre (4.03 m x cos plus its rear corner, 0.94 m x sin of the 1.41 deg turn) stays inside it.
  const turnCos=Math.abs(scenario.baseline.axis[0]*scenario.axis[0]+scenario.baseline.axis[2]*scenario.axis[2]);
  near(scenario.limits.maximumSetback,raw.crossing.forwardM-4.03*turnCos-scenario.whatIf.cornerOffset,1e-9,'maximum setback');
  // The stationary vehicle is annotated cuboid 396f15b5 exactly (centre, heading, dimensions), and the proposed what-if
  // starts as that same box.
  const box=s.ground.cuboids.items.find(b=>b.track===raw.vehicle.track);
  assert.equal(raw.vehicle.track,'396f15b5-9cf3-462b-9164-4b6f71c4e7d6');assert.equal(scenario.baseline.pose,'annotated');
  assert.deepEqual(scenario.baseline.center,box.center);assert.deepEqual(scenario.baseline.vehicle,{width:box.width,height:box.height,length:box.length});
  close3(scenario.baseline.axis,raw.vehicle.headingWorld.map(n=>-n),1e-12,'axis is the reversed annotated heading');
  assert.deepEqual(scenario.proposed,scenario.baseline);
  close3(placementPreview(s,setup.pixels,setup.vehicle,setup.targetHeight).vehicle.center,box.center,0);
  // First-load headline: the annotated box misses the driver-eye-to-head line, so Stationary Unknown and Proposed Unknown
  // (the same box; never Clear).
  const unknown={state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null};
  assert.deepEqual(scenario.currentVisibility,unknown);assert.deepEqual(scenario.proposedVisibility,unknown);
  // The slider translates the annotated pose along the lane axis: same heading and size, centre moved by the change.
  const back=createScenario(s,{...setup.input,proposedSetback:scenario.baselineSetback+1});
  close3(back.proposed.center,box.center.map((n,i)=>n+scenario.axis[i]),1e-12);
  assert.deepEqual([back.proposed.axis,back.proposed.vehicle,back.proposed.pose],[scenario.baseline.axis,scenario.baseline.vehicle,'annotated']);
  assert.deepEqual(back.baseline,scenario.baseline);assert.equal(back.proposedVisibility.state,'blocked');
  assert.deepEqual(whatIfBox(scenario.whatIf,scenario.baselineSetback+1),back.proposed);
  // The plan view frames the current box and the what-if over its whole movement, without moving as it is dragged.
  const far=createScenario(s,{...setup.input,proposedSetback:scenario.limits.maximumSetback});
  const inView=(camera,b,what)=>{for(const side of [-1,1])for(const end of [-1,1]) {
    const along=[b.axis[0],0,b.axis[2]],across=[b.axis[2],0,-b.axis[0]];
    const corner=b.center.map((n,i)=>n+along[i]*end*b.vehicle.length/2+across[i]*side*b.vehicle.width/2),q=new THREE.Vector3(...corner).project(camera);
    assert.ok(Math.abs(q.x)<.9&&Math.abs(q.y)<.9,`${what}: corner at ${q.x}, ${q.y}`);
  }};
  for(const aspect of [3.8,1,.65]) {
    const camera=planCamera(s,scenario,aspect);
    assert.deepEqual(camera.projectionMatrix.elements,planCamera(s,far,aspect).projectionMatrix.elements);
    for(const b of [scenario.baseline,far.proposed])inView(camera,b,`aspect ${aspect}`);
  }
  // A declared box runs along the lane axis through its own front, which can sit far to the side of the crossing
  // (review repro: crossing [288,150], front [640,150]); the plan view frames that box over its whole movement.
  const sideways={...setup.input,pixels:{...setup.pixels,crossing:[288,150],parking:[640,150]},vehicle:{width:'1.9',height:'1.5',length:'4'}};
  const sp=placementFromPixels(s,sideways.pixels),wide=createScenario(s,{...sideways,proposedSetback:sp.baselineSetback,driverDistance:sp.captureDistance});
  assert.equal(wide.baseline.pose,'declared');
  for(const aspect of [3.8,1,.65]) {
    const camera=planCamera(s,wide,aspect);
    for(const setback of [wide.baselineSetback,wide.limits.minimumSetback,wide.limits.maximumSetback])inView(camera,whatIfBox(wide.whatIf,setback),`sideways ${setback} at aspect ${aspect}`);
  }
  close3(scenario.target,[EXPECTED.target.groundPoint[0],EXPECTED.target.groundPoint[1]+1.1,EXPECTED.target.groundPoint[2]],1e-9);
  close3(scenario.crossing,EXPECTED.crossing.groundPoint,1e-9);
  for(const v of [scenario.currentVisibility,scenario.proposedVisibility])assert.ok(['blocked','unknown'].includes(v.state));
  assert.deepEqual(Object.fromEntries(Object.entries(scenario.support).map(([k,v])=>[k,[v.supportClass,v.hidden]])),
    {crossing:['map-only',true],parking:['map-only',true],target:['map-only',false]});
  const current={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight},status=suggestionStatus(raw,setup,current);
  assert.deepEqual(status.parts.map(p=>`${p.text} ${p.note}`),['crosswalk mapped ±1.21 m (mapped, not observed; hidden)','stationary vehicle (mapped, not observed; hidden)','child 1.1 m (default; mapped, not observed)']);
  assert.match(status.parts[1].title,/^stationary vehicle \(AV2 annotation\); reason for stopping unknown.* · annotated pose$/);
  assert.match(status.parts[0].title,/paint offset unmeasured/);
  assert.deepEqual(status.context,{crossing:'mapped crosswalk edge, map only',vehicle_front:'stationary vehicle from the dataset annotation; reason for stopping unknown',
    vehicle_size:'annotated stationary vehicle size',person_point:'declared default, map only',person_height:'declared default child'});
  assert.doesNotMatch(JSON.stringify([status.parts.map(p=>[p.text,p.note,p.title]),status.context]),/park/i);
  assert.ok(Object.values(status.context).every(v=>!/\d/.test(v)),'Context strings carry no numbers.');
  // A resized stationary vehicle, or one a declaration replaced at exactly the annotated sizes, is a declared box at
  // the annotated front pixel: current and what-if on the lane axis, in the context too.
  const resized=suggestionStatus(raw,setup,{...current,vehicle:{...setup.vehicle,width:'1.9'}});
  assert.equal(resized.context.vehicle_front,'declared vehicle, map only');assert.equal(resized.context.vehicle_size,undefined);
  const declaredStatus=suggestionStatus(raw,setup,{...current,overridden:['vehicle']});
  assert.equal(declaredStatus.context.vehicle_front,'declared vehicle, map only');assert.equal(declaredStatus.context.vehicle_size,undefined);
  assert.deepEqual(declaredStatus.parts.map(p=>p.key),['crossing','target']);
  assert.doesNotMatch(JSON.stringify([resized,declaredStatus]),/park|annotated pose/i);
  const lane={...setup.input,proposedSetback:raw.vehicle.pixelGroundGapAlongAxisM};
  for(const [name,next] of [['resized',{...lane,vehicle:{...setup.vehicle,width:'1.9'}}],['moved',{...lane,pixels:{...setup.pixels,parking:[377,189]}}],['declared',{...lane,vehicleDeclared:true}]]) {
    const declared=createScenario(s,next);
    assert.deepEqual([declared.baseline.pose,declared.proposed.pose],['declared','declared'],name);
    assert.deepEqual(declared.baseline.axis,declared.axis,name);assert.deepEqual(declared.proposed.axis,declared.axis,name);
    if(name!=='moved')near(declared.baselineSetback,.8591145531026565,1e-9,name);
    assert.deepEqual([declared.currentVisibility.state,declared.proposedVisibility.state],['blocked','blocked'],name);
  }
  assert.deepEqual(comparisonStart(s,setup.pixels,setup.vehicle,true).proposedSetback,raw.vehicle.pixelGroundGapAlongAxisM);
  // Editing loses the label; the moved crossing removes its part.
  assert.equal(isStillSuggested('parking',[377,189],setup.provenance),false);
  const moved=suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,crossing:[373,189]}});
  assert.equal(moved.edited,true);assert.deepEqual(moved.parts.map(p=>p.key),['parking','target']);
});

test('AV2 8c: Ask and Check work on first load with the suggested setup',{skip},async()=>{
  const s=await loadAV2(),setup=suggestedSetup(s,s.suggestions),scenario=createScenario(s,setup.input);
  const facts=scenarioFacts(s,scenario);
  assert.equal(facts.scene,ID);assert.ok(facts.eyeHeight_m>1&&facts.eyeHeight_m<2,`eye height ${facts.eyeHeight_m}`);
  assert.deepEqual(facts.vehicle_m,{width:1.88,height:1.49,length:4.03});assert.equal(facts.targetHeight_m,1.1);
  // The facts describe the annotated pose (turned 1.41 deg from the lane), its front-face setback and its result; the
  // what-if starts as the same box. No margin is given.
  const pose={pose:'annotated',turnedFromLane_deg:1.41,setback_m:.69,setback_ft:2.25,state:'unknown',targetInside:false,eyeInside:false};
  assert.deepEqual(facts.current,pose);assert.deepEqual(facts.proposed,pose);
  const w={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight,input:setup.input,setupInput:setup.input,run:{},vehicleLabel:'stationary vehicle'};
  const check=runTool(s,'check',w,'sweep_setbacks',{vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]});
  assert.equal(check.error,undefined);
  assert.deepEqual(check.result.vehicles.map(v=>v.label),['your vehicle','SUV']);
  for(const r of check.sweep.results)for(const row of r.rows)assert.ok(['blocked','unknown','overlap'].includes(row.state));
  // 'your vehicle' sweeps the annotated pose moved along the lane axis (front-face setbacks). The box is turned 1.41 deg,
  // so its nearer front corner is 0.94 m x sin 1.41 deg = 0.023 m nearer the crossing than the front-face centre: rows
  // start at 0.103 m, where that corner is the road tolerance (0.08 m) from the crossing. Not blocked by the vehicle at
  // 0.10-0.60 m, blocked 0.85-10.35 m (both marks), not blocked by the vehicle from 10.60 m. The SUV is a declared box.
  const [own,suv]=check.sweep.results;
  const [a,l]=[scenario.baseline.axis,scenario.axis];
  near(scenario.whatIf.cornerOffset,scenario.baseline.vehicle.width/2*Math.abs(a[0]*l[2]-a[2]*l[0]),1e-12);near(scenario.whatIf.cornerOffset,.94*Math.sin(1.4071*Math.PI/180),1e-4,'contract yaw');
  near(scenario.limits.minimumSetback,.08+scenario.whatIf.cornerOffset,1e-12);
  assert.deepEqual(own.range,[scenario.limits.minimumSetback,scenario.limits.maximumSetback]);
  assert.deepEqual(own.rows.slice(0,4).map(r=>[r2(r.setback),r.state]),[[.1,'unknown'],[.35,'unknown'],[.6,'unknown'],[.85,'blocked']]);
  assert.equal(sweepPhrase(own),'blocked 0.85-10.35 m; not blocked by the vehicle from 10.6 m (34.79 ft)');
  assert.deepEqual([own.at15ft,own.at20ft,own.stopsBlockingAt],['blocked','blocked',own.rows.find(r=>r2(r.setback)===10.6).setback]);
  assert.deepEqual(check.result.vehicles[0].blocked_m,[[.85,10.35]]);assert.deepEqual(check.result.vehicles[0].tested_m,[.1,13.53]);
  // Below the corner offset the what-if's corner would reach the crossing: refused, never drawn past it.
  assert.throws(()=>createScenario(s,{...setup.input,proposedSetback:.02}),{message:'Keep the proposed vehicle on the camera side of the crossing.'});
  assert.equal(createScenario(s,{...setup.input,proposedSetback:.05}).proposed.pose,'annotated');
  assert.equal(sweepPhrase(suv),'blocked 0.08-10.58 m; not blocked by the vehicle from 10.83 m (35.53 ft)');
  for(const row of check.sweep.results[0].rows.filter(r=>r.state!=='overlap'))
    assert.equal(createScenario(s,{...setup.input,proposedSetback:row.setback}).proposedVisibility.state,row.state,`setback ${row.setback}`);
  assert.ok(allowedNumbersFrom(facts,check.result).length>0);
  // Support for every placed point, in words the answer filter allows; an edited point keeps its support entry.
  assert.deepEqual(facts.support,{crossing:'map only',vehicle_front:'map only',person_point:'map only'});
  for(const v of Object.values(facts.support))assert.equal(checkAnswer(v).ok,true,v);
  const edited={...setup.input,pixels:{...setup.pixels,target:[486,190]}},hit=mappedGroundHit(s,[486,190]);
  assert.equal(scenarioFacts(s,createScenario(s,edited)).support.person_point,hit.supportClass==='map-only'?'map only':'with nearby LiDAR returns');
  assert.deepEqual(supportSummary(scenario.support),{text:'crossing, vehicle front, person: mapped, not observed',
    title:'crossing: mapped, not observed, pixel shows an annotated object, point behind it · vehicle front: mapped, not observed, pixel shows an annotated object, point behind it · person: mapped, not observed'});
  assert.doesNotMatch(JSON.stringify([facts,supportSummary(scenario.support)]),/park/i);
  // A crossing moved past the vehicle front asks for the camera side (browser repro: crossing at [373,200]).
  assert.throws(()=>placementFromPixels(s,{...setup.pixels,crossing:[373,200]}),{message:'Place the current vehicle on the camera side of the crossing.'});
  const ask=runTool(s,'ask',{...w,vehicleLabel:'stationary vehicle'},'declare_scenario',
    {vehicle_label:'box truck',vehicle_width_m:2.5,vehicle_height_m:3.5,vehicle_length_m:8,target_label:'child',target_height_m:1.1,setback_value:20,setback_unit:'ft'});
  assert.equal(ask.error,undefined);assert.equal(ask.input.proposedSetback,20*.3048);assert.equal(ask.input.vehicleDeclared,true);
  assert.deepEqual(ask.result,scenarioFacts(s,createScenario(s,ask.input)));
  // A declaration at exactly the annotated sizes still replaces the stationary vehicle: declared boxes on the lane axis
  // from the front pixel, in the facts and in a later distance check.
  const same=runTool(s,'ask',{...w,run:{}},'declare_scenario',
    {vehicle_label:'stationary vehicle',vehicle_width_m:1.88,vehicle_height_m:1.49,vehicle_length_m:4.03,target_label:'child',target_height_m:1.1});
  // Same sizes, but the distance results were swept in the annotated pose, so they are cleared.
  assert.equal(same.error,undefined);assert.equal(same.input.vehicleDeclared,true);assert.equal(same.clearSweep,true);
  assert.equal(check.sweep.vehicleDeclared,false);
  assert.deepEqual(same.result.current,{setback_m:.86,setback_ft:2.82,state:'blocked',targetInside:false,eyeInside:false});
  // The kept 0.69 m what-if is now a declared lane-axis box (no pose field), which blocks there.
  assert.deepEqual(same.result.proposed,{setback_m:.69,setback_ft:2.25,state:'blocked',targetInside:false,eyeInside:false});
  const after=runTool(s,'check',{...w,input:same.input},'sweep_setbacks',{vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]});
  assert.equal(sweepPhrase(after.sweep.results[0]),'blocked 0.83-10.58 m; not blocked by the vehicle from 10.83 m (35.53 ft)');
  assert.equal(after.sweep.vehicleDeclared,true);
  // A further declaration over a declared box at the same sizes keeps those (declared) distance results.
  const again=runTool(s,'ask',{...w,input:same.input,run:{}},'declare_scenario',
    {vehicle_label:'stationary vehicle',vehicle_width_m:1.88,vehicle_height_m:1.49,vehicle_length_m:4.03,target_label:'child',target_height_m:1.1});
  assert.equal(again.clearSweep,false);
  assert.equal(applyDeclaration(s,setup.pixels,null,{vehicle_label:'van',vehicle_width_m:2,vehicle_height_m:2.5,vehicle_length_m:5.5,target_label:'adult',target_height_m:1.5}).error,undefined);
  assert.equal(classifySightline(s,scenario.eye,scenario.target,scenario.vehicle,scenario.axis,[1000,0,1000]).state,'unknown');
});

test('AV2 8c: stale, tampered or reworded suggestions and ground fail the load',{skip},async()=>{
  const original=await av2Bytes('suggestions.json'),raw=JSON.parse(original.toString('utf8'));
  const reseal=mutate=>{const next=structuredClone(raw);mutate(next);const bytes=Buffer.from(JSON.stringify(next));return [{'suggestions.json':bytes},{suggestions:{file:'suggestions.json',sha256:sha(bytes)}}];};
  const cases={
    'moved pixel':[r=>{r.crossing.pixel=[373,189];r.crossing.pixelSearch.offsetPx=[0,0];},/crossing disagrees with the runtime mapped-ground rule/],
    'hidden flipped':[r=>{r.vehicle.hidden=false;},/vehicle disagrees/],
    'support class promoted':[r=>{r.person.supportClass='observed-nearby';r.person.supportLabel='observed nearby';},/person disagrees/],
    'support label':[r=>{r.person.supportLabel='observed nearby';},/supportClass and supportLabel must match/],
    'ground point drift':[r=>{r.person.groundPoint[2]+=2e-6;},/person disagrees/],
    'pixel search offset':[r=>{r.vehicle.pixelSearch.offsetPx=[0,-2];},/pixelSearch/],
    'annotated track':[r=>{r.vehicle.track='00000000-0000-0000-0000-000000000000';},/must match exactly one annotated cuboid/],
    'annotated size':[r=>{r.vehicle.dimensions.width=1.89;},/must match exactly one annotated cuboid/],
    'annotated heading':[r=>{r.vehicle.headingWorld=[0.6,0,0.8];},/must match exactly one annotated cuboid/],
    'annotated motion':[r=>{r.vehicle.motion=null;},/must match exactly one annotated cuboid/],
    'banned wording':[r=>{r.vehicle.reason='parked vehicle';},/banned vehicle-status wording/],
    'banned label':[r=>{r.limitations.push('Not parking safety.');},/banned vehicle-status wording/],
    'schema 1':[r=>{r.schemaVersion=1;},/schemaVersion must be 2/],
    'extra field':[r=>{r.crossCheck={status:'ran'};},/unexpected field file\.crossCheck/],
    'tolerance':[r=>{r.tolerances.roadToleranceMeters=.3;},/tolerances must equal/],
    'road axis':[r=>{r.roadAxis.directionOfTravel=[0.6,0,0.8];},/roadAxis must be the mapped ego lane chord/],
    'license':[r=>{r.license.id='CC-BY-4.0';},/license must equal/],
    'protocol':[r=>{r.protocolSha256='0'.repeat(64);},/name the protocol used/],
    'scene input':[r=>{r.inputs['scene.json']='0'.repeat(64);},/different scene inputs/],
    'ground input':[r=>{r.inputs['ground.json']='0'.repeat(64);},/different scene inputs/],
    'protocol input':[r=>{r.inputs['protocol.json']=r.protocolSha256='0'.repeat(64);},/different scene inputs/],
    'other scene':[r=>{r.sceneId='other';},/sceneId/],
    'unknown with stray pixel only':[r=>{r.person={status:'unknown',supportClass:'unknown',reason:'No curb proxy.',pixel:[486,186],surprise:1};},/unexpected field person\.surprise/],
  };
  for(const [name,[mutate,pattern]] of Object.entries(cases)){const [replace,overrides]=reseal(mutate);await assert.rejects(loadAV2(replace,overrides),pattern,name);}
  // An unknown person stays unplaced with its reason.
  const [replace,overrides]=reseal(r=>{r.person={status:'unknown',label:'child',supportClass:'unknown',reason:'No curb proxy on the near edge.'};});
  const s=await loadAV2(replace,overrides),setup=suggestedSetup(s,s.suggestions);
  assert.deepEqual(setup.provenance.target,{status:'unplaced',reason:'No curb proxy on the near edge.'});assert.equal(setup.input,null);
  // Bytes that drift from the catalog pin, a missing ground pin or a tampered ground layer.
  await assert.rejects(loadAV2({'suggestions.json':Buffer.concat([original,Buffer.from(' ')])}),/Placement suggestions checksum mismatch/);
  await assert.rejects(loadAV2({},{ground:undefined}),/Invalid mapped ground descriptor/);
  const ground=await av2Bytes('ground.json');ground[100]^=1;
  await assert.rejects(loadAV2({'ground.json':ground}),/Mapped ground checksum mismatch/);
  const terrain=await av2Bytes('terrain.bin');terrain[4]^=1;
  await assert.rejects(loadAV2({'terrain.bin':terrain}),/Mapped ground terrain checksum mismatch/);
  await assert.rejects(loadAV2({'terrain.bin':null}),/Mapped ground terrain unavailable \(404\)/);
  // A schema 2 file cannot be applied to a measured-depth scene.
  const bytes=Buffer.from(JSON.stringify(raw));
  await assert.rejects(loadScene(async path=>path.endsWith('/suggestions.json')?new Response(bytes):fetchAsset(path),legacyScene.id,undefined,undefined,{file:'suggestions.json',sha256:sha(bytes)}),
    /Invalid placement suggestions: /);
});

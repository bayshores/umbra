import assert from 'node:assert/strict';
import test from 'node:test';
import { measuredPoint, placementPreview, createScenario, segmentIntersectsBox, classifySightline, SIZE_LIMITS,
  mappedGroundHit,mappedGroundHeight,mappedGroundReferenceHeight } from '../src/geometry.mjs';
import { loadScene } from '../src/scene.mjs';
import { scene,input,fetchAsset } from './scene-input.mjs';
import {analytic,loadAnalytic} from './mapped-fixture.mjs';
import {suggestionConsistency} from '../scripts/evaluation/pao-baseline.mjs';
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
  assert.throws(()=>placementPreview(scene,input.pixels,{...input.vehicle,width:1e6},input.targetHeight),{message:'Vehicle width must be a number from 0.5 to 3.5 m.'});
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
test('implausible or malformed sizes and setbacks below the slider minimum are refused, never read or clamped',()=>{
  // Sizes: SIZE_LIMITS (the Nemotron tool ranges), typed numbers as plain decimals only.
  assert.deepEqual(SIZE_LIMITS,{width:[.5,3.5],height:[.5,4.5],length:[1,20],targetHeight:[.3,2.5]});
  const refused=(patch,message)=>assert.throws(()=>createScenario(scene,{...input,...patch}),{message},JSON.stringify(patch));
  for(const width of ['50',50,'0x2','0.4',' 2','2 ','+2','-2','1,9','',null,NaN,Infinity,true])refused({vehicle:{...input.vehicle,width}},'Vehicle width must be a number from 0.5 to 3.5 m.');
  refused({vehicle:{...input.vehicle,height:'50'}},'Vehicle height must be a number from 0.5 to 4.5 m.');
  refused({vehicle:{...input.vehicle,length:.01}},'Vehicle length must be a number from 1 to 20 m.');
  for(const targetHeight of [150,'100','0x1',.2])refused({targetHeight},'Target height must be a number from 0.3 to 2.5 m.');
  for(const width of ['1.9','.50','3.5',.5,'1.','2e0'])assert.doesNotThrow(()=>createScenario(scene,{...input,vehicle:{...input.vehicle,width}}),String(width));
  // Setbacks: plain positive decimals, at least the slider minimum (the road tolerance here).
  const {limits}=createScenario(scene,input);
  assert.equal(limits.minimumSetback,scene.processing.roadToleranceMeters);
  for(const proposedSetback of ['0x2','-1',0])refused({proposedSetback},'Proposed setback must be a positive number.');
  for(const proposedSetback of [.01,.05,limits.minimumSetback-1e-6])refused({proposedSetback},'Proposed setback must be at least 0.08 m from the crossing (the slider minimum).');
  for(const proposedSetback of [limits.minimumSetback,Number(limits.minimumSetback.toPrecision(15)),String(limits.minimumSetback)])
    assert.equal(createScenario(scene,{...input,proposedSetback}).proposedSetback,limits.minimumSetback);
  assert.throws(()=>placementPreview(scene,input.pixels,{...input.vehicle,length:'0x2'},input.targetHeight),{message:'Vehicle length must be a number from 1 to 20 m.'});
  assert.throws(()=>placementPreview(scene,input.pixels,input.vehicle,'150'),{message:'Target height must be a number from 0.3 to 2.5 m.'});
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

function measuredGrid(ground,slope=[.01,.02]) {
  const columns=21,rows=61,originXZ=[-5,-30],cellMeters=.5,
    heights=Array.from({length:columns*rows},(_,i)=>-2+slope[0]*(originXZ[0]+i%columns*cellMeters)+slope[1]*(originXZ[1]+Math.floor(i/columns)*cellMeters)),
    sweeps=['1000000000','2000000000','3000000000','4000000000'].map(timestampNs=>({timestampNs,sha256:'e'.repeat(64)}));
  ground.schemaVersion=4;ground.support={rows:[],count:0};
  ground.alongRayUncertainty={method:'supportSweepHoldout',heightBandMeters:null,rule:'analytic independent integer-click errors',quantile:.975,minimumCount:100,
    bins:[{minRangeM:0,maxRangeM:30,meters:.3,count:100}]};
  ground.measuredGround={version:'measured-ground-v1',columns,rows,originXZ,cellMeters,heights,heightBounds:[Math.min(...heights),Math.max(...heights)],
    partition:{frameTimestampNs:'124',excludedNearNs:250000000,fit:sweeps.slice(0,3),calibration:sweeps.slice(3)}};
}
const projectMeasured=(s,p)=>{
  const m=s.transform.worldFromCamera,q=[0,1,2].map(k=>p[0]*m[0][k]+p[1]*m[1][k]+p[2]*m[2][k]),c=s.camera;
  return [c.fx*q[0]/q[2]+c.cx,c.fy*q[1]/q[2]+c.cy];
};
test('measured ground: sloped continuous triangles hit exactly across grid cells and diagonals',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;}}));
  for(const [x,z] of [[0,-12],[-.27,-14.43],[.23,-16.71],[-.75,-19.25],[1,-20],[-1.21,-18.37]]) {
    const y=-2+.01*x+.02*z,hit=mappedGroundHit(s,projectMeasured(s,[x,y,z]));
    assert.ok(Math.hypot(hit.point[0]-x,hit.point[1]-y,hit.point[2]-z)<1e-6);
    assert.ok(Math.abs(mappedGroundHeight(s,x,z)-y)<1e-12);
    assert.equal(hit.hitKind,'top');assert.equal(hit.alongRayUncertaintyM,.3);
  }
});
test('measured ground: an unmeasured gap returns Unknown while the reference raster remains available',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;
    for(let row=20;row<30;row++)for(let col=8;col<=12;col++)g.measuredGround.heights[row*21+col]=null;
  }}));
  assert.equal(mappedGroundHeight(s,0,-18),null);
  assert.notEqual(mappedGroundReferenceHeight(s,0,-18),null);
  assert.throws(()=>mappedGroundHit(s,projectMeasured(s,[0,-2,-18])),/No independently measured ground support/);
});
test('measured ground: empty support never falls back to mapped terrain',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g);g.measuredGround.heights.fill(null);g.measuredGround.heightBounds=null;
    g.measuredGround.partition.fit=[];g.measuredGround.partition.calibration=[];g.alongRayUncertainty.bins=[];}}));
  assert.throws(()=>mappedGroundHit(s,[2,5]),/No independently measured ground support/);
  assert.equal(mappedGroundHeight(s,0,-10),null);
});
test('measured ground: independent bands above the gate are displayed unchanged and missing calibration refuses',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.alongRayUncertainty.bins[0].meters=.91;}}));
  assert.equal(mappedGroundHit(s,[2,6]).alongRayUncertaintyM,.91);
  s.ground.alongRayUncertainty.bins=[];
  assert.throws(()=>mappedGroundHit(s,[2,6]),/No independent click-error calibration/);
});
test('measured ground: loader refuses scoring-frame leakage, calibration overlap and false height bounds',async()=>{
  for(const change of [
    g=>g.measuredGround.partition.fit[0].timestampNs='124',
    g=>g.measuredGround.partition.calibration[0].timestampNs='1000000000',
    g=>g.measuredGround.heightBounds[0]-=1,
    g=>{g.support={rows:[0],count:1};},
  ])await assert.rejects(loadAnalytic(analytic({ground:g=>{measuredGrid(g);change(g);}})),/Invalid mapped ground/);
});
test('measured ground: legacy layer hit and reference height keep their existing exact values',async()=>{
  const s=await loadAnalytic(analytic()),hit=mappedGroundHit(s,[2,5]);
  assert.deepEqual(hit.point,[0,-1.4375000000000002,-9.583333333333336]);
  assert.equal(mappedGroundHeight(s,0,-10),mappedGroundReferenceHeight(s,0,-10));
  assert.equal(hit.hitKind,'top');assert.deepEqual(hit.terrainCell,[10,14]);
});
test('measured ground: moving a declared actor stops at measured holes despite a complete publisher raster',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;
    for(let row=47;row<=53;row++)for(let column=0;column<21;column++)g.measuredGround.heights[row*21+column]=null;
  }}));
  const setup={pixels:{crossing:[2,4],parking:[2,6],target:[4,4]},vehicle:{width:1,height:1.5,length:2},targetHeight:1.1,proposedSetback:10,driverDistance:20};
  const baseline=createScenario(s,setup);
  assert.ok(baseline.limits.maximumSetback<14);
  assert.equal(mappedGroundHeight(s,0,-5),null);
  assert.throws(()=>createScenario(s,{...setup,proposedSetback:14}),/setback|extent|between|outside|range/i);
});
test('measured ground: C preserves a valid fractional hit and reports a rounded diagnostic in a hole as Unknown',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;
    for(let column=0;column<21;column++)g.measuredGround.heights[20*21+column]=null;
  }})),point=[0,-2,-18.7],uv=projectMeasured(s,point);
  s.suggestions={roadAxis:{directionOfTravel:[0,0,-1]},crossing:{status:'mapped',mappedPoint:point,pixelUV:uv,pixel:uv.map(n=>Math.floor(n+.5)),pixelOffsetFromMappedM:0},
    vehicle:{status:'unknown',reason:'analytic'},person:{status:'unknown',reason:'analytic'}};
  const result=suggestionConsistency(s).roles.crossing;
  assert.equal(result.metricC.pointUsedMet,true);assert.equal(result.metricC.pixelUVRayHitMet,true);
  assert.equal(result.integerAnchorAlone.status,'unknown');assert.match(result.integerAnchorAlone.reason,/measured ground support/);
});

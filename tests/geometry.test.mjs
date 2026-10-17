import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import { measuredPoint, placementPreview, createScenario, segmentIntersectsBox, classifySightline, SIZE_LIMITS,
  mappedGroundHit,mappedGroundHeight,mappedGroundReferenceHeight,measuredGroundBand,measuredGroundTier,anchorPoint,comparisonStart } from '../src/geometry.mjs';
import { loadScene } from '../src/scene.mjs';
import { scene,input,fetchAsset } from './scene-input.mjs';
import {analytic,loadAnalytic,sha,TIER0_FILE} from './mapped-fixture.mjs';
import {suggestionConsistency,clickToGround} from '../scripts/evaluation/pao-baseline.mjs';
import {suggestedSetup} from '../src/suggestions.mjs';
import {mapSurface} from '../src/map-surface.mjs';
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
    sweeps=['1000000000','2000000000','3000000000','4000000000'].map((timestampNs,i)=>({timestampNs,sha256:String(i+1).repeat(64)}));
  ground.schemaVersion=4;ground.support={rows:[],count:0};
  ground.alongRayUncertainty={method:'supportSweepHoldout',heightBandMeters:null,rule:'analytic independent integer-click errors',quantile:.975,minimumCount:100,
    bins:[{minRangeM:0,maxRangeM:30,meters:.3,count:100}],coverage:{eligible:100,placed:100,unknown:{}}};
  ground.measuredGround={version:'measured-ground-v1',columns,rows,originXZ,cellMeters,heights,heightBounds:[Math.min(...heights),Math.max(...heights)],
    partition:{frameTimestampNs:'124',excludedNearNs:250000000,fit:sweeps.slice(0,3),calibration:sweeps.slice(3)}};
}
// v1 changes the tier-0 (v1) layer before it is saved as tier-0-ground.json; the v2 layer starts as its exact copy.
function measuredGridV2(g,v1=()=>{}) {
  measuredGrid(g,[0,0]);
  const edges=[0,10,12,15,18,20,22,25,30],bins=meters=>edges.slice(1).map((maxRangeM,i)=>({minRangeM:edges[i],maxRangeM,meters,count:100}));
  g.alongRayUncertainty.bins=bins(.3);v1(g);
  const tier0=Buffer.from(JSON.stringify(g)),s=g.measuredGround,p=s.partition,old={fit:p.fit,calibration:p.calibration},
    extra={fit:[{timestampNs:'5000000000',sha256:'5'.repeat(64)}],calibration:[{timestampNs:'6000000000',sha256:'6'.repeat(64)}]};
  Object.assign(s,{version:'measured-ground-v2',tiers:s.heights.map(h=>h===null?null:0),tier0:{groundSha256:sha(tier0),path:'../tier-0'}});
  Object.assign(p,{tiers:[old,extra],fit:[...old.fit,...extra.fit],calibration:[...old.calibration,...extra.calibration],excluded:[]});
  const band=g.alongRayUncertainty;
  Object.assign(band,{bins:[],tiers:[{bins:band.bins,coverage:band.coverage},{bins:bins(.9),coverage:{eligible:100,placed:100,unknown:{}}}]});
  g[TIER0_FILE]=tier0;
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
test('measured ground: continuous calibrated range widths interpolate without clipping and invalid functions refuse loading',async()=>{
  const linear=g=>{measuredGrid(g,[0,0]);Object.assign(g.alongRayUncertainty,{rangeRule:'linear',bins:[],knots:[
    {rangeM:0,meters:.2,count:100},{rangeM:20,meters:.8,count:100},{rangeM:30,meters:1.6,count:100}]});};
  const s=await loadAnalytic(analytic({ground:linear}));
  assert.equal(measuredGroundBand(s,0),.2);assert.equal(measuredGroundBand(s,20),.8);assert.equal(measuredGroundBand(s,30),1.6);
  assert.ok(Math.abs(measuredGroundBand(s,25)-1.2)<1e-12);assert.equal(measuredGroundBand(s,30.01),null);
  for(const change of [g=>g.alongRayUncertainty.knots[1].rangeM=0,g=>g.alongRayUncertainty.knots[0].meters=-1,
    g=>g.alongRayUncertainty.knots.pop()&&g.alongRayUncertainty.knots.pop(),g=>g.measuredGround.partition.calibration=[]])
    await assert.rejects(loadAnalytic(analytic({ground:g=>{linear(g);change(g);}})),/Invalid mapped ground/);
});
test('measured ground: exact calibration permits a zero width without a fabricated floor',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.alongRayUncertainty.bins[0].meters=0;}}));
  assert.equal(mappedGroundHit(s,[2,6]).alongRayUncertaintyM,0);
  const linear=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);Object.assign(g.alongRayUncertainty,{rangeRule:'linear',bins:[],
    knots:[{rangeM:0,meters:0,count:100},{rangeM:30,meters:0,count:100}]});}}));
  assert.equal(mappedGroundHit(linear,[2,6]).alongRayUncertaintyM,0);
});
test('measured ground: loader refuses scoring-frame leakage, calibration overlap and false height bounds',async()=>{
  for(const change of [
    g=>g.measuredGround.partition.fit[0].timestampNs='124',
    g=>g.measuredGround.partition.calibration[0].timestampNs='1000000000',
    g=>g.measuredGround.partition.calibration[0].timestampNs='01000000000',
    g=>g.measuredGround.partition.calibration[0].sha256=g.measuredGround.partition.fit[0].sha256,
    g=>g.measuredGround.partition.calibration=[],
    g=>g.alongRayUncertainty.coverage.placed=99,
    g=>g.measuredGround.heightBounds[0]-=1,
    g=>{g.support={rows:[0],count:1};},
  ])await assert.rejects(loadAnalytic(analytic({ground:g=>{measuredGrid(g);change(g);}})),/Invalid mapped ground/);
});
test('measured ground: loader requires a valid scoring sweep identity',async()=>{
  for(const change of [p=>delete p.originalSources,p=>delete p.originalSources.lidar,p=>delete p.originalSources.lidar.sha256,
    p=>p.originalSources.lidar.sha256='E'.repeat(64),p=>p.originalSources.lidar.sha256='e'.repeat(63),p=>p.originalSources.lidar.sha256=null]) {
    const fixture=analytic({ground:g=>measuredGrid(g)});change(fixture.scene.measurements.provenance);
    await assert.rejects(loadAnalytic(fixture),/Invalid mapped ground: measured support needs the scoring sweep sha256/);
  }
  await loadAnalytic(analytic({ground:g=>measuredGrid(g)}));
});
test('measured ground: loader refuses a renamed copy of the known scoring sweep',async()=>{
  const fixture=analytic({ground:g=>measuredGrid(g)});
  fixture.scene.measurements.provenance.originalSources={lidar:{sha256:'1'.repeat(64)}};
  await assert.rejects(loadAnalytic(fixture),/Invalid mapped ground/);
});
test('measured ground: grid entry inside the possible height envelope stays Unknown',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;}}));
  const grid=s.ground.measuredGround;
  grid.originXZ=[-5,-30];grid.rows=31;grid.heights=Array(grid.columns*grid.rows).fill(-2);
  grid.heights[0]=-3;grid.heights[1]=-1;grid.heightBounds=[-3,-1];
  assert.throws(()=>mappedGroundHit(s,projectMeasured(s,[0,-2,-20])),/No independently measured ground support/);
  grid.heights.fill(-2);grid.heightBounds=[-2,-2];
  assert.ok(Math.abs(mappedGroundHit(s,projectMeasured(s,[0,-2,-20])).point[2]+20)<1e-6);
});
test('measured ground: origins below ground and rays approaching a back face are refused',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;}}));
  const grid=s.ground.measuredGround;
  grid.rows=62;grid.heights=Array.from({length:grid.columns*grid.rows},(_,i)=>1+.2*(grid.originXZ[1]+Math.floor(i/grid.columns)*grid.cellMeters));grid.heightBounds=[-5,1.1];
  assert.throws(()=>mappedGroundHit(s,projectMeasured(s,[0,-1,-10])),/Camera origin is not above/);
  for(let row=59;row<grid.rows;row++)for(let col=0;col<grid.columns;col++)grid.heights[row*grid.columns+col]=null;
  assert.throws(()=>mappedGroundHit(s,projectMeasured(s,[0,-1,-10])),/No independently measured ground support/);
});
test('measured ground: legacy layer hit and reference height keep their existing exact values',async()=>{
  const s=await loadAnalytic(analytic()),hit=mappedGroundHit(s,[2,5]);
  assert.deepEqual(hit.point,[0,-1.4375000000000002,-9.583333333333336]);
  assert.equal(mappedGroundHeight(s,0,-10),mappedGroundReferenceHeight(s,0,-10));
  assert.equal(hit.hitKind,'top');assert.deepEqual(hit.terrainCell,[10,14]);
});
test('measured ground: vehicle movement crosses holes while placements there stay Unknown',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;
    for(let row=47;row<=53;row++)for(let column=0;column<21;column++)g.measuredGround.heights[row*21+column]=null;
  }}));
  const setup={pixels:{crossing:[2,4],parking:[2,6],target:[4,4]},vehicle:{width:1,height:1.5,length:2},targetHeight:1.1,proposedSetback:10,driverDistance:20};
  const baseline=createScenario(s,setup);
  assert.ok(Math.abs(baseline.limits.maximumSetback-18)<1e-8);
  assert.equal(mappedGroundHeight(s,0,-5),null);
  assert.throws(()=>mappedGroundHit(s,projectMeasured(s,[0,-2,-6])),/No independently measured ground support/);
  for(const proposedSetback of [baseline.limits.minimumSetback,10,14,baseline.limits.maximumSetback]) {
    const moved=createScenario(s,{...setup,proposedSetback});assert.equal(moved.whatIf.groundHeight,-2);
  }
  s.ground.drivableAreas[0].outline=[[-3,-3],[3,-3],[3,-30],[-3,-30]];
  assert.ok(Math.abs(createScenario(s,setup).limits.maximumSetback-15)<1e-8);
  s.ground.measuredGround.rows=51;s.ground.measuredGround.heights.length=51*21;
  assert.ok(Math.abs(createScenario(s,setup).limits.maximumSetback-13)<1e-8);
});
test('measured ground: schema 3 movement still stops at a publisher-raster hole',async()=>{
  const setup={pixels:{crossing:[2,4],parking:[6,6],target:[4,4]},vehicle:{width:1,height:1.5,length:2},targetHeight:1.1,proposedSetback:4,driverDistance:14.375};
  const make=hole=>loadAnalytic(analytic({terrain:h=>{if(hole)h[8*25+11]=NaN;}}));
  const holeScene=await make(true),full=createScenario(await make(false),setup),hole=createScenario(holeScene,setup);
  assert.ok(hole.limits.maximumSetback<full.limits.maximumSetback);
  assert.throws(()=>createScenario(holeScene,{...setup,proposedSetback:full.limits.maximumSetback}),/captured road extent/);
});
test('measured ground v2: hits use the highest vertex tier and its own band and label',async()=>{
  const s=await loadAnalytic(analytic({ground:measuredGridV2})),p=[.12,-2,-13.71],uv=projectMeasured(s,p);
  assert.equal(mappedGroundHit(s,uv).tier,0);assert.equal(mappedGroundHit(s,uv).supportLabel,'observed nearby');
  const grid=s.ground.measuredGround,column=Math.floor((p[0]+5)/.5),row=Math.floor((p[2]+30)/.5);
  grid.tiers[row*grid.columns+column+grid.columns+1]=1;
  const hit=mappedGroundHit(s,uv);
  assert.equal(hit.tier,1);assert.equal(hit.alongRayUncertaintyM,.9);
  assert.equal(hit.supportLabel,'measured at another moment of the same recording');
  assert.equal(measuredGroundTier(s,p[0],p[2]),1);
  assert.equal(measuredGroundBand(s,13,0),.3);assert.equal(measuredGroundBand(s,13,1),.9);
  s.ground.alongRayUncertainty.tiers[1].bins=[];
  assert.equal(mappedGroundHeight(s,p[0],p[2]),-2);
  assert.throws(()=>mappedGroundHit(s,uv),/No click-error calibration for measured ground tier 1 at this range/);
});
test('measured ground v2: malformed tiers, partitions and calibration records refuse loading',async()=>{
  for(const change of [
    g=>g.measuredGround.tiers.pop(),g=>g.measuredGround.tiers[0]=2,g=>g.measuredGround.tiers[0]=null,
    g=>g.measuredGround.heights[0]=null,
    g=>delete g.measuredGround.partition.tiers,
    g=>g.measuredGround.partition.tiers[1].fit.push(g.measuredGround.partition.tiers[0].calibration[0]),
    g=>g.measuredGround.partition.tiers[1].calibration[0].sha256=g.measuredGround.partition.fit[0].sha256,
    g=>g.measuredGround.partition.excluded.push({timestampNs:'7000000000',sha256:'e'.repeat(64)}),
    g=>g.measuredGround.partition.excluded.push({...g.measuredGround.partition.fit[0]}),
    g=>g.measuredGround.partition.tiers[1].fit[0].timestampNs='124',
    g=>g.measuredGround.partition.tiers[1].fit[0].sha256='e'.repeat(64),
    g=>g.alongRayUncertainty.tiers.pop(),g=>delete g.alongRayUncertainty.tiers[1].bins,
    g=>g.alongRayUncertainty.tiers[1].bins[0].meters=-1,
    g=>g.alongRayUncertainty.tiers[1].coverage.placed=99,
    g=>g.alongRayUncertainty.tiers[1].bins[0].minRangeM=1,
  ])await assert.rejects(loadAnalytic(analytic({ground:g=>{measuredGridV2(g);change(g);}})),/Invalid mapped ground/);
  for(const change of [p=>delete p.originalSources,p=>delete p.originalSources.lidar.sha256]) {
    const f=analytic({ground:measuredGridV2});change(f.scene.measurements.provenance);
    await assert.rejects(loadAnalytic(f),/measured support needs the scoring sweep sha256/);
  }
});
test('measured ground v2: tier 0 must be the pinned v1 layer, and tier 1 may fill only its holes',async()=>{
  // Vertex k is a v1 hole that v2 fills from other sweeps (tier 1); vertex j is a v1 height (tier 0).
  const k=31*21+10,j=31*21+11,filled=g=>{measuredGridV2(g,v1=>{v1.measuredGround.heights[k]=null;});g.measuredGround.heights[k]=-2;g.measuredGround.tiers[k]=1;};
  const load=change=>loadAnalytic(analytic({ground:g=>{filled(g);change(g);}}));
  const s=await load(()=>{});
  assert.equal(measuredGroundTier(s,0,.5*31-30),1);assert.equal(measuredGroundTier(s,2,.5*31-30),0);
  for(const change of [
    g=>g.measuredGround.tiers[k]=0,// the review's relabel: a tier-1 vertex claiming tier 0 and its narrower band
    g=>g.measuredGround.tiers[j]=1,
    g=>{g.measuredGround.heights[j]=-1.999;g.measuredGround.heightBounds=[-2,-1.999];},
  ])await assert.rejects(load(change),/tier 0 must keep every v1 height exactly, and tier-1 heights may fill only v1 holes/);
  for(const change of [
    g=>g.alongRayUncertainty.tiers[0].bins[3].meters=.2,
    g=>g.alongRayUncertainty.tiers[0].coverage.unknown.hole=0,
    g=>{const p=g.measuredGround.partition;p.tiers[0].fit.reverse();p.fit=p.tiers.flatMap(t=>t.fit);},
  ])await assert.rejects(load(change),/tier 0 must keep the v1 sweep partition and calibration band/);
  for(const change of [g=>g.measuredGround.originXZ=[-5.5,-30],g=>g.drivableAreas[0].outline[0]=[-3.5,0],g=>g.measuredGround.parameters={minimumReturns:11}])
    await assert.rejects(load(change),/tier 0 must be a measured-ground-v1 layer of the same scene, grid and fit/);
  await assert.rejects(load(g=>delete g.measuredGround.tier0),/measured vertex tiers must match supported heights/);
  await assert.rejects(load(g=>g.measuredGround.tier0.groundSha256='f'.repeat(64)),/Measured ground tier 0 checksum mismatch/);
  await assert.rejects(load(g=>delete g[TIER0_FILE]),/Measured ground tier 0 unavailable \(404\)/);
  await assert.rejects(load(g=>{const v2=Buffer.from(JSON.stringify(g));g[TIER0_FILE]=v2;g.measuredGround.tier0.groundSha256=sha(v2);}),/Invalid mapped ground/);
  // A valid schema-3 (publisher raster) layer of the same scene is not a tier 0 either (re-review minor: it threw a TypeError).
  await assert.rejects(loadAnalytic(analytic({ground:g=>{const s3=Buffer.from(JSON.stringify(g));filled(g);g[TIER0_FILE]=s3;g.measuredGround.tier0.groundSha256=sha(s3);}})),
    /tier 0 must be a measured-ground-v1 layer of the same scene, grid and fit/);
});
test('measured ground v2: map triangles and sampled heights share the same holes and diagonal',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>measuredGridV2(g,v1=>{v1.measuredGround.heights[31*21+10]=null;})})),mesh=mapSurface(s);
  for(let i=0;i<mesh.triangles.length;i+=3) {
    const p=[0,1,2].map(k=>[0,1,2].reduce((sum,j)=>sum+mesh.positions[3*mesh.triangles[i+j]+k],0)/3);
    assert.ok(Math.abs(mappedGroundHeight(s,p[0],p[2])-p[1])<1e-8);
  }
  assert.equal(mappedGroundHeight(s,.01,-14.49),null);
});
test('suggested setup names every unplaced role and its Unknown reason',async()=>{
  const s=await loadAnalytic(analytic({ground:measuredGridV2}));
  const suggestions={schemaVersion:3,crossing:{status:'unknown',reason:'No measured ground support at the source anchor.'},
    vehicle:{status:'unknown',reason:'Front ground remains hidden.'},person:{status:'unknown',reason:'No calibrated band.'}};
  const setup=suggestedSetup(s,suggestions);
  assert.equal(setup.input,null);
  assert.equal(setup.error,'Suggested crossing unavailable: No measured ground support at the source anchor. Suggested vehicle front unavailable: Front ground remains hidden. Suggested person unavailable: No calibrated band.');
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
test('measured ground: crossing source keeps its own support and hidden state and refuses a source hole',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.placementRangeLimitMeters=30;g.maxRangeMeters=40;}}));
  const grid=s.ground.measuredGround,point=[0,-2,-18],uv=projectMeasured(s,point),pixel=uv.map(n=>Math.floor(n+.5));
  for(let row=33;row<=37;row++)for(let col=0;col<grid.columns;col++)grid.heights[row*grid.columns+col]=-1.2;
  grid.heightBounds=[-2,-1.2];
  s.ground.crossing.edge1=[[-3,-2,-18],[3,-2,-18]];s.ground.crossing.edge2=[[-3,-2,-21],[3,-2,-21]];
  s.ground.cuboids.items[0].center=[0,-1,-15];
  s.suggestions={schemaVersion:3,sourceClass:'mappedGround',sceneId:s.id,crossing:{status:'mapped',nearEdge:'edge1',mappedPoint:point,pixelUV:uv,pixel},vehicle:{status:'unknown'}};
  const ray=mappedGroundHit(s,uv),source=anchorPoint(s,'crossing',pixel);
  assert.ok(Math.abs(ray.point[2]-point[2])>1);assert.equal(ray.hidden,false);
  assert.deepEqual(source.point,point);assert.equal(source.supportClass,'observed-nearby');assert.equal(source.hidden,true);
  const moved=anchorPoint(s,'crossing',[pixel[0]+1,pixel[1]]);
  assert.notDeepEqual(moved.point,point);
  grid.heights[24*grid.columns+10]=null;
  assert.throws(()=>anchorPoint(s,'crossing',pixel),/no supported mapped drivable ground point/);
});
test('measured ground: the evaluation-only refusal option preserves reference candidates and leaves default placement unchanged',async()=>{
  const s=await loadAnalytic(analytic({ground:g=>{measuredGrid(g,[0,0]);g.measuredGround.heights.fill(-1.4375);g.measuredGround.heightBounds=[-1.4375,-1.4375];
    g.cuboids.items=[];g.placementRangeLimitMeters=30;g.maxRangeMeters=40;g.alongRayUncertainty.bins[0].meters=.91;}}));
  const normal=clickToGround(s),option=clickToGround(s,{refuseOverBandMeters:.76});
  assert.deepEqual(option.bands.map(b=>b.candidates),normal.bands.map(b=>b.candidates));
  assert.equal(normal.metricBPoints.points.length,1);assert.equal(option.metricBPoints.points.length,0);
  assert.equal(option.bands[2].unknown['Too uncertain to place.'],1);
  assert.equal(mappedGroundHit(s,[2,6]).alongRayUncertaintyM,.91);
});

// Development artifact (local only, never shipped): the round-2 measured fd5c scene with its suggestions, loaded whole.
const R2=new URL('../.local/evaluation/pao-accuracy/ground-r2-2026-10-16/final/fd5c6932-2659589/',import.meta.url);
const r2Local=existsSync(new URL('grounded/suggestions.json',R2))&&existsSync(new URL('assembled/positions.bin',R2));
const r2Skip=r2Local?false:'round-2 measured development scene is local-only and not present';
// A built site folder: grounded/ and assembled/ files, plus a v2 build's tier-0-ground.json beside them.
const builtRead=(root,name)=>readFile(new URL(name==='tier-0-ground.json'?name:(['scene.json','ground.json','terrain.bin','suggestions.json','catalog-entry.json'].includes(name)?'grounded/':'assembled/')+name,root));
const loadR2=options=>loadBuilt(R2,options);
async function loadBuilt(root,{scene:mutateScene,ground:mutateGround,suggestions:mutateSuggestions}={}) {
  // Each changed file is repinned where the next names it: ground.json in scene.json, both in suggestions.json, then the catalog.
  const r2Read=name=>builtRead(root,name),entry=JSON.parse(await r2Read('catalog-entry.json')),bytes={},json=async name=>JSON.parse(await r2Read(name));
  if(mutateGround){const g=await json('ground.json');mutateGround(g);bytes['ground.json']=Buffer.from(JSON.stringify(g));}
  if(mutateScene||bytes['ground.json']) {
    const x=await json('scene.json');
    if(bytes['ground.json'])Object.assign(x.measurements.ground,{sha256:sha(bytes['ground.json']),bytes:bytes['ground.json'].length});
    mutateScene?.(x);bytes['scene.json']=Buffer.from(JSON.stringify(x));
  }
  if(mutateSuggestions||bytes['scene.json']) {
    const x=await json('suggestions.json');
    for(const name of ['ground.json','scene.json'])if(bytes[name])x.inputs[name]=sha(bytes[name]);
    mutateSuggestions?.(x);bytes['suggestions.json']=Buffer.from(JSON.stringify(x));
  }
  const pin=(name,record)=>bytes[name]?{...record,sha256:sha(bytes[name])}:record;
  const fetch=async path=>{const name=path.split('/').at(-1);
    if(!path.startsWith(`/scenes/${entry.id}/`))return new Response(null,{status:404});
    if(bytes[name])return new Response(bytes[name]);
    try{return new Response(await r2Read(name));}catch{return new Response(null,{status:404});}};
  return loadScene(fetch,entry.id,bytes['scene.json']?sha(bytes['scene.json']):entry.metadataSha256,undefined,
    pin('suggestions.json',entry.suggestions),pin('ground.json',entry.ground));
}
test('measured ground: a full development scene with suggestions loads with exact zero-width bands',{skip:r2Skip},async()=>{
  const zero=v=>{if(v&&typeof v==='object'){if(Object.hasOwn(v,'alongRayUncertaintyM')&&typeof v.alongRayUncertaintyM==='number')v.alongRayUncertaintyM=0;Object.values(v).forEach(zero);}};
  const s=await loadR2({ground:g=>g.alongRayUncertainty.bins.forEach(b=>b.meters=0),suggestions:zero});
  assert.equal(s.ground.schemaVersion,4);
  assert.equal(mappedGroundHit(s,s.suggestions.crossing.pixelUV).alongRayUncertaintyM,0);
  await assert.rejects(loadR2({ground:g=>g.alongRayUncertainty.bins.forEach(b=>b.meters=0)}),/runtime mapped-ground rule/);
  await assert.rejects(loadR2({ground:g=>g.alongRayUncertainty.bins.forEach(b=>b.meters=-.001),suggestions:zero}),/Invalid mapped ground/);
});
test('measured ground: the full development scene refuses a missing scoring sweep identity even when its hash is absent from the partition',{skip:r2Skip},async()=>{
  await loadR2();
  await assert.rejects(loadR2({scene:x=>delete x.measurements.provenance.originalSources}),/measured support needs the scoring sweep sha256/);
});
// Development artifacts (local only): the measured-ground-v2 builds of e1 and fd5c, with their tier-0 v1 files.
const V2=site=>new URL(`../.local/evaluation/pao-accuracy/ground-v2-full-2026-10-16/build/${site}/`,import.meta.url);
const v2Relabels=[['e1d68dde-5987490',12821],['fd5c6932-2659589',13465]];
const v2Skip=v2Relabels.every(([site])=>existsSync(new URL('tier-0-ground.json',V2(site)))&&existsSync(new URL('assembled/positions.bin',V2(site))))?false:
  'measured-ground-v2 development builds are local-only and not present';
test('measured ground v2: the e1 and fd5c development builds load, and relabelling a tier-1 vertex as tier 0 is refused',{skip:v2Skip},async()=>{
  for(const [site,vertex] of v2Relabels) {
    const s=await loadBuilt(V2(site));
    assert.equal(s.ground.measuredGround.tiers[vertex],1);
    await assert.rejects(loadBuilt(V2(site),{ground:g=>g.measuredGround.tiers[vertex]=0}),/tier-1 heights may fill only v1 holes/);
  }
});

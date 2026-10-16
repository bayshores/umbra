import assert from 'node:assert/strict';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {assessSightline,createScenario,mappedGroundHeight} from '../src/geometry.mjs';
import {suggestedSetup} from '../src/suggestions.mjs';
import {E_GRID,eGridPoints,footprint,metricEGrid} from '../scripts/evaluation/metric-e-grid.mjs';
import {loadSceneFrom,TARGET_HEIGHTS_M} from '../scripts/evaluation/metric-f.mjs';
import {devDataPresent,DEV,GROUNDED_DIR} from '../scripts/evaluation/pao-baseline.mjs';

const DEG=Math.PI/180;
// A 4 m x 2 m box 15 m ahead of the eye (+Z), turned 20 degrees, its axis toward the camera.
const turned=a=>({center:[1,.75,15],axis:[-Math.sin(a),0,-Math.cos(a)],vehicle:{width:2,height:1.5,length:4}});

test('metric E grid: points on the 0.25 m grid, beyond the box, inside the widened bearings, with mapped ground',()=>{
  const eye=[0,1.47,0],box=turned(20*DEG),flat=()=>0;
  const {points,noGround,limits}=eGridPoints(eye,box,flat);
  const corners=footprint(box),dist=([x,z])=>Math.hypot(x,z),bearing=([x,z])=>Math.atan2(x,z)-Math.atan2(1,15);
  const near=Math.max(...corners.map(dist))+.5,lo=Math.min(...corners.map(bearing))-2*DEG,hi=Math.max(...corners.map(bearing))+2*DEG;
  assert.ok(Math.abs(limits.nearM-near)<1e-12&&Math.abs(limits.farM-near-10)<1e-12);
  assert.equal(noGround,0);
  assert.ok(points.length>100);
  for(const p of points) {
    assert.equal(p.x/.25,Math.round(p.x/.25));assert.equal(p.z/.25,Math.round(p.z/.25));
    assert.ok(p.d>=near&&p.d<=near+10,`distance ${p.d}`);
    const b=bearing([p.x,p.z]);assert.ok(b>=lo-1e-12&&b<=hi+1e-12,`bearing ${b/DEG}`);
  }
  // Every grid point inside the sector is kept (brute force over the same square).
  let expected=0;
  for(let i=-160;i<=160;i++)for(let k=-160;k<=160;k++) {
    const x=i*.25,z=k*.25,d=Math.hypot(x,z),b=Math.atan2(x,z)-Math.atan2(1,15);
    if(d>=near&&d<=near+10&&b>=lo&&b<=hi)expected++;
  }
  assert.equal(points.length,expected);
  // Nearest first.
  for(let i=1;i<points.length;i++)assert.ok(points[i].d>=points[i-1].d);
  // No mapped ground: dropped and counted.
  const half=eGridPoints(eye,box,(x)=>x<1?null:0);
  assert.equal(half.points.length+half.noGround,points.length);
  assert.ok(half.points.every(p=>p.x>=1));
  assert.throws(()=>eGridPoints([1,1.47,15],box,flat),/inside the box footprint/);
});

test('metric E grid: every head point gets the app result of the recorded vehicle at its pose, eye at the camera',()=>{
  const box={...turned(0),shape:null},camera=[0,1.47,0],calls=[];
  const scene={suggestions:{vehicle:{status:'annotated',track:'t-1'}},camera:{eye:camera}};
  const app={
    createScenario:()=>({eye:[...camera],driverDistance:20,limits:{captureDistance:20},baseline:box,baselineSetback:3,whatIf:{pose:'annotated',setback:3}}),
    mappedGroundHeight:()=>.1,
    assessSightline:(s,line,b,opt)=>{calls.push({line,b,opt});return line.target[1]>1.4?{state:'unknown',reasonCode:'too-close-to-call',gapM:-.05}:{state:'blocked',gapM:-.3};}};
  const out=metricEGrid(scene,{input:{}},{heights:{child:1.1,adult:1.5},app});
  assert.equal(out.track,'t-1');
  assert.equal(out.rows.length,2*out.counts.points);
  assert.equal(calls.length,out.rows.length);
  assert.ok(calls.every(c=>c.b===box&&c.opt.ownPosition===true&&c.line.eye.every((n,i)=>n===camera[i])));
  const r=out.rows[0];
  assert.deepEqual(r.targetGround,[r.xM,.1,r.zM]);
  assert.ok(Math.abs(r.target[1]-1.2)<1e-12);
  assert.equal(r.state,'blocked');assert.equal(r.reasonCode,null);
  assert.equal(out.rows.at(-1).height,'adult');assert.equal(out.rows.at(-1).reasonCode,'too-close-to-call');
  assert.deepEqual(out.counts.byState,{'child/blocked':out.counts.points,'adult/unknown':out.counts.points});
  // Refusals: no annotated vehicle, a declared pose, an eye that is not the camera.
  assert.match(metricEGrid({...scene,suggestions:{vehicle:{status:'unknown'}}},{input:{}},{heights:{child:1.1},app}).error,/annotated/);
  const declared={...app,createScenario:()=>({...app.createScenario(),whatIf:{pose:'declared'}})};
  assert.match(metricEGrid(scene,{input:{}},{heights:{child:1.1},app:declared}).error,/annotated pose/);
  const moved={...app,createScenario:()=>({...app.createScenario(),driverDistance:18})};
  assert.match(metricEGrid(scene,{input:{}},{heights:{child:1.1},app:moved}).error,/recorded camera/);
  assert.match(metricEGrid(scene,{error:'no setup'},{heights:{child:1.1},app}).error,/no setup/);
});

const skip=devDataPresent()?false:'AV2 8c development data is local-only and not present';
test('metric E grid on the 8c development scene: rows equal a direct currentVisibility-style call',{skip},async()=>{
  const dir=fileURLToPath(DEV.dir),{scene}=await loadSceneFrom({assembled:dir+'assembled/',grounded:dir+GROUNDED_DIR});
  const setup=suggestedSetup(scene,scene.suggestions),out=metricEGrid(scene,setup,{heights:TARGET_HEIGHTS_M});
  assert.equal(out.error,undefined);
  assert.deepEqual(out.eye,scene.camera.eye);
  assert.equal(out.rule.stepM,E_GRID.stepM);
  const sc=createScenario(scene,setup.input);
  assert.deepEqual(out.box.center,sc.baseline.center);
  assert.equal(out.box.shape.track,scene.carShape.track);
  for(const r of out.rows.filter((_,i)=>i%97===0)) {
    assert.equal(r.targetGround[1],mappedGroundHeight(scene,r.xM,r.zM));
    const s=assessSightline(scene,{eye:sc.eye,target:r.target,targetGround:r.targetGround},sc.baseline,{ownPosition:true});
    assert.equal(r.state,s.state);assert.equal(r.reasonCode,s.reasonCode??null);assert.equal(r.gapM,s.gapM);
  }
  assert.ok(out.rows.some(r=>r.state==='blocked'),'some child head points are Blocked');
});

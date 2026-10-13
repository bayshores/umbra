import assert from 'node:assert/strict';
import test from 'node:test';
import {markStyle,footprint,mapLayers,rulerDots,rulerPoint,setbackAt,approachDistance} from '../src/map-layers.mjs';
import {GLYPHS,dotLayout} from '../src/dot-matrix.mjs';
import {createScenario} from '../src/geometry.mjs';
import {setbackSweep} from '../src/scenario-assistant.mjs';
import {suggestedSetup} from '../src/suggestions.mjs';
import {devDataPresent,loadDevelopmentScene} from '../scripts/evaluation/pao-baseline.mjs';

const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);
const nearPoint=(a,b,tolerance,what)=>a.forEach((v,i)=>near(v,b[i],tolerance,`${what??''}[${i}]`));

// ------------------------------------------------------------------ analytic (no local data)
test('map layers: lane paint styles come only from painted AV2 mark types',()=>{
  for(const t of [undefined,null,'','NONE','UNKNOWN','SOLID_BLUE'])assert.equal(markStyle(t),null,String(t));
  assert.deepEqual(markStyle('DOUBLE_SOLID_YELLOW'),{color:'yellow',double:true,dashed:false});
  assert.deepEqual(markStyle('SOLID_WHITE'),{color:'white',double:false,dashed:false});
  assert.deepEqual(markStyle('DASHED_WHITE'),{color:'white',double:false,dashed:true});
  assert.deepEqual(markStyle('DOUBLE_DASH_YELLOW'),{color:'yellow',double:true,dashed:true});
});

test('map layers: a cuboid footprint is its rotated length-by-width rectangle',()=>{
  const a=Math.PI/6,c=Math.cos(a),s=Math.sin(a);
  // worldFromBox columns: length axis (cos, 0, sin), width axis (-sin, 0, cos), height axis up.
  const item={center:[10,.5,-3],length:4,width:2,height:1.5,worldFromBox:[[c,-s,0],[0,0,1],[s,c,0]]};
  const f=footprint(item),side=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1]);
  assert.equal(f.length,4);
  near(side(f[0],f[1]),2,1e-12,'width edge');near(side(f[1],f[2]),4,1e-12,'length edge');
  near(side(f[2],f[3]),2,1e-12,'width edge');near(side(f[3],f[0]),4,1e-12,'length edge');
  nearPoint([f.reduce((n,p)=>n+p[0],0)/4,f.reduce((n,p)=>n+p[1],0)/4],[10,-3],1e-12,'centre');
  nearPoint(f[0],[10+2*c-s,-3+2*s+c],1e-12,'front-left corner');
});

test('map layers: built only from the scene map, null without one',()=>{
  for(const g of [null,undefined,{},{drivableAreas:[],egoLane:{}}])assert.equal(mapLayers(g),null);
  const square=[[0,0],[1,0],[1,1],[0,1]];
  const ground={drivableAreas:[{outline:square},{outline:[[0,0],[1,1]]}],
    egoLane:{leftBoundary:[[0,0,0],[0,0,5]],leftMarkType:'DOUBLE_SOLID_YELLOW',rightBoundary:[[3,0,0],[3,0,5]],rightMarkType:'NONE'},
    crossing:{edge1:[[0,0,0],[3,0,0]],edge2:[[0,0,-2],[3,0,-2]]},
    cuboids:{items:['REGULAR_VEHICLE','PEDESTRIAN','BICYCLE','BOLLARD'].map((category,i)=>({category,center:[i,1,0],length:1,width:1,height:2,
      worldFromBox:[[1,0,0],[0,0,1],[0,1,0]]}))}};
  const m=mapLayers(ground);
  assert.deepEqual(m.areas,[square],'outlines under three points are dropped');
  assert.deepEqual(m.lanes.map(l=>l.style.color),['yellow'],'unpainted boundaries are not drawn');
  assert.deepEqual(m.crossing,[[0,0,0],[3,0,0],[3,0,-2],[0,0,-2]],'crossing polygon runs edge1 then edge2 reversed');
  assert.deepEqual(m.objects.map(o=>o.kind),['vehicle','person','cyclist','other']);
  assert.deepEqual(m.objects.map(o=>o.ground),[0,0,0,0],'footprints sit at each box bottom');
  assert.deepEqual(mapLayers({...ground,cuboids:undefined}).objects,[]);
});

// A what-if vehicle 2 m wide and 4 m long whose front is at the crossing (origin) at setback 0, driver along +z.
const synthetic=(axis=[0,0,1])=>{
  const n=Math.hypot(...axis),u=axis.map(v=>v/n);
  return {axis:u,crossing:[0,0,0],limits:{captureDistance:10},
    whatIf:{pose:'annotated',center:[u[0]*5,1,u[2]*5],setback:3,axis:u,boxAxis:u,vehicle:{width:2,height:2,length:4}}};
};

test('map layers: ruler dots run from the crossing to the driver beside the vehicle line',()=>{
  const s=synthetic(),dots=rulerDots(s);
  assert.equal(dots.length,41,'every 0.25 m from 0 m to the 10 m driver distance');
  assert.ok(dots.every(d=>d.state===null),'nothing is coloured before a distance check');
  // Right of a driver facing the crossing (-z) is +x; the ruler sits half the width plus 0.7 m out, on the ground.
  nearPoint(dots[0].position,[1.7,0,0],1e-12,'0 m');nearPoint(dots.at(-1).position,[1.7,0,10],1e-12,'10 m');
  nearPoint(rulerPoint(s,4),[1.7,0,4],1e-12);nearPoint(rulerPoint(s,4,1.6),[3.3,0,4],1e-12,'labels sit farther out');
  for(const axis of [[0,0,1],[1,0,1],[-.58,.02,-.81]]) {
    const t=synthetic(axis);
    for(const m of [0,.42,3,9.75])for(const extra of [0,1.6])near(setbackAt(t,rulerPoint(t,m,extra)),m,1e-9,`axis ${axis} at ${m}`);
  }
});

test('map layers: ruler colours only checked distances, each from the nearest checked row',()=>{
  // Checked rows from 1 m to 3 m (a sweep's rows start at its slider minimum, never at 0 m).
  const rows=[{setback:1,state:'blocked'},{setback:1.25,state:'blocked'},{setback:1.5,state:'unknown'},{setback:2.1,state:'unknown'},{setback:3,state:'overlap'}];
  const at=m=>rulerDots(synthetic(),rows).find(d=>Math.abs(d.setback-m)<1e-9).state;
  assert.equal(at(0),null,'below the checked range');assert.equal(at(.75),null,'below the checked range');
  assert.equal(at(1),'blocked');assert.equal(at(1.25),'blocked');assert.equal(at(1.5),'unknown');
  assert.equal(at(1.75),null,'more than half a step from every row');assert.equal(at(2),'unknown','0.1 m from the 2.1 m row');
  assert.equal(at(3),'overlap');assert.equal(at(3.25),null,'above the checked range');
  assert.ok(rulerDots(synthetic(),[]).every(d=>d.state===null));
});

test('approach replay: the driver moves from far to exactly near, never back, slowing down, within the limits',()=>{
  const far=17.4412,closest=.2187,steps=50,d=Array.from({length:steps+1},(_,i)=>approachDistance(i,steps,far,closest));
  near(d[0],far,.005,'first step');
  assert.equal(d.at(-1),closest,'the last step is the nearest allowed distance');
  assert.equal(approachDistance(steps+3,steps,far,closest),closest);
  d.forEach((v,i)=>{
    assert.ok(v>=closest&&v<=far,`step ${i}: ${v} m`);
    if(i)assert.ok(v<=d[i-1],`step ${i} moves nearer`);
  });
  assert.ok(d[0]-d[1]>d[steps-1]-d[steps],'faster at the start than at the end');
});

test('dot lettering: every glyph is a 7-row grid and the on-screen words are fully lettered',()=>{
  for(const [ch,rows] of Object.entries(GLYPHS)) {
    assert.equal(rows.length,7,ch);
    assert.ok(rows.every(r=>r.length===rows[0].length&&/^[#.]+$/.test(r)),ch);
  }
  // The app's dot-matrix words: the wordmark and the three result states (never Clear).
  for(const word of ['Umbra','Blocked','Unknown','Overlap','0.42'])
    assert.ok([...word].every(ch=>GLYPHS[ch]),`${word} has a glyph for every character`);
  const {lit,unlit,columns,rows}=dotLayout('Bo');
  assert.equal(rows,7);assert.equal(columns,11,'two 5-column glyphs and a 1-column gap');
  assert.equal(lit.length+unlit.length,70);
  assert.deepEqual(dotLayout('B~o'),dotLayout('Bo'),'unknown characters are skipped');
  assert.deepEqual(dotLayout(''),{lit:[],unlit:[],columns:0,rows:7});
});

// ------------------------------------------------------------------ AV2 8c development scene (local only)
const skip=devDataPresent()?false:'AV2 8c development data is local-only and not present';

test('approach replay on AV2 8c: every step is a valid comparison from the recorded position to the nearest',{skip},async()=>{
  const {scene}=await loadDevelopmentScene(),setup=suggestedSetup(scene,scene.suggestions),s=createScenario(scene,setup.input);
  const {captureDistance:far,minimumDriverDistance:near}=s.limits;
  for(let i=0;i<=50;i++){
    const driverDistance=approachDistance(i,50,far,near),step=createScenario(scene,{...setup.input,driverDistance});
    assert.equal(step.driverDistance,driverDistance);
  }
});

test('map layers on AV2 8c: real map layers and a ruler that agrees with the comparison',{skip},async()=>{
  const {scene}=await loadDevelopmentScene(),setup=suggestedSetup(scene,scene.suggestions),s=createScenario(scene,setup.input);
  const m=mapLayers(scene.ground);
  assert.equal(m.areas.length,scene.ground.drivableAreas.length);
  assert.deepEqual(m.lanes.map(l=>l.style),[{color:'yellow',double:true,dashed:false},{color:'white',double:false,dashed:false}],
    'ego lane: double solid yellow left, solid white right');
  assert.equal(m.objects.length,scene.ground.cuboids.items.length);
  // The ruler's 0 m is the crossing, its setback reading is the comparison's own setback and driver distance.
  near(setbackAt(s,s.crossing),0,1e-9,'crossing');
  const b=s.proposed,front=[b.center[0]-b.axis[0]*b.vehicle.length/2,0,b.center[2]-b.axis[2]*b.vehicle.length/2];
  near(setbackAt(s,front),s.proposedSetback,1e-9,'proposed front');
  near(setbackAt(s,s.eye),s.driverDistance,1e-9,'driver');
  // After a distance check: only the checked range is coloured, with the sweep's own states.
  const sweep=setbackSweep(scene,setup.input,{label:'checked',...setup.input.vehicle}),dots=rulerDots(s,sweep.rows);
  const states=new Set(sweep.rows.map(r=>r.state));
  for(const d of dots) {
    const inside=d.setback>=sweep.rows[0].setback-1e-9&&d.setback<=sweep.rows.at(-1).setback+1e-9;
    if(!inside)assert.equal(d.state,null,`${d.setback} m is outside the checked range`);
    else assert.ok(d.state===null||states.has(d.state),`${d.setback} m`);
  }
  assert.equal(dots[0].state,null,'0 m is below the slider minimum and never checked');
  assert.ok(dots.some(d=>d.state==='blocked'),'the checked vehicle blocks the person at some distances');
});

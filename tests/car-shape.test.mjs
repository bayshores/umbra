// shape-v1 synthetic tests (.local/evaluation/pao-accuracy/shape-v1/design.md 16.2, JavaScript groups 1-12; the frozen
// protocol.json resultRule, deepTest, representation and loaderValidation). No scene data: every shape, line and scene
// here is built in memory. Car-local frame: x forward (front face at x = +L/2), y to the car's left, z above the box
// bottom; cell (i, j) counts i from the front face and j from the left side.
import assert from 'node:assert/strict';
import test from 'node:test';
import {loadScene} from '../src/scene.mjs';
import {assessSightline,prepareCarShape,shapeDeep,shapeCut,segmentIntersectsBox,sightlineGap,SHAPE_REASONS,CAR_SHAPE_REFUSALS,TOO_CLOSE_TO_CALL_M,
  UNKNOWN_REASONS,whatIfBox,createScenario,comparisonStart,annotatedVehicle,mappedGroundHit,placementFromPixels} from '../src/geometry.mjs';
import {setbackSweep,sweepPhrase,sweepShort,sweepFacts,checkAnswer,allowedNumbersFrom,resultPhrase,scenarioFacts} from '../src/scenario-assistant.mjs';
import {carShapeRecord} from './car-shape-fixture.mjs';
import {analytic,loadAnalytic,sha} from './mapped-fixture.mjs';

const D=TOO_CLOSE_TO_CALL_M;
const BOX_BLOCKED='Declared vehicle volume intersects the sightline.';
const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);
// Seeded generator (mulberry32): every run tests the same shapes and lines.
function random(seed) {
  let a=seed>>>0;
  return ()=>{a=a+0x6D2B79F5>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};
}

// ------------------------------------------------------------------ synthetic box (groups 1-8)
// assessSightline without a mapped scene: no real annotated objects, and classifySightline's later diagnostics give one
// fixed Unknown sentence (indexedSparse), so every result comes from the box and shape rules alone. The box is centred at
// the origin with axis +z, so its box-local [side, y, along] coordinates are the world coordinates.
const SPARSE={measurements:{kind:'indexedSparse'}};
const boxOf=(dims,P)=>({center:[0,0,0],axis:[0,0,1],vehicle:dims,...(P&&{shape:P})});
// Car-local [x, y, z] to box-local (= world here): side = -y, y_app = z - H/2, along = -x.
const world=([x,y,z],H)=>[-y,z-H/2,-x];
// The person's ground point is 1 m under the target, so the person column is vertical at the target.
const line=(eye,target)=>({eye,target,targetGround:[target[0],target[1]-1,target[2]]});
// A random line around a box: ends on a horizontal circle outside the footprint (sometimes, with inside, anywhere near
// the box, so the eye or the person can be inside it), heights from below the box bottom to above its top.
function randomLine(rand,{length:L,width:W,height:H},{inside=false}={}) {
  const end=()=>{
    if(inside&&rand()<.12)return world([(rand()-.5)*1.3*L,(rand()-.5)*1.3*W,(rand()*1.4-.2)*H],H);
    const a=rand()*2*Math.PI,R=Math.hypot(L,W)/2+.05+rand()*1.5;
    return world([R*Math.cos(a),R*Math.sin(a),(rand()*1.5-.25)*H],H);
  };
  return line(end(),end());
}
const record=(dims,top,refusal=null)=>carShapeRecord({track:'t',...dims,top,refusal});
const REFUSED=code=>({code,reason:CAR_SHAPE_REFUSALS[code]});
const results=(s,dims,P,diagnostics)=>[assessSightline(SPARSE,s,boxOf(dims),{diagnostics}),assessSightline(SPARSE,s,boxOf(dims,P),{diagnostics})];
const kind=r=>r.state==='blocked'?'blocked':r.reasonCode??'plain';
// Table 8.2: a box Blocked becomes Blocked (the box sentence through the 6a shortcut, the shape sentence otherwise) or
// Unknown height-not-proven with the same gapM; every other row is the identical object.
function allowedChange(box,shaped,P,what) {
  if(box.state!=='blocked')return assert.deepEqual(shaped,box,what);
  assert.equal(shaped.gapM,box.gapM,what);
  if(P.allTopsAtLeastH)return assert.deepEqual(shaped,box,what);
  if(shaped.state==='blocked')assert.equal(shaped.reason,SHAPE_REASONS.blocked,what);
  else {
    assert.equal(shaped.reasonCode,'height-not-proven',what);
    assert.equal(shaped.reason,P.status==='refused'?SHAPE_REASONS.heightNotProvenRefused(P.refusal.reason):SHAPE_REASONS.heightNotProvenMeasured,what);
  }
}

// The deep test's meaning, checked directly from the ground.json record (not from the prepared shape): some point of
// the line has the whole cube of half-size d around it inside the columns, i.e. at least d inside the footprint, at least
// d above the box bottom and at least d below the solid top (null = 0) of every cell its square overlaps with some area.
// Candidate line parameters: every value where one of those conditions can change, the midpoints between them and a
// uniform grid; candidates themselves are allowed 1e-9 m.
function cubeInside(c,x,y,z,d,tolerance=1e-9) {
  const {length:L,width:W}=c.box,n=c.grid.along,m=c.grid.lateral,xe=i=>i===n?-L/2:L/2-i*(L/n),ye=j=>j===m?-W/2:W/2-j*(W/m);
  if(Math.abs(x)>L/2-d+tolerance||Math.abs(y)>W/2-d+tolerance||z<d-tolerance)return false;
  for(let i=0;i<n;i++)if(xe(i+1)<x+d-tolerance&&x-d+tolerance<xe(i))
    for(let j=0;j<m;j++)if(ye(j+1)<y+d-tolerance&&y-d+tolerance<ye(j)&&z+d>(c.solidTopM[i*m+j]??0)+tolerance)return false;
  return true;
}
function bruteDeep(c,e,t,d=D) {
  const {length:L,width:W,height:H}=c.box,n=c.grid.along,m=c.grid.lateral;
  const x0=-e[2],y0=-e[0],z0=e[1]+H/2,dx=e[2]-t[2],dy=e[0]-t[0],dz=t[1]-e[1],cuts=[0,1];
  const at=(p0,dp,v)=>{if(dp!==0){const s=(v-p0)/dp;if(s>0&&s<1)cuts.push(s);}};
  for(let i=0;i<=n;i++)for(const v of [-d,d])at(x0,dx,(i===n?-L/2:L/2-i*(L/n))+v);
  for(let j=0;j<=m;j++)for(const v of [-d,d])at(y0,dy,(j===m?-W/2:W/2-j*(W/m))+v);
  for(const v of [-L/2+d,L/2-d])at(x0,dx,v);
  for(const v of [-W/2+d,W/2-d])at(y0,dy,v);
  for(const v of [d,...new Set(c.solidTopM.map(top=>(top??0)-d))])at(z0,dz,v);
  for(let k=1;k<100;k++)cuts.push(k/100);
  cuts.sort((a,b)=>a-b);
  const params=[...cuts];
  for(let k=1;k<cuts.length;k++)params.push((cuts[k-1]+cuts[k])/2);
  return params.some(s=>cubeInside(c,x0+dx*s,y0+dy*s,z0+dz*s,d));
}

const CAR8C={length:4.03,width:1.8825,height:1.489};// AV2 8c's averaged stationary vehicle sizes

test('shape-v1 1: a fully measured car (every solid top at least the box height) gives exactly the box results',()=>{
  const dims=CAR8C,{length:L,width:W,height:H}=dims,half=[W/2,H/2,L/2];
  const P=prepareCarShape(record(dims,()=>H));
  assert.equal(P.allTopsAtLeastH,true);
  assert.deepEqual([P.along,P.lateral],[41,19]);
  const rand=random(1),lines=[];
  for(let k=0;k<2000;k++)lines.push(randomLine(rand,dims,{inside:true}));
  // Boundary lines (box-local ends): level lines that cut exactly 0.10 m and 0.10 m +/- 1e-12 into the top, the bottom,
  // each side and each end; lines that touch a face, an edge or a corner exactly or miss by 1e-12; the eye or the person
  // on a face.
  const cut=(axis,depth,sign)=>{
    const e=[0,0,0];e[axis]=sign*(half[axis]-depth);
    const t=[...e],free=axis===0?2:0;e[free]=-6;t[free]=6;
    return line(e,t);
  };
  for(const depth of [D,D-1e-12,D+1e-12,0,-1e-12,1e-12,.05,.2])for(const axis of [0,1,2])for(const sign of [-1,1])lines.push(cut(axis,depth,sign));
  for(const depth of [D,D-1e-12,D+1e-12,0,-1e-12]) {
    // Lines that meet the box shrunk by `depth` only at one top edge or one top corner (box-local, along +z).
    const through=(p,dir)=>line(p.map((v,i)=>v-3*dir[i]),p.map((v,i)=>v+3*dir[i]));
    lines.push(through([0,H/2-depth,L/2-depth],[0,-1,1]),through([W/2-depth,H/2-depth,0],[1,-1,0]),through([W/2-depth,H/2-depth,L/2-depth],[1,1,-1]));
  }
  lines.push(line([0,0,L/2],[0,0,12]),line([0,0,-12],[0,H/2,L/2]),line([W/2,0,0],[3,.5,5]),line([-5,0,9],[0,-H/2,0]));
  const seen=new Set();
  for(const [k,s] of lines.entries())for(const diagnostics of [true,false]) {
    const [box,shaped]=results(s,dims,P,diagnostics);
    assert.deepEqual(shaped,box,`line ${k} diagnostics ${diagnostics}`);
    seen.add(kind(box));
  }
  assert.deepEqual([...seen].sort(),['blocked','driver-inside-vehicle','person-inside-vehicle','plain','too-close-to-call']);
});

test('shape-v1 2: the piece test on an all-H shape equals the box shrunk by 0.10 m',()=>{
  const dims=CAR8C,{length:L,width:W,height:H}=dims,half=[W/2,H/2,L/2];
  const P=prepareCarShape(record(dims,()=>H)),rand=random(2);
  let deep=0,shallow=0;
  for(let k=0;deep+shallow<2000;k++) {
    const {eye:e,target:t}=randomLine(rand,dims);
    if(Math.abs(-sightlineGap(e,t,half)-D)<1e-6)continue;
    const shrunk=segmentIntersectsBox(e,t,half.map(n=>D-n),half.map(n=>n-D));
    assert.equal(shapeDeep(P,e,t),shrunk,`line ${k}`);
    if(shrunk)deep++;else shallow++;
  }
  assert.ok(deep>=300&&shallow>=300,`${deep} deep, ${shallow} not deep`);
});

test('shape-v1 3: a low hood blocks only lines at least 0.10 m below its top; higher lines over it are not decided',()=>{
  const dims=CAR8C,{length:L,height:H}=dims,c=record(dims,i=>i<12?.9:H),P=prepareCarShape(c),x=L/2-.6;
  assert.equal(P.allTopsAtLeastH,false);
  const at=z=>line(world([x,3,z],H),world([x,-3,z],H));
  // Every cell under the 0.20 m square around x = L/2 - 0.6 is a hood cell (i < 12 reaches L/2 - 1.18 m).
  const below=at(.8-1e-6),[box,shaped]=results(below,dims,P,true);
  assert.deepEqual(box,{state:'blocked',reason:BOX_BLOCKED,supportedFraction:null,gapM:box.gapM});
  near(box.gapM,-.6,1e-8);
  assert.deepEqual({...shaped,shape:undefined},{state:'blocked',reason:SHAPE_REASONS.blocked,supportedFraction:null,gapM:box.gapM,shape:undefined});
  assert.equal(shaped.shape.status,'measured');near(shaped.shape.solidCutM,.9-(.8-1e-6),2e-9);near(shaped.shape.hitXM,x,1e-12);
  assert.deepEqual(results(below,dims,P,false)[1],{state:'blocked',reason:SHAPE_REASONS.blocked,supportedFraction:null,gapM:box.gapM});
  for(const [z,cutM] of [[.8+1e-6,.9-(.8+1e-6)],[1.2,null]]) {
    const s=at(z),[box,shaped]=results(s,dims,P,true);
    assert.equal(box.state,'blocked',String(z));
    assert.deepEqual({...shaped,shape:undefined},{state:'unknown',reason:SHAPE_REASONS.heightNotProvenMeasured,supportedFraction:null,reasonCode:'height-not-proven',
      gapM:box.gapM,shape:undefined},String(z));
    assert.equal(shaped.shape.hitXM,null,String(z));
    if(cutM===null)assert.equal(shaped.shape.solidCutM,null,String(z));else near(shaped.shape.solidCutM,cutM,2e-9,String(z));
    assert.deepEqual(results(s,dims,P,false)[1],{state:'unknown',reason:SHAPE_REASONS.heightNotProvenMeasured,supportedFraction:null,reasonCode:'height-not-proven',gapM:box.gapM});
  }
  for(const [z,expected] of [[H-.05,'too-close-to-call'],[H+.05,'plain']])for(const diagnostics of [true,false]) {
    const [box,shaped]=results(at(z),dims,P,diagnostics);
    assert.equal(kind(box),expected);assert.deepEqual(shaped,box,`${z} ${diagnostics}`);
  }
  assert.equal(resultPhrase(results(at(1.2),dims,P,true)[1]),'unknown (vehicle height not proven)');
});

test('shape-v1 4: a line cutting 0.20 m into an empty (unproven) corner is not decided; the same cut into a measured corner blocks',()=>{
  const dims=CAR8C,{length:L,width:W,height:H}=dims,P=prepareCarShape(record(dims,(i,j)=>i<5&&j<5?null:H));
  // Level lines at mid-height through the point 0.20 m inside the corner, at 45 degrees, so the box cut is 0.20 m and
  // every point of the line inside the box lies within 0.40 m of the corner (front-left: cells i < 5, j < 5 reach 0.49 m).
  const corner=(cx,cy,sx,sy)=>{const p=[cx,cy,H/2],u=[sx,-sy,0].map(v=>v*3/Math.SQRT2);return line(world(p.map((v,i)=>v-u[i]),H),world(p.map((v,i)=>v+u[i]),H));};
  const empty=corner(L/2-.2,W/2-.2,1,1),[box,shaped]=results(empty,dims,P,true);
  assert.equal(box.state,'blocked');near(box.gapM,-.2,1e-8);
  assert.deepEqual([shaped.state,shaped.reasonCode,shaped.reason,shaped.gapM],['unknown','height-not-proven',SHAPE_REASONS.heightNotProvenMeasured,box.gapM]);
  const measured=corner(-L/2+.2,-W/2+.2,1,1),[box2,shaped2]=results(measured,dims,P,true);
  assert.equal(box2.state,'blocked');near(box2.gapM,-.2,1e-8);
  assert.deepEqual([shaped2.state,shaped2.reason],['blocked',SHAPE_REASONS.blocked]);
});

test('shape-v1 5: one tall noise cell among lower neighbours never blocks a line at 1.20 m',()=>{
  const dims=CAR8C,{length:L,width:W,height:H}=dims,ci=20,cj=9;
  const P=prepareCarShape(record(dims,(i,j)=>i===ci&&j===cj?H:Math.abs(i-ci)<=1&&Math.abs(j-cj)<=1?.5:null));
  const x=L/2-(ci+.5)*(L/41),y=W/2-(cj+.5)*(W/19);
  for(let k=0;k<24;k++) {
    const a=k*Math.PI/24,u=[Math.cos(a)*3,Math.sin(a)*3],s=line(world([x-u[0],y-u[1],1.2],H),world([x+u[0],y+u[1],1.2],H));
    for(const diagnostics of [true,false]) {
      const [box,shaped]=results(s,dims,P,diagnostics);
      assert.equal(box.state,'blocked',`${k}`);
      assert.deepEqual([shaped.state,shaped.reasonCode],['unknown','height-not-proven'],`${k}`);
      allowedChange(box,shaped,P,`${k}`);
    }
  }
});

test('shape-v1 6: an unmeasured or refused shape turns every box Blocked into height-not-proven and changes nothing else',()=>{
  const dims=CAR8C;
  for(const [name,refusal] of [['measured, all null',null],['refused, too few sweeps',REFUSED('too-few-sweeps')],['refused, not stationary',REFUSED('not-stationary')]]) {
    const P=prepareCarShape(record(dims,()=>null,refusal)),rand=random(6),reason=refusal?SHAPE_REASONS.heightNotProvenRefused(refusal.reason):SHAPE_REASONS.heightNotProvenMeasured;
    assert.equal(P.status,refusal?'refused':'measured');
    if(refusal)assert.match(reason,new RegExp(`\\(${refusal.reason}\\)`));
    let blocked=0;
    for(let k=0;k<2000;k++) {
      const s=randomLine(rand,dims,{inside:true});
      for(const diagnostics of [true,false]) {
        const [box,shaped]=results(s,dims,P,diagnostics);
        if(box.state!=='blocked'){assert.deepEqual(shaped,box,`${name} ${k}`);continue;}
        blocked+=diagnostics;
        const local=[s.eye,s.target];// box-local = world
        const expected={state:'unknown',reason,supportedFraction:null,reasonCode:'height-not-proven',gapM:box.gapM,
          ...(diagnostics&&{shape:{status:P.status,...shapeCut(P,...local)}})};
        assert.deepEqual(shaped,expected,`${name} ${k}`);
        // A zero-height column is touched only by a line crossing the box bottom; nothing is ever deep in it.
        if(diagnostics)assert.ok(shaped.shape.hitXM===null&&(shaped.shape.solidCutM===null||shaped.shape.solidCutM===0),`${name} ${k}`);
      }
    }
    assert.ok(blocked>=300,`${name}: ${blocked} box-Blocked lines`);
  }
});

test('shape-v1 7 and 8: on random small shapes every shape Blocked has a line point whose 0.10 m cube is inside the columns, and every row obeys table 8.2',()=>{
  const rand=random(7),count={blocked:0,shapeBlocked:0,notProven:0,shortcut:0,refused:0};
  for(let k=0;k<5000;k++) {
    const dims={length:.3+rand()*.9,width:.3+rand()*.9,height:.4+rand()*.6},H=dims.height,style=rand(),level=()=>.25+Math.floor(rand()*(H-.25)*1000)/1000;
    // Styles: the box, all null, refused, two levels (a lower block of front rows or left columns), and per-cell noise.
    const rows=1+Math.floor(rand()*5),low=level(),byRow=rand()<.5;
    const top=style<.08?()=>H:style<.12?()=>null:style<.5?(i,j)=>(byRow?i:j)<rows?low:H:()=>{const r=rand();return r<.15?null:r<.55?H:level();};
    const refusal=style>=.12&&style<.15?REFUSED(rand()<.5?'too-few-sweeps':'not-stationary'):null;
    const c=record(dims,refusal?()=>null:top,refusal),P=prepareCarShape(c),s=randomLine(rand,dims,{inside:true});
    for(const diagnostics of [true,false]) {
      const [box,shaped]=results(s,dims,P,diagnostics);
      allowedChange(box,shaped,P,`case ${k} diagnostics ${diagnostics}`);
      if(diagnostics&&shaped.state==='blocked')assert.ok(bruteDeep(c,s.eye,s.target),`case ${k}: Blocked without a deep point`);
      if(diagnostics&&box.state==='blocked') {
        count.blocked++;
        if(P.allTopsAtLeastH)count.shortcut++;else if(shaped.state==='blocked')count.shapeBlocked++;else count.notProven++;
        if(P.status==='refused')count.refused++;
      }
    }
  }
  assert.ok(count.shapeBlocked>=200&&count.notProven>=200&&count.shortcut>=20&&count.refused>=5,JSON.stringify(count));
});

// ------------------------------------------------------------------ mapped scene with the recorded vehicle (groups 9-12)
// The analytic mapped scene (tests/mapped-fixture.mjs) with its cuboid moved into the lane: centre x 0.2, front face at
// z -13 (1.375 m before the crossing at z -14.375), facing the direction of travel (-z), 4 x 1.8 x 1.3 m, averaged from
// 3 annotations. suggestions.json names it as the annotated stationary vehicle and ground.json carries its car shape:
// by default a low hood (rows i < 12, the front 1.17 m, solid to 0.90 m; the rest to the box height).
const CAR={length:4,width:1.8,height:1.3},FIELDS={width:'1.8',height:'1.3',length:'4'},FRONT=[.2,-1.4375,-13];
const PIXELS={crossing:[2,4],parking:[2,5],target:[3,4]};
const hood=i=>i<12?.9:CAR.height;
function placeCar(g) {
  Object.assign(g.cuboids.items[0],{center:[.2,-1.4375+.65,-11],...CAR,worldFromBox:[[0,-1,0],[0,0,1],[-1,0,0]],
    poseAverage:{rule:'averageAllAnnotations',rows:3,centreSpreadM:{alongStd:.02,lateralStd:.01},yawSpreadDeg:{std:.1}}});
}
const vehicleFixture=({top=hood,refusal=null,shape=()=>{},ground=()=>{}}={})=>analytic({ground:g=>{
  placeCar(g);g.carShape=carShapeRecord({track:'analytic',...CAR,top,refusal});shape(g.carShape,g);ground(g);
}});
// The vehicle suggestion's ray results at pixel [2, 5], from the same scene loaded without a car shape (they do not
// depend on it).
let hits;
const vehicleHit=()=>hits??=loadAnalytic(analytic({ground:placeCar})).then(s=>mappedGroundHit(s,[2,5]));
function suggestionsFor(f,hit,mutate) {
  const g=f.ground,m=f.scene.measurements,offset=Math.hypot(hit.point[0]-FRONT[0],hit.point[2]-FRONT[2]),none={status:'unknown',supportClass:'unknown',reason:'Analytic fixture: not suggested.'};
  const raw={schemaVersion:3,sceneId:f.scene.id,sourceClass:'mappedGround',generatedUtc:'2026-10-13T00:00:00Z',
    generator:{script:'scripts/california/suggest_av2_placements.py',sha256:'1'.repeat(64),sharedScript:'scripts/california/prepare_av2_ground.py',sharedSha256:'2'.repeat(64),python:'analytic',numpy:'analytic'},
    inputs:{'scene.json':sha(JSON.stringify(f.scene)),'ground.json':m.ground.sha256,'terrain.bin':g.terrain.sha256,'recorded.png':f.scene.recordedImage.sha256,
      'referencepositions.bin':m.artifactHashes.referencePositions,'referenceeligible.bin':m.artifactHashes.referenceEligible,'protocol.json':g.protocol.sha256},
    protocolSha256:g.protocol.sha256,license:g.license,
    tolerances:{roadToleranceMeters:g.roadToleranceMeters,supportHeightAgreementMeters:g.supportHeightAgreementMeters,supportRadiusMeters:g.supportRadiusMeters,
      maxRangeMeters:g.maxRangeMeters,placementRangeLimitMeters:g.placementRangeLimitMeters,alongRayHeightBandMeters:g.alongRayUncertainty.heightBandMeters},
    crossing:{...none},roadAxis:{status:'unknown',reason:none.reason},curb:{status:'unknown',label:'curb',reason:none.reason},
    vehicle:{status:'annotated',label:'stationary vehicle',reason:'Analytic fixture.',pixel:[2,5],pixelUV:[2,5],groundPoint:hit.point,supportClass:hit.supportClass,
      supportLabel:hit.supportLabel,hidden:hit.hidden,alongRayUncertaintyM:hit.alongRayUncertaintyM,rangeM:hit.rangeM,horizontalRangeM:Math.hypot(hit.point[0],hit.point[2]),
      hitKind:hit.hitKind,terrainCityZ:hit.terrainCityZ,terrainCell:hit.terrainCell,mappedPoint:FRONT,pixelOffsetFromMappedM:offset,
      integerPixelCheck:{groundPoint:hit.point,residualM:offset},supportEvidence:{},dimensions:{width:1.8,height:1.3,length:4},track:'analytic',
      category:'REGULAR_VEHICLE',motion:null,pose:{kind:'averaged',annotations:3},headingWorld:[0,0,-1],headingToRoadAxisDeg:0,yawToRoadAxisDeg:0,
      frontGapToCrossingM:1.375,appModel:'averaged annotated box',frontResidualToAnnotatedM:{alongRoadAxis:0,horizontal:0,lateralRight:0}},
    person:{...none},limitations:['Analytic fixture only.']};
  mutate?.(raw);
  return raw;
}
// Loads through the app loader with suggestions.json (suggest false: no suggestions file).
async function loadVehicle(f,{suggest=true,mutate}={}) {
  let descriptor;
  if(suggest) {
    const bytes=Buffer.from(JSON.stringify(suggestionsFor(f,await vehicleHit(),mutate)));
    f.bytes.set('suggestions.json',bytes);descriptor={file:'suggestions.json',sha256:sha(bytes)};
  }
  return loadScene(f.fetch,f.scene.id,sha(JSON.stringify(f.scene)),undefined,descriptor,f.pin);
}
const withShape=(s,top,refusal=null)=>({...s,carShape:prepareCarShape(carShapeRecord({track:'analytic',...CAR,top,refusal}))});
const startInput=(s,targetHeight=1.1)=>({pixels:PIXELS,vehicle:FIELDS,targetHeight,...comparisonStart(s,PIXELS,FIELDS)});
const sweepOf=(s,targetHeight=1.1,driverDistance)=>setbackSweep(s,{pixels:PIXELS,driverDistance:driverDistance??comparisonStart(s,PIXELS,FIELDS).driverDistance},
  {label:'your vehicle',width:1.8,height:1.3,length:4},targetHeight);
const sub=(a,b)=>a.map((n,i)=>n-b[i]);

test('shape-v1 9: the car shape moves rigidly with the recorded vehicle; declared, resized and replacement vehicles keep the box',async()=>{
  const s=await loadVehicle(vehicleFixture());
  assert.equal(s.carShape.track,'analytic');assert.equal(s.carShape.status,'measured');assert.equal(Object.isFrozen(s.carShape),true);
  assert.deepEqual(JSON.parse(JSON.stringify(s.carShape)),{schema:'shape-v1',status:'measured',track:'analytic'});
  const input=startInput(s),scenario=createScenario(s,input);
  assert.equal(scenario.baseline.pose,'annotated');
  for(const box of [scenario.baseline,scenario.proposed,scenario.whatIf])assert.equal(box.shape,s.carShape);
  assert.equal(annotatedVehicle(s,PIXELS,FIELDS).shape,s.carShape);
  // Translation: the result for the vehicle moved along the lane equals the result for the vehicle where it is with the
  // sightline moved back by the same offset.
  const kinds=new Set(),{minimumSetback:min,maximumSetback:max}=scenario.limits;
  for(let setback=min;setback<=max;setback+=.173) {
    const box=whatIfBox(scenario.whatIf,setback),offset=sub(box.center,scenario.baseline.center);
    assert.equal(box.shape,s.carShape);assert.deepEqual([box.axis,box.vehicle,box.pose],[scenario.baseline.axis,scenario.baseline.vehicle,'annotated']);
    const moved=assessSightline(s,scenario,box),back=assessSightline(s,{eye:sub(scenario.eye,offset),target:sub(scenario.target,offset),
      targetGround:sub(scenario.targetGround,offset)},scenario.baseline);
    const what=`setback ${setback}`;
    assert.deepEqual([moved.state,moved.reasonCode,moved.reason],[back.state,back.reasonCode,back.reason],what);
    near(moved.gapM,back.gapM,1e-9,what);
    assert.equal(Object.hasOwn(moved,'shape'),Object.hasOwn(back,'shape'),what);
    if(moved.shape) {
      assert.equal(moved.shape.hitXM===null,back.shape.hitXM===null,what);if(moved.shape.hitXM!==null)near(moved.shape.hitXM,back.shape.hitXM,1e-9,what);
      assert.equal(moved.shape.solidCutM===null,back.shape.solidCutM===null,what);if(moved.shape.solidCutM!==null)near(moved.shape.solidCutM,back.shape.solidCutM,1e-6,what);
    }
    kinds.add(moved.state==='blocked'?moved.reason:moved.reasonCode??'plain');
  }
  assert.deepEqual([...kinds].sort(),[SHAPE_REASONS.blocked,'height-not-proven','plain','too-close-to-call'].sort());
  // A vehicle declared in its place (even at the annotated sizes), a resized one and one whose front pixel moved away
  // are declared boxes: no shape on any of their boxes, and no height-not-proven row in their sweeps.
  const lane={...input,proposedSetback:placementFromPixels(s,PIXELS).baselineSetback};
  for(const [name,next] of [['declared',{...lane,vehicleDeclared:true}],['resized',{...lane,vehicle:{...FIELDS,width:'1.9'}}],
    ['declared nudged',{...lane,pixels:{...PIXELS,parking:[3,6]},vehicleDeclared:true}],['moved front',{...lane,pixels:{...PIXELS,parking:[2,8]}}]]) {
    const declared=createScenario(s,next);
    for(const box of [declared.baseline,declared.proposed,declared.whatIf]) {
      assert.equal(box.pose,'declared',name);assert.equal(Object.hasOwn(box,'shape'),false,name);
    }
    const sweep=setbackSweep(s,{pixels:next.pixels,driverDistance:next.driverDistance,...(next.vehicleDeclared&&{vehicleDeclared:true})},
      {label:'van',width:Number(next.vehicle.width),height:Number(next.vehicle.height),length:Number(next.vehicle.length)},1.1);
    assert.equal(sweep.error,undefined,name);
    assert.ok(sweep.rows.length>10&&sweep.rows.every(r=>r.reason!=='height-not-proven'),name);
  }
});

test('shape-v1 10: setback sweeps with a car shape never disagree with the full result rule, and change only box-Blocked rows',async()=>{
  const s=await loadVehicle(vehicleFixture()),rand=random(10);
  const shapes=[['hood',hood],['box',()=>CAR.height],['empty front',i=>i<10?null:CAR.height],['all null',()=>null],
    ...Array.from({length:12},(_,k)=>[`noise ${k}`,()=>{const r=rand();return r<.25?null:r<.6?CAR.height:.25+Math.floor(rand()*(CAR.height-.25)*1000)/1000;}])];
  const box=withShape(s,()=>CAR.height),capture=comparisonStart(s,PIXELS,FIELDS).driverDistance,changed={blocked:0,notProven:0};
  for(const [name,top] of [...shapes,['refused',()=>null,REFUSED('too-few-sweeps')]]) {
    const scene=name==='hood'?s:withShape(s,top,shapes.find(x=>x[0]===name)?null:REFUSED('too-few-sweeps'));
    for(const targetHeight of [.5,1.1,1.5])for(const driverDistance of [capture,capture-2.5]) {
      const what=`${name} target ${targetHeight} driver ${driverDistance}`,sweep=sweepOf(scene,targetHeight,driverDistance),reference=sweepOf(box,targetHeight,driverDistance);
      assert.equal(sweep.error,undefined,what);assert.equal(reference.error,undefined,what);
      assert.deepEqual(sweep.rows.map(r=>r.setback),reference.rows.map(r=>r.setback),what);
      sweep.rows.forEach((row,i)=>{
        const ref=reference.rows[i];
        if(ref.state!=='blocked'){assert.deepEqual(row,ref,`${what} row ${i}`);return;}
        assert.equal(row.gapM,ref.gapM,`${what} row ${i}`);
        assert.ok(row.state==='blocked'&&!row.reason||row.state==='unknown'&&row.reason==='height-not-proven',`${what} row ${i}`);
        if(row.state==='blocked')changed.blocked++;else changed.notProven++;
      });
      if(targetHeight!==1.1||driverDistance!==capture)continue;
      // Every row (not only the first of each run): the fast path (diagnostics false) and the full path agree.
      const scenario=createScenario(scene,{...startInput(scene),driverDistance});
      for(const row of sweep.rows) {
        const b=whatIfBox(scenario.whatIf,row.setback),own=row.setback===scenario.baselineSetback;
        const fast=assessSightline(scene,scenario,b,{diagnostics:false,ownPosition:own}),full=assessSightline(scene,scenario,b,{ownPosition:own});
        assert.deepEqual([full.state,full.reasonCode],[fast.state,fast.reasonCode],`${what} setback ${row.setback}`);
        if(full.shape){const {shape:_,...rest}=full;assert.deepEqual(rest,fast,`${what} setback ${row.setback}`);}
      }
    }
  }
  assert.ok(changed.blocked>=50&&changed.notProven>=50,JSON.stringify(changed));
});

test('shape-v1 11: the loader accepts valid car shapes and rejects every malformed one, with no box fallback',async()=>{
  // Valid: the hood, a fully measured car, an all-null measured shape and both refusals.
  for(const [name,options,status,full] of [['hood',{},'measured',false],['box',{top:()=>CAR.height},'measured',true],['all null',{top:()=>null},'measured',false],
    ['too few sweeps',{refusal:REFUSED('too-few-sweeps')},'refused',false],['not stationary',{refusal:REFUSED('not-stationary')},'refused',false]]) {
    const s=await loadVehicle(vehicleFixture(options));
    assert.deepEqual([s.carShape.status,s.carShape.allTopsAtLeastH,s.carShape.track,s.carShape.along,s.carShape.lateral],[status,full,'analytic',40,18],name);
    assert.deepEqual(s.carShape.refusal,options.refusal??null,name);
  }
  const SOLID='solid tops must be null or in [0.25 m, box height] with returns from at least 2 sweeps',REFUSAL='refused needs a known refusal, no solid tops and zero counts';
  const LISTS='shape and scoring sweeps must be increasing decimal timestamps with SHA-256 hashes',MEASURED='measured needs no refusal and at least 2 shape sweeps';
  const measured={
    'solid tops of the wrong length':[c=>{c.solidTopM.pop();},SOLID],'counts of the wrong length':[c=>{c.counts.returns.pop();},'counts'],
    'more return sweeps than returns':[c=>{c.counts.returnSweeps[0]=8;},'counts'],
    'a solid top with returns from 1 sweep':[c=>{c.counts.returnSweeps[0]=1;},SOLID],'a solid top above the box height':[c=>{c.solidTopM[0]=1.3001;},SOLID],
    'a solid top below 0.25 m':[c=>{c.solidTopM[0]=.249;},SOLID],'a solid top as text':[c=>{c.solidTopM[0]='0.9';},SOLID],
    'a track no averaged cuboid has':[c=>{c.track='other';},'track must name exactly one averaged annotated cuboid'],
    'a cuboid without its averaged pose':[(c,g)=>{delete g.cuboids.items[0].poseAverage;},'track must name exactly one averaged annotated cuboid'],
    'a box that differs from the cuboid':[c=>{c.box.length=4.001;},'box must equal the averaged cuboid sizes exactly'],
    'a grid not recomputed from the box':[c=>{c.grid.along=41;},'grid must be recomputed from the box'],
    'another grid order':[c=>{c.grid.order='index = j*along + i';},'grid must be recomputed from the box'],
    'schema shape-v0':[c=>{c.schema='shape-v0';},'schema, status, track or frame'],'an unknown status':[c=>{c.status='partial';},'schema, status, track or frame'],
    'an extra field':[c=>{c.boxFallback=true;},'has unexpected or missing fields'],'a missing field':[c=>{delete c.diagnostics;},'has unexpected or missing fields'],
    'a negative removed count':[c=>{c.removed.belowFloor=-1;},'removed counts'],'another sweep rule':[c=>{c.sweeps.rule.maxKept=41;},'sweep rule'],
    'a plan without its hash':[c=>{c.sweeps.planSha256='plan';},'sweep plan record'],
    'overlapping shape and scoring sweeps':[c=>{c.sweeps.scoring[0].timestampNs=c.sweeps.shape[0].timestampNs;},'sweep lists must be disjoint'],
    'a dropped sweep also used':[c=>{c.sweeps.downloadDropped[0].timestampNs=c.sweeps.shape[1].timestampNs;},'sweep lists must be disjoint'],
    'lists not adding up to kept':[c=>{c.sweeps.kept=7;},'sweep lists must add up to the kept count'],
    'decreasing timestamps':[c=>{c.sweeps.shape.reverse();},LISTS],'a numeric timestamp':[c=>{c.sweeps.shape[0].timestampNs=1000;},LISTS],
    'a sweep without its hash':[c=>{delete c.sweeps.scoring[0].sha256;},LISTS],
    'a dropped download after 3 attempts':[c=>{c.sweeps.downloadDropped[0].attempts=3;},'dropped downloads'],
    'measured from 1 shape sweep':[c=>{c.sweeps.shape.splice(1);c.sweeps.kept=4;},MEASURED],
    'measured with a refusal':[c=>{c.refusal=REFUSED('too-few-sweeps');},MEASURED],
    'a negative diagnostic count':[c=>{c.diagnostics.returnsBeyondEnds.front=-1;},'diagnostics'],
    'placement protocol 3.2':[c=>{c.protocol.placementProtocolVersion='3.2';},'protocol record'],
    'another code path':[c=>{c.code.path='scripts/evaluation/metric_f_car_top.py';},'code record'],
  };
  const refused={
    'refused with a non-null top':[c=>{c.solidTopM[0]=.9;c.counts.returns[0]=7;c.counts.returnSweeps[0]=3;},REFUSAL],
    'refused with returns':[c=>{c.counts.returns[0]=1;},REFUSAL],'refused with an unknown code':[c=>{c.refusal={code:'too-dark',reason:'too dark'};},REFUSAL],
    'refused with another sentence':[c=>{c.refusal.reason='not enough sweeps';},REFUSAL],'refused without a refusal':[c=>{c.refusal=null;},REFUSAL],
  };
  for(const [cases,refusal] of [[measured,null],[refused,REFUSED('too-few-sweeps')]])for(const [name,[mutate,detail]] of Object.entries(cases))
    await assert.rejects(loadVehicle(vehicleFixture({refusal,shape:mutate})),{message:`Invalid mapped ground: car shape ${detail}.`},name);
  // Presence: exactly when suggestions name the annotated stationary vehicle, and for that track.
  const presence='Invalid mapped ground: car shape must be present exactly when the suggestions name an annotated stationary vehicle.';
  await assert.rejects(loadVehicle(vehicleFixture({ground:g=>{g.carShape=null;}})),{message:presence},'null for an annotated vehicle');
  await assert.rejects(loadVehicle(vehicleFixture(),{suggest:false}),{message:presence},'a car shape without suggestions');
  const other=vehicleFixture({ground:g=>{
    g.cuboids.items.push({...structuredClone(g.cuboids.items[0]),track:'other',center:[-2,-1.4375+.75,-22],length:4.5,width:1.9,height:1.5});
    g.carShape=carShapeRecord({track:'other',length:4.5,width:1.9,height:1.5});
  }});
  await assert.rejects(loadVehicle(other),{message:'Invalid mapped ground: car shape track differs from the suggested vehicle.'},'track mismatch');
  // Earlier ground layers (schema 2: placement protocol v3.2, no car shape) are history.
  await assert.rejects(loadVehicle(vehicleFixture({ground:g=>{g.schemaVersion=2;delete g.carShape;}})),
    {message:'Invalid mapped ground: schemaVersion must be 3 (placement protocol v3.3 with the measured car shape; earlier ground layers are history).'},'schema 2');
  // Prepared shapes: an invalid record is refused, and null stays null (no box stand-in).
  assert.equal(prepareCarShape(null),null);
  for(const bad of [{...carShapeRecord({track:'t',...CAR}),solidTopM:[1]},{...carShapeRecord({track:'t',...CAR}),box:{...CAR,height:0}}])
    assert.throws(()=>prepareCarShape(bad),{message:'Invalid car shape.'});
  const mismatched={...boxOf({width:1.8,height:1.3,length:4}),shape:prepareCarShape(carShapeRecord({track:'t',length:4,width:1.8,height:1.31}))};
  assert.throws(()=>assessSightline(SPARSE,line([-3,0,0],[3,0,0]),mismatched),{message:'The measured car shape does not match the vehicle box.'});
});

test('shape-v1 12: phrases, facts and answers name "vehicle height not proven"; no new sentence uses withheld wording',async()=>{
  const s=await loadVehicle(vehicleFixture()),sweep=sweepOf(s);
  const runs=reason=>sweep.rows.reduce((out,r,i)=>{if(r.reason===reason){if(i&&sweep.rows[i-1].reason===reason)out.at(-1)[1]=r.setback;else out.push([r.setback,r.setback]);}return out;},[]);
  const notProven=runs('height-not-proven'),fmt=n=>String(Number(n.toFixed(2)));
  assert.equal(notProven.length,1);
  const phrase=sweepPhrase(sweep);
  assert.match(phrase,new RegExp(`; vehicle height not proven ${fmt(notProven[0][0])}-${fmt(notProven[0][1])} m$`));
  assert.match(phrase,/^blocked [\d.]+-[\d.]+ m; too close to call [\d.]+-[\d.]+ m; not blocked by the vehicle from [\d.]+ m \([\d.]+ ft\); vehicle height/);
  const facts=sweepFacts([sweep]),vehicle=facts.vehicles[0];
  assert.deepEqual(vehicle.heightNotProven_m,notProven.map(([a,b])=>[Number(a.toFixed(2)),Number(b.toFixed(2))]));
  assert.equal(vehicle.result,phrase);
  // The answer check accepts the label with the computed range; the current position's own phrase names it too.
  const allowed=allowedNumbersFrom(facts),[a,b]=vehicle.heightNotProven_m[0];
  for(const text of [`For your vehicle, the vehicle height is not proven from ${a} to ${b} m.`,`Vehicle height not proven ${a}-${b} m.`])
    assert.deepEqual(checkAnswer(text,allowed,{mapped:true}),{ok:true,reason:'',text},text);
  const position=createScenario(s,{...startInput(s),proposedSetback:(notProven[0][0]+notProven[0][1])/2});
  const positionFacts=scenarioFacts(s,position);
  assert.deepEqual([positionFacts.proposed.state,positionFacts.proposed.unknownReason,positionFacts.proposed.result],['unknown','vehicle height not proven','unknown (vehicle height not proven)']);
  // height-not-proven rows touch the line: "not blocked by the vehicle" needs a later plain row, and an undecided tail is
  // named.
  const r=(rows,stop=null)=>({rows:rows.map(([state,reason],i)=>({setback:i,state,gapM:-.3,...(reason&&{reason})})),stopsBlockingAt:stop});
  const hnp=['unknown','height-not-proven'];
  assert.equal(sweepShort(r([['blocked'],hnp,hnp])),'unknown from 1 m (vehicle height not proven)');
  assert.equal(sweepShort(r([hnp,hnp])),'unknown from 0 m (vehicle height not proven)');
  assert.equal(sweepShort(r([hnp,['unknown']],1)),'not blocked by the vehicle from 1 m (3.28 ft)');
  assert.equal(sweepPhrase(r([hnp,['unknown']],1)),'not blocked by the vehicle from 1 m (3.28 ft); vehicle height not proven at 0 m');
  assert.equal(sweepPhrase(r([hnp,hnp])),'vehicle height not proven 0-1 m');
  // Several runs of one reason are named once, with every range (as blocked runs are).
  assert.equal(sweepPhrase(r([hnp,['blocked'],hnp,hnp,['blocked'],['unknown']],5)),
    'blocked at 1 m, at 4 m; not blocked by the vehicle from 5 m (16.4 ft); vehicle height not proven at 0 m, 2-3 m');
  assert.doesNotMatch(sweepPhrase(r([['unknown'],hnp])),/not blocked/);
  // Every new sentence and label: no word built on "park", and no withheld pattern for a mapped scene.
  const texts=[UNKNOWN_REASONS['height-not-proven'],SHAPE_REASONS.blocked,SHAPE_REASONS.heightNotProvenMeasured,...Object.values(CAR_SHAPE_REFUSALS),
    ...Object.values(CAR_SHAPE_REFUSALS).map(SHAPE_REASONS.heightNotProvenRefused),'unknown (vehicle height not proven)'];
  for(const text of texts) {
    assert.doesNotMatch(text,/park/i,text);
    assert.deepEqual(checkAnswer(text,[2],{mapped:true}),{ok:true,reason:'',text},text);
  }
});

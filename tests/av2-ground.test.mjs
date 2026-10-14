import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {loadScene} from '../src/scene.mjs';
import {roadPoint,mappedGroundHit,mappedGroundHeight,placementFromPixels,placementPreview,createScenario,classifySightline,isMappedGround,roadTolerance,annotatedVehicle,comparisonStart,whatIfBox,ANNOTATED_FRONT_NUDGE_PX,
  evaluatedPosition,anchorPoint,assessSightline,prepareCarShape} from '../src/geometry.mjs';
import {planCamera} from '../src/plan-camera.mjs';
import * as THREE from 'three';
import {suggestedSetup,suggestionStatus,isStillSuggested,supportSummary,validateSuggestions,crossingText} from '../src/suggestions.mjs';
import {scenarioFacts,supportFacts,runTool,setbackSweep,applyDeclaration,allowedNumbersFrom,checkAnswer,sweepPhrase,withheldNote} from '../src/scenario-assistant.mjs';
import {SYSTEM_PROMPTS,TOOLS} from '../worker/nemotron.js';
import {scene as legacyScene,input as legacyInput,fetchAsset} from './scene-input.mjs';
import {carShapeRecord} from './car-shape-fixture.mjs';
import {historyApp} from './history-app.mjs';
import {analytic,loadAnalytic,FLAT} from './mapped-fixture.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);
const close3=(a,b,tolerance,what)=>a.forEach((n,i)=>near(n,b[i],tolerance,what));
const r2=n=>Number(n.toFixed(2));

// The analytic mapped-ground fixture (analytic, loadAnalytic, FLAT) lives in tests/mapped-fixture.mjs (shared with
// tests/car-shape.test.mjs).

test('mapped ground: runtime rule returns the first drivable hit, support class, hidden flag and each Unknown reason',async()=>{
  const s=await loadAnalytic(analytic());
  assert.equal(isMappedGround(s),true);assert.equal(roadTolerance(s),.08);assert.equal(s.ground.license.id,'CC-BY-NC-SA-4.0');
  const observed=mappedGroundHit(s,[2,5]);
  close3(observed.point,[0,-1.4375,-28.75/3],1e-12);
  assert.deepEqual([observed.supportClass,observed.supportLabel,observed.hidden,observed.hitKind,observed.terrainCityZ,observed.terrainCell],
    ['observed-nearby','observed nearby',false,'top',FLAT,[10,14]]);
  // The ground layer's fitted height band (0.075 m) / sin(depression angle); no float16 half-step term (protocol v3).
  near(observed.alongRayUncertaintyM,.075/(.15/Math.hypot(.15,1)),1e-12);
  // Crop positions may be fractional (protocol v3): row 4.5 descends 0.125 m per metre, so it meets the ground at 11.5 m.
  close3(mappedGroundHit(s,[2,4.5]).point,[0,-1.4375,-11.5],1e-12);close3(mappedGroundHit(s,[2.5,5]).point,[.025/.15*1.4375,-1.4375,-1.4375/.15],1e-12);
  const hidden=mappedGroundHit(s,[2,4]);
  close3(hidden.point,[0,-1.4375,-14.375],1e-12);
  assert.deepEqual([hidden.supportClass,hidden.supportLabel,hidden.hidden],['map-only','mapped, not observed',true]);
  const step=mappedGroundHit(s,[2,7]);// enters the raised cell below its top
  close3(step.point,[0,-1.25,-5],1e-12);assert.equal(step.hitKind,'step');assert.equal(step.hidden,false);
  assert.deepEqual(roadPoint(s,[2,4],'crossing'),hidden.point);
  // User positions are whole pixels (v3.1 review): roadPoint and anchorPoint refuse a fractional one, while
  // mappedGroundHit itself still evaluates it (a suggestion's pixelUV, the loader and the harness).
  for(const position of [[2,4.5],[2.5,5],[2.000001,4]]) {
    assert.ok(mappedGroundHit(s,position).point,String(position));
    assert.throws(()=>roadPoint(s,position,'crossing'),{message:'Select a pixel inside the recorded image.'},String(position));
    for(const key of ['crossing','parking','target'])assert.throws(()=>anchorPoint(s,key,position),{message:'Select a pixel inside the recorded image.'},`${key} ${position}`);
  }
  // Only a placed suggestion is evaluated at its pixelUV. An unknown one may still carry pixel and pixelUV (e.g. refused
  // as too far); a click on that pixel is evaluated as given.
  const marker={pixel:[2,4],pixelUV:[2,4.5]},withPerson=person=>({...s,suggestions:{person}});
  assert.deepEqual(evaluatedPosition(withPerson({status:'declared-default',...marker}),'target',[2,4]),[2,4.5]);
  close3(anchorPoint(withPerson({status:'declared-default',...marker}),'target',[2,4]).point,[0,-1.4375,-11.5],1e-12);
  const unknownPerson=withPerson({status:'unknown',supportClass:'unknown',reason:'Too far to place (over 15 m).',...marker});
  assert.deepEqual(evaluatedPosition(unknownPerson,'target',[2,4]),[2,4]);
  assert.deepEqual(anchorPoint(unknownPerson,'target',[2,4]).point,hidden.point);
  for(const [pixel,reason] of [[[2,1],'Pixel ray does not descend toward the ground.'],[[2,2],'Pixel ray does not descend toward the ground.'],
    [[2,3],'No mapped ground within 20 m along the pixel ray.'],[[0,4],'Ray crosses a cell without mapped ground height before reaching the ground.'],
    [[7,4],'First mapped-ground hit is outside the mapped drivable area.'],[[8,4],'Select a pixel inside the recorded image.'],[[7.01,4],'Select a pixel inside the recorded image.'],
    [[2,-.5],'Select a pixel inside the recorded image.'],[[2,NaN],'Select a pixel inside the recorded image.'],[['2',4],'Select a pixel inside the recorded image.'],
    // Row 3.9 meets the ground 15.13 m away: beyond the 15 m placement limit (checked before the drivable test).
    [[2,3.9],'Too far to place (over 15 m).']])
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
  // This person point lies on the box's front face (same depth as the front pixel), so the box holds the person: Unknown
  // with that reason, never Blocked (it was reported Blocked before M1).
  assert.deepEqual([scenario.currentVisibility.state,scenario.currentVisibility.reasonCode],['unknown','person-inside-vehicle']);
  // A person at the crossing: the line cuts deep into the box (Blocked) and passes over a low box (Unknown, no reason).
  const atCrossing={...input,pixels:{...pixels,target:[2,4]}},deep=createScenario(s,atCrossing);
  assert.equal(deep.currentVisibility.state,'blocked');assert.ok(deep.currentVisibility.gapM<=-.1);
  const low=createScenario(s,{...atCrossing,vehicle:{...input.vehicle,height:.5}});
  assert.deepEqual({...low.currentVisibility,gapM:undefined},{state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null,gapM:undefined});
  assert.ok(low.currentVisibility.gapM>0);
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
  // A front beyond the crossing asks for the camera side, in placement and in the preview, naming the measured gap.
  for(const run of [()=>placementFromPixels(s,{...pixels,crossing:[2,5],parking:[2,4]}),()=>placementPreview(s,{crossing:[2,5],parking:[2,4]},input.vehicle,'')])
    assert.throws(run,{message:'Place the current vehicle on the camera side of the crossing (its front is 4.79 m past the crossing along the lane).'});
  assert.throws(()=>placementFromPixels(s,{...pixels,parking:[2,4]}),{message:'Current vehicle front must be separated from the crossing (its front is 0 m from it along the lane; more than 0.08 m is needed).'});
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
  // shape-v1: the annotated vehicle carries its LiDAR shape (scene.mjs prepares it while loading). Every solid top equals
  // the box height here, so every result is the box result (tests/car-shape.test.mjs covers other shapes).
  s.carShape=prepareCarShape(carShapeRecord({track:'analytic',length:4,width:1.8,height:1.6}));
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
  assert.deepEqual({...scenario.baseline,axis:undefined},{center:[2.2,-.6,-12],axis:undefined,vehicle:{width:1.8,height:1.6,length:4},pose:'annotated',shape:s.carShape});
  // The what-if starts as the annotated box exactly, so both results are the same (this box lies beside the line: Unknown).
  assert.deepEqual(scenario.proposed,scenario.baseline);assert.equal(scenario.baselineSetback,start.proposedSetback);
  assert.equal(scenario.currentVisibility.state,'unknown');assert.deepEqual(scenario.proposedVisibility,scenario.currentVisibility);
  // Its own centre line (x 2.2 + 2 sin 10 deg) runs on mapped drivable ground to the camera; the turned box reaches
  // 4 cos 10 deg + 0.9 sin 10 deg (its rear corner) behind its front-face centre along the lane axis.
  near(scenario.limits.maximumSetback,14.375-4*cos-.9*sin,1e-12);
  const facts=scenarioFacts(s,scenario),annotated={pose:'annotated',turnedFromLane_deg:10,setback_m:.41,setback_ft:1.33,state:'unknown',result:'not blocked by the vehicle (rest unknown)',targetInside:false,eyeInside:false};
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
  // Between that and the slider minimum (corner within the road tolerance) a typed or declared setback is refused too.
  assert.throws(()=>createScenario(s,{...input,proposedSetback:.9*sin+1e-6}),{message:'Proposed setback must be at least 0.24 m from the crossing (the slider minimum).'});
  assert.throws(()=>createScenario(s,{...input,proposedSetback:scenario.limits.minimumSetback-1e-6}),/at least 0\.24 m/);
  assert.ok(nearest(createScenario(s,{...input,proposedSetback:scenario.limits.minimumSetback}).proposed)>0);
  // The distance check sweeps the same translated pose; every row agrees with the comparison.
  const sweep=setbackSweep(s,{pixels,driverDistance:p.captureDistance},{label:'your vehicle',width:1.8,height:1.6,length:4},1.1);
  assert.deepEqual(sweep.range,[scenario.limits.minimumSetback,scenario.limits.maximumSetback]);
  for(const row of sweep.rows)assert.equal(createScenario(s,{...input,proposedSetback:row.setback}).proposedVisibility.state,row.state,`setback ${row.setback}`);
  // A turned current vehicle whose nearer corner is within the road tolerance of the crossing is refused, like a front.
  const close=await loadAnalytic(analytic({ground:g=>{Object.assign(g.cuboids.items[0],{center:[2.2,-.6,.15-14.375+2*cos],length:4,width:1.8,height:1.6,
    worldFromBox:[[sin,-cos,0],[0,0,1],[-cos,-sin,0]]});}}));
  close.suggestions=s.suggestions;close.carShape=s.carShape;
  assert.throws(()=>createScenario(close,{...input,proposedSetback:1}),
    {message:'Current vehicle front must be separated from the crossing (its nearest front corner is 0 m from it along the lane; more than 0.08 m is needed).'});
  close3(placementPreview(s,pixels,fields,'1.1').vehicle.center,pose.center,0);
  // Nudging the front pixel by up to ANNOTATED_FRONT_NUDGE_PX (2) on each axis keeps the annotated vehicle exactly (plan
  // M1d): same box, setback, limits and results. The nudged pixel only marks the vehicle, so a nudge whose ground hit
  // lands on the crossing ([2,4]; a declared front there is refused) is not refused.
  assert.equal(ANNOTATED_FRONT_NUDGE_PX,2);
  const same=sc=>[sc.baseline,sc.proposed,sc.baselineSetback,sc.limits,sc.currentVisibility,sc.proposedVisibility];
  for(const parking of [[2,6],[2,7],[3,5],[4,7],[4,4],[1,6],[2,4]]) {
    const nudged={...pixels,parking};
    assert.notEqual(annotatedVehicle(s,nudged,fields),null,String(parking));
    const start2=comparisonStart(s,nudged,fields);assert.equal(start2.proposedSetback,start.proposedSetback,String(parking));
    assert.deepEqual(same(createScenario(s,{...input,pixels:nudged})),same(scenario),String(parking));
    close3(placementPreview(s,nudged,fields,'1.1').vehicle.center,pose.center,0);
  }
  // Protocol v3.1 declaredReplacement: while the front pixel names the stationary vehicle (within the nudge), the vehicle
  // front point is that vehicle's front-face centre on the mapped ground, never the pixel's own ray hit. The front of the
  // turned box: centre + 2 m along its heading (x 2.2 + 2 sin 10 deg, z -12 - 2 cos 10 deg), at the ground height -1.4375.
  const frontPoint=[2.2+2*sin,-1.4375,-12-2*cos];
  close3(p.parking,frontPoint,1e-12,'front point');
  assert.deepEqual(p.support.parking,{supportClass:'map-only',supportLabel:'mapped, not observed',hidden:true,alongRayUncertaintyM:null});
  for(const parking of [[2,4],[4,7]])close3(placementFromPixels(s,{...pixels,parking}).parking,frontPoint,1e-12,String(parking));
  // Moving the front pixel further (3 px) makes both the current vehicle and the what-if declared boxes on the lane axis
  // at that pixel's ground hit. Changing a size or a declaration (even at the annotated sizes) while the front pixel still
  // names the stationary vehicle replaces it with a declared box on the lane axis whose front is the stationary vehicle's
  // front point: its setback equals the annotated box's (here 0.41 m; this box lies beside the line, so Unknown).
  const lane={...input,proposedSetback:p.baselineSetback};
  for(const [name,next] of [['moved',{...lane,pixels:{...pixels,parking:[2,8]}}],['resized',{...lane,vehicle:{...fields,width:'1.9'}}],
    ['declared',{...lane,vehicleDeclared:true}],['declared nudged',{...lane,pixels:{...pixels,parking:[3,6]},vehicleDeclared:true}]]) {
    const declared=createScenario(s,next),q=placementFromPixels(s,next.pixels);
    if(!name.startsWith('declared'))assert.equal(annotatedVehicle(s,next.pixels,next.vehicle),null,name);
    assert.equal(declared.baseline.pose,'declared',name);assert.equal(declared.proposed.pose,'declared',name);
    assert.deepEqual(declared.baseline.axis,declared.axis,name);assert.deepEqual(declared.proposed.axis,declared.axis,name);
    close3(declared.baseline.center,[q.parking[0],q.parking[1]+Number(next.vehicle.height)/2,q.parking[2]+2],1e-12,name);
    assert.equal(declared.baselineSetback,q.baselineSetback,name);
    assert.equal(Object.hasOwn(scenarioFacts(s,declared).current,'pose'),false,name);
    if(name==='moved') {
      close3(q.parking,mappedGroundHit(s,[2,8]).point,0,name);assert.equal(declared.currentVisibility.state,'blocked',name);
    }else{
      close3(q.parking,frontPoint,1e-12,name);near(declared.baselineSetback,setback,1e-12,name);
      assert.equal(declared.currentVisibility.state,'unknown',name);
    }
  }
  assert.equal(placementPreview(s,pixels,fields,'1.1',true).vehicle.annotated,undefined);
  close3(placementPreview(s,pixels,fields,'1.1',true).vehicle.center,[frontPoint[0],frontPoint[1]+.8,frontPoint[2]+2],1e-12);
  const van=setbackSweep(s,{pixels,driverDistance:p.captureDistance,vehicleDeclared:true},{label:'van',width:1.8,height:1.6,length:4},1.1);
  assert.deepEqual([van.error,van.rows[0].state],[undefined,'unknown']);
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
  // An annotated vehicle without its LiDAR shape (or with another track's) is an error, never the box.
  for(const shape of [null,undefined,prepareCarShape(carShapeRecord({track:'other',length:4,width:1.8,height:1.6}))]) {
    const bare={...s,carShape:shape};
    assert.throws(()=>createScenario(bare,input),{message:'The measured car shape is missing for the annotated vehicle.'},String(shape?.track));
    assert.throws(()=>annotatedVehicle(bare,pixels,fields),{message:'The measured car shape is missing for the annotated vehicle.'},String(shape?.track));
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
    // Ground schema 3 (placement protocol v3.3): the placement limit and the along-ray band are required and bounded; schema
    // 1 (protocol v1/v2) and schema 2 (protocol v3-v3.2, no car shape) layers are history and refused.
    'schema 1 (protocol v2) layer':x=>{x.schemaVersion=1;delete x.placementRangeLimitMeters;delete x.alongRayUncertainty;delete x.carShape;},
    'schema 2 (protocol v3.2) layer':x=>{x.schemaVersion=2;delete x.carShape;},'no car shape key':x=>{delete x.carShape;},
    'no placement limit':x=>{delete x.placementRangeLimitMeters;},'placement limit beyond the ray range':x=>{x.placementRangeLimitMeters=21;},
    'no band':x=>{delete x.alongRayUncertainty;},'zero band':x=>{x.alongRayUncertainty.heightBandMeters=0;},'band over 1 m':x=>{x.alongRayUncertainty.heightBandMeters=1.5;},
    'unknown band method':x=>{x.alongRayUncertainty.method='guess';},'fixed band without its development source':x=>{x.alongRayUncertainty.method='fixed';},
    'fixed band recorded as a fit':x=>{Object.assign(x.alongRayUncertainty,{method:'fixed',fittedOn:'dev'});},
    'averaged pose without its spread':x=>{x.cuboids.items[0].poseAverage={rule:'averageAllAnnotations',rows:3};},
    'averaged pose with another rule':x=>{x.cuboids.items[0].poseAverage={rule:'latest',rows:3,centreSpreadM:{alongStd:0,lateralStd:0},yawSpreadDeg:{std:0}};},
  };
  for(const [name,mutate] of Object.entries(invalid))await assert.rejects(loadAnalytic(analytic({ground:mutate})),/Invalid mapped ground: /,name);
  for(const version of [1,2])await assert.rejects(loadAnalytic(analytic({ground:x=>{x.schemaVersion=version;}})),
    {message:'Invalid mapped ground: schemaVersion must be 3 (placement protocol v3.3 with the measured car shape; earlier ground layers are history).'},`schema ${version}`);
  // A fixed (held-out) band with its test record, and an averaged pose with its spread, load.
  await loadAnalytic(analytic({ground:x=>{Object.assign(x.alongRayUncertainty,{method:'fixed',fittedOn:'dev'});delete x.alongRayUncertainty.fit;x.alongRayUncertainty.heldOutResiduals={count:0};
    x.cuboids.items[0].poseAverage={rule:'averageAllAnnotations',rows:3,centreSpreadM:{alongStd:.1,lateralStd:.05},yawSpreadDeg:{std:.01}};}}));
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

test('mapped ground: a box over a real annotated object, or a person or driver inside one, is Unknown with that reason',async()=>{
  // The analytic cuboid (track analytic, at z -11.25 to -12.75) is a real object while no suggestion names it.
  // The person stands at the crossing (z -14.375), so no tested box holds the person.
  const s=await loadAnalytic(analytic()),pixels={crossing:[2,4],parking:[2,5],target:[2,4]},p=placementFromPixels(s,pixels);
  const input={pixels,vehicle:{width:1.9,height:1.5,length:4},targetHeight:1.1,proposedSetback:p.baselineSetback,driverDistance:p.captureDistance,vehicleDeclared:true};
  const sweep=setbackSweep(s,{pixels,driverDistance:p.captureDistance},{label:'van',...input.vehicle},1.1);
  // A what-if box starting at setback s spans z -14.375 + s to -10.375 + s, so it overlaps the cuboid for s <= 3.125.
  // Only the last row (the box reaching the camera) holds the driver eye.
  assert.deepEqual(sweep.rows.filter(r=>r.state==='overlap').map(r=>[r.setback,r.person,r.driver]),[[sweep.range[1],false,true]]);
  for(const row of sweep.rows.slice(0,-1)) {
    assert.equal(row.reason==='overlaps-real-object',row.setback<=3.125,`setback ${row.setback}`);
    const v=createScenario(s,{...input,proposedSetback:row.setback}).proposedVisibility;
    assert.deepEqual([v.state,v.reasonCode],[row.state,row.reason],`setback ${row.setback}`);
  }
  const over=createScenario(s,{...input,proposedSetback:2}),facts=scenarioFacts(s,over);
  assert.deepEqual([over.proposedVisibility.state,over.proposedVisibility.reasonCode,over.proposedVisibility.object],['unknown','overlaps-real-object',{category:'REGULAR_VEHICLE',track:'analytic'}]);
  assert.equal(over.proposedVisibility.reason,'The vehicle box overlaps a real annotated object (regular vehicle), so this placement is not tested.');
  assert.deepEqual([facts.proposed.state,facts.proposed.unknownReason],['unknown','overlaps a real object']);
  assert.match(sweepPhrase(sweep),/overlaps a real object 0\.08-3\.08 m/);
  // Named by the suggestion as the stationary vehicle itself, the cuboid is no longer another object.
  s.suggestions={vehicle:{status:'annotated',track:'analytic',pixel:[7,9],dimensions:{width:1.5,height:1.5,length:1.5}}};
  assert.equal(createScenario(s,{...input,proposedSetback:2}).proposedVisibility.reasonCode,undefined);
  // A person standing in a real object, or a driver eye inside one: Unknown at every setback, never Blocked.
  const atCrossing=await loadAnalytic(analytic({ground:g=>{g.cuboids.items[0].center=[0,-.75,-14.375];}}));
  const person=createScenario(atCrossing,input);
  for(const v of [person.currentVisibility,person.proposedVisibility])assert.deepEqual([v.state,v.reasonCode],['unknown','person-inside-object']);
  assert.equal(scenarioFacts(atCrossing,person).current.unknownReason,'person inside a real object');
  const personSweep=setbackSweep(atCrossing,{pixels,driverDistance:p.captureDistance},{label:'van',...input.vehicle},1.1);
  assert.ok(personSweep.rows.every(r=>r.state!=='blocked'&&r.reason));assert.equal(personSweep.stopsBlockingAt,null);
  const eye=await loadAnalytic(analytic({ground:g=>{g.cuboids.items.push({...g.cuboids.items[0],track:'around-eye',category:'BOLLARD',center:[0,-.2,0],length:1,width:1,height:1});}}));
  const driver=createScenario(eye,input);
  for(const v of [driver.currentVisibility,driver.proposedVisibility])assert.deepEqual([v.state,v.reasonCode],['unknown','driver-inside-object']);
  assert.match(driver.currentVisibility.reason,/real annotated object \(bollard\)/);
});

test('mapped ground: an edited vehicle front or crossing gives a declared result or a refusal that names the gap',async()=>{
  // Analytic fixture: front pixel rows 5 and 6 hit the ground 9.58 m and 7.19 m from the camera, the crossing row 4 at
  // 14.38 m; swapping them puts the front past the crossing.
  const s=await loadAnalytic(analytic()),base={crossing:[2,4],parking:[2,5],target:[3,5]};
  const refusal=/^(?:Place the current vehicle on the camera side of the crossing \(its (?:front|nearest front corner) is \d+(?:\.\d+)? m past the crossing along the lane\)|Current vehicle front must be separated from the crossing \(its (?:front|nearest front corner) is \d+(?:\.\d+)? m from it along the lane; more than 0\.08 m is needed\))\.$/;
  for(const pixels of [{...base,parking:[2,6]},{...base,crossing:[2,5],parking:[2,4]},{...base,parking:[2,4]}]) {
    try{
      const start=comparisonStart(s,pixels,{width:1.9,height:1.5,length:4});
      const v=createScenario(s,{pixels,vehicle:{width:1.9,height:1.5,length:4},targetHeight:1.1,...start});
      assert.equal(v.baseline.pose,'declared');assert.ok(['blocked','unknown'].includes(v.currentVisibility.state));
    }catch(e){assert.match(e.message,refusal,JSON.stringify(pixels));}
  }
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

// ------------------------------------------------------------------ AV2 8c local data (current v3.3, history v3.2)
// The prepared AV2 scene is CC BY-NC-SA data kept out of public/ and dist/; it lives in ignored local evaluation folders.
// The current folder is grounded-v3.3/ (protocol v3.3, shape-v1, app contract A13): grounded-v3.2 plus the recorded
// vehicle's LiDAR shape (ground schema 3). The current app and scripts/evaluation/pao-baseline.mjs read it.
// grounded-v3.2/ (app contract sections A0-A12) is history: the same geometry and results as grounded-v3.1/, with the
// crosswalk paint-offset bound filled in ("±0.18 m"). The current app refuses it (ground schema 2, no car shape; tested
// first below), so the v3.2 contract tests that follow run against the app source of commit c7a39eb
// (tests/history-app.mjs), the code that produced the v3.2 numbers. grounded-v3.1/ is older history; grounded-v3/ is
// refused, and grounded-v2/ and grounded/ use earlier schemas. Files from two grounded folders are never mixed.
const AV2=new URL('../.local/evaluation/av2-map-first-2026-10-11/',import.meta.url),ID='av2-pao-8c52d911-2658849',V32_DIR='grounded-v3.2/'; // history (see above)
const local=existsSync(new URL(V32_DIR+'scene.json',AV2))&&existsSync(new URL('assembled/positions.bin',AV2));
const skip=local?false:'AV2 8c data is local-only and not present';
const GROUNDED=new Set(['scene.json','ground.json','terrain.bin','suggestions.json']);
const av2Bytes=name=>readFile(new URL((GROUNDED.has(name)?V32_DIR:'assembled/')+name,AV2));
// Serves the contract's directory: assembled buffers plus every grounded-v3.2 file, with optional replacements.
const av2Fetch=(replace={})=>async path=>{
  const name=path.split('/').at(-1);
  if(Object.hasOwn(replace,name))return replace[name]===null?new Response(null,{status:404}):new Response(replace[name]);
  if(!path.startsWith(`/scenes/${ID}/`)||['catalog-entry.json','LICENSE-AND-ATTRIBUTION.md'].includes(name))return new Response(null,{status:404});
  try{return new Response(await av2Bytes(name));}catch{return new Response(null,{status:404});}
};
const entry=local?JSON.parse(await readFile(new URL(V32_DIR+'catalog-entry.json',AV2),'utf8')):null;

test('AV2 8c: the current app refuses the grounded-v3.2 ground layer (schema 2, no car shape)',{skip},async()=>{
  assert.equal(JSON.parse(await av2Bytes('ground.json')).schemaVersion,2);
  await assert.rejects(loadScene(av2Fetch(),ID,entry.metadataSha256,undefined,entry.suggestions,entry.ground),
    {message:'Invalid mapped ground: schemaVersion must be 3 (placement protocol v3.3 with the measured car shape; earlier ground layers are history).'});
});

// App contract A13: grounded-v3.3 is grounded-v3.2 plus the recorded stationary vehicle's shape-v1 LiDAR shape (ground
// schemaVersion 3, placement protocol v3.3). The current app loads it; the 21 runtime vectors (runtime-test-vectors-v3.3.json,
// equal to the v3.2 vectors value for value) reproduce; the first load is unchanged (the line misses the box, so step 6
// never runs); the setback-sweep counts are those of the 8.2 check (scene-builder-v4.2/allowed-changes-8.2.json).
const V33_DIR='grounded-v3.3/',local33=existsSync(new URL(V33_DIR+'ground.json',AV2))&&existsSync(new URL('assembled/positions.bin',AV2));
const skip33=local33?false:'AV2 8c grounded-v3.3 data is local-only and not present';
const v33Fetch=async path=>{
  const name=path.split('/').at(-1);
  if(!path.startsWith(`/scenes/${ID}/`)||['catalog-entry.json','LICENSE-AND-ATTRIBUTION.md'].includes(name))return new Response(null,{status:404});
  try{return new Response(await readFile(new URL((GROUNDED.has(name)?V33_DIR:'assembled/')+name,AV2)));}catch{return new Response(null,{status:404});}
};
test('AV2 8c grounded-v3.3 (shape-v1): the current app loads the pinned layer with its measured car shape and reproduces the 21 runtime vectors',{skip:skip33},async()=>{
  const e=JSON.parse(await readFile(new URL(V33_DIR+'catalog-entry.json',AV2),'utf8'));
  assert.equal(e.metadataSha256,'cb048d0f3453638158feca8bad497a47e6271679e6b546b3b5e4b4a8297f1908');
  assert.deepEqual(e.ground,{file:'ground.json',sha256:'bb3226a74216ca0848f386ff60d69f1997e14b2576bffef38dfd23612898f316'});
  assert.deepEqual(e.suggestions,{file:'suggestions.json',sha256:'50049872e11ac33d9bc1cb9efa761c71e03c2b45e26aa10bb05f75cf33783c8c'});
  const s=await loadScene(v33Fetch,ID,e.metadataSha256,undefined,e.suggestions,e.ground);
  assert.equal(s.ground.schemaVersion,3);assert.equal(s.ground.protocol.sha256,'67117e04f0596866aa0f8feb8cb01e054865557576ea67237db24a9a9cad5af9');
  assert.equal(s.ground.terrain.sha256,'2f7a2914ddfb89593a086514ad9ceab6c8f7eb25087e467cae4c4fc371b8f082');
  const raw=s.ground.carShape,shape=s.carShape;
  assert.equal(raw.protocol.shapeProtocolSha256,'07b9535ffc6ca3d5ba0c8d7b9e8f471cd5642cbfa8e1b0da853ab2c40bb85825');
  assert.deepEqual([shape.status,shape.refusal,shape.track,shape.along,shape.lateral,shape.allTopsAtLeastH],['measured',null,'396f15b5-9cf3-462b-9164-4b6f71c4e7d6',41,19,false]);
  assert.deepEqual([raw.solidTopM.length,raw.solidTopM.filter(v=>v!==null).length,raw.sweeps.shape.length,raw.sweeps.scoring.length],[779,384,8,8]);
  const vectors=JSON.parse(await readFile(new URL('runtime-test-vectors-v3.3.json',AV2),'utf8'));
  assert.deepEqual([vectors.groundSha256,vectors.suggestionsSha256,vectors.protocolSha256],[e.ground.sha256,e.suggestions.sha256,s.ground.protocol.sha256]);
  assert.deepEqual(vectors.vectors,JSON.parse(await readFile(new URL('runtime-test-vectors-v3.2.json',AV2),'utf8')).vectors);
  assert.equal(vectors.vectors.length,21);
  for(const v of vectors.vectors) {
    if(v.supportClass==='unknown'){assert.throws(()=>mappedGroundHit(s,v.position),{message:v.reason},`${v.kind} ${v.position}`);continue;}
    const hit=mappedGroundHit(s,v.position);
    close3(hit.point,v.groundPoint,1e-6,String(v.position));
    assert.deepEqual([hit.supportClass,hit.hidden,hit.hitKind,hit.terrainCityZ,hit.terrainCell],[v.supportClass,v.hidden,v.hitKind,v.terrainCityZ,v.terrainCell],String(v.position));
    near(hit.rangeM,v.rangeM,1e-6);near(hit.alongRayUncertaintyM,v.alongRayUncertaintyM,1e-6);near(Math.hypot(hit.point[0],hit.point[2]),v.horizontalRangeM,1e-6);
  }
  const setup=suggestedSetup(s,s.suggestions),scenario=createScenario(s,setup.input);
  assert.equal(setup.error,'');assert.equal(scenario.whatIf.shape,shape);assert.equal(scenario.baseline.shape,shape);
  for(const v of [scenario.currentVisibility,scenario.proposedVisibility]){assert.equal(v.state,'unknown');assert.equal(v.reasonCode,undefined);near(v.gapM,.06241,5e-5,'first-load gap');}
  const vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,Number(setup.vehicle[k])]));
  for(const [height,want] of [[1.1,{rows:57,blocked:12,heightNotProven:26}],[1.5,{rows:57,blocked:0,heightNotProven:17}]]) {
    const rows=setbackSweep(s,{pixels:setup.pixels,driverDistance:setup.input.driverDistance},{label:'your vehicle',...vehicle},height).rows;
    assert.deepEqual({rows:rows.length,blocked:rows.filter(r=>r.state==='blocked').length,heightNotProven:rows.filter(r=>r.reason==='height-not-proven').length},want,`target ${height} m`);
  }
});

// The rest of this file: the app source of commit c7a39eb (placement protocol v3.2), whose names shadow the imports.
{
const history=local?await historyApp():null;
const {loadScene}=history?.scene??{};
const {roadPoint,mappedGroundHit,mappedGroundHeight,placementFromPixels,placementPreview,createScenario,classifySightline,isMappedGround,roadTolerance,annotatedVehicle,
  comparisonStart,whatIfBox,ANNOTATED_FRONT_NUDGE_PX,evaluatedPosition,anchorPoint,assessSightline}=history?.geometry??{};
const {suggestedSetup,suggestionStatus,isStillSuggested,supportSummary,validateSuggestions,crossingText}=history?.suggestions??{};
const {scenarioFacts,supportFacts,runTool,setbackSweep,applyDeclaration,allowedNumbersFrom,checkAnswer,sweepPhrase,withheldNote}=history?.assistant??{};
const loadAV2=(replace,overrides={})=>{const e={...entry,...overrides};return loadScene(av2Fetch(replace),ID,e.metadataSha256,undefined,e.suggestions,e.ground);};
// Contract section A5 (grounded-v3.2; the same geometry as grounded-v3.1 and grounded-v3), asserted to 1e-6 m. pixel: the integer anchor
// (marker and nudge anchor); pixelUV: the exact projection the app evaluates while the suggestion is untouched;
// anchorResidual: how far the integer anchor alone would land from the mapped point.
const EXPECTED={
  crossing:{pixel:[371,189],pixelUV:[371.14534464547864,188.52832143204816],groundPoint:[7.624036104136133,-1.2992260826921616,16.000816858496343],hidden:true,
    alongRay:1.0259120621499416,range:17.77188946258275,horizontalRange:17.724335436238846,residual:1.8e-15,anchorResidual:0.9763681089631214},
  parking:{pixel:[374,191],pixelUV:[374.45809937658737,190.84864102416427],groundPoint:[6.948579963694091,-1.2367260826921616,14.723709059401997],hidden:true,
    alongRay:0.9901881039636496,range:16.327886065911244,horizontalRange:16.280981910737676,residual:1.0284829677138683,anchorResidual:1.0657773365933012},
  target:{pixel:[484,185],pixelUV:[484.3272234524834,185.2703140345646],groundPoint:[5.704329063935546,-1.2523510826921616,17.216837073671623],hidden:false,
    alongRay:1.0887766320635415,range:18.180407919662684,horizontalRange:18.137222744704147,residual:4.4e-15,anchorResidual:0.30605104214648887},
};
const ROLE_OF={crossing:'crossing',parking:'vehicle',target:'person'};
const TRAVEL=[0.5833967978610233,0,0.8121872790468369];
// The averaged stationary-vehicle front point (protocol v3.1 declaredReplacement; suggestions vehicle.mappedPoint).
const FRONT=[7.387527453571465,-1.3148510826921616,15.653817825089043];

test('AV2 8c: the contract pins load and the runtime rule reproduces every prepared vector, fractional positions and the 30 m limit',{skip},async()=>{
  assert.deepEqual(Object.keys(entry).sort(),['ground','id','label','metadataSha256','suggestions']);
  assert.equal(entry.metadataSha256,'c2453ac1afbc35aca794ccd93f42c6f8c43f40da87bb9b34bc6fba18ac17206a');
  assert.deepEqual(entry.ground,{file:'ground.json',sha256:'556bc31b922c59853fe9e6e1d4cf9bcb8d8f7f6863de46835272f53235236115'});
  assert.deepEqual(entry.suggestions,{file:'suggestions.json',sha256:'86b2e5a134fbed2fc8ee778da78341aa60da86df044b08ffbc4ca87be32f7dae'});
  const s=await loadAV2();
  assert.equal(s.measurements.groundPolicy,'mappedGround');assert.equal(s.ground.terrain.sha256,'2f7a2914ddfb89593a086514ad9ceab6c8f7eb25087e467cae4c4fc371b8f082');
  assert.equal(s.ground.schemaVersion,2);assert.equal(s.ground.protocol.sha256,'1cef13e625ba8d0c0c00203c0bda6e256536173c4607ac79807e3e40dd173cf7');
  assert.deepEqual([s.ground.placementRangeLimitMeters,s.ground.alongRayUncertainty.heightBandMeters,s.ground.alongRayUncertainty.method],[30,.075,'fitThisDevelopmentScene']);
  assert.equal(s.ground.support.count,3055);assert.equal(s.groundSupportXZ.length,2*3055);assert.equal(s.terrainHeights.length,409*408);
  assert.equal(s.ground.license.attribution,'(c) 2021 Argo AI, LLC');assert.equal(roadTolerance(s),.08);
  for(const [key,e] of Object.entries(EXPECTED)) {
    const raw=s.suggestions[ROLE_OF[key]];
    assert.deepEqual([raw.pixel,raw.pixelUV],[e.pixel,e.pixelUV],key);
    // Untouched, the app evaluates the exact sub-pixel position; any other pixel is evaluated as given.
    assert.deepEqual(evaluatedPosition(s,key,e.pixel),e.pixelUV,key);
    assert.deepEqual(evaluatedPosition(s,key,[e.pixel[0],e.pixel[1]+1]),[e.pixel[0],e.pixel[1]+1],key);
    const hit=mappedGroundHit(s,e.pixelUV);
    close3(hit.point,e.groundPoint,1e-6,key);
    assert.deepEqual([hit.supportClass,hit.supportLabel,hit.hidden,hit.hitKind],['map-only','mapped, not observed',e.hidden,'top'],key);
    near(hit.alongRayUncertaintyM,e.alongRay,1e-6,key);near(hit.rangeM,e.range,1e-6,key);near(Math.hypot(hit.point[0],hit.point[2]),e.horizontalRange,1e-6,key);
    near(Math.hypot(hit.point[0]-raw.mappedPoint[0],hit.point[2]-raw.mappedPoint[2]),e.residual,1e-6,key);
    const anchor=mappedGroundHit(s,e.pixel).point;near(Math.hypot(anchor[0]-raw.mappedPoint[0],anchor[2]-raw.mappedPoint[2]),e.anchorResidual,1e-6,key);
  }
  // The 21 runtime vectors: the three pixelUV positions, the three integer anchors, the 13 v2 vector pixels and a 30 m
  // boundary pair 0.002 px apart on column 373 (v 150.595 is refused, v 150.597 is placed 29.998 m away). Protocol v3.2
  // changes no runtime rule: its vectors equal the v3.1 ones exactly (v3.2-app/make-runtime-vectors-v3.2.py checks this).
  const vectors=JSON.parse(await readFile(new URL('runtime-test-vectors-v3.2.json',AV2),'utf8'));
  assert.deepEqual([vectors.groundSha256,vectors.suggestionsSha256,vectors.protocolSha256],[entry.ground.sha256,entry.suggestions.sha256,s.ground.protocol.sha256]);
  assert.equal(vectors.vectors.length,21);
  const kinds={};for(const v of vectors.vectors)kinds[v.kind]=(kinds[v.kind]??0)+1;
  assert.deepEqual(kinds,{pixelUV:3,integerAnchor:3,v2VectorPixel:13,rangeLimitBeyond:1,rangeLimitWithin:1});
  for(const v of vectors.vectors) {
    if(v.supportClass==='unknown'){assert.throws(()=>mappedGroundHit(s,v.position),{message:v.reason},`${v.kind} ${v.position}`);continue;}
    const hit=mappedGroundHit(s,v.position);
    close3(hit.point,v.groundPoint,1e-6,String(v.position));
    assert.deepEqual([hit.supportClass,hit.hidden,hit.hitKind,hit.terrainCityZ,hit.terrainCell],[v.supportClass,v.hidden,v.hitKind,v.terrainCityZ,v.terrainCell],String(v.position));
    near(hit.rangeM,v.rangeM,1e-6);near(hit.alongRayUncertaintyM,v.alongRayUncertaintyM,1e-6);near(Math.hypot(hit.point[0],hit.point[2]),v.horizontalRangeM,1e-6);
  }
  const [beyond,within]=vectors.vectors.filter(v=>v.kind.startsWith('rangeLimit'));
  assert.deepEqual([beyond.position,within.position],[[373,150.595],[373,150.597]]);
  assert.throws(()=>mappedGroundHit(s,beyond.position),{message:'Too far to place (over 30 m).'});
  assert.ok(Math.hypot(mappedGroundHit(s,within.position).point[0],mappedGroundHit(s,within.position).point[2])<=30);
  // The along-ray uncertainty is the fitted band over sin(depression), with no float16 half-step (protocol v3).
  for(const e of Object.values(EXPECTED)){const hit=mappedGroundHit(s,e.pixelUV);near(hit.alongRayUncertaintyM,.075*hit.rangeM/-hit.point[1],1e-9);}
});

test('AV2 8c: suggestions open the comparison with no clicks, in the averaged annotated pose, labelled stationary and mapped, not observed',{skip},async()=>{
  const s=await loadAV2(),raw=s.suggestions,setup=suggestedSetup(s,raw);
  assert.equal(raw.schemaVersion,3);assert.equal(setup.error,'');
  assert.deepEqual(setup.pixels,{crossing:EXPECTED.crossing.pixel,parking:EXPECTED.parking.pixel,target:EXPECTED.target.pixel});
  assert.deepEqual(setup.vehicle,{width:'1.88',height:'1.49',length:'4.03'});assert.equal(setup.targetHeight,'1.1');
  assert.deepEqual(raw.vehicle.pose,{annotations:114,kind:'averaged'});
  assert.deepEqual(Object.fromEntries(Object.entries(setup.provenance).map(([k,p])=>[k,[p.status,p.supportLabel,p.hidden]])),
    {crossing:['mapped','mapped, not observed',true],parking:['annotated','mapped, not observed',true],target:['default','mapped, not observed',false],
      vehicle:['annotated',undefined,undefined],targetHeight:['default',undefined,undefined]});
  const scenario=createScenario(s,setup.input);
  close3(scenario.axis,TRAVEL.map(n=>-n),1e-12,'axis is the reversed ego-lane chord');
  // Setback of the annotated pose: lane-axis distance from the app crossing point (the exact mapped near edge on the box
  // centre line) to the centre of the averaged box's front face: the prepared front gap along the box heading, times the
  // cosine of the 1.41 deg turn (contract A6: 0.41981 m; protocol v2: 0.68655 m).
  near(scenario.baselineSetback,.41980659018081645,1e-9);
  near(scenario.baselineSetback,raw.vehicle.frontGapToCrossingM*Math.cos(raw.vehicle.yawToRoadAxisDeg*Math.PI/180),1e-9);
  close3(scenario.crossing,raw.crossing.mappedPoint,1e-9,'crossing is the mapped point');
  assert.equal(setup.input.proposedSetback,scenario.baselineSetback);assert.equal(scenario.proposedSetback,scenario.baselineSetback);
  near(scenario.limits.captureDistance,raw.crossing.forwardM,1e-9);near(scenario.limits.captureDistance,17.44349815675875,1e-9);
  // The mapped drivable run along the box line reaches the recorded camera; the turned box's lane-axis extent behind its
  // front-face centre (4.03 m x cos plus its rear corner, 0.94 m x sin of the 1.41 deg turn) stays inside it.
  const turnCos=Math.abs(scenario.baseline.axis[0]*scenario.axis[0]+scenario.baseline.axis[2]*scenario.axis[2]);
  near(scenario.limits.maximumSetback,raw.crossing.forwardM-4.03*turnCos-scenario.whatIf.cornerOffset,1e-9,'maximum setback');
  // The stationary vehicle is the averaged box of track 396f15b5 exactly (centre, heading, dimensions), and the proposed
  // what-if starts as that same box.
  const box=s.ground.cuboids.items.find(b=>b.track===raw.vehicle.track);
  assert.equal(raw.vehicle.track,'396f15b5-9cf3-462b-9164-4b6f71c4e7d6');assert.equal(box.poseAverage.rows,114);assert.equal(scenario.baseline.pose,'annotated');
  assert.deepEqual(scenario.baseline.center,box.center);assert.deepEqual(scenario.baseline.center,[6.252671055143537,-0.5370298468527466,13.988787508564087]);
  assert.deepEqual(scenario.baseline.vehicle,{width:box.width,height:box.height,length:box.length});
  close3(scenario.baseline.axis,raw.vehicle.headingWorld.map(n=>-n),1e-12,'axis is the reversed annotated heading');
  assert.deepEqual(scenario.proposed,scenario.baseline);
  close3(placementPreview(s,setup.pixels,setup.vehicle,setup.targetHeight).vehicle.center,box.center,0);
  // First load: the averaged box misses the driver-eye-to-head line by 6.2 cm (contract A6: +0.06241 m; protocol v2:
  // +0.02515 m), so Stationary Unknown and Proposed Unknown (the same box; never Clear). A miss carries no reason code.
  const unknown={state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null};
  for(const v of [scenario.currentVisibility,scenario.proposedVisibility]){const {gapM,...rest}=v;assert.deepEqual(rest,unknown);near(gapM,.06241,5e-5,'first-load gap');}
  // Eye: the recording camera's position and height (1.47 m above the mapped ground), labelled as such.
  assert.equal(scenario.eyeLabel,'recording camera height');near(scenario.eyeHeightM,1.471,.001,'eye height');
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
  // (review repro: crossing [288,150], front [640,150], now beyond the 30 m limit; the same pair two rows lower is within
  // it); the plan view frames that box over its whole movement.
  assert.throws(()=>placementFromPixels(s,{...setup.pixels,crossing:[288,150],parking:[640,150]}),{message:'Too far to place (over 30 m).'});
  const sideways={...setup.input,pixels:{...setup.pixels,crossing:[288,152],parking:[640,152]},vehicle:{width:'1.9',height:'1.5',length:'4'}};
  const sp=placementFromPixels(s,sideways.pixels),wide=createScenario(s,{...sideways,proposedSetback:sp.baselineSetback,driverDistance:sp.captureDistance});
  assert.equal(wide.baseline.pose,'declared');
  for(const aspect of [3.8,1,.65]) {
    const camera=planCamera(s,wide,aspect);
    for(const setback of [wide.baselineSetback,wide.limits.minimumSetback,wide.limits.maximumSetback])inView(camera,whatIfBox(wide.whatIf,setback),`sideways ${setback} at aspect ${aspect}`);
  }
  close3(scenario.target,[EXPECTED.target.groundPoint[0],EXPECTED.target.groundPoint[1]+1.1,EXPECTED.target.groundPoint[2]],1e-9);
  close3(scenario.crossing,EXPECTED.crossing.groundPoint,1e-9);
  for(const v of [scenario.currentVisibility,scenario.proposedVisibility])assert.ok(['blocked','unknown'].includes(v.state));
  // The vehicle front's evidence is that of the point the comparison uses (the averaged front point): mapped, not
  // observed and behind the vehicle itself; it is not placed by a ray, so it carries no along-ray uncertainty.
  assert.deepEqual(Object.fromEntries(Object.entries(scenario.support).map(([k,v])=>[k,[v.supportClass,v.hidden,v.alongRayUncertaintyM]])),
    {crossing:['map-only',true,EXPECTED.crossing.alongRay],parking:['map-only',true,null],target:['map-only',false,EXPECTED.target.alongRay]});
  const current={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight},status=suggestionStatus(raw,setup,current);
  // The crosswalk chip shows the crossing's own error (protocol v3.2 crossingError, measured from paint protocol v4), never
  // its along-ray figure: ±0.18 m is a bound (|measured offset| + 2 × spread, rounded up), never the offset itself.
  assert.deepEqual(raw.crossing.crossingError,{basis:'mapped near edge',status:'measured',label:'±0.18 m',plusMinusM:.18,
    evidence:{file:'.local/evaluation/pao-paint/results/8c52d911-2658849-v4.json',sha256:'2680ec78ac03b6348dc8bb10ca2c78a5c623e664203b5b6fc1b98684dbb6f240'}});
  assert.deepEqual(status.parts.map(p=>`${p.text} ${p.note}`),['crosswalk mapped ±0.18 m (mapped, not observed; hidden)',
    'stationary vehicle (mapped, not observed; hidden)','child 1.1 m (default; mapped, not observed)']);
  assert.equal(crossingText(raw.crossing),'crosswalk mapped ±0.18 m');
  assert.ok(status.parts[0].title.endsWith('mapped near edge; ±0.18 m bounds the paint offset (|measured offset| + 2 × spread; development-grade, not calibrated)'),status.parts[0].title);
  assert.doesNotMatch(status.parts[0].title,/along the pixel ray|1\.03|offset is/);
  for(const [e,text] of [[{status:'measured',label:'±0.25 m',plusMinusM:.25},'crosswalk mapped ±0.25 m'],[{status:'unmeasured',label:'paint offset unmeasured'},'crosswalk mapped · paint offset unmeasured'],
    [{status:'unresolved',label:'paint offset unresolved'},'crosswalk mapped · paint offset unresolved']])assert.equal(crossingText({label:'crosswalk mapped',crossingError:e}),text);
  assert.match(status.parts[1].title,/^stationary vehicle \(AV2 annotation\); reason for stopping unknown.* · annotated pose$/);
  assert.deepEqual(status.context,{crossing:'mapped crosswalk edge, map only',vehicle_front:'stationary vehicle from the dataset annotation; reason for stopping unknown',
    vehicle_size:'annotated stationary vehicle size',person_point:'declared default, map only',person_height:'declared default child'});
  assert.doesNotMatch(JSON.stringify([status.parts.map(p=>[p.text,p.note,p.title]),status.context]),/park/i);
  assert.ok(Object.values(status.context).every(v=>!/\d/.test(v)),'Context strings carry no numbers.');
  // A resized stationary vehicle, or one a declaration replaced at exactly the annotated sizes, is a declared vehicle in
  // its place (protocol v3.1 declaredReplacement): in the context too.
  const resized=suggestionStatus(raw,setup,{...current,vehicle:{...setup.vehicle,width:'1.9'}});
  assert.equal(resized.context.vehicle_front,'declared vehicle, map only');assert.equal(resized.context.vehicle_size,undefined);
  const declaredStatus=suggestionStatus(raw,setup,{...current,overridden:['vehicle']});
  assert.equal(declaredStatus.context.vehicle_front,'declared vehicle, map only');assert.equal(declaredStatus.context.vehicle_size,undefined);
  assert.deepEqual(declaredStatus.parts.map(p=>p.key),['crossing','target']);
  assert.doesNotMatch(JSON.stringify([resized,declaredStatus]),/park|annotated pose/i);
  // A front moved 3 px (past the 2 px nudge) is a declared box at that pixel's ground hit; this one cuts 0.14 m into the
  // line: Blocked.
  const movedPixels={...setup.pixels,parking:[377,191]},q=placementFromPixels(s,movedPixels);
  const movedFront=createScenario(s,{...setup.input,pixels:movedPixels,proposedSetback:q.baselineSetback});
  assert.deepEqual([movedFront.baseline.pose,movedFront.proposed.pose],['declared','declared']);close3(q.parking,mappedGroundHit(s,[377,191]).point,0);
  assert.equal(movedFront.currentVisibility.state,'blocked');assert.ok(movedFront.currentVisibility.gapM<=-.1);
  // Editing loses the label, even within the nudge; the nudged front still names the stationary vehicle in the context.
  assert.equal(isStillSuggested('parking',[374,192],setup.provenance),false);
  const nudgedStatus=suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,parking:[374,192]}});
  assert.equal(nudgedStatus.edited,true);assert.deepEqual(nudgedStatus.parts.map(p=>p.key),['crossing','target']);
  assert.equal(nudgedStatus.context.vehicle_front,'stationary vehicle from the dataset annotation; reason for stopping unknown');
  assert.equal(suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,parking:[377,191]}}).context.vehicle_front,undefined);
  const moved=suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,crossing:[371,188]}});
  assert.equal(moved.edited,true);assert.deepEqual(moved.parts.map(p=>p.key),['parking','target']);
});

test('AV2 8c: a vehicle declared in place of the stationary vehicle keeps its averaged front (protocol v3.1; audit finding 1)',{skip},async()=>{
  // The vehicle front pixel's exact ray meets a raster cell 0.078 m higher 1.03 m before the averaged front (a raster
  // shadow), so its ray hit lands 1.4313 m from the crossing while the averaged front is 0.4199 m from it. Under protocol
  // v3 a declared box took that hit as its front; protocol v3.1 keeps the averaged front for any vehicle in its place.
  const s=await loadAV2(),raw=s.suggestions,setup=suggestedSetup(s,raw),first=createScenario(s,setup.input);
  near(raw.vehicle.pixelOffsetFromMappedM,1.0284829677138683,1e-9,'the stored front pixel hit is in the raster shadow');
  near(raw.vehicle.pixelGroundGapAlongAxisM,1.4313096578253834,1e-9);
  close3(raw.vehicle.mappedPoint,FRONT,1e-9,'mappedPoint is the averaged front');
  const p=placementFromPixels(s,setup.pixels);
  close3(p.parking,FRONT,1e-9,'the vehicle front point is the averaged front point, not the pixel hit');
  close3(anchorPoint(s,'parking',setup.pixels.parking).point,FRONT,1e-9);
  assert.equal(anchorPoint(s,'parking',[377,191]).alongRayUncertaintyM>0,true,'beyond the nudge the pixel is placed by its ray');
  // The finding's test: an untouched front plus a same-size declared vehicle gives a baseline setback within 0.05 m of
  // the annotated box front (here equal to 1e-9 m: the declared box's front-face centre is the averaged front point).
  const lane={...setup.input,proposedSetback:p.baselineSetback};
  for(const [name,next] of [['declared',{...lane,vehicleDeclared:true}],['resized',{...lane,vehicle:{...setup.vehicle,width:'1.9'}}],
    ['declared nudged',{...lane,vehicleDeclared:true,pixels:{...setup.pixels,parking:[375,192]}}]]) {
    const declared=createScenario(s,next);
    assert.deepEqual([declared.baseline.pose,declared.proposed.pose],['declared','declared'],name);
    assert.ok(Math.abs(declared.baselineSetback-first.baselineSetback)<=.05,name);near(declared.baselineSetback,first.baselineSetback,1e-9,name);
    close3(declared.vehicleOrigin,[FRONT[0]-declared.axis[0]*declared.baselineSetback,FRONT[1],FRONT[2]-declared.axis[2]*declared.baselineSetback],1e-9,name);
    near(declared.vehicleGroundHeight,FRONT[1],1e-12,name);
    // The same-size box in place of the stationary vehicle cuts 0.008 m into the child's line: Unknown, too close to call
    // (the resized 1.9 m wide box 0.016 m), never the false Blocked of the shadowed pixel hit.
    for(const v of [declared.currentVisibility,declared.proposedVisibility])assert.deepEqual([v.state,v.reasonCode],['unknown','too-close-to-call'],name);
  }
  near(createScenario(s,{...lane,vehicleDeclared:true}).currentVisibility.gapM,-.00812,5e-5);
  // Regression guard: the protocol v3 model (front at the shadowed pixel hit) put the same box 1.4313 m from the crossing,
  // where it cut 0.107 m into the line (Blocked).
  const hit=raw.vehicle.groundPoint,axis=first.axis,crossing=first.crossing,old=(hit[0]-crossing[0])*axis[0]+(hit[2]-crossing[2])*axis[2],v={width:1.88,height:1.49,length:4.03};
  near(old,1.4313,1e-4);
  const oldBox={center:[hit[0]+axis[0]*v.length/2,hit[1]+v.height/2,hit[2]+axis[2]*v.length/2],axis,vehicle:v};
  const oldResult=assessSightline(s,{eye:first.eye,target:first.target,targetGround:first.targetGround},oldBox);
  assert.equal(oldResult.state,'blocked');near(oldResult.gapM,-.1072,5e-4);
  // comparisonStart, Nemotron declarations and the distance check use the same front.
  near(comparisonStart(s,setup.pixels,setup.vehicle,true).proposedSetback,first.baselineSetback,1e-9);
  near(applyDeclaration(s,setup.pixels,null,{vehicle_label:'box truck',vehicle_width_m:2.5,vehicle_height_m:3.5,vehicle_length_m:8,target_label:'child',target_height_m:1.1}).input.proposedSetback,
    first.baselineSetback,1e-9);
  const van=setbackSweep(s,{pixels:setup.pixels,driverDistance:setup.input.driverDistance,vehicleDeclared:true},{label:'van',width:2,height:2.6,length:5.9},1.1);
  assert.equal(van.error,undefined);
});

test('AV2 8c: a typed fractional position is refused, never a declared Blocked (v3.1 review, major finding)',{skip},async()=>{
  // namesAnnotatedFront takes whole pixels only. Before this fix a fractional front within 2 px of the anchor became a
  // declared box at its own raster-shadowed ray hit: [374.5,191] gave 1.468 m and Blocked (0.111 m cut), the vehicle's own
  // pixelUV 1.431 m and Blocked (0.107 m cut), through the pixel fields, createScenario and a Nemotron declaration.
  const s=await loadAV2(),raw=s.suggestions,setup=suggestedSetup(s,raw),refused={message:'Select a pixel inside the recorded image.'};
  const declaration={vehicle_label:'car',vehicle_width_m:1.88,vehicle_height_m:1.49,vehicle_length_m:4.03,target_label:'child',target_height_m:1.1};
  const typed=[['parking',[374.5,191]],['parking',raw.vehicle.pixelUV],['parking',[374.458,190.849]],['parking',[374,191.5]],['parking',[376,192.999]],
    ['crossing',[371.5,189]],['crossing',raw.crossing.pixelUV],['target',[484.5,185]],['target',raw.person.pixelUV]];
  for(const [key,position] of typed) {
    const what=`${key} ${position}`,pixels={...setup.pixels,[key]:position};
    assert.throws(()=>roadPoint(s,position,'comparison actor'),refused,what);
    assert.throws(()=>anchorPoint(s,key,position),refused,what);
    assert.throws(()=>placementFromPixels(s,pixels),refused,what);
    assert.throws(()=>comparisonStart(s,pixels,setup.vehicle),refused,what);
    assert.throws(()=>placementPreview(s,pixels,setup.vehicle,setup.targetHeight),refused,what);
    for(const vehicleDeclared of [false,true])assert.throws(()=>createScenario(s,{...setup.input,pixels,...(vehicleDeclared&&{vehicleDeclared})}),refused,what);
    assert.equal(applyDeclaration(s,pixels,null,declaration).error,refused.message,what);
    assert.equal(applyDeclaration(s,pixels,setup.input,declaration).error,refused.message,what);
    assert.equal(setbackSweep(s,{pixels,driverDistance:setup.input.driverDistance},{label:'car',width:1.88,height:1.49,length:4.03},1.1).error,refused.message,what);
    if(key==='parking')assert.equal(annotatedVehicle(s,pixels,setup.vehicle),null,what);
  }
  // The fractional position itself is still evaluated by the runtime rule (the loader, the harness, an untouched suggestion).
  close3(mappedGroundHit(s,raw.vehicle.pixelUV).point,raw.vehicle.groundPoint,1e-9);
  // Whole pixels within the nudge still name the stationary vehicle, and a same-size declaration in its place stays at the
  // averaged front: Unknown (too close to call), never the shadowed hit's Blocked.
  for(const front of [[374,191],[375,191],[376,193]]) {
    const pixels={...setup.pixels,parking:front},sc=createScenario(s,{...setup.input,pixels});
    assert.equal(sc.baseline.pose,'annotated',String(front));close3(anchorPoint(s,'parking',front).point,FRONT,1e-9,String(front));
    const d=applyDeclaration(s,pixels,null,declaration),v=createScenario(s,d.input).currentVisibility;
    near(d.input.proposedSetback,sc.baselineSetback,1e-9,String(front));assert.deepEqual([v.state,v.reasonCode],['unknown','too-close-to-call'],String(front));
  }
});

test('AV2 8c: Ask and Check work on first load with the suggested setup',{skip},async()=>{
  const s=await loadAV2(),setup=suggestedSetup(s,s.suggestions),scenario=createScenario(s,setup.input);
  const facts=scenarioFacts(s,scenario);
  assert.equal(facts.scene,ID);assert.equal(facts.eyeHeight_m,1.47);assert.equal(facts.eye,'recording camera height');
  assert.deepEqual(facts.vehicle_m,{width:1.88,height:1.49,length:4.03});assert.equal(facts.targetHeight_m,1.1);
  // The facts describe the averaged annotated pose (turned 1.41 deg from the lane), its front-face setback and its
  // result; the what-if starts as the same box. No margin is given.
  const pose={pose:'annotated',turnedFromLane_deg:1.41,setback_m:.42,setback_ft:1.38,state:'unknown',result:'not blocked by the vehicle (rest unknown)',targetInside:false,eyeInside:false};
  assert.deepEqual(facts.current,pose);assert.deepEqual(facts.proposed,pose);
  const w={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight,input:setup.input,setupInput:setup.input,run:{},vehicleLabel:'stationary vehicle'};
  const check=runTool(s,'check',w,'sweep_setbacks',{vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]});
  assert.equal(check.error,undefined);
  assert.deepEqual(check.result.vehicles.map(v=>v.label),['your vehicle','SUV']);
  for(const r of check.sweep.results)for(const row of r.rows)assert.ok(['blocked','unknown','overlap'].includes(row.state));
  // 'your vehicle' sweeps the averaged annotated pose moved along the lane axis (front-face setbacks). The box is turned
  // 1.41 deg, so its nearer front corner is 0.94 m x sin 1.41 deg = 0.023 m nearer the crossing than the front-face
  // centre: rows start at 0.103 m. Not blocked by the vehicle at 0.10-0.60 m; too close to call at 0.85 m and 10.10-10.35 m;
  // Blocked 1.10-9.85 m; not blocked by the vehicle from 10.60 m; from 12.10 m the moved box overlaps the real annotated
  // vehicle cf467f3d beside the recording car (Unknown, overlaps a real object). The SUV is a declared box in place of the
  // stationary vehicle, from its averaged front.
  const [own,suv]=check.sweep.results;
  const [a,l]=[scenario.baseline.axis,scenario.axis];
  near(scenario.whatIf.cornerOffset,scenario.baseline.vehicle.width/2*Math.abs(a[0]*l[2]-a[2]*l[0]),1e-12);near(scenario.whatIf.cornerOffset,.94*Math.sin(1.4121*Math.PI/180),1e-4,'contract yaw');
  near(scenario.limits.minimumSetback,.08+scenario.whatIf.cornerOffset,1e-12);
  assert.deepEqual(own.range,[scenario.limits.minimumSetback,scenario.limits.maximumSetback]);
  assert.deepEqual(own.rows.slice(0,6).map(r=>[r2(r.setback),r.state,r.reason]),
    [[.1,'unknown',undefined],[.35,'unknown',undefined],[.6,'unknown',undefined],[.85,'unknown','too-close-to-call'],[1.1,'blocked',undefined],[1.35,'blocked',undefined]]);
  assert.equal(sweepPhrase(own),'blocked 1.1-9.85 m; too close to call at 0.85 m, 10.1-10.35 m; not blocked by the vehicle from 10.6 m (34.79 ft); overlaps a real object 12.1-13.39 m');
  assert.deepEqual([own.at15ft,own.at20ft,own.stopsBlockingAt],['blocked','blocked',own.rows.find(r=>r2(r.setback)===10.6).setback]);
  const ownFacts=check.result.vehicles[0];
  assert.deepEqual([ownFacts.blocked_m,ownFacts.tooCloseToCall_m,ownFacts.overlapsRealObject_m,ownFacts.tested_m],[[[1.1,9.85]],[[.85,.85],[10.1,10.35]],[[12.1,13.39]],[.1,13.39]]);
  // Every row carries the signed gap; Blocked rows cut in by at least 10 cm, too-close rows by less.
  for(const row of own.rows)assert.ok(row.state==='blocked'?row.gapM<=-.1:row.reason==='too-close-to-call'?row.gapM<=0&&row.gapM>-.1:true,`setback ${row.setback}`);
  assert.equal(own.rows.find(r=>r.reason==='overlaps-real-object')&&scenarioFacts(s,createScenario(s,{...setup.input,proposedSetback:12.1})).proposed.unknownReason,'overlaps a real object');
  // Below the corner offset the what-if's corner would reach the crossing: refused, never drawn past it.
  assert.throws(()=>createScenario(s,{...setup.input,proposedSetback:.02}),{message:'Keep the proposed vehicle on the camera side of the crossing.'});
  // Below the slider minimum (0.103 m) a typed or declared setback is refused.
  assert.throws(()=>createScenario(s,{...setup.input,proposedSetback:.05}),{message:'Proposed setback must be at least 0.11 m from the crossing (the slider minimum).'});
  assert.equal(createScenario(s,{...setup.input,proposedSetback:scenario.limits.minimumSetback}).proposed.pose,'annotated');
  for(const setback of [.01,.05])
    assert.match(applyDeclaration(s,setup.pixels,setup.input,{vehicle_label:'SUV',vehicle_width_m:2,vehicle_height_m:1.8,vehicle_length_m:5,target_label:'child',target_height_m:1.1,setback_value:setback,setback_unit:'m'}).error,/^Proposed setback must be at least 0\.08 m/);
  assert.equal(sweepPhrase(suv),'blocked 0.08-10.08 m; too close to call 10.33-10.58 m; not blocked by the vehicle from 10.83 m (35.53 ft); overlaps a real object 11.08-12.44 m');
  for(const row of check.sweep.results[0].rows.filter(r=>r.state!=='overlap')) {
    const v=createScenario(s,{...setup.input,proposedSetback:row.setback}).proposedVisibility;
    assert.deepEqual([v.state,v.reasonCode],[row.state,row.reason],`setback ${row.setback}`);
  }
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
  assert.throws(()=>placementFromPixels(s,{...setup.pixels,crossing:[373,200]}),{message:'Place the current vehicle on the camera side of the crossing (its front is 2.51 m past the crossing along the lane).'});
  const ask=runTool(s,'ask',{...w,vehicleLabel:'stationary vehicle'},'declare_scenario',
    {vehicle_label:'box truck',vehicle_width_m:2.5,vehicle_height_m:3.5,vehicle_length_m:8,target_label:'child',target_height_m:1.1,setback_value:20,setback_unit:'ft'});
  assert.equal(ask.error,undefined);assert.equal(ask.input.proposedSetback,20*.3048);assert.equal(ask.input.vehicleDeclared,true);
  assert.deepEqual(ask.result,scenarioFacts(s,createScenario(s,ask.input)));
  // A declaration at exactly the annotated sizes still replaces the stationary vehicle: a declared box on the lane axis
  // from the averaged front point (protocol v3.1), in the facts and in a later distance check.
  const same=runTool(s,'ask',{...w,run:{}},'declare_scenario',
    {vehicle_label:'stationary vehicle',vehicle_width_m:1.88,vehicle_height_m:1.49,vehicle_length_m:4.03,target_label:'child',target_height_m:1.1});
  // Same sizes, but the distance results were swept in the annotated pose, so they are cleared.
  assert.equal(same.error,undefined);assert.equal(same.input.vehicleDeclared,true);assert.equal(same.clearSweep,true);
  assert.equal(check.sweep.vehicleDeclared,false);
  // The declared box at 0.42 m cuts into the line by under 10 cm: too close to call (protocol v3 put it at 1.43 m: Blocked).
  const declaredFacts={setback_m:.42,setback_ft:1.38,state:'unknown',unknownReason:'too close to call',result:'unknown (too close to call)',targetInside:false,eyeInside:false};
  assert.deepEqual(same.result.current,declaredFacts);assert.deepEqual(same.result.proposed,declaredFacts);
  const after=runTool(s,'check',{...w,input:same.input},'sweep_setbacks',{vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]});
  assert.equal(sweepPhrase(after.sweep.results[0]),
    'blocked 0.83-9.83 m; too close to call at 0.58 m, 10.08-10.33 m; not blocked by the vehicle from 10.58 m (34.71 ft); overlaps a real object 12.08-13.41 m');
  assert.equal(after.sweep.vehicleDeclared,true);
  // A further declaration over a declared box at the same sizes keeps those (declared) distance results.
  const again=runTool(s,'ask',{...w,input:same.input,run:{}},'declare_scenario',
    {vehicle_label:'stationary vehicle',vehicle_width_m:1.88,vehicle_height_m:1.49,vehicle_length_m:4.03,target_label:'child',target_height_m:1.1});
  assert.equal(again.clearSweep,false);
  assert.equal(applyDeclaration(s,setup.pixels,null,{vehicle_label:'van',vehicle_width_m:2,vehicle_height_m:2.5,vehicle_length_m:5.5,target_label:'adult',target_height_m:1.5}).error,undefined);
  assert.equal(classifySightline(s,scenario.eye,scenario.target,scenario.vehicle,scenario.axis,[1000,0,1000]).state,'unknown');
});

test('AV2 8c: a 1-2 px nudge of the vehicle front keeps the stationary vehicle; a crossing nudge keeps it or is refused naming the gap',{skip},async()=>{
  // The suggested crossing [371,189] and vehicle front [374,191] anchors are 3 px apart; near the horizon one pixel row
  // moves a ground point by about a metre. Plan M1(d): a front pixel within 2 px of the anchor (each axis) still names the
  // annotated stationary vehicle, so its exact box and results stay. A crossing nudge evaluates the nudged pixel as given
  // (only the untouched anchor uses the exact sub-pixel position); one row toward the camera puts the crossing past the
  // box: refused with the measured gap. Every result is Blocked or Unknown; no other error appears.
  const s=await loadAV2(),setup=suggestedSetup(s,s.suggestions),outcomes={},first=createScenario(s,setup.input);
  const refusal=/^Place the current vehicle on the camera side of the crossing \(its nearest front corner is \d+(?:\.\d+)? m past the crossing along the lane\)\.$/;
  for(const [role,reach] of [['crossing',1],['parking',2],['target',1]])for(let dv=-reach;dv<=reach;dv++)for(let du=-reach;du<=reach;du++) {
    if(!du&&!dv)continue;
    const [u,v]=setup.pixels[role],pixels={...setup.pixels,[role]:[u+du,v+dv]};
    let key;
    try{
      const sc=createScenario(s,{...setup.input,pixels,...comparisonStart(s,pixels,setup.vehicle)});
      assert.equal(sc.baseline.pose,'annotated',`${role} ${du},${dv}`);
      if(role==='parking') {
        // The nudged front changes nothing about the tested box: same setback, limits, boxes and results.
        assert.deepEqual([sc.baseline,sc.proposed,sc.baselineSetback,sc.limits,sc.currentVisibility,sc.proposedVisibility],
          [first.baseline,first.proposed,first.baselineSetback,first.limits,first.currentVisibility,first.proposedVisibility],`${du},${dv}`);
        assert.equal(setbackSweep(s,{pixels,driverDistance:setup.input.driverDistance},{label:'your vehicle',width:1.88,height:1.49,length:4.03},1.1).stopsBlockingAt,
          setbackSweep(s,{pixels:setup.pixels,driverDistance:setup.input.driverDistance},{label:'your vehicle',width:1.88,height:1.49,length:4.03},1.1).stopsBlockingAt);
      }
      key=[sc.baseline.pose,...[sc.currentVisibility,sc.proposedVisibility].map(x=>x.state+(x.reasonCode?`(${x.reasonCode})`:''))].join(' ');
    }catch(e){assert.match(e.message,refusal,`${role} ${du},${dv}`);key='refused: camera side';}
    (outcomes[role]??={})[key]=(outcomes[role][key]??0)+1;
  }
  assert.deepEqual(outcomes,{
    crossing:{'annotated unknown unknown':4,'refused: camera side':4},
    parking:{'annotated unknown unknown':24},
    target:{'annotated unknown unknown':8}});
  const v=createScenario(s,{...setup.input,pixels:{...setup.pixels,parking:[374,193]}}).currentVisibility;
  assert.equal(v.state,'unknown');near(v.gapM,.06241,5e-5);
});

test('AV2 8c: answers say "not blocked by the vehicle" only where Sightline computed it, in its own result phrases',{skip},async()=>{
  // Audit (October 12): with the adult (1.5 m) the rows up to 6 m cut under 10 cm into the box (too close to call), yet
  // the prompt told the model to call Unknown "not blocked by the vehicle (rest unknown)" and the check let it.
  const s=await loadAV2(),setup=suggestedSetup(s,s.suggestions);
  for(const [height,setback,gap] of [[1.5,5,-.0856],[1.1,.85,-.0414],[1.1,10.35,-.0256]]) {
    const sc=createScenario(s,{...setup.input,targetHeight:String(height),proposedSetback:setback}),facts=scenarioFacts(s,sc),allowed=allowedNumbersFrom(facts);
    assert.deepEqual([sc.proposedVisibility.state,sc.proposedVisibility.reasonCode],['unknown','too-close-to-call']);near(sc.proposedVisibility.gapM,gap,5e-4);
    assert.equal(facts.proposed.result,'unknown (too close to call)');assert.equal(facts.current.result,'not blocked by the vehicle (rest unknown)');
    const at=String(setback),ft=String(facts.proposed.setback_ft);
    for(const bad of [`At ${at} m the person is not blocked by the vehicle (rest unknown).`,`The proposed vehicle is not blocked by the vehicle (rest unknown).`,
      `At ${at} m (${ft} ft) the line isn't blocked by the vehicle.`,`The what-if at ${at} m does not block the line.`,`From ${at} m it no longer blocks.`]) {
      const r=checkAnswer(bad,allowed,{mapped:true});assert.deepEqual([r.ok,r.kind],[false,'result'],bad);
      assert.equal(withheldNote(r),'Nemotron\'s answer was withheld: described a result Sightline did not compute.');
    }
    for(const good of [`At ${at} m (${ft} ft) the result is unknown (too close to call).`,'The stationary vehicle at 0.42 m is not blocked by the vehicle (rest unknown).',
      `Stationary vehicle: not blocked by the vehicle (rest unknown); proposed at ${at} m: unknown (too close to call).`])
      assert.deepEqual(checkAnswer(good,allowed,{mapped:true}),{ok:true,reason:'',text:good},good);
  }
  // First load: both positions are plain Unknown (the line misses the box by 6.2 cm), so the unqualified claim is true.
  const first=allowedNumbersFrom(scenarioFacts(s,createScenario(s,setup.input)));
  assert.equal(checkAnswer('The child is not blocked by the vehicle (rest unknown).',first,{mapped:true}).ok,true);
  // A delivery van (2.0 x 2.6 x 5.9 m), a declared box in place of the stationary vehicle: blocked to 9.83 m, then the
  // moved box overlaps the real vehicle cf467f3d from 10.08 m. Its short phrase says so instead of "no tested distance
  // where it stops blocking".
  const w={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight,input:setup.input,setupInput:setup.input,run:{},vehicleLabel:'stationary vehicle'};
  const check=runTool(s,'check',w,'sweep_setbacks',{vehicles:[{label:'delivery van',width_m:2,height_m:2.6,length_m:5.9}]}),van=check.result.vehicles[1];
  assert.deepEqual([van.label,van.result,van.short,van.stopsBlockingAt_m],['delivery van','blocked 0.08-9.83 m; overlaps a real object 10.08-11.54 m','unknown from 10.08 m (overlaps a real object)',null]);
  const sweepAllowed=allowedNumbersFrom(scenarioFacts(s,createScenario(s,setup.input)),check.result);
  assert.equal(checkAnswer(`Delivery van: ${van.short}. Your vehicle: ${check.result.vehicles[0].short}.`,sweepAllowed,{mapped:true}).ok,true);
  assert.equal(checkAnswer('Delivery van: not blocked by the vehicle from 10.08 m.',sweepAllowed,{mapped:true}).kind,'result');
  // An annotated stationary vehicle outside the supported sizes fails the suggestion load by name (here 4.6 m tall).
  const track=s.suggestions.vehicle.track,tall={...s,ground:{...s.ground,cuboids:{...s.ground.cuboids,items:s.ground.cuboids.items.map(b=>b.track===track?{...b,height:4.6}:b)}}};
  const raw=structuredClone(s.suggestions);raw.vehicle.dimensions.height=4.6;
  assert.throws(()=>validateSuggestions(raw,tall),{message:'Invalid placement suggestions: the annotated vehicle (width 1.88 m, height 4.6 m, length 4.03 m) is outside the supported sizes (width 0.5-3.5 m, height 0.5-4.5 m, length 1-20 m).'});
  assert.equal(validateSuggestions(structuredClone(s.suggestions),s).vehicle.track,track);
});

test('AV2 8c: answer check catches invented stop reasons, sources, car models and synonyms for the crossing',{skip},async()=>{
  const s=await loadAV2(),setup=suggestedSetup(s,s.suggestions),allowed=allowedNumbersFrom(scenarioFacts(s,createScenario(s,setup.input)));
  // Fixed bad-sentence set (audit, October 12): each passed the M1 check on AV2.
  const bad={wording:['The car pulled over 0.42 m from the crossing.','The car is out of gas at 0.42 m.','It has a flat tire.','It is letting a passenger out.',
      'The crossing position comes from LiDAR.','The LiDAR shows the crossing 0.42 m away.','The stationary vehicle is exactly 0.42 m back.','The setback is verified.',
      'This is ground truth.','The stationary vehicle is a Tesla Model S.','It is a Toyota.'],
    role:['The stationary vehicle is 1.49 m from the intersection.','It is set back 1.49 m.','It sits 1.88 m from the stop line.','The setback is 1.49 m.']};
  for(const [kind,list] of Object.entries(bad))for(const text of list) {
    const r=checkAnswer(text,allowed,{mapped:true});assert.deepEqual([r.ok,r.kind],[false,kind],text);
  }
  // Correct sentences that the M1 check withheld, and the support class's own words.
  for(const good of ['The child is behind a 1.49 m tall red delivery van.','The stationary vehicle is 0.42 m from the intersection.','It is set back 0.42 m (1.38 ft).',
    'The vehicle front has nearby LiDAR returns.','The vehicle is 1.49 m tall and the child is 1.1 m tall.'])
    assert.deepEqual(checkAnswer(good,allowed,{mapped:true}),{ok:true,reason:'',text:good},good);
  // The prompt no longer says "measured" (the mapped-scene check withholds it), so the model is never led to it.
  for(const prompt of Object.values(SYSTEM_PROMPTS))assert.equal(checkAnswer(prompt.match(/The crossing[^.]*\./)[0],[],{mapped:true}).ok,true);
});

test('AV2 8c: stale, tampered, reworded or earlier-schema suggestions and ground fail the load',{skip},async()=>{
  const original=await av2Bytes('suggestions.json'),raw=JSON.parse(original.toString('utf8'));
  const reseal=mutate=>{const next=structuredClone(raw);mutate(next);const bytes=Buffer.from(JSON.stringify(next));return [{'suggestions.json':bytes},{suggestions:{file:'suggestions.json',sha256:sha(bytes)}}];};
  const cases={
    'anchor not the pixel containing pixelUV':[r=>{r.crossing.pixel=[371,190];},/crossing\.pixelUV must be a crop position and pixel the pixel containing it/],
    'pixelUV drift':[r=>{r.crossing.pixelUV[1]+=.01;},/crossing disagrees with the runtime mapped-ground rule/],
    'pixelUV outside the crop':[r=>{r.person.pixelUV[0]=736;r.person.pixel=[736,185];},/person\.pixel must be an integer pixel inside the recorded image/],
    'hidden flipped':[r=>{r.vehicle.hidden=false;},/vehicle disagrees/],
    'support class promoted':[r=>{r.person.supportClass='observed-nearby';r.person.supportLabel='observed nearby';},/person disagrees/],
    'support label':[r=>{r.person.supportLabel='observed nearby';},/supportClass and supportLabel must match/],
    'ground point drift':[r=>{r.person.groundPoint[2]+=2e-6;},/person disagrees/],
    'horizontal range':[r=>{r.person.horizontalRangeM+=1e-3;},/person disagrees/],
    'residual understated':[r=>{r.vehicle.pixelOffsetFromMappedM=0;},/vehicle disagrees/],
    'integer anchor check':[r=>{r.person.integerPixelCheck.residualM=0;},/person\.integerPixelCheck disagrees/],
    'integer anchor reason':[r=>{r.person.integerPixelCheck={reason:'Too far to place (over 30 m).'};},/person\.integerPixelCheck disagrees/],
    'uncertainty':[r=>{r.crossing.alongRayUncertaintyM=1.2111531809435245;},/crossing disagrees/],
    'crossing error status':[r=>{r.crossing.crossingError.status='pending-measurement';},/crossingError must be a measured offset/],
    'crossing error status unmeasured':[r=>{r.crossing.crossingError.status='unmeasured';},/crossingError must be a measured offset/],
    'crossing error bound':[r=>{r.crossing.crossingError.plusMinusM=.17;},/crossingError must be a measured offset/],
    'crossing error bound over 5 m':[r=>{Object.assign(r.crossing.crossingError,{plusMinusM:5.5,label:'±5.50 m'});},/crossingError must be a measured offset/],
    'crossing error evidence':[r=>{r.crossing.crossingError.evidence=null;},/crossingError must be a measured offset/],
    'crossing error evidence hash':[r=>{r.crossing.crossingError.evidence.sha256='not-a-hash';},/crossingError must be a measured offset/],
    'crossing error label':[r=>{r.crossing.crossingError.label='±1.03 m';},/crossingError must be a measured offset/],
    'crossing error fields':[r=>{r.crossing.paintOffset={status:'unmeasured'};},/unexpected field crossing\.paintOffset/],
    'averaged pose count':[r=>{r.vehicle.pose.annotations=113;},/must match exactly one annotated \(averaged\) cuboid/],
    'front point moved':[r=>{r.vehicle.mappedPoint[1]+=.05;},/mappedPoint must be the averaged box front point/],
    'annotated track':[r=>{r.vehicle.track='00000000-0000-0000-0000-000000000000';},/must match exactly one annotated/],
    'annotated size':[r=>{r.vehicle.dimensions.width=1.89;},/must match exactly one annotated/],
    'annotated heading':[r=>{r.vehicle.headingWorld=[0.6,0,0.8];},/must match exactly one annotated/],
    'annotated motion':[r=>{r.vehicle.motion=null;},/must match exactly one annotated/],
    'banned wording':[r=>{r.vehicle.reason='parked vehicle';},/banned vehicle-status wording/],
    'banned label':[r=>{r.limitations.push('Not parking safety.');},/banned vehicle-status wording/],
    'schema 2 (protocol v2)':[r=>{r.schemaVersion=2;},/schemaVersion must be 3/],
    'v2 search record':[r=>{r.crossing.pixelSearch={windowPx:3};},/unexpected field crossing\.pixelSearch/],
    'extra field':[r=>{r.crossCheck={status:'ran'};},/unexpected field file\.crossCheck/],
    'tolerance':[r=>{r.tolerances.roadToleranceMeters=.3;},/tolerances must equal/],
    'band tolerance':[r=>{r.tolerances.alongRayHeightBandMeters=.08;},/tolerances must equal/],
    'range limit tolerance':[r=>{r.tolerances.placementRangeLimitMeters=60;},/tolerances must equal/],
    'road axis':[r=>{r.roadAxis.directionOfTravel=[0.6,0,0.8];},/roadAxis must be the mapped ego lane chord/],
    'license':[r=>{r.license.id='CC-BY-4.0';},/license must equal/],
    'protocol':[r=>{r.protocolSha256='0'.repeat(64);},/name the protocol used/],
    'scene input':[r=>{r.inputs['scene.json']='0'.repeat(64);},/different scene inputs/],
    'ground input':[r=>{r.inputs['ground.json']='0'.repeat(64);},/different scene inputs/],
    'protocol input (protocol v3 file)':[r=>{r.inputs['protocol.json']=r.protocolSha256='c91066034d6b13ce06ab1ee14cd70a148a4531f93692ccac7e21b58ec2f796eb';},/different scene inputs/],
    'other scene':[r=>{r.sceneId='other';},/sceneId/],
    'unknown with stray pixel only':[r=>{r.person={status:'unknown',supportClass:'unknown',reason:'No curb proxy.',pixel:[484,185],surprise:1};},/unexpected field person\.surprise/],
    'unknown with a wrong anchor':[r=>{r.person={status:'unknown',supportClass:'unknown',reason:'Too far to place (over 30 m).',pixelUV:[484.3,185.3],pixel:[485,185]};},/person\.pixelUV and pixel/],
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
  // The superseded protocol v2 folder (ground schema 1, suggestions schema 2) is refused by its own consistent pins, so a
  // stale catalog entry fails loudly instead of loading history.
  if(existsSync(new URL('grounded-v2/catalog-entry.json',AV2))) {
    const v2=JSON.parse(await readFile(new URL('grounded-v2/catalog-entry.json',AV2),'utf8'));
    const v2Fetch=async path=>{const name=path.split('/').at(-1);try{return new Response(await readFile(new URL((GROUNDED.has(name)?'grounded-v2/':'assembled/')+name,AV2)));}catch{return new Response(null,{status:404});}};
    await assert.rejects(loadScene(v2Fetch,ID,v2.metadataSha256,undefined,v2.suggestions,v2.ground),/Invalid mapped ground: schemaVersion must be 2/);
  }
  // The superseded protocol v3 folder has the current schema numbers and consistent pins, so the loader refuses it by its
  // protocol hash (v3.1 review): the app never applies the v3.1 vehicle-front rule to v3 data.
  if(existsSync(new URL('grounded-v3/catalog-entry.json',AV2))) {
    const v3=JSON.parse(await readFile(new URL('grounded-v3/catalog-entry.json',AV2),'utf8'));
    const v3Fetch=async path=>{const name=path.split('/').at(-1);try{return new Response(await readFile(new URL((GROUNDED.has(name)?'grounded-v3/':'assembled/')+name,AV2)));}catch{return new Response(null,{status:404});}};
    await assert.rejects(loadScene(v3Fetch,ID,v3.metadataSha256,undefined,v3.suggestions,v3.ground),
      {message:'Invalid mapped ground: protocol v3 is superseded (protocol v3.1 or later is required).'});
  }
  // A schema 3 file cannot be applied to a measured-depth scene.
  const bytes=Buffer.from(JSON.stringify(raw));
  await assert.rejects(loadScene(async path=>path.endsWith('/suggestions.json')?new Response(bytes):fetchAsset(path),legacyScene.id,undefined,undefined,{file:'suggestions.json',sha256:sha(bytes)}),
    /Invalid placement suggestions: /);
});
}

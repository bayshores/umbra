import assert from 'node:assert/strict';
import test from 'node:test';
import { createScenario } from '../src/geometry.mjs';
import { sweepShort,FOOT,toMeters,RANGES,validateDeclaration,validateSweep,actorConflicts,scenarioFacts,applyDeclaration,setbackSweep,sweepFacts,sweepPhrase,sweepMarks,
  checkAnswer,allowedNumbersFrom,withheldNote,declaredLine,vehiclePart,runTool } from '../src/scenario-assistant.mjs';
import { placementFromPixels } from '../src/geometry.mjs';
import { scene,input } from './scene-input.mjs';

// Analytic flat street: eye 1.5 m above ground y=0 looking along -z; pixel column u=10 holds exact ground depths.
// Never a runtime fixture. The person stands at the crossing, so the sightline lies in the vehicle's centre plane x=0;
// one far ground return at farDepth bounds the fitted road extent.
function flatScene(f,farDepth) {
  const size=21,c=10,h=1.5,depth=new Float64Array(size*size);
  for(let v=c+1;v<size;v++)depth[v*size+c]=h*f/(v-c);
  return {id:'analytic',units:'meters',camera:{fx:f,fy:f,cx:c,cy:c,width:size,height:size,eye:[0,h,0]},
    transform:{cameraOriginGround:[0,h,0],worldFromCamera:[[1,0,0],[0,-1,0],[0,0,-1]]},processing:{depthDivisor:1,roadToleranceMeters:.08},
    depth,positions:new Float32Array([0,0,-farDepth]),triangles:new Uint32Array(),bounds:{min:[-5,0,-20],max:[5,3,0]}};
}
const near=(a,b,tolerance=1e-9)=>assert.ok(Math.abs(a-b)<=tolerance,`${a} differs from ${b}`);
const declaration={vehicle_label:'test van',vehicle_width_m:1.9,vehicle_height_m:2.4,vehicle_length_m:5.4,target_label:'adult',target_height_m:1.3};
const vehicles=[{label:'your vehicle',...input.vehicle},{label:'compact car',width:1.8,height:1.5,length:4.5},{label:'SUV',width:2,height:1.8,length:5},{label:'box truck',width:2.5,height:3.5,length:8},{label:'low trailer',width:1.8,height:.5,length:4.5}];

test('feet convert exactly and unknown units are rejected',()=>{
  assert.equal(FOOT,.3048);assert.equal(toMeters(3,'m'),3);
  near(toMeters(20,'ft'),6.096,1e-12);near(toMeters(15,'ft'),4.572,1e-12);
  assert.throws(()=>toMeters(1,'yd'),/m or ft/);
});

test('declarations accept range endpoints and reject everything else without clamping',()=>{
  const ok=validateDeclaration(declaration);
  assert.equal(ok.ok,true);
  assert.deepEqual(ok.value,{vehicleLabel:'test van',vehicle:input.vehicle,targetLabel:'adult',targetHeight:1.3,setback:null});
  for(const [key,[lo,hi]] of [['vehicle_width_m',RANGES.width],['vehicle_height_m',RANGES.height],['vehicle_length_m',RANGES.length],['target_height_m',RANGES.targetHeight]]) {
    for(const value of [lo,hi])assert.equal(validateDeclaration({...declaration,[key]:value}).ok,true);
    for(const value of [lo-.01,hi+.01,String(lo),NaN,Infinity,null,undefined]) {
      const r=validateDeclaration({...declaration,[key]:value});
      assert.equal(r.ok,false);assert.match(r.error,new RegExp(key));assert.equal(r.value,undefined);
    }
  }
  for(const label of ['',' ','x'.repeat(41),7])assert.equal(validateDeclaration({...declaration,target_label:label}).ok,false);
  assert.equal(validateDeclaration({...declaration,vehicle_label:'x'.repeat(40)}).ok,true);
  assert.match(validateDeclaration({...declaration,model:'other'}).error,/Unexpected field model/);
  for(const bad of [null,[],'text',3])assert.equal(validateDeclaration(bad).ok,false);
  assert.deepEqual(validateDeclaration({...declaration,setback_value:10,setback_unit:'ft'}).value.setback,{value:10,unit:'ft',meters:10*FOOT});
  assert.equal(validateDeclaration({...declaration,setback_value:null,setback_unit:null}).value.setback,null);
  for(const pair of [{setback_value:3},{setback_unit:'m'},{setback_value:0,setback_unit:'m'},{setback_value:-1,setback_unit:'m'},{setback_value:'3',setback_unit:'m'},{setback_value:3,setback_unit:'yd'}])
    assert.equal(validateDeclaration({...declaration,...pair}).ok,false);
});

test('sweep declarations hold one to four vehicles in the same ranges',()=>{
  const v={label:'SUV',width_m:2,height_m:1.8,length_m:5};
  assert.deepEqual(validateSweep({vehicles:[v],target_height_m:1.1}),{ok:true,value:{vehicles:[{label:'SUV',width:2,height:1.8,length:5}],targetHeight:1.1}});
  assert.equal(validateSweep({vehicles:[v,v,v,v]}).value.targetHeight,undefined);
  for(const bad of [{vehicles:[]},{vehicles:[v,v,v,v,v]},{vehicles:v},{},{vehicles:[v],target_height_m:2.6},{vehicles:[v],extra:1},
    {vehicles:[v,{...v,width_m:3.6}]},{vehicles:[{...v,label:''}]},{vehicles:[{...v,model:'x'}]},{vehicles:[null]}])assert.equal(validateSweep(bad).ok,false);
  assert.match(validateSweep({vehicles:[v,{...v,length_m:.9}]}).error,/vehicles\[1\]\.length_m/);
});

test('applying a declaration equals the direct measured comparison',()=>{
  const fresh=applyDeclaration(scene,input.pixels,null,declaration);
  assert.equal(fresh.error,undefined);
  assert.deepEqual(fresh.input,input);
  assert.deepEqual(fresh.scenario,createScenario(scene,input));
  assert.deepEqual(fresh.facts,scenarioFacts(scene,fresh.scenario));
  assert.equal(fresh.declared.setback,null);assert.match(fresh.declared.summary,/^test van 1\.9 × 2\.4 × 5\.4 m .* adult 1\.3 m$/);
  const current={...input,proposedSetback:9,driverDistance:20},kept=applyDeclaration(scene,input.pixels,current,declaration);
  assert.deepEqual(kept.scenario,createScenario(scene,current));
  const feet=applyDeclaration(scene,input.pixels,current,{...declaration,setback_value:10,setback_unit:'ft'});
  assert.equal(feet.input.proposedSetback,10*FOOT);assert.equal(feet.input.driverDistance,20);
  assert.deepEqual(feet.scenario,createScenario(scene,{...current,proposedSetback:10*FOOT}));
  assert.deepEqual(feet.declared.setback,{value:10,unit:'ft',m:3.05,ft:10});
  assert.match(feet.declared.summary,/3\.05 m \(10 ft\) from crossing$/);
});

test('declarations never invent placements or substitute out-of-extent values',()=>{
  assert.match(applyDeclaration(scene,{crossing:input.pixels.crossing,parking:input.pixels.parking},null,declaration).error,/Place the crossing/);
  assert.match(applyDeclaration(scene,input.pixels,null,{...declaration,vehicle_width_m:9}).error,/vehicle_width_m/);
  assert.match(applyDeclaration(scene,input.pixels,null,{...declaration,setback_value:100,setback_unit:'m'}).error,/captured road extent/);
  assert.match(applyDeclaration(scene,input.pixels,{...input,proposedSetback:''},declaration).error,/Proposed setback/);
});

test('scenario facts are scalar, unit-labelled and carry no pixels or coordinates',()=>{
  const facts=scenarioFacts(scene,createScenario(scene,input));
  assert.deepEqual(Object.keys(facts),['scene','units','eyeHeight_m','vehicle_m','targetHeight_m','maximumSetback_m','maximumSetback_ft','current','proposed']);
  const values=v=>v&&typeof v==='object'?Object.values(v).flatMap(values):[v];
  assert.ok(values(facts).every(v=>!Array.isArray(v)&&['string','number','boolean'].includes(typeof v)));
  assert.equal(facts.units,'meters');near(facts.eyeHeight_m,scene.camera.eye[1],.005);
  assert.deepEqual(facts.current,{setback_m:6.55,setback_ft:21.5,state:'blocked',targetInside:false,eyeInside:false});
  assert.equal(JSON.stringify(facts).includes(String(input.pixels.crossing[0])),false);
});

test('a declared vehicle over the measured person is an overlap, never blocked',()=>{
  const vehicle={width:2,height:1.8,length:5};
  for(const proposedSetback of [.5,2]) {
    const s=createScenario(scene,{...input,vehicle,proposedSetback});
    assert.equal(s.proposedVisibility.state,'blocked');
    assert.deepEqual(actorConflicts(s,'proposed'),{targetInside:true,eyeInside:false});
    assert.deepEqual(actorConflicts(s,'current'),{targetInside:false,eyeInside:false});
    assert.equal(scenarioFacts(scene,s).proposed.state,'overlap');
  }
  const rows=setbackSweep(scene,input,{label:'SUV',...vehicle}).rows;
  for(const setback of [.58,2.08])assert.equal(rows.find(r=>Math.abs(r.setback-setback)<1e-9).state,'overlap');
  assert.throws(()=>actorConflicts(createScenario(scene,input),'driver'),/current, baseline or proposed/);
});

test('every sweep row matches the measured comparison and never yields Clear',()=>{
  for(const vehicle of vehicles) {
    const sweep=setbackSweep(scene,input,vehicle);
    assert.equal(sweep.error,undefined);
    for(const {setback,state} of sweep.rows) {
      const s=createScenario(scene,{...input,vehicle:{width:vehicle.width,height:vehicle.height,length:vehicle.length},proposedSetback:setback});
      const c=actorConflicts(s,'proposed');
      if(state==='overlap')assert.ok(c.targetInside||c.eyeInside);
      else {assert.deepEqual(c,{targetInside:false,eyeInside:false});assert.equal(state,s.proposedVisibility.state);}
    }
    assert.ok(sweep.rows.every(r=>['blocked','unknown','overlap'].includes(r.state)));
    assert.doesNotMatch(JSON.stringify([sweep,sweepFacts([sweep])]),/clear|visible|safe/i);
  }
});

test('sweep steps 0.25 m across the fitted range and includes in-range 15 ft and 20 ft marks',()=>{
  const sweep=setbackSweep(scene,input,vehicles[1]),setbacks=sweep.rows.map(r=>r.setback),max=createScenario(scene,{...input,vehicle:{width:1.8,height:1.5,length:4.5}}).limits.maximumSetback;
  assert.equal(sweep.step,.25);assert.deepEqual(sweep.range,[scene.processing.roadToleranceMeters,max]);
  assert.ok(setbacks.every((n,i)=>!i||n>setbacks[i-1]));
  assert.ok(setbacks.includes(20*FOOT)&&setbacks.includes(15*FOOT));
  assert.equal(setbacks.filter(n=>n!==20*FOOT&&n!==15*FOOT&&n!==max).every(n=>Math.abs((n-.08)/.25-Math.round((n-.08)/.25))<1e-9),true);
  assert.equal(sweep.at20ft,sweep.rows.find(r=>r.setback===20*FOOT).state);
  // Same measured inputs without a driver position use the recorded capture distance.
  assert.deepEqual(setbackSweep(scene,{pixels:input.pixels,targetHeight:input.targetHeight},vehicles[1]),sweep);
  // Fitted range below 15 ft: neither mark is tested or reported.
  const short=setbackSweep(flatScene(20,4.5),{pixels:{crossing:[10,13],parking:[10,14],target:[10,13]},targetHeight:.5},{label:'cart',width:2,height:1,length:1});
  assert.equal(short.error,undefined);near(short.range[1],4.5,1e-6);
  assert.equal(short.at15ft,null);assert.equal(short.at20ft,null);
  assert.ok(short.rows.every(r=>r.setback<15*FOOT));
});

test('analytic transition: blocking stops at the first tested setback above 5.25 m',()=>{
  // Sightline height at distance a from the crossing is 0.5 + a/15; a 0.85 m box starting at s blocks iff s <= 5.25.
  const flat=flatScene(10,1),base={pixels:{crossing:[10,11],parking:[10,12],target:[10,11]},targetHeight:.5};
  const sweep=setbackSweep(flat,base,{label:'low trailer',width:2,height:.85,length:4});
  assert.equal(sweep.error,undefined);near(sweep.range[1],10,1e-6);
  for(const r of sweep.rows)assert.equal(r.state,r.setback<=5.25?'blocked':'unknown');
  near(sweep.stopsBlockingAt,5.33);near(sweep.notBlockedFrom,5.33);
  assert.equal(sweep.at15ft,'blocked');assert.equal(sweep.at20ft,'unknown');
  const facts=sweepFacts([sweep,{label:'too long',error:'Vehicle extends beyond the captured road extent.'}]);
  assert.deepEqual(facts.vehicles[0],{label:'low trailer',vehicle_m:{width:2,height:.85,length:4},targetHeight_m:.5,tested_m:[.08,10],tested_ft:[.26,32.81],step_m:.25,
    stopsBlockingAt_m:5.33,stopsBlockingAt_ft:17.49,at15ft:'blocked',at20ft:'unknown',blocked_m:[[.08,5.08]],overlap_m:[],
    result:'blocked 0.08-5.08 m; not blocked by the vehicle from 5.33 m (17.49 ft)',short:'not blocked by the vehicle from 5.33 m (17.49 ft)'});
  assert.deepEqual(facts.vehicles[1],{label:'too long',error:'Vehicle extends beyond the captured road extent.'});
  const always=setbackSweep(flat,base,{label:'tall',width:2,height:2,length:4});
  assert.equal(always.stopsBlockingAt,null);assert.equal(always.notBlockedFrom,null);
  assert.deepEqual(sweepFacts([always]).vehicles[0].blocked_m,[[.08,10]]);
  assert.equal(sweepFacts([always]).vehicles[0].result,'blocked at every tested distance');
});

test('blocking stops where every later row is unknown, not at the first unknown row',()=>{
  // Measured scene: unknown at 0.08 m, blocked from 0.33 m, then unknown to the end of the tested range.
  const sweep=setbackSweep(scene,input,{label:'small car',width:1.6,height:1.4,length:3},.5),first=sweep.rows.findIndex(r=>r.state==='blocked');
  assert.equal(sweep.rows[0].state,'unknown');assert.ok(first>0);
  near(sweep.stopsBlockingAt,21.58);near(sweep.notBlockedFrom,21.58);
  assert.equal(sweepPhrase(sweep),'blocked 0.33-21.33 m; not blocked by the vehicle from 21.58 m (70.8 ft)');
  assert.ok(sweep.rows.filter(r=>r.setback<21.58-1e-9).some(r=>r.state==='blocked'));
  assert.ok(sweep.rows.filter(r=>r.setback>=21.58-1e-9).every(r=>r.state==='unknown'));
  // A vehicle under the analytic sightline never blocks: every row is unknown from the first tested setback.
  const never=setbackSweep(flatScene(10,1),{pixels:{crossing:[10,11],parking:[10,12],target:[10,11]},targetHeight:.5},{label:'low cart',width:2,height:.5,length:4});
  assert.ok(never.rows.every(r=>r.state==='unknown'));
  assert.equal(never.stopsBlockingAt,never.range[0]);assert.equal(never.notBlockedFrom,never.range[0]);
  assert.equal(sweepPhrase(never),'not blocked by the vehicle at any tested distance (rest unknown)');
});

test('an eye inside the declared box is an overlap even where only the eye is inside',()=>{
  // Driver just past the minimum distance: a wide, tall box near the crossing holds the eye (1.46 m off its centre line).
  const vehicle={width:3.5,height:4.5,length:6},base={...input,driverDistance:createScenario(scene,input).limits.minimumDriverDistance+.2};
  const sweep=setbackSweep(scene,base,{label:'tall',...vehicle}),eyeOnly=[];
  for(const r of sweep.rows.filter(r=>r.state==='overlap')) {
    const s=createScenario(scene,{...base,vehicle,proposedSetback:r.setback}),c=actorConflicts(s,'proposed');
    if(c.eyeInside&&!c.targetInside)eyeOnly.push({r,s,c});
  }
  assert.ok(eyeOnly.length>=1);
  for(const {r,s,c} of eyeOnly) {
    assert.deepEqual(c,{targetInside:false,eyeInside:true});assert.equal(r.state,'overlap');assert.deepEqual([r.person,r.driver],[false,true]);
    assert.equal(s.proposedVisibility.state,'blocked');assert.equal(scenarioFacts(scene,s).proposed.state,'overlap');
  }
  // Overlap rows never count as blocking; the unknown run after them starts the not-blocked range.
  assert.equal(sweep.rows.some(r=>r.state==='blocked'),false);
  near(sweep.stopsBlockingAt,sweep.rows.find(r=>r.state==='unknown').setback);
  assert.equal(sweepPhrase(sweep),'not blocked by the vehicle at any tested distance (rest unknown); overlaps person and driver 0.08-2.58 m');
});

test('a vehicle that fits at no setback returns an error instead of rows',()=>{
  const r=setbackSweep(scene,input,{label:'long trailer',width:2.5,height:4,length:20});
  assert.match(r.error,/captured road extent/);assert.equal(r.rows,undefined);
});

test('answers fail on withheld claims and on numbers absent from computed facts',()=>{
  const facts=sweepFacts([setbackSweep(scene,input,vehicles[1])]),allowed=allowedNumbersFrom(facts);
  const good='Compact car: not blocked by the vehicle (rest unknown) from 4.33 m (14.21 ft); blocked from 2.58 to 4.08 m.';
  assert.deepEqual(checkAnswer(good,allowed),{ok:true,reason:'',text:good});
  assert.equal(checkAnswer('Not blocked by the vehicle (rest unknown) from 4.3 m.',allowed).ok,true);
  for(const claim of ['The crossing is clear.','The crossing is clearly open.','The child is visible.','Visibility is good.','This is safe.','Safer at 5 m.',
    'Parking here is legal.','That is illegal.','It is compliant.','That violates 22500(n).','This is dangerous.','Drivers can see the child.',"Drivers can't see her.",
    'Drivers cannot see her.','The driver will see her.','The driver sees the child.','Drivers see the child from 4.33 m.','The driver can still see the child from 4.33 m.',
    'Drivers are able to see her from 4.33 m.','A driver would be able to see the child.','Drivers can now see the child.','The driver saw her.',
    'The child is in view from 4.33 m.','The child is in sight from 4.33 m.','The path is unobstructed from 4.33 m.','Nothing blocks the line from 4.33 m.',
    'The view is unblocked from 4.33 m.','From 4.33 m the driver spots the child.','Drivers notice the child.','No obstruction from 4.33 m.','The crossing is open from 4.33 m.']) {
    const r=checkAnswer(claim,allowed);assert.equal(r.ok,false,claim);assert.match(r.reason,/withheld wording/);assert.equal(r.text,undefined);
  }
  assert.match(checkAnswer('Not blocked by the vehicle from 7.5 m.',allowed).reason,/7\.5/);
  assert.equal(checkAnswer('Not blocked by the vehicle from 4.4 m.',allowed).ok,false);
  // A leading decimal point is read as written: .5 is 0.5, not the 5 m SUV length.
  const four=allowedNumbersFrom(sweepFacts(vehicles.slice(0,4).map(v=>setbackSweep(scene,input,v))));
  assert.ok(four.includes(5));assert.match(checkAnswer('Not blocked by the vehicle from .5 m.',four).reason,/Number \.5 /);
  assert.equal(checkAnswer('Blocked at the 2.58 m parking spot.',allowed).ok,true);
  for(const empty of ['',' ',null])assert.equal(checkAnswer(empty,allowed).ok,false);
  // Ids and labels never ground numbers; only computed numeric values do.
  assert.deepEqual(allowedNumbersFrom({scene:'2018-10-19',vehicles:[{label:'26 ft box truck',stopsBlockingAt_m:null,tested_m:[1,20]}]}),[1,20]);
  assert.deepEqual(allowedNumbersFrom({label:'van 2'},{label:'4.33'}),[]);
});

test('sweep stays under 100 ms per vehicle on the legacy scene',t=>{
  // The low trailer is the worst case: it blocks no row, so its unknown run is confirmed by a full classification.
  for(const vehicle of vehicles) {
    let best=Infinity;
    // Best of three removes concurrent test-runner noise; the bound applies to each vehicle.
    for(let i=0;i<3;i++){const start=performance.now();setbackSweep(scene,input,vehicle);best=Math.min(best,performance.now()-start);}
    t.diagnostic(`${vehicle.label}: ${best.toFixed(1)} ms`);
    assert.ok(best<100,`${vehicle.label} sweep took ${best.toFixed(1)} ms`);
  }
});

test('overlong answers fail on length alone',()=>{
  const r=checkAnswer('Blocked. '.repeat(40),[]);
  assert.match(r.reason,/longer/);assert.equal(r.kind,'length');
});

// Audit C1/H1 pixel sets on the measured scene; roadPoint accepts every pixel, so a user can place them.
const sets={A:{crossing:[360,252],parking:[429,270],target:[630,324]},B:{crossing:[78,294],parking:[348,309],target:[423,255]}};
const own={label:'your vehicle',...input.vehicle};
const swept=(pixels,v=own)=>setbackSweep(scene,{pixels,driverDistance:placementFromPixels(scene,pixels).captureDistance},v,1.3);

test('result phrases name only the states their rows hold',()=>{
  // Default annotation: overlap rows, then blocked to the end. No stop distance is not "blocked at every distance".
  const def=setbackSweep(scene,input,own);
  assert.equal(def.stopsBlockingAt,null);
  assert.equal(sweepFacts([def]).vehicles[0].result,'blocked 2.58-19.82 m; overlaps person 0.08-2.33 m');
  // Set A: unknown and overlap rows only, not one blocked row.
  const a=swept(sets.A);
  assert.equal(a.rows.some(r=>r.state==='blocked'),false);assert.equal(a.stopsBlockingAt,null);
  assert.equal(sweepPhrase(a),'not blocked by the vehicle at any tested distance (rest unknown); overlaps person 5.83-10.49 m');
  // Set B: unknown rows before a blocked tail are never called blocked.
  const b=swept(sets.B);
  assert.deepEqual(b.rows.slice(0,4).map(r=>r.state),['unknown','unknown','unknown','blocked']);assert.equal(b.stopsBlockingAt,null);
  assert.equal(sweepPhrase(b),'blocked 0.83-7.08 m');
  // Every phrase is grounded as written; "every tested distance" only when every row is blocked; a stop distance only
  // after blocking.
  for(const pixels of [input.pixels,sets.A,sets.B,{...input.pixels,target:[592,318]}])for(const v of vehicles) {
    const r=swept(pixels,v);
    if(r.error)continue;
    const facts=sweepFacts([r]),phrase=facts.vehicles[0].result;
    assert.equal(phrase,sweepPhrase(r));
    assert.equal(/every tested distance/.test(phrase),r.rows.every(x=>x.state==='blocked'),phrase);
    assert.equal(/not blocked by the vehicle from/.test(phrase),r.stopsBlockingAt!==null&&r.rows.some(x=>x.state==='blocked'),phrase);
    assert.ok(checkAnswer(`${v.label}: ${phrase}.`,allowedNumbersFrom(facts)).ok,phrase);
  }
  assert.equal(sweepPhrase({error:'Vehicle extends beyond the captured road extent.'}),'Vehicle extends beyond the captured road extent.');
});

test('the app states the 20 ft and 15 ft marks itself',()=>{
  assert.equal(sweepMarks(setbackSweep(scene,input,own)),'20 ft: Blocked · 15 ft: Blocked');
  assert.equal(sweepMarks(swept(sets.A)),'20 ft: Overlap · 15 ft: Unknown');
  assert.equal(sweepMarks({at20ft:null,at15ft:'unknown'}),'20 ft: out of range · 15 ft: Unknown');
});

test('reference numbers are never grounded on their own',()=>{
  // Audit H3: 'your vehicle' is blocked at 20 ft with no stop distance, so 6.1 m or 20 ft cannot pass as one.
  const allowed=allowedNumbersFrom(sweepFacts([setbackSweep(scene,input,own)]));
  for(const claim of ['Your vehicle stops blocking at 6.1 m (20 ft).','At the 20 ft (6.1 m) mark your vehicle is not blocked by the vehicle (rest unknown).',
    'California Vehicle Code 22500(n) uses 20 ft (6.1 m); 15 ft is 4.6 m.']) {
    const r=checkAnswer(claim,allowed);assert.equal(r.ok,false,claim);assert.equal(r.kind,'number',claim);
  }
  // Ask mode without a sweep: no reference number is grounded either.
  assert.equal(checkAnswer('Your vehicle stops blocking at 6.1 m (20 ft).',allowedNumbersFrom(scenarioFacts(scene,createScenario(scene,input)))).ok,false);
});

test('legal, visibility and bare not-blocked wording is withheld',()=>{
  const allowed=allowedNumbersFrom(sweepFacts([setbackSweep(scene,input,vehicles[1])]));
  // Audit H2 strings (2.58 is in these facts), then the wider forms.
  for(const claim of ['At 20 ft your vehicle complies with Vehicle Code 22500(n).','Parking 20 ft back meets 22500(n), so it is permitted.',
    'At 20 ft parking is permitted and lawful.','At 20 ft you satisfy California 22500(n).','From 2.58 m the child is not blocked.',
    'From 2.58 m the driver has a line of sight to the child.','From 2.58 m the child is fully exposed.','From 2.58 m the driver has an open view of the child.',
    'Visibility improves from 2.58 m.','From 2.58 m the child is viewable.','From 2.58 m the crossing is free of obstruction.',
    'Parking at 2.58 m is allowed.','Parking under 2.58 m is prohibited.','Parking at 2.58 m is unlawful.','It complied at 2.58 m.','Your vehicle is in compliance at 2.58 m.',
    'This setback meets the crosswalk rule.','It satisfies the code at 2.58 m.','The 22500(n) requirement is met at 2.58 m.','The crossing is obstruction-free from 2.58 m.',
    'The driver has a full view from 2.58 m.','The driver has a clear view.','The driver has an unobstructed view.','The child is more exposed from 2.58 m.',
    'The driver can make out the child from 2.58 m.','Moving to 2.58 m reveals the child.','The child is no longer hidden from 2.58 m.','The child is hidden behind the van.',
    'The child comes into view from 2.58 m.','The child emerges from 2.58 m.','The child is uncovered from 2.58 m.','Your vehicle is not blocked from 2.58 m.',
    "The child isn't blocked from 2.58 m.",'The child is never blocked.','The child is no longer blocked from 2.58 m.','The child is invisible below 2.58 m.']) {
    const r=checkAnswer(claim,allowed);assert.equal(r.ok,false,claim);assert.equal(r.kind,'wording',claim);assert.match(r.reason,/withheld wording/);
  }
  for(const good of ['Compact car: not blocked by the vehicle from 4.33 m (14.21 ft).','Compact car: blocked 2.58-4.08 m; overlaps person 0.08-2.33 m.',
    'Not blocked by the vehicle (rest unknown) from 4.33 m.',"The compact car isn't blocked by the vehicle from 4.33 m.",'The compact car stops blocking at 4.33 m (14.21 ft).',
    'The emergency van is blocked from 2.58 to 4.08 m.','Blocked at the 2.58 m parking spot.'])
    assert.deepEqual(checkAnswer(good,allowed),{ok:true,reason:'',text:good},good);
  const yours=allowedNumbersFrom(sweepFacts([setbackSweep(scene,input,own)]));
  assert.equal(checkAnswer('Your vehicle: blocked 2.58-19.82 m; overlaps person 0.08-2.33 m.',yours).ok,true);
});

test('labels are plain names that carry no number or result',()=>{
  // Audit H7 labels: each once passed the old label check and carried digits past the number check.
  for(const label of ['not blocked from 2 m','stops blocking at 1 m','complies at 20 ft van','lawful SUV','blocked 3 m','box truck 10 ft back',
    'F-150','van 2','visible child','safe car','unknown van',"isn't blocking",'clear van','van at curb','overlap van',' van','x'.repeat(41)]) {
    const d=validateDeclaration({...declaration,vehicle_label:label}),t=validateDeclaration({...declaration,target_label:label}),s=validateSweep({vehicles:[{label,width_m:2,height_m:1.8,length_m:5}]});
    assert.equal(d.ok,false,label);assert.match(d.error,/^vehicle_label must be a plain name/);
    assert.equal(t.ok,false,label);assert.match(t.error,/^target_label must be a plain name/);
    assert.equal(s.ok,false,label);assert.match(s.error,/^vehicles\[0\]\.label must be a plain name/);
  }
  for(const label of ['your vehicle','current vehicle','current person','box truck','SUV',"driver's van",'parent & child','pick-up, long bed','x'.repeat(40)])
    assert.equal(validateSweep({vehicles:[{label,width_m:2,height_m:1.8,length_m:5}]}).ok,true,label);
  const w={pixels:input.pixels,vehicle:{width:'',height:'',length:''},targetHeight:'',input:null,setupInput:null};
  assert.match(runTool(scene,'ask',w,'declare_scenario',{...declaration,vehicle_label:'box truck 10 ft back'}).error,/plain name/);
  assert.match(runTool(scene,'check',{...w,input},'sweep_setbacks',{vehicles:[{label:'blocked 3 m',width_m:2,height_m:1.8,length_m:5}]}).error,/plain name/);
  assert.equal(checkAnswer('The box truck 10 ft back is blocked.',allowedNumbersFrom({vehicles:[{label:'box truck 10 ft back',tested_m:[1,20]}]})).ok,false);
});

test('a sweep needs a declared or entered vehicle and never changes the person height',()=>{
  // Audit H4: three points placed, empty vehicle fields, typed height 1.3, Nemotron sweeps with 1.1.
  const sweepArgs={vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]},p=placementFromPixels(scene,input.pixels);
  const empty={pixels:input.pixels,vehicle:{width:'',height:'',length:''},targetHeight:'1.3',input:null,setupInput:null};
  assert.deepEqual(runTool(scene,'ask',empty,'sweep_setbacks',{...sweepArgs,target_height_m:1.1}),{error:'Declare or enter a vehicle first.'});
  assert.deepEqual(runTool(scene,'ask',empty,'sweep_setbacks',sweepArgs),{error:'Declare or enter a vehicle first.'});
  // Typed fields (the app's setup input): the sweep runs on them, at the typed height only.
  const setupInput={pixels:input.pixels,vehicle:{width:'1.9',height:'2.4',length:'5.4'},targetHeight:'1.3',proposedSetback:p.baselineSetback,driverDistance:p.captureDistance};
  const typed={...empty,vehicle:setupInput.vehicle,setupInput};
  assert.deepEqual(runTool(scene,'ask',typed,'sweep_setbacks',{...sweepArgs,target_height_m:1.1}),{error:'Change the person height with declare_scenario first.'});
  assert.deepEqual(runTool(scene,'ask',{...typed,input},'sweep_setbacks',{...sweepArgs,target_height_m:1.295}),{error:'Change the person height with declare_scenario first.'});
  for(const asked of [undefined,1.3,1.304]) {
    const out=runTool(scene,'ask',typed,'sweep_setbacks',{...sweepArgs,...(asked!==undefined&&{target_height_m:asked})});
    assert.equal(out.error,undefined);assert.equal(out.fields,undefined);assert.equal(out.parts,undefined);
    assert.equal(out.input,setupInput);assert.equal(out.sweep.targetHeight,1.3);assert.equal(out.sweep.pixels,input.pixels);
    assert.deepEqual(out.sweep.results,[setbackSweep(scene,input,{label:'SUV',width:2,height:1.8,length:5})]);
    assert.deepEqual(out.result,sweepFacts(out.sweep.results));assert.deepEqual(out.sweep.own,setupInput.vehicle);
  }
});

test('editing one field drops only that field\'s Nemotron credit',()=>{
  const w={pixels:input.pixels,vehicle:{width:'',height:'',length:''},targetHeight:'',input:null,setupInput:null};
  const {parts,fields,input:declared}=runTool(scene,'ask',w,'declare_scenario',{...declaration,setback_value:10,setback_unit:'ft'});
  assert.equal(declaredLine(parts,declared),'test van 1.9 × 2.4 × 5.4 m (W × H × L) · adult 1.3 m · 3.05 m (10 ft) from crossing');
  // Audit H5: the user edits only the width (fields), then compares (setup input at the baseline setback).
  const p=placementFromPixels(scene,input.pixels),edited={...fields.vehicle,width:'2'};
  assert.equal(declaredLine(parts,{vehicle:edited,targetHeight:fields.targetHeight}),'test van height 2.4 m, length 5.4 m · adult 1.3 m');
  assert.equal(declaredLine(parts,{pixels:input.pixels,vehicle:edited,targetHeight:fields.targetHeight,proposedSetback:p.baselineSetback,driverDistance:p.captureDistance}),
    'test van height 2.4 m, length 5.4 m · adult 1.3 m');
  assert.equal(declaredLine(parts,{vehicle:fields.vehicle,targetHeight:'1.1'}),'test van 1.9 × 2.4 × 5.4 m (W × H × L)');
  assert.equal(declaredLine(parts,{vehicle:{width:'2',height:'2',length:'5'},targetHeight:'1.1'}),'');
  // A value restored exactly regains its credit; a later declared vehicle takes over while it holds.
  assert.equal(declaredLine(parts,{vehicle:fields.vehicle,targetHeight:'1.3'}),'test van 1.9 × 2.4 × 5.4 m (W × H × L) · adult 1.3 m');
  const more=[...parts,vehiclePart('SUV',{width:2,height:1.8,length:5})];
  assert.equal(declaredLine(more,{vehicle:{width:'2',height:'1.8',length:'5'},targetHeight:'1.3'}),'SUV 2 × 1.8 × 5 m (W × H × L) · adult 1.3 m');
  assert.equal(declaredLine(more,{vehicle:fields.vehicle,targetHeight:'1.3'}),'test van 1.9 × 2.4 × 5.4 m (W × H × L) · adult 1.3 m');
  assert.equal(declaredLine(parts,null),'');
});

test('a withheld answer gets a note true to why',()=>{
  const allowed=allowedNumbersFrom(sweepFacts([setbackSweep(scene,input,vehicles[1])]));
  // Audit H6: an answer that was only too long was once said to claim more than the geometry shows.
  const long='Compact car: blocked 2.58-4.08 m. '.repeat(9);
  assert.equal(checkAnswer(long.slice(0,34),allowed).ok,true);
  assert.equal(withheldNote(checkAnswer(long,allowed)),'Nemotron\'s answer was withheld: too long.');
  assert.equal(withheldNote(checkAnswer('Blocked from 2.6 m to 9.25 m.',allowed)),'Nemotron\'s answer was withheld: used a number Sightline did not compute.');
  assert.equal(withheldNote(checkAnswer('The child is visible from 4.33 m.',allowed)),'Nemotron\'s answer was withheld: used wording Sightline does not allow.');
  assert.equal(withheldNote(checkAnswer('  ',allowed)),'Nemotron returned no wording.');
});

test('a declaration that changes the vehicle size clears distance results',()=>{
  // Audit C2: a later declaration with the same person height once kept the previous vehicle's distance rows.
  const w={pixels:input.pixels,vehicle:{width:'1.9',height:'2.4',length:'5.4'},targetHeight:'1.3',input,setupInput:null};
  const truck=runTool(scene,'ask',w,'declare_scenario',{...declaration,vehicle_label:'box truck',vehicle_width_m:2.5,vehicle_height_m:3.5,vehicle_length_m:8});
  assert.equal(truck.error,undefined);assert.equal(truck.input.targetHeight,1.3);assert.equal(truck.clearSweep,true);
  for(const k of ['vehicle_width_m','vehicle_height_m','vehicle_length_m'])
    assert.equal(runTool(scene,'ask',w,'declare_scenario',{...declaration,[k]:declaration[k]+.001}).clearSweep,true,k);
  // Typed fields without a comparison count too.
  assert.equal(runTool(scene,'ask',{...w,input:null},'declare_scenario',{...declaration,vehicle_width_m:2}).clearSweep,true);
  assert.equal(runTool(scene,'ask',w,'declare_scenario',{...declaration,target_label:'child',target_height_m:1}).clearSweep,false);
  // Declarations are ask-only.
  assert.deepEqual(runTool(scene,'check',w,'declare_scenario',declaration),{error:'Unknown tool declare_scenario.'});
});

test('short summary phrases and extra withheld paraphrases',()=>{
  const r=(states,stop=null)=>({rows:states.map((state,i)=>({setback:i,state})),stopsBlockingAt:stop});
  assert.equal(sweepShort(r(['blocked','unknown'],1)),'not blocked by the vehicle from 1 m (3.28 ft)');
  assert.equal(sweepShort(r(['unknown','overlap'])),'not blocked by the vehicle at any tested distance');
  assert.equal(sweepShort(r(['overlap','blocked','overlap'])),'no tested distance where it stops blocking');
  for(const claim of ["The child is within the driver's view.","It stays in the field of view.","The line of sight is restored.","There is nothing in the way.","The child is in the open.","The child appears.","It is fine to park there.","You can park further back.","The driver's view is unaffected."])
    assert.equal(checkAnswer(claim,[]).ok,false,claim);
  assert.equal(checkAnswer('Compact car: not blocked by the vehicle from 4.33 m (14.21 ft). Your vehicle, SUV and delivery van: no tested distance where it stops blocking.',[4.33,14.21]).ok,true);
});

test('the distance check always includes the vehicle on screen',()=>{
  const p=placementFromPixels(scene,input.pixels),from={...input,proposedSetback:p.baselineSetback,driverDistance:p.captureDistance};
  const args={vehicles:[{label:'SUV',width_m:2,height_m:1.8,length_m:5}]};
  const check=runTool(scene,'check',{input:from,pixels:input.pixels},'sweep_setbacks',args);
  assert.deepEqual(check.sweep.results.map(r=>r.label),['your vehicle','SUV']);
  const ask=runTool(scene,'ask',{input:from,pixels:input.pixels},'sweep_setbacks',args);
  assert.deepEqual(ask.sweep.results.map(r=>r.label),['SUV']);
});


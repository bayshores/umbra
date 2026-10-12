import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {loadScene} from '../src/scene.mjs';
import {createScenario,measuredPoint} from '../src/geometry.mjs';
import {validateSuggestions,suggestedSetup,isStillSuggested,suggestionStatus} from '../src/suggestions.mjs';
import {scene,fetchAsset} from './scene-input.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const sceneFile=name=>readFile(new URL(`../public/scenes/${scene.id}/${name}`,import.meta.url));
const inputs={'scene.json':sha(await sceneFile('scene.json')),'recorded.jpg':sha(await sceneFile('recorded.jpg')),'depth.bin':scene.artifactHashes.depth};
// Test fixture: the iteration-1 geometry prototype pixels with the contract's declared defaults. Never imported by the app.
const fixture=()=>({schemaVersion:1,sceneId:scene.id,generatedUtc:'2026-10-11T21:09:14Z',
  generator:{script:'scripts/suggest-placements.py',sha256:'a'.repeat(64),python:'3.13.5',numpy:'2.3.1',pillow:'11.3.0'},
  inputs:{...inputs},
  crossing:{pixel:[572,242],status:'detected',method:'zebra-stripe detector on the depth-registered ground plane',alongRoadUncertaintyM:.5,distanceFromCameraM:29.35,barCount:11},
  roadAxis:{status:'detected',source:'curb face fit',barAgreementDeg:.34},
  curb:{status:'detected',rmsM:.05,stepHeightM:.22},
  vehicle:{pixel:[603,259],status:'declared-default',label:'SUV',dimensions:{width:1.9,height:1.8,length:4.9},setbackM:6.096,reason:'No parked vehicle detected in the curb lane; 20 ft default.'},
  person:{pixel:[593,242],status:'declared-default',label:'child',heightM:1.1,reason:'Curb end of the crossing near edge.'},
  detection:{status:'ran',model:'RF-DETR Medium',package:'rfdetr 1.11.2',license:'Apache-2.0',weightsSha256:'b'.repeat(64),detectionsSha256:'e'.repeat(64),curbLaneParked:0},
  limitations:['One frame and one scene.']});
const encode=value=>new TextEncoder().encode(JSON.stringify(value));
// Serves the fixture bytes as suggestions.json; every other asset is the real prepared scene.
const serving=bytes=>async path=>path===`/scenes/${scene.id}/suggestions.json`?new Response(bytes):fetchAsset(path);
// The first pixel (row-major) meeting a condition on the real depth image.
function findPixel(condition) {
  const c=scene.camera;
  for(let v=0;v<c.height;v++)for(let u=0;u<c.width;u++)if(condition([u,v]))return [u,v];
  throw new Error('No such pixel in the scene.');
}
const noDepth=findPixel(([u,v])=>scene.depth[v*scene.camera.width+u]===0);
const offRoad=findPixel(([u,v])=>scene.depth[v*scene.camera.width+u]>0&&measuredPoint(scene,[u,v])[1]>1);

test('a complete suggestion validates through roadPoint and createScenario with no clicks',()=>{
  const raw=fixture();
  assert.equal(validateSuggestions(raw,scene),raw);
  const setup=suggestedSetup(scene,raw);
  assert.deepEqual(setup.pixels,{crossing:[572,242],parking:[603,259],target:[593,242]});
  assert.deepEqual(setup.vehicle,{width:'1.9',height:'1.8',length:'4.9'});
  assert.equal(setup.targetHeight,'1.1');
  assert.equal(setup.error,'');
  assert.deepEqual(Object.fromEntries(Object.entries(setup.provenance).map(([k,p])=>[k,p.status])),
    {crossing:'detected',parking:'default',target:'default',vehicle:'default',targetHeight:'default'});
  const scenario=createScenario(scene,setup.input);
  assert.equal(setup.input.pixels,setup.pixels);
  assert.equal(scenario.proposedSetback,scenario.baselineSetback);
  assert.equal(setup.input.driverDistance,scenario.limits.captureDistance);
  // The nearest measured-road vehicle front lands within 0.1 m of the declared 20 ft default.
  assert.ok(Math.abs(scenario.baselineSetback-6.096)<.1,`baseline ${scenario.baselineSetback}`);
});

test('loader verifies, validates and returns the suggestions with the scene',async()=>{
  const bytes=encode(fixture()),loaded=await loadScene(serving(bytes),scene.id,undefined,undefined,{file:'suggestions.json',sha256:sha(bytes)});
  assert.deepEqual(loaded.suggestions,fixture());
  assert.equal(Object.hasOwn(await loadScene(fetchAsset,scene.id),'suggestions'),false,'No catalog entry means no suggestions.');
});

test('a bad, missing or stale suggestions file fails the load explicitly',async()=>{
  const bytes=encode(fixture()),load=(served,descriptor)=>loadScene(serving(served),scene.id,undefined,undefined,descriptor);
  await assert.rejects(load(bytes,{file:'suggestions.json',sha256:'c'.repeat(64)}),/Placement suggestions checksum mismatch/);
  await assert.rejects(load(null,{file:'missing.json',sha256:sha(bytes)}),/Placement suggestions unavailable \(404\)/);
  for(const descriptor of [null,{},{file:'suggestions.json'},{file:'../scene.json',sha256:sha(bytes)},{file:'suggestions.json',sha256:sha(bytes),extra:1}])
    await assert.rejects(load(bytes,descriptor),/Invalid placement suggestions descriptor/);
  const text=new TextEncoder().encode('{"schemaVersion":1');
  await assert.rejects(load(text,{file:'suggestions.json',sha256:sha(text)}),/not UTF-8 JSON/);
  const invalid=encode({...fixture(),extra:true});
  await assert.rejects(load(invalid,{file:'suggestions.json',sha256:sha(invalid)}),/unexpected field file\.extra/);
  for(const key of ['scene.json','recorded.jpg','depth.bin']) {
    const stale=encode({...fixture(),inputs:{...inputs,[key]:'d'.repeat(64)}});
    await assert.rejects(load(stale,{file:'suggestions.json',sha256:sha(stale)}),/prepared from different scene inputs/);
  }
});

test('strict schema rejects unknown keys, out-of-range pixels, non-positive numbers and unknown statuses',()=>{
  const cases={
    'not an object':()=>[],
    'extra top-level key':r=>{r.confidence=1;},
    'missing limitations':r=>{delete r.limitations;},
    'schema version':r=>{r.schemaVersion=2;},
    'other scene':r=>{r.sceneId='other';},
    'timestamp':r=>{r.generatedUtc='yesterday';},
    'generator script':r=>{r.generator.script='detect_geometry.py';},
    'input hash':r=>{r.inputs['recorded.jpg']='abc';},
    'fractional pixel':r=>{r.crossing.pixel=[572.5,242];},
    'pixel right of image':r=>{r.crossing.pixel=[scene.camera.width,242];},
    'pixel below image':r=>{r.person.pixel=[593,scene.camera.height];},
    'negative pixel':r=>{r.vehicle.pixel=[-1,259];},
    'three-value pixel':r=>{r.vehicle.pixel=[603,259,0];},
    'string pixel':r=>{r.vehicle.pixel=['603','259'];},
    'zero width':r=>{r.vehicle.dimensions.width=0;},
    'negative height':r=>{r.person.heightM=-1.1;},
    'string length':r=>{r.vehicle.dimensions.length='4.9';},
    'missing dimension':r=>{delete r.vehicle.dimensions.height;},
    'zero uncertainty':r=>{r.crossing.alongRoadUncertaintyM=0;},
    'extra element key':r=>{r.crossing.confidence=.9;},
    'unknown status':r=>{r.crossing.status='guessed';},
    'detected person':r=>{r.person.status='detected';},
    'unknown with pixel':r=>{r.person={status:'unknown',reason:'No curb.',pixel:[593,242]};},
    'unknown without reason':r=>{r.vehicle={status:'unknown'};},
    'default without reason':r=>{delete r.vehicle.reason;},
    'label with number':r=>{r.vehicle.label='SUV 2';},
    'detected vehicle without a parked detection':r=>{r.vehicle.status='detected';},
    'defaults without a crossing':r=>{r.crossing={status:'unknown',reason:'No crosswalk bars.'};},
    'detector without weights hash':r=>{delete r.detection.weightsSha256;},
    'detector without output hash':r=>{delete r.detection.detectionsSha256;},
    'detector output hash':r=>{r.detection.detectionsSha256='E'.repeat(64);},
    'detector status':r=>{r.detection.status='skipped';},
    'negative parked count':r=>{r.detection.curbLaneParked=-1;},
    'bar agreement':r=>{r.roadAxis.barAgreementDeg=-1;},
    'limitation type':r=>{r.limitations=[3];},
  };
  for(const [name,mutate] of Object.entries(cases)) {
    const raw=fixture(),replaced=mutate(raw);
    assert.throws(()=>validateSuggestions(replaced??raw,scene),/^Error: Invalid placement suggestions: /,name);
  }
  // Allowed variations stay valid: unknown elements with reasons and a detector that did not run.
  const raw=fixture();
  Object.assign(raw,{roadAxis:{status:'unknown',reason:'No curb face.'},curb:{status:'unknown',reason:'No depth step.'},
    vehicle:{status:'unknown',reason:'No curb lane.'},person:{status:'unknown',reason:'No curb.'},detection:{status:'not-run'}});
  assert.equal(validateSuggestions(raw,scene),raw);
});

test('unknown elements and pixels failing the road rule stay unplaced with their reason',()=>{
  const raw=fixture();
  raw.vehicle={status:'unknown',reason:'No curb lane established.'};
  raw.person.pixel=noDepth;raw.crossing.pixel=offRoad;
  validateSuggestions(raw,scene);
  const setup=suggestedSetup(scene,raw);
  assert.deepEqual(setup.pixels,{});
  assert.deepEqual(setup.vehicle,{width:'',height:'',length:''});
  assert.equal(setup.targetHeight,'1.1');
  assert.equal(setup.input,null);
  assert.deepEqual(setup.provenance.parking,{status:'unplaced',reason:'No curb lane established.'});
  assert.equal(setup.provenance.vehicle,undefined);
  assert.equal(setup.provenance.target.status,'unplaced');
  assert.match(setup.provenance.target.reason,/No measured depth at this pixel/);
  assert.match(setup.provenance.crossing.reason,/Choose crossing on the measured road surface/);
  const status=suggestionStatus(raw,setup,{pixels:{},vehicle:setup.vehicle,targetHeight:setup.targetHeight});
  assert.deepEqual(status.parts.map(p=>p.text),['crossing unknown','vehicle unknown','person unknown']);
  assert.equal(status.parts[1].title,'No curb lane established.');
  assert.equal(status.edited,false);
  assert.deepEqual(status.context,{person_height:'declared default child'});
  // Once the user places the object, its unknown note goes away.
  assert.deepEqual(suggestionStatus(raw,setup,{pixels:{parking:[600,250]},vehicle:setup.vehicle,targetHeight:''}).parts.map(p=>p.text),['crossing unknown','person unknown']);
});

test('isStillSuggested follows the value on screen; edits lose the label and the line',()=>{
  const raw=fixture(),setup=suggestedSetup(scene,raw),p=setup.provenance;
  assert.equal(isStillSuggested('crossing',[572,242],p),true);
  assert.equal(isStillSuggested('crossing',[573,242],p),false);
  assert.equal(isStillSuggested('target',undefined,p),false);
  assert.equal(isStillSuggested('vehicle',{width:'1.9',height:'1.8',length:'4.9'},p),true);
  assert.equal(isStillSuggested('vehicle',{width:'1.90',height:'1.8',length:'4.9'},p),true);
  assert.equal(isStillSuggested('vehicle',{width:'2',height:'1.8',length:'4.9'},p),false);
  assert.equal(isStillSuggested('vehicle',{width:'',height:'1.8',length:'4.9'},p),false);
  assert.equal(isStillSuggested('targetHeight','1.1',p),true);
  assert.equal(isStillSuggested('targetHeight','1.3',p),false);
  assert.equal(isStillSuggested('targetHeight','',p),false);
  assert.equal(isStillSuggested('other',[572,242],p),false);
  assert.equal(isStillSuggested('crossing',[572,242],{crossing:{status:'unplaced',reason:'x'}}),false);
  const current={pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight},initial=suggestionStatus(raw,setup,current);
  assert.deepEqual(initial.parts.map(p=>p.text),['crosswalk detected ±0.5 m','SUV at 20 ft (default, no parked vehicle)','child 1.1 m (default)']);
  assert.equal(initial.edited,false);
  assert.deepEqual(Object.keys(initial.context),['crossing','vehicle_front','vehicle_size','person_point','person_height']);
  assert.ok(Object.values(initial.context).every(v=>!/\d/.test(v)),'Context strings carry no numbers.');
  const moved=suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,target:[600,250]}});
  assert.deepEqual(moved.parts.map(p=>p.text),['crosswalk detected ±0.5 m','SUV at 20 ft (default, no parked vehicle)']);
  assert.equal(moved.edited,true);
  assert.equal(moved.context.person_point,undefined);
  assert.equal(moved.context.person_height,'declared default child');
  const resized=suggestionStatus(raw,setup,{...current,vehicle:{...setup.vehicle,length:'5.4'}});
  assert.deepEqual(resized.parts.map(p=>p.text),['crosswalk detected ±0.5 m','child 1.1 m (default)']);
  assert.equal(resized.context.vehicle_size,undefined);
  // The 20 ft is measured from the suggested crossing; once the crossing moves the distance is no longer shown.
  const crossed=suggestionStatus(raw,setup,{...current,pixels:{...setup.pixels,crossing:[572,247]}});
  assert.deepEqual(crossed.parts.map(p=>p.text),['SUV (default, no parked vehicle)','child 1.1 m (default)']);
  assert.equal(crossed.context.vehicle_front,'declared default, not a parked car in the image');
  // Without a detector run the default makes no claim about parked vehicles.
  const unrun=fixture();unrun.detection={status:'not-run'};
  assert.deepEqual(suggestionStatus(unrun,suggestedSetup(scene,unrun),current).parts.map(p=>p.text),
    ['crosswalk detected ±0.5 m','SUV at 20 ft (default)','child 1.1 m (default)']);
});

test('a detected vehicle front keeps the declared default size labelled as a default',()=>{
  const raw=fixture();
  raw.vehicle={pixel:[603,259],status:'detected',label:'SUV',dimensions:{width:1.9,height:1.8,length:4.9},setbackM:6.1,
    reason:'Parked car detected (score 0.90); front from measured depth; SUV dimensions declared.'};
  raw.detection.curbLaneParked=1;
  validateSuggestions(raw,scene);
  const setup=suggestedSetup(scene,raw);
  assert.equal(setup.provenance.parking.status,'detected');
  assert.equal(setup.provenance.vehicle.status,'default');
  const status=suggestionStatus(raw,setup,{pixels:setup.pixels,vehicle:setup.vehicle,targetHeight:setup.targetHeight});
  assert.deepEqual(status.parts.map(p=>p.text),['crosswalk detected ±0.5 m','parked vehicle front detected','SUV size (default)','child 1.1 m (default)']);
  assert.equal(status.context.vehicle_front,'parked car in the image');
  assert.equal(status.context.vehicle_size,'declared default SUV');
});

test('the prepared catalog suggestions load and open a valid comparison with no clicks',async()=>{
  const catalog=JSON.parse(await readFile(new URL('../public/scenes/index.json',import.meta.url),'utf8')),entry=catalog.find(e=>e.id===scene.id);
  const loaded=await loadScene(fetchAsset,entry.id,entry.metadataSha256,undefined,entry.suggestions);
  const setup=suggestedSetup(loaded,loaded.suggestions);
  assert.equal(setup.error,'');
  assert.deepEqual(Object.keys(setup.pixels).sort(),['crossing','parking','target']);
  createScenario(loaded,setup.input);
});

import { roadPoint, placementFromPixels, createScenario } from './geometry.mjs';

// Placement suggestions come from a one-time preparation step (suggestions.json, schemaVersion 1), never from runtime
// guessing. A declared default is a hypothetical starting value, not a measurement. Anything the file marks unknown, or
// that fails the measured-road rule here, stays unplaced with its reason.
const FOOT=.3048,DIMS=['width','height','length'];
const ROLES=[['crossing','crossing'],['parking','vehicle'],['target','person']];
const SOURCE={detected:'detected','declared-default':'default'};
const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const text=v=>typeof v==='string'&&v.trim().length>0&&v.length<=1000;
const name=v=>typeof v==='string'&&/^[A-Za-z][A-Za-z ,'&-]{0,39}$/.test(v);
const positive=v=>typeof v==='number'&&Number.isFinite(v)&&v>0;
const filled=v=>v!==''&&v!=null;
const fmt=n=>String(Number(n.toFixed(2)));
function fail(detail) {throw new Error(`Invalid placement suggestions: ${detail}.`);}
function fields(v,required,optional,where) {
  if(!object(v))fail(`${where} must be an object`);
  const extra=Object.keys(v).find(k=>!required.includes(k)&&!optional.includes(k));
  if(extra)fail(`unexpected field ${where}.${extra}`);
  const missing=required.find(k=>!Object.hasOwn(v,k));
  if(missing)fail(`missing field ${where}.${missing}`);
}
function pixel(v,scene,where) {
  const c=scene.camera;
  if(!Array.isArray(v)||v.length!==2||!v.every(Number.isSafeInteger)||v[0]<0||v[0]>=c.width||v[1]<0||v[1]>=c.height)
    fail(`${where}.pixel must be an integer pixel inside the recorded image`);
}
// Returns the element's status after checking its fields; unknown elements carry a reason and nothing else.
function element(v,scene,where,statuses,check) {
  if(!object(v)||![...statuses,'unknown'].includes(v.status))fail(`${where}.status must be ${[...statuses,'unknown'].join(' or ')}`);
  if(v.status==='unknown'){fields(v,['status','reason'],[],where);if(!text(v.reason))fail(`${where}.reason must be text`);}
  else check(v,where);
  return v.status;
}
function numbers(v,keys,where) {
  const bad=keys.find(k=>!positive(v[k]));
  if(bad)fail(`${where}.${bad} must be a finite positive number`);
}

export function validateSuggestions(raw,scene) {
  if(scene?.measurements)fail('placement is unavailable for this scene');
  fields(raw,['schemaVersion','sceneId','generatedUtc','generator','inputs','crossing','roadAxis','curb','vehicle','person','detection','limitations'],[],'file');
  if(raw.schemaVersion!==1)fail('schemaVersion must be 1');
  if(raw.sceneId!==scene.id)fail('sceneId does not match the loaded scene');
  if(typeof raw.generatedUtc!=='string'||!Number.isFinite(Date.parse(raw.generatedUtc)))fail('generatedUtc must be a timestamp');
  fields(raw.generator,['script','sha256','python','numpy','pillow'],[],'generator');
  if(raw.generator.script!=='scripts/suggest-placements.py'||!hash(raw.generator.sha256)||!['python','numpy','pillow'].every(k=>text(raw.generator[k])))
    fail('generator must name scripts/suggest-placements.py, its sha256 and versions');
  fields(raw.inputs,['scene.json','recorded.jpg','depth.bin'],[],'inputs');
  if(!Object.values(raw.inputs).every(hash))fail('inputs must be sha256 hashes');
  const crossing=element(raw.crossing,scene,'crossing',['detected'],(v,w)=>{
    fields(v,['status','pixel','method','alongRoadUncertaintyM','distanceFromCameraM','barCount'],[],w);
    pixel(v.pixel,scene,w);numbers(v,['alongRoadUncertaintyM','distanceFromCameraM'],w);
    if(!text(v.method))fail(`${w}.method must be text`);
    if(!Number.isSafeInteger(v.barCount)||v.barCount<1)fail(`${w}.barCount must be a positive integer`);
  });
  element(raw.roadAxis,scene,'roadAxis',['detected'],(v,w)=>{
    fields(v,['status','source','barAgreementDeg'],[],w);
    if(!text(v.source))fail(`${w}.source must be text`);
    if(typeof v.barAgreementDeg!=='number'||!(v.barAgreementDeg>=0&&v.barAgreementDeg<=90))fail(`${w}.barAgreementDeg must be 0 to 90`);
  });
  element(raw.curb,scene,'curb',['detected'],(v,w)=>{fields(v,['status','rmsM','stepHeightM'],[],w);numbers(v,['rmsM','stepHeightM'],w);});
  const detection=raw.detection,ran=detection?.status==='ran';
  if(!object(detection)||!['ran','not-run'].includes(detection.status))fail('detection.status must be ran or not-run');
  // detectionsSha256 pins the detector output that curbLaneParked was counted from.
  const detector=['model','package','license','weightsSha256','detectionsSha256','curbLaneParked'];
  fields(detection,ran?['status',...detector]:['status'],ran?[]:[...detector,'reason'],'detection');
  if(['model','package','license','reason'].some(k=>Object.hasOwn(detection,k)&&!text(detection[k]))||
    ['weightsSha256','detectionsSha256'].some(k=>Object.hasOwn(detection,k)&&!hash(detection[k]))||
    (Object.hasOwn(detection,'curbLaneParked')&&!(Number.isSafeInteger(detection.curbLaneParked)&&detection.curbLaneParked>=0)))
    fail('detection must name its model, package, license, weights and output sha256 and parked count');
  const vehicle=element(raw.vehicle,scene,'vehicle',['detected','declared-default'],(v,w)=>{
    const declared=v.status==='declared-default';
    fields(v,declared?['status','pixel','label','dimensions','setbackM','reason']:['status','pixel','label','dimensions'],declared?[]:['setbackM','reason'],w);
    pixel(v.pixel,scene,w);
    if(!name(v.label))fail(`${w}.label must be a plain name`);
    fields(v.dimensions,DIMS,[],w+'.dimensions');numbers(v.dimensions,DIMS,w+'.dimensions');
    if(Object.hasOwn(v,'setbackM'))numbers(v,['setbackM'],w);
    if(Object.hasOwn(v,'reason')&&!text(v.reason))fail(`${w}.reason must be text`);
  });
  const person=element(raw.person,scene,'person',['declared-default'],(v,w)=>{
    fields(v,['status','pixel','label','heightM','reason'],[],w);
    pixel(v.pixel,scene,w);numbers(v,['heightM'],w);
    if(!name(v.label))fail(`${w}.label must be a plain name`);
    if(!text(v.reason))fail(`${w}.reason must be text`);
  });
  // A detected vehicle needs a detector run that found one; defaults are placed from the crossing, so they need it.
  if(vehicle==='detected'&&!(ran&&detection.curbLaneParked>0))fail('a detected vehicle requires a detector run with a parked vehicle');
  if(crossing==='unknown'&&(vehicle==='declared-default'||person!=='unknown'))fail('defaults cannot be placed without a crossing');
  if(!Array.isArray(raw.limitations)||!raw.limitations.every(text))fail('limitations must be a list of text');
  return raw;
}

// The app's starting values from a validated file: pixels (only those that pass roadPoint here), field strings, and
// provenance per value. input is the comparison input when everything is placed and createScenario accepts it.
export function suggestedSetup(scene,suggestions) {
  const pixels={},provenance={},s=suggestions;
  for(const [key,role] of ROLES) {
    const e=s[role];
    if(e.status==='unknown'){provenance[key]={status:'unplaced',reason:e.reason};continue;}
    try{roadPoint(scene,e.pixel,role);}catch(error){provenance[key]={status:'unplaced',reason:error.message};continue;}
    pixels[key]=[...e.pixel];provenance[key]={status:SOURCE[e.status],value:[...e.pixel]};
    if(role==='vehicle'&&e.status==='declared-default')provenance[key].setbackFt=Math.round(e.setbackM/FOOT);
  }
  // Schema v1 detects at most the vehicle front; its size is always the declared default.
  if(s.vehicle.status!=='unknown')provenance.vehicle={status:'default',value:{...s.vehicle.dimensions}};
  if(s.person.status!=='unknown')provenance.targetHeight={status:SOURCE[s.person.status],value:s.person.heightM};
  const vehicle=Object.fromEntries(DIMS.map(k=>[k,provenance.vehicle?String(s.vehicle.dimensions[k]):''])),
    targetHeight=provenance.targetHeight?String(s.person.heightM):'';
  let input=null,error='';
  if(ROLES.every(([key])=>pixels[key])&&provenance.vehicle&&provenance.targetHeight) {
    try {
      const placement=placementFromPixels(scene,pixels),next={pixels,vehicle,targetHeight,proposedSetback:placement.baselineSetback,driverDistance:placement.captureDistance};
      createScenario(scene,next);input=next;
    }catch(e){error=e.message;}
  }
  return {pixels,vehicle,targetHeight,provenance,input,error};
}

// key: crossing, parking or target (current is a pixel), vehicle (the size fields) or targetHeight (the height field).
export function isStillSuggested(key,current,provenance) {
  const p=provenance?.[key];
  if(!p||p.status==='unplaced')return false;
  if(key==='vehicle')return DIMS.every(k=>filled(current?.[k])&&Number(current[k])===p.value[k]);
  if(key==='targetHeight')return filled(current)&&Number(current)===p.value;
  return Array.isArray(current)&&current.length===2&&current[0]===p.value[0]&&current[1]===p.value[1];
}

// The status line parts for the values still on screen, whether anything suggested was edited, and the Nemotron
// context entry. A vehicle or person is listed only while both its point and its size still hold the suggestion; the
// default vehicle's distance is shown only while the crossing it was measured from is also still suggested.
// Context strings carry no numbers: Sightline's computed facts already hold every value.
export function suggestionStatus(suggestions,setup,current) {
  const p=setup.provenance,s=suggestions,
    value=key=>key==='vehicle'?current.vehicle:key==='targetHeight'?current.targetHeight:current.pixels?.[key],
    // A value Nemotron declared replaces the suggestion's label even when it happens to equal the suggested value.
    still=key=>!current.overridden?.includes(key)&&isStillSuggested(key,value(key),p);
  const edited=Object.keys(p).some(key=>p[key].status!=='unplaced'&&!still(key));
  const parts=[],context={};
  const unknown=(key,word)=>{if(p[key].status==='unplaced'&&!current.pixels?.[key])parts.push({key,text:`${word} unknown`,title:p[key].reason});};
  if(still('crossing'))parts.push({key:'crossing',text:`crosswalk detected ±${fmt(s.crossing.alongRoadUncertaintyM)} m`,title:`${s.crossing.method} · ±${fmt(s.crossing.alongRoadUncertaintyM)} m along the road`});
  unknown('crossing','crossing');
  if(still('parking')&&still('vehicle')) {
    if(p.parking.status==='detected') {
      parts.push({key:'parking',text:'parked vehicle front detected',title:s.vehicle.reason});
      parts.push({key:'vehicle',text:`${s.vehicle.label} size (default)`,title:s.vehicle.reason});
    }else{
      const at=still('crossing')&&positive(s.vehicle.setbackM)?` at ${fmt(s.vehicle.setbackM/FOOT)} ft`:'',
        none=s.detection.status==='ran'&&s.detection.curbLaneParked===0?', no parked vehicle':'';
      parts.push({key:'parking',text:`${s.vehicle.label}${at} (default${none})`,title:s.vehicle.reason});
    }
  }
  unknown('parking','vehicle');
  if(still('target')&&still('targetHeight'))parts.push({key:'target',text:`${s.person.label} ${fmt(s.person.heightM)} m (default)`,title:s.person.reason});
  unknown('target','person');
  const source={detected:'parked car in the image',default:'declared default, not a parked car in the image'};
  if(still('crossing'))context.crossing='measured crosswalk stripes';
  if(still('parking'))context.vehicle_front=source[p.parking.status];
  if(still('vehicle'))context.vehicle_size=`declared default ${s.vehicle.label}`;
  if(still('target'))context.person_point='declared default';
  if(still('targetHeight'))context.person_height=`declared default ${s.person.label}`;
  return {parts,edited,context:Object.keys(context).length?context:null};
}

import { roadPoint, anchorPoint, comparisonStart, createScenario, isMappedGround, mappedGroundHit, mappedGroundHeight, mappedTravel, namesAnnotatedFront, SUPPORT_LABELS, SUPPORT_CONTEXT, SIZE_LIMITS } from './geometry.mjs';

// Placement suggestions come from a one-time preparation step (suggestions.json), never from runtime guessing. Schema 1
// is the measured-depth (DrivingStereo) file; schema 3 is the map-first (mapped ground, protocol v3) file. A declared default is a
// hypothetical starting value, not a measurement. Anything the file marks unknown, or that fails the scene's ground rule
// here, stays unplaced with its reason.
const FOOT=.3048,DIMS=['width','height','length'];
const ROLES=[['crossing','crossing'],['parking','vehicle'],['target','person']];
const SOURCE={detected:'detected','declared-default':'default',mapped:'mapped',annotated:'annotated'};
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

// The DrivingStereo detector block; crossCheck (an optional second detector run) has exactly the same fields.
const DETECTOR=['model','package','license','weightsSha256','detectionsSha256','curbLaneParked'];
function detectorFields(block,ran,where) {
  fields(block,ran?['status',...DETECTOR]:['status'],ran?[]:[...DETECTOR,'reason'],where);
  if(['model','package','license','reason'].some(k=>Object.hasOwn(block,k)&&!text(block[k]))||
    ['weightsSha256','detectionsSha256'].some(k=>Object.hasOwn(block,k)&&!hash(block[k]))||
    (Object.hasOwn(block,'curbLaneParked')&&!(Number.isSafeInteger(block.curbLaneParked)&&block.curbLaneParked>=0)))
    fail(`${where} must name its model, package, license, weights and output sha256 and parked count`);
}

// ---------------------------------------------------------------- schema 3: mapped ground (map-first AV2 scenes, protocol v3)
// Schema 2 (protocol v1/v2) files are history and are refused.
const V3_TOP=['schemaVersion','sceneId','sourceClass','generatedUtc','generator','inputs','protocolSha256','license','tolerances',
  'crossing','roadAxis','curb','vehicle','person','limitations'];
const V3_INPUTS=['scene.json','ground.json','terrain.bin','recorded.png','referencepositions.bin','referenceeligible.bin','protocol.json'];
const V3_TOLERANCES=['roadToleranceMeters','supportHeightAgreementMeters','supportRadiusMeters','maxRangeMeters','placementRangeLimitMeters'];
const V3_HIT=['status','label','reason','pixel','pixelUV','groundPoint','supportClass','supportLabel','hidden','alongRayUncertaintyM','rangeM','horizontalRangeM',
  'hitKind','terrainCityZ','terrainCell','mappedPoint','pixelOffsetFromMappedM','integerPixelCheck','supportEvidence'];
// Per role and status: [required extra fields, optional extra fields].
const V3_EXTRA={
  crossing:{mapped:[['nearEdge','crossingError','forwardM'],[]]},
  vehicle:{annotated:[['dimensions','track','category','motion','pose','headingWorld','headingToRoadAxisDeg','yawToRoadAxisDeg','frontGapToCrossingM','appModel',
    'frontResidualToAnnotatedM'],['pixelGroundGapAlongAxisM']],
  'declared-default':[['dimensions','setbackM','annotatedVehicleRejected','curbLateralAlongLengthM','lateralFromCameraM'],['pixelGroundGapAlongAxisM']]},
  person:{'declared-default':[['heightM'],['topHiddenDiagnostic']]},
};
// protocol v3 crossingError: the crossing stays the mapped near edge; its shown error is the paint offset or a status.
const CROSSING_ERROR_LABELS={unmeasured:'paint offset unmeasured','pending-measurement':'paint offset pending measurement',unresolved:'paint offset unresolved'};
const finiteNumber=v=>typeof v==='number'&&Number.isFinite(v);
const finiteList=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finiteNumber);
const intPair=v=>Array.isArray(v)&&v.length===2&&v.every(Number.isSafeInteger);
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const inCrop=(v,c)=>finiteList(v,2)&&v[0]>=0&&v[0]<=c.width-1&&v[1]>=0&&v[1]<=c.height-1;
const anchorOf=uv=>[Math.floor(uv[0]+.5),Math.floor(uv[1]+.5)];
// The runtime rule's result as the file stores it: within 1e-6 m for points, ranges and the uncertainty, the rest equal.
function sameHit(hit,v) {
  return !hit.point.some((n,i)=>Math.abs(n-v.groundPoint[i])>1e-6)&&hit.supportClass===v.supportClass&&hit.hidden===v.hidden&&hit.hitKind===v.hitKind&&
    hit.terrainCityZ===v.terrainCityZ&&same(hit.terrainCell,v.terrainCell)&&Math.abs(hit.rangeM-v.rangeM)<=1e-6&&
    Math.abs(hit.alongRayUncertaintyM-v.alongRayUncertaintyM)<=1e-6;
}
// One placed (or unknown) object: its fields, then the runtime rule re-run at its stored pixelUV (the position the app
// evaluates while the suggestion is untouched) must reproduce it; pixel is the integer pixel containing pixelUV.
function mappedElement(v,scene,role,check) {
  const statuses=Object.keys(V3_EXTRA[role]),c=scene.camera;
  if(!object(v)||![...statuses,'unknown'].includes(v.status))fail(`${role}.status must be ${[...statuses,'unknown'].join(' or ')}`);
  if(v.status==='unknown') {
    const optional=new Set(Object.values(V3_EXTRA[role]).flat(2));['label','mappedPoint','pixelUV','pixel'].forEach(k=>optional.add(k));
    fields(v,['status','supportClass','reason'],[...optional],role);
    if(v.supportClass!=='unknown'||!text(v.reason))fail(`${role} unknown must carry supportClass unknown and a reason`);
    if(Object.hasOwn(v,'pixelUV')&&(!inCrop(v.pixelUV,c)||!same(v.pixel,anchorOf(v.pixelUV))))fail(`${role}.pixelUV and pixel must be a crop position and the pixel containing it`);
    return v.status;
  }
  const [required,optional]=V3_EXTRA[role][v.status];
  fields(v,[...V3_HIT,...required],optional,role);
  pixel(v.pixel,scene,role);
  if(!inCrop(v.pixelUV,c)||!same(v.pixel,anchorOf(v.pixelUV)))fail(`${role}.pixelUV must be a crop position and pixel the pixel containing it`);
  if(!text(v.label)||!text(v.reason))fail(`${role}.label and reason must be text`);
  if(!['observed-nearby','map-only'].includes(v.supportClass)||v.supportLabel!==SUPPORT_LABELS[v.supportClass])fail(`${role}.supportClass and supportLabel must match`);
  if(scene.ground.cuboids.available?typeof v.hidden!=='boolean':v.hidden!==null)fail(`${role}.hidden must be a flag, or null without annotations`);
  if(!finiteList(v.groundPoint,3)||!finiteList(v.mappedPoint,3)||!['top','step'].includes(v.hitKind)||!finiteNumber(v.terrainCityZ)||!intPair(v.terrainCell)||
    !['alongRayUncertaintyM','rangeM','horizontalRangeM'].every(k=>positive(v[k]))||!(finiteNumber(v.pixelOffsetFromMappedM)&&v.pixelOffsetFromMappedM>=0))
    fail(`${role} ray diagnostics must be finite`);
  if(!object(v.supportEvidence)||!Object.values(v.supportEvidence).every(n=>n===null||finiteNumber(n)&&n>=0))fail(`${role}.supportEvidence must hold counts`);
  let hit;
  try{hit=mappedGroundHit(scene,v.pixelUV);}catch(error){fail(`${role}.pixelUV does not reach mapped ground (${error.message})`);}
  const horizontal=(a,b)=>Math.hypot(a[0]-b[0],a[2]-b[2]);
  if(!sameHit(hit,v)||Math.abs(Math.hypot(v.groundPoint[0],v.groundPoint[2])-v.horizontalRangeM)>1e-6||
    Math.abs(horizontal(v.groundPoint,v.mappedPoint)-v.pixelOffsetFromMappedM)>1e-6)
    fail(`${role} disagrees with the runtime mapped-ground rule`);
  // The integer anchor's own result is a diagnostic, but it must be true.
  const ic=v.integerPixelCheck;
  let anchor=null;try{anchor=mappedGroundHit(scene,v.pixel);}catch(error){anchor=error.message;}
  if(!object(ic)||(typeof anchor==='string'?!same(Object.keys(ic),['reason'])||ic.reason!==anchor
    :Object.keys(ic).sort().join()!=='groundPoint,residualM'||!finiteList(ic.groundPoint,3)||anchor.point.some((n,i)=>Math.abs(n-ic.groundPoint[i])>1e-6)||
      Math.abs(horizontal(anchor.point,v.mappedPoint)-ic.residualM)>1e-6))
    fail(`${role}.integerPixelCheck disagrees with the runtime mapped-ground rule`);
  check(v);
  return v.status;
}
function validateMapped(raw,scene) {
  if(!object(raw)||raw.schemaVersion!==3)fail('schemaVersion must be 3 for a mapped-ground scene (protocol v3; earlier files are history)');
  fields(raw,V3_TOP,[],'file');
  if(raw.sceneId!==scene.id)fail('sceneId does not match the loaded scene');
  if(raw.sourceClass!=='mappedGround')fail('sourceClass must be mappedGround');
  if(typeof raw.generatedUtc!=='string'||!Number.isFinite(Date.parse(raw.generatedUtc)))fail('generatedUtc must be a timestamp');
  // Sean (October 11): no word built on "park" anywhere for this vehicle, so the file may not carry one.
  if(/park/i.test(JSON.stringify(raw)))fail('mapped-ground suggestions must not use the banned vehicle-status wording');
  fields(raw.generator,['script','sha256','sharedScript','sharedSha256','python','numpy'],[],'generator');
  if(raw.generator.script!=='scripts/california/suggest_av2_placements.py'||raw.generator.sharedScript!=='scripts/california/prepare_av2_ground.py'||
    !hash(raw.generator.sha256)||!hash(raw.generator.sharedSha256)||!['python','numpy'].every(k=>text(raw.generator[k])))
    fail('generator must name the map-first scripts, their sha256 and versions');
  fields(raw.inputs,V3_INPUTS,[],'inputs');
  if(!Object.values(raw.inputs).every(hash)||raw.protocolSha256!==raw.inputs['protocol.json'])fail('inputs must be sha256 hashes and name the protocol used');
  const g=scene.ground;
  if(!same(raw.license,g.license))fail('license must equal the mapped ground license');
  fields(raw.tolerances,[...V3_TOLERANCES,'alongRayHeightBandMeters'],[],'tolerances');
  if(V3_TOLERANCES.some(k=>raw.tolerances[k]!==g[k])||raw.tolerances.alongRayHeightBandMeters!==g.alongRayUncertainty.heightBandMeters)
    fail('tolerances must equal the mapped ground tolerances');
  mappedElement(raw.crossing,scene,'crossing',v=>{
    const e=v.crossingError;
    if(!['edge1','edge2'].includes(v.nearEdge)||!finiteNumber(v.forwardM))fail('crossing must name its near edge and forward distance');
    if(!object(e)||Object.keys(e).sort().join()!=='basis,evidence,label,plusMinusM,status'||e.basis!=='mapped near edge')fail('crossing.crossingError must hold basis, status, label, plusMinusM and evidence');
    const evidence=x=>object(x)&&Object.keys(x).sort().join()==='file,sha256'&&text(x.file)&&hash(x.sha256);
    const ok=e.status==='measured'?finiteNumber(e.plusMinusM)&&e.plusMinusM>=0&&e.plusMinusM<=5&&e.label===`±${e.plusMinusM.toFixed(2)} m`&&evidence(e.evidence)
      :Object.hasOwn(CROSSING_ERROR_LABELS,e.status)&&e.label===CROSSING_ERROR_LABELS[e.status]&&e.plusMinusM===null&&(e.status==='unresolved'?evidence(e.evidence):e.evidence===null);
    if(!ok)fail('crossing.crossingError must be a measured offset with its evidence, or a named status');
  });
  mappedElement(raw.vehicle,scene,'vehicle',v=>{
    if(!name(v.label))fail('vehicle.label must be a plain name');
    fields(v.dimensions,DIMS,[],'vehicle.dimensions');numbers(v.dimensions,DIMS,'vehicle.dimensions');
    if(Object.hasOwn(v,'pixelGroundGapAlongAxisM')&&!finiteNumber(v.pixelGroundGapAlongAxisM))fail('vehicle.pixelGroundGapAlongAxisM must be finite');
    if(v.status==='annotated') {
      const r=v.frontResidualToAnnotatedM;
      if(!g.cuboids.available||!text(v.track)||!text(v.category)||!(v.motion===null||object(v.motion))||!finiteList(v.headingWorld,3)||v.headingWorld[1]!==0||
        !['headingToRoadAxisDeg','yawToRoadAxisDeg','frontGapToCrossingM'].every(k=>finiteNumber(v[k]))||!text(v.appModel)||
        !object(r)||Object.keys(r).sort().join()!=='alongRoadAxis,horizontal,lateralRight'||!Object.values(r).every(finiteNumber))
        fail('an annotated vehicle must carry its annotation, heading and front residual');
      // The app draws and tests this vehicle with the cuboid's exact (averaged) pose, so the file must name exactly one
      // cuboid whose category, motion, heading, (0.01 m) size and averaged-annotation count it records.
      const boxes=g.cuboids.items.filter(b=>b.track===v.track),box=boxes[0],M=box?.worldFromBox,n=box&&Math.hypot(M[0][0],M[2][0]);
      if(boxes.length!==1||box.category!==v.category||!same(box.motion,v.motion)||DIMS.some(k=>Number(box[k].toFixed(2))!==v.dimensions[k])||
        Math.abs(v.headingWorld[0]-M[0][0]/n)>1e-6||Math.abs(v.headingWorld[2]-M[2][0]/n)>1e-6||!box.poseAverage||
        !object(v.pose)||Object.keys(v.pose).sort().join()!=='annotations,kind'||v.pose.kind!=='averaged'||v.pose.annotations!==box.poseAverage.rows)
        fail('an annotated vehicle must match exactly one annotated (averaged) cuboid');
      // mappedPoint is the averaged box's front point at the mapped height of its cell: the front a vehicle in its place keeps.
      const front=[box.center[0]+M[0][0]/n*box.length/2,box.center[2]+M[2][0]/n*box.length/2],y=mappedGroundHeight(scene,front[0],front[1]);
      if(Math.abs(v.mappedPoint[0]-front[0])>1e-6||Math.abs(v.mappedPoint[2]-front[1])>1e-6||y===null||Math.abs(v.mappedPoint[1]-y)>1e-9)
        fail('an annotated vehicle mappedPoint must be the averaged box front point on the mapped ground');
      // The app tests this cuboid at its exact sizes and fills the size fields with them, so a vehicle outside the
      // supported sizes (geometry.mjs SIZE_LIMITS, e.g. a semi tractor-trailer) is refused here, by name, instead of
      // opening the scene into a size refusal.
      if(DIMS.some(k=>box[k]<SIZE_LIMITS[k][0]||box[k]>SIZE_LIMITS[k][1]))
        fail(`the annotated vehicle (${DIMS.map(k=>`${k} ${fmt(box[k])} m`).join(', ')}) is outside the supported sizes (${DIMS.map(k=>`${k} ${SIZE_LIMITS[k][0]}-${SIZE_LIMITS[k][1]} m`).join(', ')})`);
    }else if(!positive(v.setbackM)||!Array.isArray(v.annotatedVehicleRejected)||!v.annotatedVehicleRejected.every(text)||
      !finiteList(v.curbLateralAlongLengthM,2)||!finiteNumber(v.lateralFromCameraM))
      fail('a declared default vehicle must carry its setback and curb evidence');
  });
  mappedElement(raw.person,scene,'person',v=>{
    if(!name(v.label))fail('person.label must be a plain name');
    numbers(v,['heightM'],'person');
    if(Object.hasOwn(v,'topHiddenDiagnostic')&&typeof v.topHiddenDiagnostic!=='boolean')fail('person.topHiddenDiagnostic must be a flag');
  });
  element(raw.roadAxis,scene,'roadAxis',['mapped'],(v,w)=>{
    fields(v,['status','label','laneId','directionOfTravel','skewDeg','nearEdge','cameraInLane','reason'],[],w);
    const travel=mappedTravel(scene);
    if(v.laneId!==g.egoLane.id||!finiteList(v.directionOfTravel,3)||v.directionOfTravel[1]!==0||v.directionOfTravel.some((n,i)=>Math.abs(n-travel[i])>1e-9)||
      !finiteNumber(v.skewDeg)||typeof v.cameraInLane!=='boolean'||!['label','nearEdge','reason'].every(k=>text(v[k])))
      fail('roadAxis must be the mapped ego lane chord');
  });
  if(object(raw.curb)&&raw.curb.status==='unknown'){fields(raw.curb,['status','label','reason'],[],'curb');if(!text(raw.curb.reason))fail('curb.reason must be text');}
  else element(raw.curb,scene,'curb',['mapped'],(v,w)=>{
    fields(v,['status','label','side','reason','nearEdgeXZ','nearEdgeGroundY','nearEdgeLateralM'],[],w);
    if(!['label','side','reason'].every(k=>text(v[k]))||!finiteList(v.nearEdgeXZ,2)||!(v.nearEdgeGroundY===null||finiteNumber(v.nearEdgeGroundY))||!finiteNumber(v.nearEdgeLateralM))
      fail('curb must carry its mapped edge proxy');
  });
  if(!Array.isArray(raw.limitations)||!raw.limitations.every(text))fail('limitations must be a list of text');
  return raw;
}

export function validateSuggestions(raw,scene) {
  if(isMappedGround(scene))return validateMapped(raw,scene);
  if(scene?.measurements)fail('placement is unavailable for this scene');
  fields(raw,['schemaVersion','sceneId','generatedUtc','generator','inputs','crossing','roadAxis','curb','vehicle','person','detection','limitations'],['crossCheck'],'file');
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
  detectorFields(detection,ran,'detection');
  // An optional second detector (e.g. RF-DETR checking NV-Grounding-DINO). The preparation script writes it only when
  // the two agree; it never sets the suggestion, so it needs the primary run and a different detections file.
  if(Object.hasOwn(raw,'crossCheck')) {
    if(!object(raw.crossCheck)||raw.crossCheck.status!=='ran'||!ran)fail('crossCheck.status must be ran, after a detector run');
    detectorFields(raw.crossCheck,true,'crossCheck');
    if(raw.crossCheck.detectionsSha256===detection.detectionsSha256)fail('crossCheck must come from a different detections file');
    if(raw.crossCheck.weightsSha256===detection.weightsSha256)fail('crossCheck must come from different detector weights');
    // Agreement: both detectors count the same curb-lane candidates (scripts/suggest-placements.py uses the same rule
    // and refuses to write a file where the counts differ).
    if(raw.crossCheck.curbLaneParked!==detection.curbLaneParked)fail('crossCheck must count the same curb-lane vehicles as detection');
  }
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

// The app's starting values from a validated file: pixels (only those whose evaluated point is placed here), field strings,
// and provenance per value. input is the comparison input when everything is placed and createScenario accepts it.
// Schema 3: each pixel is the suggestion's integer anchor; geometry.mjs evaluates its exact sub-pixel position (pixelUV)
// while it stays there, and the support shown is that of the point the comparison uses (anchorPoint: for the annotated
// stationary vehicle, its averaged front point).
export function suggestedSetup(scene,suggestions) {
  const pixels={},provenance={},s=suggestions,mapped=s.schemaVersion===3;
  for(const [key,role] of ROLES) {
    const e=s[role];
    if(e.status==='unknown'){provenance[key]={status:'unplaced',reason:e.reason};continue;}
    let evidence;
    try{evidence=mapped?anchorPoint(scene,key,e.pixel):roadPoint(scene,e.pixel,role);}catch(error){provenance[key]={status:'unplaced',reason:error.message};continue;}
    pixels[key]=[...e.pixel];provenance[key]={status:SOURCE[e.status],value:[...e.pixel]};
    // Mapped points keep their support class ("mapped, not observed") and hidden flag (behind an annotated object).
    if(mapped)Object.assign(provenance[key],{supportClass:evidence.supportClass,supportLabel:evidence.supportLabel,hidden:evidence.hidden});
    if(role==='vehicle'&&e.status==='declared-default')provenance[key].setbackFt=Math.round(e.setbackM/FOOT);
  }
  // Schema v1 detects at most the vehicle front; its size is always the declared default. Schema v3 takes an annotated
  // stationary vehicle's size from its annotation.
  if(s.vehicle.status!=='unknown')provenance.vehicle={status:s.vehicle.status==='annotated'?'annotated':'default',value:{...s.vehicle.dimensions}};
  if(s.person.status!=='unknown')provenance.targetHeight={status:SOURCE[s.person.status],value:s.person.heightM};
  const vehicle=Object.fromEntries(DIMS.map(k=>[k,provenance.vehicle?String(s.vehicle.dimensions[k]):''])),
    targetHeight=provenance.targetHeight?String(s.person.heightM):'';
  let input=null,error='';
  if(ROLES.every(([key])=>pixels[key])&&provenance.vehicle&&provenance.targetHeight) {
    try {
      // The what-if starts on the current vehicle (an annotated vehicle: its annotated pose and setback).
      const next={pixels,vehicle,targetHeight,...comparisonStart(scene,pixels,vehicle)};
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
// Context strings carry no numbers: umbra's computed facts already hold every value.
export function suggestionStatus(suggestions,setup,current) {
  const p=setup.provenance,s=suggestions,
    value=key=>key==='vehicle'?current.vehicle:key==='targetHeight'?current.targetHeight:current.pixels?.[key],
    // A value Nemotron declared replaces the suggestion's label even when it happens to equal the suggested value.
    still=key=>!current.overridden?.includes(key)&&isStillSuggested(key,value(key),p);
  const edited=Object.keys(p).some(key=>p[key].status!=='unplaced'&&!still(key));
  const parts=[],context={};
  const unknown=(key,word)=>{if(p[key].status==='unplaced'&&!current.pixels?.[key])parts.push({key,text:`${word} unknown`,title:p[key].reason});};
  // Schema 3: the annotated vehicle keeps its annotated pose exactly while its front pixel stays within the nudge of the
  // suggestion, its size holds the suggestion and no declaration replaced the vehicle (geometry.mjs annotatedVehicle and
  // createScenario use the same test with vehicleDeclared).
  const pose=p.parking?.status==='annotated'&&!current.overridden?.includes('parking')&&namesAnnotatedFront(p.parking.value,value('parking'))&&still('vehicle');
  if(s.schemaVersion===3)return mappedStatus(s,p,still,unknown,parts,context,edited,pose);
  if(still('crossing'))parts.push({key:'crossing',text:`crosswalk detected ±${fmt(s.crossing.alongRoadUncertaintyM)} m`,title:`${s.crossing.method} · ±${fmt(s.crossing.alongRoadUncertaintyM)} m along the road`});
  unknown('crossing','crossing');
  if(still('parking')&&still('vehicle')) {
    // The hover text names the cross-check detector, when one agreed with the detection.
    const title=s.vehicle.reason+(s.crossCheck?` · cross-checked with ${s.crossCheck.model}`:'');
    if(p.parking.status==='detected') {
      parts.push({key:'parking',text:'parked vehicle front detected',title});
      parts.push({key:'vehicle',text:`${s.vehicle.label} size (default)`,title});
    }else{
      const at=still('crossing')&&positive(s.vehicle.setbackM)?` at ${fmt(s.vehicle.setbackM/FOOT)} ft`:'',
        none=s.detection.status==='ran'&&s.detection.curbLaneParked===0?', no parked vehicle':'';
      parts.push({key:'parking',text:`${s.vehicle.label}${at} (default${none})`,title});
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

// The crosswalk chip text (protocol v3.1 crossingError.display): the mapped near edge with a bound on its paint offset
// (|measured offset| + 2 × spread, e.g. "crosswalk mapped ±0.18 m") or the offset's status ("crosswalk mapped · paint offset pending measurement"). The
// crossing's along-ray uncertainty describes its pixel ray, not the crossing position, so it is never shown here.
export function crossingText(crossing) {
  const e=crossing.crossingError;
  return e.status==='measured'?`${crossing.label} ${e.label}`:`${crossing.label} · ${e.label}`;
}
// Schema 3 (mapped ground): each still-suggested point shows its support label ("mapped, not observed") and, when its
// pixel shows an annotated object instead of the ground, "hidden", as a separate note so narrow screens can wrap it. The annotated vehicle is labelled stationary; nothing
// says why it stopped. Nemotron context strings carry no numbers and none of the answer filter's withheld wording.
function mappedStatus(s,p,still,unknown,parts,context,edited,pose) {
  const support=key=>p[key].supportLabel+(p[key].hidden?'; hidden':''),
    title=(key,reason)=>`${reason} · ${p[key].supportLabel}`+(p[key].hidden?' · pixel shows an annotated object, point behind it':'');
  const where=key=>SUPPORT_CONTEXT[p[key].supportClass];
  if(still('crossing')) {
    const e=s.crossing.crossingError;
    parts.push({key:'crossing',text:crossingText(s.crossing),note:`(${support('crossing')})`,
      title:`${title('crossing',s.crossing.reason)} · ${e.status==='measured'?`mapped near edge; ${e.label} bounds the paint offset (measured offset + 2 × spread; development-grade, not calibrated)`:`mapped near edge; ${e.label}`}`});
  }
  unknown('crossing','crossing');
  if(still('parking')&&still('vehicle')) {
    const at=p.parking.status==='default'&&still('crossing')&&positive(s.vehicle.setbackM)?` at ${fmt(s.vehicle.setbackM/FOOT)} ft`:'',
      kind=p.parking.status==='default'?'default; ':'';
    parts.push({key:'parking',text:`${s.vehicle.label}${at}`,note:`(${kind}${support('parking')})`,title:title('parking',s.vehicle.reason)+(pose?' · annotated pose':'')});
  }
  unknown('parking','vehicle');
  if(still('target')&&still('targetHeight'))
    parts.push({key:'target',text:`${s.person.label} ${fmt(s.person.heightM)} m`,note:`(default; ${support('target')})`,title:title('target',s.person.reason)});
  unknown('target','person');
  if(still('crossing'))context.crossing=`mapped crosswalk edge, ${where('crossing')}`;
  // A resized annotated vehicle is a declared box at the annotated front pixel, not the annotation. A nudged front still
  // names the annotated vehicle (its pixel is no longer the suggested one, so its status part is gone).
  if(still('parking')||pose)context.vehicle_front=p.parking.status!=='annotated'?`declared default, ${where('parking')}`
    :pose?'stationary vehicle from the dataset annotation; reason for stopping unknown':`declared vehicle, ${where('parking')}`;
  if(still('vehicle'))context.vehicle_size=p.vehicle.status==='annotated'?'annotated stationary vehicle size':`declared default ${s.vehicle.label}`;
  if(still('target'))context.person_point=`declared default, ${where('target')}`;
  if(still('targetHeight'))context.person_height=`declared default ${s.person.label}`;
  return {parts,edited,context:Object.keys(context).length?context:null};
}

// Mapped-ground comparison: each placed point's support label, grouped ("crossing, vehicle front: mapped, not observed"),
// with the hidden flag in the hover text. support: {crossing,parking,target} -> {supportLabel,hidden} or null.
const POINT_NAMES=[['crossing','crossing'],['parking','vehicle front'],['target','person']];
export function supportSummary(support) {
  const groups=new Map(),title=[];
  for(const [key,name] of POINT_NAMES) {
    const p=support?.[key];
    if(!p?.supportLabel)continue;
    groups.set(p.supportLabel,[...(groups.get(p.supportLabel)??[]),name]);
    title.push(`${name}: ${p.supportLabel}`+(p.hidden?', pixel shows an annotated object, point behind it':''));
  }
  return {text:[...groups].map(([label,names])=>`${names.join(', ')}: ${label}`).join(' · '),title:title.join(' · ')};
}

import { createScenario, classifySightline, placementFromPixels, segmentIntersectsBox, isMappedGround, mappedGroundHeight, whatIfBox, annotatedVehicle, comparisonStart, SUPPORT_CONTEXT } from './geometry.mjs';

// Nemotron only declares what-if values. blocked/unknown come only from geometry.mjs; overlap marks a declared box
// over the person or eye. There is no Clear.
export const FOOT=.3048;
export function toMeters(value,unit) {
  if(unit==='m')return value;
  if(unit==='ft')return value*FOOT;
  throw new Error('Unit must be m or ft.');
}
export const RANGES={width:[.5,3.5],height:[.5,4.5],length:[1,20],targetHeight:[.3,2.5],label:40};
const STEP=.25,MARKS={at15ft:15*FOOT,at20ft:20*FOOT},EPS=1e-9,DIMS=['width','height','length'];
const r2=n=>Number(n.toFixed(2)),fmt=n=>String(r2(n));
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const absent=v=>v===undefined||v===null;
export const dims=v=>`${fmt(Number(v.width))} × ${fmt(Number(v.height))} × ${fmt(Number(v.length))} m`;
// Sizes compared at the 2 decimals Nemotron is given in the facts.
export const sameVehicle=(a,b)=>DIMS.every(k=>fmt(Number(a?.[k]))===fmt(Number(b?.[k])));

// Wording never shown from Nemotron (answers or labels). Strict on purpose: a withheld answer falls back to the app's own
// computed sentence.
const FORBIDDEN=[/\bclear(?:ly|ed|er|est|ance|s)?\b/i,/\b(?:in|non-?)?visib\w*/i,/\b(?:un)?safe(?:ly|r|st|ty)?\b/i,
  /\b(?:danger|hazard|risk|crash|collision|accident)\w*/i,
  // Law in any form: Sightline never establishes that parking is legal, permitted or meets a rule.
  /\b(?:il)?legal\w*/i,/\b(?:un)?lawful\w*/i,/\b(?:non-?)?complian\w*/i,/\bcompl(?:y|ies|ied|ying)\b/i,/\bviolat\w*/i,
  /\bpermi(?:t|ss)\w*/i,/\ballow(?:s|ed|able|ing|ance)?\b/i,/\b(?:prohibit|forbid|forbad)\w*/i,/\bconform\w*/i,/\baccordance\b/i,
  /\b(?:meets?|met|meeting|satisf\w*)\b.{0,40}?(?:\b(?:rules?|codes?|laws?|regulations?|requirements?)\b|22500)/i,
  /(?:\b(?:rules?|codes?|laws?|regulations?|requirements?)|22500\S*)\s+(?:is\s+|are\s+)?(?:met|satisfied)\b/i,
  // Perception in any form: geometry never establishes that anyone sees, spots, notices or loses sight of anything.
  /\b(?:see|sees|seen|seeing|saw)\b/i,/\b(?:notic|observ|detect|recogni[sz]|percei|percept)\w*/i,/(?<!parking\s+)\bspot(?:s|ted|ting)?\b/i,
  /\b(?:mak(?:e|es|ing)|made)\s+out\b/i,/\bview(?:able|ed|ing)\b/i,/\bviews?\s+(?:of|to)\b/i,/\b(?:open|full|good|better|direct|plain|wide)\s+views?\b/i,
  /\b(?:in|into|out\s+of)\s+(?:plain\s+|full\s+)?(?:view|sight)\b/i,/\blines?\s+of\s+sight\s+(?:to|on|with)\b/i,/\bsight\s*lines?\s+(?:to|on)\b/i,
  /\bexpos(?:e|es|ed|ing|ure)\b/i,/\b(?:hid(?:e|es|den|ing)|unhid\w*|obscur\w*|conceal\w*|reveal\w*|uncover\w*|unveil\w*|unmask\w*|emerg(?:e|es|ed|ing))\b/i,
  /\b(?:not|no\s+longer|never)\s+(?:hidden|obscured|concealed|covered|occluded|masked)\b|\bunoccluded\b/i,
  // Clear in other words. "Not blocked" is always "not blocked by the vehicle": Unknown is never presented as Clear.
  /\b(?:is|are|stays?|remains?|becomes?)\s+open\b/i,/\bun(?:obstructed|blocked|impeded|hindered)\b/i,
  /\bnothing\s+(?:blocks|obstructs|is\s+in\s+the\s+way)\b/i,/\bno\s+(?:obstructions?|blockage)\b/i,
  /\b(?:obstruction|obstacle|blockage)s?[-\s]*free\b/i,/\bfield\s+of\s+view\b/i,/\b(?:within|out\s+of)\s+(?:the\s+)?(?:driver'?s?\s+)?(?:view|sight)\b/i,
  /\blines?\s+of\s+sight\b(?![^.;]*\bblocked\b)/i,/\bnothing\s+in\s+the\s+way\b/i,/\bin\s+the\s+open\b/i,/\b(?:appears?|appeared|appearing)\b/i,
  /\b(?:restored|unaffected)\b/i,/\b(?:is|are)\s+free\b/i,/\b(?:fine|ok|okay|good|safe)\s+to\s+park\b/i,/\bcan\s+(?:still\s+)?park\b/i,/\bfree\s+(?:of|from)\s+(?:any\s+)?(?:obstruct|obstacl|block)\w*/i,
  /(?:\bnot|n't|\bnever|\bno\s+longer)\s+(?:be\s+|being\s+|been\s+)?blocked\b(?!\s+by\s+the\s+vehicle\b)/i];
// Mapped-ground (AV2) scenes: the current vehicle is a stationary annotated vehicle whose reason for stopping is unknown,
// so no model-written text (answer or label) containing any word built on "park" is shown there. Measured-depth scenes
// keep their parking wording.
const PARK=/\w*park\w*/i;
const withheld=(text,mapped=false)=>(mapped?[...FORBIDDEN,PARK]:FORBIDDEN).map(p=>text.match(p)).find(Boolean)?.[0];
// Model-written labels are shown word for word, so they are plain names: letters only, no result or state words and
// none of the withheld wording. A label never carries a number or a result.
const NAME=/^[A-Za-z][A-Za-z ,'&-]{0,39}$/,STATE_WORDS=/\b(?:un)?block\w*|\b(?:unknown|not|stops?|stopped|from|at|clear\w*|overlap\w*)\b|n't\b/i;

// Each check returns plain error text or undefined; values are rejected, never clamped or substituted.
// Mapped-scene errors never repeat model-written text, so a field name like "parked_vehicle" cannot reach the app's own line.
const unexpected=(args,keys,mapped=false)=>{const key=Object.keys(args).find(k=>!keys.includes(k));return key&&(mapped?'Unexpected field.':`Unexpected field ${key}.`);};
// The mapped-scene error never repeats the rejected word (tool errors can reach the app's own line).
const nameError=(v,name,mapped=false)=>typeof v==='string'&&mapped&&PARK.test(v)?`${name} must name only the type, like "SUV" or "child", with no status words.`
  :typeof v==='string'&&NAME.test(v)&&!STATE_WORDS.test(v)&&!withheld(v)?undefined
  :`${name} must be a plain name like "box truck" or "child": letters only, up to ${RANGES.label} characters, no numbers or results.`;
const ranged=(v,name,[lo,hi])=>typeof v==='number'&&Number.isFinite(v)&&v>=lo&&v<=hi?undefined:`${name} must be a number from ${lo} to ${hi} m.`;

// mapped: the scene is a mapped-ground (AV2) scene, where labels must not use any word built on "park".
export function validateDeclaration(args,{mapped=false}={}) {
  if(!object(args))return {ok:false,error:'Arguments must be an object.'};
  const a=args,setback=!absent(a.setback_value)||!absent(a.setback_unit);
  const error=unexpected(a,['vehicle_label','vehicle_width_m','vehicle_height_m','vehicle_length_m','target_label','target_height_m','setback_value','setback_unit'],mapped)
    ??nameError(a.vehicle_label,'vehicle_label',mapped)??ranged(a.vehicle_width_m,'vehicle_width_m',RANGES.width)
    ??ranged(a.vehicle_height_m,'vehicle_height_m',RANGES.height)??ranged(a.vehicle_length_m,'vehicle_length_m',RANGES.length)
    ??nameError(a.target_label,'target_label',mapped)??ranged(a.target_height_m,'target_height_m',RANGES.targetHeight)
    ??(setback&&!(typeof a.setback_value==='number'&&Number.isFinite(a.setback_value)&&a.setback_value>0)?'setback_value must be a positive number.':undefined)
    ??(setback&&!['m','ft'].includes(a.setback_unit)?'setback_unit must be m or ft.':undefined);
  if(error)return {ok:false,error};
  return {ok:true,value:{vehicleLabel:a.vehicle_label,vehicle:{width:a.vehicle_width_m,height:a.vehicle_height_m,length:a.vehicle_length_m},
    targetLabel:a.target_label,targetHeight:a.target_height_m,setback:setback?{value:a.setback_value,unit:a.setback_unit,meters:toMeters(a.setback_value,a.setback_unit)}:null}};
}
export function validateSweep(args,{mapped=false}={}) {
  if(!object(args))return {ok:false,error:'Arguments must be an object.'};
  let error=unexpected(args,['vehicles','target_height_m'],mapped)
    ??(Array.isArray(args.vehicles)&&args.vehicles.length>=1&&args.vehicles.length<=4?undefined:'vehicles must list 1 to 4 vehicles.')
    ??(absent(args.target_height_m)?undefined:ranged(args.target_height_m,'target_height_m',RANGES.targetHeight));
  for(let i=0;!error&&i<args.vehicles.length;i++) {
    const v=args.vehicles[i],name=`vehicles[${i}]`;
    error=object(v)?unexpected(v,['label','width_m','height_m','length_m'],mapped)??nameError(v.label,name+'.label',mapped)??ranged(v.width_m,name+'.width_m',RANGES.width)
      ??ranged(v.height_m,name+'.height_m',RANGES.height)??ranged(v.length_m,name+'.length_m',RANGES.length):`${name} must be an object.`;
  }
  if(error)return {ok:false,error};
  return {ok:true,value:{vehicles:args.vehicles.map(v=>({label:v.label,width:v.width_m,height:v.height_m,length:v.length_m})),
    targetHeight:absent(args.target_height_m)?undefined:args.target_height_m}};
}

// classifySightline's axis-local frame and box with geometry's own segment-box test (same local values, same result).
// The person's ground-to-head column touching the box (covers ground or head point inside) or the eye inside it is
// an overlap; such a setback is never reported as blocked. box: {center, axis, vehicle}, as geometry.mjs builds the
// current and what-if boxes (an annotated pose keeps its own heading and size).
function boxTests(scenario,{center,axis,vehicle}) {
  const {eye,target}=scenario,side=[axis[2],0,-axis[0]],half=[vehicle.width/2,vehicle.height/2,vehicle.length/2],low=half.map(n=>-n);
  const local=p=>{const d=[p[0]-center[0],p[1]-center[1],p[2]-center[2]];return [dot(d,side),d[1],dot(d,axis)];};
  return {targetInside:segmentIntersectsBox(local(scenario.targetGround),local(target),low,half),
    eyeInside:local(eye).every((n,i)=>Math.abs(n)<=half[i]),sightline:segmentIntersectsBox(local(eye),local(target),low,half)};
}
export function actorConflicts(scenario,which) {
  const box=['current','baseline','proposed'].includes(which)?scenario?.[which==='proposed'?which:'baseline']:null;
  if(!box?.center)throw new Error('Vehicle position must be current, baseline or proposed.');
  const {targetInside,eyeInside}=boxTests(scenario,box);
  return {targetInside,eyeInside};
}
// Unsigned angle between two horizontal unit axes, in degrees.
const turnDeg=(a,b)=>Math.atan2(Math.abs(a[0]*b[2]-a[2]*b[0]),a[0]*b[0]+a[2]*b[2])*180/Math.PI;

// Mapped-ground scenes: every placed point's support class (suggested or edited), keyed like the suggestion context.
// support: {crossing,parking,target} -> {supportClass} or null; points without a mapped-ground hit are left out.
export function supportFacts(support) {
  const out={};
  for(const [key,name] of [['crossing','crossing'],['parking','vehicle_front'],['target','person_point']])
    if(Object.hasOwn(SUPPORT_CONTEXT,support?.[key]?.supportClass??''))out[name]=SUPPORT_CONTEXT[support[key].supportClass];
  return out;
}
// Model-facing facts: scalars only, no pixels or world coordinates.
export function scenarioFacts(scene,scenario) {
  const s=scenario,v=s.vehicle,position=(which,setback,visibility)=>{
    const c=actorConflicts(s,which),state=c.targetInside||c.eyeInside?'overlap':visibility.state;
    // An annotated pose (current vehicle and its what-if) is turned from the lane axis; its setback is the lane-axis
    // distance from the crossing to its front-face centre (geometry.mjs).
    const box=s[which],pose=box.pose==='annotated'?{pose:'annotated',turnedFromLane_deg:r2(turnDeg(box.axis,s.axis))}:{};
    return {...pose,setback_m:r2(setback),setback_ft:r2(setback/FOOT),state,...c};
  };
  // Mapped scenes put the camera at the world origin, so eye height is measured to the mapped ground under the eye
  // (null when the map has no value there); measured-depth scenes have the road at Y 0.
  const ground=isMappedGround(scene)?mappedGroundHeight(scene,s.eye[0],s.eye[2]):0;
  return {scene:scene.id,units:scene.units,eyeHeight_m:ground===null?null:r2(s.eye[1]-ground),vehicle_m:{width:r2(v.width),height:r2(v.height),length:r2(v.length)},
    targetHeight_m:r2(s.target[1]-s.targetGround[1]),maximumSetback_m:r2(s.limits.maximumSetback),maximumSetback_ft:r2(s.limits.maximumSetback/FOOT),
    current:position('baseline',s.baselineSetback,s.currentVisibility),proposed:position('proposed',s.proposedSetback,s.proposedVisibility),
    ...(s.support&&{support:supportFacts(s.support)})};
}

export function applyDeclaration(scene,pixels,currentInput,args) {
  const checked=validateDeclaration(args,{mapped:isMappedGround(scene)});
  if(!checked.ok)return {error:checked.error};
  if(!['crossing','parking','target'].every(k=>pixels?.[k]))return {error:`Place the crossing, vehicle and person on the ${isMappedGround(scene)?'mapped ground':'measured road'} first.`};
  const d=checked.value;
  try {
    const placement=currentInput?null:placementFromPixels(scene,pixels);
    // A declared vehicle ends the annotated pose of a mapped-ground scene, even at the annotated sizes.
    const input={pixels,vehicle:{...d.vehicle},targetHeight:d.targetHeight,...(isMappedGround(scene)&&{vehicleDeclared:true}),
      proposedSetback:d.setback?d.setback.meters:currentInput?currentInput.proposedSetback:placement.baselineSetback,
      driverDistance:currentInput?currentInput.driverDistance:placement.captureDistance};
    const scenario=createScenario(scene,input),setback=d.setback&&{value:d.setback.value,unit:d.setback.unit,m:r2(d.setback.meters),ft:r2(d.setback.meters/FOOT)};
    const summary=`${d.vehicleLabel} ${fmt(d.vehicle.width)} × ${fmt(d.vehicle.height)} × ${fmt(d.vehicle.length)} m (W × H × L) · ${d.targetLabel} ${fmt(d.targetHeight)} m`
      +(setback?` · ${setback.m} m (${setback.ft} ft) from crossing`:'');
    return {input,scenario,facts:scenarioFacts(scene,scenario),
      declared:{vehicleLabel:d.vehicleLabel,vehicle:{...d.vehicle},targetLabel:d.targetLabel,targetHeight:d.targetHeight,setback,summary}};
  }catch(e){return {error:e.message};}
}

export function setbackSweep(scene,baseInput,vehicle,targetHeight=baseInput?.targetHeight) {
  const label=vehicle?.label,dims={width:vehicle?.width,height:vehicle?.height,length:vehicle?.length};
  let s;
  try {
    // One validation pass at the vehicle's own start (createScenario checks that it fits; one that does not fits nowhere).
    // A vehicle with the annotated sizes, on an input no declaration changed, is swept in the annotated pose.
    const start=comparisonStart(scene,baseInput?.pixels,dims,!!baseInput?.vehicleDeclared);
    s=createScenario(scene,{pixels:baseInput?.pixels,vehicle:dims,targetHeight,driverDistance:baseInput?.driverDistance??start.driverDistance,
      proposedSetback:start.proposedSetback,...(baseInput?.vehicleDeclared&&{vehicleDeclared:true})});
  }catch(e){return {label,vehicle:dims,error:e.message};}
  // Rows start where the slider does: the box's nearest point at the road tolerance.
  const min=s.limits.minimumSetback,max=s.limits.maximumSetback,setbacks=[];
  for(let k=0;min+k*STEP<=max+EPS;k++)setbacks.push(min+k*STEP);
  if(max-setbacks.at(-1)>1e-6)setbacks.push(max);
  for(const mark of Object.values(MARKS))if(mark>=min&&mark<=max&&!setbacks.some(n=>Math.abs(n-mark)<EPS))setbacks.push(mark);
  setbacks.sort((a,b)=>a-b);
  const {vehicle:v,eye,target}=s,rows=[];
  for(const setback of setbacks) {
    // The same what-if box as createScenario's proposed vehicle at this setback (geometry.mjs whatIfBox).
    const box=whatIfBox(s.whatIf,setback),c=boxTests(s,box),state=c.targetInside||c.eyeInside?'overlap':c.sightline?'blocked':'unknown';
    // classifySightline stays the authority: it confirms the first row of every run. Its later mesh and pixel
    // diagnostics ignore the vehicle and can only yield unknown, so repeating them per row (~1 ms each) is skipped.
    if(state!=='overlap'&&state!==rows.at(-1)?.state&&classifySightline(scene,eye,target,box.vehicle,box.axis,box.center).state!==state)
      throw new Error('Sweep disagrees with the measured comparison.');
    rows.push(state==='overlap'?{setback,state,person:c.targetInside,driver:c.eyeInside}:{setback,state});
  }
  let tail=rows.length;while(tail>0&&rows[tail-1].state==='unknown')tail--;
  const lastBlocked=rows.findLastIndex(r=>r.state==='blocked'),at=mark=>rows.find(r=>Math.abs(r.setback-mark)<EPS)?.state??null;
  return {label,vehicle:{...v},targetHeight:Number(targetHeight),rows,step:STEP,range:[min,rows.at(-1).setback],
    // Every larger tested setback is unknown from here; notBlockedFrom only requires no later blocked row (overlaps allowed).
    stopsBlockingAt:tail<rows.length?rows[tail].setback:null,
    notBlockedFrom:rows.find((r,i)=>i>lastBlocked&&r.state==='unknown')?.setback??null,
    at15ft:at(MARKS.at15ft),at20ft:at(MARKS.at20ft)};
}

// Consecutive rows in one state, as row groups.
const runs=(rows,state)=>rows.reduce((out,r,i)=>{
  if(r.state===state){if(rows[i-1]?.state===state)out.at(-1).push(r);else out.push([r]);}
  return out;
},[]);
const span=g=>{const a=fmt(g[0].setback),b=fmt(g.at(-1).setback);return a===b?`at ${a} m`:`${a}-${b} m`;};
// The one result phrase per swept vehicle (model facts, Distance check rows and the app's own line), read from the rows:
// blocked runs, the all-unknown tail, then overlaps. Tested distances it does not name are unknown. Every number in it
// is in sweepFacts.
export function sweepPhrase(r) {
  if(r.error)return r.error;
  if(r.rows.every(x=>x.state==='blocked'))return 'blocked at every tested distance';
  const blocked=runs(r.rows,'blocked'),parts=[];
  if(blocked.length)parts.push('blocked '+blocked.map(span).join(', '));
  if(blocked.length&&r.stopsBlockingAt!==null)parts.push(`not blocked by the vehicle from ${fmt(r.stopsBlockingAt)} m (${fmt(r.stopsBlockingAt/FOOT)} ft)`);
  else if(!blocked.length&&r.rows.some(x=>x.state==='unknown'))parts.push('not blocked by the vehicle at any tested distance (rest unknown)');
  for(const g of runs(r.rows,'overlap')){const p=g.some(x=>x.person),d=g.some(x=>x.driver);parts.push(`overlaps ${p&&d?'person and driver':p?'person':'driver'} ${span(g)}`);}
  return parts.join('; ');
}
const MARK_WORDS={blocked:'Blocked',unknown:'Unknown',overlap:'Overlap'};
// The app's own 20 ft / 15 ft line for one swept vehicle; Nemotron's answers never carry the marks.
// One short phrase per vehicle for Nemotron's two-sentence summary; the full phrase stays in the panel.
export function sweepShort(r) {
  if(r.error)return r.error;
  if(r.stopsBlockingAt!==null&&r.rows.some(x=>x.state==='blocked'))return `not blocked by the vehicle from ${fmt(r.stopsBlockingAt)} m (${fmt(r.stopsBlockingAt/FOOT)} ft)`;
  if(!r.rows.some(x=>x.state==='blocked'))return 'not blocked by the vehicle at any tested distance';
  return 'no tested distance where it stops blocking';
}
export const sweepMarks=r=>`20 ft: ${MARK_WORDS[r.at20ft]??'out of range'} · 15 ft: ${MARK_WORDS[r.at15ft]??'out of range'}`;

export function sweepFacts(results) {
  const pairs=(rows,state)=>runs(rows,state).map(g=>[r2(g[0].setback),r2(g.at(-1).setback)]),ft=m=>m===null?null:r2(m/FOOT);
  return {vehicles:results.map(r=>r.error?{label:r.label,error:r.error}:{label:r.label,
    vehicle_m:{width:r2(r.vehicle.width),height:r2(r.vehicle.height),length:r2(r.vehicle.length)},targetHeight_m:r2(r.targetHeight),
    tested_m:r.range.map(r2),tested_ft:r.range.map(ft),step_m:r.step,
    stopsBlockingAt_m:r.stopsBlockingAt===null?null:r2(r.stopsBlockingAt),stopsBlockingAt_ft:ft(r.stopsBlockingAt),
    at15ft:r.at15ft,at20ft:r.at20ft,blocked_m:pairs(r.rows,'blocked'),overlap_m:pairs(r.rows,'overlap'),result:sweepPhrase(r),short:sweepShort(r)})};
}

const NUMBER=/\d*\.?\d+/g;
// Only computed numeric values ground an answer: no reference numbers, and labels or ids never carry numbers.
export function allowedNumbersFrom(...facts) {
  const out=new Set(),walk=v=>{
    if(typeof v==='number'){if(Number.isFinite(v))out.add(v);}
    else if(v&&typeof v==='object')Object.values(v).forEach(walk);
  };
  facts.forEach(walk);
  return [...out];
}
export const MAX_ANSWER=280;
// mapped: a mapped-ground (AV2) scene, where an answer using any word built on "park" is withheld too.
export function checkAnswer(text,allowedNumbers=[],{mapped=false}={}) {
  if(typeof text!=='string'||!text.trim())return {ok:false,kind:'empty',reason:'Empty answer.'};
  if(text.length>MAX_ANSWER)return {ok:false,kind:'length',reason:'Answer is longer than two short sentences.'};
  const word=withheld(text,mapped);
  if(word)return {ok:false,kind:'wording',reason:`Uses withheld wording "${word}".`};
  const key=n=>Math.round(n*10),allowed=new Set(allowedNumbers.filter(Number.isFinite).map(key));
  const stray=(text.match(NUMBER)??[]).find(n=>!allowed.has(key(Number(n))));
  if(stray)return {ok:false,kind:'number',reason:`Number ${stray} is not in Sightline's computed facts.`};
  return {ok:true,reason:'',text};
}
const WITHHELD={length:'too long.',number:'used a number Sightline did not compute.',wording:'used wording Sightline does not allow.'};
// The quiet note shown beside the app's own line, true to why the answer was not shown.
export const withheldNote=checked=>WITHHELD[checked.kind]?`Nemotron's answer was withheld: ${WITHHELD[checked.kind]}`:'Nemotron returned no wording.';

// "Declared by Nemotron" parts. The line shows, per field, the latest declared value the field still holds: a field the
// user edits loses its credit, untouched fields keep theirs and a value restored exactly regains it.
export const vehiclePart=(label,v)=>({key:'vehicle',label,text:`${label} ${dims(v)} (W × H × L)`,value:{width:v.width,height:v.height,length:v.length}});
export function declaredLine(parts,basis) {
  if(!basis)return '';
  const last=key=>parts.findLast(p=>p.key===key&&(key==='vehicle'?sameVehicle(basis.vehicle,p.value):Number(key==='person'?basis.targetHeight:basis.proposedSetback)===p.value));
  // Without a whole declared vehicle, the latest one keeps credit for the sizes the user left untouched.
  const latest=parts.findLast(p=>p.key==='vehicle'),kept=latest?DIMS.filter(k=>fmt(Number(basis.vehicle?.[k]))===fmt(latest.value[k])):[];
  const vehicle=last('vehicle')?.text??(kept.length?`${latest.label} ${kept.map(k=>`${k} ${fmt(latest.value[k])} m`).join(', ')}`:'');
  return [vehicle,last('person')?.text,last('setback')?.text].filter(Boolean).join(' · ');
}

// One Nemotron tool call against the latest app values w={pixels,vehicle,targetHeight,input,setupInput}. Pure: returns the
// model's result and the state the app applies, or {error}. declare_scenario is ask-only. sweep_setbacks runs on the
// vehicle and person already declared or entered and never changes the person height.
export function runTool(scene,mode,w,name,args) {
  if(name==='declare_scenario'&&mode==='ask') {
    const res=applyDeclaration(scene,w.pixels,w.input,args);
    if(res.error)return {error:res.error};
    const d=res.declared,v=res.input.vehicle,before=w.input?.vehicle??w.vehicle,prev=w.input??w.setupInput;
    // Small models tend to copy the on-screen size for a newly named vehicle type; ask once per run for that type's size.
    const prior=typeof w.vehicleLabel==='string'?w.vehicleLabel.trim().toLowerCase():'';
    if(w.run&&!w.run.sizeChecked&&prior&&d.vehicleLabel.trim().toLowerCase()!==prior&&DIMS.every(k=>Number(before?.[k])===v[k])) {
      w.run.sizeChecked=true;
      return {error:`Those are the ${w.vehicleLabel}'s dimensions. Declare typical dimensions for a ${d.vehicleLabel}, or repeat the call if the user gave these sizes.`};
    }
    return {result:res.facts,facts:res.facts,input:res.input,
      fields:{vehicle:{width:String(v.width),height:String(v.height),length:String(v.length)},targetHeight:String(res.input.targetHeight)},
      parts:[vehiclePart(d.vehicleLabel,d.vehicle),{key:'person',text:`${d.targetLabel} ${fmt(d.targetHeight)} m`,value:d.targetHeight},
        ...(d.setback?[{key:'setback',text:`${d.setback.m} m (${d.setback.ft} ft) from crossing`,value:res.input.proposedSetback}]:[])],
      // Distance results describe the vehicle they were run with; a declaration that changes its size, or that replaces
      // an annotated pose (even at the same sizes), clears them.
      clearSweep:!DIMS.every(k=>Number(before?.[k])===v[k])||!!(prev&&!prev.vehicleDeclared&&annotatedVehicle(scene,prev.pixels,prev.vehicle))};
  }
  if(name==='sweep_setbacks') {
    const checked=validateSweep(args,{mapped:isMappedGround(scene)});
    if(!checked.ok)return {error:checked.error};
    const from=w.input??w.setupInput;
    if(!from)return {error:'Declare or enter a vehicle first.'};
    const height=Number(from.targetHeight),driverDistance=Number(from.driverDistance),asked=checked.value.targetHeight;
    if(asked!==undefined&&!(Math.abs(asked-height)<.005))return {error:'Change the person height with declare_scenario first.'};
    // The distance check always tests the vehicle on screen, even if Nemotron left it out.
    const own={label:'your vehicle',...Object.fromEntries(DIMS.map(k=>[k,Number(from.vehicle[k])]))};
    const listed=checked.value.vehicles,mine=v=>DIMS.every(k=>Math.abs(v[k]-own[k])<.005);
    const vehicles=mode==='check'&&!listed.some(mine)?[own,...listed].slice(0,4):listed;
    const results=vehicles.map(v=>setbackSweep(scene,{pixels:from.pixels,driverDistance,vehicleDeclared:from.vehicleDeclared},v,height));
    // vehicleDeclared: whether the checked vehicle was a declared box (false: an annotated pose may have been swept).
    return {result:sweepFacts(results),input:from,sweep:{pixels:from.pixels,targetHeight:height,driverDistance,results,own:{...from.vehicle},vehicleDeclared:!!from.vehicleDeclared}};
  }
  return {error:`Unknown tool ${name}.`};
}

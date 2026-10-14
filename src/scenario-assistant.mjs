import { createScenario, assessSightline, placementFromPixels, segmentIntersectsBox, isMappedGround, whatIfBox, annotatedVehicle, comparisonStart, SUPPORT_CONTEXT, SIZE_LIMITS, UNKNOWN_REASONS } from './geometry.mjs';

// Nemotron only declares what-if values. blocked/unknown come only from geometry.mjs (assessSightline). In sweep rows and
// Nemotron facts, overlap is the Unknown reason "person or driver inside the vehicle box" (spec vocabulary); every other
// Unknown reason travels as a reason next to state unknown. There is no Clear.
export const FOOT=.3048;
export function toMeters(value,unit) {
  if(unit==='m')return value;
  if(unit==='ft')return value*FOOT;
  throw new Error('Unit must be m or ft.');
}
// The declared size ranges are geometry's SIZE_LIMITS, so typed fields and Nemotron declarations share one set.
export const RANGES={...SIZE_LIMITS,label:40};
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
// Mapped-ground scenes also withhold any reason for the vehicle being stopped (nothing in the data says why: Sean,
// October 11) and any claim that a position was measured, surveyed or is precise (the crossing and points are mapped,
// not measured). Kept narrow so the app's own phrases ("stops blocking", "delivery van") still pass.
const STOP_REASON=new RegExp('\\b(?:'+['stop\\s*signs?','(?:traffic|red|green|amber|yellow)\\s+(?:lights?|signals?)','signals?','wait(?:s|ed|ing)?','yield(?:s|ed|ing)?',
  'stall(?:s|ed|ing)?','idl(?:e|es|ed|ing)','queu(?:e|es|ed|ing)','broke(?:n)?[-\\s]+down','break(?:s|ing)?[-\\s]+down','breakdown','disabled',
  'double[-\\s]?park\\w*','drop(?:s|ped|ping)?[-\\s]+(?:\\w+[-\\s]+){0,2}?off','drop-?offs?','pick(?:s|ed|ing)[-\\s]+(?:\\w+[-\\s]+){0,2}?up',
  '(?:un)?load(?:s|ed|ing)','deliver(?:s|ed|ing)','(?:for|in|because\\s+of)\\s+traffic','stopped\\s+(?:for|because|to)',
  // Audit (October 12): more invented reasons.
  'pull(?:s|ed|ing)?[-\\s]+(?:\\w+[-\\s]+){0,2}?over','out\\s+of\\s+(?:gas|fuel|petrol|battery|charge|power)','flat\\s+(?:tires?|tyres?)',
  'let(?:s|ting)?[-\\s]+(?:\\w+[-\\s]+){0,2}?(?:out|off)','(?:board|alight|disembark)(?:s|ed|ing)?','errands?','engine\\s+(?:trouble|failure|problems?)',
  'about\\s+to\\s+(?:turn|leave|move|go|pull|merge)','turning\\s+(?:left|right)','revers(?:e|es|ed|ing)','backing\\s+up','merg(?:e|es|ed|ing)'].join('|')+')\\b','i');
// Claims that a position is measured, surveyed, exact, verified or comes from LiDAR. The support class's own words
// ("with nearby LiDAR returns") still pass.
const INVENTED_SOURCE=/\b(?:measur|survey|precis|accura|exact|verif|certif)\w*|\bground[-\s]*truth\w*|\b(?:from|by|via|using|on|with)\s+(?:the\s+)?lidar\b|\bthe\s+lidar\b(?!\s+returns)|\blidar[-\s]+(?:show|confirm|place|put|locat|scan|data|point|map|survey|base|deriv|verif|measur)\w*/i;
// Naming the stationary vehicle's make or model is invented geometry (challenge item 4): nothing in the data says what car it is.
const MAKE=/\b(?:tesla|toyota|honda|ford|chevrolet|chevy|nissan|bmw|mercedes(?:-benz)?|audi|volkswagen|vw|hyundai|kia|subaru|mazda|lexus|jeep|dodge|gmc|cadillac|buick|chrysler|volvo|porsche|jaguar|land\s+rover|mitsubishi|infiniti|acura|lincoln|rivian|lucid|polestar|prius|camry|corolla|civic|accord|tacoma|sprinter|model\s+[3sxy])\b/i;
const withheld=(text,mapped=false)=>(mapped?[...FORBIDDEN,PARK,STOP_REASON,INVENTED_SOURCE,MAKE]:FORBIDDEN).map(p=>text.match(p)).find(Boolean)?.[0];
// Model-written labels are shown word for word, so they are plain names: letters only, no result or state words and
// none of the withheld wording. A label never carries a number or a result.
const NAME=/^[A-Za-z][A-Za-z ,'&-]{0,39}$/,STATE_WORDS=/\b(?:un)?block\w*|\b(?:unknown|not|stops?|stopped|from|at|clear\w*|overlap\w*)\b|n't\b/i;

// Each check returns plain error text or undefined; values are rejected, never clamped or substituted.
// Mapped-scene errors never repeat model-written text, so a field name like "parked_vehicle" cannot reach the app's own line.
const unexpected=(args,keys,mapped=false)=>{const key=Object.keys(args).find(k=>!keys.includes(k));return key&&(mapped?'Unexpected field.':`Unexpected field ${key}.`);};
// The mapped-scene error never repeats the rejected word (tool errors can reach the app's own line).
const nameError=(v,name,mapped=false)=>typeof v==='string'&&mapped&&(PARK.test(v)||STOP_REASON.test(v))?`${name} must name only the type, like "SUV" or "child", with no status words.`
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
// an overlap (assessSightline's person- and driver-inside-vehicle Unknown reasons); such a setback is never reported as
// blocked. box: {center, axis, vehicle}, as geometry.mjs builds the current and what-if boxes (an annotated pose keeps
// its own heading and size).
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
// A result's state in the facts and sweep rows: overlap for the person or driver inside the tested vehicle box, else
// geometry's state (blocked or unknown).
const INSIDE=new Set(['person-inside-vehicle','driver-inside-vehicle']);
const factState=visibility=>INSIDE.has(visibility.reasonCode)?'overlap':visibility.state;
// The one phrase for a computed result (facts, the app's own line and Nemotron's wording): "blocked"; "not blocked by the
// vehicle (rest unknown)" only for a plain Unknown (the line misses the box and nothing else made it Unknown); otherwise
// "unknown (<reason>)", e.g. "unknown (too close to call)" or "unknown (person inside a vehicle)".
export const PLAIN_UNKNOWN='not blocked by the vehicle (rest unknown)';
export function resultPhrase(visibility) {
  if(visibility?.state==='blocked')return 'blocked';
  const reason=UNKNOWN_REASONS[visibility?.reasonCode];
  return reason?`unknown (${reason})`:PLAIN_UNKNOWN;
}
// Model-facing facts: scalars only, no pixels or world coordinates. A position whose result is Unknown for a reason other
// than a plain miss carries that reason's short text (unknownReason); every position carries its result phrase.
export function scenarioFacts(scene,scenario) {
  const s=scenario,v=s.vehicle,position=(which,setback,visibility)=>{
    const c=actorConflicts(s,which),reason=UNKNOWN_REASONS[visibility.reasonCode];
    // An annotated pose (current vehicle and its what-if) is turned from the lane axis; its setback is the lane-axis
    // distance from the crossing to its front-face centre (geometry.mjs).
    const box=s[which],pose=box.pose==='annotated'?{pose:'annotated',turnedFromLane_deg:r2(turnDeg(box.axis,s.axis))}:{};
    return {...pose,setback_m:r2(setback),setback_ft:r2(setback/FOOT),state:factState(visibility),...(reason&&{unknownReason:reason}),result:resultPhrase(visibility),...c};
  };
  // The driver eye is the recording camera's position and height (createScenario): on a mapped scene measured to the
  // mapped ground under it (null when the map has no value there); measured-depth scenes have the road at Y 0.
  return {scene:scene.id,units:scene.units,eye:s.eyeLabel,eyeHeight_m:s.eyeHeightM===null?null:r2(s.eyeHeightM),vehicle_m:{width:r2(v.width),height:r2(v.height),length:r2(v.length)},
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
  const {vehicle:v}=s,rows=[],own=s.whatIf.pose==='annotated';
  for(const setback of setbacks) {
    // The same what-if box and result rule as createScenario's proposed vehicle at this setback (geometry.mjs whatIfBox and
    // assessSightline) without classifySightline's later diagnostics, which ignore the vehicle and can only give unknown.
    const box=whatIfBox(s.whatIf,setback),r=assessSightline(scene,s,box,{diagnostics:false,ownPosition:own&&setback===s.baselineSetback});
    const row={setback,state:factState(r),...(r.reasonCode&&{reason:r.reasonCode}),gapM:r.gapM};
    if(row.state==='overlap'){const c=boxTests(s,box);Object.assign(row,{person:c.targetInside,driver:c.eyeInside});}
    // assessSightline with its diagnostics stays the authority: it confirms the first row of every run (state and reason
    // only, so the car-shape diagnostics record is skipped).
    const prev=rows.at(-1);
    if(row.state!=='overlap'&&(row.state!==prev?.state||row.reason!==prev?.reason)) {
      const full=assessSightline(scene,s,box,{ownPosition:own&&setback===s.baselineSetback,shapeCut:false});
      if(full.state!==row.state||full.reasonCode!==r.reasonCode)throw new Error('Sweep disagrees with the measured comparison.');
    }
    rows.push(row);
  }
  // The line misses the tested box with nothing else in the way of a result: plain unknown rows, and rows whose only
  // problem is the box overlapping a real object. "Not blocked by the vehicle from" needs every later row to be one,
  // and starts at the first plain row.
  const plain=r=>r.state==='unknown'&&!r.reason,misses=r=>plain(r)||r.reason==='overlaps-real-object'&&r.gapM>0;
  let tail=rows.length;while(tail>0&&misses(rows[tail-1]))tail--;
  const touches=rows.findLastIndex(lineTouches),at=mark=>rows.find(r=>Math.abs(r.setback-mark)<EPS)?.state??null;
  return {label,vehicle:{...v},targetHeight:Number(targetHeight),rows,step:STEP,range:[min,rows.at(-1).setback],
    stopsBlockingAt:rows.slice(tail).find(plain)?.setback??null,
    // notBlockedFrom only requires no later row where the line touches the box (other Unknown rows allowed after it).
    notBlockedFrom:rows.find((r,i)=>i>touches&&plain(r))?.setback??null,
    at15ft:at(MARKS.at15ft),at20ft:at(MARKS.at20ft)};
}
// The line touches the tested box: Blocked, or Unknown only because it is too close to call or because the recorded
// stationary vehicle's LiDAR shape does not prove its height there (car shape; the line cuts into its box), or the box
// overlaps a real object while touching the line.
const lineTouches=r=>r.state==='blocked'||r.reason==='too-close-to-call'||r.reason==='height-not-proven'||r.reason==='overlaps-real-object'&&r.gapM<=0;
// A row whose placement is not tested: Unknown (or overlap) for any reason except too close to call.
const untested=r=>r.state!=='blocked'&&!!r.reason&&r.reason!=='too-close-to-call';

// Consecutive rows in one state (or matching a test), as row groups.
const runs=(rows,state)=>{const is=typeof state==='function'?state:r=>r.state===state;return rows.reduce((out,r,i)=>{
  if(is(r)){if(i>0&&is(rows[i-1]))out.at(-1).push(r);else out.push([r]);}
  return out;
},[]);};
const plainRow=r=>r.state==='unknown'&&!r.reason;
// Unknown reasons named per row range in the phrase and facts (overlap rows keep their own "overlaps person" wording).
const ROW_REASONS=[['overlaps-real-object','overlapsRealObject_m'],['person-inside-object','personInsideRealObject_m'],['driver-inside-object','driverInsideRealObject_m'],
  ['height-not-proven','heightNotProven_m']];
const span=g=>{const a=fmt(g[0].setback),b=fmt(g.at(-1).setback);return a===b?`at ${a} m`:`${a}-${b} m`;};
// The one result phrase per swept vehicle (model facts, Distance check rows and the app's own line), read from the rows:
// blocked runs, too-close-to-call runs, the stop distance (or that no tested row touches the vehicle), then overlaps and
// the other Unknown reasons. Tested distances it does not name are unknown. Every number in it is in sweepFacts.
export function sweepPhrase(r) {
  if(r.error)return r.error;
  if(r.rows.every(x=>x.state==='blocked'))return 'blocked at every tested distance';
  const blocked=runs(r.rows,'blocked'),close=runs(r.rows,x=>x.reason==='too-close-to-call'),touched=r.rows.some(lineTouches),parts=[];
  if(blocked.length)parts.push('blocked '+blocked.map(span).join(', '));
  if(close.length)parts.push('too close to call '+close.map(span).join(', '));
  if(touched&&r.stopsBlockingAt!==null)parts.push(`not blocked by the vehicle from ${fmt(r.stopsBlockingAt)} m (${fmt(r.stopsBlockingAt/FOOT)} ft)`);
  else if(!touched&&r.rows.some(plainRow))parts.push('not blocked by the vehicle at any tested distance (rest unknown)');
  for(const g of runs(r.rows,'overlap')){const p=g.some(x=>x.person),d=g.some(x=>x.driver);parts.push(`overlaps ${p&&d?'person and driver':p?'person':'driver'} ${span(g)}`);}
  for(const [code] of ROW_REASONS){const g=runs(r.rows,x=>x.reason===code);if(g.length)parts.push(`${UNKNOWN_REASONS[code]} ${g.map(span).join(', ')}`);}
  return parts.join('; ');
}
const MARK_WORDS={blocked:'Blocked',unknown:'Unknown',overlap:'Overlap'};
// The app's own 20 ft / 15 ft line for one swept vehicle; Nemotron's answers never carry the marks.
// One short phrase per vehicle for Nemotron's two-sentence summary; the full phrase stays in the panel.
export function sweepShort(r) {
  if(r.error)return r.error;
  const touched=r.rows.some(lineTouches);
  if(touched&&r.stopsBlockingAt!==null)return `not blocked by the vehicle from ${fmt(r.stopsBlockingAt)} m (${fmt(r.stopsBlockingAt/FOOT)} ft)`;
  if(touched) {
    // The last rows are not tested (the box overlaps a real object, or holds the person or driver): name where that
    // starts instead of claiming the vehicle blocks to the end (AV2 8c delivery van: blocked to 10.08 m, then the moved
    // box overlaps a real vehicle from 10.33 m, missing the line from 10.83 m). Each run start is in sweepFacts.
    let i=r.rows.length;while(i>0&&untested(r.rows[i-1]))i--;
    if(i<r.rows.length) {
      const tail=r.rows.slice(i),reasons=new Set(tail.map(x=>x.reason));
      return `unknown from ${fmt(tail[0].setback)} m`+(reasons.size===1?` (${UNKNOWN_REASONS[[...reasons][0]]})`:'');
    }
    return 'no tested distance where it stops blocking';
  }
  if(r.rows.some(plainRow))return 'not blocked by the vehicle at any tested distance';
  // No row touches the vehicle and none is plain: every tested distance is Unknown for another reason.
  const reasons=new Set(r.rows.map(x=>x.reason)),only=reasons.size===1?UNKNOWN_REASONS[[...reasons][0]]:undefined;
  return only?`unknown at every tested distance (${only})`:'unknown at every tested distance';
}
export const sweepMarks=r=>`20 ft: ${MARK_WORDS[r.at20ft]??'out of range'} · 15 ft: ${MARK_WORDS[r.at15ft]??'out of range'}`;

export function sweepFacts(results) {
  const pairs=(rows,state)=>runs(rows,state).map(g=>[r2(g[0].setback),r2(g.at(-1).setback)]),ft=m=>m===null?null:r2(m/FOOT);
  // Other Unknown ranges appear only when a vehicle has them.
  const unknownRanges=rows=>Object.fromEntries([['too-close-to-call','tooCloseToCall_m'],...ROW_REASONS].map(([code,key])=>[key,pairs(rows,x=>x.reason===code)]).filter(([,v])=>v.length));
  return {vehicles:results.map(r=>r.error?{label:r.label,error:r.error}:{label:r.label,
    vehicle_m:{width:r2(r.vehicle.width),height:r2(r.vehicle.height),length:r2(r.vehicle.length)},targetHeight_m:r2(r.targetHeight),
    tested_m:r.range.map(r2),tested_ft:r.range.map(ft),step_m:r.step,
    stopsBlockingAt_m:r.stopsBlockingAt===null?null:r2(r.stopsBlockingAt),stopsBlockingAt_ft:ft(r.stopsBlockingAt),
    at15ft:r.at15ft,at20ft:r.at20ft,blocked_m:pairs(r.rows,'blocked'),overlap_m:pairs(r.rows,'overlap'),...unknownRanges(r.rows),result:sweepPhrase(r),short:sweepShort(r)})};
}

const NUMBER=/\d*\.?\d+/g;
const key=n=>Math.round(n*10);
// Which quantity a computed number measures, from its fact key: vehicle sizes, the person's height, the eye height, and
// distances from the crossing in metres or feet (setbacks, stop distances, tested and blocked ranges). Other numbers
// (step, turn angle) only ground an answer through the plain number check.
const ROLE_NAMES={width:'vehicle width',length:'vehicle length',vehicleHeight:'vehicle height',person:'person height',eye:'eye height',
  height:'height',distance:'distance from the crossing',distance_m:'distance from the crossing in m',distance_ft:'distance from the crossing in ft'};
function roleOf(name,parent) {
  if(parent==='vehicle_m')return {width:'width',length:'length',height:'vehicleHeight'}[name]??null;
  if(name==='targetHeight_m')return 'person';
  if(name==='eyeHeight_m')return 'eye';
  if(name==='step_m'||name==='vehicle_m')return null;
  return name?.endsWith('_ft')?'distance_ft':name?.endsWith('_m')?'distance_m':null;
}
// Only computed numeric values ground an answer: no reference numbers, and labels or ids never carry numbers. The list
// also carries (as its non-enumerable roles property) each number's quantity, so checkAnswer can refuse a computed
// number used for another quantity ("1.4 m wide" from a 1.41 degree turn, "1.49 m from the crossing" from a height),
// and (as notBlocked) where "not blocked by the vehicle" holds, so it can refuse that claim anywhere else.
export function allowedNumbersFrom(...facts) {
  const out=new Set(),roles=Object.fromEntries(Object.keys(ROLE_NAMES).filter(k=>!['height','distance'].includes(k)).map(k=>[k,new Set()]));
  const walk=(v,name,parent)=>{
    if(typeof v==='number'){if(Number.isFinite(v)){out.add(v);const role=roleOf(name,parent);if(role)roles[role].add(key(v));}}
    else if(Array.isArray(v))v.forEach(x=>walk(x,name,parent));
    else if(v&&typeof v==='object')for(const [k,x] of Object.entries(v))walk(x,k,name);
  };
  facts.forEach(f=>walk(f));
  const list=[...out];
  Object.defineProperty(list,'roles',{value:roles});
  Object.defineProperty(list,'notBlocked',{value:notBlockedFacts(facts)});
  return list;
}
// Where the computed facts support "not blocked by the vehicle": positions (scenarioFacts current and proposed, by
// setback) whose result is a plain Unknown, and swept vehicles' stop distances (sweepFacts stopsBlockingAt) or, for a
// vehicle not blocked at any tested distance, its tested range. at: those distances (as keys); current/proposed: whether
// that position is a plain Unknown (null when absent); unnamed: whether a claim with no distance and no position name is
// true of everything computed (every position plain, or with no positions some swept vehicle plain).
function notBlockedFacts(facts) {
  const nb={at:new Set(),current:null,proposed:null,anyTestedDistance:false},add=(...n)=>n.forEach(x=>{if(Number.isFinite(x))nb.at.add(key(x));});
  let positions=0,plain=0,sweepPlain=false;
  const scan=(v,name)=>{
    if(Array.isArray(v)){v.forEach(x=>scan(x,name));return;}
    if(!v||typeof v!=='object')return;
    if(typeof v.state==='string'&&Number.isFinite(v.setback_m)) {
      const p=v.state==='unknown'&&!v.unknownReason;
      positions++;if(p){plain++;add(v.setback_m,v.setback_ft);}
      if(name==='current'||name==='proposed')nb[name]=(nb[name]??true)&&p;
    }
    if(Object.hasOwn(v,'stopsBlockingAt_m')) {
      if(Number.isFinite(v.stopsBlockingAt_m)){sweepPlain=true;add(v.stopsBlockingAt_m,v.stopsBlockingAt_ft);}
      if(typeof v.short==='string'&&v.short.startsWith('not blocked by the vehicle at any tested distance')) {
        sweepPlain=nb.anyTestedDistance=true;add(...(Array.isArray(v.tested_m)?v.tested_m:[]),...(Array.isArray(v.tested_ft)?v.tested_ft:[]));
      }
    }
    for(const [k,x] of Object.entries(v))scan(x,k);
  };
  facts.forEach(f=>scan(f));
  nb.unnamed=positions?plain===positions:sweepPlain;
  return nb;
}
// A number tied to a quantity by the words around it: "1.4 m wide", "2 m tall", "4 m long", "0.7 m (2.3 ft) from the
// crossing", "3 m back", "set back 3 m", "a 1.1 m child", "width of 1.9 m", "eye at 1.47 m". Tall or high takes its
// quantity from the noun it describes (SUBJECT below). The intersection, stop line or corner count as the crossing.
const UNIT=String.raw`(?:(?:(m|meters?|metres?)|(ft|feet|foot))\b)`,CROSSING=String.raw`(?:crossing|crosswalk|intersection|stop\s+line|corner)`;
const AFTER=new RegExp(String.raw`^\s*${UNIT}?\s*\)?\s*(?:\([^)]*\)\s*)?[-\s]*(?:(wide|in\s+width)|(tall|high|in\s+height)|(long|in\s+length)|((?:away\s+)?from\s+the\s+${CROSSING}|(?:before|behind|short\s+of|in\s+front\s+of|back\s+from)\s+the\s+${CROSSING}|back|set[-\s]*back)|((?:tall\s+|high\s+)?(?:child|kid|toddler|person|adult|pedestrian|teen\w*|target)))\b`,'i');
const BEFORE=new RegExp(String.raw`(?:\b(width)|\b(height)|\b(length)|\b(set[-\s]*back(?:\s+by)?|distance\s+(?:from|to)\s+the\s+${CROSSING})|\b(eye|camera)(?:\s+height)?)\s*(?:(?:is|was)(?:\s+at)?|of|at|:|=)?\s*(?:about\s+|around\s+|roughly\s+)?$`,'i');
const NOUNS=String.raw`\b(?:(child|children|kid|toddler|person|people|adult|pedestrian|teen\w*|target|man|woman|boy|girl)|(vehicles?|cars?|suvs?|vans?|trucks?|bus|buses|trailers?|sedans?|pickups?|box|motorcycles?|wagons?|hatchbacks?)|(eye|camera|driver))\b`;
// The noun a "tall" or "high" number describes: one of the next three words ("1.49 m tall vehicle", "1.5 m tall delivery
// van", "1.49 m tall red delivery van"; small joining words such as "and" or "the" end the search), else the last one
// before it in the sentence ("The child is 1.1 m tall").
const JOINING=String.raw`(?:and|or|but|the|a|an|is|are|was|were|than|while|with|of|to|for|from|at|in|on|by|as|so|then|if)\b`;
const SUBJECT=new RegExp(NOUNS,'gi'),SUBJECT_NEXT=new RegExp(String.raw`^[ \t]*(?:(?!${JOINING})[A-Za-z-]+[ \t]+){0,2}`+NOUNS,'i');
// Every number in the text with its quantity (null when no words tie it to one) and unit.
function numberRoles(text) {
  return [...text.matchAll(NUMBER)].map(m=>{
    const end=m.index+m[0].length,after=text.slice(end).match(AFTER),sentence=text.slice(0,m.index).split(/[.;!?](?=\s)/).at(-1),before=sentence.match(BEFORE);
    let unit=after?.[1]?'m':after?.[2]?'ft':null;
    let role=after?(after[3]?'width':after[4]?'height':after[5]?'length':after[6]?'distance':'person')
      :before?(before[1]?'width':before[2]?'height':before[3]?'length':before[4]?'distance':'eye'):null;
    if(role==='height') {
      const subject=(after&&text.slice(end+after[0].length).match(SUBJECT_NEXT))??[...sentence.matchAll(SUBJECT)].at(-1);
      role=subject?.[1]?'person':subject?.[2]?'vehicleHeight':subject?.[3]?'eye':'height';
    }
    // A number with a length unit and no other quantity reads as a distance ("at 5 m", "10.6 m (34.79 ft)", "1.35-10.1 m").
    const plainUnit=!role&&text.slice(end).match(/^\s*(?:(?:-|–|to)\s*\d*\.?\d+\s*)?(?:(m|meters?|metres?)|(ft|feet|foot))\b/i);
    if(plainUnit)unit=plainUnit[1]?'m':'ft';
    return {raw:m[0],value:Number(m[0]),index:m.index,role,unit,distanceLike:role==='distance'||!!plainUnit};
  });
}
function roleMismatch(text,roles) {
  for(const {raw,value,role,unit} of numberRoles(text)) {
    if(!role)continue;
    const sets=role==='distance'?(unit==='ft'?[roles.distance_ft]:unit==='m'?[roles.distance_m]:[roles.distance_m,roles.distance_ft])
      // Sizes and heights are computed in metres only, so a size written in feet never matches.
      :unit==='ft'?[]:role==='height'?[roles.vehicleHeight,roles.person,roles.eye]:[roles[role]];
    if(!sets.some(set=>set.has(key(value))))return `Number ${raw} is not Sightline's computed ${ROLE_NAMES[role==='distance'&&unit?`distance_${unit}`:role]}.`;
  }
}
// A claim that the vehicle does not block the line, in any wording the other checks let through.
const NOT_BLOCKED=/\bnot\s+blocked\s+by\s+the\s+vehicle\b|n't\s+blocked\s+by\s+the\s+vehicle\b|\b(?:does|do|did)\s*(?:not|n't)\s+block\b|\bstops?\s+blocking\s+(?:at|from|beyond|past|after)\b|\bno\s+longer\s+block(?:s|ing)?\b|\bmiss(?:es|ed)?\s+the\s+(?:line|sightline)\b/i;
// "Not blocked by the vehicle" only where a computed result says so (audit, October 12: on AV2 8c the adult at 5 m is
// Unknown, too close to call, yet "At 5 m the person is not blocked by the vehicle (rest unknown)" passed). Per clause
// (split at . ; ! ? and at but / while / whereas / however): every distance in a claiming clause must be one where the
// claim holds; a clause naming the proposed or current (stationary) position needs that position's plain result; a
// claim with neither needs every computed position to be plain. Withholding a true sentence only shows the app's line.
function notBlockedMismatch(text,nb) {
  const numbers=numberRoles(text);
  const clauses=[],split=/[.;!?](?=\s|$)|,?\s+\b(?:but|while|whereas|however)\b/gi;
  let from=0;for(const m of text.matchAll(split)){clauses.push([from,m.index]);from=m.index+m[0].length;}
  clauses.push([from,text.length]);
  for(const [a,b] of clauses) {
    const clause=text.slice(a,b);
    if(!NOT_BLOCKED.test(clause))continue;
    const proposed=/\b(?:proposed|what-?if)\b/i.test(clause),current=/\b(?:current|stationary|baseline|existing)\b/i.test(clause);
    if(proposed&&nb.proposed!==true)return 'Calls the proposed position not blocked by the vehicle; Sightline did not compute that.';
    if(current&&nb.current!==true)return 'Calls the current position not blocked by the vehicle; Sightline did not compute that.';
    const distances=numbers.filter(n=>n.index>=a&&n.index<b&&n.distanceLike);
    const bad=distances.find(n=>!nb.at.has(key(n.value)));
    if(bad)return `Says not blocked by the vehicle at ${bad.raw} ${bad.unit??'m'}, where Sightline did not compute that.`;
    if(!distances.length&&!proposed&&!current&&!(/\bat\s+any\s+tested\s+distance\b/i.test(clause)?nb.anyTestedDistance:nb.unnamed))
      return 'Says not blocked by the vehicle without a distance, but not every computed result says so.';
  }
}
export const MAX_ANSWER=280;
// mapped: a mapped-ground (AV2) scene, where an answer using any word built on "park", a reason for the vehicle being
// stopped, a measured-source claim or a car make or model is withheld too.
export function checkAnswer(text,allowedNumbers=[],{mapped=false}={}) {
  if(typeof text!=='string'||!text.trim())return {ok:false,kind:'empty',reason:'Empty answer.'};
  if(text.length>MAX_ANSWER)return {ok:false,kind:'length',reason:'Answer is longer than two short sentences.'};
  const word=withheld(text,mapped);
  if(word)return {ok:false,kind:'wording',reason:`Uses withheld wording "${word}".`};
  const allowed=new Set(allowedNumbers.filter(Number.isFinite).map(key));
  const stray=(text.match(NUMBER)??[]).find(n=>!allowed.has(key(Number(n))));
  if(stray)return {ok:false,kind:'number',reason:`Number ${stray} is not in Sightline's computed facts.`};
  // Lists from allowedNumbersFrom know each number's quantity and where "not blocked" holds; a plain list is checked by
  // value only.
  const misplaced=allowedNumbers.roles&&roleMismatch(text,allowedNumbers.roles);
  if(misplaced)return {ok:false,kind:'role',reason:misplaced};
  const unsupported=allowedNumbers.notBlocked&&notBlockedMismatch(text,allowedNumbers.notBlocked);
  if(unsupported)return {ok:false,kind:'result',reason:unsupported};
  return {ok:true,reason:'',text};
}
const WITHHELD={length:'too long.',number:'used a number Sightline did not compute.',role:'used a computed number for the wrong quantity.',wording:'used wording Sightline does not allow.',
  result:'described a result Sightline did not compute.'};
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
    // The distance check always tests the vehicle on screen, even if Nemotron left it out. Its typed sizes and the person
    // height go to the sweep as entered, so createScenario's strict number check applies to them too.
    const own={label:'your vehicle',...Object.fromEntries(DIMS.map(k=>[k,from.vehicle[k]]))};
    const listed=checked.value.vehicles,mine=v=>DIMS.every(k=>Math.abs(v[k]-Number(own[k]))<.005);
    const vehicles=mode==='check'&&!listed.some(mine)?[own,...listed].slice(0,4):listed;
    const results=vehicles.map(v=>setbackSweep(scene,{pixels:from.pixels,driverDistance,vehicleDeclared:from.vehicleDeclared},v,from.targetHeight));
    // vehicleDeclared: whether the checked vehicle was a declared box (false: an annotated pose may have been swept).
    return {result:sweepFacts(results),input:from,sweep:{pixels:from.pixels,targetHeight:height,driverDistance,results,own:{...from.vehicle},vehicleDeclared:!!from.vehicleDeclared}};
  }
  return {error:`Unknown tool ${name}.`};
}

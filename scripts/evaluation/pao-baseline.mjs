#!/usr/bin/env node
// Palo Alto accuracy baseline (task M0). Loads the local AV2 8c development scene the way tests/av2-ground.test.mjs does
// and measures the app's own geometry (src/*.mjs, unchanged) against named references. Development-grade,
// AV2-referenced numbers from one development scene: not surveyed, not held-out, never a passed gate.
//
//   node scripts/evaluation/pao-baseline.mjs [--date YYYY-MM-DD] [--out DIR] [--no-coverage] [--samples N]
//
// Writes DIR/baseline-<UTC date>.json (default DIR .local/evaluation/pao-accuracy/). Reads only local files.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {existsSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {performance} from 'node:perf_hooks';
import os from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {loadScene} from '../../src/scene.mjs';
import {mappedGroundHit,mappedGroundHeight,createScenario,comparisonStart,segmentIntersectsBox,whatIfBox,annotatedVehicle,TOO_CLOSE_TO_CALL_M} from '../../src/geometry.mjs';
import {setbackSweep,actorConflicts} from '../../src/scenario-assistant.mjs';
import {suggestedSetup} from '../../src/suggestions.mjs';

export const ROOT=new URL('../../',import.meta.url);
export const DEV={id:'av2-pao-8c52d911-2658849',dir:new URL('.local/evaluation/av2-map-first-2026-10-11/',ROOT)};
export const OUT_DIR=new URL('.local/evaluation/pao-accuracy/',ROOT);
// Range bands for click-to-ground (task M0) and the planning lane's bands (reproduction check only).
export const BANDS=[[0,10],[10,12],[12,25],[25,30],[30,60]];
export const LANE_BANDS=[[0,10],[10,15],[15,20],[20,30],[30,60]];
// A camera-visible LiDAR return counts as a ground return when its height is within this of the mapped ground under it.
export const GROUND_GATE_M=.25,STRICT_GROUND_GATE_M=.08;
// Sean (2026-10-12): a line that cuts into the box by less than 10 cm is "too close to call"; the margin is still to
// be justified by the locked error budget. Since M1 the app applies it (geometry.mjs TOO_CLOSE_TO_CALL_M); here it
// counts borderline rows and checks the app's states against this file's own clearance.
export const BORDERLINE_M=.10;
// The app's line touches the tested box (Blocked, or Unknown because too close to call: M1) when its signed gap is <= 0.
const lineTouches=v=>v.gapM<=0;
export const ADULT_HEIGHT_M=1.5;
const GROUNDED=new Set(['scene.json','ground.json','terrain.bin','suggestions.json']);
const r=(n,d=4)=>n==null||!Number.isFinite(n)?n:Number(n.toFixed(d));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
// Output names for the app's pixel keys (its vehicle-front key is not used in written results).
const pixelsOut=p=>({crossing:p.crossing,vehicleFront:p.parking,person:p.target});

export const devDataPresent=()=>existsSync(new URL('grounded-v2/scene.json',DEV.dir))&&existsSync(new URL('assembled/positions.bin',DEV.dir))
  &&existsSync(new URL('grounded-v2/catalog-entry.json',DEV.dir));

// Serves the contract's directory like tests/av2-ground.test.mjs: assembled buffers plus every grounded-v2 file. cache
// holds file bytes, so a second load with the same cache measures loadScene without disk reads.
export async function loadDevelopmentScene(cache=new Map()) {
  const entry=JSON.parse(await readFile(new URL('grounded-v2/catalog-entry.json',DEV.dir),'utf8'));
  const fetch=async path=>{
    const name=path.split('/').at(-1);
    if(!path.startsWith(`/scenes/${DEV.id}/`)||['catalog-entry.json','LICENSE-AND-ATTRIBUTION.md'].includes(name))return new Response(null,{status:404});
    try{
      if(!cache.has(name))cache.set(name,await readFile(new URL((GROUNDED.has(name)?'grounded-v2/':'assembled/')+name,DEV.dir)));
      return new Response(cache.get(name));
    }catch{return new Response(null,{status:404});}
  };
  return {scene:await loadScene(fetch,DEV.id,entry.metadataSha256,undefined,entry.suggestions,entry.ground),entry};
}

// Nearest-rank percentile (sorted[ceil(p n) - 1]); the median is the lower middle value for an even count.
export function percentile(values,p) {
  if(!values.length)return null;
  const s=Float64Array.from(values).sort();
  return s[Math.max(0,Math.ceil(p*s.length)-1)];
}
const stats=values=>({median:r(percentile(values,.5)),p95:r(percentile(values,.95)),max:r(values.length?Math.max(...values):null)});

// Box in classifySightline's local frame: side = [axis.z, 0, -axis.x], local = [d·side, d.y, d·axis], half sizes
// [width, height, length] / 2. box: {center, axis (horizontal unit), vehicle: {width, height, length}}.
function boxFrame({center:c,axis:x,vehicle:v}) {
  const side=[x[2],0,-x[0]];
  return {local:p=>{const d=[p[0]-c[0],p[1]-c[1],p[2]-c[2]];return [d[0]*side[0]+d[2]*side[2],d[1],d[0]*x[0]+d[2]*x[2]];},half:[v.width/2,v.height/2,v.length/2]};
}
// Signed sightline-to-box clearance: the smallest uniform change of all three half sizes at which the segment touches
// the box. Positive: the line misses the box by that much; negative: it cuts in and the box would have to shrink by
// that much on every side to miss. It equals min over t in [0, 1] of max_i(|p_i(t)| - half_i), a convex function of t,
// so a golden-section search finds it. t is the fraction along the line (start = eye) where the minimum sits.
export function signedClearance(start,end,box) {
  const {local,half}=boxFrame(box),a=local(start),b=local(end);
  const f=t=>Math.max(Math.abs(a[0]+t*(b[0]-a[0]))-half[0],Math.abs(a[1]+t*(b[1]-a[1]))-half[1],Math.abs(a[2]+t*(b[2]-a[2]))-half[2]);
  const g=(Math.sqrt(5)-1)/2;
  let lo=0,hi=1,x1=hi-g*(hi-lo),x2=lo+g*(hi-lo),f1=f(x1),f2=f(x2);
  for(let k=0;k<120&&hi-lo>1e-15;k++) {
    if(f1<=f2){hi=x2;x2=x1;f2=f1;x1=hi-g*(hi-lo);f1=f(x1);}
    else{lo=x1;x1=x2;f1=f2;x2=lo+g*(hi-lo);f2=f(x2);}
  }
  let best={t:(lo+hi)/2,clearanceM:f((lo+hi)/2)};
  for(const t of [0,1])if(f(t)<best.clearanceM)best={t,clearanceM:f(t)};
  return best;
}
// The app's own segment-box test in the same frame (true: the line touches or crosses the box).
export function appIntersects(start,end,box) {
  const {local,half}=boxFrame(box);
  return segmentIntersectsBox(local(start),local(end),half.map(n=>-n),half);
}

// Annotated cuboid -> upright box (horizontal heading from the box x column; sub-degree pitch and roll unused, as in the app).
export function cuboidBox(cuboid) {
  const M=cuboid.worldFromBox,n=Math.hypot(M[0][0],M[2][0]);
  return {center:[...cuboid.center],axis:[M[0][0]/n,0,M[2][0]/n],vehicle:{width:cuboid.width,height:cuboid.height,length:cuboid.length}};
}
// Upright boxes overlap: oriented footprints overlap (separating-axis test, touching counts) and height ranges overlap.
export function boxesOverlap(A,B) {
  if(Math.abs(A.center[1]-B.center[1])>(A.vehicle.height+B.vehicle.height)/2)return false;
  const rect=b=>{const u=[b.axis[0],b.axis[2]];return {c:[b.center[0],b.center[2]],u,v:[-u[1],u[0]],hl:b.vehicle.length/2,hw:b.vehicle.width/2};};
  const a=rect(A),b=rect(B);
  for(const ax of [a.u,a.v,b.u,b.v]) {
    const span=R=>{const c=R.c[0]*ax[0]+R.c[1]*ax[1],e=R.hl*Math.abs(R.u[0]*ax[0]+R.u[1]*ax[1])+R.hw*Math.abs(R.v[0]*ax[0]+R.v[1]*ax[1]);return [c-e,c+e];};
    const [a0,a1]=span(a),[b0,b1]=span(b);
    if(a1<b0||b1<a0)return false;
  }
  return true;
}
// Closed segment camera origin -> point against every annotated cuboid (geometry.mjs hiddenByCuboids, same slab test).
function behindAnyCuboid(ground,point) {
  return ground.cuboids.items.some(box=>{
    const M=box.worldFromBox,c=box.center,local=p=>[0,1,2].map(k=>(p[0]-c[0])*M[0][k]+(p[1]-c[1])*M[1][k]+(p[2]-c[2])*M[2][k]);
    const half=[box.length/2,box.width/2,box.height/2];
    return segmentIntersectsBox(local([0,0,0]),local(point),half.map(n=>-n),half);
  });
}

// ---------------------------------------------------------------- B. click to ground vs camera-visible LiDAR
// Reference: LiDAR returns the left camera can see (prepared referenceEligible = finite, positive camera depth, inside
// the crop; and the camera-origin-to-return segment crosses no annotated cuboid: protocol v2 cameraVisibleReturn) whose
// height is within the gate of the mapped ground. The "click" is the return's nearest integer pixel; the app's
// mappedGroundHit places it. Error = horizontal distance from the placed point to the return. Partly independent:
// the publisher built the ground raster from LiDAR, and the ground gate uses that raster.
export function clickToGround(scene) {
  const m=scene.measurements,c=scene.camera,g=scene.ground,rows=[],skipped={notCameraEligible:0,nonFinite:0,noMappedHeight:0,notGroundReturn:0,behindAnnotatedCuboid:0,outsideImage:0};
  for(let i=0;i<m.count;i++) {
    if(m.referenceEligible[i]!==1){skipped.notCameraEligible++;continue;}
    const p=[m.referencePositions[3*i],m.referencePositions[3*i+1],m.referencePositions[3*i+2]];
    if(!p.every(Number.isFinite)){skipped.nonFinite++;continue;}
    const h=mappedGroundHeight(scene,p[0],p[2]);
    if(h==null){skipped.noMappedHeight++;continue;}
    const dy=p[1]-h;
    if(Math.abs(dy)>GROUND_GATE_M){skipped.notGroundReturn++;continue;}
    if(behindAnyCuboid(g,p)){skipped.behindAnnotatedCuboid++;continue;}
    const px=[Math.floor(m.referenceUV[2*i]+.5),Math.floor(m.referenceUV[2*i+1]+.5)];
    if(px[0]<0||px[0]>=c.width||px[1]<0||px[1]>=c.height){skipped.outsideImage++;continue;}
    const range=Math.hypot(p[0],p[2]),row={range,dy};
    try{
      const hit=mappedGroundHit(scene,px),dx=hit.point[0]-p[0],dz=hit.point[2]-p[2],u=[p[0]/range,p[2]/range];
      Object.assign(row,{error:Math.hypot(dx,dz),along:dx*u[0]+dz*u[1],shown:hit.alongRayUncertaintyM});
    }catch(e){row.unknown=e.message;}
    rows.push(row);
  }
  const bin=(bands,gate)=>bands.map(([a,b])=>{
    const inBand=rows.filter(x=>x.range>=a&&x.range<b&&Math.abs(x.dy)<=gate),placed=inBand.filter(x=>x.unknown===undefined),unknown={};
    for(const x of inBand)if(x.unknown!==undefined)unknown[x.unknown]=(unknown[x.unknown]??0)+1;
    const e=placed.map(x=>x.error),shown=placed.map(x=>x.shown);
    return {band:`${a}-${b} m`,candidates:inBand.length,placed:placed.length,unknown,
      horizontalErrorM:stats(e),medianSignedAlongRayM:r(percentile(placed.map(x=>x.along),.5)),
      shownUncertaintyCoverage:placed.length?r(placed.filter(x=>x.error<=x.shown).length/placed.length,3):null,
      shownUncertaintyM:{median:r(percentile(shown,.5)),p95:r(percentile(shown,.95))}};
  });
  return {
    reference:'camera-visible LiDAR ground returns of the scene sweep (protocol v2 cameraVisibleReturn; ground gate on mapped height)',
    independence:'partly independent: LiDAR is a direct observation, but the publisher ground raster the app uses was built from LiDAR and the ground gate uses that raster',
    click:'nearest integer pixel to the return projection (floor(uv + 0.5)); the app accepts integer pixels only',
    errorDefinition:'horizontal (world X/Z) distance from the app placed point to the return; signed along-ray > 0 means placed beyond the return',
    groundGateM:GROUND_GATE_M,skipped,
    bands:bin(BANDS,GROUND_GATE_M),
    strictGate:{groundGateM:STRICT_GROUND_GATE_M,bands:bin(BANDS,STRICT_GROUND_GATE_M)},
    laneBandsForReproduction:bin(LANE_BANDS,GROUND_GATE_M),
  };
}

// Every crop pixel through the app's placement rule. Returns the placeable pixel list for sampling.
export function placeablePixels(scene) {
  const {width,height}=scene.camera,reasons={},list=[];
  let observed=0,hidden=0;
  for(let v=0;v<height;v++)for(let u=0;u<width;u++) {
    try{const hit=mappedGroundHit(scene,[u,v]);list.push([u,v]);if(hit.supportClass==='observed-nearby')observed++;if(hit.hidden)hidden++;}
    catch(e){reasons[e.message]=(reasons[e.message]??0)+1;}
  }
  return {list,summary:{total:width*height,placed:list.length,fraction:r(list.length/(width*height)),observedNearby:observed,mapOnly:list.length-observed,hiddenPlaced:hidden,unknownReasons:reasons}};
}

// ---------------------------------------------------------------- C. suggestion internal consistency
// The stored suggestion pixel's runtime ground hit vs the intended point the suggestion was made from (mapped crossing
// point, annotated vehicle front-face centre, mapped person point). These are the app's own inputs: a self-check, not
// accuracy.
export function suggestionConsistency(scene) {
  const s=scene.suggestions,travel=s.roadAxis.directionOfTravel,right=[-travel[2],0,travel[0]],out={};
  for(const [name,role] of [['crossing','crossing'],['vehicleFront','vehicle'],['person','person']]) {
    const e=s[role];
    if(e.status==='unknown'){out[name]={status:'unknown',reason:e.reason};continue;}
    const hit=mappedGroundHit(scene,e.pixel),d=[hit.point[0]-e.mappedPoint[0],hit.point[1]-e.mappedPoint[1],hit.point[2]-e.mappedPoint[2]];
    out[name]={pixel:e.pixel,horizontalM:r(Math.hypot(d[0],d[2])),alongTravelM:r(d[0]*travel[0]+d[2]*travel[2]),
      rightOfTravelM:r(d[0]*right[0]+d[2]*right[2]),verticalM:r(d[1]),storedResidualM:r(e.pixelOffsetFromMappedM),
      pixelFromProjectionPx:r(Math.hypot(e.pixel[0]-e.projectedUV[0],e.pixel[1]-e.projectedUV[1]),3),rangeM:r(hit.rangeM,3)};
  }
  return {label:'internal consistency (the app checked against its own map and annotation inputs), not accuracy',roles:out};
}

// ---------------------------------------------------------------- setback sweep with clearance
const pose=(scene,setup)=>annotatedVehicle(scene,setup.pixels,setup.vehicle);
function sweepRows(scene,setup,targetHeight) {
  const input={...setup.input,targetHeight:String(targetHeight)},sc=createScenario(scene,input),vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,Number(setup.vehicle[k])]));
  const sweep=setbackSweep(scene,{pixels:setup.pixels,driverDistance:input.driverDistance},{label:'your vehicle',...vehicle},targetHeight);
  if(sweep.error)throw new Error(sweep.error);
  const track=pose(scene,setup)?.track,others=scene.ground.cuboids.items.filter(b=>b.track!==track);
  let signMismatches=0;
  const rows=sweep.rows.map(row=>{
    const box=whatIfBox(sc.whatIf,row.setback),{clearanceM}=signedClearance(sc.eye,sc.target,box);
    if((clearanceM<=0)!==appIntersects(sc.eye,sc.target,box)&&Math.abs(clearanceM)>1e-9)signMismatches++;
    const overlaps=others.filter(b=>boxesOverlap(box,cuboidBox(b))).map(b=>`${b.category}:${b.track.slice(0,8)}`);
    return {setbackM:r(row.setback),state:row.state,...(row.reason&&{reason:row.reason}),clearanceM:r(clearanceM),...(row.state==='overlap'&&{person:row.person,driver:row.driver}),...(overlaps.length&&{overlapsAnnotated:overlaps})};
  });
  return {sc,sweep,rows,signMismatches};
}
// Intervals of the what-if along its slider range where the eye-to-target line cuts at least cut metres into the box
// (signed clearance <= -cut), from sign changes (scan step 0.05 m, then bisection to 1e-6 m). cut 0: the line touches
// the box (the app's Blocked or too close to call; its stopsBlockingAt meaning). cut TOO_CLOSE_TO_CALL_M: the app's
// Blocked (a cut of at least 0.10 m), the result metric H counts. Overlaps with real objects are not removed here; the
// caller confirms each interval with the app. Uses whatIfBox and the clearance.
function cutIntervals(sc,eye,target,cut,boxAt=s=>whatIfBox(sc.whatIf,s)) {
  const lo=sc.limits.minimumSetback,hi=sc.limits.maximumSetback,c=s=>signedClearance(eye,target,boxAt(s)).clearanceM+cut,edges=[];
  const step=.05,n=Math.ceil((hi-lo)/step);
  let prevS=lo,prev=c(lo);
  for(let k=1;k<=n;k++) {
    const s=Math.min(hi,lo+k*step),v=c(s);
    if((prev<=0)!==(v<=0)) {
      let a=prevS,b=s;
      for(let i=0;i<50&&b-a>1e-7;i++){const mid=(a+b)/2;if((c(mid)<=0)===(prev<=0))a=mid;else b=mid;}
      edges.push({setbackM:(a+b)/2,to:v<=0?'in':'out'});
    }
    prevS=s;prev=v;
  }
  const intervals=[];let open=c(lo)<=0?lo:null;
  for(const e of edges){if(e.to==='in')open=e.setbackM;else{intervals.push([open,e.setbackM]);open=null;}}
  if(open!==null)intervals.push([open,hi]);
  return intervals;
}
const lineTouchIntervals=(sc,eye,target,boxAt)=>cutIntervals(sc,eye,target,0,boxAt);
const appBlockedIntervals=(sc,eye,target,boxAt)=>cutIntervals(sc,eye,target,TOO_CLOSE_TO_CALL_M,boxAt);
export function setbackSweepBaseline(scene,setup) {
  const child=Number(setup.targetHeight),out={definition:{
    sweep:'app setbackSweep (src/scenario-assistant.mjs) at the suggested driver distance: rows every 0.25 m from limits.minimumSetback, plus the last setback and the 15 ft / 20 ft marks',
    vehicle:'annotated pose of the suggested stationary vehicle moved along the lane axis (whatIfBox), as the app does',
    clearance:'signed sightline-to-box clearance, metres (positive: the line misses the box; negative: it cuts in)',
    overlapsAnnotated:'what-if box overlaps another real annotated cuboid (upright footprints and height ranges; the vehicle own track excluded)'}};
  for(const [name,height] of [['child',child],['adult',ADULT_HEIGHT_M]]) {
    const {sc,sweep,rows,signMismatches}=sweepRows(scene,setup,height);
    const touchIv=lineTouchIntervals(sc,sc.eye,sc.target),blockedIv=appBlockedIntervals(sc,sc.eye,sc.target);
    // The app confirms each interval: createScenario inside and just outside its ends. Line touches: Blocked or too close
    // to call (M1); app Blocked: state blocked.
    const at=s=>createScenario(scene,{...setup.input,targetHeight:String(height),proposedSetback:s}).proposedVisibility;
    const confirm=(iv,test)=>iv.every(([a,b])=>test(at((a+b)/2))&&(a<=sc.limits.minimumSetback+1e-6||!test(at(a-1e-4)))&&(b>=sc.limits.maximumSetback-1e-6||!test(at(b+1e-4))));
    const nonOverlap=rows.filter(x=>x.state!=='overlap');
    // Fine scan (0.01 m) of the setbacks where the what-if overlaps another real annotated cuboid.
    const track=pose(scene,setup)?.track,ranges={};
    for(let s=sc.limits.minimumSetback;s<=sc.limits.maximumSetback+1e-9;s+=.01) {
      const box=whatIfBox(sc.whatIf,Math.min(s,sc.limits.maximumSetback));
      for(const b of scene.ground.cuboids.items)if(b.track!==track&&boxesOverlap(box,cuboidBox(b))){const k=`${b.category}:${b.track.slice(0,8)}`;(ranges[k]??=[s,s])[1]=s;}
    }
    out[name]={targetHeightM:height,rows,count:rows.length,blockedRows:rows.filter(x=>x.state==='blocked').length,
      lineTouchesBoxRows:rows.filter(x=>x.clearanceM<=0&&x.state!=='overlap').length,tooCloseToCallRows:rows.filter(x=>x.reason==='too-close-to-call').map(x=>x.setbackM),
      whatIfOverlapsAnnotatedSetbacksM:Object.fromEntries(Object.entries(ranges).map(([k,v])=>[k,v.map(n=>r(n,2))])),
      stopsBlockingAtRowM:r(sweep.stopsBlockingAt),at15ft:sweep.at15ft,at20ft:sweep.at20ft,
      lineTouchesIntervalsM:touchIv.map(x=>x.map(n=>r(n))),lineTouchesIntervalsConfirmedByApp:confirm(touchIv,lineTouches),
      appBlockedIntervalsM:blockedIv.map(x=>x.map(n=>r(n))),appBlockedIntervalsConfirmedByApp:confirm(blockedIv,v=>v.state==='blocked'),clearanceSignMismatches:signMismatches,
      borderline:{blockedCutInLessThanM:nonOverlap.filter(x=>x.state==='blocked'&&x.clearanceM>-BORDERLINE_M).map(x=>x.setbackM),
        // Misses only (positive clearance): since M1 Unknown rows can also cut in (too close to call).
        unknownMissByLessThanM:nonOverlap.filter(x=>x.state==='unknown'&&x.clearanceM>0&&x.clearanceM<BORDERLINE_M).map(x=>x.setbackM)}};
  }
  return out;
}

// Where the line stops touching the box and where the app's Blocked ends, and how much each moves with the driver
// position and small box errors (diagnostic only). lineTouches*: the line touches the box (the app's stopsBlockingAt
// meaning; Blocked or too close to call). appBlocked*: the app's Blocked (a cut of at least 0.10 m).
export function blockedEndRange(scene,setup) {
  const sc=createScenario(scene,setup.input),L=sc.limits,ends=[],iv3=iv=>iv.map(x=>x.map(n=>r(n,3)));
  for(let k=0;k<=40;k++) {
    const d=L.minimumDriverDistance+(L.captureDistance-L.minimumDriverDistance)*k/40,eye=createScenario(scene,{...setup.input,driverDistance:d}).eye;
    ends.push({driverDistanceM:r(d,3),lineTouchesM:iv3(lineTouchIntervals(sc,eye,sc.target)),appBlockedM:iv3(appBlockedIntervals(sc,eye,sc.target))});
  }
  const lastEnd=k=>{const e=ends.filter(x=>x[k].length).map(x=>x[k].at(-1)[1]);return e.length?[r(Math.min(...e),3),r(Math.max(...e),3)]:null;};
  const perturb=(name,fn)=>{const boxAt=s=>fn(whatIfBox(sc.whatIf,s));
    return {perturbation:name,lineTouchesM:iv3(lineTouchIntervals(sc,sc.eye,sc.target,boxAt)),appBlockedM:iv3(appBlockedIntervals(sc,sc.eye,sc.target,boxAt))};};
  const turn=(deg)=>b=>{const a=deg*Math.PI/180,c=Math.cos(a),s=Math.sin(a);return {...b,axis:[b.axis[0]*c-b.axis[2]*s,0,b.axis[0]*s+b.axis[2]*c]};};
  const shift=(m,dir)=>b=>{const side=[b.axis[2],0,-b.axis[0]],v=dir==='side'?side:b.axis;return {...b,center:b.center.map((n,i)=>n+v[i]*m)};};
  // The stationary vehicle's own (current) result across the driver slider, at its annotated setback (401 positions).
  const current=[];
  for(let k=0;k<=400;k++){const d=L.minimumDriverDistance+(L.captureDistance-L.minimumDriverDistance)*k/400,v=createScenario(scene,{...setup.input,driverDistance:d}).currentVisibility;current.push([d,v.state,v.reasonCode]);}
  const unknownRuns=[];
  current.forEach(([d,st],i)=>{if(st==='blocked')return;if(i>0&&current[i-1][1]!=='blocked')unknownRuns.at(-1).to=d;else unknownRuns.push({from:d,to:d});});
  const touch=lineTouchIntervals(sc,sc.eye,sc.target)[0]??null,blocked=appBlockedIntervals(sc,sc.eye,sc.target)[0]??null;
  return {
    eye:'driver eye = recorded camera moved along the lane axis by (driverDistance - captureDistance), height unchanged (createScenario)',
    atNamedEye:{driverDistanceM:r(L.captureDistance,4),lineTouchesFromM:touch&&r(touch[0],4),lineTouchesUntilM:touch&&r(touch[1],4),
      appBlockedFromM:blocked&&r(blocked[0],4),appBlockedUntilM:blocked&&r(blocked[1],4),
      note:'line touches: clearance <= 0 (app Blocked or too close to call); app Blocked: clearance <= -0.10 m. Both confirmed with createScenario (setbackSweep.child.*ConfirmedByApp)'},
    laneReconciliation:[
      {lane:'accuracy metrics',value:'10.50 / 10.60 m',explanation:'same eye as here; 10.50 is the last row of a 0.05 m grid whose line touches the box and 10.60 the first row of the app 0.25 m sweep after it (stopsBlockingAt). Both bracket where the line stops touching the box at the named eye (atNamedEye.lineTouchesUntilM), not where the app Blocked ends (appBlockedUntilM).'},
      {lane:'error budget',value:'7.69 / 7.99 m',explanation:'loop limits, not transitions: its scans stopped at setback change <= 7 m (0.6865 + 7 = 7.69 m) and setback <= 8 m (last 0.01 m sample 7.993 m) while still Blocked (scratch budget.mjs).'},
      {lane:'error budget facts',value:'11.37 m',explanation:'withdrawn (2026-10-12): it has no source. No saved lane output contains it, and no driver position on the slider or listed box perturbation gives it here.'},
      {lane:'robustness',value:'10.5342 m',explanation:'same eye as here, 13,000-step scan; agrees with atNamedEye.lineTouchesUntilM.'},
      {lane:'challenge',value:'385 of 401 driver positions Blocked',explanation:'this is the stationary vehicle own result across the driver slider (currentResultAcrossDriverPositions), not where the what-if stops blocking.'},
    ],
    currentResultAcrossDriverPositions:{positions:current.length,blocked:current.filter(x=>x[1]==='blocked').length,tooCloseToCall:current.filter(x=>x[2]==='too-close-to-call').length,
      notBlockedRangesM:unknownRuns.map(x=>[r(x.from,3),r(x.to,3)])},
    driverPositions:ends,
    lastLineTouchesEndAcrossDriverPositionsM:lastEnd('lineTouchesM'),lastAppBlockedEndAcrossDriverPositionsM:lastEnd('appBlockedM'),
    driverPositionsWithAnyLineTouches:`${ends.filter(e=>e.lineTouchesM.length).length} of ${ends.length}`,
    driverPositionsWithAnyAppBlocked:`${ends.filter(e=>e.appBlockedM.length).length} of ${ends.length}`,
    boxSensitivity:[perturb('none',b=>b),perturb('yaw +1 deg',turn(1)),perturb('yaw -1 deg',turn(-1)),perturb('box 0.10 m right of travel',shift(.1,'side')),
      perturb('box 0.10 m left of travel',shift(-.1,'side')),perturb('box 0.10 m toward camera',shift(.1,'axis')),perturb('box 0.10 m away from camera',shift(-.1,'axis'))],
    sideConvention:'right of travel = [axis.z, 0, -axis.x] of the what-if box (axis points toward the camera; classifySightline local x)',
  };
}

// ---------------------------------------------------------------- honesty counters
// mulberry32: a small, fixed, seeded generator so the sample is reproducible from this file alone.
function rng(seed) {
  let a=seed>>>0;
  return ()=>{a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}
const VEHICLES=[{label:'compact car',width:1.8,height:1.45,length:4.4},{label:'SUV',width:2,height:1.8,length:5},{label:'delivery van',width:2.1,height:2.6,length:5.9},
  {label:'box truck',width:2.5,height:3.5,length:8},{label:'bus',width:2.6,height:3.2,length:12},{label:'annotated size',width:1.88,height:1.49,length:4.03}];
const HEIGHTS=[.5,.8,1.1,1.5,1.7,2];
// Valid random placements (half 'uniform': three random placeable pixels; half 'structured': front 1-30 px below the
// crossing, person within 100 px across and 8 px up or down), each with a vehicle and person height, the what-if at its
// start values. Pixels are drawn over the whole crop and kept only if the app places them.
export function samplePlacements(scene,{count=3000,seed=12345,maxAttempts=400000}={}) {
  const rand=rng(seed),{width,height}=scene.camera,ri=(a,b)=>a+Math.floor(rand()*(b-a+1));
  const placeable=([u,v])=>{if(u<0||u>=width||v<0||v>=height)return false;try{mappedGroundHit(scene,[u,v]);return true;}catch{return false;}};
  const anyPixel=()=>{for(;;){const p=[ri(0,width-1),ri(0,height-1)];if(placeable(p))return p;}};
  const out=[],errors={};let attempts=0;
  while(out.length<count&&attempts<maxAttempts) {
    attempts++;
    const kind=out.length%2?'structured':'uniform',c=anyPixel();
    const pixels=kind==='uniform'?{crossing:c,parking:anyPixel(),target:anyPixel()}:{crossing:c,parking:[c[0]+ri(-40,40),c[1]+ri(1,30)],target:[c[0]+ri(-100,100),c[1]+ri(-8,8)]};
    const v=rand()<.7?VEHICLES[ri(0,VEHICLES.length-1)]:{label:'random',width:r(.5+rand()*3,3),height:r(.5+rand()*4,3),length:r(1+rand()*19,3)};
    const h=rand()<.7?HEIGHTS[ri(0,HEIGHTS.length-1)]:r(.3+rand()*2.2,3);
    if(!placeable(pixels.parking)||!placeable(pixels.target)){errors['vehicle front or person pixel not placeable']=(errors['vehicle front or person pixel not placeable']??0)+1;continue;}
    const vehicle={width:String(v.width),height:String(v.height),length:String(v.length)};
    try{
      const start=comparisonStart(scene,pixels,vehicle),input={pixels,vehicle,targetHeight:String(h),...start};
      createScenario(scene,input);out.push({kind,input,label:v.label});
    }catch(e){errors[e.message]=(errors[e.message]??0)+1;}
  }
  return {placements:out,attempts,errors,generator:`mulberry32 seed ${seed}`};
}
export function honestyCounters(scene,setup,placements) {
  const suggestionTrack=pose(scene,setup)?.track,cuboids=scene.ground.cuboids.items;
  const suggestionBox=cuboidBox(cuboids.find(b=>b.track===suggestionTrack));
  const tally={person:{rows:0,blocked:0,unknown:0},driver:{rows:0,blocked:0,unknown:0}};
  const start={placements:0,conflictReportedBlocked:0,conflicts:0};
  const realOverlap={currentOverlapsReal:0,currentOverlapsReplacedSuggestionVehicle:0,whatIfRowsOverlappingReal:0,whatIfRowsOverlappingRealReportedBlocked:0,whatIfRowsOverlappingRealReportedUnknown:0,byCategory:{}};
  const borderline={rows:0,blockedCutInLessThan:0,tooCloseToCall:0,unknownMissByLessThan:0};
  let declaredUnknownWithSuggestionVehicleOnLine=0,declaredUnknown=0,sweepErrors=0,signMismatches=0,stateMismatches=0;
  const unknownReasons={};
  const examples={conflictBlocked:[],realOverlap:[]};
  for(const {input,label} of placements) {
    // Real objects exclude the suggested stationary vehicle: it is the current vehicle itself (annotated pose) or the
    // vehicle a declaration replaced (declared box).
    const sc=createScenario(scene,input),track=sc.baseline.pose==='annotated'?suggestionTrack:null,real=cuboids.filter(b=>b.track!==suggestionTrack);
    start.placements++;
    for(const [which,vis] of [['current',sc.currentVisibility],['proposed',sc.proposedVisibility]]) {
      const c=actorConflicts(sc,which);
      if(c.targetInside||c.eyeInside){start.conflicts++;if(vis.state==='blocked'){start.conflictReportedBlocked++;if(examples.conflictBlocked.length<3)examples.conflictBlocked.push({pixels:pixelsOut(input.pixels),which,label});}}
    }
    if(real.some(b=>boxesOverlap(sc.baseline,cuboidBox(b))))realOverlap.currentOverlapsReal++;
    if(!track&&boxesOverlap(sc.baseline,suggestionBox))realOverlap.currentOverlapsReplacedSuggestionVehicle++;
    if(sc.baseline.pose==='declared'&&sc.currentVisibility.state==='unknown') {
      declaredUnknown++;
      if(appIntersects(sc.eye,sc.target,suggestionBox))declaredUnknownWithSuggestionVehicleOnLine++;
    }
    const vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,Number(input.vehicle[k])]));
    const sweep=setbackSweep(scene,{pixels:input.pixels,driverDistance:input.driverDistance},{label,...vehicle},Number(input.targetHeight));
    if(sweep.error){sweepErrors++;continue;}
    for(const row of sweep.rows) {
      const box=whatIfBox(sc.whatIf,row.setback);
      if(row.state==='overlap') {
        const state=createScenario(scene,{...input,proposedSetback:row.setback}).proposedVisibility.state;
        for(const who of ['person','driver'])if(row[who]){tally[who].rows++;tally[who][state]=(tally[who][state]??0)+1;}
        continue;
      }
      const {clearanceM}=signedClearance(sc.eye,sc.target,box);
      if(row.reason)unknownReasons[row.reason]=(unknownReasons[row.reason]??0)+1;
      // Sign: the app's own signed gap against this file's clearance. State: Blocked only at a cut of 10 cm or more;
      // too close to call only for a smaller cut (1e-9 m for the two searches' rounding).
      if((clearanceM<=0)!==(row.gapM<=0)&&Math.abs(clearanceM)>1e-9)signMismatches++;
      if(row.state==='blocked'&&clearanceM>-BORDERLINE_M+1e-9||row.reason==='too-close-to-call'&&!(clearanceM<=1e-9&&clearanceM>-BORDERLINE_M-1e-9))stateMismatches++;
      borderline.rows++;
      if(row.state==='blocked'&&clearanceM>-BORDERLINE_M)borderline.blockedCutInLessThan++;
      if(row.reason==='too-close-to-call')borderline.tooCloseToCall++;
      if(row.state==='unknown'&&clearanceM>0&&clearanceM<BORDERLINE_M)borderline.unknownMissByLessThan++;
      const hits=real.filter(b=>boxesOverlap(box,cuboidBox(b)));
      if(hits.length) {
        realOverlap.whatIfRowsOverlappingReal++;
        realOverlap[row.state==='blocked'?'whatIfRowsOverlappingRealReportedBlocked':'whatIfRowsOverlappingRealReportedUnknown']++;
        for(const b of hits)realOverlap.byCategory[b.category]=(realOverlap.byCategory[b.category]??0)+1;
        if(examples.realOverlap.length<3)examples.realOverlap.push({pixels:pixelsOut(input.pixels),setbackM:r(row.setback),state:row.state,objects:hits.map(b=>`${b.category}:${b.track.slice(0,8)}`)});
      }
    }
  }
  return {
    definitions:{
      personOrDriverInsideBox:'scenario-assistant boxTests: the person ground-to-head column touches the box, or the eye is inside it (the Distance check and Nemotron facts call this overlap); counted when createScenario reports that same box as Blocked',
      realObjectOverlap:'the current vehicle or the what-if box overlaps a real annotated cuboid other than itself (upright footprints and height ranges). A declared current vehicle replaces the suggested stationary vehicle (decision 12), so overlap with that one vehicle is counted separately',
      borderline:`non-overlap sweep rows with |signed clearance| < ${BORDERLINE_M} m: Blocked rows that cut in by less (0 since M1: they are Unknown, too close to call), rows the app reports too close to call, and Unknown rows that miss by less`,
    },
    sweepRowsWithPersonInsideBox:tally.person,sweepRowsWithDriverInsideBox:tally.driver,
    startValues:start,realObjectOverlap:realOverlap,borderline,
    informational:{declaredCurrentUnknown:declaredUnknown,declaredCurrentUnknownWithSuggestionVehicleOnLine:declaredUnknownWithSuggestionVehicleOnLine,
      note:'today a declared current vehicle replaces the suggested stationary vehicle (decision 12, keep and label); listed only so the count is visible'},
    unknownReasonsInNonOverlapRows:unknownReasons,sweepErrors,clearanceSignMismatches:signMismatches,clearanceStateMismatches:stateMismatches,examples,
  };
}

// ---------------------------------------------------------------- first load
export function firstLoad(scene,setup) {
  const sc=createScenario(scene,setup.input),{clearanceM,t}=signedClearance(sc.eye,sc.target,sc.baseline);
  const p=sc.eye.map((n,i)=>n+(sc.target[i]-n)*t),ground=mappedGroundHeight(scene,p[0],p[2]);
  return {current:sc.currentVisibility.state,proposed:sc.proposedVisibility.state,currentPose:sc.baseline.pose,baselineSetbackM:r(sc.baselineSetback,6),
    clearanceM:r(clearanceM,5),closestFractionAlongSightline:r(t,4),sightlineHeightAboveMappedGroundAtClosestM:ground==null?null:r(p[1]-ground,3),
    limits:Object.fromEntries(Object.entries(sc.limits).map(([k,v])=>[k,r(v,4)])),
    eye:{world:sc.eye.map(n=>r(n,6)),heightAboveMappedGroundM:r(sc.eye[1]-mappedGroundHeight(scene,sc.eye[0],sc.eye[2]),3)},
    targetHeightM:Number(setup.targetHeight),targetGround:sc.targetGround.map(n=>r(n,6))};
}

// ---------------------------------------------------------------- timing (Node only; browser timing is a later task)
// firstLoadMs: the process's first loadDevelopmentScene (disk reads, cold code paths), measured by the caller.
export async function timing(setup,{loads=10,creates=300,sweeps=100,firstLoadMs=null}={}) {
  const ms=f=>{const t=performance.now();f();return performance.now()-t;};
  const cache=new Map();
  let t=performance.now();const {scene}=await loadDevelopmentScene(cache);const cold=performance.now()-t;
  const warm=[];
  for(let i=0;i<loads;i++){t=performance.now();await loadDevelopmentScene(cache);warm.push(performance.now()-t);}
  const create=[],sweep=[],vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,Number(setup.vehicle[k])]));
  for(let i=0;i<creates;i++)create.push(ms(()=>createScenario(scene,setup.input)));
  for(let i=0;i<sweeps;i++)sweep.push(ms(()=>setbackSweep(scene,{pixels:setup.pixels,driverDistance:setup.input.driverDistance},{label:'your vehicle',...vehicle},Number(setup.targetHeight))));
  const ps=a=>({p50:r(percentile(a,.5),3),p95:r(percentile(a,.95),3),n:a.length});
  return {sceneLoadMs:{firstInProcess:r(firstLoadMs,1),emptyByteCacheWarmProcess:r(cold,1),warmBytesInMemory:ps(warm)},createScenarioMs:ps(create),setbackSweepMs:ps(sweep),
    environment:{node:process.version,platform:`${process.platform} ${process.arch}`,cpu:os.cpus()[0]?.model??null,cores:os.cpus().length,memoryGB:r(os.totalmem()/2**30,1)},
    note:'Node only, single process, warm module cache except the cold load; browser load and drag latency are measured separately'};
}

// ---------------------------------------------------------------- reproduction checks against the planning lanes
function reproduction(b) {
  const lane=b.clickToGround.laneBandsForReproduction,checks=[];
  const check=(name,expected,measured,tolerance)=>checks.push({name,expected,measured,tolerance,pass:measured!=null&&Math.abs(measured-expected)<=tolerance});
  check('first-load clearance (m)',.0251,b.firstLoad.clearanceM,.001);
  if(b.placeablePixels)check('placeable crop pixels',107661,b.placeablePixels.placed,0);
  for(const [i,p95,cov] of [[0,.144,1],[1,.881,.954],[2,.713,.98],[3,1.417,1],[4,5.755,.842]]) {
    check(`lane band ${lane[i].band} p95 horizontal error (m)`,p95,lane[i].horizontalErrorM.p95,.01);
    check(`lane band ${lane[i].band} shown-uncertainty coverage`,cov,lane[i].shownUncertaintyCoverage,.002);
  }
  for(const [k,v] of [['crossing',.1437],['vehicleFront',.1767],['person',.2096]])check(`suggestion ${k} residual (m)`,v,b.suggestionConsistency.roles[k].horizontalM,.0005);
  // The lanes counted rows whose line touches the box; since M1 three of those 41 are Unknown, too close to call.
  check('child sweep rows whose line touches the box',41,b.setbackSweep.child.lineTouchesBoxRows,0);
  check('child sweep stopsBlockingAt row (m)',10.603,b.setbackSweep.child.stopsBlockingAtRowM,.001);
  return {source:'planning lanes 2026-10-12 (measure-out.json, coverage.mjs, budget-out.json, app-contract.md section 8)',checks,allPass:checks.every(c=>c.pass)};
}

function provenance() {
  const file=p=>{try{return sha(readFileSync(new URL(p,ROOT)));}catch{return null;}};
  let head=null,dirty=null;
  try{head=execFileSync('git',['-C',fileURLToPath(ROOT),'rev-parse','HEAD'],{encoding:'utf8'}).trim();
    dirty=execFileSync('git',['-C',fileURLToPath(ROOT),'status','--porcelain','--','src','scripts/evaluation/pao-baseline.mjs'],{encoding:'utf8'}).trim().split('\n').filter(Boolean);}catch{}
  return {repositoryHead:head,uncommittedPaths:dirty,
    sha256:Object.fromEntries(['scripts/evaluation/pao-baseline.mjs','src/geometry.mjs','src/scene.mjs','src/scenario-assistant.mjs','src/suggestions.mjs'].map(p=>[p,file(p)]))};
}

// The whole baseline. coverage: run every crop pixel (about 4 s); samples: random placements for honesty counters.
export async function computeBaseline({coverage=true,samples=3000,timingRuns}={}) {
  const t0=performance.now(),{scene,entry}=await loadDevelopmentScene(),firstLoadMs=performance.now()-t0,setup=suggestedSetup(scene,scene.suggestions);
  if(setup.error||!setup.input)throw new Error(`Suggested setup unavailable: ${setup.error}`);
  const sc=createScenario(scene,setup.input);
  const lock=existsSync(new URL('split.lock.json',OUT_DIR))?JSON.parse(await readFile(new URL('split.lock.json',OUT_DIR),'utf8')):null;
  const splitBytes=existsSync(new URL('split.json',OUT_DIR))?await readFile(new URL('split.json',OUT_DIR)):null;
  const b={schemaVersion:1,kind:'pao-accuracy-baseline',generatedUtc:new Date().toISOString(),
    label:'Development scene only (8c52d911 / crossing 2658849): development-grade, AV2-referenced, not surveyed, not held-out. Never a passed gate.',
    provenance:provenance(),
    split:{file:'split.json',lockSha256:lock?.sha256??null,fileSha256:splitBytes?sha(splitBytes):null,matchesLock:!!(lock&&splitBytes&&sha(splitBytes)===lock.sha256)},
    scene:{id:DEV.id,metadataSha256:entry.metadataSha256,ground:entry.ground,suggestions:entry.suggestions},
    conventions:{
      driverEye:`recorded stereo_front_left camera centre (world origin [0, 0, 0]), ${r(sc.eye[1]-mappedGroundHeight(scene,0,0),3)} m above the mapped ground under it; driverDistance = limits.captureDistance (${r(sc.limits.captureDistance,4)} m from the crossing point along the lane axis). Camera height, not a seated driver eye (decision: camera eye, labeled).`,
      target:`suggested person point; child ${setup.targetHeight} m (suggestion default) and adult ${ADULT_HEIGHT_M} m`,
      percentile:'nearest-rank: sorted[ceil(p n) - 1]',
      distances:'metres; horizontal means world X/Z',
      results:'Blocked or Unknown only (Clear removed); overlap is the sweep and facts label for the Unknown reason person or driver inside the vehicle box; other Unknown reasons (M1) are row reasons',
    },
    firstLoad:firstLoad(scene,setup),
    clickToGround:clickToGround(scene),
    suggestionConsistency:suggestionConsistency(scene),
    setbackSweep:setbackSweepBaseline(scene,setup),
    blockedEnd:blockedEndRange(scene,setup),
  };
  let list=null;
  if(coverage){const p=placeablePixels(scene);list=p.list;b.placeablePixels=p.summary;}
  if(samples>0) {
    const sample=samplePlacements(scene,{count:samples});
    b.honesty={sample:{count:sample.placements.length,attempts:sample.attempts,generator:sample.generator,rejected:sample.errors},
      ...honestyCounters(scene,setup,sample.placements)};
  }
  b.timing=await timing(setup,{...timingRuns,firstLoadMs});
  b.reproduction=reproduction(b);
  return b;
}

async function main(argv) {
  const arg=k=>{const i=argv.indexOf(k);return i>=0?argv[i+1]:undefined;};
  if(!devDataPresent())throw new Error('AV2 8c development data is local-only and not present.');
  const date=arg('--date')??new Date().toISOString().slice(0,10);
  if(!/^\d{4}-\d{2}-\d{2}$/.test(date))throw new Error('--date must be YYYY-MM-DD');
  const out=arg('--out')?pathToFileURL(arg('--out').replace(/\/?$/,'/')):OUT_DIR,samples=arg('--samples')?Number(arg('--samples')):3000;
  const b=await computeBaseline({coverage:!argv.includes('--no-coverage'),samples});
  const text=JSON.stringify(b,null,1)+'\n';
  if(/park/i.test(text))throw new Error('Baseline text uses banned vehicle-status wording.');
  await mkdir(out,{recursive:true});
  const target=new URL(`baseline-${date}.json`,out);
  await writeFile(target,text);
  console.log(JSON.stringify({wrote:fileURLToPath(target),sha256:sha(text),reproductionAllPass:b.reproduction.allPass}));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)await main(process.argv.slice(2));

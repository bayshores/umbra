#!/usr/bin/env node
// Metric F (false Blocked caused by the box shape), app side. Exports the app's own setback-sweep rows for a scene, with
// each row's eye and person-head point in the car-local frame of that row's moved box, so the LiDAR car-top scorer
// (scripts/evaluation/metric_f_car_top.py) can compare the line with the measured car top. Row states and reasons come
// from the app (setbackSweep -> assessSightline / classifySightline); nothing here decides Blocked or Unknown.
// Report only. Results are Blocked or Unknown; there is no Clear.
//
//   node scripts/evaluation/metric-f.mjs --assembled DIR --grounded DIR --protocol FILE --protocol-lock FILE --scene NAME
//        --out FILE [--app-src DIR] [--check-baseline FILE] [--eval-shape FILE]
//
// --app-src: a folder holding geometry.mjs, scene.mjs, scenario-assistant.mjs and suggestions.mjs of another app version
// (for example src/ of an earlier commit copied with git show); default: this repository's src/.
// --check-baseline: a pao-baseline JSON whose setbackSweep.child and .adult rows these rows must reproduce (setbacks,
// states and clearances to 1e-4 m), else the run stops.
// --eval-shape: metric F shape-v1 approach-half split only (split-addendum.json; report only). An evaluation-only shape
// file written by scripts/california/car_shape.py approach-split for this grounded scene; the app's own car shape of
// the loaded scene is replaced by it (built from the far half of the sweeps) before any row is made. Never used by the app.
//
// Metric F shape-v1 (protocol v3): when the tested vehicle carries its LiDAR shape (ground.json carShape, placement
// protocol v3.3), each row also gets the box result (the same row with the shape removed from the box: the result before
// shape-v1) and the shape diagnostics (solidCutM, hitXM), and every row is checked against the shape-v1 allowed changes
// (protocol resultRule.allowedChanges) on the fast and the full path. A violation is recorded (shapeV1.allowedChanges),
// never hidden; a row whose app result differs from its recomputed shape result stops the run.
import {readFile,writeFile} from 'node:fs/promises';
import {isDeepStrictEqual} from 'node:util';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import * as geometry from '../../src/geometry.mjs';
import * as assistant from '../../src/scenario-assistant.mjs';
import * as suggestions from '../../src/suggestions.mjs';
import * as sceneModule from '../../src/scene.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const GROUNDED=new Set(['scene.json','ground.json','terrain.bin','suggestions.json']);
export const APP_FILES=['geometry.mjs','scene.mjs','scenario-assistant.mjs','suggestions.mjs'];
export const TARGET_HEIGHTS_M=Object.freeze({child:1.1,adult:1.5});

// The functions this file uses from an app version. CURRENT_APP is this repository's src/.
const pick=(g,a,s,sc)=>({createScenario:g.createScenario,whatIfBox:g.whatIfBox,mappedGroundHeight:g.mappedGroundHeight,
  assessSightline:g.assessSightline,prepareCarShape:g.prepareCarShape,SHAPE_REASONS:g.SHAPE_REASONS,
  setbackSweep:a.setbackSweep,suggestedSetup:s.suggestedSetup,loadScene:sc.loadScene});
export const CURRENT_APP=Object.freeze(pick(geometry,assistant,suggestions,sceneModule));
// Functions an earlier app version may lack (the v3.2 app of commit c7a39eb has no car shape).
const OPTIONAL=new Set(['mappedGroundHeight','assessSightline','prepareCarShape','SHAPE_REASONS']);
// Another app version from a folder with the four module files (they import each other relatively).
export async function loadApp(dir) {
  const url=f=>pathToFileURL(path.join(path.resolve(dir),f)).href;
  // Same order as APP_FILES: geometry, scene, scenario-assistant, suggestions.
  const [g,sc,a,s]=await Promise.all(APP_FILES.map(f=>import(url(f))));
  const app=pick(g,a,s,sc);
  for(const [k,v] of Object.entries(app))if(!OPTIONAL.has(k)&&typeof v!=='function')throw new Error(`${dir} has no ${k}`);
  return app;
}

// Car-local coordinates of a world point for an upright app box {center, axis, vehicle}: x forward along the heading
// (the app's axis points toward the camera, so heading = -axis), y left of the heading, z above the box bottom (world Y
// minus centre Y plus half the height). The app's own box frame (assessSightline) is [d·side, d.y, d·axis] with
// side = [axis.z, 0, -axis.x], which is minus left.
export function carLocal(box,p) {
  const h=[-box.axis[0],0,-box.axis[2]],left=[h[2],0,-h[0]],d=sub(p,box.center);
  return [dot(d,h),dot(d,left),d[1]+box.vehicle.height/2];
}
// The app's box-local coordinates [side, y from centre, along axis] of a car-local point.
export const appLocalFromCar=(q,vehicle)=>[-q[1],q[2]-vehicle.height/2,-q[0]];
const halfSizes=v=>[v.width/2,v.height/2,v.length/2];

// shape-v1 resultRule.allowedChanges for one result (protocol v3 metric F shape-v1; the same table as
// scene-builder-v4.2/check_allowed_changes_8_2.mjs). box: the result with the shape removed from the box; shape: the
// result with it. Returns null when the change is allowed, else what is wrong. A box result that is not Blocked must stay
// the identical object; a box Blocked may become the shape Blocked sentence or Unknown height-not-proven, gapM unchanged.
export function allowedChange(box,shape,{diagnostics,reasons=geometry.SHAPE_REASONS}) {
  if(box.state!=='blocked')return isDeepStrictEqual(box,shape)?null:'box result is not Blocked, but the shape result is not the identical object';
  const keys=Object.keys(shape).sort(),extra=diagnostics?['shape']:[];
  if(shape.gapM!==box.gapM)return 'gapM changed';
  if(diagnostics&&!(shape.shape&&typeof shape.shape.status==='string'&&'solidCutM' in shape.shape&&'hitXM' in shape.shape))return 'diagnostics shape record missing';
  if(shape.state==='blocked') {
    if(!isDeepStrictEqual(keys,['gapM','reason','state','supportedFraction',...extra].sort()))return `shape Blocked has fields ${keys}`;
    return shape.reason===reasons.blocked&&shape.supportedFraction===null?null:'shape Blocked has the wrong sentence or supportedFraction';
  }
  if(shape.state==='unknown'&&shape.reasonCode==='height-not-proven') {
    if(!isDeepStrictEqual(keys,['gapM','reason','reasonCode','state','supportedFraction',...extra].sort()))return `height-not-proven has fields ${keys}`;
    const ok=shape.reason===reasons.heightNotProvenMeasured||/^The sightline cuts into the recorded stationary vehicle's box, but no LiDAR shape could be built for it \(.+\), so this placement is not decided\.$/.test(shape.reason);
    return ok&&shape.supportedFraction===null?null:'height-not-proven has the wrong sentence or supportedFraction';
  }
  return `box Blocked became ${shape.state} ${shape.reasonCode??''}`;
}
const INSIDE=new Set(['person-inside-vehicle','driver-inside-vehicle']);
const factState=r=>INSIDE.has(r.reasonCode)?'overlap':r.state;
const withoutShape=({shape,...box})=>box;
export const ALLOWED_CHANGES_RULE='shape-v1 resultRule.allowedChanges on every setback-sweep row, fast path (diagnostics off) and full path (diagnostics on): '+
  'a row the box does not call Blocked is the identical object; a box Blocked row becomes the shape Blocked sentence (supportedFraction null) or Unknown '+
  'height-not-proven, with gapM unchanged; no row is Blocked with the shape and not with the box';

// The app's setback-sweep rows for each target height, built exactly as scripts/evaluation/pao-baseline.mjs builds them
// (sweepRows): createScenario on the suggested input at that target height for the eye and target, setbackSweep for the
// rows and whatIfBox for each row's box. gapM is the app's own (row.gapM); an app version without it gets the current
// sightlineGap (a pure geometric function) on the same box-local points. app: functions as in CURRENT_APP.
export function metricFRows(scene,setup,{heights=TARGET_HEIGHTS_M,app=CURRENT_APP,gap=geometry.sightlineGap}={}) {
  if(!setup?.input)return {error:setup?.error||'No suggested setup.'};
  const track=scene.suggestions?.vehicle?.status==='annotated'?scene.suggestions.vehicle.track:null;
  if(!track)return {error:'Metric F applies to an annotated stationary vehicle only; this scene has none.'};
  const vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,Number(setup.vehicle[k])]));
  const out={track,suggestedTargetHeightM:Number(setup.targetHeight),heights:{}};
  const violations=[];let checked=0,shape=null;
  for(const [name,height] of Object.entries(heights)) {
    const input={...setup.input,targetHeight:String(height)},sc=app.createScenario(scene,input),w=sc.whatIf;
    if(w.pose!=='annotated')throw new Error('The suggested vehicle is not swept in its annotated pose.');
    const sweep=app.setbackSweep(scene,{pixels:setup.pixels,driverDistance:input.driverDistance},{label:'your vehicle',...vehicle},height);
    if(sweep.error){out.heights[name]={targetHeightM:height,error:sweep.error};continue;}
    const own=app.whatIfBox(w,w.setback);
    out.testedBox??={center:own.center,axisToCamera:own.axis,heading:[-own.axis[0],0,-own.axis[2]],vehicle:{...own.vehicle},setbackM:w.setback,laneAxis:w.axis};
    if(w.shape)shape??=w.shape;
    const rows=sweep.rows.map(row=>{
      const box=app.whatIfBox(w,row.setback),eye=carLocal(box,sc.eye),target=carLocal(box,sc.target),targetGround=carLocal(box,sc.targetGround);
      const appGap=row.gapM!==undefined,g=appGap?row.gapM:gap(appLocalFromCar(eye,box.vehicle),appLocalFromCar(target,box.vehicle),halfSizes(box.vehicle));
      const groundY=app.mappedGroundHeight?app.mappedGroundHeight(scene,box.center[0],box.center[2]):undefined;
      const out={setbackM:row.setback,state:row.state,...(row.reason&&{reason:row.reason}),gapM:g,gapSource:appGap?'app':'current sightlineGap',
        eyeCar:eye,targetCar:target,targetGroundCar:targetGround,boxCenter:box.center,
        // Diagnostic: mapped ground under the moved box centre minus the box bottom (the box is not re-grounded).
        mappedGroundMinusBoxBottomM:groundY==null?null:groundY-(box.center[1]-box.vehicle.height/2)};
      if(!box.shape)return out;
      // shape-v1: the row's own result recomputed with and without the shape, on both paths (setbackSweep's ownPosition).
      if(typeof app.assessSightline!=='function')throw new Error('The vehicle box has a car shape, but the app has no assessSightline.');
      const own=row.setback===sc.baselineSetback,r={};
      for(const diagnostics of [false,true]) {
        const opt={diagnostics,ownPosition:own},b=app.assessSightline(scene,sc,withoutShape(box),opt),s=app.assessSightline(scene,sc,box,opt);
        const why=allowedChange(b,s,{diagnostics,reasons:app.SHAPE_REASONS});checked++;
        if(why)violations.push({height:name,setbackM:row.setback,path:diagnostics?'full':'fast',what:why});
        if(s.state==='blocked'&&b.state!=='blocked')violations.push({height:name,setbackM:row.setback,path:diagnostics?'full':'fast',what:'new Blocked'});
        r[diagnostics?'full':'fast']={b,s};
      }
      const {fast,full}=r;
      if(row.state!==factState(fast.s)||(row.reason??null)!==(fast.s.reasonCode??null)||row.gapM!==fast.s.gapM)
        throw new Error(`${name} ${row.setback}: the app sweep row differs from its recomputed shape result`);
      if(fast.s.state!==full.s.state||fast.s.reasonCode!==full.s.reasonCode)
        violations.push({height:name,setbackM:row.setback,path:'both',what:'fast and full shape results differ'});
      out.box={state:factState(fast.b),reason:fast.b.reasonCode??null,gapM:fast.b.gapM};
      out.shape=full.s.shape?{status:full.s.shape.status,solidCutM:full.s.shape.solidCutM,hitXM:full.s.shape.hitXM}:null;
      return out;
    });
    out.heights[name]={targetHeightM:height,eye:sc.eye,target:sc.target,targetGround:sc.targetGround,rows,step:sweep.step,range:sweep.range,
      stopsBlockingAt:sweep.stopsBlockingAt??null,notBlockedFrom:sweep.notBlockedFrom??null,at15ft:sweep.at15ft??null,at20ft:sweep.at20ft??null};
  }
  if(shape) {
    const all=Object.values(out.heights).flatMap(h=>h.rows??[]);
    if(!all.every(r=>r.box))throw new Error('Some rows have no box result although the vehicle has a car shape.');
    out.shapeV1={carShape:{schema:shape.schema,status:shape.status,refusal:shape.refusal,track:shape.track,allTopsAtLeastH:shape.allTopsAtLeastH,
      along:shape.along,lateral:shape.lateral,box:{length:shape.length,width:shape.width,height:shape.height}},
      allowedChanges:{rule:ALLOWED_CHANGES_RULE,violations,pass:violations.length===0,resultsChecked:checked}};
  }
  return out;
}

// Serves a scene like scene-builder-v4.1/harness-v4.1.mjs: assembled buffers plus every grounded file (catalog entry from
// the grounded folder). loadScene: the app version's loader.
export async function loadSceneFrom({assembled,grounded},loadScene=CURRENT_APP.loadScene) {
  if(typeof loadScene!=='function')throw new Error('No scene loader.');
  const entry=JSON.parse(await readFile(path.join(grounded,'catalog-entry.json'),'utf8')),id=entry.id;
  const fetch=async p=>{
    const name=p.split('/').at(-1);
    if(!p.startsWith(`/scenes/${id}/`)||['catalog-entry.json','LICENSE-AND-ATTRIBUTION.md'].includes(name))return new Response(null,{status:404});
    try{return new Response(await readFile(path.join(GROUNDED.has(name)?grounded:assembled,name)));}
    catch{return new Response(null,{status:404});}
  };
  return {scene:await loadScene(fetch,id,entry.metadataSha256,undefined,entry.suggestions,entry.ground),entry};
}

// Rows against a pao-baseline JSON (setbackSweep.<name>.rows: setbackM, state, clearanceM rounded to 1e-4).
export function checkAgainstBaseline(result,baseline) {
  const problems=[];
  for(const name of Object.keys(result.heights)) {
    const ref=baseline.setbackSweep?.[name]?.rows,rows=result.heights[name].rows;
    if(!ref){problems.push(`${name}: no baseline rows`);continue;}
    if(ref.length!==rows.length){problems.push(`${name}: ${rows.length} rows, baseline ${ref.length}`);continue;}
    rows.forEach((row,i)=>{
      const b=ref[i];
      if(Math.abs(row.setbackM-b.setbackM)>1e-4+1e-9)problems.push(`${name} row ${i}: setback ${row.setbackM} vs ${b.setbackM}`);
      if(row.state!==b.state)problems.push(`${name} row ${i}: state ${row.state} vs ${b.state}`);
      if(b.clearanceM!=null&&Math.abs(row.gapM-b.clearanceM)>1e-4+1e-9)problems.push(`${name} row ${i}: gap ${row.gapM} vs ${b.clearanceM}`);
    });
  }
  return problems;
}

function args(argv) {
  const out={};
  for(let i=0;i<argv.length;i+=2) {
    if(!argv[i].startsWith('--')||argv[i+1]===undefined)throw new Error(`Bad argument ${argv[i]}`);
    out[argv[i].slice(2)]=argv[i+1];
  }
  for(const k of ['assembled','grounded','protocol','protocol-lock','scene','out'])if(!out[k])throw new Error(`Missing --${k}`);
  return out;
}

async function main() {
  const a=args(process.argv.slice(2));
  const protocolBytes=await readFile(a.protocol),lock=JSON.parse(await readFile(a['protocol-lock'],'utf8'));
  if(sha(protocolBytes)!==lock.sha256)throw new Error('protocol.json changed after its lock');
  const protocol=JSON.parse(protocolBytes),heights=protocol.thresholds.targetHeightsM;
  const appDir=a['app-src']??fileURLToPath(new URL('../../src/',import.meta.url)),app=a['app-src']?await loadApp(appDir):CURRENT_APP;
  const appHashes=Object.fromEntries(await Promise.all(APP_FILES.map(async f=>[f,sha(await readFile(path.join(appDir,f)))])));
  const {scene,entry}=await loadSceneFrom({assembled:a.assembled,grounded:a.grounded},app.loadScene);
  const groundSha=sha(await readFile(path.join(a.grounded,'ground.json')));
  let evalShape=null;
  if(a['eval-shape']) {
    // Approach-half split (report only): the evaluation-only shape replaces the app's car shape of this loaded scene.
    const bytes=await readFile(a['eval-shape']),doc=JSON.parse(bytes);
    if(doc.schema!=='shape-v1-approach-split')throw new Error('--eval-shape is not an approach-split shape file.');
    if(doc.scene?.ground?.sha256!==groundSha)throw new Error('--eval-shape was made from another ground.json.');
    if(!scene.carShape||doc.carShape?.track!==scene.carShape.track)throw new Error('--eval-shape names another vehicle than the scene car shape.');
    if(typeof app.prepareCarShape!=='function')throw new Error('The app has no prepareCarShape.');
    scene.carShape=app.prepareCarShape(doc.carShape);
    if(scene.carShape.length!==doc.carShape.box.length)throw new Error('The evaluation shape was not applied.');
    evalShape={file:path.basename(a['eval-shape']),sha256:sha(bytes),status:doc.carShape.status,K:doc.approachSplit?.K,h:doc.approachSplit?.h};
  }
  const setup=app.suggestedSetup(scene,scene.suggestions),result=metricFRows(scene,setup,{heights,app});
  if(result.error)throw new Error(result.error);
  if(result.shapeV1) {
    Object.assign(result.shapeV1.carShape,evalShape?{source:evalShape.file,sourceSha256:evalShape.sha256}:{source:'ground.json',sourceSha256:groundSha});
    if(evalShape)result.shapeV1.evalShape=evalShape;
  }else if(evalShape)throw new Error('--eval-shape given, but the rows carry no car shape.');
  if(a['check-baseline']) {
    const problems=checkAgainstBaseline(result,JSON.parse(await readFile(a['check-baseline'],'utf8')));
    if(problems.length)throw new Error(`Rows do not reproduce the baseline:\n${problems.slice(0,20).join('\n')}`);
    result.baselineCheck={file:path.basename(a['check-baseline']),sha256:sha(await readFile(a['check-baseline'])),rule:'setbacks, states and clearances to 1e-4 m',pass:true};
  }
  const doc={kind:'metric-f app rows',scene:a.scene,sceneId:entry.id,protocolSha256:lock.sha256,
    frames:{world:'ground.json frame (X = cityX - originX, Y = cityZ - originZ up, Z = -(cityY - originY))',
      car:'x forward along the tested box heading, y left, z above the box bottom (box centre Y - H/2; the moved box keeps its centre Y)'},
    app:{dir:a['app-src']?'--app-src':'src/',sha256:appHashes},groundSha256:groundSha,catalogEntry:entry,...result};
  const text=JSON.stringify(doc,null,1)+'\n';
  if(text.toLowerCase().includes('park'))throw new Error('Output text contains a refused word.');
  await writeFile(a.out,text,{flag:'wx'});
  console.log(sha(Buffer.from(text)));
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error(e.message);process.exit(1);});

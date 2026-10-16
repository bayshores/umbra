#!/usr/bin/env node
// Metric E (result at the recorded view vs a blind photo outline), app side. Exports the app's own result for a fixed grid
// of person-head points with the recorded vehicle at its recorded pose and the eye at the recorded camera, so the D/E
// scorer can compare each result with the head point's pixel against the blind labels of that vehicle in the recorded
// photo. With the eye at the camera, the eye-to-head line is the photo ray through that pixel. Results come from the app
// (assessSightline, the same call as createScenario's currentVisibility); nothing here decides Blocked or Unknown.
// Report only. Results are Blocked or Unknown; there is no Clear.
//
//   node scripts/evaluation/metric-e-grid.mjs --assembled DIR --grounded DIR --protocol FILE --protocol-lock FILE --scene NAME
//        --out FILE
//
// The grid uses the loaded scene alone (never labels or a photo outline): see E_GRID_RULE.
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {APP_FILES,CURRENT_APP,loadSceneFrom} from './metric-f.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const DEG=Math.PI/180;
export const E_GRID=Object.freeze({stepM:.25,beyondM:.5,depthM:10,widenDeg:2});
export const E_GRID_RULE='Ground points on a 0.25 m grid of world X and Z (multiples of 0.25 m) that have mapped ground under them, '+
  'whose horizontal distance from the eye is from the farthest corner of the recorded vehicle\'s box footprint plus 0.5 m to '+
  'that plus 10 m, and whose horizontal bearing from the eye lies inside the bearings of the four footprint corners widened '+
  'by 2 degrees on each side. Each point is tested at every target height, with the head that height above the mapped '+
  'ground. The eye is the recorded camera (the suggested setup at the full captured distance), the box is the recorded '+
  'vehicle at its recorded pose with its car shape (the current vehicle of the suggested setup), and each result is '+
  'assessSightline(scene, {eye, target, targetGround}, box, {ownPosition: true}).';

// Footprint corners [x, z] of an upright app box {center, axis, vehicle} (axis: unit, horizontal, toward the camera).
export function footprint(box) {
  const a=box.axis,side=[a[2],0,-a[0]],hl=box.vehicle.length/2,hw=box.vehicle.width/2;
  return [[1,1],[1,-1],[-1,1],[-1,-1]].map(([s,t])=>[box.center[0]+a[0]*hl*s+side[0]*hw*t,box.center[2]+a[2]*hl*s+side[2]*hw*t]);
}
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));

// The grid's ground points (x, z) for an eye and a box, nearest first (then by x). groundHeight(x, z): mapped ground or null.
// Returns the kept points with their ground height, the distance and bearing limits and the count dropped for no ground.
export function eGridPoints(eye,box,groundHeight,rule=E_GRID) {
  const corners=footprint(box),dist=([x,z])=>Math.hypot(x-eye[0],z-eye[2]);
  const near=Math.max(...corners.map(dist))+rule.beyondM,far=near+rule.depthM;
  const centre=Math.atan2(box.center[0]-eye[0],box.center[2]-eye[2]);
  const offsets=corners.map(([x,z])=>wrap(Math.atan2(x-eye[0],z-eye[2])-centre));
  if(Math.max(...offsets)-Math.min(...offsets)>=Math.PI)throw new Error('The eye is inside the box footprint.');
  const lo=Math.min(...offsets)-rule.widenDeg*DEG,hi=Math.max(...offsets)+rule.widenDeg*DEG,s=rule.stepM;
  const points=[];let noGround=0;
  for(let i=Math.floor((eye[0]-far)/s);i<=Math.ceil((eye[0]+far)/s);i++) {
    for(let k=Math.floor((eye[2]-far)/s);k<=Math.ceil((eye[2]+far)/s);k++) {
      const x=i*s,z=k*s,d=Math.hypot(x-eye[0],z-eye[2]);
      if(d<near||d>far)continue;
      const b=wrap(Math.atan2(x-eye[0],z-eye[2])-centre);
      if(b<lo||b>hi)continue;
      const g=groundHeight(x,z);
      if(g===null||g===undefined||!Number.isFinite(g)){noGround++;continue;}
      points.push({x,z,g,d});
    }
  }
  points.sort((p,q)=>p.d-q.d||p.x-q.x||p.z-q.z);
  return {points,noGround,limits:{nearM:near,farM:far,bearingFromBoxCentreDeg:[lo/DEG,hi/DEG]}};
}

// The app's results on the grid. app: functions as in CURRENT_APP. heights: {name: metres}.
export function metricEGrid(scene,setup,{heights,app=CURRENT_APP,rule=E_GRID}={}) {
  if(!setup?.input)return {error:setup?.error||'No suggested setup.'};
  const track=scene.suggestions?.vehicle?.status==='annotated'?scene.suggestions.vehicle.track:null;
  if(!track)return {error:'Metric E applies to an annotated recorded vehicle only; this scene has none.'};
  const sc=app.createScenario(scene,setup.input),w=sc.whatIf;
  if(w.pose!=='annotated')return {error:'The suggested vehicle is not the recorded vehicle in its annotated pose.'};
  if(sc.driverDistance!==sc.limits.captureDistance||sc.eye.some((n,i)=>n!==scene.camera.eye[i]))
    return {error:'The suggested eye is not the recorded camera.'};
  const box=sc.baseline;
  if(sc.baselineSetback!==w.setback)throw new Error('The current vehicle is not at its own setback.');
  const {points,noGround,limits}=eGridPoints(sc.eye,box,(x,z)=>app.mappedGroundHeight(scene,x,z),rule);
  const rows=[];
  for(const [name,height] of Object.entries(heights)) {
    for(const p of points) {
      const targetGround=[p.x,p.g,p.z],target=[p.x,p.g+height,p.z];
      const r=app.assessSightline(scene,{eye:sc.eye,target,targetGround},box,{ownPosition:true});
      rows.push({height:name,xM:p.x,zM:p.z,target,targetGround,state:r.state,reasonCode:r.reasonCode??null,gapM:r.gapM});
    }
  }
  return {track,eye:sc.eye,rule:{text:E_GRID_RULE,...rule},limits,targetHeightsM:{...heights},
    box:{center:box.center,axisToCamera:box.axis,vehicle:{...box.vehicle},setbackM:sc.baselineSetback,
      shape:box.shape?{schema:box.shape.schema,status:box.shape.status,track:box.shape.track}:null},
    counts:{points:points.length,droppedNoGround:noGround,rows:rows.length,
      byState:rows.reduce((c,r)=>(c[`${r.height}/${r.state}`]=(c[`${r.height}/${r.state}`]??0)+1,c),{})},rows};
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
  const heights=JSON.parse(protocolBytes).thresholds.targetHeightsM;
  const appDir=fileURLToPath(new URL('../../src/',import.meta.url));
  const appHashes=Object.fromEntries(await Promise.all(APP_FILES.map(async f=>[f,sha(await readFile(path.join(appDir,f)))])));
  const {scene,entry}=await loadSceneFrom({assembled:a.assembled,grounded:a.grounded});
  const groundSha=sha(await readFile(path.join(a.grounded,'ground.json')));
  const result=metricEGrid(scene,CURRENT_APP.suggestedSetup(scene,scene.suggestions),{heights});
  if(result.error)throw new Error(result.error);
  const doc={kind:'pao-m7-e-grid',scene:a.scene,sceneId:entry.id,protocolSha256:lock.sha256,
    frames:{world:'ground.json frame (X = cityX - originX, Y = cityZ - originZ up, Z = -(cityY - originY)); the recorded left camera at the origin'},
    app:{dir:'src/',sha256:appHashes},groundSha256:groundSha,...result};
  const text=JSON.stringify(doc,null,1)+'\n';
  if(text.toLowerCase().includes('park'))throw new Error('Output text contains a refused word.');
  await writeFile(a.out,text,{flag:'wx'});
  console.log(sha(Buffer.from(text)));
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error(e.message);process.exit(1);});

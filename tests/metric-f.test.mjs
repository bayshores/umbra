import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {whatIfBox,sightlineGap,TOO_CLOSE_TO_CALL_M} from '../src/geometry.mjs';
import {suggestedSetup} from '../src/suggestions.mjs';
import {carLocal,appLocalFromCar,metricFRows,loadSceneFrom,checkAgainstBaseline,loadApp,CURRENT_APP,TARGET_HEIGHTS_M} from '../scripts/evaluation/metric-f.mjs';
import {devDataPresent,DEV,GROUNDED_DIR} from '../scripts/evaluation/pao-baseline.mjs';

const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);
const sub=(a,b)=>a.map((n,i)=>n-b[i]),dot=(a,b)=>a.reduce((s,n,i)=>s+n*b[i],0);
// assessSightline's own box-local frame.
const appLocal=(box,p)=>{const side=[box.axis[2],0,-box.axis[0]],d=sub(p,box.center);return [dot(d,side),d[1],dot(d,box.axis)];};

// ------------------------------------------------------------------ synthetic (no local data)
test('metric F frame: car-local points map back to the app box frame, and the gap is unchanged',()=>{
  const a=Math.PI*.3,box={center:[3,-.4,12],axis:[-Math.cos(a),0,-Math.sin(a)],vehicle:{width:1.9,height:1.5,length:4.2}};
  // heading = -axis; a point 1 m ahead of the centre, 0.5 m to the left, 0.2 m above the centre height.
  const h=[Math.cos(a),0,Math.sin(a)],left=[h[2],0,-h[0]],p=box.center.map((c,i)=>c+h[i]+.5*left[i]+(i===1?.2:0));
  const q=carLocal(box,p);
  near(q[0],1,1e-12,'forward');near(q[1],.5,1e-12,'left');near(q[2],.2+.75,1e-12,'above bottom');
  appLocalFromCar(q,box.vehicle).forEach((n,i)=>near(n,appLocal(box,p)[i],1e-12,`app axis ${i}`));
  const half=[.95,.75,2.1];
  for(let k=0;k<20;k++) {
    const e=[-8+k*.3,1.7,-1+k*.1].map((n,i)=>n+box.center[i]),t=[6,1.1-k*.02,2].map((n,i)=>n+box.center[i]);
    near(sightlineGap(appLocalFromCar(carLocal(box,e),box.vehicle),appLocalFromCar(carLocal(box,t),box.vehicle),half),
      sightlineGap(appLocal(box,e),appLocal(box,t),half),1e-9,`gap ${k}`);
  }
});

test('metric F rows: app rows pass through unchanged; an app without gapM gets the geometric gap',()=>{
  const vehicle={width:2,height:1.5,length:4},w={pose:'annotated',center:[0,.75,10],setback:1,axis:[0,0,-1],boxAxis:[0,0,-1],vehicle};
  const scene={suggestions:{vehicle:{status:'annotated',track:'t-1'}}};
  const setup={input:{pixels:{},vehicle:{width:'2',height:'1.5',length:'4'},targetHeight:'1.1'},pixels:{},vehicle:{width:'2',height:'1.5',length:'4'},targetHeight:'1.1'};
  const rows=[{setback:1,state:'blocked',reason:undefined},{setback:1.25,state:'unknown',reason:'too-close-to-call',gapM:-.04}];
  const app={createScenario:(s,input)=>({eye:[0,1.6,-10],target:[0,Number(input.targetHeight),20],targetGround:[0,0,20],whatIf:w}),
    whatIfBox,setbackSweep:()=>({rows,step:.25,range:[1,1.25],stopsBlockingAt:null,notBlockedFrom:null})};
  const out=metricFRows(scene,setup,{heights:{child:1.1},app});
  assert.equal(out.track,'t-1');
  assert.deepEqual(out.testedBox.heading,[-0,0,1]);
  const [r0,r1]=out.heights.child.rows;
  // Box centre (0, 0.75, 10), heading +Z: the eye 20 m behind the centre, 1.6 m above the box bottom (Y 0).
  near(r0.eyeCar[0],-20,1e-12);near(r0.eyeCar[1],0,1e-12);near(r0.eyeCar[2],1.6,1e-12);
  assert.equal(r0.gapSource,'current sightlineGap');
  near(r0.gapM,sightlineGap(appLocal(whatIfBox(w,1),[0,1.6,-10]),appLocal(whatIfBox(w,1),[0,1.1,20]),[1,.75,2]),1e-12);
  assert.ok(r0.gapM<0,'the level line cuts the box');
  // Row 2: the box moved 0.25 m further along the lane axis (toward -Z, the camera); app gap kept.
  assert.equal(r1.gapSource,'app');assert.equal(r1.gapM,-.04);assert.equal(r1.reason,'too-close-to-call');
  near(r1.eyeCar[0],-19.75,1e-12);
  assert.deepEqual(checkAgainstBaseline(out,{setbackSweep:{child:{rows:[{setbackM:1,state:'blocked',clearanceM:Number(r0.gapM.toFixed(4))},{setbackM:1.25,state:'unknown',clearanceM:-.04}]}}}),[]);
  assert.equal(checkAgainstBaseline(out,{setbackSweep:{child:{rows:[{setbackM:1,state:'unknown'},{setbackM:1.25,state:'unknown'}]}}}).length,1);
  assert.match(metricFRows({suggestions:{vehicle:{status:'unknown'}}},setup,{app}).error,/annotated stationary vehicle/);
});

test('metric F loadApp: an app folder gives each function from its own module',async()=>{
  const app=await loadApp(fileURLToPath(new URL('../src/',import.meta.url)));
  for(const k of Object.keys(CURRENT_APP))assert.equal(app[k],CURRENT_APP[k],k);
});

// ------------------------------------------------------------------ AV2 8c development scene (local only)
const skip=devDataPresent()?false:'AV2 8c development data is local-only and not present';
test('metric F rows on AV2 8c (protocol v3.2): the app box, car-local points and gaps agree with the app',{skip},async()=>{
  const dir=fileURLToPath(DEV.dir),{scene}=await loadSceneFrom({assembled:dir+'assembled/',grounded:dir+GROUNDED_DIR});
  const setup=suggestedSetup(scene,scene.suggestions),out=metricFRows(scene,setup);
  const ground=JSON.parse(readFileSync(dir+GROUNDED_DIR+'ground.json','utf8')),[cuboid]=ground.cuboids.items.filter(b=>b.track===out.track);
  // The tested box is the annotated cuboid: same centre, horizontal heading of worldFromBox column 0, unrounded sizes.
  const M=cuboid.worldFromBox,n=Math.hypot(M[0][0],M[2][0]);
  out.testedBox.center.forEach((v,i)=>near(v,cuboid.center[i],1e-9,'centre'));
  near(out.testedBox.heading[0],M[0][0]/n,1e-12);near(out.testedBox.heading[2],M[2][0]/n,1e-12);
  assert.deepEqual([out.testedBox.vehicle.length,out.testedBox.vehicle.width,out.testedBox.vehicle.height],[cuboid.length,cuboid.width,cuboid.height]);
  assert.equal(out.suggestedTargetHeightM,TARGET_HEIGHTS_M.child);
  const half=[cuboid.width/2,cuboid.height/2,cuboid.length/2];
  for(const [name,h] of Object.entries(out.heights)) {
    assert.ok(h.rows.length>20,name);
    for(const row of h.rows) {
      // The app's own gap, recomputed from the exported car-local points: the frame is the app's.
      near(sightlineGap(appLocalFromCar(row.eyeCar,out.testedBox.vehicle),appLocalFromCar(row.targetCar,out.testedBox.vehicle),half),row.gapM,1e-7,`${name} ${row.setbackM}`);
      assert.equal(row.gapSource,'app');
      if(row.state==='blocked')assert.ok(row.gapM<=-TOO_CLOSE_TO_CALL_M+1e-9);
      if(row.reason==='too-close-to-call')assert.ok(row.gapM>-TOO_CLOSE_TO_CALL_M&&row.gapM<=0);
      // The eye is behind the car and the person ahead of its front (car-local x).
      assert.ok(row.eyeCar[0]<-cuboid.length/2&&row.targetCar[0]>cuboid.length/2,`${name} ${row.setbackM}`);
    }
    assert.ok(h.rows.some(r=>r.state==='blocked'),`${name} has Blocked rows`);
  }
});

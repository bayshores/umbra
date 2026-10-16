import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {whatIfBox,sightlineGap,assessSightline,prepareCarShape,SHAPE_REASONS,TOO_CLOSE_TO_CALL_M} from '../src/geometry.mjs';
import {suggestedSetup} from '../src/suggestions.mjs';
import {carLocal,appLocalFromCar,metricFRows,loadSceneFrom,checkAgainstBaseline,loadApp,allowedChange,CURRENT_APP,TARGET_HEIGHTS_M} from '../scripts/evaluation/metric-f.mjs';
import {devDataPresent,DEV,GROUNDED_DIR} from '../scripts/evaluation/pao-baseline.mjs';
import {historyApp} from './history-app.mjs';

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

// shape-v1: a 4 m x 2 m x 1.5 m annotated vehicle whose LiDAR shape has every solid top at `top` (below the box height,
// so step 6 runs), and a level-ish line from the eye 1.6 m high to the person's head that cuts about 0.23 m into the box.
function shapeSetup(top) {
  const vehicle={width:2,height:1.5,length:4};
  const shape=prepareCarShape({schema:'car-shape-v1',status:'measured',refusal:null,track:'t-1',box:{length:4,width:2,height:1.5},
    grid:{along:16,lateral:8},solidTopM:Array(128).fill(top)});
  const w={pose:'annotated',center:[0,.75,10],setback:1,axis:[0,0,-1],boxAxis:[0,0,-1],vehicle,shape};
  // Sparse measurements: classifySightline's later diagnostics return Unknown at once for a line that misses the box.
  const scene={suggestions:{vehicle:{status:'annotated',track:'t-1'}},measurements:{kind:'indexedSparse'}};
  const setup={input:{pixels:{},vehicle:{width:'2',height:'1.5',length:'4'},targetHeight:'1.1'},pixels:{},vehicle:{width:'2',height:'1.5',length:'4'},targetHeight:'1.1'};
  const scenario=input=>({eye:[0,1.6,-10],target:[0,Number(input.targetHeight),20],targetGround:[0,0,20],whatIf:w,baselineSetback:1});
  const INSIDE=new Set(['person-inside-vehicle','driver-inside-vehicle']);
  // The app's own sweep rows: assessSightline without diagnostics, ownPosition at the vehicle's own setback.
  const sweep=(scene,_base,_v,height)=>{
    const s=scenario({targetHeight:String(height)});
    return {rows:[1,1.25,25].map(setback=>{
      const r=assessSightline(scene,s,whatIfBox(w,setback),{diagnostics:false,ownPosition:setback===1});
      return {setback,state:INSIDE.has(r.reasonCode)?'overlap':r.state,...(r.reasonCode&&{reason:r.reasonCode}),gapM:r.gapM};
    }),step:.25,range:[1,25]};
  };
  const app={createScenario:(_scene,input)=>scenario(input),whatIfBox,assessSightline,SHAPE_REASONS,setbackSweep:sweep};
  return {scene,setup,app,w};
}

test('metric F shape-v1 rows: box result next to the shape result, diagnostics, and the allowed-change check',()=>{
  const low=shapeSetup(.9),out=metricFRows(low.scene,low.setup,{heights:{child:1.1},app:low.app});
  const [own,next,far]=out.heights.child.rows;
  // The line passes about 1.27 m above the box bottom: the box calls it Blocked, the 0.9 m shape does not prove it.
  assert.deepEqual([own.box.state,own.box.reason],['blocked',null]);
  assert.deepEqual([own.state,own.reason],['unknown','height-not-proven']);
  assert.equal(own.box.gapM,own.gapM);
  assert.equal(own.shape.status,'measured');assert.equal(own.shape.solidCutM,null);assert.equal(own.shape.hitXM,null);
  assert.equal(next.reason,'height-not-proven');
  // Far row: the box is behind the eye, the line misses it, step 6 never runs: identical results, no diagnostics.
  assert.deepEqual([far.state,far.reason,far.box.state,far.shape],['unknown',undefined,'unknown',null]);
  assert.deepEqual(out.shapeV1.allowedChanges.violations,[]);
  assert.equal(out.shapeV1.allowedChanges.resultsChecked,6);
  assert.equal(out.shapeV1.carShape.track,'t-1');assert.equal(out.shapeV1.carShape.allTopsAtLeastH,false);
  // A 1.4 m shape top is more than 0.1 m above the line: shape Blocked, with a hit and a solid cut.
  const high=shapeSetup(1.4),rows=metricFRows(high.scene,high.setup,{heights:{child:1.1},app:high.app}).heights.child.rows;
  assert.deepEqual([rows[0].state,rows[0].reason,rows[0].box.state],['blocked',undefined,'blocked']);
  assert.ok(rows[0].shape.solidCutM>TOO_CLOSE_TO_CALL_M&&rows[0].shape.hitXM!==null);
  // An app row that differs from its recomputed shape result stops the run.
  const bad=shapeSetup(.9),sweep=bad.app.setbackSweep;
  bad.app.setbackSweep=(...a)=>{const r=sweep(...a);r.rows[0]={...r.rows[0],state:'blocked',reason:undefined};return r;};
  assert.throws(()=>metricFRows(bad.scene,bad.setup,{heights:{child:1.1},app:bad.app}),/differs from its recomputed shape result/);
});

test('metric F shape-v1 allowed changes: the protocol table on full result objects',()=>{
  const box={state:'blocked',reason:'Declared vehicle volume intersects the sightline.',supportedFraction:null,gapM:-.3};
  const diag={shape:{status:'measured',solidCutM:.2,hitXM:.5}};
  assert.equal(allowedChange(box,{state:'blocked',reason:SHAPE_REASONS.blocked,supportedFraction:null,gapM:-.3},{diagnostics:false}),null);
  assert.equal(allowedChange(box,{state:'blocked',reason:SHAPE_REASONS.blocked,supportedFraction:null,gapM:-.3,...diag},{diagnostics:true}),null);
  const hnp={state:'unknown',reason:SHAPE_REASONS.heightNotProvenMeasured,supportedFraction:null,reasonCode:'height-not-proven',gapM:-.3};
  assert.equal(allowedChange(box,hnp,{diagnostics:false}),null);
  assert.equal(allowedChange(box,{...hnp,reason:SHAPE_REASONS.heightNotProvenRefused('fewer than 2 LiDAR sweeps of it were available')},{diagnostics:false}),null);
  assert.equal(allowedChange(box,{...hnp,gapM:-.31},{diagnostics:false}),'gapM changed');
  assert.match(allowedChange(box,hnp,{diagnostics:true}),/diagnostics shape record missing/);
  assert.match(allowedChange(box,{...hnp,reasonCode:'too-close-to-call'},{diagnostics:false}),/box Blocked became unknown too-close-to-call/);
  assert.match(allowedChange(box,{state:'blocked',reason:box.reason,supportedFraction:null,gapM:-.3},{diagnostics:false}),/wrong sentence/);
  const unknown={state:'unknown',reason:'x',supportedFraction:null,reasonCode:'too-close-to-call',gapM:-.05};
  assert.equal(allowedChange(unknown,{...unknown},{diagnostics:true}),null);
  assert.match(allowedChange(unknown,{state:'blocked',reason:SHAPE_REASONS.blocked,supportedFraction:null,gapM:-.05},{diagnostics:false}),/not the identical object/);
});

test('metric F loadApp: an app folder gives each function from its own module',async()=>{
  const app=await loadApp(fileURLToPath(new URL('../src/',import.meta.url)));
  for(const k of Object.keys(CURRENT_APP))assert.equal(app[k],CURRENT_APP[k],k);
});

// ------------------------------------------------------------------ AV2 8c development scene (local only)
const skip=devDataPresent()?false:'AV2 8c development data is local-only and not present';
// The v3.2 rows are history: they keep reading grounded-v3.2 (pao-baseline.mjs now serves grounded-v3.4).
const V32_DIR='grounded-v3.2/';
// shape-v1: the current app refuses grounded-v3.2 (ground schema 2, no car shape); like metric-f.mjs --app-src, the rows
// come from the app source of commit c7a39eb (tests/history-app.mjs), the code that produced the v3.2 rows.
test('metric F rows on AV2 8c (protocol v3.2): the app box, car-local points and gaps agree with the app',{skip},async()=>{
  const app=await loadApp((await historyApp()).srcDir);
  const dir=fileURLToPath(DEV.dir),{scene}=await loadSceneFrom({assembled:dir+'assembled/',grounded:dir+V32_DIR},app.loadScene);
  const setup=app.suggestedSetup(scene,scene.suggestions),out=metricFRows(scene,setup,{app});
  assert.equal(out.shapeV1,undefined,'the v3.2 app has no car shape');
  const ground=JSON.parse(readFileSync(dir+V32_DIR+'ground.json','utf8')),[cuboid]=ground.cuboids.items.filter(b=>b.track===out.track);
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

// shape-v2 (placement protocol v3.4): the current app on grounded-v3.4. Each row's box result (the shape removed) is the
// v3.2 app's row exactly, every row obeys the allowed changes (unchanged from shape-v1), and the 8c adult rows keep no
// Blocked (shape-v2 prediction 9). The rows' car-shape section keeps the report key shapeV1 (shape-v2 changes.names).
test('metric F shape-v2 rows on AV2 8c (protocol v3.4): box results reproduce the v3.2 rows; allowed changes hold',{skip},async()=>{
  assert.equal(GROUNDED_DIR,'grounded-v3.4/');
  const dir=fileURLToPath(DEV.dir),{scene}=await loadSceneFrom({assembled:dir+'assembled/',grounded:dir+GROUNDED_DIR});
  const out=metricFRows(scene,suggestedSetup(scene,scene.suggestions));
  assert.ok(out.shapeV1,'the rows carry the car shape');
  assert.deepEqual(out.shapeV1.allowedChanges.violations,[]);
  assert.equal(out.shapeV1.carShape.status,'measured');
  const old=await loadApp((await historyApp()).srcDir),{scene:s32}=await loadSceneFrom({assembled:dir+'assembled/',grounded:dir+V32_DIR},old.loadScene);
  const v32=metricFRows(s32,old.suggestedSetup(s32,s32.suggestions),{app:old});
  for(const [name,h] of Object.entries(out.heights)) {
    const ref=v32.heights[name].rows;
    assert.equal(h.rows.length,ref.length,name);
    h.rows.forEach((row,i)=>{
      assert.equal(row.setbackM,ref[i].setbackM);
      assert.deepEqual([row.box.state,row.box.reason??undefined,row.box.gapM],[ref[i].state,ref[i].reason,ref[i].gapM],`${name} ${row.setbackM}`);
      if(row.state==='blocked')assert.equal(row.box.state,'blocked','no new Blocked');
    });
  }
  assert.equal(out.heights.adult.rows.filter(r=>r.state==='blocked').length,0,'8c adult Blocked stays 0');
  assert.equal(out.heights.child.rows.filter(r=>r.state==='blocked').length,21,'8c child Blocked with the shape (contract A14)');
  assert.ok(out.heights.adult.rows.some(r=>r.box.state==='blocked'),'the box still calls some adult rows Blocked');
});

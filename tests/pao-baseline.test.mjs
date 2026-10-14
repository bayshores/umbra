import assert from 'node:assert/strict';
import test from 'node:test';
import {suggestedSetup,crossingText} from '../src/suggestions.mjs';
import {devDataPresent,loadDevelopmentScene,signedClearance,appIntersects,boxesOverlap,percentile,clickToGround,suggestionConsistency,
  setbackSweepBaseline,blockedEndRange,firstLoad,samplePlacements,honestyCounters,GROUNDED_DIR} from '../scripts/evaluation/pao-baseline.mjs';
import {historyApp} from './history-app.mjs';

const near=(a,b,tolerance,what)=>assert.ok(Math.abs(a-b)<=tolerance,`${what??''} ${a} differs from ${b}`);

// ------------------------------------------------------------------ analytic (no local data)
test('baseline helpers: signed clearance agrees with the app segment test and its own definition',()=>{
  // Box 2 x 2 x 4 at the origin, axis +z (classifySightline frame: local x = [axis.z, 0, -axis.x] = +x).
  const box={center:[0,0,0],axis:[0,0,1],vehicle:{width:2,height:2,length:4}};
  // A line 0.3 m beside the right face (x = 1.3), running along z: misses by 0.3.
  near(signedClearance([1.3,0,-10],[1.3,0,10],box).clearanceM,.3,1e-9,'miss');
  assert.equal(appIntersects([1.3,0,-10],[1.3,0,10],box),false);
  // Through the centre: the box must shrink by its smallest half size (1 m) to clear it.
  near(signedClearance([0,0,-10],[0,0,10],box).clearanceM,-1,1e-9,'through');
  assert.equal(appIntersects([0,0,-10],[0,0,10],box),true);
  // Ending 0.5 m short of the near face: the closest point is the line end (t = 1).
  const end=signedClearance([0,0,-10],[0,0,-2.5],box);near(end.clearanceM,.5,1e-9,'short');near(end.t,1,1e-12);
  // Touching the face counts as intersecting in both.
  near(signedClearance([1,0,-10],[1,0,10],box).clearanceM,0,1e-9,'touch');assert.equal(appIntersects([1,0,-10],[1,0,10],box),true);
  // A diagonal line agrees in sign with the app test across offsets.
  for(let x=-2;x<=2;x+=.05) {
    const a=[x,.4,-8],b=[x+1.5,1.4,8],c=signedClearance(a,b,box).clearanceM;
    if(Math.abs(c)>1e-9)assert.equal(c<=0,appIntersects(a,b,box),`x ${x}`);
  }
});

test('baseline helpers: upright box overlap and nearest-rank percentile',()=>{
  const a={center:[0,0,0],axis:[0,0,1],vehicle:{width:2,height:1.5,length:4}};
  assert.equal(boxesOverlap(a,{...a,center:[1.9,0,0]}),true);
  assert.equal(boxesOverlap(a,{...a,center:[2.1,0,0]}),false);
  assert.equal(boxesOverlap(a,{...a,center:[0,1.6,0]}),false,'stacked above');
  const s=Math.SQRT1_2;assert.equal(boxesOverlap(a,{...a,center:[2.6,0,0],axis:[s,0,s]}),true,'turned corner reaches');
  assert.equal(percentile([5,1,4,2,3],.5),3);assert.equal(percentile([1,2,3,4],.5),2);assert.equal(percentile([1,2,3,4],.95),4);assert.equal(percentile([],.5),null);
});

// ------------------------------------------------------------------ AV2 8c development scene (local only)
const skip=devDataPresent()?false:'AV2 8c development data is local-only and not present';
// The expected numbers belong to these data pins (grounded-v3.2, protocol v3.2; the same numbers as grounded-v3.1);
// other pins run the invariant checks only.
const PINS={ground:'556bc31b922c59853fe9e6e1d4cf9bcb8d8f7f6863de46835272f53235236115',suggestions:'86b2e5a134fbed2fc8ee778da78341aa60da86df044b08ffbc4ca87be32f7dae'};
// grounded-v3.3 (placement protocol v3.3, shape-v1): grounded-v3.2 plus the recorded vehicle's LiDAR shape (app contract A13).
const PINS33={ground:'bb3226a74216ca0848f386ff60d69f1997e14b2576bffef38dfd23612898f316',suggestions:'50049872e11ac33d9bc1cb9efa761c71e03c2b45e26aa10bb05f75cf33783c8c'};

// History: the current app refuses grounded-v3.2 (ground schema 2, no car shape), so this baseline runs pao-baseline.mjs
// and the app source of commit c7a39eb, the code that produced these numbers (tests/history-app.mjs). The current
// harness (pao-baseline.mjs GROUNDED_DIR) reads grounded-v3.3; it is tested after this one.
test('baseline on AV2 8c (protocol v3.2, history): contract values, metric C, the Blocked-end range and its own invariants',{skip},async()=>{
  assert.equal(GROUNDED_DIR,'grounded-v3.3/');
  const {baseline,suggestions}=await historyApp(),{suggestedSetup,crossingText}=suggestions;
  const {loadDevelopmentScene,clickToGround,suggestionConsistency,setbackSweepBaseline,blockedEndRange,firstLoad,samplePlacements,honestyCounters}=baseline;
  assert.equal(baseline.GROUNDED_DIR,'grounded-v3.2/');
  const {scene,entry}=await loadDevelopmentScene(),setup=suggestedSetup(scene,scene.suggestions);
  const pinned=entry.ground.sha256===PINS.ground&&entry.suggestions.sha256===PINS.suggestions;
  const first=firstLoad(scene,setup),click=clickToGround(scene),consistency=suggestionConsistency(scene),sweep=setbackSweepBaseline(scene,setup),end=blockedEndRange(scene,setup);
  for(const k of ['child','adult']) {
    assert.equal(sweep[k].clearanceSignMismatches,0,k);
    assert.equal(sweep[k].lineTouchesIntervalsConfirmedByApp,true,k);assert.equal(sweep[k].appBlockedIntervalsConfirmedByApp,true,k);
    assert.ok(sweep[k].rows.every(r=>['blocked','unknown','overlap'].includes(r.state)),k);
  }
  assert.ok(['blocked','unknown'].includes(first.current));assert.equal(first.eye.heightAboveMappedGroundM>1,true);
  const sample=samplePlacements(scene,{count:40}),honesty=honestyCounters(scene,setup,sample.placements);
  assert.equal(sample.placements.length,40);assert.equal(honesty.sweepErrors,0);assert.equal(honesty.clearanceSignMismatches,0);
  // M1: no Blocked result with the person or driver inside the box, none that cuts in by under 10 cm.
  assert.equal(honesty.clearanceStateMismatches,0);assert.equal(honesty.borderline.blockedCutInLessThan,0);
  assert.equal(honesty.startValues.conflictReportedBlocked,0);
  for(const who of ['person','driver'])assert.equal(honesty[`sweepRowsWith${who[0].toUpperCase()+who.slice(1)}InsideBox`].blocked??0,0,who);
  // Every click band keeps its integer (M0) and fractional columns; nothing is placed beyond 30 m.
  assert.deepEqual(click.fractionalClickBands.map(x=>x.band),click.bands.map(x=>x.band));
  assert.doesNotMatch(JSON.stringify([first,click,consistency,sweep,end,honesty]),/park/i);
  if(!pinned)return;
  // Protocol v3.2: the crosswalk chip shows the measured paint-offset bound (the only first-load change from v3.1).
  assert.equal(crossingText(scene.suggestions.crossing),'crosswalk mapped ±0.18 m');
  // Contract A6: first load Unknown / Unknown, the averaged box misses the line by 0.0624 m at a 0.4198 m setback.
  assert.deepEqual([first.current,first.proposed],['unknown','unknown']);near(first.clearanceM,.06241,1e-4,'first-load clearance');near(first.baselineSetbackM,.41981,1e-4);
  // Metric C: the points the app uses lie on their mapped points; the stored vehicle-front pixelUV ray hit does not (raster
  // shadow, 1.0285 m: target <= 0.05 m NOT met there), and no result uses it (protocol v3.1 declaredReplacement).
  for(const k of ['crossing','vehicleFront','person'])near(consistency.roles[k].pointUsed.horizontalM,0,1e-6,k);
  assert.equal(consistency.roles.vehicleFront.pointUsed.source,'averaged front point');
  near(consistency.roles.vehicleFront.pixelUVRayHit.horizontalM,1.028483,1e-6);
  assert.deepEqual(Object.values(consistency.roles).map(x=>[x.metricC.pointUsedMet,x.metricC.pixelUVRayHitMet]),[[true,true],[true,false],[true,true]]);
  for(const [k,v] of [['crossing',.976368],['vehicleFront',1.065777],['person',.306051]])near(consistency.roles[k].integerAnchorAlone.horizontalM,v,1e-6,k);
  // Metric B (M0 integer click): 12-25 m coverage 0.962; 516 of 562 returns at 30-60 m refused as too far.
  const band=x=>click.bands.find(b=>b.band===x);
  near(band('12-25 m').shownUncertaintyCoverage,.962,1e-3);assert.equal(band('30-60 m').unknown['Too far to place (over 30 m).'],516);
  // Contract A6 intervals (bisected; A6 lists a 0.005 m grid).
  assert.deepEqual([sweep.child.lineTouchesIntervalsM,sweep.child.appBlockedIntervalsM,sweep.adult.appBlockedIntervalsM],[[[.6784,10.4562]],[[1.0928,10.0419]],[[6.0274,10.0419]]]);
  assert.deepEqual([sweep.child.lineTouchesBoxRows,sweep.child.blockedRows,sweep.child.tooCloseToCallRows],[41,38,[.8532,10.1032,10.3532]]);
  near(sweep.child.stopsBlockingAtRowM,10.6032,1e-3);
  assert.deepEqual(sweep.child.whatIfOverlapsAnnotatedSetbacksM,{'REGULAR_VEHICLE:cf467f3d':[11.88,13.38]});
  // Plan M4: of the five rows metric F marked likely false on protocol v2, 1.10 and 9.85 m are still plain Blocked; 0.85,
  // 10.10 and 10.35 m are Unknown (too close to call). The margin was not tuned to remove them.
  assert.deepEqual(sweep.likelyFalseRowsV2.map(x=>[x.v2SetbackM,x.state,x.plainBlocked]),
    [[.85,'unknown',false],[1.1,'blocked',true],[9.85,'blocked',true],[10.1,'unknown',false],[10.35,'unknown',false]]);
  // Protocol v3.1 evaluation.stopsBlockingRange: 81 combinations, the 27 with the driver 0.40 m beyond the recording
  // camera skipped; the child's line-touch end 10.46 m ranges 9.97-10.71 m and the app's Blocked end 10.04 m 9.56-10.30 m.
  const range=end.stopsBlockingRange.child;
  assert.deepEqual([range.combinations,range.skippedDriverPositions,range.lineTouchEndM.withoutInterval,range.blockedEndM.withoutInterval],[81,27,0,0]);
  near(range.lineTouchEndM.nominal,10.4562,1e-4);near(range.blockedEndM.nominal,10.0419,1e-4);
  assert.deepEqual([range.lineTouchEndM.range,range.blockedEndM.range],[[9.9651,10.7128],[9.558,10.2984]]);
  assert.match(end.laneReconciliation.find(x=>x.value==='11.37 m').explanation,/^withdrawn/);
  assert.deepEqual([end.currentResultAcrossDriverPositions.blocked,end.currentResultAcrossDriverPositions.tooCloseToCall],[332,37]);
});

// The current harness on grounded-v3.3 (shape-v1). The geometry is the v3.2 geometry, so every box number of the v3.2
// test above holds as a box-deep number; the app's own Blocked now also needs the recorded vehicle's LiDAR shape (step 6):
// the 8.2 check's counts (child 12 Blocked and 26 height-not-proven of 57 rows, adult 0 and 17) and no new Blocked.
test('baseline on AV2 8c (protocol v3.3, shape-v1): box-deep values unchanged, the app Blocked inside them, contract A13 counts',{skip},async()=>{
  assert.equal(GROUNDED_DIR,'grounded-v3.3/');
  const {scene,entry}=await loadDevelopmentScene(),setup=suggestedSetup(scene,scene.suggestions);
  assert.equal(scene.ground.schemaVersion,3);assert.equal(scene.carShape.status,'measured');
  const pinned=entry.ground.sha256===PINS33.ground&&entry.suggestions.sha256===PINS33.suggestions;
  const first=firstLoad(scene,setup),click=clickToGround(scene),consistency=suggestionConsistency(scene),sweep=setbackSweepBaseline(scene,setup),end=blockedEndRange(scene,setup);
  for(const k of ['child','adult']) {
    assert.equal(sweep[k].clearanceSignMismatches,0,k);
    assert.equal(sweep[k].lineTouchesIntervalsConfirmedByApp,true,k);assert.equal(sweep[k].boxDeepIntervalsConfirmedByApp,true,k);
    assert.equal(sweep[k].appBlockedInsideBoxDeep,true,`${k}: no Blocked outside the box-deep intervals`);
    assert.ok(sweep[k].rows.every(r=>['blocked','unknown','overlap'].includes(r.state)),k);
    // Every row the app calls Blocked cuts at least 0.10 m into the box (no new Blocked).
    assert.ok(sweep[k].rows.filter(r=>r.state==='blocked').every(r=>r.clearanceM<=-.1+1e-4),k);
  }
  const sample=samplePlacements(scene,{count:40}),honesty=honestyCounters(scene,setup,sample.placements);
  assert.equal(sample.placements.length,40);assert.equal(honesty.sweepErrors,0);assert.equal(honesty.clearanceSignMismatches,0);
  assert.equal(honesty.clearanceStateMismatches,0);assert.equal(honesty.borderline.blockedCutInLessThan,0);
  assert.equal(honesty.startValues.conflictReportedBlocked,0);
  assert.doesNotMatch(JSON.stringify([first,click,consistency,sweep,end,honesty]),/park/i);
  if(!pinned)return;
  assert.equal(crossingText(scene.suggestions.crossing),'crosswalk mapped ±0.18 m');
  // Shape-v1 prediction 2: the first load is unchanged (the line misses the box, so step 6 never runs).
  assert.deepEqual([first.current,first.proposed],['unknown','unknown']);near(first.clearanceM,.06241,1e-4,'first-load clearance');near(first.baselineSetbackM,.41981,1e-4);
  for(const k of ['crossing','vehicleFront','person'])near(consistency.roles[k].pointUsed.horizontalM,0,1e-6,k);
  near(consistency.roles.vehicleFront.pixelUVRayHit.horizontalM,1.028483,1e-6);
  const band=x=>click.bands.find(b=>b.band===x);
  near(band('12-25 m').shownUncertaintyCoverage,.962,1e-3);assert.equal(band('30-60 m').unknown['Too far to place (over 30 m).'],516);
  // The v3.2 contract A6 intervals, now box-deep.
  assert.deepEqual([sweep.child.lineTouchesIntervalsM,sweep.child.boxDeepIntervalsM,sweep.adult.boxDeepIntervalsM],[[[.6784,10.4562]],[[1.0928,10.0419]],[[6.0274,10.0419]]]);
  assert.deepEqual([sweep.child.lineTouchesBoxRows,sweep.child.boxDeepRows,sweep.child.tooCloseToCallRows],[41,38,[.8532,10.1032,10.3532]]);
  // Contract A13: the app's own rows with the shape.
  assert.deepEqual([sweep.child.count,sweep.child.blockedRows,sweep.child.heightNotProvenRows],[57,12,26]);
  assert.deepEqual([sweep.adult.count,sweep.adult.blockedRows,sweep.adult.heightNotProvenRows,sweep.adult.boxDeepRows],[57,0,17,17]);
  assert.deepEqual(sweep.adult.appBlockedIntervalsM,[]);
  near(sweep.child.stopsBlockingAtRowM,10.6032,1e-3);
  const range=end.stopsBlockingRange.child;
  assert.deepEqual([range.combinations,range.skippedDriverPositions,range.lineTouchEndM.withoutInterval,range.boxDeepEndM.withoutInterval],[81,27,0,0]);
  near(range.lineTouchEndM.nominal,10.4562,1e-4);near(range.boxDeepEndM.nominal,10.0419,1e-4);
  assert.deepEqual([range.lineTouchEndM.range,range.boxDeepEndM.range],[[9.9651,10.7128],[9.558,10.2984]]);
  // The stationary vehicle's own result across the driver slider: the v3.2 Blocked positions (332) are now Blocked or
  // height-not-proven; too close to call is unchanged.
  const across=end.currentResultAcrossDriverPositions;
  assert.equal(across.blocked+across.heightNotProven,332);assert.equal(across.tooCloseToCall,37);
});

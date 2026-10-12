import assert from 'node:assert/strict';
import test from 'node:test';
import {suggestedSetup} from '../src/suggestions.mjs';
import {devDataPresent,loadDevelopmentScene,signedClearance,appIntersects,boxesOverlap,percentile,clickToGround,suggestionConsistency,
  setbackSweepBaseline,blockedEndRange,firstLoad,samplePlacements,honestyCounters} from '../scripts/evaluation/pao-baseline.mjs';

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
// The reproduced numbers belong to these data pins; other pins run the invariant checks only.
const PINS={ground:'6dc1a733c9ea37c89c50e58839b0ca41b6625848fec57134426c36bfe183acd7',suggestions:'94b294f3b7f7abd1e9fc58994e8e0a72f277b59c5e6490e3596157d05533d248'};

test('baseline on AV2 8c: reproduces the planning lanes and keeps its own invariants',{skip},async()=>{
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
  assert.doesNotMatch(JSON.stringify([first,click,consistency,sweep,end,honesty]),/park/i);
  if(!pinned)return;
  assert.deepEqual([first.current,first.proposed],['unknown','unknown']);near(first.clearanceM,.0251,.001,'first-load clearance');
  const lane=click.laneBandsForReproduction;
  for(const [i,p95,coverage] of [[0,.144,1],[1,.881,.954],[2,.713,.98],[3,1.417,1],[4,5.755,.842]]) {
    near(lane[i].horizontalErrorM.p95,p95,.01,lane[i].band);near(lane[i].shownUncertaintyCoverage,coverage,.002,lane[i].band);
  }
  for(const [k,v] of [['crossing',.1437],['vehicleFront',.1767],['person',.2096]])near(consistency.roles[k].horizontalM,v,.0005,k);
  // M1: of the 41 rows whose line touches the box, 0.85, 1.10 and 10.35 m are too close to call (child); for the adult
  // all 41 are. The stop distance is unchanged; the stationary vehicle's own result is Blocked at 346 driver positions
  // and too close to call at 39 (385 touched the line before M1).
  assert.deepEqual([sweep.child.lineTouchesBoxRows,sweep.child.blockedRows,sweep.child.tooCloseToCallRows],[41,38,[.8531,1.1031,10.3531]]);
  assert.deepEqual([sweep.adult.lineTouchesBoxRows,sweep.adult.blockedRows,sweep.adult.tooCloseToCallRows.length],[41,0,41]);
  near(sweep.child.stopsBlockingAtRowM,10.603,.001);
  assert.deepEqual(sweep.child.whatIfOverlapsAnnotatedSetbacksM,{'REGULAR_VEHICLE:cf467f3d':[12.02,13.52]});
  assert.deepEqual(sweep.child.rows.filter(x=>x.reason==='overlaps-real-object').map(x=>x.setbackM),sweep.child.rows.filter(x=>x.overlapsAnnotated).map(x=>x.setbackM));
  // Where the line stops touching the box (the lanes' 10.53 m and the app's stopsBlockingAt meaning) and where the app's
  // Blocked (a cut of at least 0.10 m, metric H) ends are different numbers: 10.53 m and 10.12 m for the child; the
  // adult has no Blocked interval at all.
  near(end.atNamedEye.lineTouchesUntilM,10.5338,.0005);near(end.atNamedEye.appBlockedFromM,1.2027,.0005);near(end.atNamedEye.appBlockedUntilM,10.1214,.0005);
  assert.deepEqual([sweep.child.lineTouchesIntervalsM,sweep.child.appBlockedIntervalsM,sweep.adult.appBlockedIntervalsM],[[[.7903,10.5338]],[[1.2027,10.1214]],[]]);
  assert.match(end.laneReconciliation.find(x=>x.value==='11.37 m').explanation,/^withdrawn/);
  assert.deepEqual([end.currentResultAcrossDriverPositions.blocked,end.currentResultAcrossDriverPositions.tooCloseToCall],[346,39]);
});

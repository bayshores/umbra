import { useEffect, useMemo, useRef, useState } from 'react';
import { RoadHorizon, Cube, PersonSimple, CrosshairSimple, XCircle, ArrowCounterClockwise, CaretDown, Play, Stop } from '@phosphor-icons/react';
import '@fontsource/overpass-mono/400.css';
import '@fontsource/overpass-mono/600.css';
import '@fontsource/overpass-mono/700.css';
import StreetView from './StreetView.jsx';
import AssistantBar from './AssistantBar.jsx';
import DistanceCheck, { sweepText } from './DistanceCheck.jsx';
import { PeriHost, PeriSlot } from './PeriHost.jsx';
import DotText from './DotText.jsx';
import { useReducedMotion } from './motion.mjs';
import { approachDistance } from './map-layers.mjs';
import { roadPoint, comparisonStart, placementPreview, createScenario, isMappedGround, mappedGroundHit, annotatedVehicle } from './geometry.mjs';
import { runTool, scenarioFacts, supportFacts, checkAnswer, allowedNumbersFrom, withheldNote, declaredLine, vehiclePart, sameVehicle, dims } from './scenario-assistant.mjs';
import { runAssistant } from './nemotron-client.mjs';
import { loadScene } from './scene.mjs';
import { suggestedSetup, isStillSuggested, suggestionStatus, supportSummary } from './suggestions.mjs';

const anchors = [
  {key:'crossing',name:'Crossing',label:'Crossing edge',Icon:RoadHorizon},
  {key:'parking',name:'Vehicle',label:'Current vehicle front',Icon:Cube},
  {key:'target',name:'Person',label:'Pedestrian ground point',Icon:PersonSimple},
];
const meters = n => n.toFixed(2);
// Current (baseline) vehicle words per kind: comparison badge, plan-view tag, vehicle-front label.
const BASELINE={current:['Current','Current vehicle','Current vehicle front'],annotated:['Stationary','Stationary vehicle','Stationary vehicle front'],
  declared:['Declared','Declared vehicle','Declared vehicle front'],default:['Default','Default vehicle','Vehicle front']};
const CC_BY_NC_SA='https://creativecommons.org/licenses/by-nc-sa/4.0/';
const filled = v => v !== '' && v != null;
const positive = v => filled(v) && Number.isFinite(Number(v)) && Number(v) > 0;
function roadSelectionError(scene,pixel) {
  try {
    roadPoint(scene,pixel,'comparison actor');
    return '';
  }catch(e){return e.message;}
}
const ASSISTANT_ERRORS={assistant_unavailable:'Nemotron isn\'t configured on this server.',budget_exhausted:'Nemotron\'s spending cap is reached.',
  busy:'Nemotron is busy with another request. Try again shortly.',upstream_timeout:'Nemotron didn\'t answer in time.',upstream_error:'Nemotron returned an error.',
  network:'The Nemotron server couldn\'t be reached.'};
const STATE_WORDS={blocked:'blocked',unknown:'not blocked by the vehicle (rest unknown)',overlap:'vehicle overlaps person or driver'};
// The result words shown on screen (never Clear).
const STATE_TITLE={blocked:'Blocked',unknown:'Unknown',overlap:'Overlap'};
// Peri's reaction to a result. There is no happy "sees clearly" mood: Sightline never reports Clear.
const MOOD={blocked:'blocked',unknown:'unknown',overlap:'unknown'};
let messageId=0;
const unwrapNote=note=>(note??'').replace(/^\((.*)\)$/,'$1').replace(/; /g,', ');
// The app's own one-line result for a run, shown when Nemotron's wording is withheld. baseline: the left badge's word.
function computedLine(run,baseline) {
  const f=run.facts,parts=[];
  if(f)parts.push(`${baseline} ${STATE_WORDS[f.current.state]} · proposed ${STATE_WORDS[f.proposed.state]} at ${f.proposed.setback_m} m (${f.proposed.setback_ft} ft)`);
  // The Distance check panel lists every vehicle, so this line names only the first.
  const [first,...more]=run.sweep??[];
  if(first)parts.push(`${first.label}: ${sweepText(first)}`+(more.length?` · ${more.length} more below`:''));
  return parts.join(' · ')||run.toolError||'';
}
function imagePixel(event,scene) {
  const rect=event.currentTarget.getBoundingClientRect();
  return [Math.floor((event.clientX-rect.left)/rect.width*scene.camera.width),Math.floor((event.clientY-rect.top)/rect.height*scene.camera.height)];
}

// Opened details may sit below the visible part of a scrolling case column; bring them into view once, when they open.
const revealOnOpen=el=>{ el?.scrollIntoView({block:'nearest'}); };

// Approach replay: REPLAY_STEPS driver positions over REPLAY_MS, then the last one stays for REPLAY_HOLD_MS.
const REPLAY_MS=5000,REPLAY_HOLD_MS=800,REPLAY_STEPS=50,REPLAY_STEP_MS=REPLAY_MS/REPLAY_STEPS;
// Input that stops the replay, except keys that only move focus or modify another key.
const REPLAY_STOPS=['pointerdown','keydown','click','change'],REPLAY_KEEP_KEYS=['Tab','Shift','Control','Alt','Meta'];

// Phones and tablets (this width and below) get the page layout: a pinned header with Peri, the two driver views one
// above the other, the street map as its own layer, and the chat further down the page. Wider windows keep the map
// stage and the side column.
const ARTICLE='(max-width: 1050px)';
const LAYER_OUT_MS=180;
function useMediaQuery(query) {
  const [matches,setMatches]=useState(()=>typeof matchMedia==='function'&&matchMedia(query).matches);
  useEffect(()=>{
    if(typeof matchMedia!=='function')return undefined;
    const list=matchMedia(query),change=()=>setMatches(list.matches);
    change();
    list.addEventListener('change',change);
    return ()=>list.removeEventListener('change',change);
  },[query]);
  return matches;
}

// One screen per case: the same camera and scene, with only that case's vehicle drawn (split 1 = current, 0 = proposed).
// plain (the page layout): one caption line under the screen with the result as a plain word.
function CaseView({scene,scenario,which,name,plain,onError}) {
  const result=which==='current'?scenario.currentVisibility:scenario.proposedVisibility;
  const setback=which==='current'?scenario.baselineSetback:scenario.proposedSetback;
  // Counts result changes after the first one shown; each change lights the lettering dot by dot and scans the view in once.
  const [shown,setShown]=useState(result.state),[changes,setChanges]=useState(0);
  if(shown!==result.state){ setShown(result.state); setChanges(changes+1); }
  const screen=<div className="case-screen"><StreetView scene={scene} scenario={scenario} eye={scenario.eye} mode="compare" split={which==='current'?1:0} onError={onError}/>
    {changes>0&&<span key={changes} className="case-scan" aria-hidden="true"/>}</div>;
  if(plain)return <figure className={'case case--'+which}>
    {screen}
    <figcaption className="case-head case-head--plain">
      <span className="case-label">{name}<span className="case-sep" aria-hidden="true"> · </span><span className="case-at">{meters(setback)} m</span></span>
      <span key={'state-'+changes} className={'case-word case-word--'+result.state+(changes>0?' case-word--new':'')} title={result.reason}>{STATE_TITLE[result.state]}</span>
    </figcaption>
  </figure>;
  return <figure className={'case case--'+which}>
    <figcaption className="case-head">
      <span className="case-name">{name}</span>
      <span className="case-distance">{meters(setback)} m</span>
      <DotText key={'state-'+changes} text={STATE_TITLE[result.state]} pitch={3} className={'case-state dot-text--'+result.state} title={result.reason} animate={changes>0}/>
    </figcaption>
    {screen}
  </figure>;
}

export function App() {
  const [catalog,setCatalog]=useState(null),[sceneId,setSceneId]=useState(''),[scene,setScene]=useState(null),[fatal,setFatal]=useState('');
  const [loadAttempt,setLoadAttempt]=useState(0);
  const [recordedImage,setRecordedImage]=useState(null);
  // The corner window shows the top-down map (where the proposed vehicle can be dragged) or the recorded photo.
  const [inset,setInset]=useState('map');
  // The result card keeps the per-object evidence folded away until asked for.
  const [showEvidence,setShowEvidence]=useState(false);
  const [pixels,setPixels]=useState({}),[active,setActive]=useState('crossing'),[cursor,setCursor]=useState(['','']);
  const [vehicle,setVehicle]=useState({width:'',height:'',length:''}),[targetHeight,setTargetHeight]=useState('');
  const [input,setInput]=useState(null),[editing,setEditing]=useState(true),[message,setMessage]=useState('');
  // suggestion: the prepared setup for this scene (suggestedSetup), or null when the scene has no suggestions.
  const [suggestion,setSuggestion]=useState(null);
  // The suggested default vehicle front is a declared what-if with no parked vehicle measured, so it is not "current".
  const defaultFront=pixel=>suggestion?.provenance.parking?.status==='default'&&isStillSuggested('parking',pixel,suggestion.provenance);
  const [question,setQuestion]=useState(''),[running,setRunning]=useState(null),[reply,setReply]=useState(null),[sweep,setSweep]=useState(null),[declared,setDeclared]=useState([]);
  // The chat log: your messages, Peri's replies (Nemotron's words) and Sightline's own computed lines, oldest first.
  const [messages,setMessages]=useState([]);
  const say=(...items)=>setMessages(list=>[...list,...items.filter(Boolean).map(m=>({id:++messageId,...m}))]);
  // A Nemotron declaration or an applied distance-check vehicle replaced the vehicle on screen (it stays replaced until
  // Reset; on a mapped-ground scene the checked vehicle's own distance row restores it as it was checked).
  const [vehicleDeclared,setVehicleDeclared]=useState(false);
  // Which current (baseline) vehicle the values describe. Mapped-ground scenes: the annotated stationary vehicle in its
  // annotated pose (geometry.mjs annotatedVehicle, the same test the comparison uses; labelled stationary, never why it
  // stopped) until its front or size is changed or a declaration replaces it (d), after which it is a declared box.
  // Measured-depth scenes: the user's current vehicle.
  const baselineKind=(p,v,d)=>defaultFront(p?.parking)?'default':!isMappedGround(scene)?'current':!d&&annotatedVehicle(scene,p,v)?'annotated':'declared';
  const baselineWord=(p,v,d)=>{const kind=baselineKind(p,v,d);return kind==='default'&&isStillSuggested('crossing',p.crossing,suggestion.provenance)?`${suggestion.provenance.parking.setbackFt} ft default`:BASELINE[kind][0];};
  const labelOf=a=>a.key==='parking'?BASELINE[baselineKind(pixels,vehicle,vehicleDeclared)][2]:a.label;
  // runRef: the one in-flight Nemotron run. work: latest scenario values for synchronous tool calls during a run.
  const runRef=useRef(null),work=useRef({});
  // replay: the approach replay while it plays ({from: the comparison input it replaces until it ends, far and near:
  // the driver distances it moves between}). stopReplay ends the one playing.
  const [replay,setReplay]=useState(null),stopReplay=useRef(null),reduced=useReducedMotion();
  // The page layout's street map layer: null, 'open', or 'closing' while it fades out. Focus moves into it when it opens
  // and back to the Map button when it closes; the page behind it does not scroll.
  const article=useMediaQuery(ARTICLE),[layer,setLayer]=useState(null),layerRef=useRef(null),mapButton=useRef(null),layerWas=useRef(null);
  const layerShown=!!layer;
  useEffect(()=>{if(!article)setLayer(null);},[article]);
  useEffect(()=>{
    if(!layerShown)return undefined;
    const root=document.documentElement;
    root.classList.add('layer-open');
    return ()=>root.classList.remove('layer-open');
  },[layerShown]);
  useEffect(()=>{
    const was=layerWas.current;
    layerWas.current=layer;
    if(layer==='open'&&was!=='open')layerRef.current?.focus({preventScroll:true});
    if(was==='open'&&layer!=='open')mapButton.current?.focus({preventScroll:true});
    if(layer!=='closing')return undefined;
    const timer=setTimeout(()=>setLayer(null),LAYER_OUT_MS);
    return ()=>clearTimeout(timer);
  },[layer]);
  const closeLayer=()=>setLayer(l=>l==='open'?(reduced?null:'closing'):l);
  useEffect(()=>{
    let live=true;
    fetch('/scenes/index.json').then(async r=>{if(!r.ok)throw new Error('Scene catalog unavailable ('+r.status+').');return r.json();})
      .then(entries=>{if(!Array.isArray(entries)||!entries.length||entries.some(e=>!e.id||!e.label))throw new Error('No prepared measured scene is available.');if(live){setCatalog(entries);setSceneId(entries[0].id);}})
      .catch(e=>{if(live)setFatal(e.message);});
    return ()=>{live=false;};
  },[]);
  useEffect(()=>{
    if(!sceneId||!catalog)return;
    let live=true;const controller=new AbortController();setScene(null);setInput(null);setPixels({});setEditing(true);setActive('crossing');setCursor(['','']);setMessage('');setFatal('');
    runRef.current=null;setRunning(null);setReply(null);setSweep(null);setDeclared([]);setVehicleDeclared(false);setSuggestion(null);setMessages([]);setReplay(null);
    const entry=catalog.find(entry=>entry.id===sceneId);
    loadScene(fetch,sceneId,entry.metadataSha256,controller.signal,entry.suggestions,entry.ground).then(s=>{
      if(!live)return;
      setScene(s);
      if(s.suggestions){const next=suggestedSetup(s,s.suggestions);setSuggestion(next);applySuggestion(next);}
    }).catch(e=>{if(live)setFatal(e.message);});
    return ()=>{live=false;controller.abort();runRef.current?.abort();};
  },[sceneId,loadAttempt,catalog]);
  useEffect(()=>{
    if(!scene?.recordedImageBlob||fatal){setRecordedImage(null);return;}
    const url=URL.createObjectURL(scene.recordedImageBlob);setRecordedImage({scene,url});
    return ()=>URL.revokeObjectURL(url);
  },[scene,fatal]);
  // The driver moves from the recorded camera position to the nearest one allowed, slowing like a car nearing the
  // crossing; each step is a new comparison input, so the driver views and results are computed for it. When it ends,
  // or on any other click or key, the input on screen before comes back. Its clock is wall time minus the time the tab
  // was hidden, so it pauses while hidden and keeps its length on a slow screen; turning on Reduce Motion stops it.
  useEffect(()=>{
    if(!replay)return undefined;
    const {from,far,near}=replay;
    let timer=0,start=performance.now(),hiddenAt=0,shown=-1,over=false;
    const end=()=>{if(over)return;over=true;clearTimeout(timer);setInput(from);setReplay(null);};
    if(reduced){end();return undefined;}
    stopReplay.current=end;
    // One timer per step; the last step stays until the hold is over.
    const tick=()=>{
      if(over||hiddenAt)return;
      const elapsed=performance.now()-start;
      if(elapsed>=REPLAY_MS+REPLAY_HOLD_MS){end();return;}
      const step=Math.min(REPLAY_STEPS,Math.floor(elapsed/REPLAY_STEP_MS));
      if(step!==shown){
        shown=step;
        setInput({...from,driverDistance:approachDistance(step,REPLAY_STEPS,far,near)});
      }
      timer=setTimeout(tick,(step<REPLAY_STEPS?(step+1)*REPLAY_STEP_MS:REPLAY_MS+REPLAY_HOLD_MS)-elapsed);
    };
    const visibility=()=>{
      const now=performance.now();
      if(document.hidden){if(!hiddenAt){hiddenAt=now;clearTimeout(timer);}}
      else if(hiddenAt){start+=now-hiddenAt;hiddenAt=0;tick();}
    };
    if(document.hidden)hiddenAt=start;else tick();
    // Clicks and changes count too: screen-reader activation and native select popups send no pointerdown or keydown.
    // A finger on the screen only counts once it taps (a click), so scrolling the page to watch keeps it playing.
    const interrupt=e=>{if(e.type==='pointerdown'&&e.pointerType==='touch')return;
      if(!e.target?.closest?.('.approach-replay')&&!REPLAY_KEEP_KEYS.includes(e.key))end();};
    REPLAY_STOPS.forEach(type=>document.addEventListener(type,interrupt,true));
    document.addEventListener('visibilitychange',visibility);
    return ()=>{over=true;clearTimeout(timer);stopReplay.current=null;document.removeEventListener('visibilitychange',visibility);
      REPLAY_STOPS.forEach(type=>document.removeEventListener(type,interrupt,true));};
  },[replay,reduced]);
  const setup=useMemo(()=>{
    if(!scene||anchors.some(a=>!pixels[a.key])||Object.values(vehicle).some(v=>!filled(v))||!filled(targetHeight))return {scenario:null,input:null,error:''};
    try {
      // The what-if starts on the current vehicle (an annotated vehicle: its annotated pose and setback).
      const next={pixels,vehicle,targetHeight,...comparisonStart(scene,pixels,vehicle,vehicleDeclared),...(vehicleDeclared&&isMappedGround(scene)&&{vehicleDeclared:true})};
      return {scenario:createScenario(scene,next),input:next,error:''};
    }catch(e){return {scenario:null,input:null,error:e.message};}
  },[scene,pixels,vehicle,targetHeight,vehicleDeclared]);
  const preview=useMemo(()=>{
    if(!scene||scene.measurements?.kind==='indexedSparse'&&!isMappedGround(scene))return {actors:null,error:''};
    try{return {actors:placementPreview(scene,pixels,vehicle,targetHeight,vehicleDeclared),error:''};}
    catch(e){return {actors:null,error:e.message};}
  },[scene,pixels,vehicle,targetHeight,vehicleDeclared]);
  const result=useMemo(()=>{
    if(!scene||!input)return {scenario:null,error:''};
    try{return {scenario:createScenario(scene,input),error:''};}catch(e){return {scenario:null,error:e.message};}
  },[scene,input]);
  const scenario=result.scenario;
  const candidate=useMemo(()=>{
    if(!scene||cursor.some(v=>!filled(v)))return {valid:false,error:''};
    const error=roadSelectionError(scene,cursor.map(Number));
    return {valid:!error,error};
  },[scene,cursor]);
  // Mapped-ground scenes: each placed point's support class and hidden flag, from the same runtime rule as placement.
  const support=useMemo(()=>{
    if(!scene||!isMappedGround(scene))return {};
    return Object.fromEntries(anchors.filter(a=>pixels[a.key]).map(a=>{try{return [a.key,mappedGroundHit(scene,pixels[a.key])];}catch{return [a.key,null];}}));
  },[scene,pixels]);
  useEffect(()=>{work.current={pixels,vehicle,targetHeight,input,setupInput:setup.input};});
  const selected=anchors.find(a=>a.key===active);
  // Edits that reset the comparison also cancel a run and clear Nemotron's reply and distance results. Declared parts
  // stay: the line credits only the fields that still hold them.
  function resetComparison() {runRef.current?.abort();runRef.current=null;setRunning(null);setInput(null);setReply(null);setSweep(null);}
  // A complete suggestion opens in the comparison (as if Compare positions were pressed); otherwise placement continues
  // at the first unplaced object.
  function applySuggestion(next) {
    const first=anchors.find(a=>!next.pixels[a.key])?.key??'crossing';
    setPixels(next.pixels);setVehicle(next.vehicle);setTargetHeight(next.targetHeight);setActive(first);setCursor(next.pixels[first]??['','']);setMessage('');
    setInput(next.input&&{...next.input});setEditing(!next.input);
  }
  function resetToSuggested() {resetComparison();setDeclared([]);setVehicleDeclared(false);applySuggestion(suggestion);}
  function selectObject(key) {setActive(key);setCursor(pixels[key]??['','']);setEditing(true);setMessage('');}
  function selectPoint(pixel) {
    setCursor(pixel);
    const error=roadSelectionError(scene,pixel);
    if(error){setMessage(error);return;}
    setPixels(previous=>({...previous,[active]:pixel}));setMessage('');resetComparison();
  }
  function setCursorPoint() {if(cursor.some(v=>!filled(v))){setMessage('Enter both pixel coordinates.');return;}selectPoint(cursor.map(Number));}
  function compare() {if(setup.scenario){if(!scenario)setInput(setup.input);setEditing(false);setMessage('');}}
  function moveVan(value) {setInput(previous=>({...previous,proposedSetback:value}));setMessage('');}
  function moveDriver(value) {setInput(previous=>({...previous,driverDistance:value}));setMessage('');}
  function playApproach() {
    if(stopReplay.current){stopReplay.current();return;}
    if(scenario&&!running)setReplay({from:input,far:scenario.limits.captureDistance,near:scenario.limits.minimumDriverDistance});
  }
  // A distance-check row is a declared what-if: its vehicle and Sightline's computed stop distance go into the editable
  // fields. A row matching the checked vehicle (to the 2 decimals Nemotron was given) restores its exact sizes and credit.
  function applySweepRow(r) {
    if(!input)return;
    const own=!!sweep.own&&sameVehicle(r.vehicle,sweep.own),v=own?sweep.own:r.vehicle,mapped=isMappedGround(scene);
    const keep=sameVehicle(input.vehicle,v),next={...input,...(!keep&&{vehicle:{...v}}),proposedSetback:r.stopsBlockingAt??input.proposedSetback};
    // Another vehicle's row declares that vehicle. On a mapped-ground scene the checked vehicle's own row restores it as
    // it was checked (its stop distance was computed in that pose: the annotated pose when it was not declared).
    const nowDeclared=own&&mapped?sweep.vehicleDeclared:vehicleDeclared||!keep&&!own;
    if(mapped){delete next.vehicleDeclared;if(nowDeclared)next.vehicleDeclared=true;}
    if(!keep){
      setVehicle({width:String(v.width),height:String(v.height),length:String(v.length)});
      if(!own)setDeclared(p=>[...p,vehiclePart(r.label,v)]);
    }
    setVehicleDeclared(nowDeclared);setInput(next);setMessage('');
    // A distance-check answer still describes this panel; any other answer was about the previous values.
    setReply(p=>p?.mode==='check'&&p.basis===input?{...p,basis:next}:null);
  }
  // Explicit Ask/Check only; never from effects, drags, sliders or numeric edits. One run at a time, no retry.
  async function runNemotron(mode) {
    if(running||replay||(mode==='ask'?askHint||!question.trim():checkHint))return;
    // run.basis is the comparison input the reply describes; the reply is hidden once the input changes.
    const s=scene,controller=new AbortController(),base=input??setup.input,run={basis:input,facts:null,sweep:null,toolError:null,parts:[]};
    say({from:'you',text:mode==='ask'?question.trim():'Check distances'});
    if(mode==='ask')setQuestion('');
    runRef.current?.abort();runRef.current=controller;
    // Facts describe only the input on screen. An invalid input sends its problem, never the baseline setup's facts.
    const known=input?scenario:setup.scenario,problem=input?result.error:setup.error;
    // Without a computed scenario: only what is declared so far, never pixels or coordinates.
    const context=known?scenarioFacts(s,known):{scene:s.id,units:s.units,placed:anchors.filter(a=>pixels[a.key]).map(a=>a.name.toLowerCase()),
      vehicle_m:Object.fromEntries(Object.entries(input?.vehicle??vehicle).filter(([,v])=>positive(v)).map(([k,v])=>[k,Number(v)])),
      ...(positive(input?.targetHeight??targetHeight)&&{targetHeight_m:Number(input?.targetHeight??targetHeight)}),...(problem&&{problem}),
      ...(isMappedGround(s)&&{support:supportFacts(support)})};
    // Which values on screen are still prepared suggestions or declared defaults.
    if(suggested?.context)context.suggested=suggested.context;
    if(mode==='check'){work.current.input=base;run.basis=base;if(!input)setInput(base);setEditing(false);setMessage('');}
    setRunning(mode);setReply(null);
    // runTool decides (labels, sizes, heights, sweeps); this applies its result to the fields, comparison and panel.
    const executeTool=(name,args)=>{
      if(runRef.current!==controller)return {error:'Request was cancelled.'};
      const w=work.current,out=runTool(s,mode,{...w,run,vehicleLabel:onScreenVehicleLabel()},name,args);
      if(out.error){run.toolError=out.error;return {error:out.error};}
      if(out.fields){Object.assign(w,out.fields);setVehicle(w.vehicle);setTargetHeight(w.targetHeight);}
      if(out.clearSweep)setSweep(null);
      if(out.parts){run.parts.push(...out.parts);setDeclared(p=>[...p,...out.parts]);if(out.parts.some(p=>p.key==='vehicle'))setVehicleDeclared(true);}
      if(out.facts)run.facts=out.facts;
      w.input=out.input;run.basis=out.input;setInput(out.input);setEditing(false);setMessage('');
      if(out.sweep){run.sweep=out.sweep.results;setSweep({scene:s,...out.sweep,reveal:true});}
      return out.result;
    };
    const done=r=>{
      setReply({mode,basis:run.basis,...r});
      const set=run.parts.length&&run.basis?declaredLine(run.parts,run.basis):'';
      const first=run.sweep?.find(v=>!v.error);
      // Peri speaks only Nemotron's own answer (or, without one, the values it declared). A withheld answer, an error and
      // every computed result are Sightline's lines. basis: an error leaves Peri sleepy while the comparison is unchanged.
      say(r.text?{from:'peri',text:r.text,declared:set}:set&&{from:'peri',text:`Set ${set}`},
        r.error&&{from:'sightline',kind:'error',text:r.error,basis:run.basis},
        r.note&&{from:'sightline',kind:'note',text:r.note},
        r.computed&&{from:'sightline',text:r.computed},
        mode==='ask'&&run.facts&&{from:'sightline',kind:'result',state:run.facts.proposed.state,setback:run.facts.proposed.setback_m},
        mode==='check'&&first&&{from:'sightline',kind:'sweep',text:`${first.label}: ${sweepText(first)}`});
    };
    try {
      const out=await runAssistant({mode,question:question.trim(),context},executeTool,{signal:controller.signal});
      if(runRef.current!==controller)return;
      if(out.error){done({error:'Nemotron didn\'t finish.'});return;}
      // Mapped-ground scenes also withhold any answer using a word built on "park" (the vehicle is stationary, reason unknown).
      const checked=checkAnswer(out.answer,allowedNumbersFrom(context,...out.trace.map(t=>t.result)),{mapped:isMappedGround(s)});
      done(checked.ok?{text:out.answer.trim()}:{computed:computedLine(run,baselineWord(run.basis?.pixels,run.basis?.vehicle,run.basis?.vehicleDeclared)),note:withheldNote(checked)});
    }catch(e){
      if(runRef.current===controller)done({error:ASSISTANT_ERRORS[e?.code]??e?.message??'The Nemotron request failed.'});
    }finally{
      if(runRef.current===controller){runRef.current=null;setRunning(null);}
    }
  }
  // A failed load and the loading screen hold Peri in the middle of the window; it flies to the page once the scene opens.
  if(fatal)return <PeriHost><main className="app app--loading"><h1 className="sr-only">Sightline</h1>
    <section className="unavailable" role="alert"><PeriSlot mood="sleepy" size={88}/><h2>Scene unavailable</h2><p>{fatal}</p><button onClick={()=>sceneId?setLoadAttempt(n=>n+1):location.reload()}>Retry loading</button></section></main></PeriHost>;
  if(!scene)return <PeriHost><main className="app app--loading"><div className="backdrop" aria-hidden="true"><i/><i/><i/></div><div className="loading" role="status"><PeriSlot mood="working" size={96}/><span>Loading measured scene…</span></div></main></PeriHost>;
  const mapped=isMappedGround(scene),placementUnavailable=scene.measurements?.kind==='indexedSparse'&&!mapped;
  const surface=mapped?'mapped ground':'measured road';
  const supportText=key=>support[key]?support[key].supportLabel+(support[key].hidden?', hidden':''):'';
  // Hover text only: the annotated stationary vehicle is drawn from its annotated pose.
  const annotatedNote=key=>key==='parking'&&baselineKind(pixels,vehicle,vehicleDeclared)==='annotated'?', annotated pose':'';
  const recordedUrl=recordedImage?.scene===scene?recordedImage.url:undefined;
  const recordedSource=scene.measurements?recordedUrl:'/scenes/'+scene.id+'/recorded.jpg';
  const ownsImage=image=>image.isConnected&&image.getAttribute('src')===recordedSource;
  const evidence=scene.measurements?`${scene.reconstruction?'NVIDIA estimate from projected references':'Projected LiDAR'}${mapped?' on mapped ground':''}`:`${scene.reconstruction?'NVIDIA estimate from measured anchors':'Measured depth'} ${(scene.coverage*100).toFixed(1)}%`;
  // Mapped-ground (AV2) scenes carry their CC BY-NC-SA attribution in the evidence key.
  // The changes notice is in the hover text and, for screen readers, in the line itself.
  const attribution=mapped?<span className="attribution" title={scene.ground.license.changes}><span>{scene.ground.license.source}</span><span>{scene.ground.license.attribution}</span><a href={CC_BY_NC_SA} target="_blank" rel="noopener noreferrer">CC BY-NC-SA 4.0</a><span className="sr-only">{scene.ground.license.changes}</span></span>:null;
  const complete={crossing:!!pixels.crossing,parking:!!pixels.parking&&Object.values(vehicle).every(positive),target:!!pixels.target&&positive(targetHeight)};
  const point=pixels[active];
  const inImage=cursor.every((v,i)=>filled(v)&&Number.isInteger(Number(v))&&Number(v)>=0&&Number(v)<(i?scene.camera.height:scene.camera.width));
  const pending=inImage&&(!point||cursor.some((v,i)=>Number(v)!==point[i]));
  const bubbleStyle=point?{'--anchor-x':((point[0]+.5)/scene.camera.width*100)+'%','--anchor-y':((point[1]+.5)/scene.camera.height*100)+'%'}:{};
  const error=message||(editing?preview.error||setup.error:result.error);
  const placed=anchors.every(a=>pixels[a.key]);
  const askHint=placed?'':'Place crossing, vehicle, person first';
  // A comparison input always holds valid sizes, so the sweep can run from it even while its proposed setback is invalid.
  const checkHint=!placed?'Place crossing, vehicle and person to check distances':input||setup.scenario?''
    :setup.error?`Check distances needs a vehicle and person that fit the ${surface}`:'Enter vehicle size and person height to check distances';
  // Distance results stay tied to the scene, points, person height and driver position they were computed with.
  const sweepFor=i=>!editing&&i&&sweep?.scene===scene&&sweep.pixels===i.pixels&&sweep.targetHeight===Number(i.targetHeight)&&sweep.driverDistance===Number(i.driverDistance);
  const sweepShown=sweepFor(input);
  const overridden=[...(vehicleDeclared?['vehicle']:[]),...(declared.some(d=>d.key==='person')?['targetHeight']:[])];
  const suggested=suggestion&&suggestionStatus(scene.suggestions,suggestion,{pixels,vehicle,targetHeight,overridden});
  // Label of the vehicle on screen: the latest Nemotron declaration still in the fields, else a still-suggested default.
  const onScreenVehicleLabel=()=>{const w=work.current.input?.vehicle??work.current.vehicle,v=declared.findLast(d=>d.key==='vehicle');
    if(v&&sameVehicle(v.value,w))return v.label;
    return suggestion&&!overridden.includes('vehicle')&&isStillSuggested('vehicle',w,suggestion.provenance)?scene.suggestions.vehicle.label:null;};
  const suggestedTag=(key,value)=>suggestion&&!overridden.includes(key)&&isStillSuggested(key,value,suggestion.provenance)?<span className="suggested-tag"><span className="sr-only">, </span>Suggested, {suggestion.provenance[key].status}</span>:null;
  const selectedLabel=labelOf(selected);
  // Mapped-ground comparison: the support of the points it was computed from, suggested or edited (hover text).
  const compareSupport=mapped&&scenario?supportSummary(scenario.support):{text:'',title:''};
  const kindNow=input?baselineKind(input.pixels,input.vehicle,input.vehicleDeclared):baselineKind(pixels,vehicle,vehicleDeclared);
  // One evidence line per placed object: the suggestion it still holds (or its size) and where its point comes from.
  const evidenceRows=anchors.filter(a=>pixels[a.key]).map(({key,name})=>{
    const part=suggested?.parts.find(p=>p.key===key),s=support[key],v=input?.vehicle??vehicle,h=input?.targetHeight??targetHeight;
    const value=key==='parking'?`${BASELINE[kindNow][1]}, ${Object.values(v).every(positive)?dims(v):'size not set'}`
      :part?.text??(key==='target'?(positive(h)?`${meters(Number(h))} m`:'height not set'):'placed');
    const note=part?.note?unwrapNote(part.note):s?s.supportLabel+(s.hidden?', hidden':''):'';
    return {key,name,value,note,title:part?.title??((supportText(key)+annotatedNote(key))||undefined)};
  });
  // The place is the catalog label's first part ("Palo Alto · AV2 …" shows as "Palo Alto").
  const placeName=label=>label.split(' · ')[0];
  const placeLabel=placeName(catalog.find(e=>e.id===sceneId)?.label??'');
  // The page layout (phones and tablets) uses no icons on its controls.
  const resetButton=suggested?.edited&&<button type="button" className="suggestion-reset" onClick={resetToSuggested}>{!article&&<ArrowCounterClockwise size={14} aria-hidden="true"/>}Reset to suggested</button>;
  const credits=<p className="credits"><span>{evidence}{!scene.reconstruction&&!scene.measurements&&', dark gaps unknown'}</span><span title="This geometric comparison is not a safety assessment.">Declared what-if actors</span>{attribution}</p>;
  // The map ruler shows the checked distances for the vehicle on screen, while those results still apply.
  const rulerRows=sweepShown?sweep.results.find(r=>!r.error&&sameVehicle(r.vehicle,input.vehicle))?.rows??null:null;
  const lastSweep=messages.findLast(m=>m.kind==='sweep')?.id;
  // While the approach replay plays, the chat's Distance check stays in place, dimmed and inactive: it describes the
  // input that comes back when the replay ends (the map ruler follows the moving driver and hides).
  const chatInput=replay?.from??input;
  const renderSweep=m=>m.id===lastSweep&&sweepFor(chatInput)?<div className={replay?'distance-paused':undefined} inert={replay?true:undefined}><DistanceCheck results={sweep.results} vehicle={chatInput.vehicle} setback={chatInput.proposedSetback} codeNote={!mapped} disabled={!!running||!!replay} reveal={false} onRevealed={()=>{}} onApply={applySweepRow}/></div>:null;
  // Peri's mood wherever it is: scanning while it works, sleepy after a failed request (until the comparison changes),
  // reacting to the proposed result, idle while placing.
  const last=messages.at(-1);
  const mood=running?'working':last?.kind==='error'&&last.basis===input?'sleepy':!editing&&scenario?MOOD[scenario.proposedVisibility.state]:'idle';

  // Parts both layouts share.
  const placeSelect=catalog.length>1&&<select className="place-select" aria-label="Measured scene" value={sceneId} onChange={e=>setSceneId(e.target.value)}>{catalog.map(e=><option key={e.id} value={e.id} title={e.label}>{placeName(e.label)}</option>)}</select>;
  const placeControl=placeSelect||<span className="place-name">{placeLabel}</span>;
  const brand=<a className="brand" href="/" aria-label="Sightline home"><DotText text="Sightline" pitch={3.4}/></a>;
  const modeChoice=(className,thumb)=><nav className={className} aria-label="Mode" data-active={editing?0:1}>
    {thumb&&<span className="segmented-thumb" aria-hidden="true"/>}
    <button type="button" aria-pressed={editing} onClick={()=>setEditing(true)}>Place</button>
    <button type="button" aria-pressed={!editing} disabled={editing&&!setup.scenario} title={editing&&!setup.scenario?'Place crossing, vehicle and person first':undefined} onClick={compare}>Compare</button>
  </nav>;
  // The recorded photo where the objects are placed, with their markers and the pixel cursor.
  const editorFrame=<div className="recorded-frame">
    <div className={'recorded-picker'+(placementUnavailable?' recorded-picker--readonly':'')} role={placementUnavailable?undefined:'button'} tabIndex={placementUnavailable?undefined:0} aria-label={placementUnavailable?undefined:'Select '+selectedLabel+' on '+surface+'. Arrow keys move the pixel cursor; Enter selects.'} title={placementUnavailable?undefined:'Place '+selectedLabel.toLowerCase()+(mapped?' on mapped ground':' on a supported road pixel')}
      onClick={placementUnavailable?undefined:e=>selectPoint(imagePixel(e,scene))}
      onPointerMove={placementUnavailable?undefined:e=>{if(e.pointerType!=='mouse')return;const next=imagePixel(e,scene);setCursor(previous=>previous.every((v,i)=>Number(v)===next[i]&&filled(v))?previous:next);setMessage('');}}
      onKeyDown={placementUnavailable?undefined:e=>{if(e.key==='Enter'){e.preventDefault();setCursorPoint();}else if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();if(cursor.some(v=>!filled(v))){setMessage('Enter a pixel coordinate first, then use arrow keys.');return;}setCursor(([u,v])=>[Number(u)+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0),Number(v)+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0)]);setMessage('');}}}>
      <img src={recordedSource} alt="Recorded street" onError={e=>{if(ownsImage(e.currentTarget))setFatal('Recorded image unavailable.');}} onLoad={e=>{if(ownsImage(e.currentTarget)&&(e.currentTarget.naturalWidth!==scene.camera.width||e.currentTarget.naturalHeight!==scene.camera.height))setFatal('Recorded image dimensions disagree with calibration.');}}/>
      {!scene.measurements&&<img className="support-mask" src={'/scenes/'+scene.id+'/support.png'} alt="" onError={()=>setFatal('Depth support mask unavailable.')}/>}
      {!placementUnavailable&&<div className="recorded-actors"><StreetView scene={scene} preview={preview.actors} mode="recorded" onError={setFatal}/></div>}
    </div>
    {!placementUnavailable&&anchors.filter(a=>pixels[a.key]).map(({key,name,Icon})=>{
      // A hidden point (its pixel shows an annotated object; the point lies behind it) is drawn as a hollow marker.
      const hidden=support[key]?.hidden===true,selected=active===key,label=supportText(key),title=label+annotatedNote(key);
      return <button key={key} className={'anchor-marker'+(selected?' anchor-marker--selected':'')+(hidden?' anchor-marker--hidden':'')} tabIndex={selected?-1:0} aria-label={'Edit '+name+' placement'+(label?', '+label:'')} title={title||undefined} onClick={()=>selectObject(key)}
        style={{left:((pixels[key][0]+.5)/scene.camera.width*100)+'%',top:((pixels[key][1]+.5)/scene.camera.height*100)+'%'}}><Icon size={17} weight="bold"/></button>;})}
    {!placementUnavailable&&pending&&<span className={'pixel-cursor'+(candidate.valid?' pixel-cursor--supported':' pixel-cursor--unsupported')} role="img" aria-label={candidate.valid?(mapped?'Mapped ground point':'Measured road point'):candidate.error} style={{left:((Number(cursor[0])+.5)/scene.camera.width*100)+'%',top:((Number(cursor[1])+.5)/scene.camera.height*100)+'%'}}>{candidate.valid?<CrosshairSimple size={28} weight="bold"/>:<XCircle size={28} weight="fill"/>}</span>}
  </div>;
  // The vehicle's size or the person's height (on desktop it floats beside the selected point).
  const fieldBubble=!placementUnavailable&&active!=='crossing'&&<div className={'field-bubble'+(point?' field-bubble--anchored':'')} aria-label={selected.name+' dimensions'}>
    <div className="bubble-title">{selected.name==='Vehicle'?'Vehicle size':'Person height'} <span>m</span></div>
    {active==='parking'?suggestedTag('vehicle',vehicle):suggestedTag('targetHeight',targetHeight)}
    {active==='parking'?['width','height','length'].map(key=><label key={key}><span>{key}</span><input type="number" aria-label={'Vehicle '+key+' in meters'} aria-invalid={filled(vehicle[key])&&!positive(vehicle[key])} min="0" step="any" value={vehicle[key]} onChange={e=>{setVehicle(v=>({...v,[key]:e.target.value}));resetComparison();}}/></label>)
      :<input type="number" aria-label="Target height in meters" aria-invalid={filled(targetHeight)&&!positive(targetHeight)} min="0" step="any" value={targetHeight} onChange={e=>{setTargetHeight(e.target.value);resetComparison();}}/>}
  </div>;
  const pixelControls=!placementUnavailable&&<div className="pixel-controls"><span className="selected-label">{!article&&<selected.Icon size={17}/>}{selectedLabel}{suggestedTag(active,pixels[active])}{!!supportText(active)&&<span className="suggested-tag support-tag" title={supportText(active)+annotatedNote(active)}><span className="sr-only">, </span>{supportText(active)}</span>}</span><div className="pixel-inputs"><span>Pixel</span>{['X','Y'].map((name,i)=><label key={name}><span className="sr-only">Pixel {name}</span><input aria-label={'Pixel '+name} aria-invalid={!!candidate.error} aria-describedby="point-feedback" type="number" min="0" max={i?scene.camera.height-1:scene.camera.width-1} step="1" value={cursor[i]} onKeyDown={e=>{if(e.key==='Enter')setCursorPoint();}} onChange={e=>{setCursor(p=>p.map((v,j)=>i===j?e.target.value:v));setMessage('');}}/></label>)}<button disabled={!candidate.valid} aria-describedby="point-feedback" onClick={setCursorPoint}>Set point</button></div><span id="point-feedback" className="sr-only" role="status">{candidate.valid?(mapped?'Mapped ground point':'Measured road point'):candidate.error}</span></div>;
  const placePreview=<StreetView scene={scene} preview={preview.actors} onError={setFatal}/>;
  const planView=scenario?<StreetView scene={scene} scenario={scenario} eye={scenario.eye} mode="plan" framing="context" ruler={rulerRows} baselineLabel={BASELINE[kindNow][1]} onSetbackChange={moveVan} onDriverChange={moveDriver} onError={setFatal}/>:<div className="unavailable">Comparison unavailable</div>;
  const photoView=recordedSource&&<img className="recorded-still" src={recordedSource} alt="Recorded camera frame"/>;
  const evidenceList=showEvidence&&evidenceRows.length>0&&<dl className="evidence" id="evidence-details" ref={revealOnOpen} title={compareSupport.title||undefined}>
    {evidenceRows.map(r=><div key={r.key} title={r.title}><dt>{r.name}</dt><dd>{r.value}{r.note&&<span className="evidence-note">{r.note}</span>}</dd></div>)}
    {resetButton&&<div><dt className="sr-only">Suggestions</dt><dd>{resetButton}</dd></div>}
  </dl>;
  const chat=(extra)=>!placementUnavailable&&<AssistantBar messages={messages} question={question} onQuestion={setQuestion} askHint={askHint} checkHint={checkHint} running={running}
    onAsk={()=>runNemotron('ask')} onCheck={()=>runNemotron('check')} renderSweep={renderSweep} {...extra}/>;
  const backdrop=<div className="backdrop" aria-hidden="true"><i/><i/><i/></div>;
  const sceneAspect={'--scene-aspect':scene.camera.width+' / '+scene.camera.height};
  const proposed=scenario?.proposedVisibility;

  // Phones and tablets: one page with a pinned header (Peri, the name, Place or Compare), then the two driver views one
  // above the other, the buttons, the credits and the chat. The street map opens as its own layer over the page.
  if(article)return <PeriHost><main className={'app app--article'+(editing?' app--placing':' app--comparing')} style={sceneAspect}>
    {backdrop}
    <div className="article" inert={layer==='open'?true:undefined}>
      <header className="article-head">
        <PeriSlot rank={2} mood={mood} alive className="article-peri"/>
        {/* With more than one place, the name is shown as text with the menu laid invisibly over it: a native menu is as
            wide as its longest place. */}
        <div className="article-brand">{brand}{placeSelect?<span className="place-pick"><span className="place-name" aria-hidden="true">{placeLabel}</span>{placeSelect}</span>:placeControl}</div>
        {!placementUnavailable&&modeChoice('choice choice--mode')}
      </header>
      <div className="article-body">
        {placementUnavailable&&<p className="scene-support-state" role="status">Ground unvalidated · placement unavailable</p>}
        {error&&<p className="inline-error" role="alert">{error}</p>}
        {editing?<>
          {!placementUnavailable&&<div className="article-objects">
            <nav className="choice choice--objects" aria-label="Comparison objects">
              {anchors.map(({key,name})=><button key={key} type="button" className={complete[key]?undefined:'object--missing'} aria-label={name+(complete[key]?'':', not placed yet')} aria-pressed={active===key} onClick={()=>selectObject(key)}><span>{name}</span></button>)}
            </nav>
            {resetButton}
          </div>}
          <div className="recorded-editor article-editor">{editorFrame}</div>
          {!placementUnavailable&&<div className="article-place">{pixelControls}{fieldBubble}</div>}
          <figure className="case article-preview">
            <div className="case-screen">{placePreview}</div>
            <figcaption className="case-head case-head--plain"><span className="case-label">Driver view<span className="case-sep" aria-hidden="true"> · </span><span className="case-at">Dark gaps are unknown</span></span></figcaption>
          </figure>
        </>:scenario?<>
          <div className="article-cases">
            <CaseView scene={scene} scenario={scenario} which="current" name={BASELINE[kindNow][1]} plain onError={setFatal}/>
            <CaseView scene={scene} scenario={scenario} which="proposed" name="Proposed vehicle" plain onError={setFatal}/>
          </div>
          <div className="article-notes">
            {!!compareSupport.text&&<p className="support-line" title={compareSupport.title||undefined}>{compareSupport.text}</p>}
            <div className="article-actions">
              {!reduced&&<button type="button" className="approach-replay" disabled={!replay&&!!running} aria-label={replay?'Stop the approach':undefined}
                title={replay?undefined:'Move the driver from the recorded camera position to the nearest one'} onClick={playApproach}>{replay?'Stop':'Approach'}</button>}
              <button type="button" ref={mapButton} aria-haspopup="dialog" onClick={()=>setLayer('open')}>Map</button>
              {evidenceRows.length>0&&<button type="button" aria-expanded={showEvidence} aria-controls="evidence-details" onClick={()=>setShowEvidence(v=>!v)}>Details</button>}
            </div>
            {evidenceList}
          </div>
        </>:<div className="unavailable">Enter valid comparison values.</div>}
        {credits}
        {chat({plain:true})}
      </div>
    </div>
    {layer&&<div ref={layerRef} className={'map-layer'+(layer==='closing'?' map-layer--closing':'')} role="dialog" aria-modal="true" aria-label="Street map" tabIndex={-1}
      onKeyDown={e=>{if(e.key==='Escape'){e.preventDefault();closeLayer();}}}>
      <div className="map-layer-top">
        <div className="choice" role="group" aria-label="Map or photo">
          <button type="button" aria-pressed={inset==='map'} onClick={()=>setInset('map')}>Map</button>
          <button type="button" aria-pressed={inset==='photo'} onClick={()=>setInset('photo')}>Photo</button>
        </div>
        <button type="button" className="map-layer-close" onClick={closeLayer}>Close</button>
      </div>
      <div className={'map-layer-view map-layer-view--'+inset}><div className="map-layer-fill">{inset==='map'?planView:photoView}</div></div>
      <div className="map-layer-foot">
        {proposed&&<p className="map-layer-result"><span className="case-label">Proposed vehicle</span><span className={'case-word case-word--'+proposed.state} title={proposed.reason}>{STATE_TITLE[proposed.state]}</span></p>}
        {layer==='open'&&<PeriSlot rank={3} mood={mood} alive className="map-layer-peri"/>}
      </div>
      {credits}
    </div>}
  </main></PeriHost>;

  return <PeriHost><main className={'app'+(editing?' app--placing':' app--comparing')} style={sceneAspect}>
    {backdrop}
    <section className="stage" aria-label={editing?(placementUnavailable?'Recorded observation':'Place comparison actors'):'Street map'}>
      {editing?<div className="stage-photo stage-photo--editor"><div className="recorded-editor" style={bubbleStyle}>{editorFrame}{fieldBubble}</div></div>
      :inset==='map'?<div className="stage-map">{planView}</div>
      :<div className="stage-photo">{photoView}</div>}
      <div className="stage-brand">{brand}{placeControl}</div>
      {!editing&&<div className="stage-controls">
        <div className="segmented segmented--small" role="tablist" aria-label="Stage view" data-active={inset==='map'?0:1}>
          <span className="segmented-thumb" aria-hidden="true"/>
          <button type="button" role="tab" aria-selected={inset==='map'} aria-pressed={inset==='map'} onClick={()=>setInset('map')}>Map</button>
          <button type="button" role="tab" aria-selected={inset==='photo'} aria-pressed={inset==='photo'} onClick={()=>setInset('photo')}>Photo</button>
        </div>
        {!reduced&&(replay||scenario&&!running)&&<button type="button" className="approach-replay" aria-label={replay?'Stop the approach':undefined}
          title={replay?undefined:'Move the driver from the recorded camera position to the nearest one'} onClick={playApproach}>
          {replay?<Stop size={12} weight="fill" aria-hidden="true"/>:<Play size={12} weight="fill" aria-hidden="true"/>}<span>{replay?'Stop':'Approach'}</span>
        </button>}
      </div>}
      <div className="stage-credits">{credits}</div>
    </section>
    <aside className="side">
      <header className="side-top">
        {placementUnavailable?<span className="scene-support-state" role="status">Ground unvalidated · placement unavailable</span>:modeChoice('segmented',true)}
      {editing&&!placementUnavailable&&<nav className="object-selectors" aria-label="Comparison objects">
        {anchors.map(({key,name,Icon})=><button key={key} type="button" className={complete[key]?undefined:'object--missing'} aria-label={name+(complete[key]?'':', not placed yet')} aria-pressed={active===key} onClick={()=>selectObject(key)}><Icon size={18} aria-hidden="true"/><span>{name}</span></button>)}
        {resetButton}
      </nav>}
      </header>
      {error&&<p className="inline-error" role="alert">{error}</p>}
      {editing?<div className="side-place">
        {pixelControls}
        <div className="card-head"><span>Driver view</span><span>Dark gaps are unknown</span></div>
        <div className="preview-viewport">{placePreview}</div>
      </div>
      :<div className="side-cases">
        {scenario?<>
          <CaseView scene={scene} scenario={scenario} which="current" name={BASELINE[kindNow][1]} onError={setFatal}/>
          <CaseView scene={scene} scenario={scenario} which="proposed" name="Proposed vehicle" onError={setFatal}/>
        </>:<div className="unavailable">Enter valid comparison values.</div>}
        {scenario&&<div className="side-evidence">
      {!!compareSupport.text&&<p className="support-line" title={compareSupport.title||undefined}>{compareSupport.text}</p>}
      {evidenceRows.length>0&&<button type="button" className="details-toggle" aria-expanded={showEvidence} aria-controls="evidence-details" onClick={()=>setShowEvidence(v=>!v)}>{showEvidence?'Hide details':'Details'}<CaretDown size={12} weight="bold" aria-hidden="true"/></button>}
      {evidenceList}
        </div>}
      </div>}
      {chat({head:<PeriSlot rank={1} mood={mood} alive size={44}/>})}
    </aside>
  </main></PeriHost>;
}

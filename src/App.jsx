import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { SquaresFour, RoadHorizon, Cube, PersonSimple, CheckCircle, Eye, ArrowsHorizontal, CrosshairSimple, XCircle, ArrowCounterClockwise } from '@phosphor-icons/react';
import '@fontsource/geist-mono/400.css';
import '@fontsource/geist-mono/500.css';
import StreetView from './StreetView.jsx';
import AssistantBar from './AssistantBar.jsx';
import DistanceCheck, { sweepText } from './DistanceCheck.jsx';
import { roadPoint, placementFromPixels, placementPreview, createScenario } from './geometry.mjs';
import { runTool, scenarioFacts, checkAnswer, allowedNumbersFrom, withheldNote, declaredLine, vehiclePart, sameVehicle } from './scenario-assistant.mjs';
import { runAssistant } from './nemotron-client.mjs';
import { loadScene } from './scene.mjs';
import { suggestedSetup, isStillSuggested, suggestionStatus } from './suggestions.mjs';

const anchors = [
  {key:'crossing',name:'Crossing',label:'Crossing edge',Icon:RoadHorizon},
  {key:'parking',name:'Vehicle',label:'Current vehicle front',Icon:Cube},
  {key:'target',name:'Person',label:'Pedestrian ground point',Icon:PersonSimple},
];
const meters = n => n.toFixed(2);
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

function DriverComparison({scene,scenario,baseline,onError}) {
  const [split,setSplit]=useState(.5);
  return <section className="comparison-view" aria-label={baseline+' and proposed comparison'}>
    <StreetView scene={scene} scenario={scenario} eye={scenario.eye} mode="compare" split={split} onError={onError}/>
    {split>0&&<div className="comparison-badge comparison-badge--current"><span>{baseline}</span><span className={'visibility-state visibility-state--'+scenario.currentVisibility.state} title={scenario.currentVisibility.reason}>{scenario.currentVisibility.state}</span></div>}
    {split<1&&<div className="comparison-badge comparison-badge--proposed"><span>Proposed</span><span className={'visibility-state visibility-state--'+scenario.proposedVisibility.state} title={scenario.proposedVisibility.reason}>{scenario.proposedVisibility.state}</span></div>}
    <div className="wipe-seam" style={{left:(split*100)+'%'}} aria-hidden="true"/>
    <input className="wipe-control" type="range" aria-label="Comparison split" aria-valuetext={Math.round(split*100)+' percent '+baseline.toLowerCase()+', '+Math.round((1-split)*100)+' percent proposed'} min="0" max="100" step="1" value={split*100} onChange={e=>setSplit(Number(e.target.value)/100)}/>
    <ArrowsHorizontal className="wipe-icon" size={20} weight="bold" style={{left:(split*100)+'%'}} aria-hidden="true"/>
  </section>;
}

export function App() {
  const [catalog,setCatalog]=useState(null),[sceneId,setSceneId]=useState(''),[scene,setScene]=useState(null),[fatal,setFatal]=useState('');
  const [loadAttempt,setLoadAttempt]=useState(0);
  const [recordedImage,setRecordedImage]=useState(null);
  const [pixels,setPixels]=useState({}),[active,setActive]=useState('crossing'),[cursor,setCursor]=useState(['','']);
  const [vehicle,setVehicle]=useState({width:'',height:'',length:''}),[targetHeight,setTargetHeight]=useState('');
  const [input,setInput]=useState(null),[editing,setEditing]=useState(true),[message,setMessage]=useState('');
  // suggestion: the prepared setup for this scene (suggestedSetup), or null when the scene has no suggestions.
  const [suggestion,setSuggestion]=useState(null);
  // The suggested default vehicle front is a declared what-if with no parked vehicle measured, so it is not "current".
  const defaultFront=pixel=>suggestion?.provenance.parking?.status==='default'&&isStillSuggested('parking',pixel,suggestion.provenance);
  const baselineWord=pixel=>defaultFront(pixel)?(isStillSuggested('crossing',pixels.crossing,suggestion.provenance)?`${suggestion.provenance.parking.setbackFt} ft default`:'Default'):'Current';
  const labelOf=a=>a.key==='parking'&&defaultFront(pixels.parking)?'Vehicle front':a.label;
  const [question,setQuestion]=useState(''),[running,setRunning]=useState(null),[reply,setReply]=useState(null),[sweep,setSweep]=useState(null),[declared,setDeclared]=useState([]);
  // runRef: the one in-flight Nemotron run. work: latest scenario values for synchronous tool calls during a run.
  const runRef=useRef(null),work=useRef({});
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
    runRef.current=null;setRunning(null);setReply(null);setSweep(null);setDeclared([]);setSuggestion(null);
    const entry=catalog.find(entry=>entry.id===sceneId);
    loadScene(fetch,sceneId,entry.metadataSha256,controller.signal,entry.suggestions).then(s=>{
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
  const setup=useMemo(()=>{
    if(!scene||anchors.some(a=>!pixels[a.key])||Object.values(vehicle).some(v=>!filled(v))||!filled(targetHeight))return {scenario:null,input:null,error:''};
    try {
      const p=placementFromPixels(scene,pixels),next={pixels,vehicle,targetHeight,proposedSetback:p.baselineSetback,driverDistance:p.captureDistance};
      return {scenario:createScenario(scene,next),input:next,error:''};
    }catch(e){return {scenario:null,input:null,error:e.message};}
  },[scene,pixels,vehicle,targetHeight]);
  const preview=useMemo(()=>{
    if(!scene||scene.measurements?.kind==='indexedSparse')return {actors:null,error:''};
    try{return {actors:placementPreview(scene,pixels,vehicle,targetHeight),error:''};}
    catch(e){return {actors:null,error:e.message};}
  },[scene,pixels,vehicle,targetHeight]);
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
  function resetToSuggested() {resetComparison();setDeclared([]);applySuggestion(suggestion);}
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
  // A distance-check row is a declared what-if: its vehicle and Sightline's computed stop distance go into the editable
  // fields. A row matching the checked vehicle (to the 2 decimals Nemotron was given) restores its exact sizes and credit.
  function applySweepRow(r) {
    if(!input)return;
    const own=!!sweep.own&&sameVehicle(r.vehicle,sweep.own),v=own?sweep.own:r.vehicle;
    const keep=sameVehicle(input.vehicle,v),next={...input,...(!keep&&{vehicle:{...v}}),proposedSetback:r.stopsBlockingAt??input.proposedSetback};
    if(!keep){
      setVehicle({width:String(v.width),height:String(v.height),length:String(v.length)});
      if(!own)setDeclared(p=>[...p,vehiclePart(r.label,v)]);
    }
    setInput(next);setMessage('');
    // A distance-check answer still describes this panel; any other answer was about the previous values.
    setReply(p=>p?.mode==='check'&&p.basis===input?{...p,basis:next}:null);
  }
  // Explicit Ask/Check only; never from effects, drags, sliders or numeric edits. One run at a time, no retry.
  async function runNemotron(mode) {
    if(running||(mode==='ask'?askHint||!question.trim():checkHint))return;
    // run.basis is the comparison input the reply describes; the reply is hidden once the input changes.
    const s=scene,controller=new AbortController(),base=input??setup.input,run={basis:input,facts:null,sweep:null,toolError:null};
    runRef.current?.abort();runRef.current=controller;
    // Facts describe only the input on screen. An invalid input sends its problem, never the baseline setup's facts.
    const known=input?scenario:setup.scenario,problem=input?result.error:setup.error;
    // Without a computed scenario: only what is declared so far, never pixels or coordinates.
    const context=known?scenarioFacts(s,known):{scene:s.id,units:s.units,placed:anchors.filter(a=>pixels[a.key]).map(a=>a.name.toLowerCase()),
      vehicle_m:Object.fromEntries(Object.entries(input?.vehicle??vehicle).filter(([,v])=>positive(v)).map(([k,v])=>[k,Number(v)])),
      ...(positive(input?.targetHeight??targetHeight)&&{targetHeight_m:Number(input?.targetHeight??targetHeight)}),...(problem&&{problem})};
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
      if(out.parts)setDeclared(p=>[...p,...out.parts]);
      if(out.facts)run.facts=out.facts;
      w.input=out.input;run.basis=out.input;setInput(out.input);setEditing(false);setMessage('');
      if(out.sweep){run.sweep=out.sweep.results;setSweep({scene:s,...out.sweep,reveal:true});}
      return out.result;
    };
    const done=r=>setReply({mode,basis:run.basis,...r});
    try {
      const out=await runAssistant({mode,question:question.trim(),context},executeTool,{signal:controller.signal});
      if(runRef.current!==controller)return;
      if(out.error){done({error:'Nemotron didn\'t finish.'});return;}
      const checked=checkAnswer(out.answer,allowedNumbersFrom(context,...out.trace.map(t=>t.result)));
      done(checked.ok?{text:out.answer.trim()}:{computed:computedLine(run,baselineWord(run.basis?.pixels?.parking)),note:withheldNote(checked)});
    }catch(e){
      if(runRef.current===controller)done({error:ASSISTANT_ERRORS[e?.code]??e?.message??'The Nemotron request failed.'});
    }finally{
      if(runRef.current===controller){runRef.current=null;setRunning(null);}
    }
  }
  if(fatal)return <main className="app"><h1>Sightline</h1><section className="unavailable" role="alert"><h2>Scene unavailable</h2><p>{fatal}</p><button onClick={()=>sceneId?setLoadAttempt(n=>n+1):location.reload()}>Retry loading</button></section></main>;
  if(!scene)return <main className="app"><h1>Sightline</h1><p role="status">Loading measured scene…</p></main>;
  const placementUnavailable=scene.measurements?.kind==='indexedSparse';
  const recordedUrl=recordedImage?.scene===scene?recordedImage.url:undefined;
  const recordedSource=scene.measurements?recordedUrl:'/scenes/'+scene.id+'/recorded.jpg';
  const ownsImage=image=>image.isConnected&&image.getAttribute('src')===recordedSource;
  const evidence=scene.measurements?`${scene.reconstruction?'NVIDIA estimate · projected references':'Projected LiDAR'}`:`${scene.reconstruction?'NVIDIA estimate · measured anchors':'Measured depth'} ${(scene.coverage*100).toFixed(1)}%`;
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
    :setup.error?'Check distances needs a vehicle and person that fit the measured road':'Enter vehicle size and person height to check distances';
  // Distance results stay tied to the scene, points, person height and driver position they were computed with.
  const sweepShown=!editing&&input&&sweep?.scene===scene&&sweep.pixels===input.pixels&&sweep.targetHeight===Number(input.targetHeight)&&sweep.driverDistance===Number(input.driverDistance);
  // Nemotron's lines describe the input they were made for; any later change hides them.
  const shownReply=reply&&reply.basis===input?reply:null;
  // Credit follows the values on screen: the comparison input, or the fields while they are being edited.
  const declaredText=declaredLine(declared,input??{vehicle,targetHeight});
  const overridden=declared.flatMap(d=>d.key==='vehicle'?['vehicle']:d.key==='person'?['targetHeight']:[]);
  const suggested=suggestion&&suggestionStatus(scene.suggestions,suggestion,{pixels,vehicle,targetHeight,overridden});
  // Label of the vehicle on screen: the latest Nemotron declaration still in the fields, else a still-suggested default.
  const onScreenVehicleLabel=()=>{const w=work.current.input?.vehicle??work.current.vehicle,v=declared.findLast(d=>d.key==='vehicle');
    if(v&&sameVehicle(v.value,w))return v.label;
    return suggestion&&!overridden.includes('vehicle')&&isStillSuggested('vehicle',w,suggestion.provenance)?scene.suggestions.vehicle.label:null;};
  const suggestedTag=(key,value)=>suggestion&&!overridden.includes(key)&&isStillSuggested(key,value,suggestion.provenance)?<span className="suggested-tag"><span className="sr-only">, </span>Suggested · {suggestion.provenance[key].status}</span>:null;
  const selectedLabel=labelOf(selected);
  return <main className="app" style={{'--scene-aspect':scene.camera.width+' / '+scene.camera.height}}>
    <header className="app-header">
      <a className="brand" href="/" aria-label="Sightline home"><SquaresFour size={26} weight="fill"/><span>Sightline</span></a>
      {placementUnavailable?<span className="scene-support-state" role="status">Ground unvalidated · placement unavailable</span>:<nav className="object-selectors" aria-label="Comparison objects">{anchors.map(({key,name,Icon})=><button key={key} aria-label={name} aria-pressed={active===key} onClick={()=>selectObject(key)}><Icon size={20}/><span>{name}</span>{complete[key]&&<CheckCircle className="completion-icon" size={13} weight="fill" aria-label="Complete"/>}</button>)}</nav>}
      {!placementUnavailable&&<div className="header-action">{editing?<button className="compare-button" disabled={!setup.scenario} onClick={compare}>Compare positions</button>:<button onClick={()=>setEditing(true)}>Edit placements</button>}</div>}
    </header>
    {!placementUnavailable&&<AssistantBar question={question} onQuestion={setQuestion} askHint={askHint} checkHint={checkHint} running={running} onAsk={()=>runNemotron('ask')} onCheck={()=>runNemotron('check')} declared={declaredText} reply={shownReply}/>}
    {!placementUnavailable&&suggested&&(suggested.parts.length>0||suggested.edited)&&<div className="suggestion-line">
      {suggested.parts.length>0&&<span>Suggested{suggested.parts.map(p=><Fragment key={p.key}>{' '}<span title={p.title}>· {p.text}</span></Fragment>)}</span>}
      {suggested.edited&&<button type="button" className="suggestion-reset" onClick={resetToSuggested}><ArrowCounterClockwise size={13} aria-hidden="true"/>Reset to suggested</button>}
    </div>}
    {catalog.length>1&&<select aria-label="Measured scene" value={sceneId} onChange={e=>setSceneId(e.target.value)}>{catalog.map(e=><option key={e.id} value={e.id}>{e.label}</option>)}</select>}
    <div className="workspace">
      {error&&<p className="inline-error" role="alert">{error}</p>}
      {editing?<div className="placement-layout">
        <section className="placement-panel" aria-label={placementUnavailable?'Recorded observation':'Place comparison actors'}>
          <div className="recorded-editor" style={bubbleStyle}>
            <div className="recorded-frame">
            <div className={'recorded-picker'+(placementUnavailable?' recorded-picker--readonly':'')} role={placementUnavailable?undefined:'button'} tabIndex={placementUnavailable?undefined:0} aria-label={placementUnavailable?undefined:'Select '+selectedLabel+' on measured road. Arrow keys move the pixel cursor; Enter selects.'} title={placementUnavailable?undefined:'Place '+selectedLabel.toLowerCase()+' on a supported road pixel'}
              onClick={placementUnavailable?undefined:e=>selectPoint(imagePixel(e,scene))}
              onPointerMove={placementUnavailable?undefined:e=>{if(e.pointerType!=='mouse')return;const next=imagePixel(e,scene);setCursor(previous=>previous.every((v,i)=>Number(v)===next[i]&&filled(v))?previous:next);setMessage('');}}
              onKeyDown={placementUnavailable?undefined:e=>{if(e.key==='Enter'){e.preventDefault();setCursorPoint();}else if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();if(cursor.some(v=>!filled(v))){setMessage('Enter a pixel coordinate first, then use arrow keys.');return;}setCursor(([u,v])=>[Number(u)+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0),Number(v)+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0)]);setMessage('');}}}>
              <img src={recordedSource} alt="Recorded street" onError={e=>{if(ownsImage(e.currentTarget))setFatal('Recorded image unavailable.');}} onLoad={e=>{if(ownsImage(e.currentTarget)&&(e.currentTarget.naturalWidth!==scene.camera.width||e.currentTarget.naturalHeight!==scene.camera.height))setFatal('Recorded image dimensions disagree with calibration.');}}/>
              {!scene.measurements&&<img className="support-mask" src={'/scenes/'+scene.id+'/support.png'} alt="" onError={()=>setFatal('Depth support mask unavailable.')}/>}
              {!placementUnavailable&&<div className="recorded-actors"><StreetView scene={scene} preview={preview.actors} mode="recorded" onError={setFatal}/></div>}
            </div>
            {!placementUnavailable&&anchors.filter(a=>pixels[a.key]).map(({key,name,Icon})=><button key={key} className={'anchor-marker'+(active===key?' anchor-marker--selected':'')} tabIndex={active===key?-1:0} aria-label={'Edit '+name+' placement'} onClick={()=>selectObject(key)} style={{left:((pixels[key][0]+.5)/scene.camera.width*100)+'%',top:((pixels[key][1]+.5)/scene.camera.height*100)+'%'}}><Icon size={17} weight="bold"/></button>)}
            {!placementUnavailable&&pending&&<span className={'pixel-cursor'+(candidate.valid?' pixel-cursor--supported':' pixel-cursor--unsupported')} role="img" aria-label={candidate.valid?'Measured road point':candidate.error} style={{left:((Number(cursor[0])+.5)/scene.camera.width*100)+'%',top:((Number(cursor[1])+.5)/scene.camera.height*100)+'%'}}>{candidate.valid?<CrosshairSimple size={28} weight="bold"/>:<XCircle size={28} weight="fill"/>}</span>}
            </div>
            {!placementUnavailable&&active!=='crossing'&&<div className={'field-bubble'+(point?' field-bubble--anchored':'')} aria-label={selected.name+' dimensions'}>
              <div className="bubble-title">{selected.name==='Vehicle'?'Vehicle size':'Person height'} <span>m</span></div>
              {active==='parking'?suggestedTag('vehicle',vehicle):suggestedTag('targetHeight',targetHeight)}
              {active==='parking'?['width','height','length'].map(key=><label key={key}><span>{key}</span><input type="number" aria-label={'Vehicle '+key+' in meters'} aria-invalid={filled(vehicle[key])&&!positive(vehicle[key])} min="0" step="any" value={vehicle[key]} onChange={e=>{setVehicle(v=>({...v,[key]:e.target.value}));resetComparison();}}/></label>):<input type="number" aria-label="Target height in meters" aria-invalid={filled(targetHeight)&&!positive(targetHeight)} min="0" step="any" value={targetHeight} onChange={e=>{setTargetHeight(e.target.value);resetComparison();}}/>}
            </div>}
          </div>
          {!placementUnavailable&&<div className="pixel-controls"><span className="selected-label"><selected.Icon size={17}/>{selectedLabel}{suggestedTag(active,pixels[active])}</span><div className="pixel-inputs"><span>Pixel</span>{['X','Y'].map((name,i)=><label key={name}><span className="sr-only">Pixel {name}</span><input aria-label={'Pixel '+name} aria-invalid={!!candidate.error} aria-describedby="point-feedback" type="number" min="0" max={i?scene.camera.height-1:scene.camera.width-1} step="1" value={cursor[i]} onKeyDown={e=>{if(e.key==='Enter')setCursorPoint();}} onChange={e=>{setCursor(p=>p.map((v,j)=>i===j?e.target.value:v));setMessage('');}}/></label>)}<button disabled={!candidate.valid} aria-describedby="point-feedback" onClick={setCursorPoint}>Set point</button></div><span id="point-feedback" className="sr-only" role="status">{candidate.valid?'Measured road point':candidate.error}</span></div>}
        </section>
        <section className="driver-preview" aria-label="Placement driver preview"><div className="view-heading"><h2><Eye size={18}/>Driver view</h2></div><div className="preview-viewport"><StreetView scene={scene} preview={preview.actors} onError={setFatal}/></div><div className="evidence-key"><span>{evidence}</span><span>Dark gaps · unknown</span></div></section>
      </div>:<>
        {scenario?<DriverComparison scene={scene} scenario={scenario} baseline={baselineWord(input.pixels.parking)} onError={setFatal}/>:<div className="unavailable">Enter valid comparison values.</div>}
        <div className="comparison-tools">
          <section className="approach-panel"><div className="control-heading"><h2>Driver position</h2>{scenario&&<span>{meters(scenario.driverDistance)} m</span>}</div>{scenario&&<><input aria-label="Driver position" type="range" min={scenario.limits.minimumDriverDistance} max={scenario.limits.captureDistance} step="any" value={input.driverDistance} onChange={e=>setInput(p=>({...p,driverDistance:Number(e.target.value)}))}/><div className="range-ends"><span>Near crossing</span><span>Recorded camera</span></div></>}</section>
          <section className="plan-panel"><div className="control-heading"><h2>Proposed movement</h2><label className="setback-input"><span className="sr-only">Vehicle distance from crossing</span><input aria-label="Exact vehicle distance in meters" type="number" min="0" step="any" value={typeof input.proposedSetback==='number'?Number(input.proposedSetback.toFixed(2)):input.proposedSetback} onChange={e=>moveVan(e.target.value)}/><span>m</span></label></div><div className="plan-viewport">{scenario?<StreetView scene={scene} scenario={scenario} eye={scenario.eye} mode="plan" baselineLabel={defaultFront(input.pixels.parking)?'Default vehicle':'Current vehicle'} onSetbackChange={moveVan} onError={setFatal}/>:<div className="unavailable">Comparison unavailable</div>}</div>{scenario&&<input aria-label="Vehicle distance from crossing" type="range" min={scene.processing.roadToleranceMeters} max={scenario.limits.maximumSetback} step="any" value={Number(input.proposedSetback)} onChange={e=>moveVan(Number(e.target.value))}/>}</section>
          {sweepShown&&<DistanceCheck results={sweep.results} vehicle={input.vehicle} setback={input.proposedSetback} disabled={!!running} reveal={sweep.reveal&&!running} onRevealed={()=>setSweep(p=>p&&{...p,reveal:false})} onApply={applySweepRow}/>}
        </div>
        <div className="evidence-key"><span>{evidence}{!scene.reconstruction&&!scene.measurements&&' · dark gaps unknown'}</span><span title="This geometric comparison is not a safety assessment.">Declared what-if actors</span></div>
      </>}
    </div>
  </main>;
}

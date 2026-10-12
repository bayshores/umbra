import { useEffect, useRef } from 'react';
import { FOOT, dims, sameVehicle, sweepPhrase, sweepMarks } from './scenario-assistant.mjs';

const MARKS=[{ft:15,m:15*FOOT},{ft:20,m:20*FOOT}];
const LEGEND=[['blocked','Blocked'],['unknown','Unknown'],['overlap','Overlap']];
const short=n=>String(Number(n.toFixed(2)));
// The app's own sentence for one swept vehicle: the shared phrase, read only from the geometry sweep rows. Word joiners
// keep each range ("2.58-19.82") on one line.
export const sweepText=r=>{const t=sweepPhrase(r);return (t[0].toUpperCase()+t.slice(1)).replace(/(\d)-(\d)/g,'$1\u2060-\u2060$2');};

// Each tested setback colours the strip up to the next tested setback.
function segments(rows) {
  const out=[];
  rows.forEach((r,i)=>{
    const to=rows[i+1]?.setback??r.setback;
    if(out.at(-1)?.state===r.state)out.at(-1).to=to;else out.push({state:r.state,from:r.setback,to});
  });
  return out;
}

// codeNote: false on mapped-ground scenes, whose current vehicle is a stationary annotated vehicle (no parking reference).
export default function DistanceCheck({results,vehicle,setback,codeNote=true,disabled,reveal,onRevealed,onApply}) {
  const panel=useRef(null);
  // A finished check brings the panel into view once; focus stays on the control that started it.
  useEffect(()=>{
    if(!reveal)return;
    panel.current?.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
    onRevealed();
  },[reveal]);
  return <section ref={panel} className="distance-panel" aria-label="Distance check">
    <div className="control-heading"><h2>Distance check</h2><span className="distance-legend" aria-hidden="true">{LEGEND.map(([state,name])=><span key={state}><span className={'distance-swatch distance-fill--'+state}/>{name}</span>)}</span></div>
    <div className="distance-rows">{results.map((r,i)=>{
      const name=<span className="distance-name"><span>{r.label}</span><span>{dims(r.vehicle)}</span></span>;
      if(r.error)return <div key={i} className="distance-row distance-row--error">{name}<span className="distance-result">{r.error}</span></div>;
      const max=r.range[1],pct=n=>(n/max*100)+'%',applied=sameVehicle(vehicle,r.vehicle);
      // Without a stop distance the current setback is kept, so it must already fit this vehicle.
      const fits=r.stopsBlockingAt!==null||(Number(setback)>0&&Number(setback)<=max+1e-9),reason=`Set proposed within ${short(max)} m to apply`;
      return <button key={i} className={'distance-row'+(applied?' distance-row--applied':'')} aria-pressed={applied} aria-disabled={!fits||undefined} title={fits?'Apply this vehicle and distance':reason} disabled={disabled} onClick={()=>{if(fits)onApply(r);}}>
        {name}
        <span className="distance-strip" aria-hidden="true">
          {segments(r.rows).map((s,j)=><span key={j} className={'distance-segment distance-fill--'+s.state} style={{left:pct(s.from),width:pct(Math.max(s.to-s.from,max/400))}}/>)}
          {MARKS.filter(k=>k.m>=r.range[0]&&k.m<=max).map(k=><span key={k.ft} className={'distance-tick distance-tick--'+k.ft} style={{left:pct(k.m)}}><span>{k.ft} ft</span></span>)}
        </span>
        <span className="distance-result">{sweepText(r)}{!fits&&' · '+reason}</span>
        <span className="distance-result distance-marks">{sweepMarks(r)}</span>
      </button>;
    })}</div>
    {codeNote&&<p className="distance-note">20 ft: California Vehicle Code §22500(n) parking distance near crosswalks, reference only</p>}
  </section>;
}

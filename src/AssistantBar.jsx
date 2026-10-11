import { Sparkle, ArrowRight, Ruler } from '@phosphor-icons/react';

const STATUS={ask:'Nemotron is setting up…',check:'Nemotron is checking distances…'};
export const SOURCE='NVIDIA Nemotron · Nebius Token Factory';

// One compact row: Nemotron only declares what-if values; Sightline computes every result shown.
// While a run is in progress the controls are aria-disabled rather than disabled, so keyboard focus stays where it was.
export default function AssistantBar({question,onQuestion,askHint,checkHint,running,onAsk,onCheck,declared,reply}) {
  const busy=!!running,ready=!askHint&&!busy;
  // The source label sits on the line that carries Nemotron's own contribution.
  const source=<span className="assistant-source"> · {SOURCE}</span>;
  return <section className="assistant" aria-label="Nemotron assistant">
    <div className={'assistant-bar'+(ready?' assistant-bar--ready':'')}>
      <span className="assistant-label"><Sparkle size={16} weight="fill" aria-hidden="true"/>Nemotron</span>
      <form className="assistant-form" onSubmit={e=>{e.preventDefault();if(ready&&question.trim())onAsk();}}>
        <input type="text" aria-label="Describe a what-if for Nemotron" maxLength={500} value={question} disabled={!!askHint} readOnly={busy} aria-disabled={busy||undefined}
          placeholder={askHint||'e.g. box truck 10 ft back, child'} title={askHint||undefined} onChange={e=>onQuestion(e.target.value)}/>
        <button className="assistant-submit" type="submit" aria-label="Ask Nemotron" title={askHint||'Ask Nemotron'} disabled={!!askHint||!question.trim()} aria-disabled={busy||undefined}><ArrowRight size={17} weight="bold"/></button>
      </form>
      <button className="assistant-check" type="button" title={checkHint||'Check distances with Nemotron'} disabled={!!checkHint} aria-disabled={busy||undefined} onClick={()=>{if(!busy)onCheck();}}><Ruler size={16}/>Check distances</button>
    </div>
    <div className="assistant-output" role="status">
      {running?<p>{STATUS[running]}</p>:<>
        {!!declared&&<p className="assistant-declared">Declared by Nemotron · {declared}{!reply?.text&&!reply?.note&&source}</p>}
        {!!reply?.text&&<p>{reply.text}{source}</p>}
        {!!reply?.computed&&<p>{reply.computed}</p>}
        {!!reply?.note&&<p className="assistant-note">{reply.note}{source}</p>}
        {!!reply?.error&&<p className="assistant-note">{reply.error}</p>}
        {!reply&&!askHint&&!!checkHint&&<p className="assistant-note">{checkHint}</p>}
      </>}
    </div>
  </section>;
}

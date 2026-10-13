import { useEffect, useRef } from 'react';
import { ArrowUp, Ruler } from '@phosphor-icons/react';
import Bloop from './Bloop.jsx';
import DotText from './DotText.jsx';

const STATUS = { ask: 'Bloop is setting up…', check: 'Bloop is checking distances…' };
export const SOURCE = 'NVIDIA Nemotron · Nebius Token Factory';
const STATE_TITLE = { blocked: 'Blocked', unknown: 'Unknown', overlap: 'Overlap' };

// The chat with Bloop, Sightline's assistant persona. Bloop's words come from NVIDIA Nemotron through Nebius Token
// Factory (named in the header); Nemotron only declares what-if values, and Sightline computes every result, which the
// log shows as its own "Sightline" lines. The one Bloop on screen lives in this header and shows the app's mood.
// While a run is in progress the controls are aria-disabled rather than disabled, so keyboard focus stays where it was.
export default function AssistantBar({ messages, question, onQuestion, askHint, checkHint, running, onAsk, onCheck, mood, renderSweep }) {
  const busy = !!running, ready = !askHint && !busy, log = useRef(null), field = useRef(null);
  useEffect(() => {
    const el = log.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, running]);
  // The message box starts one line tall and grows with its text, up to its CSS max-height.
  useEffect(() => {
    const el = field.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [question]);
  const send = () => { if (ready && question.trim()) onAsk(); };
  return <section className="chat" aria-label="Chat with Bloop">
    <header className="chat-head">
      <Bloop key={mood.key} mood={mood.name} loop={mood.loop} size={64} className="chat-bloop"/>
      <div className="chat-title"><h2>Bloop</h2><p className="chat-source">runs on {SOURCE}</p></div>
    </header>
    <ol className="chat-log" ref={log} aria-live="polite" aria-label="Messages">
      {!messages.length && !running && <li className="chat-empty">Try “box truck 10 ft back, child”</li>}
      {messages.map(m => <li key={m.id} className={'msg msg--' + m.from + (m.kind ? ' msg--' + m.kind : '')}>
        <span className="msg-author">{m.from === 'you' ? 'You' : m.from === 'bloop' ? 'Bloop' : 'Sightline'}</span>
        {m.kind === 'result' ? <p className="msg-result">
            <DotText text={STATE_TITLE[m.state] ?? m.state} pitch={3} className={'dot-text--' + m.state} animate/>
            <span className="msg-result-at">{m.setback} m</span></p>
          : m.kind === 'sweep' ? (renderSweep(m) ?? <p className="msg-bubble">{m.text}</p>)
          : <p className="msg-bubble">{m.text}</p>}
        {!!m.declared && <p className="msg-declared">Set {m.declared}</p>}
      </li>)}
      {running && <li className="msg msg--bloop msg--typing"><span className="msg-author">Bloop</span>
        <p className="msg-bubble"><span className="typing" aria-hidden="true"><i/><i/><i/></span>{STATUS[running]}</p></li>}
    </ol>
    <form className="composer" onSubmit={e => { e.preventDefault(); send(); }}>
      <div className="composer-field">
        <button className="composer-check" type="button" title={checkHint || 'Ask Bloop to check distances'} disabled={!!checkHint} aria-disabled={busy || undefined}
          onClick={() => { if (!busy) onCheck(); }}><Ruler size={15} aria-hidden="true"/>Check distances</button>
        <textarea ref={field} aria-label="Message Bloop" rows={1} maxLength={500} value={question} disabled={!!askHint} readOnly={busy} aria-disabled={busy || undefined}
          placeholder="Message Bloop" title={askHint || undefined} onChange={e => onQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}/>
        <button className="composer-send" type="submit" aria-label="Send to Bloop" title={askHint || 'Send'} disabled={!!askHint || !question.trim()} aria-disabled={busy || undefined}>
          <ArrowUp size={17} weight="bold"/></button>
      </div>
      {!!askHint && <p className="composer-hint">{askHint}</p>}
      {!askHint && !!checkHint && <p className="composer-hint">{checkHint}</p>}
    </form>
  </section>;
}

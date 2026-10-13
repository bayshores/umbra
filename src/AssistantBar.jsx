import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Ruler } from '@phosphor-icons/react';
import DotText from './DotText.jsx';
import { prefersReducedMotion, usePageVisible, useReducedMotion } from './motion.mjs';

const STATUS = { ask: 'Setting up…', check: 'Checking distances…' };
export const SOURCE = 'NVIDIA Nemotron · Nebius Token Factory';
const STATE_TITLE = { blocked: 'Blocked', unknown: 'Unknown', overlap: 'Overlap' };
// Requests Peri can set up (declare_scenario, or sweep_setbacks for "how far back"), shown one at a time in the empty chat.
const EXAMPLES = ['box truck 10 ft back, child', 'SUV 6 m back, adult', 'how far back must a delivery van be?', 'pickup 20 ft back, wheelchair user'];

// The empty chat's suggestion changes every few seconds while the tab is visible; with Reduce Motion it stays on the
// first one. Screen readers get the first one only, so the polite live log is not re-read on every change.
function ExampleRequest() {
  const reduced = useReducedMotion(), visible = usePageVisible(), [shown, setShown] = useState(0);
  useEffect(() => {
    if (reduced || !visible) return undefined;
    const timer = setTimeout(() => setShown(n => (n + 1) % EXAMPLES.length), 4200);
    return () => clearTimeout(timer);
  }, [shown, reduced, visible]);
  const i = reduced ? 0 : shown;
  return <li className="chat-empty">
    <span className="sr-only">Try “{EXAMPLES[0]}”</span>
    <span key={i} className="chat-example" aria-hidden="true">Try “{EXAMPLES[i]}”</span>
  </li>;
}

// The chat with Peri, Sightline's assistant persona. Peri's words come from NVIDIA Nemotron through Nebius Token
// Factory (named in the header); Nemotron only declares what-if values, and Sightline computes every result, which the
// log shows as its own "Sightline" lines. head: shown before the title (on desktop, the place where Peri sits).
// plain (phones and tablets): the chat is part of the page, with plain result words, no icons and no send button
// (Enter sends); the page, not the log, scrolls, and it stays at the bottom as the conversation grows if it was there.
// While a run is in progress the controls are aria-disabled rather than disabled, so keyboard focus stays where it was.
export default function AssistantBar({ messages, question, onQuestion, askHint, checkHint, running, onAsk, onCheck, renderSweep, head, plain = false }) {
  const busy = !!running, ready = !askHint && !busy, log = useRef(null), field = useRef(null), atBottom = useRef(true);
  useEffect(() => {
    if (!plain) return undefined;
    const root = document.documentElement, check = () => { atBottom.current = innerHeight + scrollY >= root.scrollHeight - 120; };
    check();
    addEventListener('scroll', check, { passive: true });
    return () => removeEventListener('scroll', check);
  }, [plain]);
  useEffect(() => {
    if (!plain) { const el = log.current; if (el) el.scrollTop = el.scrollHeight; return; }
    if (messages.length && atBottom.current) scrollTo({ top: document.documentElement.scrollHeight, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }, [messages.length, running, plain]);
  // The message box starts one line tall and grows with its text, up to its CSS max-height.
  useEffect(() => {
    const el = field.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [question]);
  const send = () => { if (ready && question.trim()) onAsk(); };
  return <section className="chat" aria-label="Chat with Peri">
    <header className="chat-head">
      {head}
      <div className="chat-title"><h2>Peri</h2><p className="chat-source">runs on {SOURCE}</p></div>
    </header>
    <ol className="chat-log" ref={log} aria-live="polite" aria-label="Messages">
      {!messages.length && !running && <ExampleRequest/>}
      {messages.map(m => <li key={m.id} className={'msg msg--' + m.from + (m.kind ? ' msg--' + m.kind : '')}>
        <span className="msg-author">{m.from === 'you' ? 'You' : m.from === 'peri' ? 'Peri' : 'Sightline'}</span>
        {m.kind === 'result' ? <p className="msg-result">
            {plain ? <span className={'msg-result-word msg-result-word--' + m.state}>{STATE_TITLE[m.state] ?? m.state}</span>
              : <DotText text={STATE_TITLE[m.state] ?? m.state} pitch={3} className={'dot-text--' + m.state} animate/>}
            <span className="msg-result-at">{m.setback} m</span></p>
          : m.kind === 'sweep' ? (renderSweep(m) ?? <p className="msg-bubble">{m.text}</p>)
          : <p className="msg-bubble">{m.text}</p>}
        {!!m.declared && <p className="msg-declared">Set {m.declared}</p>}
      </li>)}
      {running && <li className="msg msg--peri msg--typing"><span className="msg-author">Peri</span>
        <p className="msg-bubble">{!plain && <span className="typing" aria-hidden="true"><i/><i/><i/></span>}{STATUS[running]}</p></li>}
    </ol>
    <form className="composer" onSubmit={e => { e.preventDefault(); send(); }}>
      <div className="composer-field">
        <button className="composer-check" type="button" title={checkHint || 'Ask Peri to check distances'} disabled={!!checkHint} aria-disabled={busy || undefined}
          onClick={() => { if (!busy) onCheck(); }}>{!plain && <Ruler size={15} aria-hidden="true"/>}Check distances</button>
        <textarea ref={field} aria-label="Message Peri" rows={1} maxLength={500} value={question} disabled={!!askHint} readOnly={busy} aria-disabled={busy || undefined}
          placeholder="Message Peri" title={askHint || undefined} enterKeyHint="send" onChange={e => onQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}/>
        {!plain && <button className="composer-send" type="submit" aria-label="Send to Peri" title={askHint || 'Send'} disabled={!!askHint || !question.trim()} aria-disabled={busy || undefined}>
          <ArrowUp size={17} weight="bold"/></button>}
      </div>
      {!!askHint && <p className="composer-hint">{askHint}</p>}
      {!askHint && !!checkHint && <p className="composer-hint">{checkHint}</p>}
    </form>
  </section>;
}

import { useEffect, useId, useState } from 'react';
import BLOOP from './bloop-frames.json';

// Bloop, Sightline's mascot: vector frames traced (potrace) from mood sheets generated with OpenAI image generation in Codex. Each frame is five
// layers (body, two rim crescents, highlight, face) drawn with the brand palette. Moods: idle, bounce (working), blocked,
// unsure (an Unknown result: Bloop never has a "sees clearly" mood, because Sightline never reports Clear) and sleepy
// (empty or unavailable). Frames advance with timers only while the page is visible; reduced motion shows the first frame.
const FILLS = { blue: '#A9C7E8', pink: '#E8B9C6', white: '#FFFFFF', ink: '#12141C' };
// With loop off, a mood stops on its clearest frame: squished for blocked, puzzled for unsure; idle stays still (the logo).
const HOLD = { blocked: 3, unsure: 4, idle: 0 };
const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export default function Bloop({ mood = 'idle', size = 64, loop = true, className = '', label }) {
  const state = BLOOP.states[mood] ?? BLOOP.states.idle;
  const [frame, setFrame] = useState(0), [visible, setVisible] = useState(() => typeof document === 'undefined' || !document.hidden);
  // A gradient id safe inside url(#…) whatever characters React's useId uses.
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const end = loop ? -1 : Math.min(HOLD[mood] ?? state.frames.length - 1, state.frames.length - 1);
  useEffect(() => setFrame(0), [mood]);
  useEffect(() => {
    const change = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', change);
    return () => document.removeEventListener('visibilitychange', change);
  }, []);
  useEffect(() => {
    if (!visible || frame === end) return undefined;
    // Reduced motion: no frame changes; a reaction shows its held frame right away.
    if (reducedMotion()) { if (end >= 0) setFrame(end); return undefined; }
    const timer = setTimeout(() => setFrame(n => (n + 1) % state.frames.length), state.durations[frame]);
    return () => clearTimeout(timer);
  }, [mood, frame, end, visible, state]);
  const f = state.frames[Math.min(frame, state.frames.length - 1)];
  const [x, y, w, h] = BLOOP.viewBox;
  return <svg className={'bloop ' + className} viewBox={`${x} ${y} ${w} ${h}`} width={size} height={size * h / w}
    role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : 'true'} focusable="false">
    <defs>
      {/* The traced paths are y-flipped (potrace), so the light sits at the bottom of their local box: top left on screen. */}
      <radialGradient id={'bloop-pearl-' + id} cx=".42" cy=".62" r=".78">
        <stop offset="0" stopColor="#EFEBF8"/><stop offset=".6" stopColor="#DCD5EE"/><stop offset="1" stopColor="#C3BADF"/>
      </radialGradient>
    </defs>
    <g transform={`matrix(${f.matrix.join(' ')})`}>
      <path fill={`url(#bloop-pearl-${id})`} d={f.paths.body}/>
      {['blue', 'pink', 'white', 'ink'].map(k => f.paths[k] && <path key={k} fill={FILLS[k]} d={f.paths[k]}/>)}
    </g>
  </svg>;
}

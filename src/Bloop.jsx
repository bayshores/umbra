import { useEffect, useId, useRef, useState } from 'react';
import BLOOP from './bloop-frames.json';
import { usePageVisible, useReducedMotion } from './motion.mjs';
import { REST, faceParts } from './bloop-face.mjs';

// Bloop, umbra's mascot: vector frames traced (potrace) from mood sheets generated with OpenAI image generation in Codex. Each frame is five
// layers (body, two rim crescents, highlight, face) drawn with the brand palette. Moods: idle, bounce (working), blocked,
// unsure (an Unknown result: Bloop never has a "sees clearly" mood, because umbra never reports Clear) and sleepy
// (empty or unavailable). Frames advance with timers only while the page is visible; reduced motion shows the first frame.
// The chat's Bloop is alive: after a reaction he settles onto a calm face, blinks now and then, glances toward a drag or
// the field you type in, and dozes off after a minute with no input. All of that stops while the tab is hidden and stays
// off with Reduce Motion.
const FILLS = { blue: '#A9C7E8', pink: '#E8B9C6', white: '#FFFFFF', ink: '#12141C' };
// With loop off, a mood stops on its clearest frame: squished for blocked, puzzled for unsure; idle stays still (the logo).
const HOLD = { blocked: 3, unsure: 4, idle: 0 };
const SETTLE_MS = 2400, DOZE_MS = 60000, GLANCE_MS = 1200;
// Any of these is input: it wakes a dozing Bloop and restarts the one-minute clock.
const INPUT = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'];
// How far the face moves toward what you are doing, in viewBox units (the body is about 400 wide), and the distance in
// screen pixels at which the full offset is reached.
const LOOK = [26, 16], LOOK_RANGE = 160;
const listen = (types, handler) => {
  types.forEach(type => document.addEventListener(type, handler, { capture: true, passive: true }));
  return () => types.forEach(type => document.removeEventListener(type, handler, { capture: true }));
};

export default function Bloop({ mood = 'idle', size = 64, loop = true, className = '', label, alive = false }) {
  const reduced = useReducedMotion(), visible = usePageVisible(), live = alive && !reduced && visible;
  const [frame, setFrame] = useState(0), [settled, setSettled] = useState(false), [dozing, setDozing] = useState(false);
  // blink counts blinks: each new value redraws the eyes, which restarts their CSS blink.
  const [blink, setBlink] = useState(0), dozingRef = useRef(false), svg = useRef(null), look = useRef(null);
  const base = BLOOP.states[mood] ?? BLOOP.states.idle, state = dozing ? BLOOP.states.sleepy : base, last = base.frames.length - 1;
  // A live idle Bloop rests on his first frame instead of looping the idle frames.
  const loops = loop && !(alive && mood === 'idle');
  const hold = loops ? -1 : Math.min(HOLD[mood] ?? last, last);
  const rest = alive && REST[mood] !== undefined && !loops ? Math.min(REST[mood], last) : hold;
  const end = dozing ? -1 : settled ? rest : hold;
  // A gradient id safe inside url(#…) whatever characters React's useId uses.
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  useEffect(() => { setFrame(0); setSettled(false); }, [mood]);
  useEffect(() => {
    if (!visible || frame === end) return undefined;
    // Reduced motion: no frame changes; a reaction shows its held frame right away.
    if (reduced) { if (end >= 0) setFrame(end); return undefined; }
    // A step queued before he dozed off or woke up is dropped, so it cannot move him off the frame that change set.
    const timer = setTimeout(() => { if (dozingRef.current === dozing) setFrame(n => (n + 1) % state.frames.length); }, state.durations[frame]);
    return () => clearTimeout(timer);
  }, [frame, end, visible, reduced, state, dozing]);
  // Settling: the held reaction stays for SETTLE_MS, then the remaining frames play once up to the resting face.
  useEffect(() => {
    if (!live || dozing || settled || hold < 0 || rest === hold || frame !== hold) return undefined;
    const timer = setTimeout(() => setSettled(true), SETTLE_MS);
    return () => clearTimeout(timer);
  }, [live, dozing, settled, hold, rest, frame]);
  // Dozing: after DOZE_MS with no input Bloop loops his sleepy frames; input wakes him straight onto his resting face.
  // He never dozes while working (bounce).
  useEffect(() => {
    if (!live || mood === 'bounce') return undefined;
    let quietSince = performance.now(), timer = 0, stopped = false;
    const wake = () => {
      quietSince = performance.now();
      if (!dozingRef.current) return;
      dozingRef.current = false;
      setDozing(false); setSettled(true); setFrame(Math.max(rest, 0));
      // The one-minute clock starts again, so he can doze off again later.
      if (!stopped) { clearTimeout(timer); timer = setTimeout(check, DOZE_MS); }
    };
    const check = () => {
      const quiet = performance.now() - quietSince;
      if (quiet < DOZE_MS) { timer = setTimeout(check, DOZE_MS - quiet); return; }
      dozingRef.current = true;
      setDozing(true); setFrame(0);
    };
    timer = setTimeout(check, DOZE_MS);
    const stop = listen(INPUT, wake);
    // Hiding the tab or turning on Reduce Motion also wakes him, so a frozen sleepy frame is never left on screen.
    return () => { stopped = true; clearTimeout(timer); stop(); wake(); };
  }, [live, mood, rest]);
  const f = state.frames[Math.min(frame, state.frames.length - 1)];
  const parts = alive && !dozing && REST[mood] !== undefined && frame === end && end === rest ? faceParts(f) : null;
  // Blinking: on a resting face, every 2.6-6.4 s, sometimes twice in a row.
  const blinks = live && !!parts;
  useEffect(() => {
    if (!blinks) return undefined;
    let timer = 0;
    const next = () => {
      timer = setTimeout(() => {
        setBlink(n => n + 1);
        if (Math.random() < .2) timer = setTimeout(() => { setBlink(n => n + 1); next(); }, 280);
        else next();
      }, 2600 + Math.random() * 3800);
    };
    next();
    return () => clearTimeout(timer);
  }, [blinks]);
  // Glancing: the face turns toward a drag (any pressed pointer) or the focused field while you type, and back after
  // GLANCE_MS without either. The offset is written straight to the face group (a CSS transition eases it).
  const glances = live && !dozing && mood !== 'bounce';
  useEffect(() => {
    const face = look.current, box = svg.current;
    if (!glances || !face || !box) return undefined;
    let target = null, frameRequest = 0, back = 0;
    const apply = () => {
      frameRequest = 0;
      let point = target;
      // A key press is read one frame later, so a Tab that moves focus turns him toward the newly focused element.
      if (point === 'focus') {
        const el = document.activeElement;
        if (!el || el === document.body || box.contains(el)) return;
        const e = el.getBoundingClientRect();
        point = [e.left + e.width / 2, e.top + e.height / 2];
      }
      const r = box.getBoundingClientRect(), dx = point[0] - (r.left + r.width / 2), dy = point[1] - (r.top + r.height / 2);
      const d = Math.hypot(dx, dy), k = d < 1 ? 0 : Math.min(1, d / LOOK_RANGE) / d;
      face.style.transform = `translate(${(dx * k * LOOK[0]).toFixed(1)}px, ${(dy * k * LOOK[1]).toFixed(1)}px)`;
      clearTimeout(back);
      back = setTimeout(() => { face.style.transform = ''; }, GLANCE_MS);
    };
    const toward = point => { target = point; frameRequest ||= requestAnimationFrame(apply); };
    const stopPointer = listen(['pointerdown', 'pointermove'], e => { if (e.type === 'pointerdown' || e.buttons) toward([e.clientX, e.clientY]); });
    const stopKeys = listen(['keydown'], () => toward('focus'));
    return () => { stopPointer(); stopKeys(); cancelAnimationFrame(frameRequest); clearTimeout(back); face.style.transform = ''; };
  }, [glances]);
  const [x, y, w, h] = BLOOP.viewBox, matrix = `matrix(${f.matrix.join(' ')})`;
  return <svg ref={svg} className={'bloop ' + className} viewBox={`${x} ${y} ${w} ${h}`} width={size} height={size * h / w}
    role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : 'true'} focusable="false">
    <defs>
      {/* The traced paths are y-flipped (potrace), so the light sits at the bottom of their local box: top left on screen. */}
      <radialGradient id={'bloop-pearl-' + id} cx=".42" cy=".62" r=".78">
        <stop offset="0" stopColor="#EFEBF8"/><stop offset=".6" stopColor="#DCD5EE"/><stop offset="1" stopColor="#C3BADF"/>
      </radialGradient>
    </defs>
    <g transform={matrix}>
      <path fill={`url(#bloop-pearl-${id})`} d={f.paths.body}/>
      {['blue', 'pink', 'white'].map(k => f.paths[k] && <path key={k} fill={FILLS[k]} d={f.paths[k]}/>)}
    </g>
    <g ref={look} className="bloop-look"><g transform={matrix} fill={FILLS.ink}>
      {parts ? <>
          {parts.eyes.map((d, i) => <path key={i + '-' + blink} className={blink ? 'bloop-eye bloop-eye--blink' : 'bloop-eye'} d={d}/>)}
          <path d={parts.mouth}/>
        </>
        : f.paths.ink && <path d={f.paths.ink}/>}
    </g></g>
  </svg>;
}

import { useEffect, useId, useRef, useState } from 'react';
import { usePageVisible, useReducedMotion } from './motion.mjs';
import { CENTER, DISC, DOTS, EYE, EYES, GAP, MOODS, RING, inGap } from './peri-shape.mjs';

// Peri, the app's mascot, drawn in code: LiDAR-style dots inside a thin ring, with two capsule eyes (the geometry is in
// peri-shape.mjs). A change of mood recolours the dots from the middle outward: rose for Blocked, a gap with no dots
// for Unknown, dimmed with shut eyes for sleepy. While working, a bright band sweeps across the dots like a LiDAR scan.
// The app's one Peri (PeriHost.jsx moves it between places) is alive: it blinks now and then, glances toward a drag or
// the field you type in, and dozes off after a minute with no input. All of that stops while the tab is hidden and
// stays off with Reduce Motion.
const DOZE_MS = 60000, GLANCE_MS = 1200, BLINK_MS = 200;
// Any of these is input: it wakes a dozing Peri and restarts the one-minute clock.
const INPUT = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'];
// How far the eyes move toward what you are doing, in viewBox units (the ball is 94 wide), and the distance in screen
// pixels at which the full offset is reached.
const LOOK = [5, 3.5], LOOK_RANGE = 160;
const listen = (types, handler) => {
  types.forEach(type => document.addEventListener(type, handler, { capture: true, passive: true }));
  return () => types.forEach(type => document.removeEventListener(type, handler, { capture: true }));
};

export default function Peri({ mood = 'idle', size = 44, className = '', label, alive = false }) {
  const reduced = useReducedMotion(), visible = usePageVisible(), live = alive && !reduced && visible;
  // blinking: the eyes carry the blink class only while a blink plays, so moving Peri to another place (which restarts
  // CSS animations) or a mood change never replays it, on both eyes or on one as a wink.
  const [dozing, setDozing] = useState(false), [blinking, setBlinking] = useState(false), svg = useRef(null), look = useRef(null);
  const known = MOODS.includes(mood) ? mood : 'idle', face = dozing ? 'sleepy' : known;
  // An id safe inside url(#…) whatever characters React's useId uses.
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  // Dozing: after DOZE_MS with no input Peri shuts its eyes; input wakes it. It never dozes while working.
  useEffect(() => {
    if (!live || known === 'working' || known === 'sleepy') return undefined;
    let quietSince = performance.now(), timer = 0, asleep = false;
    const check = () => {
      const quiet = performance.now() - quietSince;
      if (quiet < DOZE_MS) { timer = setTimeout(check, DOZE_MS - quiet); return; }
      asleep = true;
      setDozing(true);
    };
    const wake = () => {
      quietSince = performance.now();
      if (!asleep) return;
      asleep = false;
      setDozing(false);
      clearTimeout(timer);
      timer = setTimeout(check, DOZE_MS);
    };
    timer = setTimeout(check, DOZE_MS);
    const stop = listen(INPUT, wake);
    // A new mood, a hidden tab or Reduce Motion also wakes it, so shut eyes are never left on screen.
    return () => { clearTimeout(timer); stop(); setDozing(false); };
  }, [live, known]);
  // Blinking: with open eyes, every 2.6-6.4 s, sometimes twice in a row.
  const blinks = live && (face === 'idle' || face === 'unknown');
  useEffect(() => {
    if (!blinks) return undefined;
    let timer = 0, open = 0;
    const shut = () => { setBlinking(true); clearTimeout(open); open = setTimeout(() => setBlinking(false), BLINK_MS); };
    const next = () => {
      timer = setTimeout(() => {
        shut();
        if (Math.random() < .2) timer = setTimeout(() => { shut(); next(); }, 280);
        else next();
      }, 2600 + Math.random() * 3800);
    };
    next();
    return () => { clearTimeout(timer); clearTimeout(open); setBlinking(false); };
  }, [blinks]);
  // Glancing: the eyes turn toward a drag (any pressed pointer) or the focused field while you type, and back after
  // GLANCE_MS without either. The offset is written straight to the eye group (a CSS transition eases it).
  const glances = live && face !== 'working' && face !== 'sleepy';
  useEffect(() => {
    const eyes = look.current, box = svg.current;
    if (!glances || !eyes || !box) return undefined;
    let target = null, frameRequest = 0, back = 0;
    const apply = () => {
      frameRequest = 0;
      let point = target;
      // A key press is read one frame later, so a Tab that moves focus turns the eyes toward the newly focused element.
      if (point === 'focus') {
        const el = document.activeElement;
        if (!el || el === document.body || box.contains(el)) return;
        const e = el.getBoundingClientRect();
        point = [e.left + e.width / 2, e.top + e.height / 2];
      }
      const r = box.getBoundingClientRect(), dx = point[0] - (r.left + r.width / 2), dy = point[1] - (r.top + r.height / 2);
      const d = Math.hypot(dx, dy), k = d < 1 ? 0 : Math.min(1, d / LOOK_RANGE) / d;
      eyes.style.transform = `translate(${(dx * k * LOOK[0]).toFixed(1)}px, ${(dy * k * LOOK[1]).toFixed(1)}px)`;
      clearTimeout(back);
      back = setTimeout(() => { eyes.style.transform = ''; }, GLANCE_MS);
    };
    const toward = point => { target = point; frameRequest ||= requestAnimationFrame(apply); };
    const stopPointer = listen(['pointerdown', 'pointermove'], e => { if (e.type === 'pointerdown' || e.buttons) toward([e.clientX, e.clientY]); });
    const stopKeys = listen(['keydown'], () => toward('focus'));
    return () => { stopPointer(); stopKeys(); cancelAnimationFrame(frameRequest); clearTimeout(back); eyes.style.transform = ''; };
  }, [glances]);
  const eye = EYES[face], gap = face === 'unknown', sweep = face === 'working' && !reduced && visible;
  return <svg ref={svg} className={`peri peri--${face} ${className}`.trim()} viewBox="0 0 100 100" width={size} height={size}
    role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : 'true'} focusable="false">
    <defs>
      {/* Unknown: the disc and rim leave out the gap (peri-out), where only a faint dashed rim remains (peri-in). */}
      <mask id={'peri-out-' + id}><rect width="100" height="100" fill="#fff"/><circle cx={GAP.x} cy={GAP.y} r={GAP.r} fill="#000"/></mask>
      <mask id={'peri-in-' + id}><circle cx={GAP.x} cy={GAP.y} r={GAP.r} fill="#fff"/></mask>
      {sweep && <clipPath id={'peri-dots-' + id}>{DOTS.map((d, i) => <circle key={i} cx={d.x} cy={d.y} r={d.r + .4}/>)}</clipPath>}
    </defs>
    <g mask={gap ? `url(#peri-out-${id})` : undefined}>
      <circle className="peri-disc" cx={CENTER} cy={CENTER} r={DISC}/>
      <circle className="peri-ring" cx={CENTER} cy={CENTER} r={RING}/>
    </g>
    {gap && <circle className="peri-ring peri-ring--gap" cx={CENTER} cy={CENTER} r={RING} mask={`url(#peri-in-${id})`}/>}
    <g>{DOTS.map((d, i) => <circle key={i} className={inGap(d) ? 'peri-dot peri-dot--gap' : 'peri-dot'} cx={d.x} cy={d.y} r={d.r}
      style={{ transitionDelay: d.delay + 'ms' }}/>)}</g>
    {sweep && <g clipPath={`url(#peri-dots-${id})`}><rect className="peri-sweep" x="-24" y="0" width="22" height="100"/></g>}
    <g ref={look} className="peri-look">
      {EYE.x.map((x, i) => {
        // Unknown leaves the right eye as a dashed outline: half of what Peri would see is missing.
        const unseen = gap && i === 1;
        return <rect key={i} className={'peri-eye' + (unseen ? ' peri-eye--unknown' : '') + (blinking ? ' peri-eye--blink' : '')}
          x={x} y={eye.y} width={EYE.w} height={eye.h} rx={Math.min(EYE.w, eye.h) / 2}/>;
      })}
    </g>
  </svg>;
}

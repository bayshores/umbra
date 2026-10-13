import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Peri from './Peri.jsx';
import { prefersReducedMotion } from './motion.mjs';

// The one Peri on screen moves as the page changes: from the loading screen to the page header (phones and tablets)
// or the chat header (desktop), and into the street map layer and back. Each of those places renders a PeriSlot. The
// host draws Peri once, inside a carrier element that rests in the highest-ranked visible slot, so it scrolls and clips
// with that place like any other content. When the place changes, the carrier flies from where Peri is now to the new
// slot inside a fixed layer above the page: a transform animation, which the browser runs off the main thread, so it
// stays smooth while a new view starts up. It lands inside the slot at the end. Peri keeps its old mood in flight and
// takes the new place's mood just after landing, so its dots recolour where you can see them. With Reduce Motion, or
// while the tab is hidden, it moves at once and fades in instead.
const FLY_MS = 520, FADE_MS = 200, EASE = 'cubic-bezier(.32,.72,.16,1)';
const Host = createContext(null);

// slots: each mounted slot element and a ref to its {mood, alive, rank}; show: sets the face Peri draws.
function createMover(carrier, show) {
  const slots = new Map();
  let sky = null, home = null, flight = null, queued = false, shown = { mood: 'working', alive: false };
  const box = el => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width }; };
  const visible = el => el.isConnected && el.getClientRects().length > 0;
  // The highest rank wins; on a tie, the slot mounted last.
  const pick = () => {
    let best = null, rank = -Infinity;
    for (const [el, state] of slots) if (state.current.rank >= rank && visible(el)) { best = el; rank = state.current.rank; }
    return best;
  };
  // Moves the carrier without restarting its animations where the browser can (Element.moveBefore).
  const put = parent => {
    if (carrier.parentNode === parent) return;
    if (parent.moveBefore && carrier.isConnected && parent.isConnected) {
      try { parent.moveBefore(carrier, null); return; } catch { /* appendChild below */ }
    }
    parent.appendChild(carrier);
  };
  // Every animation on the carrier except keep (the flight just started).
  const stop = keep => { for (const a of carrier.getAnimations()) if (a !== keep) a.cancel(); };
  const toSky = (r, size) => {
    Object.assign(carrier.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', width: size + 'px', height: size + 'px',
      transformOrigin: '0 0', transform: `translate(${r.x}px, ${r.y}px) scale(${r.w / size})` });
    put(sky);
  };
  const face = () => {
    const state = home && !flight && slots.get(home)?.current;
    if (!state || state.mood === shown.mood && state.alive === shown.alive) return;
    shown = { mood: state.mood, alive: state.alive };
    show(shown);
  };
  const land = (to, fade) => {
    carrier.removeAttribute('style');
    put(to);
    stop(null);
    if (fade) carrier.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_MS, easing: 'ease-out' });
    // Reading its box applies the landed style now, so the mood change a frame later is a transition.
    box(carrier);
    requestAnimationFrame(face);
  };
  const fly = to => {
    // Where Peri is now, mid-flight included (the box includes the running transform).
    const from = carrier.isConnected ? box(carrier) : null, end = box(to);
    home = to;
    flight = null;
    if (!from || !from.w || document.hidden || prefersReducedMotion()) { land(to, !!from && !document.hidden); return; }
    // Drawn at the larger of the two sizes and scaled down, so it stays sharp all the way.
    const size = Math.max(from.w, end.w), at = r => `translate(${r.x}px, ${r.y}px) scale(${r.w / size})`;
    toSky(from, size);
    const animation = carrier.animate([{ transform: at(from) }, { transform: at(end) }], { duration: FLY_MS, easing: EASE, fill: 'forwards' });
    stop(animation);
    flight = animation;
    animation.onfinish = () => {
      if (flight !== animation) return;
      flight = null;
      // The place may have moved or changed while Peri flew: it flies on from where it is.
      if (pick() !== to) { update(); return; }
      const now = box(to);
      if (Math.abs(now.x - end.x) > 1 || Math.abs(now.y - end.y) > 1 || Math.abs(now.w - end.w) > 1) fly(to);
      else land(to, false);
    };
  };
  const update = () => {
    queued = false;
    if (!sky) return;
    const target = pick();
    if (!target) { flight = null; home = null; stop(null); carrier.remove(); return; }
    if (target !== home) fly(target);
    else face();
  };
  const schedule = () => { if (!queued) { queued = true; queueMicrotask(update); } };
  return {
    schedule,
    // The fixed layer that flights cross.
    sky: node => { sky = node; if (node) schedule(); },
    register(el, state) {
      slots.set(el, state);
      schedule();
      // A slot leaving the page takes Peri's place with it: Peri waits in the sky where it was, then flies on.
      return () => {
        slots.delete(el);
        if (el === home) {
          home = null;
          if (carrier.parentNode === el && sky) { const r = box(carrier); if (r.w) toSky(r, r.w); else carrier.remove(); }
        }
        schedule();
      };
    },
  };
}

export function PeriHost({ children }) {
  const [carrier] = useState(() => Object.assign(document.createElement('span'), { className: 'peri-carrier' }));
  const [face, setFace] = useState({ mood: 'working', alive: false });
  const [mover] = useState(() => createMover(carrier, setFace));
  // A new window size can show a different slot.
  useEffect(() => {
    addEventListener('resize', mover.schedule);
    return () => removeEventListener('resize', mover.schedule);
  }, [mover]);
  return <Host.Provider value={mover}>
    {children}
    <div className="peri-sky" ref={mover.sky} aria-hidden="true"/>
    {createPortal(<Peri mood={face.mood} alive={face.alive} size="100%"/>, carrier)}
  </Host.Provider>;
}

// A place where Peri can be. rank: the visible slot with the highest rank has Peri; size: its width and height in
// pixels (or set in CSS).
export function PeriSlot({ mood = 'idle', alive = false, rank = 0, size, className = '' }) {
  const mover = useContext(Host), state = useRef({ mood, alive, rank });
  useLayoutEffect(() => {
    const s = state.current;
    if (s.mood === mood && s.alive === alive && s.rank === rank) return;
    state.current = { mood, alive, rank };
    mover?.schedule();
  }, [mood, alive, rank, mover]);
  const ref = useCallback(el => (el && mover ? mover.register(el, state) : undefined), [mover]);
  return <span ref={ref} className={('peri-slot ' + className).trim()} style={size ? { width: size, height: size } : undefined} aria-hidden="true"/>;
}

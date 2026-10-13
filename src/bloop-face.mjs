// The live chat Bloop's resting faces (Bloop.jsx). After a reaction he leaves the held frame for a calm face with open
// eyes: the last frame of blocked and unsure, the first of idle. Only these faces blink.
export const REST = { blocked: 5, unsure: 5, idle: 0 };

// A face's eyes and mouth: its three outlines each start with an absolute M, and the traced y axis points up, so the
// mouth is the one that starts lowest. A face of any other shape returns null and is drawn whole.
const FACES = new WeakMap();
const startY = part => Number(part.match(/^M\s*-?[\d.]+[\s,]+(-?[\d.]+)/)?.[1]);
export function faceParts(frame) {
  if (!FACES.has(frame)) {
    const parts = frame.paths.ink?.match(/M[^M]*/g);
    const sorted = parts?.length === 3 && parts.every(p => Number.isFinite(startY(p))) ? [...parts].sort((a, b) => startY(b) - startY(a)) : null;
    FACES.set(frame, sorted && { eyes: sorted.slice(0, 2), mouth: sorted[2] });
  }
  return FACES.get(frame);
}

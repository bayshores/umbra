// Peri, Sightline's mascot (src/Peri.jsx): a ball of LiDAR-style dots inside a thin ring, with two capsule eyes. This
// module holds the drawing's geometry in a 100 × 100 viewBox, so tests can check it without React.
export const CENTER = 50, RING = 46, DISC = 47;
// Peri's moods. working: a request is running. blocked and unknown: the proposed result. sleepy: a failed request, an
// unavailable scene, or a minute with no input. There is no "sees clearly" mood, because Sightline never reports Clear.
export const MOODS = ['idle', 'working', 'blocked', 'unknown', 'sleepy'];

// The dots: a 7-unit grid, keeping only whole dots that sit inside the ring. Each dot's delay grows with its distance
// from the middle, so a change of mood spreads outward dot by dot.
const STEP = 7, DOT = 2.1, INNER = 40, SPREAD_MS = 360;
export const DOTS = [];
for (let y = CENTER - 6 * STEP; y <= CENTER + 6 * STEP; y += STEP) {
  for (let x = CENTER - 6 * STEP; x <= CENTER + 6 * STEP; x += STEP) {
    const d = Math.hypot(x - CENTER, y - CENTER);
    if (d <= INNER) DOTS.push({ x, y, r: DOT, delay: Math.round(d / INNER * SPREAD_MS) });
  }
}

// Unknown opens a gap in the lower right with no dots and no rim, like the dark gaps in the driver view where nothing
// was measured.
export const GAP = { x: 80, y: 76, r: 30 };
export const inGap = ({ x, y }) => Math.hypot(x - GAP.x, y - GAP.y) < GAP.r;

// The eyes: two capsules side by side, centred on the ball. Blocked squints, sleepy closes them to thin lines.
export const EYE = { x: [33, 56], w: 11 };
export const EYES = {
  idle: { y: 32, h: 22 }, working: { y: 32, h: 22 }, unknown: { y: 32, h: 22 },
  blocked: { y: 42, h: 9 }, sleepy: { y: 44, h: 3 },
};

import assert from 'node:assert/strict';
import test from 'node:test';
import { CENTER, DISC, DOTS, EYE, EYES, GAP, MOODS, RING, inGap } from '../src/peri-shape.mjs';

// The geometry of Peri, the mascot the app draws (src/Peri.jsx), in its 100 × 100 viewBox.
test('Peri has no mood that could read as a Clear result', () => {
  assert.deepEqual([...MOODS].sort(), ['blocked', 'idle', 'sleepy', 'unknown', 'working']);
  assert.ok(!MOODS.some(mood => /clear|happy|visible|safe/i.test(mood)));
  assert.deepEqual(Object.keys(EYES).sort(), [...MOODS].sort());
});

test('every dot is whole, inside the ring, and lights up from the middle outward', () => {
  assert.ok(DOTS.length > 100);
  assert.ok(RING < DISC && DISC <= CENTER);
  for (const d of DOTS) {
    assert.ok(Math.hypot(d.x - CENTER, d.y - CENTER) + d.r < RING - 1, `${d.x},${d.y}`);
    assert.ok(Number.isInteger(d.delay) && d.delay >= 0 && d.delay <= 400);
  }
  const middle = DOTS.find(d => d.x === CENTER && d.y === CENTER), far = DOTS.reduce((a, b) => (a.delay > b.delay ? a : b));
  assert.equal(middle?.delay, 0);
  assert.ok(Math.hypot(far.x - CENTER, far.y - CENTER) > 35);
  // A dot never lights up before one nearer the middle.
  const reach = d => Math.hypot(d.x - CENTER, d.y - CENTER), byReach = [...DOTS].sort((a, b) => reach(a) - reach(b));
  byReach.forEach((d, i) => assert.ok(!i || d.delay >= byReach[i - 1].delay, `${d.x},${d.y} lights before a nearer dot`));
  assert.equal(new Set(DOTS.map(d => d.x + ',' + d.y)).size, DOTS.length);
});

test('the Unknown gap removes some dots but leaves both eyes on the ball', () => {
  const gone = DOTS.filter(inGap).length;
  assert.ok(gone > 10 && gone < DOTS.length / 3, String(gone));
  assert.ok(GAP.x > CENTER && GAP.y > CENTER);
  for (const [mood, { y, h }] of Object.entries(EYES)) {
    assert.ok(h > 0 && y > CENTER - RING && y + h < CENTER + RING, mood);
    // The solid left eye stays whole outside the gap; the right one (dashed in Unknown) may touch its edge.
    const [left, right] = EYE.x;
    assert.ok([[left, y], [left + EYE.w, y], [left, y + h], [left + EYE.w, y + h]].every(([px, py]) => !inGap({ x: px, y: py })), `${mood} left eye`);
    assert.ok(!inGap({ x: right + EYE.w / 2, y: y + h / 2 }), `${mood} right eye centre`);
    for (const x of EYE.x) assert.ok(Math.hypot(x + EYE.w / 2 - CENTER, y + h / 2 - CENTER) < RING / 2, `${mood} eye near the middle`);
  }
  // The two eyes sit side by side, the same distance from the middle.
  assert.equal(EYE.x[0] + EYE.w / 2 - CENTER, -(EYE.x[1] + EYE.w / 2 - CENTER));
});

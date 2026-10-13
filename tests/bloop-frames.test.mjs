import assert from 'node:assert/strict';
import test from 'node:test';
import BLOOP from '../src/bloop-frames.json' with { type: 'json' };
import { REST, faceParts } from '../src/bloop-face.mjs';

// The traced mascot frames the app draws (src/Bloop.jsx).
test('every Bloop mood has six complete, timed vector frames', () => {
  assert.deepEqual(Object.keys(BLOOP.states).sort(), ['blocked', 'bounce', 'idle', 'sleepy', 'unsure']);
  assert.equal(BLOOP.viewBox.length, 4);
  assert.ok(BLOOP.viewBox.every(Number.isFinite) && BLOOP.viewBox[2] > 0 && BLOOP.viewBox[3] > 0);
  for (const [mood, state] of Object.entries(BLOOP.states)) {
    assert.equal(state.frames.length, 6, mood);
    assert.equal(state.durations.length, state.frames.length, mood);
    assert.ok(state.durations.every(ms => Number.isInteger(ms) && ms >= 60 && ms <= 2000), mood);
    for (const frame of state.frames) {
      assert.equal(frame.matrix.length, 6, mood);
      assert.ok(frame.matrix.every(Number.isFinite), mood);
      for (const layer of ['body', 'ink']) assert.match(frame.paths[layer], /^M/i, `${mood} ${layer}`);
      assert.ok(Object.keys(frame.paths).every(k => ['body', 'blue', 'pink', 'white', 'ink'].includes(k)), mood);
    }
  }
});

test('Bloop has no mood that could read as a Clear result', () => {
  assert.ok(!Object.keys(BLOOP.states).some(mood => /clear|happy|visible|safe/i.test(mood)));
});

// The chat Bloop blinks by squeezing the two eyes of a resting face (src/bloop-face.mjs).
test('each resting face splits into two side-by-side eyes above a mouth', () => {
  const start = d => d.match(/^M\s*(-?[\d.]+)[\s,]+(-?[\d.]+)/).slice(1).map(Number);
  for (const [mood, index] of Object.entries(REST)) {
    const frame = BLOOP.states[mood].frames[index], face = faceParts(frame);
    assert.ok(face, mood);
    assert.equal(face.eyes.join('') + face.mouth, frame.paths.ink.match(/M[^M]*/g).sort((a, b) => start(b)[1] - start(a)[1]).join(''), mood);
    const [left, right] = face.eyes.map(start).sort((a, b) => a[0] - b[0]), mouth = start(face.mouth);
    assert.ok(right[0] - left[0] > 100, `${mood} eyes apart`);
    assert.ok(Math.abs(right[1] - left[1]) < 30, `${mood} eyes level`);
    assert.ok(Math.min(left[1], right[1]) - mouth[1] > 100, `${mood} mouth below the eyes`);
  }
  assert.equal(faceParts({ paths: { ink: 'M0 0 l1 1z' } }), null);
});

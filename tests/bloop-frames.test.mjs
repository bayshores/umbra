import assert from 'node:assert/strict';
import test from 'node:test';
import BLOOP from '../src/bloop-frames.json' with { type: 'json' };

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

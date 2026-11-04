/// <reference types="node" />
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { motionSamples, type RawMotion } from './motion.ts';

test('motion samples use the sensor clock, shifted onto the recording clock', () => {
  const start = 1_000_000;
  // Sensor readings every 20 ms from 5000 s since boot; the listener ran 10–40 ms later (jitter).
  const raw: RawMotion[] = [0, 1, 2, 3, 4].map((i) => [
    start + 100 + i * 20 + [10, 40, 10, 25, 15][i],
    5000 + i * 0.02,
    0.1 * i,
    -0.2,
    0.3,
  ]);
  const s = motionSamples(raw, start);
  assert.equal(s.length, 5);
  // Evenly spaced like the sensor's own times, offset by the median listener delay (15 ms).
  s.forEach(([t], i) => assert.ok(Math.abs(t - (0.1 + i * 0.02 + 0.015)) < 1e-9, `t=${t}`));
  assert.deepEqual(s[2].slice(1), [0.2, -0.2, 0.3]);
});

test('motion samples fall back to the listener clock and drop broken readings', () => {
  const raw: RawMotion[] = [
    [1500, 0, 0.1, 0.2, 0.3], // no sensor timestamp
    [1520, 0, NaN, 0.2, 0.3],
    [1540, 0, 0.1, 0.2, 0.3],
  ];
  assert.deepEqual(
    motionSamples(raw, 1000).map((r) => r[0]),
    [0.5, 0.54],
  );
  assert.deepEqual(motionSamples([], 1000), []);
});

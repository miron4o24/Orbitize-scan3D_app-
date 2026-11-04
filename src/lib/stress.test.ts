/// <reference types="node" />
import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { StressSummary } from './api.ts';
import { effective, outcome, stressColor, UNTESTED } from './stress.ts';

const summary = (kind: StressSummary['case'], extra: Partial<StressSummary> = {}): StressSummary => ({
  case: kind,
  ref_mm: 100,
  max_mpa: 2,
  weak: [0, 0, 0],
  volume_cm3: 50,
  elements: 1000,
  pitch_mm: 1.5,
  dropped: 0,
  ...extra,
});

const close = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);

test('infill weakens layered prints, not resin', () => {
  close(effective('pla', 100).mpa, 50 * 0.7);
  assert.ok(effective('pla', 20).mpa < effective('pla', 50).mpa);
  assert.deepEqual(effective('resin', 20), effective('resin', 100));
});

test('a weight: stress grows with it, falls with the size squared; the limit is where it breaks', () => {
  const s = summary('top');
  const one = outcome(s, 1, 'pla', 100, 100);
  close(one.peak, 2);
  close(outcome(s, 3, 'pla', 100, 100).peak, 6);
  close(outcome(s, 1, 'pla', 100, 200).peak, 0.5);
  close(outcome(s, one.limit, 'pla', 100, 100).factor, 1);
  assert.equal(outcome(s, 1, 'pla', 100, 100).verdict, 'safe');
  assert.equal(outcome(s, one.limit * 1.01, 'pla', 100, 100).verdict, 'break');
  close(one.massG, 50 * 1.24);
});

test('a drop: higher is worse, and the limit is the height where it breaks', () => {
  const s = summary('drop', { max_mpa: 0.002, sag_m: 1e-7 });
  const low = outcome(s, 0.3, 'pla', 100, 100);
  const high = outcome(s, 2, 'pla', 100, 100);
  assert.ok(high.peak > low.peak);
  close(outcome(s, low.limit, 'pla', 100, 100).factor, 1, 1e-6);
});

test('colours run from blue to red; untested pieces are grey', () => {
  assert.deepEqual(stressColor(-1), UNTESTED);
  const [r0, , b0] = stressColor(0);
  const [r1, , b1] = stressColor(1);
  assert.ok(b0 > r0 && r1 > b1);
  assert.deepEqual(stressColor(5), stressColor(1));
});

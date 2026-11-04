/// <reference types="node" />
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { elevation, isUsableAim, viewDir, type Vec3 } from './coverage.ts';
import { OrbitTracker } from './orbit.ts';
import { judge, measure } from './quality.ts';

const W = 128;
const H = 96;

function image(fn: (x: number, y: number) => number) {
  const px = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const v = fn(x, y);
      px.set([v, v, v, 255], (y * W + x) * 4);
    }
  return measure(px, W, H);
}

// Deterministic pseudo-random texture, like a real textured object.
const noise = (x: number, y: number) => ((Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1 + 1) % 1;

test('sharp texture passes, flat image is rejected as blurry', () => {
  const sharp = image((x, y) => 60 + 140 * noise(x, y));
  assert.equal(judge(sharp).reject, false, `sharpness ${sharp.sharpness}`);
  const flat = image(() => 128);
  assert.equal(judge(flat).reject, true);
});

test('blurring a texture lowers sharpness below the threshold', () => {
  const blurred = image((x, y) => {
    let s = 0;
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) s += noise(x + dx, y + dy);
    return 60 + (140 * s) / 81;
  });
  assert.equal(judge(blurred).reject, true, `sharpness ${blurred.sharpness}`);
});

test('dark and overexposed frames warn but are kept', () => {
  const dark = judge(image((x, y) => 10 + 25 * noise(x, y)));
  assert.ok(dark.warnings.some((w) => w.startsWith('Too dark')));
  const bright = judge(image((x, y) => (noise(x, y) > 0.5 ? 255 : 120)));
  assert.ok(bright.warnings.some((w) => w.startsWith('Overexposed')));
  assert.equal(bright.reject, false);
});

const close = (a: Vec3, b: Vec3) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);

test('view direction: flat phone looks down, upright phone looks north', () => {
  assert.ok(close(viewDir({ alpha: 0, beta: 0, gamma: 0 }), [0, 0, -1]));
  assert.ok(close(viewDir({ alpha: 0, beta: Math.PI / 2, gamma: 0 }), [0, 1, 0]));
  // Turning left by 90° (alpha) points the upright phone west.
  assert.ok(close(viewDir({ alpha: Math.PI / 2, beta: Math.PI / 2, gamma: 0 }), [-1, 0, 0]));
});

test('elevation and usable aim', () => {
  const tilt = (deg: number) => (deg * Math.PI) / 180;
  const v = viewDir({ alpha: 0, beta: tilt(60), gamma: 0 }); // 30° below horizon
  assert.equal(Math.round(elevation(v)), -30);
  assert.equal(isUsableAim(v), true);
  assert.equal(isUsableAim(viewDir({ alpha: 0, beta: tilt(120), gamma: 0 })), false); // aimed upward
  assert.equal(isUsableAim(viewDir({ alpha: 0, beta: 0, gamma: 0 })), false); // straight down
});

// Walks `turnDeg` around the object at `degPerS`, looking `downDeg` below the horizon; 20 Hz like the app.
function walk(o: OrbitTracker, from: number, turnDeg: number, degPerS: number, downDeg: number, t0: number) {
  const rad = (d: number) => (d * Math.PI) / 180;
  const steps = Math.round((turnDeg / degPerS) * 20);
  for (let i = 1; i <= steps; i++)
    o.update(viewDir({ alpha: rad(from + (turnDeg * i) / steps), beta: rad(90 - downDeg), gamma: 0 }), degPerS, t0 + i / 20);
  return t0 + steps / 20;
}

test('orbit: two circles, the second from higher up, complete the scan', () => {
  const o = new OrbitTracker();
  let t = walk(o, 0, 200, 12, 10, 0);
  assert.equal(o.state.loop, 0);
  assert.equal(o.state.pace, 'good');
  assert.ok(o.firstLoopFraction > 0.5 && o.firstLoopFraction < 1);
  t = walk(o, 200, 160, 12, 10, t);
  assert.equal(o.state.loop, 1);
  // Going round again at the same height does not count for the second circle.
  t = walk(o, 360, 120, 12, 10, t);
  assert.equal(o.state.aim, 'higher');
  assert.equal(o.state.covered[1].filter(Boolean).length, 0);
  walk(o, 480, 370, 12, 45, t);
  assert.equal(o.done, true);
});

test('orbit: the first circle only counts near the object’s height', () => {
  const o = new OrbitTracker();
  walk(o, 0, 360, 12, 40, 0); // filmed from well above, like a 6 cm speaker from standing height
  assert.equal(o.state.aim, 'lower');
  assert.equal(o.state.covered[0].filter(Boolean).length, 0);
});

test('orbit: the second circle must not look down too steeply', () => {
  const o = new OrbitTracker();
  const t = walk(o, 0, 360, 12, 10, 0);
  assert.equal(o.state.loop, 1);
  walk(o, 360, 360, 12, 70, t); // nearly straight down: no overlap with the first circle
  assert.equal(o.state.aim, 'lower');
  assert.equal(o.state.covered[1].filter(Boolean).length, 0);
});

test('orbit: turning too fast covers nothing', () => {
  const o = new OrbitTracker();
  walk(o, 0, 360, 45, 10, 0);
  assert.equal(o.state.pace, 'fast');
  // Only the first ~0.5 s counts, while the smoothed speed catches up.
  assert.ok(o.state.covered[0].filter(Boolean).length <= 4, `${o.state.covered[0].filter(Boolean).length}`);
  assert.equal(o.state.loop, 0);
});

test('print export: Y-up model becomes Z-up and is scaled to the requested size', async () => {
  const { BoxGeometry, Mesh } = await import('three');
  const { exportForPrint } = await import('./exportModel.ts');
  const tall = new Mesh(new BoxGeometry(1, 4, 2)); // height on Y, like a glTF model
  const stl = exportForPrint(tall, 'stl', 80) as Uint8Array;
  const dv = new DataView(stl.buffer, stl.byteOffset, stl.byteLength);
  const tris = dv.getUint32(80, true);
  assert.equal(stl.byteLength, 84 + 50 * tris);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let t = 0; t < tris; t++)
    for (let v = 0; v < 3; v++)
      for (let a = 0; a < 3; a++) {
        const x = dv.getFloat32(84 + t * 50 + 12 + v * 12 + a * 4, true);
        min[a] = Math.min(min[a], x);
        max[a] = Math.max(max[a], x);
      }
  const ext = max.map((m, a) => m - min[a]);
  assert.ok(Math.abs(ext[2] - 80) < 1e-3, `z extent ${ext[2]}`); // old Y (height) is now Z
  assert.ok(Math.abs(ext[0] - 20) < 1e-3 && Math.abs(ext[1] - 40) < 1e-3, `${ext}`);
  assert.match(exportForPrint(tall, 'obj', 80) as string, /^v /m);
});

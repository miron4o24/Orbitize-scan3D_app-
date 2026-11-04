// The phone's orientation while filming, for the server's route for plain objects (backend/app/motion.py): it places
// the frames by how the phone was turned when photo matching cannot. Each sample is
// [seconds since the recording was requested, alpha, beta, gamma] (radians, W3C device orientation).
export type MotionSample = [number, number, number, number];

// One reading as the listener got it: [phone clock ms, sensor timestamp s, alpha, beta, gamma].
export type RawMotion = [number, number, number, number, number];

const round = (x: number, digits: number) => Math.round(x * 10 ** digits) / 10 ** digits;

// The sensor's own timestamps are exact but count from another origin; the times the listener ran are on the
// recording's clock but late by a varying delay. So: sensor times, shifted by the median difference.
export function motionSamples(raw: RawMotion[], startedAt: number): MotionSample[] {
  const ok = raw.filter((r) => r.every(Number.isFinite));
  if (ok.length === 0) return [];
  const sensorClock = ok.every((r) => r[1] > 0);
  const diffs = ok.map(([ms, ts]) => (ms - startedAt) / 1000 - ts).sort((a, b) => a - b);
  const offset = diffs[Math.floor(diffs.length / 2)];
  return ok.map(([ms, ts, a, b, g]) => [
    round(sensorClock ? ts + offset : (ms - startedAt) / 1000, 4),
    round(a, 5),
    round(b, 5),
    round(g, 5),
  ]);
}

// Guided orbit: two circles around the object (the second from higher up), tracked from DeviceMotion.
// The view azimuth turns 360° per circle while the phone stays pointed at the object.

import { elevation, isUsableAim, type Vec3 } from './coverage.ts';

export const SECTORS = 24; // 15° each
export const LOOPS = 2;
const NEEDED = SECTORS - 2; // a circle counts as done with at most 30° missing
// The first circle must look at the object almost level: seen only from above, a flat top cannot be told apart
// from a cone on it (from 40° above, a cone up to 0.84x the top's radius; from 20°, 0.36x).
const LOOP1_MAX_DOWN_DEG = 20;
const LOOP2_DROP_DEG = 15; // the second circle must look down at least this much more than the first
// ...but not much more: frames of the two circles must still share enough of the view to be linked (from straight
// above, a plain table and the object's top are all there is, and that circle came out disconnected).
const LOOP2_MAX_DROP_DEG = 45;
export const GOOD_DEG_PER_S = [4, 24] as const; // ~15–90 s per circle
const SHAKE_DEG_PER_S = 70;
const SMOOTH_S = 0.6;

export type Pace = 'still' | 'good' | 'fast';

export type OrbitState = {
  loop: number; // 0-based circle being filmed; LOOPS when finished
  covered: boolean[][]; // [loop][sector], sectors counted from where filming started
  sector: number; // where the phone is now
  pace: Pace;
  aim: Aim; // whether the phone's height suits the circle being filmed
};

// 'lower': the phone looks down too steeply for this circle. 'higher': second circle, phone not raised enough.
export type Aim = 'ok' | 'lower' | 'higher';

const deg = (r: number) => (r * 180) / Math.PI;
export const azimuth = (v: Vec3) => (deg(Math.atan2(v[0], v[1])) + 360) % 360;
const wrap180 = (a: number) => ((((a + 180) % 360) + 360) % 360) - 180;

export class OrbitTracker {
  private startAz: number | null = null;
  private lastAz = 0;
  private lastT = 0;
  private speed = 0; // smoothed °/s of azimuth change
  private loop1Elevations: number[] = [];
  state: OrbitState = {
    loop: 0,
    covered: Array.from({ length: LOOPS }, () => Array<boolean>(SECTORS).fill(false)),
    sector: 0,
    pace: 'still',
    aim: 'ok',
  };

  get done() {
    return this.state.loop >= LOOPS;
  }

  /** Fraction of the first circle filmed: below 1, the scan will likely be incomplete. */
  get firstLoopFraction() {
    return Math.min(1, this.state.covered[0].filter(Boolean).length / NEEDED);
  }

  /** view: where the camera points; turnRate: total rotation rate in °/s; t: seconds. Returns true on change. */
  update(view: Vec3, turnRate: number, t: number): boolean {
    const az = azimuth(view);
    if (this.startAz === null) {
      this.startAz = az;
      this.lastAz = az;
      this.lastT = t;
    }
    const dt = Math.max(1e-3, t - this.lastT);
    const k = Math.min(1, dt / SMOOTH_S);
    this.speed += k * (Math.abs(wrap180(az - this.lastAz)) / dt - this.speed);
    this.lastAz = az;
    this.lastT = t;

    const s = this.state;
    const pace: Pace =
      this.speed > GOOD_DEG_PER_S[1] || turnRate > SHAKE_DEG_PER_S
        ? 'fast'
        : this.speed < GOOD_DEG_PER_S[0]
          ? 'still'
          : 'good';
    const sector = Math.floor((((az - this.startAz) % 360) + 360) % 360 / (360 / SECTORS)) % SECTORS;
    const el = elevation(view);
    let aim: Aim = 'ok';
    let changed = pace !== s.pace || sector !== s.sector;

    // Sectors passed too fast stay open: those frames are blurred, so the user goes over them again.
    if (!this.done && isUsableAim(view) && pace !== 'fast') {
      if (s.loop === 0) {
        if (el < -LOOP1_MAX_DOWN_DEG) aim = 'lower';
        else this.loop1Elevations.push(el);
      } else {
        const loop1 = this.loop1Elevations.reduce((a, b) => a + b, 0) / Math.max(1, this.loop1Elevations.length);
        if (el > Math.max(-80, loop1 - LOOP2_DROP_DEG)) aim = 'higher';
        else if (el < loop1 - LOOP2_MAX_DROP_DEG) aim = 'lower';
      }
      if (aim === 'ok' && !s.covered[s.loop][sector]) {
        const covered = s.covered.map((row) => row.slice());
        covered[s.loop][sector] = true;
        const count = covered[s.loop].filter(Boolean).length;
        this.state = { ...s, covered, loop: count >= NEEDED ? s.loop + 1 : s.loop };
        changed = true;
      }
    }
    changed ||= aim !== s.aim;
    this.state = { ...this.state, sector, pace, aim };
    return changed;
  }
}

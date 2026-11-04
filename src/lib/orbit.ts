// Насочена обиколка: два кръга около предмета (вторият по-отвисоко), следени чрез DeviceMotion. Азимутът на изгледа
// се завърта на 360° за кръг, докато телефонът остава насочен към предмета.

import { elevation, isUsableAim, type Vec3 } from './coverage.ts';

export const SECTORS = 24; // по 15°
export const LOOPS = 2;
const NEEDED = SECTORS - 2; // кръгът се счита за завършен, ако липсват най-много 30°
// Първият кръг трябва да гледа предмета почти хоризонтално: само отгоре плоският връх не се различава от конус върху
// него (от 40° отгоре: конус до 0,84 от радиуса на върха; от 20°: 0,36).
const LOOP1_MAX_DOWN_DEG = 20;
const LOOP2_DROP_DEG = 15; // вторият кръг трябва да гледа надолу поне с толкова повече от първия
// ...но не много повече: кадрите от двата кръга трябва да споделят достатъчна част от изгледа, за да се свържат
// (право отгоре се виждат само еднообразна маса и върхът на предмета и този кръг оставаше несвързан).
const LOOP2_MAX_DROP_DEG = 45;
export const GOOD_DEG_PER_S = [4, 24] as const; // около 15-90 с за кръг
const SHAKE_DEG_PER_S = 70;
const SMOOTH_S = 0.6;

export type Pace = 'still' | 'good' | 'fast';

export type OrbitState = {
  loop: number; // заснеманият кръг, броен от 0; LOOPS при завършване
  covered: boolean[][]; // [кръг][сектор], секторите се броят от началото на заснемането
  sector: number; // къде е телефонът в момента
  pace: Pace;
  aim: Aim; // дали височината на телефона е подходяща за заснемания кръг
};

// 'lower': телефонът гледа твърде стръмно надолу за този кръг. 'higher': втори кръг, телефонът не е вдигнат
// достатъчно.
export type Aim = 'ok' | 'lower' | 'higher';

const deg = (r: number) => (r * 180) / Math.PI;
export const azimuth = (v: Vec3) => (deg(Math.atan2(v[0], v[1])) + 360) % 360;
const wrap180 = (a: number) => ((((a + 180) % 360) + 360) % 360) - 180;

export class OrbitTracker {
  private startAz: number | null = null;
  private lastAz = 0;
  private lastT = 0;
  private speed = 0; // заглаждана промяна на азимута в °/с
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

  /* Заснетата част от първия кръг: под 1 сканирането вероятно ще е непълно. */
  get firstLoopFraction() {
    return Math.min(1, this.state.covered[0].filter(Boolean).length / NEEDED);
  }

  /* view: накъде сочи камерата; turnRate: обща скорост на завъртане в °/с; t: секунди. Връща true при промяна. */
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

    // Секторите, минати твърде бързо, остават отворени: тези кадри са размазани и потребителят минава отново през
    // тях.
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

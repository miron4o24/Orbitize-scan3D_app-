// Числата на екрана за заснемане на AI модела. Всяко е измерено от сензорите на телефона, а не украса, и всяко казва
// нещо, от което зависи моделът на сървъра (app/ai.py): кадри от всички страни и от две височини, без размазване, на
// светло.
//
// Рязкостта по време на запис е оценка от жироскопа: камерата не дава кадрите си, докато записва. Размазването на кадър
// е скоростта на завъртане по времето на експонацията, затова бързото въртене размазва, а бавното не. Истинската
// рязкост на кадрите се мери след записа (lib/videoCheck.ts), а сървърът избира най-рязките (app/video.py).

import { type Vec3 } from './coverage.ts';
import { LOOPS, SECTORS, type OrbitState } from './orbit.ts';

// Кадрите, които AI моделът взима от видеото (app/ai.py AI_FRAMES): толкова различни рязки гледни точки са нужни.
export const TARGET_VIEWS = 50;
const VIEW_STEP_DEG = 6; // нова гледна точка: поне толкова встрани от всяка вече преброена
const SHARP_FULL_DEG_S = 10; // до тази скорост на завъртане кадрите са рязки
const SHARP_ZERO_DEG_S = 60; // от тази нататък са размазани
export const SHARP_OK = 50; // кадър, който се брои за гледна точка

/* Оценка на рязкостта 0-100 от общата скорост на завъртане на телефона (°/с). */
export function sharpness(turnDegS: number): number {
  const k = (turnDegS - SHARP_FULL_DEG_S) / (SHARP_ZERO_DEG_S - SHARP_FULL_DEG_S);
  return Math.round(100 * Math.min(1, Math.max(0, 1 - k)));
}

export type Level = 'low' | 'mid' | 'high';
export const sharpLevel = (s: number): Level => (s < 40 ? 'low' : s < 70 ? 'mid' : 'high');
export const LEVEL_TEXT: Record<Level, string> = { low: 'НИСКА', mid: 'СРЕДНА', high: 'ВИСОКА' };

export const DARK_LUX = 10; // стая, твърде тъмна за заснемане
/* Светлината от сензора (lux) като 0-100 по логаритмична скала: 3 lux е 0, 1000 lux (ярка стая) е 100. */
export function lightPercent(lux: number): number {
  const p = (Math.log10(Math.max(lux, 1)) - 0.5) / 2.5;
  return Math.round(100 * Math.min(1, Math.max(0, p)));
}

/* Дял от двата кръга (ниско и отвисоко), заснет без размазване и от правилната височина. */
export function coverage(state: OrbitState | null): number {
  if (!state) return 0;
  const n = state.covered.reduce((a, row) => a + row.filter(Boolean).length, 0);
  return n / (LOOPS * SECTORS);
}

/**
 * Различните рязки гледни точки досега: накъдето е гледала камерата, докато е била достатъчно стабилна, и поне
 * VIEW_STEP_DEG встрани от всяка преброена. Това е горе-долу колко полезни кадъра ще намери сървърът във видеото.
 */
export class ViewCounter {
  private views: Vec3[] = [];
  private readonly cos = Math.cos((VIEW_STEP_DEG * Math.PI) / 180);

  get count() {
    return this.views.length;
  }

  /* True, когато гледната точка е нова. */
  add(view: Vec3, sharp: number): boolean {
    if (sharp < SHARP_OK) return false;
    for (const v of this.views) if (v[0] * view[0] + v[1] * view[1] + v[2] * view[2] > this.cos) return false;
    this.views.push(view);
    return true;
  }
}

/* Секторите от сегашния кръг, които още липсват и са най-близо до телефона: тях да заснеме след това. */
export function nextSectors(state: OrbitState | null, n = 2): number[] {
  if (!state || state.loop >= LOOPS) return [];
  const row = state.covered[state.loop];
  const gap = (i: number) => Math.min(Math.abs(i - state.sector), SECTORS - Math.abs(i - state.sector));
  return row
    .map((on, i) => (on ? -1 : i))
    .filter((i) => i >= 0)
    .sort((a, b) => gap(a) - gap(b))
    .slice(0, n);
}

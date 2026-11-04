// Положението на телефона по време на записа, за пътя на сървъра за едноцветни предмети (backend/app/motion.py):
// поставя кадрите според завъртането на телефона, когато съпоставянето на снимките не може. Всяка проба е [секунди от
// заявяването на записа, alpha, beta, gamma] (радиани, положение на устройството по W3C).
export type MotionSample = [number, number, number, number];

// Едно показание, както го е получил слушателят: [часовник на телефона в мс, времева отметка на сензора в с, alpha,
// beta, gamma].
export type RawMotion = [number, number, number, number, number];

const round = (x: number, digits: number) => Math.round(x * 10 ** digits) / 10 ** digits;

// Отметките на сензора са точни, но се броят от друго начало; времената на слушателя са по часовника на записа, но
// закъсняват с променливо време. Затова: времена на сензора, отместени с медианната разлика.
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

// Дати на български, еднакви на всеки телефон (езикът на самия телефон може да е друг).
const MONTHS = ['яну', 'фев', 'мар', 'апр', 'май', 'юни', 'юли', 'авг', 'сеп', 'окт', 'ное', 'дек'];
const MONTHS_LONG = ['януари', 'февруари', 'март', 'април', 'май', 'юни', 'юли', 'август', 'септември', 'октомври', 'ноември', 'декември'];

const pad = (n: number) => String(n).padStart(2, '0');

/** 26.10.2026 */
export function shortDate(ms: number) {
  const d = new Date(ms);
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`;
}

/** 26 окт */
export function dayMonth(ms: number) {
  const d = new Date(ms);
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

/** октомври 2026 */
export function monthYear(ms: number) {
  const d = new Date(ms);
  return `${MONTHS_LONG[d.getMonth()]} ${d.getFullYear()}`;
}

/** 7 мин, 1 ч 05 мин */
export function duration(seconds: number) {
  const m = Math.round(seconds / 60);
  if (m < 60) return `${Math.max(1, m)} мин`;
  return `${Math.floor(m / 60)} ч ${pad(m % 60)} мин`;
}

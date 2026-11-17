import type { ProMethod } from './api';
import { getSession } from './session';

// Двата начина за създаване на модел с едни и същи имена на всеки екран: AI моделът и инженерният режим, и двата за
// всеки профил. Стандартното сканиране е премахнато; старите му модели остават в библиотеката.

// По-старите сканирания може да са "quick" (по-малко и по-малки кадри, около 10 мин); новите никога не са.
export type Quality = 'quick' | 'detailed';

// AI моделите (сървър app/ai.py, app/pipeline.py EXPRESS_*). Сега може да се стартира само "single"; останалите са тук за
// вече направени с тях сканирания и за текстови промени ("edit").
export const PRO_INFO: Record<ProMethod, { label: string; time: string; minutes: number }> = {
  express: { label: 'Експресно', time: '~5 мин', minutes: 5 },
  edit: { label: 'Промяна с AI', time: '~1–3 мин', minutes: 2 },
  single: { label: 'AI модел', time: '~15 мин', minutes: 15 },
  claude: { label: 'Claude дизайн', time: '~3 мин', minutes: 3 },
  photo: { label: 'AI от снимки', time: '~3 мин', minutes: 3 },
};

// Инженерният режим (сървър app/cad): безплатен, детайлът се строи от измерените размери.
export const CAD_INFO = { label: 'Инженерен CAD', time: '~5 мин' };

/* Как е създаден моделът, за етикета в прегледа. */
export const madeWith = (ai: ProMethod | 'cad') => (ai === 'cad' ? CAD_INFO.label : PRO_INFO[ai].label);

/* Дали може да се създаде AI модел: всеки влязъл профил, освен ако сървърът няма AI модела. */
export function aiAvailable(): boolean {
  const s = getSession();
  // Скрива го само ясното "не" на сървъра: профил, запазен преди сървърът да съобщи какъв AI има, не бива да го
  // скрива.
  return s?.kind === 'account' && s.user.ai?.single !== false;
}

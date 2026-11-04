import type { ProMethod } from './api';
import { getSession } from './session';

// Двата начина за създаване на модел с едни и същи имена на всеки екран: стандартното сканиране (за всички) и AI
// модела (Pro).

// По-старите сканирания може да са "quick" (по-малко и по-малки кадри, около 10 мин); новите никога не са.
export type Quality = 'quick' | 'detailed';

export const SCAN_INFO = { label: 'Стандартно', time: '~35 мин', minutes: 35 };

export const ONE_PHOTO_INFO = { label: 'От една снимка', time: '~1 мин', minutes: 1 };

// Pro (сървър app/ai.py, app/pipeline.py EXPRESS_*). Сега може да се стартира само "single"; останалите са тук за
// вече направени с тях сканирания и за текстови промени ("edit").
export const PRO_INFO: Record<ProMethod, { label: string; time: string; minutes: number }> = {
  express: { label: 'Експресно', time: '~5 мин', minutes: 5 },
  edit: { label: 'Промяна с AI', time: '~1–3 мин', minutes: 2 },
  single: { label: 'AI модел', time: '~15 мин', minutes: 15 },
  claude: { label: 'Claude дизайн', time: '~3 мин', minutes: 3 },
  photo: { label: 'AI от снимки', time: '~3 мин', minutes: 3 },
};

/* Единственият AI модел на безплатен профил, още неизползван. */
export function aiTrial(): boolean {
  const s = getSession();
  if (s?.kind !== 'account' || s.user.pro) return false;
  const l = s.user.limits?.ai;
  return !!l && l.limit != null && l.used < l.limit && s.user.ai?.single !== false;
}

/* Дали този профил може да създаде AI модел (Pro или безплатния му опит, и сървърът го поддържа). */
export function aiAvailable(): boolean {
  const s = getSession();
  // Скрива го само ясното "не" на сървъра: профил, запазен преди сървърът да съобщи какъв AI има, не бива да го
  // скрива.
  return s?.kind === 'account' && ((!!s.user.pro && s.user.ai?.single !== false) || aiTrial());
}

import type { ProMethod } from './api';

// Начините за създаване на модел, с едни и същи имена на всеки екран: AI модел от видео и от една снимка, и двата за
// всички профили.

// По-старите сканирания може да са "quick" (по-малко и по-малки кадри, около 10 мин); новите никога не са.
export type Quality = 'quick' | 'detailed';

// Старото стандартно сканиране е спряно за всички: името и времето му остават само за вече направените с него модели.
export const OLD_SCAN_INFO = { label: 'Стандартно', minutes: 35 };

export const ONE_PHOTO_INFO = { label: 'От една снимка', time: '~2 мин', minutes: 2 };

// Сървър app/ai.py, app/pipeline.py EXPRESS_*. Сега може да се стартира само "single"; останалите са тук за вече
// направени с тях сканирания и за текстови промени ("edit").
export const PRO_INFO: Record<ProMethod, { label: string; time: string; minutes: number }> = {
  express: { label: 'Експресно', time: '~5 мин', minutes: 5 },
  edit: { label: 'Промяна с AI', time: '~1–3 мин', minutes: 2 },
  single: { label: 'AI модел', time: 'до 8 мин', minutes: 8 },
  claude: { label: 'Claude дизайн', time: '~3 мин', minutes: 3 },
  photo: { label: 'AI от снимки', time: '~3 мин', minutes: 3 },
  cad: { label: 'CAD модел', time: '~3 мин', minutes: 3 },
};

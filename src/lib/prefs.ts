import type { Method } from './api';

// Имената и времето на всеки начин за създаване на модел, еднакви на всеки екран. Нови модели се създават само с
// "single" (AI моделът от видео); останалите са тук за вече направени с тях модели и за текстови промени ("edit").
export const PRO_INFO: Record<Method, { label: string; time: string; minutes: number }> = {
  edit: { label: 'Промяна с AI', time: '~1–3 мин', minutes: 2 },
  single: { label: 'AI модел', time: '~15 мин', minutes: 15 },
  claude: { label: 'Claude дизайн', time: '~3 мин', minutes: 3 },
  photo: { label: 'AI от снимки', time: '~3 мин', minutes: 3 },
};

// AI моделът по предишния начин (сървър: без посоките на повърхността от app/normals.py): по-бърз, но по-груб по
// гладките части (лице, плешива глава).
export const CLASSIC_INFO = { label: 'Предишен AI модел', time: '~12 мин', minutes: 12 };

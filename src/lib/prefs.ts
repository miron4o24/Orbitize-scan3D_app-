import type { ProMethod } from './api';
import { getSession } from './session';

// The two ways to make a model, named the same on every screen: the standard scan (everyone) and the AI model (Pro).

// Older scans may be "quick" (fewer, smaller frames, ~10 min); new ones are never.
export type Quality = 'quick' | 'detailed';

export const SCAN_INFO = { label: 'Стандартно', time: '~35 мин', minutes: 35 };

// Pro (backend app/ai.py, app/pipeline.py EXPRESS_*). Only "single" can be started now; the others stay here for
// scans already made with them, and for text changes ("edit").
export const PRO_INFO: Record<ProMethod, { label: string; time: string; minutes: number }> = {
  express: { label: 'Експресно', time: '~5 мин', minutes: 5 },
  edit: { label: 'Промяна с AI', time: '~1–3 мин', minutes: 2 },
  single: { label: 'AI модел', time: '~15 мин', minutes: 15 },
  claude: { label: 'Claude дизайн', time: '~3 мин', minutes: 3 },
  photo: { label: 'AI от снимки', time: '~3 мин', minutes: 3 },
};

/** A free account's one AI model, not yet used. */
export function aiTrial(): boolean {
  const s = getSession();
  if (s?.kind !== 'account' || s.user.pro) return false;
  const l = s.user.limits?.ai;
  return !!l && l.limit != null && l.used < l.limit && s.user.ai?.single !== false;
}

/** Whether this account can make an AI model (Pro or its free one, and the server has it). */
export function aiAvailable(): boolean {
  const s = getSession();
  // Only a server's clear "no" hides it: a profile saved before the server told which AI it has must not.
  return s?.kind === 'account' && ((!!s.user.pro && s.user.ai?.single !== false) || aiTrial());
}

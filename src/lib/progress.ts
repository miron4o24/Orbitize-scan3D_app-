import { File, Paths } from 'expo-file-system';
import { useSyncExternalStore } from 'react';

import { ApiError, getProgress, type Badge, type Progress } from './api';
import { currentUserId } from './session';

// The account's points, level, badges and weekly challenges (backend app/progress.py), and the moments worth a
// celebration: points earned, a challenge done, a badge, a new level. What was last seen is kept per account in a
// file, so each is celebrated once, on whatever screen the app is when it is noticed (components/Celebration.tsx).

export type Celebration =
  | { kind: 'points'; points: number; label: string }
  | { kind: 'challenge'; title: string; points: number }
  | { kind: 'badge'; badge: Badge }
  | { kind: 'level'; number: number; name: string }
  | { kind: 'welcome'; points: number; badges: number };

type Seen = { points: number; level: number; badges: string[]; challenges: string[] };

export type Queued = Celebration & { id: number };
type State = { owner: string | null; progress: Progress | null; queue: Queued[] };
let serial = 0;

let state: State = { owner: null, progress: null, queue: [] };
let unsupported = false; // a server from before points
let lastCheck = 0;
let running: Promise<void> | null = null;
const listeners = new Set<() => void>();

function set(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

/** The signed-in account's progress (null until first fetched, for guests, or on a server without points). */
export function useProgress() {
  const s = useSyncExternalStore(subscribe, () => state);
  return s.owner === currentUserId() ? s.progress : null;
}

export const useCelebrations = () => useSyncExternalStore(subscribe, () => state.queue);

/** The celebration on screen is done with. */
export const dismissCelebration = () => set({ queue: state.queue.slice(1) });

const seenFile = (userId: string) => new File(Paths.document, `progress-${userId}.json`);

function loadSeen(userId: string): Seen | null {
  try {
    const f = seenFile(userId);
    return f.exists ? (JSON.parse(f.textSync()) as Seen) : null;
  } catch {
    return null;
  }
}

function saveSeen(userId: string, p: Progress, before: Seen | null) {
  const week = p.challenges.items.filter((c) => c.done).map((c) => `${p.challenges.week}:${c.id}`);
  const seen: Seen = {
    points: p.points,
    level: p.level.number,
    badges: p.badges.filter((b) => b.unlocked).map((b) => b.id),
    challenges: [...new Set([...(before?.challenges ?? []), ...week])].slice(-30),
  };
  try {
    seenFile(userId).writeSync(JSON.stringify(seen));
  } catch {} // not kept: celebrated again next time, no worse
}

/** What changed since `seen`, in the order it is shown: the points, the challenges, the badges, the level last. */
export function celebrations(p: Progress, seen: Seen | null): Celebration[] {
  const unlocked = p.badges.filter((b) => b.unlocked);
  // First time on this phone: the history so far in one card, not a card per old badge.
  if (!seen) return p.points > 0 ? [{ kind: 'welcome', points: p.points, badges: unlocked.length }] : [];
  const out: Celebration[] = [];
  const challenges = p.challenges.items.filter((c) => c.done && !seen.challenges.includes(`${p.challenges.week}:${c.id}`));
  const earned = p.points - seen.points - challenges.reduce((sum, c) => sum + c.points, 0);
  if (earned > 0) out.push({ kind: 'points', points: earned, label: p.recent.find((r) => r.kind !== 'challenge')?.label ?? 'Точки' });
  for (const c of challenges) out.push({ kind: 'challenge', title: c.title, points: c.points });
  for (const b of unlocked) if (!seen.badges.includes(b.id)) out.push({ kind: 'badge', badge: b });
  if (p.level.number > seen.level) out.push({ kind: 'level', number: p.level.number, name: p.level.name });
  return out;
}

/** Fetches the progress and queues what is new. Often called (after a scan, a post, a like, a test, on returning to
 * the app): at most every few seconds unless `now`. */
export function checkProgress(now = false): Promise<void> {
  const userId = currentUserId();
  if (!userId || unsupported) return Promise.resolve();
  if (running) return running;
  if (!now && Date.now() - lastCheck < 5000 && state.owner === userId) return Promise.resolve();
  running = (async () => {
    try {
      const p = await getProgress();
      lastCheck = Date.now();
      if (currentUserId() !== userId) return; // signed out meanwhile
      const seen = loadSeen(userId);
      const fresh = state.owner === userId ? state.queue : [];
      set({ owner: userId, progress: p, queue: [...fresh, ...celebrations(p, seen).map((c) => ({ ...c, id: ++serial }))] });
      saveSeen(userId, p, seen);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404 && e.message === 'Not Found') unsupported = true;
    } finally {
      running = null;
    }
  })();
  return running;
}

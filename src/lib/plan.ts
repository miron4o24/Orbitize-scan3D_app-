import type { LimitKind, Profile } from './session';

// The free plan's limits as the app shows them (backend app/plans.py counts them): stress tests on a few models, AR
// for a few scans; Pro has none. A model already counted can be used again freely.

export type Allowance = { unlimited: true } | { unlimited: false; left: number; limit: number; counted: boolean };

/** What the account may still do of `kind` for this model; null: no limits to show (guests, older servers). */
export function allowance(user: Profile | null | undefined, kind: LimitKind, jobId: string | undefined): Allowance | null {
  const l = user?.limits?.[kind];
  if (!l) return null;
  if (l.limit == null) return { unlimited: true };
  return { unlimited: false, left: Math.max(0, l.limit - l.used), limit: l.limit, counted: !!jobId && l.refs.includes(jobId) };
}

/** True when the free plan's uses are spent and this model is not one of them. */
export const blocked = (a: Allowance | null) => !!a && !a.unlimited && !a.counted && a.left === 0;

/** A short line under the button: how many are left, or nothing for Pro and models already counted. */
export function allowanceNote(a: Allowance | null): string | undefined {
  if (!a || a.unlimited || a.counted) return undefined;
  if (a.left === 0) return 'Използвахте всички безплатни. С Pro са без ограничение.';
  return `Остават ${a.left} от ${a.limit}`;
}

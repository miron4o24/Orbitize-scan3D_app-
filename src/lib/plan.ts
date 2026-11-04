import type { LimitKind, Profile } from './session';

// Ограниченията на безплатния план, както ги показва приложението (сървър app/plans.py ги брои): тестове за здравина
// на няколко модела и AR за няколко сканирания; Pro няма ограничения. Вече преброен модел може да се използва отново
// свободно.

export type Allowance = { unlimited: true } | { unlimited: false; left: number; limit: number; counted: boolean };

/**
 * Какво още може профилът от вид `kind` за този модел; null: няма ограничения за показване (гости, по-стари сървъри).
 */
export function allowance(user: Profile | null | undefined, kind: LimitKind, jobId: string | undefined): Allowance | null {
  const l = user?.limits?.[kind];
  if (!l) return null;
  if (l.limit == null) return { unlimited: true };
  return { unlimited: false, left: Math.max(0, l.limit - l.used), limit: l.limit, counted: !!jobId && l.refs.includes(jobId) };
}

/* True, когато възможностите на безплатния план са изчерпани и този модел не е сред тях. */
export const blocked = (a: Allowance | null) => !!a && !a.unlimited && !a.counted && a.left === 0;

/* Кратък ред под бутона: колко остават или нищо за Pro и вече преброени модели. */
export function allowanceNote(a: Allowance | null): string | undefined {
  if (!a || a.unlimited || a.counted) return undefined;
  if (a.left === 0) return 'Използвахте всички безплатни. С Pro са без ограничение.';
  return `Остават ${a.left} от ${a.limit}`;
}

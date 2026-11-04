import { File, Paths } from 'expo-file-system';
import * as SecureStore from 'expo-secure-store';
import { useSyncExternalStore } from 'react';

import type { ProMethod } from './api';

// Кой използва приложението: влязъл профил (сесийният ключ е в защитено хранилище, а профилът във файл, за да се
// отваря приложението влязло и без мрежа) или все още никой (екран за вход).

export type Profile = {
  id: string;
  email: string;
  name: string;
  google_picture: string | null;
  has_avatar: boolean;
  created: number; // секунди по Unix
  stats: { scans: number; done: number; seconds: number; shared: number; published?: number }; // published: по-нови сървъри
  // Pro: може да създава AI модели; ai: за кои сървърът има ключове; owner: страницата за Pro (по-нови сървъри)
  pro?: boolean;
  pro_until?: number | null; // Pro чрез плащане: до кога (секунди по Unix)
  billing?: boolean; // Pro може да се купи в приложението (сървърът е настроен за това)
  pro_product?: string; // продуктът му в магазина
  owner?: boolean;
  ai?: Partial<Record<ProMethod, boolean>>;
  // Броените използвания от безплатния план (сървър app/plans.py): limit е null при Pro; refs: вече преброените
  // модели
  limits?: Record<LimitKind, { used: number; limit: number | null; refs: string[] }>;
};

export type LimitKind = 'stress' | 'ar' | 'ai';

export type Session = { kind: 'account'; token: string; user: Profile } | null;

const TOKEN_KEY = 'session_token';
const profileFile = () => new File(Paths.document, 'profile.json');
const guestFile = () => new File(Paths.document, 'guest'); // остатък от по-стари версии, които имаха гости

function load(): Session {
  try {
    if (guestFile().exists) guestFile().delete();
    const token = SecureStore.getItem(TOKEN_KEY);
    const f = profileFile();
    if (token && f.exists) return { kind: 'account', token, user: JSON.parse(f.textSync()) as Profile };
  } catch {} // не се чете: влезте отново
  return null;
}

let current: Session = load();
let avatarVersion = Date.now(); // променя се при смяна на собствената снимка, за да се презаредят кешираните изображения
const listeners = new Set<() => void>();

function emit(next: Session) {
  current = next;
  listeners.forEach((l) => l());
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

export const useSession = () => useSyncExternalStore(subscribe, () => current);
export const getSession = () => current;
export const sessionToken = () => (current?.kind === 'account' ? current.token : null);
export const currentUserId = () => (current?.kind === 'account' ? current.user.id : null);
export const getAvatarVersion = () => avatarVersion;

export async function signedIn(token: string, user: Profile) {
  await SecureStore.setItemAsync(TOKEN_KEY, token);
  profileFile().writeSync(JSON.stringify(user));
  avatarVersion = Date.now();
  emit({ kind: 'account', token, user });
}

/* Нов профил от сървъра (променени име, снимка или статистика). */
export function updateProfile(user: Profile) {
  if (current?.kind !== 'account') return;
  if (user.has_avatar !== current.user.has_avatar) avatarVersion = Date.now();
  profileFile().writeSync(JSON.stringify(user));
  emit({ ...current, user });
}

export function avatarChanged() {
  avatarVersion = Date.now();
  if (current) emit({ ...current });
}

/* Забравя профила на този телефон (сесията на сървъра се приключва от auth.signOut). */
export async function clearSession() {
  await SecureStore.deleteItemAsync(TOKEN_KEY).catch(() => {});
  if (profileFile().exists) profileFile().delete();
  emit(null);
}

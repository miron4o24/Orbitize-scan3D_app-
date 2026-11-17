import { useSyncExternalStore } from 'react';

// Прозорците на приложението в неговия дизайн, вместо системните (Alert.alert): същото извикване, show(title,
// message, buttons). Показва ги components/DialogHost.tsx (в app/_layout.tsx). С няколко избора и `sheet` (или икони
// по бутоните) излиза отдолу като списък; иначе е прозорец в средата.

export type DialogButton = {
  text: string;
  style?: 'default' | 'cancel' | 'destructive';
  onPress?: () => void;
  icon?: string; // име на Ionicons (за избор от списък)
  sub?: string; // малък ред под избора (например колко време отнема)
};

export type Dialog = { id: number; title: string; message?: string; buttons: DialogButton[]; sheet: boolean };

let current: Dialog | null = null;
let next = 1;
const queue: Dialog[] = [];
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

/* Показва прозорец; без бутони има само "OK". Следващ прозорец чака, докато предният се затвори. */
export function showDialog(title: string, message?: string, buttons?: DialogButton[], options?: { sheet?: boolean }) {
  const list = buttons?.length ? buttons : [{ text: 'OK' }];
  const d: Dialog = { id: next++, title, message, buttons: list, sheet: !!options?.sheet || list.some((b) => b.icon) };
  if (current) queue.push(d);
  else current = d;
  emit();
}

/* Затваря прозореца (и изпълнява избраното след това, за да може то да отвори нов прозорец). */
export function closeDialog(button?: DialogButton) {
  current = queue.shift() ?? null;
  emit();
  button?.onPress?.();
}

export const useDialog = () =>
  useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => current,
  );

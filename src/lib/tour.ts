import { File, Paths } from 'expo-file-system';
import { useSyncExternalStore } from 'react';

// Обиколката за първи път: стъпка по стъпка какво прави всеки бутон на началния екран и в лентата долу. Показва се
// веднъж (tour.json); собственикът на приложението може да я пусне отново от профила, за проба.

export type TourStep = {
  target?: string;
  icon: string;
  title: string;
  text: string;
};

export const STEPS: TourStep[] = [
  {
    icon: 'planet-outline',
    title: 'Добре дошли в Orbitize',
    text: 'Тук от видео или снимки на предмет става 3D модел, който можете да отпечатате в реален размер. Ето какво прави всеки бутон.',
  },
  {
    target: 'home-ai',
    icon: 'sparkles-outline',
    title: 'AI модел',
    text: 'Обиколете предмета с камерата (около минута). Сървърът прави 3D модел с цветовете му. Сложете предмета върху нещо шарено и снимайте на равна светлина.',
  },
  {
    target: 'home-cad',
    icon: 'construct-outline',
    title: 'Инженерен режим',
    text: 'За детайли и резервни части: снимки от всички страни и няколко мерки с шублер дават точен CAD модел 1:1.',
  },
  {
    target: 'home-upload',
    icon: 'cloud-upload-outline',
    title: 'Готово видео',
    text: 'Имате видео около предмета, заснето по-рано? Качете го оттук, без да снимате отново.',
  },
  {
    target: 'tab-scan',
    icon: 'add',
    title: 'Ново сканиране',
    text: 'Този бутон е винаги долу в средата: започва нов модел от всеки екран.',
  },
  {
    target: 'tab-scans',
    icon: 'cube-outline',
    title: 'Сканирания',
    text: 'Всичките ви модели. Докоснете модел, за да го отворите; задръжте, за да го изтриете.',
  },
  {
    target: 'tab-community',
    icon: 'people-outline',
    title: 'Общност',
    text: 'Модели, които други са споделили. Вашите също могат да са тук: „Публикувай в общността“ при модела.',
  },
  {
    target: 'tab-profile',
    icon: 'person-outline',
    title: 'Профил',
    text: 'Вашето ниво, тема (тъмна или светла) и настройки на профила.',
  },
  {
    icon: 'hand-left-outline',
    title: 'При готовия модел',
    text: 'Плъзнете с пръст, за да го завъртите, щипнете за приближаване, с два пръста го местите, двойно докосване го връща. „Без цвят“ показва самата форма, а STL е файлът за принтера.',
  },
];

type Rect = { x: number; y: number; width: number; height: number };
type State = { step: number | null; targets: Record<string, Rect> };

let state: State = { step: null, targets: {} };
const listeners = new Set<() => void>();
const emit = (next: Partial<State>) => {
  state = { ...state, ...next };
  for (const l of listeners) l();
};
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

const seenFile = () => new File(Paths.document, 'tour.json');

export function tourSeen(): boolean {
  try {
    return seenFile().exists;
  } catch {
    return true; // без достъп до диска: по-добре без обиколка, отколкото всеки път
  }
}

export function startTour() {
  emit({ step: 0 });
}

export function nextStep() {
  if (state.step == null) return;
  if (state.step + 1 >= STEPS.length) endTour();
  else emit({ step: state.step + 1 });
}

export function endTour() {
  emit({ step: null });
  try {
    seenFile().write(JSON.stringify({ seen: Date.now() }));
  } catch {}
}

/* Отново от началото (бутонът на собственика в профила). */
export function restartTour() {
  try {
    if (seenFile().exists) seenFile().delete();
  } catch {}
  startTour();
}

export function setTarget(id: string, rect: Rect) {
  const old = state.targets[id];
  if (old && old.x === rect.x && old.y === rect.y && old.width === rect.width && old.height === rect.height) return;
  emit({ targets: { ...state.targets, [id]: rect } });
}

export function useTour(): State {
  return useSyncExternalStore(subscribe, () => state);
}

import { File, Paths } from 'expo-file-system';
import { useSyncExternalStore } from 'react';
import { type ImageStyle, StyleSheet, type TextStyle, type ViewStyle } from 'react-native';

// Визията на Orbitize: технически инструмент за сканиране. Тъмна по подразбиране (изборът, профил → Тема, се пази в
// theme.txt). Циан за действия, избор, връзки и напредъка на сканирането; кехлибарен "сигнал" само за мерки, живата
// точка на сканирането и предупрежденията при заснемане. Линии от 1px вместо сенки, остри ъгли (до 8, освен
// хапчетата), IBM Plex Sans за текст и IBM Plex Mono за всяко число, етикет, размер, номер, време и процент
// (шрифтовете се зареждат в app/_layout.tsx). Без градиенти освен фона на 3D изгледа и сцената.

export type Scheme = 'dark' | 'light';

export type Palette = {
  scheme: Scheme;
  bg: string;
  surface: string;
  surface2: string;
  surface3: string;
  line: string;
  lineStrong: string;
  text: string;
  text2: string;
  muted: string;
  accent: string;
  accentStrong: string;
  accentSoft: string;
  success: string;
  successSoft: string;
  warning: string;
  warningSoft: string;
  danger: string;
  dangerSoft: string;
  primary: string; // основният бутон
  onPrimary: string;
  accentLine: string; // рамка в цвета за акцент (препоръчан избор)
  glass: string; // прозрачни ленти и етикети върху съдържание
  viewer: readonly [string, string]; // радиалният фон на 3D изгледа, от центъра към ръба
  stage: readonly [string, string, string]; // радиалният фон на сцената с изображението, от центъра към ръба
  backdrop: string; // зад диалоговите прозорци
  signal: string; // measurements, live scan point, capture warnings. Never an action.
  tint: readonly [string, string, string]; // soft fills behind model thumbnails, rotate by index
};

const dark: Palette = {
  scheme: 'dark',
  bg: '#080b10',
  surface: '#0f141b',
  surface2: '#161d26',
  surface3: '#1e2733',
  line: '#1f2a36',
  lineStrong: '#56687a',
  text: '#e6edf5',
  text2: '#a9b8c7',
  muted: '#8795a5',
  accent: '#35d4ff',
  accentStrong: '#00b8e6',
  accentSoft: '#0c2b3a',
  success: '#3ddc84',
  successSoft: 'rgba(61,220,132,0.12)',
  warning: '#ffc24d',
  warningSoft: 'rgba(255,194,77,0.12)',
  danger: '#ff6b6b',
  dangerSoft: 'rgba(255,107,107,0.12)',
  primary: '#35d4ff',
  onPrimary: '#00131c',
  accentLine: 'rgba(53,212,255,0.4)',
  glass: 'rgba(8,11,16,0.82)',
  viewer: ['#161d26', '#080b10'],
  stage: ['#161d26', '#0b0f14', '#080b10'],
  backdrop: 'rgba(0,0,0,0.65)',
  signal: '#ffb829',
  tint: ['#131d28', '#12202a', '#171c2b'],
};

const light: Palette = {
  scheme: 'light',
  bg: '#edf0f4',
  surface: '#ffffff',
  surface2: '#e1e6ed',
  surface3: '#d3dae3',
  line: '#cbd3dc',
  lineStrong: '#7a8896',
  text: '#0b1116',
  text2: '#33404d',
  muted: '#566370',
  accent: '#0866ad',
  accentStrong: '#065490',
  accentSoft: '#d9eaf7',
  success: '#116b34',
  successSoft: 'rgba(17,107,52,0.1)',
  warning: '#8a5a00',
  warningSoft: 'rgba(138,90,0,0.1)',
  danger: '#b3261e',
  dangerSoft: 'rgba(179,38,30,0.09)',
  primary: '#0866ad',
  onPrimary: '#ffffff',
  accentLine: '#9cc3e3',
  glass: 'rgba(237,240,244,0.85)',
  viewer: ['#ffffff', '#e1e6ed'],
  stage: ['#ffffff', '#e4e9ee', '#d3dae3'],
  backdrop: 'rgba(11,17,22,0.4)',
  signal: '#8f5200',
  tint: ['#dde6f0', '#d8e3ea', '#e4e4ee'],
};

export const palettes: Record<Scheme, Palette> = { dark, light };

export const brand = ['#35d4ff', '#0866ad'] as const;
export const brandDeep = ['#0866ad', '#065490'] as const; // legacy only, remove when unused

export const font = {
  regular: 'IBMPlexSans_400Regular',
  medium: 'IBMPlexSans_500Medium',
  semibold: 'IBMPlexSans_600SemiBold',
  mono: 'IBMPlexMono_400Regular',
  monoMedium: 'IBMPlexMono_500Medium',
};

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };
// sm: buttons, inputs, chips, thumbnails. md: cards, panels. pill: status pills, record button.
export const radius = { sm: 4, md: 8, pill: 999 };

// Само размери: цветът идва от палитрата (components/ui.tsx). Шрифтовете носят дебелината си в семейството: без
// fontWeight (Android би минал към системния шрифт).
export const text = {
  display: { fontFamily: font.semibold, fontSize: 40, lineHeight: 44, letterSpacing: -0.8 },
  title: { fontFamily: font.semibold, fontSize: 28, lineHeight: 34, letterSpacing: -0.28 },
  h2: { fontFamily: font.semibold, fontSize: 18, lineHeight: 24, letterSpacing: 0 },
  h3: { fontFamily: font.semibold, fontSize: 15, lineHeight: 22, letterSpacing: 0 },
  lead: { fontFamily: font.regular, fontSize: 15, lineHeight: 22 },
  small: { fontFamily: font.regular, fontSize: 13, lineHeight: 18 },
  label: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 14, letterSpacing: 0.66, textTransform: 'uppercase' },
  mono: { fontFamily: font.mono, fontSize: 13, lineHeight: 18 },
  number: { fontFamily: font.monoMedium, fontSize: 22, lineHeight: 28, letterSpacing: 0 },
} satisfies Record<string, TextStyle>;

// --- Избраната тема ------------------------------------------------------------------------------------------------

const file = () => new File(Paths.document, 'theme.txt');

function load(): Scheme {
  try {
    const f = file();
    return f.exists && f.textSync().trim() === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

let scheme = load();
const listeners = new Set<() => void>();
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

export function setScheme(next: Scheme) {
  scheme = next;
  try {
    file().writeSync(next);
  } catch {} // не се пази: при следващото стартиране отново е тъмна
  listeners.forEach((l) => l());
}

export const useScheme = () => useSyncExternalStore(subscribe, () => scheme);
export const useColors = () => palettes[useScheme()];

/**
 * Стилове, които използват палитрата, създадени веднъж за всяка тема: `const useStyles = makeStyles((c) => ({ ...
 * }))`.
 */
export function makeStyles<T extends { [K in keyof T]: ViewStyle | TextStyle | ImageStyle }>(build: (c: Palette) => T) {
  const made: Partial<Record<Scheme, T>> = {};
  return () => {
    const s = useScheme();
    return (made[s] ??= StyleSheet.create(build(palettes[s])));
  };
}

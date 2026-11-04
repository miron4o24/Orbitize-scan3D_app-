import { File, Paths } from 'expo-file-system';
import { useSyncExternalStore } from 'react';
import { type ImageStyle, StyleSheet, type TextStyle, type ViewStyle } from 'react-native';

// Scanly's look, after the website: calm and flat, with the brand gradient on the one main action of a screen. Surfaces
// with 1px lines instead of shadows, Onest for text,
// JetBrains Mono for numbers, sizes, dates and small uppercase labels (fonts loaded in app/_layout.tsx). Dark by
// default; the choice (profile → Тема) is kept in theme.txt.

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
  primary: string; // the primary button
  onPrimary: string;
  accentLine: string; // a border in the accent colour (a recommended choice)
  glass: string; // see-through bars and labels over content
  viewer: readonly [string, string]; // the 3D view's radial background, centre → edge
  stage: readonly [string, string, string]; // the picture stage's radial background, centre → edge
  backdrop: string; // behind dialogs
};

const dark: Palette = {
  scheme: 'dark',
  bg: '#09090b',
  surface: '#131317',
  surface2: '#1a1a1f',
  surface3: '#25252b',
  line: 'rgba(255,255,255,0.09)',
  lineStrong: 'rgba(255,255,255,0.16)',
  text: '#f2f2f4',
  text2: '#bcbcc4',
  muted: '#8c8c96',
  accent: '#9d9bff',
  accentStrong: '#6d6af7',
  accentSoft: 'rgba(109,106,247,0.14)',
  success: '#4ade80',
  successSoft: 'rgba(74,222,128,0.12)',
  warning: '#fbbf24',
  warningSoft: 'rgba(251,191,36,0.12)',
  danger: '#f87171',
  dangerSoft: 'rgba(248,113,113,0.12)',
  primary: '#f2f2f4',
  onPrimary: '#09090b',
  accentLine: 'rgba(157,155,255,0.38)',
  glass: 'rgba(9,9,11,0.78)',
  viewer: ['#1d1d23', '#0c0c0f'],
  stage: ['#19191f', '#0d0d10', '#09090b'],
  backdrop: 'rgba(0,0,0,0.6)',
};

const light: Palette = {
  scheme: 'light',
  bg: '#fafafa',
  surface: '#ffffff',
  surface2: '#f4f4f5',
  surface3: '#ececee',
  line: '#e7e7ea',
  lineStrong: '#d6d6db',
  text: '#0a0a0c',
  text2: '#3f3f46',
  muted: '#71717a',
  accent: '#4f46e5',
  accentStrong: '#4338ca',
  accentSoft: '#eef0ff',
  success: '#15803d',
  successSoft: 'rgba(21,128,61,0.09)',
  warning: '#b45309',
  warningSoft: 'rgba(180,83,9,0.09)',
  danger: '#dc2626',
  dangerSoft: 'rgba(220,38,38,0.08)',
  primary: '#0a0a0c',
  onPrimary: '#ffffff',
  accentLine: '#c7cbfb',
  glass: 'rgba(250,250,250,0.82)',
  viewer: ['#ffffff', '#efeff2'],
  stage: ['#ffffff', '#f1f1f4', '#ebebef'],
  backdrop: 'rgba(10,10,12,0.35)',
};

export const palettes: Record<Scheme, Palette> = { dark, light };

/** The logo and the avatar. */
export const brand = ['#3b6cf6', '#8b5cf6'] as const;
/** Under white text and icons: the one main action of a screen. */
export const brandDeep = ['#2f5ee8', '#7c3aed'] as const;

export const font = {
  regular: 'Onest_400Regular',
  medium: 'Onest_500Medium',
  semibold: 'Onest_600SemiBold',
  mono: 'JetBrainsMono_400Regular',
  monoMedium: 'JetBrainsMono_500Medium',
};

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 };
export const radius = { sm: 12, md: 18, pill: 999 };

// Sizes only: the colour comes from the palette (components/ui.tsx). Custom fonts carry their weight in the family:
// no fontWeight (Android would fall back to the system font).
// The website's scale: big tight headlines, quiet body text, mono details.
export const text = {
  display: { fontFamily: font.semibold, fontSize: 40, lineHeight: 44, letterSpacing: -1.8 },
  title: { fontFamily: font.semibold, fontSize: 32, lineHeight: 36, letterSpacing: -1.2 },
  h2: { fontFamily: font.semibold, fontSize: 22, lineHeight: 27, letterSpacing: -0.6 },
  h3: { fontFamily: font.semibold, fontSize: 16, lineHeight: 21, letterSpacing: -0.25 },
  lead: { fontFamily: font.regular, fontSize: 16.5, lineHeight: 25 },
  body: { fontFamily: font.regular, fontSize: 15, lineHeight: 22 },
  small: { fontFamily: font.regular, fontSize: 13.5, lineHeight: 19 },
  label: { fontFamily: font.mono, fontSize: 11, lineHeight: 14, letterSpacing: 1, textTransform: 'uppercase' },
  mono: { fontFamily: font.mono, fontSize: 13, lineHeight: 18 },
  number: { fontFamily: font.semibold, fontSize: 24, lineHeight: 30, letterSpacing: -0.8 },
} satisfies Record<string, TextStyle>;

// --- The chosen theme ---------------------------------------------------------------------------------------------

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
  } catch {} // not kept: back to dark at the next start
  listeners.forEach((l) => l());
}

export const useScheme = () => useSyncExternalStore(subscribe, () => scheme);
export const useColors = () => palettes[useScheme()];

/** Styles that use the palette, made once per theme: `const useStyles = makeStyles((c) => ({ ... }))`. */
export function makeStyles<T extends { [K in keyof T]: ViewStyle | TextStyle | ImageStyle }>(build: (c: Palette) => T) {
  const made: Partial<Record<Scheme, T>> = {};
  return () => {
    const s = useScheme();
    return (made[s] ??= StyleSheet.create(build(palettes[s])));
  };
}

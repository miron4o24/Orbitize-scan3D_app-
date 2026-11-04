import type { StressCase, StressSummary } from './api';

// The stress test's numbers for the user's choices. The server solves each kind of test once per model, at a 100 mm
// longest side, for 1 kg (own weight: at 1 g/cm³), with stresses that do not depend on the plastic's stiffness
// (backend app/stress.py). Here they are scaled, while the user picks: by the weight (proportional), by the real size
// (a given weight: 1/size²; own weight: size), and by the material and infill picked.

export type Material = 'pla' | 'petg' | 'abs' | 'resin';
export type Infill = 20 | 50 | 100;
export type Verdict = 'safe' | 'edge' | 'break';

// Printed parts, typical datasheet values: stiffness (GPa), tensile strength (MPa), density (g/cm³). layered: printed
// in layers (FDM), which hold together less well than the plastic itself, and with infill; resin prints are solid.
export const MATERIALS: Record<Material, { label: string; gpa: number; mpa: number; density: number; layered: boolean }> = {
  pla: { label: 'PLA', gpa: 3.5, mpa: 50, density: 1.24, layered: true },
  petg: { label: 'PETG', gpa: 2.1, mpa: 48, density: 1.27, layered: true },
  abs: { label: 'ABS', gpa: 2.2, mpa: 40, density: 1.04, layered: true },
  resin: { label: 'Смола', gpa: 2.4, mpa: 45, density: 1.15, layered: false },
};
export const INFILLS: Infill[] = [20, 50, 100];
const LAYERS = 0.7; // strength between FDM layers, of the plastic's own
const SAFE = 2; // a safety factor this big: holds with room to spare

/** Strength (MPa), stiffness (GPa) and density (g/cm³) of the part as printed. Infill: the walls carry most of the
 * load, so strength and stiffness fall much less than the infill share does. */
export function effective(material: Material, infill: Infill) {
  const m = MATERIALS[material];
  const p = m.layered ? infill / 100 : 1;
  return {
    mpa: m.mpa * (m.layered ? LAYERS : 1) * (0.45 + 0.55 * p),
    gpa: m.gpa * (0.4 + 0.6 * p),
    density: m.density * (0.3 + 0.7 * p),
  };
}

export type Outcome = {
  scale: number; // the server's stresses times this are the real ones (MPa)
  peak: number; // the highest stress, MPa
  strength: number; // MPa
  factor: number; // strength / peak
  verdict: Verdict;
  limit: number; // what breaks it: kg, or for a drop the height in m (0: not even a drop from nothing)
  massG: number;
};

/** The test at `load` (kg; for a drop the height in m), for the material, infill and real longest side (mm). */
export function outcome(s: StressSummary, load: number, material: Material, infill: Infill, longestMm: number): Outcome {
  const e = effective(material, infill);
  const k = longestMm / s.ref_mm;
  let scale: number;
  let limit: number;
  if (s.case === 'drop') {
    // Landing on its base, the object stops in the distance it squashes: its own weight's stresses times
    // 1 + √(1 + 2h / sag), the sag being how far its own weight presses it down.
    const own = e.density * k; // own-weight stresses grow with density and size
    const sag = Math.max(((s.sag_m ?? 0) * e.density * k ** 2) / e.gpa, 1e-15);
    scale = own * (1 + Math.sqrt(1 + (2 * load) / sag));
    const breaks = e.mpa / (s.max_mpa * own); // the factor that breaks it
    limit = breaks <= 2 ? 0 : (sag * ((breaks - 1) ** 2 - 1)) / 2;
  } else {
    scale = load / k ** 2;
    limit = (e.mpa * k ** 2) / s.max_mpa;
  }
  const peak = s.max_mpa * scale;
  const factor = e.mpa / peak;
  return {
    scale,
    peak,
    strength: e.mpa,
    factor,
    verdict: factor >= SAFE ? 'safe' : factor >= 1 ? 'edge' : 'break',
    limit,
    massG: s.volume_cm3 * k ** 3 * e.density,
  };
}

export const CASES: { value: StressCase; label: string; about: string; icon: string }[] = [
  { value: 'top', label: 'Отгоре', about: 'Нещо тежко натиска върха.', icon: 'arrow-down-outline' },
  { value: 'hang', label: 'Висене', about: 'Основата е закрепена, тежестта виси от другия край.', icon: 'arrow-up-outline' },
  { value: 'side', label: 'Отстрани', about: 'Основата е закрепена, върхът се бута настрани.', icon: 'arrow-forward-outline' },
  { value: 'drop', label: 'Падане', about: 'Пада върху основата си на твърд под.', icon: 'download-outline' },
];

export const WEIGHTS = [0.2, 0.5, 1, 2, 5, 10, 20, 50]; // kg
export const HEIGHTS = [0.3, 0.5, 1, 1.5, 2, 3]; // m

const num = (v: number) => (v >= 100 ? Math.round(v).toString() : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1).replace(/\.0$/, '') : v.toFixed(2).replace(/0$/, ''));
export const kgText = (kg: number) => (kg >= 1000 ? `${num(kg / 1000)} т` : kg >= 1 ? `${num(kg)} кг` : `${Math.round(kg * 1000)} г`);
export const mText = (m: number) => (m >= 1 ? `${num(m)} м` : `${Math.round(m * 100)} см`);
export const loadText = (kind: StressCase, load: number) => (kind === 'drop' ? mText(load) : kgText(load));

/** "Holds up to": the limit in words, kept to what is believable. */
export function limitText(kind: StressCase, limit: number) {
  if (kind === 'drop') return limit <= 0 ? 'Не издържа падане' : limit > 20 ? 'Над 20 м' : mText(limit);
  return limit > 5000 ? 'Над 5 т' : kgText(limit);
}

// --- Colours --------------------------------------------------------------------------------------------------

// From no stress to the strength: blue, cyan, green, yellow, orange, red (sRGB, 0..1); at or past it, red.
const STOPS: [number, [number, number, number]][] = [
  [0, [0.15, 0.35, 0.92]],
  [0.15, [0.02, 0.71, 0.83]],
  [0.35, [0.13, 0.77, 0.37]],
  [0.6, [0.92, 0.7, 0.03]],
  [0.8, [0.98, 0.45, 0.09]],
  [1, [0.94, 0.27, 0.27]],
];
export const UNTESTED: [number, number, number] = [0.42, 0.42, 0.45];

/** The colour (sRGB) for a stress as a share of the strength; negative: not tested. */
export function stressColor(share: number): [number, number, number] {
  if (share < 0) return UNTESTED;
  if (share >= 1) return STOPS[STOPS.length - 1][1];
  let i = 1;
  while (STOPS[i][0] < share) i++;
  const [a, ca] = STOPS[i - 1];
  const [b, cb] = STOPS[i];
  const t = (share - a) / (b - a);
  return [0, 1, 2].map((j) => ca[j] + (cb[j] - ca[j]) * t) as [number, number, number];
}

/** The legend's gradient, as CSS-like stops for expo-linear-gradient. */
export const LEGEND = STOPS.map(([at, [r, g, b]]) => ({ at, color: `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})` }));

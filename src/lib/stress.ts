import type { StressCase, StressSummary } from './api';

// Числата от теста за здравина за изборите на потребителя. Сървърът решава всеки вид тест веднъж за модел, при
// най-дълга страна 100 мм, за 1 кг (собствено тегло: при 1 г/см³), с напрежения, които не зависят от твърдостта на
// пластмасата (сървър app/stress.py). Тук те се мащабират, докато потребителят избира: по теглото (пропорционално),
// по реалния размер (при зададено тегло: 1/размер²; при собствено тегло: размер) и по избраните материал и пълнеж.

export type Material = 'pla' | 'petg' | 'abs' | 'resin';
export type Infill = 20 | 50 | 100;
export type Verdict = 'safe' | 'edge' | 'break';

// Отпечатани части, типични стойности от технически листове: твърдост (GPa), якост на опън (MPa), плътност (г/см³).
// layered: отпечатани на слоеве (FDM), които се държат заедно по-слабо от самата пластмаса и имат пълнеж;
// отпечатаното със смола е плътно.
export const MATERIALS: Record<Material, { label: string; gpa: number; mpa: number; density: number; layered: boolean }> = {
  pla: { label: 'PLA', gpa: 3.5, mpa: 50, density: 1.24, layered: true },
  petg: { label: 'PETG', gpa: 2.1, mpa: 48, density: 1.27, layered: true },
  abs: { label: 'ABS', gpa: 2.2, mpa: 40, density: 1.04, layered: true },
  resin: { label: 'Смола', gpa: 2.4, mpa: 45, density: 1.15, layered: false },
};
export const INFILLS: Infill[] = [20, 50, 100];
const LAYERS = 0.7; // якост между слоевете при FDM като дял от тази на пластмасата
const SAFE = 2; // коефициент на сигурност с такава големина: държи с резерв

/**
 * Якост (MPa), твърдост (GPa) и плътност (г/см³) на отпечатаната част. Пълнеж: стените носят по-голямата част от
 * товара, затова якостта и твърдостта падат много по-малко от дела на пълнежа.
 */
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
  scale: number; // напреженията от сървъра, умножени по това, са реалните (MPa)
  peak: number; // най-високото напрежение, MPa
  strength: number; // MPa
  factor: number; // якост / пик
  verdict: Verdict;
  limit: number; // какво го чупи: кг, а при падане височината в м (0: дори падане от нулева височина)
  massG: number;
};

/* Тестът при `load` (кг; при падане височината в м) за материала, пълнежа и реалната най-дълга страна (мм). */
export function outcome(s: StressSummary, load: number, material: Material, infill: Infill, longestMm: number): Outcome {
  const e = effective(material, infill);
  const k = longestMm / s.ref_mm;
  let scale: number;
  let limit: number;
  if (s.case === 'drop') {
    // При приземяване на основата си предметът спира в разстоянието, на което се сплесква: напреженията от
    // собственото му тегло, умножени по 1 + √(1 + 2h / sag), където sag показва колко го притиска собственото му
    // тегло.
    const own = e.density * k; // напреженията от собствено тегло растат с плътността и размера
    const sag = Math.max(((s.sag_m ?? 0) * e.density * k ** 2) / e.gpa, 1e-15);
    scale = own * (1 + Math.sqrt(1 + (2 * load) / sag));
    const breaks = e.mpa / (s.max_mpa * own); // множителят, който го чупи
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

/* "Издържа до": границата с думи, в рамките на правдоподобното. */
export function limitText(kind: StressCase, limit: number) {
  if (kind === 'drop') return limit <= 0 ? 'Не издържа падане' : limit > 20 ? 'Над 20 м' : mText(limit);
  return limit > 5000 ? 'Над 5 т' : kgText(limit);
}

// --- Цветове -------------------------------------------------------------------------------------------------------

// От липса на напрежение до якостта: синьо, циан, зелено, жълто, оранжево, червено (sRGB, 0..1); при нея и над нея:
// червено.
const STOPS: [number, [number, number, number]][] = [
  [0, [0.15, 0.35, 0.92]],
  [0.15, [0.02, 0.71, 0.83]],
  [0.35, [0.13, 0.77, 0.37]],
  [0.6, [0.92, 0.7, 0.03]],
  [0.8, [0.98, 0.45, 0.09]],
  [1, [0.94, 0.27, 0.27]],
];
export const UNTESTED: [number, number, number] = [0.42, 0.42, 0.45];

/* Цветът (sRGB) за напрежение като дял от якостта; отрицателно: не е тествано. */
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

/* Градиентът на легендата като точки за expo-linear-gradient, подобни на CSS. */
export const LEGEND = STOPS.map(([at, [r, g, b]]) => ({ at, color: `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})` }));

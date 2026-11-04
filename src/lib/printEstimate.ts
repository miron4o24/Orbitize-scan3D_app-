import { File, Paths } from 'expo-file-system';

import { MATERIALS, type Infill, type Material } from './stress';

// Колко време отнема печатът, колко пластмаса и на каква цена: груба оценка от обема и повърхността на модела (сървър
// mesh.geometry, при най-дълга страна 100 мм; обемът расте с куба на размера, повърхността с квадрата), както би го
// изчислила програма за печат за обикновен принтер. FDM: стени с дебелина WALL_MM около повърхността (2 линии от 0,4
// мм; горните и долните слоеве са горе-долу същото), дял на пълнежа от останалото, печат средно с FLOW_MM3_S
// (външните стени по-бавно, пълнежът по-бързо) плюс LAYER_S на слой за движения и смяна на слоя. Смола: плътно,
// времето зависи само от слоевете. Без подпори: те зависят от това как е поставен моделът върху плочата.

const WALL_MM = 0.8;
const LAYER_MM = 0.2;
const FLOW_MM3_S = 8;
const LAYER_S = 4;
const RESIN_LAYER_MM = 0.05;
const RESIN_LAYER_S = 9;
const POWER_KW: Record<Material, number> = { pla: 0.12, petg: 0.14, abs: 0.18, resin: 0.06 }; // средна консумация на принтера
export const KWH_EUR = 0.13; // приблизителна цена на тока в България
export const DEFAULT_EUR_PER_KG: Record<Material, number> = { pla: 20, petg: 22, abs: 20, resin: 35 };

export type PrintEstimate = { grams: number; seconds: number; plasticEur: number; powerEur: number };

/* volume100, area100, height100: при размер 100 мм; longestMm: реалната най-дълга страна. */
export function estimate(
  volume100: number,
  area100: number,
  height100: number,
  longestMm: number,
  material: Material,
  infill: Infill,
  eurPerKg: number,
): PrintEstimate {
  const k = longestMm / 100;
  const volume = volume100 * k ** 3;
  const area = area100 * k ** 2;
  const height = height100 * k;
  let plastic: number; // mm³
  let seconds: number;
  if (material === 'resin') {
    plastic = volume;
    seconds = (height / RESIN_LAYER_MM) * RESIN_LAYER_S;
  } else {
    const shell = Math.min(volume, area * WALL_MM);
    plastic = shell + (volume - shell) * (infill / 100);
    seconds = plastic / FLOW_MM3_S + (height / LAYER_MM) * LAYER_S;
  }
  const grams = (plastic / 1000) * MATERIALS[material].density;
  return {
    grams,
    seconds,
    plasticEur: (grams / 1000) * eurPerKg,
    powerEur: (seconds / 3600) * POWER_KW[material] * KWH_EUR,
  };
}

export function printTime(seconds: number): string {
  const minutes = Math.max(1, Math.round(seconds / 60));
  if (minutes < 60) return `${minutes} мин`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `${h} ч ${m} мин` : `${h} ч`;
}

export const euro = (v: number) => (v < 0.01 ? '< 0,01 €' : `${v.toFixed(2).replace('.', ',')} €`);

// Собствените цени на пластмасата на потребителя (€ за кг), пазени на телефона.
const pricesFile = () => new File(Paths.document, 'plastic-prices.json');

export function getPrices(): Record<Material, number> {
  try {
    const f = pricesFile();
    if (f.exists) return { ...DEFAULT_EUR_PER_KG, ...(JSON.parse(f.textSync()) as Partial<Record<Material, number>>) };
  } catch {}
  return { ...DEFAULT_EUR_PER_KG };
}

export function setPrice(material: Material, eurPerKg: number) {
  pricesFile().writeSync(JSON.stringify({ ...getPrices(), [material]: eurPerKg }));
}

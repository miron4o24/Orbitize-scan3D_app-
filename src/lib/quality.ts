// Проверка на качеството на всеки кадър върху малко RGBA копие на снимката (около 512 px широко).

export type FrameMetrics = {
  sharpness: number; // дисперсия на лапласиана на яркостта; ниска = размазано
  brightness: number; // средна яркост 0..255
  clipped: number; // дял на пикселите с яркост >= 250
};

// опростяване: праговете са първоначални стойности за кадри от 512 px; настройват се с показанията на екрана за
// заснемане.
export const THRESHOLDS = { minSharpness: 40, minBrightness: 45, maxClipped: 0.15 };

export function measure(rgba: Uint8Array, width: number, height: number): FrameMetrics {
  const n = width * height;
  const lum = new Float32Array(n);
  let sum = 0;
  let clipped = 0;
  for (let i = 0; i < n; i++) {
    const y = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    lum[i] = y;
    sum += y;
    if (y >= 250) clipped++;
  }

  // Текуща дисперсия по Уелфорд на лапласиана с 4 съседа.
  let count = 0;
  let mean = 0;
  let m2 = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const l = lum[i - 1] + lum[i + 1] + lum[i - width] + lum[i + width] - 4 * lum[i];
      count++;
      const d = l - mean;
      mean += d / count;
      m2 += d * (l - mean);
    }
  }

  return { sharpness: count ? m2 / count : 0, brightness: sum / n, clipped: clipped / n };
}

export function judge(m: FrameMetrics) {
  const warnings: string[] = [];
  if (m.brightness < THRESHOLDS.minBrightness) warnings.push('Твърде тъмно: добавете светлина');
  if (m.clipped > THRESHOLDS.maxClipped) warnings.push('Твърде светло: избягвайте пряка светлина и отблясъци');
  const blurry = m.sharpness < THRESHOLDS.minSharpness;
  if (blurry) warnings.push('Размазано: дръжте телефона стабилно');
  // Размазването разваля съпоставянето на особеностите, затова кадърът отпада; за проблеми с осветеността само се
  // предупреждава.
  return { reject: blurry, warnings };
}

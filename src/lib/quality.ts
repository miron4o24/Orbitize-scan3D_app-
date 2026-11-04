// Per-frame quality check on a small RGBA copy of the photo (~512 px wide).

export type FrameMetrics = {
  sharpness: number; // variance of the Laplacian of luminance; low = blurry
  brightness: number; // mean luminance 0..255
  clipped: number; // fraction of pixels at >= 250 luminance
};

// shortcut: thresholds are first guesses for 512 px frames; tune with the readout on the capture screen.
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

  // Welford running variance of the 4-neighbour Laplacian.
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
  // Blur ruins feature matching, so drop the frame; exposure issues are only warned about.
  return { reject: blurry, warnings };
}

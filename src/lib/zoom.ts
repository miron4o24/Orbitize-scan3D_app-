// Увеличение на снимка около средата ѝ: точка p от снимката (0..w, 0..h) е на рамката в c + t + (p - c) * s.

export type Zoom = { s: number; x: number; y: number }; // увеличение и отместване (пиксели)

export const ZOOM_MAX = 6;

/* Увеличение от 1 до ZOOM_MAX, а отместването - така, че снимката да покрива рамката. */
export function clampZoom(v: Zoom, w: number, h: number): Zoom {
  const s = Math.min(ZOOM_MAX, Math.max(1, v.s));
  const mx = ((s - 1) * w) / 2;
  const my = ((s - 1) * h) / 2;
  return { s, x: Math.min(mx, Math.max(-mx, v.x)), y: Math.min(my, Math.max(-my, v.y)) };
}

/* Точката от снимката под точката (qx, qy) на рамката. */
export function unproject(v: Zoom, w: number, h: number, qx: number, qy: number) {
  return { x: w / 2 + (qx - w / 2 - v.x) / v.s, y: h / 2 + (qy - h / 2 - v.y) / v.s };
}

/* Новото увеличение s с два пръста: точката от снимката, която беше под пръстите (m0 при start), е под тях (m). */
export function pinch(start: Zoom, w: number, h: number, m0: { x: number; y: number }, m: { x: number; y: number }, s: number): Zoom {
  const p = unproject(start, w, h, m0.x, m0.y);
  const ns = Math.min(ZOOM_MAX, Math.max(1, s));
  return clampZoom({ s: ns, x: m.x - w / 2 - (p.x - w / 2) * ns, y: m.y - h / 2 - (p.y - h / 2) * ns }, w, h);
}

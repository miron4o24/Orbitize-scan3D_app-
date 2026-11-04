// Накъде сочи задната камера, по завъртането от DeviceMotion (радиани, ред Z-X'-Y'' по W3C). Световна система: x на
// изток, y на север, z нагоре. Изгледът е единичният вектор, по който сочи задната камера.

export type Vec3 = [number, number, number];

export function viewDir({ alpha, beta, gamma }: { alpha: number; beta: number; gamma: number }): Vec3 {
  const [ca, sa, cb, sb, cg, sg] = [
    Math.cos(alpha),
    Math.sin(alpha),
    Math.cos(beta),
    Math.sin(beta),
    Math.cos(gamma),
    Math.sin(gamma),
  ];
  // -1 × третата колона на R = Rz(alpha)·Rx(beta)·Ry(gamma): оста -Z на устройството в световни координати.
  return [-(ca * sg + cg * sa * sb), -(sa * sg - ca * cg * sb), -cb * cg];
}

const deg = (r: number) => (r * 180) / Math.PI;

export function elevation(v: Vec3) {
  return deg(Math.asin(Math.min(1, Math.max(-1, v[2]))));
}

// Насочена разумно: не към тавана и не право надолу.
export function isUsableAim(v: Vec3) {
  const el = elevation(v);
  return el <= 10 && el >= -80;
}

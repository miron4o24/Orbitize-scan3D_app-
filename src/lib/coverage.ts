// Where the back camera points, from DeviceMotion rotation (radians, W3C Z-X'-Y'' order).
// World frame: x east, y north, z up. A view is the unit vector the back camera points along.

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
  // -1 × third column of R = Rz(alpha)·Rx(beta)·Ry(gamma): the device's -Z axis in world coordinates.
  return [-(ca * sg + cg * sa * sb), -(sa * sg - ca * cg * sb), -cb * cg];
}

const deg = (r: number) => (r * 180) / Math.PI;

export function elevation(v: Vec3) {
  return deg(Math.asin(Math.min(1, Math.max(-1, v[2]))));
}

// Aimed sensibly: not at the ceiling or straight down.
export function isUsableAim(v: Vec3) {
  const el = elevation(v);
  return el <= 10 && el >= -80;
}

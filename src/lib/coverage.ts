// Orbit coverage from DeviceMotion rotation (radians, W3C Z-X'-Y'' order).
// World frame: x east, y north, z up. A view is the unit vector the back camera points along.

export type Vec3 = [number, number, number];

export const SECTORS = 24; // 15° each
export const BANDS = 3; // 0 = low (near horizontal), 1 = mid, 2 = high (looking down)
const BAND_EDGES = [-25, -50]; // view elevation in degrees

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

export function angleBetween(u: Vec3, v: Vec3) {
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  return deg(Math.acos(Math.min(1, Math.max(-1, dot))));
}

export function elevation(v: Vec3) {
  return deg(Math.asin(Math.min(1, Math.max(-1, v[2]))));
}

export function cellOf(v: Vec3) {
  const el = elevation(v);
  const az = (deg(Math.atan2(v[0], v[1])) + 360) % 360;
  const band = el > BAND_EDGES[0] ? 0 : el > BAND_EDGES[1] ? 1 : 2;
  return { band, sector: Math.floor(az / (360 / SECTORS)) % SECTORS };
}

export const cellKey = (c: { band: number; sector: number }) => `${c.band}:${c.sector}`;

// Auto-capture only when aimed sensibly (not at the ceiling or straight down) and far enough from every earlier shot.
export function isNewAngle(v: Vec3, shots: Vec3[], minDeg = 12) {
  const el = elevation(v);
  if (el > 10 || el < -80) return false;
  return shots.every((s) => angleBetween(v, s) >= minDeg);
}

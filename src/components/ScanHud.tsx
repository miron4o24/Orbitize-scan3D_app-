import { memo, useMemo } from 'react';
import Svg, { Circle, G, Line, Path } from 'react-native-svg';

import { LOOPS, SECTORS, type OrbitState } from '@/lib/orbit';

// Средата на екрана за заснемане: рамка около предмета, двата кръга около него (долният за обиколката ниско, горният
// за тази отвисоко), разделени на сектори по 15°, които светват, когато са заснети рязко и от правилната височина
// (lib/orbit.ts), и къде е телефонът сега. Точките в средата са същото покритие, нарисувано върху предмета: страните
// от долния кръг и горната част от горния. Не са възстановената форма (тя се прави на сървъра), а кои страни вече са
// заснети.

export const HUD = {
  cyan: '#38bdf8',
  amber: '#f59e0b',
  dim: 'rgba(255,255,255,0.26)',
  line: 'rgba(255,255,255,0.55)',
  panel: 'rgba(8,12,17,0.9)',
};

const TICKS = SECTORS * 2; // по две черти на сектор
const POINTS = 240;

type Geo = { cx: number; cy: number; rx: number; ry: number }[]; // по един елипс за кръг

function geometry(w: number, h: number): Geo {
  const rx = w * 0.4;
  const ry = rx * 0.27;
  const cy = h * 0.66;
  return [
    { cx: w / 2, cy, rx, ry }, // ниско
    { cx: w / 2, cy: cy - h * 0.2, rx: rx * 0.7, ry: ry * 0.7 }, // отвисоко
  ];
}

// Ъгъл 0: там, където е започнал записът, отпред (долу на елипса).
const onEllipse = (e: Geo[number], deg: number, k = 1): [number, number] => {
  const t = (deg * Math.PI) / 180;
  return [e.cx + e.rx * k * Math.sin(t), e.cy + e.ry * k * Math.cos(t)];
};

function random(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Dot = { x: number; y: number; r: number; loop: number; sector: number };

function dots(w: number, h: number): Dot[] {
  const [low] = geometry(w, h);
  const R = low.rx * 0.42;
  const tall = h * 0.3;
  const rnd = random(7);
  const out: Dot[] = [];
  for (let i = 0; i < POINTS; i++) {
    const top = i % 10 < 3; // 30% по горната страна
    const deg = rnd() * 360;
    const t = (deg * Math.PI) / 180;
    const k = top ? Math.sqrt(rnd()) : 0.86 + rnd() * 0.14;
    const y = top ? 1 : rnd();
    const front = Math.cos(t);
    out.push({
      x: low.cx + R * k * Math.sin(t),
      y: low.cy + R * (low.ry / low.rx) * k * front - y * tall,
      r: 1.2 + 0.7 * (front + 1) * 0.5,
      loop: top ? 1 : 0,
      sector: Math.floor(deg / (360 / SECTORS)) % SECTORS,
    });
  }
  return out;
}

const Rings = memo(function Rings({ w, h, orbit }: { w: number; h: number; orbit: OrbitState | null }) {
  const geo = useMemo(() => geometry(w, h), [w, h]);
  const cloud = useMemo(() => dots(w, h), [w, h]);
  const active = orbit?.loop ?? 0;
  return (
    <G>
      {cloud.map((d, i) => {
        const on = !!orbit?.covered[d.loop]?.[d.sector];
        return <Circle key={i} cx={d.x} cy={d.y} r={d.r} fill={on ? HUD.cyan : '#ffffff'} opacity={on ? 0.75 : 0.16} />;
      })}
      {geo.map((e, loop) =>
        Array.from({ length: TICKS }, (_, i) => {
          const deg = (i * 360) / TICKS;
          const sector = Math.floor(i / 2);
          const on = !!orbit?.covered[loop]?.[sector];
          const front = Math.cos((deg * Math.PI) / 180); // отпред чертите са по-дебели: перспектива
          const [x1, y1] = onEllipse(e, deg, 0.93);
          const [x2, y2] = onEllipse(e, deg, 1.07);
          const now = loop === Math.min(active, LOOPS - 1);
          return (
            <Line
              key={`${loop}-${i}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={on ? HUD.cyan : HUD.dim}
              strokeWidth={(loop === 0 ? 1.6 : 1.2) + (front + 1) * 0.7}
              strokeLinecap="round"
              opacity={(now || on ? 1 : 0.45) * (0.6 + 0.4 * (front + 1) * 0.5)}
            />
          );
        }),
      )}
    </G>
  );
});

const Frame = memo(function Frame({ w, h }: { w: number; h: number }) {
  const m = w * 0.1; // рамката: предметът да е вътре и да я запълва
  const top = h * 0.04;
  const bottom = h * 0.96;
  const L = w * 0.07;
  const c = (x: number, y: number, dx: number, dy: number) => `M${x + dx * L},${y} L${x},${y} L${x},${y + dy * L}`;
  const cx = w / 2;
  const cy = h * 0.12;
  return (
    <G stroke={HUD.line} strokeWidth={1.6} fill="none">
      <Path d={c(m, top, 1, 1)} />
      <Path d={c(w - m, top, -1, 1)} />
      <Path d={c(m, bottom, 1, -1)} />
      <Path d={c(w - m, bottom, -1, -1)} />
      <Path d={`M${cx - 9},${cy} L${cx - 3},${cy} M${cx + 3},${cy} L${cx + 9},${cy} M${cx},${cy - 9} L${cx},${cy - 3} M${cx},${cy + 3} L${cx},${cy + 9}`} />
    </G>
  );
});

type Props = { w: number; h: number; orbit: OrbitState | null; angle: number; recording: boolean };

export function ScanStage({ w, h, orbit, angle, recording }: Props) {
  const geo = geometry(w, h);
  const ring = geo[Math.min(orbit?.loop ?? 0, LOOPS - 1)];
  const [mx, my] = onEllipse(ring, angle);
  return (
    <Svg width={w} height={h}>
      <Frame w={w} h={h} />
      <Rings w={w} h={h} orbit={orbit} />
      {recording && (
        <G>
          <Line x1={mx} y1={my - 12} x2={mx} y2={0} stroke={HUD.amber} strokeWidth={1.4} strokeDasharray="3 4" opacity={0.85} />
          <Circle cx={mx} cy={my} r={11} stroke={HUD.amber} strokeWidth={1.6} fill="rgba(245,158,11,0.18)" />
          <Circle cx={mx} cy={my} r={5.5} fill={HUD.amber} />
        </G>
      )}
    </Svg>
  );
}

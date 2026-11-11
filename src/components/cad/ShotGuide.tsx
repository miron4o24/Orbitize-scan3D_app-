import Svg, { G, Line, Path, Polygon, Rect, Circle } from 'react-native-svg';

import type { CadPhoto } from '@/lib/api';

// Картинката към всяка снимка: вляво предметът, както стои, с оранжевата страна, която трябва да се снима; вдясно
// същият предмет, обърнат така, че тази страна да гледа нагоре, и телефонът над него. Предметът е кутия в
// изометрия, гледана отпред-отдясно-отгоре (както я вижда човекът пред масата).

type Face = CadPhoto;
type P3 = [number, number, number];

const ORANGE = '#f97316';
const EDGE = 'rgba(255,255,255,0.85)';
const HIDDEN = 'rgba(255,255,255,0.45)';
const SHADES: Record<'top' | 'front' | 'right', string> = { top: '#9ca3af', front: '#6b7280', right: '#4b5563' };
const BOX: P3 = [1.7, 1.0, 0.6]; // дължина, дълбочина, височина

// Изометрия: x надясно-надолу, y навътре (назад), z нагоре; гледаме от предния десен горен ъгъл.
const project = ([x, y, z]: P3): [number, number] => [(x + y) / Math.SQRT2, (x - y - 2 * z) / Math.sqrt(6)];

function faces([a, b, c]: P3): Record<Face, P3[]> {
  return {
    top: [
      [0, 0, c],
      [a, 0, c],
      [a, b, c],
      [0, b, c],
    ],
    bottom: [
      [0, 0, 0],
      [a, 0, 0],
      [a, b, 0],
      [0, b, 0],
    ],
    front: [
      [0, 0, 0],
      [a, 0, 0],
      [a, 0, c],
      [0, 0, c],
    ],
    back: [
      [0, b, 0],
      [a, b, 0],
      [a, b, c],
      [0, b, c],
    ],
    left: [
      [0, 0, 0],
      [0, b, 0],
      [0, b, c],
      [0, 0, c],
    ],
    right: [
      [a, 0, 0],
      [a, b, 0],
      [a, b, c],
      [a, 0, c],
    ],
  };
}

const VISIBLE: Face[] = ['front', 'right', 'top'];

// Размерите на кутията, когато страната `up` гледа нагоре (претъркаляна напред/назад или изправена на край).
function turned(up: Face): P3 {
  const [a, b, c] = BOX;
  if (up === 'front' || up === 'back') return [a, c, b];
  if (up === 'left' || up === 'right') return [c, b, a];
  return [a, b, c];
}

function Box({ dims, mark, at, scale }: { dims: P3; mark: Face; at: [number, number]; scale: number }) {
  const f = faces(dims);
  const pts = (face: Face) =>
    f[face]
      .map(project)
      .map(([x, y]) => `${at[0] + x * scale},${at[1] + y * scale}`)
      .join(' ');
  const hiddenMark = !VISIBLE.includes(mark);
  // Скритите ръбове (до задния долен ляв ъгъл) с пунктир: така и скритата страна се вижда къде е.
  const [a, b, c] = dims;
  const corner = project([0, b, 0]);
  const ends = [project([a, b, 0]), project([0, 0, 0]), project([0, b, c])];
  return (
    <G>
      {VISIBLE.map((face) => (
        <Polygon key={face} points={pts(face)} fill={face === mark ? ORANGE : SHADES[face as 'top']} stroke={EDGE} strokeWidth={1.2} strokeLinejoin="round" />
      ))}
      {ends.map(([x, y], k) => (
        <Line
          key={k}
          x1={at[0] + corner[0] * scale}
          y1={at[1] + corner[1] * scale}
          x2={at[0] + x * scale}
          y2={at[1] + y * scale}
          stroke={HIDDEN}
          strokeWidth={1}
          strokeDasharray="3,3"
        />
      ))}
      {hiddenMark && <Polygon points={pts(mark)} fill="rgba(249,115,22,0.35)" stroke={ORANGE} strokeWidth={2} strokeDasharray="4,3" />}
    </G>
  );
}

// Мащаб и място, така че кутията да се събере в правоъгълника (x, y, w, h).
function fit(dims: P3, x: number, y: number, w: number, h: number): { at: [number, number]; scale: number } {
  const all = Object.values(faces(dims)).flat().map(project);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const scale = Math.min(w / (x1 - x0), h / (y1 - y0));
  return { scale, at: [x + (w - (x1 - x0) * scale) / 2 - x0 * scale, y + (h - (y1 - y0) * scale) / 2 - y0 * scale] };
}

function Phone({ cx, top }: { cx: number; top: number }) {
  return (
    <G>
      <Rect x={cx - 15} y={top} width={30} height={10} rx={3} fill="#111827" stroke={EDGE} strokeWidth={1.2} />
      <Circle cx={cx - 8} cy={top + 5} r={2} fill={EDGE} />
      <Path d={`M ${cx} ${top + 13} L ${cx} ${top + 21} M ${cx - 4} ${top + 17} L ${cx} ${top + 22} L ${cx + 4} ${top + 17}`} stroke={ORANGE} strokeWidth={2} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </G>
  );
}

export function ShotGuide({ photo, width = 220, height = 96 }: { photo: CadPhoto; width?: number; height?: number }) {
  if (photo === 'top') {
    const box = fit(BOX, width * 0.25, 30, width * 0.5, height - 34);
    return (
      <Svg width={width} height={height}>
        <Phone cx={width / 2} top={2} />
        <Box dims={BOX} mark="top" {...box} />
      </Svg>
    );
  }
  const before = fit(BOX, 4, 18, width * 0.36, height - 22);
  const after = turned(photo);
  const right = fit(after, width * 0.6, 30, width * 0.38, height - 34);
  const mid = width * 0.48;
  return (
    <Svg width={width} height={height}>
      <Box dims={BOX} mark={photo} {...before} />
      <Path
        d={`M ${mid - 14} ${height / 2 + 6} L ${mid + 12} ${height / 2 + 6} M ${mid + 5} ${height / 2} L ${mid + 13} ${height / 2 + 6} L ${mid + 5} ${height / 2 + 12}`}
        stroke={EDGE}
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Phone cx={width * 0.79} top={2} />
      <Box dims={after} mark="top" {...right} />
    </Svg>
  );
}

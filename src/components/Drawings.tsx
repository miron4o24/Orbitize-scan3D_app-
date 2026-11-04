import { Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useDerivedValue, useSharedValue, type SharedValue } from 'react-native-reanimated';
import Svg, { Line, Path, Text as SvgText } from 'react-native-svg';

import { font, makeStyles, useColors } from '@/theme';
import { useReveal } from './motion';

// Рисунките за функциите от сайта (index.html .bento): тънки линии, акцент върху важната част и малко движение, което
// се изпълнява, щом плочката се превърти в изгледа (motion.Reveal).

/* Докъде е стигнала появата на плочката (0..1), започвайки `from` след началото и разтегната до края. */
function useStage(from = 0) {
  const fallback = useSharedValue(1);
  const p = useReveal() ?? fallback;
  return useDerivedValue(() => Math.min(1, Math.max(0, (p.value - from) / (1 - from))));
}

const W = 260; // рисунките са с тази ширина и центрирани в плочката си

/* Линийка, чийто маркер се плъзга до 100 мм. */
export function RulerDrawing() {
  const c = useColors();
  const s = useStyles();
  const p = useStage(0.15);
  const knob = useAnimatedStyle(() => ({ transform: [{ translateX: -150 * (1 - p.value) }] }));
  const ticks = [];
  for (let i = 0; i <= 20; i++) {
    const x = 14 + i * 11.6;
    ticks.push(<Line key={i} x1={x} y1={70} x2={x} y2={i % 5 === 0 ? 54 : 62} stroke={c.lineStrong} strokeWidth={1} />);
  }
  return (
    <View style={s.vis}>
      <Svg width={W} height={110}>
        <Line x1={8} y1={70} x2={252} y2={70} stroke={c.text2} strokeWidth={1.4} />
        {ticks}
        <SvgText x={14} y={96} fill={c.muted} fontFamily={font.mono} fontSize={10}>
          0
        </SvgText>
        <SvgText x={200} y={96} fill={c.muted} fontFamily={font.mono} fontSize={10}>
          100 ММ
        </SvgText>
      </Svg>
      <Animated.View style={[s.knob, knob]}>
        <View style={[s.knobDot, { backgroundColor: c.accent }]} />
        <View style={[s.knobLine, { backgroundColor: c.accent }]} />
      </Animated.View>
    </View>
  );
}

function Chip({ p, i, name, note, first }: { p: SharedValue<number>; i: number; name: string; note: string; first?: boolean }) {
  const s = useStyles();
  const lift = useAnimatedStyle(() => {
    const t = Math.min(1, Math.max(0, (p.value - i * 0.12) / 0.64));
    return { opacity: 0.3 + 0.7 * t, transform: [{ translateY: 8 - 13 * t }] };
  });
  return (
    <Animated.View style={[s.chip, first && s.chipFirst, lift]}>
      <Text style={s.chipName}>{name}</Text>
      <Text style={s.chipNote}>{note}</Text>
    </Animated.View>
  );
}

/* Трите файла, появяващи се един след друг; този за печат е отбелязан. */
export function FilesDrawing() {
  const s = useStyles();
  const p = useStage(0.1);
  return (
    <View style={[s.vis, s.row]}>
      <Chip p={p} i={0} name="STL" note="за слайсер" first />
      <Chip p={p} i={1} name="OBJ" note="3D програми" />
      <Chip p={p} i={2} name="GLB" note="с цветове" />
    </View>
  );
}

/* Под в перспектива и моделът, който израства върху него в реалния си размер. */
export function RoomDrawing() {
  const c = useColors();
  const s = useStyles();
  const p = useStage(0.2);
  const grow = useAnimatedStyle(() => ({ opacity: p.value, transform: [{ translateY: 18 * (1 - p.value) }, { scale: 0.4 + 0.6 * p.value }] }));
  const lines = [];
  for (let i = 0; i <= 8; i++) {
    const x = 130 + (i - 4) * 16; // далечен ръб
    const bx = 130 + (i - 4) * 46; // близък ръб
    lines.push(<Line key={`d${i}`} x1={x} y1={52} x2={bx} y2={112} stroke={c.line} strokeWidth={1} />);
  }
  for (const [y, half] of [
    [52, 64],
    [64, 92],
    [80, 128],
    [112, 184],
  ] as const)
    lines.push(<Line key={`h${y}`} x1={130 - half} y1={y} x2={130 + half} y2={y} stroke={c.line} strokeWidth={1} />);
  return (
    <View style={s.vis}>
      <Svg width={W} height={116}>
        {lines}
        <SvgText x={8} y={14} fill={c.muted} fontFamily={font.mono} fontSize={10}>
          AR · 1:1
        </SvgText>
      </Svg>
      <Animated.View style={[s.roomModel, grow]}>
        <Svg width={56} height={60} viewBox="0 0 200 214">
          <Path d="M100 18 172 58 100 98 28 58Z" fill={c.accentSoft} stroke={c.accent} strokeWidth={5} />
          <Path d="M28 58 100 98V182L28 142Z" fill={c.accentSoft} stroke={c.accent} strokeWidth={5} />
          <Path d="M172 58 100 98V182L172 142Z" fill={c.accentSoft} stroke={c.accent} strokeWidth={5} />
        </Svg>
      </Animated.View>
    </View>
  );
}

/* Линк за споделяне с плъзгащ се превключвател. */
export function ShareDrawing() {
  const c = useColors();
  const s = useStyles();
  const p = useStage(0.4);
  const knob = useAnimatedStyle(() => ({ transform: [{ translateX: -16 * (1 - p.value) }] }));
  const track = useAnimatedStyle(() => ({ opacity: 0.25 + 0.75 * p.value }));
  return (
    <View style={s.vis}>
      <View style={s.pill}>
        <Text style={s.pillText}>scanly…/s/a8f3</Text>
        <View style={s.toggle}>
          <Animated.View style={[s.toggleTrack, { backgroundColor: c.accentSoft, borderColor: c.accent }, track]} />
          <Animated.View style={[s.toggleKnob, { backgroundColor: c.accent }, knob]} />
        </View>
      </View>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  vis: { height: 120, alignItems: 'center', justifyContent: 'center' },
  row: { flexDirection: 'row', gap: 10 },
  knob: { position: 'absolute', top: 18, left: '50%', marginLeft: W / 2 - 18, alignItems: 'center' }, // над отметката за 100 мм
  knobDot: { width: 8, height: 8, borderRadius: 4 },
  knobLine: { width: 1.4, height: 40 },
  chip: {
    paddingTop: 14,
    paddingBottom: 10,
    paddingHorizontal: 14,
    borderWidth: 1,
    borderColor: c.lineStrong,
    borderTopLeftRadius: 6,
    borderTopRightRadius: 16,
    borderBottomLeftRadius: 6,
    borderBottomRightRadius: 6,
    backgroundColor: c.surface,
  },
  chipFirst: { borderColor: c.accent },
  chipName: { fontFamily: font.monoMedium, fontSize: 14, color: c.text },
  chipNote: { fontFamily: font.mono, fontSize: 9.5, color: c.muted, marginTop: 3 },
  roomModel: { position: 'absolute', top: 38 },
  pill: {
    width: W - 30,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: c.lineStrong,
    backgroundColor: c.surface2,
    flexDirection: 'row',
    alignItems: 'center',
    paddingLeft: 18,
    paddingRight: 8,
  },
  pillText: { flex: 1, fontFamily: font.mono, fontSize: 12, color: c.text2 },
  toggle: { width: 40, height: 24, justifyContent: 'center' },
  toggleTrack: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, borderRadius: 12, borderWidth: 1.2 },
  toggleKnob: { position: 'absolute', right: 4, width: 16, height: 16, borderRadius: 8 },
}));

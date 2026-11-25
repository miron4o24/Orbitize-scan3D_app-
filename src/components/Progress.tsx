import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import Animated, { useAnimatedProps, useAnimatedStyle, useReducedMotion, useSharedValue, withDelay, withTiming } from 'react-native-reanimated';
import Svg, { Circle, Polygon } from 'react-native-svg';

import type { Progress } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { font, makeStyles, radius, space, text, useColors } from '@/theme';
import { CountUp, ease, PressScale } from './motion';
import { Card, H2, Label, Small, type IconName } from './ui';
import { showDialog } from '@/lib/dialog';

// Точките на профила (lib/progress.ts) в профила и на началния екран: нивото в пръстен, който се запълва към
// следващото, седмичните предизвикателства и значките.

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

// Шестоъгълникът на значка (50 точки): отключена - рамка в цвета за акцент, заключена - пунктир при 50%.
const HEX = 50;
const COLUMNS = 3;
const GAP = 8;
const HEX_POINTS = Array.from({ length: 6 }, (_, k) => {
  const a = ((-90 + 60 * k) * Math.PI) / 180;
  return `${HEX / 2 + (HEX / 2 - 1.5) * Math.cos(a)},${HEX / 2 + (HEX / 2 - 1.5) * Math.sin(a)}`;
}).join(' ');

const share = (p: Progress) => Math.min(1, Math.max(0, (p.points - p.level.from) / Math.max(1, p.level.to - p.level.from)));

/* Номерът на нивото в пръстен, запълнен толкова, колкото точките са напреднали към следващото ниво. */
function LevelRing({ progress, size }: { progress: Progress; size: number }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const stroke = size > 50 ? 5 : 4;
  const r = (size - stroke) / 2;
  const length = 2 * Math.PI * r;
  const filled = useSharedValue(reduce ? share(progress) : 0);
  useEffect(() => {
    filled.set(reduce ? share(progress) : withDelay(200, withTiming(share(progress), { duration: 1100, easing: ease })));
  }, [progress, reduce, filled]);
  const arc = useAnimatedProps(() => ({ strokeDashoffset: length * (1 - filled.value) }));
  return (
    <View style={{ width: size, height: size }}>
      <Svg width={size} height={size} style={s.ring}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={c.surface3} strokeWidth={stroke} fill="none" />
        <AnimatedCircle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={c.accent}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="butt"
          strokeDasharray={`${length} ${length}`}
          animatedProps={arc}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </Svg>
      <View style={s.ringInside}>
        <Text style={[s.ringNumber, size < 50 && s.ringNumberSmall]}>{progress.level.number}</Text>
      </View>
    </View>
  );
}

/* Тънка лента, запълнена до `value` (0..1), която нараства при появата. */
export function Bar({ value, color }: { value: number; color?: string }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const w = useSharedValue(reduce ? value : 0);
  useEffect(() => {
    w.set(reduce ? value : withTiming(value, { duration: 900, easing: ease }));
  }, [value, reduce, w]);
  const fill = useAnimatedStyle(() => ({ width: `${Math.min(1, Math.max(0, w.value)) * 100}%` }));
  return (
    <View style={s.track}>
      <Animated.View style={[s.fill, { backgroundColor: color ?? c.accent }, fill]} />
    </View>
  );
}

const nextName = (p: Progress) => (p.level.to - p.points > 0 ? `${p.level.to - p.points} точки до следващото ниво` : 'Следващото ниво е близо');

/* Нивото в профила: пръстен, име, точки и колко остава до следващото ниво. */
export function LevelCard({ progress }: { progress: Progress }) {
  const c = useColors();
  const s = useStyles();
  return (
    <Card style={s.level}>
      <View style={s.levelHead}>
        <LevelRing progress={progress} size={64} />
        <View style={s.flex}>
          <Label>Ниво {progress.level.number}</Label>
          <H2 numberOfLines={1}>{progress.level.name}</H2>
          <Small>{nextName(progress)}</Small>
        </View>
        <View style={s.points}>
          <CountUp value={progress.points} style={s.pointsValue} />
          <Label>Точки</Label>
        </View>
      </View>
      {progress.points_week > 0 && (
        <View style={s.week}>
          <Ionicons name="trending-up-outline" size={14} color={c.success} />
          <Text style={s.weekText}>+{progress.points_week} точки тази седмица</Text>
        </View>
      )}
    </Card>
  );
}

/* Нивото на началния екран: един ред, който отваря профила. */
export function LevelStrip({ progress, onPress }: { progress: Progress; onPress: () => void }) {
  const c = useColors();
  const s = useStyles();
  return (
    <PressScale onPress={onPress} accessibilityRole="button" accessibilityLabel={`Ниво ${progress.level.number}, ${progress.points} точки. Към профила`}>
      <Card style={s.strip}>
        <LevelRing progress={progress} size={44} />
        <View style={s.flex}>
          <Text style={s.stripTitle} numberOfLines={1}>
            {progress.level.name} · {progress.points} точки
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={16} color={c.muted} />
      </Card>
    </PressScale>
  );
}

const daysLeft = (ends: number) => {
  const days = Math.ceil((ends * 1000 - Date.now()) / 86_400_000);
  return days <= 1 ? 'последен ден' : `остават ${days} дни`;
};

export function Challenges({ progress }: { progress: Progress }) {
  const c = useColors();
  const s = useStyles();
  const items = progress.challenges.items;
  return (
    <Card style={s.list}>
      <Text style={s.listNote}>{daysLeft(progress.challenges.ends)}</Text>
      {items.map((x, i) => (
        <View key={x.id} style={[s.challenge, i < items.length - 1 && s.rowLine]}>
          <View style={[s.icon, { borderColor: x.done ? c.success : c.line }]}>
            <Ionicons name={(x.done ? 'checkmark' : x.icon) as IconName} size={18} color={x.done ? c.success : c.text2} />
          </View>
          <View style={s.challengeBody}>
            <Text style={[s.challengeTitle, x.done && s.doneText]} numberOfLines={2}>
              {x.title}
            </Text>
            <View style={s.challengeBar}>
              <View style={s.flex}>
                <Bar value={x.progress / x.target} color={x.done ? c.success : c.accent} />
              </View>
              <Text style={s.count}>
                {x.progress}/{x.target}
              </Text>
            </View>
          </View>
          <Text style={[s.reward, x.done && { color: c.success, borderColor: c.success }]}>+{x.points}</Text>
        </View>
      ))}
    </Card>
  );
}

export function Badges({ progress }: { progress: Progress }) {
  const c = useColors();
  const s = useStyles();
  const open = (b: Progress['badges'][number]) =>
    showDialog(
      b.title,
      b.unlocked ? `${b.description}.\nОтключена на ${shortDate((b.unlocked_at ?? 0) * 1000)}.` : `${b.description}.\nДосега: ${b.progress} от ${b.target}.`,
    );
  // Три колони с точна ширина от измерената (процентите плюс отстъпите не се събират на тесни екрани и редовете се
  // разместваха); всички значки са с еднаква височина.
  const [width, setWidth] = useState(0);
  const cell = width ? Math.floor((width - GAP * (COLUMNS - 1)) / COLUMNS) : 0;
  return (
    <View style={s.grid} onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
      {!!cell && progress.badges.map((b) => (
        <Pressable
          key={b.id}
          onPress={() => open(b)}
          style={({ pressed }) => [s.badge, { width: cell }, pressed && s.pressed]}
          accessibilityRole="button"
          accessibilityLabel={`${b.title}, ${b.unlocked ? 'отключена' : `${b.progress} от ${b.target}`}`}
        >
          <View style={[s.badgeIcon, !b.unlocked && s.badgeLocked]}>
            <Svg width={HEX} height={HEX} style={s.ring}>
              <Polygon
                points={HEX_POINTS}
                fill="none"
                stroke={b.unlocked ? c.accent : c.lineStrong}
                strokeWidth={1.5}
                strokeDasharray={b.unlocked ? undefined : '3 3'}
              />
            </Svg>
            <Ionicons name={b.icon as IconName} size={20} color={b.unlocked ? c.accent : c.muted} />
            {!b.unlocked && (
              <View style={s.lock}>
                <Ionicons name="lock-closed-outline" size={10} color={c.muted} />
              </View>
            )}
          </View>
          <Text style={[s.badgeTitle, !b.unlocked && s.lockedText]} numberOfLines={2}>
            {b.title}
          </Text>
          <View style={s.badgeFoot}>
            {b.unlocked ? (
              <Text style={s.badgeSub}>{shortDate((b.unlocked_at ?? 0) * 1000)}</Text>
            ) : b.target > 1 ? (
              <View style={s.badgeBar}>
                <Bar value={b.progress / b.target} color={c.muted} />
              </View>
            ) : (
              <Text style={s.badgeSub}>заключена</Text>
            )}
          </View>
        </Pressable>
      ))}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  ring: { position: 'absolute' },
  ringInside: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  ringNumber: { fontFamily: font.monoMedium, fontSize: 22, color: c.text },
  ringNumberSmall: { fontSize: 17 },
  level: { gap: space.md },
  levelHead: { flexDirection: 'row', alignItems: 'center', gap: space.lg },
  points: { alignItems: 'flex-end', gap: 2 },
  pointsValue: { fontFamily: font.monoMedium, fontSize: 22, lineHeight: 28, color: c.text },
  week: { flexDirection: 'row', alignItems: 'center', gap: 6, borderTopWidth: 1, borderTopColor: c.line, paddingTop: space.md },
  weekText: { fontFamily: font.mono, fontSize: 12, color: c.success },
  strip: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  stripTitle: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  track: { height: 3, backgroundColor: c.surface3, overflow: 'hidden' },
  fill: { height: 3 },
  list: { paddingVertical: 0 },
  listNote: { ...text.mono, fontSize: 11.5, color: c.muted, paddingTop: space.md },
  rowLine: { borderBottomWidth: 1, borderBottomColor: c.line },
  challenge: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: 14 },
  icon: { width: 36, height: 36, borderRadius: radius.sm, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  challengeBody: { flex: 1, gap: 8 },
  challengeTitle: { fontFamily: font.medium, fontSize: 14.5, lineHeight: 19, color: c.text },
  doneText: { color: c.text2 },
  challengeBar: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  count: { fontFamily: font.mono, fontSize: 11.5, color: c.muted },
  reward: {
    fontFamily: font.monoMedium,
    fontSize: 12.5,
    color: c.accent,
    borderWidth: 1,
    borderColor: c.accentLine,
    borderRadius: radius.pill,
    paddingHorizontal: 9,
    paddingVertical: 3,
    overflow: 'hidden',
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: GAP, minHeight: 1 },
  badge: {
    alignItems: 'center',
    gap: 6,
    paddingVertical: 14,
    paddingHorizontal: 6,
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.sm,
  },
  pressed: { opacity: 0.7 },
  badgeIcon: { width: HEX, height: HEX, alignItems: 'center', justifyContent: 'center' },
  badgeLocked: { opacity: 0.5 },
  lock: {
    position: 'absolute',
    right: -3,
    bottom: -3,
    width: 18,
    height: 18,
    borderRadius: radius.sm,
    backgroundColor: c.surface3,
    borderWidth: 1,
    borderColor: c.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Мястото на две реда, за да са значките с еднаква височина и с подравнени дати и ленти.
  badgeTitle: { fontFamily: font.medium, fontSize: 12.5, lineHeight: 16, height: 32, textAlign: 'center', textAlignVertical: 'center', color: c.text },
  badgeFoot: { height: 16, alignSelf: 'stretch', alignItems: 'center', justifyContent: 'center' },
  lockedText: { color: c.muted },
  badgeSub: { fontFamily: font.mono, fontSize: 10.5, color: c.muted },
  badgeBar: { alignSelf: 'stretch', paddingHorizontal: 10 },
}));

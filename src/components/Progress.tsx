import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect } from 'react';
import { Alert, Pressable, Text, View } from 'react-native';
import Animated, { useAnimatedProps, useAnimatedStyle, useReducedMotion, useSharedValue, withDelay, withTiming } from 'react-native-reanimated';
import Svg, { Circle } from 'react-native-svg';

import type { Progress } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { font, makeStyles, radius, space, text, useColors } from '@/theme';
import { CountUp, ease, PressScale } from './motion';
import { Card, H2, Label, Small, type IconName } from './ui';

// The account's points (lib/progress.ts) on the profile and the home screen: the level in a ring that fills towards
// the next, the week's challenges, the badges.

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const share = (p: Progress) => Math.min(1, Math.max(0, (p.points - p.level.from) / Math.max(1, p.level.to - p.level.from)));

/** The level's number in a ring filled as far as the points have come towards the next level. */
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
          strokeLinecap="round"
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

/** A thin bar filled to `value` (0..1), growing in. */
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

/** The level on the profile: ring, name, points and how far the next level is. */
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

/** The level on the home screen: one line that opens the profile. */
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
          <View style={[s.icon, { backgroundColor: x.done ? c.successSoft : c.surface2 }]}>
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
    Alert.alert(
      b.title,
      b.unlocked ? `${b.description}.\nОтключена на ${shortDate((b.unlocked_at ?? 0) * 1000)}.` : `${b.description}.\nДосега: ${b.progress} от ${b.target}.`,
    );
  return (
    <View style={s.grid}>
      {progress.badges.map((b) => (
        <Pressable
          key={b.id}
          onPress={() => open(b)}
          style={({ pressed }) => [s.badge, pressed && s.pressed]}
          accessibilityRole="button"
          accessibilityLabel={`${b.title}, ${b.unlocked ? 'отключена' : `${b.progress} от ${b.target}`}`}
        >
          <View style={[s.badgeIcon, b.unlocked ? { backgroundColor: c.warningSoft, borderColor: c.warning } : s.badgeLocked]}>
            <Ionicons name={b.icon as IconName} size={22} color={b.unlocked ? c.warning : c.muted} />
            {!b.unlocked && (
              <View style={s.lock}>
                <Ionicons name="lock-closed-outline" size={10} color={c.muted} />
              </View>
            )}
          </View>
          <Text style={[s.badgeTitle, !b.unlocked && s.lockedText]} numberOfLines={2}>
            {b.title}
          </Text>
          {b.unlocked ? (
            <Text style={s.badgeSub}>{shortDate((b.unlocked_at ?? 0) * 1000)}</Text>
          ) : b.target > 1 ? (
            <View style={s.badgeBar}>
              <Bar value={b.progress / b.target} color={c.muted} />
            </View>
          ) : (
            <Text style={s.badgeSub}>заключена</Text>
          )}
        </Pressable>
      ))}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  ring: { position: 'absolute' },
  ringInside: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  ringNumber: { fontFamily: font.semibold, fontSize: 24, letterSpacing: -0.8, color: c.text },
  ringNumberSmall: { fontSize: 17 },
  level: { gap: space.md },
  levelHead: { flexDirection: 'row', alignItems: 'center', gap: space.lg },
  points: { alignItems: 'flex-end', gap: 2 },
  pointsValue: { fontFamily: font.monoMedium, fontSize: 22, lineHeight: 28, color: c.text },
  week: { flexDirection: 'row', alignItems: 'center', gap: 6, borderTopWidth: 1, borderTopColor: c.line, paddingTop: space.md },
  weekText: { fontFamily: font.mono, fontSize: 12, color: c.success },
  strip: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  stripTitle: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  track: { height: 6, borderRadius: 3, backgroundColor: c.surface3, overflow: 'hidden' },
  fill: { height: 6, borderRadius: 3 },
  list: { paddingVertical: 0 },
  listNote: { ...text.mono, fontSize: 11.5, color: c.muted, paddingTop: space.md },
  rowLine: { borderBottomWidth: 1, borderBottomColor: c.line },
  challenge: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: 14 },
  icon: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center' },
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
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  badge: {
    width: '31.8%',
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
  badgeIcon: { width: 48, height: 48, borderRadius: 24, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  badgeLocked: { backgroundColor: c.surface2, borderColor: c.line },
  lock: {
    position: 'absolute',
    right: -3,
    bottom: -3,
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: c.surface3,
    borderWidth: 1,
    borderColor: c.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  badgeTitle: { fontFamily: font.medium, fontSize: 12.5, lineHeight: 16, textAlign: 'center', color: c.text },
  lockedText: { color: c.muted },
  badgeSub: { fontFamily: font.mono, fontSize: 10.5, color: c.muted },
  badgeBar: { alignSelf: 'stretch', paddingHorizontal: 10, paddingTop: 4 },
}));

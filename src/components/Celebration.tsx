import Ionicons from '@expo/vector-icons/Ionicons';
import * as Haptics from 'expo-haptics';
import { useEffect } from 'react';
import { Pressable, Text, View } from 'react-native';
import Animated, {
  Easing,
  FadeInDown,
  FadeOutUp,
  ReduceMotion,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withDelay,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { dismissCelebration, useCelebrations, type Queued } from '@/lib/progress';
import { font, makeStyles, radius, space, text, useColors, type Palette } from '@/theme';
import type { IconName } from './ui';
import { ease } from './motion';

// Спечелени точки, изпълнено седмично предизвикателство, значка, ново ниво: най-горе на отворения екран се спуска
// карта с избухваща икона и изчезва сама (или при докосване). Няколко поредни идват една след друга.

const SPARKS = 10;

/* Иконата в пръстена си: пръстенът набъбва и избледнява, а искри излитат (само при големи моменти). */
function Burst({ icon, color, soft, sparks }: { icon: IconName; color: string; soft: string; sparks: boolean }) {
  const s = useStyles();
  const reduce = useReducedMotion();
  const t = useSharedValue(reduce ? 1 : 0);
  const pop = useSharedValue(reduce ? 1 : 0.4);
  useEffect(() => {
    if (reduce) return;
    pop.set(withTiming(1, { duration: 420, easing: Easing.out(Easing.back(2.2)) }));
    t.set(withDelay(80, withTiming(1, { duration: 900, easing: Easing.out(Easing.cubic) })));
  }, [reduce, t, pop]);
  const ring = useAnimatedStyle(() => ({ opacity: 0.6 * (1 - t.value), transform: [{ scale: 1 + 0.9 * t.value }] }));
  const core = useAnimatedStyle(() => ({ transform: [{ scale: pop.value }] }));
  return (
    <View style={s.burst}>
      {!reduce && <Animated.View style={[s.ring, { borderColor: color }, ring]} />}
      {!reduce && sparks && Array.from({ length: SPARKS }, (_, i) => <Spark key={i} angle={(i / SPARKS) * 2 * Math.PI} color={color} t={t} />)}
      <Animated.View style={[s.core, { backgroundColor: soft }, core]}>
        <Ionicons name={icon} size={22} color={color} />
      </Animated.View>
    </View>
  );
}

function Spark({ angle, color, t }: { angle: number; color: string; t: SharedValue<number> }) {
  const s = useStyles();
  const style = useAnimatedStyle(() => {
    const d = 18 + 22 * t.value;
    return {
      opacity: t.value < 0.05 ? 0 : 1 - t.value,
      transform: [{ translateX: Math.cos(angle) * d }, { translateY: Math.sin(angle) * d }, { scale: 1 - 0.5 * t.value }],
    };
  });
  return <Animated.View style={[s.spark, { backgroundColor: color }, style]} />;
}

type Look = { kicker: string; title: string; sub?: string; chip?: string; icon: IconName; color: string; soft: string; big: boolean };

function look(item: Queued, c: Palette): Look {
  switch (item.kind) {
    case 'points':
      return { kicker: `+${item.points} точки`, title: item.label, icon: 'sparkles-outline', color: c.accent, soft: c.accentSoft, big: false };
    case 'challenge':
      return { kicker: 'Предизвикателство изпълнено', title: item.title, chip: `+${item.points}`, icon: 'flag-outline', color: c.success, soft: c.successSoft, big: true };
    case 'badge':
      return { kicker: 'Нова значка', title: item.badge.title, sub: item.badge.description, icon: item.badge.icon as IconName, color: c.warning, soft: c.warningSoft, big: true };
    case 'level':
      return { kicker: 'Ново ниво', title: `Ниво ${item.number} · ${item.name}`, sub: 'Браво! Продължавайте в същия дух.', icon: 'trophy-outline', color: c.accent, soft: c.accentSoft, big: true };
    case 'welcome':
      return {
        kicker: 'Награди',
        title: `Вече имате ${item.points} точки`,
        sub: item.badges > 0 ? `и ${item.badges} ${item.badges === 1 ? 'значка' : 'значки'} за досегашната си работа` : 'за досегашните си сканирания',
        icon: 'ribbon-outline',
        color: c.warning,
        soft: c.warningSoft,
        big: true,
      };
  }
}

function Card({ item }: { item: Queued }) {
  const c = useColors();
  const s = useStyles();
  const insets = useSafeAreaInsets();
  const x = look(item, c);
  useEffect(() => {
    if (x.big) Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    else Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    const timer = setTimeout(dismissCelebration, x.big ? 4200 : 2800);
    return () => clearTimeout(timer);
  }, [x.big]);
  return (
    <Animated.View
      entering={FadeInDown.duration(420).easing(ease).reduceMotion(ReduceMotion.System)}
      exiting={FadeOutUp.duration(260).reduceMotion(ReduceMotion.System)}
      style={[s.wrap, { top: insets.top + space.sm }]}
    >
      <Pressable
        onPress={dismissCelebration}
        style={[s.card, { borderColor: x.color }]}
        accessibilityRole="alert"
        accessibilityLiveRegion="polite"
        accessibilityLabel={[x.kicker, x.title, x.sub].filter(Boolean).join('. ')}
      >
        <Burst icon={x.icon} color={x.color} soft={x.soft} sparks={x.big} />
        <View style={s.words}>
          <Text style={[s.kicker, { color: x.color }]} numberOfLines={1}>
            {x.kicker}
          </Text>
          <Text style={s.title} numberOfLines={2}>
            {x.title}
          </Text>
          {x.sub && (
            <Text style={s.sub} numberOfLines={2}>
              {x.sub}
            </Text>
          )}
        </View>
        {x.chip && <Text style={[s.chip, { color: x.color, borderColor: x.color }]}>{x.chip}</Text>}
      </Pressable>
    </Animated.View>
  );
}

/* Монтира се веднъж, над всички екрани (app/_layout.tsx). */
export function Celebrations() {
  const queue = useCelebrations();
  return queue.length ? <Card key={queue[0].id} item={queue[0]} /> : null;
}

const useStyles = makeStyles((c) => ({
  wrap: { position: 'absolute', left: space.lg, right: space.lg, zIndex: 100 },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    padding: 14,
    paddingRight: 16,
    backgroundColor: c.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    elevation: 10,
  },
  burst: { width: 48, height: 48, alignItems: 'center', justifyContent: 'center' },
  ring: { position: 'absolute', width: 48, height: 48, borderRadius: 24, borderWidth: 2 },
  core: { width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center' },
  spark: { position: 'absolute', width: 5, height: 5, borderRadius: 2.5 },
  words: { flex: 1, gap: 2 },
  kicker: { ...text.label, fontFamily: font.monoMedium },
  title: { fontFamily: font.semibold, fontSize: 16, lineHeight: 21, letterSpacing: -0.3, color: c.text },
  sub: { ...text.small, color: c.muted },
  chip: { fontFamily: font.monoMedium, fontSize: 13, borderWidth: 1, borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 4, overflow: 'hidden' },
}));

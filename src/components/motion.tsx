import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Pressable,
  Text,
  useWindowDimensions,
  View,
  type PressableProps,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import Animated, {
  cancelAnimation,
  Easing,
  FadeIn,
  FadeInDown,
  measure,
  ReduceMotion,
  useAnimatedReaction,
  useAnimatedRef,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withDelay,
  withRepeat,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';

import { font, makeStyles, useColors } from '@/theme';

// The website's motion (assets/style.css): one calm curve, things rise 12px as they fade in, headlines come up line
// by line, parts further down come up as they scroll into view, numbers count up. All of it follows the phone's
// "reduce motion" setting.
export const ease = Easing.bezier(0.22, 0.8, 0.24, 1);
export const easeOut = ease;
const STAGGER_MS = 60;
const MAX_STAGGER = 8;

/** Content arriving: fades in and rises 12px, item `index` 60 ms after the one before. */
export const enter = (index = 0) =>
  FadeInDown.duration(700)
    .delay(Math.min(index, MAX_STAGGER) * STAGGER_MS)
    .easing(ease)
    .withInitialValues({ opacity: 0, transform: [{ translateY: 12 }] })
    .reduceMotion(ReduceMotion.System);

/** A soft fade. */
export const fade = FadeIn.duration(260).easing(ease).reduceMotion(ReduceMotion.System);

/** A section switching in: a fade with a small rise. */
export const swap = FadeInDown.duration(380)
  .easing(ease)
  .withInitialValues({ opacity: 0, transform: [{ translateY: 8 }] })
  .reduceMotion(ReduceMotion.System);

export function Appear({ index, children, style }: { index?: number; children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return (
    <Animated.View entering={enter(index)} style={style}>
      {children}
    </Animated.View>
  );
}

// --- Headlines ----------------------------------------------------------------------------------------------------

export type Line = string | { soft: string };

/** A headline that comes up line by line from under its own baseline (the website's hero). */
export function RiseLines({ lines, style, softColor, delay = 0 }: { lines: Line[]; style: TextStyle; softColor: string; delay?: number }) {
  const reduce = useReducedMotion();
  const h = (style.lineHeight ?? 44) * 1.08;
  return (
    <View accessible accessibilityRole="header" accessibilityLabel={lines.map((l) => (typeof l === 'string' ? l : l.soft)).join(' ')}>
      {lines.map((l, i) => {
        const wait = delay + i * 90;
        const entering = reduce
          ? undefined
          : () => {
              'worklet';
              return {
                initialValues: { transform: [{ translateY: h }] },
                animations: { transform: [{ translateY: withDelay(wait, withTiming(0, { duration: 1000, easing: ease })) }] },
              };
            };
        return (
          <View key={i} style={{ overflow: 'hidden', height: h, marginBottom: (style.lineHeight ?? 44) - h }}>
            <Animated.Text entering={entering} style={[style, typeof l !== 'string' && { color: softColor }]} numberOfLines={1} adjustsFontSizeToFit>
              {typeof l === 'string' ? l : l.soft}
            </Animated.Text>
          </View>
        );
      })}
    </View>
  );
}

// --- Scroll-in ----------------------------------------------------------------------------------------------------

const ScrollContext = createContext<SharedValue<number> | null>(null);
const RevealContext = createContext<SharedValue<number> | null>(null);

/** Around a scroll view's content: lets each Reveal inside know when it scrolls into view. */
export function RevealScroll({ y, children }: { y: SharedValue<number>; children: ReactNode }) {
  return <ScrollContext.Provider value={y}>{children}</ScrollContext.Provider>;
}

/** 0 → 1 as the nearest Reveal comes into view, for drawings that play then. */
export const useReveal = () => useContext(RevealContext);

/** Comes up (fades in, rises 18px) once it has scrolled into view, like the website's sections. */
export function Reveal({ children, style, delay = 0 }: { children: ReactNode; style?: StyleProp<ViewStyle>; delay?: number }) {
  const scrollY = useContext(ScrollContext);
  const reduce = useReducedMotion();
  const { height } = useWindowDimensions();
  const ref = useAnimatedRef<Animated.View>();
  const progress = useSharedValue(reduce ? 1 : 0);
  const laidOut = useSharedValue(0);
  const started = useSharedValue(reduce ? 1 : 0);
  useAnimatedReaction(
    () => [scrollY ? scrollY.value : 0, laidOut.value],
    () => {
      if (started.value || !laidOut.value) return;
      const m = measure(ref);
      if (!m || m.pageY > height * 0.92) return;
      started.set(1);
      progress.set(withDelay(delay, withTiming(1, { duration: 900, easing: ease })));
    },
  );
  const style2 = useAnimatedStyle(() => ({ opacity: progress.value, transform: [{ translateY: 18 * (1 - progress.value) }] }));
  return (
    <RevealContext.Provider value={progress}>
      <Animated.View ref={ref} style={[style, style2]} onLayout={() => laidOut.set(1)}>
        {children}
      </Animated.View>
    </RevealContext.Provider>
  );
}

// --- Touch --------------------------------------------------------------------------------------------------------

/** A light tick under the finger: 'select' for a choice, 'tap' for a main action. Silent where there is none. */
export function haptic(kind: 'select' | 'tap' = 'select') {
  (kind === 'select' ? Haptics.selectionAsync() : Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)).catch(() => {});
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

/** A Pressable that shrinks to 98% while held; `feedback` adds a light tick on press. */
export function PressScale({
  style,
  onPressIn,
  onPressOut,
  onPress,
  feedback,
  ...rest
}: Omit<PressableProps, 'style'> & { style?: StyleProp<ViewStyle>; feedback?: boolean }) {
  const reduce = useReducedMotion();
  const scale = useSharedValue(1);
  const animated = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  return (
    <AnimatedPressable
      {...rest}
      onPressIn={(e) => {
        if (!reduce) scale.set(withTiming(0.98, { duration: 150, easing: ease }));
        onPressIn?.(e);
      }}
      onPressOut={(e) => {
        scale.set(withTiming(1, { duration: 250, easing: ease }));
        onPressOut?.(e);
      }}
      onPress={
        onPress &&
        ((e) => {
          if (feedback) haptic('tap');
          onPress(e);
        })
      }
      style={[style, animated]}
    />
  );
}

// --- Loading and live states ------------------------------------------------------------------------------------

/** A loading placeholder with a band of light passing over it (still when motion is reduced). */
export function Skeleton({ style }: { style?: StyleProp<ViewStyle> }) {
  const c = useColors();
  const reduce = useReducedMotion();
  const [w, setW] = useState(0);
  const t = useSharedValue(0);
  useEffect(() => {
    if (reduce) return;
    t.set(withRepeat(withTiming(1, { duration: 1300, easing: Easing.linear }), -1, false));
    return () => cancelAnimation(t);
  }, [reduce, t]);
  const band = useAnimatedStyle(() => ({ transform: [{ translateX: -w + 2 * w * t.value }] }));
  return (
    <View style={[{ borderRadius: 6, overflow: 'hidden', backgroundColor: c.surface2 }, style]} onLayout={(e) => setW(e.nativeEvent.layout.width)}>
      {!reduce && w > 0 && (
        <Animated.View style={[{ position: 'absolute', top: 0, bottom: 0, width: w }, band]}>
          <LinearGradient colors={[c.surface2, c.surface3, c.surface2]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={{ flex: 1 }} />
        </Animated.View>
      )}
    </View>
  );
}

/** A small dot that breathes: "working on it". */
export function LiveDot({ color, size = 6 }: { color: string; size?: number }) {
  const reduce = useReducedMotion();
  const t = useSharedValue(0);
  useEffect(() => {
    if (reduce) return;
    t.set(withRepeat(withTiming(1, { duration: 1400, easing: Easing.out(Easing.quad) }), -1, false));
    return () => cancelAnimation(t);
  }, [reduce, t]);
  const ring = useAnimatedStyle(() => ({ opacity: 0.5 * (1 - t.value), transform: [{ scale: 1 + 1.6 * t.value }] }));
  const dot = { width: size, height: size, borderRadius: size / 2, backgroundColor: color };
  return (
    <View style={{ width: size, height: size }}>
      {!reduce && <Animated.View style={[dot, { position: 'absolute' }, ring]} />}
      <View style={dot} />
    </View>
  );
}

/** A number that counts up to `value` in 1.4 s (and glides on to a new one). */
export function CountUp({
  value,
  style,
  format = (n) => String(Math.round(n)),
  duration = 1400,
}: {
  value: number;
  style?: StyleProp<TextStyle>;
  format?: (n: number) => string;
  duration?: number;
}) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    if (reduce) return;
    const begin = from.current;
    const start = Date.now();
    let frame = requestAnimationFrame(function step() {
      const t = Math.min(1, (Date.now() - start) / duration);
      const v = begin + (value - begin) * (1 - (1 - t) ** 3);
      from.current = v;
      setShown(v);
      if (t < 1) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [value, reduce, duration]);
  return (
    <Text style={style} numberOfLines={1} adjustsFontSizeToFit>
      {format(reduce ? value : shown)}
    </Text>
  );
}

/** Words running slowly sideways without end (the website's ticker). */
export function Ticker({ items, style }: { items: string[]; style?: StyleProp<ViewStyle> }) {
  const s = useTickerStyles();
  const reduce = useReducedMotion();
  const [w, setW] = useState(0); // one copy of the words
  const x = useSharedValue(0);
  useEffect(() => {
    if (!w || reduce) return;
    x.set(0);
    x.set(withRepeat(withTiming(-w, { duration: w * 28, easing: Easing.linear }), -1, false));
    return () => cancelAnimation(x);
  }, [w, reduce, x]);
  const track = useAnimatedStyle(() => ({ transform: [{ translateX: x.value }] }));
  const copy = (key: string, measureIt?: boolean) => (
    <View key={key} style={s.copy} onLayout={measureIt ? (e) => setW(e.nativeEvent.layout.width) : undefined}>
      {items.map((t) => (
        <View key={t} style={s.item}>
          <Text style={s.star}>✦</Text>
          <Text style={s.word}>{t}</Text>
        </View>
      ))}
    </View>
  );
  return (
    <View style={[s.ticker, style]} accessibilityLabel={items.join(', ')}>
      <Animated.View style={[s.track, track]}>
        {copy('a', true)}
        {copy('b')}
      </Animated.View>
    </View>
  );
}

const useTickerStyles = makeStyles((c) => ({
  ticker: { overflow: 'hidden', borderTopWidth: 1, borderBottomWidth: 1, borderColor: c.line, paddingVertical: 16 },
  track: { flexDirection: 'row', width: 10000 },
  copy: { flexDirection: 'row' },
  item: { flexDirection: 'row', alignItems: 'center' },
  star: { fontSize: 10, color: c.lineStrong, marginHorizontal: 20 },
  word: { fontFamily: font.medium, fontSize: 19, letterSpacing: -0.6, color: c.muted },
}));

// --- Scroll -------------------------------------------------------------------------------------------------------

/** For the bar at the top of a page: its back and line appear once the page has scrolled a little. */
export function useBarStyle(y: SharedValue<number>) {
  return useAnimatedStyle(() => {
    const t = Math.min(1, Math.max(0, (y.value - 24) / 32));
    return { opacity: t };
  });
}

/** For a title in that bar: it comes in once the page's big title has gone. */
export function useBarTitleStyle(y: SharedValue<number>, after = 60) {
  return useAnimatedStyle(() => {
    const t = Math.min(1, Math.max(0, (y.value - after) / 28));
    return { opacity: t, transform: [{ translateY: 6 * (1 - t) }] };
  });
}

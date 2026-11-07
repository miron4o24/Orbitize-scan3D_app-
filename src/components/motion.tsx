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

import { useColors } from '@/theme';

// Движението от сайта (assets/style.css): една спокойна крива, елементите се вдигат с 12px, докато се появяват,
// заглавията излизат ред по ред, по-долните части се появяват при превъртане, числата се броят нагоре. Всичко следва
// настройката на телефона за намалено движение.
export const ease = Easing.bezier(0.22, 0.8, 0.24, 1);
export const easeOut = ease;
const STAGGER_MS = 60;
const MAX_STAGGER = 8;

/* Появяващо се съдържание: избледнява навътре и се вдига с 12px, елемент `index` идва 60 мс след предходния. */
export const enter = (index = 0) =>
  FadeInDown.duration(700)
    .delay(Math.min(index, MAX_STAGGER) * STAGGER_MS)
    .easing(ease)
    .withInitialValues({ opacity: 0, transform: [{ translateY: 12 }] })
    .reduceMotion(ReduceMotion.System);

/* Меко появяване. */
export const fade = FadeIn.duration(260).easing(ease).reduceMotion(ReduceMotion.System);

/* Смяна на раздел: появяване с малко вдигане. */
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

// --- Заглавия
// -------------------------------------------------------------------------------------------------------

export type Line = string | { soft: string };

/* Заглавие, което излиза ред по ред изпод собствената си базова линия (главният раздел на сайта). */
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

// --- Поява при превъртане
// -------------------------------------------------------------------------------------------

const ScrollContext = createContext<SharedValue<number> | null>(null);

/* Около съдържанието на превъртащ се изглед: съобщава на всеки Reveal вътре кога влиза в изгледа. */
export function RevealScroll({ y, children }: { y: SharedValue<number>; children: ReactNode }) {
  return <ScrollContext.Provider value={y}>{children}</ScrollContext.Provider>;
}

/* Появява се (избледнява навътре и се вдига с 18px), щом се превърти в изгледа, като разделите на сайта. */
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
    <Animated.View ref={ref} style={[style, style2]} onLayout={() => laidOut.set(1)}>
      {children}
    </Animated.View>
  );
}

// --- Докосване
// ------------------------------------------------------------------------------------------------------

/* Леко потрепване под пръста: 'select' за избор, 'tap' за главно действие. Без усещане, където го няма. */
export function haptic(kind: 'select' | 'tap' = 'select') {
  (kind === 'select' ? Haptics.selectionAsync() : Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)).catch(() => {});
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

/* Pressable, който се свива до 98%, докато е натиснат; `feedback` добавя леко потрепване при натискане. */
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

// --- Зареждане и живи състояния
// -------------------------------------------------------------------------------------

/* Заместител при зареждане с преминаваща светла лента (неподвижен при намалено движение). */
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

/* Малка пулсираща точка: "работи се". */
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

/* Число, което се брои до `value` за 1,4 с (и плавно преминава към ново). */
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

// --- Превъртане
// -----------------------------------------------------------------------------------------------------

/* За лентата най-горе на страницата: фонът и линията ѝ се появяват, щом страницата се превърти малко. */
export function useBarStyle(y: SharedValue<number>) {
  return useAnimatedStyle(() => {
    const t = Math.min(1, Math.max(0, (y.value - 24) / 32));
    return { opacity: t };
  });
}

/* За заглавие в тази лента: появява се, щом голямото заглавие на страницата изчезне. */
export function useBarTitleStyle(y: SharedValue<number>, after = 60) {
  return useAnimatedStyle(() => {
    const t = Math.min(1, Math.max(0, (y.value - after) / 28));
    return { opacity: t, transform: [{ translateY: 6 * (1 - t) }] };
  });
}

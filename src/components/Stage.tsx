import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useState } from 'react';
import { Text, View, type StyleProp, type ViewStyle } from 'react-native';
import Animated, {
  cancelAnimation,
  Easing,
  FadeInDown,
  FadeOutUp,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import Svg, { Circle, Defs, Ellipse, Mask, Path, Pattern, RadialGradient, Rect, Stop } from 'react-native-svg';

import { font, makeStyles, radius, useColors } from '@/theme';
import { ease } from './motion';
import { Brackets } from './ui';

// Сцената: радиална основа, мрежа от точки, която избледнява към краищата, ъглови скоби и в средата кубът (плоски
// стени в цветовете на темата, ръбове в цвета за акцент), носещ се, докато през него минава равнина на сканиране, а
// пътят на телефона го обикаля. Етикет отдолу минава през стъпките.

const CUBE = 124; // нарисуван размер на куба (рисунката му е от 200 единици)
const ORBIT_W = 250;
const ORBIT_H = 64;

/* Една точка от пътя на телефона около куба; зад куба потъмнява и намалява. */
function Orbiter({ t, phase, color }: { t: SharedValue<number>; phase: number; color: string }) {
  const style = useAnimatedStyle(() => {
    const a = (t.value + phase) * 2 * Math.PI;
    const front = Math.sin(a);
    return {
      opacity: 0.35 + 0.65 * ((front + 1) / 2),
      zIndex: front > 0 ? 3 : 0,
      transform: [
        { translateX: (ORBIT_W / 2) * Math.cos(a) },
        { translateY: (ORBIT_H / 2) * front },
        { scale: 0.7 + 0.45 * ((front + 1) / 2) },
      ],
    };
  });
  return <Animated.View style={[{ position: 'absolute', width: 8, height: 8, borderRadius: radius.pill, backgroundColor: color }, style]} />;
}

export type Journey = { title: string; icon: keyof typeof Ionicons.glyphMap }[];

/* Стъкленият етикет отдолу: една стъпка след друга, с линия, която се запълва, докато стъпката се показва. */
function JourneyPill({ steps }: { steps: Journey }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const [at, setAt] = useState(0);
  const fill = useSharedValue(0);
  useEffect(() => {
    if (reduce) return;
    fill.set(0);
    fill.set(withTiming(1, { duration: 2600, easing: Easing.linear }));
    const timer = setTimeout(() => setAt((i) => (i + 1) % steps.length), 2800);
    return () => clearTimeout(timer);
  }, [at, reduce, steps.length, fill]);
  const track = useAnimatedStyle(() => ({ width: `${fill.value * 100}%` }));
  const step = steps[at];
  return (
    <View style={s.journey}>
      <View style={s.journeyIcon}>
        <Ionicons name={step.icon} size={14} color={c.onPrimary} />
      </View>
      <Animated.View key={at} entering={reduce ? undefined : FadeInDown.duration(380).easing(ease)} exiting={reduce ? undefined : FadeOutUp.duration(200)} style={s.journeyText}>
        <Text style={s.journeyTitle} numberOfLines={1}>
          {step.title}
        </Text>
      </Animated.View>
      <View style={s.journeyTrack}>
        <Animated.View style={[s.journeyFill, track]} />
      </View>
    </View>
  );
}

export function Stage({
  height = 300,
  journey,
  style,
}: {
  height?: number;
  journey?: Journey;
  style?: StyleProp<ViewStyle>;
}) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const float = useSharedValue(0.5);
  const sweep = useSharedValue(0.15);
  const orbit = useSharedValue(0.1);
  useEffect(() => {
    if (reduce) return;
    float.set(withRepeat(withTiming(1, { duration: 2800, easing: Easing.inOut(Easing.sin) }), -1, true));
    sweep.set(
      withRepeat(
        withSequence(
          withTiming(1, { duration: 2400, easing: Easing.inOut(Easing.cubic) }),
          withTiming(0, { duration: 2400, easing: Easing.inOut(Easing.cubic) }),
        ),
        -1,
        false,
      ),
    );
    orbit.set(withRepeat(withTiming(1.1, { duration: 9000, easing: Easing.linear }), -1, false));
    return () => [float, sweep, orbit].forEach(cancelAnimation);
  }, [reduce, float, sweep, orbit]);

  const k = CUBE / 200; // единици от рисунката → точки
  const cube = useAnimatedStyle(() => ({ transform: [{ translateY: -5 + 10 * float.value }] }));
  // Равнината минава от горната страна на куба (y 58) до долната (y 142), в единици от рисунката.
  const plane = useAnimatedStyle(() => ({
    transform: [{ translateY: (-5 + 10 * float.value + 84 * sweep.value) * k }],
    opacity: 0.55 + 0.45 * Math.sin(sweep.value * Math.PI),
  }));

  return (
    <View style={[s.stage, { height }, style]} pointerEvents="none" accessibilityElementsHidden importantForAccessibility="no-hide-descendants">
      <Svg style={s.fill} width="100%" height="100%">
        <Defs>
          <RadialGradient id="ground" cx="55%" cy="40%" rx="90%" ry="75%">
            <Stop offset="0" stopColor={c.stage[0]} />
            <Stop offset="0.6" stopColor={c.stage[1]} />
            <Stop offset="1" stopColor={c.stage[2]} />
          </RadialGradient>
          <Pattern id="dots" patternUnits="userSpaceOnUse" width="22" height="22">
            <Circle cx="11" cy="11" r="1" fill={c.lineStrong} />
          </Pattern>
          <RadialGradient id="fadeOut" cx="50%" cy="45%" rx="70%" ry="65%">
            <Stop offset="0" stopColor={c.text} stopOpacity="1" />
            <Stop offset="0.85" stopColor={c.text} stopOpacity="0" />
          </RadialGradient>
          <Mask id="dotsMask">
            <Rect width="100%" height="100%" fill="url(#fadeOut)" />
          </Mask>
        </Defs>
        <Rect width="100%" height="100%" fill="url(#ground)" />
        <Rect width="100%" height="100%" fill="url(#dots)" mask="url(#dotsMask)" opacity={0.6} />
      </Svg>

      <View style={s.centre}>
        {/* пътят на телефона, под куба */}
        <Svg width={ORBIT_W + 8} height={ORBIT_H + 8} style={s.orbit}>
          <Ellipse
            cx={(ORBIT_W + 8) / 2}
            cy={(ORBIT_H + 8) / 2}
            rx={ORBIT_W / 2}
            ry={ORBIT_H / 2}
            stroke={c.lineStrong}
            strokeWidth={1}
            strokeDasharray="3 5"
            fill="none"
          />
        </Svg>
        <View style={s.orbiters}>
          <Orbiter t={orbit} phase={0} color={c.signal} />
          <Orbiter t={orbit} phase={0.5} color={c.text2} />
        </View>
        <Animated.View style={[s.cube, cube]}>
          <Svg width={CUBE} height={CUBE} viewBox="0 0 200 200">
            <Path d="M100 18 172 58 100 98 28 58Z" fill={c.surface3} />
            <Path d="M28 58 100 98V182L28 142Z" fill={c.surface2} />
            <Path d="M172 58 100 98V182L172 142Z" fill={c.surface} />
            <Path d="M100 18 172 58V142L100 182 28 142V58ZM100 98V182M28 58 100 98 172 58" stroke={c.accent} strokeWidth={1.5} fill="none" />
          </Svg>
        </Animated.View>
        {/* равнината на сканиране, която го пресича */}
        <Animated.View style={[s.cube, s.plane, plane]}>
          <Svg width={CUBE} height={CUBE} viewBox="0 0 200 200" style={{ overflow: 'visible' }}>
            <Path d="M100 4 196 58 100 112 4 58Z" fill={c.accentSoft} stroke={c.accent} strokeWidth={1.4} />
          </Svg>
        </Animated.View>
      </View>

      <Brackets inset={10} />
      {journey && <JourneyPill steps={journey} />}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  stage: { borderRadius: radius.md, borderWidth: 1, borderColor: c.line, overflow: 'hidden', alignItems: 'center', justifyContent: 'center' },
  fill: { position: 'absolute', left: 0, top: 0, right: 0, bottom: 0 },
  centre: { width: ORBIT_W + 8, height: CUBE + 20, alignItems: 'center', justifyContent: 'center', marginTop: -28 },
  orbit: { position: 'absolute', bottom: -ORBIT_H / 2 + 6 },
  orbiters: { position: 'absolute', bottom: 10, left: 0, right: 0, alignItems: 'center', justifyContent: 'center', height: 0 },
  cube: { width: CUBE, height: CUBE, zIndex: 1 },
  plane: { position: 'absolute', zIndex: 2 },
  journey: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 9,
    paddingHorizontal: 12,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.lineStrong,
    backgroundColor: c.glass,
  },
  journeyIcon: { width: 26, height: 26, borderRadius: radius.sm, backgroundColor: c.primary, alignItems: 'center', justifyContent: 'center' },
  journeyText: { flex: 1, minWidth: 0 },
  journeyTitle: { fontFamily: font.semibold, fontSize: 13, color: c.text },
  journeyTrack: { width: 56, height: 2, backgroundColor: c.surface3, overflow: 'hidden' },
  journeyFill: { height: 2, backgroundColor: c.accent },
}));

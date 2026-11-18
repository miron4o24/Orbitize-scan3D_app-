import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useRef, type ReactNode } from 'react';
import { Modal, Pressable, Text, useWindowDimensions, View, type HostInstance, type StyleProp, type ViewStyle } from 'react-native';
import Animated, { FadeIn } from 'react-native-reanimated';

import { endTour, nextStep, setTarget, STEPS, useTour } from '@/lib/tour';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { Button, type IconName } from './ui';

/* Нещо, което обиколката посочва (lib/tour.ts STEPS[].target): мястото му на екрана се мери при показване и при
   всяка стъпка (превъртане междувременно). */
export function TourTarget({ id, children, style }: { id: string; children: ReactNode; style?: StyleProp<ViewStyle> }) {
  const ref = useRef<HostInstance>(null);
  const { step } = useTour();
  const measure = () =>
    ref.current?.measureInWindow((x, y, width, height) => {
      if (width && height) setTarget(id, { x, y, width, height });
    });
  useEffect(() => {
    if (step == null) return;
    const t = setTimeout(measure, 60);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);
  return (
    <View ref={ref} style={style} onLayout={measure} collapsable={false}>
      {children}
    </View>
  );
}

const PAD = 6; // колко около посоченото се осветява
const CARD_GAP = 14;

/* Обиколката: затъмнен екран с "прозорец" около посочения бутон и карта с обяснението. */
export function TourOverlay() {
  const { step, targets } = useTour();
  const c = useColors();
  const s = useStyles();
  const win = useWindowDimensions();
  if (step == null) return null;
  const now = STEPS[step];
  const rect = now.target ? targets[now.target] : undefined;
  const last = step === STEPS.length - 1;
  const hole = rect && {
    x: Math.max(0, rect.x - PAD),
    y: Math.max(0, rect.y - PAD),
    w: Math.min(win.width, rect.width + 2 * PAD),
    h: rect.height + 2 * PAD,
  };
  // Картата под посоченото, ако то е в горната половина, иначе над него; без цел - в средата.
  const below = hole && hole.y + hole.h / 2 < win.height / 2;
  const cardPos: ViewStyle = hole ? (below ? { top: hole.y + hole.h + CARD_GAP } : { bottom: win.height - hole.y + CARD_GAP }) : { top: win.height * 0.28 };
  return (
    <Modal visible transparent animationType="fade" statusBarTranslucent navigationBarTranslucent onRequestClose={endTour}>
      <View style={s.fill}>
        {hole ? (
          <>
            <View style={[s.shade, { left: 0, right: 0, top: 0, height: hole.y }]} />
            <View style={[s.shade, { left: 0, right: 0, top: hole.y + hole.h, bottom: 0 }]} />
            <View style={[s.shade, { left: 0, width: hole.x, top: hole.y, height: hole.h }]} />
            <View
              style={[
                s.shade,
                {
                  left: hole.x + hole.w,
                  right: 0,
                  top: hole.y,
                  height: hole.h,
                },
              ]}
            />
            <Animated.View
              key={`ring-${step}`}
              entering={FadeIn.duration(220)}
              style={[s.ring, { left: hole.x, top: hole.y, width: hole.w, height: hole.h }]}
              pointerEvents="none"
            />
          </>
        ) : (
          <View style={[s.shade, s.fill]} />
        )}
        <Animated.View key={`card-${step}`} entering={FadeIn.duration(220)} style={[s.card, cardPos]}>
          <View style={s.head}>
            <View style={s.icon}>
              <Ionicons name={now.icon as IconName} size={18} color={c.accent} />
            </View>
            <Text style={s.count}>
              {String(step + 1).padStart(2, '0')} / {String(STEPS.length).padStart(2, '0')}
            </Text>
          </View>
          <Text style={s.title}>{now.title}</Text>
          <Text style={s.text}>{now.text}</Text>
          <View style={s.dots}>
            {STEPS.map((_, i) => (
              <View key={i} style={[s.dot, i === step && s.dotOn]} />
            ))}
          </View>
          <View style={s.buttons}>
            {!last && (
              <Pressable onPress={endTour} hitSlop={8} accessibilityRole="button" accessibilityLabel="Пропусни обиколката">
                <Text style={s.skip}>Пропусни</Text>
              </Pressable>
            )}
            <Button title={last ? 'Готово' : 'Напред'} icon={last ? 'checkmark' : 'arrow-forward'} onPress={nextStep} style={s.next} />
          </View>
        </Animated.View>
      </View>
    </Modal>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1 },
  shade: { position: 'absolute', backgroundColor: 'rgba(0,0,0,0.72)' },
  ring: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: c.accent,
    borderRadius: radius.md,
  },
  card: {
    position: 'absolute',
    left: 16,
    right: 16,
    padding: space.lg,
    gap: space.sm,
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.accentLine,
    borderRadius: radius.md,
  },
  head: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  icon: {
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: c.accentLine,
    borderRadius: radius.sm,
  },
  count: {
    fontFamily: font.monoMedium,
    fontSize: 12,
    letterSpacing: 0.6,
    color: c.muted,
  },
  title: { fontFamily: font.semibold, fontSize: 18, color: c.text },
  text: {
    fontFamily: font.regular,
    fontSize: 14.5,
    lineHeight: 21,
    color: c.text2,
  },
  dots: { flexDirection: 'row', gap: 5, marginTop: space.xs },
  dot: {
    width: 6,
    height: 6,
    borderRadius: radius.pill,
    backgroundColor: c.surface3,
  },
  dotOn: { width: 16, backgroundColor: c.accent },
  buttons: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: space.sm,
    gap: space.md,
  },
  skip: {
    fontFamily: font.monoMedium,
    fontSize: 12.5,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    color: c.muted,
  },
  next: { marginLeft: 'auto' },
}));

import Ionicons from '@expo/vector-icons/Ionicons';
import { LinearGradient } from 'expo-linear-gradient';
import { Tabs } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useReducedMotion, useSharedValue, withSequence, withTiming } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { easeOut, haptic, PressScale } from '@/components/motion';
import type { IconName } from '@/components/ui';
import { startScan } from '@/lib/startScan';
import { brandDeep, font, makeStyles, space, useColors } from '@/theme';

// По два раздела от двете страни на бутона за сканиране, който е на позиция 2.
const ITEMS: Record<string, { label: string; icon: IconName; active: IconName; slot: number }> = {
  index: { label: 'Начало', icon: 'home-outline', active: 'home', slot: 0 },
  community: { label: 'Общност', icon: 'people-outline', active: 'people', slot: 1 },
  scans: { label: 'Сканирания', icon: 'cube-outline', active: 'cube', slot: 3 },
  profile: { label: 'Профил', icon: 'person-outline', active: 'person', slot: 4 },
};
const SLOTS = 5;
const MARK = 20; // линията над отворения раздел

type BarProps = { state: { index: number; routes: { key: string; name: string }[] }; navigateToTab: (routeKey: string) => void };

/* Иконата на раздел, която леко подскача при отварянето му. */
function TabIcon({ name, on, color }: { name: IconName; on: boolean; color: string }) {
  const reduce = useReducedMotion();
  const scale = useSharedValue(1);
  useEffect(() => {
    if (on && !reduce) scale.set(withSequence(withTiming(1.12, { duration: 120, easing: easeOut }), withTiming(1, { duration: 200, easing: easeOut })));
  }, [on, reduce, scale]);
  const style = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  return (
    <Animated.View style={style}>
      <Ionicons name={name} size={22} color={color} />
    </Animated.View>
  );
}

function TabBar({ state, navigateToTab }: BarProps) {
  const insets = useSafeAreaInsets();
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const current = state.routes[state.index]?.name;
  const slot = ITEMS[current]?.slot;
  const [width, setWidth] = useState(0);
  const x = useSharedValue(0);
  const shown = useSharedValue(0);
  const placed = useRef(false);
  useEffect(() => {
    if (!width) return;
    const each = width / SLOTS;
    shown.set(withTiming(slot == null ? 0 : 1, { duration: 200 }));
    if (slot == null) return;
    const to = each * slot + (each - MARK) / 2;
    x.set(placed.current && !reduce ? withTiming(to, { duration: 300, easing: easeOut }) : to);
    placed.current = true;
  }, [slot, width, reduce, x, shown]);
  const mark = useAnimatedStyle(() => ({ opacity: shown.value, transform: [{ translateX: x.value }] }));

  const item = (name: string) => {
    const it = ITEMS[name];
    const on = current === name;
    return (
      <Pressable
        key={name}
        style={s.item}
        onPress={() => {
          if (on) return;
          haptic();
          const route = state.routes.find((r) => r.name === name);
          if (route) navigateToTab(route.key);
        }}
        accessibilityRole="tab"
        accessibilityState={{ selected: on }}
        accessibilityLabel={it.label}
      >
        <TabIcon name={on ? it.active : it.icon} on={on} color={on ? c.text : c.muted} />
        <Text style={[s.label, on && s.labelOn]}>{it.label}</Text>
      </Pressable>
    );
  };
  return (
    <View style={[s.bar, { paddingBottom: Math.max(insets.bottom, space.sm) }]} onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
      <Animated.View style={[s.mark, mark]} />
      {item('index')}
      {item('community')}
      <View style={s.item}>
        <PressScale onPress={startScan} feedback style={s.scan} accessibilityRole="button" accessibilityLabel="Ново сканиране">
          <LinearGradient colors={brandDeep} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={StyleSheet.absoluteFill} />
          <Ionicons name="add" size={26} color="#ffffff" />
        </PressScale>
      </View>
      {item('scans')}
      {item('profile')}
    </View>
  );
}

export default function TabsLayout() {
  const c = useColors();
  return (
    <Tabs
      tabBar={(props) => <TabBar {...props} />}
      screenOptions={{ headerShown: false, sceneStyle: { backgroundColor: c.bg }, animation: 'fade' }}
    >
      <Tabs.Screen name="index" />
      <Tabs.Screen name="community" />
      <Tabs.Screen name="scans" />
      <Tabs.Screen name="profile" />
    </Tabs>
  );
}

const useStyles = makeStyles((c) => ({
  bar: { flexDirection: 'row', backgroundColor: c.bg, paddingTop: space.sm, borderTopWidth: 1, borderTopColor: c.line },
  mark: { position: 'absolute', top: -1, left: 0, width: MARK, height: 2, borderRadius: 1, backgroundColor: c.accent },
  item: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 3, minHeight: 48 },
  label: { fontFamily: font.medium, fontSize: 11, color: c.muted },
  labelOn: { color: c.text },
  scan: { width: 54, height: 42, borderRadius: 14, overflow: 'hidden', alignItems: 'center', justifyContent: 'center' },
}));

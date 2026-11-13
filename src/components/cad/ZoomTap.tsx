import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { PanResponder, Pressable, View, type GestureResponderEvent, type PanResponderInstance, type ViewStyle } from 'react-native';

import { clampZoom as clamp, pinch, unproject, ZOOM_MAX as MAX, type Zoom as View_ } from '@/lib/zoom';
import { makeStyles, radius, useColors } from '@/theme';

// Снимка, която се увеличава с два пръста (или с бутоните) и се мести с един, а докосване (без местене) връща
// точката върху снимката, от 0 до 1 - за избиране на малки детайли (очертанието, отвор за махане). Без нови
// библиотеки: PanResponder от React Native.

const STEP = 1.8; // увеличение с един бутон
const SLOP = 8; // толкова пиксела движение още е докосване

export function ZoomTap({
  ratio,
  disabled,
  onTap,
  children,
  style,
  label,
}: {
  ratio: number;
  disabled?: boolean;
  onTap: (x: number, y: number) => void;
  children: ReactNode;
  style?: ViewStyle;
  label: string;
}) {
  const s = useStyles();
  const c = useColors();
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View_>({ s: 1, x: 0, y: 0 });
  // Текущото състояние за жеста (PanResponder се създава веднъж).
  const live = useRef({ view, size, disabled, onTap });
  useLayoutEffect(() => {
    live.current = { view, size, disabled, onTap };
  });
  const gesture = useRef({ start: { s: 1, x: 0, y: 0 }, moved: false, pinched: false, d0: 0, m0: { x: 0, y: 0 }, touches: 0 });
  const [responder, setResponder] = useState<PanResponderInstance | null>(null);

  useEffect(() => {
    const begin = (e: GestureResponderEvent) => {
      const t = e.nativeEvent.touches;
      const g = gesture.current;
      g.start = live.current.view;
      g.touches = t.length;
      if (t.length >= 2) {
        g.pinched = true;
        g.d0 = Math.hypot(t[0].locationX - t[1].locationX, t[0].locationY - t[1].locationY) || 1;
        g.m0 = { x: (t[0].locationX + t[1].locationX) / 2, y: (t[0].locationY + t[1].locationY) / 2 };
      }
    };
    const created = PanResponder.create({
      onStartShouldSetPanResponder: () => !live.current.disabled,
      onMoveShouldSetPanResponder: () => !live.current.disabled,
      onMoveShouldSetPanResponderCapture: (e) => !live.current.disabled && (e.nativeEvent.touches.length >= 2 || live.current.view.s > 1),
      onPanResponderTerminationRequest: () => false,
      onPanResponderGrant: (e) => {
        gesture.current.moved = false;
        gesture.current.pinched = false;
        begin(e);
      },
      onPanResponderMove: (e, gs) => {
        const g = gesture.current;
        const { w, h } = live.current.size;
        const t = e.nativeEvent.touches;
        if (t.length !== g.touches) begin(e); // пръст добавен или вдигнат: нова основа
        if (t.length >= 2) {
          const d = Math.hypot(t[0].locationX - t[1].locationX, t[0].locationY - t[1].locationY) || 1;
          const m = { x: (t[0].locationX + t[1].locationX) / 2, y: (t[0].locationY + t[1].locationY) / 2 };
          setView(pinch(g.start, w, h, g.m0, m, (g.start.s * d) / g.d0)); // точката под пръстите остава под тях
          g.moved = true;
          return;
        }
        if (Math.abs(gs.dx) > SLOP || Math.abs(gs.dy) > SLOP) g.moved = true;
        if (g.moved && !g.pinched && g.start.s > 1) setView(clamp({ ...g.start, x: g.start.x + gs.dx, y: g.start.y + gs.dy }, w, h));
      },
      onPanResponderRelease: (e) => {
        const g = gesture.current;
        const { w, h } = live.current.size;
        if (g.moved || g.pinched || !w || !h) return;
        const p = unproject(live.current.view, w, h, e.nativeEvent.locationX, e.nativeEvent.locationY);
        live.current.onTap(Math.min(1, Math.max(0, p.x / w)), Math.min(1, Math.max(0, p.y / h)));
      },
    });
    setResponder(created);
  }, []);

  const zoom = (k: number) => setView((v) => clamp({ s: v.s * k, x: v.x * k, y: v.y * k }, size.w, size.h));

  return (
    <View style={[s.frame, { aspectRatio: ratio }, style]}>
      <View
        style={s.fill}
        onLayout={(e) => setSize({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}
        accessibilityRole="imagebutton"
        accessibilityLabel={label}
        {...responder?.panHandlers}
      >
        <View pointerEvents="none" style={[s.fill, { transform: [{ translateX: view.x }, { translateY: view.y }, { scale: view.s }] }]}>
          {children}
        </View>
      </View>
      <View style={s.tools} pointerEvents="box-none">
        <Pressable style={s.tool} onPress={() => zoom(STEP)} disabled={view.s >= MAX} accessibilityRole="button" accessibilityLabel="Увеличи">
          <Ionicons name="add" size={20} color={c.text} />
        </Pressable>
        <Pressable style={s.tool} onPress={() => zoom(1 / STEP)} disabled={view.s <= 1} accessibilityRole="button" accessibilityLabel="Намали">
          <Ionicons name="remove" size={20} color={c.text} />
        </Pressable>
        {view.s > 1 && (
          <Pressable style={s.tool} onPress={() => setView({ s: 1, x: 0, y: 0 })} accessibilityRole="button" accessibilityLabel="Цялата снимка">
            <Ionicons name="contract-outline" size={18} color={c.text} />
          </Pressable>
        )}
      </View>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  frame: { width: '100%', borderRadius: radius.md, overflow: 'hidden', backgroundColor: c.surface2 },
  fill: { width: '100%', height: '100%' },
  tools: { position: 'absolute', right: 8, top: 8, gap: 8 },
  tool: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.line,
    opacity: 0.94,
  },
}));

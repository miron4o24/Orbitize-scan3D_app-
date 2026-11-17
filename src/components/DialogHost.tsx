import Ionicons from '@expo/vector-icons/Ionicons';
import { Modal, Pressable, Text, View } from 'react-native';
import Animated, { FadeIn, FadeInDown, useReducedMotion } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { closeDialog, useDialog, type DialogButton } from '@/lib/dialog';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { haptic } from './motion';
import type { IconName } from './ui';

// Прозорците на приложението (lib/dialog.ts) в техническия дизайн: плоска повърхност с рамка от 1px и остри ъгли.
// Списък от избори се плъзга отдолу (лист); потвърждение е в средата. Отказ е винаги отделен и последен.

export function DialogHost() {
  const d = useDialog();
  const c = useColors();
  const s = useStyles();
  const insets = useSafeAreaInsets();
  const reduce = useReducedMotion();
  if (!d) return null;
  const cancel = d.buttons.find((b) => b.style === 'cancel');
  const actions = d.buttons.filter((b) => b !== cancel);
  const choose = (b?: DialogButton) => {
    haptic();
    closeDialog(b);
  };
  const color = (b: DialogButton) => (b.style === 'destructive' ? c.danger : c.accent);

  return (
    <Modal visible transparent statusBarTranslucent navigationBarTranslucent animationType="fade" onRequestClose={() => closeDialog(cancel)}>
      <Pressable style={[s.backdrop, d.sheet ? s.bottom : s.middle]} onPress={() => cancel && choose(cancel)} accessibilityLabel="Затвори">
        <Animated.View
          key={d.id}
          entering={reduce ? undefined : d.sheet ? FadeInDown.duration(220) : FadeIn.duration(160)}
          style={[d.sheet ? s.sheet : s.box, d.sheet && { paddingBottom: Math.max(insets.bottom, space.md) + space.sm }]}
        >
          <Pressable accessible={false}>
            <View style={s.head}>
              <Text style={s.title}>{d.title}</Text>
              {!!d.message && <Text style={s.message}>{d.message}</Text>}
            </View>
            {d.sheet ? (
              <View style={s.options}>
                {actions.map((b) => (
                  <Pressable key={b.text} onPress={() => choose(b)} style={({ pressed }) => [s.option, pressed && s.pressed]} accessibilityRole="button" accessibilityLabel={b.sub ? `${b.text}. ${b.sub}` : b.text}>
                    {b.icon && (
                      <View style={[s.optionIcon, b.style === 'destructive' && { borderColor: c.danger }]}>
                        <Ionicons name={b.icon as IconName} size={20} color={color(b)} />
                      </View>
                    )}
                    <View style={s.flex}>
                      <Text style={[s.optionText, b.style === 'destructive' && { color: c.danger }]}>{b.text}</Text>
                      {!!b.sub && <Text style={s.optionSub}>{b.sub}</Text>}
                    </View>
                    <Ionicons name="chevron-forward" size={16} color={c.muted} />
                  </Pressable>
                ))}
              </View>
            ) : (
              <View style={[s.buttons, actions.length + (cancel ? 1 : 0) > 2 && s.buttonsColumn]}>
                {cancel && (
                  <Pressable onPress={() => choose(cancel)} style={({ pressed }) => [s.button, s.secondary, pressed && s.pressed]} accessibilityRole="button">
                    <Text style={s.secondaryText}>{cancel.text}</Text>
                  </Pressable>
                )}
                {actions.map((b) => (
                  <Pressable
                    key={b.text}
                    onPress={() => choose(b)}
                    style={({ pressed }) => [s.button, b.style === 'destructive' ? s.danger : s.primary, pressed && s.pressed]}
                    accessibilityRole="button"
                  >
                    <Text style={[s.primaryText, b.style === 'destructive' && { color: c.danger }]}>{b.text}</Text>
                  </Pressable>
                ))}
              </View>
            )}
            {d.sheet && cancel && (
              <Pressable onPress={() => choose(cancel)} style={({ pressed }) => [s.button, s.secondary, s.sheetCancel, pressed && s.pressed]} accessibilityRole="button">
                <Text style={s.secondaryText}>{cancel.text}</Text>
              </Pressable>
            )}
          </Pressable>
        </Animated.View>
      </Pressable>
    </Modal>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  backdrop: { flex: 1, backgroundColor: c.backdrop },
  middle: { justifyContent: 'center', padding: space.xl },
  bottom: { justifyContent: 'flex-end' },
  box: { backgroundColor: c.surface, borderWidth: 1, borderColor: c.lineStrong, borderRadius: radius.md, padding: space.lg },
  sheet: {
    backgroundColor: c.surface,
    borderTopWidth: 1,
    borderColor: c.lineStrong,
    borderTopLeftRadius: radius.md,
    borderTopRightRadius: radius.md,
    paddingHorizontal: space.lg,
    paddingTop: space.lg,
  },
  head: { gap: 6, marginBottom: space.lg },
  title: { fontFamily: font.semibold, fontSize: 18, lineHeight: 24, color: c.text },
  message: { fontFamily: font.regular, fontSize: 14, lineHeight: 20, color: c.text2 },
  options: { gap: space.sm },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    minHeight: 60,
    paddingHorizontal: space.md,
    paddingVertical: space.sm,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.md,
    backgroundColor: c.surface2,
  },
  optionIcon: { width: 40, height: 40, borderRadius: radius.sm, borderWidth: 1, borderColor: c.accentLine, alignItems: 'center', justifyContent: 'center' },
  optionText: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  optionSub: { fontFamily: font.mono, fontSize: 12, color: c.muted, marginTop: 2 },
  buttons: { flexDirection: 'row', gap: space.sm },
  buttonsColumn: { flexDirection: 'column-reverse' },
  button: { flex: 1, minHeight: 44, borderRadius: radius.sm, alignItems: 'center', justifyContent: 'center', paddingHorizontal: space.md, borderWidth: 1 },
  primary: { backgroundColor: c.primary, borderColor: c.primary },
  danger: { backgroundColor: c.dangerSoft, borderColor: c.danger },
  secondary: { backgroundColor: c.surface, borderColor: c.lineStrong },
  sheetCancel: { flex: 0, marginTop: space.md },
  primaryText: { fontFamily: font.semibold, fontSize: 15, color: c.onPrimary },
  secondaryText: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  pressed: { opacity: 0.75 },
}));

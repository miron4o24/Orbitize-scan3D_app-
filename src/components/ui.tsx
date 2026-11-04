import Ionicons from '@expo/vector-icons/Ionicons';
import { LinearGradient } from 'expo-linear-gradient';
import { useEffect, useRef, useState, type ComponentProps, type ReactNode, type Ref } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  type StyleProp,
  type TextInputInstance,
  type TextInputProps,
  type TextProps,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import Animated, {
  LinearTransition,
  useAnimatedScrollHandler,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';

import { brandDeep, font, makeStyles, radius, space, text, useColors, type Palette } from '@/theme';
import { Appear, CountUp, ease, haptic, LiveDot, PressScale, swap, useBarStyle, useBarTitleStyle } from './motion';

export type IconName = ComponentProps<typeof Ionicons>['name'];

// --- Text ---------------------------------------------------------------------------------------------------------

function styledText(base: TextStyle, color: keyof Palette) {
  return function StyledText({ style, ...rest }: TextProps) {
    const c = useColors();
    return <Text {...rest} style={[base, { color: c[color] as string }, style]} />;
  };
}

export const Display = styledText(text.display, 'text');
export const Title = styledText(text.title, 'text');
export const H2 = styledText(text.h2, 'text');
export const H3 = styledText(text.h3, 'text');
export const Lead = styledText(text.lead, 'text2');
export const Body = styledText(text.body, 'text2');
export const Small = styledText(text.small, 'muted');
/** Small UPPERCASE mono label ("РАЗМЕР"). */
export const Label = styledText(text.label, 'muted');
export const Mono = styledText(text.mono, 'text2');

/** The website's kicker: a green dot with a soft ring, then small mono capitals. */
export function Kicker({ children, style }: { children: string; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  return (
    <View style={[s.kicker, style]}>
      <View style={s.kickerDot} />
      <Label>{children}</Label>
    </View>
  );
}

// --- Layout -------------------------------------------------------------------------------------------------------

/** How far a page has scrolled, for a ScrollBar: pass `scrollY.onScroll` to an Animated scroll view. */
export function useScrollY() {
  const y = useSharedValue(0);
  const onScroll = useAnimatedScrollHandler((e) => {
    y.set(e.contentOffset.y);
  });
  return { y, onScroll };
}

/** A scrolling page on the background colour. */
export function Screen({
  children,
  style,
  scrollY,
}: {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  scrollY?: ReturnType<typeof useScrollY>;
}) {
  const s = useStyles();
  return (
    <Animated.ScrollView
      style={s.screen}
      contentContainerStyle={[s.screenContent, style]}
      keyboardShouldPersistTaps="handled"
      showsVerticalScrollIndicator={false}
      onScroll={scrollY?.onScroll}
      scrollEventThrottle={16}
    >
      {children}
    </Animated.ScrollView>
  );
}

/** The bar at the top of a page, like the website's header: see-through ground and a line under it once the page
 * scrolls; its title comes in once the page's big title has gone. `overlay` lays it over the page. */
export function ScrollBar({
  title,
  y,
  left,
  right,
  overlay,
  titleAfter,
}: {
  title?: ReactNode;
  y: SharedValue<number>;
  left?: ReactNode;
  right?: ReactNode;
  overlay?: boolean;
  titleAfter?: number;
}) {
  const s = useStyles();
  const back = useBarStyle(y);
  const titleIn = useBarTitleStyle(y, titleAfter);
  return (
    <View style={[s.scrollBar, overlay && s.scrollBarOverlay]} pointerEvents="box-none">
      <Animated.View style={[s.scrollBarBack, back]} pointerEvents="none" />
      {left}
      <Animated.View style={[s.flex, typeof title === 'string' && titleIn]} pointerEvents="none">
        {typeof title === 'string' ? (
          <Text style={s.scrollBarTitle} numberOfLines={1}>
            {title}
          </Text>
        ) : (
          title
        )}
      </Animated.View>
      {right}
    </View>
  );
}

/** One clear title per screen, with an optional kicker above it and a short mono line under it. */
export function Header({ kicker, title, sub, right }: { kicker?: string; title: string; sub?: string; right?: ReactNode }) {
  const s = useStyles();
  return (
    <Appear style={s.header}>
      <View style={s.flex}>
        {kicker && <Label style={s.headerKicker}>{kicker}</Label>}
        <Title numberOfLines={2}>{title}</Title>
        {sub && <Mono style={s.sub}>{sub}</Mono>}
      </View>
      {right}
    </Appear>
  );
}

export function Card({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  return <View style={[s.card, style]}>{children}</View>;
}

/** A group under a small mono label, with an optional arrow link on the right. */
export function Section({
  label,
  title,
  action,
  children,
  style,
}: {
  label?: string;
  title?: string;
  action?: { label: string; onPress: () => void };
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
}) {
  const s = useStyles();
  return (
    <View style={[s.section, style]}>
      {(label || title || action) && (
        <View style={s.sectionHead}>
          <View style={s.flex}>
            {label && <Label>{label}</Label>}
            {title && <H2 style={s.sectionTitle}>{title}</H2>}
          </View>
          {action && <ArrowLink label={action.label} onPress={action.onPress} />}
        </View>
      )}
      {children}
    </View>
  );
}

/** "Всички →": a quiet text link whose arrow steps forward when pressed. */
export function ArrowLink({ label, onPress, style }: { label: string; onPress: () => void; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  const reduce = useReducedMotion();
  const x = useSharedValue(0);
  const arrow = useAnimatedStyle(() => ({ transform: [{ translateX: x.value }] }));
  return (
    <Pressable
      onPress={onPress}
      onPressIn={() => !reduce && x.set(withTiming(3, { duration: 200, easing: ease }))}
      onPressOut={() => x.set(withTiming(0, { duration: 200, easing: ease }))}
      hitSlop={10}
      accessibilityRole="button"
      accessibilityLabel={label}
      style={[s.arrowLink, style]}
    >
      <Text style={s.arrowText}>{label}</Text>
      <Animated.Text style={[s.arrowText, arrow]}>→</Animated.Text>
    </Pressable>
  );
}

/** The website's specs: small mono names over big numbers, between thin lines. */
export function Specs({ items, style }: { items: { label: string; value: number | string; suffix?: string }[]; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  return (
    <View style={[s.specs, style]}>
      {items.map((it, i) => (
        <View key={it.label} style={[s.spec, i < items.length - 1 && s.specLine]}>
          <Label numberOfLines={1}>{it.label}</Label>
          {typeof it.value === 'number' ? (
            <CountUp value={it.value} format={(v) => `${Math.round(v)}${it.suffix ?? ''}`} style={s.specValue} />
          ) : (
            <Text style={s.specValue} numberOfLines={1} adjustsFontSizeToFit>
              {it.value}
            </Text>
          )}
        </View>
      ))}
    </View>
  );
}

/** A settings line: label and a short description on the left, a control (or value and chevron) on the right. */
export function Row({
  label,
  description,
  value,
  right,
  onPress,
  danger,
  last,
}: {
  label: string;
  description?: string;
  value?: string;
  right?: ReactNode;
  onPress?: () => void;
  danger?: boolean;
  last?: boolean;
}) {
  const c = useColors();
  const s = useStyles();
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      accessibilityRole={onPress ? 'button' : undefined}
      style={({ pressed }) => [s.row, !last && s.rowLine, pressed && s.rowPressed]}
    >
      <View style={s.flex}>
        <Text style={[s.rowLabel, danger && { color: c.danger }]} numberOfLines={1}>
          {label}
        </Text>
        {description && <Small>{description}</Small>}
      </View>
      {value != null && (
        <Mono style={s.rowValue} numberOfLines={1}>
          {value}
        </Mono>
      )}
      {right}
      {onPress && !right && <Ionicons name="chevron-forward" size={16} color={danger ? c.danger : c.muted} />}
    </Pressable>
  );
}

/** A question-and-answer line like the website's FAQ: a plus that turns into a minus, the answer rising in. */
export function Fold({ title, number, children, last }: { title: string; number?: string; children: ReactNode; last?: boolean }) {
  const s = useStyles();
  const reduce = useReducedMotion();
  const [open, setOpen] = useState(false);
  const turn = useSharedValue(90);
  useEffect(() => {
    turn.set(reduce ? (open ? 0 : 90) : withTiming(open ? 0 : 90, { duration: 300, easing: ease }));
  }, [open, reduce, turn]);
  const bar = useAnimatedStyle(() => ({ transform: [{ rotate: `${turn.value}deg` }] }));
  return (
    <Animated.View layout={LinearTransition.duration(300).easing(ease)} style={[s.fold, !last && s.rowLine]}>
      <Pressable
        onPress={() => {
          haptic();
          setOpen((o) => !o);
        }}
        style={s.foldHead}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
      >
        {number && <Text style={s.foldNumber}>{number}</Text>}
        <Text style={s.foldTitle}>{title}</Text>
        <View style={s.plus}>
          <View style={s.plusBar} />
          <Animated.View style={[s.plusBar, s.plusAbs, bar]} />
        </View>
      </Pressable>
      {open && (
        <Animated.View entering={swap} style={[s.foldBody, number != null && s.foldIndent]}>
          {children}
        </Animated.View>
      )}
    </Animated.View>
  );
}

/** Two to four choices side by side (a theme, a section, a quality); the raised piece slides to the chosen one. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  style,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  style?: StyleProp<ViewStyle>;
}) {
  const s = useStyles();
  const reduce = useReducedMotion();
  const [width, setWidth] = useState(0); // inside the padding
  const at = Math.max(0, options.findIndex((o) => o.value === value));
  const segment = width / options.length;
  const x = useSharedValue(0);
  const placed = useRef(false); // the first placing does not slide in from the left
  useEffect(() => {
    if (!segment) return;
    x.set(placed.current && !reduce ? withTiming(at * segment, { duration: 320, easing: ease }) : at * segment);
    placed.current = true;
  }, [at, segment, reduce, x]);
  const slide = useAnimatedStyle(() => ({ transform: [{ translateX: x.value }] }));
  return (
    <View style={[s.segmented, style]} accessibilityRole="tablist" onLayout={(e) => setWidth(e.nativeEvent.layout.width - 8)}>
      {segment > 0 && <Animated.View style={[s.segmentOn, { width: segment }, slide]} />}
      {options.map((o) => {
        const on = o.value === value;
        return (
          <Pressable
            key={o.value}
            onPress={() => {
              if (!on) haptic();
              onChange(o.value);
            }}
            style={s.segment}
            accessibilityRole="tab"
            accessibilityState={{ selected: on }}
          >
            <Text style={[s.segmentText, on && s.segmentTextOn]} numberOfLines={1}>
              {o.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

// --- Controls -----------------------------------------------------------------------------------------------------

type ButtonProps = {
  title: string;
  sub?: string; // a small line under the title
  onPress?: () => void;
  variant?: 'brand' | 'primary' | 'secondary' | 'danger' | 'ghost'; // brand: the one main action of a screen
  size?: 'md' | 'lg';
  icon?: IconName;
  loading?: boolean;
  disabled?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function Button({ title, sub, onPress, variant = 'primary', size = 'md', icon, loading, disabled, style }: ButtonProps) {
  const c = useColors();
  const s = useStyles();
  const off = disabled || loading;
  const fg = { brand: '#ffffff', primary: c.onPrimary, secondary: c.text, danger: c.danger, ghost: c.text2 }[variant];
  return (
    <PressScale
      onPress={onPress}
      feedback={variant === 'primary' || variant === 'brand'}
      disabled={off}
      accessibilityRole="button"
      accessibilityLabel={sub ? `${title}. ${sub}` : title}
      accessibilityState={{ disabled: !!off, busy: !!loading }}
      style={[s.button, size === 'lg' && s.buttonLg, !!sub && s.buttonTwoLines, s[variant], off && s.off, style]}
    >
      {variant === 'brand' && <LinearGradient colors={brandDeep} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={StyleSheet.absoluteFill} />}
      {loading ? (
        <ActivityIndicator color={fg} size="small" />
      ) : (
        <>
          {icon && <Ionicons name={icon} size={size === 'lg' ? 19 : 17} color={fg} />}
          <View style={s.buttonLines}>
            <Text style={[s.buttonText, size === 'lg' && s.buttonTextLg, { color: fg }]} numberOfLines={1}>
              {title}
            </Text>
            {sub && (
              <Text style={[s.buttonSub, { color: fg }]} numberOfLines={1}>
                {sub}
              </Text>
            )}
          </View>
        </>
      )}
    </PressScale>
  );
}

/** A square glass tool button (the website's viewer tools), e.g. over the 3D view. */
export function IconButton({ icon, onPress, label, active }: { icon: IconName; onPress: () => void; label: string; active?: boolean }) {
  const c = useColors();
  const s = useStyles();
  return (
    <PressScale onPress={onPress} accessibilityRole="button" accessibilityLabel={label} hitSlop={6} style={[s.iconButton, active && s.iconButtonOn]}>
      <Ionicons name={icon} size={18} color={active ? c.bg : c.text2} />
    </PressScale>
  );
}

const BADGES = {
  done: { label: 'Готов', color: 'success', soft: 'successSoft' },
  working: { label: 'Обработка', color: 'accent', soft: 'accentSoft' },
  waiting: { label: 'Не е качен', color: 'warning', soft: 'warningSoft' },
  failed: { label: 'Грешка', color: 'danger', soft: 'dangerSoft' },
} as const;
export type BadgeKind = keyof typeof BADGES;

/** The website's status pill: soft ground, a dot (breathing while it works) and the word. */
export function Badge({ kind, label }: { kind: BadgeKind; label?: string }) {
  const c = useColors();
  const s = useStyles();
  const b = BADGES[kind];
  return (
    <View style={[s.badge, { backgroundColor: c[b.soft] }]}>
      {kind === 'working' ? <LiveDot color={c[b.color]} size={6} /> : <View style={[s.badgeDot, { backgroundColor: c[b.color] }]} />}
      <Text style={[s.badgeText, { color: c[b.color] }]}>{label ?? b.label}</Text>
    </View>
  );
}

/** A text field whose border turns to the accent, with a soft ring, while it is in use. */
export function Field(props: TextInputProps & { invalid?: boolean; ref?: Ref<TextInputInstance> }) {
  const c = useColors();
  const s = useStyles();
  const [focused, setFocused] = useState(false);
  const { invalid, style, onFocus, onBlur, ...rest } = props;
  return (
    <TextInput
      placeholderTextColor={c.muted}
      selectionColor={c.accent}
      cursorColor={c.accent}
      keyboardAppearance={c.scheme}
      {...rest}
      onFocus={(e) => {
        setFocused(true);
        onFocus?.(e);
      }}
      onBlur={(e) => {
        setFocused(false);
        onBlur?.(e);
      }}
      style={[s.field, focused && s.fieldFocused, invalid && s.invalid, style]}
    />
  );
}

export function ErrorText({ children }: { children: ReactNode }) {
  const c = useColors();
  const s = useStyles();
  return (
    <View style={s.error}>
      <Ionicons name="alert-circle-outline" size={17} color={c.danger} />
      <Text style={[text.small, s.flex, s.errorText]}>{children}</Text>
    </View>
  );
}

/** A quiet remark: an info icon and muted text, `warn` in the warning colour. */
export function Note({ children, warn }: { children: ReactNode; warn?: boolean }) {
  const c = useColors();
  const s = useStyles();
  const color = warn ? c.warning : c.muted;
  return (
    <View style={[s.note, warn && { backgroundColor: c.warningSoft, borderColor: 'transparent' }]}>
      <Ionicons name="information-circle-outline" size={17} color={color} />
      <Text style={[text.small, s.flex, { color: warn ? c.warning : c.muted }]}>{children}</Text>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  screen: { flex: 1, backgroundColor: c.bg },
  screenContent: { padding: 18, paddingTop: space.sm, gap: space.xl, paddingBottom: space.xxl * 2 },
  kicker: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  kickerDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: c.success, outlineColor: c.successSoft, outlineWidth: 3, outlineStyle: 'solid' },
  header: { flexDirection: 'row', alignItems: 'flex-end', gap: space.md, paddingTop: space.lg, paddingBottom: space.xs },
  headerKicker: { marginBottom: space.md },
  sub: { marginTop: space.md, color: c.muted, fontSize: 12, letterSpacing: 0.2 },
  card: { backgroundColor: c.surface, borderRadius: radius.md, borderWidth: 1, borderColor: c.line, padding: 20, overflow: 'hidden' },
  scrollBar: { height: 56, flexDirection: 'row', alignItems: 'center', gap: space.md, paddingHorizontal: 18 },
  scrollBarOverlay: { position: 'absolute', left: 0, right: 0, top: 0, zIndex: 10 },
  scrollBarBack: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, backgroundColor: c.glass, borderBottomWidth: 1, borderBottomColor: c.line },
  scrollBarTitle: { fontFamily: font.semibold, fontSize: 16, letterSpacing: -0.3, color: c.text },
  section: { gap: space.lg },
  sectionHead: { flexDirection: 'row', alignItems: 'flex-end', gap: space.md },
  sectionTitle: { marginTop: space.sm },
  arrowLink: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 2 },
  arrowText: { fontFamily: font.medium, fontSize: 14, color: c.text },
  specs: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: c.line },
  spec: { flex: 1, paddingTop: space.lg, paddingRight: space.md, marginRight: space.md, gap: 6 },
  specLine: { borderRightWidth: 1, borderRightColor: c.line },
  specValue: { ...text.number, color: c.text },
  // Full card width (over the card's padding), so a pressed row lights up edge to edge.
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingVertical: 14,
    minHeight: 60,
    marginHorizontal: -20,
    paddingHorizontal: 20,
  },
  rowLabel: { fontFamily: font.medium, fontSize: 15.5, letterSpacing: -0.2, color: c.text },
  rowLine: { borderBottomWidth: 1, borderBottomColor: c.line },
  rowPressed: { backgroundColor: c.surface2 },
  rowValue: { maxWidth: '45%', color: c.muted },
  fold: { overflow: 'hidden' },
  foldHead: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: 18 },
  foldNumber: { fontFamily: font.mono, fontSize: 12, color: c.muted, width: 28 },
  foldTitle: { flex: 1, fontFamily: font.medium, fontSize: 16, letterSpacing: -0.25, color: c.text },
  foldBody: { paddingBottom: 20, paddingRight: 36 },
  foldIndent: { paddingLeft: 40 },
  plus: { width: 13, height: 13, alignItems: 'center', justifyContent: 'center' },
  plusBar: { width: 13, height: 1.5, backgroundColor: c.text2 },
  plusAbs: { position: 'absolute' },
  segmented: { flexDirection: 'row', backgroundColor: c.surface2, borderRadius: 13, padding: 3, borderWidth: 1, borderColor: c.line },
  segment: { flex: 1, minHeight: 38, borderRadius: 10, alignItems: 'center', justifyContent: 'center', paddingHorizontal: space.sm },
  segmentOn: { position: 'absolute', left: 3, top: 3, bottom: 3, borderRadius: 10, backgroundColor: c.surface3 },
  segmentText: { fontFamily: font.medium, fontSize: 14, color: c.muted },
  segmentTextOn: { color: c.text },
  button: {
    minHeight: 46,
    borderRadius: radius.sm,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.sm,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  buttonLg: { minHeight: 54, borderRadius: 14, paddingHorizontal: 22 },
  buttonText: { fontFamily: font.semibold, fontSize: 15, letterSpacing: -0.15 },
  buttonTextLg: { fontSize: 16 },
  buttonLines: { alignItems: 'center', flexShrink: 1 },
  buttonTwoLines: { paddingVertical: 8 },
  buttonSub: { fontFamily: font.regular, fontSize: 12, lineHeight: 16, opacity: 0.7, marginTop: 1 },
  brand: { overflow: 'hidden' },
  primary: { backgroundColor: c.primary },
  secondary: { backgroundColor: c.surface2, borderColor: c.line },
  danger: { backgroundColor: c.dangerSoft },
  ghost: { backgroundColor: 'transparent' },
  off: { opacity: 0.45 },
  iconButton: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: c.glass,
    borderWidth: 1,
    borderColor: c.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconButtonOn: { backgroundColor: c.text, borderColor: c.text },
  badge: { alignSelf: 'flex-start', flexDirection: 'row', alignItems: 'center', gap: 6, borderRadius: 999, paddingHorizontal: 9, paddingVertical: 3 },
  badgeDot: { width: 6, height: 6, borderRadius: 3 },
  badgeText: { fontFamily: font.medium, fontSize: 12 },
  field: {
    minHeight: 50,
    backgroundColor: c.surface2,
    borderWidth: 1,
    borderColor: c.lineStrong,
    borderRadius: radius.sm,
    paddingHorizontal: 13,
    paddingVertical: 11,
    fontSize: 16,
    fontFamily: font.regular,
    color: c.text,
  },
  fieldFocused: { borderColor: c.accent, outlineColor: c.accentSoft, outlineWidth: 3, outlineStyle: 'solid' },
  invalid: { borderColor: c.danger },
  error: { flexDirection: 'row', gap: space.sm, alignItems: 'flex-start', backgroundColor: c.dangerSoft, borderRadius: radius.sm, padding: space.md },
  errorText: { color: c.danger, fontFamily: font.medium },
  note: {
    flexDirection: 'row',
    gap: space.sm,
    alignItems: 'flex-start',
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.line,
    backgroundColor: c.surface2,
    padding: space.md,
  },
}));

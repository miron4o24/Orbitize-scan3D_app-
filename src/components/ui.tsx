import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useRef, useState, type ComponentProps, type ReactNode, type Ref } from 'react';
import {
  ActivityIndicator,
  Pressable,
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
  useAnimatedScrollHandler,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';

import Svg, { Circle, Defs, Path, Pattern, Rect } from 'react-native-svg';

import { font, makeStyles, radius, space, text, useColors, type Palette } from '@/theme';
import { Appear, CountUp, ease, haptic, LiveDot, PressScale, useBarStyle, useBarTitleStyle } from './motion';

export type IconName = ComponentProps<typeof Ionicons>['name'];

const ABSOLUTE = { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0 } as const;

// --- Текст
// ----------------------------------------------------------------------------------------------------------

function styledText(base: TextStyle, color: keyof Palette) {
  return function StyledText({ style, ...rest }: TextProps) {
    const c = useColors();
    return <Text {...rest} style={[base, { color: c[color] as string }, style]} />;
  };
}

export const Title = styledText(text.title, 'text');
export const H2 = styledText(text.h2, 'text');
export const H3 = styledText(text.h3, 'text');
export const Lead = styledText(text.lead, 'text2');
export const Small = styledText(text.small, 'muted');
/** Small UPPERCASE mono label ("РАЗМЕР"). */
export const Label = styledText(text.label, 'muted');
export const Mono = styledText(text.mono, 'text2');

/* Надзаглавие: зелена точка, после малки моноширинни главни букви. */
export function Kicker({ children, style }: { children: string; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  return (
    <View style={[s.kicker, style]}>
      <View style={s.kickerDot} />
      <Label>{children}</Label>
    </View>
  );
}

// --- Подредба
// -------------------------------------------------------------------------------------------------------

/* Докъде е превъртяна страницата, за ScrollBar: подайте `scrollY.onScroll` на Animated ScrollView. */
export function useScrollY() {
  const y = useSharedValue(0);
  const onScroll = useAnimatedScrollHandler((e) => {
    y.set(e.contentOffset.y);
  });
  return { y, onScroll };
}

/* Превъртаща се страница върху цвета на фона. */
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

/**
 * Лентата най-горе на страницата, като заглавната част на сайта: прозрачна основа и линия под нея, щом страницата се
 * превърти; заглавието ѝ се появява, щом голямото заглавие изчезне. `overlay` я поставя върху страницата.
 */
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

/* Едно ясно заглавие на екран, с надзаглавие отгоре и кратък моноширинен ред отдолу по желание. */
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

/* Група под малък моноширинен етикет, с връзка със стрелка вдясно по желание. */
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

/* Спецификациите от сайта: малки моноширинни имена над големи числа, между тънки линии. */
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

/* Ред с настройка: етикет и кратко описание вляво, контрола (или стойност и стрелка) вдясно. */
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

/* Два до четири избора един до друг (тема, раздел, качество) в една рамкирана група; избраният е в цвета за акцент. */
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
  const [width, setWidth] = useState(0); // вътре в отстъпа
  const at = Math.max(0, options.findIndex((o) => o.value === value));
  const segment = width / options.length;
  const x = useSharedValue(0);
  const placed = useRef(false); // първото поставяне не се плъзга отляво
  useEffect(() => {
    if (!segment) return;
    x.set(placed.current && !reduce ? withTiming(at * segment, { duration: 320, easing: ease }) : at * segment);
    placed.current = true;
  }, [at, segment, reduce, x]);
  const slide = useAnimatedStyle(() => ({ transform: [{ translateX: x.value }] }));
  return (
    <View style={[s.segmented, style]} accessibilityRole="tablist" onLayout={(e) => setWidth(e.nativeEvent.layout.width - 2)}>
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

// --- Контроли
// -------------------------------------------------------------------------------------------------------

type ButtonProps = {
  title: string;
  sub?: string; // малък ред под заглавието
  onPress?: () => void;
  variant?: 'brand' | 'primary' | 'secondary' | 'danger' | 'ghost'; // brand: като primary (името остава за извикванията)
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
  const fg = { brand: c.onPrimary, primary: c.onPrimary, secondary: c.text, danger: c.danger, ghost: c.text2 }[variant];
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

/* Квадратен бутон за инструмент, например върху 3D изгледа. */
export function IconButton({ icon, onPress, label, active }: { icon: IconName; onPress: () => void; label: string; active?: boolean }) {
  const c = useColors();
  const s = useStyles();
  return (
    <PressScale onPress={onPress} accessibilityRole="button" accessibilityLabel={label} hitSlop={6} style={[s.iconButton, active && s.iconButtonOn]}>
      <Ionicons name={icon} size={18} color={active ? c.onPrimary : c.text2} />
    </PressScale>
  );
}

const BADGES = {
  done: { label: 'Готов', color: 'success' },
  working: { label: 'Обработка', color: 'accent' },
  waiting: { label: 'Чака', color: 'warning' },
  failed: { label: 'Грешка', color: 'danger' },
} as const;
export type BadgeKind = keyof typeof BADGES;

/* Етикет за състояние: рамка от 1px и думата в цвета на състоянието (никога само цвят); пулсира, докато работи. */
export function Badge({ kind, label }: { kind: BadgeKind; label?: string }) {
  const c = useColors();
  const s = useStyles();
  const b = BADGES[kind];
  return (
    <View style={[s.badge, { borderColor: c[b.color] }]}>
      {kind === 'working' && <LiveDot color={c[b.color]} size={5} />}
      <Text style={[s.badgeText, { color: c[b.color] }]}>{label ?? b.label}</Text>
    </View>
  );
}

/**
 * Ред от списък (сканиране): миниатюра от 50px върху c.tint[index % 3], името и моноширинен ред под него ("ORB-0142 ·
 * 84×84×96"), вдясно етикетът за състояние. `progress` (0..1): лента от 3px под реда, докато се обработва.
 */
export function ListRow({
  title,
  caption,
  index = 0,
  thumb,
  right,
  progress,
  onPress,
}: {
  title: string;
  caption?: string;
  index?: number;
  thumb?: ReactNode;
  right?: ReactNode;
  progress?: number | null;
  onPress?: () => void;
}) {
  const c = useColors();
  const s = useStyles();
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      accessibilityRole={onPress ? 'button' : undefined}
      style={({ pressed }) => [s.listRow, pressed && s.rowPressed]}
    >
      <View style={[s.listThumb, { backgroundColor: c.tint[index % 3] }]}>{thumb}</View>
      <View style={s.flex}>
        <Text style={s.listTitle} numberOfLines={1}>
          {title}
        </Text>
        {caption && (
          <Text style={s.listCaption} numberOfLines={1}>
            {caption}
          </Text>
        )}
        {progress != null && (
          <View style={s.listTrack}>
            <View style={[s.listFill, { width: `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%` }]} />
          </View>
        )}
      </View>
      {right}
    </Pressable>
  );
}

/* Таблица ключ-стойност: редове с линии от 1px, ключът като Label, стойността моноширинна вдясно. */
export function KeyValue({ rows, style }: { rows: { key: string; value: ReactNode }[]; style?: StyleProp<ViewStyle> }) {
  const s = useStyles();
  return (
    <View style={[s.kv, style]}>
      {rows.map((r, i) => (
        <View key={r.key} style={[s.kvRow, i < rows.length - 1 && s.rowLine]}>
          <Label style={s.flex}>{r.key}</Label>
          {typeof r.value === 'string' ? <Text style={s.kvValue}>{r.value}</Text> : r.value}
        </View>
      ))}
    </View>
  );
}

/* Четирите ъглови скоби (L от 10px, 1,5px в цвета за акцент) около област за сканиране или 3D изглед. */
export function Brackets({ color, size = 10, inset = 0 }: { color?: string; size?: number; inset?: number }) {
  const c = useColors();
  const col = color ?? c.accent;
  const at = { position: 'absolute' as const, width: size, height: size, borderColor: col };
  const w = 1.5;
  return (
    <View style={[ABSOLUTE, { margin: inset }]} pointerEvents="none">
      <View style={[at, { left: 0, top: 0, borderLeftWidth: w, borderTopWidth: w }]} />
      <View style={[at, { right: 0, top: 0, borderRightWidth: w, borderTopWidth: w }]} />
      <View style={[at, { left: 0, bottom: 0, borderLeftWidth: w, borderBottomWidth: w }]} />
      <View style={[at, { right: 0, bottom: 0, borderRightWidth: w, borderBottomWidth: w }]} />
    </View>
  );
}

/* Мрежа от точки (стъпка 18px, точки от 1px в c.lineStrong при 50%) зад 3D изгледа и прегледа при обработка. */
export function DotGrid() {
  const c = useColors();
  return (
    <Svg style={ABSOLUTE} pointerEvents="none">
      <Defs>
        <Pattern id="dots" width={18} height={18} patternUnits="userSpaceOnUse">
          <Circle cx={9} cy={9} r={0.75} fill={c.lineStrong} opacity={0.5} />
        </Pattern>
      </Defs>
      <Rect width="100%" height="100%" fill="url(#dots)" />
    </Svg>
  );
}

/* Линия за размер (в c.signal): тънка линия с чертички в краищата и моноширинна стойност в средата. */
export function DimLine({ length, value, vertical }: { length: number; value: string; vertical?: boolean }) {
  const c = useColors();
  const s = useStyles();
  const t = 6;
  const d = vertical ? `M${t},0 L${t},${length} M0,0 L${2 * t},0 M0,${length} L${2 * t},${length}` : `M0,${t} L${length},${t} M0,0 L0,${2 * t} M${length},0 L${length},${2 * t}`;
  return (
    <View style={vertical ? { height: length, width: 2 * t, alignItems: 'center', justifyContent: 'center' } : { width: length, height: 2 * t, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={vertical ? 2 * t : length} height={vertical ? length : 2 * t} style={ABSOLUTE}>
        <Path d={d} stroke={c.signal} strokeWidth={1} />
      </Svg>
      <Text style={s.dimValue}>{value}</Text>
    </View>
  );
}

/* Текстово поле (стойността моноширинна), чиято рамка приема цвета за акцент с пръстен от 1px, докато се използва. */
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

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  screen: { flex: 1, backgroundColor: c.bg },
  screenContent: { padding: 18, paddingTop: space.sm, gap: space.xl, paddingBottom: space.xxl * 2 },
  kicker: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  kickerDot: { width: 6, height: 6, borderRadius: radius.pill, backgroundColor: c.success },
  header: { flexDirection: 'row', alignItems: 'flex-end', gap: space.md, paddingTop: space.lg, paddingBottom: space.xs },
  headerKicker: { marginBottom: space.md },
  sub: { marginTop: space.md, color: c.muted, fontSize: 12, letterSpacing: 0.2 },
  card: { backgroundColor: c.surface, borderRadius: radius.md, borderWidth: 1, borderColor: c.line, padding: 16, overflow: 'hidden' },
  scrollBar: { height: 56, flexDirection: 'row', alignItems: 'center', gap: space.md, paddingHorizontal: 18 },
  scrollBarOverlay: { position: 'absolute', left: 0, right: 0, top: 0, zIndex: 10 },
  scrollBarBack: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, backgroundColor: c.glass, borderBottomWidth: 1, borderBottomColor: c.line },
  scrollBarTitle: { fontFamily: font.semibold, fontSize: 16, color: c.text },
  section: { gap: space.lg },
  sectionHead: { flexDirection: 'row', alignItems: 'flex-end', gap: space.md },
  sectionTitle: { marginTop: space.sm },
  arrowLink: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 2 },
  arrowText: { fontFamily: font.medium, fontSize: 14, color: c.accent },
  specs: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: c.line },
  spec: { flex: 1, paddingTop: space.lg, paddingRight: space.md, marginRight: space.md, gap: 6 },
  specLine: { borderRightWidth: 1, borderRightColor: c.line },
  specValue: { ...text.number, color: c.text },
  // На цялата ширина на картата (над отстъпа ѝ), така че натиснат ред да светне от край до край.
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingVertical: 14,
    minHeight: 60,
    marginHorizontal: -16,
    paddingHorizontal: 16,
  },
  rowLabel: { fontFamily: font.medium, fontSize: 15, color: c.text },
  rowLine: { borderBottomWidth: 1, borderBottomColor: c.line },
  rowPressed: { backgroundColor: c.surface2 },
  rowValue: { maxWidth: '45%', color: c.muted },
  segmented: { flexDirection: 'row', borderRadius: radius.sm, borderWidth: 1, borderColor: c.lineStrong, overflow: 'hidden' },
  segment: { flex: 1, minHeight: 36, alignItems: 'center', justifyContent: 'center', paddingHorizontal: space.sm },
  segmentOn: { position: 'absolute', left: 0, top: 0, bottom: 0, backgroundColor: c.accent },
  segmentText: { fontFamily: font.monoMedium, fontSize: 10.5, letterSpacing: 0.6, textTransform: 'uppercase', color: c.text2 },
  segmentTextOn: { color: c.onPrimary },
  button: {
    minHeight: 44,
    borderRadius: radius.sm,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.sm,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  buttonLg: { minHeight: 50, paddingHorizontal: 20 },
  buttonText: { fontFamily: font.semibold, fontSize: 15 },
  buttonTextLg: { fontSize: 15 },
  buttonLines: { alignItems: 'center', flexShrink: 1 },
  buttonTwoLines: { paddingVertical: 8 },
  buttonSub: { fontFamily: font.regular, fontSize: 12, lineHeight: 16, opacity: 0.7, marginTop: 1 },
  brand: { backgroundColor: c.primary },
  primary: { backgroundColor: c.primary },
  secondary: { backgroundColor: c.surface, borderColor: c.lineStrong },
  danger: { backgroundColor: c.dangerSoft, borderColor: c.danger },
  ghost: { backgroundColor: 'transparent' },
  off: { opacity: 0.45 },
  iconButton: {
    width: 40,
    height: 40,
    borderRadius: radius.sm,
    backgroundColor: c.glass,
    borderWidth: 1,
    borderColor: c.lineStrong,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconButtonOn: { backgroundColor: c.accent, borderColor: c.accent },
  badge: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  badgeText: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 14 },
  listRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.md,
    paddingVertical: 7,
    paddingHorizontal: 10,
  },
  listThumb: { width: 50, height: 50, borderRadius: radius.sm, borderWidth: 1, borderColor: c.line, overflow: 'hidden' },
  listTitle: { fontFamily: font.semibold, fontSize: 15, lineHeight: 20, color: c.text },
  listCaption: { fontFamily: font.mono, fontSize: 11.5, lineHeight: 16, color: c.muted, marginTop: 2 },
  listTrack: { height: 3, marginTop: 6, backgroundColor: c.surface3, overflow: 'hidden' },
  listFill: { height: 3, backgroundColor: c.signal },
  kv: { borderTopWidth: 1, borderTopColor: c.line },
  kvRow: { flexDirection: 'row', alignItems: 'center', gap: space.md, minHeight: 40, paddingVertical: space.sm },
  kvValue: { fontFamily: font.mono, fontSize: 13, color: c.text, textAlign: 'right' },
  dimValue: {
    fontFamily: font.monoMedium,
    fontSize: 11,
    color: c.signal,
    backgroundColor: c.bg,
    paddingHorizontal: 4,
  },
  field: {
    minHeight: 42,
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.lineStrong,
    borderRadius: radius.sm,
    paddingHorizontal: 12,
    paddingVertical: 9,
    fontSize: 13,
    fontFamily: font.mono,
    color: c.text,
  },
  fieldFocused: { borderColor: c.accent, outlineColor: c.accentLine, outlineWidth: 1, outlineStyle: 'solid' },
  invalid: { borderColor: c.danger },
  error: { flexDirection: 'row', gap: space.sm, alignItems: 'flex-start', backgroundColor: c.dangerSoft, borderRadius: radius.sm, padding: space.md },
  errorText: { color: c.danger, fontFamily: font.medium },
}));

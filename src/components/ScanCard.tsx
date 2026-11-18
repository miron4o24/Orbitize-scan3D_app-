import Ionicons from '@expo/vector-icons/Ionicons';
import { Image } from 'expo-image';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useReducedMotion, useSharedValue, withTiming, type SharedValue } from 'react-native-reanimated';
import Svg, { Defs, RadialGradient, Rect, Stop } from 'react-native-svg';

import { authHeaders, thumbUrl } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { badgeOf, isAi, type Entry } from '@/lib/library';
import { font, makeStyles, radius, useColors } from '@/theme';
import { ease, Skeleton } from './motion';
import { Badge, ListRow } from './ui';

/* Меката радиална основа на 3D изгледа, зад изображение или икона. */
export function ViewerGround({ id }: { id: string }) {
  const c = useColors();
  return (
    <Svg style={StyleSheet.absoluteFill} width="100%" height="100%">
      <Defs>
        <RadialGradient id={id} cx="50%" cy="38%" rx="75%" ry="75%">
          <Stop offset="0" stopColor={c.viewer[0]} />
          <Stop offset="1" stopColor={c.viewer[1]} />
        </RadialGradient>
      </Defs>
      <Rect width="100%" height="100%" fill={`url(#${id})`} />
    </Svg>
  );
}

/* Изображението на сканирането от сървъра (рисунка на модела, щом е готов) или икона, докато няма такова. plain: без
   основата на 3D изгледа (миниатюра в ред върху c.tint). fill: запълва родителя с каквато и да е форма, с целия модел
   видим (голямата карта на началния екран). */
export function ScanThumb({ entry, zoom, plain, fill, size = 36 }: { entry: Entry; zoom?: SharedValue<number>; plain?: boolean; fill?: boolean; size?: number }) {
  const c = useColors();
  const s = useStyles();
  const [failed, setFailed] = useState(false);
  const show = entry.thumb && !!entry.jobId && !failed;
  // Друг адрес след завършване: кешът на изображения на телефона не бива да пази кадъра от видеото, показван по време
  // на обработката.
  const uri = `${thumbUrl(entry.jobId ?? '')}${entry.status === 'done' ? '?v=model' : ''}`;
  const working = entry.status !== 'done' && entry.status !== 'failed';
  const scaled = useAnimatedStyle(() => ({ transform: [{ scale: zoom ? zoom.value : 1 }] }));
  return (
    <View style={[s.thumb, (plain || fill) && s.thumbPlain]}>
      {!plain && <ViewerGround id={`g-${entry.key}`} />}
      {show ? (
        <Animated.View style={[StyleSheet.absoluteFill, scaled]}>
          <Image
            source={{ uri, headers: authHeaders() }}
            style={StyleSheet.absoluteFill}
            contentFit={fill ? 'contain' : 'cover'}
            transition={300}
            recyclingKey={entry.key}
            onError={() => setFailed(true)}
          />
        </Animated.View>
      ) : (
        <Ionicons
          name={entry.status === 'failed' ? 'alert-circle-outline' : working ? 'hourglass-outline' : 'cube-outline'}
          size={size}
          color={entry.status === 'failed' ? c.danger : c.lineStrong}
        />
      )}
    </View>
  );
}

/**
 * Сканиране в мрежа: изображение върху основата на 3D изгледа, после заглавие и моноширинна дата. Върху
 * изображението: "AI" при AI модел и състоянието, ако не е готово (обичайният случай не се нуждае от дума). Всички
 * карти са с еднаква височина. При натискане картата хлътва и изображението леко се навежда (ефектът при посочване от
 * сайта).
 */
export function ScanCard({ entry, onPress, onLongPress }: { entry: Entry; onPress: () => void; onLongPress?: () => void }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const shared = !!(entry.server?.share ?? entry.scan?.share);
  const press = useSharedValue(0);
  const zoom = useSharedValue(1);
  const card = useAnimatedStyle(() => ({ transform: [{ scale: 1 - 0.02 * press.value }], borderColor: press.value > 0.5 ? c.lineStrong : c.line }));
  const to = (v: number) => {
    if (reduce) return;
    press.set(withTiming(v, { duration: 220, easing: ease }));
    zoom.set(withTiming(1 + 0.04 * v, { duration: 600, easing: ease }));
  };
  return (
    <Pressable
      onPress={onPress}
      onLongPress={onLongPress}
      onPressIn={() => to(1)}
      onPressOut={() => to(0)}
      accessibilityRole="button"
      accessibilityLabel={entry.title}
      accessibilityHint={onLongPress ? 'Задръжте за изтриване' : undefined}
    >
      <Animated.View style={[s.card, card]}>
        <View>
          <ScanThumb entry={entry} zoom={zoom} />
          {isAi(entry) && (
            <View style={s.aiChip}>
              <Ionicons name="sparkles" size={11} color={c.accent} />
              <Text style={s.aiText}>AI</Text>
            </View>
          )}
          {entry.status !== 'done' && (
            <View style={s.status}>
              <Badge kind={badgeOf(entry.status)} />
            </View>
          )}
        </View>
        <View style={s.meta}>
          <Text style={s.title} numberOfLines={1}>
            {entry.title}
          </Text>
          <View style={s.row}>
            <Text style={s.date}>{shortDate(entry.createdAt)}</Text>
            {shared && <Ionicons name="link-outline" size={13} color={c.muted} />}
          </View>
        </View>
      </Animated.View>
    </Pressable>
  );
}

/* Кратък номер на модела от задачата на сървъра ("ORB-1A2B"), моноширинен в редовете. */
export const modelId = (entry: Entry) => `ORB-${(entry.jobId ?? entry.key).replace(/[^0-9a-z]/gi, '').slice(0, 4).toUpperCase()}`;

/* Размерите в мм, когато са истински: инженерният режим или избраният от потребителя реален размер. */
export function sizeText(entry: Entry): string | null {
  const report = entry.server?.report ?? entry.scan?.report;
  const dims = report?.dimensions_mm;
  if (!dims) return null;
  const longest = report?.ai === 'cad' ? Math.max(...dims) : entry.scan?.longestMm;
  if (!longest) return null;
  const k = longest / Math.max(...dims);
  return dims.map((d) => Math.round(d * k)).join('×');
}

/* Сканиране като ред от списък: миниатюра, име, номер и размер (или дата), вдясно състоянието. */
export function ScanRow({ entry, index, onPress, onLongPress }: { entry: Entry; index: number; onPress: () => void; onLongPress?: () => void }) {
  const caption = [modelId(entry), sizeText(entry) ?? shortDate(entry.createdAt)].join(' · ');
  return (
    <Pressable onLongPress={onLongPress} accessibilityHint={onLongPress ? 'Задръжте за изтриване' : undefined}>
      <ListRow
        title={entry.title}
        caption={isAi(entry) ? `${caption} · AI` : caption}
        index={index}
        thumb={<ScanThumb entry={entry} plain size={22} />}
        right={<Badge kind={badgeOf(entry.status)} />}
        onPress={onPress}
      />
    </Pressable>
  );
}

/* Формата на карта, докато списъкът се зарежда. */
export function ScanCardSkeleton() {
  const s = useStyles();
  return (
    <View style={s.card}>
      <Skeleton style={s.skeletonThumb} />
      <View style={s.meta}>
        <Skeleton style={s.skeletonLine} />
        <Skeleton style={s.skeletonShort} />
      </View>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  card: { backgroundColor: c.surface, borderRadius: radius.md, borderWidth: 1, borderColor: c.line, overflow: 'hidden' },
  thumb: { aspectRatio: 1, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  thumbPlain: { width: '100%', height: '100%', aspectRatio: undefined },
  meta: { paddingTop: 12, paddingHorizontal: 14, paddingBottom: 14, gap: 4, borderTopWidth: 1, borderTopColor: c.line },
  aiChip: {
    position: 'absolute',
    top: 8,
    left: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.accentLine,
    backgroundColor: c.glass,
  },
  aiText: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 15, color: c.text },
  status: { position: 'absolute', left: 8, bottom: 8, borderRadius: radius.pill, backgroundColor: c.glass },
  title: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  date: { fontFamily: font.mono, fontSize: 12, color: c.muted },
  skeletonThumb: { aspectRatio: 1, borderRadius: 0 },
  skeletonLine: { height: 14, width: '75%' },
  skeletonShort: { height: 11, width: '45%' },
}));

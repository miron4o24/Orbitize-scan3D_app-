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
import { Badge } from './ui';

/** The 3D view's soft radial ground, behind a picture or an icon. */
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

/** The scan's picture from the server (a drawing of the model once it is done), or an icon while there is none. */
export function ScanThumb({ entry, zoom }: { entry: Entry; zoom?: SharedValue<number> }) {
  const c = useColors();
  const s = useStyles();
  const [failed, setFailed] = useState(false);
  const show = entry.thumb && !!entry.jobId && !failed;
  // Another URL once done: the phone's image cache must not keep the video frame shown while processing.
  const uri = `${thumbUrl(entry.jobId ?? '')}${entry.status === 'done' ? '?v=model' : ''}`;
  const working = entry.status !== 'done' && entry.status !== 'failed';
  const scaled = useAnimatedStyle(() => ({ transform: [{ scale: zoom ? zoom.value : 1 }] }));
  return (
    <View style={s.thumb}>
      <ViewerGround id={`g-${entry.key}`} />
      {show ? (
        <Animated.View style={[StyleSheet.absoluteFill, scaled]}>
          <Image
            source={{ uri, headers: authHeaders() }}
            style={StyleSheet.absoluteFill}
            contentFit="cover"
            transition={300}
            recyclingKey={entry.key}
            onError={() => setFailed(true)}
          />
        </Animated.View>
      ) : (
        <Ionicons
          name={entry.status === 'failed' ? 'alert-circle-outline' : working ? 'hourglass-outline' : 'cube-outline'}
          size={36}
          color={entry.status === 'failed' ? c.danger : c.lineStrong}
        />
      )}
    </View>
  );
}

/** A scan in a grid: picture on the 3D view's ground, then title and mono date. Over the picture: "AI" on an AI model,
 * and the status unless it is done (the usual case needs no word). All cards are the same height. Pressed, the card
 * dips and the picture leans in a little (the website's hover). */
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

/** The shape of a card while the list loads. */
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
  meta: { paddingTop: 12, paddingHorizontal: 14, paddingBottom: 14, gap: 4, borderTopWidth: 1, borderTopColor: c.line },
  aiChip: { position: 'absolute', top: 8, left: 8, flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 3, borderRadius: radius.pill, backgroundColor: c.glass },
  aiText: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 15, color: c.text },
  status: { position: 'absolute', left: 8, bottom: 8, borderRadius: radius.pill, backgroundColor: c.glass },
  title: { fontFamily: font.semibold, fontSize: 15, letterSpacing: -0.2, color: c.text },
  row: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  date: { fontFamily: font.mono, fontSize: 12, color: c.muted },
  skeletonThumb: { aspectRatio: 1, borderRadius: 0 },
  skeletonLine: { height: 14, width: '75%' },
  skeletonShort: { height: 11, width: '45%' },
}));

import Ionicons from '@expo/vector-icons/Ionicons';
import { Image } from 'expo-image';
import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withSequence,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';

import { authHeaders, authorPhotoUrl, postThumbUrl, type Post } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { font, makeStyles, radius, useColors } from '@/theme';
import { Avatar } from './Avatar';
import { ease, easeOut, haptic, Skeleton } from './motion';
import { ViewerGround } from './ScanCard';

/* Авторът на публикация в общността: снимка (или инициали), име и ниво. */
export function AuthorLine({ post, size = 18, date }: { post: Post; size?: number; date?: boolean }) {
  const s = useStyles();
  const a = post.author;
  return (
    <View style={s.author}>
      <Avatar user={a} size={size} photo={a.has_avatar ? authorPhotoUrl(a.id) : undefined} />
      <Text style={[s.authorName, size > 20 && s.authorBig]} numberOfLines={1}>
        {post.mine ? 'Вие' : a.name}
      </Text>
      {a.level != null && a.level > 1 && (
        <Text style={s.level} accessibilityLabel={`Ниво ${a.level}`}>
          Н{a.level}
        </Text>
      )}
      {date && <Text style={s.date}>· {shortDate(post.published * 1000)}</Text>}
    </View>
  );
}

/* Сърцето с броя. Подскача, когато се запълни. */
export function LikeButton({ liked, likes, onPress, big }: { liked: boolean; likes: number; onPress: () => void; big?: boolean }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const pop = useSharedValue(1);
  const was = useRef(liked);
  useEffect(() => {
    if (liked && !was.current && !reduce)
      pop.set(withSequence(withTiming(1.35, { duration: 140, easing: easeOut }), withTiming(1, { duration: 260, easing: ease })));
    was.current = liked;
  }, [liked, reduce, pop]);
  const style = useAnimatedStyle(() => ({ transform: [{ scale: pop.value }] }));
  return (
    <Pressable
      onPress={() => {
        haptic();
        onPress();
      }}
      hitSlop={8}
      style={[s.like, big && s.likeBig, liked && s.likeOn]}
      accessibilityRole="button"
      accessibilityState={{ selected: liked }}
      accessibilityLabel={liked ? `Харесано, ${likes}. Махни харесването` : `Харесай, ${likes}`}
    >
      <Animated.View style={style}>
        <Ionicons name={liked ? 'heart' : 'heart-outline'} size={big ? 20 : 15} color={liked ? c.danger : c.text} />
      </Animated.View>
      <Text style={[s.likeCount, big && s.likeCountBig]}>{likes}</Text>
    </Pressable>
  );
}

function PostThumb({ post, zoom }: { post: Post; zoom: SharedValue<number> }) {
  const c = useColors();
  const s = useStyles();
  const [failed, setFailed] = useState(false);
  const scaled = useAnimatedStyle(() => ({ transform: [{ scale: zoom.value }] }));
  return (
    <View style={s.thumb}>
      <ViewerGround id={`p-${post.id}`} />
      {failed ? (
        <Ionicons name="cube-outline" size={36} color={c.lineStrong} />
      ) : (
        <Animated.View style={[StyleSheet.absoluteFill, scaled]}>
          <Image
            source={{ uri: postThumbUrl(post.id), headers: authHeaders() }}
            style={StyleSheet.absoluteFill}
            contentFit="cover"
            transition={300}
            recyclingKey={post.id}
            onError={() => setFailed(true)}
          />
        </Animated.View>
      )}
    </View>
  );
}

/**
 * Публикация в мрежата на общността: изображение на модела с харесванията, после името и авторът. При натискане
 * хлътва и изображението се навежда, както при картите на сканирания.
 */
export function PostCard({ post, onPress }: { post: Post; onPress: () => void }) {
  const c = useColors();
  const s = useStyles();
  const reduce = useReducedMotion();
  const press = useSharedValue(0);
  const zoom = useSharedValue(1);
  const card = useAnimatedStyle(() => ({ transform: [{ scale: 1 - 0.02 * press.value }], borderColor: press.value > 0.5 ? c.lineStrong : c.line }));
  const to = (v: number) => {
    if (reduce) return;
    press.set(withTiming(v, { duration: 220, easing: ease }));
    zoom.set(withTiming(1 + 0.04 * v, { duration: 600, easing: ease }));
  };
  return (
    <Pressable onPress={onPress} onPressIn={() => to(1)} onPressOut={() => to(0)} accessibilityRole="button" accessibilityLabel={`${post.title}, от ${post.author.name}`}>
      <Animated.View style={[s.card, card]}>
        <PostThumb post={post} zoom={zoom} />
        <View style={s.count} pointerEvents="none">
          <Ionicons name={post.liked ? 'heart' : 'heart-outline'} size={12} color={post.liked ? c.danger : c.text} />
          <Text style={s.countText}>{post.likes}</Text>
        </View>
        <View style={s.meta}>
          <Text style={s.title} numberOfLines={1}>
            {post.title}
          </Text>
          <AuthorLine post={post} />
        </View>
      </Animated.View>
    </Pressable>
  );
}

export function PostCardSkeleton() {
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
  count: {
    position: 'absolute',
    top: 10,
    right: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: c.glass,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  countText: { fontFamily: font.monoMedium, fontSize: 11, color: c.text },
  meta: { paddingTop: 12, paddingHorizontal: 14, paddingBottom: 14, gap: 8, borderTopWidth: 1, borderTopColor: c.line },
  title: { fontFamily: font.semibold, fontSize: 15, letterSpacing: -0.2, color: c.text },
  author: { flexDirection: 'row', alignItems: 'center', gap: 7, minWidth: 0 },
  authorName: { flexShrink: 1, fontFamily: font.regular, fontSize: 12.5, color: c.muted },
  authorBig: { fontFamily: font.medium, fontSize: 14.5, color: c.text },
  date: { fontFamily: font.mono, fontSize: 12, color: c.muted },
  level: {
    fontFamily: font.monoMedium,
    fontSize: 10,
    color: c.accent,
    backgroundColor: c.accentSoft,
    borderRadius: 999,
    paddingHorizontal: 6,
    paddingVertical: 1,
    overflow: 'hidden',
  },
  like: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: c.lineStrong,
    borderRadius: 999,
    paddingHorizontal: 12,
    height: 34,
  },
  likeBig: { height: 44, paddingHorizontal: 16, gap: 8 },
  likeOn: { borderColor: c.danger },
  likeCount: { fontFamily: font.monoMedium, fontSize: 13, color: c.text },
  likeCountBig: { fontSize: 15 },
  skeletonThumb: { aspectRatio: 1, borderRadius: 0 },
  skeletonLine: { height: 14, width: '75%' },
  skeletonShort: { height: 11, width: '45%' },
}));

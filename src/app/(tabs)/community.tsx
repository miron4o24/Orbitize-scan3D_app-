import { router, useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { ActivityIndicator, RefreshControl, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Appear } from '@/components/motion';
import { PostCard, PostCardSkeleton } from '@/components/PostCard';
import { Stage } from '@/components/Stage';
import { Button, ErrorText, Lead, ScrollBar, Segmented, Small, Title, useScrollY } from '@/components/ui';
import { communityFeed, errorText, type Post, type PostSort } from '@/lib/api';
import { clearSession, useSession } from '@/lib/session';
import { makeStyles, space, useColors } from '@/theme';

export default function Community() {
  const session = useSession();
  const c = useColors();
  const s = useStyles();
  const scrollY = useScrollY();
  const account = session?.kind === 'account';
  const [sort, setSort] = useState<PostSort>('new');
  const [posts, setPosts] = useState<Post[] | null>(null); // null: the first page is loading
  const [next, setNext] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const asked = useRef(0); // only the latest request's answer is shown

  const load = useCallback(
    async (pulled?: boolean) => {
      if (!account) return;
      const ticket = ++asked.current;
      if (pulled) setRefreshing(true);
      try {
        const page = await communityFeed(sort);
        if (ticket !== asked.current) return;
        setPosts(page.posts);
        setNext(page.next);
        setError(null);
      } catch (e) {
        if (ticket === asked.current) {
          setError(errorText(e));
          setPosts((p) => p ?? []);
        }
      } finally {
        if (ticket === asked.current) setRefreshing(false);
      }
    },
    [account, sort],
  );

  // Fresh on every visit (likes change on a post's page, and others post meanwhile) and on another sort.
  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  async function loadMore() {
    if (next === null || more || !posts) return;
    setMore(true);
    const ticket = asked.current;
    try {
      const page = await communityFeed(sort, next);
      if (ticket !== asked.current) return;
      const have = new Set(posts.map((p) => p.id));
      setPosts([...posts, ...page.posts.filter((p) => !have.has(p.id))]);
      setNext(page.next);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setMore(false);
    }
  }

  function changeSort(v: PostSort) {
    if (v === sort) return;
    setPosts(null);
    setSort(v); // a new load, so the focus effect asks again
  }

  const open = (p: Post) => router.push({ pathname: '/community/[id]', params: { id: p.id } });

  const header = (
    <View style={s.header}>
      <Appear>
        <Title style={s.title}>Общност</Title>
      </Appear>
      {account && (
        <Appear index={3} style={s.sort}>
          <Segmented<PostSort>
            value={sort}
            onChange={changeSort}
            options={[
              { value: 'new', label: 'Нови' },
              { value: 'top', label: 'Най-харесвани' },
            ]}
          />
        </Appear>
      )}
      {error && <ErrorText>{error}</ErrorText>}
    </View>
  );

  if (!account)
    return (
      <SafeAreaView style={s.fill} edges={['top']}>
        <ScrollBar title="Общност" y={scrollY.y} overlay titleAfter={70} />
        <Animated.ScrollView style={s.fill} contentContainerStyle={s.list} onScroll={scrollY.onScroll} scrollEventThrottle={16}>
          {header}
          <Appear index={3}>
            <Stage height={240} />
          </Appear>
          <Appear index={4} style={s.guest}>
            <Small>Влезте, за да виждате и споделяте модели.</Small>
            <Button variant="brand" size="lg" title="Вход или регистрация" icon="log-in-outline" onPress={() => clearSession()} />
          </Appear>
        </Animated.ScrollView>
      </SafeAreaView>
    );

  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <ScrollBar title="Общност" y={scrollY.y} overlay titleAfter={70} />
      <Animated.FlatList
        style={s.fill}
        onScroll={scrollY.onScroll}
        scrollEventThrottle={16}
        data={posts ?? []}
        keyExtractor={(p) => p.id}
        numColumns={2}
        columnWrapperStyle={s.columns}
        contentContainerStyle={s.list}
        showsVerticalScrollIndicator={false}
        onEndReached={loadMore}
        onEndReachedThreshold={0.6}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} colors={[c.text]} progressBackgroundColor={c.surface} tintColor={c.muted} />
        }
        ListHeaderComponent={header}
        ListFooterComponent={more ? <ActivityIndicator style={s.more} color={c.muted} /> : undefined}
        ListEmptyComponent={
          posts === null ? (
            <View style={s.skeletons}>
              {[0, 1].map((row) => (
                <View key={row} style={s.columns}>
                  <View style={s.cell}>
                    <PostCardSkeleton />
                  </View>
                  <View style={s.cell}>
                    <PostCardSkeleton />
                  </View>
                </View>
              ))}
            </View>
          ) : error ? undefined : (
            <View style={s.empty}>
              <Appear index={2}>
                <Stage height={240} />
              </Appear>
              <Appear index={3} style={s.emptyText}>
                <Title style={s.emptyTitle}>Бъдете първи.</Title>
                <Lead>Отворете свой модел и го публикувайте.</Lead>
              </Appear>
              <Appear index={4}>
                <Button variant="brand" size="lg" title="Моите сканирания" icon="cube-outline" onPress={() => router.navigate('/scans')} />
              </Appear>
            </View>
          )
        }
        renderItem={({ item, index }) => (
          <Appear index={Math.min(index, 8) + 2} style={[s.cell, (posts?.length ?? 0) % 2 === 1 && index === (posts?.length ?? 0) - 1 && s.lastOdd]}>
            <PostCard post={item} onPress={() => open(item)} />
          </Appear>
        )}
      />
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  list: { paddingHorizontal: 18, paddingBottom: 48, gap: 14, flexGrow: 1 },
  columns: { flexDirection: 'row', gap: 14 },
  cell: { flex: 1 },
  lastOdd: { maxWidth: '50%', paddingRight: 7 },
  header: { gap: space.md, paddingTop: space.xl, paddingBottom: space.lg },
  title: { fontSize: 38, lineHeight: 42, letterSpacing: -1.7, marginTop: space.sm },
  sort: { marginTop: space.sm },
  guest: { gap: space.lg, marginTop: space.xl },
  skeletons: { gap: 14 },
  more: { paddingVertical: space.lg },
  empty: { gap: space.xl, paddingTop: space.sm },
  emptyText: { gap: space.md },
  emptyTitle: { fontSize: 28, lineHeight: 32 },
}));

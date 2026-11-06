import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, RefreshControl, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Appear } from '@/components/motion';
import { ScanCard, ScanCardSkeleton } from '@/components/ScanCard';
import { Stage } from '@/components/Stage';
import { Button, ErrorText, ScrollBar, Segmented, Title, useScrollY } from '@/components/ui';
import { errorText } from '@/lib/api';
import { deleteEntry, isAi, openEntry, useLibrary, type Entry } from '@/lib/library';
import { makeStyles, space, useColors } from '@/theme';

type Group = 'ai' | 'other';

export default function MyScans() {
  const { entries, refresh, loading, error } = useLibrary();
  const c = useColors();
  const s = useStyles();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const scrollY = useScrollY();

  // AI моделите отделно от останалите, щом има и от двата вида; отваря се на последно създадения вид.
  const [chosen, setChosen] = useState<Group | null>(null);
  const ai = entries.filter(isAi);
  const other = entries.filter((e) => !isAi(e));
  const both = ai.length > 0 && other.length > 0;
  const group: Group = chosen ?? (entries[0] && isAi(entries[0]) ? 'ai' : 'other');
  const shown = !both ? entries : group === 'ai' ? ai : other;

  function confirmDelete(entry: Entry) {
    Alert.alert(`Изтриване на „${entry.title}“?`, 'Това не може да се върне.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Изтрий',
        style: 'destructive',
        onPress: () =>
          deleteEntry(entry).then(
            () => {
              setDeleteError(null);
              refresh();
            },
            (e) => setDeleteError(`Не се изтри: ${errorText(e)}`),
          ),
      },
    ]);
  }

  const skeleton = loading && entries.length === 0;
  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <ScrollBar title="Сканирания" y={scrollY.y} overlay titleAfter={70} />
      <Animated.FlatList
        style={s.fill}
        onScroll={scrollY.onScroll}
        scrollEventThrottle={16}
        data={shown}
        keyExtractor={(e) => e.key}
        numColumns={2}
        columnWrapperStyle={s.columns}
        contentContainerStyle={s.list}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl refreshing={loading && !skeleton} onRefresh={refresh} colors={[c.text]} progressBackgroundColor={c.surface} tintColor={c.muted} />}
        ListHeaderComponent={
          <View style={s.header}>
            <Appear>
              <Title style={s.title}>Сканирания</Title>
            </Appear>
            {both && (
              <Appear index={1} style={s.groups}>
                <Segmented<Group>
                  value={group}
                  onChange={setChosen}
                  options={[
                    { value: 'ai', label: `AI модели · ${ai.length}` },
                    { value: 'other', label: `Legacy · ${other.length}` },
                  ]}
                />
              </Appear>
            )}
            {error && <ErrorText>{error}</ErrorText>}
            {deleteError && <ErrorText>{deleteError}</ErrorText>}
          </View>
        }
        ListEmptyComponent={
          skeleton ? (
            <View style={s.skeletons}>
              {[0, 1].map((row) => (
                <View key={row} style={s.columns}>
                  <View style={s.cell}>
                    <ScanCardSkeleton />
                  </View>
                  <View style={s.cell}>
                    <ScanCardSkeleton />
                  </View>
                </View>
              ))}
            </View>
          ) : (
            <View style={s.empty}>
              <Appear index={2}>
                <Stage height={240} />
              </Appear>
              <Appear index={3}>
                <Title style={s.emptyTitle}>Още няма сканирания.</Title>
              </Appear>
              <Appear index={4}>
                <Button variant="brand" size="lg" title="Start 1:1 photo test" icon="construct-outline" onPress={() => router.push('/part')} />
              </Appear>
            </View>
          )
        }
        renderItem={({ item, index }: { item: Entry; index: number }) => (
          <Appear index={index + 2} style={[s.cell, shown.length % 2 === 1 && index === shown.length - 1 && s.lastOdd]}>
            <ScanCard entry={item} onPress={() => openEntry(item)} onLongPress={() => confirmDelete(item)} />
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
  groups: { marginTop: space.sm },
  skeletons: { gap: 14 },
  empty: { gap: space.xl, paddingTop: space.sm },
  emptyTitle: { fontSize: 28, lineHeight: 32 },
}));

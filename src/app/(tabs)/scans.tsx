import Ionicons from '@expo/vector-icons/Ionicons';
import { useState } from 'react';
import { Pressable, RefreshControl, ScrollView, Text, useWindowDimensions, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Appear } from '@/components/motion';
import { ScanCard, ScanCardSkeleton } from '@/components/ScanCard';
import { Stage } from '@/components/Stage';
import { Button, ErrorText, Field, Label, ScrollBar, Title, useScrollY } from '@/components/ui';
import { errorText } from '@/lib/api';
import { badgeOf, deleteEntry, isAi, openEntry, useLibrary, type Entry } from '@/lib/library';
import { startScan } from '@/lib/startScan';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { showDialog } from '@/lib/dialog';

// Сканиранията като мрежа от карти с рисунката на всеки модел, с търсене по име и бързи филтри (вид и състояние).
// Докосване отваря модела, задържане го изтрива.

type Filter = 'all' | 'ai' | 'cad' | 'working' | 'failed';

const isCad = (e: Entry) => (e.server?.report ?? e.scan?.report)?.ai === 'cad';
const MATCH: Record<Filter, (e: Entry) => boolean> = {
  all: () => true,
  ai: (e) => isAi(e) && !isCad(e),
  cad: isCad,
  working: (e) => ['working', 'waiting'].includes(badgeOf(e.status)),
  failed: (e) => badgeOf(e.status) === 'failed',
};
const LABEL: Record<Filter, string> = { all: 'Всички', ai: 'AI модели', cad: 'Инженерни', working: 'В работа', failed: 'Неуспешни' };

export default function MyScans() {
  const { entries, refresh, loading, error } = useLibrary();
  const c = useColors();
  const s = useStyles();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const scrollY = useScrollY();
  // Точно половината от реда: сама карта в последния ред не се разтяга на цялата ширина.
  const cellWidth = (useWindowDimensions().width - 2 * 16 - space.sm) / 2;

  const q = query.trim().toLowerCase();
  const shown = entries.filter((e) => MATCH[filter](e) && (!q || e.title.toLowerCase().includes(q)));
  // Филтрите с поне един модел (и "Всички" винаги), с броя си.
  const filters = (Object.keys(MATCH) as Filter[]).filter((f) => f === 'all' || f === filter || entries.some(MATCH[f]));

  function confirmDelete(entry: Entry) {
    showDialog(`Изтриване на „${entry.title}“?`, 'Това не може да се върне.', [
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
        data={skeleton ? [] : shown}
        keyExtractor={(e) => e.key}
        numColumns={2}
        columnWrapperStyle={s.row}
        contentContainerStyle={s.list}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        refreshControl={<RefreshControl refreshing={loading && !skeleton} onRefresh={refresh} colors={[c.text]} progressBackgroundColor={c.surface} tintColor={c.muted} />}
        ListHeaderComponent={
          <View style={s.header}>
            <Appear>
              <View style={s.titleRow}>
                <Title style={s.title}>Сканирания</Title>
                {!skeleton && <Text style={s.count}>{entries.length}</Text>}
              </View>
            </Appear>
            {entries.length > 3 && (
              <Appear index={1}>
                <View style={s.search}>
                  <Ionicons name="search" size={16} color={c.muted} />
                  <Field
                    style={s.searchField}
                    value={query}
                    onChangeText={setQuery}
                    placeholder="Търсене по име"
                    returnKeyType="search"
                    accessibilityLabel="Търсене по име"
                  />
                  {!!query && (
                    <Pressable onPress={() => setQuery('')} hitSlop={8} accessibilityRole="button" accessibilityLabel="Изчисти търсенето">
                      <Ionicons name="close-circle" size={18} color={c.muted} />
                    </Pressable>
                  )}
                </View>
              </Appear>
            )}
            {filters.length > 1 && (
              <Appear index={2}>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.chips} style={s.chipsOut}>
                  {filters.map((f) => {
                    const on = f === filter;
                    return (
                      <Pressable
                        key={f}
                        onPress={() => setFilter(f)}
                        style={[s.chip, on && s.chipOn]}
                        accessibilityRole="button"
                        accessibilityState={{ selected: on }}
                      >
                        <Text style={[s.chipText, on && s.chipTextOn]}>{LABEL[f]}</Text>
                        <Text style={[s.chipCount, on && s.chipTextOn]}>{entries.filter(MATCH[f]).length}</Text>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              </Appear>
            )}
            {error && <ErrorText>{error}</ErrorText>}
            {deleteError && <ErrorText>{deleteError}</ErrorText>}
            {!skeleton && shown.length > 0 && <Label>Задръжте модел, за да го изтриете</Label>}
          </View>
        }
        ListEmptyComponent={
          skeleton ? (
            <View style={s.skeletons}>
              {[0, 1].map((row) => (
                <View key={row} style={s.row}>
                  <View style={{ width: cellWidth }}>
                    <ScanCardSkeleton />
                  </View>
                  <View style={{ width: cellWidth }}>
                    <ScanCardSkeleton />
                  </View>
                </View>
              ))}
            </View>
          ) : entries.length > 0 ? (
            <View style={s.none}>
              <Ionicons name="search-outline" size={28} color={c.lineStrong} />
              <Text style={s.noneText}>Нищо не отговаря на търсенето.</Text>
              <Button
                title="Покажи всички"
                variant="secondary"
                onPress={() => {
                  setQuery('');
                  setFilter('all');
                }}
              />
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
                <Button variant="brand" size="lg" title="Първо сканиране" icon="videocam-outline" onPress={startScan} />
              </Appear>
            </View>
          )
        }
        renderItem={({ item, index }: { item: Entry; index: number }) => (
          <Appear index={Math.min(index, 8) + 2} style={{ width: cellWidth }}>
            <ScanCard entry={item} onPress={() => openEntry(item)} onLongPress={() => confirmDelete(item)} />
          </Appear>
        )}
      />
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  list: { paddingHorizontal: 16, paddingBottom: 48, gap: space.sm, flexGrow: 1 },
  row: { flexDirection: 'row', gap: space.sm },
  header: { gap: space.md, paddingTop: space.xl, paddingBottom: space.sm },
  titleRow: { flexDirection: 'row', alignItems: 'baseline', gap: space.sm, marginTop: space.sm },
  title: { marginBottom: space.xs },
  count: { fontFamily: font.monoMedium, fontSize: 15, color: c.muted },
  search: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
    paddingHorizontal: space.md,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.sm,
    backgroundColor: c.surface,
  },
  searchField: { flex: 1, borderWidth: 0, backgroundColor: 'transparent', paddingHorizontal: 0 },
  chipsOut: { marginHorizontal: -16 },
  chips: { paddingHorizontal: 16, gap: space.sm },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.pill,
    backgroundColor: c.surface,
  },
  chipOn: { borderColor: c.accent, backgroundColor: c.accentSoft },
  chipText: { fontFamily: font.medium, fontSize: 13, color: c.text2 },
  chipCount: { fontFamily: font.monoMedium, fontSize: 12, color: c.muted },
  chipTextOn: { color: c.accent },
  skeletons: { gap: space.sm },
  none: { alignItems: 'center', gap: space.md, paddingTop: space.xxl },
  noneText: { fontFamily: font.regular, fontSize: 14, color: c.text2 },
  empty: { gap: space.xl, paddingTop: space.sm },
  emptyTitle: { fontSize: 22, lineHeight: 28 },
}));

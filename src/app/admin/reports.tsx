import { Image } from 'expo-image';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, Text, View } from 'react-native';

import { Chip } from '@/components/admin';
import { Button, Card, ErrorText, Small } from '@/components/ui';
import { adminKeepPost, adminRemovePost, adminReports, authHeaders, errorText, thumbUrl, type AdminReport } from '@/lib/api';
import { showDialog } from '@/lib/dialog';
import { font, makeStyles, radius, space, useColors } from '@/theme';

/* Докладваните модели от общността: остави (сигналите се изчистват) или махни от общността (моделът остава на автора). */
export default function AdminReports() {
  const c = useColors();
  const s = useStyles();
  const [list, setList] = useState<AdminReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      setList(await adminReports());
      setError(null);
    } catch (e) {
      setError(errorText(e));
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  async function act(id: string, work: () => Promise<unknown>) {
    setBusy(id);
    try {
      await work();
      await load();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={async () => {
            setRefreshing(true);
            await load();
            setRefreshing(false);
          }}
          tintColor={c.muted}
          colors={[c.accent]}
        />
      }
    >
      <Small>Модел с 3 сигнала се скрива от общността, докато не решите какво да стане с него.</Small>
      {error && <ErrorText>{error}</ErrorText>}
      {!list ? (
        <ActivityIndicator color={c.muted} style={s.loading} />
      ) : list.length === 0 ? (
        <Small style={s.empty}>Няма сигнали. Общността е чиста.</Small>
      ) : (
        list.map((post) => (
          <Card key={post.id} style={s.card}>
            <View style={s.head}>
              <View style={[s.thumb, { backgroundColor: c.tint[0] }]}>
                <Image source={{ uri: thumbUrl(post.id), headers: authHeaders() }} style={s.image} contentFit="cover" transition={200} />
              </View>
              <View style={s.flex}>
                <Text style={s.title} numberOfLines={2}>
                  {post.title || 'Без име'}
                </Text>
                <View style={s.chips}>
                  <Chip label={post.reports === 1 ? '1 сигнал' : `${post.reports} сигнала`} tone="warning" icon="flag-outline" />
                  {post.hidden && <Chip label="Скрит" tone="danger" icon="eye-off-outline" />}
                </View>
              </View>
            </View>
            <View style={s.reasons}>
              {post.reasons.map((r, i) => (
                <Text key={i} style={s.reason}>
                  „{r || 'без причина'}“
                </Text>
              ))}
            </View>
            <View style={s.buttons}>
              <Button title="Остави" variant="secondary" icon="checkmark" onPress={() => act(post.id, () => adminKeepPost(post.id))} loading={busy === post.id} style={s.flex} />
              <Button
                title="Махни"
                variant="danger"
               
                icon="trash-outline"
                onPress={() =>
                  showDialog('Махане от общността?', 'Моделът остава при автора, но никой друг не го вижда.', [
                    { text: 'Отказ', style: 'cancel' },
                    { text: 'Махни', style: 'destructive', onPress: () => act(post.id, () => adminRemovePost(post.id)) },
                  ])
                }
                disabled={busy === post.id}
                style={s.flex}
              />
            </View>
            <Button title="Профилът на автора" variant="ghost" onPress={() => router.push(`/admin/user/${post.author}`)} />
          </Card>
        ))
      )}
    </ScrollView>
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.lg },
  loading: { marginTop: space.xxl },
  empty: { textAlign: 'center', marginTop: space.xl },
  card: { gap: space.md },
  head: { flexDirection: 'row', gap: space.md, alignItems: 'center' },
  thumb: { width: 64, height: 64, borderRadius: radius.sm, borderWidth: 1, borderColor: c.line, overflow: 'hidden' },
  image: { width: '100%', height: '100%' },
  flex: { flex: 1, gap: 6 },
  title: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  chips: { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
  reasons: { gap: 4 },
  reason: { fontFamily: font.regular, fontSize: 13, lineHeight: 18, color: c.text2 },
  buttons: { flexDirection: 'row', gap: space.sm },
}));

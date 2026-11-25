import { router, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, RefreshControl, View } from 'react-native';

import { UserLine } from '@/components/admin';
import { Button, ErrorText, Field, Label, Segmented, Small } from '@/components/ui';
import { adminUsers, errorText, type AdminFilter, type AdminUser } from '@/lib/api';
import { makeStyles, radius, space, useColors } from '@/theme';

const FILTERS: { value: AdminFilter; label: string }[] = [
  { value: 'all', label: 'Всички' },
  { value: 'unverified', label: 'Чакат' },
  { value: 'blocked', label: 'Блок.' },
  { value: 'admins', label: 'Админи' },
];

/* Всички профили: търсене по име или имейл и филтър по статус; натискане отваря профила. */
export default function AdminUsers() {
  const c = useColors();
  const s = useStyles();
  const params = useLocalSearchParams<{ show?: AdminFilter }>();
  const [show, setShow] = useState<AdminFilter>(params.show ?? 'all');
  const [q, setQ] = useState('');
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [total, setTotal] = useState(0);
  const [next, setNext] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const asked = useRef(0); // само последната заявка се показва (бързо писане в търсенето)

  const load = useCallback(async (query: string, filter: AdminFilter) => {
    const mine = ++asked.current;
    try {
      const r = await adminUsers(query, filter);
      if (mine !== asked.current) return;
      setUsers(r.users);
      setTotal(r.total);
      setNext(r.next);
      setError(null);
    } catch (e) {
      if (mine === asked.current) setError(errorText(e));
    }
  }, []);

  // Търсенето тръгва, след като човек спре да пише за момент.
  useEffect(() => {
    const t = setTimeout(() => load(q, show), q ? 300 : 0);
    return () => clearTimeout(t);
  }, [q, show, load]);

  // Обратно от профил (блокиран, изтрит): списъкът се опреснява.
  useFocusEffect(
    useCallback(() => {
      if (users) load(q, show);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []),
  );

  async function more() {
    if (next == null || busy) return;
    setBusy(true);
    try {
      const r = await adminUsers(q, show, next);
      setUsers((u) => [...(u ?? []), ...r.users]);
      setNext(r.next);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <FlatList
      style={s.screen}
      contentContainerStyle={s.content}
      data={users ?? []}
      keyExtractor={(u) => u.id}
      keyboardShouldPersistTaps="handled"
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={async () => {
            setRefreshing(true);
            await load(q, show);
            setRefreshing(false);
          }}
          tintColor={c.muted}
          colors={[c.accent]}
        />
      }
      ListHeaderComponent={
        <View style={s.head}>
          <Field
            value={q}
            onChangeText={setQ}
            placeholder="Търсене по име или имейл"
            accessibilityLabel="Търсене по име или имейл"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="search"
          />
          <Segmented<AdminFilter> value={show} onChange={setShow} options={FILTERS} />
          {error && <ErrorText>{error}</ErrorText>}
          {users && <Label>{total === 1 ? '1 профил' : `${total} профила`}</Label>}
        </View>
      }
      renderItem={({ item, index }) => (
        <View style={[s.item, index === 0 && s.first, index === (users?.length ?? 0) - 1 && s.last]}>
          <UserLine user={item} last={index === (users?.length ?? 0) - 1} onPress={() => router.push(`/admin/user/${item.id}`)} />
        </View>
      )}
      ListEmptyComponent={users ? <Small style={s.empty}>Няма профили тук.</Small> : <ActivityIndicator color={c.muted} style={s.empty} />}
      ListFooterComponent={next != null ? <Button title="Още" variant="secondary" onPress={more} loading={busy} style={s.more} /> : undefined}
    />
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2 },
  head: { gap: space.md, marginBottom: space.md },
  item: { backgroundColor: c.surface, borderLeftWidth: 1, borderRightWidth: 1, borderColor: c.line, paddingHorizontal: space.md },
  first: { borderTopWidth: 1, borderTopLeftRadius: radius.md, borderTopRightRadius: radius.md },
  last: { borderBottomWidth: 1, borderBottomLeftRadius: radius.md, borderBottomRightRadius: radius.md },
  empty: { marginTop: space.xl, textAlign: 'center' },
  more: { marginTop: space.lg },
}));

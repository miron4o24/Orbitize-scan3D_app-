import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, View } from 'react-native';

import { Chip, Tiles } from '@/components/admin';
import { Appear } from '@/components/motion';
import { Card, ErrorText, Header, Row, Section, Small } from '@/components/ui';
import { adminOverview, errorText, type AdminOverview } from '@/lib/api';
import { restartTour } from '@/lib/tour';
import { makeStyles, space, useColors } from '@/theme';

/* Администриране: числата за сървъра и профилите и входовете към списъците. Само за собственика и администраторите
   (сървърът отказва на всички други, app/admin.py). */
export default function AdminHome() {
  const c = useColors();
  const s = useStyles();
  const [data, setData] = useState<AdminOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await adminOverview());
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

  async function refresh() {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }

  const users = (show?: string) => router.push(show ? { pathname: '/admin/users', params: { show } } : '/admin/users');

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} tintColor={c.muted} colors={[c.accent]} />}
    >
      <Header kicker="Само за администратора" title="Администриране" sub={data ? `Сървър ${data.server.version}` : undefined} />
      {error && <ErrorText>{error}</ErrorText>}
      {!data && !error && <ActivityIndicator color={c.muted} style={s.loading} />}

      {data && (
        <>
          <Appear index={1}>
            <Section label="Профили" action={{ label: 'Всички', onPress: () => users() }}>
              <Tiles
                items={[
                  { label: 'профила общо', value: data.users.total, onPress: () => users() },
                  { label: 'нови за 7 дни', value: data.users.new_week, tone: 'accent' },
                  { label: 'активни за 7 дни', value: data.users.active_week, tone: 'success' },
                  { label: 'чакат потвърждение', value: data.users.unverified, tone: data.users.unverified ? 'warning' : undefined, onPress: () => users('unverified') },
                  { label: 'блокирани', value: data.users.blocked, tone: data.users.blocked ? 'danger' : undefined, onPress: () => users('blocked') },
                  { label: 'администратори', value: data.users.admins, onPress: () => users('admins') },
                ]}
              />
            </Section>
          </Appear>

          <Appear index={2}>
            <Section label="Обработка" action={{ label: 'Опашка', onPress: () => router.push('/admin/jobs') }}>
              <Tiles
                items={[
                  { label: 'обработват се сега', value: data.jobs.working, tone: data.jobs.working ? 'warning' : undefined, onPress: () => router.push('/admin/jobs') },
                  { label: 'чакат на опашката', value: data.jobs.queued, onPress: () => router.push('/admin/jobs') },
                  { label: 'неуспешни за 7 дни', value: data.jobs.failed_week, tone: data.jobs.failed_week ? 'danger' : undefined, onPress: () => router.push({ pathname: '/admin/jobs', params: { show: 'failed' } }) },
                  { label: 'готови модела', value: data.jobs.done, tone: 'success' },
                ]}
              />
            </Section>
          </Appear>

          <Appear index={3}>
            <Section label="Общност и сървър">
              <Card style={s.list}>
                <Row
                  label="Сигнали в общността"
                  description={`${data.community.published} публикувани модела`}
                  value={String(data.community.reported)}
                  onPress={() => router.push('/admin/reports')}
                />
                <Row
                  label="Потвърждение по имейл"
                  description={data.server.mail ? 'Новите профили получават код' : 'Изключено: профилите не се потвърждават'}
                  right={<Chip label={data.server.mail ? 'Включено' : 'Изключено'} tone={data.server.mail ? 'success' : 'warning'} />}
                  onPress={() => router.push('/admin/mail')}
                />
                <Row label="Свободно място" value={`${data.server.disk_free_gb} от ${data.server.disk_total_gb} GB`} />
                <Row label="База данни" value={data.server.database} last />
              </Card>
            </Section>
          </Appear>

          {data.me.owner && (
            <Appear index={4}>
              <Section label="Собственик">
                <Card style={s.list}>
                  <Row
                    label="Покажи обиколката отново"
                    description="Обиколката за първи път, за проба"
                    onPress={() => {
                      router.navigate('/');
                      setTimeout(restartTour, 600);
                    }}
                    last
                  />
                </Card>
              </Section>
            </Appear>
          )}
          <View style={s.note}>
            <Small>Промените се записват веднага. Собственикът се задава само на сървъра (OWNER_EMAILS).</Small>
          </View>
        </>
      )}
    </ScrollView>
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.xl },
  loading: { marginTop: space.xxl },
  list: { paddingVertical: 0 },
  note: { paddingHorizontal: space.xs },
}));

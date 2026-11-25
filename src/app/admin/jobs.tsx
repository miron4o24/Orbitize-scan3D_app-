import { router, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, Modal, Pressable, RefreshControl, ScrollView, Text, View } from 'react-native';

import { Chip, STATE_TEXT, ago, methodText, stateTone } from '@/components/admin';
import { Button, Card, ErrorText, H2, Segmented, Small } from '@/components/ui';
import { adminDeleteJob, adminJobs, errorText, getLog, type AdminJob } from '@/lib/api';
import { duration } from '@/lib/dates';
import { showDialog } from '@/lib/dialog';
import { font, makeStyles, radius, space, useColors } from '@/theme';

type Show = 'active' | 'failed' | 'all';

/* Опашката на сървъра: какво се обработва и чака сега, какво не успя; дневник, спиране и профилът на автора. */
export default function AdminJobs() {
  const c = useColors();
  const s = useStyles();
  const params = useLocalSearchParams<{ show?: Show }>();
  const [show, setShow] = useState<Show>(params.show ?? 'active');
  const [list, setList] = useState<AdminJob[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [log, setLog] = useState<{ title: string; text: string } | null>(null);

  const load = useCallback(async () => {
    try {
      setList(await adminJobs(show));
      setError(null);
    } catch (e) {
      setError(errorText(e));
    }
  }, [show]);

  useFocusEffect(
    useCallback(() => {
      load();
      if (show !== 'active') return;
      const t = setInterval(load, 10_000); // докато се гледа опашката, тя се опреснява сама
      return () => clearInterval(t);
    }, [load, show]),
  );

  function pick(next: Show) {
    setList(null); // новият списък, не старият, докато се зарежда
    setShow(next);
  }

  async function run(what: string, work: () => Promise<void>) {
    setBusy(what);
    try {
      await work();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  function options(job: AdminJob) {
    const working = job.state !== 'done' && job.state !== 'failed' && job.state !== 'cancelled';
    showDialog(job.title || 'Сканиране', `${methodText(job.method)} · ${STATE_TEXT[job.state]}${job.user ? ` · ${job.user.name}` : ''}`, [
      { text: 'Дневник на обработката', onPress: () => run(job.id, async () => setLog({ title: job.title || 'Сканиране', text: (await getLog(job.id, 120)) || 'Дневникът е празен.' })) },
      ...(job.user ? [{ text: 'Профилът на автора', onPress: () => router.push(`/admin/user/${job.user!.id}`) }] : []),
      ...(working
        ? [
            {
              text: 'Спри и изтрий',
              style: 'destructive' as const,
              onPress: () =>
                showDialog('Спиране на обработката?', 'Сканирането се изтрива. Авторът ще трябва да го направи отново.', [
                  { text: 'Отказ', style: 'cancel' },
                  { text: 'Спри', style: 'destructive', onPress: () => run(job.id, async () => (await adminDeleteJob(job.id), await load())) },
                ]),
            },
          ]
        : []),
      { text: 'Отказ', style: 'cancel' },
    ]);
  }

  return (
    <>
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
        <Segmented<Show>
          value={show}
          onChange={pick}
          options={[
            { value: 'active', label: 'Сега' },
            { value: 'failed', label: 'Неуспешни' },
            { value: 'all', label: 'Всички' },
          ]}
        />
        {error && <ErrorText>{error}</ErrorText>}
        {!list ? (
          <ActivityIndicator color={c.muted} style={s.loading} />
        ) : list.length === 0 ? (
          <Small style={s.empty}>{show === 'active' ? 'Опашката е празна: нищо не се обработва.' : 'Няма такива.'}</Small>
        ) : (
          <Card style={s.list}>
            {list.map((job, i) => (
              <Pressable
                key={job.id}
                onPress={() => options(job)}
                accessibilityRole="button"
                accessibilityLabel={job.title || 'Сканиране'}
                style={({ pressed }) => [s.row, i < list.length - 1 && s.line, pressed && s.pressed]}
              >
                <View style={s.flex}>
                  <Text style={s.title} numberOfLines={1}>
                    {job.title || 'Без име'}
                    {job.user ? <Text style={s.who}>{`  ${job.user.name}`}</Text> : null}
                  </Text>
                  <Text style={s.meta} numberOfLines={1}>
                    {[methodText(job.method), ago(job.created), job.step, job.seconds ? duration(job.seconds) : null].filter(Boolean).join(' · ')}
                  </Text>
                  {job.state === 'failed' && job.error ? (
                    <Text style={s.error} numberOfLines={2}>
                      {job.error}
                    </Text>
                  ) : null}
                </View>
                {busy === job.id ? <ActivityIndicator size="small" color={c.muted} /> : <Chip label={STATE_TEXT[job.state]} tone={stateTone(job.state)} />}
              </Pressable>
            ))}
          </Card>
        )}
      </ScrollView>

      <Modal visible={!!log} transparent animationType="fade" onRequestClose={() => setLog(null)}>
        <View style={s.backdrop}>
          <Card style={s.dialog}>
            <H2 numberOfLines={1}>{log?.title}</H2>
            <ScrollView style={s.logBox} contentContainerStyle={s.logContent}>
              <Text style={s.log} selectable>
                {log?.text}
              </Text>
            </ScrollView>
            <Button title="Затвори" variant="secondary" onPress={() => setLog(null)} />
          </Card>
        </View>
      </Modal>
    </>
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.lg },
  loading: { marginTop: space.xxl },
  empty: { textAlign: 'center', marginTop: space.xl },
  list: { paddingVertical: 0 },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  line: { borderBottomWidth: 1, borderBottomColor: c.line },
  pressed: { opacity: 0.7 },
  flex: { flex: 1, gap: 3 },
  title: { fontFamily: font.semibold, fontSize: 14.5, color: c.text },
  who: { fontFamily: font.regular, fontSize: 13, color: c.muted },
  meta: { fontFamily: font.mono, fontSize: 11.5, color: c.muted },
  error: { fontFamily: font.regular, fontSize: 12.5, lineHeight: 17, color: c.danger },
  backdrop: { flex: 1, backgroundColor: c.backdrop, justifyContent: 'center', padding: space.xl },
  dialog: { gap: space.md, borderRadius: radius.md, maxHeight: '80%' },
  logBox: { borderWidth: 1, borderColor: c.line, borderRadius: radius.sm, backgroundColor: c.bg },
  logContent: { padding: space.md },
  log: { fontFamily: font.mono, fontSize: 11, lineHeight: 16, color: c.text2 },
}));

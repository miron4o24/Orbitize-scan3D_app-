import Ionicons from '@expo/vector-icons/Ionicons';
import { Image } from 'expo-image';
import { router, useFocusEffect, useLocalSearchParams, useNavigation } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Modal, Pressable, ScrollView, Text, View } from 'react-native';

import { Chip, UserChips, ago, methodText, STATE_TEXT, stateTone } from '@/components/admin';
import { Avatar } from '@/components/Avatar';
import { Appear } from '@/components/motion';
import { Button, Card, ErrorText, Field, H2, KeyValue, Mono, Row, Section, Small, Title } from '@/components/ui';
import {
  adminAvatarUrl,
  adminChangeUser,
  adminDeleteJob,
  adminDeleteUser,
  adminSignOutUser,
  adminUser,
  authHeaders,
  errorText,
  getLog,
  thumbUrl,
  type AdminScan,
  type AdminUserDetail,
} from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { showDialog } from '@/lib/dialog';
import { useSession } from '@/lib/session';
import { font, makeStyles, radius, space, useColors } from '@/theme';

/* Един профил за администратора: данни, статус (блокиране, администратор, потвърждение), изход от всички телефони,
   изтриване и сканиранията му с дневника на обработката. */
export default function AdminUserScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const c = useColors();
  const s = useStyles();
  const nav = useNavigation();
  const session = useSession();
  const iAmOwner = session?.kind === 'account' && !!session.user.owner;
  const [user, setUser] = useState<AdminUserDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [log, setLog] = useState<{ title: string; text: string } | null>(null);
  const [renaming, setRenaming] = useState(false);

  const load = useCallback(async () => {
    try {
      setUser(await adminUser(id));
      setError(null);
    } catch (e) {
      setError(errorText(e));
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  useEffect(() => {
    if (user) nav.setOptions({ title: user.name });
  }, [nav, user]);

  async function run(what: string, work: () => Promise<void>) {
    setBusy(what);
    setError(null);
    try {
      await work();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  if (!user)
    return (
      <View style={s.center}>{error ? <ErrorText>{error}</ErrorText> : <ActivityIndicator color={c.muted} />}</View>
    );

  // Какво не може да се променя оттук (сървърът проверява същото, app/admin.py).
  const locked = user.is_me
    ? 'Това е вашият профил.'
    : user.owner
      ? 'Собственикът се задава само на сървъра.'
      : user.admin && !iAmOwner
        ? 'Само собственикът променя администраторите.'
        : null;
  const spinner = (what: string) => (busy === what ? <ActivityIndicator size="small" color={c.muted} /> : undefined);
  const change = (what: string, patch: Parameters<typeof adminChangeUser>[1]) =>
    run(what, async () => setUser(await adminChangeUser(user.id, patch)));

  function askBlock() {
    if (user!.blocked) return change('block', { blocked: false });
    showDialog('Блокиране на профила?', `${user!.name} излиза от всички телефони и не може да влезе, докато не го отблокирате. Моделите му остават.`, [
      { text: 'Отказ', style: 'cancel' },
      { text: 'Блокирай', style: 'destructive', onPress: () => change('block', { blocked: true }) },
    ]);
  }

  function askAdmin() {
    const on = !user!.admin;
    showDialog(
      on ? 'Права на администратор?' : 'Без права на администратор?',
      on ? `${user!.name} ще вижда „Администриране“: профилите, опашката и сигналите, и ще може да блокира обикновени профили.` : `${user!.name} отново е обикновен профил.`,
      [
        { text: 'Отказ', style: 'cancel' },
        { text: on ? 'Направи администратор' : 'Махни правата', style: on ? 'default' : 'destructive', onPress: () => change('admin', { admin: on }) },
      ],
    );
  }

  function askSignOut() {
    showDialog('Изход от всички телефони?', 'Профилът остава, но трябва да влезе отново.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Изход навсякъде',
        onPress: () =>
          run('signout', async () => {
            await adminSignOutUser(user!.id);
            await load();
          }),
      },
    ]);
  }

  function askDelete() {
    showDialog('Изтриване на профила?', `Изтриват се ${user!.name} и всички ${user!.scans} сканирания, завинаги.`, [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Изтрий профила',
        style: 'destructive',
        onPress: () =>
          run('delete', async () => {
            await adminDeleteUser(user!.id);
            router.back();
          }),
      },
    ]);
  }

  function scanOptions(scan: AdminScan) {
    showDialog(scan.title || 'Сканиране', `${methodText(scan.method)} · ${STATE_TEXT[scan.state]} · ${shortDate(scan.created * 1000)}`, [
      {
        text: 'Дневник на обработката',
        onPress: () =>
          run(`log-${scan.id}`, async () => setLog({ title: scan.title || 'Сканиране', text: (await getLog(scan.id, 120)) || 'Дневникът е празен.' })),
      },
      {
        text: 'Изтрий сканирането',
        style: 'destructive',
        onPress: () =>
          showDialog('Изтриване на сканирането?', 'Изтрива се завинаги, и от общността.', [
            { text: 'Отказ', style: 'cancel' },
            {
              text: 'Изтрий',
              style: 'destructive',
              onPress: () =>
                run(`scan-${scan.id}`, async () => {
                  await adminDeleteJob(scan.id);
                  await load();
                }),
            },
          ]),
      },
      { text: 'Отказ', style: 'cancel' },
    ]);
  }

  return (
    <>
      <ScrollView style={s.screen} contentContainerStyle={s.content}>
        <Appear style={s.head}>
          <Avatar user={user} size={64} photo={user.has_avatar ? adminAvatarUrl(user.id) : undefined} />
          <View style={s.flex}>
            <Title numberOfLines={2}>{user.name}</Title>
            <Mono numberOfLines={1} style={s.email} selectable>
              {user.email}
            </Mono>
            <UserChips user={user} />
          </View>
        </Appear>

        {error && <ErrorText>{error}</ErrorText>}

        <Appear index={1}>
          <KeyValue
            rows={[
              { key: 'Създаден', value: shortDate(user.created * 1000) },
              { key: 'Последно активен', value: ago(user.last_seen) },
              { key: 'Вход', value: user.google ? 'Google' : 'Имейл и парола' },
              { key: 'Влезли телефони', value: String(user.sessions) },
              { key: 'Сканирания', value: String(user.scans) },
            ]}
          />
        </Appear>

        <Appear index={2}>
          <Section label="Статус">
            {locked && <Small>{locked}</Small>}
            <Card style={s.list}>
              <Row
                label={user.blocked ? 'Отблокирай' : 'Блокирай'}
                description={user.blocked ? 'Може пак да влиза' : 'Без вход, излиза навсякъде'}
                onPress={locked ? undefined : askBlock}
                right={spinner('block')}
                danger={!user.blocked && !locked}
              />
              {!user.verified && (
                <Row label="Потвърди имейла" description="Без кода от писмото" onPress={locked ? undefined : () => change('verify', { verified: true })} right={spinner('verify')} />
              )}
              <Row label="Смени името" value={user.name} onPress={locked ? undefined : () => setRenaming(true)} right={spinner('name')} />
              {iAmOwner && !user.owner && !user.is_me && (
                <Row
                  label={user.admin ? 'Махни правата на администратор' : 'Направи администратор'}
                  onPress={askAdmin}
                  right={spinner('admin')}
                />
              )}
              <Row label="Изход от всички телефони" onPress={locked || !user.sessions ? undefined : askSignOut} right={spinner('signout')} />
              <Row label="Изтрий профила" onPress={locked ? undefined : askDelete} right={spinner('delete')} danger={!locked} last />
            </Card>
          </Section>
        </Appear>

        <Appear index={3}>
          <Section label={`Сканирания · ${user.scans_list.length}`}>
            {user.scans_list.length === 0 ? (
              <Small>Още няма сканирания.</Small>
            ) : (
              <Card style={s.list}>
                {user.scans_list.map((scan, i) => (
                  <Pressable
                    key={scan.id}
                    onPress={() => scanOptions(scan)}
                    accessibilityRole="button"
                    accessibilityLabel={scan.title || 'Сканиране'}
                    style={({ pressed }) => [s.scan, i < user.scans_list.length - 1 && s.line, pressed && s.pressed]}
                  >
                    <View style={[s.thumb, { backgroundColor: c.tint[i % 3] }]}>
                      {scan.thumb ? (
                        <Image source={{ uri: thumbUrl(scan.id), headers: authHeaders() }} style={s.thumbImage} contentFit="cover" transition={200} />
                      ) : (
                        <Ionicons name="cube-outline" size={20} color={c.lineStrong} />
                      )}
                    </View>
                    <View style={s.flex}>
                      <Text style={s.scanTitle} numberOfLines={1}>
                        {scan.title || 'Без име'}
                      </Text>
                      <Text style={s.scanMeta} numberOfLines={1}>
                        {methodText(scan.method)} · {shortDate(scan.created * 1000)}
                        {scan.published ? ' · в общността' : ''}
                      </Text>
                    </View>
                    {busy === `log-${scan.id}` || busy === `scan-${scan.id}` ? (
                      <ActivityIndicator size="small" color={c.muted} />
                    ) : (
                      <Chip label={STATE_TEXT[scan.state]} tone={stateTone(scan.state)} />
                    )}
                  </Pressable>
                ))}
              </Card>
            )}
          </Section>
        </Appear>
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

      <NameDialog
        visible={renaming}
        name={user.name}
        saving={busy === 'name'}
        onClose={() => setRenaming(false)}
        onSave={(name) => {
          change('name', { name });
          setRenaming(false);
        }}
      />
    </>
  );
}

function NameDialog(props: { visible: boolean; name: string; saving: boolean; onClose: () => void; onSave: (n: string) => void }) {
  const s = useStyles();
  const [value, setValue] = useState(props.name);
  const valid = value.trim().length > 0 && value.trim().length <= 100;
  return (
    <Modal visible={props.visible} transparent animationType="fade" onRequestClose={props.onClose} onShow={() => setValue(props.name)}>
      <View style={s.backdrop}>
        <Card style={s.dialog}>
          <H2>Име на профила</H2>
          <Field value={value} onChangeText={setValue} autoFocus maxLength={100} placeholder="Име" accessibilityLabel="Име" invalid={!valid} />
          <View style={s.buttons}>
            <Button title="Отказ" variant="secondary" onPress={props.onClose} style={s.flex} />
            <Button title="Запази" onPress={() => props.onSave(value.trim())} disabled={!valid} loading={props.saving} style={s.flex} />
          </View>
        </Card>
      </View>
    </Modal>
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.xl },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: c.bg },
  flex: { flex: 1, gap: 2 },
  head: { flexDirection: 'row', alignItems: 'center', gap: space.lg },
  email: { color: c.muted, marginTop: 2 },
  list: { paddingVertical: 0 },
  line: { borderBottomWidth: 1, borderBottomColor: c.line },
  pressed: { opacity: 0.7 },
  scan: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  thumb: { width: 44, height: 44, borderRadius: radius.sm, borderWidth: 1, borderColor: c.line, alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  thumbImage: { width: '100%', height: '100%' },
  scanTitle: { fontFamily: font.semibold, fontSize: 14.5, color: c.text },
  scanMeta: { fontFamily: font.mono, fontSize: 11.5, color: c.muted },
  backdrop: { flex: 1, backgroundColor: c.backdrop, justifyContent: 'center', padding: space.xl },
  dialog: { gap: space.md, borderRadius: radius.md, maxHeight: '80%' },
  logBox: { borderWidth: 1, borderColor: c.line, borderRadius: radius.sm, backgroundColor: c.bg },
  logContent: { padding: space.md },
  log: { fontFamily: font.mono, fontSize: 11, lineHeight: 16, color: c.text2 },
  buttons: { flexDirection: 'row', gap: space.sm },
}));

import Ionicons from '@expo/vector-icons/Ionicons';
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ActivityIndicator, Alert, Modal, View } from 'react-native';
import type { SharedValue } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Avatar } from '@/components/Avatar';
import { Appear, PressScale } from '@/components/motion';
import { Badges, Challenges, LevelCard } from '@/components/Progress';
import {
  Button,
  Card,
  ErrorText,
  Field,
  H2,
  Header,
  Mono,
  Row,
  Screen,
  ScrollBar,
  Section,
  Segmented,
  Small,
  Specs,
  Title,
  useScrollY,
} from '@/components/ui';
import { deleteAccount, errorText, removeAvatar, renameMe, uploadAvatar } from '@/lib/api';
import { refreshProfile, signOut } from '@/lib/auth';
import { checkProgress, useProgress } from '@/lib/progress';
import { deleteScan, listScans } from '@/lib/scans';
import { avatarChanged, clearSession, updateProfile, useSession, type Profile } from '@/lib/session';
import { makeStyles, radius, setScheme, space, useColors, useScheme, type Scheme } from '@/theme';

const AVATAR_SIDE = 512;

export default function ProfileScreen() {
  const session = useSession();
  if (session?.kind !== 'account') return <Guest />;
  return <Account user={session.user} />;
}

function TopBar({ title, y }: { title: string; y: SharedValue<number> }) {
  return <ScrollBar title={title} y={y} titleAfter={110} />;
}

function ThemeRow({ last }: { last?: boolean }) {
  const scheme = useScheme();
  const s = useStyles();
  return (
    <Row
      label="Тема"
      last={last}
      right={
        <Segmented<Scheme>
          style={s.themeSwitch}
          value={scheme}
          onChange={setScheme}
          options={[
            { value: 'dark', label: 'Тъмна' },
            { value: 'light', label: 'Светла' },
          ]}
        />
      }
    />
  );
}

function Guest() {
  const s = useStyles();
  const scrollY = useScrollY();
  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <TopBar title="Профил" y={scrollY.y} />
      <Screen scrollY={scrollY}>
        <Header title="Не сте влезли" />
        <Appear index={1} style={s.guest}>
          <Small>Влезте, за да пазите моделите си на всеки телефон.</Small>
          {/* Екранът за вход се връща, щом отметката за гост изчезне. */}
          <Button title="Вход или регистрация" icon="log-in-outline" onPress={() => clearSession()} />
        </Appear>
        <Appear index={2}>
          <Section label="Настройки">
            <Card style={s.list}>
              <ThemeRow last />
            </Card>
          </Section>
        </Appear>
      </Screen>
    </SafeAreaView>
  );
}

function Account({ user }: { user: Profile }) {
  const c = useColors();
  const s = useStyles();
  const [busy, setBusy] = useState<string | null>(null); // какво работи в момента, за индикатора му
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const scrollY = useScrollY();

  const progress = useProgress();

  useFocusEffect(
    useCallback(() => {
      refreshProfile().catch(() => {}); // нова статистика; без връзка остава последната
      checkProgress(true);
    }, []),
  );

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

  async function pickPhoto() {
    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    if (picked.canceled || !picked.assets?.length) return;
    const a = picked.assets[0];
    await run('photo', async () => {
      const side = Math.min(AVATAR_SIDE, a.width, a.height);
      const context = ImageManipulator.manipulate(a.uri).resize(a.width <= a.height ? { width: side } : { height: side });
      const image = await context.renderAsync();
      const out = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.85 });
      const file = new File(out.uri);
      try {
        updateProfile(await uploadAvatar(file));
        avatarChanged();
      } finally {
        if (file.exists) file.delete();
      }
    });
  }

  function photoOptions() {
    const options: { text: string; style?: 'cancel' | 'destructive'; onPress?: () => void }[] = [
      { text: 'Избери снимка', onPress: pickPhoto },
    ];
    if (user.has_avatar)
      options.push({
        text: user.google_picture ? 'Използвай снимката от Google' : 'Премахни снимката',
        style: 'destructive',
        onPress: () =>
          run('photo', async () => {
            updateProfile(await removeAvatar());
            avatarChanged();
          }),
      });
    options.push({ text: 'Отказ', style: 'cancel' });
    Alert.alert('Снимка на профила', undefined, options);
  }

  function confirmSignOut() {
    Alert.alert('Изход?', 'Сканиранията остават в профила.', [
      { text: 'Отказ', style: 'cancel' },
      { text: 'Изход', style: 'destructive', onPress: () => run('signout', signOut) },
    ]);
  }

  function confirmDelete() {
    Alert.alert(
      'Изтриване на профила?',
      `Изтриват се профилът и всички ${user.stats.scans} сканирания, завинаги.`,
      [
        { text: 'Отказ', style: 'cancel' },
        {
          text: 'Изтрий профила',
          style: 'destructive',
          onPress: () =>
            run('delete', async () => {
              await deleteAccount();
              for (const sc of listScans()) if (sc.userId === user.id) deleteScan(sc.id);
              await signOut();
            }),
        },
      ],
    );
  }

  const stats: [number, string][] = [
    [user.stats.scans, 'Сканирания'],
    user.stats.published != null ? [user.stats.published, 'В общността'] : [user.stats.shared, 'Споделени'],
  ];
  const spinner = (what: string) => (busy === what ? <ActivityIndicator size="small" color={c.muted} /> : undefined);
  // Ограниченията на безплатния план; при Pro няма какво да се покаже.
  const limits = (
    [
      ['stress', 'Тестове за здравина'],
      ['ar', 'Виж в стаята'],
    ] as const
  ).filter(([kind]) => user.limits?.[kind]?.limit != null);

  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <TopBar title={user.name} y={scrollY.y} />
      <Screen scrollY={scrollY}>
        <View style={s.head}>
          <Appear>
            <PressScale onPress={photoOptions} accessibilityRole="button" accessibilityLabel="Смяна на снимката" style={s.avatar}>
              <Avatar user={user} size={84} />
              <View style={s.camera}>
                {busy === 'photo' ? <ActivityIndicator size="small" color={c.text} /> : <Ionicons name="camera-outline" size={14} color={c.text} />}
              </View>
            </PressScale>
          </Appear>
          <Appear index={1}>
            <Title numberOfLines={2} style={s.name}>
              {user.name}
            </Title>
            <Mono numberOfLines={1} style={s.email}>
              {user.email}
            </Mono>
          </Appear>
        </View>

        <Appear index={3}>
          <Specs items={stats.map(([value, label]) => ({ label, value }))} />
        </Appear>

        {error && <ErrorText>{error}</ErrorText>}

        {progress && (
          <>
            <Appear index={4}>
              <LevelCard progress={progress} />
            </Appear>
            <Appear index={4}>
              <Section label="Седмични предизвикателства">
                <Challenges progress={progress} />
              </Section>
            </Appear>
            <Appear index={5}>
              <Section label={`Значки · ${progress.badges.filter((b) => b.unlocked).length} от ${progress.badges.length}`}>
                <Badges progress={progress} />
              </Section>
            </Appear>
          </>
        )}

        {user.limits && (
          <Appear index={2}>
            <Section label="План">
              <Card style={s.list}>
                <Row
                  label={user.pro ? 'Pro' : 'Безплатен'}
                  description={user.pro ? undefined : 'С Pro: промени с AI и без ограничения'}
                  onPress={() => router.push('/pro')}
                  last={limits.length === 0}
                />
                {limits.map(([kind, label], i) => {
                  const l = user.limits![kind];
                  return <Row key={kind} label={label} value={`${Math.min(l.used, l.limit!)} от ${l.limit}`} last={i === limits.length - 1} />;
                })}
              </Card>
            </Section>
          </Appear>
        )}

        <Appear index={2}>
          <Section label="Настройки">
            <Card style={s.list}>
              <ThemeRow />
              <Row label="Име" value={user.name} onPress={() => setEditing(true)} />
              <Row label="Снимка на профила" onPress={photoOptions} right={spinner('photo')} last />
            </Card>
          </Section>
        </Appear>

        {user.owner && (
          <Appear index={3}>
            <Section label="Собственик">
              <Card style={s.list}>
                <Row label="Pro" description="Кой има Pro" onPress={() => router.push('/owner')} last />
              </Card>
            </Section>
          </Appear>
        )}

        <Appear index={3}>
          <Section>
            <Card style={s.list}>
              <Row label="Изход" onPress={confirmSignOut} right={spinner('signout')} />
              <Row label="Изтриване на профила" onPress={confirmDelete} right={spinner('delete')} danger last />
            </Card>
          </Section>
        </Appear>
      </Screen>

      <NameEditor
        visible={editing}
        name={user.name}
        onClose={() => setEditing(false)}
        onSave={(name) =>
          run('name', async () => {
            updateProfile(await renameMe(name));
            setEditing(false);
          })
        }
        saving={busy === 'name'}
      />
    </SafeAreaView>
  );
}

function NameEditor(props: { visible: boolean; name: string; saving: boolean; onClose: () => void; onSave: (n: string) => void }) {
  const s = useStyles();
  const [value, setValue] = useState(props.name);
  const valid = value.trim().length > 0 && value.trim().length <= 100;
  return (
    <Modal visible={props.visible} transparent animationType="fade" onRequestClose={props.onClose} onShow={() => setValue(props.name)}>
      <View style={s.backdrop}>
        <Card style={s.dialog}>
          <H2>Вашето име</H2>
          <Field value={value} onChangeText={setValue} autoFocus maxLength={100} placeholder="Име" accessibilityLabel="Име" invalid={!valid} />
          <View style={s.dialogButtons}>
            <Button title="Отказ" variant="secondary" onPress={props.onClose} style={s.flex} />
            <Button title="Запази" onPress={() => props.onSave(value.trim())} disabled={!valid} loading={props.saving} style={s.flex} />
          </View>
        </Card>
      </View>
    </Modal>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  guest: { gap: space.lg },
  head: { gap: space.lg, paddingTop: space.sm },
  avatar: { alignSelf: 'flex-start', marginBottom: space.sm },
  name: { fontSize: 34, lineHeight: 38, letterSpacing: -1.4 },
  email: { color: c.muted, marginTop: space.sm },
  camera: {
    position: 'absolute',
    right: -2,
    bottom: -2,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: c.surface3,
    borderWidth: 1,
    borderColor: c.lineStrong,
    alignItems: 'center',
    justifyContent: 'center',
  },
  list: { paddingVertical: 0 },
  themeSwitch: { width: 150 },
  backdrop: { flex: 1, backgroundColor: c.backdrop, justifyContent: 'center', padding: space.xl },
  dialog: { gap: space.md, borderRadius: radius.md },
  dialogButtons: { flexDirection: 'row', gap: space.sm, marginTop: space.xs },
}));

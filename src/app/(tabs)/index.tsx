import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Avatar } from '@/components/Avatar';
import { Brand } from '@/components/Logo';
import { Appear, enter, PressScale, Reveal, RevealScroll } from '@/components/motion';
import { LevelStrip } from '@/components/Progress';
import { ScanRow } from '@/components/ScanCard';
import { Stage, type Journey } from '@/components/Stage';
import { ArrowLink, Button, IconButton, Kicker, Label, ScrollBar, Section, Title, useScrollY } from '@/components/ui';
import { errorText } from '@/lib/api';
import { refreshProfile } from '@/lib/auth';
import { importVideo } from '@/lib/importVideo';
import { aiAvailable } from '@/lib/prefs';
import { badgeOf, openEntry, useLibrary } from '@/lib/library';
import { checkProgress, useProgress } from '@/lib/progress';
import { useSession } from '@/lib/session';
import { startScan } from '@/lib/startScan';
import { afterTransition } from '@/lib/afterTransition';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { showDialog } from '@/lib/dialog';

// Начален екран (библиотеката): марката с връзката до сървъра, моделите с броячите им, новото сканиране (AI модел
// или инженерният режим, startScan) и последните модели като редове. До първото сканиране сцената с куба минава през
// трите стъпки.

const JOURNEY: Journey = [
  { title: 'Обиколете предмета', icon: 'phone-portrait-outline' },
  { title: 'Моделът се изгражда сам', icon: 'sparkles-outline' },
  { title: 'Отпечатайте го в реален размер', icon: 'print-outline' },
];

export default function Home() {
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;
  const { entries, loading, error } = useLibrary();
  const c = useColors();
  const s = useStyles();
  const [importing, setImporting] = useState(false);
  const recent = entries.slice(0, 5);
  const ready = entries.filter((e) => badgeOf(e.status) === 'done').length;
  const working = entries.filter((e) => badgeOf(e.status) === 'working').length;
  const scrollY = useScrollY();
  const hasScans = entries.length > 0;
  const firstTime = !loading && !hasScans;
  const progress = useProgress();

  const canAi = !!user && aiAvailable(); // прочита се отново при всяка промяна на сесията (профила)

  useFocusEffect(
    useCallback(() => {
      // След анимацията на прехода (връщане от модела), за да не я накъса.
      return afterTransition(() => {
        checkProgress();
        refreshProfile().catch(() => {}); // AI моделът, включен или изключен на сървъра
      });
    }, []),
  );

  async function pickVideo() {
    if (importing) return;
    setImporting(true);
    try {
      await importVideo();
    } catch (e) {
      showDialog('Видеото не може да се използва', errorText(e));
    } finally {
      setImporting(false);
    }
  }

  // Без AI модел: вход или съобщение (сървърът го е изключил).
  const needAi = () => {
    if (!user) return router.push('/sign-in');
    showDialog('AI моделът временно не е достъпен', 'Опитайте по-късно или използвайте инженерния режим.');
  };
  const upload = () => (canAi ? pickVideo() : needAi()); // видео, заснето по-рано

  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <ScrollBar
        y={scrollY.y}
        left={
          <View style={s.brandRow}>
            <Brand />
            {!loading && (
              <View style={[s.chip, { borderColor: error ? c.danger : c.success }]}>
                <View style={[s.chipDot, { backgroundColor: error ? c.danger : c.success }]} />
                <Text style={[s.chipText, { color: error ? c.danger : c.success }]}>{error ? 'ОФЛАЙН' : 'ОНЛАЙН'}</Text>
              </View>
            )}
          </View>
        }
        right={
          user ? (
            <PressScale onPress={() => router.push('/profile')} accessibilityRole="button" accessibilityLabel="Профил">
              <Avatar user={user} size={32} />
            </PressScale>
          ) : (
            <IconButton icon="person-outline" label="Профил" onPress={() => router.push('/profile')} />
          )
        }
      />
      <Animated.ScrollView contentContainerStyle={s.content} showsVerticalScrollIndicator={false} onScroll={scrollY.onScroll} scrollEventThrottle={16}>
        <RevealScroll y={scrollY.y}>
          <View style={s.hero}>
            {user && (
              <Appear>
                <Kicker>{`Здравейте, ${user.name.split(' ')[0]}`}</Kicker>
              </Appear>
            )}
            <Appear index={1}>
              <Title>Моите модели</Title>
            </Appear>
            <Appear index={2} style={s.stats}>
              {(
                [
                  ['Модели', entries.length, c.text],
                  ['Готови', ready, c.text],
                  ['В работа', working, c.signal],
                ] as const
              ).map(([label, value, color], i) => (
                <View key={label} style={[s.stat, i < 2 && s.statLine]}>
                  <Label numberOfLines={1}>{label}</Label>
                  <Text style={[s.statValue, { color }]}>{loading && !entries.length ? '—' : value}</Text>
                </View>
              ))}
            </Appear>
            <Appear index={3} style={s.cta}>
              <Button title="Нов скан" icon="add" size="lg" onPress={startScan} />
              <ArrowLink label={importing ? 'Отваряне…' : 'Качи готово видео'} onPress={upload} style={s.ctaLink} />
            </Appear>
            {progress && (
              <Appear index={7}>
                <LevelStrip progress={progress} onPress={() => router.push('/profile')} />
              </Appear>
            )}
          </View>

          {firstTime && (
            <Appear index={8}>
              <Stage height={310} journey={JOURNEY} />
            </Appear>
          )}

          {hasScans && (
            <Reveal>
              <Section label="Последни" action={{ label: 'Всички', onPress: () => router.push('/scans') }}>
                <View style={s.recent}>
                  {recent.map((e, i) => (
                    <Animated.View key={e.key} entering={enter(i)}>
                      <ScanRow entry={e} index={i} onPress={() => openEntry(e)} />
                    </Animated.View>
                  ))}
                </View>
              </Section>
            </Reveal>
          )}
        </RevealScroll>
      </Animated.ScrollView>
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  content: { paddingHorizontal: 16, paddingBottom: 48, gap: space.xl },
  hero: { paddingTop: space.md, gap: space.lg },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: 1, borderRadius: radius.pill, paddingHorizontal: 7, paddingVertical: 2 },
  chipDot: { width: 5, height: 5, borderRadius: radius.pill },
  chipText: { fontFamily: font.monoMedium, fontSize: 10, letterSpacing: 0.6 },
  stats: { flexDirection: 'row', borderWidth: 1, borderColor: c.line, borderRadius: radius.md, backgroundColor: c.surface },
  stat: { flex: 1, paddingVertical: space.md, paddingHorizontal: space.md, gap: 4 },
  statLine: { borderRightWidth: 1, borderRightColor: c.line },
  statValue: { fontFamily: font.monoMedium, fontSize: 20, lineHeight: 26 },
  cta: { gap: space.md },
  ctaLink: { alignSelf: 'center' },
  recent: { gap: space.sm },
}));

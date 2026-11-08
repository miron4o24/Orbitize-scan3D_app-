import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, ScrollView, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Avatar } from '@/components/Avatar';
import { Brand } from '@/components/Logo';
import { Appear, enter, PressScale, Reveal, RevealScroll, RiseLines } from '@/components/motion';
import { LevelStrip } from '@/components/Progress';
import { ScanCard, ScanCardSkeleton } from '@/components/ScanCard';
import { Stage, type Journey } from '@/components/Stage';
import { ScanChoice } from '@/components/ScanChoice';
import { ArrowLink, IconButton, Kicker, ScrollBar, Section, useScrollY } from '@/components/ui';
import { errorText, type Method } from '@/lib/api';
import { refreshProfile } from '@/lib/auth';
import { importVideo } from '@/lib/importVideo';
import { aiAvailable, aiTrial, PRO_INFO, SCAN_INFO } from '@/lib/prefs';
import { openEntry, useLibrary } from '@/lib/library';
import { checkProgress, useProgress } from '@/lib/progress';
import { useSession } from '@/lib/session';
import { makeStyles, space, text, useColors } from '@/theme';

// Начален екран: заглавие, бутонът за започване на сканиране и сканиранията. До първото сканиране сцената с куба
// минава през трите стъпки.

const JOURNEY: Journey = [
  { title: 'Обиколете предмета', icon: 'phone-portrait-outline' },
  { title: 'Моделът се изгражда сам', icon: 'sparkles-outline' },
  { title: 'Отпечатайте го в реален размер', icon: 'print-outline' },
];

export default function Home() {
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;
  const { entries, loading } = useLibrary();
  const c = useColors();
  const s = useStyles();
  const [importing, setImporting] = useState(false);
  const recent = entries.slice(0, 8);
  const scrollY = useScrollY();
  const hasScans = entries.length > 0;
  const firstTime = !loading && !hasScans;
  const progress = useProgress();

  const canAi = !!user && aiAvailable(); // прочита се отново при всяка промяна на сесията (профила)
  const aiTitle = `${PRO_INFO.single.label} · ${PRO_INFO.single.time}`;
  const scanTitle = `${SCAN_INFO.label} · ${SCAN_INFO.time}`;

  useFocusEffect(
    useCallback(() => {
      checkProgress();
      refreshProfile().catch(() => {}); // Pro, включен или изключен от собственика
    }, []),
  );

  async function pickVideo(method: Method = 'scan') {
    if (importing) return;
    setImporting(true);
    try {
      await importVideo(method);
    } catch (e) {
      Alert.alert('Видеото не може да се използва', errorText(e));
    } finally {
      setImporting(false);
    }
  }

  const scan = () => router.push('/capture');
  const filmAi = () => router.push({ pathname: '/capture', params: { ai: '1' } });
  const engineer = () => router.push('/engineer');

  // Видео, заснето по-рано: при наличие на двата начина потребителят казва кой.
  const upload = () => {
    if (!canAi) return pickVideo();
    Alert.alert('Какъв модел?', undefined, [
      { text: 'Отказ', style: 'cancel' },
      { text: scanTitle, onPress: () => pickVideo('scan') },
      { text: aiTitle, onPress: () => pickVideo('single') },
    ]);
  };

  return (
    <SafeAreaView style={s.fill} edges={['top']}>
      <ScrollBar
        y={scrollY.y}
        left={<Brand />}
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
            <RiseLines
              key={hasScans ? 'back' : 'new'}
              lines={hasScans ? ['Какво ще', { soft: 'сканирате?' }] : ['Сканирайте предмет.', 'Отпечатайте го', { soft: 'в реален размер.' }]}
              style={{ ...text.display, color: c.text }}
              softColor={c.muted}
              delay={60}
            />
            <Appear index={5} style={s.cta}>
              {canAi ? (
                <>
                  <ScanChoice highlight title={PRO_INFO.single.label} time={PRO_INFO.single.time} sub={aiTrial() ? 'Един безплатен опит' : 'Най-бързият начин'} icon="sparkles" onPress={filmAi} />
                  <ScanChoice title={SCAN_INFO.label} time={SCAN_INFO.time} sub="Може да е малко по-добър от AI модела" icon="videocam-outline" onPress={scan} />
                </>
              ) : (
                <ScanChoice highlight title="Започни сканиране" time={SCAN_INFO.time} icon="videocam" onPress={scan} />
              )}
              <ScanChoice title={PRO_INFO.cad.label} time={PRO_INFO.cad.time} sub="Детайл 1:1 от снимки и мерки: STEP за CAD" icon="construct-outline" onPress={engineer} />
              <ArrowLink label={importing ? 'Отваряне…' : 'Качи готово видео'} onPress={upload} style={s.ctaLink} />
              {!!user && !user.pro && <ArrowLink label="Виж Orbitize Pro" onPress={() => router.push('/pro')} style={s.ctaLink} />}
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

          {hasScans ? (
            <Reveal>
              <Section title="Вашите модели" action={{ label: 'Всички', onPress: () => router.push('/scans') }}>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.bleed} contentContainerStyle={s.recent}>
                  {recent.map((e, i) => (
                    <Animated.View key={e.key} entering={enter(i)} style={s.recentCard}>
                      <ScanCard entry={e} onPress={() => openEntry(e)} />
                    </Animated.View>
                  ))}
                </ScrollView>
              </Section>
            </Reveal>
          ) : loading ? (
            <View style={[s.recent, s.bleed]}>
              <View style={s.recentCard}>
                <ScanCardSkeleton />
              </View>
              <View style={s.recentCard}>
                <ScanCardSkeleton />
              </View>
            </View>
          ) : null}
        </RevealScroll>
      </Animated.ScrollView>
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  content: { paddingHorizontal: 18, paddingBottom: 48, gap: 44 },
  hero: { paddingTop: space.lg, gap: 22 },
  cta: { gap: 12, marginTop: 4 },
  ctaLink: { alignSelf: 'center', marginTop: 6 },
  bleed: { marginHorizontal: -18 },
  recent: { flexDirection: 'row', gap: space.md, paddingHorizontal: 18 },
  recentCard: { width: 168 },
}));

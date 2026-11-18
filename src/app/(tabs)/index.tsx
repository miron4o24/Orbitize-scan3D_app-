import Ionicons from '@expo/vector-icons/Ionicons';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ScrollView, Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Avatar } from '@/components/Avatar';
import { Brand } from '@/components/Logo';
import { Appear, enter, PressScale, Reveal, RevealScroll } from '@/components/motion';
import { LevelStrip } from '@/components/Progress';
import { ScanCard, ScanThumb, modelId, sizeText } from '@/components/ScanCard';
import { Stage, type Journey } from '@/components/Stage';
import { TourTarget } from '@/components/Tour';
import { ArrowLink, IconButton, Kicker, Label, ScrollBar, Section, Title, useScrollY, type IconName } from '@/components/ui';
import { errorText } from '@/lib/api';
import { refreshProfile } from '@/lib/auth';
import { shortDate } from '@/lib/dates';
import { importVideo } from '@/lib/importVideo';
import { aiAvailable, CAD_INFO, PRO_INFO } from '@/lib/prefs';
import { badgeOf, isAi, openEntry, useLibrary, type Entry } from '@/lib/library';
import { checkProgress, useProgress } from '@/lib/progress';
import { useSession } from '@/lib/session';
import { afterTransition } from '@/lib/afterTransition';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { showDialog } from '@/lib/dialog';

// Начален екран: марката с връзката до сървъра, последният готов модел отблизо, двата начина за нов модел като
// големи плочки (AI модел от видео, инженерният режим от снимки и шублер), броячите, съвет за по-добро сканиране и
// последните модели в лента за плъзгане. До първото сканиране сцената с куба минава през трите стъпки.

const JOURNEY: Journey = [
  { title: 'Обиколете предмета', icon: 'phone-portrait-outline' },
  { title: 'Моделът се изгражда сам', icon: 'sparkles-outline' },
  { title: 'Отпечатайте го в реален размер', icon: 'print-outline' },
];

// Съветите за сканиране: всеки път, когато екранът се отвори, следващият. Първите три са от истински лоши сканирания
// (The Rock и Фродо върху черна кутия: 14-17 мин и двойни очи; на шарен поднос - 7 мин и чисто).
const TIPS: { icon: IconName; text: string }[] = [
  { icon: 'grid-outline', text: 'Сложете предмета върху нещо шарено: вестник, списание или шарена кърпа. Камерата се хваща за шарката.' },
  { icon: 'sunny-outline', text: 'Равна светлина в стаята или до прозорец. Без светлината на телефона: прави отблясъци по лъскави предмети.' },
  { icon: 'sync-outline', text: 'Два бавни кръга: първо на височината на предмета, после отвисоко. Дръжте предмета в рамката.' },
  { icon: 'resize-outline', text: 'Приближете се: предметът да заема поне половината от кадъра. Далечен предмет губи детайлите си.' },
  { icon: 'construct-outline', text: 'За резервни части и детайли с точни мерки: инженерният режим. Шублерът дава размери до милиметър.' },
];
let tipIndex = 0;

export default function Home() {
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;
  const { entries, loading, error } = useLibrary();
  const c = useColors();
  const s = useStyles();
  const [importing, setImporting] = useState(false);
  const [tip, setTip] = useState(() => TIPS[tipIndex % TIPS.length]);
  const latest = entries.find((e) => badgeOf(e.status) === 'done');
  const recent = entries.filter((e) => e !== latest).slice(0, 8);
  const ready = entries.filter((e) => badgeOf(e.status) === 'done').length;
  const working = entries.filter((e) => badgeOf(e.status) === 'working').length;
  const scrollY = useScrollY();
  const hasScans = entries.length > 0;
  const firstTime = !loading && !hasScans;
  const progress = useProgress();

  const canAi = !!user && aiAvailable(); // прочита се отново при всяка промяна на сесията (профила)

  useFocusEffect(
    useCallback(() => {
      setTip(TIPS[tipIndex++ % TIPS.length]);
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
  const scanAi = () => (canAi ? router.push('/capture') : needAi());

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
              <Title>{hasScans ? 'Моите модели' : 'Първият ви 3D модел'}</Title>
            </Appear>

            {latest && (
              <Appear index={2}>
                <LatestCard entry={latest} />
              </Appear>
            )}

            <Appear index={3}>
              <Label>Нов модел</Label>
              <View style={s.modes}>
                <TourTarget id="home-ai" style={s.flex}>
                  <ModeTile
                    icon="sparkles-outline"
                    title={PRO_INFO.single.label}
                    sub="Видео около предмета"
                    time={PRO_INFO.single.time}
                    primary
                    onPress={scanAi}
                  />
                </TourTarget>
                <TourTarget id="home-cad" style={s.flex}>
                  <ModeTile
                    icon="construct-outline"
                    title="Инженерен"
                    sub="Снимки и шублер, 1:1"
                    time={CAD_INFO.time}
                    onPress={() => router.push('/engineer')}
                  />
                </TourTarget>
              </View>
              <TourTarget id="home-upload" style={s.uploadWrap}>
                <ArrowLink label={importing ? 'Отваряне…' : 'Качи готово видео'} onPress={upload} />
              </TourTarget>
            </Appear>

            <Appear index={4} style={s.stats}>
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

            {progress && (
              <Appear index={5}>
                <LevelStrip progress={progress} onPress={() => router.push('/profile')} />
              </Appear>
            )}
          </View>

          <Appear index={6}>
            <View style={s.tip}>
              <View style={s.tipIcon}>
                <Ionicons name={tip.icon} size={18} color={c.signal} />
              </View>
              <View style={s.flex}>
                <Label>Съвет за по-добър модел</Label>
                <Text style={s.tipText}>{tip.text}</Text>
              </View>
            </View>
          </Appear>

          {firstTime && (
            <Appear index={7}>
              <Stage height={280} journey={JOURNEY} />
            </Appear>
          )}

          {recent.length > 0 && (
            <Reveal>
              <Section label="Последни" action={{ label: 'Всички', onPress: () => router.push('/scans') }}>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.rail} style={s.railOut}>
                  {recent.map((e, i) => (
                    <Animated.View key={e.key} entering={enter(i)} style={s.railItem}>
                      <ScanCard entry={e} onPress={() => openEntry(e)} />
                    </Animated.View>
                  ))}
                </ScrollView>
              </Section>
            </Reveal>
          )}
        </RevealScroll>
      </Animated.ScrollView>
    </SafeAreaView>
  );
}

/* Последният готов модел, голям: рисунката му, номерът, размерът и как е създаден. */
function LatestCard({ entry }: { entry: Entry }) {
  const c = useColors();
  const s = useStyles();
  const size = sizeText(entry);
  return (
    <PressScale onPress={() => openEntry(entry)} style={s.latest} accessibilityRole="button" accessibilityLabel={`Последен модел: ${entry.title}`}>
      <View style={s.latestThumb}>
        <ScanThumb entry={entry} size={56} fill />
        <View style={s.latestTag}>
          <Text style={s.latestTagText}>ПОСЛЕДЕН</Text>
        </View>
      </View>
      <View style={s.latestMeta}>
        <View style={s.flex}>
          <Text style={s.latestTitle} numberOfLines={1}>
            {entry.title}
          </Text>
          <Text style={s.latestSub} numberOfLines={1}>
            {[modelId(entry), size ? `${size} мм` : shortDate(entry.createdAt), isAi(entry) ? 'AI' : null].filter(Boolean).join(' · ')}
          </Text>
        </View>
        <View style={s.open}>
          <Ionicons name="arrow-forward" size={18} color={c.onPrimary} />
        </View>
      </View>
    </PressScale>
  );
}

/* Голяма плочка за начин на създаване: икона, име, какво е нужно и колко време. */
function ModeTile(props: { icon: IconName; title: string; sub: string; time: string; primary?: boolean; onPress: () => void }) {
  const c = useColors();
  const s = useStyles();
  return (
    <PressScale
      onPress={props.onPress}
      style={[s.mode, props.primary && s.modePrimary]}
      accessibilityRole="button"
      accessibilityLabel={`${props.title}: ${props.sub}`}
    >
      <View style={[s.modeIcon, props.primary && s.modeIconPrimary]}>
        <Ionicons name={props.icon} size={22} color={props.primary ? c.onPrimary : c.accent} />
      </View>
      <Text style={s.modeTitle}>{props.title}</Text>
      <Text style={s.modeSub}>{props.sub}</Text>
      <Text style={s.modeTime}>{props.time}</Text>
    </PressScale>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  content: { paddingHorizontal: 16, paddingBottom: 48, gap: space.xl },
  hero: { paddingTop: space.md, gap: space.lg },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: 1, borderRadius: radius.pill, paddingHorizontal: 7, paddingVertical: 2 },
  chipDot: { width: 5, height: 5, borderRadius: radius.pill },
  chipText: { fontFamily: font.monoMedium, fontSize: 10, letterSpacing: 0.6 },
  latest: { borderWidth: 1, borderColor: c.line, borderRadius: radius.md, backgroundColor: c.surface, overflow: 'hidden' },
  latestThumb: { height: 190 },
  latestTag: {
    position: 'absolute',
    top: 10,
    left: 10,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.accentLine,
    backgroundColor: c.glass,
  },
  latestTagText: { fontFamily: font.monoMedium, fontSize: 10.5, letterSpacing: 0.8, color: c.accent },
  latestMeta: { flexDirection: 'row', alignItems: 'center', gap: space.md, padding: space.md, borderTopWidth: 1, borderTopColor: c.line },
  latestTitle: { fontFamily: font.semibold, fontSize: 16, color: c.text },
  latestSub: { fontFamily: font.mono, fontSize: 12, color: c.muted, marginTop: 2 },
  open: { width: 36, height: 36, alignItems: 'center', justifyContent: 'center', borderRadius: radius.sm, backgroundColor: c.primary },
  modes: { flexDirection: 'row', gap: space.sm, marginTop: space.sm },
  mode: { flex: 1, gap: 4, padding: space.md, borderWidth: 1, borderColor: c.line, borderRadius: radius.md, backgroundColor: c.surface, minHeight: 148 },
  modePrimary: { borderColor: c.accentLine, backgroundColor: c.accentSoft },
  modeIcon: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.accentLine,
    marginBottom: space.sm,
  },
  modeIconPrimary: { backgroundColor: c.primary, borderColor: c.primary },
  modeTitle: { fontFamily: font.semibold, fontSize: 16, color: c.text },
  modeSub: { fontFamily: font.regular, fontSize: 13, lineHeight: 18, color: c.text2 },
  modeTime: { fontFamily: font.monoMedium, fontSize: 12, color: c.muted, marginTop: 'auto' },
  uploadWrap: { alignSelf: 'center', marginTop: space.md },
  stats: { flexDirection: 'row', borderWidth: 1, borderColor: c.line, borderRadius: radius.md, backgroundColor: c.surface },
  stat: { flex: 1, paddingVertical: space.md, paddingHorizontal: space.md, gap: 4 },
  statLine: { borderRightWidth: 1, borderRightColor: c.line },
  statValue: { fontFamily: font.monoMedium, fontSize: 20, lineHeight: 26 },
  tip: {
    flexDirection: 'row',
    gap: space.md,
    padding: space.md,
    borderWidth: 1,
    borderColor: c.line,
    borderLeftWidth: 3,
    borderLeftColor: c.signal,
    borderRadius: radius.md,
    backgroundColor: c.surface,
  },
  tipIcon: { width: 32, height: 32, alignItems: 'center', justifyContent: 'center' },
  tipText: { fontFamily: font.regular, fontSize: 14, lineHeight: 20, color: c.text2, marginTop: 4 },
  railOut: { marginHorizontal: -16 },
  rail: { paddingHorizontal: 16, gap: space.sm },
  railItem: { width: 150 },
}));

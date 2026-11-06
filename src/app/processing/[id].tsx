import Ionicons from '@expo/vector-icons/Ionicons';
import { File } from 'expo-file-system';
import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ScrollView, Switch, Text, View } from 'react-native';
import Animated, {
  cancelAnimation,
  Easing,
  ReduceMotion,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withRepeat,
  withTiming,
  ZoomIn,
} from 'react-native-reanimated';

import { Appear, CountUp, LiveDot } from '@/components/motion';
import { Button, Card, ErrorText, H3, Header, Mono, Row, Screen, Small } from '@/components/ui';
import { OLD_SCAN_INFO, PRO_INFO } from '@/lib/prefs';
import { refreshProfile } from '@/lib/auth';
import { checkProgress } from '@/lib/progress';
import { setUpPush } from '@/lib/push';
import { useSession } from '@/lib/session';
import { font, makeStyles, radius, space, text, useColors } from '@/theme';

import {
  ApiError,
  cancelJob,
  createJob,
  errorText,
  failureText,
  getJob,
  getLog,
  retrying,
  startJob,
  uploadMotion,
  uploadPhoto,
  uploadVideo,
  type Method,
  type ProMethod,
} from '@/lib/api';
import { deleteScan, getScan, motionFile, photosDir, scanTitle, updateScan, videoFile, type Scan } from '@/lib/scans';

const ACTIVE = ['queued', 'reconstructing', 'meshing', 'cleaning'];
const STEPS: [string, string][] = [
  ['queued', 'На опашка'],
  ['reconstructing', 'Изграждане на предмета в 3D'],
  ['meshing', 'Изграждане на повърхността'],
  ['cleaning', 'Подготовка за печат'],
];
// Стъпките на инструментите на сървъра в "reconstructing", с думи за потребителя.
const SUBSTEPS: Record<string, string> = {
  'reading video': 'Четене на видеото',
  'picking frames': 'Избор на най-добрите кадри',
  'feature extraction': 'Разглеждане на кадрите',
  'photo matching': 'Свързване на кадрите',
  'camera solving': 'Подреждане на кадрите в 3D',
  undistorting: 'Подготовка на кадрите',
  'finding the object': 'Отделяне на предмета от фона',
  'estimating depth': 'Измерване на формата',
  'dense setup': 'Подготовка на кадрите',
  'dense matching': 'Измерване на повърхността',
  'sharpening the outlines': 'Уточняване на очертанията',
  'fusing depth': 'Изграждане на повърхността',
  'adding detail': 'Добавяне на фините детайли',
  texturing: 'Оцветяване със снимките',
  'shape from motion sensors': 'Изграждане на формата',
  // AI модели (Pro)
  'claude designing': 'Проектиране на предмета',
  'claude checking': 'Проверка на модела',
  'ai designing': 'Проектиране на предмета',
  'ai checking': 'Сравняване със сканирането и поправяне',
  'choosing frames': 'Избор на най-добрите кадри',
  'ai modelling': 'AI изгражда модела',
  'downloading the model': 'Получаване на модела',
  'understanding the request': 'Разчитане на промяната',
  'finding the part': 'Намиране на частта от модела',
};
// Текстовата промяна (Pro) минава през тези вместо това.
const EDIT_STEPS: [string, string][] = [
  ['queued', 'На опашка'],
  ['reconstructing', 'Разчитане на промяната'],
  ['cleaning', 'Прилагане на промените'],
];

const pop = ZoomIn.duration(240).easing(Easing.out(Easing.back(2))).reduceMotion(ReduceMotion.System); // отметка на стъпка

// Секундите на всяка стъпка на сървъра при сканиране на дрелка с 80 кадъра и пълни детайли (общо 24 мин): общата
// лента за напредък минава през тях. Бързото сканиране заема приблизително същите дялове от по-кратко време.
const PLAN: [string, number][] = [
  ['reading video', 4],
  ['picking frames', 2],
  ['feature extraction', 15],
  ['photo matching', 311],
  ['camera solving', 100],
  ['finding the object', 156],
  ['undistorting', 2],
  ['dense setup', 2],
  ['dense matching', 525],
  ['meshing', 22],
  ['sharpening the outlines', 98],
  ['estimating depth', 57],
  ['fusing depth', 146],
  ['cleaning', 6],
  ['texturing', 25],
];
// Същото за методите на Pro: експресното е мащабирано от стъпките за 80 кадъра до 30 по-малки кадъра, а останалото е
// приблизително.
// Модел от една снимка: сървърът търси предмета и AI изгражда формата.
const ONE_PHOTO_PLAN: [string, number][] = [
  ['finding the object', 5],
  ['ai modelling', 70],
  ['cleaning', 10],
];
const PRO_PLANS: Record<ProMethod, [string, number][]> = {
  edit: [
    ['reconstructing', 3],
    ['understanding the request', 45],
    ['cleaning', 10],
    ['finding the part', 40],
  ],
  express: [
    ['reading video', 3],
    ['picking frames', 2],
    ['feature extraction', 6],
    ['photo matching', 37],
    ['camera solving', 12],
    ['finding the object', 58],
    ['undistorting', 2],
    ['dense setup', 1],
    ['dense matching', 45],
    ['meshing', 10],
    ['estimating depth', 30],
    ['fusing depth', 60],
    ['cleaning', 4],
    ['texturing', 20],
  ],
  // 48 разположени кадъра, предметът е изрязан от всичките 48, а дълбочините им са слети в очертанията му (сървър
  // app/ai.py, outline.py).
  single: [
    ['reading video', 4],
    ['picking frames', 4],
    ['finding the object', 40],
    ['feature extraction', 8],
    ['photo matching', 30],
    ['camera solving', 12],
    ['undistorting', 3],
    ['sharpening the outlines', 30],
    ['dense setup', 1],
    ['dense matching', 90],
    ['meshing', 10],
    ['estimating depth', 15],
    ['fusing depth', 40],
    ['cleaning', 10],
    ['adding detail', 45],
    ['texturing', 20],
  ],
  claude: [
    ['reading video', 5],
    ['claude designing', 90],
    ['claude checking', 90],
    ['cleaning', 5],
  ],
  part: [
    ['finding the object', 40],
    ['camera solving', 30],
    ['dense matching', 90],
    ['fusing depth', 40],
    ['ai designing', 60],
    ['ai checking', 120],
    ['cleaning', 5],
  ],
  cad: [
    ['reconstructing', 5],
    ['ai designing', 60],
    ['ai checking', 150],
    ['cleaning', 5],
  ],
  photo: [
    ['reading video', 5],
    ['choosing frames', 20],
    ['ai modelling', 150],
    ['downloading the model', 10],
    ['cleaning', 5],
  ],
};

/* Дял от извършената работа (0..1) след `elapsed` мс в стъпка `name`; null за стъпки извън плана. */
function planned(plan: [string, number][], name: string, elapsed: number): number | null {
  const i = plan.findIndex(([n]) => n === name);
  if (i < 0) return null;
  const total = plan.reduce((a, [, s]) => a + s, 0);
  const before = plan.slice(0, i).reduce((a, [, s]) => a + s, 0);
  const own = plan[i][1];
  return (before + Math.min(elapsed / 1000, own * 0.95)) / total;
}

/* Тънка плоска лента, която плавно приема новата си дължина, с мек отблясък, докато работата продължава. */
function ProgressBar({ share }: { share: number }) {
  const s = useStyles();
  const reduce = useReducedMotion();
  const width = Math.max(2, Math.min(100, share * 100));
  const shown = useSharedValue(width);
  const shine = useSharedValue(0);
  useEffect(() => {
    shown.set(reduce ? width : withTiming(width, { duration: 400, easing: Easing.out(Easing.cubic) }));
  }, [width, reduce, shown]);
  useEffect(() => {
    if (reduce) return;
    shine.set(withRepeat(withTiming(1, { duration: 1600, easing: Easing.inOut(Easing.quad) }), -1, false));
    return () => cancelAnimation(shine);
  }, [reduce, shine]);
  const fill = useAnimatedStyle(() => ({ width: `${shown.value}%` }));
  const sweep = useAnimatedStyle(() => ({ left: `${-30 + 130 * shine.value}%` }));
  return (
    <View style={s.bar}>
      <Animated.View style={[s.barFill, fill]}>
        {!reduce && share < 1 && <Animated.View style={[s.barShine, sweep]} />}
      </Animated.View>
    </View>
  );
}

export default function Processing() {
  const { id, auto } = useLocalSearchParams<{ id: string; auto?: string }>();
  const c = useColors();
  const s = useStyles();
  const [scan, setScan] = useState<Scan | null>(() => getScan(id));
  const [uploaded, setUploaded] = useState<number | null>(null); // изпратени снимки или байтове; различно от null по време на качване
  const [netError, setNetError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [step, setStep] = useState<string | null>(null); // какво прави сървърът в момента
  const [ahead, setAhead] = useState<number | null>(null); // докато чака: колко са преди него
  const [log, setLog] = useState<string | null>(null);
  const update = (patch: Partial<Scan>) => setScan(updateScan(id, patch));
  const [progress, setProgress] = useState(0); // дял от извършената работа; никога не намалява (някои стъпки работят два пъти)

  const active = !!scan && ACTIVE.includes(scan.status);
  const jobId = scan?.jobId;
  const method: Method = scan?.method ?? 'scan';
  const pro = method === 'scan' ? null : method;
  const plan = scan?.photoCount === 1 ? ONE_PHOTO_PLAN : pro ? PRO_PLANS[pro] : PLAN;
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;

  useEffect(() => {
    if (!active || !jobId) return;
    let stopped = false;
    let current = { name: '', since: 0 };
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        if (stopped) return;
        setNetError(null);
        const name = job.step ?? job.state;
        const now = Date.now();
        if (current.name !== name) current = { name, since: now };
        setStep(name);
        setAhead(job.ahead ?? null);
        const p = planned(plan, name, now - current.since);
        if (p != null) setProgress((x) => Math.max(x, p));
        const r = job.report;
        const was = getScan(id);
        setScan(
          updateScan(id, {
            status: job.state,
            error: job.error ?? undefined,
            report: r ?? undefined,
            // AI модел носи името си и приблизителния реален размер, ако потребителят не е дал такъв.
            ...(r?.object && !was?.title && { title: r.object }),
            ...(r?.real_mm && !was?.longestMm && { longestMm: r.real_mm }),
            // Точна част по мерките на потребителя: размерът е истинският, без въпрос след това.
            ...(r?.real_mm && r.size_given && { longestMm: r.real_mm, sizeSet: true }),
          }),
        );
        if (job.state === 'done') {
          checkProgress(true); // завършено сканиране носи точки
          router.replace({ pathname: '/viewer/[id]', params: { id } });
        }
      } catch (e) {
        if (stopped) return;
        if (e instanceof ApiError && e.status === 404)
          setScan(
            updateScan(id, {
              status: 'failed',
              jobId: undefined,
              error: 'Обработката беше прекъсната. Качете видеото отново.',
            }),
          );
        else setNetError('Няма връзка в момента. Опитвам отново…');
      }
    };
    tick();
    const timer = setInterval(tick, 5000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [active, jobId, id, plan]);

  const video = videoFile(id);
  const hasVideo = video.exists;
  const photos =
    hasVideo || !photosDir(id).exists
      ? []
      : photosDir(id)
          .list()
          .filter((f): f is File => f instanceof File && f.name.endsWith('.jpg'));
  // Напредък в байтове за видео, иначе в снимки.
  const total = hasVideo ? video.size : photos.length;

  async function uploadAndStart() {
    setNetError(null);
    setInfo(null);
    setLog(null);
    setUploaded(0);
    // Заключен екран спира приложението, а с него и качването.
    await activateKeepAwakeAsync('upload').catch(() => {});
    try {
      const known = scan!.jobId;
      const old = known
        ? await retrying(() => getJob(known)).catch((e) => {
            if (e instanceof ApiError && e.status === 404) return null; // изтекло на сървъра: започваме нова задача
            throw e;
          })
        : null;
      const job = old ?? (await retrying(() => createJob()));
      if (!old) update({ jobId: job.id });
      // По-ранен опит го е стартирал, но отговорът се е загубил: просто го следваме.
      if (ACTIVE.includes(job.state)) {
        update({ status: job.state, error: undefined });
        return;
      }
      if (hasVideo) {
        setUploaded(job.video_bytes);
        await uploadVideo(job.id, video, job.video_bytes, setUploaded);
        const motion = motionFile(id);
        if (motion.exists) await uploadMotion(job.id, motion);
      } else {
        const have = new Set(job.received);
        let done = photos.filter((f) => have.has(f.name)).length;
        setUploaded(done);
        for (const f of photos) {
          if (have.has(f.name)) continue;
          await uploadPhoto(job.id, f);
          setUploaded(++done);
        }
      }
      // Изборите в момента: могат да се променят, докато качването върви.
      const now = getScan(id) ?? scan!;
      // AI моделът, а точната част (снимки и мерки) остава такава; и по-старо сканиране се опитва отново като AI модел.
      const way = now.method === 'part' ? 'part' : 'single';
      update({ method: way, quality: undefined });
      const started = await retrying(() =>
        startJob(job.id, {
          flat_base: now.flatBase ?? true,
          quality: 'detailed',
          ...(hasVideo && { video_bytes: video.size }),
          method: way,
          ...(now.extendFrom && { extend: now.extendFrom }),
          ...(way === 'part' && { size_mm: now.partSize ?? [], notes: now.partNotes ?? '' }),
        }),
      ).catch((e) => {
        // Опит, който е минал, но е загубил отговора си, оставя задачата вече стартирана.
        if (e instanceof ApiError && e.status === 409) return retrying(() => getJob(job.id));
        throw e;
      });
      update({ status: started.state, error: undefined });
      refreshProfile().catch(() => {});
      setUpPush(true); // моментът, в който "кажи ми, когато е готово" има смисъл
    } catch (e) {
      setNetError(`Качването спря. ${errorText(e)}`);
    } finally {
      setUploaded(null);
      deactivateKeepAwake('upload').catch(() => {});
    }
  }

  // Дневникът на задачата на сървъра: само за собственика на приложението, за да види защо сканирането се е
  // провалило.
  const owner = !!user?.owner;
  async function showLog() {
    if (!scan!.jobId) return setLog('Няма данни за това сканиране.');
    setLog('Зареждане…');
    setLog(await getLog(scan!.jobId).catch((e) => `Данните не се заредиха: ${errorText(e)}`));
  }

  async function cancel() {
    if (scan!.jobId) await cancelJob(scan!.jobId).catch(() => {}); // ако вече го няма, няма проблем
    update({ status: 'uploading', jobId: undefined, error: undefined });
    setInfo('Обработката е спряна.');
  }

  function remove() {
    if (scan!.jobId) cancelJob(scan!.jobId).catch(() => {});
    deleteScan(id);
    router.back();
  }

  // Веднага след заснемане или избор на видео: не е нужно да се натиска качване.
  const autoStarted = useRef(false);
  useEffect(() => {
    const sc = getScan(id);
    if (auto !== '1' || autoStarted.current || !sc || sc.status !== 'uploading' || sc.jobId) return;
    autoStarted.current = true;
    uploadAndStart();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once, on arrival
  }, []);

  if (!scan)
    return (
      <Screen>
        <Header title="Сканирането не е намерено." />
      </Screen>
    );

  if (active) {
    const total = scan.photoCount === 1 ? 2 : pro ? PRO_INFO[pro].minutes : scan.quality === 'quick' ? 10 : OLD_SCAN_INFO.minutes;
    const queued = scan.status === 'queued';
    const share = queued ? 0 : progress;
    const left = Math.max(1, Math.round(total * (1 - share)));
    const steps = method === 'edit' || method === 'cad' ? EDIT_STEPS : STEPS;
    const at = steps.findIndex(([st]) => st === scan.status);
    return (
      <Screen>
        <Header kicker={pro ? PRO_INFO[pro].label : OLD_SCAN_INFO.label} title={scanTitle(scan)} />
        <Appear index={1}>
          <Card style={s.overall}>
            <View style={s.row}>
              {queued ? (
                <Text style={[s.percent, s.flex]}>—</Text>
              ) : (
                <CountUp value={share * 100} format={(v) => `${Math.round(v)}%`} style={[s.percent, s.flex]} />
              )}
              <Mono style={s.muted}>{queued ? (ahead ? `преди вас: ${ahead}` : ahead === 0 ? 'следвате вие' : 'на опашка') : `≈ ${left} мин`}</Mono>
            </View>
            <ProgressBar share={share} />
          </Card>
        </Appear>
        <Appear index={2}>
          <Card style={s.list}>
            {steps.map(([state, label], i) => {
              const here = i === at;
              const past = i < at;
              return (
                <View key={state} style={[s.step, i < steps.length - 1 && s.line, !here && !past && s.ahead]}>
                  <View style={[s.dot, past && s.dotDone, here && s.dotHere]}>
                    {here ? (
                      <LiveDot color={c.accent} size={8} />
                    ) : past ? (
                      <Animated.View entering={pop}>
                        <Ionicons name="checkmark" size={14} color={c.success} />
                      </Animated.View>
                    ) : null}
                  </View>
                  <View style={s.flex}>
                    <H3 style={!here && !past && s.later}>{label}</H3>
                    {here && step && SUBSTEPS[step] && <Small>{SUBSTEPS[step]}</Small>}
                  </View>
                </View>
              );
            })}
          </Card>
        </Appear>
        <Appear index={3}>
          <Small>{queued && ahead ? `Изчакване около ${ahead * total} мин. Можете да затворите приложението.` : 'Можете да затворите приложението.'}</Small>
        </Appear>
        {netError && <ErrorText>{netError}</ErrorText>}
        <Button title="Спри обработката" variant="danger" icon="close" onPress={cancel} />
      </Screen>
    );
  }

  // Текстовата промяна няма какво да качва или повтаря: оригиналът е непроменен и потребителят я описва отново там.
  if (method === 'edit' || method === 'cad')
    return (
      <Screen>
        <Header
          title={
            method === 'cad'
              ? scan.status === 'failed'
                ? 'CAD моделът не успя'
                : 'CAD моделът е спрян'
              : scan.status === 'failed'
                ? 'Промяната не успя'
                : 'Промяната е спряна'
          }
        />
        <Appear index={1} style={s.failed}>
          {scan.status === 'failed' && <ErrorText>{failureText(scan.error)}</ErrorText>}
          <Small>{method === 'cad' ? 'Сканирането е непроменено. Опитайте отново по-късно.' : 'Оригиналът е непроменен. Опишете промяната по-просто.'}</Small>
          {scan.status === 'failed' &&
            owner &&
            (log === null ? (
              <Button title="Данни от сървъра" variant="secondary" onPress={showLog} />
            ) : (
              <ScrollView horizontal style={s.logBox}>
                <Text style={s.log} selectable>
                  {log}
                </Text>
              </ScrollView>
            ))}
        </Appear>
        <Button title="Изтрий" variant="danger" icon="trash-outline" onPress={remove} />
      </Screen>
    );

  const uploading = uploaded !== null;
  const failed = scan.status === 'failed';
  return (
    <Screen>
      <Header title={failed ? 'Сканирането не успя' : 'Готово за качване'} />
      {failed && (
        <Appear index={1} style={s.failed}>
          <ErrorText>{failureText(scan.error)}</ErrorText>
          <Small>Опитайте с ново видео: бавно, отблизо и на светло.</Small>
          {owner &&
            (log === null ? (
              <Button title="Данни от сървъра" variant="secondary" onPress={showLog} />
            ) : (
              <ScrollView horizontal style={s.logBox}>
                <Text style={s.log} selectable>
                  {log}
                </Text>
              </ScrollView>
            ))}
        </Appear>
      )}
      {info && <Small>{info}</Small>}
      <Appear index={3}>
        <Card style={s.list}>
          <Row
            label="Равна основа"
            last
            right={
              <Switch
                value={scan.flatBase ?? true}
                onValueChange={(v) => update({ flatBase: v })}
                trackColor={{ true: c.accentStrong, false: c.surface3 }}
                thumbColor="#ffffff"
                ios_backgroundColor={c.surface3}
                accessibilityLabel="Равна основа"
              />
            }
          />
        </Card>
      </Appear>
      {uploading ? (
        <Card style={s.overall}>
          <View style={s.row}>
            <H3 style={s.flex}>Качване</H3>
            <Mono>{Math.round((uploaded / Math.max(1, total)) * 100)}%</Mono>
          </View>
          <ProgressBar share={uploaded / Math.max(1, total)} />
          <Small>Оставете приложението отворено.</Small>
        </Card>
      ) : (
        <Button
          title={failed ? 'Опитай отново' : 'Създай модел'}
          icon={failed ? 'refresh' : 'cloud-upload-outline'}
          onPress={uploadAndStart}
          disabled={total === 0}
        />
      )}
      {netError && <ErrorText>{netError}</ErrorText>}
      {!uploading && <Button title="Изтрий сканирането" variant="danger" icon="trash-outline" onPress={remove} />}
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  muted: { color: c.muted },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  overall: { gap: space.md },
  percent: { ...text.display, fontSize: 64, lineHeight: 68, letterSpacing: -3, color: c.text },
  bar: { height: 3, backgroundColor: c.lineStrong, borderRadius: 99, overflow: 'hidden' },
  barFill: { height: 3, borderRadius: 99, backgroundColor: c.accent, overflow: 'hidden' },
  barShine: { position: 'absolute', top: 0, bottom: 0, width: '30%', backgroundColor: c.scheme === 'dark' ? 'rgba(9,9,11,0.35)' : 'rgba(255,255,255,0.45)', borderRadius: 99 },
  dotHere: { borderColor: c.accent, backgroundColor: c.accentSoft },
  list: { paddingVertical: 0 },
  line: { borderBottomWidth: 1, borderBottomColor: c.line },
  step: { flexDirection: 'row', alignItems: 'flex-start', gap: space.md, paddingVertical: space.md },
  dot: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: c.lineStrong,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: -1,
  },
  dotDone: { borderColor: c.success, backgroundColor: c.successSoft },
  later: { color: c.muted },
  ahead: { opacity: 0.45 },
  failed: { gap: space.md },
  logBox: { maxHeight: 320, backgroundColor: c.surface2, borderRadius: radius.sm, borderWidth: 1, borderColor: c.line, padding: space.sm },
  log: { fontFamily: font.mono, fontSize: 10, color: c.text2 },
}));

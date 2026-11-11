import { useCameraPermissions } from 'expo-camera';
import { Directory, File, Paths } from 'expo-file-system';
import { useKeepAwake } from 'expo-keep-awake';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Measure, mmText } from '@/components/cad/Measure';
import { Photos } from '@/components/cad/Photos';
import { Review } from '@/components/cad/Review';
import { SHOTS, Shoot } from '@/components/cad/Shoot';
import { Button, Card, ErrorText, Header, Note, Screen, Section, Small } from '@/components/ui';
import {
  cadAssemble,
  cadBuild,
  cadEdit,
  cadPrepare,
  cadRead,
  cancelJob,
  createJob,
  errorText,
  retrying,
  uploadPhoto,
  type CadOp,
  type CadOverallId,
  type CadPhoto,
  type CadPlan,
  type CadQuestion,
  type CadState,
} from '@/lib/api';
import { CAD_INFO } from '@/lib/prefs';
import { createScan, getScan, modelFile, updateScan } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { font, makeStyles, space } from '@/theme';

// Инженерният режим: мерене по указания (сървър app/cad). Снимките казват какво има (очертания, стъпала, отвори),
// шублерът казва колко е голямо, а моделът се строи само от числата.
// 1. До шест снимки отгоре, всяка с друга страна на предмета нагоре (качват се и се четат, докато се снима).
// 2. Преглед на снимките. 3. Общите размери, всеки с линия върху снимката. 4. "Какво виждам" с 3D по оценките.
// 5. Подробните мерки, по една, с линия върху снимката. 6. Моделът и файловете.
// С ?scan=<id> се отваря построен детайл за промяна на мерките или елементите.

type Phase = 'intro' | 'shoot' | 'photos' | 'reading' | 'overall' | 'review' | 'measure' | 'building';

export default function Engineer() {
  useKeepAwake();
  const { scan: editId } = useLocalSearchParams<{ scan?: string }>();
  const editing = editId ? getScan(editId) : null;
  const s = useStyles();
  const [permission, requestPermission] = useCameraPermissions();
  const [folder] = useState(() => {
    const d = new Directory(Paths.cache, `cad-${Date.now().toString(36)}`);
    d.create({ intermediates: true, idempotent: true });
    return d;
  });
  const [phase, setPhase] = useState<Phase>(editing ? 'reading' : 'intro');
  const [shot, setShot] = useState(0);
  const [single, setSingle] = useState(false); // снимане отново на една снимка от прегледа
  const [photos, setPhotos] = useState<Partial<Record<CadPhoto, File>>>({});
  const [uploading, setUploading] = useState(0);
  const [jobId, setJobId] = useState<string | null>(editing?.jobId ?? null);
  const [plan, setPlan] = useState<CadPlan | null>(null);
  const [oi, setOi] = useState(0);
  const [sizes, setSizes] = useState<Partial<Record<CadOverallId, number>>>({});
  const [state, setState] = useState<CadState | null>(null);
  const [asked, setAsked] = useState<CadQuestion[]>([]);
  const [qi, setQi] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number | null>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const finished = useRef(false);
  const jobRef = useRef<string | null>(null);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const failed = useRef<Set<CadPhoto>>(new Set());

  // Излизане без готов детайл: задачата на сървъра се изтрива (иначе би останала като недовършено сканиране).
  useEffect(
    () => () => {
      if (!finished.current && jobRef.current && !editId) cancelJob(jobRef.current).catch(() => {});
      if (folder.exists) folder.delete();
    },
    [folder, editId],
  );

  // Промяна на построен детайл: описанието му от сървъра (последното построено).
  useEffect(() => {
    if (!editing?.cad || !editing.jobId) return;
    cadEdit(editing.jobId, editing.cad, { values: {} })
      .then((st) => {
        setState(st);
        setPhase('review');
      })
      .catch((e) => {
        setError(`Детайлът вече не е на сървъра: ${errorText(e)}`);
        setPhase('review');
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function ensureJob(): Promise<string> {
    if (jobRef.current) return jobRef.current;
    const id = (await retrying(() => createJob())).id;
    jobRef.current = id;
    setJobId(id);
    return id;
  }

  // Всяка снимка се качва веднага, една след друга, и сървърът започва да я чете, докато се снима следващата.
  function upload(photo: CadPhoto, file: File) {
    setUploading((n) => n + 1);
    queue.current = queue.current
      .then(async () => {
        const id = await ensureJob();
        await uploadPhoto(id, file);
        failed.current.delete(photo);
        cadPrepare(id).catch(() => {}); // само за да е по-бързо накрая; грешките се виждат при четенето
      })
      .catch(() => {
        failed.current.add(photo);
      })
      .finally(() => setUploading((n) => n - 1));
  }

  async function read() {
    setError(null);
    setPhase('reading');
    try {
      await queue.current;
      const id = await ensureJob();
      for (const photo of [...failed.current]) {
        const file = photos[photo];
        if (file) await uploadPhoto(id, file); // повторен опит за неуспешно качените
        failed.current.delete(photo);
      }
      const p = await cadRead(id);
      setPlan(p);
      setOi(0);
      setPhase('overall');
    } catch (e) {
      setError(errorText(e));
      setPhase('photos');
    }
  }

  async function assemble(values: Partial<Record<CadOverallId, number>>) {
    if (!jobId) return;
    setBusy(true);
    setError(null);
    setPhase('reading');
    try {
      const st = await cadAssemble(jobId, values);
      setState(st);
      setPhase('review');
    } catch (e) {
      setError(errorText(e));
      setPhase('overall');
    } finally {
      setBusy(false);
    }
  }

  async function apply(op: CadOp) {
    if (!state || !jobId) return;
    setBusy(true);
    setError(null);
    try {
      setState(await cadEdit(jobId, state.design, op));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  function startMeasuring() {
    if (!state) return;
    const list = state.questions.filter((q) => (editing ? true : q.src !== 'measured'));
    if (!list.length) return build({});
    setAsked(list);
    // Вече въведените (връщане към прегледа и обратно) остават; при промяна на детайл: досегашните стойности.
    setAnswers((prev) =>
      Object.fromEntries(
        list.filter((q) => q.id in prev || editing).map((q) => [q.id, q.id in prev ? prev[q.id] : q.value]),
      ),
    );
    setQi(0);
    setPhase('measure');
  }

  async function build(values: Record<string, number | null>) {
    if (!state || !jobId) return;
    setBusy(true);
    setError(null);
    try {
      const measured = await cadEdit(jobId, state.design, { values });
      setState(measured);
      const job = await cadBuild(jobId, measured.design);
      const report = job.report!;
      finished.current = true;
      const size = report.cad?.size_mm.map(mmText).join(' × ');
      if (editing) {
        const local = modelFile(editing.id);
        if (local.exists) local.delete(); // новият модел се изтегля наново
        updateScan(editing.id, {
          report,
          longestMm: report.real_mm ?? undefined,
          cad: measured.design,
          status: 'done',
        });
        router.back();
        return;
      }
      const scan = createScan(currentUserId() ?? undefined, 'cad');
      updateScan(scan.id, {
        status: 'done',
        jobId,
        report,
        longestMm: report.real_mm ?? undefined,
        cad: measured.design,
        title: `CAD детайл ${size ?? ''} мм`.trim(),
      });
      router.replace({ pathname: '/viewer/[id]', params: { id: scan.id } });
    } catch (e) {
      setError(errorText(e));
      setPhase((p) => (p === 'building' ? 'measure' : p));
    } finally {
      setBusy(false);
    }
  }

  function answer(value: number | null) {
    const q = asked[qi];
    const next = { ...answers, [q.id]: value };
    setAnswers(next);
    if (qi < asked.length - 1) {
      setQi(qi + 1);
      return;
    }
    build(next);
  }

  function answerOverall(value: number | null) {
    if (!plan || value == null) return;
    const q = plan.overall[oi];
    const next = { ...sizes, [q.id]: value };
    setSizes(next);
    if (oi < plan.overall.length - 1) {
      setOi(oi + 1);
      return;
    }
    assemble(next);
  }

  // --- Екраните ---

  if (phase === 'shoot') {
    if (!permission) return <View style={s.black} />;
    if (!permission.granted)
      return (
        <Screen>
          <Header title="Достъп до камерата" />
          {permission.canAskAgain ? (
            <Button title="Разреши камерата" icon="camera-outline" onPress={requestPermission} />
          ) : (
            <Small>Достъпът до камерата е отказан. Разрешете го от настройките на телефона.</Small>
          )}
          <Button title="Назад" variant="ghost" onPress={() => setPhase('intro')} />
        </Screen>
      );
    const photo = SHOTS[shot].photo;
    const done = (file: File | null) => {
      if (file) {
        setPhotos((p) => ({ ...p, [photo]: file }));
        upload(photo, file);
      }
      if (single || shot === SHOTS.length - 1) {
        setSingle(false);
        setPhase('photos');
      } else setShot(shot + 1);
    };
    const close = () => {
      if (!single) return router.back();
      setSingle(false);
      setPhase('photos');
    };
    return <Shoot key={`${shot}-${single}`} index={shot} only={single} folder={folder} onTaken={done} onSkip={() => done(null)} onClose={close} />;
  }

  if (phase === 'photos')
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Photos
          photos={photos}
          uploading={uploading > 0}
          error={error}
          onRetake={(p) => {
            setShot(SHOTS.findIndex((x) => x.photo === p));
            setSingle(true);
            setError(null);
            setPhase('shoot');
          }}
          onNext={read}
          onBack={() => setPhase('intro')}
        />
      </SafeAreaView>
    );

  if (phase === 'reading' || phase === 'building')
    return (
      <SafeAreaView style={s.safe}>
        <View style={s.wait}>
          <ActivityIndicator size="large" />
          <Text style={s.waitTitle}>
            {phase === 'building' ? 'Строя модела…' : editing ? 'Зареждане на детайла…' : plan ? 'Сглобявам формата…' : 'Чета снимките…'}
          </Text>
          <Small>
            {phase === 'building'
              ? 'Само от вашите мерки. Няколко секунди.'
              : plan
                ? 'Очертания, стъпала и отвори от всички снимки. Няколко секунди.'
                : 'Намирам предмета на всяка снимка. До половин минута.'}
          </Small>
        </View>
      </SafeAreaView>
    );

  if (phase === 'overall' && plan && jobId && plan.overall[oi]) {
    const q = plan.overall[oi];
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Measure
          key={q.id}
          jobId={jobId}
          stamp={plan.stamp}
          kicker="Общите размери ·"
          question={{ ...q, est: null, value: sizes[q.id] ?? null, src: 'photo', optional: false, kind: q.id === 'D' ? 'diameter' : 'length' }}
          index={oi}
          count={plan.overall.length}
          answer={sizes[q.id]}
          busy={busy}
          error={error ?? (oi === 0 && plan.warnings.length ? plan.warnings.join(' ') : null)}
          onAnswer={answerOverall}
          onBack={() => (oi > 0 ? setOi(oi - 1) : setPhase('photos'))}
        />
      </SafeAreaView>
    );
  }

  if (phase === 'review' && state && jobId)
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Review
          jobId={jobId}
          state={state}
          busy={busy}
          error={error}
          onOp={apply}
          onNext={startMeasuring}
          onRetake={
            editing
              ? undefined
              : () => {
                  setError(null);
                  setPhase('photos');
                }
          }
        />
      </SafeAreaView>
    );

  if (phase === 'measure' && state && jobId && asked[qi])
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Measure
          key={asked[qi].id}
          jobId={jobId}
          stamp={state.stamp}
          question={asked[qi]}
          index={qi}
          count={asked.length}
          answer={answers[asked[qi].id]}
          busy={busy}
          error={error}
          onAnswer={answer}
          onBack={() => (qi > 0 ? setQi(qi - 1) : setPhase('review'))}
        />
      </SafeAreaView>
    );

  if (phase === 'review' && error)
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Screen>
          <Header title="Детайлът не се зареди" />
          <ErrorText>{error}</ErrorText>
          <Button title="Назад" variant="secondary" onPress={() => router.back()} />
        </Screen>
      </SafeAreaView>
    );

  const started = Object.keys(photos).length > 0;
  return (
    <SafeAreaView style={s.safe} edges={['top']}>
      <Screen>
        <Header
          kicker={CAD_INFO.label}
          title="Детайл 1:1 от снимки и мерки"
          sub="Снимките казват каква е формата, вашите мерки казват колко е голяма. Моделът е само от вашите числа."
        />
        <Section label="Как става">
          <Card style={s.card}>
            {STEPS.map(([title, text], k) => (
              <View key={title} style={s.step}>
                <Text style={s.stepNum}>{k + 1}</Text>
                <View style={s.flex}>
                  <Text style={s.label}>{title}</Text>
                  <Small>{text}</Small>
                </View>
              </View>
            ))}
          </Card>
        </Section>
        <Note>
          Нужен е шублер (или линийка за по-груби мерки) и контрастна повърхност: светъл предмет на тъмна маса или обратно. За
          плочи, скоби, капаци, кутии, валове, цилиндри; не за свободни (органични) форми.
        </Note>
        <Button
          title={started ? 'Към снимките' : 'Започни'}
          variant="brand"
          size="lg"
          icon="camera-outline"
          onPress={() => {
            if (started) return setPhase('photos');
            setShot(0);
            setPhase('shoot');
          }}
        />
        <Button title="Назад" variant="ghost" onPress={() => router.back()} />
      </Screen>
    </SafeAreaView>
  );
}

const STEPS: [string, string][] = [
  [
    'Снимки от всички страни',
    'До 6 снимки отгоре: за всяка обръщате предмета с друга страна нагоре. Картинка показва коя и как; нивелирът казва кога телефонът е прав.',
  ],
  ['Общите размери', 'Дължина, ширина, височина: оранжева линия върху вашата снимка показва какво да измерите.'],
  ['Какво виждам', 'Сървърът разчита очертанията, стъпалата и отворите; проверявате ги в 3D и добавяте пропуснатото.'],
  ['Мерките, по една', 'Всяка с линия върху снимката. Каквото не можете да измерите, остава с оценката от снимките.'],
  ['CAD файловете', 'STEP за SolidWorks, Fusion, FreeCAD, Onshape; кодът на модела; STL; чертеж.'],
];

const useStyles = makeStyles((c) => ({
  safe: { flex: 1, backgroundColor: c.bg },
  black: { flex: 1, backgroundColor: '#000' },
  flex: { flex: 1 },
  card: { gap: space.lg },
  label: { fontFamily: font.medium, fontSize: 15, color: c.text },
  step: { flexDirection: 'row', gap: space.md, alignItems: 'flex-start' },
  stepNum: {
    width: 22,
    color: c.accent,
    fontFamily: font.semibold,
    fontSize: 15,
  },
  wait: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.md,
    padding: space.xl,
  },
  waitTitle: { color: c.text, fontFamily: font.semibold, fontSize: 18 },
}));

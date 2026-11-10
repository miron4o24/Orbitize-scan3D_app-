import { useCameraPermissions } from 'expo-camera';
import { Directory, File, Paths } from 'expo-file-system';
import { useKeepAwake } from 'expo-keep-awake';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Measure, mmOf, mmText } from '@/components/cad/Measure';
import { Review } from '@/components/cad/Review';
import { SHOTS, Shoot } from '@/components/cad/Shoot';
import { Button, Card, ErrorText, Field, Header, Note, Screen, Section, Small } from '@/components/ui';
import {
  cadBuild,
  cadEdit,
  cadRead,
  cancelJob,
  createJob,
  errorText,
  retrying,
  uploadPhoto,
  type CadOp,
  type CadQuestion,
  type CadState,
  type CadView,
} from '@/lib/api';
import { CAD_INFO } from '@/lib/prefs';
import { createScan, getScan, modelFile, updateScan } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { font, makeStyles, space } from '@/theme';

// Инженерният режим: мерене по указания (сървър app/cad). Снимките казват какво има (очертания, стъпала, отвори),
// шублерът казва колко е голямо, а моделът се строи само от числата. 1. Три снимки отгоре с нивелир. 2. Общите
// размери. 3. Сървърът чете снимките: "Какво виждам" с 3D по оценките. 4. Мерките, по една. 5. Моделът и файловете.
// С ?scan=<id> се отваря построен детайл за промяна на мерките или елементите.

type Phase = 'intro' | 'shoot' | 'overall' | 'reading' | 'review' | 'measure' | 'building';

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
  const [photos, setPhotos] = useState<Partial<Record<CadView, File>>>({});
  const [sizes, setSizes] = useState({ L: '', W: '', H: '' });
  const [jobId, setJobId] = useState<string | null>(editing?.jobId ?? null);
  const [state, setState] = useState<CadState | null>(null);
  const [asked, setAsked] = useState<CadQuestion[]>([]);
  const [qi, setQi] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number | null>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const finished = useRef(false);
  const jobRef = useRef<string | null>(null);

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

  async function read() {
    const L = mmOf(sizes.L);
    const W = mmOf(sizes.W);
    const H = mmOf(sizes.H);
    if (!L || !W || !H) return setError('Въведете трите общи размера в милиметри, от 0 до 1000.');
    setError(null);
    setPhase('reading');
    try {
      let id = jobRef.current;
      if (!id) {
        id = (await retrying(() => createJob())).id;
        jobRef.current = id;
        setJobId(id);
      }
      for (const f of Object.values(photos)) if (f) await uploadPhoto(id, f);
      const st = await cadRead(id, L, W, H);
      setState(st);
      setPhase('review');
    } catch (e) {
      setError(errorText(e));
      setPhase('overall');
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
    setAnswers(editing ? Object.fromEntries(list.map((q) => [q.id, q.value])) : {});
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
        updateScan(editing.id, { report, longestMm: report.real_mm ?? undefined, cad: measured.design, status: 'done' });
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
    const done = (file: File | null) => {
      if (file) setPhotos((p) => ({ ...p, [SHOTS[shot].view]: file }));
      else
        setPhotos((p) => {
          const { [SHOTS[shot].view]: _skip, ...rest } = p;
          return rest;
        });
      if (shot < SHOTS.length - 1) setShot(shot + 1);
      else setPhase('overall');
    };
    return <Shoot key={shot} index={shot} folder={folder} onTaken={done} onSkip={() => done(null)} onClose={() => router.back()} />;
  }

  if (phase === 'reading' || phase === 'building')
    return (
      <SafeAreaView style={s.safe}>
        <View style={s.wait}>
          <ActivityIndicator size="large" />
          <Text style={s.waitTitle}>{phase === 'reading' ? (editing ? 'Зареждане на детайла…' : 'Чета снимките…') : 'Строя модела…'}</Text>
          <Small>{phase === 'reading' ? 'Очертания, стъпала и отвори. Около половин минута.' : 'Само от вашите мерки. Няколко секунди.'}</Small>
        </View>
      </SafeAreaView>
    );

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
          onRetake={() => {
            setShot(0);
            setPhotos({});
            setError(null);
            setPhase('shoot');
          }}
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

  if (phase === 'overall')
    return (
      <SafeAreaView style={s.safe} edges={['top']}>
        <Screen>
          <Header kicker={`${CAD_INFO.label} · снимките са готови`} title="Общите размери" sub="С шублер, в милиметри. От тях се разчита мащабът на снимките." />
          <Card style={s.card}>
            {(
              [
                ['L', 'Дължина', 'най-дългата страна, както лежи'],
                ['W', 'Ширина', 'другата страна, гледано отгоре'],
                ['H', 'Височина', 'от масата до най-високата точка'],
              ] as const
            ).map(([key, label, hint]) => (
              <View key={key} style={s.line}>
                <View style={s.flex}>
                  <Text style={s.label}>{label}</Text>
                  <Small>{hint}</Small>
                </View>
                <Field
                  style={s.mm}
                  value={sizes[key]}
                  onChangeText={(t) => setSizes((v) => ({ ...v, [key]: t }))}
                  keyboardType="decimal-pad"
                  placeholder="мм"
                  accessibilityLabel={`${label} в милиметри`}
                  invalid={!!sizes[key].trim() && !mmOf(sizes[key])}
                />
              </View>
            ))}
          </Card>
          <Note>Кръгъл детайл (вал, втулка): ширината и височината са диаметърът.</Note>
          {error && <ErrorText>{error}</ErrorText>}
          <Button title="Прочети снимките" variant="brand" size="lg" icon="scan-outline" onPress={read} />
          <Button
            title="Снимай отново"
            variant="ghost"
            icon="camera-outline"
            onPress={() => {
              setShot(0);
              setPhotos({});
              setPhase('shoot');
            }}
          />
        </Screen>
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

  return (
    <SafeAreaView style={s.safe} edges={['top']}>
      <Screen>
        <Header kicker={CAD_INFO.label} title="Детайл 1:1 от снимки и шублер" sub="Снимките казват какво има, шублерът казва колко е голямо. Моделът е само от вашите числа." />
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
        <Note>Нужен е шублер. За плочи, скоби, капаци, кутии, валове и втулки; не за свободни (органични) форми.</Note>
        <Button title="Започни" variant="brand" size="lg" icon="camera-outline" onPress={() => setPhase('shoot')} />
        <Button title="Назад" variant="ghost" onPress={() => router.back()} />
      </Screen>
    </SafeAreaView>
  );
}

const STEPS: [string, string][] = [
  ['Три снимки отгоре', 'Детайлът легнал, на страна и изправен. Нивелирът казва кога телефонът е прав. Контрастна повърхност.'],
  ['Общите размери', 'Дължина, ширина и височина с шублера.'],
  ['Какво виждам', 'Сървърът разчита очертанията, стъпалата и отворите; проверявате ги в 3D и добавяте пропуснатото.'],
  ['Мерките, по една', 'Картинка показва какво точно да измерите. Обикновено 5 до 15 мерки.'],
  ['CAD файловете', 'STEP за SolidWorks, Fusion, FreeCAD, Onshape; кодът на модела; STL; чертеж.'],
];

const useStyles = makeStyles((c) => ({
  safe: { flex: 1, backgroundColor: c.bg },
  black: { flex: 1, backgroundColor: '#000' },
  flex: { flex: 1 },
  card: { gap: space.lg },
  line: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  label: { fontFamily: font.medium, fontSize: 15, color: c.text },
  mm: { width: 104, textAlign: 'right' },
  step: { flexDirection: 'row', gap: space.md, alignItems: 'flex-start' },
  stepNum: { width: 22, color: c.accent, fontFamily: font.semibold, fontSize: 15 },
  wait: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.md, padding: space.xl },
  waitTitle: { color: c.text, fontFamily: font.semibold, fontSize: 18 },
}));

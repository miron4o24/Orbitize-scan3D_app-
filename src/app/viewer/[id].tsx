import Ionicons from '@expo/vector-icons/Ionicons';
import { File, Paths } from 'expo-file-system';
import { router, useLocalSearchParams, type ErrorBoundaryProps } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Modal, Pressable, Share, Switch, Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { Object3D } from 'three';

import { FullscreenModel } from '@/components/FullscreenModel';
import { CadPanel } from '@/components/cad/CadPanel';
import { MeasureTool } from '@/components/MeasureTool';
import { ModelView, type Spot } from '@/components/ModelView';
import { Appear, fade, PressScale, Skeleton, swap } from '@/components/motion';
import { PrintHelper } from '@/components/PrintHelper';
import { SpotPicker } from '@/components/SpotPicker';
import { Badge, Button, Card, ErrorText, Field, H2, H3, IconButton, Label, Note, Row, Screen, Section, Segmented, Small, Title } from '@/components/ui';
import {
  downloadResult,
  downloadSpin,
  editModel,
  errorText,
  getGeometry,
  nameplateModel,
  publishScan,
  renameScan,
  shareScan,
  shareUrl,
  unpublishScan,
  unshareScan,
  type Report,
} from '@/lib/api';
import { arAvailable, arSupport, viewInRoom, type ArSupport } from '@/lib/ar';
import { refreshProfile } from '@/lib/auth';
import { duration, shortDate } from '@/lib/dates';
import { exportForPrint } from '@/lib/exportModel';
import { badgeOf, deleteEntry } from '@/lib/library';
import { parseGlb } from '@/lib/model';
import { allowance, allowanceNote, blocked } from '@/lib/plan';
import { madeWith, SCAN_INFO } from '@/lib/prefs';
import { estimate, euro, getPrices, printTime, setPrice } from '@/lib/printEstimate';
import { checkProgress } from '@/lib/progress';
import { createScan, getScan, modelFile, scanTitle, updateScan } from '@/lib/scans';
import { currentUserId, useSession } from '@/lib/session';
import { INFILLS, MATERIALS, type Infill, type Material } from '@/lib/stress';
import { font, makeStyles, radius, space, text, useColors } from '@/theme';

async function loadModel(id: string, jobId: string | undefined): Promise<Object3D> {
  const file = modelFile(id);
  if (!file.exists) {
    if (!jobId) throw new Error('Моделът вече не е наличен.');
    await downloadResult(jobId, 'glb', file);
  }
  try {
    return await parseGlb(file, `scan-${id}`);
  } catch (e) {
    if (jobId) file.delete(); // вероятно прекъснато изтегляне; повторният опит го изтегля отново
    throw e;
  }
}

const FORMATS = [
  { format: 'stl', mime: 'model/stl', note: '' },
  { format: 'obj', mime: 'model/obj', note: 'За 3D програми' },
  { format: 'glb', mime: 'model/gltf-binary', note: 'С цветовете' },
] as const;
type Format = (typeof FORMATS)[number]['format'];
// Файлове за печат: "smooth" идва от сървъра с изгладени вълни от сканирането (сървър app/mesh.py smooth), "detail" е
// моделът, както е показан.
type Surface = 'smooth' | 'detail';
const SMOOTH_PASSES = 2;
// Под този дял от ширината на кадъра предметът няма достатъчно пиксели за детайлите си (колонка от 6 см, заснета от
// разстояние на ръка, заема 18%).
const SMALL_OBJECT = 0.5;

type Tab = 'size' | 'files' | 'share';

const FRAMES_KEPT_MS = 7 * 86_400_000; // сървърът пази кадрите на модела толкова дълго (сървър JOB_TTL_DAYS)

const mmText = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

// Грешка при показване на модела (издадената версия би затворила приложението) се показва тук.
export function ErrorBoundary({ error, retry }: ErrorBoundaryProps) {
  const s = useStyles();
  return (
    <View style={[s.fill, s.center]}>
      <ErrorText>Моделът не може да се покаже на този телефон.</ErrorText>
      <Button title="Опитай отново" icon="refresh" onPress={retry} />
    </View>
  );
}

export default function Viewer() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const c = useColors();
  const s = useStyles();
  const [scan, setScan] = useState(() => getScan(id));
  const [model, setModel] = useState<Object3D | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [full, setFull] = useState(false); // моделът на целия екран
  const [mmInput, setMmInput] = useState(String(scan?.longestMm ?? 100));
  const [busy, setBusy] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [realSize, setRealSize] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [tab, setTab] = useState<Tab>('size');
  const [surface, setSurface] = useState<Surface>('smooth');
  const [support, setSupport] = useState<ArSupport | null>(null); // AR на този телефон; null, докато се пита
  const [editing, setEditing] = useState(false); // диалогът за текстова промяна (Pro)
  const [editText, setEditText] = useState(''); // пази се, докато потребителят посочва върху модела
  const [spot, setSpot] = useState<Spot | null>(null); // мястото върху модела, за което е промяната ("smooth here")
  const [picking, setPicking] = useState(false); // посочване върху модела на целия екран
  const [measuring, setMeasuring] = useState(false); // две докосвания върху модела на целия екран
  const [plating, setPlating] = useState(false); // диалогът за табелка с име (Pro)
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;
  const jobId = scan?.jobId;
  const account = !!currentUserId() && !!jobId; // запазено в профила: преименуване на сървъра, споделяне с линк
  const arDone = arAvailable && !!jobId && scan?.status === 'done';
  const showAr = arDone && support !== null && support !== 'unsupported';
  const stressAllowed = allowance(user, 'stress', jobId);
  const arAllowed = allowance(user, 'ar', jobId);
  const canEdit = !!user?.pro && !!user.ai?.edit && !!jobId && scan?.status === 'done';
  // Създаден от видео, чиито кадри още са на сървъра: може да се добави липсваща страна.
  const [openedAt] = useState(() => Date.now());
  const canExtend = !!jobId && scan?.status === 'done' && !scan.report?.edit && openedAt - scan.createdAt < FRAMES_KEPT_MS;

  useEffect(() => {
    if (arDone) arSupport().then(setSupport);
  }, [arDone]);

  // Обем и повърхност за оценката на печата, за сканирания с отчет отпреди те да бъдат включени в него.
  const needGeometry = scan?.status === 'done' && !!jobId && !!scan.report && scan.report.volume_mm3 == null;
  useEffect(() => {
    if (!needGeometry) return;
    let cancelled = false;
    getGeometry(jobId!).then(
      (g) => {
        const now = getScan(id);
        if (!cancelled && now?.report) setScan(updateScan(id, { report: { ...now.report, volume_mm3: g.volume_mm3, area_mm2: g.area_mm2 } }));
      },
      () => {}, // по-стар сървър: без оценка
    );
    return () => {
      cancelled = true;
    };
  }, [needGeometry, jobId, id]);

  useEffect(() => {
    let cancelled = false;
    loadModel(id, jobId).then(
      (m) => !cancelled && setModel(m),
      (e) => !cancelled && setLoadError(`Моделът не се зареди. ${errorText(e)}`),
    );
    return () => {
      cancelled = true;
    };
  }, [id, jobId, attempt]);

  const mm = Number(mmInput.replace(',', '.'));
  const mmValid = Number.isFinite(mm) && mm >= 1 && mm <= 2000;
  const report = scan?.report;
  // Сървърът нормализира модела до най-дълга страна 100 мм; размерите се мащабират линейно. Двете страни на стъпката
  // идват в произволен ред: по-дългата се показва като дължина.
  const dims = (() => {
    if (!report || !mmValid) return null;
    const [x, y, z] = report.dimensions_mm.map((d) => (d * mm) / 100);
    return [Math.max(x, y), Math.min(x, y), z];
  })();

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

  const exportAs = (format: Format) =>
    run(format, async () => {
      if (!model || !mmValid) return;
      setScan(updateScan(id, { longestMm: mm }));
      let file = modelFile(id);
      if (format !== 'glb') {
        file = new File(Paths.cache, `scan-${id}-${Math.round(mm)}mm.${format}`);
        if (surface === 'smooth' && jobId) {
          if (file.exists) file.delete();
          await downloadResult(jobId, format, file, mm, SMOOTH_PASSES);
        } else file.writeSync(exportForPrint(model, format, mm));
      }
      const mime = FORMATS.find((f) => f.format === format)!.mime;
      await Sharing.shareAsync(file.uri, { mimeType: mime, dialogTitle: 'Изнасяне на 3D модел' });
    });

  const viewAr = () => {
    if (blocked(arAllowed)) return Alert.alert('Виж в стаята', allowanceNote(arAllowed));
    run('ar', async () => {
      if (!mmValid || !scan) return;
      setScan(updateScan(id, { longestMm: mm }));
      await viewInRoom(jobId!, mm, scanTitle(scan), realSize);
      if (arAllowed && !arAllowed.unlimited && !arAllowed.counted) refreshProfile().catch(() => {}); // остава един по-малко
    });
  };

  const stressTest = () => {
    if (blocked(stressAllowed)) return Alert.alert('Тест за здравина', allowanceNote(stressAllowed));
    router.push({ pathname: '/stress/[id]', params: { id: jobId!, source: 'scan', mm: String(mm), title: scanTitle(scan!) } });
  };

  // Pro: нов модел, създаден от този чрез текстова промяна; следи се на екрана за обработка като сканиране.
  const edit = (prompt: string) =>
    run('edit', async () => {
      if (!mmValid || !scan) return;
      const title = `${scanTitle(scan)} · промяна`.slice(0, 100);
      const job = await editModel(jobId!, prompt, mm, title, spot);
      const next = createScan(currentUserId() ?? undefined, 'edit');
      updateScan(next.id, { jobId: job.id, status: job.state, title });
      setEditing(false);
      setEditText('');
      setSpot(null);
      router.push({ pathname: '/processing/[id]', params: { id: next.id } });
    });

  const nameplate = (text: string, shape: 'round' | 'square') =>
    run('plate', async () => {
      if (!mmValid || !scan) return;
      const newTitle = `${scanTitle(scan)} · ${text}`.slice(0, 100);
      const job = await nameplateModel(jobId!, text, shape, mm, newTitle);
      const next = createScan(currentUserId() ?? undefined, 'edit');
      updateScan(next.id, { jobId: job.id, status: job.state, title: newTitle });
      setPlating(false);
      router.push({ pathname: '/processing/[id]', params: { id: next.id } });
    });

  const extend = () =>
    Alert.alert('Добавяне на липсваща страна', 'Заснемете само това, което липсва. Предметът трябва да е на същото място. Този модел остава.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Заснеми',
        onPress: () =>
          router.push({
            pathname: '/capture',
            params: { extend: jobId!, title: `${scanTitle(scan!)} · допълнен`.slice(0, 100), ...(report?.ai === 'single' && { ai: '1' }) },
          }),
      },
    ]);

  const shareVideo = () =>
    run('spin', async () => {
      const file = new File(Paths.cache, `orbitize-${id}.mp4`);
      if (file.exists) file.delete();
      await downloadSpin(jobId!, file);
      await Sharing.shareAsync(file.uri, { mimeType: 'video/mp4', dialogTitle: 'Споделяне на видео' });
    });

  const rename = (title: string) =>
    run('rename', async () => {
      if (account) await renameScan(jobId!, title);
      setScan(updateScan(id, { title: title || undefined }));
      setRenaming(false);
    });

  const publish = () =>
    Alert.alert('Публикуване в общността?', 'Всички ще виждат модела и името ви.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Публикувай',
        onPress: () =>
          run('publish', async () => {
            const post = await publishScan(jobId!, scanTitle(scan!), mm);
            setScan(updateScan(id, { published: true, title: post.title, longestMm: mm }));
            checkProgress(true);
          }),
      },
    ]);

  const unpublish = () =>
    Alert.alert('Махане от общността?', 'Моделът остава при вас. Харесванията се изтриват.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Махни',
        style: 'destructive',
        onPress: () =>
          run('publish', async () => {
            await unpublishScan(jobId!);
            setScan(updateScan(id, { published: undefined }));
          }),
      },
    ]);

  const sendLink = (slug: string) => Share.share({ message: shareUrl(slug) }).catch(() => {});

  const share = () =>
    run('share', async () => {
      const { share: slug } = await shareScan(jobId!);
      setScan(updateScan(id, { share: slug }));
      await sendLink(slug);
    });

  const unshare = () =>
    Alert.alert('Спиране на споделянето?', 'Линкът спира да работи.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Спри',
        style: 'destructive',
        onPress: () =>
          run('share', async () => {
            await unshareScan(jobId!);
            setScan(updateScan(id, { share: undefined }));
          }),
      },
    ]);

  const remove = () =>
    Alert.alert('Изтриване на сканирането?', 'Това не може да се върне.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Изтрий',
        style: 'destructive',
        onPress: () =>
          run('delete', async () => {
            await deleteEntry({ key: id, scan: scan!, jobId, title: '', createdAt: 0, status: 'done', thumb: false });
            router.back();
          }),
      },
    ]);

  if (!scan)
    return (
      <SafeAreaView style={[s.fill, s.center]}>
        <H2>Сканирането не е намерено.</H2>
        <Button title="Назад" variant="secondary" onPress={() => router.back()} />
      </SafeAreaView>
    );

  const title = scanTitle(scan);
  const done = scan.status === 'done';
  const kind = badgeOf(scan.status);
  // Кой от двата начина го е създал; безплатният профил има само един, затова няма какво да се различава.
  const way = report?.ai ? madeWith(report.ai) : report && !report.edit && user?.pro ? SCAN_INFO.label : null;
  const tabs: { value: Tab; label: string }[] = [
    { value: 'size', label: 'Размер' },
    { value: 'files', label: 'Файлове' },
  ];
  if (account && done) tabs.push({ value: 'share', label: 'Споделяне' });

  return (
    <View style={s.fill}>
      <View style={s.preview}>
        {model ? (
          <Animated.View entering={fade} style={s.flex}>
            {!full && !picking && <ModelView model={model} wireframe={wireframe} background={c.viewer} />}
          </Animated.View>
        ) : loadError ? (
          <View style={s.center}>
            <ErrorText>{loadError}</ErrorText>
            <Button
              title="Опитай отново"
              icon="refresh"
              variant="secondary"
              onPress={() => {
                setLoadError(null);
                setAttempt((a) => a + 1);
              }}
            />
          </View>
        ) : (
          <View style={s.loading}>
            <Skeleton style={s.loadingBlock} />
            <Label>Зареждане на модела…</Label>
          </View>
        )}
        <SafeAreaView edges={['top']} style={s.floatTop} pointerEvents="box-none">
          <IconButton icon="chevron-back" label="Назад" onPress={() => router.back()} />
          {model && <IconButton icon="expand-outline" label="Цял екран" onPress={() => setFull(true)} />}
        </SafeAreaView>
      </View>

      <Screen style={s.content}>
        <Appear>
          <Pressable style={s.titleRow} onPress={() => setRenaming(true)} accessibilityRole="button" accessibilityLabel="Преименуване">
            <Title style={s.flex} numberOfLines={2}>
              {title}
            </Title>
            <Ionicons name="pencil-outline" size={16} color={c.muted} style={s.pencil} />
          </Pressable>
          <View style={s.metaLine}>
            {kind !== 'done' && <Badge kind={kind} />}
            <Text style={s.info}>
              {way ? `${way} · ` : ''}
              {shortDate(scan.createdAt)}
            </Text>
          </View>
        </Appear>

        {showAr && (
          <Appear index={1} style={s.withNote}>
            <Button
              title="Виж в стаята"
              icon="cube-outline"
              size="lg"
              onPress={viewAr}
              loading={busy === 'ar'}
              disabled={!mmValid || (!!busy && busy !== 'ar')}
            />
            {allowanceNote(arAllowed) && <Small style={s.noteText}>{allowanceNote(arAllowed)}</Small>}
          </Appear>
        )}

        {done && !!jobId && (
          <Appear index={2} style={s.withNote}>
            <Button title="Тест за здравина" icon="barbell-outline" variant="secondary" size="lg" disabled={!mmValid} onPress={stressTest} />
            {allowanceNote(stressAllowed) && <Small style={s.noteText}>{allowanceNote(stressAllowed)}</Small>}
          </Appear>
        )}

        {canEdit && (
          <Appear index={2}>
            <Button title="Промени с AI" icon="color-wand-outline" variant="secondary" size="lg" onPress={() => setEditing(true)} disabled={!mmValid || !!busy} />
          </Appear>
        )}

        {canEdit && (
          <Appear index={2}>
            <Button title="Табелка с име" icon="pricetag-outline" variant="secondary" size="lg" onPress={() => setPlating(true)} disabled={!mmValid || !!busy} />
          </Appear>
        )}

        {canExtend && (
          <Appear index={2}>
            <Button title="Добави липсваща страна" icon="add-circle-outline" variant="secondary" size="lg" onPress={extend} disabled={!!busy} />
          </Appear>
        )}

        {error && <ErrorText>{error}</ErrorText>}

        {report?.cad && !!jobId && done && (
          <Appear index={3}>
            <CadPanel scanId={id} jobId={jobId} cad={report.cad} stamp={String(report.cad.volume_mm3)} />
          </Appear>
        )}

        <Appear index={3}>
          <Segmented<Tab> value={tab} onChange={setTab} options={tabs} />
        </Appear>

        <Animated.View key={tab} entering={swap} style={s.tab}>
          {tab === 'size' && (
            <Section label="Реален размер">
              <Card style={s.list}>
                <View style={s.mmRow}>
                  <H3 style={s.flex}>Най-дълга страна</H3>
                  <View style={[s.mmBox, !mmValid && s.mmInvalid]}>
                    <Field
                      style={s.mm}
                      value={mmInput}
                      onChangeText={setMmInput}
                      keyboardType="decimal-pad"
                      accessibilityLabel="Най-дълга страна в милиметри"
                      selectTextOnFocus
                    />
                    <Text style={s.mmUnit}>мм</Text>
                  </View>
                </View>
                {!mmValid && <Small style={s.danger}>Въведете размер от 1 до 2000 мм.</Small>}
                {model && mmValid && (
                  <View style={s.measure}>
                    <Button title="Измери на модела" icon="resize-outline" variant="ghost" onPress={() => setMeasuring(true)} />
                  </View>
                )}
                {dims && (
                  <View style={s.dims}>
                    {(['Дължина', 'Ширина', 'Височина'] as const).map((label, i) => (
                      <View key={label} style={[s.dim, i > 0 && s.cellLine]}>
                        <Text style={s.dimValue}>{mmText(dims[i])}</Text>
                        <Label>{label}</Label>
                      </View>
                    ))}
                  </View>
                )}
                {showAr && (
                  <View style={s.lineTop}>
                    <Row
                      label="Реален размер в стаята"
                      last
                      right={
                        <Switch
                          value={realSize}
                          onValueChange={setRealSize}
                          trackColor={{ true: c.accentStrong, false: c.surface3 }}
                          thumbColor="#ffffff"
                          ios_backgroundColor={c.surface3}
                          accessibilityLabel="Реален размер в AR"
                        />
                      }
                    />
                  </View>
                )}
              </Card>
            </Section>
          )}

          {tab === 'size' && report && mmValid && <PrintCard report={report} mm={mm} />}

          {tab === 'files' && (
            <>
              <Section label={`Изтегляне · ${mmValid ? `${mmText(mm)} мм` : 'в милиметри'}`}>
                <View style={s.formats}>
                  {FORMATS.map((f, i) => {
                    const off = !model || !mmValid || (!!busy && busy !== f.format);
                    const rec = f.format === 'stl';
                    return (
                      <PressScale
                        key={f.format}
                        onPress={() => exportAs(f.format)}
                        disabled={off || !!busy}
                        style={[s.format, rec && s.formatRec, i === FORMATS.length - 1 && s.formatWide, off && s.dim40]}
                        accessibilityRole="button"
                        accessibilityLabel={`Изнеси ${f.format.toUpperCase()}`}
                      >
                        {rec && <Text style={s.formatBest}>За печат</Text>}
                        <View style={s.formatHead}>
                          <Text style={s.formatName}>{f.format.toUpperCase()}</Text>
                          {busy === f.format ? (
                            <ActivityIndicator size="small" color={c.muted} />
                          ) : (
                            <Ionicons name="arrow-down" size={16} color={rec ? c.accent : c.muted} />
                          )}
                        </View>
                        {!!f.note && <Small>{f.note}</Small>}
                      </PressScale>
                    );
                  })}
                </View>
              </Section>

              {!!jobId && done && (
                <Section label="Повърхност">
                  <Segmented<Surface>
                    value={surface}
                    onChange={setSurface}
                    options={[
                      { value: 'smooth', label: 'Гладка' },
                      { value: 'detail', label: 'Детайлна' },
                    ]}
                  />
                </Section>
              )}

              {!!jobId && done && mmValid && <PrintHelper jobId={jobId} mm={mm} />}
            </>
          )}

          {tab === 'share' && (
            <>
              <Section label="Видео">
                <Button title="Видео за споделяне" icon="videocam-outline" variant="secondary" onPress={shareVideo} loading={busy === 'spin'} disabled={!!busy && busy !== 'spin'} />
                {busy === 'spin' && <Small>Подготвя се, около 2 минути.</Small>}
              </Section>

              <Section label="Линк">
                {scan.share ? (
                  <View style={s.buttons}>
                    <Button title="Изпрати линк" icon="share-outline" onPress={() => sendLink(scan.share!)} style={s.flex} />
                    <Button title="Спри" variant="danger" onPress={unshare} loading={busy === 'share'} />
                  </View>
                ) : (
                  <Button title="Създай линк" icon="link-outline" variant="secondary" onPress={share} loading={busy === 'share'} />
                )}
              </Section>

              <Section label="Общност">
                {scan.published ? (
                  <View style={s.buttons}>
                    <Button
                      title="Виж го там"
                      icon="people-outline"
                      variant="secondary"
                      onPress={() => router.push({ pathname: '/community/[id]', params: { id: jobId! } })}
                      style={s.flex}
                    />
                    <Button title="Махни" variant="danger" onPress={unpublish} loading={busy === 'publish'} />
                  </View>
                ) : (
                  <Button
                    title="Публикувай в общността"
                    icon="people-outline"
                    onPress={publish}
                    loading={busy === 'publish'}
                    disabled={!mmValid || (!!busy && busy !== 'publish')}
                  />
                )}
              </Section>
            </>
          )}
        </Animated.View>

        {!!report?.edit?.changes.length && (
          <Section label="Направени промени">
            <Card style={s.share}>
              {report.edit.changes.map((line) => (
                <Small key={line}>• {line}</Small>
              ))}
            </Card>
          </Section>
        )}

        {report?.object_width != null && report.object_width < SMALL_OBJECT && (
          <Note warn>Предметът беше далеч от камерата. Следващия път се приближете.</Note>
        )}

        {/* Само за собственика на приложението: как е създаден моделът и колко време е отнело. */}
        {!!user?.owner && !!report && (
          <Section label="Собственик">
            <Card style={s.list}>
              <Row label="Начин" value={report.ai ? madeWith(report.ai) : report.edit ? 'Промяна с AI' : 'Сканиране'} last={report.seconds == null} />
              {report.seconds != null && <Row label="Време за обработка" value={duration(report.seconds)} last />}
            </Card>
          </Section>
        )}

        <Button title="Изтрий сканирането" variant="danger" icon="trash-outline" onPress={remove} loading={busy === 'delete'} />
      </Screen>

      {model && (
        <FullscreenModel
          visible={full}
          model={model}
          title={title}
          wireframe={wireframe}
          onWireframe={() => setWireframe((w) => !w)}
          onClose={() => setFull(false)}
        />
      )}
      <RenameDialog visible={renaming} initial={scan.title ?? ''} saving={busy === 'rename'} onClose={() => setRenaming(false)} onSave={rename} />
      <PlateDialog visible={plating} saving={busy === 'plate'} error={plating ? error : null} onClose={() => setPlating(false)} onSave={nameplate} />
      {model && mmValid && (
        <MeasureTool
          visible={measuring}
          model={model}
          mm={mm}
          onClose={() => setMeasuring(false)}
          onApply={(longest) => {
            const rounded = Math.round(longest * 10) / 10;
            setMmInput(String(rounded));
            setScan(updateScan(id, { longestMm: rounded }));
            setMeasuring(false);
          }}
        />
      )}
      <EditDialog
        visible={editing}
        saving={busy === 'edit'}
        error={editing ? error : null}
        value={editText}
        onChange={setEditText}
        spot={spot}
        onPickSpot={
          model
            ? () => {
                setEditing(false);
                setPicking(true);
              }
            : undefined
        }
        onClearSpot={() => setSpot(null)}
        onClose={() => setEditing(false)}
        onSave={edit}
      />
      {model && (
        <SpotPicker
          visible={picking}
          model={model}
          initial={spot}
          onClose={() => {
            setPicking(false);
            setEditing(true);
          }}
          onDone={(picked) => {
            setSpot(picked);
            setPicking(false);
            setEditing(true);
          }}
        />
      )}
    </View>
  );
}

function PlateDialog(props: { visible: boolean; saving: boolean; error: string | null; onClose: () => void; onSave: (text: string, shape: 'round' | 'square') => void }) {
  const s = useStyles();
  const [value, setValue] = useState('');
  const [shape, setShape] = useState<'round' | 'square'>('square');
  const name = value.trim();
  return (
    <Modal visible={props.visible} transparent animationType="fade" onRequestClose={props.onClose}>
      <View style={s.backdrop}>
        <Card style={s.dialog}>
          <H2>Табелка с име</H2>
          <Field value={value} onChangeText={setValue} autoFocus maxLength={40} placeholder="напр. Иван" accessibilityLabel="Име на табелката" />
          <Segmented
            value={shape}
            onChange={setShape}
            options={[
              { value: 'square', label: 'Правоъгълна' },
              { value: 'round', label: 'Кръгла' },
            ]}
          />
          {props.error && <ErrorText>{props.error}</ErrorText>}
          <View style={s.buttons}>
            <Button title="Отказ" variant="secondary" onPress={props.onClose} style={s.flex} />
            <Button title="Направи" onPress={() => props.onSave(name, shape)} loading={props.saving} disabled={!name} style={s.flex} />
          </View>
        </Card>
      </View>
    </Modal>
  );
}

function RenameDialog(props: { visible: boolean; initial: string; saving: boolean; onClose: () => void; onSave: (t: string) => void }) {
  const s = useStyles();
  const [value, setValue] = useState(props.initial);
  return (
    <Modal visible={props.visible} transparent animationType="fade" onRequestClose={props.onClose} onShow={() => setValue(props.initial)}>
      <View style={s.backdrop}>
        <Card style={s.dialog}>
          <H2>Име на сканирането</H2>
          <Field value={value} onChangeText={setValue} autoFocus maxLength={100} placeholder="напр. Бормашина" accessibilityLabel="Име на сканирането" />
          <View style={s.buttons}>
            <Button title="Отказ" variant="secondary" onPress={props.onClose} style={s.flex} />
            <Button title="Запази" onPress={() => props.onSave(value.trim())} loading={props.saving} style={s.flex} />
          </View>
        </Card>
      </View>
    </Modal>
  );
}

// Заявки, които сървърът разбира точно (сървър app/edit.py, правила); останалите отиват при езиковия му модел. При
// посочено място върху модела "here" и "this" означават него.
const EXAMPLES = ['make the hair blue', 'make it 15 cm tall', 'add a keychain loop', 'hollow it with 2 mm walls'];
const SPOT_EXAMPLES = ['make this red', 'smooth here', 'add a 3 mm hole here'];

type EditDialogProps = {
  visible: boolean;
  saving: boolean;
  error: string | null;
  value: string;
  onChange: (text: string) => void;
  spot: Spot | null;
  onPickSpot?: () => void; // липсва, докато моделът не се зареди
  onClearSpot: () => void;
  onClose: () => void;
  onSave: (prompt: string) => void;
};

function EditDialog(props: EditDialogProps) {
  const s = useStyles();
  const c = useColors();
  const prompt = props.value.trim();
  return (
    <Modal visible={props.visible} transparent animationType="fade" onRequestClose={props.onClose}>
      <View style={s.backdrop}>
        <Card style={s.dialog}>
          <H2>Промени с AI</H2>
          <Small>Опишете промяната на английски.</Small>
          <Field
            value={props.value}
            onChangeText={props.onChange}
            autoFocus
            multiline
            maxLength={500}
            autoCapitalize="none"
            placeholder={props.spot ? 'smooth here and make this red' : 'make the hair blue and add a keychain loop'}
            accessibilityLabel="Каква промяна"
            style={s.prompt}
          />
          {props.spot ? (
            <View style={s.spotRow}>
              <Ionicons name="locate" size={18} color={c.accent} />
              <Text style={s.spotText}>Мястото е посочено</Text>
              {props.onPickSpot && <Button title="Смени" variant="secondary" onPress={props.onPickSpot} />}
              <IconButton icon="close" label="Махни посоченото място" onPress={props.onClearSpot} />
            </View>
          ) : (
            props.onPickSpot && <Button title="Посочи място на модела" icon="locate-outline" variant="secondary" onPress={props.onPickSpot} />
          )}
          <View style={s.examples}>
            {(props.spot ? SPOT_EXAMPLES : EXAMPLES).map((ex) => (
              <Pressable key={ex} onPress={() => props.onChange(ex)} style={s.example} accessibilityRole="button" accessibilityLabel={ex}>
                <Text style={s.exampleText}>{ex}</Text>
              </Pressable>
            ))}
          </View>
          {props.error && <ErrorText>{props.error}</ErrorText>}
          <View style={s.buttons}>
            <Button title="Отказ" variant="secondary" onPress={props.onClose} style={s.flex} />
            <Button title="Направи" onPress={() => props.onSave(prompt)} loading={props.saving} disabled={prompt.length < 2} style={s.flex} />
          </View>
        </Card>
      </View>
    </Modal>
  );
}

/* Колко време отнема печатът, пластмасата и цената при зададения по-горе размер (lib/printEstimate.ts). */
function PrintCard({ report, mm }: { report: Report; mm: number }) {
  const s = useStyles();
  const [material, setMaterial] = useState<Material>('pla');
  const [infill, setInfill] = useState<Infill>(20);
  const [priceInput, setPriceInput] = useState(() => String(getPrices().pla));
  const price = Number(priceInput.replace(',', '.'));
  const priceValid = Number.isFinite(price) && price >= 0 && price <= 1000;

  const pick = (m: Material) => {
    setMaterial(m);
    setPriceInput(String(getPrices()[m]));
  };
  const changePrice = (text: string) => {
    setPriceInput(text);
    const v = Number(text.replace(',', '.'));
    if (Number.isFinite(v) && v >= 0 && v <= 1000) setPrice(material, v); // следващите модели започват с него
  };

  if (report.volume_mm3 == null || report.area_mm2 == null)
    return (
      <Section label="Печат · ориентировъчно">
        <Small>Изчисляване…</Small>
      </Section>
    );
  const e = estimate(report.volume_mm3, report.area_mm2, report.dimensions_mm[2], mm, material, infill, priceValid ? price : 0);
  return (
    <Section label="Печат · ориентировъчно">
      <Segmented<Material> value={material} onChange={pick} options={(Object.keys(MATERIALS) as Material[]).map((m) => ({ value: m, label: MATERIALS[m].label }))} />
      {material !== 'resin' && (
        <Segmented<string> value={String(infill)} onChange={(v) => setInfill(Number(v) as Infill)} options={INFILLS.map((p) => ({ value: String(p), label: `Пълнеж ${p}%` }))} />
      )}
      <Card style={s.list}>
        <Row label="Време за печат" value={`~${printTime(e.seconds)}`} />
        <Row label="Пластмаса" value={`${Math.max(1, Math.round(e.grams))} г`} />
        <Row
          label="Цена"
          value={priceValid ? `~${euro(e.plasticEur + e.powerEur)}` : '—'}
        />
        <View style={s.mmRow}>
          <H3 style={s.flex}>Цена на пластмасата</H3>
          <View style={[s.mmBox, !priceValid && s.mmInvalid]}>
            <Field
              style={s.mm}
              value={priceInput}
              onChangeText={changePrice}
              keyboardType="decimal-pad"
              accessibilityLabel="Цена на пластмасата в евро за килограм"
              selectTextOnFocus
            />
            <Text style={s.mmUnit}>€/кг</Text>
          </View>
        </View>
      </Card>
    </Section>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.md, padding: space.xl },
  preview: { height: '42%', backgroundColor: c.viewer[1], borderBottomWidth: 1, borderBottomColor: c.line },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.md },
  loadingBlock: { width: 120, height: 120, borderRadius: radius.md },
  floatTop: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: space.lg,
    paddingTop: space.sm,
  },
  content: { paddingTop: space.xl, gap: 20 },
  titleRow: { flexDirection: 'row', alignItems: 'flex-start', gap: space.sm },
  pencil: { marginTop: 12 },
  metaLine: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 12, flexWrap: 'wrap' },
  info: { ...text.mono, fontSize: 12, color: c.muted, letterSpacing: 0.2 },
  cellLine: { borderLeftWidth: 1, borderLeftColor: c.line },
  tab: { gap: space.xl },
  list: { paddingVertical: 0 },
  lineTop: { borderTopWidth: 1, borderTopColor: c.line },
  mmRow: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  mmBox: {
    width: 112,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: c.surface2,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.lineStrong,
    paddingRight: space.md,
  },
  mmInvalid: { borderColor: c.danger },
  mm: { flex: 1, minWidth: 0, textAlign: 'right', fontFamily: font.monoMedium, fontSize: 16, backgroundColor: 'transparent', borderWidth: 0, paddingRight: 6 },
  mmUnit: { fontFamily: font.mono, fontSize: 13, color: c.muted },
  measure: { borderTopWidth: 1, borderTopColor: c.line, paddingVertical: space.xs },
  danger: { color: c.danger, paddingBottom: space.sm },
  dims: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: c.line, marginHorizontal: -space.lg },
  dim: { flex: 1, alignItems: 'center', gap: space.xs, paddingVertical: space.md },
  dimValue: { fontFamily: font.monoMedium, fontSize: 17, color: c.text },
  spotRow: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  spotText: { flex: 1, fontFamily: font.medium, fontSize: 14, color: c.text },
  formats: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  format: {
    flexBasis: '48%',
    flexGrow: 1,
    gap: 2,
    paddingVertical: 12,
    paddingHorizontal: 14,
    backgroundColor: c.surface2,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.sm,
  },
  formatRec: { backgroundColor: c.accentSoft, borderColor: c.accentLine },
  formatWide: { flexBasis: '100%' },
  formatBest: { fontFamily: font.monoMedium, fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase', color: c.accent, marginBottom: 4 },
  formatHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  formatName: { fontFamily: font.monoMedium, fontSize: 16, color: c.text },
  dim40: { opacity: 0.4 },
  share: { gap: space.md },
  withNote: { gap: space.sm },
  noteText: { textAlign: 'center' },
  prompt: { minHeight: 72, textAlignVertical: 'top' },
  examples: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  example: { borderWidth: 1, borderColor: c.lineStrong, borderRadius: 999, paddingHorizontal: 10, paddingVertical: 5 },
  exampleText: { fontFamily: font.mono, fontSize: 11, color: c.text2 },
  buttons: { flexDirection: 'row', gap: space.sm },
  backdrop: { flex: 1, backgroundColor: c.backdrop, justifyContent: 'center', padding: space.xl },
  dialog: { gap: space.md },
}));

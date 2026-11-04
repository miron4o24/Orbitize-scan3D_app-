import Ionicons from '@expo/vector-icons/Ionicons';
import { File, Paths } from 'expo-file-system';
import { router, useLocalSearchParams, type ErrorBoundaryProps } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import type { Object3D } from 'three';

import { FullscreenModel } from '@/components/FullscreenModel';
import { ModelView } from '@/components/ModelView';
import { Appear, fade, PressScale, Skeleton } from '@/components/motion';
import { AuthorLine, LikeButton } from '@/components/PostCard';
import { PrintPhotos } from '@/components/PrintPhotos';
import { Button, Card, ErrorText, IconButton, Label, Section, Small, Title } from '@/components/ui';
import { ApiError, downloadPost, errorText, getPost, likePost, NOT_AVAILABLE, reportPost, unpublishScan, type Post } from '@/lib/api';
import { exportForPrint } from '@/lib/exportModel';
import { parseGlb } from '@/lib/model';
import { checkProgress } from '@/lib/progress';
import { listScans, updateScan } from '@/lib/scans';
import { font, makeStyles, radius, space, useColors } from '@/theme';

// Модел от общността: завъртане, харесване и изтегляне в размера, зададен от автора.

const FORMATS = [
  { format: 'stl', mime: 'model/stl', note: '' },
  { format: 'obj', mime: 'model/obj', note: 'За 3D програми' },
  { format: 'glb', mime: 'model/gltf-binary', note: 'С цветовете' },
] as const;
type Format = (typeof FORMATS)[number]['format'];

const mmText = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

async function loadModel(id: string): Promise<Object3D> {
  const file = new File(Paths.cache, `post-${id}.glb`);
  if (!file.exists) await downloadPost(id, file);
  try {
    return await parseGlb(file, `post-${id}`);
  } catch (e) {
    file.delete(); // вероятно прекъснато изтегляне; повторният опит го изтегля отново
    throw e;
  }
}

export function ErrorBoundary({ retry }: ErrorBoundaryProps) {
  const s = useStyles();
  return (
    <View style={[s.fill, s.center]}>
      <ErrorText>Моделът не може да се покаже.</ErrorText>
      <Button title="Опитай отново" icon="refresh" onPress={retry} />
    </View>
  );
}

export default function CommunityPost() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const c = useColors();
  const s = useStyles();
  const [post, setPost] = useState<Post | null>(null);
  const [gone, setGone] = useState<string | null>(null); // публикацията не можа да се получи
  const [model, setModel] = useState<Object3D | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [wireframe, setWireframe] = useState(false);
  const [full, setFull] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getPost(id).then(
      (p) => !cancelled && setPost(p),
      (e) => !cancelled && setGone(e instanceof ApiError && e.status === 404 && e.message !== NOT_AVAILABLE ? 'Моделът вече не е в общността.' : errorText(e)),
    );
    return () => {
      cancelled = true;
    };
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    loadModel(id).then(
      (m) => !cancelled && setModel(m),
      (e) => !cancelled && setLoadError(`Моделът не се зареди. ${errorText(e)}`),
    );
    return () => {
      cancelled = true;
    };
  }, [id, attempt]);

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

  // Веднага на екрана; после идва броят от сървъра (или се връща старото състояние).
  async function toggleLike() {
    if (!post) return;
    const before = post;
    const on = !post.liked;
    setPost({ ...post, liked: on, likes: post.likes + (on ? 1 : -1) });
    try {
      const r = await likePost(post.id, on);
      setPost((p) => p && { ...p, ...r });
      if (on) checkProgress(); // харесването се брои за някои седмични предизвикателства
    } catch (e) {
      setPost(before);
      setError(errorText(e));
    }
  }

  const exportAs = (format: Format) =>
    run(format, async () => {
      if (!model || !post) return;
      // Изнасянето е с пълните детайли, а показваният модел е лекият файл за преглед.
      const full = new File(Paths.cache, `post-${id}-full.glb`);
      if (!full.exists) await downloadPost(id, full, false);
      let file = full;
      if (format !== 'glb') {
        file = new File(Paths.cache, `post-${id}-${Math.round(post.longest_mm)}mm.${format}`);
        file.writeSync(exportForPrint(await parseGlb(full, `post-${id}-full`), format, post.longest_mm));
      }
      const mime = FORMATS.find((f) => f.format === format)!.mime;
      await Sharing.shareAsync(file.uri, { mimeType: mime, dialogTitle: 'Изнасяне на 3D модел' });
    });

  const report = (reason: string) =>
    run('report', async () => {
      await reportPost(id, reason);
      Alert.alert('Благодарим', 'Ще прегледаме модела.');
    });

  const askReport = () =>
    Alert.alert('Докладване на модела', 'Какъв е проблемът?', [
      { text: 'Отказ', style: 'cancel' },
      { text: 'Спам или реклама', onPress: () => report('spam') },
      { text: 'Неподходящо съдържание', onPress: () => report('inappropriate') },
    ]);

  const takeDown = () =>
    Alert.alert('Махане от общността?', 'Моделът остава при вас. Харесванията се изтриват.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Махни',
        style: 'destructive',
        onPress: () =>
          run('remove', async () => {
            await unpublishScan(id);
            const local = listScans().find((sc) => sc.jobId === id);
            if (local) updateScan(local.id, { published: undefined });
            router.back();
          }),
      },
    ]);

  const dims = (() => {
    if (!post?.dimensions_mm) return null;
    const [x, y, z] = post.dimensions_mm.map((d) => (d * post.longest_mm) / 100);
    return [Math.max(x, y), Math.min(x, y), z];
  })();

  return (
    <View style={s.fill}>
      <View style={s.preview}>
        {model ? (
          <Animated.View entering={fade} style={s.flex}>
            {!full && <ModelView model={model} wireframe={wireframe} background={c.viewer} />}
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

      <ScrollView style={s.flex} contentContainerStyle={s.content} showsVerticalScrollIndicator={false}>
        {gone ? (
          <ErrorText>{gone}</ErrorText>
        ) : !post ? (
          <View style={s.skeleton}>
            <Skeleton style={s.skeletonTitle} />
            <Skeleton style={s.skeletonLine} />
          </View>
        ) : (
          <>
            <Appear>
              <Title numberOfLines={2}>{post.title}</Title>
              <View style={s.byRow}>
                <View style={s.flex}>
                  <AuthorLine post={post} size={28} date />
                </View>
                <LikeButton liked={post.liked} likes={post.likes} onPress={toggleLike} big />
              </View>
            </Appear>

            {dims && (
              <Appear index={1}>
                <Card style={s.strip}>
                  {(['Дължина', 'Ширина', 'Височина'] as const).map((label, i) => (
                    <View key={label} style={[s.cell, i > 0 && s.cellLine]}>
                      <Label>{label}</Label>
                      <Text style={s.cellValue}>{mmText(dims[i])} мм</Text>
                    </View>
                  ))}
                </Card>
              </Appear>
            )}

            {error && <ErrorText>{error}</ErrorText>}

            <Appear index={2}>
              <Section label={`Изтегляне · ${mmText(post.longest_mm)} мм`}>
                <View style={s.formats}>
                  {FORMATS.map((f, i) => {
                    const off = !model || (!!busy && busy !== f.format);
                    const rec = f.format === 'stl';
                    return (
                      <PressScale
                        key={f.format}
                        onPress={() => exportAs(f.format)}
                        disabled={off || !!busy}
                        style={[s.format, rec && s.formatRec, i === FORMATS.length - 1 && s.formatWide, off && s.dim40]}
                        accessibilityRole="button"
                        accessibilityLabel={`Изтегли ${f.format.toUpperCase()}`}
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
            </Appear>

            <Appear index={3}>
              <PrintPhotos postId={post.id} mine={post.mine} />
            </Appear>

            <Appear index={3}>
              <Button
                title="Тест за здравина"
                icon="barbell-outline"
                variant="secondary"
                size="lg"
                onPress={() => router.push({ pathname: '/stress/[id]', params: { id, source: 'post', mm: String(post.longest_mm), title: post.title } })}
              />
            </Appear>

            <Appear index={4}>
              {post.mine ? (
                <Button title="Махни от общността" variant="danger" icon="eye-off-outline" onPress={takeDown} loading={busy === 'remove'} />
              ) : (
                <Button title="Докладвай" variant="ghost" icon="flag-outline" onPress={askReport} loading={busy === 'report'} />
              )}
            </Appear>
          </>
        )}
      </ScrollView>

      {model && (
        <FullscreenModel
          visible={full}
          model={model}
          title={post?.title ?? ''}
          wireframe={wireframe}
          onWireframe={() => setWireframe((w) => !w)}
          onClose={() => setFull(false)}
        />
      )}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.md, padding: space.xl },
  preview: { height: '46%', backgroundColor: c.viewer[1], borderBottomWidth: 1, borderBottomColor: c.line },
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
  content: { padding: 18, paddingTop: space.xl, paddingBottom: 48, gap: 20 },
  byRow: { flexDirection: 'row', alignItems: 'center', gap: space.md, marginTop: space.md },
  skeleton: { gap: space.md },
  skeletonTitle: { height: 30, width: '70%' },
  skeletonLine: { height: 14, width: '45%' },
  strip: { flexDirection: 'row', padding: 0, borderRadius: radius.sm },
  cell: { flex: 1, gap: 6, paddingVertical: 12, paddingHorizontal: 14 },
  cellLine: { borderLeftWidth: 1, borderLeftColor: c.line },
  cellValue: { fontFamily: font.monoMedium, fontSize: 15, lineHeight: 21, color: c.text },
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
}));

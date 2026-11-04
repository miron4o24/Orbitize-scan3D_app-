import Ionicons from '@expo/vector-icons/Ionicons';
import { File, Paths } from 'expo-file-system';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { router, useLocalSearchParams, type ErrorBoundaryProps } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  Box3,
  BoxGeometry,
  BufferAttribute,
  Color,
  ConeGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  SphereGeometry,
  SRGBColorSpace,
  Vector3,
  type BufferGeometry,
  type Object3D,
} from 'three';

import { ModelView } from '@/components/ModelView';
import { Appear, fade, haptic, Skeleton, swap } from '@/components/motion';
import { Button, Card, ErrorText, Field, H3, IconButton, Kicker, Label, Note, Section, Segmented, Small, Title, type IconName } from '@/components/ui';
import { downloadStressModel, errorText, stressSummary, type StressCase, type StressSource, type StressSummary, type Vec3 } from '@/lib/api';
import { refreshProfile } from '@/lib/auth';
import { parseGlb } from '@/lib/model';
import { checkProgress } from '@/lib/progress';
import {
  CASES,
  HEIGHTS,
  INFILLS,
  LEGEND,
  limitText,
  loadText,
  MATERIALS,
  outcome,
  stressColor,
  WEIGHTS,
  type Infill,
  type Material,
  type Verdict,
} from '@/lib/stress';
import { font, makeStyles, radius, space, useColors } from '@/theme';

// The stress test: will the printed object hold? The server works out the stresses for each kind of load once per
// model (backend app/stress.py); here they are scaled to the weight or height, the plastic and the real size picked,
// and painted onto the model, blue (calm) to red (breaks), live as the choices change.

// painted: the share of the strength per server MPa it is coloured for now.
type Loaded = { summary: StressSummary; model: Object3D; values: ArrayLike<number>; colors: BufferAttribute; painted?: number };

const DEFAULT_LOAD: Record<StressCase, number> = { top: 2, hang: 1, side: 0.5, drop: 1 };

const VERDICTS: Record<Verdict, { title: string; icon: IconName }> = {
  safe: { title: 'Издържа спокойно', icon: 'shield-checkmark-outline' },
  edge: { title: 'На ръба', icon: 'warning-outline' },
  break: { title: 'Ще се счупи', icon: 'close-circle-outline' },
};

/** An arrow along +y from the origin, `length` long. */
function arrow(length: number, size: number, color: string) {
  const head = 0.07 * size;
  const material = new MeshBasicMaterial({ color });
  const shaft = new Mesh(new CylinderGeometry(0.012 * size, 0.012 * size, length - head, 12), material);
  shaft.position.y = (length - head) / 2;
  const tip = new Mesh(new ConeGeometry(0.035 * size, head, 16), material);
  tip.position.y = length - head / 2;
  const g = new Group();
  g.add(shaft, tip);
  return g;
}

/** The base it is held by, the force's arrow and the weakest point, around the model. */
function addGuides(group: Group, model: Object3D, s: StressSummary, accent: string) {
  const box = new Box3().setFromObject(model);
  const dims = box.getSize(new Vector3());
  const size = Math.max(dims.x, dims.y, dims.z);
  const centre = box.getCenter(new Vector3());

  const plateH = 0.015 * size;
  const plate = new Mesh(
    new BoxGeometry(dims.x * 1.15, plateH, dims.z * 1.15),
    new MeshStandardMaterial({ color: '#6b7280', roughness: 0.9, metalness: 0, transparent: true, opacity: 0.85 }),
  );
  plate.position.set(centre.x, box.min.y - plateH / 2, centre.z);
  group.add(plate);

  const length = 0.28 * size;
  const gap = 0.02 * size;
  const up = new Vector3(0, 1, 0);
  let dir: Vector3;
  let tail: Vector3;
  if (s.case === 'drop') {
    dir = new Vector3(0, -1, 0); // the fall: over the object, pointing down
    tail = new Vector3(centre.x, box.max.y + gap + length, centre.z);
  } else {
    dir = new Vector3(...(s.load_dir ?? [0, -1, 0])).normalize();
    const at = new Vector3(...(s.load_at ?? [centre.x, box.max.y, centre.z]));
    // A push ends at the object; a pull (hanging) starts at it.
    tail = s.case === 'hang' ? at.clone().addScaledVector(dir, gap) : at.clone().addScaledVector(dir, -(gap + length));
  }
  const a = arrow(length, size, accent);
  a.position.copy(tail);
  a.quaternion.setFromUnitVectors(up, dir);
  group.add(a);

  // The weakest point, seen through the model.
  const weak = new Vector3(...(s.weak as Vec3));
  const r = 0.022 * size;
  const halo = new Mesh(new SphereGeometry(r * 1.9, 20, 14), new MeshBasicMaterial({ color: '#ef4444', transparent: true, opacity: 0.35, depthTest: false }));
  const dot = new Mesh(new SphereGeometry(r, 16, 12), new MeshBasicMaterial({ color: '#ffffff', depthTest: false }));
  for (const m of [halo, dot]) {
    m.position.copy(weak);
    m.renderOrder = 10;
    group.add(m);
  }
}

async function loadCase(id: string, source: StressSource, kind: StressCase, accent: string): Promise<Loaded> {
  const summary = await stressSummary(id, source, kind);
  // Named by the result too: a scan processed again gets new stresses.
  const file = new File(Paths.cache, `stress-${source}-${id}-${kind}-${summary.elements}-${Math.round(summary.max_mpa * 1e6)}.glb`);
  if (!file.exists) await downloadStressModel(id, source, kind, file);
  let scene: Object3D;
  try {
    scene = await parseGlb(file, `stress-${id}-${kind}`);
  } catch (e) {
    file.delete(); // likely a truncated download: the next try fetches it again
    throw e;
  }
  let mesh: Mesh | null = null;
  scene.traverse((o) => {
    if (!mesh && o instanceof Mesh) mesh = o;
  });
  const geometry = (mesh as Mesh | null)?.geometry as BufferGeometry | undefined;
  const values = geometry?.getAttribute('_stress');
  if (!mesh || !geometry || !values) throw new Error('Резултатът от теста не се зареди. Опитайте отново.');
  const colors = new BufferAttribute(new Float32Array(values.count * 3), 3);
  geometry.setAttribute('color', colors);
  (mesh as Mesh).material = new MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0 });
  const group = new Group();
  group.add(scene);
  addGuides(group, scene, summary, accent);
  return { summary, model: group, values: values.array as ArrayLike<number>, colors };
}

/** Colours the model for stresses `per` times the server's, as a share of `strength` (unless it already is). */
function paint(l: Loaded, per: number, strength: number) {
  const k = per / strength;
  if (l.painted === k) return;
  l.painted = k;
  const out = l.colors.array as Float32Array;
  const c = new Color();
  for (let i = 0; i < l.values.length; i++) {
    const v = l.values[i];
    const [r, g, b] = stressColor(v < 0 ? -1 : v * k);
    c.setRGB(r, g, b, SRGBColorSpace);
    out[3 * i] = c.r;
    out[3 * i + 1] = c.g;
    out[3 * i + 2] = c.b;
  }
  l.colors.needsUpdate = true;
}

export function ErrorBoundary({ retry }: ErrorBoundaryProps) {
  const s = useStyles();
  return (
    <View style={[s.fill, s.center]}>
      <ErrorText>Тестът не може да се покаже.</ErrorText>
      <Button title="Опитай отново" icon="refresh" onPress={retry} />
    </View>
  );
}

/** While the server works it out: the seconds so far, so a long first test does not look stuck. */
function Waiting() {
  const s = useStyles();
  const [waited, setWaited] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setWaited((w) => w + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <View style={s.center}>
      <Skeleton style={s.loadingBlock} />
      <Label>Изчисляване… {waited > 2 ? `${waited} с` : ''}</Label>
    </View>
  );
}

const gramText = (g: number) => (g >= 1000 ? `${(g / 1000).toFixed(1)} кг` : `${Math.max(1, Math.round(g))} г`);

export default function StressTest() {
  const params = useLocalSearchParams<{ id: string; source?: StressSource; mm?: string; title?: string }>();
  const id = params.id;
  const source: StressSource = params.source === 'post' ? 'post' : 'scan';
  const c = useColors();
  const s = useStyles();
  const [kind, setKind] = useState<StressCase>('top');
  const [loads, setLoads] = useState(DEFAULT_LOAD);
  const [material, setMaterial] = useState<Material>('pla');
  const [infill, setInfill] = useState<Infill>(20);
  const [mmInput, setMmInput] = useState(params.mm ?? '100');
  const [results, setResults] = useState<Partial<Record<StressCase, Loaded>>>({});
  const [errors, setErrors] = useState<Partial<Record<StressCase, string>>>({});
  const [attempt, setAttempt] = useState(0);
  const started = useRef(new Set<string>());

  const current = results[kind];
  const failed = errors[kind];

  useEffect(() => {
    const key = `${kind}-${attempt}`;
    if (results[kind] || started.current.has(key)) return;
    started.current.add(key);
    setErrors((e) => ({ ...e, [kind]: undefined }));
    loadCase(id, source, kind, c.accent).then(
      (l) => {
        setResults((r) => ({ ...r, [kind]: l }));
        checkProgress(true); // a first test of this kind earns points
        refreshProfile().catch(() => {}); // the free plan's count of tested models
      },
      (e) => setErrors((er) => ({ ...er, [kind]: errorText(e) })),
    );
  }, [id, source, kind, attempt, results, c.accent]);

  const mm = Number(mmInput.replace(',', '.'));
  const mmValid = Number.isFinite(mm) && mm >= 5 && mm <= 2000;
  const load = loads[kind];
  const o = current ? outcome(current.summary, load, material, infill, mmValid ? mm : current.summary.ref_mm) : null;

  const scale = o?.scale ?? 0;
  const strength = o?.strength ?? 1;

  // A tick when the verdict turns: worse is felt.
  const verdict = o?.verdict;
  const lastVerdict = useRef<Verdict | undefined>(undefined);
  useEffect(() => {
    if (verdict && lastVerdict.current && verdict !== lastVerdict.current)
      Haptics.notificationAsync(verdict === 'break' ? Haptics.NotificationFeedbackType.Error : Haptics.NotificationFeedbackType.Success).catch(() => {});
    lastVerdict.current = verdict;
  }, [verdict]);

  const info = CASES.find((x) => x.value === kind)!;
  const choices = kind === 'drop' ? HEIGHTS : WEIGHTS;
  const tone = verdict === 'safe' ? c.success : verdict === 'edge' ? c.warning : c.danger;
  const toneSoft = verdict === 'safe' ? c.successSoft : verdict === 'edge' ? c.warningSoft : c.dangerSoft;

  return (
    <View style={s.fill}>
      <View style={s.preview}>
        {current ? (
          <Animated.View key={kind} entering={fade} style={s.flex}>
            <ModelView model={current.model} wireframe={false} background={c.viewer} beforeDraw={() => paint(current, scale, strength)} />
          </Animated.View>
        ) : failed ? (
          <View style={s.center}>
            <ErrorText>{failed}</ErrorText>
            <Button title="Опитай отново" icon="refresh" variant="secondary" onPress={() => setAttempt((a) => a + 1)} />
          </View>
        ) : (
          <Waiting key={`${kind}-${attempt}`} />
        )}
        <SafeAreaView edges={['top']} style={s.floatTop} pointerEvents="box-none">
          <IconButton icon="chevron-back" label="Назад" onPress={() => router.back()} />
        </SafeAreaView>
        {current && (
          <Animated.View entering={fade} style={s.legend} pointerEvents="none">
            <LinearGradient
              colors={LEGEND.map((l) => l.color) as [string, string, ...string[]]}
              locations={LEGEND.map((l) => l.at) as [number, number, ...number[]]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={s.legendBar}
            />
            <View style={s.legendEnds}>
              <Text style={s.legendLabel}>Спокойно</Text>
              <View style={s.weakKey}>
                <View style={s.weakDot} />
                <Text style={s.legendLabel}>Най-слабо място</Text>
              </View>
              <Text style={s.legendLabel}>Чупи се</Text>
            </View>
          </Animated.View>
        )}
      </View>

      <ScrollView style={s.flex} contentContainerStyle={s.content} showsVerticalScrollIndicator={false}>
        <Appear>
          <Kicker>Тест за здравина</Kicker>
          <Title numberOfLines={2} style={s.title}>
            {params.title || 'Модел'}
          </Title>
        </Appear>

        <Appear index={1}>
          <Segmented<StressCase> value={kind} onChange={setKind} options={CASES.map((x) => ({ value: x.value, label: x.label }))} />
        </Appear>

        <Animated.View key={kind} entering={swap} style={s.about}>
          <Ionicons name={info.icon as IconName} size={18} color={c.accent} />
          <Small style={s.flex}>{info.about}</Small>
        </Animated.View>

        {o && current && (
          <Animated.View entering={fade}>
            <Card style={[s.verdict, { borderColor: tone }]}>
              <View style={s.verdictHead}>
                <View style={[s.verdictIcon, { backgroundColor: toneSoft }]}>
                  <Ionicons name={VERDICTS[o.verdict].icon} size={22} color={tone} />
                </View>
                <Text style={[s.flex, s.verdictTitle, { color: tone }]}>{VERDICTS[o.verdict].title}</Text>
              </View>
              <View style={s.facts}>
                <View style={s.fact}>
                  <Label>{kind === 'drop' ? 'Издържа падане до' : 'Издържа до'}</Label>
                  <Text style={s.factValue}>{limitText(kind, o.limit)}</Text>
                </View>
                <View style={[s.fact, s.factLine]}>
                  <Label>Тегло</Label>
                  <Text style={s.factValue}>{gramText(o.massG)}</Text>
                </View>
              </View>
            </Card>
          </Animated.View>
        )}

        <Section label={kind === 'drop' ? 'Височина на падане' : 'Тежест'}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={s.bleed} contentContainerStyle={s.chips}>
            {choices.map((v) => {
              const on = v === load;
              return (
                <Pressable
                  key={v}
                  onPress={() => {
                    if (!on) haptic();
                    setLoads((l) => ({ ...l, [kind]: v }));
                  }}
                  style={[s.chip, on && s.chipOn]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: on }}
                >
                  <Text style={[s.chipText, on && s.chipTextOn]}>{loadText(kind, v)}</Text>
                </Pressable>
              );
            })}
          </ScrollView>
        </Section>

        <Section label="Материал">
          <Segmented<Material> value={material} onChange={setMaterial} options={(Object.keys(MATERIALS) as Material[]).map((m) => ({ value: m, label: MATERIALS[m].label }))} />
        </Section>

        {MATERIALS[material].layered && (
          <Section label="Запълване">
            <Segmented<`${Infill}`>
              value={`${infill}`}
              onChange={(v) => setInfill(Number(v) as Infill)}
              options={INFILLS.map((p) => ({ value: `${p}` as `${Infill}`, label: p === 100 ? 'Плътно' : `${p}%` }))}
            />
          </Section>
        )}

        <Section label="Реален размер">
          <Card style={s.mmRow}>
            <View style={s.flex}>
              <H3>Най-дълга страна</H3>
            </View>
            <View style={[s.mmBox, !mmValid && s.mmInvalid]}>
              <Field style={s.mm} value={mmInput} onChangeText={setMmInput} keyboardType="decimal-pad" accessibilityLabel="Най-дълга страна в милиметри" selectTextOnFocus />
              <Text style={s.mmUnit}>мм</Text>
            </View>
          </Card>
          {!mmValid && <Small style={s.danger}>Въведете размер от 5 до 2000 мм.</Small>}
        </Section>

        {current && current.summary.dropped > 0.01 && (
          <Note warn>Сивите части са твърде тънки и не са тествани.</Note>
        )}

        <Small>Приблизителна оценка.</Small>
      </ScrollView>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: space.md, padding: space.xl },
  preview: { height: '44%', backgroundColor: c.viewer[1], borderBottomWidth: 1, borderBottomColor: c.line },
  loadingBlock: { width: 120, height: 120, borderRadius: radius.md },
  floatTop: { position: 'absolute', left: 0, right: 0, top: 0, flexDirection: 'row', paddingHorizontal: space.lg, paddingTop: space.sm },
  legend: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 12,
    gap: 6,
    backgroundColor: c.glass,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  legendLabel: { fontFamily: font.mono, fontSize: 10.5, color: c.muted },
  legendBar: { height: 6, borderRadius: 3 },
  legendEnds: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  weakKey: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  weakDot: { width: 9, height: 9, borderRadius: 4.5, backgroundColor: '#ffffff', borderWidth: 2.5, borderColor: 'rgba(239,68,68,0.6)' },
  content: { padding: 18, paddingTop: space.xl, paddingBottom: 48, gap: 20 },
  title: { marginTop: space.sm },
  about: { flexDirection: 'row', gap: space.md, alignItems: 'flex-start' },
  verdict: { gap: space.md, borderWidth: 1 },
  verdictHead: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  verdictIcon: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  verdictTitle: { fontFamily: font.semibold, fontSize: 19, lineHeight: 24, letterSpacing: -0.4 },
  facts: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: c.line, marginHorizontal: -space.lg },
  fact: { flex: 1, gap: 6, paddingVertical: 12, paddingHorizontal: space.lg },
  factLine: { borderLeftWidth: 1, borderLeftColor: c.line },
  factValue: { fontFamily: font.monoMedium, fontSize: 16, lineHeight: 22, color: c.text },
  bleed: { marginHorizontal: -18 },
  chips: { flexDirection: 'row', gap: 8, paddingHorizontal: 18 },
  chip: { borderWidth: 1, borderColor: c.lineStrong, borderRadius: radius.pill, paddingHorizontal: 14, height: 38, justifyContent: 'center' },
  chipOn: { backgroundColor: c.primary, borderColor: c.primary },
  chipText: { fontFamily: font.monoMedium, fontSize: 13.5, color: c.text },
  chipTextOn: { color: c.onPrimary },
  mmRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
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
  danger: { color: c.danger },
}));

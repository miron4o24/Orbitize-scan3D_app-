import Ionicons from '@expo/vector-icons/Ionicons';
import { File, Paths } from 'expo-file-system';
import { Image } from 'expo-image';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';
import type { Object3D } from 'three';

import { ModelView } from '@/components/ModelView';
import { Button, Card, ErrorText, Header, Note, Screen, Section, Small } from '@/components/ui';
import {
  authHeaders,
  cadPhotoUrl,
  downloadCadPreview,
  errorText,
  type CadAdd,
  type CadEdges,
  type CadFace,
  type CadFeatureType,
  type CadItem,
  type CadOp,
  type CadPhoto,
  type CadState,
} from '@/lib/api';
import { parseGlb } from '@/lib/model';
import { font, makeStyles, radius, space, useColors } from '@/theme';

// "Какво виждам": снимките с разпознатото очертание, детайлът в 3D по оценките (за да се види, че е разчетен
// правилно: посоки, стъпала, отвори) и списъкът на елементите, които могат да се махнат, обърнат или добавят.

const PHOTO_TITLE: Record<CadPhoto, string> = {
  top: 'Отгоре',
  front: 'Отпред',
  back: 'Отзад',
  right: 'Отдясно',
  left: 'Отляво',
  bottom: 'Отдолу',
};

/* Снимките, на които има бутони за обръщане: първата от всяка двойка отстрани (обръщането важи и за другата). */
function flippable(views: CadPhoto[]): CadPhoto[] {
  const out: CadPhoto[] = [];
  for (const pair of [
    ['front', 'back'],
    ['right', 'left'],
  ] as const) {
    const first = pair.find((p) => views.includes(p));
    if (first) out.push(first);
  }
  return out;
}

export function Review({
  jobId,
  state,
  busy,
  error,
  onOp,
  onNext,
  onRetake,
}: {
  jobId: string;
  state: CadState;
  busy: boolean;
  error: string | null;
  onOp: (op: CadOp) => void;
  onNext: () => void;
  onRetake?: () => void; // няма при промяна на построен детайл
}) {
  const s = useStyles();
  const c = useColors();
  const [adding, setAdding] = useState(false);
  const turned = state.design.kind === 'turned';
  const left = state.questions.filter((q) => q.src !== 'measured').length;
  const feature = (id: string) => state.design.features.find((f) => f.id === id);
  const canFlip = flippable(state.views);

  return (
    <Screen>
      <Header kicker="Инженерен CAD · проверка" title="Какво виждам" sub="Проверете, че предметът е разчетен правилно. Подробните мерки идват после, по една." />

      <Section label="В 3D (по оценките от снимките)">
        <Preview key={state.stamp} jobId={jobId} stamp={state.stamp} />
        <Small>Завъртете го с пръст. Ако нещо е наобратно (стъпалото отдолу, отворът в грешния край), обърнете снимката отпред или отдясно по-долу.</Small>
      </Section>

      <Section label="Разпознато">
        <Card style={s.list}>
          {state.items.map((item, k) => (
            <ItemRow key={item.id} item={item} last={k === state.items.length - 1} busy={busy} through={!feature(item.id)?.depth} onOp={onOp} />
          ))}
        </Card>
        {!adding ? (
          <Button title="Добави елемент" sub="отвор, джоб, канал, заобляне, фаска" icon="add" variant="secondary" onPress={() => setAdding(true)} disabled={busy} />
        ) : (
          <AddFeature
            turned={turned}
            busy={busy}
            onCancel={() => setAdding(false)}
            onAdd={(add) => {
              setAdding(false);
              onOp({ add });
            }}
          />
        )}
      </Section>

      <Section label="Снимките с очертанието">
        {state.views.map((v) => (
          <Card key={v} style={s.photoCard}>
            <View style={s.photoHead}>
              <Text style={s.photoTitle}>{PHOTO_TITLE[v]}</Text>
              {canFlip.includes(v) && (
                <View style={s.flipRow}>
                  <Pressable style={s.chip} onPress={() => onOp({ flip: [v, 0] })} disabled={busy} accessibilityRole="button" accessibilityLabel="Обърни ляво-дясно">
                    <Ionicons name="swap-horizontal" size={16} color={c.text2} />
                  </Pressable>
                  <Pressable style={s.chip} onPress={() => onOp({ flip: [v, 1] })} disabled={busy} accessibilityRole="button" accessibilityLabel="Обърни горе-долу">
                    <Ionicons name="swap-vertical" size={16} color={c.text2} />
                  </Pressable>
                </View>
              )}
            </View>
            <Image source={{ uri: cadPhotoUrl(jobId, v, state.stamp), headers: authHeaders() }} style={s.photo} contentFit="contain" transition={150} />
          </Card>
        ))}
        <Small>Оранжевото е това, което ще се построи. Ако не пасва на предмета, снимайте отново.</Small>
        {onRetake && <Button title="Снимай отново" icon="camera-outline" variant="ghost" onPress={onRetake} disabled={busy} />}
      </Section>

      {[...state.warnings, ...state.notes].map((n) => (
        <Note key={n} warn>
          {n}
        </Note>
      ))}
      {error && <ErrorText>{error}</ErrorText>}
      <Button
        title={left ? `Към мерките (${left})` : 'Построй модела'}
        sub="с шублер, по една"
        variant="brand"
        size="lg"
        icon="resize-outline"
        loading={busy}
        onPress={onNext}
      />
    </Screen>
  );
}

function ItemRow({ item, last, busy, through, onOp }: { item: CadItem; last: boolean; busy: boolean; through: boolean; onOp: (op: CadOp) => void }) {
  const s = useStyles();
  const c = useColors();
  const isFeature = item.kind !== 'body' && item.kind !== 'view';
  const plain = item.kind === 'view' && item.text.includes('правоъгълник') && !item.text.includes('заобл');
  const cut = item.kind === 'hole' || item.kind === 'rect' || item.kind === 'slot';
  return (
    <View style={[s.item, !last && s.itemLine]}>
      <Ionicons name={ICON[item.kind] ?? 'ellipse-outline'} size={18} color={c.accent} />
      <Text style={s.itemText}>{item.text}</Text>
      {cut && (
        <Pressable style={s.chip} onPress={() => onOp({ through: [item.id, !through] })} disabled={busy} accessibilityRole="button">
          <Text style={s.chipText}>{through ? 'проходен' : 'глух'}</Text>
        </Pressable>
      )}
      {(isFeature || (item.kind === 'view' && !plain)) && (
        <Pressable
          style={s.chip}
          onPress={() => onOp({ remove: item.id })}
          disabled={busy}
          accessibilityRole="button"
          accessibilityLabel={isFeature ? 'Махни' : 'Опрости до правоъгълник'}
        >
          <Ionicons name={isFeature ? 'trash-outline' : 'square-outline'} size={16} color={c.text2} />
        </Pressable>
      )}
    </View>
  );
}

const ICON: Partial<Record<CadItem['kind'], 'cube-outline' | 'scan-outline' | 'ellipse-outline' | 'square-outline' | 'remove-outline' | 'radio-button-off-outline' | 'triangle-outline'>> = {
  body: 'cube-outline',
  view: 'scan-outline',
  hole: 'radio-button-off-outline',
  rect: 'square-outline',
  slot: 'remove-outline',
  fillet: 'ellipse-outline',
  chamfer: 'triangle-outline',
};

/* 3D по оценките: създава се наново (key) при всяка промяна на описанието. */
function Preview({ jobId, stamp }: { jobId: string; stamp: string }) {
  const s = useStyles();
  const c = useColors();
  const [model, setModel] = useState<Object3D | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const file = new File(Paths.cache, `cad-preview-${jobId}-${stamp}.glb`);
    (async () => {
      try {
        if (!file.exists) await downloadCadPreview(jobId, file);
        const m = await parseGlb(file, `cad-${jobId}-${stamp}`);
        if (!cancelled) setModel(m);
      } catch (e) {
        if (!cancelled) setError(errorText(e));
      } finally {
        if (file.exists) file.delete();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId, stamp]);
  return (
    <View style={s.preview}>
      {model ? (
        <ModelView model={model} wireframe={false} background={c.viewer} />
      ) : error ? (
        <View style={s.center}>
          <Small>{error}</Small>
        </View>
      ) : (
        <View style={s.center}>
          <ActivityIndicator color={c.accent} />
        </View>
      )}
    </View>
  );
}

// --- Добавяне на елемент -------------------------------------------------------------------------------------------

const TYPES: { value: CadFeatureType; label: string }[] = [
  { value: 'hole', label: 'Отвор' },
  { value: 'rect', label: 'Правоъгълен изрез' },
  { value: 'slot', label: 'Канал' },
  { value: 'fillet', label: 'Заобляне' },
  { value: 'chamfer', label: 'Фаска' },
];
const FACES: { value: CadFace; label: string }[] = [
  { value: 'top', label: 'Отгоре' },
  { value: 'front', label: 'Отпред' },
  { value: 'left', label: 'Отляво' },
  { value: 'bottom', label: 'Отдолу' },
  { value: 'back', label: 'Отзад' },
  { value: 'right', label: 'Отдясно' },
];
const EDGES: { value: CadEdges; label: string }[] = [
  { value: 'vertical', label: 'Вертикалните' },
  { value: 'top', label: 'Горните' },
  { value: 'bottom', label: 'Долните' },
  { value: 'all', label: 'Всички' },
];
// Осите в равнината на всяка стена (за "в средата по ..."), както ги брои сървърът (design.plane_axes).
const ALONG: Record<CadFace, [string, string]> = {
  top: ['дължината', 'ширината'],
  bottom: ['дължината', 'ширината'],
  front: ['дължината', 'височината'],
  back: ['дължината', 'височината'],
  left: ['ширината', 'височината'],
  right: ['ширината', 'височината'],
};
const THREADS = ['', 'M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12'];

function AddFeature({ turned, busy, onAdd, onCancel }: { turned: boolean; busy: boolean; onAdd: (a: CadAdd) => void; onCancel: () => void }) {
  const s = useStyles();
  const [t, setT] = useState<CadFeatureType>('hole');
  const [face, setFace] = useState<CadFace>(turned ? 'left' : 'top');
  const [through, setThrough] = useState(true);
  const [centered, setCentered] = useState<[boolean, boolean]>([true, true]);
  const [edges, setEdges] = useState<CadEdges>(turned ? 'ends' : 'vertical');
  const [thread, setThread] = useState('');
  const edgeType = t === 'fillet' || t === 'chamfer';
  const faces = turned ? FACES.filter((f) => f.value === 'left' || f.value === 'top') : FACES;
  const along = turned ? (face === 'left' ? ['', ''] : ['дължината', '']) : ALONG[face];
  return (
    <Card style={s.add}>
      <Text style={s.addTitle}>Нов елемент</Text>
      <Chips options={TYPES} value={t} onChange={setT} />
      {edgeType ? (
        <>
          <Small>Кои ръбове</Small>
          <Chips options={turned ? [{ value: 'ends', label: 'Краищата' }, { value: 'all', label: 'Всички' }] : EDGES} value={edges} onChange={setEdges} />
        </>
      ) : (
        <>
          <Small>{turned ? 'Къде' : 'От коя страна започва'}</Small>
          <Chips options={turned ? faces.map((f) => ({ value: f.value, label: f.value === 'left' ? 'По оста' : 'Напречно' })) : faces} value={face} onChange={setFace} />
          <Chips
            options={[
              { value: 'through', label: 'Проходен' },
              { value: 'blind', label: 'Глух (с дълбочина)' },
            ]}
            value={through ? 'through' : 'blind'}
            onChange={(v) => setThrough(v === 'through')}
          />
          {along.map(
            (a, k) =>
              !!a && (
                <Chips
                  key={a + k}
                  options={[
                    { value: 'c', label: `В средата по ${a}` },
                    { value: 'm', label: 'Ще го измеря' },
                  ]}
                  value={centered[k] ? 'c' : 'm'}
                  onChange={(v) => setCentered((cur) => (k === 0 ? [v === 'c', cur[1]] : [cur[0], v === 'c']))}
                />
              ),
          )}
          {t === 'hole' && (
            <>
              <Small>Резба (по желание; в модела отворът е с диаметъра, който измерите)</Small>
              <Chips options={THREADS.map((v) => ({ value: v, label: v || 'Без' }))} value={thread} onChange={setThread} />
            </>
          )}
        </>
      )}
      <View style={s.addButtons}>
        <Button title="Отказ" variant="ghost" onPress={onCancel} style={s.flex} />
        <Button
          title="Добави"
          icon="add"
          loading={busy}
          style={s.flex}
          onPress={() =>
            onAdd(
              edgeType
                ? { t, edges }
                : {
                    t,
                    face,
                    through,
                    centered: turned ? [face === 'left', true] : centered,
                    ...(t === 'hole' && thread && { thread }),
                  },
            )
          }
        />
      </View>
    </Card>
  );
}

function Chips<T extends string>({ options, value, onChange }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  const s = useStyles();
  return (
    <View style={s.chips}>
      {options.map((o) => (
        <Pressable
          key={o.value}
          style={[s.pick, o.value === value && s.pickOn]}
          onPress={() => onChange(o.value)}
          accessibilityRole="button"
          accessibilityState={{ selected: o.value === value }}
        >
          <Text style={[s.pickText, o.value === value && s.pickTextOn]}>{o.label}</Text>
        </Pressable>
      ))}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  flex: { flex: 1 },
  list: { paddingVertical: 0 },
  item: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  itemLine: { borderBottomWidth: 1, borderBottomColor: c.line },
  itemText: { flex: 1, color: c.text, fontFamily: font.regular, fontSize: 14.5, lineHeight: 20 },
  chip: {
    minWidth: 36,
    height: 32,
    paddingHorizontal: 10,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: c.surface2,
    borderWidth: 1,
    borderColor: c.line,
  },
  chipText: { color: c.text2, fontFamily: font.medium, fontSize: 12.5 },
  photoCard: { gap: space.sm, padding: space.sm },
  photoHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.sm, paddingTop: 4 },
  photoTitle: { color: c.text, fontFamily: font.semibold, fontSize: 14.5 },
  flipRow: { flexDirection: 'row', gap: space.sm },
  photo: { width: '100%', aspectRatio: 4 / 3, borderRadius: radius.sm, backgroundColor: c.surface2 },
  preview: { height: 260, borderRadius: radius.md, overflow: 'hidden', backgroundColor: c.surface2 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.lg },
  add: { gap: space.md },
  addTitle: { color: c.text, fontFamily: font.semibold, fontSize: 15 },
  addButtons: { flexDirection: 'row', gap: space.md },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: space.sm },
  pick: { paddingHorizontal: 12, height: 34, borderRadius: 17, justifyContent: 'center', borderWidth: 1, borderColor: c.line, backgroundColor: c.surface },
  pickOn: { backgroundColor: c.accentSoft, borderColor: c.accentLine },
  pickText: { color: c.text2, fontFamily: font.medium, fontSize: 13 },
  pickTextOn: { color: c.accent },
}));

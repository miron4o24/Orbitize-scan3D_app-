import Ionicons from '@expo/vector-icons/Ionicons';
import { Image } from 'expo-image';
import { useState } from 'react';
import { ActivityIndicator, Pressable, Text, View, type GestureResponderEvent } from 'react-native';

import { Button, ErrorText, Header, Note, Screen, Segmented } from '@/components/ui';
import { authHeaders, cadOutlineUrl, cadRetrace, errorText, type CadPhoto, type CadPoint } from '@/lib/api';
import { font, makeStyles, radius, space, useColors } from '@/theme';

import { SHOTS } from './Shoot';

// Проверка на очертанията, преди да се мери: на всяка снимка оранжевата линия трябва да следва точно детайла.
// Автоматичното изрязване понякога взима и ръката, която го държи, телефона или листа под него, или сянката: тогава
// всичко след това (размери, части, отвори) излиза грешно. Докосване върху детайла го изрязва наново (MobileSAM на
// сървъра); докосване в режим "Не е детайлът" маха това, което не трябва да е вътре.

const title = (p: CadPhoto) => SHOTS.find((x) => x.photo === p)?.title ?? p;

export function Outlines({
  jobId,
  photos,
  error,
  onNext,
  onBack,
  onRetake,
}: {
  jobId: string;
  photos: CadPhoto[];
  error: string | null;
  onNext: (changed: boolean) => void;
  onBack: () => void;
  onRetake: (photo: CadPhoto) => void;
}) {
  const s = useStyles();
  const c = useColors();
  const [open, setOpen] = useState<CadPhoto | null>(null);
  const [stamps, setStamps] = useState<Partial<Record<CadPhoto, number>>>({});
  const [changed, setChanged] = useState(false);

  if (open)
    return (
      <Fix
        jobId={jobId}
        photo={open}
        onChanged={() => {
          setChanged(true);
          setStamps((st) => ({ ...st, [open]: Date.now() }));
        }}
        onDone={() => setOpen(null)}
        onRetake={() => {
          setOpen(null);
          onRetake(open);
        }}
      />
    );

  return (
    <Screen>
      <Header
        kicker="Инженерен CAD · проверка"
        title="Очертанията"
        sub="Оранжевата линия трябва да следва точно предмета на всяка снимка. Ако хваща ръката, телефона или сянката, или изпуска част, докоснете снимката."
      />
      <View style={s.grid}>
        {photos.map((p) => (
          <Pressable key={p} style={s.tile} onPress={() => setOpen(p)} accessibilityRole="button" accessibilityLabel={`${title(p)}: поправи очертанието`}>
            <Image
              source={{ uri: cadOutlineUrl(jobId, p, String(stamps[p] ?? 0)), headers: authHeaders() }}
              style={s.thumb}
              contentFit="cover"
              transition={150}
            />
            <View style={s.caption}>
              <Text style={s.title}>{title(p)}</Text>
              <Ionicons name="hand-left-outline" size={15} color={c.text2} />
            </View>
          </Pressable>
        ))}
      </View>
      {error && <ErrorText>{error}</ErrorText>}
      <Button title="Всички са точни: напред" variant="brand" size="lg" icon="arrow-forward" onPress={() => onNext(changed)} />
      <Button title="Назад към снимките" variant="ghost" icon="arrow-back" onPress={onBack} />
    </Screen>
  );
}

type Mode = 'add' | 'remove';

function Fix({ jobId, photo, onChanged, onDone, onRetake }: { jobId: string; photo: CadPhoto; onChanged: () => void; onDone: () => void; onRetake: () => void }) {
  const s = useStyles();
  const c = useColors();
  const [mode, setMode] = useState<Mode>('add');
  const [points, setPoints] = useState<CadPoint[]>([]);
  const [stamp, setStamp] = useState(() => String(Date.now()));
  const [ratio, setRatio] = useState(4 / 3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(next: CadPoint[]) {
    setBusy(true);
    setError(null);
    try {
      await cadRetrace(jobId, photo, next);
      setPoints(next);
      setStamp(String(Date.now()));
      onChanged();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  function tap(e: GestureResponderEvent, width: number, height: number) {
    if (busy || !width || !height) return;
    const x = Math.min(1, Math.max(0, e.nativeEvent.locationX / width));
    const y = Math.min(1, Math.max(0, e.nativeEvent.locationY / height));
    if (mode === 'remove' && !points.some((p) => p.length === 2)) {
      setError('Първо докоснете самия предмет.');
      return;
    }
    const point: CadPoint = mode === 'add' ? [x, y] : [x, y, 0];
    void send([...points, point].slice(-12)); // сървърът приема до 12 точки
  }

  const [size, setSize] = useState({ width: 0, height: 0 });
  return (
    <Screen>
      <Header
        kicker={`Очертанието · ${title(photo)}`}
        title={mode === 'add' ? 'Докоснете предмета' : 'Докоснете това, което не е предметът'}
        sub={
          mode === 'add'
            ? 'Докоснете върху самия предмет. Ако линията изпусне част, докоснете и нея.'
            : 'Докоснете ръката, телефона, листа или сянката, които линията хваща.'
        }
      />
      <Segmented
        options={[
          { value: 'add', label: 'Това е предметът' },
          { value: 'remove', label: 'Не е предметът' },
        ]}
        value={mode}
        onChange={setMode}
      />
      <Pressable
        onPress={(e) => tap(e, size.width, size.height)}
        onLayout={(e) => setSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
        style={[s.photo, { aspectRatio: ratio }]}
        accessibilityRole="imagebutton"
        accessibilityLabel={mode === 'add' ? 'Снимката: докоснете предмета' : 'Снимката: докоснете това, което не е предметът'}
      >
        <Image
          source={{ uri: cadOutlineUrl(jobId, photo, stamp), headers: authHeaders() }}
          style={s.fill}
          contentFit="fill"
          transition={120}
          onLoad={(e) => e.source.width && e.source.height && setRatio(e.source.width / e.source.height)}
        />
        {busy && (
          <View style={s.busy}>
            <ActivityIndicator color={c.accent} size="large" />
            <Text style={s.busyText}>Изрязвам предмета…</Text>
          </View>
        )}
      </Pressable>
      {error ? <ErrorText>{error}</ErrorText> : <Note>Зелените точки са върху предмета, червените не са. Линията се обновява след всяко докосване.</Note>}
      <Button title="Готово" variant="brand" icon="checkmark" disabled={busy} onPress={onDone} />
      {points.length > 0 && <Button title="Обратно към автоматичното" variant="secondary" icon="refresh" disabled={busy} onPress={() => send([])} />}
      <Button title="Снимай тази страна отново" variant="ghost" icon="camera-outline" disabled={busy} onPress={onRetake} />
    </Screen>
  );
}

const absolute = { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0 } as const;

const useStyles = makeStyles((c) => ({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
  tile: { width: '47.5%', borderRadius: radius.md, overflow: 'hidden', backgroundColor: c.surface, borderWidth: 1, borderColor: c.line },
  thumb: { width: '100%', aspectRatio: 4 / 3, backgroundColor: c.surface2 },
  caption: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.md, paddingVertical: space.sm },
  title: { color: c.text, fontFamily: font.medium, fontSize: 14 },
  photo: { width: '100%', borderRadius: radius.md, overflow: 'hidden', backgroundColor: c.surface2 },
  fill: { width: '100%', height: '100%' },
  busy: { ...absolute, alignItems: 'center', justifyContent: 'center', gap: space.sm, backgroundColor: 'rgba(0,0,0,0.45)' },
  busyText: { color: '#fff', fontFamily: font.medium, fontSize: 15 },
}));

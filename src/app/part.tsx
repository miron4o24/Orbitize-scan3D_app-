import Ionicons from '@expo/vector-icons/Ionicons';
import { File } from 'expo-file-system';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, ScrollView, View } from 'react-native';

import { Button, Card, ErrorText, Field, H3, Header, Note, Screen, Section, Small } from '@/components/ui';
import { errorText } from '@/lib/api';
import { PRO_INFO } from '@/lib/prefs';
import { createScan, deleteScan, photosDir, updateScan } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { makeStyles, radius, space, useColors } from '@/theme';

// Точна част (инженерен режим, сървър app/cadfit.py run_part): по една снимка на всяка от шестте страни, посочена от
// потребителя (сървърът не гадае коя е коя), колкото още снимки иска и мерките с линийка. Моделът е само формата, сив.
const MAX_EXTRA = 50;
const MAX_BYTES = 15 * 1024 * 1024; // MAX_PHOTO_BYTES на сървъра

type Side = 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right';
const SIDES: { side: Side; label: string; how: string }[] = [
  { side: 'top', label: 'Отгоре', how: 'Телефонът точно над частта; дължината отляво надясно.' },
  { side: 'bottom', label: 'Отдолу', how: 'Обърнете частта; дължината пак отляво надясно.' },
  { side: 'front', label: 'Отпред', how: 'Дългата страна към вас, на нивото на масата.' },
  { side: 'back', label: 'Отзад', how: 'Другата дълга страна, на нивото на масата.' },
  { side: 'left', label: 'Отляво', how: 'Левият край, на нивото на масата.' },
  { side: 'right', label: 'Отдясно', how: 'Десният край, на нивото на масата.' },
];
const SIZES = [
  { label: 'Дължина', hint: 'отляво надясно' },
  { label: 'Ширина', hint: 'отпред назад' },
  { label: 'Височина', hint: 'от масата догоре' },
] as const;

const number = (text: string) => {
  const v = Number(text.replace(',', '.'));
  return Number.isFinite(v) && v > 0 && v <= 2000 ? v : null;
};

async function choose(source: 'camera' | 'gallery', many: boolean): Promise<string[]> {
  if (source === 'camera') {
    const access = await ImagePicker.requestCameraPermissionsAsync();
    if (!access.granted) {
      Alert.alert('Камера', 'Разрешете достъпа до камерата от настройките на телефона.');
      return [];
    }
    const shot = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.95 });
    return shot.canceled ? [] : shot.assets.map((a) => a.uri);
  }
  const got = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    quality: 0.95,
    allowsMultipleSelection: many,
    selectionLimit: many ? MAX_EXTRA : 1,
  });
  return got.canceled ? [] : got.assets.map((a) => a.uri);
}

export default function Part() {
  const c = useColors();
  const s = useStyles();
  const [sides, setSides] = useState<Partial<Record<Side, string>>>({});
  const [extra, setExtra] = useState<string[]>([]);
  const [sizes, setSizes] = useState(['', '', '']);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setSide = async (side: Side, source: 'camera' | 'gallery') => {
    const [uri] = await choose(source, false);
    if (uri) setSides((now) => ({ ...now, [side]: uri }));
  };
  const addExtra = async (source: 'camera' | 'gallery') => {
    const uris = await choose(source, true);
    if (uris.length) setExtra((now) => [...now, ...uris].slice(0, MAX_EXTRA));
  };

  const measures = sizes.map(number);
  const bad = sizes.some((t, i) => t.trim() !== '' && measures[i] === null);
  const missing = SIDES.filter((x) => !sides[x.side]);
  const ready = missing.length === 0 && !bad && measures.some((v) => v !== null);

  function start() {
    setError(null);
    if (missing.length) return setError(`Липсва снимка: ${missing.map((x) => x.label.toLowerCase()).join(', ')}.`);
    if (bad) return setError('Мерките са в милиметри, от 0,1 до 2000.');
    if (!measures.some((v) => v !== null)) return setError('Нужна е поне една мярка с линийка. С трите копието е 1:1 по всяка ос.');
    setBusy(true);
    const scan = createScan(currentUserId() ?? undefined, 'part');
    try {
      const copy = (uri: string, name: string) => {
        const dest = new File(photosDir(scan.id), name);
        new File(uri).copySync(dest);
        if (dest.size > MAX_BYTES) throw new Error('Една от снимките е твърде голяма. Изберете по-малка.');
      };
      SIDES.forEach((x) => copy(sides[x.side]!, `view_${x.side}.jpg`));
      extra.forEach((uri, i) => copy(uri, `img_${String(i + 1).padStart(4, '0')}.jpg`));
      updateScan(scan.id, {
        status: 'uploading',
        photoCount: SIDES.length + extra.length,
        partSize: measures.map((v) => v ?? 0) as [number, number, number],
        longestMm: Math.max(...measures.map((v) => v ?? 0)),
      });
    } catch (e) {
      deleteScan(scan.id);
      setBusy(false);
      return setError(errorText(e));
    }
    setBusy(false);
    router.replace({ pathname: '/processing/[id]', params: { id: scan.id, auto: '1' } });
  }

  return (
    <Screen>
      <Header kicker={`${PRO_INFO.part.label} · ${PRO_INFO.part.time}`} title="Шестте страни и мерки" sub="Копие 1:1 за печат: само формата." />

      <Note>Частта на еднородна маса, без ръка в кадъра. За всяка страна по една снимка отблизо, частта да изпълва кадъра. После добавете колкото още искате.</Note>

      <Section label={`Шестте страни · ${SIDES.length - missing.length} от ${SIDES.length}`}>
        <View style={s.grid}>
          {SIDES.map((x) => (
            <Card key={x.side} style={s.slot}>
              {sides[x.side] ? (
                <Image source={{ uri: sides[x.side] }} style={s.thumb} contentFit="cover" />
              ) : (
                <View style={[s.thumb, s.empty]}>
                  <Ionicons name="camera-outline" size={22} color={c.muted} />
                </View>
              )}
              <H3>{x.label}</H3>
              <Small>{x.how}</Small>
              <View style={s.row}>
                <Button title="Снимай" onPress={() => setSide(x.side, 'camera')} style={s.flex} />
                <Button title="Галерия" variant="secondary" onPress={() => setSide(x.side, 'gallery')} style={s.flex} />
              </View>
            </Card>
          ))}
        </View>
      </Section>

      <Section label={`Още снимки (по желание) · ${extra.length}`}>
        {extra.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.thumbs}>
            {extra.map((uri, i) => (
              <Pressable key={`${uri}-${i}`} onPress={() => setExtra((now) => now.filter((_, j) => j !== i))} accessibilityRole="button" accessibilityLabel={`Махни снимка ${i + 1}`}>
                <Image source={{ uri }} style={s.small} contentFit="cover" />
                <View style={s.remove}>
                  <Ionicons name="close" size={14} color={c.text} />
                </View>
              </Pressable>
            ))}
          </ScrollView>
        )}
        <View style={s.row}>
          <Button title="Снимай" icon="camera-outline" variant="secondary" onPress={() => addExtra('camera')} style={s.flex} />
          <Button title="От галерията" icon="images-outline" variant="secondary" onPress={() => addExtra('gallery')} style={s.flex} />
        </View>
        <Small>Отвсякъде и отблизо: с 10 и повече снимки общо сървърът опитва и пълно сканиране.</Small>
      </Section>

      <Section label="Мерки с линийка или шублер (мм)">
        <Card style={s.sizes}>
          {SIZES.map((size, i) => (
            <View key={size.label} style={s.sizeRow}>
              <View style={s.flex}>
                <H3>{size.label}</H3>
                <Small>{size.hint}</Small>
              </View>
              <Field
                style={s.mm}
                value={sizes[i]}
                onChangeText={(t) => setSizes((now) => now.map((x, j) => (j === i ? t : x)))}
                keyboardType="decimal-pad"
                placeholder="мм"
                accessibilityLabel={`${size.label} в милиметри`}
                invalid={sizes[i].trim() !== '' && measures[i] === null}
              />
            </View>
          ))}
        </Card>
        <Small>Поне една мярка. С трите размерът е точен по всяка ос.</Small>
      </Section>

      {error && <ErrorText>{error}</ErrorText>}
      <Button title="Направи модела" icon="construct-outline" size="lg" onPress={start} loading={busy} disabled={!ready} />
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: space.sm },
  slot: { width: '48.5%', gap: space.xs },
  thumb: { width: '100%', aspectRatio: 4 / 3, borderRadius: radius.sm, backgroundColor: c.surface2 },
  empty: { alignItems: 'center', justifyContent: 'center' },
  thumbs: { gap: space.sm, paddingBottom: space.sm },
  small: { width: 76, height: 76, borderRadius: radius.sm, backgroundColor: c.surface2 },
  remove: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 22,
    height: 22,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: c.surface,
  },
  row: { flexDirection: 'row', gap: space.sm },
  flex: { flex: 1 },
  sizes: { gap: space.md },
  sizeRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  mm: { width: 110, textAlign: 'right' },
}));

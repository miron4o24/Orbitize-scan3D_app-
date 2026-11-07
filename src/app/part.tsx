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

// Точна част (инженерен режим, сървър app/part.py): снимки на шестте страни на малкия детайл,
// допълнителни снимки по желание и мерките му с линийка или шублер. Моделът се изгражда в реален размер 1:1.
const MAX_EXTRA = 50;
const MAX_BYTES = 15 * 1024 * 1024;

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
    const shot = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 1 });
    return shot.canceled ? [] : shot.assets.map((a) => a.uri);
  }
  const got = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images'],
    quality: 1,
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
  const sideCount = Object.keys(sides).length;
  const hasMinPhotos = SIDES.every(({ side }) => !!sides[side]);
  const hasMinMeasures = measures.every((v) => v !== null);
  const ready = hasMinPhotos && !bad && hasMinMeasures;

  function start() {
    setError(null);
    if (!hasMinPhotos) return setError('Направете и шестте обозначени снимки на частта.');
    if (bad) return setError('Мерките са в милиметри, от 0,1 до 2000.');
    if (!hasMinMeasures) return setError('Въведете дължина, ширина и височина с линийка или шублер.');

    setBusy(true);
    const scan = createScan(currentUserId() ?? undefined, 'part');
    try {
      const copy = (uri: string, name: string) => {
        const dest = new File(photosDir(scan.id), name);
        new File(uri).copy(dest);
        if (dest.size > MAX_BYTES) throw new Error('Снимката е твърде голяма (до 15 MB).');
      };

      for (const [side, uri] of Object.entries(sides)) {
        if (uri) copy(uri, `view_${side}.jpg`);
      }
      extra.forEach((uri, i) => copy(uri, `photo_${String(i + 1).padStart(3, '0')}.jpg`));

      updateScan(scan.id, {
        status: 'uploading',
        photoCount: sideCount + extra.length,
        measures,
        title: 'Точна част 1:1',
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
      <Header kicker={`${PRO_INFO.part.label} · ${PRO_INFO.part.time}`} title="Точна част 1:1" sub="Копие за 3D печат по снимки и мерки с шублер." />

      <Note>Поставете детайла на матова еднородна повърхност. Снимайте всяка страна точно от 90°: камерата успоредна на страната, детайлът в центъра и без перспектива. За инженерно копие са нужни всички шест страни и трите мерки с шублер.</Note>

      <Section label={`Задължителни страни · ${sideCount} от ${SIDES.length}`}>
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

      <Section label={`Още снимки по желание · ${extra.length}`}>
        {extra.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.thumbs}>
            {extra.map((uri, i) => (
              <Pressable
                key={`${uri}-${i}`}
                onPress={() => setExtra((now) => now.filter((_, j) => j !== i))}
                accessibilityRole="button"
                accessibilityLabel={`Махни снимка ${i + 1}`}
              >
                <Image source={{ uri }} style={s.small} contentFit="cover" />
                <View style={s.remove}>
                  <Ionicons name="close" size={14} color={c.text} />
                </View>
              </Pressable>
            ))}
          </ScrollView>
        )}
        <View style={s.row}>
          <Button title="Снимай още" icon="camera-outline" variant="secondary" onPress={() => addExtra('camera')} style={s.flex} />
          <Button title="От галерията" icon="images-outline" variant="secondary" onPress={() => addExtra('gallery')} style={s.flex} />
        </View>
        <Small>Допълнителните снимки се пазят към сканирането. Формата 1:1 се определя от шестте перпендикулярни снимки и трите мерки.</Small>
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
        <Small>И трите мерки са задължителни. Снимките определят формата, а шублерът определя реалния мащаб по всяка ос.</Small>
      </Section>

      {error && <ErrorText>{error}</ErrorText>}
      <Button title="Направи модела 1:1" icon="construct-outline" size="lg" onPress={start} loading={busy} disabled={!ready} />
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

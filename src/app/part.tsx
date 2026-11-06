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

// Точна част (инженерен режим, сървър app/cadfit.py run_part): снимки от всички страни и мерките с линийка. С 10 и повече
// снимки сървърът първо сканира формата, после AI я проектира като чист CAD модел и я сверява със сканирането; размерът
// става точно мерките.
const MIN_PHOTOS = 3; // PART_MIN_PHOTOS на сървъра
const SCAN_PHOTOS = 10; // AI.MIN_FRAMES на сървъра: от толкова снимки формата се и сканира
const MAX_PHOTOS = 40;
const MAX_BYTES = 15 * 1024 * 1024; // MAX_PHOTO_BYTES на сървъра
const SIZES = [
  { label: 'Дължина', hint: 'най-дългата страна' },
  { label: 'Ширина', hint: '' },
  { label: 'Височина', hint: 'както лежи' },
] as const;

const number = (text: string) => {
  const v = Number(text.replace(',', '.'));
  return Number.isFinite(v) && v > 0 && v <= 2000 ? v : null;
};

export default function Part() {
  const c = useColors();
  const s = useStyles();
  const [photos, setPhotos] = useState<string[]>([]); // адресите на снимките в телефона
  const [sizes, setSizes] = useState(['', '', '']);
  const [notes, setNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = (uris: string[]) => setPhotos((now) => [...now, ...uris].slice(0, MAX_PHOTOS));

  async function shoot() {
    const access = await ImagePicker.requestCameraPermissionsAsync();
    if (!access.granted) return Alert.alert('Камера', 'Разрешете достъпа до камерата от настройките на телефона.');
    const shot = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.92 });
    if (!shot.canceled && shot.assets?.length) add(shot.assets.map((a) => a.uri));
  }

  async function pick() {
    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.92,
      allowsMultipleSelection: true,
      selectionLimit: MAX_PHOTOS,
    });
    if (!picked.canceled && picked.assets?.length) add(picked.assets.map((a) => a.uri));
  }

  const measures = sizes.map(number);
  const badSize = sizes.some((t, i) => t.trim() !== '' && measures[i] === null);

  function start() {
    setError(null);
    if (photos.length < MIN_PHOTOS) return setError(`Нужни са поне ${MIN_PHOTOS} снимки от различни страни.`);
    if (badSize) return setError('Мерките са в милиметри, от 0,1 до 2000.');
    setBusy(true);
    const scan = createScan(currentUserId() ?? undefined, 'part');
    try {
      photos.forEach((uri, i) => {
        const dest = new File(photosDir(scan.id), `img_${String(i + 1).padStart(4, '0')}.jpg`);
        new File(uri).copySync(dest);
        if (dest.size > MAX_BYTES) throw new Error('Една от снимките е твърде голяма. Изберете по-малка.');
      });
      updateScan(scan.id, {
        status: 'uploading',
        photoCount: photos.length,
        partSize: measures.map((v) => v ?? 0) as [number, number, number],
        partNotes: notes.trim() || undefined,
        ...(measures[0] && { longestMm: Math.max(...measures.map((v) => v ?? 0)) }),
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
      <Header kicker={`${PRO_INFO.part.label} · ${PRO_INFO.part.time}`} title="Снимки и мерки" sub="За дребни части: копие 1:1 за печат." />

      <Note>
        {`Сложете частта на равна маса, по възможност върху лист на квадратчета. Снимайте я отвсякъде: отгоре, отстрани, отдолу и отблизо отворите. От ${SCAN_PHOTOS} снимки нагоре формата се и сканира и е най-точна.`}
      </Note>

      <Section label={`Снимки · ${photos.length}`}>
        {photos.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.thumbs}>
            {photos.map((uri, i) => (
              <Pressable
                key={`${uri}-${i}`}
                onPress={() => setPhotos((now) => now.filter((_, j) => j !== i))}
                accessibilityRole="button"
                accessibilityLabel={`Махни снимка ${i + 1}`}
              >
                <Image source={{ uri }} style={s.thumb} contentFit="cover" />
                <View style={s.remove}>
                  <Ionicons name="close" size={14} color={c.text} />
                </View>
              </Pressable>
            ))}
          </ScrollView>
        )}
        <View style={s.row}>
          <Button title="Снимай" icon="camera-outline" onPress={shoot} style={s.flex} disabled={photos.length >= MAX_PHOTOS} />
          <Button title="От галерията" icon="images-outline" variant="secondary" onPress={pick} style={s.flex} disabled={photos.length >= MAX_PHOTOS} />
        </View>
        {photos.length > 0 && photos.length < SCAN_PHOTOS && (
          <Small>{`Още ${SCAN_PHOTOS - photos.length} снимки и формата ще се сканира, не само ще се проектира.`}</Small>
        )}
      </Section>

      <Section label="Мерки с линийка (мм)">
        <Card style={s.sizes}>
          {SIZES.map((size, i) => (
            <View key={size.label} style={s.sizeRow}>
              <View style={s.flex}>
                <H3>{size.label}</H3>
                {!!size.hint && <Small>{size.hint}</Small>}
              </View>
              <Field
                style={s.mm}
                value={sizes[i]}
                onChangeText={(t) => setSizes((now) => now.map((v, j) => (j === i ? t : v)))}
                keyboardType="decimal-pad"
                placeholder="мм"
                accessibilityLabel={`${size.label} в милиметри`}
                invalid={sizes[i].trim() !== '' && measures[i] === null}
              />
            </View>
          ))}
        </Card>
        <Small>Поне една мярка е нужна за копие 1:1. Всички три правят модела най-точен.</Small>
      </Section>

      <Section label="Бележки (по желание)">
        <Field value={notes} onChangeText={setNotes} maxLength={300} multiline placeholder="напр. отвор 5 мм в средата, стена 2 мм" accessibilityLabel="Бележки за частта" />
      </Section>

      {error && <ErrorText>{error}</ErrorText>}
      <Button title="Направи модела" icon="sparkles" size="lg" onPress={start} loading={busy} disabled={photos.length < MIN_PHOTOS} />
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  thumbs: { gap: space.sm, paddingBottom: space.sm },
  thumb: { width: 84, height: 84, borderRadius: radius.sm, backgroundColor: c.surface2 },
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

import Ionicons from '@expo/vector-icons/Ionicons';
import { File } from 'expo-file-system';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useState } from 'react';
import { Alert, View } from 'react-native';

import { Button, Card, ErrorText, Field, H3, Header, Note, Screen, Section, Small } from '@/components/ui';
import { errorText } from '@/lib/api';
import { PRO_INFO } from '@/lib/prefs';
import { createScan, deleteScan, photosDir, updateScan } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { makeStyles, radius, space, useColors } from '@/theme';

// Точна част (инженерен режим, сървър app/views3d.py): три снимки като чертеж (отгоре, отстрани, от края) и мерките с
// линийка. Сървърът изрязва частта от всяка снимка почти пиксел по пиксел, мащабира очертанията до мерките и тялото е
// общото на трите: размерите са измерени, не предположени.
const MAX_BYTES = 15 * 1024 * 1024; // MAX_PHOTO_BYTES на сървъра

type ViewName = 'top' | 'side' | 'end';
const VIEWS: { name: ViewName; label: string; how: string; icon: 'arrow-down' | 'arrow-forward' | 'arrow-back'; needed: boolean }[] = [
  { name: 'top', label: 'Отгоре', how: 'Телефонът точно над частта, хоризонтално; частта по дължина отляво надясно.', icon: 'arrow-down', needed: true },
  { name: 'side', label: 'Отстрани', how: 'Без да местите частта: откъм вас, на нивото на масата.', icon: 'arrow-forward', needed: true },
  { name: 'end', label: 'От края', how: 'Пак без да я местите: откъм дясната ѝ страна, на нивото на масата.', icon: 'arrow-back', needed: false },
];
const SIZES = [
  { label: 'Дължина', hint: 'отляво надясно на снимката отгоре' },
  { label: 'Ширина', hint: 'отпред назад' },
  { label: 'Височина', hint: 'от масата догоре' },
] as const;

const number = (text: string) => {
  const v = Number(text.replace(',', '.'));
  return Number.isFinite(v) && v > 0 && v <= 2000 ? v : null;
};

export default function Part() {
  const c = useColors();
  const s = useStyles();
  const [photos, setPhotos] = useState<Partial<Record<ViewName, string>>>({});
  const [sizes, setSizes] = useState(['', '', '']);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function take(view: ViewName, source: 'camera' | 'gallery') {
    if (source === 'camera') {
      const access = await ImagePicker.requestCameraPermissionsAsync();
      if (!access.granted) return Alert.alert('Камера', 'Разрешете достъпа до камерата от настройките на телефона.');
    }
    const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.95 };
    const got = source === 'camera' ? await ImagePicker.launchCameraAsync(options) : await ImagePicker.launchImageLibraryAsync(options);
    if (!got.canceled && got.assets?.length) setPhotos((now) => ({ ...now, [view]: got.assets[0].uri }));
  }

  const measures = sizes.map(number);
  const badSize = sizes.some((t, i) => t.trim() !== '' && measures[i] === null);
  const missing = VIEWS.filter((v) => v.needed && !photos[v.name]);
  const ready = missing.length === 0 && measures.some((v) => v !== null) && !badSize;

  function start() {
    setError(null);
    if (missing.length) return setError(`Липсва снимка: ${missing.map((v) => v.label.toLowerCase()).join(', ')}.`);
    if (badSize) return setError('Мерките са в милиметри, от 0,1 до 2000.');
    if (!measures.some((v) => v !== null)) return setError('Нужна е поне една мярка с линийка, най-добре дължината.');
    setBusy(true);
    const scan = createScan(currentUserId() ?? undefined, 'part');
    try {
      const chosen = VIEWS.filter((v) => photos[v.name]);
      for (const v of chosen) {
        const dest = new File(photosDir(scan.id), `view_${v.name}.jpg`);
        new File(photos[v.name]!).copySync(dest);
        if (dest.size > MAX_BYTES) throw new Error('Една от снимките е твърде голяма. Изберете по-малка.');
      }
      updateScan(scan.id, {
        status: 'uploading',
        photoCount: chosen.length,
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
      <Header kicker={`${PRO_INFO.part.label} · ${PRO_INFO.part.time}`} title="Три снимки и мерки" sub="Като чертеж: размерите се измерват от снимките." />

      <Note>Сложете частта на равна маса с еднороден цвят. Снимайте отблизо, частта да заема по-голямата част от снимката, и не я местете между снимките.</Note>

      {VIEWS.map((v) => (
        <Section key={v.name} label={v.needed ? v.label : `${v.label} (по желание)`}>
          <Card style={s.view}>
            <View style={s.viewRow}>
              {photos[v.name] ? (
                <Image source={{ uri: photos[v.name] }} style={s.thumb} contentFit="cover" />
              ) : (
                <View style={[s.thumb, s.empty]}>
                  <Ionicons name={v.icon} size={26} color={c.muted} />
                </View>
              )}
              <Small style={s.flex}>{v.how}</Small>
            </View>
            <View style={s.row}>
              <Button title={photos[v.name] ? 'Снимай пак' : 'Снимай'} icon="camera-outline" onPress={() => take(v.name, 'camera')} style={s.flex} />
              <Button title="От галерията" icon="images-outline" variant="secondary" onPress={() => take(v.name, 'gallery')} style={s.flex} />
            </View>
          </Card>
        </Section>
      ))}

      <Section label="Мерки с линийка (мм)">
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
  view: { gap: space.md },
  viewRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  thumb: { width: 76, height: 76, borderRadius: radius.sm, backgroundColor: c.surface2 },
  empty: { alignItems: 'center', justifyContent: 'center' },
  row: { flexDirection: 'row', gap: space.sm },
  flex: { flex: 1 },
  sizes: { gap: space.md },
  sizeRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  mm: { width: 110, textAlign: 'right' },
}));

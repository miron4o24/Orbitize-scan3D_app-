import { router } from 'expo-router';
import { useState } from 'react';
import { View } from 'react-native';

import { Button, Card, ErrorText, Field, H3, Header, Note, Screen, Section, Small } from '@/components/ui';
import { PRO_INFO } from '@/lib/prefs';
import { makeStyles, space } from '@/theme';

// Точна част (инженерен режим): мерките с линийка, после видео около частта. Сървърът прави AI модела на място (без
// външни услуги и лимити, app/ai.py), а после размерът по всяка ос става точно мярката (app/truesize.py): копие 1:1.
const SIZES = [
  { label: 'Дължина', hint: 'най-дългата страна, както лежи' },
  { label: 'Ширина', hint: 'по-късата страна отгоре' },
  { label: 'Височина', hint: 'от масата догоре' },
] as const;

const number = (text: string) => {
  const v = Number(text.replace(',', '.'));
  return Number.isFinite(v) && v > 0 && v <= 2000 ? v : null;
};

export default function Part() {
  const s = useStyles();
  const [sizes, setSizes] = useState(['', '', '']);
  const [error, setError] = useState<string | null>(null);
  const measures = sizes.map(number);
  const bad = sizes.some((t, i) => t.trim() !== '' && measures[i] === null);
  const ready = !bad && measures.some((v) => v !== null);

  function film() {
    setError(null);
    if (bad) return setError('Мерките са в милиметри, от 0,1 до 2000.');
    if (!ready) return setError('Нужна е поне една мярка. С трите копието е 1:1 по всяка ос.');
    router.push({ pathname: '/capture', params: { part: measures.map((v) => v ?? 0).join(',') } });
  }

  return (
    <Screen>
      <Header kicker={`${PRO_INFO.part.label} · ${PRO_INFO.part.time}`} title="Мерки, после видео" sub="Копие 1:1 за печат: формата от видеото, размерът от линийката." />

      <Section label="1. Мерки с линийка или шублер (мм)">
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
        <Small>Мерете частта така, както ще лежи на масата при видеото. С трите мерки размерът е точен по всяка ос.</Small>
      </Section>

      <Section label="2. Видео около частта">
        <Note>Частта на еднородна маса, без да я пипате. Обиколете я бавно 2 пъти, отблизо, единия път по-ниско, другия отгоре.</Note>
      </Section>

      {error && <ErrorText>{error}</ErrorText>}
      <Button title="Запиши видео" icon="videocam-outline" size="lg" onPress={film} disabled={!ready} />
    </Screen>
  );
}

const useStyles = makeStyles(() => ({
  flex: { flex: 1 },
  sizes: { gap: space.md },
  sizeRow: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  mm: { width: 110, textAlign: 'right' },
}));

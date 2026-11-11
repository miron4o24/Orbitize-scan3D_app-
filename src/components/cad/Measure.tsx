import { Image } from 'expo-image';
import { useEffect, useRef, useState } from 'react';
import { Text, View, type TextInputInstance } from 'react-native';

import { Button, ErrorText, Field, Header, Screen, Small } from '@/components/ui';
import { authHeaders, cadQuestionUrl, type CadQuestion } from '@/lib/api';
import { font, makeStyles, radius, space } from '@/theme';

// Една мярка на екран: картинката от сървъра е снимката на човека с оранжева линия точно там, където да се мери
// (без снимка на този изглед: чертеж), оценката от снимките е само за ориентир. "Не мога да го измеря" оставя
// оценката и тя е отбелязана като такава в модела, кода и чертежа.

export const mmText = (v: number) => `${Math.round(v * 100) / 100}`.replace('.', ',');

/* Милиметри от полето (запетая или точка); undefined за празно или извън 0 до 1000. */
export function mmOf(t: string): number | undefined {
  const v = Number(t.trim().replace(',', '.'));
  return t.trim() && Number.isFinite(v) && v > 0 && v <= 1000 ? v : undefined;
}

/* Заглавието: името без "(от ... края)" - посоката се вижда по линията на снимката, която може да е завъртяна. */
const titleOf = (q: CadQuestion) => q.name.replace(/\s*\([^)]*\)\s*$/, '');

export function Measure({
  jobId,
  stamp,
  question,
  index,
  count,
  kicker = 'Инженерен CAD · мярка',
  answer,
  busy,
  error,
  onAnswer,
  onBack,
}: {
  jobId: string;
  stamp: string;
  question: CadQuestion;
  kicker?: string;
  index: number;
  count: number;
  answer: number | null | undefined; // undefined: още не е отговорено; null: пропуснато
  busy: boolean;
  error: string | null;
  onAnswer: (value: number | null) => void;
  onBack: () => void;
}) {
  const s = useStyles();
  const [text, setText] = useState(answer ? mmText(answer) : '');
  const [invalid, setInvalid] = useState(false);
  const field = useRef<TextInputInstance>(null);
  // Екранът се създава наново за всяка мярка (key в engineer.tsx), затова полето започва с отговора ѝ.
  useEffect(() => {
    const t = setTimeout(() => field.current?.focus(), 250);
    return () => clearTimeout(t);
  }, []);
  const last = index === count - 1;

  function next() {
    const v = mmOf(text);
    if (v === undefined) return setInvalid(true);
    onAnswer(v);
  }

  return (
    <Screen>
      <Header
        kicker={`${kicker} ${index + 1} от ${count}`}
        title={titleOf(question)}
        sub={`Измерете по оранжевата линия на снимката, между двете стрелки.${question.hint ? ` ${question.hint}` : ''}`}
      />
      <View style={s.progress}>
        <View style={[s.bar, { width: `${((index + 1) / count) * 100}%` }]} />
      </View>
      <Image source={{ uri: cadQuestionUrl(jobId, question.id, stamp), headers: authHeaders() }} style={s.picture} contentFit="contain" transition={120} />
      <View style={s.line}>
        <Field
          ref={field}
          style={s.input}
          value={text}
          onChangeText={(t) => {
            setText(t);
            setInvalid(false);
          }}
          keyboardType="decimal-pad"
          placeholder="0,00"
          returnKeyType="done"
          onSubmitEditing={next}
          accessibilityLabel={`${question.name} в милиметри`}
          invalid={invalid}
        />
        <Text style={s.unit}>мм</Text>
      </View>
      {question.est != null && <Small>{`От снимката ≈ ${mmText(question.est)} мм (само ориентир)`}</Small>}
      {invalid && <ErrorText>Въведете размера в милиметри, от 0 до 1000 (например 12,35).</ErrorText>}
      {error && <ErrorText>{error}</ErrorText>}
      <Button title={last ? 'Построй модела' : 'Напред'} variant="brand" size="lg" icon={last ? 'cube-outline' : 'arrow-forward'} loading={busy} onPress={next} />
      {question.est != null && (
        <Button
          title="Не мога да го измеря"
          sub={`ще остане ${mmText(question.est)} мм, отбелязано като оценка`}
          variant="secondary"
          disabled={busy}
          onPress={() => onAnswer(null)}
        />
      )}
      <Button title="Назад" variant="ghost" icon="arrow-back" disabled={busy} onPress={onBack} />
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  progress: { height: 4, borderRadius: 2, backgroundColor: c.surface2, overflow: 'hidden' },
  bar: { height: 4, borderRadius: 2, backgroundColor: c.accent },
  picture: { width: '100%', aspectRatio: 4 / 3, borderRadius: radius.md, backgroundColor: '#18181b' },
  line: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  input: { flex: 1, fontSize: 28, height: 64, fontFamily: font.semibold, textAlign: 'right' },
  unit: { color: c.muted, fontFamily: font.medium, fontSize: 18 },
}));

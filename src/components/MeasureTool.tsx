import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import type { Object3D } from 'three';

import { ModelView, type Spot } from '@/components/ModelView';
import { Button, Field, IconButton, Small } from '@/components/ui';
import { font, makeStyles, radius, space, useColors } from '@/theme';

type Props = {
  visible: boolean;
  model: Object3D;
  mm: number; // the model's longest side now
  onClose: () => void;
  onApply: (longestMm: number) => void;
};

const mmText = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

// Two taps on the model give the distance between them at the size set. Typing what that distance really is sets the
// whole model's size: the way to true size for a part that must fit.
export function MeasureTool({ visible, model, mm, onClose, onApply }: Props) {
  const c = useColors();
  const s = useStyles();
  const [marks, setMarks] = useState<Spot[]>([]);
  const [box, setBox] = useState<readonly [number, number, number] | null>(null);
  const [real, setReal] = useState('');

  const pick = (spot: Spot, size: readonly [number, number, number]) => {
    setBox(size);
    setReal('');
    setMarks((now) => (now.length === 1 ? [now[0], spot] : [spot]));
  };

  const distance =
    marks.length === 2 && box
      ? (Math.hypot(...marks[0].map((v, i) => (v - marks[1][i]) * box[i])) * mm) / Math.max(...box)
      : null;
  const typed = Number(real.replace(',', '.'));
  const next = distance && typed > 0 ? (mm * typed) / distance : null;
  const valid = next != null && Number.isFinite(next) && next >= 1 && next <= 2000;

  return (
    <Modal
      visible={visible}
      animationType="fade"
      statusBarTranslucent
      navigationBarTranslucent
      onShow={() => {
        setMarks([]);
        setReal('');
      }}
      onRequestClose={onClose}
    >
      <SafeAreaProvider>
        <KeyboardAvoidingView style={s.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <View style={s.fill}>
            {visible && <ModelView model={model} wireframe={false} background={c.viewer} marks={marks} onPick={pick} />}
            <SafeAreaView edges={['top']} style={s.top} pointerEvents="box-none">
              <IconButton icon="close" label="Затвори" onPress={onClose} />
              <Text style={s.title} numberOfLines={1}>
                Измерване
              </Text>
              <View style={s.side} />
            </SafeAreaView>
            <SafeAreaView edges={['bottom']} style={s.bottom} pointerEvents="box-none">
              <View style={s.card}>
                {distance == null ? (
                  <Small style={s.center}>{marks.length ? 'Докоснете втората точка' : 'Докоснете две точки на модела'}</Small>
                ) : (
                  <>
                    <Text style={s.value}>{mmText(distance)} мм</Text>
                    <View style={s.row}>
                      <Field
                        value={real}
                        onChangeText={setReal}
                        keyboardType="decimal-pad"
                        placeholder="Реално, мм"
                        accessibilityLabel="Реално разстояние в милиметри"
                        style={s.field}
                      />
                      <Button title="Приложи" disabled={!valid} onPress={() => valid && onApply(next)} />
                    </View>
                    <Small>Въведете истинското разстояние и целият модел става в реален размер.</Small>
                  </>
                )}
              </View>
            </SafeAreaView>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaProvider>
    </Modal>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.viewer[1] },
  top: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingHorizontal: space.lg,
    paddingTop: space.sm,
  },
  title: { flex: 1, textAlign: 'center', fontFamily: font.semibold, fontSize: 15, letterSpacing: -0.2, color: c.text },
  side: { width: 44 },
  bottom: { position: 'absolute', left: 0, right: 0, bottom: 0, paddingHorizontal: space.lg, paddingBottom: space.md },
  card: { backgroundColor: c.glass, borderWidth: 1, borderColor: c.line, borderRadius: radius.md, padding: space.lg, gap: space.sm },
  center: { textAlign: 'center' },
  value: { fontFamily: font.monoMedium, fontSize: 24, color: c.text, textAlign: 'center' },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  field: { flex: 1 },
}));

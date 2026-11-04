import { useState } from 'react';
import { Modal, Text, View } from 'react-native';
import Animated from 'react-native-reanimated';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import type { Object3D } from 'three';

import { ModelView, type Spot } from '@/components/ModelView';
import { fade } from '@/components/motion';
import { Button, IconButton } from '@/components/ui';
import { font, makeStyles, space, useColors } from '@/theme';

type Props = {
  visible: boolean;
  model: Object3D;
  initial: Spot | null;
  onClose: () => void;
  onDone: (spot: Spot) => void;
};

// The model over the whole screen to point at the place a text change is for ("smooth here", "make this red"): a tap
// marks it with a dot, a drag turns the model. As FullscreenModel, the viewer drops its own small view meanwhile.
export function SpotPicker({ visible, model, initial, onClose, onDone }: Props) {
  const c = useColors();
  const s = useStyles();
  const [spot, setSpot] = useState<Spot | null>(initial);
  return (
    <Modal
      visible={visible}
      animationType="fade"
      statusBarTranslucent
      navigationBarTranslucent
      onShow={() => setSpot(initial)}
      onRequestClose={onClose}
    >
      <SafeAreaProvider>
        <View style={s.fill}>
          {visible && <ModelView model={model} wireframe={false} background={c.viewer} spot={spot} onPick={setSpot} />}
          <SafeAreaView edges={['top']} style={s.top} pointerEvents="box-none">
            <IconButton icon="close" label="Отказ" onPress={onClose} />
            <Text style={s.title} numberOfLines={1}>
              Посочете мястото
            </Text>
            <View style={s.side} />
          </SafeAreaView>
          <SafeAreaView edges={['bottom']} style={s.bottom} pointerEvents="box-none">
            <Animated.View entering={fade} style={s.hint} pointerEvents="none">
              <Text style={s.hintText}>{spot ? 'Мястото е отбелязано. Докоснете друго, за да го смените.' : 'Докоснете мястото · плъзнете, за да завъртите'}</Text>
            </Animated.View>
            <Button title="Готово" icon="checkmark" size="lg" disabled={!spot} onPress={() => spot && onDone(spot)} style={s.done} />
          </SafeAreaView>
        </View>
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
  bottom: { position: 'absolute', left: 0, right: 0, bottom: 0, alignItems: 'center', gap: space.md, paddingHorizontal: space.lg, paddingBottom: space.lg },
  hint: {
    backgroundColor: c.glass,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  hintText: { fontFamily: font.mono, fontSize: 11, color: c.muted, textAlign: 'center' },
  done: { alignSelf: 'stretch' },
}));

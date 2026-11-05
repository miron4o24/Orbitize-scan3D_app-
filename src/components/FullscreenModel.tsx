import { Modal, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import type { Object3D } from 'three';

import { ModelView } from '@/components/ModelView';
import { IconButton } from '@/components/ui';
import { font, makeStyles, space, useColors } from '@/theme';

type Props = {
  visible: boolean;
  model: Object3D;
  title: string;
  wireframe: boolean;
  onWireframe: () => void;
  plain: boolean;
  onPlain: () => void;
  onClose: () => void;
};

// Моделът на целия екран. Модел може да стои само в един изглед наведнъж, затова екранът за преглед маха малкия си
// изглед, докато това е отворено; бутонът за връщане го затваря.
export function FullscreenModel({ visible, model, title, wireframe, onWireframe, plain, onPlain, onClose }: Props) {
  const c = useColors();
  const s = useStyles();
  return (
    <Modal visible={visible} animationType="fade" statusBarTranslucent navigationBarTranslucent onRequestClose={onClose}>
      <SafeAreaProvider>
        <View style={s.fill}>
          {visible && <ModelView model={model} wireframe={wireframe} plain={plain} background={c.viewer} />}
          <SafeAreaView edges={['top']} style={s.top} pointerEvents="box-none">
            <IconButton icon="close" label="Затвори цял екран" onPress={onClose} />
            <Text style={s.title} numberOfLines={1}>
              {title}
            </Text>
            <IconButton icon="color-palette-outline" label={plain ? 'Покажи цветовете' : 'Покажи без цветове'} active={plain} onPress={onPlain} />
            <IconButton icon="grid-outline" label={wireframe ? 'Покажи повърхността' : 'Покажи мрежата'} active={wireframe} onPress={onWireframe} />
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
}));

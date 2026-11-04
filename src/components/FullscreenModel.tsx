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
  onClose: () => void;
};

// The model over the whole screen. A model can sit in one view at a time, so the viewer drops its own small view
// while this is open; the back button closes it.
export function FullscreenModel({ visible, model, title, wireframe, onWireframe, onClose }: Props) {
  const c = useColors();
  const s = useStyles();
  return (
    <Modal visible={visible} animationType="fade" statusBarTranslucent navigationBarTranslucent onRequestClose={onClose}>
      <SafeAreaProvider>
        <View style={s.fill}>
          {visible && <ModelView model={model} wireframe={wireframe} background={c.viewer} />}
          <SafeAreaView edges={['top']} style={s.top} pointerEvents="box-none">
            <IconButton icon="close" label="Затвори цял екран" onPress={onClose} />
            <Text style={s.title} numberOfLines={1}>
              {title}
            </Text>
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

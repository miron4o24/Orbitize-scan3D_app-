import { Image } from 'expo-image';
import { Text, View } from 'react-native';

import { font, makeStyles } from '@/theme';

const LOGO = require('../../assets/logo.png');

/* Иконата на приложението като малък знак (собствена черна плочка, еднаква и в двете теми). */
export function Logo({ size = 30 }: { size?: number }) {
  return <Image source={LOGO} style={{ width: size, height: size }} contentFit="contain" accessibilityIgnoresInvertColors />;
}

/* Знакът и името, горе вляво на началния екран и екрана за вход. */
export function Brand() {
  const s = useStyles();
  return (
    <View style={s.brand}>
      <Logo />
      <Text style={s.name}>Orbitize</Text>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  brand: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  name: { fontFamily: font.semibold, fontSize: 18, letterSpacing: -0.5, color: c.text },
}));

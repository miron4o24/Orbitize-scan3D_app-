import { Image } from 'expo-image';
import { Text, View } from 'react-native';

import { font, makeStyles } from '@/theme';

const LOGO = require('../../assets/logo.png');

/** The app icon as a small mark (its own black tile, the same in both themes). */
export function Logo({ size = 30 }: { size?: number }) {
  return <Image source={LOGO} style={{ width: size, height: size }} contentFit="contain" accessibilityIgnoresInvertColors />;
}

/** The mark and the name, top left of Home and the sign-in screen. */
export function Brand() {
  const s = useStyles();
  return (
    <View style={s.brand}>
      <Logo />
      <Text style={s.name}>Scanly</Text>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  brand: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  name: { fontFamily: font.semibold, fontSize: 18, letterSpacing: -0.5, color: c.text },
}));

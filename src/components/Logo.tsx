import { Text, View } from 'react-native';
import Svg, { Circle, Ellipse } from 'react-native-svg';

import { font, makeStyles, useColors } from '@/theme';

/* Знакът на Orbitize: предмет в средата, орбита около него и точката на камерата върху нея (в цвета на текста). */
export function Logo({ size = 24 }: { size?: number }) {
  const c = useColors();
  return (
    <Svg width={size} height={size} viewBox="0 0 64 64" accessibilityRole="image" accessibilityLabel="Orbitize">
      <Circle cx={32} cy={32} r={16} stroke={c.text} strokeWidth={4} fill="none" />
      <Ellipse cx={32} cy={32} rx={28} ry={10} rotation={-24} origin="32, 32" stroke={c.text} strokeWidth={3} fill="none" />
      <Circle cx={41.3} cy={18.4} r={5} fill={c.text} />
    </Svg>
  );
}

/* Знакът и името, горе вляво на началния екран и екрана за вход. */
export function Brand() {
  const s = useStyles();
  return (
    <View style={s.brand}>
      <Logo size={23} />
      <Text style={s.name}>Orbitize</Text>
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  brand: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  name: { fontFamily: font.semibold, fontSize: 17, letterSpacing: -0.34, color: c.text },
}));

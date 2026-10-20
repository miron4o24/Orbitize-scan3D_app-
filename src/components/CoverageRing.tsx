import { StyleSheet, View } from 'react-native';

import { BANDS, cellKey, SECTORS } from '@/lib/coverage';

const SIZE = 150;
const DOT = 9;
// Outer ring = low band, inner ring = looking down from above, like a dome seen from the top.
const RADII = [66, 48, 30];

type Props = { covered: Set<string>; current: { band: number; sector: number } | null };

export function CoverageRing({ covered, current }: Props) {
  const dots = [];
  for (let band = 0; band < BANDS; band++)
    for (let sector = 0; sector < SECTORS; sector++) {
      const a = ((sector + 0.5) / SECTORS) * 2 * Math.PI;
      const key = cellKey({ band, sector });
      const isCurrent = current?.band === band && current.sector === sector;
      dots.push(
        <View
          key={key}
          style={[
            styles.dot,
            {
              left: SIZE / 2 + RADII[band] * Math.sin(a) - DOT / 2,
              top: SIZE / 2 - RADII[band] * Math.cos(a) - DOT / 2,
              backgroundColor: covered.has(key) ? '#4caf50' : 'rgba(255,255,255,0.35)',
            },
            isCurrent && styles.current,
          ]}
        />,
      );
    }
  return (
    <View
      style={styles.ring}
      accessible
      accessibilityLabel={`Coverage: ${covered.size} of ${BANDS * SECTORS} angles captured`}
    >
      {dots}
    </View>
  );
}

const styles = StyleSheet.create({
  ring: { width: SIZE, height: SIZE },
  dot: { position: 'absolute', width: DOT, height: DOT, borderRadius: DOT / 2 },
  current: { borderWidth: 2, borderColor: '#ffeb3b' },
});

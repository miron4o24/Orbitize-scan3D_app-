import { StyleSheet, View } from 'react-native';

import { LOOPS, SECTORS, type OrbitState } from '@/lib/orbit';
import { palettes } from '@/theme';

const DOT = 10;
const GAP = 16; // between the two rings of dots

type Props = { size: number; orbit: OrbitState | null };

// The circle the object should stay inside, ringed by one row of dots per circle walked (outer = first).
// Dots turn green as the user walks past; the yellow one is where the phone is now.
export function OrbitGuide({ size, orbit }: Props) {
  const dots = [];
  for (let loop = 0; loop < LOOPS; loop++) {
    const r = size / 2 - DOT / 2 - loop * GAP;
    for (let sector = 0; sector < SECTORS; sector++) {
      const a = ((sector + 0.5) / SECTORS) * 2 * Math.PI;
      const covered = orbit?.covered[loop][sector];
      const active = orbit ? orbit.loop === loop : loop === 0;
      const here = active && orbit?.sector === sector;
      dots.push(
        <View
          key={`${loop}:${sector}`}
          style={[
            styles.dot,
            {
              left: size / 2 + r * Math.sin(a) - DOT / 2,
              top: size / 2 - r * Math.cos(a) - DOT / 2,
              backgroundColor: covered ? palettes.dark.success : active ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.2)',
            },
            here && styles.here,
          ]}
        />,
      );
    }
  }
  const inner = size - 2 * (DOT + LOOPS * GAP);
  const done = orbit?.covered.map((row) => row.filter(Boolean).length) ?? [0, 0];
  return (
    <View
      style={{ width: size, height: size }}
      pointerEvents="none"
      accessible
      accessibilityLabel={`Кръг 1: заснети ${done[0]} от ${SECTORS} стъпки. Кръг 2: ${done[1]} от ${SECTORS}.`}
    >
      <View
        style={[
          styles.target,
          { width: inner, height: inner, borderRadius: inner / 2, left: (size - inner) / 2, top: (size - inner) / 2 },
        ]}
      />
      {dots}
    </View>
  );
}

const styles = StyleSheet.create({
  dot: { position: 'absolute', width: DOT, height: DOT, borderRadius: DOT / 2 },
  here: { backgroundColor: palettes.dark.warning, transform: [{ scale: 1.5 }] },
  target: { position: 'absolute', borderWidth: 1.5, borderStyle: 'dashed', borderColor: 'rgba(255,255,255,0.6)' },
});

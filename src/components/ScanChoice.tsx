import Ionicons from '@expo/vector-icons/Ionicons';
import { LinearGradient } from 'expo-linear-gradient';
import { StyleSheet, Text, View } from 'react-native';

import { brandDeep, font, makeStyles, radius, space, useColors } from '@/theme';
import { PressScale } from './motion';
import type { IconName } from './ui';

const ON_BRAND = '#ffffff';

/**
 * Начин за започване на модел на началния екран: икона, име, колко време отнема и един малък ред. `highlight` е този
 * върху градиента на марката.
 */
export function ScanChoice({
  title,
  time,
  sub,
  icon,
  highlight,
  onPress,
}: {
  title: string;
  time: string;
  sub?: string;
  icon: IconName;
  highlight?: boolean;
  onPress: () => void;
}) {
  const c = useColors();
  const s = useStyles();
  const fg = highlight ? ON_BRAND : c.text;
  return (
    <PressScale
      onPress={onPress}
      feedback
      accessibilityRole="button"
      accessibilityLabel={[title, time, sub].filter(Boolean).join('. ')}
      style={[s.card, !highlight && s.plain]}
    >
      {highlight && <LinearGradient colors={brandDeep} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={StyleSheet.absoluteFill} />}
      <View style={[s.icon, highlight ? s.iconOnBrand : s.iconPlain]}>
        <Ionicons name={icon} size={21} color={fg} />
      </View>
      <View style={s.lines}>
        <View style={s.top}>
          <Text style={[s.title, { color: fg }]} numberOfLines={1}>
            {title}
          </Text>
          <View style={[s.time, highlight ? s.timeOnBrand : s.timePlain]}>
            <Text style={[s.timeText, { color: fg }]}>{time}</Text>
          </View>
        </View>
        {sub && (
          <Text style={[s.sub, { color: highlight ? ON_BRAND : c.muted }]} numberOfLines={2}>
            {sub}
          </Text>
        )}
      </View>
      <Ionicons name="chevron-forward" size={18} color={highlight ? ON_BRAND : c.muted} />
    </PressScale>
  );
}

const useStyles = makeStyles((c) => ({
  card: {
    minHeight: 84,
    borderRadius: radius.md,
    overflow: 'hidden',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    paddingHorizontal: space.lg,
    paddingVertical: 14,
  },
  plain: { backgroundColor: c.surface, borderWidth: 1, borderColor: c.lineStrong },
  icon: { width: 44, height: 44, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
  iconOnBrand: { backgroundColor: 'rgba(255,255,255,0.18)' },
  iconPlain: { backgroundColor: c.surface3 },
  lines: { flex: 1, minWidth: 0, gap: 3 },
  top: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  title: { flexShrink: 1, fontFamily: font.semibold, fontSize: 17, lineHeight: 22, letterSpacing: -0.3 },
  time: { borderRadius: radius.pill, paddingHorizontal: space.sm, paddingVertical: 2 },
  timeOnBrand: { backgroundColor: 'rgba(255,255,255,0.18)' },
  timePlain: { backgroundColor: c.surface3 },
  timeText: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 15 },
  sub: { fontFamily: font.regular, fontSize: 13, lineHeight: 18 },
}));

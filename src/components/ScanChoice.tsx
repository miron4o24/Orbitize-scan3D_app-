import Ionicons from '@expo/vector-icons/Ionicons';
import { Text, View } from 'react-native';

import { font, makeStyles, radius, space, useColors } from '@/theme';
import { PressScale } from './motion';
import type { IconName } from './ui';

/**
 * Начин за започване на модел на началния екран: икона, име, колко време отнема и един малък ред. `highlight` е
 * препоръчаният: мек фон в цвета за акцент с рамка в него.
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
  const fg = highlight ? c.accent : c.text;
  return (
    <PressScale
      onPress={onPress}
      feedback
      accessibilityRole="button"
      accessibilityLabel={[title, time, sub].filter(Boolean).join('. ')}
      style={[s.card, highlight ? s.highlight : s.plain]}
    >
      <View style={[s.icon, highlight ? s.iconOn : s.iconPlain]}>
        <Ionicons name={icon} size={20} color={fg} />
      </View>
      <View style={s.lines}>
        <View style={s.top}>
          <Text style={s.title} numberOfLines={1}>
            {title}
          </Text>
          <View style={[s.time, highlight && s.timeOn]}>
            <Text style={[s.timeText, highlight && { color: c.accent }]}>{time}</Text>
          </View>
        </View>
        {sub && (
          <Text style={s.sub} numberOfLines={2}>
            {sub}
          </Text>
        )}
      </View>
      <Ionicons name="chevron-forward" size={18} color={highlight ? c.accent : c.muted} />
    </PressScale>
  );
}

const useStyles = makeStyles((c) => ({
  card: {
    minHeight: 76,
    borderRadius: radius.md,
    borderWidth: 1,
    overflow: 'hidden',
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
    paddingHorizontal: space.lg,
    paddingVertical: space.md,
  },
  plain: { backgroundColor: c.surface, borderColor: c.lineStrong },
  highlight: { backgroundColor: c.accentSoft, borderColor: c.accentLine },
  icon: { width: 40, height: 40, borderRadius: radius.sm, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  iconOn: { borderColor: c.accentLine },
  iconPlain: { borderColor: c.line, backgroundColor: c.surface2 },
  lines: { flex: 1, minWidth: 0, gap: 3 },
  top: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  title: { flexShrink: 1, fontFamily: font.semibold, fontSize: 16, lineHeight: 22, color: c.text },
  time: { borderRadius: radius.pill, borderWidth: 1, borderColor: c.lineStrong, paddingHorizontal: 7, paddingVertical: 1 },
  timeOn: { borderColor: c.accentLine },
  timeText: { fontFamily: font.monoMedium, fontSize: 11, lineHeight: 15, color: c.text2 },
  sub: { fontFamily: font.regular, fontSize: 13, lineHeight: 18, color: c.text2 },
}));

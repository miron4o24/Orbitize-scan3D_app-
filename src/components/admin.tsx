import Ionicons from '@expo/vector-icons/Ionicons';
import { Pressable, Text, View } from 'react-native';

import { adminAvatarUrl, type AdminUser, type JobState } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { font, makeStyles, radius, space, useColors } from '@/theme';
import { Avatar } from './Avatar';

type IconName = keyof typeof Ionicons.glyphMap;
type Tone = 'accent' | 'success' | 'warning' | 'danger' | 'muted';

/* Колко отдавна, накратко: „току-що“, „преди 5 мин“, „преди 3 ч“, „преди 2 дни“, после датата. */
export function ago(unixSeconds: number | null | undefined): string {
  if (!unixSeconds) return 'никога';
  const s = Math.max(0, Date.now() / 1000 - unixSeconds);
  if (s < 90) return 'току-що';
  if (s < 3600) return `преди ${Math.round(s / 60)} мин`;
  if (s < 86400) return `преди ${Math.round(s / 3600)} ч`;
  if (s < 7 * 86400) return `преди ${Math.round(s / 86400)} ${Math.round(s / 86400) === 1 ? 'ден' : 'дни'}`;
  return shortDate(unixSeconds * 1000);
}

/* Малък етикет с рамка: дума в цвета на значението си (никога само цвят). */
export function Chip({ label, tone = 'muted', icon }: { label: string; tone?: Tone; icon?: IconName }) {
  const c = useColors();
  const s = useStyles();
  const color = tone === 'muted' ? c.muted : c[tone];
  return (
    <View style={[s.chip, { borderColor: color }]}>
      {icon && <Ionicons name={icon} size={11} color={color} />}
      <Text style={[s.chipText, { color }]}>{label}</Text>
    </View>
  );
}

/* Етикетите на профил: собственик, администратор, блокиран, непотвърден. */
export function UserChips({ user }: { user: Pick<AdminUser, 'owner' | 'admin' | 'blocked' | 'verified'> }) {
  const s = useStyles();
  const chips = [
    user.owner ? <Chip key="o" label="Собственик" tone="accent" icon="star-outline" /> : user.admin ? <Chip key="a" label="Админ" tone="accent" icon="shield-checkmark-outline" /> : null,
    user.blocked ? <Chip key="b" label="Блокиран" tone="danger" icon="ban-outline" /> : null,
    !user.verified ? <Chip key="v" label="Непотвърден" tone="warning" icon="mail-unread-outline" /> : null,
  ].filter(Boolean);
  return chips.length ? <View style={s.chips}>{chips}</View> : null;
}

/* Профил в списъка: снимка, име, имейл и етикетите, вдясно сканиранията и кога е бил активен. */
export function UserLine({ user, onPress, last }: { user: AdminUser; onPress: () => void; last?: boolean }) {
  const c = useColors();
  const s = useStyles();
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={`${user.name}, ${user.email}`}
      style={({ pressed }) => [s.user, !last && s.line, pressed && s.pressed]}
    >
      <Avatar user={user} size={40} photo={user.has_avatar ? adminAvatarUrl(user.id) : undefined} />
      <View style={s.flex}>
        <Text style={[s.name, user.blocked && { color: c.muted }]} numberOfLines={1}>
          {user.name}
        </Text>
        <Text style={s.email} numberOfLines={1}>
          {user.email}
        </Text>
        <UserChips user={user} />
      </View>
      <View style={s.right}>
        <Text style={s.count}>{user.scans}</Text>
        <Text style={s.when}>{ago(user.last_seen)}</Text>
      </View>
      <Ionicons name="chevron-forward" size={16} color={c.muted} />
    </Pressable>
  );
}

/* Плочки с числа (2 в ред): голямо число, име отдолу, по желание натискане към списъка. */
export function Tiles({ items }: { items: { label: string; value: number | string; tone?: Tone; onPress?: () => void }[] }) {
  const c = useColors();
  const s = useStyles();
  return (
    <View style={s.tiles}>
      {items.map((it) => (
        <Pressable
          key={it.label}
          onPress={it.onPress}
          disabled={!it.onPress}
          accessibilityRole={it.onPress ? 'button' : undefined}
          style={({ pressed }) => [s.tile, pressed && s.pressed]}
        >
          <Text style={[s.tileValue, it.tone && it.tone !== 'muted' && { color: c[it.tone] }]} numberOfLines={1} adjustsFontSizeToFit>
            {it.value}
          </Text>
          <Text style={s.tileLabel} numberOfLines={2}>
            {it.label}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

export const STATE_TEXT: Record<JobState, string> = {
  uploading: 'Качва се',
  queued: 'Чака',
  reconstructing: 'Обработва се',
  meshing: 'Обработва се',
  cleaning: 'Обработва се',
  done: 'Готово',
  failed: 'Грешка',
  cancelled: 'Спряно',
};

export const stateTone = (state: JobState): Tone =>
  state === 'done' ? 'success' : state === 'failed' ? 'danger' : state === 'cancelled' ? 'muted' : 'warning';

export const methodText = (m?: string | null) =>
  m === 'cad' ? 'Инженерен режим' : m === 'single' ? 'AI модел' : m === 'edit' ? 'Промяна' : m === 'scan' || m === 'express' ? 'Сканиране' : m ? `AI (${m})` : '—';

const useStyles = makeStyles((c) => ({
  flex: { flex: 1, gap: 2 },
  pressed: { opacity: 0.7 },
  line: { borderBottomWidth: 1, borderBottomColor: c.line },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 4, borderWidth: 1, borderRadius: radius.pill, paddingHorizontal: 7, paddingVertical: 1 },
  chipText: { fontFamily: font.monoMedium, fontSize: 10.5, lineHeight: 15 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 4 },
  user: { flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.md },
  name: { fontFamily: font.semibold, fontSize: 15, color: c.text },
  email: { fontFamily: font.mono, fontSize: 12, color: c.muted },
  right: { alignItems: 'flex-end', gap: 2 },
  count: { fontFamily: font.monoMedium, fontSize: 15, color: c.text },
  when: { fontFamily: font.mono, fontSize: 10.5, color: c.muted },
  tiles: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: space.sm },
  tile: {
    width: '48.5%',
    minHeight: 84,
    justifyContent: 'space-between',
    padding: space.md,
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.line,
    borderRadius: radius.md,
  },
  tileValue: { fontFamily: font.monoMedium, fontSize: 26, lineHeight: 32, color: c.text },
  tileLabel: { fontFamily: font.regular, fontSize: 12.5, lineHeight: 16, color: c.muted },
}));

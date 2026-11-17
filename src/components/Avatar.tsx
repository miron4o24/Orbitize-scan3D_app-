import Ionicons from '@expo/vector-icons/Ionicons';
import { Image } from 'expo-image';
import { useState } from 'react';
import { Text, View } from 'react-native';

import { authHeaders, avatarUrl } from '@/lib/api';
import { getAvatarVersion, type Profile } from '@/lib/session';
import { font, makeStyles, radius, useColors } from '@/theme';

type Person = Pick<Profile, 'name' | 'has_avatar' | 'google_picture'>;

/**
 * Собствената качена снимка, иначе снимката от Google, иначе инициалите (или знак за човек) в квадрат с рамка. photo:
 * къде е качената снимка на някой друг (автор в общността); по подразбиране тази на влезлия потребител.
 */
export function Avatar({ user, size, photo }: { user: Person | null; size: number; photo?: string }) {
  const c = useColors();
  const s = useStyles();
  const version = getAvatarVersion();
  const [failed, setFailed] = useState<string | null>(null);
  const source = user?.has_avatar
    ? { uri: photo ?? avatarUrl(version), headers: authHeaders() }
    : user?.google_picture
      ? { uri: user.google_picture }
      : null;
  const initials = (user?.name ?? '')
    .split(/\s+/)
    .map((w) => w[0])
    .filter(Boolean)
    .slice(0, 2)
    .join('')
    .toUpperCase();
  const box = { width: size, height: size, borderRadius: size >= 48 ? radius.md : radius.sm };
  return (
    <View style={[s.box, box]}>
      {source && failed !== source.uri ? (
        <Image source={source} style={{ width: size, height: size }} contentFit="cover" transition={150} onError={() => setFailed(source.uri)} />
      ) : initials ? (
        <Text style={[s.initials, { fontSize: size * 0.36 }]}>{initials}</Text>
      ) : (
        <Ionicons name="person-outline" size={size * 0.5} color={c.text2} />
      )}
    </View>
  );
}

const useStyles = makeStyles((c) => ({
  box: {
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    backgroundColor: c.surface2,
    borderWidth: 1,
    borderColor: c.lineStrong,
  },
  initials: { color: c.text2, fontFamily: font.monoMedium },
}));

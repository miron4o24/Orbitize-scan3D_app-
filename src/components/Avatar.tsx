import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { authHeaders, avatarUrl } from '@/lib/api';
import { getAvatarVersion, type Profile } from '@/lib/session';
import { brand, font } from '@/theme';

type Person = Pick<Profile, 'name' | 'has_avatar' | 'google_picture'>;

/**
 * Собствената качена снимка, иначе снимката от Google, иначе инициалите върху градиента на марката. photo: къде е
 * качената снимка на някой друг (автор в общността); по подразбиране тази на влезлия потребител.
 */
export function Avatar({ user, size, photo }: { user: Person | null; size: number; photo?: string }) {
  const version = getAvatarVersion();
  const [failed, setFailed] = useState<string | null>(null);
  const source = user?.has_avatar
    ? { uri: photo ?? avatarUrl(version), headers: authHeaders() }
    : user?.google_picture
      ? { uri: user.google_picture }
      : null;
  const initials = (user?.name ?? '?')
    .split(/\s+/)
    .map((w) => w[0])
    .filter(Boolean)
    .slice(0, 2)
    .join('')
    .toUpperCase();
  const round = { width: size, height: size, borderRadius: size / 2 };
  return (
    <View style={[styles.circle, round]}>
      {source && failed !== source.uri ? (
        <Image source={source} style={round} contentFit="cover" transition={150} onError={() => setFailed(source.uri)} />
      ) : (
        <>
          <LinearGradient colors={brand} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={StyleSheet.absoluteFill} />
          <Text style={[styles.initials, { fontSize: size * 0.36 }]}>{initials}</Text>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  circle: { alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
  initials: { color: '#ffffff', fontFamily: font.semibold },
});

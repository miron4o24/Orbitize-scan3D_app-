import { router } from 'expo-router';
import { Alert } from 'react-native';

import { PRO_INFO } from './prefs';

/**
 * Ново сканиране от всяко място освен бутоните на началния екран: точна част (снимки и мерки) или видео (фигури и
 * всичко останало).
 */
export function startScan() {
  Alert.alert('Какъв модел?', undefined, [
    { text: 'Отказ', style: 'cancel' },
    { text: `${PRO_INFO.part.label} · ${PRO_INFO.part.time}`, onPress: () => router.push('/part') },
    { text: `${PRO_INFO.single.label} · ${PRO_INFO.single.time}`, onPress: () => router.push('/capture') },
  ]);
}

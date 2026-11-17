import { router } from 'expo-router';
import { Alert } from 'react-native';

import { aiAvailable, CAD_INFO, PRO_INFO } from './prefs';

/**
 * Нов модел от всяко място освен бутоните на началния екран: потребителят казва кой начин (AI модел, ако е наличен,
 * или инженерният режим за детайл 1:1 от снимки и шублер).
 */
export function startScan() {
  const engineer = { text: `${CAD_INFO.label} · ${CAD_INFO.time}`, onPress: () => router.push('/engineer') };
  Alert.alert('Какъв модел?', undefined, [
    { text: 'Отказ', style: 'cancel' },
    ...(aiAvailable() ? [{ text: `${PRO_INFO.single.label} · ${PRO_INFO.single.time}`, onPress: () => router.push('/capture') }] : []),
    engineer,
  ]);
}

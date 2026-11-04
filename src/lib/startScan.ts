import { router } from 'expo-router';
import { Alert } from 'react-native';

import { startFromPhoto } from './importPhoto';
import { aiAvailable, ONE_PHOTO_INFO, PRO_INFO, SCAN_INFO } from './prefs';

/**
 * Ново сканиране от всяко място освен бутоните на началния екран: при наличие на двата начина потребителят казва кой.
 */
export function startScan() {
  if (!aiAvailable()) {
    router.push('/capture');
    return;
  }
  Alert.alert('Какъв модел?', undefined, [
    { text: 'Отказ', style: 'cancel' },
    { text: `${ONE_PHOTO_INFO.label} · ${ONE_PHOTO_INFO.time}`, onPress: startFromPhoto },
    { text: `${SCAN_INFO.label} · ${SCAN_INFO.time}`, onPress: () => router.push('/capture') },
    { text: `${PRO_INFO.single.label} · ${PRO_INFO.single.time}`, onPress: () => router.push({ pathname: '/capture', params: { ai: '1' } }) },
  ]);
}

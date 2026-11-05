import { router } from 'expo-router';
import { Alert } from 'react-native';

import { startFromPhoto } from './importPhoto';
import { ONE_PHOTO_INFO, PRO_INFO } from './prefs';

/**
 * Ново сканиране от всяко място освен бутоните на началния екран: потребителят казва от видео или от снимка.
 */
export function startScan() {
  Alert.alert('Какъв модел?', undefined, [
    { text: 'Отказ', style: 'cancel' },
    { text: `${ONE_PHOTO_INFO.label} · ${ONE_PHOTO_INFO.time}`, onPress: startFromPhoto },
    { text: `${PRO_INFO.single.label} · ${PRO_INFO.single.time}`, onPress: () => router.push('/capture') },
  ]);
}

import { router } from 'expo-router';

import { aiAvailable, CAD_INFO, PRO_INFO } from './prefs';
import { showDialog } from './dialog';

/**
 * Нов модел от всяко място освен бутоните на началния екран: потребителят казва кой начин (AI модел, ако е наличен,
 * или инженерният режим за детайл 1:1 от снимки и шублер).
 */
export function startScan() {
  const engineer = {
    text: CAD_INFO.label,
    sub: `${CAD_INFO.time} · детайл 1:1 от снимки и шублер`,
    icon: 'construct-outline',
    onPress: () => router.push('/engineer'),
  };
  showDialog('Какъв модел?', undefined, [
    ...(aiAvailable()
      ? [{ text: PRO_INFO.single.label, sub: `${PRO_INFO.single.time} · видео около предмета`, icon: 'sparkles-outline', onPress: () => router.push('/capture') }]
      : []),
    engineer,
    { text: 'Отказ', style: 'cancel' as const },
  ]);
}

import Constants, { ExecutionEnvironment } from 'expo-constants';
import { router } from 'expo-router';
import { Platform } from 'react-native';

import { forgetPushToken, savePushToken } from './api';
import { listScans } from './scans';

// "Вашият модел е готов", когато приложението е затворено. Сървърът го изпраща през услугата за известия на Expo
// (сървър app/notify.py). Не работи в Expo Go, което няма известия на Android: тогава нищо тук не прави нищо. Същото,
// когато компилацията няма файл на Firebase (google-services.json): телефонът не получава ключ, а приложението работи
// без известия.

type Notifications = typeof import('expo-notifications');

let loaded: Notifications | null | undefined;
function load(): Notifications | null {
  if (loaded === undefined) {
    try {
      loaded =
        Constants.executionEnvironment === ExecutionEnvironment.StoreClient
          ? null
          : // eslint-disable-next-line @typescript-eslint/no-require-imports -- must load lazily, see above
            (require('expo-notifications') as Notifications);
    } catch {
      loaded = null;
    }
  }
  return loaded;
}

let token: string | null = null;

/**
 * Регистрира този телефон за влезлия профил. ask: може да покаже въпроса на системата; иначе само ако вече е
 * разрешено.
 */
export async function setUpPush(ask: boolean) {
  const N = load();
  if (!N) return;
  try {
    if (Platform.OS === 'android')
      await N.setNotificationChannelAsync('scans', { name: 'Готови модели', importance: N.AndroidImportance.HIGH });
    let { granted, canAskAgain } = await N.getPermissionsAsync();
    if (!granted && ask && canAskAgain) ({ granted } = await N.requestPermissionsAsync());
    if (!granted) return;
    const projectId = Constants.expoConfig?.extra?.eas?.projectId as string | undefined;
    const { data } = await N.getExpoPushTokenAsync({ projectId });
    token = data;
    await savePushToken(data);
  } catch {} // няма известия, нищо друго не се губи
}

/* Изход от профила: телефонът спира да получава известия за него. */
export async function stopPush() {
  if (token) await forgetPushToken(token).catch(() => {});
  token = null;
}

function open(jobId: unknown) {
  const scan = listScans().find((s) => s.jobId === jobId);
  if (scan) router.push({ pathname: scan.status === 'done' ? '/viewer/[id]' : '/processing/[id]', params: { id: scan.id } });
  else router.push('/scans');
}

/* Показва се и при отворено приложение; докосването отваря модела. Връща как да спре слушането. */
export function listenToPush(): () => void {
  const N = load();
  if (!N) return () => {};
  N.setNotificationHandler({
    handleNotification: async () => ({ shouldShowBanner: true, shouldShowList: true, shouldPlaySound: true, shouldSetBadge: false }),
  });
  const sub = N.addNotificationResponseReceivedListener((r) => open(r.notification.request.content.data?.jobId));
  N.getLastNotificationResponseAsync()
    .then((r) => {
      if (!r) return;
      open(r.notification.request.content.data?.jobId);
      return N.clearLastNotificationResponseAsync();
    })
    .catch(() => {});
  return () => sub.remove();
}

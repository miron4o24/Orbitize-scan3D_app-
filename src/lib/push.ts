import Constants, { ExecutionEnvironment } from 'expo-constants';
import { router } from 'expo-router';
import { Platform } from 'react-native';

import { forgetPushToken, savePushToken } from './api';
import { listScans } from './scans';

// "Your model is ready", when the app is closed. The server sends it through Expo's push service (backend
// app/notify.py). Not in Expo Go, which has no push on Android: everything here then does nothing. The same when the
// build has no Firebase file (google-services.json): the phone gets no token, and the app works without notifications.

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

/** Registers this phone for the signed-in account. ask: may show the system's question; else only if already allowed. */
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
  } catch {} // no notifications, nothing else lost
}

/** Signing out: this phone stops hearing about the account. */
export async function stopPush() {
  if (token) await forgetPushToken(token).catch(() => {});
  token = null;
}

function open(jobId: unknown) {
  const scan = listScans().find((s) => s.jobId === jobId);
  if (scan) router.push({ pathname: scan.status === 'done' ? '/viewer/[id]' : '/processing/[id]', params: { id: scan.id } });
  else router.push('/scans');
}

/** Shown also while the app is open; a tap opens the model. Returns how to stop listening. */
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

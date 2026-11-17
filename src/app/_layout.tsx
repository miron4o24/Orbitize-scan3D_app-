import { IBMPlexMono_400Regular, IBMPlexMono_500Medium } from '@expo-google-fonts/ibm-plex-mono';
import { IBMPlexSans_400Regular, IBMPlexSans_500Medium, IBMPlexSans_600SemiBold, useFonts } from '@expo-google-fonts/ibm-plex-sans';
import { NavigationBar } from 'expo-navigation-bar';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import * as SystemUI from 'expo-system-ui';
import { useEffect } from 'react';
import { Appearance, AppState } from 'react-native';

import { Celebrations } from '@/components/Celebration';
import { DialogHost } from '@/components/DialogHost';
import { refreshProfile } from '@/lib/auth';
import { checkProgress } from '@/lib/progress';
import { listenToPush, setUpPush } from '@/lib/push';
import { useSession } from '@/lib/session';
import { font, useColors } from '@/theme';

SplashScreen.preventAutoHideAsync().catch(() => {});

export default function RootLayout() {
  const session = useSession();
  const c = useColors();
  // Шрифт, който не се зарежда, преминава към системния, вместо да задържа началния екран.
  const [loaded, failed] = useFonts({
    IBMPlexSans_400Regular,
    IBMPlexSans_500Medium,
    IBMPlexSans_600SemiBold,
    IBMPlexMono_400Regular,
    IBMPlexMono_500Medium,
  });
  const ready = loaded || !!failed;
  useEffect(() => {
    if (ready) SplashScreen.hideAsync().catch(() => {});
  }, [ready]);

  // Системните части на телефона следват темата на приложението: системните диалози, клавиатурата, прозорецът зад
  // екраните.
  useEffect(() => {
    Appearance.setColorScheme(c.scheme);
    SystemUI.setBackgroundColorAsync(c.bg).catch(() => {});
  }, [c]);

  // Точките, спечелени междувременно (завършено на сървъра сканиране, харесване на публикация), се празнуват при
  // връщане; профилът също се опреснява, защото копието на телефона решава кои сканирания да се предлагат.
  const account = session?.kind === 'account' ? session.user.id : null;
  useEffect(() => {
    if (!account) return;
    checkProgress(true);
    refreshProfile().catch(() => {});
    setUpPush(false); // самият въпрос идва с първото сканиране (екран за обработка)
    const stopListening = listenToPush();
    const sub = AppState.addEventListener('change', (st) => {
      if (st !== 'active') return;
      checkProgress();
      refreshProfile().catch(() => {});
    });
    return () => {
      sub.remove();
      stopListening();
    };
  }, [account]);

  if (!ready) return null;

  const bars = c.scheme === 'dark' ? 'light' : 'dark'; // светли икони при тъмна тема
  const inside = session !== null;
  return (
    <>
      <StatusBar style={bars} />
      <NavigationBar style={bars} />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: c.bg },
          headerShadowVisible: false,
          headerTintColor: c.text,
          headerTitleStyle: { color: c.text, fontFamily: font.semibold, fontSize: 16 },
          contentStyle: { backgroundColor: c.bg },
          animation: 'slide_from_right', // страницата се отваря отстрани; камерата се вдига отдолу
        }}
      >
        <Stack.Protected guard={inside}>
          <Stack.Screen name="(tabs)" options={{ headerShown: false, animation: 'fade' }} />
          <Stack.Screen name="capture" options={{ headerShown: false, animation: 'fade_from_bottom' }} />
          <Stack.Screen name="engineer" options={{ headerShown: false, animation: 'fade_from_bottom' }} />
          <Stack.Screen name="processing/[id]" options={{ title: '' }} />
          <Stack.Screen name="viewer/[id]" options={{ headerShown: false }} />
          <Stack.Screen name="community/[id]" options={{ headerShown: false }} />
          <Stack.Screen name="stress/[id]" options={{ headerShown: false }} />
        </Stack.Protected>
        <Stack.Protected guard={!inside}>
          <Stack.Screen name="sign-in" options={{ headerShown: false, animation: 'fade' }} />
        </Stack.Protected>
      </Stack>
      <Celebrations />
      <DialogHost />
    </>
  );
}

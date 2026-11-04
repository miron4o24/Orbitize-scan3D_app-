import { JetBrainsMono_400Regular, JetBrainsMono_500Medium } from '@expo-google-fonts/jetbrains-mono';
import { Onest_400Regular, Onest_500Medium, Onest_600SemiBold, useFonts } from '@expo-google-fonts/onest';
import { NavigationBar } from 'expo-navigation-bar';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import * as SystemUI from 'expo-system-ui';
import { useEffect } from 'react';
import { Appearance, AppState } from 'react-native';

import { Celebrations } from '@/components/Celebration';
import { refreshProfile } from '@/lib/auth';
import { checkProgress } from '@/lib/progress';
import { listenToPush, setUpPush } from '@/lib/push';
import { useSession } from '@/lib/session';
import { font, useColors } from '@/theme';

SplashScreen.preventAutoHideAsync().catch(() => {});

export default function RootLayout() {
  const session = useSession();
  const c = useColors();
  // A font that fails to load falls back to the system one rather than keeping the splash screen up.
  const [loaded, failed] = useFonts({
    Onest_400Regular,
    Onest_500Medium,
    Onest_600SemiBold,
    JetBrainsMono_400Regular,
    JetBrainsMono_500Medium,
  });
  const ready = loaded || !!failed;
  useEffect(() => {
    if (ready) SplashScreen.hideAsync().catch(() => {});
  }, [ready]);

  // The phone's own parts follow the app's theme: system dialogs, the keyboard, the window behind the screens.
  useEffect(() => {
    Appearance.setColorScheme(c.scheme);
    SystemUI.setBackgroundColorAsync(c.bg).catch(() => {});
  }, [c]);

  // Points earned meanwhile (a scan finished on the server, a like on a post) are celebrated on coming back; the
  // profile (Pro) comes fresh too, as the phone's copy is what decides which scans are offered.
  const account = session?.kind === 'account' ? session.user.id : null;
  useEffect(() => {
    if (!account) return;
    checkProgress(true);
    refreshProfile().catch(() => {});
    setUpPush(false); // the question itself comes with the first scan (processing screen)
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

  const bars = c.scheme === 'dark' ? 'light' : 'dark'; // light icons on the dark theme
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
          animation: 'slide_from_right', // a page opens from the side; the camera rises from below
        }}
      >
        <Stack.Protected guard={inside}>
          <Stack.Screen name="(tabs)" options={{ headerShown: false, animation: 'fade' }} />
          <Stack.Screen name="capture" options={{ headerShown: false, animation: 'fade_from_bottom' }} />
          <Stack.Screen name="processing/[id]" options={{ title: '' }} />
          <Stack.Screen name="viewer/[id]" options={{ headerShown: false }} />
          <Stack.Screen name="community/[id]" options={{ headerShown: false }} />
          <Stack.Screen name="stress/[id]" options={{ headerShown: false }} />
          <Stack.Screen name="owner" options={{ title: '' }} />
          <Stack.Screen name="pro" options={{ title: '' }} />
        </Stack.Protected>
        <Stack.Protected guard={!inside}>
          <Stack.Screen name="sign-in" options={{ headerShown: false, animation: 'fade' }} />
        </Stack.Protected>
      </Stack>
      <Celebrations />
    </>
  );
}

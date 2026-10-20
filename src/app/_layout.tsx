import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="auto" />
      <Stack>
        <Stack.Screen name="index" options={{ title: '3D Scan' }} />
        <Stack.Screen name="capture" options={{ title: 'Capture', headerShown: false }} />
        <Stack.Screen name="processing/[id]" options={{ title: 'Processing' }} />
        <Stack.Screen name="viewer/[id]" options={{ title: 'Model' }} />
        <Stack.Screen name="history" options={{ title: 'History' }} />
      </Stack>
    </>
  );
}

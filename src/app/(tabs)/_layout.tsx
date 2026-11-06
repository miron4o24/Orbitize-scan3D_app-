import Ionicons from '@expo/vector-icons/Ionicons';
import { Tabs } from 'expo-router';

import { useColors } from '@/theme';

export default function TabsLayout() {
  const c = useColors();
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: c.accent,
        tabBarInactiveTintColor: c.muted,
        tabBarStyle: { backgroundColor: c.bg, borderTopColor: c.line },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{ title: 'Test bench', tabBarIcon: ({ color, size }) => <Ionicons name="terminal-outline" size={size} color={color as string} /> }}
      />
      <Tabs.Screen
        name="scans"
        options={{ title: 'Jobs', tabBarIcon: ({ color, size }) => <Ionicons name="list-outline" size={size} color={color as string} /> }}
      />
      <Tabs.Screen name="community" options={{ href: null }} />
      <Tabs.Screen name="profile" options={{ href: null }} />
    </Tabs>
  );
}

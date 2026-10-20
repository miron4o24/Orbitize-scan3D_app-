import { Link } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { API_URL } from '@/lib/api';

const TIPS = [
  'Matte, textured objects work best.',
  'Avoid shiny, transparent or plain single-colour objects.',
  'Use even, diffuse light. No harsh shadows.',
  'Never move the object. You walk around it.',
  'Put it on a non-reflective surface with some texture (newspaper works).',
];

export default function Home() {
  return (
    <View style={styles.container}>
      <Link href="/capture" asChild>
        <Pressable style={styles.primary} accessibilityRole="button">
          <Text style={styles.primaryText}>New scan</Text>
        </Pressable>
      </Link>
      <Link href="/history" asChild>
        <Pressable style={styles.secondary} accessibilityRole="button">
          <Text style={styles.secondaryText}>History</Text>
        </Pressable>
      </Link>
      <Text style={styles.heading}>Tips for a good scan</Text>
      {TIPS.map((t) => (
        <Text key={t} style={styles.tip}>
          • {t}
        </Text>
      ))}
      <Text style={styles.footer}>Backend: {API_URL || 'not configured'}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 12, backgroundColor: '#fff' },
  primary: { backgroundColor: '#1565c0', padding: 18, borderRadius: 12, alignItems: 'center' },
  primaryText: { color: '#fff', fontSize: 18, fontWeight: '600' },
  secondary: { borderColor: '#1565c0', borderWidth: 1, padding: 14, borderRadius: 12, alignItems: 'center' },
  secondaryText: { color: '#1565c0', fontSize: 16 },
  heading: { marginTop: 16, fontSize: 16, fontWeight: '600' },
  tip: { fontSize: 14, color: '#333' },
  footer: { marginTop: 'auto', fontSize: 12, color: '#888' },
});

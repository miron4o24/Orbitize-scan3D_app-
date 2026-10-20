import { Link, useLocalSearchParams } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';

import { getScan } from '@/lib/scans';

// Placeholder until milestone 4 (upload + polling).
export default function Processing() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const scan = getScan(id);

  if (!scan) return <Text style={styles.container}>Scan not found.</Text>;

  return (
    <View style={styles.container}>
      <Text>
        {scan.photoCount} photos · {scan.status}
      </Text>
      <Link href={{ pathname: '/viewer/[id]', params: { id } }}>Open viewer</Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 12 },
});

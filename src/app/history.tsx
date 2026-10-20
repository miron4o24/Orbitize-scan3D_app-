import { Link, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';

import { listScans, type Scan } from '@/lib/scans';

export default function History() {
  const [scans, setScans] = useState<Scan[]>([]);
  useFocusEffect(useCallback(() => setScans(listScans()), []));

  return (
    <FlatList
      data={scans}
      keyExtractor={(s) => s.id}
      contentContainerStyle={styles.list}
      ListEmptyComponent={<Text style={styles.empty}>No scans yet.</Text>}
      renderItem={({ item }) => (
        <Link
          href={
            item.status === 'done'
              ? { pathname: '/viewer/[id]', params: { id: item.id } }
              : { pathname: '/processing/[id]', params: { id: item.id } }
          }
        >
          <View style={styles.row}>
            <Text style={styles.title}>{new Date(item.createdAt).toLocaleString()}</Text>
            <Text style={styles.sub}>
              {item.photoCount} photos · {item.status}
            </Text>
          </View>
        </Link>
      )}
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: 16, gap: 8 },
  empty: { textAlign: 'center', marginTop: 40, color: '#888' },
  row: { padding: 12, borderRadius: 8, backgroundColor: '#f2f4f7' },
  title: { fontSize: 15, fontWeight: '500' },
  sub: { fontSize: 13, color: '#555' },
});

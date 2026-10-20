import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, FlatList, Pressable, StyleSheet, Text } from 'react-native';

import { cancelJob } from '@/lib/api';
import { deleteScan, listScans, type Scan } from '@/lib/scans';

export default function History() {
  const [scans, setScans] = useState<Scan[]>([]);
  useFocusEffect(useCallback(() => setScans(listScans()), []));

  function confirmDelete(scan: Scan) {
    Alert.alert('Delete scan?', 'Photos and model on this phone are removed. This cannot be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          if (scan.jobId) cancelJob(scan.jobId).catch(() => {}); // server copy expires anyway
          deleteScan(scan.id);
          setScans(listScans());
        },
      },
    ]);
  }

  return (
    <FlatList
      data={scans}
      keyExtractor={(s) => s.id}
      contentContainerStyle={styles.list}
      ListEmptyComponent={<Text style={styles.empty}>No scans yet. Tap New scan on the home screen.</Text>}
      ListFooterComponent={scans.length ? <Text style={styles.empty}>Long press a scan to delete it.</Text> : null}
      renderItem={({ item }) => (
        <Pressable
          style={styles.row}
          accessibilityRole="button"
          accessibilityHint="Long press to delete"
          onPress={() =>
            router.push({ pathname: item.status === 'done' ? '/viewer/[id]' : '/processing/[id]', params: { id: item.id } })
          }
          onLongPress={() => confirmDelete(item)}
        >
          <Text style={styles.title}>{new Date(item.createdAt).toLocaleString()}</Text>
          <Text style={styles.sub}>
            {item.photoCount} photos · {item.status}
          </Text>
        </Pressable>
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

import { router } from 'expo-router';
import { useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';

import { createScan, updateScan } from '@/lib/scans';

// Placeholder until milestone 2 (camera, coverage, auto-capture).
export default function Capture() {
  const [scan] = useState(createScan);

  return (
    <View style={styles.container}>
      <Text>Camera comes in milestone 2. Scan {scan.id}</Text>
      <Button
        title="Finish"
        onPress={() => {
          updateScan(scan.id, { status: 'uploading' });
          router.replace({ pathname: '/processing/[id]', params: { id: scan.id } });
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16 },
});

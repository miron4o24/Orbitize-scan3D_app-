import { File, Paths } from 'expo-file-system';
import { useLocalSearchParams } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Switch, Text, TextInput, View } from 'react-native';
import type { Object3D } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import { ModelView } from '@/components/ModelView';
import { downloadResult, errorText } from '@/lib/api';
import { exportForPrint } from '@/lib/exportModel';
import { getScan, modelFile, updateScan } from '@/lib/scans';

async function loadModel(id: string, jobId: string | undefined): Promise<Object3D> {
  const file = modelFile(id);
  if (!file.exists) {
    if (!jobId) throw new Error('The model is not on this phone and the scan has no server job.');
    await downloadResult(jobId, 'glb', file);
  }
  const bytes = await file.bytes();
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  try {
    return (await new GLTFLoader().parseAsync(buffer, '')).scene;
  } catch (e) {
    if (jobId) file.delete(); // likely a truncated download; Retry fetches it again
    throw e;
  }
}

const MIME = { stl: 'model/stl', obj: 'model/obj', glb: 'model/gltf-binary' } as const;

export default function Viewer() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [scan, setScan] = useState(() => getScan(id));
  const [model, setModel] = useState<Object3D | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [mmText, setMmText] = useState(String(scan?.longestMm ?? 100));
  const [busy, setBusy] = useState(false);
  const jobId = scan?.jobId;

  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    loadModel(id, jobId).then(
      (m) => !cancelled && setModel(m),
      (e) => !cancelled && setError(`Could not load the model: ${errorText(e)}`),
    );
    return () => {
      cancelled = true;
    };
  }, [id, jobId, attempt]);

  const mm = Number(mmText.replace(',', '.'));
  const mmValid = Number.isFinite(mm) && mm >= 1 && mm <= 2000;
  const report = scan?.report;
  // The server normalises the model to a 100 mm longest side; dimensions scale linearly.
  const dims = report && mmValid ? report.dimensions_mm.map((d) => (d * mm) / 100) : null;

  async function share(format: keyof typeof MIME) {
    if (!model || !mmValid) return;
    setBusy(true);
    try {
      setScan(updateScan(id, { longestMm: mm }));
      let file = modelFile(id);
      if (format !== 'glb') {
        file = new File(Paths.cache, `scan-${id}-${Math.round(mm)}mm.${format}`);
        file.write(exportForPrint(model, format, mm));
      }
      await Sharing.shareAsync(file.uri, { mimeType: MIME[format], dialogTitle: 'Export 3D model' });
    } catch (e) {
      setError(`Export failed: ${errorText(e)}`);
    } finally {
      setBusy(false);
    }
  }

  if (!scan) return <Text style={styles.pad}>Scan not found.</Text>;

  return (
    <View style={styles.fill}>
      <View style={styles.preview}>
        {model ? (
          <ModelView model={model} wireframe={wireframe} />
        ) : error ? (
          <View style={styles.center}>
            <Text style={styles.error}>{error}</Text>
            <Button
              title="Retry"
              onPress={() => {
                setError(null);
                setAttempt((a) => a + 1);
              }}
            />
          </View>
        ) : (
          <ActivityIndicator style={styles.center} size="large" />
        )}
      </View>
      <ScrollView contentContainerStyle={styles.pad}>
        <View style={styles.row}>
          <Text style={styles.flex}>Wireframe</Text>
          <Switch value={wireframe} onValueChange={setWireframe} accessibilityLabel="Wireframe" />
        </View>
        <View style={styles.row}>
          <Text style={styles.flex}>Real length of the longest side (mm)</Text>
          <TextInput
            style={[styles.input, !mmValid && styles.invalid]}
            value={mmText}
            onChangeText={setMmText}
            keyboardType="decimal-pad"
            accessibilityLabel="Longest side in millimetres"
          />
        </View>
        {!mmValid && <Text style={styles.error}>Enter a size between 1 and 2000 mm.</Text>}
        {report && (
          <View style={styles.report}>
            <Text style={styles.bold}>Print report</Text>
            {dims && <Text>Size: {dims.map((d) => d.toFixed(1)).join(' × ')} mm</Text>}
            <Text>Triangles: {report.triangles.toLocaleString()}</Text>
            <Text>Watertight: {report.watertight ? 'yes' : 'no. The slicer may need to repair it'}</Text>
            {report.photos_used != null && (
              <Text>
                Photos used: {report.photos_used} / {report.photos_total}
              </Text>
            )}
            {report.seconds != null && <Text>Processing time: {Math.round(report.seconds / 60)} min</Text>}
          </View>
        )}
        {error && model && <Text style={styles.error}>{error}</Text>}
        <View style={styles.row}>
          {(['stl', 'obj', 'glb'] as const).map((f) => (
            <View key={f} style={styles.flex}>
              <Button title={f.toUpperCase()} onPress={() => share(f)} disabled={!model || !mmValid || busy} />
            </View>
          ))}
        </View>
        <Text style={styles.note}>STL and OBJ are scaled to the size above, in millimetres, Z up.</Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#fff' },
  preview: { height: '55%' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 20 },
  pad: { padding: 16, gap: 12 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  flex: { flex: 1 },
  input: { borderWidth: 1, borderColor: '#bbb', borderRadius: 6, paddingHorizontal: 10, paddingVertical: 6, minWidth: 90 },
  invalid: { borderColor: '#b00020' },
  report: { backgroundColor: '#f2f4f7', padding: 12, borderRadius: 8, gap: 4 },
  bold: { fontWeight: '600' },
  error: { color: '#b00020' },
  note: { color: '#555', fontSize: 13 },
});

import { CameraView, useCameraPermissions } from 'expo-camera';
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { router } from 'expo-router';
import { DeviceMotion } from 'expo-sensors';
import jpeg from 'jpeg-js';
import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Pressable, StyleSheet, Switch, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { CoverageRing } from '@/components/CoverageRing';
import { cellKey, cellOf, isNewAngle, viewDir, type Vec3 } from '@/lib/coverage';
import { judge, measure, type FrameMetrics } from '@/lib/quality';
import { createScan, deleteScan, photosDir, updateScan } from '@/lib/scans';

const MIN_PHOTOS = 20;
const TARGET_PHOTOS = 60;
const MAX_PHOTOS = 120;
const STEADY_DEG_PER_S = 20;
const STEADY_MS = 300;

// Largest 4:3 size with long side <= 2600 px: full sensor view, ~5 MP, keeps uploads small.
function pickPictureSize(sizes: string[]) {
  return sizes
    .map((s) => ({ s, w: Number(s.split('x')[0]), h: Number(s.split('x')[1]) }))
    .filter(({ w, h }) => Math.abs(Math.max(w, h) / Math.min(w, h) - 4 / 3) < 0.02 && Math.max(w, h) <= 2600)
    .sort((a, b) => b.w * b.h - a.w * a.h)[0]?.s;
}

async function checkQuality(uri: string): Promise<FrameMetrics> {
  const small = await (await ImageManipulator.manipulate(uri).resize({ width: 512 }).renderAsync()).saveAsync({
    format: SaveFormat.JPEG,
    compress: 0.9,
  });
  const file = new File(small.uri);
  const { data, width, height } = jpeg.decode(await file.bytes(), { useTArray: true });
  file.delete();
  return measure(data, width, height);
}

export default function Capture() {
  const [permission, requestPermission] = useCameraPermissions();
  const [scan] = useState(createScan);
  const camera = useRef<CameraView>(null);
  const [pictureSize, setPictureSize] = useState<string>();
  const [auto, setAuto] = useState(true);
  const [shots, setShots] = useState<Vec3[]>([]);
  const [covered, setCovered] = useState(() => new Set<string>());
  const [current, setCurrent] = useState<{ band: number; sector: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [lastMetrics, setLastMetrics] = useState<FrameMetrics | null>(null);

  // Mutable mirrors of state for the 20 Hz sensor callback.
  const busy = useRef(false);
  const shotsRef = useRef<Vec3[]>([]);
  const autoRef = useRef(auto);
  const finished = useRef(false);

  async function capture(view: Vec3) {
    if (busy.current || !camera.current || shotsRef.current.length >= MAX_PHOTOS) return;
    busy.current = true;
    try {
      const pic = await camera.current.takePictureAsync({ quality: 0.9, exif: true, shutterSound: false });
      const metrics = await checkQuality(pic.uri);
      setLastMetrics(metrics);
      const verdict = judge(metrics);
      setMessage(verdict.warnings[0] ?? null);
      if (verdict.reject) {
        new File(pic.uri).delete();
        return;
      }
      const n = shotsRef.current.length + 1;
      new File(pic.uri).move(new File(photosDir(scan.id), `img_${String(n).padStart(4, '0')}.jpg`));
      shotsRef.current = [...shotsRef.current, view];
      setShots(shotsRef.current);
      setCovered((prev) => new Set(prev).add(cellKey(cellOf(view))));
      updateScan(scan.id, { photoCount: n });
    } catch (e) {
      setMessage(`Capture failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      busy.current = false;
    }
  }

  const lastView = useRef<Vec3 | null>(null);
  useEffect(() => {
    DeviceMotion.requestPermissionsAsync();
    DeviceMotion.setUpdateInterval(50);
    let steadySince = 0;
    let lastUi = 0;
    const sub = DeviceMotion.addListener((e) => {
      if (!e.rotation) return;
      const view = viewDir(e.rotation);
      lastView.current = view;
      const r = e.rotationRate;
      const speed = r ? Math.hypot(r.alpha, r.beta, r.gamma) : 0;
      const now = Date.now();
      if (speed > STEADY_DEG_PER_S) steadySince = now;
      if (now - lastUi > 150) {
        lastUi = now;
        setCurrent(cellOf(view));
      }
      if (autoRef.current && now - steadySince > STEADY_MS && isNewAngle(view, shotsRef.current)) capture(view);
    });
    return () => sub.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- subscribe once; callback reads refs
  }, []);

  // Leaving without any photo should not leave an empty scan in history.
  useEffect(
    () => () => {
      if (!finished.current && shotsRef.current.length === 0) deleteScan(scan.id);
    },
    [scan.id],
  );

  function finish() {
    const go = () => {
      finished.current = true;
      updateScan(scan.id, { status: 'uploading', photoCount: shotsRef.current.length });
      router.replace({ pathname: '/processing/[id]', params: { id: scan.id } });
    };
    if (shots.length >= MIN_PHOTOS) return go();
    Alert.alert(
      'Few photos',
      `Only ${shots.length} photos. Reconstruction usually needs at least ${MIN_PHOTOS}, ideally ${TARGET_PHOTOS}+.`,
      [{ text: 'Keep scanning' }, { text: 'Process anyway', onPress: go }],
    );
  }

  if (!permission) return <View style={styles.fill} />;
  if (!permission.granted)
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>The camera is needed to photograph the object from all sides.</Text>
        {permission.canAskAgain ? (
          <Button title="Allow camera" onPress={requestPermission} />
        ) : (
          <Text style={styles.permText}>Camera access was denied. Enable it in system settings.</Text>
        )}
        <Button title="Back" onPress={() => router.back()} />
      </SafeAreaView>
    );

  return (
    <View style={styles.fill}>
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        animateShutter={false}
        pictureSize={pictureSize}
        onCameraReady={async () => {
          const sizes = await camera.current?.getAvailablePictureSizesAsync();
          setPictureSize(sizes && pickPictureSize(sizes));
        }}
      />
      <SafeAreaView style={[StyleSheet.absoluteFill, styles.overlay]} pointerEvents="box-none">
        <View style={styles.top}>
          <Text style={styles.count}>
            {shots.length} / {TARGET_PHOTOS} photos
          </Text>
          <Text style={styles.hint}>Walk slowly around the object at 2–3 heights. Keep it centred.</Text>
          {message && <Text style={styles.warn}>{message}</Text>}
          {__DEV__ && lastMetrics && (
            <Text style={styles.debug}>
              sharp {lastMetrics.sharpness.toFixed(0)} · bright {lastMetrics.brightness.toFixed(0)} · clip{' '}
              {(lastMetrics.clipped * 100).toFixed(1)}%
            </Text>
          )}
        </View>
        <View style={styles.bottom}>
          <CoverageRing covered={covered} current={current} />
          <View style={styles.controls}>
            <View style={styles.autoRow}>
              <Text style={styles.label}>Auto</Text>
              <Switch
                value={auto}
                onValueChange={(v) => {
                  autoRef.current = v;
                  setAuto(v);
                }}
                accessibilityLabel="Auto capture"
              />
            </View>
            <Pressable
              style={styles.shutter}
              accessibilityRole="button"
              accessibilityLabel="Take photo"
              onPress={() => lastView.current && capture(lastView.current)}
            />
            <Button title="Done" onPress={finish} />
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 },
  permText: { textAlign: 'center', fontSize: 16 },
  overlay: { justifyContent: 'space-between' },
  top: { padding: 16, gap: 4, backgroundColor: 'rgba(0,0,0,0.35)' },
  count: { color: '#fff', fontSize: 20, fontWeight: '600' },
  hint: { color: '#eee', fontSize: 13 },
  warn: { color: '#ffeb3b', fontSize: 15, fontWeight: '600' },
  debug: { color: '#aaa', fontSize: 11 },
  bottom: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16 },
  controls: { alignItems: 'center', gap: 12 },
  autoRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  label: { color: '#fff' },
  shutter: { width: 72, height: 72, borderRadius: 36, backgroundColor: '#fff', borderWidth: 4, borderColor: '#bbb' },
});

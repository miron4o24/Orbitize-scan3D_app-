import Ionicons from '@expo/vector-icons/Ionicons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { File } from 'expo-file-system';
import { useKeepAwake } from 'expo-keep-awake';
import { router, useLocalSearchParams } from 'expo-router';
import { DeviceMotion, LightSensor } from 'expo-sensors';
import { useEffect, useRef, useState } from 'react';
import { Alert, Platform, Pressable, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { OrbitGuide } from '@/components/OrbitGuide';
import { Button, Header, Small } from '@/components/ui';
import { errorText } from '@/lib/api';
import { viewDir } from '@/lib/coverage';
import { motionSamples, type RawMotion } from '@/lib/motion';
import { OrbitTracker, type OrbitState } from '@/lib/orbit';
import { aiAvailable, PRO_INFO, SCAN_INFO } from '@/lib/prefs';
import { createScan, deleteScan, motionFile, updateScan, videoFile } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { checkVideo } from '@/lib/videoCheck';
import { font, makeStyles, palettes, radius, space } from '@/theme';

// Two slow circles take 1–2 minutes; the cap only guards against a forgotten recording (~320 MB).
const MAX_SECONDS = 180;
const MIN_SECONDS = 10; // an AI model's video: shorter cannot go round the object
const MIN_ROUND = 0.75; // an AI model's video: of the first circle, below which sides will be missing
const DARK_LUX = 10; // the phone's light sensor: a room too dim to film in
// 4K: the server wraps the frames onto the model as its texture, and small print reads only with the pixels for it
// (at 1080p a drill's battery label was just readable). 14 Mbit/s: ~105 MB per minute to upload, twice 1080p at
// 6 Mbit/s; a slow walk-around changes little between frames, so the encoder keeps detail at that rate. Phones
// without 4K record their highest quality instead.
const BITRATE = 14_000_000;

const TIPS = [
  'Само предметът, върху шарена подложка.',
  'Ярка светлина, без светкавица.',
  'Приближете се: предметът да запълва кръга.',
  'Два бавни кръга: първо ниско, после отгоре.',
];

const PHASES = [
  ['Кръг 1 от 2: отстрани', 'Приклекнете до височината на предмета.'],
  ['Кръг 2 от 2: отгоре', 'Вдигнете телефона и гледайте надолу.'],
  ['Готово!', 'Запазване…'],
];

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

export default function Capture() {
  useKeepAwake(); // the screen must not lock mid-circle
  const [permission, requestPermission] = useCameraPermissions();
  // The AI model films with nothing but a point in the middle; the standard scan with the guide around the object.
  // extend: this video adds a missing side to that model: the camera as plain, whichever way the model was made.
  const params = useLocalSearchParams<{ ai?: string; extend?: string; title?: string }>();
  const [ai] = useState(() => params.ai === '1' && aiAvailable());
  const [scan] = useState(() => {
    const made = createScan(currentUserId() ?? undefined, ai ? 'single' : 'scan');
    return params.extend ? updateScan(made.id, { extendFrom: params.extend, title: params.title }) ?? made : made;
  });
  const extending = !!params.extend;
  const plain = ai || extending;
  const camera = useRef<CameraView>(null);
  const { width, height } = useWindowDimensions();
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [orbit, setOrbit] = useState<OrbitState | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [showTips, setShowTips] = useState(true);
  const [dim, setDim] = useState(false); // the room is too dim, by the light sensor (where the phone has one)
  const [checking, setChecking] = useState(false); // looking over the video just filmed

  // Mutable mirrors of state for the 20 Hz sensor callback.
  const recordingRef = useRef(false);
  const tracker = useRef(new OrbitTracker());
  const stopReason = useRef<'user' | 'done' | null>(null);
  const startedAt = useRef(0);
  const motion = useRef<RawMotion[]>([]); // the phone's orientation while recording, sent with the video
  const finished = useRef(false);
  const unmounted = useRef(false);

  function stop(reason: 'user' | 'done') {
    if (!recordingRef.current || stopReason.current) return;
    stopReason.current = reason;
    camera.current?.stopRecording();
  }

  useEffect(() => {
    DeviceMotion.requestPermissionsAsync();
    DeviceMotion.setUpdateInterval(50);
    const sub = DeviceMotion.addListener((e) => {
      if (!e.rotation || !recordingRef.current) return;
      const { alpha, beta, gamma, timestamp } = e.rotation;
      motion.current.push([Date.now(), timestamp, alpha, beta, gamma]);
      const r = e.rotationRate;
      const turnRate = r ? Math.hypot(r.alpha, r.beta, r.gamma) : 0;
      const t = tracker.current;
      if (t.update(viewDir(e.rotation), turnRate, Date.now() / 1000)) setOrbit(t.state);
      if (t.done && !plain) stop('done');
    });
    return () => sub.remove();
  }, [plain]);

  useEffect(() => {
    let sub: { remove: () => void } | null = null;
    let cancelled = false;
    LightSensor.isAvailableAsync()
      .then((has) => {
        if (!has || cancelled) return;
        LightSensor.setUpdateInterval(1000);
        sub = LightSensor.addListener(({ illuminance }) => setDim(illuminance < DARK_LUX));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      sub?.remove();
    };
  }, []);

  useEffect(() => {
    if (!recording) return;
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - startedAt.current) / 1000)), 500);
    return () => clearInterval(timer);
  }, [recording]);

  // Leaving without a kept video should not leave an empty scan in history.
  useEffect(
    () => () => {
      unmounted.current = true;
      if (recordingRef.current) camera.current?.stopRecording();
      if (!finished.current) deleteScan(scan.id);
    },
    [scan.id],
  );

  function keep(uri: string, secs: number) {
    const dest = videoFile(scan.id);
    if (dest.exists) dest.delete();
    new File(uri).moveSync(dest);
    motionFile(scan.id).writeSync(JSON.stringify({ samples: motionSamples(motion.current, startedAt.current) }));
    finished.current = true;
    updateScan(scan.id, { status: 'uploading', videoSeconds: secs });
    router.replace({ pathname: '/processing/[id]', params: { id: scan.id, auto: '1' } }); // upload at once
  }

  /** A look over the video before it goes up: what is wrong with it can still be filmed again. */
  async function review(uri: string, secs: number) {
    setChecking(true);
    const found = await checkVideo(uri, secs);
    if (unmounted.current) return;
    setChecking(false);
    if (ai && !extending && tracker.current.firstLoopFraction < MIN_ROUND) found.push('Не обиколихте целия предмет.');
    if (!found.length) return keep(uri, secs);
    Alert.alert('Да запишем ли отново?', `${found.join('\n')}\n\nМоделът може да излезе по-лош.`, [
      { text: 'Запиши отново', onPress: () => new File(uri).delete() },
      { text: 'Използвай все пак', onPress: () => keep(uri, secs) },
    ]);
  }

  async function record() {
    if (!camera.current || recordingRef.current) return;
    tracker.current = new OrbitTracker();
    stopReason.current = null;
    setOrbit(tracker.current.state);
    setMessage(null);
    setSeconds(0);
    motion.current = [];
    startedAt.current = Date.now();
    recordingRef.current = true;
    setRecording(true);
    try {
      // Resolves when stopRecording() is called or MAX_SECONDS is reached.
      const video = await camera.current.recordAsync({
        maxDuration: MAX_SECONDS,
        ...(Platform.OS === 'ios' && { codec: 'avc1' as const }), // iOS applies videoBitrate only with a codec
      });
      const secs = Math.round((Date.now() - startedAt.current) / 1000);
      if (unmounted.current) {
        if (video) new File(video.uri).delete();
        return;
      }
      if (!video) throw new Error('Видеото не се запази.');
      if (plain && secs < MIN_SECONDS) {
        new File(video.uri).delete();
        setMessage('Твърде кратко видео. Обиколете целия предмет.');
        return;
      }
      // Stopped by the time limit before even one full circle: let the user decide.
      if (!plain && !stopReason.current && tracker.current.firstLoopFraction < 1)
        Alert.alert('Времето изтече', 'Първият кръг не е завършен, затова моделът ще има дупки.', [
          { text: 'Запиши отново', onPress: () => new File(video.uri).delete() },
          { text: 'Използвай все пак', onPress: () => keep(video.uri, secs) },
        ]);
      else await review(video.uri, secs);
    } catch (e) {
      if (!unmounted.current) setMessage(`Записът не успя. ${errorText(e)}`);
    } finally {
      recordingRef.current = false;
      if (!unmounted.current) setRecording(false);
    }
  }

  function onShutter() {
    if (!recording) return record();
    if (plain) return stop('user'); // no circles counted: the user knows when it has gone round
    const part = tracker.current.firstLoopFraction;
    if (part >= 1) return stop('user');
    // Recording continues while the alert is open.
    Alert.alert(
      'Още не сте обиколили докрай',
      `Заснети са около ${Math.round(part * 100)}% от първия кръг. Незаснетите страни ще липсват.`,
      [
        { text: 'Продължи', style: 'cancel' },
        { text: 'Спри все пак', style: 'destructive', onPress: () => stop('user') },
      ],
    );
  }

  if (!permission) return <View style={styles.fill} />;
  if (!permission.granted) return <CameraPermission canAsk={permission.canAskAgain} onAllow={requestPermission} />;

  const guide = Math.min(width - 32, height * 0.5);
  const phase = PHASES[Math.min(orbit?.loop ?? 0, PHASES.length - 1)];
  const pace = orbit?.pace;
  const way = ai ? PRO_INFO.single : SCAN_INFO;
  const noPanel = plain ? recording && !message : !recording && !showTips && !message;
  return (
    <View style={styles.fill}>
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        mode="video"
        mute
        videoQuality="2160p"
        videoBitrate={BITRATE}
        // Stabilisation crops and warps each frame differently, which breaks the fixed-lens model COLMAP fits.
        videoStabilizationMode="off"
        onCameraReady={() => setReady(true)}
      />
      <View style={[StyleSheet.absoluteFill, styles.middle]} pointerEvents="none">
        {plain ? <View style={styles.point} /> : <OrbitGuide size={guide} orbit={orbit} />}
      </View>
      <SafeAreaView style={[StyleSheet.absoluteFill, styles.overlay]} pointerEvents="box-none">
        <View style={styles.top}>
          {!recording && (
            <View style={styles.topRow}>
              <Pressable
                style={styles.close}
                onPress={() => router.back()}
                accessibilityRole="button"
                accessibilityLabel="Затвори"
              >
                <Ionicons name="close" size={20} color={WHITE} />
              </Pressable>
              {aiAvailable() && (
                <View style={styles.tipsToggle}>
                  <Text style={styles.tipsToggleText}>
                    {way.label} · {way.time}
                  </Text>
                </View>
              )}
              {!plain && (
                <Pressable
                  style={styles.tipsToggle}
                  onPress={() => setShowTips((t) => !t)}
                  accessibilityRole="button"
                >
                  <Text style={styles.tipsToggleText}>{showTips ? 'Скрий' : 'Съвети'}</Text>
                </Pressable>
              )}
            </View>
          )}
          <View style={[styles.panel, noPanel && styles.hidden]}>
          {plain ? (
            !recording && (
              <Text style={styles.hint}>
                {extending
                  ? 'Заснемете само липсващата страна. Предметът да не е местен.'
                  : 'Дръжте предмета на точката и го обиколете бавно, отблизо.'}
              </Text>
            )
          ) : recording ? (
            <>
              <Text style={styles.title}>{phase[0]}</Text>
              <Text style={styles.hint}>{phase[1]}</Text>
              {orbit?.aim === 'lower' && (
                <Text style={styles.warn}>
                  {orbit.loop === 0
                    ? 'По-ниско'
                    : 'Свалете малко телефона'}
                </Text>
              )}
              {orbit?.aim === 'higher' && (
                <Text style={styles.warn}>По-високо</Text>
              )}
            </>
          ) : (
            showTips && (
              <>
                {TIPS.map((t, i) => (
                  <View key={t} style={styles.tipRow}>
                    <Text style={styles.tipNum}>{String(i + 1).padStart(2, '0')}</Text>
                    <Text style={[styles.hint, styles.flex]}>{t}</Text>
                  </View>
                ))}
              </>
            )
          )}
          {message && <Text style={styles.warn}>{message}</Text>}
          </View>
          {dim && !checking && (
            <View style={styles.darkPill}>
              <Ionicons name="moon-outline" size={15} color={dark.warning} />
              <Text style={styles.darkText}>Твърде тъмно: добавете светлина</Text>
            </View>
          )}
          {checking && (
            <View style={styles.darkPill}>
              <Text style={styles.checkText}>Проверка на видеото…</Text>
            </View>
          )}
        </View>
        <View style={styles.bottom}>
          <View style={styles.side}>
            {recording && <Text style={styles.time}>● {fmt(seconds)}</Text>}
          </View>
          <Pressable
            style={[styles.shutter, recording && styles.stop, !ready && styles.disabled]}
            accessibilityRole="button"
            accessibilityLabel={recording ? 'Спри записа' : 'Започни запис'}
            disabled={!ready || checking}
            onPress={onShutter}
          >
            <View style={recording ? styles.stopIcon : styles.recIcon} />
          </Pressable>
          <View style={styles.side}>
            {recording && pace && (plain ? pace === 'fast' : pace !== 'still' || seconds >= 3) && (
              <Text style={[styles.pace, styles[pace]]}>
                {pace === 'fast' ? 'Твърде бързо!\nЗабавете' : pace === 'good' ? 'Добро темпо' : 'Тръгнете'}
              </Text>
            )}
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

function CameraPermission({ canAsk, onAllow }: { canAsk: boolean; onAllow: () => void }) {
  const s = usePermissionStyles();
  return (
    <SafeAreaView style={s.fill}>
      <View style={s.content}>
        <Header title="Достъп до камерата" />
        {canAsk ? (
          <Button title="Разреши камерата" icon="camera-outline" onPress={onAllow} />
        ) : (
          <Small>Достъпът до камерата е отказан. Разрешете го от настройките на телефона.</Small>
        )}
        <Button title="Назад" variant="ghost" onPress={() => router.back()} />
      </View>
    </SafeAreaView>
  );
}

const usePermissionStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  content: { flex: 1, justifyContent: 'center', padding: space.xl, gap: space.lg },
}));

// Over the camera picture: always the dark theme's colours.
const dark = palettes.dark;
const WHITE = '#ffffff';
const GLASS = 'rgba(9,9,11,0.72)';

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  middle: { alignItems: 'center', justifyContent: 'center' },
  point: { width: 14, height: 14, borderRadius: 7, backgroundColor: WHITE, borderWidth: 2, borderColor: 'rgba(0,0,0,0.55)' },
  overlay: { justifyContent: 'space-between' },
  flex: { flex: 1 },
  top: { padding: space.md, gap: space.sm },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  close: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tipsToggle: {
    height: 36,
    justifyContent: 'center',
    paddingHorizontal: 14,
    borderRadius: 18,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  tipsToggleText: { color: WHITE, fontSize: 13.5, fontFamily: font.medium },
  darkPill: {
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 18,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  darkText: { color: dark.warning, fontSize: 13, fontFamily: font.medium },
  checkText: { color: WHITE, fontSize: 13, fontFamily: font.medium },
  panel: { padding: space.lg, gap: space.sm, borderRadius: radius.md, backgroundColor: GLASS, borderWidth: 1, borderColor: dark.line },
  hidden: { display: 'none' },
  tipRow: { flexDirection: 'row', gap: space.md, alignItems: 'flex-start' },
  tipNum: { fontFamily: font.mono, fontSize: 12, lineHeight: 18, color: dark.muted, width: 20 },
  title: { color: dark.text, fontSize: 16, lineHeight: 22, fontFamily: font.semibold, letterSpacing: -0.2 },
  hint: { color: dark.text2, fontSize: 13, lineHeight: 18, fontFamily: font.regular },
  warn: { color: dark.warning, fontSize: 14.5, lineHeight: 20, fontFamily: font.medium },
  bottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    margin: space.md,
    paddingVertical: space.md,
    paddingHorizontal: space.sm,
    borderRadius: 24,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  side: { width: 96, alignItems: 'center' },
  time: { color: WHITE, fontSize: 17, fontFamily: font.monoMedium },
  pace: {
    fontSize: 13,
    lineHeight: 17,
    fontFamily: font.medium,
    textAlign: 'center',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  still: { color: dark.text2, backgroundColor: dark.surface3 },
  good: { color: dark.success, backgroundColor: dark.successSoft },
  fast: { color: dark.danger, backgroundColor: dark.dangerSoft },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stop: { borderColor: '#e53935' },
  disabled: { opacity: 0.4 },
  recIcon: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#e53935' },
  stopIcon: { width: 28, height: 28, borderRadius: 4, backgroundColor: '#e53935' },
});

import Ionicons from '@expo/vector-icons/Ionicons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { File } from 'expo-file-system';
import { useKeepAwake } from 'expo-keep-awake';
import { router, useLocalSearchParams } from 'expo-router';
import { DeviceMotion, LightSensor } from 'expo-sensors';
import { useEffect, useRef, useState } from 'react';
import { Alert, Platform, Pressable, StyleSheet, Text, View, type LayoutChangeEvent } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { HUD, ScanStage } from '@/components/ScanHud';
import { Button, Header, Small } from '@/components/ui';
import { errorText } from '@/lib/api';
import { elevation, viewDir } from '@/lib/coverage';
import { motionSamples, type RawMotion } from '@/lib/motion';
import { GOOD_DEG_PER_S, OrbitTracker, SECTORS, type OrbitState } from '@/lib/orbit';
import { PRO_INFO } from '@/lib/prefs';
import { coverage, DARK_LUX, LEVEL_TEXT, lightPercent, sharpLevel, sharpness, TARGET_VIEWS, ViewCounter } from '@/lib/scanHud';
import { createScan, deleteScan, motionFile, updateScan, videoFile } from '@/lib/scans';
import { currentUserId } from '@/lib/session';
import { checkVideo } from '@/lib/videoCheck';
import { font, makeStyles, palettes, space } from '@/theme';

// Заснемане за AI модела: предметът в рамката, два кръга около него (ниско, после отвисоко). Всичко на екрана е
// измерено от сензорите на телефона (lib/scanHud.ts) и казва на потребителя какво още трябва на модела: кои страни
// липсват, дали кадрите ще са размазани, дали е достатъчно светло и на каква височина е телефонът. Записът спира сам,
// когато и двата кръга са заснети.

// Обиколката отнема около минута; границата само пази от забравен запис (около 320 MB).
const MAX_SECONDS = 180;
const MIN_SECONDS = 10; // по-кратко видео не може да обиколи предмета
const MIN_ROUND = 0.75; // дял от първия кръг, под който ще липсват страни
// 4K: сървърът нанася кадрите върху модела като текстура, а дребният текст се чете само ако има достатъчно пиксели
// (при 1080p етикетът на батерия на дрелка едва се четеше). 14 Mbit/s: около 105 MB в минута за качване, двойно
// спрямо 1080p при 6 Mbit/s; при бавна обиколка кадрите се променят малко, затова кодекът запазва детайла при тази
// скорост. Телефони без 4K записват с най-високото си качество.
const BITRATE = 14_000_000;

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
const LIVE_MS = 160; // колко често се обновяват числата на екрана

// Това, което сензорите казват в момента, за числата на екрана.
type Live = { angle: number; speed: number; tilt: number; sharp: number; views: number };
const NO_LIVE: Live = { angle: 0, speed: 0, tilt: 0, sharp: 100, views: 0 };

export default function Capture() {
  useKeepAwake(); // екранът не бива да се заключва по средата на кръга
  const [permission, requestPermission] = useCameraPermissions();
  // extend: това видео добавя липсваща страна към модела.
  const params = useLocalSearchParams<{ extend?: string; title?: string }>();
  const [scan] = useState(() => {
    const made = createScan(currentUserId() ?? undefined, 'single');
    return params.extend ? updateScan(made.id, { extendFrom: params.extend, title: params.title }) ?? made : made;
  });
  const extending = !!params.extend;
  const camera = useRef<CameraView>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [orbit, setOrbit] = useState<OrbitState | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [lux, setLux] = useState<number | null>(null); // сензорът за светлина (ако телефонът има такъв)
  const [live, setLive] = useState<Live>(NO_LIVE);
  const [stage, setStage] = useState<{ w: number; h: number } | null>(null);
  const [checking, setChecking] = useState(false); // проверка на току-що заснетото видео

  // Променливи копия на състоянието за извикванията от сензора с 20 Hz.
  const recordingRef = useRef(false);
  const tracker = useRef(new OrbitTracker());
  const stopReason = useRef<'user' | 'done' | null>(null);
  const startedAt = useRef(0);
  const motion = useRef<RawMotion[]>([]); // положението на телефона по време на записа, изпраща се с видеото
  const views = useRef(new ViewCounter());
  const now = useRef<Live>(NO_LIVE);
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
      if (!e.rotation) return;
      const r = e.rotationRate;
      const turnRate = r ? Math.hypot(r.alpha, r.beta, r.gamma) : 0;
      const view = viewDir(e.rotation);
      const sharp = sharpness(turnRate);
      const t = tracker.current;
      // Наклонът и рязкостта се показват и преди записа: потребителят вижда височината и колко спокойно държи.
      now.current = { ...now.current, tilt: elevation(view), sharp: Math.round(0.7 * now.current.sharp + 0.3 * sharp) };
      if (!recordingRef.current) return;
      const { alpha, beta, gamma, timestamp } = e.rotation;
      motion.current.push([Date.now(), timestamp, alpha, beta, gamma]);
      if (t.update(view, turnRate, Date.now() / 1000)) setOrbit(t.state);
      views.current.add(view, sharp);
      now.current = { ...now.current, angle: t.angle, speed: t.orbitSpeed, views: views.current.count };
      if (t.done && !extending) stop('done'); // и двата кръга са заснети
    });
    return () => sub.remove();
  }, [extending]);

  useEffect(() => {
    const timer = setInterval(() => setLive(now.current), LIVE_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let sub: { remove: () => void } | null = null;
    let cancelled = false;
    LightSensor.isAvailableAsync()
      .then((has) => {
        if (!has || cancelled) return;
        LightSensor.setUpdateInterval(1000);
        sub = LightSensor.addListener(({ illuminance }) => setLux(illuminance));
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

  // Излизане без запазено видео не бива да оставя празно сканиране в историята.
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
    router.replace({ pathname: '/processing/[id]', params: { id: scan.id, auto: '1' } }); // качване веднага
  }

  /* Преглед на видеото преди качване: което не е наред, още може да се заснеме отново. */
  async function review(uri: string, secs: number) {
    setChecking(true);
    const found = await checkVideo(uri, secs);
    if (unmounted.current) return;
    setChecking(false);
    if (!extending && tracker.current.firstLoopFraction < MIN_ROUND) found.push('Не обиколихте целия предмет.');
    if (!found.length) return keep(uri, secs);
    Alert.alert('Да запишем ли отново?', `${found.join('\n')}\n\nМоделът може да излезе по-лош.`, [
      { text: 'Запиши отново', onPress: () => new File(uri).delete() },
      { text: 'Използвай все пак', onPress: () => keep(uri, secs) },
    ]);
  }

  async function record() {
    if (!camera.current || recordingRef.current) return;
    tracker.current = new OrbitTracker();
    views.current = new ViewCounter();
    now.current = { ...now.current, angle: 0, speed: 0, views: 0 };
    stopReason.current = null;
    setOrbit(tracker.current.state);
    setMessage(null);
    setSeconds(0);
    motion.current = [];
    startedAt.current = Date.now();
    recordingRef.current = true;
    setRecording(true);
    try {
      // Завършва при извикване на stopRecording() или при достигане на MAX_SECONDS.
      const video = await camera.current.recordAsync({
        maxDuration: MAX_SECONDS,
        ...(Platform.OS === 'ios' && { codec: 'avc1' as const }), // iOS прилага videoBitrate само при зададен кодек
      });
      const secs = Math.round((Date.now() - startedAt.current) / 1000);
      if (unmounted.current) {
        if (video) new File(video.uri).delete();
        return;
      }
      if (!video) throw new Error('Видеото не се запази.');
      if (secs < MIN_SECONDS) {
        new File(video.uri).delete();
        setMessage('Твърде кратко видео. Обиколете целия предмет.');
        return;
      }
      await review(video.uri, secs);
    } catch (e) {
      if (!unmounted.current) setMessage(`Записът не успя. ${errorText(e)}`);
    } finally {
      recordingRef.current = false;
      if (!unmounted.current) setRecording(false);
    }
  }

  function onShutter() {
    if (!recording) return record();
    stop('user');
  }

  if (!permission) return <View style={styles.fill} />;
  if (!permission.granted) return <CameraPermission canAsk={permission.canAskAgain} onAllow={requestPermission} />;

  const way = PRO_INFO.single;
  const dim = lux != null && lux < DARK_LUX;
  const level = sharpLevel(live.sharp);
  const share = coverage(orbit);
  const fast = recording && orbit?.pace === 'fast';
  const slowOk = live.speed >= GOOD_DEG_PER_S[0] && live.speed <= GOOD_DEG_PER_S[1];
  // Какво да направи потребителят сега: най-важното първо.
  const guide = checking
    ? 'Проверка на видеото…'
    : message
      ? message
      : dim
        ? 'Твърде тъмно: добавете светлина'
        : !recording
          ? extending
            ? 'Заснемете само липсващата страна. Предметът да не е местен.'
            : 'Предметът в рамката. Обиколете го бавно: първо ниско, после отвисоко.'
          : fast
            ? 'Твърде бързо: забавете'
            : orbit?.aim === 'lower'
              ? orbit.loop === 0
                ? 'Свалете телефона до височината на предмета'
                : 'Не толкова отгоре'
              : orbit?.aim === 'higher'
                ? 'Вдигнете телефона и гледайте надолу'
                : orbit?.loop === 1
                  ? 'Кръг 2 от 2: отвисоко, гледайте надолу'
                  : 'Кръг 1 от 2: ниско, на височината на предмета'; // fmt: skip
  const warn = !!message || dim || fast || (recording && orbit?.aim !== 'ok');
  const segments = Math.round(share * SECTORS);
  return (
    <View style={styles.fill}>
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        mode="video"
        mute
        videoQuality="2160p"
        videoBitrate={BITRATE}
        // Стабилизацията изрязва и изкривява всеки кадър различно, което разваля модела на обектива с постоянни
        // параметри, който напасва COLMAP.
        videoStabilizationMode="off"
        onCameraReady={() => setReady(true)}
      />
      <SafeAreaView style={StyleSheet.absoluteFill} pointerEvents="box-none">
        <View style={styles.top} pointerEvents="box-none">
          <View style={styles.readout}>
            <Text style={styles.mono}>
              <Text style={{ color: recording ? HUD.cyan : HUD.dim }}>● </Text>
              {recording ? `REC ${fmt(seconds)}` : 'ГОТОВ'}
            </Text>
            <Text style={styles.mono}>
              КАДРИ <Text style={styles.value}>{live.views}</Text>
              <Text style={styles.soft}> / {TARGET_VIEWS}</Text>
            </Text>
            <Text style={styles.mono}>
              РЯЗКОСТ <Text style={level === 'low' ? styles.amber : styles.value}>{LEVEL_TEXT[level]}</Text>
            </Text>
            <Text style={styles.mono}>
              СВЕТЛИНА <Text style={dim ? styles.amber : styles.value}>{lux == null ? '—' : `${lightPercent(lux)}%`}</Text>
            </Text>
          </View>
          {!recording && (
            <View style={styles.topRight}>
              <Pressable style={styles.close} onPress={() => router.back()} accessibilityRole="button" accessibilityLabel="Затвори">
                <Ionicons name="close" size={20} color={WHITE} />
              </Pressable>
              <Text style={styles.way}>
                {way.label} · {way.time}
              </Text>
            </View>
          )}
        </View>
        <View style={styles.stage} pointerEvents="none" onLayout={(e: LayoutChangeEvent) => setStage({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
          {stage && <ScanStage w={stage.w} h={stage.h} orbit={orbit} angle={live.angle} recording={recording} />}
        </View>
        <View style={styles.guide} pointerEvents="none">
          <Text style={[styles.guideText, warn && styles.amber]}>{guide}</Text>
        </View>
        <View style={styles.panel}>
          <View style={styles.row}>
            <Text style={styles.label}>ПОКРИТИЕ</Text>
            <Text style={styles.big}>{Math.round(share * 100)}%</Text>
          </View>
          <View style={styles.bar}>
            {Array.from({ length: SECTORS }, (_, i) => (
              <View key={i} style={[styles.cell, i < segments && styles.cellOn]} />
            ))}
          </View>
          <View style={styles.stats}>
            <Stat label="СКОРОСТ" value={recording ? `${Math.round(live.speed)}°/с` : '—'} warn={recording && !slowOk && seconds >= 3} />
            <Stat label="РЯЗКОСТ" value={String(live.sharp)} warn={level === 'low'} />
            <Stat label="НАКЛОН" value={`${Math.round(live.tilt)}°`} warn={recording && orbit?.aim !== 'ok'} />
          </View>
          <View style={styles.controls}>
            <Pressable
              style={[styles.shutter, recording && styles.stop, !ready && styles.disabled]}
              accessibilityRole="button"
              accessibilityLabel={recording ? 'Спри записа' : 'Започни запис'}
              disabled={!ready || checking}
              onPress={onShutter}
            >
              <View style={recording ? styles.stopIcon : styles.recIcon} />
            </Pressable>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.label}>{label}</Text>
      <Text style={[styles.statValue, warn && styles.amber]}>{value}</Text>
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

// Върху изображението от камерата: винаги тъмни цветове.
const dark = palettes.dark;
const WHITE = '#ffffff';
const GLASS = 'rgba(9,9,11,0.6)';

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  top: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', paddingHorizontal: space.lg, paddingTop: space.sm },
  readout: { gap: 2, paddingVertical: 6, paddingHorizontal: 10, borderRadius: 10, backgroundColor: GLASS },
  mono: { color: dark.text2, fontFamily: font.mono, fontSize: 12.5, lineHeight: 18, letterSpacing: 0.6 },
  value: { color: WHITE, fontFamily: font.monoMedium },
  soft: { color: HUD.dim },
  amber: { color: HUD.amber },
  topRight: { alignItems: 'flex-end', gap: space.sm },
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
  way: { color: WHITE, fontSize: 12.5, fontFamily: font.medium, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12, backgroundColor: GLASS, overflow: 'hidden' },
  stage: { flex: 1 },
  guide: { alignItems: 'center', paddingHorizontal: space.lg, paddingBottom: space.sm },
  guideText: {
    color: WHITE,
    fontSize: 14,
    lineHeight: 19,
    fontFamily: font.medium,
    textAlign: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 14,
    backgroundColor: GLASS,
    overflow: 'hidden',
  },
  panel: {
    backgroundColor: HUD.panel,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.12)',
    paddingHorizontal: space.lg,
    paddingTop: space.md,
    gap: space.sm,
  },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'baseline' },
  label: { color: dark.muted, fontFamily: font.mono, fontSize: 11, letterSpacing: 1.2 },
  big: { color: WHITE, fontFamily: font.monoMedium, fontSize: 15 },
  bar: { flexDirection: 'row', gap: 3, height: 9 },
  cell: { flex: 1, borderRadius: 1.5, backgroundColor: 'rgba(255,255,255,0.12)' },
  cellOn: { backgroundColor: HUD.cyan },
  stats: { flexDirection: 'row', marginTop: 4 },
  stat: { flex: 1, gap: 2 },
  statValue: { color: WHITE, fontFamily: font.monoMedium, fontSize: 17 },
  controls: { alignItems: 'center', paddingVertical: space.sm },
  shutter: {
    width: 68,
    height: 68,
    borderRadius: 34,
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stop: { borderColor: '#e53935' },
  disabled: { opacity: 0.4 },
  recIcon: { width: 50, height: 50, borderRadius: 25, backgroundColor: '#e53935' },
  stopIcon: { width: 24, height: 24, borderRadius: 4, backgroundColor: '#e53935' },
});

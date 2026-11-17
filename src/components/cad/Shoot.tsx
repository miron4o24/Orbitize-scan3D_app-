import Ionicons from '@expo/vector-icons/Ionicons';
import { CameraView } from 'expo-camera';
import { File, type Directory } from 'expo-file-system';
import { Image } from 'expo-image';
import { useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { errorText, type CadPhoto } from '@/lib/api';
import { font, palettes, radius, space } from '@/theme';

import { ShotGuide } from './ShotGuide';
import { LEVEL_DEG, useLevel } from './useLevel';

// Снимките на инженерния режим: до шест, всяка отгоре с телефона успореден на масата (нивелирът го пази), а
// предметът обърнат с друга страна нагоре. Картинката (ShotGuide) показва коя страна и как. Сървърът сам разбира
// как точно е лежал предметът (app/cad/photos.py), затова посоките не трябва да са точни.

export const SHOTS: { photo: CadPhoto; title: string; how: string; required?: boolean }[] = [
  {
    photo: 'top',
    title: 'Отгоре',
    how: 'Сложете предмета на масата, както стои обикновено, с дългата страна наляво-надясно.',
    required: true,
  },
  { photo: 'front', title: 'Отпред', how: 'Претърколете го от себе си: страната, която гледаше към вас, е нагоре.' },
  { photo: 'back', title: 'Отзад', how: 'Задната страна нагоре: претърколете го към себе си.' },
  { photo: 'right', title: 'Отдясно', how: 'Изправете го на левия край: десният край е нагоре.' },
  { photo: 'left', title: 'Отляво', how: 'Изправете го на десния край: левият край е нагоре.' },
  { photo: 'bottom', title: 'Отдолу', how: 'Обърнете го с дъното нагоре.' },
];

// Върху живото изображение от камерата: тъмната палитра и постоянните циан и сигнален цвят, за да се четат и при
// двете теми.
const dark = palettes.dark;
const WHITE = dark.text;
const DARK_TEXT = dark.onPrimary;
const GLASS = dark.glass;
const GOOD = dark.success;
const CAM_ACCENT = '#35d4ff';

export function Shoot({
  index,
  step,
  count,
  only,
  folder,
  onTaken,
  onSkip,
  onClose,
}: {
  index: number;
  step: number; // коя поред е тази снимка (от 1) и колко са (плосък детайл: само отгоре и отдолу)
  count: number;
  only?: boolean; // само тази снимка (без "Пропусни" и без брояч)
  folder: Directory;
  onTaken: (file: File) => void;
  onSkip: () => void;
  onClose: () => void;
}) {
  const shot = SHOTS[index];
  const single = only != null; // снимане отново на една снимка (от прегледа на снимките)
  const camera = useRef<CameraView>(null);
  const [ready, setReady] = useState(false);
  const [torch, setTorch] = useState(false);
  const [taking, setTaking] = useState(false);
  const [taken, setTaken] = useState<File | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const level = useLevel();
  const straight = !level.available || level.tilt <= LEVEL_DEG;

  async function take() {
    if (!camera.current || taking) return;
    setTaking(true);
    setMessage(null);
    try {
      const pic = await camera.current.takePictureAsync({ quality: 0.95, shutterSound: false });
      const dest = new File(folder, `${shot.photo}.jpg`);
      if (dest.exists) dest.delete();
      new File(pic.uri).moveSync(dest);
      setTaken(dest);
    } catch (e) {
      setMessage(`Снимката не стана. ${errorText(e)}`);
    } finally {
      setTaking(false);
    }
  }

  if (taken)
    return (
      <View style={styles.fill}>
        <Image source={{ uri: taken.uri }} style={StyleSheet.absoluteFill} contentFit="contain" />
        <SafeAreaView style={[StyleSheet.absoluteFill, styles.overlay]} pointerEvents="box-none">
          <View style={styles.top}>
            <View style={styles.panel}>
              <Text style={styles.panelTitle}>Предметът целият ли се вижда и на фокус?</Text>
              <Text style={styles.hint}>Ръбовете трябва да са ясни: от тях се разчита формата.</Text>
            </View>
          </View>
          <View style={styles.choices}>
            <Pressable style={styles.secondary} onPress={() => setTaken(null)} accessibilityRole="button">
              <Ionicons name="refresh" size={18} color={WHITE} />
              <Text style={styles.secondaryText}>Отново</Text>
            </Pressable>
            <Pressable style={styles.primary} onPress={() => onTaken(taken)} accessibilityRole="button">
              <Text style={styles.primaryText}>Добре</Text>
              <Ionicons name="chevron-forward" size={18} color={DARK_TEXT} />
            </Pressable>
          </View>
        </SafeAreaView>
      </View>
    );

  // Мехурчето отива натам, накъдето телефонът е наклонен надолу (като на истински нивелир, обратно).
  const bx = Math.max(-1, Math.min(1, -level.x * 4)) * 38;
  const by = Math.max(-1, Math.min(1, level.y * 4)) * 38;
  return (
    <View style={styles.fill}>
      <CameraView ref={camera} style={StyleSheet.absoluteFill} mode="picture" enableTorch={torch} onCameraReady={() => setReady(true)} />
      <SafeAreaView style={[StyleSheet.absoluteFill, styles.overlay]} pointerEvents="box-none">
        <View style={styles.top}>
          <View style={styles.topRow}>
            <Pressable style={styles.round} onPress={onClose} accessibilityRole="button" accessibilityLabel="Затвори">
              <Ionicons name="close" size={20} color={WHITE} />
            </Pressable>
            <View style={styles.pill}>
              <Text style={styles.pillText}>{single ? `Отново · ${shot.title}` : `${step} от ${count} · ${shot.title}`}</Text>
            </View>
            <Pressable
              style={[styles.round, torch && styles.roundOn]}
              onPress={() => setTorch((t) => !t)}
              accessibilityRole="button"
              accessibilityState={{ selected: torch }}
              accessibilityLabel={torch ? 'Изключи фенерчето' : 'Включи фенерчето'}
            >
              <Ionicons name={torch ? 'flash' : 'flash-off'} size={20} color={torch ? DARK_TEXT : WHITE} />
            </Pressable>
          </View>
          <View style={[styles.panel, styles.guideRow]}>
            <ShotGuide photo={shot.photo} width={132} height={70} />
            <View style={styles.guideText}>
              <Text style={styles.panelTitle}>Снимка {shot.title.toLowerCase()}</Text>
              <Text style={styles.hint}>{shot.how}</Text>
            </View>
          </View>
          <Text style={styles.under}>
            {shot.photo === 'top'
              ? 'Телефонът над предмета, успореден на масата. Предметът в рамката.'
              : 'Оранжевата страна гледа нагоре, към телефона. Телефонът пак успореден на масата.'}
          </Text>
          {message && <Text style={styles.warn}>{message}</Text>}
        </View>

        <View style={styles.middle} pointerEvents="none">
          <View style={styles.frame}>
            <View style={[styles.corner, styles.tl]} />
            <View style={[styles.corner, styles.tr]} />
            <View style={[styles.corner, styles.bl]} />
            <View style={[styles.corner, styles.br]} />
          </View>
          {level.available && (
            <View style={[styles.bubbleRing, straight && styles.bubbleRingOk]}>
              <View style={styles.cross} />
              <View style={[styles.cross, styles.crossV]} />
              <View style={[styles.bubble, straight && styles.bubbleOk, { transform: [{ translateX: bx }, { translateY: by }] }]} />
            </View>
          )}
        </View>

        <View style={styles.bottomArea}>
          <Text style={[styles.status, straight && styles.statusOk]}>
            {!level.available ? 'Дръжте телефона успоредно на масата' : straight ? 'Право е: снимайте' : `Изправете телефона (${Math.round(level.tilt)}°)`}
          </Text>
          <View style={styles.bottom}>
            <View style={styles.side} />
            <Pressable
              style={[styles.shutter, (!ready || taking || !straight) && styles.disabled]}
              accessibilityRole="button"
              accessibilityLabel="Снимай"
              disabled={!ready || taking || !straight}
              onPress={take}
            >
              <View style={[styles.shutterIn, straight && styles.shutterOk]} />
            </Pressable>
            <View style={styles.side}>
              {!shot.required && !single && (
                <Pressable style={styles.skip} onPress={onSkip} accessibilityRole="button" accessibilityHint="Ако страната е като отсрещната или предметът не стои на нея">
                  <Text style={styles.skipText}>Пропусни</Text>
                </Pressable>
              )}
            </View>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: dark.bg },
  overlay: { justifyContent: 'space-between' },
  top: { padding: space.md, gap: space.sm },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  round: {
    width: 40,
    height: 40,
    borderRadius: radius.sm,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.lineStrong,
    alignItems: 'center',
    justifyContent: 'center',
  },
  roundOn: { backgroundColor: CAM_ACCENT, borderColor: CAM_ACCENT },
  pill: {
    height: 36,
    justifyContent: 'center',
    paddingHorizontal: 14,
    borderRadius: radius.pill,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.lineStrong,
  },
  pillText: { color: WHITE, fontSize: 12.5, fontFamily: font.monoMedium },
  panel: { padding: space.lg, gap: 6, borderRadius: radius.md, backgroundColor: GLASS, borderWidth: 1, borderColor: dark.line },
  panelTitle: { color: WHITE, fontSize: 16, fontFamily: font.semibold },
  guideRow: { flexDirection: 'row', alignItems: 'center', gap: space.md, padding: space.md },
  guideText: { flex: 1, gap: 4 },
  under: {
    color: WHITE,
    fontSize: 13,
    lineHeight: 18,
    fontFamily: font.medium,
    textAlign: 'center',
    backgroundColor: GLASS,
    borderRadius: radius.sm,
    paddingHorizontal: space.sm,
    paddingVertical: 4,
    overflow: 'hidden',
  },
  hint: { color: dark.text2, fontSize: 13.5, lineHeight: 19, fontFamily: font.regular },
  warn: { color: dark.warning, fontSize: 14.5, lineHeight: 20, fontFamily: font.medium, textAlign: 'center' },
  middle: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  // Рамката е по-малка от кадъра: телефонът е по-високо и перспективата (по-близкото изглежда по-голямо) е по-малка.
  frame: { position: 'absolute', width: '58%', aspectRatio: 1.45 },
  corner: { position: 'absolute', width: 18, height: 18, borderColor: CAM_ACCENT },
  tl: { left: 0, top: 0, borderLeftWidth: 1.5, borderTopWidth: 1.5 },
  tr: { right: 0, top: 0, borderRightWidth: 1.5, borderTopWidth: 1.5 },
  bl: { left: 0, bottom: 0, borderLeftWidth: 1.5, borderBottomWidth: 1.5 },
  br: { right: 0, bottom: 0, borderRightWidth: 1.5, borderBottomWidth: 1.5 },
  // Нивелирът е кръгъл като истинския (кръг, не ъгъл).
  bubbleRing: {
    width: 92,
    height: 92,
    borderRadius: radius.pill,
    borderWidth: 1.5,
    borderColor: dark.lineStrong,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: GLASS,
  },
  bubbleRingOk: { borderColor: GOOD },
  cross: { position: 'absolute', width: 92, height: 1, backgroundColor: dark.lineStrong },
  crossV: { width: 1, height: 92 },
  bubble: { width: 22, height: 22, borderRadius: radius.pill, backgroundColor: dark.text },
  bubbleOk: { backgroundColor: GOOD },
  bottomArea: { gap: space.sm },
  status: { color: dark.signal, textAlign: 'center', fontFamily: font.monoMedium, fontSize: 14 },
  statusOk: { color: GOOD },
  bottom: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    margin: space.md,
    marginTop: 0,
    paddingVertical: space.md,
    paddingHorizontal: space.sm,
    borderRadius: radius.md,
    backgroundColor: dark.surface,
    borderWidth: 1,
    borderColor: dark.line,
  },
  side: { width: 96, alignItems: 'center' },
  shutter: { width: 72, height: 72, borderRadius: radius.pill, borderWidth: 3, borderColor: WHITE, alignItems: 'center', justifyContent: 'center' },
  shutterIn: { width: 56, height: 56, borderRadius: radius.pill, backgroundColor: WHITE },
  shutterOk: { backgroundColor: GOOD },
  disabled: { opacity: 0.4 },
  skip: { height: 40, paddingHorizontal: 14, borderRadius: radius.sm, justifyContent: 'center', borderWidth: 1, borderColor: dark.lineStrong },
  skipText: { color: WHITE, fontFamily: font.medium, fontSize: 14 },
  choices: { flexDirection: 'row', gap: space.md, margin: space.md },
  secondary: {
    flex: 1,
    height: 48,
    borderRadius: radius.sm,
    flexDirection: 'row',
    gap: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: dark.surface,
    borderWidth: 1,
    borderColor: dark.lineStrong,
  },
  secondaryText: { color: WHITE, fontFamily: font.semibold, fontSize: 15 },
  primary: { flex: 1, height: 48, borderRadius: radius.sm, flexDirection: 'row', gap: 4, alignItems: 'center', justifyContent: 'center', backgroundColor: CAM_ACCENT },
  primaryText: { color: DARK_TEXT, fontFamily: font.semibold, fontSize: 15 },
});

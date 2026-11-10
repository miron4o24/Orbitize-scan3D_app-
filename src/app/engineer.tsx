import Ionicons from "@expo/vector-icons/Ionicons";
import { CameraView, useCameraPermissions } from "expo-camera";
import { File } from "expo-file-system";
import { Image } from "expo-image";
import { useKeepAwake } from "expo-keep-awake";
import { router } from "expo-router";
import { useEffect, useRef, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  Button,
  Card,
  ErrorText,
  Field,
  Header,
  IconButton,
  Note,
  Screen,
  Section,
  Segmented,
  Small,
} from "@/components/ui";
import {
  errorText,
  type CadEngine,
  type CadFeature,
  type CadFeatureKind,
} from "@/lib/api";
import { PRO_INFO } from "@/lib/prefs";
import { createScan, deleteScan, photosDir, updateScan } from "@/lib/scans";
import { currentUserId, useSession } from "@/lib/session";
import { font, makeStyles, palettes, radius, space } from "@/theme";

// Инженерният режим (сървър app/engineer.py): снимки на малък детайл и мерки с шублер -> CAD модел 1:1 от елементи
// (STEP за CAD програмите и кодът на CadQuery). Първо снимките, после мерките.

const MIN = 8;
const MAX = 40;
const GOOD = 12; // с толкова снимки детайлът се вижда отвсякъде с припокриване

const TIPS = [
  "Детайлът върху шарена подложка (вестник, кърпа с щампа), не върху едноцветна маса.",
  "Обиколете го: снимка през около 30°, после още няколко отгоре.",
  "Еднакво разстояние, без приближаване. Детайлът да заема половината кадър.",
  "Светло, без светкавица и без силни отблясъци.",
];

const KINDS: { value: CadFeatureKind; label: string }[] = [
  { value: "hole", label: "Отвор Ø" },
  { value: "diameter", label: "Диаметър" },
  { value: "thickness", label: "Дебелина" },
  { value: "other", label: "Друго" },
];

type Measure = {
  key: number;
  kind: CadFeatureKind;
  value: string;
  label: string;
};

/* Милиметри от полето (запетая или точка); undefined за празно или извън 0 до 1000. */
function mmOf(t: string): number | undefined {
  const v = Number(t.trim().replace(",", "."));
  return t.trim() && Number.isFinite(v) && v > 0 && v <= 1000 ? v : undefined;
}

export default function Engineer() {
  useKeepAwake();
  const [permission, requestPermission] = useCameraPermissions();
  const session = useSession();
  const user = session?.kind === "account" ? session.user : null;
  const claudeOk = !!user?.owner && !!user.ai?.cad_claude; // абонаментът е на собственика: само за неговия профил
  const [scan] = useState(() =>
    createScan(currentUserId() ?? undefined, "cad"),
  );
  const camera = useRef<CameraView>(null);
  const finished = useRef(false);
  const [ready, setReady] = useState(false);
  const [taking, setTaking] = useState(false);
  const [torch, setTorch] = useState(false);
  const [shots, setShots] = useState<File[]>([]);
  const [phase, setPhase] = useState<"photos" | "measures">("photos");
  const [message, setMessage] = useState<string | null>(null);

  // Излизане без започнат модел не бива да оставя празно сканиране в историята.
  useEffect(
    () => () => {
      if (!finished.current) deleteScan(scan.id);
    },
    [scan.id],
  );

  async function take() {
    if (!camera.current || taking || shots.length >= MAX) return;
    setTaking(true);
    setMessage(null);
    try {
      // Без skipProcessing: снимката е завъртяна както е държан телефонът.
      const pic = await camera.current.takePictureAsync({
        quality: 0.92,
        shutterSound: false,
      });
      const dest = new File(
        photosDir(scan.id),
        `img_${String(shots.length + 1).padStart(4, "0")}.jpg`,
      );
      if (dest.exists) dest.delete();
      new File(pic.uri).moveSync(dest); // временният файл на камерата се трие
      setShots((s) => [...s, dest]);
    } catch (e) {
      setMessage(`Снимката не стана. ${errorText(e)}`);
    } finally {
      setTaking(false);
    }
  }

  function undo() {
    const last = shots[shots.length - 1];
    if (!last) return;
    if (last.exists) last.delete();
    setShots((s) => s.slice(0, -1));
  }

  if (phase === "measures")
    return (
      <Measures
        photos={shots.length}
        claudeOk={claudeOk}
        onBack={() => setPhase("photos")}
        onStart={(cad) => {
          finished.current = true;
          updateScan(scan.id, {
            status: "uploading",
            photoCount: shots.length,
            method: "cad",
            cad,
            flatBase: true,
          });
          router.replace({
            pathname: "/processing/[id]",
            params: { id: scan.id, auto: "1" },
          });
        }}
      />
    );

  if (!permission) return <View style={styles.fill} />;
  if (!permission.granted)
    return (
      <Screen>
        <Header title="Достъп до камерата" />
        {permission.canAskAgain ? (
          <Button
            title="Разреши камерата"
            icon="camera-outline"
            onPress={requestPermission}
          />
        ) : (
          <Small>
            Достъпът до камерата е отказан. Разрешете го от настройките на
            телефона.
          </Small>
        )}
        <Button title="Назад" variant="ghost" onPress={() => router.back()} />
      </Screen>
    );

  const n = shots.length;
  const enough = n >= MIN;
  return (
    <View style={styles.fill}>
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        mode="picture"
        enableTorch={torch}
        onCameraReady={() => setReady(true)}
      />
      <SafeAreaView
        style={[StyleSheet.absoluteFill, styles.overlay]}
        pointerEvents="box-none"
      >
        <View style={styles.top}>
          <View style={styles.topRow}>
            <Pressable
              style={styles.round}
              onPress={() => router.back()}
              accessibilityRole="button"
              accessibilityLabel="Затвори"
            >
              <Ionicons name="close" size={20} color={WHITE} />
            </Pressable>
            <View style={styles.pill}>
              <Text style={styles.pillText}>
                {PRO_INFO.cad.label} ·{" "}
                {n < MIN
                  ? `${n} от поне ${MIN}`
                  : n < GOOD
                    ? `${n} · още ${GOOD - n} за по-добро`
                    : `${n} снимки`}
              </Text>
            </View>
            <Pressable
              style={[styles.round, torch && styles.roundOn]}
              onPress={() => setTorch((t) => !t)}
              accessibilityRole="button"
              accessibilityState={{ selected: torch }}
              accessibilityLabel={
                torch ? "Изключи фенерчето" : "Включи фенерчето"
              }
            >
              <Ionicons
                name={torch ? "flash" : "flash-off"}
                size={20}
                color={torch ? DARK_TEXT : WHITE}
              />
            </Pressable>
          </View>
          {n === 0 && (
            <View style={styles.panel}>
              {TIPS.map((t, i) => (
                <View key={t} style={styles.tipRow}>
                  <Text style={styles.tipNum}>
                    {String(i + 1).padStart(2, "0")}
                  </Text>
                  <Text style={[styles.hint, styles.flex]}>{t}</Text>
                </View>
              ))}
            </View>
          )}
          {message && <Text style={styles.warn}>{message}</Text>}
        </View>
        <View style={styles.bottomArea}>
          {n > 0 && (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.strip}
            >
              {shots.map((f) => (
                <Image
                  key={f.uri}
                  source={{ uri: f.uri }}
                  style={styles.thumb}
                  contentFit="cover"
                />
              ))}
            </ScrollView>
          )}
          <View style={styles.bottom}>
            <View style={styles.side}>
              {n > 0 && (
                <Pressable
                  style={styles.round}
                  onPress={undo}
                  accessibilityRole="button"
                  accessibilityLabel="Махни последната снимка"
                >
                  <Ionicons name="arrow-undo" size={19} color={WHITE} />
                </Pressable>
              )}
            </View>
            <Pressable
              style={[
                styles.shutter,
                (!ready || taking || n >= MAX) && styles.disabled,
              ]}
              accessibilityRole="button"
              accessibilityLabel="Снимай"
              disabled={!ready || taking || n >= MAX}
              onPress={take}
            >
              <View style={styles.shutterIn} />
            </Pressable>
            <View style={styles.side}>
              <Pressable
                style={[styles.next, !enough && styles.disabled]}
                disabled={!enough}
                onPress={() => setPhase("measures")}
                accessibilityRole="button"
                accessibilityLabel="Напред към мерките"
              >
                <Text style={styles.nextText}>Мерки</Text>
                <Ionicons name="chevron-forward" size={16} color={DARK_TEXT} />
              </Pressable>
            </View>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

/* Мерките с шублер: общите размери (поне един, колкото повече, толкова по-точно) и по желание мерки на елементи. */
function Measures({
  photos,
  claudeOk,
  onBack,
  onStart,
}: {
  photos: number;
  claudeOk: boolean;
  onBack: () => void;
  onStart: (cad: {
    length?: number;
    width?: number;
    height?: number;
    features: CadFeature[];
    engine: CadEngine;
  }) => void;
}) {
  const s = useStyles();
  const [length, setLength] = useState("");
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [rows, setRows] = useState<Measure[]>([]);
  const [engine, setEngine] = useState<CadEngine>("local");
  const [error, setError] = useState<string | null>(null);
  const next = useRef(1);

  const set = (key: number, patch: Partial<Measure>) =>
    setRows((r) => r.map((m) => (m.key === key ? { ...m, ...patch } : m)));
  const overall = [mmOf(length), mmOf(width), mmOf(height)];

  function start() {
    if (!overall.some(Boolean))
      return setError("Въведете поне един общ размер в милиметри.");
    const bad =
      [length, width, height].some((t) => t.trim() && !mmOf(t)) ||
      rows.some((m) => m.value.trim() && !mmOf(m.value));
    if (bad)
      return setError(
        "Размерите са в милиметри, от 0 до 1000 (например 12,5).",
      );
    const features = rows
      .filter((m) => mmOf(m.value))
      .map((m) => ({
        kind: m.kind,
        value: mmOf(m.value)!,
        ...(m.label.trim() && { label: m.label.trim().slice(0, 40) }),
      }));
    onStart({
      length: overall[0],
      width: overall[1],
      height: overall[2],
      features,
      engine,
    });
  }

  return (
    <SafeAreaView style={s.safe} edges={["top"]}>
      <Screen>
        <Header
          kicker={`${PRO_INFO.cad.label} · ${photos} снимки`}
          title="Мерки с шублер"
          sub="В милиметри. Моделът ги спазва точно."
        />
        <Section label="Общ размер (поне един)">
          <Card style={s.card}>
            {(
              [
                [
                  "Дължина",
                  "най-дългата страна, както лежи",
                  length,
                  setLength,
                ],
                ["Ширина", "другата страна отгоре", width, setWidth],
                ["Височина", "от масата нагоре", height, setHeight],
              ] as const
            ).map(([label, hint, value, change]) => (
              <View key={label} style={s.line}>
                <View style={s.flex}>
                  <Text style={s.label}>{label}</Text>
                  <Small>{hint}</Small>
                </View>
                <Field
                  style={s.mm}
                  value={value}
                  onChangeText={change}
                  keyboardType="decimal-pad"
                  placeholder="мм"
                  accessibilityLabel={`${label} в милиметри`}
                  invalid={!!value.trim() && !mmOf(value)}
                />
              </View>
            ))}
          </Card>
        </Section>

        <Section label="Мерки на елементи (по желание)">
          {rows.map((m) => (
            <Card key={m.key} style={s.card}>
              <Segmented<CadFeatureKind>
                value={m.kind}
                onChange={(kind) => set(m.key, { kind })}
                options={KINDS}
              />
              <View style={s.line}>
                <Field
                  style={s.flex}
                  value={m.label}
                  onChangeText={(label) => set(m.key, { label })}
                  placeholder={
                    m.kind === "other"
                      ? "Какво е (напр. стъпка на резба)"
                      : "Име по желание (напр. монтажен отвор)"
                  }
                  maxLength={40}
                />
                <Field
                  style={s.mm}
                  value={m.value}
                  onChangeText={(value) => set(m.key, { value })}
                  keyboardType="decimal-pad"
                  placeholder="мм"
                  accessibilityLabel="Стойност в милиметри"
                  invalid={!!m.value.trim() && !mmOf(m.value)}
                />
                <IconButton
                  icon="trash-outline"
                  label="Махни мярката"
                  onPress={() =>
                    setRows((r) => r.filter((x) => x.key !== m.key))
                  }
                />
              </View>
            </Card>
          ))}
          <Button
            title="Добави мярка"
            icon="add"
            variant="secondary"
            onPress={() =>
              setRows((r) => [
                ...r,
                { key: next.current++, kind: "hole", value: "", label: "" },
              ])
            }
            disabled={rows.length >= 20}
          />
          <Small>
            Отвор: диаметърът му. Диаметър: на вал или издатина. Дебелина: на
            плоча или стъпало. Еднаквите отвори се въвеждат веднъж.
          </Small>
        </Section>

        {claudeOk && (
          <Section label="Кой прави модела">
            <Segmented<CadEngine>
              value={engine}
              onChange={setEngine}
              options={[
                { value: "local", label: "Сървърът (безплатно)" },
                { value: "claude", label: "Claude" },
              ]}
            />
            <Small>
              {engine === "claude"
                ? "Claude гледа снимките и мерките и проектира детайла по елементи, проверен срещу сканирането. По вашия абонамент, без допълнително плащане. Около 5–10 мин повече."
                : "Сървърът разпознава блокове, цилиндри, отвори и изрези от сканирането."}
            </Small>
          </Section>
        )}

        <Note>
          Получавате STEP (за SolidWorks, Fusion, FreeCAD, Onshape), кода на
          модела (CadQuery) и STL в милиметри.
        </Note>
        {error && <ErrorText>{error}</ErrorText>}
        <Button
          title="Създай CAD модел"
          variant="brand"
          size="lg"
          icon="cube-outline"
          onPress={start}
        />
        <Button
          title="Още снимки"
          variant="ghost"
          icon="camera-outline"
          onPress={onBack}
        />
      </Screen>
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  safe: { flex: 1, backgroundColor: c.bg },
  flex: { flex: 1 },
  card: { gap: space.md },
  line: { flexDirection: "row", alignItems: "center", gap: space.md },
  label: { fontFamily: font.medium, fontSize: 15, color: c.text },
  mm: { width: 96, textAlign: "right" },
}));

// Върху изображението от камерата: винаги цветовете на тъмната тема.
const dark = palettes.dark;
const WHITE = "#ffffff";
const DARK_TEXT = "#09090b";
const GLASS = "rgba(9,9,11,0.72)";

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#000" },
  overlay: { justifyContent: "space-between" },
  flex: { flex: 1 },
  top: { padding: space.md, gap: space.sm },
  topRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  roundOn: { backgroundColor: "#facc15", borderColor: "#facc15" },
  round: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
    alignItems: "center",
    justifyContent: "center",
  },
  pill: {
    height: 36,
    justifyContent: "center",
    paddingHorizontal: 14,
    borderRadius: 18,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  pillText: { color: WHITE, fontSize: 13.5, fontFamily: font.medium },
  panel: {
    padding: space.lg,
    gap: space.sm,
    borderRadius: radius.md,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  tipRow: { flexDirection: "row", gap: space.md, alignItems: "flex-start" },
  tipNum: {
    fontFamily: font.mono,
    fontSize: 12,
    lineHeight: 18,
    color: dark.muted,
    width: 20,
  },
  hint: {
    color: dark.text2,
    fontSize: 13,
    lineHeight: 18,
    fontFamily: font.regular,
  },
  warn: {
    color: dark.warning,
    fontSize: 14.5,
    lineHeight: 20,
    fontFamily: font.medium,
    textAlign: "center",
  },
  bottomArea: { gap: space.sm },
  strip: { gap: 6, paddingHorizontal: space.md },
  thumb: {
    width: 48,
    height: 48,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: dark.line,
  },
  bottom: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    margin: space.md,
    marginTop: 0,
    paddingVertical: space.md,
    paddingHorizontal: space.sm,
    borderRadius: 24,
    backgroundColor: GLASS,
    borderWidth: 1,
    borderColor: dark.line,
  },
  side: { width: 96, alignItems: "center" },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: WHITE,
    alignItems: "center",
    justifyContent: "center",
  },
  shutterIn: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: WHITE,
  },
  next: {
    flexDirection: "row",
    alignItems: "center",
    gap: 2,
    height: 40,
    paddingHorizontal: 14,
    borderRadius: 20,
    backgroundColor: WHITE,
  },
  nextText: { color: DARK_TEXT, fontSize: 14, fontFamily: font.semibold },
  disabled: { opacity: 0.4 },
});

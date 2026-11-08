import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { useState } from "react";
import { View } from "react-native";

import {
  downloadCadFile,
  errorText,
  type CadFile,
  type CadReport,
} from "@/lib/api";
import { makeStyles, space } from "@/theme";
import { Button, Card, ErrorText, Row, Section, Small } from "./ui";

// Инженерният режим на екрана за преглед: какво е построено (елементите по реда в кода), колко е близо до
// сканирането и мерките, и файловете за CAD програмите.

const FILES: {
  name: CadFile;
  title: string;
  sub: string;
  mime: string;
  icon:
    "cube-outline" | "code-slash-outline" | "images-outline" | "print-outline";
}[] = [
  {
    name: "part.step",
    title: "STEP за CAD",
    sub: "SolidWorks, Fusion, FreeCAD, Onshape",
    mime: "application/step",
    icon: "cube-outline",
  },
  {
    name: "part.py",
    title: "Кодът на модела",
    sub: "CadQuery: всеки размер се променя",
    mime: "text/x-python",
    icon: "code-slash-outline",
  },
  {
    name: "compare.png",
    title: "Сравнение със сканирането",
    sub: "горе сканирането, долу моделът",
    mime: "image/png",
    icon: "images-outline",
  },
  {
    name: "part.stl",
    title: "STL в милиметри",
    sub: "за печат, в реален размер",
    mime: "model/stl",
    icon: "print-outline",
  },
];

const mm = (v: number) => `${Math.round(v * 100) / 100} мм`.replace(".", ",");

export function CadPanel({ jobId, cad }: { jobId: string; cad: CadReport }) {
  const s = useStyles();
  const [busy, setBusy] = useState<CadFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const claude = cad.engine === "claude" && cad.claude;
  const size = claude
    ? [
        cad.claude!.size_mm.length,
        cad.claude!.size_mm.width,
        cad.claude!.size_mm.height,
      ]
    : cad.local.size_mm;
  const dev = claude
    ? cad.claude!.deviation_from_scan_mm
    : cad.local.deviation_mm;

  async function share(f: (typeof FILES)[number]) {
    setBusy(f.name);
    setError(null);
    try {
      const file = new File(
        Paths.cache,
        `${f.name.replace(".", `-${jobId.slice(0, 8)}.`)}`,
      );
      if (file.exists) file.delete();
      await downloadCadFile(jobId, f.name, file);
      await Sharing.shareAsync(file.uri, {
        mimeType: f.mime,
        dialogTitle: f.title,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Section
      label={`CAD модел · ${claude ? `Claude (${cad.rounds ?? 1} проверки)` : "сървърът"}`}
    >
      <Card style={s.list}>
        <Row
          label="Размер"
          value={size.map((v) => mm(v).replace(" мм", "")).join(" × ") + " мм"}
        />
        <Row
          label="Разлика от сканирането"
          value={`средно ${mm(dev.mean)}, 90% до ${mm(dev.p90)}`}
          last
        />
      </Card>
      {cad.claude_failed && (
        <Small>Claude не успя този път: това е моделът на сървъра.</Small>
      )}
      {!claude && cad.local.elements.length > 0 && (
        <Card style={s.box}>
          {cad.local.elements.map((e, i) => (
            <Small key={`${i}-${e}`}>{`${i + 1}. ${e}`}</Small>
          ))}
        </Card>
      )}
      {cad.local.notes.length > 0 && (
        <Card style={s.box}>
          {cad.local.notes.map((n) => (
            <Small key={n}>• {n}</Small>
          ))}
        </Card>
      )}
      <View style={s.files}>
        {FILES.map((f) => (
          <Button
            key={f.name}
            title={f.title}
            sub={f.sub}
            icon={f.icon}
            variant={f.name === "part.step" ? "primary" : "secondary"}
            loading={busy === f.name}
            disabled={!!busy && busy !== f.name}
            onPress={() => share(f)}
          />
        ))}
      </View>
      {error && <ErrorText>{error}</ErrorText>}
    </Section>
  );
}

const useStyles = makeStyles(() => ({
  list: { paddingVertical: 0 },
  box: { gap: 4 },
  files: { gap: space.sm },
}));

import { File, Paths } from 'expo-file-system';
import { Image } from 'expo-image';
import { router } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useState } from 'react';
import { Text, View } from 'react-native';

import { Button, Card, ErrorText, Note, Row, Section, Small } from '@/components/ui';
import { authHeaders, cadPhotoUrl, downloadCadFile, errorText, type CadFile, type CadPhoto, type CadReport } from '@/lib/api';
import { font, makeStyles, radius, space } from '@/theme';

import { mmText } from './Measure';

// Построеният детайл на екрана за преглед: размерът, кое е измерено и кое оценено, елементите, файловете за CAD
// програмите и проверката върху снимките.

const FILES: { name: CadFile; title: string; sub: string; mime: string; icon: 'cube-outline' | 'document-text-outline' | 'code-slash-outline' | 'print-outline' }[] = [
  { name: 'part.step', title: 'STEP за CAD', sub: 'SolidWorks, Fusion, FreeCAD, Onshape', mime: 'application/step', icon: 'cube-outline' },
  { name: 'drawing.png', title: 'Чертеж', sub: 'изгледите и всички размери', mime: 'image/png', icon: 'document-text-outline' },
  { name: 'part.py', title: 'Кодът на модела', sub: 'CadQuery: всеки размер е число най-горе', mime: 'text/x-python', icon: 'code-slash-outline' },
  { name: 'part.stl', title: 'STL в милиметри', sub: 'за печат, в реален размер', mime: 'model/stl', icon: 'print-outline' },
];

export function CadPanel({ scanId, jobId, cad, stamp }: { scanId: string; jobId: string; cad: CadReport; stamp: string }) {
  const s = useStyles();
  const [busy, setBusy] = useState<CadFile | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function share(f: (typeof FILES)[number]) {
    setBusy(f.name);
    setError(null);
    try {
      const file = new File(Paths.cache, f.name.replace('.', `-${jobId.slice(0, 8)}.`));
      if (file.exists) file.delete();
      await downloadCadFile(jobId, f.name, file);
      await Sharing.shareAsync(file.uri, { mimeType: f.mime, dialogTitle: f.title });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Section label="CAD детайл">
      <Card style={s.list}>
        <Row label="Размер" value={`${cad.size_mm.map(mmText).join(' × ')} мм`} />
        <Row label="Измерени с шублер" value={String(cad.measured)} />
        <Row label="Оценени от снимките" value={String(cad.estimated)} last />
      </Card>
      {cad.estimated > 0 && (
        <Note warn>{`Оценени, не измерени: ${cad.estimated_names.join(', ')}. Проверете ги преди изработка; в чертежа са със звездичка.`}</Note>
      )}
      <Card style={s.box}>
        {cad.elements.map((e, i) => (
          <Small key={`${i}-${e}`}>{`${i + 1}. ${e}`}</Small>
        ))}
      </Card>
      <View style={s.files}>
        {FILES.map((f) => (
          <Button
            key={f.name}
            title={f.title}
            sub={f.sub}
            icon={f.icon}
            variant={f.name === 'part.step' ? 'primary' : 'secondary'}
            loading={busy === f.name}
            disabled={!!busy && busy !== f.name}
            onPress={() => share(f)}
          />
        ))}
      </View>
      {error && <ErrorText>{error}</ErrorText>}
      <Button
        title="Промени мерките или елементите"
        icon="create-outline"
        variant="ghost"
        onPress={() => router.push({ pathname: '/engineer', params: { scan: scanId } })}
      />
      <Text style={s.label}>Проверка върху снимките</Text>
      <Small>Оранжевото е построеният детайл. Трябва да лежи точно върху ръбовете на снимката.</Small>
      {(['top', 'front', 'right'] as const).map((v) => (
        <CheckPhoto key={v} jobId={jobId} photo={v} stamp={stamp} />
      ))}
    </Section>
  );
}

/* Снимка с построения детайл върху нея; ако тази страна не е снимана, не се показва. */
function CheckPhoto({ jobId, photo, stamp }: { jobId: string; photo: CadPhoto; stamp: string }) {
  const s = useStyles();
  const [missing, setMissing] = useState(false);
  if (missing) return null;
  return (
    <Image
      source={{ uri: cadPhotoUrl(jobId, photo, stamp), headers: authHeaders() }}
      style={s.photo}
      contentFit="contain"
      transition={150}
      onError={() => setMissing(true)}
    />
  );
}

const useStyles = makeStyles((c) => ({
  list: { paddingVertical: 0 },
  box: { gap: 4 },
  files: { gap: space.sm },
  label: { color: c.text, fontFamily: font.semibold, fontSize: 15, marginTop: space.sm },
  photo: { width: '100%', aspectRatio: 4 / 3, borderRadius: radius.sm, backgroundColor: c.surface2 },
}));

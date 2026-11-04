import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { useEffect, useState } from 'react';

import { Button, Card, ErrorText, Row, Section, Segmented } from '@/components/ui';
import { ApiError, downloadPrintZip, errorText, printPlan, type PrintPlan } from '@/lib/api';
import { makeStyles } from '@/theme';

// Готово за принтера: моделът е положен така, че да иска най-малко подпори, и е разрязан на части, когато е по-голям
// от плочата на принтера (сървър app/printing.py). Обичайни плочи по страна.
const BEDS = {
  s: { label: '18 см', bed: [180, 180, 180] },
  m: { label: '22 см', bed: [220, 220, 250] },
  l: { label: '25 см', bed: [256, 256, 256] },
  xl: { label: '35 см', bed: [350, 350, 350] },
} as const;
type Bed = keyof typeof BEDS;

export function PrintHelper({ jobId, mm }: { jobId: string; mm: number }) {
  const s = useStyles();
  const [bed, setBed] = useState<Bed>('m');
  const [result, setResult] = useState<{ key: string; plan?: PrintPlan; error?: string } | null>(null);
  const [missing, setMissing] = useState(false); // сървър отпреди тази функция
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const size = BEDS[bed].bed;
  const key = `${jobId}:${mm}:${bed}`;
  const plan = result?.key === key ? result.plan : undefined; // отговор за друг размер не е отговор
  const planError = result?.key === key ? result.error : undefined;

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      printPlan(jobId, mm, size).then(
        (p) => !cancelled && setResult({ key, plan: p }),
        (e) => {
          if (cancelled) return;
          if (e instanceof ApiError && (e.status === 404 || e.status === 405)) setMissing(true);
          else setResult({ key, error: errorText(e) });
        },
      );
    }, 400); // докато размерът още се въвежда
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, mm, size, key]);

  async function download() {
    setBusy(true);
    setError(null);
    try {
      const file = new File(Paths.cache, `print-${jobId.slice(0, 8)}-${Math.round(mm)}mm-${bed}.zip`);
      if (file.exists) file.delete();
      await downloadPrintZip(jobId, mm, size, file);
      await Sharing.shareAsync(file.uri, { mimeType: 'application/zip', dialogTitle: 'Файлове за печат' });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  if (missing) return null;
  const saved = plan?.turned && plan.support_before > 0 ? Math.round((1 - plan.support_after / plan.support_before) * 100) : 0;
  return (
    <Section label="Готово за принтера · плоча">
      <Segmented<Bed> value={bed} onChange={setBed} options={(Object.keys(BEDS) as Bed[]).map((b) => ({ value: b, label: BEDS[b].label }))} />
      <Card style={s.list}>
        <Row label="Части" value={plan ? String(plan.parts) : '…'} last={!saved} />
        {saved > 0 && <Row label="Завъртян за по-малко подпори" value={`−${saved}%`} last />}
      </Card>
      {(error || planError) && <ErrorText>{error ?? planError}</ErrorText>}
      <Button title="Изтегли за печат" icon="print-outline" variant="secondary" onPress={download} loading={busy} disabled={!plan} />
    </Section>
  );
}

const useStyles = makeStyles(() => ({
  list: { paddingVertical: 0 },
}));

import { File } from 'expo-file-system';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

import { ApiError, cancelJob, createJob, errorText, getJob, startJob, uploadPhoto } from '@/lib/api';
import { deleteScan, getScan, photosDir, updateScan, type Scan } from '@/lib/scans';

const ACTIVE = ['queued', 'reconstructing', 'meshing', 'cleaning'];
const STEPS: [string, string][] = [
  ['queued', 'Waiting for the server'],
  ['reconstructing', 'Locating cameras and computing dense 3D points (slowest step)'],
  ['meshing', 'Building the surface'],
  ['cleaning', 'Making it printable'],
];

export default function Processing() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [scan, setScan] = useState<Scan | null>(() => getScan(id));
  const [uploaded, setUploaded] = useState<number | null>(null); // non-null while uploading
  const [netError, setNetError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const update = (patch: Partial<Scan>) => setScan(updateScan(id, patch));

  const active = !!scan && ACTIVE.includes(scan.status);
  const jobId = scan?.jobId;

  useEffect(() => {
    if (!active || !jobId) return;
    let stopped = false;
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        if (stopped) return;
        setNetError(null);
        setScan(updateScan(id, { status: job.state, error: job.error ?? undefined, report: job.report ?? undefined }));
        if (job.state === 'done') router.replace({ pathname: '/viewer/[id]', params: { id } });
      } catch (e) {
        if (stopped) return;
        if (e instanceof ApiError && e.status === 404)
          setScan(
            updateScan(id, {
              status: 'failed',
              jobId: undefined,
              error: 'The server no longer has this job (restarted or expired). Upload again.',
            }),
          );
        else setNetError(`Can't reach the server: ${errorText(e)}. Retrying…`);
      }
    };
    tick();
    const timer = setInterval(tick, 5000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [active, jobId, id]);

  if (!scan) return <Text style={styles.pad}>Scan not found.</Text>;

  const photos = photosDir(id)
    .list()
    .filter((f): f is File => f instanceof File && f.name.endsWith('.jpg'));

  async function uploadAndStart() {
    setNetError(null);
    setInfo(null);
    setUploaded(0);
    try {
      let job = scan!.jobId
        ? await getJob(scan!.jobId).catch((e) => {
            if (e instanceof ApiError && e.status === 404) return null; // expired on the server: start a new job
            throw e;
          })
        : null;
      if (!job) {
        job = await createJob();
        update({ jobId: job.id });
      }
      const have = new Set(job.received);
      let done = photos.filter((f) => have.has(f.name)).length;
      setUploaded(done);
      for (const f of photos) {
        if (have.has(f.name)) continue;
        await uploadPhoto(job.id, f);
        setUploaded(++done);
      }
      const started = await startJob(job.id, { flat_base: scan!.flatBase ?? true });
      update({ status: started.state, error: undefined });
    } catch (e) {
      setNetError(`Upload stopped: ${errorText(e)}. Already-sent photos are kept; tap again to resume.`);
    } finally {
      setUploaded(null);
    }
  }

  async function cancel() {
    if (scan!.jobId) await cancelJob(scan!.jobId).catch(() => {}); // already gone is fine
    update({ status: 'uploading', jobId: undefined, error: undefined });
    setInfo('Processing cancelled. Your photos are still on the phone.');
  }

  function remove() {
    if (scan!.jobId) cancelJob(scan!.jobId).catch(() => {});
    deleteScan(id);
    router.back();
  }

  if (active)
    return (
      <ScrollView contentContainerStyle={styles.pad}>
        {STEPS.map(([state, label]) => {
          const here = scan.status === state;
          const past = STEPS.findIndex(([s]) => s === scan.status) > STEPS.findIndex(([s]) => s === state);
          return (
            <View key={state} style={styles.step}>
              {here ? <ActivityIndicator /> : <Text style={styles.mark}>{past ? '✓' : '·'}</Text>}
              <Text style={[styles.stepText, here && styles.bold]}>{label}</Text>
            </View>
          );
        })}
        <Text style={styles.note}>
          Processing runs on the server CPU and can take 10–60 minutes. You can leave this screen; the scan keeps
          processing and you can come back from History.
        </Text>
        {netError && <Text style={styles.error}>{netError}</Text>}
        <Button title="Cancel processing" color="#b00020" onPress={cancel} />
      </ScrollView>
    );

  const uploading = uploaded !== null;
  return (
    <ScrollView contentContainerStyle={styles.pad}>
      {scan.status === 'failed' && (
        <View style={styles.failBox}>
          <Text style={styles.bold}>Reconstruction failed</Text>
          <Text>{scan.error}</Text>
          <Text style={styles.note}>
            If the photos could not be matched, retrying will not help: rescan slower, with more overlap and
            better light.
          </Text>
        </View>
      )}
      {info && <Text style={styles.note}>{info}</Text>}
      <Text style={styles.title}>{photos.length} photos ready</Text>
      <View style={styles.row}>
        <Text style={styles.flex}>Flat base (cut the bottom flat so it stands on the print bed)</Text>
        <Switch
          value={scan.flatBase ?? true}
          onValueChange={(v) => update({ flatBase: v })}
          disabled={uploading}
          accessibilityLabel="Flat base"
        />
      </View>
      {uploading ? (
        <View style={styles.progress}>
          <Text>
            Uploading {uploaded} / {photos.length}
          </Text>
          <View style={styles.bar}>
            <View style={[styles.fill, { width: `${(100 * uploaded) / Math.max(1, photos.length)}%` }]} />
          </View>
        </View>
      ) : (
        <Button
          title={scan.status === 'failed' ? 'Try again' : 'Upload and process'}
          onPress={uploadAndStart}
          disabled={photos.length === 0}
        />
      )}
      {netError && <Text style={styles.error}>{netError}</Text>}
      {!uploading && <Button title="Delete scan" color="#b00020" onPress={remove} />}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  pad: { padding: 20, gap: 14 },
  title: { fontSize: 18, fontWeight: '600' },
  step: { flexDirection: 'row', alignItems: 'center', gap: 10, minHeight: 28 },
  mark: { width: 20, textAlign: 'center', fontSize: 16 },
  stepText: { flex: 1, fontSize: 15 },
  bold: { fontWeight: '600' },
  note: { color: '#555', fontSize: 13 },
  error: { color: '#b00020' },
  failBox: { backgroundColor: '#fdecea', padding: 12, borderRadius: 8, gap: 6 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  flex: { flex: 1 },
  progress: { gap: 6 },
  bar: { height: 8, backgroundColor: '#e0e0e0', borderRadius: 4, overflow: 'hidden' },
  fill: { height: 8, backgroundColor: '#1565c0' },
});

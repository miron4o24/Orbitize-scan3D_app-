import { router } from 'expo-router';
import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';

import { Button, Card, ErrorText, Header, H3, Note, Screen, Section, Small } from '@/components/ui';
import { API_URL, errorText, health } from '@/lib/api';
import { importVideo } from '@/lib/importVideo';
import { openEntry, useLibrary, type Entry } from '@/lib/library';
import { aiAvailable } from '@/lib/prefs';
import { font, makeStyles, space, useColors } from '@/theme';

const statusLabel: Record<Entry['status'], string> = {
  capturing: 'capturing',
  uploading: 'uploading',
  queued: 'queued',
  reconstructing: 'reconstructing',
  meshing: 'meshing',
  cleaning: 'cleaning',
  done: 'done',
  failed: 'failed',
  cancelled: 'cancelled',
};

export default function Home() {
  const c = useColors();
  const s = useStyles();
  const { entries, error, refresh } = useLibrary();
  const [backend, setBackend] = useState<'unknown' | 'ok' | 'failed'>('unknown');
  const [busy, setBusy] = useState(false);

  const checkBackend = async () => {
    setBusy(true);
    try {
      const result = await health();
      setBackend(result.ok ? 'ok' : 'failed');
    } catch {
      setBackend('failed');
    } finally {
      setBusy(false);
    }
  };

  const importTestVideo = async () => {
    setBusy(true);
    try {
      await importVideo('scan');
    } catch (e) {
      setBackend(errorText(e) ? 'failed' : 'unknown');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Screen>
      <Header kicker="DEVELOPER CONSOLE" title="Orbitize test bench" sub="Direct access to reconstruction pipelines and job state." />
      <Note>Use this screen while developing. No onboarding, marketing, community, or animated scan flow.</Note>

      <Section label="Backend">
        <Card style={s.status}>
          <View style={s.flex}>
            <H3>{backend === 'ok' ? 'reachable' : backend === 'failed' ? 'unreachable' : 'not checked'}</H3>
            <Small selectable>{API_URL || 'EXPO_PUBLIC_API_URL is empty'}</Small>
          </View>
          <Button title="Check" variant="secondary" onPress={checkBackend} loading={busy} />
        </Card>
      </Section>

      <Section label="Pipelines">
        <View style={s.actions}>
          <Button title="1:1 photos" sub="six views + three measurements" icon="construct-outline" onPress={() => router.push('/part')} size="lg" />
          <Button title="Record video" sub="standard reconstruction" icon="videocam-outline" onPress={() => router.push('/capture')} size="lg" />
          {aiAvailable() && (
            <Button title="AI video" sub="single-image AI pipeline" icon="sparkles-outline" onPress={() => router.push({ pathname: '/capture', params: { ai: '1' } })} size="lg" />
          )}
          <Button title="Import video" icon="download-outline" variant="secondary" onPress={importTestVideo} loading={busy} />
        </View>
      </Section>

      <Section label={`Recent jobs · ${entries.length}`}>
        {error && <ErrorText>{error}</ErrorText>}
        {!entries.length && <Small>No local jobs yet.</Small>}
        {entries.slice(0, 12).map((entry) => (
          <Pressable key={entry.key} onPress={() => openEntry(entry)} style={({ pressed }) => [s.job, pressed && s.pressed]}>
            <View style={s.flex}>
              <Text style={s.jobTitle} numberOfLines={1}>{entry.title}</Text>
              <Text style={s.jobMeta} selectable>{entry.jobId ?? entry.scan?.id ?? entry.key}</Text>
            </View>
            <Text style={[s.jobStatus, { color: entry.status === 'done' ? c.success : entry.status === 'failed' ? c.danger : c.warning }]}>
              {statusLabel[entry.status]}
            </Text>
          </Pressable>
        ))}
        {!!entries.length && <Button title="Refresh jobs" variant="ghost" onPress={refresh} />}
      </Section>
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  status: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  flex: { flex: 1, minWidth: 0 },
  actions: { gap: space.sm },
  job: { minHeight: 56, flexDirection: 'row', alignItems: 'center', gap: space.md, paddingVertical: space.sm, borderBottomWidth: 1, borderBottomColor: c.line },
  pressed: { opacity: 0.55 },
  jobTitle: { color: c.text, fontFamily: font.medium, fontSize: 15 },
  jobMeta: { color: c.muted, fontFamily: font.mono, fontSize: 11, marginTop: 3 },
  jobStatus: { fontFamily: font.monoMedium, fontSize: 11 },
}));

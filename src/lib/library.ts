import { router, useFocusEffect } from 'expo-router';
import { useCallback, useMemo, useState } from 'react';

import type { BadgeKind } from '@/components/ui';
import { ApiError, cancelJob, errorText, listMyScans, type ServerScan } from './api';
import { deleteScan, importScan, listScans, scanTitle, updateScan, type Scan, type ScanStatus } from './scans';
import { currentUserId, useSession } from './session';

// "My scans": the scans on this phone plus, when signed in, the account's scans from the server (made on other
// phones, or deleted here). A scan on both is one entry; the server's state wins once processing has started.

export type Entry = {
  key: string;
  scan?: Scan; // the local record, if this phone has one
  server?: ServerScan;
  jobId?: string;
  title: string;
  createdAt: number;
  status: ScanStatus;
  thumb: boolean; // the server has a picture of it
};

/** The local scans the current user may see: an account's own (with nobody signed in, those of older guest use). */
export function visibleScans(): Scan[] {
  const uid = currentUserId();
  return listScans().filter((s) => (uid ? s.userId === uid : !s.userId));
}

export function badgeOf(status: ScanStatus): BadgeKind {
  if (status === 'done') return 'done';
  if (status === 'failed' || status === 'cancelled') return 'failed';
  if (status === 'capturing' || status === 'uploading') return 'waiting';
  return 'working';
}

/** Made by AI: an AI model, or a text change of a model. */
export function isAi(entry: Entry): boolean {
  const report = entry.server?.report ?? entry.scan?.report;
  const method = entry.scan?.method;
  return !!report?.ai || !!report?.edit || (!!method && method !== 'scan' && method !== 'express');
}

/** Copies the server's state into the matching local records. */
function sync(server: ServerScan[]) {
  const byJob = new Map(server.map((s) => [s.id, s]));
  for (const scan of visibleScans()) {
    const s = scan.jobId ? byJob.get(scan.jobId) : undefined;
    if (!s) continue;
    const patch: Partial<Scan> = {};
    // While the phone is still uploading, the server's "uploading" says nothing new.
    if (s.state !== 'uploading' && s.state !== scan.status) {
      patch.status = s.state;
      patch.error = s.error ?? undefined;
      patch.report = s.report ?? undefined;
    }
    if ((s.title || undefined) !== scan.title) patch.title = s.title || undefined;
    if ((s.share ?? undefined) !== scan.share) patch.share = s.share ?? undefined;
    if (s.published !== undefined && !!s.published !== !!scan.published) patch.published = !!s.published || undefined;
    if (Object.keys(patch).length) updateScan(scan.id, patch);
  }
}

function merge(local: Scan[], server: ServerScan[] | null): Entry[] {
  const out: Entry[] = [];
  const byJob = new Map((server ?? []).map((s) => [s.id, s]));
  for (const scan of local) {
    const s = scan.jobId ? byJob.get(scan.jobId) : undefined;
    if (s) byJob.delete(s.id);
    out.push({
      key: scan.id,
      scan,
      server: s,
      jobId: scan.jobId,
      title: scanTitle(scan),
      createdAt: scan.createdAt,
      status: scan.status,
      thumb: s ? s.thumb : scan.status === 'done' && !!scan.jobId,
    });
  }
  for (const s of byJob.values()) {
    const createdAt = s.created * 1000;
    out.push({
      key: s.id,
      server: s,
      jobId: s.id,
      title: scanTitle({ title: s.title, createdAt }),
      createdAt,
      status: s.state,
      thumb: s.thumb,
    });
  }
  return out.sort((a, b) => b.createdAt - a.createdAt);
}

export function useLibrary() {
  const session = useSession();
  const uid = session?.kind === 'account' ? session.user.id : null;
  const [local, setLocal] = useState<Scan[]>(visibleScans);
  const [server, setServer] = useState<ServerScan[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLocal(visibleScans());
    if (!uid) {
      setServer(null);
      return;
    }
    setLoading(true);
    try {
      const list = await listMyScans();
      sync(list);
      setServer(list);
      setLocal(visibleScans());
      setError(null);
    } catch (e) {
      setError(`Запазените сканирания не се заредиха: ${errorText(e)}`);
    } finally {
      setLoading(false);
    }
  }, [uid]);

  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh]),
  );

  const entries = useMemo(() => merge(local, server), [local, server]);
  return { entries, refresh, loading, error };
}

/** Opens a scan: the model when it is finished, its progress or upload screen otherwise. */
export function openEntry(entry: Entry) {
  let scan = entry.scan;
  const uid = currentUserId();
  if (!scan) {
    if (!entry.server || !uid) return;
    scan = importScan(entry.server, uid);
  }
  router.push({ pathname: scan.status === 'done' ? '/viewer/[id]' : '/processing/[id]', params: { id: scan.id } });
}

/** Deletes a scan from this phone and from the server (and so from the account on every phone). */
export async function deleteEntry(entry: Entry) {
  if (entry.jobId)
    await cancelJob(entry.jobId).catch((e) => {
      // Already gone is fine. A guest's server copy expires by itself, so offline is fine too; an account's would
      // come back from the server, so that must reach it.
      if (currentUserId() && !(e instanceof ApiError && e.status === 404)) throw e;
    });
  if (entry.scan) deleteScan(entry.scan.id);
}

import { Directory, File, Paths } from 'expo-file-system';

import type { Method, Report, ServerScan } from './api';
import { dayMonth } from './dates';
import type { Quality } from './prefs';

// Scans live on disk only: Paths.document/scans/<id>/{scan.json, video.mp4 or photos/, results}.
export type ScanStatus =
  | 'capturing'
  | 'uploading'
  | 'queued'
  | 'reconstructing'
  | 'meshing'
  | 'cleaning'
  | 'done'
  | 'failed'
  | 'cancelled';

export type Scan = {
  id: string;
  createdAt: number;
  status: ScanStatus;
  photoCount: number; // older photo scans; video scans keep 0
  videoSeconds?: number;
  jobId?: string;
  error?: string;
  longestMm?: number;
  flatBase?: boolean;
  report?: Report;
  userId?: string; // the account it belongs to; none: made as a guest (joins the account at sign-in)
  title?: string;
  share?: string; // the public link's id while shared
  published?: boolean; // in the community
  quality?: Quality; // how the server processes it (missing: detailed)
  imported?: boolean; // the video came from the gallery, not the app's camera
  method?: Method; // an AI model (Pro) instead of a scan (missing: scan)
  extendFrom?: string; // this video adds a missing side to that model (its job id)
};

const root = new Directory(Paths.document, 'scans');

export const scanDir = (id: string) => new Directory(root, id);
export const photosDir = (id: string) => new Directory(scanDir(id), 'photos');
export const videoFile = (id: string) => new File(scanDir(id), 'video.mp4'); // iOS records MOV; the server reads either
export const motionFile = (id: string) => new File(scanDir(id), 'motion.json'); // the phone's orientation while filming
export const modelFile = (id: string) => new File(scanDir(id), 'model.glb');
const metaFile = (id: string) => new File(scanDir(id), 'scan.json');

const newId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

export function createScan(userId?: string, method: Method = 'scan'): Scan {
  const scan: Scan = {
    id: newId(),
    createdAt: Date.now(),
    status: 'capturing',
    photoCount: 0,
    userId,
    ...(method !== 'scan' && { method }),
  };
  photosDir(scan.id).create({ intermediates: true });
  saveScan(scan);
  return scan;
}

/** A scan of the account made on another phone (or whose local copy was deleted): a local record to open it by. */
export function importScan(server: ServerScan, userId: string): Scan {
  const scan: Scan = {
    id: newId(),
    createdAt: server.created * 1000,
    status: server.state === 'uploading' ? 'failed' : server.state,
    photoCount: 0,
    jobId: server.id,
    report: server.report ?? undefined,
    error: server.state === 'uploading' ? 'Това сканиране не беше качено докрай.' : (server.error ?? undefined),
    userId,
    title: server.title || undefined,
    share: server.share ?? undefined,
    published: !!server.published || undefined,
  };
  scanDir(scan.id).create({ intermediates: true });
  saveScan(scan);
  return scan;
}

/** The title shown for a scan: its own, or the date. */
export const scanTitle = (scan: { title?: string; createdAt: number }) => scan.title || `Сканиране ${dayMonth(scan.createdAt)}`;

export function saveScan(scan: Scan) {
  metaFile(scan.id).writeSync(JSON.stringify(scan));
}

export function getScan(id: string): Scan | null {
  const f = metaFile(id);
  return f.exists ? (JSON.parse(f.textSync()) as Scan) : null;
}

export function updateScan(id: string, patch: Partial<Scan>): Scan {
  const scan = { ...getScan(id)!, ...patch };
  saveScan(scan);
  return scan;
}

export function listScans(): Scan[] {
  if (!root.exists) return [];
  return root
    .list()
    .filter((e): e is Directory => e instanceof Directory)
    .map((d) => getScan(d.name))
    .filter((s): s is Scan => s !== null)
    .sort((a, b) => b.createdAt - a.createdAt);
}

export function deleteScan(id: string) {
  const d = scanDir(id);
  if (d.exists) d.delete();
}

import { Directory, File, Paths } from 'expo-file-system';

import type { Method, Report, ServerScan } from './api';
import { dayMonth } from './dates';

// Сканиранията са само на диска: Paths.document/scans/<id>/{scan.json, video.mp4 или photos/, results}.
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
  photoCount: number; // по-стари сканирания със снимки; при видео остава 0
  videoSeconds?: number;
  jobId?: string;
  error?: string;
  longestMm?: number;
  flatBase?: boolean;
  report?: Report;
  userId?: string; // профилът, на който принадлежи; без такъв: създадено като гост (присъединява се към профила при влизане)
  title?: string;
  share?: string; // номерът на публичния линк, докато е споделено
  published?: boolean; // в общността
  imported?: boolean; // видеото е от галерията, а не от камерата на приложението
  method?: Method; // как е създаден (по-стари сканирания: липсва, "scan" или "express")
  classic?: boolean; // AI моделът по предишния начин
  extendFrom?: string; // това видео добавя липсваща страна към този модел (номер на задачата му)
};

const root = new Directory(Paths.document, 'scans');

export const scanDir = (id: string) => new Directory(root, id);
export const photosDir = (id: string) => new Directory(scanDir(id), 'photos');
export const videoFile = (id: string) => new File(scanDir(id), 'video.mp4'); // iOS записва MOV; сървърът чете и двата формата
export const motionFile = (id: string) => new File(scanDir(id), 'motion.json'); // положението на телефона по време на записа
export const modelFile = (id: string) => new File(scanDir(id), 'model.glb');
const metaFile = (id: string) => new File(scanDir(id), 'scan.json');

const newId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

export function createScan(userId?: string, method: Method = 'single'): Scan {
  const scan: Scan = {
    id: newId(),
    createdAt: Date.now(),
    status: 'capturing',
    photoCount: 0,
    userId,
    method,
  };
  photosDir(scan.id).create({ intermediates: true });
  saveScan(scan);
  return scan;
}

/**
 * Сканиране на профила, направено на друг телефон (или с изтрито местно копие): местен запис, през който да се
 * отвори.
 */
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

/* Показваното заглавие на сканиране: неговото или датата. */
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

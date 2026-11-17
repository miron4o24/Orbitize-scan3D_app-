import { Directory, File, Paths } from 'expo-file-system';

import type { CadDesign, Method, Report, ServerScan } from './api';
import { dayMonth } from './dates';
import type { Quality } from './prefs';

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
  quality?: Quality; // как го обработва сървърът (липсва: подробно)
  imported?: boolean; // видеото е от галерията, а не от камерата на приложението
  method?: Method; // AI модел или инженерният режим (липсва: старо стандартно сканиране)
  extendFrom?: string; // това видео добавя липсваща страна към този модел (номер на задачата му)
  cad?: CadDesign; // инженерният режим: описанието на детайла с мерките (за промяна по-късно)
};

const root = new Directory(Paths.document, 'scans');

export const scanDir = (id: string) => new Directory(root, id);
export const photosDir = (id: string) => new Directory(scanDir(id), 'photos');
export const videoFile = (id: string) => new File(scanDir(id), 'video.mp4'); // iOS записва MOV; сървърът чете и двата формата
export const motionFile = (id: string) => new File(scanDir(id), 'motion.json'); // положението на телефона по време на записа
// CAD детайлите имат собствен файл с версия: при промяна в начина, по който сървърът ги показва (CAD_DISPLAY),
// приложението ги изтегля наново, вместо да показва стария запазен файл.
export const CAD_DISPLAY = 3;
export const modelFile = (id: string, cad = false) => new File(scanDir(id), cad ? `model-cad${CAD_DISPLAY}.glb` : 'model.glb');
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
    ...(server.report?.real_mm && { longestMm: server.report.real_mm }), // истинският размер (CAD детайл, AI модел)
    ...(server.report?.ai === 'cad' && { method: 'cad' as const }),
  };
  scanDir(scan.id).create({ intermediates: true });
  saveScan(scan);
  return scan;
}

/* Показваното заглавие на сканиране: неговото или датата. */
export const scanTitle = (scan: { title?: string; createdAt: number }) => scan.title || `Сканиране ${dayMonth(scan.createdAt)}`;

// Записите в паметта: дискът се чете веднъж при първото искане, а после само се записва. Иначе всяко връщане към
// началния екран или списъка четеше и разчиташе файла на всяко сканиране (десетки синхронни четения на основната
// нишка, точно по време на анимацията на връщане).
let cache: Map<string, Scan> | null = null;

function loaded(): Map<string, Scan> {
  if (cache) return cache;
  cache = new Map();
  if (!root.exists) return cache;
  for (const d of root.list()) {
    if (!(d instanceof Directory)) continue;
    const f = metaFile(d.name);
    if (!f.exists) continue;
    try {
      const scan = JSON.parse(f.textSync()) as Scan;
      cache.set(scan.id, scan);
    } catch {} // повреден запис: пропуска се, както и преди
  }
  return cache;
}

export function saveScan(scan: Scan) {
  metaFile(scan.id).writeSync(JSON.stringify(scan));
  loaded().set(scan.id, scan);
}

export function getScan(id: string): Scan | null {
  return loaded().get(id) ?? null;
}

export function updateScan(id: string, patch: Partial<Scan>): Scan {
  const scan = { ...getScan(id)!, ...patch };
  saveScan(scan);
  return scan;
}

export function listScans(): Scan[] {
  return [...loaded().values()].sort((a, b) => b.createdAt - a.createdAt);
}

export function deleteScan(id: string) {
  const d = scanDir(id);
  if (d.exists) d.delete();
  loaded().delete(id);
}

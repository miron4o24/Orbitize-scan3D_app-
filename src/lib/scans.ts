import { Directory, File, Paths } from 'expo-file-system';

import type { Report } from './api';

// Scans live on disk only: Paths.document/scans/<id>/{scan.json, photos/, results}.
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
  photoCount: number;
  jobId?: string;
  error?: string;
  longestMm?: number;
  flatBase?: boolean;
  report?: Report;
};

const root = new Directory(Paths.document, 'scans');

export const scanDir = (id: string) => new Directory(root, id);
export const photosDir = (id: string) => new Directory(scanDir(id), 'photos');
export const modelFile = (id: string) => new File(scanDir(id), 'model.glb');
const metaFile = (id: string) => new File(scanDir(id), 'scan.json');

export function createScan(): Scan {
  const scan: Scan = {
    id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    createdAt: Date.now(),
    status: 'capturing',
    photoCount: 0,
  };
  photosDir(scan.id).create({ intermediates: true });
  saveScan(scan);
  return scan;
}

export function saveScan(scan: Scan) {
  metaFile(scan.id).write(JSON.stringify(scan));
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

import { fetch as expoFetch } from 'expo/fetch';
import type { File } from 'expo-file-system';

export const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? '').replace(/\/$/, '');
// shortcut: a token shipped in the app binary can be extracted; fine for a personal backend, use per-user auth before public release.
const API_TOKEN = process.env.EXPO_PUBLIC_API_TOKEN ?? '';

export type JobState =
  | 'uploading'
  | 'queued'
  | 'reconstructing'
  | 'meshing'
  | 'cleaning'
  | 'done'
  | 'failed'
  | 'cancelled';

export type Job = {
  id: string;
  state: JobState;
  received: string[];
  error?: string;
  report?: {
    dimensions_mm: [number, number, number];
    triangles: number;
    watertight: boolean;
  };
};

export type ResultFormat = 'glb' | 'stl' | 'obj' | '3mf';

const headers = (extra?: Record<string, string>) => ({
  Authorization: `Bearer ${API_TOKEN}`,
  ...extra,
});

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (!API_URL) throw new Error('Backend URL not set (EXPO_PUBLIC_API_URL).');
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: headers(body ? { 'Content-Type': 'application/json' } : undefined),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path}: ${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

export const health = () => call<{ ok: boolean }>('GET', '/health');
export const createJob = () => call<Job>('POST', '/jobs');
export const getJob = (id: string) => call<Job>('GET', `/jobs/${id}`);
export const startJob = (id: string, params: { flat_base: boolean }) =>
  call<Job>('POST', `/jobs/${id}/start`, params);
export const cancelJob = (id: string) => call<Job>('DELETE', `/jobs/${id}`);

// Idempotent per file name, so a retry only needs to send what `received` lacks.
export async function uploadPhoto(jobId: string, file: File) {
  const res = await expoFetch(`${API_URL}/jobs/${jobId}/photos/${file.name}`, {
    method: 'PUT',
    headers: headers({ 'Content-Type': 'image/jpeg' }),
    body: file,
  });
  if (!res.ok) throw new Error(`upload ${file.name}: ${res.status} ${await res.text()}`);
}

export const resultUrl = (jobId: string, format: ResultFormat, longestMm?: number) =>
  `${API_URL}/jobs/${jobId}/result.${format}${longestMm ? `?longest_mm=${longestMm}` : ''}`;

export const authHeaders = () => headers();

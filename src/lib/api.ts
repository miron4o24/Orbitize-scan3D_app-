import { fetch as expoFetch } from 'expo/fetch';
import { File, FileMode } from 'expo-file-system';

import { clearSession, sessionToken, type Profile } from './session';

export const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? '').replace(/\/$/, '');
// Requests carry only the signed-in account's own session token. The server's admin token must never be in the app:
// anything in the app binary can be extracted.

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
  video_bytes: number;
  error?: string;
  report?: Report | null;
  step?: string | null; // the tool running now, e.g. "camera solving"
  ahead?: number | null; // while waiting: how many go first, the one being processed among them
};

export type Report = {
  dimensions_mm: [number, number, number]; // at the default 100 mm longest side
  triangles: number;
  watertight: boolean;
  photos_used?: number;
  photos_total?: number;
  seconds?: number;
  object_width?: number; // median share of the frame width the object spanned
  ai?: AiMethod; // made by an AI (Pro) rather than scanned
  object?: string; // what the AI saw, in Bulgarian
  real_mm?: number | null; // the real longest side: Claude's estimate, or a text change's result
  // at the 100 mm size, for the print estimate (lib/printEstimate.ts); older scans: getGeometry
  volume_mm3?: number;
  area_mm2?: number;
  edit?: { prompt: string; changes: string[]; source: string | null }; // made by a text change (Pro)
};

export type Geometry = { dimensions_mm: [number, number, number]; volume_mm3: number; area_mm2: number };

// Pro (backend app/ai.py): the ~5 minute express scan, and the AI models: single (free, on the server: the object's
// AI-cut outline in every frame with measured and AI depth fused within it, the photos as its texture; TripoSR from
// one frame as a fallback), claude and photo (paid services, off until the server has their keys). edit: a text change
// of a model (backend app/edit.py), made from the viewer.
export type AiMethod = 'single' | 'claude' | 'photo';
export type ProMethod = 'express' | 'edit' | AiMethod;
export type Method = 'scan' | ProMethod;

export type ResultFormat = 'glb' | 'stl' | 'obj' | '3mf';

/** What a feature says when the server does not have it yet. */
export const NOT_AVAILABLE = 'Това още не е налично. Опитайте по-късно.';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

// What the user is told when something fails: the message itself when it is written for them (in Bulgarian: the
// app's and the server's own), otherwise a plain sentence for its kind. Technical messages ("Network request
// failed", "SSLHandshakeException…", an older server's English) stay out of sight.
const CYRILLIC = /[Ѐ-ӿ]/;
const OFFLINE = /network|connect|ssl|socket|timeout|timed out|abort|fetch|host|unreachable/i;

export function errorText(e: unknown): string {
  const message = e instanceof Error ? e.message : String(e);
  if (CYRILLIC.test(message)) return message;
  if (e instanceof ApiError) {
    if (e.status === 401) return 'Влезте отново в профила си.';
    if (e.status === 413) return 'Файлът е твърде голям.';
    if (e.status === 429) return 'Твърде много опити. Изчакайте малко и опитайте отново.';
    if (e.status >= 500) return 'Услугата временно не отговаря. Опитайте след малко.';
    return 'Нещо се обърка. Опитайте отново.';
  }
  if (OFFLINE.test(message)) return 'Няма връзка. Проверете интернета и опитайте отново.';
  return 'Нещо се обърка. Опитайте отново.';
}

/** A scan's failure as the server gave it, for the user (an older server's English becomes a plain sentence). */
export const failureText = (error: string | undefined) =>
  error && CYRILLIC.test(error) ? error : 'Обработката не успя. Опитайте отново или заснемете ново видео.';

// Mobile networks and the Funnel relay drop a connection now and then (seen: "SSLHandshakeException: connection
// closed"), so network errors and 5xx answers get a few quiet retries, about a minute in all. Other answers are final.
const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 15_000, 30_000];
const final = (e: unknown) => e instanceof ApiError && e.status < 500 && e.status !== 408 && e.status !== 429;

export async function retrying<T>(work: (attempt: number) => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await work(attempt);
    } catch (e) {
      if (final(e) || attempt === RETRY_DELAYS_MS.length) throw e;
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
    }
  }
}

async function check(res: { ok: boolean; status: number; text(): Promise<string> }, what: string) {
  if (res.ok) return;
  let detail = await res.text();
  try {
    detail = JSON.parse(detail).detail ?? detail; // FastAPI error body
  } catch {}
  // The account's session was ended (signed out elsewhere, account deleted): back to the sign-in screen.
  if (res.status === 401 && sessionToken()) clearSession();
  throw new ApiError(detail ? String(detail) : `${what}: ${res.status}`, res.status);
}

const headers = (extra?: Record<string, string>): Record<string, string> => {
  const token = sessionToken();
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra };
};

/** For images loaded straight from the server (expo-image sends these). */
export const authHeaders = () => headers();

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (!API_URL) throw new Error('Няма връзка. Опитайте отново.');
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: headers(body ? { 'Content-Type': 'application/json' } : undefined),
    body: body ? JSON.stringify(body) : undefined,
  });
  await check(res, `${method} ${path}`);
  return (await res.json()) as T;
}

// accounts: the server has its account database (email and password sign-in); google_client_id: Google sign-in is
// set up, with this web client. Servers from before accounts answer just { ok }.
export type Health = { ok: boolean; accounts?: boolean; database?: string; google_client_id?: string | null };
export const health = () => call<Health>('GET', '/health');
export const createJob = () => call<Job>('POST', '/jobs');
export const getJob = (id: string) => call<Job>('GET', `/jobs/${id}`);
// quality: servers from before quick scans ignore it and process in full detail. method: an AI model (Pro).
export const startJob = (
  id: string,
  params: { flat_base: boolean; video_bytes?: number; quality?: 'quick' | 'detailed'; method?: Method; extend?: string },
) => call<Job>('POST', `/jobs/${id}/start`, params);
export const cancelJob = (id: string) => call<Job>('DELETE', `/jobs/${id}`);

// The last lines of the server's processing log, shown when a scan fails.
export async function getLog(id: string, lines = 40) {
  const res = await fetch(`${API_URL}/jobs/${id}/log?lines=${lines}`, { headers: headers() });
  await check(res, 'log');
  return res.text();
}

// Idempotent per file name, so a retry only needs to send what `received` lacks.
export const uploadPhoto = (jobId: string, file: File) =>
  retrying(async () => {
    const res = await expoFetch(`${API_URL}/jobs/${jobId}/photos/${file.name}`, {
      method: 'PUT',
      headers: headers({ 'Content-Type': 'image/jpeg' }),
      body: file,
    });
    await check(res, `upload ${file.name}`);
  });

// The phone's orientation while filming (src/lib/motion.ts). A server without this endpoint (404) just goes without.
export async function uploadMotion(jobId: string, file: File) {
  const body = await file.text();
  await retrying(async () => {
    const res = await fetch(`${API_URL}/jobs/${jobId}/motion`, {
      method: 'PUT',
      headers: headers({ 'Content-Type': 'application/json' }),
      body,
    });
    if (res.status === 404 || res.status === 405) return;
    await check(res, 'upload motion');
  });
}

const VIDEO_CHUNK_BYTES = 8 * 1024 * 1024;

// Resumes from `from` (the server's video_bytes). Chunks keep memory flat and make a dropped connection cost one chunk.
export async function uploadVideo(jobId: string, file: File, from: number, onProgress: (sent: number) => void) {
  const handle = file.open(FileMode.ReadOnly);
  try {
    let sent = from;
    while (sent < file.size) {
      const next = sent;
      sent = await retrying(async (attempt) => {
        // A chunk whose answer was lost may have arrived anyway: go on from what the server has.
        const at = attempt ? (await getJob(jobId)).video_bytes : next;
        if (at >= file.size) return at;
        handle.offset = at;
        const chunk = handle.readBytesSync(Math.min(VIDEO_CHUNK_BYTES, file.size - at));
        const res = await expoFetch(`${API_URL}/jobs/${jobId}/video?offset=${at}`, {
          method: 'PUT',
          headers: headers({ 'Content-Type': 'application/octet-stream' }),
          body: chunk,
        });
        await check(res, 'upload video');
        return at + chunk.length;
      });
      onProgress(sent);
    }
  } finally {
    handle.close();
  }
}

// GLBs with the photo texture (the viewer decodes it through expo-gl: lib/photoTexture.ts). The server sends the
// texture only on request, as older app builds fail on images inside a GLB. smooth: extra smoothing passes (1-3)
// on a print file, which takes the scan's ripples off.
export const resultUrl = (jobId: string, format: ResultFormat, longestMm?: number, smooth?: number) => {
  const query = [longestMm && `longest_mm=${longestMm}`, format === 'glb' && 'texture=true', smooth && `smooth=${smooth}`]
    .filter(Boolean)
    .join('&');
  return `${API_URL}/jobs/${jobId}/result.${format}${query ? `?${query}` : ''}`;
};

export const downloadResult = (jobId: string, format: ResultFormat, dest: File, longestMm?: number, smooth?: number) =>
  retrying(() =>
    File.downloadFileAsync(resultUrl(jobId, format, longestMm, smooth), dest, { headers: headers(), idempotent: true }),
  );

// Signed links to the model at its real size for AR (lib/ar.ts), which the browser opens without our token: `page`
// (servers from before it send only `url`, the GLB). The server makes the file first, so this takes a few seconds;
// not retried, as each try makes it again.
export async function arLink(jobId: string, longestMm: number) {
  try {
    return await call<{ url: string; page?: string; expires: number }>('POST', `/jobs/${jobId}/ar`, {
      longest_mm: longestMm,
    });
  } catch (e) {
    if (e instanceof ApiError && (e.status === 405 || (e.status === 404 && e.message === 'Not Found')))
      throw new Error(NOT_AVAILABLE);
    throw e;
  }
}

// Volume and surface for the print estimate, for scans whose report is from before they were in it.
export const getGeometry = (jobId: string) => call<Geometry>('GET', `/jobs/${jobId}/geometry`);

// Pro: a new model made from this one by a text change (in English); the server answers the new job. spot: where
// the user tapped on the model ("smooth here", "make this red"), as components/ModelView's Spot.
export const editModel = (jobId: string, prompt: string, longestMm: number, title: string, spot?: readonly number[] | null) =>
  call<Job>('POST', `/jobs/${jobId}/edit`, { prompt, longest_mm: longestMm, title, ...(spot ? { spot } : {}) });

// Pro: a new model made from this one, standing on a plate with a name.
export const nameplateModel = (jobId: string, text: string, shape: 'round' | 'square', longestMm: number, title: string) =>
  call<Job>('POST', `/jobs/${jobId}/nameplate`, { text, shape, longest_mm: longestMm, title });

// Ready for the printer (backend app/printing.py): how it would be laid and cut, and the files.
export type PrintPlan = { parts: number; support_before: number; support_after: number; turned: boolean; size_mm: [number, number, number] };
type BedSize = readonly [number, number, number];
export const printPlan = (jobId: string, longestMm: number, [x, y, z]: BedSize) =>
  call<PrintPlan>('POST', `/jobs/${jobId}/print/plan`, { longest_mm: longestMm, bed_x: x, bed_y: y, bed_z: z });
export const downloadPrintZip = (jobId: string, longestMm: number, [x, y, z]: BedSize, dest: File) =>
  retrying(() =>
    File.downloadFileAsync(`${API_URL}/jobs/${jobId}/print.zip?longest_mm=${longestMm}&bed_x=${x}&bed_y=${y}&bed_z=${z}`, dest, {
      headers: headers(),
      idempotent: true,
    }),
  );

// The spinning video to share: asked for until ready (it takes the server a minute or two), then downloaded.
const SPIN_POLL_MS = 4000;
const SPIN_MOST_MS = 8 * 60 * 1000;
export async function downloadSpin(jobId: string, dest: File) {
  const begun = Date.now();
  while (!(await retrying(() => call<{ ready: boolean }>('POST', `/jobs/${jobId}/spin`))).ready) {
    if (Date.now() - begun > SPIN_MOST_MS) throw new Error('Видеото не се получи. Опитайте отново.');
    await new Promise((resolve) => setTimeout(resolve, SPIN_POLL_MS));
  }
  return retrying(() => File.downloadFileAsync(`${API_URL}/jobs/${jobId}/spin.mp4`, dest, { headers: headers(), idempotent: true }));
}

// "Your model is ready" on this phone (lib/push.ts).
export const savePushToken = (token: string) => call<{ ok: boolean }>('PUT', '/me/push-token', { token });
export const forgetPushToken = (token: string) => call<{ ok: boolean }>('POST', '/me/push-token/forget', { token });

// Pro bought in Google Play: the server checks it with Google and answers the profile with Pro on.
export const confirmGooglePurchase = (token: string) => call<Profile>('POST', '/billing/google', { purchase_token: token });

export const thumbUrl = (jobId: string) => `${API_URL}/jobs/${jobId}/thumb.jpg`;
export const shareUrl = (slug: string) => `${API_URL}/s/${slug}`;

// --- Accounts ---------------------------------------------------------------------------------------------------

export const signInWithIdToken = (idToken: string) =>
  call<{ token: string; user: Profile }>('POST', '/auth/google', { id_token: idToken });
export const register = (email: string, password: string, name: string) =>
  call<{ token: string; user: Profile }>('POST', '/auth/register', { email, password, name });
export const login = (email: string, password: string) =>
  call<{ token: string; user: Profile }>('POST', '/auth/login', { email, password });
export const endSession = () => call<{ ok: boolean }>('POST', '/auth/logout');
export const getMe = () => call<Profile>('GET', '/me');
export const renameMe = (name: string) => call<Profile>('PATCH', '/me', { name });
export const deleteAccount = () => call<{ ok: boolean }>('DELETE', '/me');
export const removeAvatar = () => call<Profile>('DELETE', '/me/avatar');
export const avatarUrl = (version: number) => `${API_URL}/me/avatar.jpg?v=${version}`;

export async function uploadAvatar(file: File) {
  const res = await expoFetch(`${API_URL}/me/avatar`, {
    method: 'PUT',
    headers: headers({ 'Content-Type': 'image/jpeg' }),
    body: file,
  });
  await check(res, 'upload photo');
  return (await res.json()) as Profile;
}

/** A scan saved in the account (any phone). */
export type ServerScan = {
  id: string; // the job id
  title: string;
  created: number; // Unix seconds
  state: JobState;
  error?: string | null;
  report?: Report | null;
  step?: string | null;
  share: string | null; // the public link's id while shared
  published?: number | null; // Unix seconds it went into the community (servers from before it leave this out)
  thumb: boolean;
};

export const listMyScans = () => call<ServerScan[]>('GET', '/me/scans');
export const renameScan = (jobId: string, title: string) => call<ServerScan>('PATCH', `/me/scans/${jobId}`, { title });
export const shareScan = (jobId: string) => call<{ share: string }>('POST', `/me/scans/${jobId}/share`);
export const unshareScan = (jobId: string) => call<{ share: null }>('DELETE', `/me/scans/${jobId}/share`);
export const claimScans = (jobIds: string[]) => call<{ claimed: string[] }>('POST', '/me/scans/claim', { job_ids: jobIds });

// --- Community --------------------------------------------------------------------------------------------------
// Finished scans their owners posted for every signed-in user (backend app/community.py).

// level: from their points (servers from before points leave it out).
export type Author = { id: string; name: string; has_avatar: boolean; google_picture: string | null; level?: number };

export type Post = {
  id: string; // the job id
  title: string;
  published: number; // Unix seconds
  longest_mm: number; // the real size its author gave
  likes: number;
  liked: boolean; // by me
  mine: boolean;
  dimensions_mm: [number, number, number] | null; // at 100 mm longest side
  triangles: number | null;
  author: Author;
};

export type PostSort = 'new' | 'top';

// A server from before the community answers FastAPI's own "Not Found" for these paths.
async function community<T>(method: string, path: string, body?: unknown): Promise<T> {
  try {
    return await call<T>(method, path, body);
  } catch (e) {
    if (e instanceof ApiError && (e.status === 405 || (e.status === 404 && e.message === 'Not Found')))
      throw new ApiError(NOT_AVAILABLE, e.status);
    throw e;
  }
}

export const communityFeed = (sort: PostSort, offset = 0) =>
  community<{ posts: Post[]; next: number | null }>('GET', `/community?sort=${sort}&offset=${offset}`);
export const getPost = (id: string) => community<Post>('GET', `/community/${id}`);
export const publishScan = (jobId: string, title: string, longestMm: number) =>
  community<Post>('POST', `/me/scans/${jobId}/publish`, { title, longest_mm: longestMm });
export const unpublishScan = (jobId: string) => community<{ published: null }>('DELETE', `/me/scans/${jobId}/publish`);
export const likePost = (id: string, on: boolean) =>
  community<{ likes: number; liked: boolean }>(on ? 'PUT' : 'DELETE', `/community/${id}/like`);
export const reportPost = (id: string, reason: string) => community<{ ok: boolean }>('POST', `/community/${id}/report`, { reason });
export const postThumbUrl = (id: string) => `${API_URL}/community/${id}/thumb.jpg`;
export const authorPhotoUrl = (userId: string) => `${API_URL}/community/authors/${userId}/avatar.jpg`;

// Photos of a community model printed in real life.
export type PrintPhoto = { id: string; name: string; created: number; mine: boolean; can_remove: boolean };
export const listPrintPhotos = (id: string) => community<PrintPhoto[]>('GET', `/community/${id}/prints`);
export const printPhotoUrl = (photoId: string) => `${API_URL}/community/prints/${photoId}.jpg`;
export const removePrintPhoto = (photoId: string) => community<{ ok: boolean }>('DELETE', `/community/prints/${photoId}`);
export const reportPrintPhoto = (photoId: string) => community<{ ok: boolean }>('POST', `/community/prints/${photoId}/report`);
export async function uploadPrintPhoto(id: string, file: File) {
  const res = await expoFetch(`${API_URL}/community/${id}/prints`, {
    method: 'PUT',
    headers: headers({ 'Content-Type': 'image/jpeg' }),
    body: file,
  });
  await check(res, 'upload photo');
}

export const downloadPost = (id: string, dest: File) =>
  retrying(() => File.downloadFileAsync(`${API_URL}/community/${id}/model.glb`, dest, { headers: headers(), idempotent: true }));

// --- Stress test ------------------------------------------------------------------------------------------------
// The server works each kind of test out once per model (backend app/stress.py): at a ref_mm longest side and for
// 1 kg (own weight: at 1 g/cm³). lib/stress.ts scales that to the weight, height, material and size picked.

export type StressCase = 'top' | 'hang' | 'side' | 'drop';
export type Vec3 = [number, number, number];

export type StressSummary = {
  case: StressCase;
  ref_mm: number;
  max_mpa: number; // the highest stress, MPa
  weak: Vec3; // where it is: model coordinates (mm, y up) at ref_mm
  volume_cm3: number; // at ref_mm
  sag_m?: number; // drop: the own weight's mean sag at 1 GPa and 1 g/cm³
  load_at?: Vec3; // the others: where the force acts, and which way
  load_dir?: Vec3;
  elements: number;
  pitch_mm: number;
  dropped: number; // share of the model too thin to be held, so not tested
};

/** Own scans by their job id; community posts by their id. */
export type StressSource = 'scan' | 'post';
const stressPath = (id: string, source: StressSource, kind: StressCase) =>
  `${source === 'post' ? `/community/${id}` : `/jobs/${id}`}/stress/${kind}`;

// The first request for a kind works it out on the server (up to a minute). A dropped connection is tried again:
// the server finishes it anyway, and the next try gets it at once.
export async function stressSummary(id: string, source: StressSource, kind: StressCase) {
  try {
    return await retrying(() => call<StressSummary>('GET', stressPath(id, source, kind)));
  } catch (e) {
    if (e instanceof ApiError && (e.status === 405 || (e.status === 404 && e.message === 'Not Found')))
      throw new ApiError(NOT_AVAILABLE, e.status);
    throw e;
  }
}

// The model with its stress per vertex (the "_stress" attribute, MPa at the summary's size and load).
export const downloadStressModel = (id: string, source: StressSource, kind: StressCase, dest: File) =>
  retrying(() => File.downloadFileAsync(`${API_URL}${stressPath(id, source, kind)}/model.glb`, dest, { headers: headers(), idempotent: true }));

// --- Points, levels, badges, weekly challenges (backend app/progress.py) ----------------------------------------

export type Badge = {
  id: string;
  title: string;
  description: string;
  icon: string; // an Ionicons name
  target: number;
  progress: number;
  unlocked: boolean;
  unlocked_at: number | null; // Unix seconds
};

export type Challenge = { id: string; title: string; icon: string; target: number; progress: number; points: number; done: boolean };

export type Level = { number: number; name: string; from: number; to: number }; // points where it starts, the next starts

export type Progress = {
  points: number;
  points_week: number;
  level: Level;
  badges: Badge[];
  challenges: { week: string; ends: number; items: Challenge[] }; // ends: Unix seconds
  recent: { kind: string; label: string; points: number; created: number }[];
};

export const getProgress = () => call<Progress>('GET', '/me/progress');

// --- The owner's Pro page: who is Pro (backend app/accounts.py) ------------------------------------------------

export type OwnerUser = { id: string; email: string; name: string; created: number; pro: boolean; owner: boolean };

export const ownerUsers = (q: string) => call<OwnerUser[]>('GET', `/owner/users?q=${encodeURIComponent(q)}`);
export const setPro = (userId: string, pro: boolean) => call<{ id: string; pro: boolean }>('PUT', `/owner/users/${userId}/pro`, { pro });

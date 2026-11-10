import { fetch as expoFetch } from 'expo/fetch';
import { File, FileMode } from 'expo-file-system';

import { clearSession, sessionToken, type Profile } from './session';

export const API_URL = (process.env.EXPO_PUBLIC_API_URL ?? '').replace(/\/$/, '');
// Заявките носят само сесийния ключ на влезлия профил. Администраторският ключ на сървъра никога не трябва да е в
// приложението: всичко в инсталационния файл може да бъде извлечено.

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
  step?: string | null; // инструментът, който работи в момента, например "camera solving"
  ahead?: number | null; // докато чака: колко са преди него, включително обработваната в момента
};

export type Report = {
  dimensions_mm: [number, number, number]; // при подразбиращата се най-дълга страна от 100 мм
  triangles: number;
  watertight: boolean;
  photos_used?: number;
  photos_total?: number;
  seconds?: number;
  object_width?: number; // медианният дял от ширината на кадъра, който заема предметът
  ai?: AiMethod | 'cad'; // създаден от AI (Pro) или в инженерния режим, а не сканиран
  object?: string; // какво е видял AI, на български
  real_mm?: number | null; // реалната най-дълга страна: оценка на AI или резултат от текстова промяна
  // при размер 100 мм, за оценката на печата (lib/printEstimate.ts); за по-стари сканирания: getGeometry
  volume_mm3?: number;
  area_mm2?: number;
  edit?: { prompt: string; changes: string[]; source: string | null }; // създаден чрез текстова промяна (Pro)
  cad?: CadReport; // инженерният режим: детайлът, построен от измерените размери
};

export type Geometry = { dimensions_mm: [number, number, number]; volume_mm3: number; area_mm2: number };

// Pro (сървър app/ai.py): експресното сканиране за около 5 минути и AI моделите: single (безплатен, на сървъра:
// очертанията на предмета, изрязани с AI във всеки кадър, в които се сливат измерената и AI дълбочината, със снимките
// като текстура; TripoSR от един кадър е резервен вариант), claude и photo (платени услуги, изключени, докато
// сървърът няма ключове за тях). edit: текстова промяна на модел (сървър app/edit.py), направена от екрана за
// преглед.
export type AiMethod = 'single' | 'claude' | 'photo';
export type ProMethod = 'express' | 'edit' | AiMethod;
export type Method = 'scan' | 'cad' | ProMethod; // cad: инженерният режим (безплатен, не е сканиране)

export type ResultFormat = 'glb' | 'stl' | 'obj' | '3mf';

/* Какво казва дадена функция, когато сървърът още я няма. */
export const NOT_AVAILABLE = 'Това още не е налично. Опитайте по-късно.';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

// Какво се показва на потребителя при неуспех: самото съобщение, когато е написано за него (на български: от
// приложението и от сървъра), иначе обикновено изречение за вида грешка. Техническите съобщения ("Network request
// failed", "SSLHandshakeException…", английските съобщения на по-стар сървър) остават скрити.
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

/**
 * Неуспехът на сканиране, както го е дал сървърът, за потребителя (английското съобщение на по-стар сървър става
 * обикновено изречение).
 */
export const failureText = (error: string | undefined) =>
  error && CYRILLIC.test(error) ? error : 'Обработката не успя. Опитайте отново или заснемете ново видео.';

// Мобилните мрежи и препращането през Funnel понякога прекъсват връзката (наблюдавано: "SSLHandshakeException:
// connection closed"), затова мрежовите грешки и отговорите 5xx получават няколко тихи повторения, общо около минута.
// Другите отговори са окончателни.
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
  let detail: unknown = await res.text();
  try {
    detail = JSON.parse(detail as string).detail ?? detail; // тяло на грешка от FastAPI
  } catch {}
  // Проверката на FastAPI (422) дава списък. Непознат режим значи, че сървърът е отпреди него.
  if (Array.isArray(detail)) {
    const items = detail as { type?: string; loc?: unknown[]; msg?: string }[];
    detail = items.some((d) => d.type === 'literal_error' && d.loc?.includes('method'))
      ? 'Сървърът още не поддържа този режим: трябва да се обнови.'
      : items.map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`).join('; ');
  }
  // Сесията на профила е приключила (изход от другаде, изтрит профил): обратно към екрана за вход.
  if (res.status === 401 && sessionToken()) clearSession();
  throw new ApiError(detail ? String(detail) : `${what}: ${res.status}`, res.status);
}

const headers = (extra?: Record<string, string>): Record<string, string> => {
  const token = sessionToken();
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra };
};

/* За изображения, зареждани директно от сървъра (expo-image изпраща тези заглавки). */
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

// accounts: сървърът има база данни с профили (вход с имейл и парола); google_client_id: входът с Google е настроен с
// този уеб клиент. Сървъри отпреди профилите отговарят само с { ok }.
export type Health = { ok: boolean; accounts?: boolean; database?: string; google_client_id?: string | null };
export const health = () => call<Health>('GET', '/health');
export const createJob = () => call<Job>('POST', '/jobs');
export const getJob = (id: string) => call<Job>('GET', `/jobs/${id}`);
// quality: сървъри отпреди бързите сканирания го пренебрегват и обработват с пълни детайли. method: AI модел (Pro).
export const startJob = (
  id: string,
  params: { flat_base: boolean; video_bytes?: number; quality?: 'quick' | 'detailed'; method?: Method; extend?: string },
) => call<Job>('POST', `/jobs/${id}/start`, params);
export const cancelJob = (id: string) => call<Job>('DELETE', `/jobs/${id}`);

// Последните редове от дневника на обработката на сървъра, показвани при неуспешно сканиране.
export async function getLog(id: string, lines = 40) {
  const res = await fetch(`${API_URL}/jobs/${id}/log?lines=${lines}`, { headers: headers() });
  await check(res, 'log');
  return res.text();
}

// Повторението е безопасно за всяко име на файл, така че повторен опит изпраща само това, което липсва в `received`.
export const uploadPhoto = (jobId: string, file: File) =>
  retrying(async () => {
    const res = await expoFetch(`${API_URL}/jobs/${jobId}/photos/${file.name}`, {
      method: 'PUT',
      headers: headers({ 'Content-Type': 'image/jpeg' }),
      body: file,
    });
    await check(res, `upload ${file.name}`);
  });

// Положението на телефона по време на записа (src/lib/motion.ts). Сървър без тази крайна точка (404) просто минава
// без него.
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

// Продължава от `from` (video_bytes на сървъра). Частите пазят паметта постоянна, а прекъсната връзка струва само
// една част.
export async function uploadVideo(jobId: string, file: File, from: number, onProgress: (sent: number) => void) {
  const handle = file.open(FileMode.ReadOnly);
  try {
    let sent = from;
    while (sent < file.size) {
      const next = sent;
      sent = await retrying(async (attempt) => {
        // Част, чийто отговор се е загубил, може все пак да е пристигнала: продължаваме от това, което сървърът има.
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

// GLB файлове с текстура от снимките (екранът за преглед я декодира през expo-gl: lib/photoTexture.ts). Сървърът
// изпраща текстурата само при заявка, защото по-стари версии на приложението се провалят при изображения в GLB.
// smooth: допълнителни заглаждания (1-3) на файла за печат, които махат вълните от сканирането.
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

// Подписани линкове към модела в реален размер за AR (lib/ar.ts), които браузърът отваря без нашия ключ: `page`
// (сървъри отпреди него изпращат само `url`, тоест GLB). Сървърът първо създава файла, затова това отнема няколко
// секунди; не се повтаря, защото всеки опит го създава наново.
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

// Обем и повърхност за оценката на печата, за сканирания с отчет отпреди те да бъдат включени в него.
export const getGeometry = (jobId: string) => call<Geometry>('GET', `/jobs/${jobId}/geometry`);

// Pro: нов модел, създаден от този чрез текстова промяна (на английски); сървърът връща новата задача. spot: мястото,
// което потребителят е докоснал върху модела ("smooth here", "make this red"), като Spot от components/ModelView.
export const editModel = (jobId: string, prompt: string, longestMm: number, title: string, spot?: readonly number[] | null) =>
  call<Job>('POST', `/jobs/${jobId}/edit`, { prompt, longest_mm: longestMm, title, ...(spot ? { spot } : {}) });

// Pro: нов модел, създаден от този, стоящ върху плочка с име.
export const nameplateModel = (jobId: string, text: string, shape: 'round' | 'square', longestMm: number, title: string) =>
  call<Job>('POST', `/jobs/${jobId}/nameplate`, { text, shape, longest_mm: longestMm, title });

// Готово за принтера (сървър app/printing.py): как би се разположил и разрязал моделът и файловете.
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

// Видеото със завъртане за споделяне: заявява се, докато стане готово (сървърът го прави за минута-две), после се
// изтегля.
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

// "Вашият модел е готов" на този телефон (lib/push.ts).
export const savePushToken = (token: string) => call<{ ok: boolean }>('PUT', '/me/push-token', { token });
export const forgetPushToken = (token: string) => call<{ ok: boolean }>('POST', '/me/push-token/forget', { token });

// Pro, купен от Google Play: сървърът проверява покупката при Google и връща профила с включен Pro.
export const confirmGooglePurchase = (token: string) => call<Profile>('POST', '/billing/google', { purchase_token: token });

export const thumbUrl = (jobId: string) => `${API_URL}/jobs/${jobId}/thumb.jpg`;
export const shareUrl = (slug: string) => `${API_URL}/s/${slug}`;

// --- Профили -------------------------------------------------------------------------------------------------------

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

/* Сканиране, запазено в профила (на всеки телефон). */
export type ServerScan = {
  id: string; // номерът на задачата
  title: string;
  created: number; // секунди по Unix
  state: JobState;
  error?: string | null;
  report?: Report | null;
  step?: string | null;
  share: string | null; // номерът на публичния линк, докато е споделено
  published?: number | null; // секунди по Unix, откакто е в общността (сървъри отпреди нея го пропускат)
  thumb: boolean;
};

export const listMyScans = () => call<ServerScan[]>('GET', '/me/scans');
export const renameScan = (jobId: string, title: string) => call<ServerScan>('PATCH', `/me/scans/${jobId}`, { title });
export const shareScan = (jobId: string) => call<{ share: string }>('POST', `/me/scans/${jobId}/share`);
export const unshareScan = (jobId: string) => call<{ share: null }>('DELETE', `/me/scans/${jobId}/share`);
export const claimScans = (jobIds: string[]) => call<{ claimed: string[] }>('POST', '/me/scans/claim', { job_ids: jobIds });

// --- Общност --- Завършени сканирания, публикувани от собствениците им за всички влезли потребители (сървър
// app/community.py).

// level: от точките им (сървъри отпреди точките го пропускат).
export type Author = { id: string; name: string; has_avatar: boolean; google_picture: string | null; level?: number };

export type Post = {
  id: string; // номерът на задачата
  title: string;
  published: number; // секунди по Unix
  longest_mm: number; // реалният размер, зададен от автора
  likes: number;
  liked: boolean; // by me
  mine: boolean;
  dimensions_mm: [number, number, number] | null; // при най-дълга страна от 100 мм
  triangles: number | null;
  author: Author;
};

export type PostSort = 'new' | 'top';

// Сървър отпреди общността отговаря за тези пътища със стандартното "Not Found" на FastAPI.
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

// Снимки на модел от общността, отпечатан в реалността.
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

// --- Тест за здравина --- Сървърът изчислява всеки вид тест веднъж за модел (сървър app/stress.py): при най-дълга
// страна ref_mm и за 1 кг (собствено тегло: при 1 г/см³). lib/stress.ts го мащабира към избраните тегло, височина,
// материал и размер.

export type StressCase = 'top' | 'hang' | 'side' | 'drop';
export type Vec3 = [number, number, number];

export type StressSummary = {
  case: StressCase;
  ref_mm: number;
  max_mpa: number; // най-високото напрежение, MPa
  weak: Vec3; // къде е то: координати в модела (мм, y нагоре) при ref_mm
  volume_cm3: number; // при ref_mm
  sag_m?: number; // drop: средното провисване от собственото тегло при 1 GPa и 1 г/см³
  load_at?: Vec3; // останалите: къде действа силата и в каква посока
  load_dir?: Vec3;
  elements: number;
  pitch_mm: number;
  dropped: number; // дял от модела, твърде тънък, за да се закрепи, затова не е тестван
};

/* Собствените сканирания по номера на задачата; публикациите в общността по техния номер. */
export type StressSource = 'scan' | 'post';
const stressPath = (id: string, source: StressSource, kind: StressCase) =>
  `${source === 'post' ? `/community/${id}` : `/jobs/${id}`}/stress/${kind}`;

// Първата заявка за даден вид го изчислява на сървъра (до минута). При прекъсната връзка се опитва отново: сървърът
// така или иначе го довършва, а следващият опит го получава веднага.
export async function stressSummary(id: string, source: StressSource, kind: StressCase) {
  try {
    return await retrying(() => call<StressSummary>('GET', stressPath(id, source, kind)));
  } catch (e) {
    if (e instanceof ApiError && (e.status === 405 || (e.status === 404 && e.message === 'Not Found')))
      throw new ApiError(NOT_AVAILABLE, e.status);
    throw e;
  }
}

// Моделът с напрежението за всеки връх (атрибут "_stress", MPa при размера и натоварването от обобщението).
export const downloadStressModel = (id: string, source: StressSource, kind: StressCase, dest: File) =>
  retrying(() => File.downloadFileAsync(`${API_URL}${stressPath(id, source, kind)}/model.glb`, dest, { headers: headers(), idempotent: true }));

// --- Точки, нива, значки, седмични предизвикателства (сървър app/progress.py)
// ---------------------------------------

export type Badge = {
  id: string;
  title: string;
  description: string;
  icon: string; // име на икона от Ionicons
  target: number;
  progress: number;
  unlocked: boolean;
  unlocked_at: number | null; // секунди по Unix
};

export type Challenge = { id: string; title: string; icon: string; target: number; progress: number; points: number; done: boolean };

export type Level = { number: number; name: string; from: number; to: number }; // точките, от които започва това ниво и следващото

export type Progress = {
  points: number;
  points_week: number;
  level: Level;
  badges: Badge[];
  challenges: { week: string; ends: number; items: Challenge[] }; // ends: секунди по Unix
  recent: { kind: string; label: string; points: number; created: number }[];
};

export const getProgress = () => call<Progress>('GET', '/me/progress');

// --- Страницата на собственика за Pro: кой е с Pro (сървър app/accounts.py)
// -----------------------------------------

export type OwnerUser = { id: string; email: string; name: string; created: number; pro: boolean; owner: boolean };

export const ownerUsers = (q: string) => call<OwnerUser[]>('GET', `/owner/users?q=${encodeURIComponent(q)}`);
export const setPro = (userId: string, pro: boolean) => call<{ id: string; pro: boolean }>('PUT', `/owner/users/${userId}/pro`, { pro });

// --- Инженерният режим: мерене по указания (сървър app/cad) ---------------------------------------------------------
// Снимките (top.jpg, side.jpg, по желание end.jpg) се качват с uploadPhoto; сървърът ги чете и предлага описание на
// детайла, в което всеки размер е въпрос с оценка от снимките. Моделът се строи само от числата.

export type CadView = 'top' | 'side' | 'end';
export type CadDimKind = 'length' | 'position' | 'diameter' | 'radius';
export type CadDim = {
  v: number | null;
  est: number | null; // оценката от снимките (null: няма, трябва да се измери)
  src: 'measured' | 'photo' | 'skipped';
  name: string;
  ask: string;
  hint: string;
  kind: CadDimKind;
  optional: boolean;
  hidden?: boolean;
};
export type CadFeatureType = 'hole' | 'rect' | 'slot' | 'fillet' | 'chamfer';
export type CadFeature = { id: string; t: CadFeatureType; depth?: string; face?: string; thread?: string };
export type CadDesign = {
  kind: 'block' | 'turned';
  dims: Record<string, CadDim>;
  features: CadFeature[];
  notes: string[];
  flips?: Partial<Record<'side' | 'end', [boolean, boolean]>>;
  [key: string]: unknown; // нивата и изгледите: приложението само ги връща на сървъра
};
export type CadQuestion = {
  id: string;
  name: string;
  ask: string;
  hint: string;
  est: number | null;
  value: number | null;
  src: CadDim['src'];
  optional: boolean;
  kind: CadDimKind;
};
export type CadItem = { id: string; text: string; view?: string | null; kind: 'body' | 'view' | CadFeatureType };
export type CadState = {
  design: CadDesign;
  items: CadItem[];
  questions: CadQuestion[];
  notes: string[];
  warnings: string[];
  views: CadView[];
  stamp: string; // за картинките: нова стойност при всяка промяна
};
export type CadFace = 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right';
export type CadEdges = 'vertical' | 'top' | 'bottom' | 'all' | 'ends';
export type CadAdd = {
  t: CadFeatureType;
  face?: CadFace;
  through?: boolean;
  centered?: [boolean, boolean];
  edges?: CadEdges;
  thread?: string;
};
export type CadOp =
  | { values: Record<string, number | null> }
  | { remove: string }
  | { add: CadAdd }
  | { through: [string, boolean] }
  | { flip: ['side' | 'end', 0 | 1] };
export type CadReport = {
  size_mm: [number, number, number];
  volume_mm3: number;
  kind: 'block' | 'turned';
  measured: number;
  estimated: number;
  estimated_names: string[];
  elements: string[];
};
export type CadFile = 'part.step' | 'part.py' | 'part.stl' | 'part.glb' | 'drawing.png';

export const cadRead = (jobId: string, L: number, W: number, H: number) =>
  call<CadState>('POST', `/jobs/${jobId}/cad/read`, { L, W, H });
export const cadEdit = (jobId: string, design: CadDesign, op: CadOp) =>
  call<CadState>('POST', `/jobs/${jobId}/cad/edit`, { design, op });
export const cadBuild = (jobId: string, design: CadDesign) => call<Job>('POST', `/jobs/${jobId}/cad/build`, { design });
export const cadQuestionUrl = (jobId: string, dim: string, stamp: string) => `${API_URL}/jobs/${jobId}/cad/question/${dim}.png?s=${stamp}`;
export const cadPhotoUrl = (jobId: string, view: CadView, stamp: string) => `${API_URL}/jobs/${jobId}/cad/photo/${view}.jpg?s=${stamp}`;
export const downloadCadPreview = (jobId: string, dest: File) =>
  retrying(() => File.downloadFileAsync(`${API_URL}/jobs/${jobId}/cad/preview.glb`, dest, { headers: headers(), idempotent: true }));
export const downloadCadFile = (jobId: string, name: CadFile, dest: File) =>
  retrying(() => File.downloadFileAsync(`${API_URL}/jobs/${jobId}/cad/file/${name}`, dest, { headers: headers(), idempotent: true }));

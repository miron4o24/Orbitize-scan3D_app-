import { claimScans, endSession, getMe, health, login, register, signInWithIdToken } from './api';
import { stopPush } from './push';
import { listScans, updateScan } from './scans';
import { clearSession, currentUserId, signedIn, updateProfile } from './session';

// Два начина за влизане, един профил на имейл: имейл и парола (без код за потвърждение) или Google (телефонът
// получава ID токен за уеб клиента, който сървърът посочва в /health; сървърът го проверява). И в двата случая
// сървърът отговаря със собствен сесиен ключ. Вход с Google с адреса на имейл профил отваря този профил.

type GoogleModule = typeof import('@react-native-google-signin/google-signin');

// Зарежда се при първа употреба: Expo Go няма вграден вход с Google и импортирането му там спира цялото приложение.
let google: GoogleModule | null | undefined;
function loadGoogle(): GoogleModule | null {
  if (google === undefined) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports -- must load lazily, see above
      google = require('@react-native-google-signin/google-signin') as GoogleModule;
    } catch {
      google = null;
    }
  }
  return google;
}

export async function signInWithPassword(email: string, password: string) {
  const { token, user } = await login(email.trim(), password);
  await signedIn(token, user);
  await adoptLocalScans(user.id);
}

export async function createAccount(email: string, password: string, name: string) {
  const { token, user } = await register(email.trim(), password, name.trim());
  await signedIn(token, user);
  await adoptLocalScans(user.id);
}

/* Влиза в профила; false, когато потребителят затвори избора на профил в Google. */
export async function signInWithGoogle(): Promise<boolean> {
  const h = await health();
  if (!h.google_client_id) throw new Error('Използвайте имейл и парола.');
  const g = loadGoogle();
  if (!g) throw new Error('Използвайте имейл и парола.');
  const { GoogleSignin, isSuccessResponse } = g;
  GoogleSignin.configure({ webClientId: h.google_client_id });
  let idToken: string | null;
  try {
    await GoogleSignin.hasPlayServices({ showPlayServicesUpdateDialog: true });
    const res = await GoogleSignin.signIn();
    if (!isSuccessResponse(res)) return false;
    idToken = res.data.idToken;
  } catch (e) {
    throw new Error(googleProblem(g, e));
  }
  if (!idToken) throw new Error('Google не върна данни за вход. Опитайте отново.');
  const { token, user } = await signInWithIdToken(idToken);
  await signedIn(token, user);
  await adoptLocalScans(user.id);
  return true;
}

function googleProblem({ isErrorWithCode, statusCodes }: GoogleModule, e: unknown): string {
  if (isErrorWithCode(e)) {
    if (e.code === statusCodes.IN_PROGRESS) return 'Входът вече е отворен.';
    if (e.code === statusCodes.PLAY_SERVICES_NOT_AVAILABLE) return 'Входът с Google не работи на този телефон. Използвайте имейл и парола.';
    // DEVELOPER_ERROR: Google не познава тази компилация на приложението (липсва клиент за Android с нейния SHA-1).
    if (e.code === '10' || /DEVELOPER_ERROR/.test(e.message))
      return 'Входът с Google не успя. Използвайте имейл и парола.';
  }
  return 'Входът с Google не успя. Опитайте отново.';
}

/* Сканиранията, направени на този телефон преди влизане, се присъединяват към профила тук и на сървъра. */
async function adoptLocalScans(userId: string) {
  const mine = listScans().filter((s) => !s.userId);
  const jobIds = mine.map((s) => s.jobId).filter((j): j is string => !!j);
  if (jobIds.length) await claimScans(jobIds).catch(() => {}); // така или иначе остават на този телефон
  for (const s of mine) updateScan(s.id, { userId });
  if (jobIds.length) await refreshProfile();
}

export async function refreshProfile() {
  if (!currentUserId()) return;
  updateProfile(await getMe());
}

/* Приключва сесията тук и на сървъра. Сканиранията остават на сървъра и се връщат при следващото влизане. */
export async function signOut() {
  await stopPush();
  await endSession().catch(() => {}); // без връзка: сесията на сървъра просто остава неизползвана
  await loadGoogle()?.GoogleSignin.signOut().catch(() => {});
  await clearSession();
}

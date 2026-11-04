import { claimScans, endSession, getMe, health, login, register, signInWithIdToken } from './api';
import { stopPush } from './push';
import { listScans, updateScan } from './scans';
import { clearSession, currentUserId, signedIn, updateProfile } from './session';

// Two ways in, one account per email: email and password (no verification code), or Google (the phone gets an ID
// token for the web client the server names in /health; the server checks it). Either way the server answers with
// its own session token. A Google sign-in with the address of an email account opens that account.

type GoogleModule = typeof import('@react-native-google-signin/google-signin');

// Loaded on first use: Expo Go has no Google sign-in inside, and importing it there stops the whole app.
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

/** Signs in; false when the user closed Google's account picker. */
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
    // DEVELOPER_ERROR: Google does not know this app build (Android client with its SHA-1 missing).
    if (e.code === '10' || /DEVELOPER_ERROR/.test(e.message))
      return 'Входът с Google не успя. Използвайте имейл и парола.';
  }
  return 'Входът с Google не успя. Опитайте отново.';
}

/** Scans made on this phone before signing in join the account, here and on the server. */
async function adoptLocalScans(userId: string) {
  const mine = listScans().filter((s) => !s.userId);
  const jobIds = mine.map((s) => s.jobId).filter((j): j is string => !!j);
  if (jobIds.length) await claimScans(jobIds).catch(() => {}); // they stay on this phone either way
  for (const s of mine) updateScan(s.id, { userId });
  if (jobIds.length) await refreshProfile();
}

export async function refreshProfile() {
  if (!currentUserId()) return;
  updateProfile(await getMe());
}

/** Ends the session here and on the server. Scans stay on the server and come back at the next sign-in. */
export async function signOut() {
  await stopPush();
  await endSession().catch(() => {}); // offline: the server session just stays unused
  await loadGoogle()?.GoogleSignin.signOut().catch(() => {});
  await clearSession();
}

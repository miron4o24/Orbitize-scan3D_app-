import { File, Paths } from 'expo-file-system';
import { Platform } from 'react-native';

import SceneViewer from '../../modules/scene-viewer';
import { arLink, retrying } from './api';

// AR inside the app (modules/scene-viewer ArActivity: ARCore + SceneView): the camera finds a floor or table and the
// model stands on it with a shadow, at its real size. The server makes the file for it (backend app/ar.py: metres,
// base at the origin, at most 60k triangles). Android only, and only on phones Google's ARCore supports.
export const arAvailable = Platform.OS === 'android' && SceneViewer != null;

export type ArSupport = 'supported' | 'unsupported' | 'unknown';
let support: Promise<ArSupport> | null = null;

/** Whether this phone can do AR at all (asked once per app start; "unknown" when the check fails). */
export function arSupport(): Promise<ArSupport> {
  if (!SceneViewer) return Promise.resolve('unsupported');
  support ??= SceneViewer.arSupport().catch(() => 'unknown' as const);
  return support;
}

export async function viewInRoom(jobId: string, longestMm: number, title: string, realSize: boolean) {
  if (!SceneViewer) throw new Error('„Виж в стаята“ работи само на Android.');
  if ((await arSupport()) === 'unsupported')
    throw new Error('Този телефон не поддържа „Виж в стаята“.');
  // "unlit" in the name: files kept from before the server's unlit AR files (backend app/mesh.py UNLIT) are not used.
  const file = new File(Paths.cache, `ar-unlit-${jobId}-${Math.round(longestMm * 10)}.glb`);
  if (!file.exists) {
    const { url } = await arLink(jobId, longestMm);
    // identity: the raw file. The server gzips for clients that ask, and the GLB must be stored unzipped.
    await retrying(() => File.downloadFileAsync(url, file, { headers: { 'Accept-Encoding': 'identity' }, idempotent: true }));
  }
  await SceneViewer.openAr(decodeURIComponent(file.uri.replace(/^file:\/\//, '')), title, realSize);
}

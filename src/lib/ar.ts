import { File, Paths } from 'expo-file-system';
import { Platform } from 'react-native';

import SceneViewer from '../../modules/scene-viewer';
import { arLink, retrying } from './api';

// AR вътре в приложението (modules/scene-viewer ArActivity: ARCore + SceneView): камерата намира под или маса и
// моделът застава върху тях със сянка, в реалния си размер. Сървърът създава файла за това (сървър app/ar.py: метри,
// основа в началото на координатите, най-много 60 хил. триъгълника). Само за Android и само на телефони, които ARCore
// на Google поддържа.
export const arAvailable = Platform.OS === 'android' && SceneViewer != null;

export type ArSupport = 'supported' | 'unsupported' | 'unknown';
let support: Promise<ArSupport> | null = null;

/**
 * Дали телефонът изобщо може да показва AR (пита се веднъж при всяко стартиране; "unknown", когато проверката се
 * провали).
 */
export function arSupport(): Promise<ArSupport> {
  if (!SceneViewer) return Promise.resolve('unsupported');
  support ??= SceneViewer.arSupport().catch(() => 'unknown' as const);
  return support;
}

export async function viewInRoom(jobId: string, longestMm: number, title: string, realSize: boolean) {
  if (!SceneViewer) throw new Error('„Виж в стаята“ работи само на Android.');
  if ((await arSupport()) === 'unsupported')
    throw new Error('Този телефон не поддържа „Виж в стаята“.');
  // "unlit" в името: файлове, запазени отпреди неосветените AR файлове на сървъра (сървър app/mesh.py UNLIT), не се
  // използват.
  const file = new File(Paths.cache, `ar-unlit-${jobId}-${Math.round(longestMm * 10)}.glb`);
  if (!file.exists) {
    const { url } = await arLink(jobId, longestMm);
    // identity: суровият файл. Сървърът компресира с gzip за клиенти, които го искат, а GLB трябва да се пази
    // некомпресиран.
    await retrying(() => File.downloadFileAsync(url, file, { headers: { 'Accept-Encoding': 'identity' }, idempotent: true }));
  }
  await SceneViewer.openAr(decodeURIComponent(file.uri.replace(/^file:\/\//, '')), title, realSize);
}

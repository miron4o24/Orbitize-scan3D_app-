import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';

import type { Method } from './api';
import { createScan, deleteScan, updateScan, videoFile } from './scans';
import { currentUserId } from './session';

// Видео с обиколка, заснето по-рано (с камерата на телефона), а не в приложението. Сървърът го обработва като запис
// от приложението, но без дневник на положението за съвсем едноцветни предмети. AI моделите на Pro приемат всяко
// видео.

const MAX_BYTES = 600 * 1024 * 1024; // MAX_VIDEO_MB на сървъра
const MIN_SECONDS = 10;

export class ImportProblem extends Error {}

/* Дава на потребителя да избере видео и го отваря за качване. False, когато нищо не е избрано. */
export async function importVideo(method: Method = 'scan'): Promise<boolean> {
  const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'], allowsEditing: false, quality: 1 });
  if (picked.canceled || !picked.assets?.length) return false;
  const asset = picked.assets[0];
  const seconds = asset.duration != null ? Math.round(asset.duration / 1000) : undefined;
  if ((method === 'scan' || method === 'express') && seconds != null && seconds < MIN_SECONDS)
    throw new ImportProblem(`Видеото е само ${seconds} с. Използвайте такова, което обикаля целия предмет (30 с или повече).`);

  const scan = createScan(currentUserId() ?? undefined, method);
  try {
    const src = new File(asset.uri);
    const dest = videoFile(scan.id);
    // Прозорецът за избор дава собствено копие в кеша на приложението: преместването му не струва нищо.
    try {
      src.moveSync(dest);
    } catch {
      src.copySync(dest);
    }
    if (dest.size > MAX_BYTES)
      throw new ImportProblem(
        'Видеото е твърде голямо. Изберете по-кратко.',
      );
    updateScan(scan.id, { status: 'uploading', videoSeconds: seconds, imported: true });
  } catch (e) {
    deleteScan(scan.id);
    throw e;
  }
  router.push({ pathname: '/processing/[id]', params: { id: scan.id, auto: '1' } });
  return true;
}

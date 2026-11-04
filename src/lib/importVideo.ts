import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';

import type { Method } from './api';
import { createScan, deleteScan, updateScan, videoFile } from './scans';
import { currentUserId } from './session';

// A walk-around video filmed earlier (with the phone's own camera app) instead of in the app. The server treats it
// like an app recording, except that there is no orientation log for very plain objects. Pro's AI models take any
// video.

const MAX_BYTES = 600 * 1024 * 1024; // the server's MAX_VIDEO_MB
const MIN_SECONDS = 10;

export class ImportProblem extends Error {}

/** Lets the user pick a video and opens it for upload. False when nothing was picked. */
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
    // The picker hands over its own copy in the app's cache: moving it costs nothing.
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

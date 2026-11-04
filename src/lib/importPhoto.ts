import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { Alert } from 'react-native';

import { errorText } from './api';
import { ONE_PHOTO_INFO } from './prefs';
import { createScan, deleteScan, photosDir, updateScan } from './scans';
import { currentUserId } from './session';

// 3D модел от една снимка: сървърът изрязва предмета и AI (TripoSR) изгражда формата, като предполага задната страна.

const MAX_BYTES = 15 * 1024 * 1024; // MAX_PHOTO_BYTES на сървъра

class PhotoProblem extends Error {}

async function pick(source: 'camera' | 'gallery') {
  const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], allowsEditing: false, quality: 0.92 };
  if (source === 'gallery') return ImagePicker.launchImageLibraryAsync(options);
  const access = await ImagePicker.requestCameraPermissionsAsync();
  if (!access.granted) throw new PhotoProblem('Разрешете достъпа до камерата от настройките на телефона.');
  return ImagePicker.launchCameraAsync(options);
}

async function fromSource(source: 'camera' | 'gallery') {
  const picked = await pick(source);
  if (picked.canceled || !picked.assets?.length) return;
  const scan = createScan(currentUserId() ?? undefined, 'single');
  try {
    const dest = new File(photosDir(scan.id), 'img_0001.jpg');
    new File(picked.assets[0].uri).copySync(dest);
    if (dest.size > MAX_BYTES) throw new PhotoProblem('Снимката е твърде голяма. Изберете по-малка.');
    updateScan(scan.id, { status: 'uploading', photoCount: 1 });
  } catch (e) {
    deleteScan(scan.id);
    throw e;
  }
  router.push({ pathname: '/processing/[id]', params: { id: scan.id, auto: '1' } });
}

/** Пита откъде е снимката и започва модела. */
export function startFromPhoto() {
  const go = (source: 'camera' | 'gallery') => () => {
    fromSource(source).catch((e) => Alert.alert('Снимката не може да се използва', errorText(e)));
  };
  Alert.alert(ONE_PHOTO_INFO.label, 'Снимайте само предмета, отблизо и при добра светлина. AI предполага задната страна.', [
    { text: 'Отказ', style: 'cancel' },
    { text: 'От галерията', onPress: go('gallery') },
    { text: 'Снимай', onPress: go('camera') },
  ]);
}

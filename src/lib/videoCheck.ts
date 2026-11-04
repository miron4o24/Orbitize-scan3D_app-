import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { createVideoPlayer, type VideoPlayer } from 'expo-video';
import { decode } from 'jpeg-js';

import { measure, THRESHOLDS, type FrameMetrics } from './quality';

// A look at a few frames of a video on the phone, before it is uploaded: a dark or blurry video would cost a long wait
// on the server for a poor model. Any trouble here (a phone that cannot read it back) just skips the check.

const SAMPLES = [0.1, 0.3, 0.5, 0.7, 0.9]; // where in the video
const SIDE = 384; // the size the thresholds are for
const READY_MS = 4000;

function ready(player: VideoPlayer): Promise<boolean> {
  if (player.status === 'readyToPlay') return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      sub.remove();
      resolve(false);
    }, READY_MS);
    const sub = player.addListener('statusChange', ({ status }) => {
      if (status !== 'readyToPlay' && status !== 'error') return;
      clearTimeout(timer);
      sub.remove();
      resolve(status === 'readyToPlay');
    });
  });
}

const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/** What is wrong with these frames, for the user; empty when they look fine. Most of the frames must show it. */
export function problems(frames: FrameMetrics[]): string[] {
  const out: string[] = [];
  if (median(frames.map((f) => f.brightness)) < THRESHOLDS.minBrightness) out.push('Видеото е твърде тъмно. Добавете светлина.');
  if (median(frames.map((f) => f.clipped)) > THRESHOLDS.maxClipped) out.push('Има твърде силни отблясъци. Избягвайте пряка светлина.');
  if (median(frames.map((f) => f.sharpness)) < THRESHOLDS.minSharpness) out.push('Видеото е размазано. Движете се по-бавно.');
  return out;
}

export async function checkVideo(uri: string, seconds: number): Promise<string[]> {
  let player: VideoPlayer | null = null;
  try {
    player = createVideoPlayer(uri);
    if (!(await ready(player))) return [];
    const thumbs = await player.generateThumbnailsAsync(SAMPLES.map((share) => share * seconds), { maxWidth: SIDE, maxHeight: SIDE });
    const frames: FrameMetrics[] = [];
    for (const thumb of thumbs) {
      const image = await ImageManipulator.manipulate(thumb).renderAsync();
      const saved = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.9 });
      const file = new File(saved.uri);
      try {
        const { data, width, height } = decode(await file.bytes(), { useTArray: true, formatAsRGBA: true });
        frames.push(measure(data, width, height));
      } finally {
        if (file.exists) file.delete();
      }
    }
    return frames.length ? problems(frames) : [];
  } catch {
    return [];
  } finally {
    player?.release();
  }
}

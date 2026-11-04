import type { File } from 'expo-file-system';
import { Mesh, MeshStandardMaterial, type Object3D } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import { photoTexturePlugin } from './photoTexture';

/**
 * GLB от сървъра като сцена на three.js. textureKey: именува файла на текстурата от снимките (lib/photoTexture.ts).
 */
export async function parseGlb(file: File, textureKey: string): Promise<Object3D> {
  const bytes = await file.bytes();
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const scene = (await new GLTFLoader().register(photoTexturePlugin(textureKey)).parseAsync(buffer, '')).scene;
  // Сканирания отпреди сървърът да записва материал получават стандартния изцяло метален материал на glTF, който
  // изглежда почти черен, защото няма какво да отразява: показваме ги матови.
  scene.traverse((o) => {
    if (!(o instanceof Mesh)) return;
    for (const m of [o.material].flat())
      if (m instanceof MeshStandardMaterial && m.metalness === 1 && m.roughness === 1 && !m.metalnessMap) {
        m.metalness = 0;
        m.roughness = 0.85;
      }
  });
  return scene;
}

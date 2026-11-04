import type { ExpoWebGLRenderingContext } from 'expo-gl';
import { File, Paths } from 'expo-file-system';
import { ExternalTexture, Mesh, MeshStandardMaterial, type Object3D, type WebGLRenderer } from 'three';
import type { GLTFLoaderPlugin, GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader.js';

// GLB файлът от сървъра носи текстурата от снимките като JPEG изображения. three.js декодира изображения с Image или
// createImageBitmap на браузъра, които React Native няма, затова зареждането би се провалило. Вместо това всяко
// изображение се записва във файл и щом контекстът на GL съществува (ModelView), expo-gl го декодира директно в
// текстура на видеокартата: texImage2D приема { localUri } (docs.expo.dev/versions/v57.0.0/sdk/gl-view).

type Pending = { localUri: string; gl?: ExpoWebGLRenderingContext };

/**
 * Добавка за GLTFLoader: текстурите стават ExternalTexture, чието изображение чака във файл (`prefix` именува
 * файловете).
 */
export function photoTexturePlugin(prefix: string) {
  return (parser: GLTFParser): GLTFLoaderPlugin => ({
    name: 'scan3d_photo_texture',
    loadTexture: (textureIndex: number) => {
      const source: number | undefined = parser.json.textures?.[textureIndex]?.source;
      const image = source === undefined ? undefined : parser.json.images?.[source];
      if (image?.bufferView === undefined) return null;
      return (parser.getDependency('bufferView', image.bufferView) as Promise<ArrayBuffer>).then((bytes) => {
        const file = new File(Paths.cache, `${prefix}-texture-${source}.jpg`);
        file.writeSync(new Uint8Array(bytes));
        const texture = new ExternalTexture(null);
        texture.flipY = false; // координатите на текстурата в glTF започват от горната страна на изображението, както редовете във файла
        texture.userData = { localUri: file.uri } satisfies Pending;
        return texture;
      });
    },
  });
}

/* Предава текстурите на модела на видеокартата на този контекст на GL (отново, ако изгледът е пресъздаден). */
export function uploadPhotoTextures(model: Object3D, gl: ExpoWebGLRenderingContext, renderer: WebGLRenderer) {
  model.traverse((o) => {
    if (!(o instanceof Mesh)) return;
    for (const m of [o.material].flat()) {
      if (!(m instanceof MeshStandardMaterial) || !(m.map instanceof ExternalTexture)) continue;
      const map = m.map;
      const pending = map.userData as Pending;
      if (!pending.localUri || pending.gl === gl) continue;
      try {
        const texture = gl.createTexture();
        // Свързва се през състоянието на three, за да остане верен записът му за свързаното. Не с
        // renderer.resetState(): той чете gl.canvas, което контекстът на expo-gl няма, и грешката TypeError затваряше
        // приложението.
        renderer.state.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // three го задава отново преди всяко свое качване
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, { localUri: pending.localUri } as never);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        map.sourceTexture = texture;
      } catch (e) {
        console.warn('Photo texture not shown:', e);
        m.map = null; // обикновеният матов модел вместо нищо
        m.needsUpdate = true;
        continue;
      }
      pending.gl = gl;
      // Текселите са в sRGB и се качват непроменени: декодират се в шейдъра (three прави така за видео текстури).
      m.onBeforeCompile = (shader) => {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <map_fragment>',
          '#define DECODE_VIDEO_TEXTURE\n#include <map_fragment>',
        );
      };
      m.customProgramCacheKey = () => 'scan3d-srgb-map';
      m.needsUpdate = true;
    }
  });
}

import type { ExpoWebGLRenderingContext } from 'expo-gl';
import { File, Paths } from 'expo-file-system';
import { ExternalTexture, Mesh, MeshStandardMaterial, type Object3D, type WebGLRenderer } from 'three';
import type { GLTFLoaderPlugin, GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader.js';

// The server's GLB carries the photo texture as JPEG images. three.js decodes images with the browser's Image or
// createImageBitmap, which React Native does not have, so the loader would fail on them. Instead each image is
// written to a file, and once the GL context exists (ModelView) expo-gl decodes it straight into a GPU texture:
// texImage2D accepts { localUri } (docs.expo.dev/versions/v57.0.0/sdk/gl-view).

type Pending = { localUri: string; gl?: ExpoWebGLRenderingContext };

/** GLTFLoader plugin: textures become ExternalTextures whose image waits in a file (`prefix` names the files). */
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
        texture.flipY = false; // glTF's texture coordinates start at the image's top, as the file's rows do
        texture.userData = { localUri: file.uri } satisfies Pending;
        return texture;
      });
    },
  });
}

/** Hands the model's photo textures to the GPU of this GL context (again after the view was recreated). */
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
        // Bound through three's state so its record of what is bound stays right. Not renderer.resetState(): that
        // reads gl.canvas, which expo-gl's context lacks, and the TypeError closed the app.
        renderer.state.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // three sets it again before each of its own uploads
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, { localUri: pending.localUri } as never);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        map.sourceTexture = texture;
      } catch (e) {
        console.warn('Photo texture not shown:', e);
        m.map = null; // the plain matte model rather than none
        m.needsUpdate = true;
        continue;
      }
      pending.gl = gl;
      // The texels are sRGB, uploaded as they are: decode them in the shader (three does so for video textures).
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

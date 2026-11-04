import { Box3, Group, Vector3, type Object3D } from 'three';
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';

// Measured on a detached copy: the shown model sits under the viewer's fit, zoom and rotation groups.
export function longestSide(model: Object3D) {
  const size = new Box3().setFromObject(model.clone()).getSize(new Vector3());
  return Math.max(size.x, size.y, size.z);
}

// The GLB is glTF Y-up; slicers expect Z-up and millimetres.
export function exportForPrint(model: Object3D, format: 'stl' | 'obj', longestMm: number): Uint8Array | string {
  const root = new Group();
  root.rotation.x = Math.PI / 2;
  root.scale.setScalar(longestMm / longestSide(model));
  root.add(model.clone());
  root.updateMatrixWorld(true);
  if (format === 'obj') return new OBJExporter().parse(root);
  const view = new STLExporter().parse(root, { binary: true }) as DataView;
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

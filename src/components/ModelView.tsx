import { GLView, type ExpoWebGLRenderingContext } from 'expo-gl';
import { useEffect, useRef, useState } from 'react';
import { PanResponder, StyleSheet, View } from 'react-native';
import { useReducedMotion } from 'react-native-reanimated';

import { uploadPhotoTextures } from '@/lib/photoTexture';
import {
  AmbientLight,
  Box3,
  BufferGeometry,
  DirectionalLight,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Raycaster,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
  type Material,
  type Object3D,
} from 'three';

/**
 * Точка върху модела: дялове (0-1) от ограничаващата му кутия по собствените му (glTF) оси x, y и z. Сървърът я
 * превръща обратно в координати на модела (сървър app/edit.py spot_point).
 */
export type Spot = readonly [number, number, number];

/**
 * background: радиалният градиент зад модела, цвят от центъра към ръба (#rrggbb). beforeDraw: изпълнява се преди
 * всяко рисуване, а нова функция рисува отново (тестът за здравина оцветява модела на място). onPick: докосване на
 * модела (не влачене) избира точката под пръста и съобщава размера на модела в собствените му оси; spot: точка,
 * отбелязана с кръгче; marks: няколко, свързани с линия (измерване).
 */
type Props = {
  model: Object3D;
  wireframe: boolean;
  background: readonly [string, string];
  beforeDraw?: () => void;
  spot?: Spot | null;
  marks?: readonly Spot[];
  onPick?: (spot: Spot, size: readonly [number, number, number]) => void;
};

type Stage = {
  gl: ExpoWebGLRenderingContext;
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  orbit: Group; // завъртане и мащабиране от потребителя
  fit: Group; // мащабира модела до куб с ръб 2 единици
  backdrop: ShaderMaterial;
};

function rgb(hex: string) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return new Vector3(r, g, b);
}

// Градиентът като четириъгълник на целия екран, нарисуван пръв. Цветовете му излизат непроменени (без преобразуване
// на цветовото пространство в шейдъра), затова съвпадат точно с цветовете на приложението.
function backdrop() {
  const material = new ShaderMaterial({
    uniforms: { inner: { value: new Vector3() }, outer: { value: new Vector3() } },
    vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader:
      'uniform vec3 inner; uniform vec3 outer; varying vec2 vUv;' +
      'void main() { float t = clamp(length(vUv * 2.0 - 1.0) / 1.41421356, 0.0, 1.0); gl_FragColor = vec4(mix(inner, outer, t), 1.0); }',
    depthTest: false,
    depthWrite: false,
  });
  const quad = new Mesh(new PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  quad.renderOrder = -1;
  return { quad, material };
}

const INITIAL = { rx: 0.35, ry: 0.6, zoom: 1 };
const TAP_MOVE = 8; // докосване, което се мести по-малко от това (точки) и завършва в рамките на TAP_MS, е натискане, а не завъртане
const TAP_MS = 400;
const SPOT_COLOR = 0xff3d2e;
const SPOT_SIZE = 0.018; // радиусът на кръгчето като дял от най-голямата страна на модела
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// three.js рисува върху канава; expo-gl дава само контекста на WebGL, затова подаваме на three минимален заместител.
function createStage(gl: ExpoWebGLRenderingContext): Stage {
  const { drawingBufferWidth: width, drawingBufferHeight: height } = gl;
  const canvas = {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    style: {},
    addEventListener() {},
    removeEventListener() {},
    getContext: () => gl,
  };
  const renderer = new WebGLRenderer({ canvas: canvas as unknown as HTMLCanvasElement, antialias: true });
  renderer.setSize(width, height, false);

  const camera = new PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 0, 4);
  const key = new DirectionalLight(0xffffff, 1.6);
  key.position.set(3, 5, 4);
  const back = new DirectionalLight(0xffffff, 0.5);
  back.position.set(-4, -2, -3);
  const orbit = new Group();
  const fit = new Group();
  orbit.add(fit);
  const scene = new Scene();
  const { quad, material } = backdrop();
  scene.add(quad, new AmbientLight(0xffffff, 1.2), key, back, orbit);
  return { gl, renderer, scene, camera, orbit, fit, backdrop: material };
}

const INTRO_TURN = 1.1; // радианите, на които моделът се завърта на мястото си при появата
const INTRO_MS = 1400;

// Един пръст върти, два мащабират. Рисува само при промяна: след краткото завъртане при появата нищо не се анимира.
export function ModelView({ model, wireframe, background, beforeDraw, spot, marks, onPick }: Props) {
  const reduce = useReducedMotion();
  const [view, setView] = useState(() => (reduce ? INITIAL : { ...INITIAL, ry: INITIAL.ry - INTRO_TURN }));
  const [stage, setStage] = useState<Stage | null>(null);
  // touched: завъртането спира окончателно при първото докосване. Останалото е нужно при натискане и се поддържа
  // актуално за обработчиците.
  const live = useRef({
    view,
    touched: false,
    stage: null as Stage | null,
    model,
    box: null as Box3 | null,
    size: { width: 0, height: 0 },
    onPick,
  });
  live.current.stage = stage;
  live.current.model = model;
  live.current.onPick = onPick;
  const [pan] = useState(() => {
    let start = { ...INITIAL, dist: 0, at: 0 };
    const current = () => live.current.view;
    return PanResponder.create({
      // Прихващане: иначе GLView отдолу би поел докосванията.
      onStartShouldSetPanResponderCapture: () => true,
      onMoveShouldSetPanResponderCapture: () => true,
      onPanResponderGrant: () => {
        live.current.touched = true;
        start = { ...current(), dist: 0, at: Date.now() };
      },
      onPanResponderRelease: (e, g) => {
        const tap = !start.dist && Math.abs(g.dx) < TAP_MOVE && Math.abs(g.dy) < TAP_MOVE && Date.now() - start.at < TAP_MS;
        if (tap) pick(e.nativeEvent.locationX, e.nativeEvent.locationY);
      },
      onPanResponderMove: (e, g) => {
        const t = e.nativeEvent.touches;
        if (t.length >= 2) {
          const d = Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);
          if (!start.dist) start.dist = d;
          live.current.view = { ...current(), zoom: clamp((start.zoom * d) / start.dist, 0.3, 6) };
        } else if (!start.dist) {
          live.current.view = { ...current(), rx: clamp(start.rx + g.dy * 0.01, -1.5, 1.5), ry: start.ry + g.dx * 0.01 };
        }
        setView(live.current.view);
      },
    });
  });

  // Точката от модела под докосването, като Spot.
  function pick(x: number, y: number) {
    const { stage: st, model: m, box, size, onPick: picked } = live.current;
    if (!st || !box || !picked || !size.width || !size.height) return;
    st.scene.updateMatrixWorld(true);
    const ray = new Raycaster();
    ray.setFromCamera(new Vector2((x / size.width) * 2 - 1, -(y / size.height) * 2 + 1), st.camera);
    const hit = ray.intersectObject(m, true)[0];
    if (!hit) return;
    const local = m.worldToLocal(hit.point.clone());
    const d = box.getSize(new Vector3());
    const share = (v: number, lo: number, s: number) => (s > 0 ? Math.min(1, Math.max(0, (v - lo) / s)) : 0.5);
    picked([share(local.x, box.min.x, d.x), share(local.y, box.min.y, d.y), share(local.z, box.min.z, d.z)], [d.x, d.y, d.z]);
  }

  // Завъртането на място, щом повърхността на GL е налична.
  useEffect(() => {
    if (!stage || reduce) return;
    const from = INITIAL.ry - INTRO_TURN;
    const begin = Date.now();
    let frame = requestAnimationFrame(function step() {
      if (live.current.touched) return;
      const t = Math.min(1, (Date.now() - begin) / INTRO_MS);
      live.current.view = { ...live.current.view, ry: from + INTRO_TURN * (1 - (1 - t) ** 3) };
      setView(live.current.view);
      if (t < 1) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [stage, reduce]);

  useEffect(() => () => stage?.renderer.dispose(), [stage]);

  // Центрира модела и го вписва в куб с ръб 2 единици. Измерва се отделно, така че повторно изпълнение никога не
  // натрупва отместването.
  useEffect(() => {
    if (!stage) return;
    model.removeFromParent();
    const box = new Box3().setFromObject(model);
    live.current.box = box; // в собствените оси на модела: измерва се отделно
    const size = box.getSize(new Vector3());
    const centre = new Group();
    centre.position.copy(box.getCenter(new Vector3()).negate());
    centre.add(model);
    stage.fit.scale.setScalar(2 / Math.max(size.x, size.y, size.z));
    stage.fit.add(centre);
    uploadPhotoTextures(model, stage.gl, stage.renderer);
    return () => {
      stage.fit.remove(centre);
    };
  }, [stage, model]);

  useEffect(() => {
    model.traverse((o) => {
      if (o instanceof Mesh) for (const m of [o.material].flat() as Material[]) (m as Material & { wireframe: boolean }).wireframe = wireframe;
    });
  }, [model, wireframe]);

  // Избраните точки като кръгчета върху модела (в собствените му оси, за да се въртят с него), като две се свързват с
  // линия, видима през модела. Самите те не се улавят от докосвания.
  const shown = JSON.stringify(marks ?? (spot ? [spot] : []));
  useEffect(() => {
    const box = live.current.box;
    const points = JSON.parse(shown) as Spot[];
    if (!stage || !box || !points.length) return;
    const d = box.getSize(new Vector3());
    const at = ([x, y, z]: Spot) => new Vector3(box.min.x + x * d.x, box.min.y + y * d.y, box.min.z + z * d.z);
    const dots = points.map((p) => {
      const dot = new Mesh(new SphereGeometry(SPOT_SIZE * Math.max(d.x, d.y, d.z), 20, 14), new MeshBasicMaterial({ color: SPOT_COLOR }));
      dot.position.copy(at(p));
      dot.raycast = () => {};
      model.add(dot);
      return dot;
    });
    const line =
      points.length === 2
        ? new Line(new BufferGeometry().setFromPoints(points.map(at)), new LineBasicMaterial({ color: SPOT_COLOR, depthTest: false }))
        : null;
    if (line) {
      line.raycast = () => {};
      line.renderOrder = 1;
      model.add(line);
    }
    return () => {
      for (const dot of dots) {
        model.remove(dot);
        dot.geometry.dispose();
        dot.material.dispose();
      }
      if (line) {
        model.remove(line);
        line.geometry.dispose();
        line.material.dispose();
      }
    };
  }, [stage, model, shown]);

  const [inner, outer] = background;
  useEffect(() => {
    if (!stage) return;
    stage.backdrop.uniforms.inner.value.copy(rgb(inner));
    stage.backdrop.uniforms.outer.value.copy(rgb(outer));
    stage.orbit.rotation.set(view.rx, view.ry, 0);
    stage.orbit.scale.setScalar(view.zoom);
    beforeDraw?.();
    stage.renderer.render(stage.scene, stage.camera);
    stage.gl.endFrameEXP();
  }, [stage, model, view, wireframe, inner, outer, beforeDraw, shown]);

  return (
    <View
      style={styles.fill}
      {...pan.panHandlers}
      onLayout={(e) => (live.current.size = e.nativeEvent.layout)}
      accessible
      accessibilityLabel={
        onPick
          ? '3D модел. Докоснете място, за да го посочите. Плъзнете, за да го завъртите, щипнете за мащаб.'
          : '3D модел. Плъзнете, за да го завъртите, щипнете за мащаб.'
      }
    >
      <GLView style={styles.fill} onContextCreate={(gl) => setStage(createStage(gl))} />
    </View>
  );
}

const styles = StyleSheet.create({ fill: { flex: 1 } });

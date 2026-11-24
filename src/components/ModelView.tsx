import Ionicons from '@expo/vector-icons/Ionicons';
import { GLView, type ExpoWebGLRenderingContext } from 'expo-gl';
import { useEffect, useRef, useState } from 'react';
import { PanResponder, PixelRatio, Pressable, StyleSheet, View } from 'react-native';
import { useReducedMotion } from 'react-native-reanimated';

import { uploadPhotoTextures } from '@/lib/photoTexture';
import { makeStyles, radius, useColors } from '@/theme';
import {
  AmbientLight,
  Box3,
  BufferGeometry,
  DirectionalLight,
  DoubleSide,
  Group,
  HemisphereLight,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
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
 * background: радиалният градиент зад модела, цвят от центъра към ръба (#rrggbb). plain: моделът без цвета си (сив,
 * като глина), за да се види самата форма. controls: бутоните за увеличаване, намаляване и връщане на изгледа.
 * beforeDraw: изпълнява се преди всяко рисуване, а нова функция рисува отново (тестът за здравина оцветява модела на
 * място). onPick: докосване на модела (не влачене) избира точката под пръста и съобщава размера на модела в
 * собствените му оси; spot: точка, отбелязана с кръгче; marks: няколко, свързани с линия (измерване).
 */
type Props = {
  model: Object3D;
  wireframe: boolean;
  background: readonly [string, string];
  plain?: boolean;
  controls?: boolean;
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
  fit: Group; // мащабира модела до куб с ръб 2 единици, центриран в началото
  backdrop: ShaderMaterial;
};

// Камерата обикаля около точка (target): az, el в радиани, dist в единици (моделът е вписан в куб с ръб 2). Моделът
// стои на място, а приближаването мести камерата: при мащабиране на самия модел той минаваше през камерата.
type Orbit = { az: number; el: number; dist: number; tx: number; ty: number; tz: number };
const HOME: Orbit = { az: 0.6, el: 0.35, dist: 4.2, tx: 0, ty: 0, tz: 0 };
const NEAR_DIST = 1.25; // най-близо: още извън вписващия куб на повечето модели
const FAR_DIST = 11;
const EL_LIMIT = 1.45; // почти отгоре или отдолу, без да се обръща
const TURN = 0.008; // радиана на точка влачене
const ZOOM_STEP = 1.35; // бутоните + и -
const TAP_MOVE = 8; // докосване, което се мести по-малко от това (точки) и завършва в рамките на TAP_MS, е натискане
const TAP_MS = 400;
const DOUBLE_MS = 320; // второ натискане до толкова след първото връща изгледа
const GLIDE_DECAY = 0.92; // на кадър: колко от скоростта остава след пускане
const SPOT_SIZE = 0.018; // радиусът на кръгчето като дял от най-голямата страна на модела
const INTRO_TURN = 1.1; // радианите, на които моделът се завърта на мястото си при появата
const INTRO_MS = 1400;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const ease = (t: number) => 1 - (1 - t) ** 3;

function rgb(hex: string) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return new Vector3(r, g, b);
}

// Градиентът като четириъгълник на целия екран, нарисуван пръв, с мрежата от точки на темата (стъпка 18, точки от 1
// точка в c.lineStrong при 50%) зад модела. Цветовете излизат непроменени (без преобразуване на цветовото
// пространство в шейдъра), затова съвпадат точно с цветовете на приложението.
const DOT_PITCH = 18;
function backdrop() {
  const ratio = PixelRatio.get();
  const material = new ShaderMaterial({
    uniforms: {
      inner: { value: new Vector3() },
      outer: { value: new Vector3() },
      dotColor: { value: new Vector3() },
      pitch: { value: DOT_PITCH * ratio },
      size: { value: Math.max(1, ratio) },
    },
    vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader:
      'uniform vec3 inner; uniform vec3 outer; uniform vec3 dotColor; uniform float pitch; uniform float size; varying vec2 vUv;' +
      'void main() { float t = clamp(length(vUv * 2.0 - 1.0) / 1.41421356, 0.0, 1.0); vec3 base = mix(inner, outer, t);' +
      ' vec2 g = mod(gl_FragCoord.xy, pitch); float on = step(g.x, size) * step(g.y, size);' +
      ' gl_FragColor = vec4(mix(base, dotColor, 0.5 * on), 1.0); }',
    depthTest: false,
    depthWrite: false,
  });
  const quad = new Mesh(new PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  quad.renderOrder = -1;
  return { quad, material };
}

// expo-gl изпълнява само тези две настройки на pixelStorei; за всяка друга (three задава премултиплициране, цветово
// преобразуване и PACK_*/UNPACK_* при всяко качване на текстура) не прави нищо, а пише ред в конзолата. Тук те се
// пропускат тихо: резултатът е същият, без шума.
function quietPixelStore(gl: ExpoWebGLRenderingContext) {
  const native = gl.pixelStorei.bind(gl);
  const supported = new Set<number>([gl.UNPACK_FLIP_Y_WEBGL, gl.UNPACK_ALIGNMENT]);
  gl.pixelStorei = (pname: number, param: number | boolean) => {
    if (supported.has(pname)) native(pname, param as never);
  };
}

// three.js рисува върху канава; expo-gl дава само контекста на WebGL, затова подаваме на three минимален заместител.
// Основната светлина е закачена за камерата: страната, която гледате, винаги е осветена.
function createStage(gl: ExpoWebGLRenderingContext): Stage {
  quietPixelStore(gl);
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

  const camera = new PerspectiveCamera(40, width / height, 0.01, 100);
  const key = new DirectionalLight(0xffffff, 1.5);
  key.position.set(1.5, 2, 3); // спрямо камерата: отгоре вдясно, зад гледащия
  camera.add(key);
  const rim = new DirectionalLight(0xffffff, 0.45);
  rim.position.set(-3, 1, -2);
  camera.add(rim);
  const fit = new Group();
  const scene = new Scene();
  const { quad, material } = backdrop();
  scene.add(quad, new AmbientLight(0xffffff, 0.75), new HemisphereLight(0xffffff, 0x445566, 0.6), camera, fit);
  return { gl, renderer, scene, camera, fit, backdrop: material };
}

// Сивият материал на глина за plain: един за всички, двустранен (сканираните повърхности понякога имат обърнати
// стени).
const CLAY = new MeshStandardMaterial({ color: 0xc9ced6, roughness: 0.85, metalness: 0, side: DoubleSide });

/**
 * Моделът в 3D: един пръст го завърта (и след пускане леко продължава), два пръста го приближават и местят, двойно
 * докосване връща изгледа. Рисува само при промяна, направо от жестовете (без пререндериране на екрана).
 */
export function ModelView({ model, wireframe, background, plain = false, controls = true, beforeDraw, spot, marks, onPick }: Props) {
  const c = useColors();
  const s = useStyles();
  // Мерките (две точки и линия) са в сигналния цвят; посоченото място за промяна е избор (цвета за акцент).
  const markColor = marks ? c.signal : c.accent;
  const reduce = useReducedMotion();
  const [stage, setStage] = useState<Stage | null>(null);
  const live = useRef({
    orbit: reduce ? { ...HOME } : { ...HOME, az: HOME.az - INTRO_TURN },
    touched: false,
    stage: null as Stage | null,
    model,
    box: null as Box3 | null,
    size: { width: 0, height: 0 },
    onPick,
    beforeDraw,
    queued: false,
    glide: 0, // кадърът на анимацията (плъзгане след пускане или връщане), 0 - няма
    lastTap: 0,
  });
  live.current.stage = stage;
  live.current.model = model;
  live.current.onPick = onPick;
  live.current.beforeDraw = beforeDraw;

  // Рисуване най-много веднъж на кадър, от каквото и да е поискано.
  function draw() {
    const L = live.current;
    if (L.queued) return;
    L.queued = true;
    requestAnimationFrame(() => {
      L.queued = false;
      const st = L.stage;
      if (!st) return;
      const o = L.orbit;
      const cosEl = Math.cos(o.el);
      st.camera.position.set(o.tx + o.dist * cosEl * Math.sin(o.az), o.ty + o.dist * Math.sin(o.el), o.tz + o.dist * cosEl * Math.cos(o.az));
      st.camera.lookAt(o.tx, o.ty, o.tz);
      L.beforeDraw?.();
      st.renderer.render(st.scene, st.camera);
      st.gl.endFrameEXP();
    });
  }

  function stopAnimation() {
    if (live.current.glide) cancelAnimationFrame(live.current.glide);
    live.current.glide = 0;
  }

  // Плавен преход към друг изглед (връщане, бутоните за мащаб).
  function animateTo(to: Partial<Orbit>, ms = 380) {
    stopAnimation();
    const from = { ...live.current.orbit };
    const target = { ...from, ...to };
    const begin = Date.now();
    const step = () => {
      const t = Math.min(1, (Date.now() - begin) / ms);
      const k = reduce ? 1 : ease(t);
      const o = {} as Orbit;
      for (const key of Object.keys(target) as (keyof Orbit)[]) o[key] = from[key] + (target[key] - from[key]) * k;
      live.current.orbit = o;
      draw();
      live.current.glide = t < 1 && !reduce ? requestAnimationFrame(step) : 0;
    };
    step();
  }

  const zoomBy = (k: number) => {
    live.current.touched = true;
    animateTo({ dist: clamp(live.current.orbit.dist / k, NEAR_DIST, FAR_DIST) });
  };
  const reset = () => {
    live.current.touched = true;
    animateTo({ ...HOME, az: HOME.az + Math.round((live.current.orbit.az - HOME.az) / (2 * Math.PI)) * 2 * Math.PI });
  };

  const [pan] = useState(() => {
    let start = { orbit: { ...HOME }, pinch: 0, mid: { x: 0, y: 0 }, finger: { x: 0, y: 0 }, at: 0, two: false };
    const touches = (e: { nativeEvent: { touches: ArrayLike<{ pageX: number; pageY: number }> } }) => Array.from(e.nativeEvent.touches);
    const begin = (e: Parameters<typeof touches>[0]) => {
      const t = touches(e);
      start = { orbit: { ...live.current.orbit }, pinch: 0, mid: { x: 0, y: 0 }, finger: { x: 0, y: 0 }, at: start.at, two: start.two };
      if (t.length === 1) start.finger = { x: t[0].pageX, y: t[0].pageY };
      if (t.length >= 2) {
        start.pinch = Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);
        start.mid = { x: (t[0].pageX + t[1].pageX) / 2, y: (t[0].pageY + t[1].pageY) / 2 };
        start.two = true;
      }
    };
    return PanResponder.create({
      // Прихващане: иначе GLView отдолу би поел докосванията.
      onStartShouldSetPanResponderCapture: () => true,
      onMoveShouldSetPanResponderCapture: () => true,
      onPanResponderTerminationRequest: () => false,
      onPanResponderGrant: (e) => {
        live.current.touched = true;
        stopAnimation();
        start.at = Date.now();
        start.two = false;
        begin(e);
      },
      onPanResponderMove: (e, g) => {
        const t = touches(e);
        // Пръст добавен или махнат по средата на жеста: оттук нататък от новото положение, без скок.
        if ((t.length >= 2) !== start.pinch > 0) begin(e);
        if (t.length >= 2 && start.pinch > 0) {
          const d = Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);
          const mid = { x: (t[0].pageX + t[1].pageX) / 2, y: (t[0].pageY + t[1].pageY) / 2 };
          const o = { ...start.orbit, dist: clamp((start.orbit.dist * start.pinch) / Math.max(d, 1), NEAR_DIST, FAR_DIST) };
          // Местене с два пръста: в равнината на екрана, колкото пръстите (при това приближение).
          const h = live.current.size.height || 1;
          const perPoint = (2 * o.dist * Math.tan((40 * Math.PI) / 360)) / h;
          const dx = (mid.x - start.mid.x) * perPoint;
          const dy = (mid.y - start.mid.y) * perPoint;
          const right = new Vector3(Math.cos(o.az), 0, -Math.sin(o.az));
          const up = new Vector3(-Math.sin(o.el) * Math.sin(o.az), Math.cos(o.el), -Math.sin(o.el) * Math.cos(o.az));
          const shift = right.multiplyScalar(-dx).add(up.multiplyScalar(dy));
          live.current.orbit = { ...o, tx: clamp(start.orbit.tx + shift.x, -2, 2), ty: clamp(start.orbit.ty + shift.y, -2, 2), tz: clamp(start.orbit.tz + shift.z, -2, 2) };
        } else if (t.length === 1 && !start.pinch) {
          // От мястото на пръста при (пре)започването: след махнат втори пръст изгледът не скача.
          live.current.orbit = {
            ...start.orbit,
            az: start.orbit.az - (t[0].pageX - start.finger.x) * TURN,
            el: clamp(start.orbit.el + (t[0].pageY - start.finger.y) * TURN, -EL_LIMIT, EL_LIMIT),
          };
        }
        draw();
      },
      onPanResponderRelease: (e, g) => {
        const tap = !start.two && Math.abs(g.dx) < TAP_MOVE && Math.abs(g.dy) < TAP_MOVE && Date.now() - start.at < TAP_MS;
        if (tap) {
          const now = Date.now();
          // Където докосването посочва точки (мерене, място за промяна), двойното докосване е две точки.
          if (!live.current.onPick && now - live.current.lastTap < DOUBLE_MS) {
            live.current.lastTap = 0;
            reset();
          } else {
            live.current.lastTap = now;
            pick(e.nativeEvent.locationX, e.nativeEvent.locationY);
          }
          return;
        }
        // Завъртане с един пръст: продължава по инерция и спира плавно.
        if (!start.two && !reduce && Math.hypot(g.vx, g.vy) > 0.15) {
          let vx = -g.vx * 16 * TURN;
          let vy = g.vy * 16 * TURN;
          const step = () => {
            const o = live.current.orbit;
            live.current.orbit = { ...o, az: o.az + vx, el: clamp(o.el + vy, -EL_LIMIT, EL_LIMIT) };
            vx *= GLIDE_DECAY;
            vy *= GLIDE_DECAY;
            draw();
            live.current.glide = Math.abs(vx) + Math.abs(vy) > 0.0005 ? requestAnimationFrame(step) : 0;
          };
          live.current.glide = requestAnimationFrame(step);
        }
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
    const share = (v: number, lo: number, sz: number) => (sz > 0 ? Math.min(1, Math.max(0, (v - lo) / sz)) : 0.5);
    picked([share(local.x, box.min.x, d.x), share(local.y, box.min.y, d.y), share(local.z, box.min.z, d.z)], [d.x, d.y, d.z]);
  }

  // Завъртането на място, щом повърхността на GL е налична.
  useEffect(() => {
    if (!stage || reduce) return;
    const from = HOME.az - INTRO_TURN;
    const begin = Date.now();
    let frame = requestAnimationFrame(function step() {
      if (live.current.touched) return;
      const t = Math.min(1, (Date.now() - begin) / INTRO_MS);
      live.current.orbit = { ...live.current.orbit, az: from + INTRO_TURN * ease(t) };
      draw();
      if (t < 1) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [stage, reduce]);

  useEffect(
    () => () => {
      stopAnimation();
      stage?.renderer.dispose();
    },
    [stage],
  );

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
    draw();
    return () => {
      stage.fit.remove(centre);
    };
  }, [stage, model]);

  // Цветът на модела или глина; мрежата върху който и да е от двата. Собствените материали се пазят на мрежите.
  useEffect(() => {
    model.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      const own = (o.userData.ownMaterial ??= o.material) as Material | Material[];
      if (plain && !o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      o.material = plain ? CLAY : own;
      for (const m of [o.material].flat() as Material[]) (m as Material & { wireframe: boolean }).wireframe = wireframe;
    });
    draw();
  }, [model, wireframe, plain, stage]);
  useEffect(
    () => () => {
      // Глината е обща: следващият модел не бива да я наследи като мрежа.
      CLAY.wireframe = false;
      model.traverse((o) => {
        if (o instanceof Mesh && o.userData.ownMaterial) o.material = o.userData.ownMaterial;
      });
    },
    [model],
  );

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
      const dot = new Mesh(new SphereGeometry(SPOT_SIZE * Math.max(d.x, d.y, d.z), 20, 14), new MeshBasicMaterial({ color: markColor }));
      dot.position.copy(at(p));
      dot.raycast = () => {};
      model.add(dot);
      return dot;
    });
    const line =
      points.length === 2
        ? new Line(new BufferGeometry().setFromPoints(points.map(at)), new LineBasicMaterial({ color: markColor, depthTest: false }))
        : null;
    if (line) {
      line.raycast = () => {};
      line.renderOrder = 1;
      model.add(line);
    }
    draw();
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
  }, [stage, model, shown, markColor]);

  const [inner, outer] = background;
  const grid = c.lineStrong;
  useEffect(() => {
    if (!stage) return;
    stage.backdrop.uniforms.inner.value.copy(rgb(inner));
    stage.backdrop.uniforms.outer.value.copy(rgb(outer));
    stage.backdrop.uniforms.dotColor.value.copy(rgb(grid));
    draw();
  }, [stage, inner, outer, grid, beforeDraw]);

  return (
    <View style={styles.fill}>
      <View
        style={styles.fill}
        {...pan.panHandlers}
        onLayout={(e) => (live.current.size = e.nativeEvent.layout)}
        accessible
        accessibilityLabel={
          onPick
            ? '3D модел. Докоснете място, за да го посочите. Плъзнете, за да го завъртите, щипнете за мащаб, двойно докосване връща изгледа.'
            : '3D модел. Плъзнете, за да го завъртите, щипнете за мащаб, с два пръста го местите, двойно докосване връща изгледа.'
        }
      >
        <GLView style={styles.fill} onContextCreate={(gl) => setStage(createStage(gl))} />
      </View>
      {controls && (
        <View style={s.controls} pointerEvents="box-none">
          <ControlButton icon="add" label="Приближи" onPress={() => zoomBy(ZOOM_STEP)} />
          <ControlButton icon="remove" label="Отдалечи" onPress={() => zoomBy(1 / ZOOM_STEP)} />
          <ControlButton icon="scan-outline" label="Върни изгледа" onPress={reset} />
        </View>
      )}
    </View>
  );
}

function ControlButton({ icon, label, onPress }: { icon: 'add' | 'remove' | 'scan-outline'; label: string; onPress: () => void }) {
  const c = useColors();
  const s = useStyles();
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [s.control, pressed && s.controlPressed]} accessibilityRole="button" accessibilityLabel={label} hitSlop={6}>
      <Ionicons name={icon} size={18} color={c.text} />
    </Pressable>
  );
}

const styles = StyleSheet.create({ fill: { flex: 1 } });

const useStyles = makeStyles((c) => ({
  controls: { position: 'absolute', left: 12, bottom: 12, gap: 8 },
  control: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: c.line,
    backgroundColor: c.glass,
  },
  controlPressed: { borderColor: c.accentLine, backgroundColor: c.surface2 },
}));

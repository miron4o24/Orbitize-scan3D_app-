/* eslint-disable react/no-unknown-property -- three.js JSX props (position, intensity, ...) */
import { Canvas } from '@react-three/fiber/native';
import { useEffect, useMemo, useState } from 'react';
import { PanResponder, StyleSheet, View } from 'react-native';
import { Box3, Mesh, Vector3, type Material, type Object3D } from 'three';

type Props = { model: Object3D; wireframe: boolean };

const INITIAL = { rx: 0.35, ry: 0.6, zoom: 1 };
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// One finger orbits, two fingers pinch-zoom. Plain PanResponder: drei's OrbitControls is DOM-oriented.
export function ModelView({ model, wireframe }: Props) {
  const [view, setView] = useState(INITIAL);
  const [pan] = useState(() => {
    let current = INITIAL;
    let start = { ...INITIAL, dist: 0 };
    return PanResponder.create({
      // Capture: the Canvas below would otherwise claim the touches.
      onStartShouldSetPanResponderCapture: () => true,
      onMoveShouldSetPanResponderCapture: () => true,
      onPanResponderGrant: () => {
        start = { ...current, dist: 0 };
      },
      onPanResponderMove: (e, g) => {
        const t = e.nativeEvent.touches;
        if (t.length >= 2) {
          const d = Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);
          if (!start.dist) start.dist = d;
          current = { ...current, zoom: clamp((start.zoom * d) / start.dist, 0.3, 6) };
        } else if (!start.dist) {
          current = { ...current, rx: clamp(start.rx + g.dy * 0.01, -1.5, 1.5), ry: start.ry + g.dx * 0.01 };
        }
        setView(current);
      },
    });
  });

  // Centre the model and fit it into a 2-unit cube once.
  const fit = useMemo(() => {
    const box = new Box3().setFromObject(model);
    const size = box.getSize(new Vector3());
    return { center: box.getCenter(new Vector3()).negate(), scale: 2 / Math.max(size.x, size.y, size.z) };
  }, [model]);

  useEffect(() => {
    model.traverse((o) => {
      if (o instanceof Mesh) for (const m of [o.material].flat() as Material[]) (m as Material & { wireframe: boolean }).wireframe = wireframe;
    });
  }, [model, wireframe]);

  return (
    <View
      style={styles.fill}
      {...pan.panHandlers}
      accessible
      accessibilityLabel="3D model preview. Drag to rotate, pinch to zoom."
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }}>
        <color attach="background" args={['#eef1f5']} />
        <ambientLight intensity={1.2} />
        <directionalLight position={[3, 5, 4]} intensity={1.6} />
        <directionalLight position={[-4, -2, -3]} intensity={0.5} />
        <group rotation={[view.rx, view.ry, 0]} scale={view.zoom}>
          <group scale={fit.scale}>
            <primitive object={model} position={fit.center} />
          </group>
        </group>
      </Canvas>
    </View>
  );
}

const styles = StyleSheet.create({ fill: { flex: 1 } });

import { Accelerometer } from 'expo-sensors';
import { useEffect, useState } from 'react';

// Нивелирът за снимките на инженерния режим: телефонът трябва да е успореден на масата (камерата гледа право
// надолу), за да е снимката почти ортогонална проекция и очертанието на детайла да е вярно. Наклонът е ъгълът между
// силата на тежестта и оста на екрана; x и y казват накъде да се изправи.

export const LEVEL_DEG = 4; // до толкова градуса снимката се приема за права

export type Level = { available: boolean; tilt: number; x: number; y: number };

export function useLevel(): Level {
  const [level, setLevel] = useState<Level>({ available: true, tilt: 0, x: 0, y: 0 });
  useEffect(() => {
    let sub: { remove: () => void } | null = null;
    let cancelled = false;
    Accelerometer.isAvailableAsync()
      .then((has) => {
        if (cancelled) return;
        if (!has) return setLevel({ available: false, tilt: 0, x: 0, y: 0 });
        Accelerometer.setUpdateInterval(80);
        sub = Accelerometer.addListener(({ x, y, z }) => {
          const g = Math.hypot(x, y, z) || 1;
          const tilt = (Math.acos(Math.min(1, Math.abs(z) / g)) * 180) / Math.PI;
          setLevel({ available: true, tilt, x: x / g, y: y / g });
        });
      })
      .catch(() => !cancelled && setLevel({ available: false, tilt: 0, x: 0, y: 0 }));
    return () => {
      cancelled = true;
      sub?.remove();
    };
  }, []);
  return level;
}

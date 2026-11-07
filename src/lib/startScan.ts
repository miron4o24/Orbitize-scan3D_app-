import { router } from 'expo-router';

/* Ново сканиране от всяко място освен бутоните на началния екран. */
export function startScan() {
  router.push('/capture');
}

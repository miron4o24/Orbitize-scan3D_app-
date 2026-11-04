import Constants, { ExecutionEnvironment } from 'expo-constants';
import { Platform } from 'react-native';

import { confirmGooglePurchase } from './api';
import { updateProfile } from './session';

// Pro through Google Play. The phone only shows the offer and takes the payment; the server checks each purchase with
// Google before it counts (backend app/billing.py). Not in Expo Go, which has no store inside.

type Iap = typeof import('expo-iap');
export type ProOffer = { price: string; sku: string; offerToken: string | null };

let loaded: Iap | null | undefined;
function load(): Iap | null {
  if (loaded === undefined) {
    try {
      loaded =
        Platform.OS !== 'android' || Constants.executionEnvironment === ExecutionEnvironment.StoreClient
          ? null
          : // eslint-disable-next-line @typescript-eslint/no-require-imports -- must load lazily, see above
            (require('expo-iap') as Iap);
    } catch {
      loaded = null;
    }
  }
  return loaded;
}

let connected = false;
async function connect(iap: Iap) {
  if (connected) return;
  await iap.initConnection();
  connected = true;
}

/** Pro's price from the store; null when it cannot be bought on this phone. */
export async function proOffer(sku: string): Promise<ProOffer | null> {
  const iap = load();
  if (!iap) return null;
  try {
    await connect(iap);
    const found = await iap.fetchProducts({ skus: [sku], type: 'subs' });
    const product = found?.find((p) => p.id === sku);
    if (!product) return null;
    const offers = 'subscriptionOffers' in product ? product.subscriptionOffers : null;
    return { price: product.displayPrice, sku, offerToken: offers?.[0]?.offerTokenAndroid ?? null };
  } catch {
    return null;
  }
}

/** Pays and makes this account Pro. False when the user changed their mind; throws on trouble. */
export function subscribe(offer: ProOffer): Promise<boolean> {
  const iap = load();
  if (!iap) return Promise.resolve(false);
  return new Promise((resolve, reject) => {
    const stop = () => {
      bought.remove();
      failed.remove();
    };
    const bought = iap.purchaseUpdatedListener(async (purchase) => {
      if (purchase.productId !== offer.sku) return;
      stop();
      try {
        if (!purchase.purchaseToken) throw new Error('Покупката не можа да се провери.');
        updateProfile(await confirmGooglePurchase(purchase.purchaseToken));
        // Only after the server has it: Google refunds what is not finished within 3 days, so nothing is lost on a failure.
        await iap.finishTransaction({ purchase, isConsumable: false });
        resolve(true);
      } catch (e) {
        reject(e);
      }
    });
    const failed = iap.purchaseErrorListener((e) => {
      stop();
      if (/cancel/i.test(String(e.code))) resolve(false);
      else reject(new Error('Плащането не успя. Опитайте отново.'));
    });
    iap
      .requestPurchase({
        request: {
          google: {
            skus: [offer.sku],
            ...(offer.offerToken && { subscriptionOffers: [{ sku: offer.sku, offerToken: offer.offerToken }] }),
          },
        },
        type: 'subs',
      })
      .catch((e: unknown) => {
        stop();
        reject(e);
      });
  });
}

/** An earlier purchase (a new phone, a reinstall) back on this account. True when one was found. */
export async function restore(sku: string): Promise<boolean> {
  const iap = load();
  if (!iap) return false;
  await connect(iap);
  const purchases = await iap.getAvailablePurchases();
  const mine = purchases.find((p) => p.productId === sku && p.purchaseToken);
  if (!mine?.purchaseToken) return false;
  updateProfile(await confirmGooglePurchase(mine.purchaseToken));
  await iap.finishTransaction({ purchase: mine, isConsumable: false }).catch(() => {}); // already finished
  return true;
}

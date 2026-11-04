import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useState } from 'react';
import { Linking, Text, View } from 'react-native';

import { Appear } from '@/components/motion';
import { Button, Card, ErrorText, Header, IconName, Note, Screen, Small } from '@/components/ui';
import { errorText } from '@/lib/api';
import { refreshProfile } from '@/lib/auth';
import { proOffer, restore, subscribe, type ProOffer } from '@/lib/billing';
import { useSession } from '@/lib/session';
import { font, makeStyles, radius, space, useColors } from '@/theme';

// Какво дава Pro и как се получава. Цената идва от магазина, затова винаги е реалната.

const BENEFITS: { icon: IconName; title: string }[] = [
  { icon: 'sparkles', title: 'AI модели без ограничение' },
  { icon: 'color-wand-outline', title: 'Промени с AI и табелка с име' },
  { icon: 'barbell-outline', title: 'Тестове за здравина без ограничение' },
  { icon: 'cube-outline', title: 'Виж в стаята без ограничение' },
];

const SUBSCRIPTIONS = 'https://play.google.com/store/account/subscriptions';

export default function Pro() {
  const c = useColors();
  const s = useStyles();
  const session = useSession();
  const user = session?.kind === 'account' ? session.user : null;
  const [offer, setOffer] = useState<ProOffer | null>(null);
  const [busy, setBusy] = useState<'buy' | 'restore' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const product = user?.billing ? user.pro_product : undefined;

  useEffect(() => {
    if (product) proOffer(product).then(setOffer);
  }, [product]);

  async function run(what: 'buy' | 'restore', work: () => Promise<void>) {
    setBusy(what);
    setError(null);
    setInfo(null);
    try {
      await work();
      await refreshProfile().catch(() => {});
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  const buy = () =>
    run('buy', async () => {
      if (offer) await subscribe(offer);
    });
  const bringBack = () =>
    run('restore', async () => {
      if (product && !(await restore(product))) setInfo('Няма намерена покупка за този Google профил.');
    });

  return (
    <Screen>
      <Header title="Scanly Pro" />
      <Appear index={1}>
        <Card style={s.list}>
          {BENEFITS.map((b) => (
            <View key={b.title} style={s.row}>
              <View style={s.icon}>
                <Ionicons name={b.icon} size={18} color={c.accent} />
              </View>
              <Text style={s.title}>{b.title}</Text>
            </View>
          ))}
        </Card>
      </Appear>

      {error && <ErrorText>{error}</ErrorText>}
      {info && <Note>{info}</Note>}

      <Appear index={2} style={s.actions}>
        {user?.pro ? (
          <>
            <Note>{user.pro_until ? `Вашият Pro е активен до ${new Date(user.pro_until * 1000).toLocaleDateString('bg-BG')}. Подновява се автоматично.` : 'Вашият Pro е активен.'}</Note>
            {!!user.pro_until && <Button title="Управление на абонамента" variant="secondary" onPress={() => Linking.openURL(SUBSCRIPTIONS)} />}
          </>
        ) : offer ? (
          <>
            <Button variant="brand" size="lg" title={`Вземи Pro · ${offer.price} на месец`} onPress={buy} loading={busy === 'buy'} disabled={!!busy} />
            <Small style={s.center}>Спирате го по всяко време от Google Play.</Small>
            <Button title="Възстанови покупка" variant="ghost" onPress={bringBack} loading={busy === 'restore'} disabled={!!busy} />
          </>
        ) : (
          <Note>Плащането в приложението идва скоро.</Note>
        )}
      </Appear>
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  list: { gap: space.lg },
  row: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  icon: { width: 36, height: 36, borderRadius: radius.sm, backgroundColor: c.accentSoft, alignItems: 'center', justifyContent: 'center' },
  title: { flex: 1, fontFamily: font.medium, fontSize: 15, letterSpacing: -0.2, color: c.text },
  actions: { gap: space.md },
  center: { textAlign: 'center' },
}));

import { useEffect, useState } from 'react';
import { ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native';

import { Chip } from '@/components/admin';
import { Button, Card, ErrorText, Field, Label, Small } from '@/components/ui';
import { adminMail, adminSetMail, adminTestMail, errorText, type MailStatus } from '@/lib/api';
import { showDialog } from '@/lib/dialog';
import { useSession } from '@/lib/session';
import { makeStyles, space, useColors } from '@/theme';

/* Пощенската кутия, от която сървърът праща кодовете за потвърждение на нови профили. Без нея профилите се създават
   без потвърждение. Задава се от собственика; паролата не се връща от сървъра, затова празното поле я оставя. */
export default function AdminMail() {
  const c = useColors();
  const s = useStyles();
  const session = useSession();
  const owner = session?.kind === 'account' && !!session.user.owner;
  const [status, setStatus] = useState<MailStatus | null>(null);
  const [host, setHost] = useState('smtp.gmail.com');
  const [port, setPort] = useState('587');
  const [user, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [sender, setSender] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    adminMail()
      .then((m) => {
        setStatus(m);
        if (m.host) setHost(m.host);
        if (m.port) setPort(String(m.port));
        if (m.user) setUser(m.user);
        if (m.sender) setSender(m.sender);
      })
      .catch((e) => setError(errorText(e)));
  }, []);

  async function run(what: string, work: () => Promise<void>) {
    setBusy(what);
    setError(null);
    setNotice(null);
    try {
      await work();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  const save = () =>
    run('save', async () => {
      const m = await adminSetMail({ host: host.trim(), port: Number(port) || 587, user: user.trim(), password: password || undefined, sender: sender.trim() });
      setStatus(m);
      setPassword('');
      setNotice('Запазено. Изпратете пробно писмо, за да проверите.');
    });

  const test = () =>
    run('test', async () => {
      const r = await adminTestMail();
      setNotice(`Писмото е изпратено до ${r.sent_to}. Проверете пощата (и нежеланата поща).`);
    });

  const turnOff = () =>
    showDialog('Изключване на пощата?', 'Новите профили ще се създават без потвърждение по имейл.', [
      { text: 'Отказ', style: 'cancel' },
      {
        text: 'Изключи',
        style: 'destructive',
        onPress: () =>
          run('off', async () => {
            setStatus(await adminSetMail({ host: '', port: 587, user: '', sender: '' }));
          }),
      },
    ]);

  if (!status) return <View style={s.center}>{error ? <ErrorText>{error}</ErrorText> : <ActivityIndicator color={c.muted} />}</View>;

  const fromServer = status.source === 'server';
  const canEdit = owner && !fromServer;
  return (
    <KeyboardAvoidingView style={s.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView style={s.screen} contentContainerStyle={s.content} keyboardShouldPersistTaps="handled">
        <Card style={s.state}>
          <View style={s.flex}>
            <Label>Потвърждение на нови профили</Label>
            <Small>
              {status.configured
                ? `Кодът се праща от ${status.user}${fromServer ? ' (настройка на сървъра)' : ''}.`
                : 'Изключено: новите профили не се потвърждават по имейл.'}
            </Small>
          </View>
          <Chip label={status.configured ? 'Включено' : 'Изключено'} tone={status.configured ? 'success' : 'warning'} />
        </Card>

        {error && <ErrorText>{error}</ErrorText>}
        {notice && <Small style={{ color: c.success }}>{notice}</Small>}

        {canEdit ? (
          <>
            <Small>
              С Gmail: включете потвърждаване в две стъпки в Google профила, после „Пароли за приложения“ и създайте парола за
              Orbitize. Въведете адреса и тази парола от 16 букви (не обичайната парола).
            </Small>
            <View style={s.form}>
              <Label>Сървър за поща</Label>
              <Field value={host} onChangeText={setHost} autoCapitalize="none" autoCorrect={false} placeholder="smtp.gmail.com" accessibilityLabel="Сървър за поща" />
              <Label>Порт</Label>
              <Field value={port} onChangeText={setPort} keyboardType="number-pad" placeholder="587" accessibilityLabel="Порт" />
              <Label>Имейл на пощенската кутия</Label>
              <Field value={user} onChangeText={setUser} autoCapitalize="none" autoCorrect={false} keyboardType="email-address" placeholder="orbitize@gmail.com" accessibilityLabel="Имейл" />
              <Label>Парола за приложения</Label>
              <Field
                value={password}
                onChangeText={setPassword}
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                placeholder={status.configured ? 'запазена (оставете празно)' : 'abcd efgh ijkl mnop'}
                accessibilityLabel="Парола за приложения"
              />
              <Label>Подател (по желание)</Label>
              <Field value={sender} onChangeText={setSender} autoCorrect={false} placeholder="Orbitize <orbitize@gmail.com>" accessibilityLabel="Подател" />
            </View>
            <Button title="Запази" icon="save-outline" onPress={save} loading={busy === 'save'} disabled={!host.trim() || !user.trim() || (!password && !status.configured)} />
          </>
        ) : (
          <Small>{fromServer ? 'Пощата е зададена в .env на сървъра и се променя там.' : 'Само собственикът може да настройва пощата.'}</Small>
        )}

        {status.configured && <Button title="Изпрати пробно писмо до мен" variant="secondary" icon="mail-outline" onPress={test} loading={busy === 'test'} />}
        {canEdit && status.configured && <Button title="Изключи потвърждението" variant="ghost" onPress={turnOff} loading={busy === 'off'} />}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.lg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: c.bg },
  state: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  flex: { flex: 1, gap: 4 },
  form: { gap: space.sm },
}));

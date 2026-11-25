import { useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, View } from 'react-native';

import { Chip } from '@/components/admin';
import { Button, Card, ErrorText, Label, Small } from '@/components/ui';
import { adminMail, adminTestMail, errorText, type MailStatus } from '@/lib/api';
import { makeStyles, space, useColors } from '@/theme';

/* Дали новите профили се потвърждават с код по имейл. Пощенската кутия се задава само в .env на сървъра (SMTP_*,
   app/mail.py), не от приложението; тук се вижда състоянието и се праща пробно писмо. */
export default function AdminMail() {
  const c = useColors();
  const s = useStyles();
  const [status, setStatus] = useState<MailStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    adminMail()
      .then(setStatus)
      .catch((e) => setError(errorText(e)));
  }, []);

  async function test() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const r = await adminTestMail();
      setNotice(`Писмото е изпратено до ${r.sent_to}. Проверете пощата (и нежеланата поща).`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  if (!status) return <View style={s.center}>{error ? <ErrorText>{error}</ErrorText> : <ActivityIndicator color={c.muted} />}</View>;

  return (
    <ScrollView style={s.screen} contentContainerStyle={s.content}>
      <Card style={s.state}>
        <View style={s.flex}>
          <Label>Потвърждение на нови профили</Label>
          <Small>{status.configured ? `Кодът се праща от ${status.user}.` : 'Изключено: новите профили не се потвърждават по имейл.'}</Small>
        </View>
        <Chip label={status.configured ? 'Включено' : 'Изключено'} tone={status.configured ? 'success' : 'warning'} />
      </Card>
      {error && <ErrorText>{error}</ErrorText>}
      {notice && <Small style={{ color: c.success }}>{notice}</Small>}
      <Small>Пощенската кутия се задава в .env на сървъра (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD), не от приложението.</Small>
      {status.configured && <Button title="Изпрати пробно писмо до мен" variant="secondary" icon="mail-outline" onPress={test} loading={busy} />}
    </ScrollView>
  );
}

const useStyles = makeStyles((c) => ({
  screen: { flex: 1, backgroundColor: c.bg },
  content: { padding: 18, paddingTop: space.sm, paddingBottom: space.xxl * 2, gap: space.lg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: space.xl, backgroundColor: c.bg },
  state: { flexDirection: 'row', alignItems: 'center', gap: space.md },
  flex: { flex: 1, gap: 4 },
}));

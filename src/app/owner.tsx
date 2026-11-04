import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Switch } from 'react-native';

import { Appear } from '@/components/motion';
import { Card, ErrorText, Field, Header, Row, Screen, Section, Small } from '@/components/ui';
import { errorText, ownerUsers, setPro, type OwnerUser } from '@/lib/api';
import { shortDate } from '@/lib/dates';
import { useSession } from '@/lib/session';
import { makeStyles, useColors } from '@/theme';

// The owner's page (backend app/accounts.py OWNER_EMAILS): who is Pro, until Pro is a paid plan. The server checks
// the owner again on every call. The AI services' keys are the app's and are set on the server, not here.

export default function Owner() {
  const session = useSession();
  const c = useColors();
  const s = useStyles();
  const owner = session?.kind === 'account' && !!session.user.owner;
  const [query, setQuery] = useState('');
  const [users, setUsers] = useState<OwnerUser[] | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback((q: string) => {
    ownerUsers(q).then(
      (list) => {
        setUsers(list);
        setError(null);
      },
      (e) => setError(errorText(e)),
    );
  }, []);

  // A short pause after typing before asking the server.
  useEffect(() => {
    if (!owner) return;
    const t = setTimeout(() => search(query), 350);
    return () => clearTimeout(t);
  }, [owner, query, search]);

  async function toggle(u: OwnerUser, pro: boolean) {
    setSwitching(u.id);
    setUsers((list) => list?.map((x) => (x.id === u.id ? { ...x, pro } : x)) ?? null);
    try {
      await setPro(u.id, pro);
    } catch (e) {
      setUsers((list) => list?.map((x) => (x.id === u.id ? { ...x, pro: !pro } : x)) ?? null);
      setError(errorText(e));
    } finally {
      setSwitching(null);
    }
  }

  if (!owner)
    return (
      <Screen>
        <Header title="Само за собственика на приложението." />
      </Screen>
    );

  return (
    <Screen>
      <Header kicker="Собственик" title="Pro" sub="Кой може да прави AI модели" />
      {error && <ErrorText>{error}</ErrorText>}

      <Appear index={1}>
        <Section label={users ? `Потребители · ${users.length}` : 'Потребители'}>
          <Field value={query} onChangeText={setQuery} placeholder="Търсене по име или имейл" autoCapitalize="none" autoCorrect={false} />
          {users === null ? (
            <ActivityIndicator color={c.muted} />
          ) : users.length === 0 ? (
            <Small>Няма такива профили.</Small>
          ) : (
            <Card style={s.list}>
              {users.map((u, i) => (
                <Row
                  key={u.id}
                  label={u.name}
                  description={`${u.email} · от ${shortDate(u.created * 1000)}`}
                  value={u.owner ? 'Собственик' : undefined}
                  last={i === users.length - 1}
                  right={
                    u.owner ? undefined : (
                      <Switch
                        value={u.pro}
                        onValueChange={(v) => toggle(u, v)}
                        disabled={switching === u.id}
                        trackColor={{ true: c.accentStrong, false: c.surface3 }}
                        thumbColor="#ffffff"
                        ios_backgroundColor={c.surface3}
                        accessibilityLabel={`Pro за ${u.name}`}
                      />
                    )
                  }
                />
              ))}
            </Card>
          )}
          <Small>Pro профилите виждат „AI модел“ на началния екран. Промяната важи веднага.</Small>
        </Section>
      </Appear>
    </Screen>
  );
}

const useStyles = makeStyles(() => ({
  list: { paddingVertical: 0 },
}));

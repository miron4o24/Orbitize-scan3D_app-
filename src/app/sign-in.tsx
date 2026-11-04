import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useRef, useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, Text, type TextInputInstance, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Brand } from '@/components/Logo';
import { Appear, RiseLines } from '@/components/motion';
import { Stage } from '@/components/Stage';
import { Button, ErrorText, Field, Label, Small } from '@/components/ui';
import { errorText, health, type Health } from '@/lib/api';
import { createAccount, signInWithGoogle, signInWithPassword } from '@/lib/auth';
import { font, makeStyles, space, text, useColors } from '@/theme';

const PASSWORD_MIN = 6; // the server's rule (accounts.PASSWORD_MIN)

export default function SignIn() {
  const c = useColors();
  const s = useStyles();
  const [server, setServer] = useState<Health | 'unreachable' | null>(null);
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const emailRef = useRef<TextInputInstance>(null);
  const passwordRef = useRef<TextInputInstance>(null);

  const check = () => {
    setServer(null);
    health().then(setServer, () => setServer('unreachable'));
  };
  useEffect(() => {
    health().then(setServer, () => setServer('unreachable'));
  }, []);

  const register = mode === 'register';
  const emailOk = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim());
  const ready = emailOk && password.length >= (register ? PASSWORD_MIN : 1);

  async function submit() {
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      // On success the layout swaps this screen for the app.
      if (register) await createAccount(email, password, name);
      else await signInWithPassword(email, password);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function google() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signInWithGoogle();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  function switchMode() {
    setMode(register ? 'login' : 'register');
    setError(null);
  }

  // The server answers but its accounts are down: nobody can sign in until they are back.
  const noAccounts = server !== null && server !== 'unreachable' && !server.accounts;

  return (
    <SafeAreaView style={s.fill}>
      <KeyboardAvoidingView style={s.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={s.content} keyboardShouldPersistTaps="handled">
          <View style={s.top}>
            <Appear>
              <Brand />
            </Appear>
            <Appear index={1}>
              <Stage height={210} />
            </Appear>
            <RiseLines
              key={mode}
              lines={register ? ['Създайте', { soft: 'профил.' }] : ['Добре дошли', { soft: 'отново.' }]}
              style={{ ...text.display, color: c.text }}
              softColor={c.muted}
              delay={120}
            />
            <Appear index={4}>
              <Small>Сканирайте предмет. Отпечатайте го в реален размер.</Small>
            </Appear>
          </View>

          <Appear index={1} style={s.form}>
            {server !== null && server !== 'unreachable' && !!server.google_client_id && (
              <>
                <Button title="Продължи с Google" icon="logo-google" variant="secondary" onPress={google} disabled={busy} />
                <View style={s.or}>
                  <View style={s.orLine} />
                  <Label>или с имейл</Label>
                  <View style={s.orLine} />
                </View>
              </>
            )}
            {register && (
              <Field
                value={name}
                onChangeText={setName}
                placeholder="Вашето име"
                autoComplete="name"
                textContentType="name"
                returnKeyType="next"
                onSubmitEditing={() => emailRef.current?.focus()}
                maxLength={100}
                accessibilityLabel="Вашето име"
              />
            )}
            <Field
              ref={emailRef}
              value={email}
              onChangeText={setEmail}
              placeholder="Имейл"
              autoCapitalize="none"
              autoCorrect={false}
              autoComplete="email"
              keyboardType="email-address"
              textContentType="emailAddress"
              returnKeyType="next"
              onSubmitEditing={() => passwordRef.current?.focus()}
              maxLength={320}
              accessibilityLabel="Имейл"
            />
            <View>
              <Field
                ref={passwordRef}
                value={password}
                onChangeText={setPassword}
                placeholder={register ? `Парола (поне ${PASSWORD_MIN} знака)` : 'Парола'}
                secureTextEntry={!showPassword}
                autoCapitalize="none"
                autoCorrect={false}
                autoComplete={register ? 'new-password' : 'current-password'}
                textContentType={register ? 'newPassword' : 'password'}
                returnKeyType="go"
                onSubmitEditing={submit}
                maxLength={200}
                style={s.password}
                accessibilityLabel="Парола"
              />
              <Pressable
                style={s.eye}
                onPress={() => setShowPassword((v) => !v)}
                hitSlop={8}
                accessibilityRole="button"
                accessibilityLabel={showPassword ? 'Скрий паролата' : 'Покажи паролата'}
              >
                <Ionicons name={showPassword ? 'eye-off-outline' : 'eye-outline'} size={19} color={c.muted} />
              </Pressable>
            </View>

            {error && <ErrorText>{error}</ErrorText>}
            {server === 'unreachable' && <ErrorText>Няма връзка. Проверете интернета и опитайте отново.</ErrorText>}
            {noAccounts && <ErrorText>Профилите временно не са достъпни. Опитайте след малко.</ErrorText>}

            {server === 'unreachable' || noAccounts ? (
              <Button title="Опитай отново" icon="refresh" onPress={check} />
            ) : (
              <Button
                variant="brand"
                title={register ? 'Създай профил' : 'Вход'}
                onPress={submit}
                loading={busy || server === null}
                disabled={!ready}
              />
            )}
            <Pressable onPress={switchMode} accessibilityRole="button" style={s.switch} hitSlop={6}>
              <Text style={s.switchText}>
                {register ? 'Имате профил? ' : 'Нямате профил? '}
                <Text style={s.link}>{register ? 'Вход' : 'Регистрация'}</Text>
              </Text>
            </Pressable>
          </Appear>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  content: { flexGrow: 1, justifyContent: 'center', padding: 18, paddingVertical: space.xl, gap: space.xl },
  top: { gap: space.lg },
  form: { gap: space.md },
  password: { paddingRight: 44 },
  eye: { position: 'absolute', right: space.md, top: 0, bottom: 0, justifyContent: 'center' },
  switch: { paddingVertical: space.xs, alignSelf: 'center' },
  switchText: { fontFamily: font.regular, fontSize: 13.5, color: c.muted },
  link: { color: c.text, fontFamily: font.medium },
  or: { flexDirection: 'row', alignItems: 'center', gap: space.md, marginVertical: space.xs },
  orLine: { flex: 1, height: 1, backgroundColor: c.line },
}));

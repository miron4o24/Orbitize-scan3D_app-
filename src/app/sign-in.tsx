import Ionicons from '@expo/vector-icons/Ionicons';
import { useEffect, useRef, useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, Text, type TextInputInstance, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Brand } from '@/components/Logo';
import { Appear, RiseLines } from '@/components/motion';
import { Stage } from '@/components/Stage';
import { Button, ErrorText, Field, Label, Small } from '@/components/ui';
import { errorText, health, type Health } from '@/lib/api';
import { confirmEmail, createAccount, sendCodeAgain, signInWithGoogle, signInWithPassword } from '@/lib/auth';
import { font, makeStyles, radius, space, text, useColors } from '@/theme';

const PASSWORD_MIN = 6; // правилото на сървъра (accounts.PASSWORD_MIN)

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
  // Имейлът, на който е изпратен код за потвърждение (нов профил): екранът минава към въвеждане на кода.
  const [pending, setPending] = useState<string | null>(null);
  const [code, setCode] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
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
      // При успех подредбата заменя този екран с приложението; при нов профил първо кодът от имейла.
      const done = register ? await createAccount(email, password, name) : await signInWithPassword(email, password);
      if (done === 'verify') {
        setPending(email.trim());
        setCode('');
        setNotice(null);
      }
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

  async function confirm() {
    if (!pending || code.replace(/\D/g, '').length !== 6 || busy) return;
    setBusy(true);
    setError(null);
    try {
      await confirmEmail(pending, code.replace(/\D/g, ''));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  async function resend() {
    if (!pending || busy) return;
    setBusy(true);
    setError(null);
    try {
      await sendCodeAgain(pending);
      setNotice('Изпратихме нов код.');
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

  // Сървърът отговаря, но профилите му не работят: никой не може да влезе, докато не се върнат.
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
              style={{ ...text.title, color: c.text }}
              softColor={c.muted}
              delay={120}
            />
            <Appear index={4}>
              <Small>Сканирайте предмет. Отпечатайте го в реален размер.</Small>
            </Appear>
          </View>

          {pending ? (
            <Appear index={1} style={s.form}>
              <View style={s.mailIcon}>
                <Ionicons name="mail-unread-outline" size={26} color={c.accent} />
              </View>
              <Text style={s.codeTitle}>Потвърдете имейла си</Text>
              <Small>
                Изпратихме 6-цифрен код на {pending}. Въведете го тук (проверете и папката за нежелана поща).
              </Small>
              <Field
                value={code}
                onChangeText={(t) => setCode(t.replace(/\D/g, '').slice(0, 6))}
                placeholder="000000"
                keyboardType="number-pad"
                textContentType="oneTimeCode"
                autoComplete="one-time-code"
                maxLength={6}
                style={s.code}
                onSubmitEditing={confirm}
                accessibilityLabel="Код от имейла"
                autoFocus
              />
              {error && <ErrorText>{error}</ErrorText>}
              {notice && !error && <Small>{notice}</Small>}
              <Button variant="brand" title="Потвърди" onPress={confirm} loading={busy} disabled={code.length !== 6} />
              <View style={s.codeLinks}>
                <Pressable onPress={resend} accessibilityRole="button" hitSlop={6} disabled={busy}>
                  <Text style={s.link}>Изпрати нов код</Text>
                </Pressable>
                <Pressable
                  onPress={() => {
                    setPending(null);
                    setError(null);
                  }}
                  accessibilityRole="button"
                  hitSlop={6}
                >
                  <Text style={s.switchText}>Друг имейл</Text>
                </Pressable>
              </View>
            </Appear>
          ) : (
          <Appear index={1} style={s.form}>
            {register && <Label>Вашето име</Label>}
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
            <Label>Имейл</Label>
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
            <Label>Парола</Label>
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
            {server !== null && server !== 'unreachable' && !!server.google_client_id && (
              <>
                <View style={s.or}>
                  <View style={s.orLine} />
                  <Label>или</Label>
                  <View style={s.orLine} />
                </View>
                <Button title="Продължи с Google" icon="logo-google" variant="secondary" onPress={google} disabled={busy} />
              </>
            )}
            <Pressable onPress={switchMode} accessibilityRole="button" style={s.switch} hitSlop={6}>
              <Text style={s.switchText}>
                {register ? 'Имате профил? ' : 'Нямате профил? '}
                <Text style={s.link}>{register ? 'Вход' : 'Регистрация'}</Text>
              </Text>
            </Pressable>
          </Appear>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const useStyles = makeStyles((c) => ({
  fill: { flex: 1, backgroundColor: c.bg },
  content: { flexGrow: 1, justifyContent: 'center', padding: 18, paddingVertical: space.xl, gap: space.xl },
  top: { gap: space.lg },
  form: { gap: space.sm },
  password: { paddingRight: 44 },
  eye: { position: 'absolute', right: space.md, top: 0, bottom: 0, justifyContent: 'center' },
  switch: { paddingVertical: space.xs, alignSelf: 'center' },
  switchText: { fontFamily: font.regular, fontSize: 13.5, color: c.muted },
  link: { color: c.accent, fontFamily: font.medium },
  or: { flexDirection: 'row', alignItems: 'center', gap: space.md, marginVertical: space.sm },
  orLine: { flex: 1, height: 1, backgroundColor: c.line },
  mailIcon: { width: 48, height: 48, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: c.accentLine, borderRadius: radius.md },
  codeTitle: { fontFamily: font.semibold, fontSize: 20, color: c.text },
  code: { fontFamily: font.monoMedium, fontSize: 26, letterSpacing: 10, textAlign: 'center', marginTop: space.sm },
  codeLinks: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: space.xs },
}));

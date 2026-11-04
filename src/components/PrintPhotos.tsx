import { Image } from 'expo-image';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import { useCallback, useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, Text, View } from 'react-native';

import { Button, ErrorText, Section } from '@/components/ui';
import { authHeaders, errorText, listPrintPhotos, printPhotoUrl, removePrintPhoto, reportPrintPhoto, uploadPrintPhoto, type PrintPhoto } from '@/lib/api';
import { checkProgress } from '@/lib/progress';
import { font, makeStyles, radius, space } from '@/theme';

const SIDE = 1600;

// "Отпечатано от" за модел от общността: негови снимки в реалността. Всеки освен автора може да добави своя;
// докосването предлага какво може да се направи със снимката.
export function PrintPhotos({ postId, mine }: { postId: string; mine: boolean }) {
  const s = useStyles();
  const [photos, setPhotos] = useState<PrintPhoto[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    listPrintPhotos(postId).then(setPhotos, () => setPhotos(null)); // по-стар сървър: няма какво да се покаже
  }, [postId]);
  useEffect(load, [load]);

  async function add() {
    const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
    if (picked.canceled || !picked.assets?.length) return;
    const a = picked.assets[0];
    setBusy(true);
    setError(null);
    try {
      const fit = a.width >= a.height ? { width: Math.min(SIDE, a.width), height: null } : { width: null, height: Math.min(SIDE, a.height) };
      const image = await ImageManipulator.manipulate(a.uri).resize(fit).renderAsync();
      const out = await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.85 });
      const file = new File(out.uri);
      try {
        await uploadPrintPhoto(postId, file);
      } finally {
        if (file.exists) file.delete();
      }
      load();
      checkProgress(true);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  function options(photo: PrintPhoto) {
    const buttons: { text: string; style?: 'cancel' | 'destructive'; onPress?: () => void }[] = [];
    if (photo.can_remove)
      buttons.push({
        text: 'Изтрий',
        style: 'destructive',
        onPress: () => removePrintPhoto(photo.id).then(load, (e) => setError(errorText(e))),
      });
    if (!photo.mine)
      buttons.push({
        text: 'Докладвай',
        onPress: () => reportPrintPhoto(photo.id).then(() => Alert.alert('Благодарим', 'Ще прегледаме снимката.'), (e) => setError(errorText(e))),
      });
    buttons.push({ text: 'Отказ', style: 'cancel' });
    Alert.alert(photo.name, undefined, buttons);
  }

  if (photos === null) return null;
  return (
    <Section label={photos.length ? `Отпечатали го · ${photos.length}` : 'Отпечатахте ли го?'}>
      {photos.length > 0 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.row}>
          {photos.map((p) => (
            <Pressable key={p.id} onPress={() => options(p)} accessibilityRole="button" accessibilityLabel={`Снимка от ${p.name}`}>
              <Image source={{ uri: printPhotoUrl(p.id), headers: authHeaders() }} style={s.photo} contentFit="cover" transition={200} />
              <View style={s.name}>
                <Text style={s.nameText} numberOfLines={1}>
                  {p.name}
                </Text>
              </View>
            </Pressable>
          ))}
        </ScrollView>
      )}
      {error && <ErrorText>{error}</ErrorText>}
      {!mine && <Button title="Добави снимка на отпечатъка" icon="camera-outline" variant="secondary" onPress={add} loading={busy} />}
    </Section>
  );
}

const useStyles = makeStyles((c) => ({
  row: { gap: space.sm },
  photo: { width: 132, height: 168, borderRadius: radius.sm, backgroundColor: c.surface2 },
  name: { position: 'absolute', left: 6, right: 6, bottom: 6, backgroundColor: c.glass, borderRadius: radius.pill, paddingHorizontal: 8, paddingVertical: 2 },
  nameText: { fontFamily: font.medium, fontSize: 11, color: c.text },
}));

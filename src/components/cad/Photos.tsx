import Ionicons from '@expo/vector-icons/Ionicons';
import type { File } from 'expo-file-system';
import { Image } from 'expo-image';
import { Pressable, Text, View } from 'react-native';

import { Button, ErrorText, Header, Note, Screen } from '@/components/ui';
import type { CadPhoto } from '@/lib/api';
import { font, makeStyles, radius, space, useColors } from '@/theme';

import { SHOTS } from './Shoot';

// Преглед на снимките след снимането: коя страна е снимана и коя пропусната; докосване снима отново само нея.

export function Photos({
  shots,
  photos,
  uploading,
  error,
  onRetake,
  onNext,
  onBack,
}: {
  shots: CadPhoto[]; // снимките на режима (плосък детайл: отгоре и отдолу)
  photos: Partial<Record<CadPhoto, File>>;
  uploading: boolean;
  error: string | null;
  onRetake: (photo: CadPhoto) => void;
  onNext: () => void;
  onBack: () => void;
}) {
  const s = useStyles();
  const c = useColors();
  const list = SHOTS.filter((shot) => shots.includes(shot.photo));
  const count = list.filter((shot) => photos[shot.photo]).length;
  return (
    <Screen>
      <Header
        kicker={`Инженерен CAD · ${count} от ${list.length} снимки`}
        title="Снимките"
        sub="Докоснете снимка, за да я направите отново. Колкото повече страни, толкова по-точно."
      />
      <View style={s.grid}>
        {list.map((shot) => {
          const file = photos[shot.photo];
          return (
            <Pressable
              key={shot.photo}
              style={s.tile}
              onPress={() => onRetake(shot.photo)}
              accessibilityRole="button"
              accessibilityLabel={`${shot.title}: ${file ? 'снимай отново' : 'снимай'}`}
            >
              {file ? (
                <Image source={{ uri: file.uri }} style={s.thumb} contentFit="cover" />
              ) : (
                <View style={[s.thumb, s.empty]}>
                  <Ionicons name="camera-outline" size={22} color={c.muted} />
                  <Text style={s.emptyText}>пропусната</Text>
                </View>
              )}
              <View style={s.caption}>
                <Text style={s.title}>{shot.title}</Text>
                <Ionicons name={file ? 'refresh' : 'add'} size={15} color={c.text2} />
              </View>
            </Pressable>
          );
        })}
      </View>
      {count < 3 && list.length > 2 && <Note>С повече страни (поне отгоре, отпред и отдясно) формата се разчита по-сигурно.</Note>}
      {error && <ErrorText>{error}</ErrorText>}
      <Button title={uploading ? 'Качване на снимките…' : 'Напред: общите размери'} variant="brand" size="lg" icon="arrow-forward" loading={uploading} onPress={onNext} />
      <Button title="Назад" variant="ghost" icon="arrow-back" onPress={onBack} />
    </Screen>
  );
}

const useStyles = makeStyles((c) => ({
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: space.md },
  tile: {
    width: '47.5%',
    borderRadius: radius.md,
    overflow: 'hidden',
    backgroundColor: c.surface,
    borderWidth: 1,
    borderColor: c.line,
  },
  thumb: { width: '100%', aspectRatio: 4 / 3, backgroundColor: c.surface2 },
  empty: { alignItems: 'center', justifyContent: 'center', gap: 4 },
  emptyText: { color: c.muted, fontFamily: font.regular, fontSize: 12.5 },
  caption: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.md, paddingVertical: space.sm },
  title: { color: c.text, fontFamily: font.medium, fontSize: 14 },
}));

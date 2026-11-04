import { requireOptionalNativeModule } from 'expo';

type SceneViewerModule = {
  arSupport(): Promise<'supported' | 'unsupported' | 'unknown'>;
  openAr(path: string, title: string, realSize: boolean): Promise<void>;
  openInPackage(url: string, packageName: string): Promise<boolean>;
};

// null на iOS и в компилации отпреди модула.
export default requireOptionalNativeModule<SceneViewerModule>('SceneViewer');

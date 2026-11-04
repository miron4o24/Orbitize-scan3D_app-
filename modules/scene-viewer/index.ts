import { requireOptionalNativeModule } from 'expo';

type SceneViewerModule = {
  arSupport(): Promise<'supported' | 'unsupported' | 'unknown'>;
  openAr(path: string, title: string, realSize: boolean): Promise<void>;
  openInPackage(url: string, packageName: string): Promise<boolean>;
};

// null on iOS and in builds from before the module.
export default requireOptionalNativeModule<SceneViewerModule>('SceneViewer');

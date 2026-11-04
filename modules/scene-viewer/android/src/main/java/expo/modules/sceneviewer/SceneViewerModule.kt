package expo.modules.sceneviewer

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import com.google.ar.core.ArCoreApk
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.functions.Queues
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class SceneViewerModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("SceneViewer")

    // Whether this phone can run ARCore at all: "supported" (maybe after installing Google Play Services for AR),
    // "unsupported" (not on Google's list of AR phones) or "unknown" (the check did not finish).
    AsyncFunction<String>("arSupport") {
      val context = appContext.reactContext ?: return@AsyncFunction "unknown"
      var availability = ArCoreApk.getInstance().checkAvailability(context)
      var tries = 0
      while (availability.isTransient && tries < 25) { // the first check asks Google's servers
        Thread.sleep(200)
        availability = ArCoreApk.getInstance().checkAvailability(context)
        tries++
      }
      when {
        availability.isSupported -> "supported"
        availability == ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE -> "unsupported"
        else -> "unknown"
      }
    }

    // The model (a local GLB in metres) in the room, in the app's own AR screen (ArActivity).
    AsyncFunction("openAr") { path: String, title: String, realSize: Boolean ->
      val activity = appContext.currentActivity ?: throw Exceptions.MissingActivity()
      activity.startActivity(
        Intent(activity, ArActivity::class.java)
          .putExtra(ArActivity.EXTRA_PATH, path)
          .putExtra(ArActivity.EXTRA_TITLE, title)
          .putExtra(ArActivity.EXTRA_REAL_SIZE, realSize)
      )
    }.runOnQueue(Queues.MAIN)

    // Opens a URL in one given app (Intent.setPackage); false when that app is not installed.
    AsyncFunction("openInPackage") { url: String, packageName: String ->
      val activity = appContext.currentActivity ?: throw Exceptions.MissingActivity()
      val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).setPackage(packageName)
      try {
        activity.startActivity(intent)
        true
      } catch (e: ActivityNotFoundException) {
        false
      }
    }.runOnQueue(Queues.MAIN)
  }
}

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

    // Дали телефонът изобщо може да работи с ARCore: "supported" (може би след инсталиране на Google Play Services for AR),
    // "unsupported" (not on Google's list of AR phones) or "unknown" (the check did not finish).
    AsyncFunction<String>("arSupport") {
      val context = appContext.reactContext ?: return@AsyncFunction "unknown"
      var availability = ArCoreApk.getInstance().checkAvailability(context)
      var tries = 0
      while (availability.isTransient && tries < 25) { // първата проверка пита сървърите на Google
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

    // Моделът (местен GLB в метри) в стаята, в собствения AR екран на приложението (ArActivity).
    AsyncFunction("openAr") { path: String, title: String, realSize: Boolean ->
      val activity = appContext.currentActivity ?: throw Exceptions.MissingActivity()
      activity.startActivity(
        Intent(activity, ArActivity::class.java)
          .putExtra(ArActivity.EXTRA_PATH, path)
          .putExtra(ArActivity.EXTRA_TITLE, title)
          .putExtra(ArActivity.EXTRA_REAL_SIZE, realSize)
      )
    }.runOnQueue(Queues.MAIN)

    // Отваря адрес в дадено приложение (Intent.setPackage); false, когато то не е инсталирано.
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

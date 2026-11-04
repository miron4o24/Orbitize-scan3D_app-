package expo.modules.sceneviewer

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.ar.core.Anchor
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.ModelNode
import java.io.File

// The model in the room, inside the app: ARCore finds a floor or table, the model is put on it as soon as one is
// found (at its real size: the file is in metres with its base at the origin) and casts a shadow there. Tap another
// spot to move it, drag to slide it, twist with two fingers to turn it; pinch only when not at real size.
class ArActivity : ComponentActivity() {
  private lateinit var sceneView: ARSceneView
  private lateinit var hint: TextView
  private var modelNode: ModelNode? = null
  private var anchorNode: AnchorNode? = null
  private var failed = false

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val path = intent.getStringExtra(EXTRA_PATH)
    val title = intent.getStringExtra(EXTRA_TITLE) ?: ""
    val realSize = intent.getBooleanExtra(EXTRA_REAL_SIZE, true)

    val root = FrameLayout(this)
    sceneView = ARSceneView(this)
    root.addView(sceneView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

    val close = pill("✕  Затвори", 15f).apply { setOnClickListener { finish() } }
    root.addView(close, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.START))
    val name = pill(title, 15f)
    if (title.isNotEmpty()) {
      root.addView(name, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.END))
    }
    hint = pill("Камерата се включва…", 16f).apply { gravity = Gravity.CENTER }
    root.addView(hint, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
    setContentView(root)

    // Keep the buttons and text clear of the status and navigation bars.
    ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
      val m = dp(16)
      for (view in listOf(close, name, hint)) {
        val params = view.layoutParams as? FrameLayout.LayoutParams ?: continue // the name is left out when empty
        if (view == hint) params.setMargins(m, m, m, bars.bottom + m) else params.setMargins(m, bars.top + m, m, m)
        view.requestLayout()
      }
      insets
    }

    sceneView.apply {
      lifecycle = this@ArActivity.lifecycle
      planeRenderer.isEnabled = true
      planeRenderer.isShadowReceiver = true
      configureSession { _, config ->
        config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
        config.lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
        config.instantPlacementMode = Config.InstantPlacementMode.DISABLED
      }
      onSessionFailed = { e ->
        failed = true
        hint.text = when (e) {
          is UnavailableDeviceNotCompatibleException ->
            "Този телефон не поддържа AR: не е в списъка на Google с AR телефони. Моделът още може да се върти " +
              "в приложението."
          is UnavailableArcoreNotInstalledException, is UnavailableUserDeclinedInstallationException,
          is UnavailableApkTooOldException ->
            "AR изисква „Google Play Services for AR“. Инсталирайте или обновете го от Play Store и опитайте отново."
          is SecurityException -> "AR изисква камерата. Разрешете достъп до камерата в настройките на телефона."
          else -> "AR не можа да започне: ${e.message ?: e.javaClass.simpleName}"
        }
      }
      onTrackingFailureChanged = { reason -> showHint(reason) }
      onSessionUpdated = { _, _ ->
        // Put the model on the first floor or table seen in the middle of the screen.
        if (anchorNode == null && modelNode != null) {
          hitTestAR(planeTypes = setOf(Plane.Type.HORIZONTAL_UPWARD_FACING))?.let { place(it.createAnchor()) }
        }
      }
      setOnGestureListener(onSingleTapConfirmed = { e, node ->
        if (node == null && modelNode != null) {
          hitTestAR(e.x, e.y, planeTypes = setOf(Plane.Type.HORIZONTAL_UPWARD_FACING))?.let { place(it.createAnchor()) }
        }
      })
    }

    val file = path?.let { File(it) }
    if (file == null || !file.exists()) {
      failed = true
      hint.text = "Файлът на модела липсва. Върнете се и опитайте отново."
      return
    }
    try {
      val instance = sceneView.modelLoader.createModelInstance(file)
      // A scan's photo colours already hold its look. Filament still gives every surface a 4% mirror shine, which
      // with the room's light (ENVIRONMENTAL_HDR) made scans look oiled: no shine at all (ior 1 = no reflection).
      for (material in instance.materialInstances) {
        if (material.material.hasParameter("roughnessFactor")) material.setParameter("roughnessFactor", 1f)
        if (material.material.hasParameter("ior")) material.setParameter("ior", 1f)
      }
      modelNode = ModelNode(modelInstance = instance, scaleToUnits = null, centerOrigin = null).apply {
        isEditable = true
        isScaleEditable = !realSize
        isShadowCaster = true
      }
      showHint(null)
    } catch (e: Exception) {
      failed = true
      hint.text = "Моделът не се зареди: ${e.message ?: e.javaClass.simpleName}"
    }
  }

  private fun place(anchor: Anchor) {
    val node = modelNode ?: return
    val current = anchorNode
    if (current != null) {
      current.detachAnchor()
      current.anchor = anchor
      return
    }
    anchorNode = AnchorNode(sceneView.engine, anchor).apply {
      isEditable = true
      addChildNode(node)
    }
    sceneView.addChildNode(anchorNode!!)
    showHint(null)
  }

  private fun showHint(reason: TrackingFailureReason?) {
    if (failed) return
    hint.text = when (reason) {
      TrackingFailureReason.INSUFFICIENT_LIGHT -> "Твърде тъмно. Пуснете повече светлина."
      TrackingFailureReason.EXCESSIVE_MOTION -> "Движете телефона по-бавно."
      TrackingFailureReason.INSUFFICIENT_FEATURES -> "Насочете камерата към маса или под с някаква шарка."
      TrackingFailureReason.CAMERA_UNAVAILABLE -> "Камерата се използва от друго приложение."
      else ->
        if (anchorNode == null) "Насочете телефона към маса или под и го движете бавно, докато моделът се появи."
        else "Докоснете друго място, за да го преместите. Плъзнете, за да го местите, завъртете с два пръста."
    }
  }

  private fun pill(label: String, size: Float) = TextView(this).apply {
    text = label
    setTextColor(Color.WHITE)
    setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
    setPadding(dp(16), dp(10), dp(16), dp(10))
    background = GradientDrawable().apply {
      cornerRadius = dp(14).toFloat()
      setColor(Color.argb(184, 9, 9, 11)) // the app's dark background, see-through
      setStroke(dp(1), Color.argb(20, 255, 255, 255))
    }
  }

  private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

  companion object {
    const val EXTRA_PATH = "path"
    const val EXTRA_TITLE = "title"
    const val EXTRA_REAL_SIZE = "realSize"
  }
}

// A model from the community: the 3D viewer, likes, files to print it at its real size (or another), report.
(function () {
  if (!S.requireAuth()) return;
  S.page({ title: "Общност", active: "community" });
  const { el, icon, api, C } = S;
  const root = document.getElementById("root");
  document.getElementById("back").append(icon("back"), "Общност");
  const id = new URLSearchParams(location.search).get("id") || "";

  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem("orbitize." + k)); } catch { return null; } },
  };

  let post; // the /community/{id} entry
  let glbBlob = null, model = null; // the viewer's file, and its points for the print files (made on first use)
  let size = null; // chosen longest side in mm; the author's real size to begin with

  async function load() {
    try {
      post = await api(`/community/${encodeURIComponent(id)}`);
    } catch (err) {
      return root.replaceChildren(el("div", { class: "card center" },
        el("p", { class: "alert alert-error" }, err.message),
        el("a", { class: "btn btn-secondary", href: "community.html" }, "Обратно към общността")));
    }
    size = Math.round(post.longest_mm);
    document.title = `${post.title} · ${C.appName}`;
    root.replaceChildren(el("div", { class: "detail" }, viewer(), panel()));
    loadModel();
    update();
  }

  // The printer chosen on the scan page (kept in this browser), for "does it fit".
  function bed() {
    if (store.get("printer") === "__custom") return { name: "вашия принтер", ...(store.get("customBed") || { x: 220, y: 220, z: 250 }) };
    return C.printers.find((p) => p.name === (store.get("printer") || "Bambu Lab A1 mini")) || C.printers[0];
  }

  // --- 3D viewer ------------------------------------------------------------------------------------------------

  function viewer() {
    return el("div", { class: "viewer", id: "viewer" },
      el("div", { class: "overlay", id: "overlay" },
        el("div", { style: "display:grid;gap:14px;justify-items:center" },
          el("div", { class: "spinner" }),
          el("span", { id: "load-text" }, "Подготвяне на 3D модела…"),
          el("div", { class: "load-bar" }, el("i", { id: "load-bar" })))));
  }

  async function loadModel() {
    const box = document.getElementById("viewer"), overlay = document.getElementById("overlay");
    const slow = setTimeout(() => {
      const text = document.getElementById("load-text");
      if (text && !text.textContent.includes("%")) text.textContent = "Сървърът подготвя модела. Първото отваряне отнема по-дълго, следващите са мигновени.";
    }, 5000);
    try {
      glbBlob = await S.cachedBlob(`cglb/${id}/${post.published}`, `/community/${id}/model.glb`, {
        onProgress(p) {
          const bar = document.getElementById("load-bar"), text = document.getElementById("load-text");
          if (bar) bar.style.width = `${Math.round(p * 100)}%`;
          if (text) text.textContent = `Зареждане на 3D модела… ${Math.round(p * 100)}%`;
        },
      });
    } catch (err) {
      clearTimeout(slow);
      overlay.replaceChildren(el("div", { style: "display:grid;gap:12px;justify-items:center" },
        el("span", {}, err.message),
        el("button", { class: "btn btn-secondary btn-sm", onclick: () => { box.replaceWith(viewer()); loadModel(); } }, "Опитай отново")));
      return;
    }
    clearTimeout(slow);
    const mv = el("model-viewer", {
      src: URL.createObjectURL(glbBlob), alt: post.title, "camera-controls": true, "touch-action": "pan-y",
      "shadow-intensity": "0.7", "shadow-softness": "0.9", exposure: "1.05", "environment-image": "neutral",
      "interaction-prompt": "none", "camera-orbit": "30deg 70deg auto", "rotation-per-second": "20deg", "auto-rotate": true,
    });
    mv.addEventListener("load", () => { overlay.remove(); geometry().then(update).catch(() => {}); });
    mv.addEventListener("error", () => { overlay.textContent = "Моделът не може да се покаже в този браузър."; });
    const spin = el("button", { class: "tool on", title: "Въртене", "aria-label": "Въртене", "aria-pressed": "true" }, icon("rotate"));
    spin.onclick = () => {
      const on = !mv.hasAttribute("auto-rotate");
      mv.toggleAttribute("auto-rotate", on);
      spin.classList.toggle("on", on);
      spin.setAttribute("aria-pressed", String(on));
    };
    const reset = el("button", { class: "tool", title: "Начален изглед", "aria-label": "Начален изглед" }, icon("target"));
    reset.onclick = () => { mv.cameraOrbit = "30deg 70deg auto"; mv.cameraTarget = "auto auto auto"; mv.fieldOfView = "auto"; };
    const full = el("button", { class: "tool", title: "Цял екран", "aria-label": "Цял екран" }, icon("expand"));
    full.onclick = () => (document.fullscreenElement ? document.exitFullscreen() : box.requestFullscreen && box.requestFullscreen());
    box.prepend(mv);
    box.append(el("div", { class: "viewer-tools" }, spin, reset, box.requestFullscreen ? full : null),
      el("span", { class: "hint" }, "Плъзнете, за да въртите · щипнете или скролнете за увеличение"));
  }

  // The model's points, read once from the viewer's file (for the size, the fit and the print files).
  async function geometry() {
    if (!model) model = Supports.fromGLB(await glbBlob.arrayBuffer());
    return model;
  }

  // --- Panel --------------------------------------------------------------------------------------------------------

  const section = (label, ...kids) => el("section", { class: "sec" }, el("div", { class: "sec-label" }, label), ...kids);

  function panel() {
    const av = el("span", { class: "avatar sm" });
    S.authorAvatar(av, post.author);
    const cell = (label, idv, text) => el("div", {}, el("small", {}, label), el("b", { id: idv }, text));
    return el("div", { class: "panel" },
      el("div", {},
        el("div", { class: "title-row" }, el("h1", { style: "cursor:auto" }, post.title)),
        el("div", { class: "meta-line" },
          el("span", { class: "author" }, av, el("span", {}, post.author.name || "Потребител")),
          el("span", {}, `· ${S.date(post.published)}`),
          post.mine ? el("span", { class: "badge ok" }, "Ваш модел") : null)),
      el("div", { class: "summary" }, cell("Размер", "sum-size", "…"), cell("Принтер", "sum-fit", "…"), cell("Харесвания", "sum-likes", post.likes)),
      el("div", { class: "row-actions" }, likeButton()),
      downloadCard(),
      post.mine ? ownerCard() : reportCard());
  }

  function likeButton() {
    const b = el("button", { class: "btn btn-secondary like-big", type: "button" });
    const show = () => {
      b.classList.toggle("on", post.liked);
      b.setAttribute("aria-pressed", String(post.liked));
      b.replaceChildren(icon("heart"), post.liked ? "Харесано" : "Харесай", el("span", { class: "n" }, post.likes));
      const s = document.getElementById("sum-likes");
      if (s) s.textContent = post.likes;
    };
    b.onclick = async () => {
      const was = post.liked;
      post.liked = !was; post.likes += was ? -1 : 1; show();
      try { Object.assign(post, await api(`/community/${id}/like`, { method: was ? "DELETE" : "PUT" })); }
      catch (err) { post.liked = was; post.likes += was ? 1 : -1; S.toast(err.message); }
      show();
    };
    show();
    return b;
  }

  // Size in mm and "fits the printer", from the model once it is read (the server's numbers until then).
  function update() {
    const sz = document.getElementById("sum-size"), fit = document.getElementById("sum-fit");
    if (!sz) return;
    let dims = null;
    if (model) {
      const ext = [0, 1, 2].map((k) => model.max[k] - model.min[k]), f = size / Math.max(...ext);
      dims = ext.map((v) => v * f);
    } else if (post.dimensions_mm) {
      const f = size / Math.max(...post.dimensions_mm);
      dims = post.dimensions_mm.map((v) => v * f);
    }
    sz.textContent = dims ? dims.map((v) => Math.round(v)).join("×") + " мм" : `${size} мм`;
    if (!model) return;
    const b = bed(), m = C.printerMarginMm || 0;
    const [x, y] = dims.slice(0, 2).sort((a, b2) => a - b2), [bx, by] = [b.x - 2 * m, b.y - 2 * m].sort((a, b2) => a - b2);
    const ok = x <= bx && y <= by && dims[2] <= b.z;
    fit.textContent = ok ? "Събира се" : "Твърде голям";
    fit.className = ok ? "ok" : "bad";
    fit.title = b.name;
  }

  // --- Files --------------------------------------------------------------------------------------------------------

  const FORMATS = [["3mf", "3MF", "Bambu Studio, PrusaSlicer"], ["stl", "STL", "Всеки слайсер"], ["glb", "GLB", "С цветовете, за Blender"]];
  const fileName = () => post.title.replace(/[^\p{L}\p{N}_\- ]+/gu, "").trim().replace(/\s+/g, "-").slice(0, 50) || "model";

  function downloadCard() {
    const input = el("input", { class: "input", type: "number", min: "5", max: "2000", step: "1", value: size, inputmode: "numeric",
      "aria-label": "Най-дълга страна в мм" });
    input.oninput = () => {
      const v = Number(input.value);
      if (v >= 5 && v <= 2000) { size = Math.round(v); update(); }
    };
    const reset = el("button", { class: "btn btn-ghost btn-sm", type: "button" }, "Истински размер");
    reset.onclick = () => { size = Math.round(post.longest_mm); input.value = size; update(); };
    const buttons = FORMATS.map(([fmt, label, note]) => {
      const b = el("button", { class: "btn btn-secondary format-btn", type: "button" }, el("b", {}, label), el("span", {}, note));
      b.onclick = async () => {
        if (!glbBlob) return S.toast("Изчакайте моделът да се зареди.");
        b.disabled = true;
        const head = b.querySelector("b");
        head.textContent = "Подготвя се…";
        await new Promise((r) => setTimeout(r, 30)); // let "Подготвя се…" show first
        try {
          let file;
          if (fmt === "glb") file = Supports.scaledGLB(await glbBlob.arrayBuffer(), size);
          else {
            const m = await geometry(), mesh = { pos: Supports.mm(m, size), tri: m.tri };
            file = fmt === "stl" ? Supports.stl(mesh) : await Supports.threeMF(mesh, bed());
          }
          S.saveBlob(file, `${fileName()}-${size}mm.${fmt}`);
        } catch { S.toast("Файлът не можа да се направи."); }
        head.textContent = label;
        b.disabled = false;
      };
      return b;
    });
    return section("Изтегляне за печат",
      el("div", { class: "size-row" }, el("label", { class: "unit-input" }, input), reset),
      el("p", { class: "hint", style: "margin:8px 0 14px" }, `Най-дългата страна. Авторът го е сканирал с ${Math.round(post.longest_mm)} мм.`),
      el("div", { class: "formats three" }, buttons));
  }

  // --- Yours: take it out; someone else's: report -------------------------------------------------------------------

  function ownerCard() {
    const b = el("button", { class: "btn btn-danger btn-sm", type: "button" }, "Махни от общността");
    b.onclick = async () => {
      const ok = await S.confirmBox({ title: "Да махнем ли модела от общността?", text: "Сканирането остава във вашия профил. Харесванията му се изтриват.", ok: "Махни", danger: true });
      if (!ok) return;
      try {
        await api(`/me/scans/${id}/publish`, { method: "DELETE" });
        const list = S.scansCache.get();
        if (list) S.scansCache.set(list.map((s) => (s.id === id ? { ...s, published: null } : s)));
        location.replace("community.html");
      } catch (err) { S.toast(err.message); }
    };
    return section("Вашият модел",
      el("div", { class: "row-actions" }, el("a", { class: "btn btn-secondary btn-sm", href: `scan.html?id=${encodeURIComponent(id)}` }, "Отвори в „Моите сканирания“"), b));
  }

  function reportCard() {
    const b = el("button", { class: "btn btn-ghost btn-sm", type: "button" }, icon("flag"), "Докладвай");
    b.onclick = () => {
      const reason = el("textarea", { class: "input notes", rows: "3", maxlength: "200", placeholder: "Напр. неподходящо съдържание, чужд модел…" });
      const send = el("button", { class: "btn btn-danger" }, "Изпрати");
      const d = el("dialog", {},
        el("h3", {}, "Докладване на модела"),
        el("p", { class: "muted" }, "Кажете накратко какво не е наред. Ще прегледаме модела."),
        el("label", { class: "field" }, el("span", {}, "Причина"), reason),
        el("div", { class: "row-actions" }, el("button", { class: "btn btn-secondary", onclick: () => d.close() }, "Отказ"), send));
      send.onclick = async () => {
        send.disabled = true;
        try {
          await api(`/community/${id}/report`, { method: "POST", json: { reason: reason.value.trim() } });
          d.close();
          b.disabled = true;
          b.replaceChildren(icon("check"), "Докладвано");
          S.toast("Благодарим! Ще прегледаме модела.");
        } catch (err) { S.toast(err.message); send.disabled = false; }
      };
      d.addEventListener("close", () => d.remove());
      document.body.append(d);
      d.showModal();
    };
    return section("Нещо не е наред?", b);
  }

  load();
})();

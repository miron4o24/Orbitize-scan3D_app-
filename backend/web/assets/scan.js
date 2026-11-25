// One scan: the 3D viewer, "does it fit my printer", downloads at the chosen size, sharing, rename and delete.
(function () {
  if (!S.requireAuth()) return;
  S.page({ title: "Сканиране", active: "scans" });
  const { el, icon, api, C } = S;
  const root = document.getElementById("root");
  document.getElementById("back").append(icon("back"), "Моите сканирания");
  const id = new URLSearchParams(location.search).get("id") || "";
  const UNTITLED = "Без име";

  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem("orbitize." + k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem("orbitize." + k, JSON.stringify(v)); } catch {} },
  };

  let scan; // the /me/scans entry: title, share, state, report
  let pollTimer;

  // Shown at once from the list saved in the browser, then checked against the server.
  function fromCache() {
    const list = S.scansCache.get();
    return list && list.find((s) => s.id === id);
  }

  async function load() {
    clearTimeout(pollTimer);
    let fresh;
    try {
      const all = await api("/me/scans");
      S.scansCache.set(all);
      fresh = all.find((s) => s.id === id);
    } catch (err) {
      if (scan) return S.toast(err.message);
      return fail(err.message);
    }
    if (!fresh) return fail("Този модел не съществува или не е във вашия профил.");
    const changed = !scan || JSON.stringify(fresh) !== JSON.stringify(scan);
    scan = fresh;
    if (changed) render();
    else if (scan.state !== "done") pollTimer = setTimeout(load, 8000);
  }

  function fail(text) {
    root.replaceChildren(el("div", { class: "card center" },
      el("p", { class: "alert alert-error" }, text),
      el("a", { class: "btn btn-secondary", href: "scans.html" }, "Обратно към сканиранията")));
  }

  let shownState = null;
  function render() {
    document.title = `${scan.title || UNTITLED} · ${C.appName}`;
    const done = scan.state === "done";
    const panel = el("div", { class: "panel" },
      titleCard(),
      done ? summary() : statusCard(),
      done ? tabs([
        ["size", "Размер", [printCard()]],
        ["print", "Печат", [costCard(), supportCard(), splitCard()]],
        ["files", "Файлове", isCad() ? [cadCard(), downloadCard()] : [downloadCard()]],
        ["more", "Още", [communityCard(), shareCard(), notesCard(), dangerCard()]],
      ]) : el("div", { class: "card" }, notesCard(), dangerCard()));
    // Keep the viewer (and its loaded model) when only the panel changes.
    const old = document.getElementById("viewer");
    if (old && shownState === scan.state) old.nextElementSibling.replaceWith(panel);
    else {
      root.replaceChildren(el("div", { class: "detail" }, viewer(done), panel));
      if (done) loadModel();
    }
    shownState = scan.state;
    if (done) update();
  }

  // A part from the engineering mode (photos + measurements): its real size is known, and it has CAD files.
  const isCad = () => !!(scan.report && scan.report.ai === "cad");

  // The CAD files of the part (backend /jobs/{id}/cad/file/{name}): STEP for CAD programs, the model's code, the
  // drawing, and the plain STL in millimetres.
  function cadCard() {
    const cad = scan.report.cad || {};
    const FILES = [["part.step", "STEP", "SolidWorks, Fusion, FreeCAD, Onshape"], ["part.stl", "STL", "в мм, за всеки слайсър"],
      ["drawing.png", "Чертеж", "с размерите"], ["part.py", "Кодът на модела", "CadQuery (Python)"]];
    const box = el("div", { class: "formats" });
    for (const [name, label, sub] of FILES) {
      const b = el("button", { class: `btn btn-secondary format-btn ${name === "part.step" ? "rec" : ""}`, type: "button" },
        name === "part.step" ? el("span", { class: "best" }, "За CAD програми") : null, el("b", {}, label), el("span", {}, sub));
      b.onclick = async () => {
        b.disabled = true;
        try {
          const blob = await api(`/jobs/${id}/cad/file/${name}`, { as: "blob" });
          S.saveBlob(blob, `${(scan.title || "detail").replace(/[^\p{L}\p{N}_-]+/gu, "_")}.${name.split(".")[1]}`);
        } catch (err) { S.toast(err.message); }
        b.disabled = false;
      };
      box.append(b);
    }
    const size = (cad.size_mm || []).map((v) => String(Math.round(v * 100) / 100).replace(".", ",")).join(" × ");
    return section("CAD модел 1:1",
      el("p", { class: "small muted" }, `${size ? size + " мм · " : ""}${cad.measured ?? "?"} измерени размера${cad.estimated ? `, ${cad.estimated} по оценка от снимките` : ""}.`),
      box,
      (cad.elements || []).length ? el("ul", { class: "small muted", style: "margin:12px 0 0;padding-left:18px" }, cad.elements.map((t) => el("li", {}, t))) : null);
  }

  // --- Panel layout: a summary of the numbers that matter, then the rest in four tabs -----------------------------

  // A section of a tab: a small label, then its parts. id: for the parts shown and hidden as a whole.
  const section = (label, ...kids) => el("section", { class: "sec" }, el("div", { class: "sec-label" }, label), ...kids);

  // Size, printer fit, and material and time: filled in by update() and drawCost().
  function summary() {
    const cell = (label, idv) => el("div", {}, el("small", {}, label), el("b", { id: idv }, "…"));
    return el("div", { class: "summary" }, cell("Размер", "sum-size"), cell("Принтер", "sum-fit"), cell("Материал · време", "sum-cost"));
  }

  // The tab chosen last stays chosen (for this browser tab), also on other scans.
  let openTab = () => {};
  function tabs(list) {
    const saved = (() => { try { return sessionStorage.getItem("orbitize.tab"); } catch { return null; } })();
    let active = list.some(([k]) => k === saved) ? saved : list[0][0];
    const bar = el("div", { class: "tab-bar", role: "tablist" });
    const panes = list.map(([key, , kids]) => el("div", { class: "pane", role: "tabpanel", "data-tab": key }, ...kids));
    const buttons = list.map(([key, label]) => el("button", { type: "button", role: "tab", "data-tab": key, onclick: () => openTab(key) }, label));
    bar.append(...buttons);
    openTab = (key) => {
      active = key;
      try { sessionStorage.setItem("orbitize.tab", key); } catch {}
      buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === key)));
      panes.forEach((p) => p.classList.toggle("hidden", p.dataset.tab !== key));
    };
    openTab(active);
    return el("div", { class: "tabs-wrap" }, bar, ...panes);
  }

  // --- 3D viewer ------------------------------------------------------------------------------------------------

  const modelKey = () => `glb/${id}/${scan.created}-${(scan.report && scan.report.seconds) || 0}`;

  function viewer(done) {
    const box = el("div", { class: "viewer", id: "viewer" });
    if (!done) {
      box.append(el("div", { class: "overlay" }, el("div", {},
        el("div", { class: "icon-box", style: "margin:0 auto 12px" }, icon("cube")),
        scan.state === "failed" ? "От това сканиране не можа да се направи модел." : "3D моделът ще се появи тук, когато е готов.")));
      return box;
    }
    box.append(el("div", { class: "overlay", id: "overlay" },
      el("div", { style: "display:grid;gap:14px;justify-items:center" },
        el("div", { class: "spinner" }),
        el("span", { id: "load-text" }, "Подготвяне на 3D модела…"),
        el("div", { class: "load-bar" }, el("i", { id: "load-bar" })))));
    return box;
  }

  async function loadModel() {
    const box = document.getElementById("viewer");
    const overlay = document.getElementById("overlay");
    // The server builds the file before it starts sending: say so if that takes a while.
    const slow = setTimeout(() => {
      const text = document.getElementById("load-text");
      if (text && !text.textContent.includes("%")) text.textContent = "Сървърът подготвя модела. Първото отваряне отнема по-дълго, следващите са мигновени.";
    }, 5000);
    let blob;
    try {
      blob = await S.cachedBlob(modelKey(), `/jobs/${id}/result.glb?texture=true`, {
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
        el("button", { class: "btn btn-secondary btn-sm", onclick: () => { overlay.remove(); box.replaceWith(viewer(true)); loadModel(); } }, "Опитай отново")));
      return;
    }
    clearTimeout(slow);
    modelURL = URL.createObjectURL(blob);
    const mv = el("model-viewer", {
      id: "mv", src: modelURL, alt: scan.title || "3D сканиране", "camera-controls": true, "touch-action": "pan-y",
      "shadow-intensity": "0.7", "shadow-softness": "0.9", exposure: "1.05", "environment-image": "neutral",
      "interaction-prompt": "none", "camera-orbit": "30deg 70deg auto", "rotation-per-second": "20deg",
    });
    // The supports are worked out once the model shows, so they don't slow down its first appearance.
    mv.addEventListener("load", () => { overlay.remove(); gotModel(blob); });
    mv.addEventListener("error", () => { overlay.textContent = "Моделът не може да се покаже в този браузър."; gotModel(blob); });

    const spin = el("button", { class: "tool", title: "Въртене", "aria-label": "Въртене", "aria-pressed": "false" }, icon("rotate"));
    spin.onclick = () => {
      const on = !mv.hasAttribute("auto-rotate");
      mv.toggleAttribute("auto-rotate", on); // only renders all the time while on
      spin.classList.toggle("on", on);
      spin.setAttribute("aria-pressed", String(on));
    };
    const reset = el("button", { class: "tool", title: "Начален изглед", "aria-label": "Начален изглед" }, icon("target"));
    reset.onclick = () => { mv.cameraOrbit = "30deg 70deg auto"; mv.cameraTarget = "auto auto auto"; mv.fieldOfView = "auto"; };
    const full = el("button", { class: "tool", title: "Цял екран", "aria-label": "Цял екран" }, icon("expand"));
    full.onclick = () => (document.fullscreenElement ? document.exitFullscreen() : box.requestFullscreen && box.requestFullscreen());
    const tools = el("div", { class: "viewer-tools" }, spin, reset, box.requestFullscreen ? full : null);

    box.prepend(mv);
    box.append(tools, el("span", { class: "hint" }, "Плъзнете, за да въртите · щипнете или скролнете за увеличение"));
  }

  // --- Title, rename ----------------------------------------------------------------------------------------------

  function titleCard() {
    const h1 = el("h1", {}, scan.title || UNTITLED);
    const editBtn = el("button", { class: "btn btn-ghost btn-sm", title: "Преименувай", "aria-label": "Преименувай" }, icon("edit"));
    const fav = el("button", { class: "btn btn-ghost btn-sm fav-btn" });
    const showFav = () => {
      const on = S.marks.fav(id);
      fav.classList.toggle("on", on);
      fav.title = on ? "Махни от любими" : "Добави в любими";
      fav.setAttribute("aria-label", fav.title);
      fav.setAttribute("aria-pressed", String(on));
    };
    fav.append(icon("star"));
    fav.onclick = () => { S.marks.setFav(id, !S.marks.fav(id)); showFav(); };
    showFav();
    const row = el("div", { class: "title-row" }, h1, editBtn, fav);
    h1.title = "Натиснете, за да преименувате";
    h1.onclick = () => editBtn.click();
    editBtn.onclick = () => {
      const input = el("input", { class: "input", value: scan.title || "", maxlength: "100", placeholder: "Име на сканирането" });
      const save = async () => {
        const title = input.value.trim();
        if (title === (scan.title || "")) return row.replaceChildren(h1, editBtn, fav);
        try {
          scan = await api(`/me/scans/${id}`, { method: "PATCH", json: { title } });
          const list = S.scansCache.get();
          if (list) S.scansCache.set(list.map((s) => (s.id === id ? scan : s)));
          S.toast("Преименувано");
          document.title = `${scan.title || UNTITLED} · ${C.appName}`;
        } catch (err) { S.toast(err.message); }
        h1.textContent = scan.title || UNTITLED;
        row.replaceChildren(h1, editBtn, fav);
      };
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") input.blur();
        if (e.key === "Escape") { input.value = scan.title || ""; input.blur(); }
      });
      input.addEventListener("blur", save);
      row.replaceChildren(input);
      input.focus();
      input.select();
    };
    const r = scan.report || {};
    return el("div", {},
      row,
      el("div", { class: "meta-line" },
        S.stateBadge(scan.state),
        el("span", {}, S.date(scan.created) + (r.seconds ? ` · изградено за ${Math.max(1, Math.round(r.seconds / 60))} мин` : ""))));
  }

  // --- Not ready yet / failed -------------------------------------------------------------------------------------

  const PROGRESS = { uploading: 5, queued: 12, reconstructing: 40, meshing: 70, cleaning: 90 };
  function statusCard() {
    const c = el("div", { class: "card" });
    if (scan.state === "failed" || scan.state === "cancelled") {
      const logBox = el("div", {});
      // The server's log is for the administrators (as in the app); everyone else gets the advice.
      const details = el("button", { class: `btn btn-ghost btn-sm ${S.isAdmin(S.auth.user()) ? "" : "hidden"}` }, "Данни от сървъра");
      details.onclick = async () => {
        details.disabled = true;
        try {
          const text = await api(`/jobs/${id}/log?lines=60`, { as: "text" });
          logBox.replaceChildren(
            scan.error ? el("p", { class: "small muted" }, scan.error) : null,
            el("pre", { class: "log" }, text || "Няма подробности."));
          details.remove();
        } catch (err) { S.toast(err.message); details.disabled = false; }
      };
      c.append(el("h3", {}, "Това сканиране не се получи"),
        el("p", { class: "alert alert-error" }, scan.error && /[\u0400-\u04FF]/.test(scan.error) ? scan.error : "Сървърът не успя да изгради модел от това видео."),
        el("p", { class: "small muted" }, "Опитайте с ново видео от приложението: бавно, отблизо и на светло, с предмета изцяло в кадъра, на еднотонна повърхност."),
        el("div", { class: "row-actions" }, details), logBox);
      return c;
    }
    const pct = PROGRESS[scan.state] || 10;
    c.append(el("h3", {}, scan.state === "uploading" ? "Още се качва" : "3D моделът се изгражда"),
      el("div", { class: "progress" }, el("i", { style: `width:${pct}%` })),
      el("p", { class: "muted small" }, scan.state === "uploading"
        ? "Видеото още се изпраща от телефона. Дръжте приложението отворено, докато приключи."
        : "AI моделът се изгражда за около 12 минути. Страницата се обновява сама."));
    pollTimer = setTimeout(load, 8000);
    return c;
  }

  // --- Does it fit the printer? -----------------------------------------------------------------------------------
  // dimensions_mm are the model's proportions (x, y, z with z up, the flat base at the bottom); the real size comes
  // from the longest side the user chooses. The server scales the download the same way (longest_mm).

  const CUSTOM = "__custom";
  let size = null; // chosen longest side in mm
  const printerKey = () => store.get("printer") || "Bambu Lab A1 mini";

  function bed() {
    if (printerKey() === CUSTOM) {
      const b = store.get("customBed") || { x: 220, y: 220, z: 250 };
      return { name: "вашия принтер", ...b };
    }
    return C.printers.find((p) => p.name === printerKey()) || C.printers[0];
  }

  function proportions() {
    const d = (scan.report && scan.report.dimensions_mm) || [1, 1, 1];
    return d.map((v) => Math.max(v, 1e-6));
  }

  function measure(L) {
    const d = proportions(), k = L / Math.max(...d);
    const [x, y, z] = d.map((v) => v * k);
    const b = bed(), m = C.printerMarginMm || 0;
    const bx = b.x - 2 * m, by = b.y - 2 * m, bz = b.z;
    const straight = x <= bx && y <= by, turned = x <= by && y <= bx;
    const ok = z <= bz && (straight || turned);
    const fitK = Math.min(bz / d[2], Math.max(Math.min(bx / d[0], by / d[1]), Math.min(bx / d[1], by / d[0])));
    const maxL = Math.floor(Math.max(...d) * fitK);
    return { x, y, z, ok, turned: ok && !straight, maxL, b, bx, by };
  }

  function printerSelect() {
    const groups = new Map();
    for (const p of C.printers) {
      if (!groups.has(p.brand)) groups.set(p.brand, el("optgroup", { label: p.brand }));
      groups.get(p.brand).append(el("option", { value: p.name, selected: p.name === printerKey() }, `${p.name} · ${p.x}×${p.y}×${p.z}`));
    }
    return el("select", { class: "input", id: "printer", "aria-label": "Вашият принтер" }, [...groups.values()],
      el("optgroup", { label: "Друг" }, el("option", { value: CUSTOM, selected: printerKey() === CUSTOM }, "Друг принтер (въведете размер)")));
  }

  function printCard() {
    // A CAD part is already in real size (report.real_mm); a scan starts at 100 mm or the most that fits.
    if (size == null) size = store.get("size." + id) || (isCad() && Math.round(scan.report.real_mm * 100) / 100) || Math.max(5, Math.min(100, measure(100).maxL));

    const select = printerSelect();
    const cb = store.get("customBed") || { x: 220, y: 220, z: 250 };
    const customInput = (axis, label) => el("label", {}, el("span", { class: "small muted" }, label),
      el("input", { class: "input", type: "number", min: "20", max: "2000", inputmode: "numeric", value: cb[axis], "data-axis": axis }));
    const custom = el("div", { class: `custom-bed ${printerKey() === CUSTOM ? "" : "hidden"}` },
      customInput("x", "Ширина мм"), customInput("y", "Дълбочина мм"), customInput("z", "Височина мм"));

    const range = el("input", { type: "range", min: "5", max: "300", step: "1", id: "size-range", "aria-label": "Най-дълга страна" });
    const num = el("input", { class: "input", type: "number", min: "1", max: "2000", step: "1", inputmode: "numeric", id: "size-num", "aria-label": "Най-дълга страна в мм" });
    // Quick sizes; "Максимум" is the largest that fits the chosen printer.
    const presets = el("div", { class: "presets", role: "group", "aria-label": "Бързи размери" },
      [50, 75, 100, 150, 200].map((v) => el("button", { type: "button", class: "chip-btn", "data-size": v,
        onclick: () => { setSize(v); store.set("size." + id, size); } }, `${v} мм`)),
      el("button", { type: "button", class: "chip-btn", "data-size": "max",
        onclick: () => { setSize(measure(size).maxL); store.set("size." + id, size); } }, "Максимум"));

    select.onchange = () => {
      store.set("printer", select.value);
      custom.classList.toggle("hidden", select.value !== CUSTOM);
      update();
    };
    custom.addEventListener("input", (e) => {
      const v = Number(e.target.value);
      if (v >= 20) { const b = store.get("customBed") || cb; b[e.target.dataset.axis] = v; store.set("customBed", b); update(); }
    });
    range.oninput = () => setSize(Number(range.value));
    range.onchange = () => store.set("size." + id, size); // saved when let go, not on every step
    num.oninput = () => { const v = Number(num.value); if (v >= 1 && v <= 2000) setSize(v, true); };
    num.onchange = () => store.set("size." + id, size);
    num.onblur = () => (num.value = size);

    return el("div", {},
      section("Принтер", select, custom),
      section("Най-дълга страна",
        el("div", { class: "size-row" }, range, el("div", { class: "unit-input" }, num)), presets,
        el("p", { class: "hint" }, isCad() ? `Реалният размер от вашите мерки: ${String(Math.round(scan.report.real_mm * 100) / 100).replace(".", ",")} мм.` : "За реален размер: измерете предмета с линийка.")),
      section("На плочата",
        el("div", { class: "dims" },
          el("div", { class: "dim" }, el("small", {}, "Ширина"), el("b", { id: "dx" })),
          el("div", { class: "dim" }, el("small", {}, "Дълбочина"), el("b", { id: "dy" })),
          el("div", { class: "dim" }, el("small", {}, "Височина"), el("b", { id: "dz" }))),
        el("div", { id: "fit" }),
        el("div", { class: "plate", id: "plate" })));
  }

  // The slider fires many times a second: redraw at most once per frame.
  let frame = 0, numberTyped = false;
  function setSize(v, fromNumber) {
    size = Math.max(1, Math.min(2000, Math.round(v)));
    numberTyped = !!fromNumber;
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; update(numberTyped); });
  }

  const mm = (v) => String(v >= 100 ? Math.round(v) : Math.round(v * 10) / 10).replace(".", ",") + " мм";

  function update(fromNumber) {
    const m = measure(size);
    const range = document.getElementById("size-range"), num = document.getElementById("size-num");
    if (!range) return;
    range.max = Math.min(2000, Math.max(300, Math.ceil(m.maxL * 1.5)));
    range.value = size;
    if (!fromNumber) num.value = size;
    document.getElementById("dx").textContent = mm(m.x);
    document.getElementById("dy").textContent = mm(m.y);
    document.getElementById("dz").textContent = mm(m.z);
    const sumSize = document.getElementById("sum-size");
    if (sumSize) {
      sumSize.textContent = [m.x, m.y, m.z].map((v) => Math.round(v)).join("×") + " мм";
      const sumFit = document.getElementById("sum-fit");
      sumFit.textContent = m.ok ? "Събира се" : "Твърде голям";
      sumFit.className = m.ok ? "ok" : "bad";
    }
    const fit = document.getElementById("fit");
    fit.replaceChildren(m.ok
      ? el("div", { class: "fit ok" }, icon("check"), el("span", {},
          `Събира се на ${m.b.name}.` + (m.turned ? " Завъртете го на четвърт оборот върху плочата (слайсърът може да го направи)." : "")))
      : el("div", { class: "fit bad" }, icon("alert"), el("span", {},
          `Твърде голям за ${m.b.name}. Най-големият, който се събира, е ${m.maxL} мм по най-дългата страна. `,
          el("a", { href: "#split-card", onclick: (e) => { e.preventDefault(); openTab("print"); document.getElementById("split-card").scrollIntoView({ behavior: "smooth", block: "center" }); } },
            "Или го разделете на части."))));
    document.querySelectorAll("[data-size]").forEach((b) =>
      b.classList.toggle("on", b.dataset.size === "max" ? size === m.maxL : Number(b.dataset.size) === size));
    const best = bestFormat(m.b);
    labelDownloads();
    scheduleSupports();
    drawSplit();
    document.querySelectorAll(".dl").forEach((b) => {
      b.classList.toggle("rec", b.dataset.fmt === best);
      const tag = b.querySelector(".best");
      if (b.dataset.fmt === best && !tag) b.append(el("i", { class: "best" }, "Препоръчано"));
      if (b.dataset.fmt !== best && tag) tag.remove();
    });
    renderPlate(m);
  }

  // Top and side view of the build plate with the model's outline, to scale.
  function renderPlate(m) {
    const box = document.getElementById("plate");
    if (!box || !m) return;
    const NS = "http://www.w3.org/2000/svg";
    const svg = (w, h, parts) => {
      const pad = Math.max(w, h) * 0.06;
      const s = document.createElementNS(NS, "svg");
      s.setAttribute("viewBox", `${-pad} ${-pad} ${w + 2 * pad} ${h + 2 * pad}`);
      s.setAttribute("role", "img");
      for (const [tag, a] of parts) {
        const n = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
        s.append(n);
      }
      return s;
    };
    // Colours from the theme (assets/style.css), so the drawing suits dark and light.
    const color = m.ok ? "var(--accent)" : "var(--danger)", fill = m.ok ? "rgba(99,102,241,.22)" : "rgba(239,68,68,.18)";
    const [fx, fy] = m.turned ? [m.y, m.x] : [m.x, m.y];
    const b = m.b, mg = C.printerMarginMm || 0, sw = Math.max(b.x, b.y, b.z) / 150;
    // Top view: plate centred; the model's footprint in the middle.
    const W = Math.max(b.x, fx), D = Math.max(b.y, fy);
    const ox = (W - b.x) / 2, oy = (D - b.y) / 2;
    const top = svg(W, D, [
      ["rect", { x: ox, y: oy, width: b.x, height: b.y, rx: sw * 4, style: "fill:var(--plate-bg);stroke:var(--plate-line)", "stroke-width": sw }],
      ["rect", { x: ox + mg, y: oy + mg, width: b.x - 2 * mg, height: b.y - 2 * mg, fill: "none", style: "stroke:var(--plate-margin)", "stroke-width": sw, "stroke-dasharray": `${sw * 4} ${sw * 3}` }],
      ["rect", { x: (W - fx) / 2, y: (D - fy) / 2, width: fx, height: fy, rx: sw * 2, fill, style: `stroke:${color}`, "stroke-width": sw * 1.5 }],
    ]);
    // Side view: the printer's height, the model standing on the plate.
    const sx = m.turned ? m.y : m.x, H = Math.max(b.z, m.z), SW = Math.max(b.x, sx);
    const side = svg(SW, H, [
      ["rect", { x: (SW - b.x) / 2, y: H - b.z, width: b.x, height: b.z, style: "fill:var(--plate-bg);stroke:var(--plate-line)", "stroke-width": sw, rx: sw * 2 }],
      ["rect", { x: (SW - sx) / 2, y: H - m.z, width: sx, height: m.z, rx: sw * 2, fill, style: `stroke:${color}`, "stroke-width": sw * 1.5 }],
      ["line", { x1: (SW - b.x) / 2, x2: (SW + b.x) / 2, y1: H, y2: H, style: "stroke:var(--plate-floor)", "stroke-width": sw * 2.5 }],
    ]);
    box.replaceChildren(
      el("figure", {}, top, el("figcaption", {}, `Плочата отгоре · ${b.x}×${b.y} мм`)),
      el("figure", {}, side, el("figcaption", {}, `Отстрани · ${b.z} мм височина`)));
  }

  // --- Supports (assets/supports.js) ------------------------------------------------------------------------------
  // Worked out in the browser from the GLB the viewer loaded, again when the size or the printer changes.
  // The user's choice is kept per scan; with none, supports are on whenever the model needs them.

  let modelURL = null, glbBlob = null, geo = null, sup = null, supTimer = 0, previewURL = null, showing = false;
  let gotModel;
  const glbWait = new Promise((r) => (gotModel = (b) => { glbBlob = b; r(b); }));
  const kind = () => (String(bed().brand || "").startsWith("Смола") ? "resin" : "fdm");
  const supportsOn = () => !!sup && sup.columns.length > 0 && store.get("supports." + id) !== false;

  function geometry() {
    return geo || (geo = glbWait.then(async (b) => {
      const buf = await b.arrayBuffer();
      return { buf, model: Supports.fromGLB(buf) };
    }));
  }

  function scheduleSupports() {
    clearTimeout(supTimer);
    supTimer = setTimeout(planSupports, 250);
  }

  async function planSupports() {
    let g;
    try { g = await geometry(); } catch { return drawSupports(true); }
    if (!sup || sup.size !== size || sup.kind !== kind()) {
      sup = Supports.plan(g.model, size, kind());
      if (showing) showSupports(true);
    }
    drawSupports();
  }

  function supportCard() {
    const check = el("input", { type: "checkbox", id: "sup-on" });
    check.onchange = () => {
      store.set("supports." + id, check.checked);
      if (!check.checked && showing) showSupports(false);
      drawSupports();
    };
    const show = el("button", { class: "btn btn-secondary btn-sm", id: "sup-show", type: "button", onclick: () => showSupports(!showing) });
    const card = section("Подпори",
      el("div", { id: "sup-status" }, el("div", { class: "fit wait" }, el("span", { class: "spinner" }),
        el("span", {}, "Проверка за надвиснали части…"))),
      el("div", { id: "sup-controls", class: "hidden" },
        el("div", { class: "row-split" },
          el("label", { class: "check", style: "margin:0" }, check, el("span", {}, "Във файла за печат")), show)));
    if (sup) setTimeout(drawSupports);
    return card;
  }

  function drawSupports(failed) {
    const status = document.getElementById("sup-status");
    labelDownloads();
    drawCost(failed === true);
    if (!status) return;
    if (failed === true || !sup) {
      if (failed === true) status.replaceChildren(el("div", { class: "fit wait" }, icon("alert"),
        el("span", {}, "Подпорите не могат да се изчислят в този браузър. Ако са нужни, включете ги в слайсъра.")));
      return;
    }
    const n = sup.columns.length, on = supportsOn();
    status.replaceChildren(!n
      ? el("div", { class: "fit ok" }, icon("check"), el("span", {}, "Не са нужни подпори."))
      : on
        ? el("div", { class: "fit ok" }, icon("check"), el("span", {},
            `Дървовидни подпори: ${n} ${n === 1 ? "точка" : "точки"} на допир, ${sup.stands} ${sup.stands === 1 ? "ствол" : "ствола"}.`))
        : el("div", { class: "fit warn" }, icon("alert"), el("span", {}, "Има надвиснали части. Без подпори може да увиснат.")));
    document.getElementById("sup-controls").classList.toggle("hidden", !n);
    document.getElementById("sup-on").checked = on;
    const show = document.getElementById("sup-show");
    show.classList.toggle("hidden", !on);
    show.replaceChildren(icon(showing ? "eyeoff" : "eye"), showing ? "Скрий в 3D" : "Покажи в 3D");
  }

  // Swaps the viewer's model for the same one with its supports (violet), and back.
  async function showSupports(on) {
    const mv = document.getElementById("mv");
    const old = previewURL;
    showing = on && !!mv && !!sup;
    if (showing && cutShown) hideCut(false);
    if (showing) {
      const g = await geometry();
      previewURL = URL.createObjectURL(Supports.preview(g.buf, sup));
      mv.src = previewURL;
    } else {
      previewURL = null;
      if (mv) mv.src = modelURL;
    }
    if (old) setTimeout(() => URL.revokeObjectURL(old), 5000);
    drawSupports();
  }

  // The file with supports, made here; null when it should come from the server as it is.
  async function supportedFile(fmt) {
    if (fmt === "glb" || !glbBlob) return null;
    let g;
    try { g = await geometry(); } catch { return null; }
    if (!sup || sup.size !== size || sup.kind !== kind()) { sup = Supports.plan(g.model, size, kind()); drawSupports(); }
    if (!supportsOn()) return null;
    await new Promise((r) => setTimeout(r, 30)); // let "Подготвя се…" show first
    const mesh = Supports.build(g.model, sup);
    return fmt === "stl" ? Supports.stl(mesh) : fmt === "obj" ? Supports.obj(mesh) : Supports.threeMF(mesh, bed());
  }

  function labelDownloads() {
    const extra = supportsOn() ? " · с подпори" : "";
    document.querySelectorAll(".dl").forEach((b) => {
      b.querySelector("span").textContent = b.dataset.note + ` · ${size} мм` + (b.dataset.fmt === "glb" ? "" : extra);
    });
  }

  // --- Material, price and time -----------------------------------------------------------------------------------
  // A rough guess from the model's volume and surface at the chosen size, supports included when they are on, so the
  // user knows before slicing. FDM: PLA, 0.2 mm layers, 0.8 mm walls, 15% infill; resin: solid, 0.05 mm layers.
  // Printers from before about 2023 move far slower than the newer fast ones (Bambu, K1, MK4…).

  const SLOW = /Ender-3( \/| V2| S1| V3 SE)|Ender-5 Plus|CR-10|Neptune 3|Vyper|SV06|Genius|Sidewinder X2|Ultimaker|Raise3D|Snapmaker|Kingroon|Tronxy|MK3S|MINI\+|Creator 4|X-Smart|вашия/;
  const MATERIAL = {
    fdm: { name: "PLA", density: 1.24, price: 20 },
    resin: { name: "смола", density: 1.1, price: 30 },
  };
  const price = () => store.get("price." + kind()) || MATERIAL[kind()].price;

  function estimate(model) {
    const k = kind(), m = measure(size);
    const s = size / Math.max(...[0, 1, 2].map((i) => model.max[i] - model.min[i]));
    let V = model.volume * s ** 3, A = model.area * s ** 2;
    if (supportsOn()) { V += sup.volume; A += sup.area; }
    let used, sec;
    if (k === "resin") {
      used = V;
      sec = (m.z / 0.05) * 6; // each layer: lit, then lifted off the film
    } else {
      const slow = SLOW.test(m.b.name), shell = Math.min(V, A * 0.8);
      used = shell + (V - shell) * 0.15;
      sec = used / (slow ? 1.8 : 4.5) + (m.z / 0.2) * (slow ? 4 : 2); // mm³ a second, and each layer's moves
    }
    const grams = (used / 1000) * MATERIAL[k].density;
    return { grams, cost: (grams / 1000) * price(), sec };
  }

  const hours = (sec) => {
    const t = Math.max(1, Math.round(sec / 60)), h = Math.floor(t / 60), r = t % 60;
    return h >= 10 ? `${Math.round(t / 60)} ч` : h ? `${h} ч` + (r ? ` ${r} мин` : "") : `${r} мин`;
  };

  function costCard() {
    const input = el("input", { class: "input", type: "number", min: "1", max: "500", step: "0.5", inputmode: "decimal",
      id: "price", "aria-label": "Цена за килограм в евро" });
    input.oninput = () => {
      const v = Number(input.value);
      if (v > 0 && v <= 500) { store.set("price." + kind(), v); drawCost(); }
    };
    if (sup) setTimeout(drawCost);
    return section("Материал, цена и време",
      el("div", { class: "dims" },
        el("div", { class: "dim" }, el("small", {}, "Материал"), el("b", { id: "c-g" }, "…")),
        el("div", { class: "dim" }, el("small", {}, "Цена"), el("b", { id: "c-eur" }, "…")),
        el("div", { class: "dim" }, el("small", {}, "Време"), el("b", { id: "c-time" }, "…"))),
      el("div", { class: "row-split" }, el("span", { id: "price-label", class: "small muted" }), el("div", { class: "unit-input eur price" }, input)),
      el("p", { class: "hint", id: "cost-note" }));
  }

  async function drawCost(failed) {
    const g = document.getElementById("c-g");
    if (!g) return;
    const k = kind(), mat = MATERIAL[k], input = document.getElementById("price");
    document.getElementById("price-label").textContent = `${k === "resin" ? "Смола" : "Филамент"}, € за кг`;
    if (document.activeElement !== input) input.value = price();
    const note = document.getElementById("cost-note");
    let geo;
    if (failed || !sup || sup.size !== size) {
      if (failed) note.textContent = "Не може да се изчисли в този браузър.";
      return;
    }
    try { geo = await geometry(); } catch { return; }
    const e = estimate(geo.model), n = (v, d) => v.toLocaleString("bg-BG", { maximumFractionDigits: d });
    g.textContent = `${n(e.grams, e.grams < 10 ? 1 : 0)} г`;
    document.getElementById("c-eur").textContent = `${n(e.cost, 2)} €`;
    document.getElementById("c-time").textContent = hours(e.sec);
    const sumCost = document.getElementById("sum-cost");
    if (sumCost) sumCost.textContent = `${n(e.grams, e.grams < 10 ? 1 : 0)} г · ${hours(e.sec)}`;
    note.textContent = "Приблизително: " + (k === "resin" ? "плътен, слой 0,05 мм" : `${mat.name}, слой 0,2 мм, 15% пълнеж`) +
      (supportsOn() ? ", с подпорите." : ".");
  }

  // --- Too big: in pieces (assets/split.js) -----------------------------------------------------------------------
  // Worked out in the browser when asked (it takes a few seconds on a big scan), again after the size or printer
  // changes. The pieces are cut from the model as it is, with holes for pins across every cut.

  let cut = null, cutKey = "", cutBusy = false, cutShown = false, cutURL = null, cutCount = null;
  const plate = () => {
    const b = bed(), mg = C.printerMarginMm || 0;
    return { x: b.x - 2 * mg - 1, y: b.y - 2 * mg - 1, z: b.z - 1 };
  };
  const splitKey = () => [size, kind(), JSON.stringify(plate())].join("|");
  const cutAngle = () => (kind() === "resin" ? 30 : 35);

  function splitCard() {
    const show = el("button", { class: "btn btn-secondary btn-sm", id: "cut-show", type: "button", onclick: () => (cutShown ? hideCut(true) : showCut()) });
    const get = el("button", { class: "btn btn-primary btn-sm", id: "cut-get", type: "button", onclick: () => downloadParts(get) },
      icon("download"), "Изтегли частите (ZIP)");
    setTimeout(drawSplit);
    const sec = section("Разделяне на части",
      el("div", { id: "cut-status" }),
      el("div", { class: "row-actions hidden", id: "cut-actions", style: "margin-top:12px" }, show, get),
      el("p", { class: "hint" }, "Всяка част се събира на плочата, с дупки за щифтове по срезовете."));
    sec.id = "split-card";
    sec.classList.add("hidden");
    return sec;
  }

  async function drawSplit() {
    const card = document.getElementById("split-card");
    if (!card) return;
    const m = measure(size), key = splitKey();
    if (cutShown && cutKey !== key) hideCut(true);
    card.classList.toggle("hidden", m.ok && !cutShown);
    if (m.ok && !cutShown) return;
    const status = document.getElementById("cut-status"), actions = document.getElementById("cut-actions");
    const say = (cls, ic, text) => status.replaceChildren(el("div", { class: `fit ${cls}` }, ic, el("span", {}, text)));
    if (cutBusy) return;
    let g;
    try { g = await geometry(); } catch { return say("wait", icon("alert"), "Разделянето не работи в този браузър. Слайсърът (Bambu Studio, PrusaSlicer, Orca) също може да реже модела."); }
    if (!cutCount || cutCount.key !== key) cutCount = { key, n: Split.layoutFor(g.model, size, plate()).count };
    if (splitKey() !== key || cutBusy) return; // changed while this ran
    const n = cutCount.n;
    if (!n) {
      actions.classList.add("hidden");
      return say("bad", icon("alert"), "Толкова голям ще е над 8 части по една страна. Намалете размера отгоре.");
    }
    actions.classList.remove("hidden");
    document.getElementById("cut-show").replaceChildren(icon(cutShown ? "eyeoff" : "eye"), cutShown ? "Скрий частите" : "Покажи частите в 3D");
    if (cut && cutKey === key) {
      const pins = cut.pins ? ` и ${cut.pins.count} щифта Ø${cut.pinD} мм (2 от тях резервни)` : "";
      return say("ok", icon("check"), `Готово: ${cut.parts.length} ${cut.parts.length === 1 ? "част" : "части"}${pins}. Всяка се събира на ${m.b.name}.`);
    }
    say("ok", icon("split"), `За ${m.b.name} моделът ще е на ${n} ${n === 1 ? "част" : "части"} в този размер.`);
  }

  // The pieces for the current size and printer (cut once, kept until either changes).
  async function pieces() {
    const key = splitKey();
    if (cut && cutKey === key) return cut;
    const status = document.getElementById("cut-status");
    cutBusy = true;
    status.replaceChildren(el("div", { class: "fit wait" }, el("span", { class: "spinner" }), el("span", {}, "Разделяне на части… Може да отнеме няколко секунди.")));
    try {
      const g = await geometry();
      await new Promise((r) => setTimeout(r, 30)); // let the message show first
      cut = await Split.split(g.model, size, plate(), kind());
      cutKey = key;
      return cut;
    } catch (err) {
      const text = err.code === "open"
        ? "Повърхността на модела има пролуки, затова не може да се реже тук. Слайсърът може: инструментът Cut в Bambu Studio, PrusaSlicer или Orca."
        : err.code === "many" ? "Толкова голям ще е над 8 части по една страна. Намалете размера отгоре."
          : "Разделянето не успя в този браузър. Опитайте в Chrome или Edge на компютър.";
      status.replaceChildren(el("div", { class: "fit bad" }, icon("alert"), el("span", {}, text)));
      return null;
    } finally {
      cutBusy = false;
    }
  }

  // The pieces pulled apart in the viewer, numbered as in the files.
  async function showCut() {
    const r = await pieces();
    const mv = document.getElementById("mv");
    if (!r || !mv) return drawSplit();
    if (showing) await showSupports(false);
    const { blob, labels } = Split.preview(r);
    const old = cutURL;
    cutURL = URL.createObjectURL(blob);
    mv.querySelectorAll(".part-tag").forEach((t) => t.remove());
    mv.src = cutURL;
    mv.append(...labels.map((l) => el("button", { class: "part-tag", slot: `hotspot-part-${l.number}`, type: "button", tabindex: "-1",
      "data-position": l.at.map((v) => `${v}m`).join(" "), "data-normal": "0m 1m 0m" }, String(l.number))));
    if (old) setTimeout(() => URL.revokeObjectURL(old), 5000);
    cutShown = true;
    drawSplit();
    document.getElementById("viewer").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideCut(restore) {
    const mv = document.getElementById("mv");
    cutShown = false;
    if (mv) {
      mv.querySelectorAll(".part-tag").forEach((t) => t.remove());
      if (restore) mv.src = modelURL;
    }
    if (cutURL) { const u = cutURL; cutURL = null; setTimeout(() => URL.revokeObjectURL(u), 5000); }
    if (restore) drawSplit();
  }

  // Where a piece goes, in words, from its place in its layer's grid.
  function where(p) {
    const [i, j, k] = p.idx, [nx, ny, nz] = p.n, words = [];
    if (nz > 1) words.push(k === 0 ? "долу" : k === nz - 1 ? "горе" : "в средата по височина");
    if (nx > 1) words.push(i === 0 ? "отляво" : i === nx - 1 ? "отдясно" : "в средата отляво надясно");
    if (ny > 1) words.push(j === 0 ? "отпред" : j === ny - 1 ? "отзад" : "в средата отпред назад");
    return words.join(", ") || "цялата";
  }

  async function downloadParts(btn) {
    btn.disabled = true;
    const label = btn.lastChild;
    try {
      const r = await pieces();
      if (!r) return;
      const name = fileName(), withSupports = store.get("supports." + id) !== false;
      const files = [];
      for (const p of r.parts) {
        label.textContent = `Част ${p.number} от ${r.parts.length}…`;
        await new Promise((res) => setTimeout(res));
        const f = Split.forPrint(p, r.dowels, r.holeR, plate(), cutAngle());
        let mesh = f.model;
        if (withSupports) {
          const plan = Supports.plan(f.model, f.longest, kind(), f.skip);
          if (plan.columns.length) mesh = Supports.build(f.model, plan);
        }
        files.push([`${name}-част-${p.number}.stl`, new Uint8Array(await Supports.stl(mesh).arrayBuffer())]);
      }
      if (r.pins) files.push([`щифтове-${r.pins.count}-бр.stl`, new Uint8Array(await Supports.stl(r.pins).arrayBuffer())]);
      const pins = r.pins
        ? [`• щифтове-${r.pins.count}-бр.stl: ${r.pins.count} щифта Ø${r.pinD} мм × ${r.pins.length} мм (2 от тях са резервни).`]
        : [];
      const text = [
        `${scan.title || UNTITLED}: ${size} мм по най-дългата страна, на ${r.parts.length} части за ${bed().name}.`, "",
        "Какво има тук",
        `• ${name}-част-1.stl … част-${r.parts.length}.stl: частите. Всяка вече е обърната на най-добрата страна за печат` +
          (withSupports ? " и има подпори, където трябват." : "."),
        ...pins, "",
        "Печат",
        "1. Отпечатайте частите една по една, както са. Не ги мащабирайте: размерът е точен.",
        ...(r.pins ? ["2. Отпечатайте щифтовете, по-добре със 100% пълнеж."] : []), "",
        "Сглобяване",
        r.pins ? "3. Наредете частите на сухо с щифтовете, без лепило. Ако щифт влиза много стегнато, шлифовайте го леко."
          : "3. Наредете частите на сухо, без лепило.",
        r.pins ? "4. Сложете малко лепило в дупките (секундно или двукомпонентно епоксидно), вкарайте щифтовете и притиснете частите за минута."
          : "4. Намажете срезовете с лепило (секундно или двукомпонентно епоксидно) и притиснете частите за минута.",
        "5. Ако искате шевовете да не се виждат: шпакловка, шкурка и боя.", "",
        "Къде отива всяка част (гледано отпред)",
        ...r.parts.map((p) => `Част ${p.number}: ${where(p)}`), "",
      ].join("\r\n");
      files.push(["Как-да-сглобите.txt", new TextEncoder().encode("\uFEFF" + text)]);
      label.textContent = "Архивиране…";
      S.saveBlob(await Supports.zip(files, "application/zip"), `${name}-${size}mm-на-${r.parts.length}-части.zip`);
      drawSplit();
    } catch (err) {
      S.toast(err.message || "Частите не можаха да се направят.");
    } finally {
      label.textContent = "Изтегли частите (ZIP)";
      btn.disabled = false;
    }
  }

  // --- Downloads --------------------------------------------------------------------------------------------------

  const FORMATS = [
    ["3mf", "3MF", "За Bambu, Prusa, Orca"],
    ["stl", "STL", "Работи във всеки слайсър"],
    ["obj", "OBJ", "За 3D програми"],
    ["glb", "GLB", "С цветове, за Blender и уеб"],
  ];

  // Bambu Studio and PrusaSlicer open 3MF best; STL works in every other slicer, resin ones included.
  function bestFormat(b) {
    return b.brand === "Bambu Lab" || b.brand === "Prusa" ? "3mf" : "stl";
  }

  // Latin and Cyrillic letters, digits, - and _ only: safe as a file name everywhere.
  const fileName = () => (scan.title || "scan").replace(/[^\p{L}\p{N}_\- ]+/gu, "").trim().replace(/\s+/g, "-").slice(0, 50) || "scan";

  function downloadCard() {
    const name = fileName();
    const buttons = FORMATS.map(([fmt, label, note]) => {
      const b = el("button", { class: "btn btn-secondary format-btn dl", "data-note": note, "data-fmt": fmt },
        el("b", {}, label), el("span", {}, note));
      b.onclick = async () => {
        b.disabled = true;
        const head = b.querySelector("b");
        head.textContent = "Подготвя се…";
        try {
          const local = await supportedFile(fmt);
          const glb = fmt === "glb" && glbBlob ? Supports.scaledGLB(await glbBlob.arrayBuffer(), size) : null;
          if (local) S.saveBlob(local, `${name}-${size}mm-с-подпори.${fmt}`);
          else if (glb) S.saveBlob(glb, `${name}-${size}mm.glb`); // the viewer's copy, resized here: no new download
          else {
            const q = `longest_mm=${size}` + (fmt === "glb" ? "&texture=true" : "");
            const blob = await api(`/jobs/${id}/result.${fmt}?${q}`, {
              as: "blob", onProgress: (p) => (head.textContent = `Изтегляне ${Math.round(p * 100)}%`),
            });
            S.saveBlob(blob, `${name}-${size}mm.${fmt}`);
          }
        } catch (err) { S.toast(err.message || "Файлът не можа да се направи."); }
        head.textContent = label;
        b.disabled = false;
      };
      return b;
    });
    return section("Изтегляне",
      el("div", { class: "formats" }, buttons),
      el("p", { class: "hint" }, "В избрания размер, в мм, стъпил на основата си. Подпорите, ако са включени, са в 3MF, STL и OBJ."));
  }

  // --- Notes (this browser only) -----------------------------------------------------------------------------------

  function notesCard() {
    const box = el("textarea", { class: "input notes", rows: "3", maxlength: "2000", placeholder: "Напр. за какво е, кой материал, какво да оправя следващия път…",
      "aria-label": "Бележки" });
    box.value = S.marks.note(id);
    let t;
    box.oninput = () => { clearTimeout(t); t = setTimeout(() => S.marks.setNote(id, box.value), 300); };
    box.onblur = () => { clearTimeout(t); S.marks.setNote(id, box.value); };
    return section("Бележки", box, el("p", { class: "hint" }, "Пазят се само в този браузър."));
  }

  // --- Community (community.html) -----------------------------------------------------------------------------------

  function communityCard() {
    const c = el("section", { class: "sec" });
    const setPublished = (v) => {
      scan.published = v;
      const list = S.scansCache.get();
      if (list) S.scansCache.set(list.map((s) => (s.id === id ? { ...s, published: v } : s)));
    };
    const draw = () => {
      const on = !!scan.published;
      const parts = [el("div", { class: "sec-label" }, "Общност"),
        el("p", { class: "hint", style: "margin:0 0 12px" }, on
          ? "Моделът е публикуван. Всеки с профил може да го види, хареса и изтегли."
          : `Публикувайте модела, за да могат всички с профил да го видят и изтеглят. Ще се вижда с името ви и размер ${size} мм.`)];
      if (on) {
        const off = el("button", { class: "btn btn-ghost btn-sm" }, "Махни от общността");
        off.onclick = async () => {
          off.disabled = true;
          try { await api(`/me/scans/${id}/publish`, { method: "DELETE" }); setPublished(null); S.toast("Махнато от общността"); }
          catch (err) { S.toast(err.message); }
          draw();
        };
        parts.push(el("div", { class: "row-actions" },
          el("a", { class: "btn btn-secondary btn-sm", href: `post.html?id=${encodeURIComponent(id)}` }, icon("globe"), "Виж в общността"), off));
      } else {
        const pub = el("button", { class: "btn btn-primary btn-sm" }, icon("globe"), "Публикувай в общността");
        pub.onclick = async () => {
          if (!scan.title) return S.toast("Първо дайте име на модела (горе, до заглавието).");
          const ok = await S.confirmBox({ title: "Да публикуваме ли модела?", ok: "Публикувай",
            text: `„${scan.title}“ ще се вижда от всички с профил, с вашето име и размер ${size} мм. Може да го махнете по всяко време.` });
          if (!ok) return;
          pub.disabled = true;
          try {
            const r = await api(`/me/scans/${id}/publish`, { method: "POST", json: { title: scan.title, longest_mm: size } });
            setPublished(r.published);
            S.toast("Публикувано в общността");
          } catch (err) { S.toast(err.message); }
          draw();
        };
        parts.push(pub);
      }
      c.replaceChildren(...parts);
    };
    draw();
    return c;
  }

  // --- Share link -------------------------------------------------------------------------------------------------

  function shareCard() {
    const c = el("section", { class: "sec" });
    const draw = () => {
      const link = scan.share ? `${C.apiUrl}/s/${scan.share}` : null;
      const toggle = el("button", { class: `btn ${link ? "btn-ghost" : "btn-primary"} btn-sm` }, link ? "Спри споделянето" : "Създай линк за споделяне");
      toggle.onclick = async () => {
        toggle.disabled = true;
        try {
          const r = await api(`/me/scans/${id}/share`, { method: link ? "DELETE" : "POST" });
          scan.share = r.share;
          const list = S.scansCache.get();
          if (list) S.scansCache.set(list.map((s) => (s.id === id ? { ...s, share: r.share } : s)));
          S.toast(link ? "Линкът вече не работи" : "Линкът е създаден");
        } catch (err) { S.toast(err.message); }
        draw();
      };
      const parts = [el("div", { class: "sec-label" }, "Споделяне"),
        el("p", { class: "hint", style: "margin:0 0 12px" }, link
          ? "Всеки с този линк може да върти модела в браузъра си и да види името ви. Никой друг не може да го намери."
          : "Създайте линк, за да могат други да видят този модел в браузъра си. Другите ви сканирания остават лични.")];
      if (link) {
        const input = el("input", { class: "input", value: link, readonly: true, "aria-label": "Линк за споделяне", onfocus: (e) => e.target.select() });
        const copy = el("button", { class: "btn btn-primary btn-sm" }, "Копирай");
        copy.onclick = async () => {
          try { await navigator.clipboard.writeText(link); S.toast("Линкът е копиран"); }
          catch { input.select(); S.toast("Натиснете Ctrl+C, за да копирате"); }
        };
        parts.push(el("div", { class: "share-box" }, input, copy),
          el("div", { class: "row-actions", style: "margin-top:10px" },
            el("a", { class: "btn btn-secondary btn-sm", href: link, target: "_blank", rel: "noopener" }, icon("link"), "Отвори"),
            navigator.share ? el("button", { class: "btn btn-secondary btn-sm",
              onclick: () => navigator.share({ title: scan.title || "3D сканиране", url: link }).catch(() => {}) }, icon("share"), "Изпрати") : null,
            toggle));
      } else parts.push(toggle);
      c.replaceChildren(...parts);
    };
    draw();
    return c;
  }

  // --- Delete -----------------------------------------------------------------------------------------------------

  function dangerCard() {
    const b = el("button", { class: "btn btn-danger btn-sm" }, icon("trash"), "Изтрий сканирането");
    b.onclick = async () => {
      const ok = await S.confirmBox({ title: "Да изтрием ли сканирането?", text: "Моделът и линкът за споделяне се изтриват завинаги, и от приложението.", ok: "Изтрий", danger: true });
      if (!ok) return;
      try {
        await api(`/jobs/${id}`, { method: "DELETE" });
        const list = S.scansCache.get();
        if (list) S.scansCache.set(list.filter((s) => s.id !== id));
        await S.forgetFiles(id);
        location.replace("scans.html");
      } catch (err) { S.toast(err.message); }
    };
    return section("Изтриване", el("p", { class: "hint", style: "margin:0 0 12px" }, "Моделът и линкът се изтриват завинаги."), b);
  }

  scan = fromCache();
  if (scan) render();
  load();
})();

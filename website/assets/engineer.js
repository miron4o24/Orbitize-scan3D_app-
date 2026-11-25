// Инженерният режим в сайта: същите стъпки и същият сървър като в приложението (src/app/engineer.tsx, backend app/cad).
// Снимките казват какво има (очертания, стъпала, отвори), мерките казват колко е голямо, а моделът се строи само от
// числата. 1. Снимки (до 6, или отгоре и отдолу за плосък детайл), качват се веднага. 2. Проверка на очертанията
// (поправка с докосване). 3. Общите размери, всеки с линия върху снимката. 4. "Какво виждам": 3D по оценките,
// елементите, обръщане на изгледите. 5. Подробните мерки, по една. 6. Моделът: страницата на сканирането.
(function () {
  if (!S.requireAuth()) return;
  S.page({ title: "Инженерен режим", active: "engineer" });
  const { el, icon, api } = S;
  const root = document.getElementById("root");

  const SHOTS = [
    { photo: "top", title: "Отгоре", how: "Сложете предмета на масата, както стои обикновено, с дългата страна наляво-надясно.", required: true },
    { photo: "front", title: "Отпред", how: "Претърколете го от себе си: страната, която гледаше към вас, е нагоре." },
    { photo: "back", title: "Отзад", how: "Задната страна нагоре: претърколете го към себе си." },
    { photo: "right", title: "Отдясно", how: "Изправете го на левия край: десният край е нагоре." },
    { photo: "left", title: "Отляво", how: "Изправете го на десния край: левият край е нагоре." },
    { photo: "bottom", title: "Отдолу", how: "Обърнете го с дъното нагоре." },
  ];
  const TITLE = Object.fromEntries(SHOTS.map((x) => [x.photo, x.title]));
  const GUESS_ASK = 1.25; // мярка, различна от снимката повече от толкова пъти: "сигурни ли сте?"

  // Състоянието на работата (като в приложението).
  let flat = false;
  let jobId = null;
  let creating = null;
  const photos = {}; // photo -> { url, status: "uploading" | "ok" | "error", file }
  let queue = Promise.resolve();
  let plan = null, state = null;
  let sizes = {}, oi = 0;
  let asked = [], answers = {}, qi = 0;
  let finished = false;

  const order = () => (flat ? ["top", "bottom"] : SHOTS.map((x) => x.photo));
  const mmText = (v) => String(Math.round(v * 100) / 100).replace(".", ",");
  function mmOf(t, zero) {
    const v = Number(String(t).trim().replace(",", "."));
    return String(t).trim() && Number.isFinite(v) && (zero ? v >= 0 : v > 0) && v <= 1000 ? v : undefined;
  }

  // Излизане без готов детайл: задачата на сървъра се изтрива (иначе би останала като недовършено сканиране).
  addEventListener("pagehide", () => {
    if (jobId && !finished) fetch(`${S.C.apiUrl}/jobs/${jobId}`, { method: "DELETE", keepalive: true, headers: { Authorization: `Bearer ${S.auth.token()}` } });
  });

  async function ensureJob() {
    if (jobId) return jobId;
    creating = creating || api("/jobs", { method: "POST" }).then((j) => (jobId = j.id));
    return creating;
  }

  // Картинките от сървъра искат сесийния ключ: изтеглят се и се показват от паметта на браузъра.
  async function picture(path) {
    return URL.createObjectURL(await api(path, { as: "blob" }));
  }

  // Снимка от телефона или компютъра → JPEG (сървърът приема JPEG до 15 MB). JPEG до 14 MB отива както е.
  async function asJpeg(file) {
    if (file.type === "image/jpeg" && file.size < 14 * 1024 * 1024) return file;
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const k = Math.min(1, 4000 / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    return new Promise((ok, bad) => c.toBlob((b) => (b ? ok(b) : bad(new Error("jpeg"))), "image/jpeg", 0.92));
  }

  function upload(photo, file) {
    photos[photo] = { url: URL.createObjectURL(file), status: "uploading", file };
    drawPhotos();
    queue = queue.then(async () => {
      try {
        const jpeg = await asJpeg(file);
        const id = await ensureJob();
        await api(`/jobs/${id}/photos/${photo}.jpg`, { method: "PUT", body: jpeg, headers: { "Content-Type": "image/jpeg" } });
        photos[photo].status = "ok";
        api(`/jobs/${id}/cad/prepare`, { method: "POST" }).catch(() => {}); // само за да е по-бързо накрая
      } catch (err) {
        photos[photo].status = "error";
        photos[photo].error = err.message || "Снимката не можа да се отвори.";
      }
      drawPhotos();
    });
  }

  // --- Общи части на екраните ------------------------------------------------------------------------------------

  function head(kicker, title, sub) {
    return el("div", { class: "eng-head" }, el("div", { class: "kicker amber" }, el("i"), kicker), el("h1", {}, title), sub ? el("p", { class: "muted" }, sub) : null);
  }
  const errorBox = (text) => (text ? el("p", { class: "alert alert-error" }, text) : null);
  function show(...kids) {
    root.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
    scrollTo({ top: 0 });
  }
  function waiting(title, sub) {
    show(el("div", { class: "card center eng-wait" }, el("div", { class: "spinner" }), el("h3", {}, title), el("p", { class: "muted" }, sub)));
  }
  function progress(i, n) {
    return el("div", { class: "eng-progress" }, el("i", { style: `width:${((i + 1) / n) * 100}%` }));
  }

  // --- 0. Въведение: обемен или плосък детайл --------------------------------------------------------------------

  const STEPS = [
    ["Снимки от всички страни", "До 6 снимки отгоре: за всяка обръщате предмета с друга страна нагоре. Плосък детайл: само отгоре и отдолу. Снимайте отвисоко, перпендикулярно на масата, на еднотонен фон."],
    ["Общите размери", "Дължина, ширина, височина: оранжева линия върху вашата снимка показва какво да измерите."],
    ["Какво виждам", "Сървърът разчита очертанията, стъпалата и отворите; проверявате ги в 3D и добавяте пропуснатото."],
    ["Мерките, по една", "Всяка с линия върху снимката. Каквото не можете да измерите, остава с оценката от снимките."],
    ["CAD файловете", "STEP за SolidWorks, Fusion, FreeCAD, Onshape; кодът на модела; STL; чертеж."],
  ];

  function intro() {
    const pick = (isFlat) => () => { flat = isFlat; drawPhotos(); };
    show(
      head("Инженерен режим", "Детайл 1:1 от снимки и мерки", "Снимките казват каква е формата, вашите мерки казват колко е голяма. Моделът е само от вашите числа."),
      el("div", { class: "eng-grid" },
        el("div", { class: "card eng-steps" }, el("div", { class: "sec-label" }, "Как става"),
          el("ol", { class: "pipe amber-pipe" }, STEPS.map(([t, d]) => el("li", {}, el("b", {}, t), el("span", {}, d))))),
        el("div", { class: "eng-choose" },
          choice("cube", "Обемен предмет", "Снимки от всички страни (до 6): кутия, флашка, капачка, вал", pick(false), true),
          choice("note", "Плосък детайл", "Само отгоре и отдолу: кука, пластина, шайба, уплътнение. Една дебелина навсякъде", pick(true)),
          el("p", { class: "small muted" }, "Нужни са ви снимки на предмета и нещо, с което да мерите: линийка или шублер."))));
  }
  function choice(ic, title, sub, onclick, primary) {
    return el("button", { type: "button", class: `eng-choice ${primary ? "primary" : ""}`, onclick }, icon(ic), el("span", {}, el("b", {}, title), el("small", {}, sub)));
  }

  // --- 1. Снимки ---------------------------------------------------------------------------------------------------

  let photoError = null;
  function drawPhotos() {
    const shots = SHOTS.filter((x) => order().includes(x.photo));
    const busy = Object.values(photos).some((p) => p.status === "uploading");
    const grid = el("div", { class: "shot-grid" }, shots.map((x, k) => {
      const p = photos[x.photo];
      const input = el("input", { type: "file", accept: "image/*", capture: "environment", class: "hidden" });
      input.onchange = () => input.files[0] && upload(x.photo, input.files[0]);
      const pic = p ? el("img", { src: p.url, alt: x.title }) : el("div", { class: "shot-empty" }, icon("plus"), el("span", {}, "Добави снимка"));
      const status = p ? (p.status === "uploading" ? el("span", { class: "badge busy" }, el("span", { class: "dot" }), "Качва се")
        : p.status === "ok" ? el("span", { class: "badge ok" }, el("span", { class: "dot" }), "Качена")
          : el("span", { class: "badge bad" }, el("span", { class: "dot" }), "Грешка")) : null;
      return el("label", { class: `shot ${p ? "has" : ""}` }, input,
        el("div", { class: "shot-pic" }, pic, status),
        el("div", { class: "shot-text" }, el("b", {}, `${k + 1}. ${x.title}${x.required ? "" : " · по желание"}`), el("span", { class: "small muted" }, flat && x.photo === "bottom" ? "Обърнете го: дъното нагоре." : x.how),
          p && p.status === "error" ? el("span", { class: "small", style: "color:var(--danger)" }, p.error) : null));
    }));
    const next = el("button", { class: "btn btn-primary amber-btn btn-lg", disabled: !photos.top || busy }, icon("ruler"), busy ? "Качва се…" : "Напред: прочети снимките");
    next.onclick = read;
    show(head("Инженерен режим · снимки", flat ? "Отгоре и отдолу" : "Снимки от всички страни",
        "Снимайте отвисоко, с телефона успореден на масата, предметът цял в кадъра. Отгоре е задължителна; колкото повече страни, толкова по-точно."),
      grid, errorBox(photoError),
      el("div", { class: "row-actions" }, next, el("button", { class: "btn btn-ghost", onclick: intro }, "Назад")));
  }

  async function read() {
    photoError = null;
    waiting("Чета снимките…", "Намирам предмета на всяка снимка. До половин минута.");
    try {
      await queue;
      const id = await ensureJob();
      for (const [photo, p] of Object.entries(photos)) {
        if (p.status !== "error") continue;
        await api(`/jobs/${id}/photos/${photo}.jpg`, { method: "PUT", body: await asJpeg(p.file), headers: { "Content-Type": "image/jpeg" } });
        p.status = "ok";
      }
      plan = await api(`/jobs/${id}/cad/read`, { method: "POST", json: { flat } });
      oi = 0;
      outlines();
    } catch (err) {
      photoError = err.message;
      drawPhotos();
    }
  }

  // --- 2. Проверка на очертанията ----------------------------------------------------------------------------------

  let outlineChanged = false;
  const stamps = {};
  async function outlines(error) {
    const grid = el("div", { class: "shot-grid" }, plan.photos.map((p) => {
      const img = el("img", { alt: TITLE[p] });
      picture(`/jobs/${jobId}/cad/outline/${p}.jpg?s=${stamps[p] || 0}`).then((u) => (img.src = u)).catch(() => {});
      return el("button", { type: "button", class: "shot has", onclick: () => fixOutline(p) },
        el("div", { class: "shot-pic" }, img), el("div", { class: "shot-text" }, el("b", {}, TITLE[p]), el("span", { class: "small muted" }, "Натиснете, за да поправите")));
    }));
    const next = el("button", { class: "btn btn-primary amber-btn btn-lg" }, "Всички са точни: напред");
    next.onclick = () => (outlineChanged ? reread() : overall());
    show(head("Инженерен CAD · проверка", "Очертанията", "Оранжевата линия трябва да следва точно предмета на всяка снимка. Ако хваща ръката, телефона или сянката, или изпуска част, натиснете снимката."),
      grid, errorBox(error),
      el("div", { class: "row-actions" }, next, el("button", { class: "btn btn-ghost", onclick: drawPhotos }, "Назад към снимките")));
  }

  // Докосване върху предмета го изрязва наново (MobileSAM на сървъра); "Не е предметът" маха това, което не трябва.
  function fixOutline(photo) {
    let mode = "add", points = [], busy = false;
    const img = el("img", { alt: TITLE[photo], class: "tap-img" });
    const wrap = el("div", { class: "tap-wrap" }, img);
    const msg = el("div");
    const title = el("h1");
    const reset = el("button", { class: "btn btn-secondary hidden" }, "Обратно към автоматичното");
    const load = () => picture(`/jobs/${jobId}/cad/outline/${photo}.jpg?s=${Date.now()}`).then((u) => (img.src = u));
    const seg = el("div", { class: "seg" });
    const setMode = (m) => {
      mode = m;
      title.textContent = m === "add" ? "Натиснете предмета" : "Натиснете това, което не е предметът";
      seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
    };
    seg.append(el("button", { type: "button", "data-m": "add", onclick: () => setMode("add") }, "Това е предметът"),
      el("button", { type: "button", "data-m": "remove", onclick: () => setMode("remove") }, "Не е предметът"));
    async function send(next) {
      busy = true;
      wrap.classList.add("busy");
      msg.replaceChildren();
      try {
        await api(`/jobs/${jobId}/cad/outline/${photo}`, { method: "POST", json: { points: next } });
        points = next;
        outlineChanged = true;
        stamps[photo] = Date.now();
        reset.classList.toggle("hidden", !points.length);
        await load();
      } catch (err) { msg.replaceChildren(errorBox(err.message)); }
      busy = false;
      wrap.classList.remove("busy");
    }
    img.onclick = (e) => {
      if (busy) return;
      const r = img.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      if (mode === "remove" && !points.some((p) => p.length === 2)) return msg.replaceChildren(errorBox("Първо натиснете самия предмет."));
      send([...points, mode === "add" ? [x, y] : [x, y, 0]].slice(-12)); // сървърът приема до 12 точки
    };
    reset.onclick = () => send([]);
    show(el("div", { class: "eng-head" }, el("div", { class: "kicker amber" }, el("i"), `Очертанието · ${TITLE[photo]}`), title,
        el("p", { class: "muted" }, "Натиснете върху самия предмет; ако линията изпусне част, натиснете и нея. В „Не е предметът“: ръката, телефона, листа или сянката.")),
      seg, wrap, msg,
      el("div", { class: "row-actions" },
        el("button", { class: "btn btn-primary amber-btn", onclick: () => outlines() }, icon("check"), "Готово"), reset,
        el("button", { class: "btn btn-ghost", onclick: drawPhotos }, "Снимай тази страна отново")));
    setMode("add");
    load();
  }

  async function reread() {
    waiting("Чета снимките отново…", "С поправените очертания. Няколко секунди.");
    try {
      plan = await api(`/jobs/${jobId}/cad/read`, { method: "POST", json: { flat } });
      outlineChanged = false;
      oi = 0;
      overall();
    } catch (err) { outlines(err.message); }
  }

  // --- 3. Общите размери ---------------------------------------------------------------------------------------------

  // Колко е общият размер по снимката, от вече измерения: ключ 54 x 24 мм, въведен като 54 x 54, ставаше квадратна плоча.
  function photoGuess(q) {
    const known = q.guess ? sizes[q.guess.of] : undefined;
    return q.guess && known ? Math.round(q.guess.k * known * 10) / 10 : null;
  }

  function overall(error) {
    const q = plan.overall[oi];
    const guess = photoGuess(q);
    measureScreen({
      kicker: `Общите размери · ${oi + 1} от ${plan.overall.length}`, q: { ...q, kind: q.id === "D" ? "diameter" : "length" },
      hint: guess ? `${q.hint || ""} По снимката: около ${mmText(guess)} мм.` : q.hint, image: `/jobs/${jobId}/cad/question/${q.id}.png?s=${plan.stamp}`,
      value: sizes[q.id], zero: false, index: oi, count: plan.overall.length,
      error: error || (oi === 0 && plan.warnings.length ? plan.warnings.join(" ") : null),
      onBack: () => (oi > 0 ? (oi--, overall()) : outlines()),
      onAnswer: async (v) => {
        if (v == null) return;
        if (guess && Math.abs(Math.log(v / guess)) > Math.log(GUESS_ASK)) {
          const sure = await S.confirmBox({ title: "Проверете мярката", ok: `Вярно е: ${mmText(v)} мм`,
            text: `По снимката ${q.name.toLowerCase()} е около ${mmText(guess)} мм, а въведохте ${mmText(v)} мм. Ако сте измерили друга страна, моделът няма да прилича на предмета.` });
          if (!sure) return;
        }
        sizes[q.id] = v;
        if (oi < plan.overall.length - 1) { oi++; return overall(); }
        assemble();
      },
    });
  }

  async function assemble() {
    waiting("Сглобявам формата…", "Очертания, стъпала и отвори от всички снимки. Няколко секунди.");
    try {
      state = await api(`/jobs/${jobId}/cad/read`, { method: "POST", json: { ...sizes, flat } });
      review();
    } catch (err) {
      // Мярка, която не пасва на снимката (сървърът казва коя): обратно на нея, не на последната.
      const w = plan.overall.findIndex((q) => q.id === "W");
      if (err.message.startsWith("Ширината") && w >= 0) oi = w;
      overall(err.message);
    }
  }

  // Една мярка на екран: снимката на човека с оранжева линия точно там, където да се мери (или чертеж).
  function measureScreen({ kicker, q, hint, image, value, zero, index, count, error, onAnswer, onBack, last }) {
    const img = el("img", { class: "q-img", alt: q.name });
    picture(image).then((u) => (img.src = u)).catch(() => {});
    const input = el("input", { class: "input mm-input", inputmode: "decimal", placeholder: "0,00", value: value != null ? mmText(value) : "", "aria-label": `${q.name} в милиметри` });
    const invalid = el("div");
    const next = el("button", { class: "btn btn-primary amber-btn btn-lg", type: "submit" }, last ? icon("cube") : null, last ? "Построй модела" : "Напред");
    const form = el("form", { class: "mm-form" }, el("div", { class: "mm-line" }, input, el("span", { class: "mono muted" }, "мм")),
      q.est != null ? el("p", { class: "small muted" }, `От снимката ≈ ${mmText(q.est)} мм (само ориентир)`) : null, invalid, errorBox(error),
      el("div", { class: "row-actions" }, next,
        q.est != null ? el("button", { class: "btn btn-secondary", type: "button", onclick: () => onAnswer(null) }, `Не мога да го измеря (остава ${mmText(q.est)} мм)`) : null,
        el("button", { class: "btn btn-ghost", type: "button", onclick: onBack }, "Назад")));
    form.onsubmit = (e) => {
      e.preventDefault();
      const v = mmOf(input.value, zero);
      if (v === undefined) {
        return invalid.replaceChildren(errorBox(zero ? "Въведете размера в милиметри, от 0 до 1000 (например 12,35). 0: няма го (стъпало, отвор) или ъгълът е остър."
          : "Въведете размера в милиметри, над 0 и до 1000 (например 12,35)."));
      }
      onAnswer(v);
    };
    const name = q.name.replace(/\s*\([^)]*\)\s*$/, ""); // посоката се вижда по линията на снимката
    show(el("div", { class: "eng-head" }, el("div", { class: "kicker amber" }, el("i"), kicker), el("h1", {}, name),
        el("p", { class: "muted" }, `Измерете по оранжевата линия на снимката, между двете стрелки.${hint ? ` ${hint}` : ""}`)),
      progress(index, count), el("div", { class: "measure" }, img, form));
    input.focus();
  }

  // --- 4. Какво виждам ------------------------------------------------------------------------------------------------

  const KIND_ICON = { body: "cube", view: "target", hole: "target", rect: "split", slot: "ruler", fillet: "rotate", chamfer: "rotate" };
  const TYPES = [["hole", "Отвор"], ["rect", "Правоъгълен изрез"], ["slot", "Канал"], ["fillet", "Заобляне"], ["chamfer", "Фаска"]];
  const FACES = [["top", "Отгоре"], ["front", "Отпред"], ["left", "Отляво"], ["bottom", "Отдолу"], ["back", "Отзад"], ["right", "Отдясно"]];
  const EDGES = [["vertical", "Вертикалните"], ["top", "Горните"], ["bottom", "Долните"], ["all", "Всички"]];
  const ALONG = { top: ["дължината", "ширината"], bottom: ["дължината", "ширината"], front: ["дължината", "височината"],
    back: ["дължината", "височината"], left: ["ширината", "височината"], right: ["ширината", "височината"] };
  const THREADS = ["", "M3", "M4", "M5", "M6", "M8", "M10", "M12"];

  let busy = false;
  async function op(o, error) {
    if (busy) return;
    busy = true;
    document.body.classList.add("eng-busy");
    try {
      state = await api(`/jobs/${jobId}/cad/edit`, { method: "POST", json: { design: state.design, op: o } });
      error = null;
    } catch (err) { error = err.message; }
    busy = false;
    document.body.classList.remove("eng-busy");
    review(error);
  }

  function review(error) {
    const left = state.questions.filter((q) => q.src !== "measured").length;
    const feature = (id) => state.design.features.find((f) => f.id === id);
    const turned = state.design.kind === "turned";

    // 3D по оценките от снимките (model-viewer, като на страницата на модела).
    const preview = el("div", { class: "viewer eng-preview" }, el("div", { class: "center" }, el("div", { class: "spinner" })));
    api(`/jobs/${jobId}/cad/preview.glb?s=${state.stamp}`, { as: "blob" }).then((b) => {
      preview.replaceChildren(el("model-viewer", { src: URL.createObjectURL(b), "camera-controls": true, "auto-rotate": true, "rotation-per-second": "18deg",
        "shadow-intensity": "0.6", exposure: "1.05", "environment-image": "neutral", "interaction-prompt": "none", "camera-orbit": "30deg 65deg auto", alt: "Детайлът в 3D" }));
    }).catch((err) => preview.replaceChildren(el("p", { class: "small muted center" }, err.message)));

    const items = el("div", { class: "card eng-items" }, state.items.map((item) => {
      const f = feature(item.id);
      const isFeature = item.kind !== "body" && item.kind !== "view";
      const plain = item.kind === "view" && item.text.includes("правоъгълник") && !item.text.includes("заобл");
      const cut = ["hole", "rect", "slot"].includes(item.kind) && !(f && f.open);
      const through = !(f && f.depth);
      return el("div", { class: "eng-item" }, icon(KIND_ICON[item.kind] || "target"), el("span", {}, item.text),
        cut ? el("button", { class: "chip-btn", type: "button", onclick: () => op({ through: [item.id, !through] }) }, through ? "проходен" : "глух") : null,
        isFeature || (item.kind === "view" && !plain)
          ? el("button", { class: "chip-btn", type: "button", title: isFeature ? "Махни" : "Опрости до правоъгълник", onclick: () => op({ remove: item.id }) }, icon(isFeature ? "trash" : "split"))
          : null);
    }));

    const addBox = el("div");
    const addBtn = el("button", { class: "btn btn-secondary", type: "button", onclick: () => { addBtn.classList.add("hidden"); addBox.replaceChildren(addForm(turned, () => { addBox.replaceChildren(); addBtn.classList.remove("hidden"); })); } },
      icon("plus"), "Добави елемент");

    // Снимките с очертанието: натиснат отвор, който го няма на предмета (отблясък), се маха след потвърждение.
    const canFlip = [["front", "back"], ["right", "left"]].map((pair) => pair.find((p) => state.views.includes(p))).filter(Boolean);
    const views = state.views.map((v) => {
      const img = el("img", { class: "tap-img", alt: TITLE[v] });
      picture(`/jobs/${jobId}/cad/photo/${v}.jpg?s=${state.stamp}`).then((u) => (img.src = u)).catch(() => {});
      const hint = el("p", { class: "small muted" });
      img.onclick = async (e) => {
        const r = img.getBoundingClientRect();
        try {
          const hit = await api(`/jobs/${jobId}/cad/tap/${v}`, { method: "POST", json: { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height } });
          if (!hit.id) return (hint.textContent = "Там няма отвор. Натиснете вътре в оранжевия кръг или контур на отвора.");
          if (await S.confirmBox({ title: "Да махна ли този елемент?", ok: "Махни", danger: true,
            text: `${hit.text || "Отвор"}: ако го няма на предмета (например отблясък от светкавицата), моделът ще е без него.` })) op({ remove: hit.id });
        } catch (err) { hint.textContent = err.message; }
      };
      const flips = canFlip.includes(v) ? el("div", { class: "row-actions" },
        el("button", { class: "chip-btn", type: "button", onclick: () => op({ flip: [v, 0] }) }, "⇆ ляво-дясно"),
        el("button", { class: "chip-btn", type: "button", onclick: () => op({ flip: [v, 1] }) }, "⇅ горе-долу")) : null;
      return el("div", { class: "card eng-view" }, el("div", { class: "eng-view-head" }, el("b", {}, TITLE[v]), flips), img, hint);
    });

    const next = el("button", { class: "btn btn-primary amber-btn btn-lg" }, icon("ruler"), left ? `Към мерките (${left})` : "Построй модела");
    next.onclick = startMeasuring;
    show(head("Инженерен CAD · проверка", "Какво виждам", "Проверете, че предметът е разчетен правилно. Подробните мерки идват после, по една."),
      el("div", { class: "eng-review" },
        el("div", {}, el("div", { class: "sec-label" }, "В 3D (по оценките от снимките)"), preview,
          el("p", { class: "small muted" }, "Въртете го с мишката или пръст. Ако нещо е наобратно (стъпалото отдолу, отворът в грешния край), обърнете снимката отпред или отдясно по-долу.")),
        el("div", {}, el("div", { class: "sec-label" }, "Разпознато"), items, addBtn, addBox)),
      el("div", { class: "sec-label", style: "margin-top:28px" }, "Снимките с очертанието"),
      el("p", { class: "small muted" }, "Оранжевото е това, което ще се построи. Отвор, който го няма (отблясък от светкавицата), натиснете, за да го махнете."),
      el("div", { class: "eng-views" }, views),
      errorBox(error),
      el("div", { class: "row-actions" }, next, el("button", { class: "btn btn-ghost", onclick: drawPhotos }, icon("back"), "Снимай отново")));
  }

  function addForm(turned, close) {
    let t = "hole", face = turned ? "left" : "top", through = true, centered = [true, true], edges = turned ? "ends" : "vertical", thread = "";
    const box = el("div", { class: "card eng-add" });
    const chips = (opts, get, set) => el("div", { class: "chips" }, opts.map(([v, label]) =>
      el("button", { type: "button", class: `chip-btn ${get() === v ? "on" : ""}`, onclick: () => { set(v); draw(); } }, label)));
    function draw() {
      const edgeType = t === "fillet" || t === "chamfer";
      const faces = turned ? FACES.filter(([v]) => v === "left" || v === "top").map(([v]) => [v, v === "left" ? "По оста" : "Напречно"]) : FACES;
      const along = turned ? (face === "left" ? ["", ""] : ["дължината", ""]) : ALONG[face];
      box.replaceChildren(el("b", {}, "Нов елемент"), chips(TYPES, () => t, (v) => (t = v)),
        edgeType
          ? [el("span", { class: "small muted" }, "Кои ръбове"), chips(turned ? [["ends", "Краищата"], ["all", "Всички"]] : EDGES, () => edges, (v) => (edges = v))]
          : [el("span", { class: "small muted" }, turned ? "Къде" : "От коя страна започва"), chips(faces, () => face, (v) => (face = v)),
            chips([["through", "Проходен"], ["blind", "Глух (с дълбочина)"]], () => (through ? "through" : "blind"), (v) => (through = v === "through")),
            along.map((a, k) => (a ? chips([["c", `В средата по ${a}`], ["m", "Ще го измеря"]], () => (centered[k] ? "c" : "m"),
              (v) => (centered = k === 0 ? [v === "c", centered[1]] : [centered[0], v === "c"])) : null)),
            t === "hole" ? [el("span", { class: "small muted" }, "Резба (по желание; в модела отворът е с диаметъра, който измерите)"),
              chips(THREADS.map((v) => [v, v || "Без"]), () => thread, (v) => (thread = v))] : null],
        el("div", { class: "row-actions" },
          el("button", { class: "btn btn-ghost", type: "button", onclick: close }, "Отказ"),
          el("button", { class: "btn btn-primary", type: "button", onclick: () => op({ add: edgeType ? { t, edges }
            : { t, face, through, centered: turned ? [face === "left", true] : centered, ...(t === "hole" && thread ? { thread } : {}) } }) }, icon("plus"), "Добави")));
    }
    draw();
    return box;
  }

  // --- 5. Мерките, по една, и строенето ------------------------------------------------------------------------------

  function startMeasuring() {
    asked = state.questions.filter((q) => q.src !== "measured");
    if (!asked.length) return build({});
    answers = Object.fromEntries(asked.filter((q) => q.id in answers).map((q) => [q.id, answers[q.id]]));
    qi = 0;
    measureOne();
  }

  function measureOne(error) {
    const q = asked[qi];
    measureScreen({
      kicker: `Инженерен CAD · мярка ${qi + 1} от ${asked.length}`, q, hint: q.hint, image: `/jobs/${jobId}/cad/question/${q.id}.png?s=${state.stamp}`,
      value: answers[q.id], zero: true, index: qi, count: asked.length, error, last: qi === asked.length - 1,
      onBack: () => (qi > 0 ? (qi--, measureOne()) : review()),
      onAnswer: (v) => {
        answers[q.id] = v;
        if (qi < asked.length - 1) { qi++; return measureOne(); }
        build(answers);
      },
    });
  }

  async function build(values) {
    waiting("Строя модела…", "Само от вашите мерки. Няколко секунди.");
    try {
      const measured = await api(`/jobs/${jobId}/cad/edit`, { method: "POST", json: { design: state.design, op: { values } } });
      state = measured;
      const job = await api(`/jobs/${jobId}/cad/build`, { method: "POST", json: { design: measured.design } });
      finished = true;
      const size = ((job.report && job.report.cad && job.report.cad.size_mm) || []).map(mmText).join(" × ");
      await api(`/me/scans/${jobId}`, { method: "PATCH", json: { title: `CAD детайл ${size} мм`.trim() } }).catch(() => {});
      location.href = `scan.html?id=${encodeURIComponent(jobId)}`;
    } catch (err) {
      if (asked.length) { qi = asked.length - 1; measureOne(err.message); } else review(err.message);
    }
  }

  intro();
})();

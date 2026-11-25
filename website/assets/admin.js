// Администриране в сайта: същото като екрана „Администриране“ в приложението (src/app/admin, backend app/admin.py).
// Преглед, профили и техният статус, опашката на сървъра, сигналите в общността и пощата за кодовете. Достъпът и
// правилата се проверяват на сървъра (собственикът и администраторите); тук само се показват.
(function () {
  if (!S.requireAuth()) return;
  S.page({ title: "Администриране", active: "admin" });
  const { el, icon, api } = S;
  const root = document.getElementById("root");

  const TABS = [["overview", "Преглед", "target"], ["users", "Профили", "users"], ["jobs", "Обработка", "queue"], ["reports", "Сигнали", "flag"], ["mail", "Поща", "mail"]];
  const STATE = { uploading: "Качва се", queued: "Чака", reconstructing: "Обработва се", meshing: "Обработва се", cleaning: "Обработва се",
    done: "Готово", failed: "Грешка", cancelled: "Спряно" };
  const tone = (st) => (st === "done" ? "ok" : st === "failed" ? "bad" : st === "cancelled" ? "" : "busy");
  const method = (m) => (m === "cad" ? "Инженерен режим" : m === "single" ? "AI модел" : m === "edit" ? "Промяна" : m === "scan" || m === "express" ? "Сканиране" : m ? `AI (${m})` : "—");

  function ago(t) {
    if (!t) return "никога";
    const s = Math.max(0, Date.now() / 1000 - t);
    if (s < 90) return "току-що";
    if (s < 3600) return `преди ${Math.round(s / 60)} мин`;
    if (s < 86400) return `преди ${Math.round(s / 3600)} ч`;
    if (s < 7 * 86400) { const d = Math.round(s / 86400); return `преди ${d} ${d === 1 ? "ден" : "дни"}`; }
    return S.date(t);
  }
  const chip = (text, cls = "") => el("span", { class: `badge ${cls}` }, el("span", { class: "dot" }), text);
  function userChips(u) {
    return el("span", { class: "adm-chips" },
      u.owner ? chip("Собственик", "info") : u.admin ? chip("Админ", "info") : null,
      u.blocked ? chip("Блокиран", "bad") : null,
      !u.verified ? chip("Непотвърден", "warn") : null);
  }
  async function avatar(u, size = 40) {
    const node = el("span", { class: "avatar adm-avatar", style: `width:${size}px;height:${size}px` }, S.initials(u.name));
    let src = null;
    if (u.has_avatar) { try { src = URL.createObjectURL(await api(`/manage/users/${u.id}/avatar.jpg`, { as: "blob" })); } catch {} }
    if (!src && u.google_picture) src = u.google_picture;
    if (src) node.replaceChildren(el("img", { src, alt: "", referrerpolicy: "no-referrer" }));
    return node;
  }
  // replaceChildren без празните части (null/false от условията), които иначе стават текст "null".
  const put = (node, ...kids) => node.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
  const errorBox = (text) => el("p", { class: "alert alert-error" }, text);
  const spinner = () => el("div", { class: "center", style: "padding:48px" }, el("div", { class: "spinner" }));

  // --- Подредба: заглавие и раздели; разделът е в адреса (#users, #user=ID, ...), за да работи „Назад“ ---------------

  const content = el("div", { class: "adm-content" });
  const bar = el("div", { class: "tab-bar adm-tabs", role: "tablist" });
  let me = { owner: false };
  root.replaceChildren(
    el("div", { class: "page-head" }, el("div", {}, el("div", { class: "kicker", style: "margin-bottom:12px" }, el("i"), "Само за администратора"), el("h1", {}, "Администриране"))),
    bar, content);
  for (const [key, label, ic] of TABS) bar.append(el("button", { type: "button", role: "tab", "data-tab": key, onclick: () => (location.hash = key) }, icon(ic), label));

  function route() {
    const h = location.hash.slice(1) || "overview";
    const [key, arg] = h.split("=");
    bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === (key === "user" ? "users" : key))));
    put(content, spinner());
    const run = { overview, users, user: () => userPage(decodeURIComponent(arg || "")), jobs, reports, mail }[key] || overview;
    run().catch((err) => put(content, errorBox(err.message)));
  }
  addEventListener("hashchange", route);

  // --- Преглед ----------------------------------------------------------------------------------------------------

  async function overview() {
    const o = await api("/manage/overview");
    me = o.me;
    const tile = (label, value, cls, hash) => el(hash ? "a" : "div", { class: `adm-tile ${cls || ""}`, href: hash ? `#${hash}` : null }, el("b", {}, value), el("span", {}, label));
    put(content, 
      el("div", { class: "sec-label" }, "Профили"),
      el("div", { class: "adm-tiles" },
        tile("профила общо", o.users.total, "", "users"), tile("нови за 7 дни", o.users.new_week, "accent"), tile("активни за 7 дни", o.users.active_week, "ok"),
        tile("чакат потвърждение", o.users.unverified, o.users.unverified ? "warn" : "", "users=unverified"),
        tile("блокирани", o.users.blocked, o.users.blocked ? "bad" : "", "users=blocked"), tile("администратори", o.users.admins, "", "users=admins")),
      el("div", { class: "sec-label" }, "Обработка"),
      el("div", { class: "adm-tiles" },
        tile("обработват се сега", o.jobs.working, o.jobs.working ? "warn" : "", "jobs"), tile("чакат на опашката", o.jobs.queued, "", "jobs"),
        tile("неуспешни за 7 дни", o.jobs.failed_week, o.jobs.failed_week ? "bad" : "", "jobs=failed"), tile("готови модела", o.jobs.done, "ok")),
      el("div", { class: "sec-label" }, "Общност и сървър"),
      el("div", { class: "card adm-kv" },
        row("Сигнали в общността", `${o.community.reported} · ${o.community.published} публикувани`, "#reports"),
        row("Потвърждение по имейл", o.server.mail ? "Включено" : "Изключено", "#mail"),
        row("Свободно място", `${o.server.disk_free_gb} от ${o.server.disk_total_gb} GB`),
        row("База данни", o.server.database), row("Версия на сървъра", o.server.version)));
  }
  function row(k, v, href) {
    return el(href ? "a" : "div", { class: "adm-row", href }, el("span", {}, k), el("b", { class: "mono" }, v), href ? icon("back") : null);
  }

  // --- Профили -------------------------------------------------------------------------------------------------------

  const FILTERS = [["all", "Всички"], ["unverified", "Чакат потвърждение"], ["blocked", "Блокирани"], ["admins", "Администратори"]];
  let query = "";
  async function users() {
    const show = (location.hash.split("=")[1] || "all");
    const search = el("input", { class: "input", type: "search", placeholder: "Търсене по име или имейл", value: query, "aria-label": "Търсене по име или имейл" });
    const list = el("div", {}, spinner());
    const chips = el("div", { class: "chips" }, FILTERS.map(([v, label]) =>
      el("button", { type: "button", class: `chip-btn ${show === v ? "on" : ""}`, onclick: () => (location.hash = v === "all" ? "users" : `users=${v}`) }, label)));
    put(content, el("div", { class: "adm-tools" }, el("label", { class: "search" }, icon("search"), search), chips), list);
    let asked = 0, offset = 0, all = [];
    async function load(more) {
      const mine = ++asked;
      const r = await api(`/manage/users?q=${encodeURIComponent(query)}&show=${show}&offset=${more ? offset : 0}`);
      if (mine !== asked) return;
      all = more ? all.concat(r.users) : r.users;
      offset = r.next || 0;
      const rows = el("div", { class: "card adm-list" });
      for (const u of all) {
        const av = el("span");
        avatar(u).then((n) => av.replaceWith(n));
        rows.append(el("a", { class: `adm-user ${u.blocked ? "dim" : ""}`, href: `#user=${u.id}` }, av,
          el("span", { class: "grow" }, el("b", {}, u.name), el("span", { class: "mono small muted" }, u.email), userChips(u)),
          el("span", { class: "adm-right" }, el("b", { class: "mono" }, u.scans), el("span", { class: "small muted" }, ago(u.last_seen)))));
      }
      put(list, el("p", { class: "small muted" }, r.total === 1 ? "1 профил" : `${r.total} профила`),
        all.length ? rows : el("p", { class: "muted" }, "Няма профили тук."),
        r.next != null ? el("button", { class: "btn btn-secondary", onclick: () => load(true) }, "Още") : null);
    }
    let t;
    search.oninput = () => { query = search.value.trim(); clearTimeout(t); t = setTimeout(() => load().catch((e) => put(list, errorBox(e.message))), 300); };
    await load();
  }

  // --- Един профил ---------------------------------------------------------------------------------------------------

  async function userPage(id) {
    const [u, o] = await Promise.all([api(`/manage/users/${id}`), api("/manage/overview")]);
    me = o.me;
    const locked = u.is_me ? "Това е вашият профил." : u.owner ? "Собственикът се задава само на сървъра." : u.admin && !me.owner ? "Само собственикът променя администраторите." : null;
    const change = async (patch, ok) => {
      try { await api(`/manage/users/${id}`, { method: "PATCH", json: patch }); S.toast(ok); route(); } catch (err) { S.toast(err.message); }
    };
    const act = (label, cls, onclick, disabled) => el("button", { class: `btn ${cls}`, type: "button", disabled: !!disabled, onclick }, label);
    const actions = el("div", { class: "row-actions adm-actions" },
      act(u.blocked ? "Отблокирай" : "Блокирай", u.blocked ? "btn-secondary" : "btn-danger", async () => {
        if (u.blocked) return change({ blocked: false }, "Отблокиран");
        if (await S.confirmBox({ title: "Блокиране на профила?", ok: "Блокирай", danger: true,
          text: `${u.name} излиза от всички устройства и не може да влезе, докато не го отблокирате. Моделите му остават.` })) change({ blocked: true }, "Блокиран");
      }, locked),
      !u.verified ? act("Потвърди имейла", "btn-secondary", () => change({ verified: true }, "Имейлът е потвърден"), locked) : null,
      act("Смени името", "btn-secondary", async () => {
        const name = prompt("Ново име на профила", u.name);
        if (name && name.trim() && name.trim() !== u.name) change({ name: name.trim() }, "Името е сменено");
      }, locked),
      me.owner && !u.owner && !u.is_me ? act(u.admin ? "Махни правата на администратор" : "Направи администратор", "btn-secondary", async () => {
        const on = !u.admin;
        if (await S.confirmBox({ title: on ? "Права на администратор?" : "Без права на администратор?", ok: on ? "Направи администратор" : "Махни правата", danger: !on,
          text: on ? `${u.name} ще вижда „Администриране“: профилите, опашката и сигналите, и ще може да блокира обикновени профили.` : `${u.name} отново е обикновен профил.` })) change({ admin: on }, on ? "Вече е администратор" : "Правата са махнати");
      }) : null,
      act("Изход от всички устройства", "btn-ghost", async () => {
        if (!(await S.confirmBox({ title: "Изход от всички устройства?", text: "Профилът остава, но трябва да влезе отново.", ok: "Изход навсякъде" }))) return;
        try { const r = await api(`/manage/users/${id}/signout`, { method: "POST" }); S.toast(`Затворени сесии: ${r.ended}`); route(); } catch (err) { S.toast(err.message); }
      }, locked || !u.sessions),
      act("Изтрий профила", "btn-danger", async () => {
        if (!(await S.confirmBox({ title: "Изтриване на профила?", text: `Изтриват се ${u.name} и всички ${u.scans} сканирания, завинаги.`, ok: "Изтрий профила", danger: true, confirmWord: "ИЗТРИЙ" }))) return;
        try { await api(`/manage/users/${id}`, { method: "DELETE" }); S.toast("Профилът е изтрит"); location.hash = "users"; } catch (err) { S.toast(err.message); }
      }, locked));

    const scans = el("div", { class: "card adm-list" });
    for (const sc of u.scans_list) {
      const pic = el("span", { class: "adm-thumb" }, icon("cube"));
      if (sc.thumb || sc.state === "done") api(`/jobs/${sc.id}/thumb.jpg`, { as: "blob" }).then((b) => pic.replaceChildren(el("img", { src: URL.createObjectURL(b), alt: "" }))).catch(() => {});
      const logBtn = el("button", { class: "chip-btn", type: "button" }, "Дневник");
      const delBtn = el("button", { class: "chip-btn", type: "button", title: "Изтрий сканирането" }, icon("trash"));
      const logBox = el("div");
      logBtn.onclick = async () => {
        try { logBox.replaceChildren(el("pre", { class: "log" }, (await api(`/jobs/${sc.id}/log?lines=120`, { as: "text" })) || "Дневникът е празен.")); } catch (err) { S.toast(err.message); }
      };
      delBtn.onclick = async () => {
        if (!(await S.confirmBox({ title: "Изтриване на сканирането?", text: "Изтрива се завинаги, и от общността.", ok: "Изтрий", danger: true }))) return;
        try { await api(`/jobs/${sc.id}`, { method: "DELETE" }); S.toast("Изтрито"); route(); } catch (err) { S.toast(err.message); }
      };
      scans.append(el("div", { class: "adm-scan" },
        el("div", { class: "adm-user" }, pic,
          el("span", { class: "grow" }, el("b", {}, sc.title || "Без име"),
            el("span", { class: "small muted mono" }, `${method(sc.method)} · ${S.date(sc.created)}${sc.published ? " · в общността" : ""}`)),
          chip(STATE[sc.state] || sc.state, tone(sc.state)), logBtn, delBtn), logBox));
    }
    const av = await avatar(u, 64);
    put(content, 
      el("a", { class: "btn btn-ghost btn-sm", href: "#users", style: "padding-left:6px;margin-bottom:12px" }, icon("back"), "Профили"),
      el("div", { class: "adm-person" }, av, el("div", {}, el("h2", {}, u.name), el("div", { class: "mono muted" }, u.email), userChips(u))),
      el("div", { class: "card adm-kv" },
        row("Създаден", S.date(u.created)), row("Последно активен", ago(u.last_seen)), row("Вход", u.google ? "Google" : "Имейл и парола"),
        row("Влезли устройства", String(u.sessions)), row("Сканирания", String(u.scans))),
      el("div", { class: "sec-label" }, "Статус"), locked ? el("p", { class: "small muted" }, locked) : null, actions,
      el("div", { class: "sec-label" }, `Сканирания · ${u.scans_list.length}`),
      u.scans_list.length ? scans : el("p", { class: "muted" }, "Още няма сканирания."));
  }

  // --- Обработка ---------------------------------------------------------------------------------------------------

  let jobsTimer;
  async function jobs() {
    clearTimeout(jobsTimer);
    const show = location.hash.split("=")[1] || "active";
    const list = await api(`/manage/jobs?show=${show}`);
    const chips = el("div", { class: "chips" }, [["active", "Сега"], ["failed", "Неуспешни"], ["all", "Всички"]].map(([v, label]) =>
      el("button", { type: "button", class: `chip-btn ${show === v ? "on" : ""}`, onclick: () => (location.hash = v === "active" ? "jobs" : `jobs=${v}`) }, label)));
    const rows = el("div", { class: "card adm-list" });
    for (const j of list) {
      const logBox = el("div");
      const working = !["done", "failed", "cancelled"].includes(j.state);
      const logBtn = el("button", { class: "chip-btn", type: "button" }, "Дневник");
      logBtn.onclick = async () => {
        try { logBox.replaceChildren(el("pre", { class: "log" }, (await api(`/jobs/${j.id}/log?lines=120`, { as: "text" })) || "Дневникът е празен.")); } catch (err) { S.toast(err.message); }
      };
      const stop = working ? el("button", { class: "chip-btn", type: "button" }, "Спри") : null;
      if (stop) stop.onclick = async () => {
        if (!(await S.confirmBox({ title: "Спиране на обработката?", text: "Сканирането се изтрива. Авторът ще трябва да го направи отново.", ok: "Спри", danger: true }))) return;
        try { await api(`/jobs/${j.id}`, { method: "DELETE" }); route(); } catch (err) { S.toast(err.message); }
      };
      rows.append(el("div", { class: "adm-scan" },
        el("div", { class: "adm-user" },
          el("span", { class: "grow" }, el("b", {}, j.title || "Без име", j.user ? el("a", { class: "small", href: `#user=${j.user.id}`, style: "margin-left:8px" }, j.user.name) : null),
            el("span", { class: "small muted mono" }, [method(j.method), ago(j.created), j.step, j.seconds ? `${Math.round(j.seconds / 60)} мин` : null].filter(Boolean).join(" · ")),
            j.state === "failed" && j.error ? el("span", { class: "small", style: "color:var(--danger)" }, j.error) : null),
          chip(STATE[j.state] || j.state, tone(j.state)), logBtn, stop), logBox));
    }
    put(content, el("div", { class: "adm-tools" }, chips),
      list.length ? rows : el("p", { class: "muted" }, show === "active" ? "Опашката е празна: нищо не се обработва." : "Няма такива."));
    if (show === "active") jobsTimer = setTimeout(() => location.hash.startsWith("#jobs") && route(), 10000); // опашката се опреснява сама
  }

  // --- Сигнали в общността --------------------------------------------------------------------------------------------

  async function reports() {
    const list = await api("/manage/reports");
    const cards = list.map((p) => {
      const pic = el("span", { class: "adm-thumb big" }, icon("cube"));
      api(`/jobs/${p.id}/thumb.jpg`, { as: "blob" }).then((b) => pic.replaceChildren(el("img", { src: URL.createObjectURL(b), alt: "" }))).catch(() => {});
      const act = (label, cls, fn) => el("button", { class: `btn ${cls}`, type: "button", onclick: async () => { try { await fn(); route(); } catch (err) { S.toast(err.message); } } }, label);
      return el("div", { class: "card adm-report" },
        el("div", { class: "adm-user" }, pic, el("span", { class: "grow" }, el("b", {}, p.title || "Без име"),
          el("span", { class: "adm-chips" }, chip(p.reports === 1 ? "1 сигнал" : `${p.reports} сигнала`, "warn"), p.hidden ? chip("Скрит", "bad") : null))),
        el("ul", { class: "small muted" }, p.reasons.map((r) => el("li", {}, `„${r || "без причина"}“`))),
        el("div", { class: "row-actions" },
          act("Остави", "btn-secondary", () => api(`/admin/community/${p.id}/keep`, { method: "POST" })),
          act("Махни от общността", "btn-danger", async () => {
            if (await S.confirmBox({ title: "Махане от общността?", text: "Моделът остава при автора, но никой друг не го вижда.", ok: "Махни", danger: true }))
              await api(`/admin/community/${p.id}`, { method: "DELETE" });
          }),
          el("a", { class: "btn btn-ghost", href: `#user=${p.author}` }, "Авторът")));
    });
    put(content, el("p", { class: "small muted" }, "Модел с 3 сигнала се скрива от общността, докато не решите какво да стане с него."),
      ...(cards.length ? cards : [el("p", { class: "muted" }, "Няма сигнали. Общността е чиста.")]));
  }

  // --- Поща (само преглед: задава се в .env на сървъра) ----------------------------------------------------------

  async function mail() {
    const m = await api("/manage/mail");
    const test = el("button", { class: "btn btn-secondary", type: "button" }, icon("mail"), "Изпрати пробно писмо до мен");
    test.onclick = async () => {
      test.disabled = true;
      try { const r = await api("/manage/mail/test", { method: "POST" }); S.toast(`Изпратено до ${r.sent_to}. Проверете пощата.`); } catch (err) { S.toast(err.message); }
      test.disabled = false;
    };
    put(content, 
      el("div", { class: "card adm-mail" },
        el("div", {}, el("b", {}, "Потвърждение на нови профили"),
          el("p", { class: "small muted" }, m.configured ? `Кодът се праща от ${m.user}.` : "Изключено: новите профили не се потвърждават по имейл.")),
        chip(m.configured ? "Включено" : "Изключено", m.configured ? "ok" : "warn")),
      el("p", { class: "small muted" }, "Пощенската кутия се задава в .env на сървъра (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD), не от сайта или приложението."),
      m.configured ? test : null);
  }

  route();
})();

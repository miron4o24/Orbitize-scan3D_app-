  if (S.requireAuth()) {
    S.page({ title: "Профил", active: "account" });
    const { el, icon, api } = S;
    const root = document.getElementById("root");

    // Any picture the browser can open becomes a square JPEG of at most 512 px (the server takes JPEG up to 2 MB).
    function toJpeg(file) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const side = Math.min(img.width, img.height), out = Math.min(512, side);
          const c = document.createElement("canvas");
          c.width = c.height = out;
          c.getContext("2d").drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, out, out);
          URL.revokeObjectURL(img.src);
          c.toBlob((b) => (b ? resolve(b) : reject(new Error("Снимката не може да се прочете."))), "image/jpeg", 0.88);
        };
        img.onerror = () => reject(new Error("Снимката не може да се прочете."));
        img.src = URL.createObjectURL(file);
      });
    }

    function render(me) {
      S.auth.setUser(me);
      const av = el("span", { class: "avatar lg" });
      S.avatarInto(av, me);
      const file = el("input", { type: "file", accept: "image/*", class: "hidden" });
      file.onchange = async () => {
        if (!file.files[0]) return;
        try {
          const jpeg = await toJpeg(file.files[0]);
          await S.forgetFiles("avatar/");
          render(await api("/me/avatar", { method: "PUT", body: jpeg, headers: { "Content-Type": "image/jpeg" } }));
          S.toast("Снимката е запазена");
        } catch (err) { S.toast(err.message); }
      };
      const photoBtns = el("div", { class: "row-actions" },
        el("button", { class: "btn btn-secondary btn-sm", onclick: () => file.click() }, me.has_avatar ? "Смени снимката" : "Добави снимка"),
        me.has_avatar ? el("button", { class: "btn btn-ghost btn-sm", onclick: async () => {
          try { await S.forgetFiles("avatar/"); render(await api("/me/avatar", { method: "DELETE" })); } catch (err) { S.toast(err.message); }
        } }, "Премахни") : null, file);

      const name = el("input", { class: "input", value: me.name, maxlength: "100", autocomplete: "name" });
      const saveName = el("button", { class: "btn btn-primary", type: "submit" }, "Запази");
      name.setAttribute("aria-label", "Име");
      const nameForm = el("form", { class: "inline-form" }, name, saveName);
      nameForm.onsubmit = async (e) => {
        e.preventDefault();
        if (!name.value.trim()) return S.toast("Името не може да е празно.");
        saveName.disabled = true;
        try { render(await api("/me", { method: "PATCH", json: { name: name.value } })); S.toast("Запазено"); }
        catch (err) { S.toast(err.message); saveName.disabled = false; }
      };

      const st = me.stats;
      // Нивото, точките и значките, като в профила в приложението (backend app/progress.py).
      const progressBox = el("div");
      api("/me/progress").then((p) => {
        document.getElementById("points").textContent = p.points;
        const lv = p.level, span = Math.max(1, (lv.to || lv.from + 1) - lv.from);
        const unlocked = p.badges.filter((b) => b.unlocked).length;
        progressBox.replaceChildren(
          el("div", { class: "card level-card" },
            el("div", { class: "level-head" }, el("b", { class: "mono" }, `Ниво ${lv.number}`), el("span", {}, lv.name),
              el("span", { class: "mono muted" }, lv.to ? `${p.points} / ${lv.to}` : `${p.points}`)),
            el("div", { class: "progress" }, el("i", { style: `width:${Math.min(100, ((p.points - lv.from) / span) * 100)}%` }))),
          el("div", { class: "sec-label", style: "margin:22px 0 10px" }, `Значки · ${unlocked} от ${p.badges.length}`),
          el("div", { class: "badge-grid" }, p.badges.map((b) => el("div", { class: `badge-card ${b.unlocked ? "on" : ""}`, title: b.description },
            el("span", { class: "hex" }, icon(b.unlocked ? "star" : "lock")), el("b", {}, b.title),
            el("small", {}, b.unlocked ? S.date(b.unlocked_at) : b.target > 1 ? `${b.progress} от ${b.target}` : "заключена")))));
      }).catch(() => { document.getElementById("points").textContent = "—"; });
      const signOut = el("button", { class: "btn btn-secondary" }, icon("logout"), "Изход");
      signOut.onclick = async () => {
        try { await api("/auth/logout", { method: "POST", noRedirect: true }); } catch {}
        S.auth.clear();
        location.replace("index.html");
      };
      const del = el("button", { class: "btn btn-danger" }, icon("trash"), "Изтрий профила ми");
      del.onclick = async () => {
        const ok = await S.confirmBox({
          title: "Да изтрием ли профила ви?",
          text: `Профилът ви и всичките ви сканирания (${st.scans}) се изтриват завинаги, в приложението и тук. Това не може да се отмени.`,
          ok: "Изтрий всичко", danger: true, confirmWord: "ИЗТРИЙ",
        });
        if (!ok) return;
        try {
          await api("/me", { method: "DELETE" });
          S.auth.clear();
          location.replace("index.html");
        } catch (err) { S.toast(err.message); }
      };

      // Theme: the same switch as the moon/sun in the header.
      const themeSeg = el("div", { class: "seg", role: "group", "aria-label": "Тема" });
      const drawTheme = () => themeSeg.replaceChildren(...[["dark", "Тъмна", "moon"], ["light", "Светла", "sun"]].map(([t, label, ic]) =>
        el("button", { type: "button", "aria-pressed": String(S.theme() === t), onclick: () => { S.setTheme(t); drawTheme(); } }, icon(ic), label)));
      drawTheme();
      const setting = (title, text, control, cls = "") =>
        el("div", { class: `setting ${cls}` }, el("div", {}, el("h3", {}, title), el("p", {}, text)), el("div", {}, control));
      const photo = el("div", { class: "photo-row" }, el("span", { class: "avatar", html: av.innerHTML }), photoBtns);
      S.avatarInto(photo.firstChild, me);

      root.replaceChildren(
        el("div", { class: "profile" }, av,
          el("div", { style: "min-width:0" }, el("h2", {}, me.name),
            el("div", { class: "mono" }, me.email), el("div", { class: "mono" }, `Член от ${S.date(me.created)}`))),
        el("div", { class: "stat-grid" },
          el("div", {}, el("small", {}, "Сканирания"), el("b", {}, st.scans)),
          el("div", {}, el("small", {}, "В общността"), el("b", {}, st.published ?? st.shared)),
          el("div", {}, el("small", {}, "Точки"), el("b", { id: "points" }, "…"))),
        progressBox,
        el("div", { class: "settings" },
          S.isAdmin(me) ? setting("Администриране", "Профили и техният статус, обработката, сигналите в общността, пощата.",
            el("a", { class: "btn btn-primary", href: "admin.html" }, icon("shield"), "Отвори")) : null,
          setting("Снимка", "Показва се в профила ви, тук и в приложението.", photo),
          setting("Име", "Вижда се, когато споделите линк към модел.", nameForm),
          setting("Тема", "Тъмна или светла. Пази се в този браузър.", themeSeg),
          setting("Изход", "Излизате от профила на това устройство.", signOut),
          setting("Изтриване на профила", "Изтрива профила, всичките ви сканирания, линковете и снимката от сървъра. Не може да се отмени.", del, "danger")));
    }

    api("/me").then(render).catch((err) => {
      root.replaceChildren(el("div", { class: "card" }, el("p", { class: "alert alert-error" }, err.message)));
    });
  }

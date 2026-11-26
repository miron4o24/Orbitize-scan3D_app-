  if (S.requireAuth()) {
    S.page({ title: "Общност", active: "community" });
    const { el, icon, api } = S;
    const list = document.getElementById("list");
    const moreBox = document.getElementById("more-box"), moreBtn = document.getElementById("more");
    const SORTS = [["new", "Най-нови"], ["top", "Най-харесвани"]];
    let sort = (() => { try { return sessionStorage.getItem("orbitize.csort") || "new"; } catch { return "new"; } })();
    let posts = null, next = null, entered = false;
    const thumbs = new Map(); // post id -> blob URL

    // Pictures need the token, so they are fetched (4 at a time, kept in the browser) instead of plain <img> links.
    async function loadThumbs(items) {
      const todo = items.filter((p) => !thumbs.has(p.id));
      const worker = async () => {
        for (let p; (p = todo.shift()); ) {
          thumbs.set(p.id, null);
          try {
            const raw = await S.cachedBlob(`cthumb/${p.id}/${p.published}`, `/community/${p.id}/thumb.jpg`);
            const url = URL.createObjectURL(await S.cutout(raw).catch(() => raw));
            thumbs.set(p.id, url);
            const box = document.querySelector(`[data-thumb="${p.id}"]`);
            if (box) box.replaceChildren(el("img", { src: url, alt: "", decoding: "async" }));
          } catch {}
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
    }

    // Heart and count; a tap changes it at once, then the server's answer is shown.
    function likeButton(p) {
      const b = el("button", { class: "like-btn", type: "button" });
      const show = () => {
        b.classList.toggle("on", p.liked);
        b.setAttribute("aria-pressed", String(p.liked));
        b.title = p.liked ? "Махни харесването" : "Харесай";
        b.setAttribute("aria-label", `${b.title} (${p.likes})`);
        b.replaceChildren(icon("heart"), el("span", {}, p.likes));
      };
      b.onclick = async (e) => {
        e.preventDefault(); e.stopPropagation();
        const was = p.liked;
        p.liked = !was; p.likes += was ? -1 : 1; show();
        try { Object.assign(p, await api(`/community/${p.id}/like`, { method: was ? "DELETE" : "PUT" })); }
        catch (err) { p.liked = was; p.likes += was ? 1 : -1; S.toast(err.message); }
        show();
      };
      show();
      return b;
    }

    // 24.10.25: short enough for a card's last line on a phone.
    const shortDate = (unix) => {
      const d = new Date(unix * 1000), two = (n) => String(n).padStart(2, "0");
      return `${two(d.getDate())}.${two(d.getMonth() + 1)}.${String(d.getFullYear()).slice(2)}`;
    };

    function card(p) {
      const pic = thumbs.get(p.id);
      const av = el("span", { class: "avatar sm" });
      S.authorAvatar(av, p.author);
      return el("a", { class: "scan-card post-card", href: `post.html?id=${encodeURIComponent(p.id)}` },
        el("div", { class: "thumb", "data-thumb": p.id }, pic ? el("img", { src: pic, alt: "", decoding: "async" }) : icon("cube")),
        p.mine ? el("span", { class: "mine-tag" }, "Ваш") : null,
        el("div", { class: "meta" },
          el("div", { class: "title" }, p.title),
          el("div", { class: "row" },
            el("span", { class: "author" }, av, el("span", {}, p.author.name || "Потребител")),
            likeButton(p)),
          el("div", { class: "size" }, `${Math.round(p.longest_mm)} мм · ${shortDate(p.published)}`)));
    }

    function ghosts(n) {
      return el("div", { class: "scan-grid" }, Array.from({ length: n }, () =>
        el("div", { class: "scan-card ghost" },
          el("div", { class: "thumb" }, el("div", { class: "skeleton" })),
          el("div", { class: "meta" }, el("div", { class: "skeleton", style: "height:16px;width:70%" }),
            el("div", { class: "skeleton", style: "height:12px;width:45%" })))));
    }

    function drawChips() {
      document.getElementById("chips").replaceChildren(...SORTS.map(([key, label]) =>
        el("button", { class: `chip-btn ${sort === key ? "on" : ""}`, type: "button", "aria-pressed": String(sort === key),
          onclick: () => {
            if (sort === key) return;
            sort = key;
            try { sessionStorage.setItem("orbitize.csort", key); } catch {}
            drawChips(); load();
          } }, label)));
    }

    function draw() {
      if (!posts) return list.replaceChildren(ghosts(8));
      moreBox.classList.toggle("hidden", next == null);
      if (!posts.length) {
        return list.replaceChildren(el("div", { class: "card center" },
          el("div", { class: "icon-box" }, icon("globe")),
          el("h3", {}, "Още няма публикувани модели"),
          el("p", { class: "muted", style: "max-width:400px" },
            "Бъдете първи: отворете готово сканиране от „Моите сканирания“ и в раздел „Още“ изберете „Публикувай в общността“."),
          el("a", { class: "btn btn-primary", href: "scans.html" }, "Моите сканирания")));
      }
      list.replaceChildren(el("div", { class: `scan-grid ${entered ? "" : "enter"}` }, posts.map(card)));
      loadThumbs(posts);
      entered = true;
    }

    async function load() {
      posts = null; next = null; entered = false;
      draw();
      try {
        const r = await api(`/community?sort=${sort}`);
        posts = r.posts; next = r.next;
      } catch (err) {
        return list.replaceChildren(el("div", { class: "card center" },
          el("p", { class: "alert alert-error" }, err.message),
          el("button", { class: "btn btn-secondary", onclick: load }, "Опитай отново")));
      }
      draw();
    }

    moreBtn.onclick = async () => {
      moreBtn.disabled = true;
      try {
        const r = await api(`/community?sort=${sort}&offset=${next}`);
        const seen = new Set(posts.map((p) => p.id)); // a post liked meanwhile can move to the next page
        posts = posts.concat(r.posts.filter((p) => !seen.has(p.id)));
        next = r.next;
        draw();
      } catch (err) { S.toast(err.message); }
      moreBtn.disabled = false;
    };

    drawChips();
    load();
  }

// Shared by every page: API calls with the user's session token, the header and footer, small helpers.
// Only the user's own session token is used here (from sign-in); the site never holds the server's admin token.
(function () {
  const C = window.ORBITIZE;
  const TOKEN_KEY = "orbitize.token";
  const USER_KEY = "orbitize.user";

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
  };

  const auth = {
    token: () => store.get(TOKEN_KEY),
    user() { try { return JSON.parse(store.get(USER_KEY) || "null"); } catch { return null; } },
    save(token, user) { forgetLocal(); store.set(TOKEN_KEY, token); store.set(USER_KEY, JSON.stringify(user)); },
    setUser(user) {
      store.set(USER_KEY, JSON.stringify(user));
      const av = document.querySelector(".header .avatar");
      if (av) avatarInto(av, user); // name or photo may have changed
    },
    clear() { store.set(TOKEN_KEY, null); store.set(USER_KEY, null); forgetLocal(); },
  };

  // --- Kept in the browser, so pages open at once -----------------------------------------------------------------
  // The scan list (shown straight away, then refreshed) and the 3D models and pictures (the server builds a model
  // file on every request, which takes a while). All of it is removed on sign-out.
  const SCANS_KEY = "orbitize.scans";
  const FILES = "orbitize-files-v1";
  const FILES_MAX = 40;

  const scansCache = {
    get() { try { return JSON.parse(store.get(SCANS_KEY) || "null"); } catch { return null; } },
    set(list) { store.set(SCANS_KEY, JSON.stringify(list)); },
  };

  // Favourites and notes, per scan. The server has no place for them, so they live in this browser only; they are
  // kept on sign-out (only the scans' owner can open those scans to see them).
  const marks = {
    favs() { try { return JSON.parse(store.get("orbitize.fav") || "{}") || {}; } catch { return {}; } },
    fav: (id) => !!marks.favs()[id],
    setFav(id, on) {
      const f = marks.favs();
      if (on) f[id] = 1; else delete f[id];
      store.set("orbitize.fav", JSON.stringify(f));
    },
    note: (id) => store.get("orbitize.note." + id) || "",
    setNote: (id, text) => store.set("orbitize.note." + id, text.trim() ? text : null),
  };

  function forgetLocal() {
    store.set(SCANS_KEY, null);
    try { caches.delete(FILES); } catch {}
  }

  const fileUrl = (key) => `${location.origin}/__files/${key}`;

  // A file from the API, from the browser's store when it was fetched before. key must change when the file does.
  async function cachedBlob(key, path, { onProgress } = {}) {
    let box = null;
    try {
      box = await caches.open(FILES);
      const hit = await box.match(fileUrl(key));
      if (hit) return await hit.blob();
    } catch { box = null; } // no Cache Storage (old browser, private mode): just fetch
    const blob = await api(path, { as: "blob", onProgress });
    if (box) {
      try {
        await box.put(fileUrl(key), new Response(blob, { headers: { "Content-Type": blob.type } }));
        const keys = await box.keys(); // oldest first
        for (const k of keys.slice(0, Math.max(0, keys.length - FILES_MAX))) await box.delete(k);
      } catch {}
    }
    return blob;
  }

  async function forgetFiles(part) {
    try {
      const box = await caches.open(FILES);
      for (const k of await box.keys()) if (k.url.includes(part)) await box.delete(k);
    } catch {}
  }

  // Reads a download in pieces to report how far it is (0..1), where the server says how big it is.
  async function readWithProgress(res, onProgress) {
    const total = Number(res.headers.get("Content-Length")) || 0;
    if (!res.body || !total) return res.blob();
    const reader = res.body.getReader(), parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.length;
      onProgress(Math.min(1, got / total));
    }
    return new Blob(parts, { type: res.headers.get("Content-Type") || "" });
  }

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  // The server answers in English: its messages in Bulgarian (exact text, or how the text starts).
  const MESSAGES = [
    ["Wrong email or password", "Грешен имейл или парола."],
    ["There is already an account with this email", "Вече има профил с този имейл. Влезте в него."],
    ["This email already has an account through Google", "Този имейл вече има профил чрез Google. Използвайте „Вход с Google“."],
    ["Enter a valid email address", "Въведете валиден имейл адрес."],
    ["The password needs at least", "Паролата трябва да е поне 6 знака."],
    ["Too many wrong passwords", "Твърде много грешни пароли. Изчакайте 15 минути и опитайте отново."],
    ["Not signed in", "Не сте влезли или входът е изтекъл. Влезте отново."],
    ["This account no longer exists", "Този профил вече не съществува."],
    ["The account database is not reachable", "Базата данни с профили не е достъпна в момента. Опитайте по-късно."],
    ["Accounts are not set up", "Профилите още не са настроени на сървъра."],
    ["Google sign-in is not set up", "Входът с Google още не е настроен на сървъра."],
    ["Google sign-in failed", "Входът с Google не успя. Опитайте отново."],
    ["No such job", "Сканирането не е намерено."],
    ["No such scan", "Сканирането не е намерено."],
    ["Result not ready", "Моделът още не е готов."],
    ["Only a finished scan can be shared", "Може да споделите само готово сканиране."],
    ["Photo too large", "Снимката е твърде голяма (до 2 MB)."],
    ["Not a JPEG", "Файлът не е JPEG снимка."],
    ["The name cannot be empty", "Името не може да е празно."],
    ["Job already started", "Обработката вече е започнала."],
    ["Too many scans are waiting", "Твърде много сканирания чакат. Опитайте по-късно."],
    ["The server is full", "Сървърът е зает в момента. Опитайте по-късно."],
    ["Need a video or at least", "Няма достатъчно кадри за нов опит. Направете ново сканиране."],
    ["No picture", "Няма снимка."],
    ["No own photo", "Няма снимка."],
    ["No photo", "Няма снимка."],
    ["This model is no longer in the community", "Този модел вече не е в общността."],
    ["Only a finished scan can go into the community", "В общността може да публикувате само готово сканиране."],
    ["Give the model a name first", "Първо дайте име на модела."],
    ["This is your own model", "Това е ваш модел."],
  ];
  const BY_STATUS = {
    0: "Няма връзка със сървъра. Проверете интернета и опитайте отново.",
    400: "Невалидна заявка.", 403: "Нямате достъп до това.", 404: "Не е намерено.",
    413: "Файлът е твърде голям.", 429: "Твърде много опити. Изчакайте малко.",
    502: "Сървърът не отговаря в момента. Опитайте по-късно.",
    503: "Сървърът не отговаря в момента. Опитайте по-късно.",
  };

  function detail(body, status) {
    const d = body && body.detail;
    if (typeof d === "string" && /[\u0400-\u04FF]/.test(d)) return d; // the server's own message, in Bulgarian
    if (typeof d === "string") {
      const hit = MESSAGES.find(([en]) => d.startsWith(en));
      if (hit) return hit[1];
    }
    return BY_STATUS[status] || `Нещо се обърка (грешка ${status}). Опитайте отново.`;
  }

  // as: "json" (default), "blob" or "text". A 401 means the sign-in has ended: back to the sign-in page.
  async function api(path, { method = "GET", json, body, as = "json", headers = {}, noRedirect, onProgress } = {}) {
    const h = { ...headers };
    const t = auth.token();
    if (t) h.Authorization = `Bearer ${t}`;
    if (json !== undefined) { h["Content-Type"] = "application/json"; body = JSON.stringify(json); }
    let res;
    try {
      res = await fetch(C.apiUrl + path, { method, headers: h, body });
    } catch {
      throw new ApiError(0, detail(null, 0));
    }
    if (!res.ok) {
      let b = null;
      try { b = await res.json(); } catch {}
      if (res.status === 401 && t && !noRedirect) {
        auth.clear();
        location.href = "login.html?next=" + encodeURIComponent(location.pathname.split("/").pop() + location.search);
      }
      throw new ApiError(res.status, detail(b, res.status));
    }
    if (as === "blob") return onProgress ? readWithProgress(res, onProgress) : res.blob();
    if (as === "text") return res.text();
    return res.json();
  }

  function requireAuth() {
    if (!auth.token()) {
      location.replace("login.html?next=" + encodeURIComponent(location.pathname.split("/").pop() + location.search));
      return false;
    }
    return true;
  }

  // Small DOM helper: el("div", {class: "x", onclick: f}, "text", child). Text is always set as text, never HTML.
  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v; // only for our own icons below
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : String(kid));
    return n;
  }

  const ICONS = {
    cube: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7 12 12l8.7-5M12 22V12"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>',
    android: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.6 9.48 19.44 6.3a.38.38 0 0 0-.66-.38l-1.86 3.22a11.5 11.5 0 0 0-9.84 0L5.22 5.92a.38.38 0 0 0-.66.38L6.4 9.48A10.8 10.8 0 0 0 1 18h22a10.8 10.8 0 0 0-5.4-8.52zM7 15.25a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5zm10 0a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5z"/></svg>',
    printer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9V2h12v7M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 14h12v8H6z"/></svg>',
    ruler: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.4 2.4 0 0 1 0-3.4l2.6-2.6a2.4 2.4 0 0 1 3.4 0zM14.5 12.5l2-2M11.5 9.5l2-2M8.5 6.5l2-2M17.5 15.5l2-2"/></svg>',
    share: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/></svg>',
    split: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
    star: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 2.5l2.9 6 6.6.9-4.8 4.6 1.2 6.5L12 17.4l-5.9 3.1 1.2-6.5-4.8-4.6 6.6-.9z"/></svg>',
    note: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
    coin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M15.5 8.5A4.5 4.5 0 1 0 15.5 15.5M7 10.5h6M7 13.5h6"/></svg>',
    edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
    video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>',
    sparkle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2z"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/></svg>',
    room: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5 12 3l9 6.5V21H3z"/><path d="M9 21v-6h6v6"/></svg>',
    back: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>',
    logout: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>',
  };
  Object.assign(ICONS, {
    heart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>',
    flag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7"/></svg>',
    globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    eyeoff: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-2.4 3.3M6.6 6.6A17 17 0 0 0 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6M9.9 9.9a3 3 0 0 0 4.2 4.2M2 2l20 20"/></svg>',
    bulb: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    rotate: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg>',
    target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
    support: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 4h18v4H3z"/><path d="M7 8v11M12 8v11M17 8v11M5 20h4M10 20h4M15 20h4"/></svg>',
    lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
    plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>',
    shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/></svg>',
    mail: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 6-10 7L2 6"/></svg>',
    ban: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.9 4.9 14.2 14.2"/></svg>',
    queue: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
    expand: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/></svg>',
  });
  const icon = (name) => el("span", { html: ICONS[name], style: "display:contents" });

  // The app's mark (src/components/Logo.tsx): an object in the middle, an orbit around it and the camera's dot on it.
  const ORBIT = '<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="16" fill="none" stroke="currentColor" stroke-width="4"/><ellipse cx="32" cy="32" rx="28" ry="10" transform="rotate(-24 32 32)" fill="none" stroke="currentColor" stroke-width="3"/><circle cx="41.3" cy="18.4" r="5" fill="currentColor"/></svg>';
  // The app icon: the scan frame around a cube of printed layers (accent shades, see --logo-* in style.css).
  const LOGO = '<svg viewBox="0 0 64 64" aria-hidden="true"><defs><mask id="logo-layers"><rect width="64" height="64" fill="#fff"/><path d="M20 29.7 32 36.7 44 29.7M20 34.3 32 41.3 44 34.3" fill="none" stroke="#000" stroke-width="1.6"/></mask></defs><path d="M8 20v-6a6 6 0 0 1 6-6h6M44 8h6a6 6 0 0 1 6 6v6M56 44v6a6 6 0 0 1-6 6h-6M20 56h-6a6 6 0 0 1-6-6v-6" fill="none" stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/><g mask="url(#logo-layers)"><path class="t" d="M32 18 44 25 32 32 20 25z"/><path class="l" d="M20 25 32 32v14l-12-7z"/><path class="r" d="M32 32 44 25v14l-12 7z"/></g></svg>';

  function initials(name) {
    return (name || "?").trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?";
  }

  // The user's picture: their own uploaded photo (needs the token), else the Google picture, else initials.
  async function avatarInto(node, user) {
    node.textContent = initials(user && user.name);
    if (!user) return;
    let src = null;
    if (user.has_avatar) {
      try { src = URL.createObjectURL(await cachedBlob(`avatar/${user.id}`, "/me/avatar.jpg")); } catch {}
    }
    if (!src && user.google_picture) src = user.google_picture;
    if (src) {
      const img = el("img", { src, alt: "", referrerpolicy: "no-referrer" });
      img.onerror = () => { node.textContent = initials(user.name); };
      node.replaceChildren(img);
    }
  }

  // Someone else's picture in the community: the server only gives the photos of people who posted there.
  async function authorAvatar(node, a) {
    node.textContent = initials(a.name);
    let src = null;
    if (a.has_avatar) {
      try { src = URL.createObjectURL(await cachedBlob(`author/${a.id}`, `/community/authors/${a.id}/avatar.jpg`)); } catch {}
    }
    if (!src && a.google_picture) src = a.google_picture;
    if (src) {
      const img = el("img", { src, alt: "", referrerpolicy: "no-referrer" });
      img.onerror = () => { node.textContent = initials(a.name); };
      node.replaceChildren(img);
    }
  }

  // Dark by default; the choice is kept in this browser. Each page's head sets it before anything is drawn.
  const THEME_KEY = "orbitize.theme";
  const theme = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    store.set(THEME_KEY, t);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = t === "light" ? "#f6f7fb" : "#07080d";
    document.querySelectorAll(".theme-btn").forEach(themeButton);
    dispatchEvent(new CustomEvent("themechange", { detail: t }));
  }
  function themeButton(b) {
    const dark = theme() === "dark";
    b.replaceChildren(icon(dark ? "sun" : "moon"));
    b.title = dark ? "Светла тема" : "Тъмна тема";
    b.setAttribute("aria-label", b.title);
  }

  const isAdmin = (u) => !!(u && (u.admin || u.owner));

  function header(active) {
    const signedIn = !!auth.token();
    const link = (href, text, key, cls = "") =>
      el("a", { href, class: `link ${cls} ${active === key ? "active" : ""}`, "aria-current": active === key ? "page" : null }, text);
    const nav = el("nav", { class: "nav", "aria-label": "Главно меню" });
    if (signedIn) {
      const user = auth.user();
      // Phones get the short names, so the links and the picture fit.
      const admin = link("admin.html", [el("span", { class: "long" }, "Администриране"), el("span", { class: "short" }, "Админ")], "admin", "admin-link");
      if (!isAdmin(user)) admin.classList.add("hidden");
      nav.append(link("scans.html", [el("span", { class: "long" }, "Моите сканирания"), el("span", { class: "short" }, "Сканирания")], "scans"),
        link("engineer.html", [el("span", { class: "long" }, "Инженерен режим"), el("span", { class: "short" }, "CAD")], "engineer"),
        link("community.html", "Общност", "community", "hide-xs"), admin);
      const av = el("span", { class: "avatar" });
      avatarInto(av, user);
      // The profile from the server on every page (name, picture, admin rights may have changed since).
      api("/me", { noRedirect: !user }).then((me) => {
        auth.setUser(me);
        admin.classList.toggle("hidden", !isAdmin(me));
      }).catch(() => {});
      nav.append(el("a", { href: "account.html", class: `account-link ${active === "account" ? "active" : ""}`,
        title: "Вашият профил", "aria-label": "Вашият профил", "aria-current": active === "account" ? "page" : null }, av));
    } else {
      nav.append(link("index.html#modes", "Как работи", "", "hide-sm"));
      nav.append(link("login.html", "Вход", "login"));
      nav.append(el("a", { href: C.apkUrl, class: "btn btn-primary btn-sm", style: "margin-left:4px" }, icon("android"), "Изтегли"));
    }
    const themeBtn = el("button", { type: "button", class: "theme-btn", onclick: () => setTheme(theme() === "dark" ? "light" : "dark") });
    themeButton(themeBtn);
    nav.prepend(themeBtn);
    const h = el("header", { class: "header" },
      el("div", { class: "wrap" },
        el("a", { href: "index.html", class: "brand", "aria-label": `${C.appName}, начална страница` },
          el("span", { class: "logo orbit", html: ORBIT }), C.appName),
        nav));
    addEventListener("scroll", () => h.classList.toggle("scrolled", scrollY > 4), { passive: true });
    return h;
  }

  function footer() {
    return el("footer", { class: "footer" },
      el("div", { class: "wrap" },
        el("div", { class: "about" },
          el("a", { href: "index.html", class: "brand" }, el("span", { class: "logo orbit", html: ORBIT }), C.appName),
          el("p", {}, "Обиколи. Измери. Отпечатай. Реален предмет → 3D модел в реален размер, готов за 3D печат.")),
        el("nav", { "aria-label": "Долно меню" },
          el("a", { href: "index.html#modes" }, "Как работи"),
          el("a", { href: "index.html#faq" }, "Въпроси"),
          el("a", { href: C.apkUrl }, "Приложение за Android"),
          el("a", { href: "privacy.html" }, "Поверителност"),
          el("a", { href: "terms.html" }, "Условия")),
        el("div", { class: "copy" }, `© ${new Date().getFullYear()} ${C.appName}. Всички права запазени.`)));
  }

  // The server's scan pictures have a light background and a soft shadow baked in (backend app/render.py: a
  // gradient from BG_TOP to BG_BOTTOM, the shadow a blend towards SHADE). Here the background is taken out and the
  // shadow kept as a see-through shade, so the model sits on whatever the page is (dark or light). Only what joins
  // the picture's edge is touched; the shadow only near the bottom, where it is drawn, so grey models stay whole.
  const BG_TOP = [246, 248, 252], BG_BOTTOM = [228, 234, 243], SHADE = [120, 130, 150];
  async function cutout(blob) {
    const bmp = await createImageBitmap(blob);
    const w = bmp.width, h = bmp.height, c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const img = ctx.getImageData(0, 0, w, h), d = img.data;
    // How much shadow lies over the background at pixel p (0 = plain background), or -1: part of the model.
    const shade = (p, y) => {
      const t = y / (h - 1);
      let sum = 0, lo = 9, hi = -9;
      for (let k = 0; k < 3; k++) {
        const b = BG_TOP[k] * (1 - t) + BG_BOTTOM[k] * t, v = (b - d[p * 4 + k]) / (b - SHADE[k]);
        sum += v; lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
      const v = sum / 3;
      return hi - lo < 0.07 && v > -0.08 && v < 0.38 ? Math.max(0, v) : -1;
    };
    const state = new Uint8Array(w * h); // 1 background, 2 model
    const flood = (seeds, test) => {
      const stack = seeds;
      while (stack.length) {
        const p = stack.pop();
        if (state[p]) continue;
        const y = (p / w) | 0, x = p - y * w;
        if (!test(p, y)) continue;
        state[p] = 1;
        if (x > 0) stack.push(p - 1);
        if (x < w - 1) stack.push(p + 1);
        if (y > 0) stack.push(p - w);
        if (y < h - 1) stack.push(p + w);
      }
    };
    const edges = [];
    for (let x = 0; x < w; x++) edges.push(x, (h - 1) * w + x);
    for (let y = 0; y < h; y++) edges.push(y * w, y * w + w - 1);
    // Plain background first; the lowest thing left is the shadow's bottom, so the shadow lies just above it.
    flood(edges.slice(), (p, y) => { const v = shade(p, y); return v >= 0 && v < 0.035; });
    let bottom = 0;
    for (let p = 0; p < w * h; p++) if (!state[p]) bottom = (p / w) | 0;
    const top = bottom - 0.18 * h, seeds = [];
    for (let p = 0; p < w * h; p++) if (state[p]) {
      const y = (p / w) | 0;
      if (y >= top) for (const q of [p - 1, p + 1, p - w, p + w]) if (q >= 0 && q < w * h && !state[q]) seeds.push(q);
    }
    flood(seeds, (p, y) => y >= top && shade(p, y) >= 0);
    // The picture is a JPEG: its edges carry light specks that are nearly background. Two rings of those go too.
    const near = (p, y) => {
      const t = y / (h - 1);
      let worst = 0;
      for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(BG_TOP[k] * (1 - t) + BG_BOTTOM[k] * t - d[p * 4 + k]));
      return worst < 26;
    };
    for (let ring = 0; ring < 2; ring++) {
      const add = [];
      for (let p = 0; p < w * h; p++) {
        if (state[p]) continue;
        const y = (p / w) | 0, x = p - y * w;
        if (((x > 0 && state[p - 1] === 1) || (x < w - 1 && state[p + 1] === 1) || (y > 0 && state[p - w] === 1) || (y < h - 1 && state[p + w] === 1)) && near(p, y)) add.push(p);
      }
      for (const p of add) state[p] = 1;
    }
    for (let p = 0; p < w * h; p++) if (state[p] === 1) {
      const v = Math.max(0, shade(p, (p / w) | 0));
      d[p * 4] = 8; d[p * 4 + 1] = 8; d[p * 4 + 2] = 14;
      d[p * 4 + 3] = Math.round(Math.min(1, v / 0.35) * 120);
    }
    ctx.putImageData(img, 0, 0);
    return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error("cutout"))), "image/png"));
  }

  // Parts marked .reveal fade in once they scroll into view.
  function reveal() {
    const items = document.querySelectorAll(".reveal");
    if (/[?&]noanim\b/.test(location.search)) { document.documentElement.classList.add("noanim"); return items.forEach((n) => n.classList.add("in")); } // screenshots
    if (!("IntersectionObserver" in window)) return items.forEach((n) => n.classList.add("in"));
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { rootMargin: "0px 0px -8% 0px" });
    items.forEach((n) => io.observe(n));
  }

  // Every page calls this: fills in the app name, adds the header and footer.
  function page({ title, active } = {}) {
    document.title = title ? `${title} · ${C.appName}` : C.appName;
    document.querySelectorAll("[data-app-name]").forEach((n) => (n.textContent = C.appName));
    document.querySelectorAll("[data-apk]").forEach((n) => n.setAttribute("href", C.apkUrl));
    const main = document.querySelector("main");
    if (main) { main.id = "main"; main.tabIndex = -1; }
    document.body.prepend(el("a", { class: "skip", href: "#main" }, "Към съдържанието"), header(active));
    document.body.append(footer());
    reveal();
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = theme() === "light" ? "#f6f7fb" : "#07080d";
    addEventListener("offline", () => toast("Няма интернет връзка."));
    addEventListener("online", () => toast("Връзката е възстановена."));
  }

  let toastTimer;
  function toast(text) {
    let t = document.querySelector(".toast");
    if (!t) { t = el("div", { class: "toast", role: "status", "aria-live": "polite" }); document.body.append(t); }
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), Math.max(2600, text.length * 60)); // longer text, longer time
  }

  // A yes/no box. Returns a promise of true/false. confirmWord: the user must type it to confirm.
  function confirmBox({ title, text, ok = "Добре", danger = false, confirmWord }) {
    return new Promise((resolve) => {
      const input = confirmWord ? el("input", { class: "input", autocomplete: "off", placeholder: confirmWord }) : null;
      const okBtn = el("button", { class: `btn ${danger ? "btn-danger" : "btn-primary"}`, disabled: !!confirmWord }, ok);
      const d = el("dialog", {},
        el("h3", {}, title), el("p", { class: "muted" }, text),
        input ? el("label", { class: "field" }, el("span", {}, `Напишете ${confirmWord}, за да потвърдите`), input) : null,
        el("div", { class: "row-actions" },
          el("button", { class: "btn btn-secondary", onclick: () => d.close("no") }, "Отказ"), okBtn));
      if (input) input.addEventListener("input", () => (okBtn.disabled = input.value.trim() !== confirmWord));
      okBtn.addEventListener("click", () => d.close("yes"));
      d.addEventListener("close", () => { resolve(d.returnValue === "yes"); d.remove(); });
      document.body.append(d);
      d.showModal();
    });
  }

  const STATES = {
    uploading: ["Качва се", "warn busy"],
    queued: ["Чака", "busy"],
    reconstructing: ["Изгражда се", "busy"],
    meshing: ["Изгражда се", "busy"],
    cleaning: ["Завършва", "busy"],
    done: ["Готово", "ok"],
    failed: ["Неуспешно", "bad"],
    cancelled: ["Отказано", "bad"],
  };
  const ACTIVE = new Set(["queued", "reconstructing", "meshing", "cleaning"]);
  function stateBadge(state) {
    const [text, cls] = STATES[state] || [state || "Неизвестно", ""];
    return el("span", { class: `badge ${cls}` }, el("span", { class: "dot" }), text);
  }

  function date(unix) {
    if (!unix) return "";
    return new Date(unix * 1000).toLocaleDateString("bg-BG", { day: "numeric", month: "long", year: "numeric" });
  }

  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  window.S = { C, isAdmin, ORBIT, LOGO, auth, api, ApiError, requireAuth, el, icon, page, toast, confirmBox, stateBadge, ACTIVE, date,
    saveBlob, avatarInto, authorAvatar, initials, scansCache, marks, cachedBlob, forgetFiles, theme, setTheme, reveal, cutout };
})();

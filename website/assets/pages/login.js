  S.page({ title: "Вход", active: "login" });
  const pwToggle = document.getElementById("pw-toggle"), pwInput = document.getElementById("password");
  pwToggle.append(S.icon("eye"));
  pwToggle.onclick = () => {
    const show = pwInput.type === "password";
    pwInput.type = show ? "text" : "password";
    pwToggle.replaceChildren(S.icon(show ? "eyeoff" : "eye"));
    pwToggle.setAttribute("aria-label", show ? "Скрий паролата" : "Покажи паролата");
    pwToggle.setAttribute("aria-pressed", String(show));
  };
  // Cursor in the email box on computers (on phones it would pop up the keyboard over the page).
  if (matchMedia("(hover: hover)").matches) document.querySelector("[name=email]").focus();
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  // Only go back to one of our own pages after sign-in, never to another site.
  const next = /^[a-z]+\.html(\?[\w=&%.-]*)?$/.test(params.get("next") || "") ? params.get("next") : "scans.html";
  if (S.auth.token()) location.replace(next);

  let mode = "in";
  function setMode(m) {
    mode = m;
    const up = m === "up";
    $("tab-in").classList.toggle("on", !up);
    $("tab-up").classList.toggle("on", up);
    $("name-field").classList.toggle("hidden", !up);
    $("pw-hint").classList.toggle("hidden", !up);
    $("submit").textContent = up ? "Създай профил" : "Вход";
    $("heading").textContent = up ? "Създайте профил" : "Добре дошли отново";
    $("sub").textContent = up ? "Моделите ви се пазят в него, в приложението и тук." : "Влезте със същия профил като в приложението.";
    $("form").password.autocomplete = up ? "new-password" : "current-password";
    showError(null);
  }
  $("tab-in").onclick = () => setMode("in");
  $("tab-up").onclick = () => setMode("up");
  if (params.get("mode") === "register") setMode("up");

  function showError(text) {
    $("msg").textContent = text || "";
    $("msg").classList.toggle("hidden", !text);
  }

  // Нов профил (или непотвърден при вход): сървърът праща 6-цифрен код на имейла и чака него, без сесия.
  let pending = null;
  function done(res) {
    if (res.verify) return askCode(res.email);
    S.auth.save(res.token, res.user);
    location.replace(next);
  }
  function askCode(email) {
    pending = email;
    $("form").classList.add("hidden");
    document.querySelector(".tabs").classList.add("hidden");
    $("code-form").classList.remove("hidden");
    $("heading").textContent = "Потвърдете имейла";
    $("sub").textContent = "Последна стъпка преди профила.";
    $("code-text").textContent = `Изпратихме 6-цифрен код на ${email}. Въведете го тук (проверете и нежеланата поща).`;
    $("code-form").code.value = "";
    $("code-form").code.focus();
    $("submit").disabled = false;
    showError(null);
  }
  $("code-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const code = e.target.code.value.replace(/\D/g, "");
    if (code.length !== 6) return showError("Кодът е 6 цифри.");
    $("code-submit").disabled = true;
    showError(null);
    try {
      done(await S.api("/auth/verify", { method: "POST", json: { email: pending, code } }));
    } catch (err) {
      showError(err.message);
      $("code-submit").disabled = false;
    }
  });
  $("code-resend").onclick = async () => {
    showError(null);
    try {
      await S.api("/auth/resend", { method: "POST", json: { email: pending } });
      S.toast("Изпратихме нов код.");
    } catch (err) { showError(err.message); }
  };
  $("code-back").onclick = () => {
    $("code-form").classList.add("hidden");
    $("form").classList.remove("hidden");
    document.querySelector(".tabs").classList.remove("hidden");
    setMode(mode);
  };

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    const email = f.email.value.trim(), password = f.password.value;
    if (!email || !password) return showError("Въведете имейл и парола.");
    if (mode === "up" && !(password.length >= 8 && /\p{L}/u.test(password) && /\d/.test(password)))
      return showError("Паролата трябва да е поне 8 знака и да има поне една буква и една цифра.");
    $("submit").disabled = true;
    showError(null);
    try {
      const json = mode === "up" ? { email, password, name: f.name.value.trim() } : { email, password };
      done(await S.api(mode === "up" ? "/auth/register" : "/auth/login", { method: "POST", json }));
    } catch (err) {
      showError(err.message);
      $("submit").disabled = false;
    }
  });

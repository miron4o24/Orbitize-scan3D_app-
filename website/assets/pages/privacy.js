  S.page({ title: "Политика за поверителност" });
  if (S.C.contactEmail) {
    const p = document.getElementById("contact");
    p.textContent = "Въпроси или искания за данните ви? Пишете ни на ";
    p.append(S.el("a", { href: "mailto:" + S.C.contactEmail }, S.C.contactEmail), ".");
  }

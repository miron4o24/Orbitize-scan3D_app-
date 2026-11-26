// Тема преди показване на страницата (без проблясване); отделен файл, защото политиката (CSP) не позволява вграден код.
(function(d){var t="dark";try{t=localStorage.getItem("orbitize.theme")||t}catch(e){}d.dataset.theme=t;d.classList.add("js")})(document.documentElement);

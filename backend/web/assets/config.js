// Orbitize website settings. Everything here is public: never put a password, the API_TOKEN or any other secret here.
// The website is served by the Orbitize server itself (backend/app/main.py), so the API is at the same address ("").
window.ORBITIZE = {
  appName: "Orbitize",
  apiUrl: "",
  // The newest Android build (EAS). Change it after every new build.
  apkUrl: "https://expo.dev/artifacts/eas/7FLF6ehuQ87Gebihi4SxF4gKjHH0k32_IHYezZmIysA.apk",
  googleClientId: "814464272568-p3lknkud9vg4alfcn1beq695bjtffc8u.apps.googleusercontent.com",
  // Shown on the Privacy and Terms pages. Fill it in before asking Google to verify the sign-in.
  contactEmail: "",
  printerMarginMm: 5, // kept free around the model on the plate
  // Build volumes in mm: [brand, model, width (X), depth (Y), height (Z)]. Add or change printers here.
  printers: [
    ["Bambu Lab", "A1 mini", 180, 180, 180],
    ["Bambu Lab", "A1", 256, 256, 256],
    ["Bambu Lab", "P1P", 256, 256, 256],
    ["Bambu Lab", "P1S", 256, 256, 256],
    ["Bambu Lab", "P2S", 256, 256, 256],
    ["Bambu Lab", "X1 Carbon", 256, 256, 256],
    ["Bambu Lab", "X1E", 256, 256, 256],
    ["Bambu Lab", "H2D", 325, 320, 325],

    ["Prusa", "MINI+", 180, 180, 180],
    ["Prusa", "MK3S+", 250, 210, 210],
    ["Prusa", "MK4 / MK4S", 250, 210, 220],
    ["Prusa", "CORE One", 250, 220, 270],
    ["Prusa", "XL", 360, 360, 360],

    ["Creality", "Ender-3 / Ender-3 Pro", 220, 220, 250],
    ["Creality", "Ender-3 V2", 220, 220, 250],
    ["Creality", "Ender-3 S1 / S1 Pro", 220, 220, 270],
    ["Creality", "Ender-3 S1 Plus", 300, 300, 300],
    ["Creality", "Ender-3 V3 SE", 220, 220, 250],
    ["Creality", "Ender-3 V3 KE", 220, 220, 240],
    ["Creality", "Ender-3 V3", 220, 220, 250],
    ["Creality", "Ender-3 V3 Plus", 300, 300, 330],
    ["Creality", "Ender-5 S1", 220, 220, 280],
    ["Creality", "Ender-5 Plus", 350, 350, 400],
    ["Creality", "CR-10 SE", 220, 220, 265],
    ["Creality", "CR-10 Smart Pro", 300, 300, 400],
    ["Creality", "K1 / K1C", 220, 220, 250],
    ["Creality", "K1 Max", 300, 300, 300],
    ["Creality", "K2 Plus", 350, 350, 350],
    ["Creality", "Hi", 260, 260, 300],

    ["Elegoo", "Neptune 3 Pro", 225, 225, 280],
    ["Elegoo", "Neptune 4 / 4 Pro", 225, 225, 265],
    ["Elegoo", "Neptune 4 Plus", 320, 320, 385],
    ["Elegoo", "Neptune 4 Max", 420, 420, 480],
    ["Elegoo", "Centauri Carbon", 256, 256, 256],

    ["Anycubic", "Kobra 2 Neo", 220, 220, 250],
    ["Anycubic", "Kobra 2 Pro", 220, 220, 250],
    ["Anycubic", "Kobra 2 Plus", 320, 320, 400],
    ["Anycubic", "Kobra 2 Max", 420, 420, 500],
    ["Anycubic", "Kobra 3", 250, 250, 260],
    ["Anycubic", "Kobra S1", 250, 250, 250],
    ["Anycubic", "Vyper", 245, 245, 260],

    ["Sovol", "SV06", 220, 220, 250],
    ["Sovol", "SV06 Plus", 300, 300, 340],
    ["Sovol", "SV07", 220, 220, 250],
    ["Sovol", "SV08", 350, 350, 345],

    ["QIDI", "Q1 Pro", 245, 245, 240],
    ["QIDI", "X-Smart 3", 175, 180, 170],
    ["QIDI", "X-Plus 3", 280, 280, 270],
    ["QIDI", "X-Max 3", 325, 325, 315],
    ["QIDI", "Plus4", 305, 305, 280],

    ["Flashforge", "Adventurer 5M / 5M Pro", 220, 220, 220],
    ["Flashforge", "AD5X", 220, 220, 220],
    ["Flashforge", "Creator 4", 400, 350, 500],

    ["Artillery", "Genius Pro", 220, 220, 250],
    ["Artillery", "Sidewinder X2", 300, 300, 400],
    ["Artillery", "Sidewinder X4 Plus", 300, 300, 400],
    ["Artillery", "Sidewinder X4 Pro", 240, 240, 250],

    ["Anker", "AnkerMake M5", 235, 235, 250],
    ["Anker", "AnkerMake M5C", 220, 220, 250],

    ["Voron", "0.2", 120, 120, 120],
    ["Voron", "Trident 250", 250, 250, 250],
    ["Voron", "Trident 300", 300, 300, 250],
    ["Voron", "Trident 350", 350, 350, 250],
    ["Voron", "2.4 (250)", 250, 250, 250],
    ["Voron", "2.4 (300)", 300, 300, 300],
    ["Voron", "2.4 (350)", 350, 350, 350],

    ["Ultimaker", "2+ Connect", 223, 220, 205],
    ["Ultimaker", "S3", 230, 190, 200],
    ["Ultimaker", "S5 / S7", 330, 240, 300],

    ["Raise3D", "E2", 330, 240, 240],
    ["Raise3D", "Pro3", 300, 300, 300],

    ["Snapmaker", "J1", 300, 200, 200],
    ["Snapmaker", "Artisan", 400, 400, 400],

    ["Други", "Kingroon KP3S", 180, 180, 180],
    ["Други", "Tronxy X5SA", 330, 330, 400],

    ["Смола (резин)", "Elegoo Mars 4", 133, 75, 150],
    ["Смола (резин)", "Elegoo Saturn 3", 219, 123, 250],
    ["Смола (резин)", "Elegoo Saturn 4 Ultra", 219, 123, 220],
    ["Смола (резин)", "Anycubic Photon Mono 4", 153, 87, 165],
    ["Смола (резин)", "Anycubic Photon Mono M5s", 219, 123, 200],
    ["Смола (резин)", "Phrozen Sonic Mini 8K S", 165, 72, 170],
    ["Смола (резин)", "Formlabs Form 3+", 145, 145, 185],
    ["Смола (резин)", "Formlabs Form 4", 200, 125, 210],
  ].map(([brand, model, x, y, z]) => ({
    brand, model, x, y, z,
    name: brand === "Други" || brand.startsWith("Смола") ? model : `${brand} ${model}`,
  })),
};

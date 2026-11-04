// app.json плюс файла на Firebase за известия, когато е в проекта. Без файла изграждането работи, но без известия
// (изграждането се проваля, ако тук е посочен липсващ файл).
const fs = require('fs');

module.exports = ({ config }) => ({
  ...config,
  android: {
    ...config.android,
    ...(fs.existsSync('./google-services.json') && { googleServicesFile: './google-services.json' }),
  },
});

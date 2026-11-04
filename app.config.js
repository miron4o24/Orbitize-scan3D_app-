// app.json, plus Firebase's file for push notifications once it is in the project. Without the file a build still works,
// just with no notifications (a build fails on a file that is named here but missing).
const fs = require('fs');

module.exports = ({ config }) => ({
  ...config,
  android: {
    ...config.android,
    ...(fs.existsSync('./google-services.json') && { googleServicesFile: './google-services.json' }),
  },
});

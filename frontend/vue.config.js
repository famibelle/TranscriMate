const { defineConfig } = require('@vue/cli-service')
const { DefinePlugin } = require('webpack');

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',
    port: 8080,
    client: {
      webSocketURL: process.env.VUE_APP_WEBSOCKET_URL
    },
    allowedHosts: 'all', // Autoriser tous les hôtes pour éviter Invalid Host header
  },
  pages: {
    index: {
      entry: 'src/main.js',  // Point d'entrée de l'application
      title: 'TranscriMate'  // Nouveau titre pour l'onglet du navigateur
    }
  },
  configureWebpack: {
    plugins: [
      new DefinePlugin({
        // Définir les feature flags Vue.js pour supprimer les warnings
        __VUE_OPTIONS_API__: JSON.stringify(true),
        __VUE_PROD_DEVTOOLS__: JSON.stringify(false),
        __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: JSON.stringify(false)
      })
    ]
  }
});
 
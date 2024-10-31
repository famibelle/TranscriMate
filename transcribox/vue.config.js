const { defineConfig } = require('@vue/cli-service')
const HtmlWebpackPlugin = require('html-webpack-plugin'); // Importer HtmlWebpackPlugin


module.exports = defineConfig({
  transpileDependencies: true
})

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',
    port: 8080,
    client: {
      // webSocketURL: 'ws://localhost:8080/ws', // Configuration WebSocket
      // webSocketURL: `${process.env.NODE_ENV === 'production' || process.env.USE_HTTPS ? 'wss' : 'ws'}://${process.env.DEV_SERVER_HOST || 'localhost'}:8080/ws`
      webSocketURL: `${process.env.NODE_ENV === 'production' || process.env.VUE_APP_USE_HTTPS ? 'wss' : 'ws'}://${process.env.VUE_APP_DEV_SERVER_HOST || 'localhost'}:8080/ws`
      // webSocketURL: process.env.NODE_ENV === 'production' ? process.env.VUE_APP_WEBSOCKET_URL : `wss://${process.env.VUE_APP_DEV_SERVER_HOST || 'localhost'}:8080/ws`
    },
    allowedHosts: 'all', // Autoriser tous les hôtes pour éviter Invalid Host header
  },
  pages: {
    index: {
      entry: 'src/main.js',  // Point d'entrée de l'application
      title: 'Transcriptor'  // Nouveau titre pour l'onglet du navigateur
    }
  }
});

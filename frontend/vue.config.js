const { defineConfig } = require('@vue/cli-service')
const HtmlWebpackPlugin = require('html-webpack-plugin'); // Importer HtmlWebpackPlugin

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
  }
});
 
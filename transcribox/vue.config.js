const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true
})

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',
    port: 8080,
    client: {
      webSocketURL: 'ws://localhost:8080/ws', // Configuration WebSocket
    },
    allowedHosts: 'all', // Autoriser tous les hôtes pour éviter Invalid Host header
  },
});

const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true
})

module.exports = {
  devServer: {
    host: 'localhost',  // Utiliser localhost comme hôte
    port: 8080,         // Le port que tu utilises
    client: {
      webSocketURL: 'ws://localhost:8080/ws', // Configuration pour le WebSocket
    },
  },
};

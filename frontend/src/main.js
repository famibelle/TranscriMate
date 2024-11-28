import { createApp } from 'vue' 
import App from './App.vue'

// Importation du composant Toggle et de son style
import Toggle from '@vueform/toggle'
import '@vueform/toggle/themes/default.css'

// Création de l'application
const app = createApp(App)

// Enregistrement du composant Toggle sous un nom multi-mots
app.component('ToggleSwitch', Toggle) // Renommage en "ToggleSwitch" par exemple

// Montage de l'application
app.mount('#app')


// import { createApp } from 'vue'
// import App from './App.vue'

// createApp(App).mount('#app')

// // Dans main.js
// import Toggle from '@vueform/toggle'
// import '@vueform/toggle/themes/default.css'

// const app = createApp(App)
// app.component('Toggle', Toggle)
// app.mount('#app')

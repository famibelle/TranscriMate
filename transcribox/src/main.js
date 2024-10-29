import { createApp } from 'vue'
import App from './App.vue'

createApp(App).mount('#app')



// Dans main.js
import Toggle from '@vueform/toggle'
import '@vueform/toggle/themes/default.css'

const app = createApp(App)
app.component('Toggle', Toggle)
app.mount('#app')

<template>
  <!-- Utilisez v-html pour afficher le contenu rendu en Markdown -->
  <div v-html="renderedMarkdown"></div>
</template>

<script>
import { marked } from 'marked';

export default {
  props: {
    content: {
      type: String,
      required: true
    },
    typingSpeed: {
      type: Number,
      default: 0.01 // Valeur par défaut en millisecondes, modifiable via la prop
    }
  },
  data() {
    return {
      displayedContent: '', // Contenu progressif pour l'effet de streaming
      intervalId: null // ID de l'intervalle pour contrôler l'animation
    };
  },
  computed: {
    renderedMarkdown() {
      return marked(this.displayedContent);
    }
  },
  watch: {
    // Surveille les changements dans `content` pour redémarrer l'effet
    content(newContent) {
      this.startStreaming(newContent);
    }
  },
  mounted() {
    // Démarre l'effet de streaming au montage du composant
    this.startStreaming(this.content);
  },
  methods: {
    startStreaming(content) {
      // Réinitialise `displayedContent` et arrête l'intervalle existant
      this.displayedContent = '';
      if (this.intervalId) clearInterval(this.intervalId);

      // Vérifie si la vitesse est définie à 0 pour afficher tout le contenu d'un coup
      if (this.typingSpeed === 0) {
        this.displayedContent = content; // Affiche tout le texte d'un coup
        return; // Arrête la fonction ici pour éviter l'intervalle
      }

      // Variable pour garder la position actuelle dans `content`
      let index = 0;

      // Déclenche un intervalle pour ajouter les caractères progressivement
      this.intervalId = setInterval(() => {
        // Ajoute le caractère suivant
        this.displayedContent += content[index];
        index++;

        // Arrête l'intervalle lorsque tout le texte est affiché
        if (index >= content.length) {
          clearInterval(this.intervalId);
          this.intervalId = null;
        }
      }, this.typingSpeed); // Utilise la valeur de `typingSpeed` pour ajuster la vitesse
    }
  },
  beforeDestroy() {
    // Nettoie l'intervalle lorsque le composant est détruit
    if (this.intervalId) clearInterval(this.intervalId);
  }
};
</script>

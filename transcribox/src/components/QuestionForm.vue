<template>
  <div>
    <!-- Affiche la réponse en Markdown, placé en dehors du formulaire -->
    <div v-if="response" style="margin-bottom: 20px;">
      <MarkdownRenderer :content="response" />
    </div>
    
    <!-- Formulaire avec le champ de texte et le bouton, alignés en ligne -->
    <form @submit.prevent="askQuestion" style="display: flex; align-items: center;">
      <textarea
        v-model="question"
        ref="questionInput"
        placeholder="Posez une question"
        class="chatbot-textarea"
      ></textarea>
      
      <!-- Bouton rond avec une flèche vers le haut -->
      <button type="submit" class="submit-button">
        <span class="arrow">↑</span>
      </button>
    </form>
  </div>
</template>

<script>
import MarkdownRenderer from './MarkdownRenderer.vue';

export default {
  components: {
    MarkdownRenderer
  },

  props: {
    fullTranscription: String, // Transcription passée en prop depuis App.vue
  },
  data() {
    return {
      question: 'Fais une synthèse structurée, mets en gras les points importants et ne dépasse pas 500 mots',
      response: '',
    };
  },

  mounted() {
    // Met le focus sur l'input et sélectionne le texte par défaut
    this.$nextTick(() => {
      this.$refs.questionInput.focus();
      this.$refs.questionInput.select();
    });
  },

  methods: {
    async askQuestion() {
      this.response = ''; // Réinitialise la réponse pour chaque nouvelle question

      // Prépare les données de la requête
      const requestData = {
        question: this.question,
        transcription: this.fullTranscription,
      };

      // Utilise fetch pour envoyer une requête POST et gérer la réponse en continu
      const response = await fetch('/ask_question/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      // Gère la lecture en streaming
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      // Lit et décode les données en continu
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        this.response += decoder.decode(value, { stream: true }); // Affiche progressivement chaque segment
      }
    },
  },
};
</script>

<style scoped>
form {
  margin-bottom: 20px;
}
input {
  /* padding: 10px; */
  font-size: 16px;
  margin-right: 10px;
  padding: 0.5em;

}

/* Ajoute des styles supplémentaires si besoin */
textarea {
  font-size: 16px;
  padding: 0.5em;
  line-height: 1.5; /* Pour espacer légèrement les lignes */
}

button {
  padding: 10px;
  font-size: 16px;
}
h3 {
  margin-top: 20px;
}




/* Styles pour le bouton rond avec une flèche pointant vers le haut */
.submit-button {
  width: 40px;                  /* Taille du bouton */
  height: 40px;                 /* Taille du bouton */
  border-radius: 50%;           /* Forme ronde */
  background-color: white;      /* Couleur de fond */
  color: black;                 /* Couleur de la flèche */
  font-size: 1.2em;             /* Taille de la flèche */
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  cursor: pointer;
  margin-left: 10px;            /* Espace entre le bouton et le textarea */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); /* Ombre pour effet de relief */
  padding: 0;                   /* Retire le padding pour garder une forme parfaite */
  box-sizing: border-box;       /* Pour assurer un alignement précis */
  transition: transform 0.2s;   /* Transition pour l'effet au clic */
}

.submit-button:active {
  transform: scale(0.95);       /* Effet de pression au clic */
}

.arrow {
  font-weight: bold;            /* Flèche en gras pour la visibilité */
}

.submit-button:hover {
  background-color: #0056b3; /* Couleur au survol */
}

/* Styles pour le textarea type chatbot */
.chatbot-textarea {
  flex-grow: 1;              /* Prend tout l'espace disponible */
  height: 2.5em;
  border: none;
  border-radius: 20px;
  padding: 0.5em 1em;
  font-size: 16px;
  background-color: #f1f1f1;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
  resize: none;
  overflow-wrap: break-word;
  outline: none;
  margin-right: 10px;       /* Espace entre le champ de texte et le bouton */
}

</style>

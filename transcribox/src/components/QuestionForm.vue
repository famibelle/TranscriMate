<template>
  <div>
    <!-- Affiche la réponse en Markdown  -->
    <div v-if="response" style="margin-bottom: 20px;">
      <MarkdownRenderer :content="response" />
      <!-- <div>{{response}}</div> -->
      <!-- Bouton de copie avec emoji 📋 -->
      <button @click="copyToClipboard" class="copy-button" title="Copier">
        📋
        <!-- Tooltip pour afficher "Copier" après un délai -->
        <span class="tooltip">Copier</span>
      </button>

    </div>
    
    <!-- Formulaire avec le champ de texte et le bouton, alignés en ligne -->
    <form @submit.prevent="askQuestion" style="display: flex; align-items: center;">
      <textarea
        v-model="question"
        ref="questionInput"
        placeholder="Posez une question"
        class="chatbot-textarea"
        @keydown.enter.prevent="askQuestion"
      ></textarea>
      
      <!-- Bouton rond avec une flèche ou un carré en fonction de l'état de streaming -->
      <button type="submit" class="submit-button" :disabled="isStreamingChatResponse">
        <span v-if="!isStreamingChatResponse" class="arrow">🡹</span> <!-- Affiche une flèche si le streaming n'est pas en cours -->
        <span v-else class="square">■</span> <!-- Affiche un carré pendant le streaming -->
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
    chat_model: String
  },
  data() {
    return {
      question: 'Fais une synthèse structurée',
      response: '',
      isStreamingChatResponse: false // État pour suivre si le streaming de réponse du chat est en cours
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

    // Fonction pour copier la réponse dans le presse-papiers
    copyToClipboard() {
      navigator.clipboard.writeText(this.response).then(() => {
        // alert("Réponse copiée dans le presse-papiers !");
      }).catch(err => {
        console.error("Erreur lors de la copie : ", err);
      });
    },

    async askQuestion() {
      // Réinitialise les états de streaming et de réponse au début de chaque requête
      this.isStreamingChatResponse = true; // Active l'état de streaming
      this.response = ''; // Réinitialise la réponse pour chaque nouvelle question

      // Prépare les données de la requête
      const requestData = {
        question: this.question,
        transcription: this.fullTranscription,
        chat_model: this.chat_model // Utilisation de la prop
      };

      console.log("Données envoyées :", requestData);

      try {
        // Utilise fetch pour envoyer une requête POST et obtenir la réponse complète
        const response = await fetch('/ask_question/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData),
        });

        // Lire la réponse en JSON
        const result = await response.json();
        this.response = result.response;
        console.log("Réponse complète :", this.response);
      } 
      catch (error) {
        console.error("Erreur lors de la récupération de la réponse :", error);
      } 
      finally {
        // Assure que l'état de traitement est désactivé à la fin, même en cas d'erreur
        this.isStreamingChatResponse = false;
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
  font-size: 16px;
  margin-right: 10px;
  padding: 0.5em;
}

/* Styles pour le bouton rond avec une flèche ou un carré */
.submit-button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: white;
  color: black;
  font-size: 1.2em;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  cursor: pointer;
  margin-left: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
  padding: 0;
  box-sizing: border-box;
  transition: transform 0.2s;
}

.submit-button:active {
  transform: scale(0.95);
}

.arrow {
  font-weight: bold;
}

.square {
  font-weight: bold; /* Affiche le carré en gras pendant le streaming */
  animation: pulse 1s ease-in-out infinite; /* Animation de pulsation */
}

.submit-button:hover {
  background-color: #0056b3;
}

/* Styles pour le textarea type chatbot */
.chatbot-textarea {
  flex-grow: 1;
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
  margin-right: 10px;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1); /* Taille initiale */
    opacity: 1;
  }
  50% {
    transform: scale(1.2); /* Taille augmentée pour effet de pulsation */
    opacity: 0.7; /* Légère transparence pour accentuer l'effet */
  }
}

</style>

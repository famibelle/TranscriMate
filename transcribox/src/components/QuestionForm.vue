<template>
  <div>
    <form @submit.prevent="askQuestion">
      <div v-if="response">
      <MarkdownRenderer :content="response" />

    </div>

    <input
        v-model="question"
        ref="questionInput"
        placeholder="Posez une question"
        style="width: 80%; height: 2em;"
      />
      <button type="submit">Envoyer</button>
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
      question: 'Fais une synthèse structurée, de la transcription, mets en gras les points importants',
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
  padding: 10px;
  font-size: 16px;
  margin-right: 10px;
}
button {
  padding: 10px;
  font-size: 16px;
}
h3 {
  margin-top: 20px;
}
</style>

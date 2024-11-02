<template>
    <div>
      <form @submit.prevent="askQuestion">
        <input
          v-model="question"
          type="text"
          placeholder="Entrez votre question ici"
          required
        />
        <button type="submit">Envoyer</button>
      </form>
      <div v-if="response">
        <h3>Réponse :</h3>
        <p>{{ response }}</p>
      </div>
    </div>
</template>
  
  <script>
  export default {
    props: {
      fullTranscription: String,
    },
    data() {
      return {
        question: '',
        response: '',
        eventSource: null,
      };
    },
    methods: {
      askQuestion() {
        this.response = '';
        if (this.eventSource) {
          this.eventSource.close();
        }
  
        // Envoie la transcription et la question en tant que paramètres de requête
        const params = new URLSearchParams({
          question: this.question,
          transcription: this.fullTranscription,
        });
  
        this.eventSource = new EventSource(`/ask_question/?${params.toString()}`);
  
        this.eventSource.onmessage = (event) => {
            console.log("Received data:", event.data);  // Log chaque segment de la réponse
            this.response += event.data;  // Ajoute progressivement chaque partie de la réponse
        };
  
        this.eventSource.onerror = () => {
            console.error("Erreur de connexion avec EventSource");
            this.eventSource.close();
        };
      },
    },
    beforeDestroy() {
      if (this.eventSource) {
        this.eventSource.close();
      }
    },
  };
</script>
  
<template>
  <div id="app">
    <!-- Vue principale affichée après l'upload du fichier -->
    <div v-if="file">
      <div class="file-header">
        <span>{{ file.name }}</span>
        <div class="controls">
          <button @click="openSettings">⚙️</button>
          <button @click="removeFile">✕</button>
        </div>
      </div>

      <div class="audio-player">
        <!-- Bouton de lecture -->
        <button @click="togglePlay">
          <span v-if="isPlaying">⏸️</span>
          <span v-else>▶️</span>
        </button>

        <!-- Barre de progression -->
        <input type="range" min="0" :max="audioDuration" v-model="currentTime" @input="seekAudio" />

        <!-- Affichage du temps actuel et de la durée totale -->
        <span>{{ formatTime(currentTime) }} / {{ formatTime(audioDuration) }}</span>
      </div>

      <!-- Liste des locuteurs et des segments de transcription -->
      <div class="transcriptions">
        <div v-for="(segment, index) in transcriptions" :key="index" class="transcription-segment">
          <strong>{{ segment.speaker }}</strong>

          <!-- Liste des chunks -->
          <span 
            v-for="(chunk, i) in segment.text.chunks" 
            :key="i" 
            class="chunk" 
            @click="playChunk(segment.audio_url, chunk.timestamp[0], chunk.timestamp[1])">
            {{ chunk.text }}
          </span>

          <!-- Player audio pour le segment -->
          <!-- <audio :src="segment.audio_url" controls></audio> -->
        </div>

      <!-- Textarea pour l'ensemble de la transcription -->
      <div v-if="transcriptions.length > 0" class="transcription-full">
        <h3>Transcription complète</h3>
        <textarea v-model="fullTranscription" readonly></textarea>
        <button @click="copyToClipboard">Copier</button>
      </div>

      </div>
    </div>

    <!-- Interface d'upload si aucun fichier n'est sélectionné -->
    <div v-else class="upload-container">
      <div 
        class="upload-box"
        @dragover.prevent
        @drop.prevent="handleDrop"
        @click="triggerFileInput"
      >
        <p>🎵 Déposez votre fichier audio ou vidéo ici</p>
        <p>ou</p>
        <button @click="triggerFileInput">Sélectionnez un fichier</button>
        <p>Formats supportés : MP3, MP4, WAV, WebM</p>
      </div>
      <input
        type="file"
        ref="fileInput"
        @change="onFileChange"
        accept=".mp3,.mp4,.wav,.webm"
        style="display: none"
      />
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      file: null,  // Stocke le fichier sélectionné ou déposé
      audio: null,  // Instance de l'objet Audio
      isPlaying: false,  // Indique si l'audio est en cours de lecture
      currentTime: 0,  // Temps actuel de la lecture
      audioDuration: 0,  // Durée totale de l'audio
      transcriptions: []  // Ce tableau sera rempli par des transcriptions réelles du backend
    };
  },

  computed: {
    // Computed property pour concaténer toute la transcription
    fullTranscription() {
      return this.transcriptions
        .map(segment => {
          const speaker = segment.speaker + ": ";
          const text = segment.text.chunks.map(chunk => chunk.text).join(' ');
          return speaker + text;
        })
        .join('\n\n');  // Ajouter une séparation entre chaque locuteur
    }
  },

  methods: {
    // Méthode pour copier la transcription complète dans le presse-papiers
    copyToClipboard() {
      navigator.clipboard.writeText(this.fullTranscription)
        .then(() => alert('Texte copié dans le presse-papiers !'))
        .catch(err => console.error('Erreur lors de la copie :', err));
    },

    // Méthode pour lire un chunk spécifique
    playChunk(audioUrl, start, end) {
      const audio = new Audio(audioUrl);

      // Démarre la lecture à partir du timestamp 'start'
      audio.currentTime = start;
      audio.play();

      // Arrêter la lecture après la durée du chunk
      const duration = (end - start) * 1000;
      setTimeout(() => {
        audio.pause();
      }, duration);
    },

    // Gère le changement de fichier
    onFileChange(event) {
      const files = event.target.files;
      if (files.length) {
        this.file = files[0];  // Stocke le fichier sélectionné
        this.setupAudio();  // Préparer l'audio
        this.uploadFile();  // Envoyer le fichier au backend et récupérer la transcription
      }
    },

    // Gère le fichier déposé via drag & drop
    handleDrop(event) {
      const files = event.dataTransfer.files;
      if (files.length) {
        this.file = files[0];  // Stocke le fichier déposé
        this.setupAudio();  // Préparer l'audio
        this.uploadFile();  // Envoyer le fichier au backend et récupérer la transcription
      }
    },

    // Prépare l'audio pour lecture
    setupAudio() {
      this.audio = new Audio(URL.createObjectURL(this.file));
      this.audio.addEventListener('loadedmetadata', () => {
        this.audioDuration = this.audio.duration;  // Obtenir la durée de l'audio
      });
      this.audio.addEventListener('timeupdate', () => {
        this.currentTime = this.audio.currentTime;  // Mettre à jour le temps actuel
      });
    },

    // Lire ou mettre en pause l'audio
    togglePlay() {
      if (this.isPlaying) {
        this.audio.pause();
      } else {
        this.audio.play();
      }
      this.isPlaying = !this.isPlaying;
    },

    // Rechercher un moment spécifique dans l'audio
    seekAudio() {
      this.audio.currentTime = this.currentTime;
    },

    // Formater le temps en minutes et secondes
    formatTime(time) {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60).toString().padStart(2, '0');
      return `${minutes}:${seconds}`;
    },

    // Ouvrir les paramètres (à personnaliser)
    openSettings() {
      alert("Ouvrir les paramètres");
    },

    // Supprimer le fichier
    removeFile() {
      this.file = null;
      this.audio = null;
      this.currentTime = 0;
      this.audioDuration = 0;
      this.isPlaying = false;
    },

    // Déclenche le dialogue de sélection de fichier
    triggerFileInput() {
      this.$refs.fileInput.click();  // Simule un clic sur le champ d'upload caché
    },

    // Envoie le fichier au backend et récupère les transcriptions
    async uploadFile() {
      const formData = new FormData();
      formData.append('file', this.file);

      // Requête POST vers ton backend pour obtenir les transcriptions
      try {
        const response = await fetch('http://127.0.0.1:8000/uploadfile/', {
          method: 'POST',
          body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let done = false;

        // Lire les données reçues en temps réel
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          buffer += decoder.decode(value, { stream: !done });

          // Diviser les segments JSON reçus par nouvelle ligne (chaque segment est séparé par '\n')
          let lines = buffer.split('\n');
          buffer = lines.pop();  // Garder la dernière ligne partielle pour la prochaine boucle

          // Traiter chaque segment JSON
          for (const line of lines) {
            if (line.trim()) {
              console.log("Segment reçu:", line);  // Ajout du log pour chaque segment reçu
              const segment = JSON.parse(line);  // Convertir le JSON en objet
              this.transcriptions.push(segment);  // Ajouter le segment à la liste
              console.log("Transcription mise à jour:", this.transcriptions);  // Log pour vérifier la mise à jour
              this.$forceUpdate();  // Forcer la mise à jour du DOM pour s'assurer que le frontend se rafraîchit
            }
          }
        }
      } catch (error) {
        console.error("Erreur lors de l'upload ou récupération des transcriptions", error);
      }
    }
  }
};
</script>

<style>
/* Style de l'interface d'upload */
.upload-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f4f4f4;
}

.upload-box {
  width: 400px;
  height: 300px;
  border: 2px dashed #aaa;
  border-radius: 10px;
  text-align: center;
  padding: 20px;
  background-color: #fff;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  cursor: pointer;
}

.upload-box:hover {
  border-color: #4CAF50;
}

.upload-box p {
  margin: 10px 0;
  font-size: 16px;
}

button {
  background-color: #4CAF50;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

button:hover {
  background-color: #45a049;
}

/* Style de la vue principale */
.file-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: #eee;
  border-bottom: 1px solid #ccc;
}

.file-header .controls {
  display: flex;
  gap: 10px;
}

.audio-player {
  display: flex;
  align-items: center;
  padding: 10px;
}

.audio-player input[type="range"] {
  margin: 0 10px;
  flex: 1;
}

.transcriptions {
  padding: 10px;
  border-top: 1px solid #ccc;
}

.transcription-segment {
  margin-bottom: 10px;
}

.chunk {
  cursor: pointer;
  color: blue;
  margin-right: 5px;
  transition: color 0.3s ease; /* Ajout d'une transition fluide pour l'effet de survol */
}

.chunk:hover {
  color: rgb(140, 0, 255); /* Change la couleur au survol */
  text-decoration: underline; /* Ajoute un soulignement en vagues au survol */
}

textarea {
  width: 100%;
  height: 200px;
  margin-top: 10px;
  font-size: 16px;
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #ccc;
  resize: none;
}
</style>

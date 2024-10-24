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

        <!-- Fenêtre modale pour les paramètres de transcription -->
        <div v-if="showSettings" class="settings-modal">
        <div class="settings-content">
          <h3>Paramètres de transcription</h3>
          <label for="model-select">Choisir le modèle de transcription :</label>
          <select id="model-select" v-model="selectedModel">
            <option value="openai/whisper-large-v3-turbo">Whisper Large v3 Turbo</option>
            <option value="openai/whisper-large-v3">Whisper Large v3</option>
            <option value="openai/whisper-tiny">Whisper Tiny</option>
            <option value="openai/whisper-small">Whisper Small</option>
            <option value="openai/whisper-medium">Whisper Medium</option>
            <option value="openai/whisper-base">Whisper Base</option>
            <option value="openai/whisper-large">Whisper Large</option>
          </select>
          <br /><br />
          <button @click="saveSettings">Enregistrer</button>
          <button @click="closeSettings">Annuler</button>
        </div>
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
      <div v-for="(segment, index) in transcriptions" :key="index" class="transcription-segment">
        <div class="message">
          <div class="message-header">
            <span 
              v-if="!segment.isEditing" 
              class="speaker" 
              @click="toggleSpeakerAudio(segment, index)" 
              @contextmenu.prevent="enableEditMode(segment, $event)">
              <span v-if="playingIndex === index">⏸️</span> 
              <span v-if="!segment.isEditing">
              {{ segment.speaker }}:
              </span>
            </span>
            <input
              v-else
              class="edit-input"
              type="text"
              v-model="segment.speaker"
              @blur="applySpeakerChange(segment)"
              @keyup.enter="applySpeakerChange(segment)"
            />
          </div>

      <!-- Texte complet du segment entouré dans une bulle -->
      <div class="message-body">
          <!-- Contenu du message (transcription) -->
            <div class="chunk-container">
              <span 
                v-for="(chunk, i) in segment.text.chunks" 
                :key="i" 
                class="chunk" 
                @click="playOrPauseChunk(segment.audio_url, chunk.timestamp[0], chunk.timestamp[1], i)">
                {{ chunk.text }}
              </span>
            </div>
      </div>
    </div>
  </div>

  
  <div>
    <!-- Barre de progression ASCII pour la transcription globale -->
    <pre>{{ updateAsciiProgressBar() }}</pre>
    <p>{{ transcriptionProgress.toFixed(2) }}% transcrit</p> <!-- Montre le pourcentage -->
  </div>

      <!-- Section pour afficher les statistiques de temps de parole -->
      <div class="stats-container">
        <h3>📊 Détection</h3>
        <p>{{ speechStats.totalSpeakers }} locuteurs identifiés</p>
        <p>Durée : {{ formatTime(speechStats.totalDuration) }}</p>

        <h3>👥 Répartition temps de parole</h3>
        <ul>
          <li v-for="(speakerStat, index) in speechStats.speakers" :key="index">
            {{ speakerStat.speaker }} : {{ speakerStat.percentage.toFixed(2) }}%
          </li>
        </ul>
      </div>

      <!-- Textarea pour l'ensemble de la transcription -->
      <div v-if="transcriptions.length > 0" class="transcription-full">
        <h3>Transcription complète</h3>
        <textarea v-model="fullTranscription" readonly></textarea>
        <button @click="copyToClipboard">Copier</button>
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
        <button @click.stop="triggerFileInput">Sélectionnez un fichier</button>
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
      transcribedTime: 0,  // Temps total déjà transcrit en secondes
      transcriptionProgress: 0,  // Progression globale en pourcentage
      playingIndex: null,  // Index du speaker en train d'être lu
      showSettings: false, // Affiche ou non les paramètres
      selectedModel: "openai/whisper-large-v3-turbo", // Modèle par défaut
      file: null,  // Stocke le fichier sélectionné ou déposé
      audio: null,  // Instance de l'objet Audio
      isPlaying: false,  // Indique si l'audio est en cours de lecture
      currentTime: 0,  // Temps actuel de la lecture
      audioDuration: 0,  // Durée totale de l'audio
      transcriptions: [],  // Ce tableau sera rempli par des transcriptions réelles du backend
      speechStats: {
        totalSpeakers: 0, // Nombre de locuteurs par défaut
        totalDuration: 0,  // Durée totale par défaut
        speakers: []       // Tableau vide pour la répartition des temps de parole
      },
        oldSpeakerName: '',  // Stocker l'ancien nom du speaker avant l'édition
      currentAudio: null, // Pour garder une référence à l'audio en cours
      currentChunkIndex: null, // Pour garder une trace du chunk en cours de lecture
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
        .join('\n');  // Ajouter une séparation entre chaque locuteur
    }
  },

  methods: {
    loadAudioMetadata(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.onloadedmetadata = () => {
      this.audioDuration = audio.duration;  // Récupérer la durée totale de l'audio
    };
    },

    // Méthode à appeler quand la transcription avance
    updateTranscriptionProgress(transcribedSeconds) {
      this.transcribedTime = transcribedSeconds;
      this.transcriptionProgress = (this.transcribedTime / this.audioDuration) * 100;
    },

    // Méthode pour générer la barre de progression en ASCII art
    updateAsciiProgressBar() {
      const barLength = 20;  // Longueur de la barre
      const filledLength = Math.round((this.transcriptionProgress / 100) * barLength);  // Portion remplie
      const emptyLength = barLength - filledLength;  // Portion vide

      const filledBar = '█'.repeat(filledLength);  // Blocs remplis
      const emptyBar = '-'.repeat(emptyLength);  // Blocs vides
      const progressBar = `[${filledBar}${emptyBar}]`;  // Barre finale

      return progressBar;
    },

    toggleSpeakerAudio(segment, index) {
      // Si un autre passage est en lecture, l'arrêter
      if (this.audio && this.playingIndex !== index) {
        this.audio.pause();  // Arrêter l'audio en cours
        this.audio = null;
        this.playingIndex = null;
      }

      // Si le passage est déjà en cours de lecture, l'arrêter
      if (this.playingIndex === index) {
        this.audio.pause();
        this.playingIndex = null;
      } else {
        // Sinon, démarrer la lecture du segment audio
        this.audio = new Audio(segment.audio_url);
        this.audio.play();
        this.playingIndex = index;

        // Gérer la fin de la lecture pour remettre l'icône ▶️
        this.audio.onended = () => {
          this.playingIndex = null;  // Remettre l'icône à ▶️ quand l'audio est terminé
        };
      }
    },

    // Ouvrir les paramètres (à personnaliser)
    openSettings() {
      this.showSettings = true; // Ouvrir la fenêtre modale des paramètres
    },

    closeSettings() {
      this.showSettings = false; // Fermer la fenêtre modale
    },
    saveSettings() {
      this.showSettings = false; // Fermer les paramètres une fois enregistrés
      alert(`Modèle de transcription sélectionné : ${this.selectedModel}`);
      // Logique supplémentaire pour enregistrer les paramètres si nécessaire
    },


    // Formater le temps en minutes et secondes
    formatTime(seconds) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.floor(seconds % 60);
      return `${minutes}:${remainingSeconds < 10 ? "0" : ""}${remainingSeconds}`;
    },

    playOrPauseChunk(audioUrl, startTime, endTime, chunkIndex) {
      // Si un audio est déjà en cours de lecture et qu'il s'agit du même chunk, on le met en pause
      if (this.currentAudio && this.currentChunkIndex === chunkIndex) {
        this.currentAudio.pause();
        this.currentAudio = null; // Réinitialiser l'état
        this.currentChunkIndex = null;
      } else {
        // Si un autre chunk est en cours de lecture, on le met en pause avant d'en jouer un nouveau
        if (this.currentAudio) {
          this.currentAudio.pause();
        }

        // Créer un nouvel objet Audio pour jouer le nouveau chunk
        const audio = new Audio(audioUrl);
        audio.currentTime = startTime;
        audio.play();

        // Définir un timeout pour arrêter l'audio à la fin du chunk
        setTimeout(() => {
          audio.pause();
        }, (endTime - startTime) * 1000); // Convertir le temps en millisecondes

        // Stocker la référence de l'audio en cours et l'index du chunk
        this.currentAudio = audio;
        this.currentChunkIndex = chunkIndex;
      }
    },

    // Activer le mode d'édition pour un segment
    enableEditMode(segment) {
      this.oldSpeakerName = segment.speaker;  // Sauvegarder le nom original avant édition
      segment.isEditing = true;  // Activer le champ d'édition
    },

    // Appliquer le changement de nom du speaker à tous les segments
    applySpeakerChange(segment) {
      const newSpeaker = segment.speaker;  // Nouveau nom du speaker
      const oldSpeaker = this.oldSpeakerName;  // Récupérer l'ancien nom du speaker sauvegardé

      // Mettre à jour tous les segments avec le même ancien nom de speaker
      this.transcriptions.forEach(seg => {
        if (seg.speaker === oldSpeaker) {
          seg.speaker = newSpeaker;  // Mettre à jour le speaker
        }
      });
      segment.isEditing = false;  // Désactiver le mode édition

      this.calculateSpeechStats();  // Recalculer les statistiques après la modification
    },

    // Méthode pour calculer les temps de parole des speakers
    calculateSpeechStats() {
      const stats = {};
      let totalDuration = 0; // Variable pour la durée totale de l'audio

      // Calculer le temps de parole pour chaque locuteur et la durée totale
      this.transcriptions.forEach(segment => {
        const speaker = segment.speaker;
        const duration = segment.end_time - segment.start_time; // Durée du segment
        totalDuration += duration; // Ajouter à la durée totale

        if (!stats[speaker]) {
          stats[speaker] = 0;  // Initialiser à zéro si ce speaker n'a pas encore été ajouté
        }
        stats[speaker] += duration;  // Ajouter la durée du segment au speaker
      });

      // Calculer les pourcentages pour chaque locuteur
      const percentageStats = Object.entries(stats).map(([speaker, time]) => {
        return {
          speaker: speaker,
          percentage: (time / totalDuration) * 100  // Calculer le pourcentage
        };
      });

      // Trier les locuteurs par ordre décroissant de temps de parole
      percentageStats.sort((a, b) => b.percentage - a.percentage);

      // Stocker les statistiques mises à jour
      this.speechStats = {
        totalDuration: totalDuration,  // Durée totale de l'audio
        speakers: percentageStats,  // Répartition des locuteurs et pourcentages
        totalSpeakers: percentageStats.length  // Nombre de locuteurs identifiés
      };
    },

    // Méthode pour jouer l'audio d'un segment complet
    playAudio(audioUrl) {
      const audio = new Audio(audioUrl);  // Créer une instance d'Audio avec l'URL du segment
      audio.play();  // Jouer l'audio
    },

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
        console.log("Fichier sélectionné :", this.file);
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
      this.$refs.fileInput.click();  // Simule un clic sur l'input file caché
    },

    // Envoie le fichier au backend et récupère les transcriptions
    async uploadFile() {
      const formData = new FormData();
      formData.append('file', this.file);

      // Requête POST vers ton backend pour obtenir les transcriptions
      try {
        const response = await fetch('http://localhost:8000/uploadfile/', {
          method: 'POST',
          body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let done = false;

        console.log("Début du streaming...");

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
              // Créer un nouveau tableau à chaque ajout
              this.transcriptions.push(segment);  // Ajouter le segment au tableau des transcriptions
              this.$nextTick(() => {
                console.log("DOM mis à jour avec le nouveau segment");
                console.log("Transcription mise à jour:", this.transcriptions);  // Log pour vérifier la mise à jour
              });  // S'assurer que Vue met à jour le DOM après chaque ajout
            }
          }
        }
        console.log("Streaming terminé.");
        this.calculateSpeechStats();  // Calculer les statistiques après réception des transcriptions

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

.message-body {
  position: relative;
  background-color: #e1ffc7;
  padding: 3px 10px 10px 10px; /* 3px en haut, 10px sur les côtés et en bas */
  border-radius: 8px;
  display: inline-block;
  max-width: 90%;
  margin-bottom: 10px;
  font-family: 'Roboto', sans-serif;
}

/* Ajout d'une petite flèche à gauche de la bulle */
.message-body::before {
  content: "";
  position: absolute;
  top: 10px; /* Ajuste pour positionner la flèche verticalement */
  left: -10px; /* Ajuste pour la position horizontale */
  border-width: 10px;
  border-style: solid;
  border-color: transparent #e1ffc7 transparent transparent; /* Flèche triangulaire vers la gauche */
}

.message-text {
  font-family: 'Roboto', sans-serif;
  font-size: 14px;
  margin: 0;
}

.chunk-container {
  display: flex;
  flex-wrap: wrap;
  margin-top: 10px;
}

.chunk {
  background-color: #e1ffc7;
  border-radius: 2px;
  cursor: pointer;
  font-size: 14px;
  line-height: 1.4;
}

.chunk:hover {
  background-color: yellow; /* Ajouter un surlignage doux lors du hover */
}

.chunk:active {
  background-color: #a5d096;  /* Changement de couleur au clic */
}

/* Style pour rendre le texte du speaker cliquable */
.speaker {
  cursor: pointer;
  font-weight: bold;  /* Mettre en gras par défaut */
  position: relative;
  background-color: #e7ffc7;
  border-radius: 8px;
  display: inline-block;
  max-width: 90%;
}

.speaker:hover {
  background-color: rgb(0, 255, 76); /* Surlignage à la manière d'un stabilo lorsqu'actif */
}

/* Ajouter l'emoji ▶️ lors du survol */
.speaker:hover::before {
  content: '▶️ ';
  font-size: 16px;
  color: inherit; /* Optionnel, pour garder la même couleur que le speaker */
  position: relative;
  left: 5px; /* Ajuste la distance entre le texte et l'emoji */
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

/* Style pour le champ d'édition */
.edit-input {
  font-size: 16px;
  padding: 2px;
  margin-left: 5px;
}
</style>

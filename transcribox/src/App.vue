<template>
  <div :class="{ dark: isDarkMode }">
    <div id="app" class="page-container">
      <!-- Vue principale affichée après l'upload du fichier -->
      <div v-if="file">
        <!-- Section fichier avec le même style que Statistiques -->
        <div class="file-container">
          <div class="file-header">📁 Fichier
            <div class="settings-group">
              <button @click="openSettings" class="settings-button">⚙️</button>
              <button @click="toggleDarkMode" class="settings-button">{{ isDarkMode ? "🌞" : "🌙" }}</button>
            </div>
          </div>
          <div class="file-body">
            <span>{{ file.name }}</span>
            <div class="controls">
              <button @click="removeFile">❌</button>
            </div>
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

        <!-- Section audio-player avec style similaire à stats-container -->
        <div class="audio-player-container">
          <div class="audio-player-header">🎵 Lecture Audio</div>
          <div class="audio-player-body">
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

          <!-- Affichage du thumbnail si disponible -->
          <div v-if="thumbnail" class="thumbnail-preview">
            <h3>Aperçu de la vidéo :</h3>
            <img :src="thumbnail" alt="Thumbnail de la vidéo" />
          </div>

          <!-- Élément vidéo caché pour capturer le thumbnail -->
          <video ref="video" style="display: none;" @loadeddata="captureThumbnail"></video>
        </div>

        <!-- Section de la barre de progression ASCII pour la transcription globale -->
        <div class="progress-bar-container">
          <div class="progress-bar-header">📈 Progression de la Transcription</div>
          <div>
            <div class="loading-message">{{ loadingMessage }}</div>
            <pre>{{ progressBarExtractionAudio }}</pre>
          </div>
          <div v-if="progressMessage">
            <span v-if="progressData.status === 'diarization_processing'" class="pulsating-emoji">🗣️</span>
            {{ progressMessage }}
            <span v-if="progressData.status === 'diarization_processing'" class="pulsating-emoji">👂</span>
          </div>
          <div class="progress-bar-body">
            <!-- Barre de progression ASCII pour la transcription globale -->
            <pre>{{ updateAsciiProgressBar() }}</pre>
            <p>{{ transcriptionProgress.toFixed(2) }}% de l'audio transcrit</p> <!-- Montre le pourcentage -->
          </div>
        </div>

        <!-- Section pour afficher les statistiques de temps de parole avec style ASCII -->
        <div class="stats-container">
          <div class="stats-header">📊 Statistiques</div>
          <div class="stats-body">
            <p>{{ speechStats.totalSpeakers }} locuteurs identifiés</p>
            <p>Durée : {{ formatTime(speechStats.totalDuration) }}</p>

            <div class="stats-subheader">👥 Répartition temps de parole</div>
            <ul>
              <li v-for="(speakerStat, index) in speechStats.speakers" :key="index" class="speaker-stat">
                <span class="speaker-label">{{ speakerStat.speaker }} : {{ speakerStat.percentage.toFixed(2) }}% du
                  temps total</span>
                <div class="bar-container">
                  <div class="bar" :style="{ width: speakerStat.percentage.toFixed(2) + '%' }"></div>
                </div>
              </li>
            </ul>
          </div>
        </div>

        <!-- Liste des locuteurs et des segments de transcription avec couleur unique par locuteur -->
        <div class="conversation-container" :class="{ dark: isDarkMode, disabled: !isTranscriptionComplete }">
          <div class="conversation-header">💬 Conversation
            <span v-if="isTranscriptionComplete" class="info-icon" title="Clic gauche pour lire, clic droit pour renommer le locuteur">ℹ️</span><span v-if="!isTranscriptionComplete" class="dots">...</span>
            </div>
          <div class="conversation-body">
            <span v-if="isTranscriptionComplete">
              <p class="instruction">Astuce : Utilisez un LLM sécurisé pour faire le compte rendu de la conversation </p>
            </span>
            <div v-for="(segment, index) in transcriptions" :key="index" class="message"
              :style="{ backgroundColor: getSpeakerColor(segment.speaker) }">
              <div class="message-header">
                <span v-if="!segment.isEditing" class="speaker"
                  @click="isTranscriptionComplete ? toggleSpeakerAudio(segment, index) : null"
                  @contextmenu.prevent="isTranscriptionComplete ? enableEditMode(segment) : null"
                  @touchstart.prevent="handleTouchStart($event, segment)" @touchend.prevent="handleTouchEnd($event)">

                  <span v-if="playingIndex === index">⏸️</span>
                  {{ segment.speaker }}:
                </span>
                <input v-else class="edit-input" type="text" v-model="segment.speaker"
                  :disabled="!isTranscriptionComplete" @blur="applySpeakerChange(segment)"
                  @keyup.enter="applySpeakerChange(segment)" />
              </div>

              <!-- Texte complet du segment entouré dans une bulle -->
              <div class="message-body">
                <div class="chunk-container">
                  <span v-for="(chunk, i) in segment.text.chunks" :key="i" class="chunk"
                    @click="isTranscriptionComplete ? playOrPauseChunk(segment.audio_url, chunk.timestamp[0], chunk.timestamp[1], i) : null">
                    {{ chunk.text }}<span v-if="i < segment.text.chunks.length - 1"> </span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Textarea pour l'ensemble de la transcription avec style encadré -->
        <div v-if="transcriptions.length > 0" class="transcription-full-container">
          <div class="transcription-header">📝 {{ isTranscriptionComplete ? "Transcription complète" : "Transcription en cours ..." }}
          </div>
          <button @click="copyToClipboard" class="copy-button">📋 Copier</button>
          <textarea v-model="fullTranscription" class="transcription-textarea" readonly
            oninput="this.style.height = ''; this.style.height = this.scrollHeight + 'px'"></textarea>
        </div>

      </div>

      <!-- Interface d'upload si aucun fichier n'est sélectionné -->
      <div v-else class="upload-container">
        <!-- Titre principal et sous-titre pour clarifier la fonction du service -->
        <div class="stats-container">
          <div class="stats-header">🎙️ Convertissez vos fichiers audio et vidéo en texte, avec identification des intervenants</div>
          <div class="stats-body">
            <p>Déposez un fichier audio ou vidéo, et notre IA extrait automatiquement la bande son, sépare les voix et transforme chaque parole en texte associé à son locuteur.</p>
          </div>
        </div>
        <div class="upload-box" @dragover.prevent @drop.prevent="handleDrop" @click="triggerFileInput">
          <p>Déposez votre fichier audio 🎙️ ou vidéo 🎬 ici</p>
          <button @click.stop="triggerFileInput">📁 Sélectionnez un fichier</button>


        <!-- Bouton d'enregistrement rond -->
        <div class="record-button-wrapper">
          <button 
            @click.stop="toggleRecording" 
            class="record-button"
            :class="{ 'record-button--recording': isRecording }"
            :title="isRecording ? 'Arrêter l\'enregistrement' : 'Commencer l\'enregistrement'"
          >
            <span class="record-button__inner">🎙️</span>
          </button>
          
          <!-- Label sous le bouton -->
          <span class="record-button__label">
            {{ isRecording ? 'Stop' : 'Enregistrer' }}
          </span>
        </div>

        

        </div>

        <input type="file" ref="fileInput" @change="onFileChange" accept="audio/*, video/*, .m4a"
          style="display: none" />

      <!-- Affichage du temps d'enregistrement -->
      <div v-if="isRecording" class="recording-timer">
        Enregistrement en cours: {{ formatTime(recordingTime) }}
      </div>

        <!-- Section "Comment ça marche ?" pour guider l'utilisateur -->
        <div class="stats-container">
          <div class="stats-header">🚀 Comment ça marche ?</div>

          <ol>
            <li><strong>Ajoutez un fichier:</strong> 📂glissez-déposez un fichier audio ou vidéo dans l’espace ci-dessus.</li>
            <li><strong>Traitement automatique:</strong> notre technologie d'IA extrait la bande son 📞, distingue chaque voix 👥 et crée une transcription complète, organisée par intervenant.</li>
            <li><strong>Copiez la transcription:</strong> obtenez un document textuel clair et structuré, prêt à être copié 📋 et utilisé où vous le souhaitez.</li>
          </ol>

        </div>

        <!-- Exemple de sortie pour montrer la séparation des voix -->
        <div class="stats-container">
          <div class="stats-header">✨ Exemple de sortie</div>
          <p><strong>Alice</strong> : Bonjour, comment allez-vous ?</p>
          <p><strong>Bob</strong> : Très bien, merci. Et vous ?</p>
          <p><strong>Clara</strong> : Impeccable !</p>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isRecording: false, // État de l'enregistrement
      mediaRecorder: null, // Instance du MediaRecorder
      audioChunks: [],
      recordingTime: 0,
      timerInterval: null,
      stream: null,

      thumbnail: null, // URL de l'image thumbnail
      extraction_audio_status: "🔄 Extraction audio en cours...",
      touchTimer: null,
      touchStartTime: null,
      loadingMessage: "🔄 Extraction audio en cours...",
      progressBarExtractionAudio: "[░░░░░░░░░░]",
      progress: 0,
      intervalId: null,
      progressMessage: '',  // Nouveau message de progression
      diarization: null,  // Stockage des données de diarisation complètes
      speakerColors: {}, // Associera chaque locuteur à une couleur unique
      isDarkMode: false, // Contrôle du mode sombre
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
    isTranscriptionComplete() {
      return this.transcriptionProgress === 100;
    },
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
    async toggleRecording() {
      if (!this.isRecording) {
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true
            }
          });

          this.mediaRecorder = new MediaRecorder(this.stream);
          
          this.mediaRecorder.ondataavailable = (event) => {
            this.audioChunks.push(event.data);
          };
          
          this.mediaRecorder.onstop = async () => {
            // Utiliser le temps d'enregistrement comme durée
            this.audioDuration = this.recordingTime;
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            const audioFile = new File([audioBlob], 'recording.wav', { 
              type: 'audio/wav',
              lastModified: Date.now()
            });
            
            // Ajouter la durée aux métadonnées du fichier
            Object.defineProperty(audioFile, 'duration', {
              value: this.recordingTime,
              writable: false
            });

            this.handleRecordedAudio(audioFile);
          };
          
          // Réinitialiser les variables
          this.audioChunks = [];
          this.recordingTime = 0;
          this.startTime = Date.now();
          
          // Démarrer l'enregistrement
          this.mediaRecorder.start();
          this.isRecording = true;
          this.startTimer();
          
        } catch (err) {
          console.error('Erreur lors de l\'accès au microphone:', err);
          alert('Impossible d\'accéder au microphone. Veuillez vérifier les permissions.');
        }
      } else {
        // Arrêt de l'enregistrement
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
          this.mediaRecorder.stop();
        }
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
        }
        this.isRecording = false;
        this.stopTimer();
      }
    },    

    startTimer() {
      this.timerInterval = setInterval(() => {
        // Calculer le temps écoulé en secondes
        this.recordingTime = Math.round((Date.now() - this.startTime) / 1000);
      }, 1000);
    },

    stopTimer() {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
    },
   
    handleRecordedAudio(audioFile) {
      // Utiliser directement this.recordingTime comme durée
      console.log('Durée de l\'enregistrement:', this.formatTime(this.recordingTime));
      
      // Appeler votre méthode de traitement du fichier
      this.onFileChange({ target: { files: [audioFile] } });
    },

    beforeDestroy() {
    this.stopTimer();
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
  },


    // Capture un thumbnail de la vidéo sélectionnée
    generateThumbnail(file) {
      // Vérification de l'existence de l'élément vidéo
      const videoElement = this.$refs.video;
      if (!videoElement) {
        console.error("L'élément vidéo n'est pas disponible");
        return;
      }

      videoElement.src = URL.createObjectURL(file); // Charger la vidéo
      videoElement.load(); // Assurez-vous que la vidéo est chargée
    },

    // Capture le thumbnail quand les données de la vidéo sont prêtes
    captureThumbnail() {
      const videoElement = this.$refs.video;
      if (!videoElement) return;

      const canvas = document.createElement("canvas");
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      this.thumbnail = canvas.toDataURL("image/png");

      URL.revokeObjectURL(videoElement.src);
    },

    handleTouchStart(event, segment) {
      if (!this.isTranscriptionComplete) return;

      this.touchStartTime = new Date().getTime();
      this.touchTimer = setTimeout(() => {
        this.enableEditMode(segment); // On appelle directement avec le segment
      }, 600);
    },

    handleTouchEnd(event) {
      clearTimeout(this.touchTimer);

      const touchDuration = new Date().getTime() - this.touchStartTime;
      if (touchDuration >= 600) {
        event.preventDefault();
      }
    },

    startProgressLoop() {
      this.intervalId = setInterval(() => {
        this.loadingMessage = "🔄 Extraction audio en cours...";
        this.progress = (this.progress + 1) % 10; // Boucle de 0 à 9 pour la progression
        const filled = '█'.repeat(this.progress);
        const empty = '░'.repeat(10 - this.progress);
        this.progressBarExtractionAudio = `[${filled}${empty}]`;
      }, 300); // Vitesse de progression en millisecondes
    },
    stopProgressLoop() {
      clearInterval(this.intervalId);
      this.progressBarExtractionAudio = "[██████████]"; // Barre pleine pour indiquer la fin
      this.loadingMessage = "Extraction audio terminée!";
    },
    getSpeakerColor(speaker) {
      // Vérifie si une couleur est déjà générée pour ce locuteur
      if (!this.speakerColors[speaker]) {
        this.speakerColors[speaker] = this.generateBaseColor(speaker);
      }
      // Retourne la couleur ajustée pour le mode actif
      return this.adjustColorForMode(this.speakerColors[speaker]);
    },
    generateBaseColor() {
      // Génère une teinte unique pour chaque locuteur en utilisant HSL
      const hue = Math.floor(Math.random() * 360);
      return `hsl(${hue}, 70%, 50%)`; // Luminosité moyenne initiale
    },
    adjustColorForMode(color) {
      // Modifie la luminosité pour s'adapter au mode sombre ou clair
      const lightness = this.isDarkMode ? '30%' : '85%'; // Plus sombre en mode sombre
      return color.replace(/(\d+%)$/, lightness); // Ajuste la dernière valeur HSL
    },

    toggleDarkMode() {
      this.isDarkMode = !this.isDarkMode;
    },

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
      if (seconds === "...") {
        return "...";
      }
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.floor(seconds % 60);
      return `${minutes}:${remainingSeconds < 10 ? "0" : ""}${remainingSeconds}`;
    },

    //  formatTime(seconds) {
    //   if (!seconds || !isFinite(seconds)) return '00:00';
    //   const minutes = Math.floor(seconds / 60);
    //   const remainingSeconds = Math.floor(seconds % 60);
    //   return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    // },

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

      // Mettre à jour tous les segments dans diarization avec le même ancien nom de speaker
      if (this.diarization) {
        this.diarization.forEach(entry => {
          if (entry.speaker === oldSpeaker) {
            entry.speaker = newSpeaker;
          }
        });
      }

      segment.isEditing = false;  // Désactiver le mode édition
      this.calculateSpeechStats();  // Recalculer les statistiques après la modification
    },

    // Méthode pour calculer les temps de parole des locuteurs
    calculateSpeechStats() {
      const stats = {};
      let totalDuration = 0; // Variable pour la durée totale de l'audio

      // Utiliser les données de this.diarization pour calculer le temps de parole de chaque locuteur
      console.log("Calcul pour la diarization :", this.diarization);
      this.diarization.forEach(entry => {
        const speaker = entry.speaker;
        const duration = entry.end_time - entry.start_time; // Durée du segment
        totalDuration += duration; // Ajouter à la durée totale de l'audio

        if (!stats[speaker]) {
          stats[speaker] = 0;  // Initialiser le compteur pour chaque locuteur
        }
        stats[speaker] += duration;  // Ajouter la durée du segment au temps total du locuteur
        console.log("speaker :", speaker);
        console.log("duration :", duration);
      });

      // Calculer les pourcentages de temps de parole pour chaque locuteur
      const percentageStats = Object.entries(stats).map(([speaker, time]) => {
        return {
          speaker: speaker,
          percentage: (time / totalDuration) * 100  // Calcul du pourcentage de temps de parole
        };
      });

      // Trier les locuteurs par ordre décroissant de temps de parole
      percentageStats.sort((a, b) => b.percentage - a.percentage);

      // Mettre à jour les statistiques avec les nouvelles données
      this.speechStats = {
        totalDuration: isNaN(totalDuration) ? "..." : totalDuration,  // Durée totale de l'audio ou "..." si NaN
        speakers: percentageStats,            // Répartition des locuteurs et pourcentages
        totalSpeakers: percentageStats.length // Nombre de locuteurs identifiés
      };
    },

    // Méthode pour jouer l'audio d'un segment complet
    playAudio(audioUrl) {
      const audio = new Audio(audioUrl);  // Créer une instance d'Audio avec l'URL du segment
      audio.play();  // Jouer l'audio
    },

    // Méthode pour copier la transcription complète dans le presse-papiers
    copyToClipboard() {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(this.fullTranscription)
          .then(() => {
            console.log("Texte copié dans le presse-papier !");
          })
          .catch(err => {
            console.error("Erreur lors de la copie du texte :", err);
          });
      } else {
        // Alternative de copie avec un champ texte temporaire
        const textarea = document.createElement("textarea");
        textarea.value = this.fullTranscription;
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand("copy");
          console.log("Texte copié dans le presse-papier !");
        } catch (err) {
          console.error("Erreur lors de la copie du texte :", err);
        }
        document.body.removeChild(textarea);
      }
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

        // Vérifie si le fichier est une vidéo
        if (this.file.type.startsWith("video/")) {
          this.generateThumbnail(this.file); // Capture le thumbnail si c'est une vidéo
        } else {
          this.thumbnail = null; // Réinitialise le thumbnail s'il ne s'agit pas d'une vidéo
        }

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
      // Réinitialiser toutes les variables liées à la transcription
      this.transcriptions = [];
      this.fullTranscription = '';
      this.currentAudio = null;
      this.currentChunkIndex = null;
      this.speechStats = {};
      this.diarization = null;
      this.transcriptionProgress = 0;
      this.progressData = {}; // Stocker le statut de progression

      this.startProgressLoop(); // Démarre la boucle de progression


      const formData = new FormData();
      formData.append('file', this.file);

      try {
        //  const response = await fetch('http://localhost:8000/uploadfile/', {
        const response = await fetch(`${process.env.VUE_APP_API_URL}/uploadfile/`, {
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
              try {
                const data = JSON.parse(line);
                console.log("Data reçue: ", data);

                if (data.extraction_audio_status === "extraction_audio_done") {
                  this.stopProgressLoop(); // Arrête la boucle de progression lorsque l'extraction est terminée
                }

                // Gestion de l'état "processing" pour afficher le message
                if (data.status === 'diarization_processing') {
                  this.progressData.message = data.message;  // Affiche "👂 Séparation des voix en cours..."
                  this.progressMessage = data.message;  // Affiche "👂 Séparation des voix en cours..." dans la progression
                  this.progressData.status = data.status;

                } else if (data.status === 'diarization_done') {
                  this.progressData.message = data.message;  // Affiche "Séparation terminée."
                  // this.progressMessage = ''; // Réinitialise le message une fois terminé
                  this.progressMessage = data.message;
                  this.progressData.status = data.status;
                }

                // Si on reçoit la diarization complète
                else if (data.diarization) {
                  this.diarization = data.diarization;
                  this.totalDuration = this.diarization.reduce((acc, entry) => acc + (entry.end_time - entry.start_time), 0);
                  this.calculateSpeechStats();
                }

                // Si on reçoit un segment de transcription
                else if (data.speaker && data.text && data.text.chunks) {
                  const segment = data;
                  this.transcriptions.push(segment);

                  // Calcul de la progression
                  const processedDuration = this.transcriptions.reduce((acc, seg) => acc + (seg.end_time - seg.start_time), 0);
                  this.transcriptionProgress = (processedDuration / this.totalDuration) * 100;

                  this.$nextTick(() => {
                    console.log("DOM mis à jour avec le nouveau segment");
                  });
                }
              } catch (error) {
                console.error("Erreur de parsing JSON :", error);
              }
            }
          }
        }
        console.log("Streaming terminé.");
      } catch (error) {
        console.error("Erreur lors de l'upload ou récupération des transcriptions", error);
      }
    }
  }
};
</script>

<style scoped>
/* Style de l'interface d'upload */


/* Fond de page et bordures en mode clair */
html,
body {
  background-color: #ffffff;
  color: #333;
  margin: 0;
  padding: 0;
}

/* Mode sombre global */
.dark html,
.dark body {
  background-color: #121212;
  /* Fond sombre pour toute la page */
  color: #e0e0e0;
  /* Texte clair pour le mode sombre */
}

/* Bordures pour tous les conteneurs principaux en mode sombre */
.dark .upload-box,
.dark .upload-container,
.dark .file-container,
.dark .stats-container,
.dark .transcription-full-container,
.dark .audio-player-container,
.dark .progress-bar-container,
.dark .conversation-container {
  border-color: #555;
  /* Bordure sombre pour s'adapter au mode dark */
  background-color: #000000;
  /* Fond sombre uniforme */
}

/* Bordures génériques pour tout autre élément */
.dark * {
  border-color: #555 !important;
}

.upload-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: auto;
  background-color: #f4f4f4;
}

.upload-box {
  width: auto;
  height: 60%;
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
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

button:hover {
  background-color: #45a049;
}

.settings-button {
  background: none;
  border: none;
  cursor: pointer;
}

.settings-group {
  display: flex;
  gap: 0.5em;
  /* Espace entre les boutons, ajustez selon votre préférence */
}

/* Style de la vue principale */
.file-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: #eee;
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 10px;
  border-radius: 5px;
  /* Correction de l'erreur '5x' */
}


.file-header .controls {
  display: flex;
  gap: 10px;
}

.audio-player {
  align-items: center;
  padding: 10px;
  display: flex;
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

.message-text {
  font-family: 'Roboto', sans-serif;
  font-size: 14px;
  margin: 0;
}



.chunk {
  font-family: 'Roboto', sans-serif;
  border-radius: 2px;
  cursor: pointer;
  font-size: 14px;
  line-height: 1.4;
  position: relative; /* Nécessaire pour positionner l'infobulle */
}

.chunk:hover {
  background-color: yellow;
  /* Ajouter un surlignage doux lors du hover */
}

.chunk::after {
  content: "🖱️ Clic pour écouter";
  position: absolute;
  top: -120%; /* Positionne l'infobulle juste au-dessus du chunk */
  left: 50%;
  transform: translateX(-50%);
  background-color: #333;
  color: #fff;
  padding: 5px;
  border-radius: 5px;
  white-space: nowrap;
  font-size: 0.9em;
  z-index: 10;
  opacity: 0;
  transition: opacity 0.2s ease-in-out;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}

.chunk:hover::after {
  opacity: 0.9; /* Affiche l'infobulle */
  transition-delay: 0.4s; /* Délai d'apparition de 400 ms */
}

/* Style pour rendre le texte du speaker cliquable */
.speaker {
  font-family: 'Roboto', sans-serif;
  cursor: pointer;
  font-weight: bold;
  /* Mettre en gras par défaut */
  position: relative;
  border-radius: 8px;
  display: inline-block;
  max-width: 90%;
}

.speaker:hover {
  background-color: rgb(0, 255, 76);
  /* Surlignage à la manière d'un stabilo lorsqu'actif */
}

/* Ajouter l'emoji ▶️ lors du survol */
.speaker:hover::before {
  content: '▶️ ';
  font-size: 16px;
  color: inherit;
  /* Optionnel, pour garder la même couleur que le speaker */
  position: relative;
  left: 5px;
  /* Ajuste la distance entre le texte et l'emoji */
}

.speaker::after {
  content: "💡 Clic droit pour renommer, clic gauche pour écouter";
  position: absolute;
  top: 50%;
  left: 105%;
  transform: translateY(-50%);
  background-color: #333;
  color: #fff;
  padding: 5px;
  border-radius: 5px;
  white-space: nowrap;
  font-size: 0.9em;
  z-index: 10;
  opacity: 0;
  transition: opacity 0.2s ease-in-out;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}

.speaker:hover::after {
  opacity: 0.9;
  transition-delay: 0.4s; /* Délai d'apparition de 400 ms */
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


.dark-mode-toggle {
  margin-top: 20px;
}

/* Mode sombre */
.dark {
  background-color: #121212;
  color: #ffffff;
}

.dark .audio-player,
.dark .message-body .dark .controls,
.dark .transcriptions,
.dark .copy-button,
.dark .statistics {
  color: #ffffff;
  border-color: #444;
}

.dark input[type="range"] {
  background-color: #444;
  color: #ffffff;
}

/* Applique un fond sombre pour .dark */
.dark .loading-message,
.dark .progress-bar-container,
.dark .audio-player-container,
.dark .conversation-container {
  background-color: #000000;
  border-color: #555;
  color: #000000;
}

/* Style pour les headers en mode sombre */
.dark .loading-message,
.dark .progress-bar-header,
.dark .audio-player-header,
.dark .conversation-header {
  color: #e0e0e0;
  border-bottom-color: #000000;
}

/* Ajuste les couleurs des sections internes */
.dark .loading-message pre,
.dark .progress-bar-body pre,
.dark .audio-player-body,
.dark .conversation-body {
  background-color: #000000;
  color: #e0e0e0;
}

/* Surlignage des chunks en mode sombre */
.dark .chunk-container .chunk:hover {
  background-color: #555;
}

/* Couleur de texte adaptative en fonction du mode */
.dark .loading-message pre,
.dark .conversation-container,
.dark .stats-container,
.dark .transcription-full-container,
.dark .audio-player-container,
.dark .progress-bar-container {
  color: #e0e0e0;
  /* Texte en blanc pour le mode sombre */
}

/* Mode sombre pour .file-container et ses parties */
.dark .file-container {
  border: 1px solid #555;
  background-color: #1e1e1e;
  color: #e0e0e0;
}

.dark .file-header,
.dark .file-body {
  background-color: #1e1e1e;
  color: #e0e0e0;
}

.dark .file-header {
  border-bottom: 1px solid #555;
}

/* Boutons de contrôle en mode sombre */
.controls button {
  color: inherit;
}

.controls button:hover {
  color: #aaa;
}

/* Couleur de texte par défaut pour le mode clair */
.conversation-container,
.stats-container,
.transcription-full-container,
.audio-player-container,
.progress-bar-container {
  color: #333;
  /* Texte en noir pour le mode clair */
}

.upload-container,
.stats-container {
  border: 1px solid #333;
  border-radius: 10px;
  width: 80%;
  /* Définit la largeur à 80% de la page */
  max-width: 800px;
  /* Optionnel : limite la largeur maximale pour les grands écrans */
  margin: 20px auto;
  /* Centre le cadre horizontalement */
  padding: 10px;
  font-family: monospace;
}

.how-it-works-header,
.service-description-header,
.stats-header {
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 10px;
}

.how-it-works-body,
.service-description-body,
.stats-body {
  padding: 5px 0;
}

.service-description-subheader,
.stats-subheader {
  font-weight: bold;
  margin-top: 10px;
  border-top: 1px solid #333;
  padding-top: 5px;
}

ul {
  list-style-type: none;
  padding-left: 0;
}

li {
  margin: 5px 0;
}


/* Conteneur principal */
.transcription-full-container {
  border-radius: 10px;
  border: 1px solid #333;
  width: 80%;
  /* Définit la largeur à 80% de la page */
  max-width: 800px;
  /* Optionnel : limite la largeur maximale pour les grands écrans */
  margin: 20px auto;
  /* Centre le cadre horizontalement */
  padding: 10px;
  font-family: monospace;
}

/* Titre de la transcription */
.transcription-header {
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 10px;
}

/* Zone de texte pour la transcription */
.transcription-textarea {
  width: 100%;
  background-color: #1e1e1e;
  color: #ffffff;
  border: 1px solid #3e3e3e;
  border-radius: 5px;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 1em;
  resize: none;
  overflow-y: auto;
  /* Autorise la barre de défilement verticale */
  box-sizing: border-box;
  transition: box-shadow 0.3s ease;
}

/* Ombre au focus */
.transcription-textarea:focus {
  outline: none;
  box-shadow: 0px 0px 5px 2px rgba(255, 255, 255, 0.2);
}

/* Style de la barre de défilement sobre */
.transcription-textarea::-webkit-scrollbar {
  width: 8px;
  /* Largeur de la barre de défilement */
}

.transcription-textarea::-webkit-scrollbar-track {
  background: #1e1e1e;
  /* Fond de la zone de défilement (même que l'arrière-plan) */
}

.transcription-textarea::-webkit-scrollbar-thumb {
  background-color: #3e3e3e;
  /* Couleur de la poignée de défilement */
  border-radius: 5px;
  /* Arrondi pour un look plus moderne */
}

.transcription-textarea::-webkit-scrollbar-thumb:hover {
  background-color: #555555;
  /* Légèrement plus clair au survol */
}

/* Bouton Copier */
.copy-button {
  background-color: #007bff;
  /* Couleur bleue pour le bouton */
  color: white;
  /* Texte en blanc */
  border: none;
  /* Pas de bordure */
  border-radius: 5px;
  /* Angles arrondis */
  padding: 10px 20px;
  /* Espacement interne */
  font-size: 1.1em;
  /* Taille un peu plus grande */
  margin-top: 15px;
  /* Espacement au-dessus du bouton */
  cursor: pointer;
  /* Curseur pointeur pour indiquer un bouton cliquable */
  display: block;
  /* Bouton affiché comme bloc */
  width: 100%;
  /* Le bouton prend toute la largeur */
  text-align: center;
  /* Texte centré dans le bouton */
}

/* Effet de survol du bouton */
.copy-button:hover {
  background-color: #0056b3;
  /* Couleur plus foncée au survol */
}

/* Styles pour le mode clair */
.light-mode .transcription-full-container {
  background-color: #f5f5f5;
  border: 1px solid #ddd;
}

.light-mode .transcription-header {
  color: #333;
}

.light-mode .transcription-textarea {
  background-color: #ffffff;
  color: #333;
  border: 1px solid #ddd;
}

.light-mode .copy-button {
  background-color: #007bff;
  color: white;
}

.light-mode .copy-button:hover {
  background-color: #0056b3;
}

.message-container {
  border: 1px solid #333;
  width: 80%;
  /* Largeur de 80% de la page pour un alignement harmonieux */
  max-width: 800px;
  /* Limite maximale de largeur */
  margin: 15px auto;
  /* Espacement vertical entre chaque message */
  padding: 10px;
  font-family: monospace;
  background-color: #f9f9f9;
  /* Fond légèrement coloré pour l'effet bulle */
  border-radius: 8px;
  /* Coins arrondis pour un effet de bulle */
}

.message-body {
  padding: 5px 0;
}

.chunk-container .chunk {
  display: inline-block;
  cursor: pointer;
  transition: background-color 0.3s;
}

.chunk-container .chunk:hover {
  background-color: #e0e0e0;
  /* Surlignage au survol */
  border-radius: 4px;
}


.edit-input {
  width: 100%;
  padding: 4px;
  font-size: 14px;
  font-family: monospace;
  border: 1px solid #ddd;
  border-radius: 4px;
}

/* Couleurs pour chaque locuteur */
.speaker-0 .message-body {
  background-color: #ffe0e0;
  /* Rouge clair */
}

.speaker-1 .message-body {
  background-color: #09886c;
  /* Vert clair */
}

.speaker-2 .message-body {
  background-color: #08089c;
  /* Bleu clair */
}

.speaker-3 .message-body {
  background-color: #fff0b3;
  /* Jaune clair */
}

.speaker-4 .message-body {
  background-color: #b4064c;
  /* Rose clair */
}

.message-body {
  padding: 5px;
  border-radius: 5px;
  margin-bottom: 10px;
}

.conversation-container {
  border: 1px solid #333;
  width: 80%;
  /* Largeur de 80% de la page */
  max-width: 800px;
  /* Limite maximale de largeur */
  margin: 20px auto;
  padding: 10px;
  font-family: monospace;
  background-color: #f9f9f9;
  /* Fond clair */
  border-radius: 8px;
}

.conversation-header {
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 15px;
}

.conversation-body {
  padding: 5px 0;
}

.message {
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 10px;
}

.message-header {
  font-weight: bold;
  color: #333;
  /* Conserve un texte lisible */
  border-radius: 4px 4px 0 0;
  padding: 5px;
}

.message-body {
  padding: 5px;
  border-radius: 0 0 4px 4px;
}

.chunk-container .chunk {
  display: inline-block;
  cursor: pointer;
}

.chunk-container .chunk:hover {
  background-color: #e0e0e0;
  /* Surlignage au survol */
  border-radius: 4px;
}

.edit-input {
  width: 100%;
  padding: 4px;
  font-size: 14px;
  font-family: monospace;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.audio-player-container {
  border: 1px solid #333;
  width: 80%;
  /* Largeur de 80% de la page */
  max-width: 800px;
  /* Limite maximale de largeur */
  margin: 20px auto;
  padding: 10px;
  font-family: monospace;
  background-color: #f9f9f9;
  /* Fond clair */
  border-radius: 8px;
}

.audio-player-header {
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 10px;
}

.audio-player-body {
  display: flex;
  align-items: center;
  gap: 10px;
}

.audio-player-body button {
  font-size: 20px;
  /* Taille du bouton de lecture */
  background: none;
  border: none;
  cursor: pointer;
}

.audio-player-body input[type="range"] {
  flex-grow: 1;
  height: 4px;
  background: #ddd;
  border-radius: 5px;
  cursor: pointer;
}

.audio-player-body span {
  font-size: 14px;
  color: #333;
}

.progress-bar-container {
  border: 1px solid #333;
  width: 80%;
  /* Largeur de 80% de la page */
  max-width: 800px;
  /* Limite maximale de largeur */
  margin: 20px auto;
  padding: 10px;
  font-family: monospace;
  background-color: #f9f9f9;
  /* Fond clair */
  border-radius: 8px;
}

.progress-bar-header {
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 5px;
  margin-bottom: 10px;
}

.progress-bar-body {
  padding: 5px 0;
}

.progress-bar-body pre {
  font-size: 16px;
  color: #333;
  /* Fond légèrement plus sombre pour contraster */
  padding: 5px;
  border-radius: 4px;
}

.progress-bar-body p {
  font-size: 14px;
  color: #333;
  margin-top: 10px;
}

.file-container {
  border: 1px solid #333;
  width: 80%;
  /* Largeur de 80% de la page */
  max-width: 800px;
  /* Limite maximale de largeur */
  margin: 20px auto;
  padding: 10px;
  font-family: monospace;
  border-radius: 8px;
}

.file-body {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.controls button {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 18px;
  padding: 5px;
}

.progress-bar-container {
  margin-top: 20px;
  text-align: center;
}

.progress-bar {
  width: 100%;
  height: 20px;
  background-color: #e0e0e0;
  border-radius: 10px;
  overflow: hidden;
}

.progress {
  height: 100%;
  background-color: #4caf50;
  /* Couleur de la barre de progression */
  transition: width 0.3s;
  /* Animation douce */
}


.highlight {
  background-color: #d1e7dd;
  /* Couleur de surbrillance légère */
  font-weight: bold;
  /* Texte en gras pendant la lecture */
  color: #333;
  /* Couleur du texte plus sombre */
}

/* Code CSS pour l'Animation de Battement */
@keyframes heartbeat {

  0%,
  100% {
    transform: scale(1);
  }

  50% {
    transform: scale(1.5);
  }
}

.pulsating-emoji {
  display: inline-block;
  animation: heartbeat 0.8s infinite;
}

.loading-message {
  font-family: monospace;
  color: #555;
  text-align: center;
  margin-bottom: 10px;
}

pre {
  font-family: monospace;
  text-align: center;
  color: #000000;
}

/* Styles globaux */
body {
  margin: 0;
  padding: 0;
  background-color: #f0f0f0;
  /* Couleur de fond claire par défaut */
  color: #333;
  /* Couleur du texte par défaut */
}

/* Conteneur principal pour centrer et encadrer le contenu */
.page-container {
  max-width: 800px;
  margin: 20px auto;
  padding: 20px;
  background-color: #ffffff;
  /* Fond clair par défaut */
  color: #333;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Mode sombre */
.dark .page-container {
  background-color: #1e1e1e;
  /* Fond sombre */
  color: #e0e0e0;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
  /* Ombre plus intense */
}


.stats-subheader {
  font-weight: bold;
  margin-bottom: 10px;
}

ul {
  list-style-type: none;
  padding: 0;
}

.speaker-stat {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.speaker-label {
  flex: 1;
  /* Prend tout l'espace disponible à gauche */
  margin-right: 10px;
}

.bar-container {
  width: 33%;
  /* Largeur fixe ou ajustable de la barre */
  background-color: #e0e0e0;
  border-radius: 5px;
  overflow: hidden;
  height: 20px;
  /* Hauteur de la barre */
}

.bar {
  height: 100%;
  /* Prend toute la hauteur du conteneur */
  background-color: #4CAF50;
  /* Couleur de la barre */
  border-radius: 5px 0 0 5px;
  transition: width 0.3s ease;
  /* Transition pour une animation douce */
}

.conversation-container.disabled {
  opacity: 0.9;
  pointer-events: none;
  /* Désactive toutes les interactions dans le conteneur */
}

.dark .loading-message,
.dark-loading-message {
  color: #c0c0c0;
  /* Texte clair pour le mode sombre */
}

.dark pre,
.dark-progress-bar {
  color: #c0c0c0;
  /* Couleur plus claire pour la barre de progression */
}

.dots {
  display: inline-block;
  margin-left: 5px;
  font-weight: bold;
  animation: blink 1.5s steps(5, end) infinite;
}
.dark .dots {
  color: #f0f0f0;
}

@keyframes blink {
  0%, 20% {
    color: transparent;
  }
  40% {
    color: rgb(133, 131, 131);
  }
  60% {
    color: transparent;
  }
  80%, 100% {
    color: black;
  }
}

.instruction {
  font-size: 0.85em;
  color: #666;
  margin-top: 5px;
  text-align: center;
}

.info-icon {
  font-size: 0.75em;
  color: #666;
  margin-left: 5px;
  cursor: help;
}

.record-button-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.record-button {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  padding: 0;
  border: none;
  background-color: #ffffff;
  cursor: pointer;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

.record-button:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.record-button__inner {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background-color: #ff4444;
  transition: all 0.3s ease;
}

.record-button--recording .record-button__inner {
  width: 20px;
  height: 20px;
  border-radius: 4px;
  animation: pulse 2s infinite;
}

.record-button__label {
  font-size: 0.875rem;
  color: #666;
  user-select: none;
}

.record-timer {
  margin-top: 1rem;
  text-align: center;
  color: #ff4444;
  font-weight: 500;
  font-size: 1.125rem;
}

@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(0.9);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

/* Si vous avez besoin de gérer la disposition des boutons */
.buttons-container {
  display: flex;
  gap: 2rem;
  justify-content: center;
  align-items: center;
  margin-top: 1rem;
}

/* Style alternatif avec bordure */
/* Décommentez ces styles si vous préférez une version avec bordure */

.record-button {
  border: 2px solid #ff4444;
}

.record-button--recording {
  border-color: #cc0000;
  animation: borderPulse 2s infinite;
}

@keyframes borderPulse {
  0% {
    border-color: #ff4444;
  }
  50% {
    border-color: #cc0000;
  }
  100% {
    border-color: #ff4444;
  }
}

.recording-timer {
  text-align: center;
  margin-top: 1rem;
  font-size: 1.2rem;
  color: #ff4444;
}
</style>
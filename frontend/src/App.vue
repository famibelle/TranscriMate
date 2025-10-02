<template>
  <div :class="{ dark: isDarkMode }">
    <div id="app" class="page-container">

      <div class="tabs-container">
        <div class="tabs-header">

          <button @click="activeTab = 'streaming'" :class="['tab-button', { active: activeTab === 'streaming' }]">
            <span class="tab-title">🔄 Mode Streaming</span>
            <span class="tab-subtitle">Upload + Affichage Progressif</span>
          </button>

          <button @click="activeTab = 'live'" :class="['tab-button', { active: activeTab === 'live' }]">
            <span class="tab-title">🎤 Mode Live</span>
            <span class="tab-subtitle">Microphone Temps Réel</span>
          </button>

          <button @click="activeTab = 'chatbot'" :class="['tab-button', { active: activeTab === 'chatbot' }]">
            <span class="tab-title">🤖 AKABot</span>
            <span class="tab-subtitle">IA Assistant</span>
          </button>

          <button @click="activeTab = 'simple'" :class="['tab-button', { active: activeTab === 'simple' }]">
            <span class="tab-title">� API Simple</span>
            <span class="tab-subtitle">Swagger/Développeurs</span>
          </button>

        </div>

        <div class="tab-content">

          <!-- MODE STREAMING : Upload fichier + Affichage progressif -->
          <div v-if="activeTab === 'streaming'">
            <div class="streaming-mode-container">
              <!-- Header du mode streaming -->
              <div class="stats-container" v-if="!file">
                <div class="stats-header">🔄 Mode Streaming - Transcription Progressive</div>
                <div class="stats-body">
                  <p>Upload de fichier avec affichage des segments en temps réel</p>
                  <div class="mode-features">
                    <span class="feature">📁 Upload fichier</span>
                    <span class="feature">🎯 Diarisation complète</span>
                    <span class="feature">📝 Affichage progressif</span>
                    <span class="feature">⚡ Server-Sent Events</span>
                  </div>
                </div>
              </div>
              
              <!-- Vue principale affichée après l'upload du fichier -->
              <div v-if="file">
              <!-- Section fichier -->
              <div class="file-container">
                <div class="file-header">📁 Fichier
                  <div class="settings-group">
                    <button @click="toggleDarkMode" class="settings-button">{{ isDarkMode ? "🌞" : "🌙" }}</button>
                  </div>
                </div>
                <div class="file-body">
                  <span>{{ file.name }}</span>
                  <div class="controls">
                    <button @click="removeFile">❌</button>
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
              <div class="progress-bar-container" v-if="!isTranscriptionComplete">
                <div class="progress-bar-header">📈 Progression de la {{ settings.task === "transcribe" ?
                  "Transcription" : "Traduction" }}
                </div>
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

              <!-- Liste des locuteurs et des segments de transcription avec couleur unique par locuteur -->
              <div class="conversation-container" :class="{ dark: isDarkMode, disabled: !isTranscriptionComplete }">
                <div class="conversation-header">💬 Conversation
                  <span v-if="isTranscriptionComplete" class="info-icon"
                    title="Clic gauche pour lire, clic droit pour renommer le locuteur">ℹ️</span><span
                    v-if="!isTranscriptionComplete" class="dots">...</span>
                </div>
                <div class="conversation-body">
                  <span v-if="isTranscriptionComplete">
                    <p class="instruction">Astuce : Editez le nom du SPEAKER (clique droit sur le nom du speaker) pour avoir un compte rendu plus précis </p>
                  </span>
                  <div v-for="(segment, index) in transcriptions" :key="index" class="message"
                    :style="{ backgroundColor: getSpeakerColor(segment.speaker) }">
                    <div class="message-header">
                      <span v-if="!segment.isEditing" class="speaker"
                        @click="isTranscriptionComplete ? toggleSpeakerAudio(segment, index) : null"
                        @contextmenu.prevent="isTranscriptionComplete ? enableEditMode(segment) : null"
                        @touchstart.prevent="handleTouchStart($event, segment)"
                        @touchend.prevent="handleTouchEnd($event)">

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
                        <span v-for="(chunk, i) in segment.text.chunks" :key="i" 
                          :class="['chunk', { 'audio-available': isTranscriptionComplete && segment.audio_url, 'no-audio': !segment.audio_url }]"
                          @click="isTranscriptionComplete && segment.audio_url ? playSegmentWithTimestamps(segment.audio_url, segment.segment_start || segment.start_time, segment.segment_end || segment.end_time, `${index}-${i}`) : null"
                          :title="segment.audio_url ? 'Cliquer pour écouter ce segment' : 'Audio non disponible en mode streaming'">
                          {{ chunk.text }}<span v-if="i < segment.text.chunks.length - 1"> </span>
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Textarea pour l'ensemble de la transcription avec style encadré -->
              <div v-if="transcriptions.length > 0" class="transcription-full-container">
                <div class="transcription-header">📝 {{ isTranscriptionComplete ? "Transcription complète" :
                  "Transcription en cours à " }} {{isTranscriptionComplete ? "" : transcriptionProgress.toFixed(2)}}
                  {{!isTranscriptionComplete ? "%" : ""}}

                </div>
                <button @click="copyToClipboard" class="copy-button">📋 Copier</button>
                <textarea v-model="fullTranscription" class="transcription-textarea" readonly
                  oninput="this.style.height = ''; this.style.height = this.scrollHeight + 'px'"></textarea>
              </div>

              <div class="stats-container" v-if="isTranscriptionComplete">
                <div class="stats-header">🤖 Chatbot</div>
                <div class="stats-body"></div>
                <div id="app">
                  <QuestionForm :fullTranscription="fullTranscription" :chat_model="settings.chat_model" />
                </div>
              </div>


              <!-- Section pour afficher les statistiques de temps de parole avec style ASCII -->
              <div class="stats-container" v-if="diarization !== null">
                <div class="stats-header">📊 Statistiques</div>
                <div class="stats-body">
                  <p>{{ speechStats.totalSpeakers }} locuteurs identifiés</p>
                  <p>Durée : {{ formatTime(speechStats.totalDuration) }}</p>

                  <div class="stats-subheader">👥 Répartition temps de parole</div>
                  <ul>
                    <li v-for="(speakerStat, index) in speechStats.speakers" :key="index" class="speaker-stat">
                      <span class="speaker-label">{{ speakerStat.speaker }} : {{ speakerStat.percentage.toFixed(2) }}%
                        du
                        temps total</span>
                      <div class="bar-container">
                        <div class="bar" :style="{ width: speakerStat.percentage.toFixed(2) + '%' }"></div>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>

              <!-- Paramètre -->
              <div class="stats-container">
                <div class="stats-header">⚙️ Paramètres du Chatbot</div>
                <div class="settings-group">
                </div>
                <!-- Fenêtre modale pour les paramètres de transcription -->
                <div class="settings-modal">
                  <div>
                    <div>
                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.chat_model === 'gpt-4'" @change="toggleModel">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.chat_model === 'gpt-4'}">OpenAI GPT</span>
                      </div>
                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.chat_model === 'chocolatine'" @change="toggleModel">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.chat_model === 'chocolatine'}">Chocolatine🍫🥖</span>
                      </div>
                    </div>
                  </div>
                  <div>
                  </div>
                  <div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Interface d'upload si aucun fichier n'est sélectionné -->
            <div v-else class="upload-container">
              <!-- Titre principal et sous-titre pour clarifier la fonction du service -->
              <div class="stats-container">
                <div class="stats-header">🎙️ Convertissez vos fichiers audio et vidéo en texte, avec identification des
                  intervenants</div>
                <div class="stats-body">
                  <p>Déposez un fichier audio ou vidéo, et notre IA extrait automatiquement la bande son, sépare les
                    voix et transforme chaque parole en texte associé à son locuteur.</p>
                </div>
              </div>
              <div
                class="upload-box"
                @dragover.prevent="!isRecording && $event"
                @drop.prevent="!isRecording && handleDrop($event)"
                @click="!isRecording && triggerFileInput()"
                :class="{ 'upload-box--disabled': isRecording }"
              >

                <p v-if="!isRecording">Déposez votre fichier audio 🎙️ ou vidéo 🎬 ici</p>
                <button v-if="!isRecording" @click.stop="triggerFileInput">📁 Sélectionnez un fichier</button>
                <p v-if="!isRecording">ou</p>
                <!-- Bouton d'enregistrement rond -->
                <div class="record-button-wrapper">
                  <!-- <pre v-html="asciiSpectrogram" v-if="isRecording"></pre> -->

                  <!-- <button @click.stop="toggleRecording" class="record-button"
                    :class="{ 'record-button--recording': isRecording }"
                    :title="isRecording ? 'Arrêter l\'enregistrement' : 'Commencer l\'enregistrement'">
                    <span class="record-button__inner">🎙️</span>
                  </button> -->

                  <!-- Label sous le bouton -->
                  <!-- <span class="recording-status">
                    {{ isRecording ? 'STOP ■' : 'REC ●' }}
                  </span> -->
                  <Dictaphone 
                    :asciiSpectrogram="asciiSpectrogram"
                    :is-recording="isRecording"
                    @click.stop="toggleRecording"
                  />

                </div>
              </div>

              <input type="file" ref="fileInput" @change="onFileChange" accept="audio/*, video/*, .m4a"
                style="display: none" />

              <!-- Affichage du temps d'enregistrement -->
              <div v-if="isRecording" class="recording-timer">
                Enregistrement en cours: {{ formatTime(recordingTime) }}
                <div>{{ transcriptionLive.text }}</div>
              </div>

              <!-- Section "Comment ça marche ?" pour guider l'utilisateur -->
              <div class="stats-container">
                <div class="stats-header">🚀 Comment ça marche ?</div>

                <ol>
                  <li><strong>Ajoutez un fichier:</strong> 📂glissez-déposez un fichier audio ou vidéo dans l’espace
                    ci-dessus.</li>
                  <li><strong>Traitement automatique:</strong> notre technologie d'IA extrait la bande son 📞, distingue
                    chaque voix 👥 et crée une transcription complète, organisée par intervenant.</li>
                  <li><strong>Copiez la transcription:</strong> obtenez un document textuel clair et structuré, prêt à
                    être copié 📋 et utilisé où vous le souhaitez.</li>
                  <li><strong>Chatbot:</strong>Demander à l'AI 🤖 d'en faire une synthèse</li>
                </ol>

              </div>

              <!-- Exemple de sortie pour montrer la séparation des voix -->
              <div class="stats-container">
                <div class="stats-header">✨ Exemple de sortie</div>
                <p><strong>Alice</strong> : Bonjour, comment allez-vous ?</p>
                <p><strong>Bob</strong> : Très bien, merci. Et vous ?</p>
                <p><strong>Clara</strong> : Impeccable !</p>
              </div>

              <!-- Paramètre -->
              <div class="stats-container">
                <div class="stats-header">⚙️ Paramètres généraux</div>
                <div class="settings-group">
                </div>
                <!-- Fenêtre modale pour les paramètres de transcription -->
                <div class="settings-modal">
                  <div>
                    <div>

                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.task === 'transcribe'" @change="toggleTask">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.task === 'transcribe' }">Transcrire</span>

                      </div>

                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.task === 'translate'" @change="toggleTask">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.task === 'translate' }">Traduire (🇬🇧)</span>
                      </div>
                    </div>
                  </div>
                  <div>
                  </div>
                  <div>
                  </div>
                </div>
              </div>
            </div>
            </div> <!-- Fermeture streaming-mode-container -->
          </div>

          <!-- MODE LIVE : Microphone temps réel -->
          <div v-if="activeTab === 'live'">
            <div class="live-mode-container">
              <!-- Header du mode live -->
              <div class="stats-container">
                <div class="stats-header">🎤 Mode Live - Transcription Temps Réel</div>
                <div class="stats-body">
                  <p>Transcription directe depuis votre microphone en temps réel</p>
                  <div class="mode-features">
                    <span class="feature">✅ Transcription instantanée</span>
                    <span class="feature">⚡ Faible latence</span>
                    <span class="feature">🔄 Flux continu</span>
                  </div>
                </div>
              </div>

              <!-- Interface microphone temps réel -->
              <div class="live-recording-container">
                <div class="record-button-wrapper">
                  <Dictaphone 
                    :isRecording="isRecording"
                    :recordingTime="recordingTime"
                    :transcriptionLive="transcriptionLive"
                    @toggleRecording="toggleRecording"
                  />
                </div>

                <!-- Affichage temps d'enregistrement -->
                <div v-if="isRecording" class="recording-timer">
                  Enregistrement en cours: {{ formatTime(recordingTime) }}
                  <div class="live-spectrogram">{{ asciiSpectrogram }}</div>
                </div>

                <!-- Transcription live accumulée -->
                <div class="transcription-live-container">
                  <div class="transcription-header">💬 Transcription Live</div>
                  <textarea 
                    ref="transcriptionArea" 
                    v-model="accumulatedTranscription" 
                    class="transcription-textarea live-textarea" 
                    readonly
                    placeholder="Commencez à parler pour voir la transcription apparaître...">
                  </textarea>
                </div>
              </div>

              <!-- Paramètres du mode live -->
              <div class="stats-container">
                <div class="stats-header">⚙️ Paramètres Mode Live</div>
                <div class="settings-modal">
                  <div class="setting-item">
                    <label>Langue:</label>
                    <select v-model="settings.lang">
                      <option value="auto">Auto-détection</option>
                      <option value="fr">Français</option>
                      <option value="en">Anglais</option>
                    </select>
                  </div>
                  <div class="setting-item">
                    <label>Mode:</label>
                    <select v-model="settings.task">
                      <option value="transcribe">Transcription</option>
                      <option value="translate">Traduction (vers anglais)</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- MODE CHATBOT : Assistant IA -->
          <div v-if="activeTab === 'chatbot'">
            <div class="chatbot-mode-container">
              <!-- Header du mode chatbot -->
              <div class="stats-container">
                <div class="stats-header">🤖 AKAbot - Assistant IA</div>
                <div class="stats-body">
                  <p>Posez vos questions sur les transcriptions ou sur AKABI</p>
                  <div class="mode-features">
                    <span class="feature">🧠 IA Chocolatine & GPT</span>
                    <span class="feature">📝 Analyse de transcription</span>
                    <span class="feature">💡 Conseils AKABI</span>
                  </div>
                </div>
              </div>

              <!-- Interface du chatbot -->
              <div class="stats-container">
                <div class="stats-header">💬 Assistant Conversationnel</div>
                <div class="stats-body">
                  <QuestionForm 
                    :defaultQuestion="'Que fait AKABI en IA?'" 
                    :fullTranscription="fullTranscription"
                    :chat_model="settings.chat_model" 
                  />
                </div>
              </div>


              <!-- Section "Comment ça marche ?" pour guider l'utilisateur -->
              <div class="stats-container">
                <div class="stats-header">🧩 Comment ça marche ?</div>
                <ol>
                  <li><strong>Chatbot: </strong>Demandez à AKAbot 🤖 comment AKABI peut vous aider dans vos projets d'IA
                  </li>
                  <li>
                    <strong>Posez une question: </strong>Demandez à AKABot de l'aide sur vos projets IA en lui posant
                    des questions spécifiques.
                    <em>Exemples de questions:</em>
                    <ul>
                      <li>"Quels sont les cas d'usage d'AKABI en IA ?"</li>
                      <li>"Comment AKABI peut m'aider avec des solutions de RAG ?"</li>
                    </ul>
                  </li>

                  <li>
                    <strong>Interaction guidée: </strong>Si vous ne savez pas par où commencer, essayez une question
                    générale, comme "Que propose AKABI dans le domaine de la prédiction ?".<br>
                    AKABot vous orientera vers les solutions IA les plus adaptées.
                  </li>

                  <li>
                    <strong>Recevez des réponses précises: </strong>AKABot est alimenté par les use cases d'AKABI, donc
                    chaque réponse est basée sur des applications concrètes et des projets réels.<br>
                    Vous obtiendrez des informations détaillées sur la manière dont AKABI aborde les problématiques
                    courantes en IA, que ce soit en traitement de données, en génération de langage, ou en
                    automatisation.
                  </li>

                  <li>
                    <strong>Demandez des conseils personnalisés: </strong>Besoin d’une solution sur mesure ? Posez des
                    questions spécifiques à votre secteur pour recevoir des recommandations d'AKABot sur les solutions
                    IA pertinentes pour vous.
                  </li>
                </ol>

              </div>
              <!-- Paramètre -->
              <div class="stats-container">
                <div class="stats-header">⚙️ Paramètres du Chatbot</div>
                <div class="settings-group">
                </div>
                <!-- Fenêtre modale pour les paramètres de transcription -->
                <div class="settings-modal">
                  <div>
                    <div>
                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.chat_model === 'gpt-4'" @change="toggleModel">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.chat_model === 'gpt-4'}">OpenAI GPT</span>
                      </div>
                      <div>
                        <label class="switch">
                          <input type="checkbox" :checked="settings.chat_model === 'chocolatine'" @change="toggleModel">
                          <span class="slider"></span>
                        </label> <span :class="{ bold: settings.chat_model === 'chocolatine'}">Chocolatine🍫🥖</span>
                      </div>
                    </div>
                  </div>
                  <div>
                  </div>
                  <div>
                  </div>
                </div>
              </div>
            </div>

          </div>

          <!-- FIN MODE CHATBOT -->

          <!-- MODE API SIMPLE : Interface directe pour développeurs -->
          <div v-if="activeTab === 'simple'">
            <h2>Mode API Simple</h2>
            <div class="tab-info">
              Envoyez vos fichiers directement à l'API et recevez la réponse complète avec les URLs audio intégrées.
              Parfait pour l'intégration dans d'autres applications.
            </div>

            <!-- Zone d'upload -->
            <div class="upload-area" 
                 @drop="handleDrop" 
                 @dragover.prevent 
                 @dragenter.prevent
                 :class="{ 'drag-over': isDragOver }">
              <div v-if="!file" class="upload-placeholder">
                <div class="upload-icon">📁</div>
                <p>Glissez-déposez votre fichier audio ici ou cliquez pour sélectionner</p>
                <input type="file" 
                       ref="fileInputSimple" 
                       @change="handleFileSelect" 
                       accept="audio/*" 
                       style="display: none;">
                <button @click="$refs.fileInputSimple.click()" class="btn btn-primary">
                  Choisir un fichier
                </button>
              </div>
              
              <div v-if="file" class="file-info">
                <div class="file-details">
                  <strong>Fichier sélectionné:</strong> {{ file.name }}<br>
                  <strong>Taille:</strong> {{ formatFileSize(file.size) }}<br>
                  <strong>Type:</strong> {{ file.type }}
                </div>
                <button @click="clearFile" class="btn btn-secondary">Supprimer</button>
              </div>
            </div>

            <!-- Bouton de traitement -->
            <div v-if="file && !isProcessing" class="upload-controls">
              <button @click="uploadFileSimple" class="btn btn-success btn-large">
                🚀 Traiter le fichier
              </button>
            </div>

            <!-- Progress -->
            <div v-if="isProcessing" class="processing-status">
              <div class="loading-spinner"></div>
              <p>Traitement en cours... Veuillez patienter.</p>
            </div>

            <!-- Résultats -->
            <div v-if="transcriptionResult && !isProcessing" class="results-container">
              <h3>Résultats de transcription</h3>
              
              <!-- Statistiques -->
              <div v-if="transcriptionResult.statistics" class="stats-simple">
                <div class="stat-item">
                  <strong>Durée:</strong> {{ formatDuration(transcriptionResult.statistics.total_duration) }}
                </div>
                <div class="stat-item">
                  <strong>Temps de traitement:</strong> {{ transcriptionResult.statistics.processing_time }}s
                </div>
                <div v-if="transcriptionResult.statistics.detected_language" class="stat-item">
                  <strong>Langue détectée:</strong> {{ transcriptionResult.statistics.detected_language }}
                </div>
              </div>

              <!-- Chunks avec audio -->
              <div class="chunks-container">
                <div v-for="(chunk, index) in transcriptionResult.chunks" :key="index" class="chunk-item">
                  <div class="chunk-header">
                    <span class="timestamp">{{ formatTimestamp(chunk.timestamp[0]) }} - {{ formatTimestamp(chunk.timestamp[1]) }}</span>
                    <button v-if="transcriptionResult.audio_url" 
                            @click="playSegmentWithTimestamps(transcriptionResult.audio_url, chunk.timestamp[0], chunk.timestamp[1])"
                            class="btn-play"
                            :class="{ 'audio-available': transcriptionResult.audio_url }">
                      🔊 Écouter
                    </button>
                    <span v-else class="audio-unavailable">🔇 Audio non disponible</span>
                  </div>
                  <div class="chunk-text">{{ chunk.text }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- FIN MODE API SIMPLE -->

        </div>
      </div>

    </div>
  </div>
</template>

<script>

import axios from 'axios';
// import TaskToggle from './components/TaskToggle.vue'
//import CustomToggle from './components/CustomToggle.vue'
import QuestionForm from './components/QuestionForm.vue'
import Dictaphone from './components/MyDictaphone.vue'



export default {
  components: {
    // TaskToggle,
    // CustomToggle,
    QuestionForm,
    Dictaphone
  },

  watch: {
    'settings': {
      // handler(newVal, oldVal) {
        handler() {
        // Appeler saveSettings chaque fois que settings.task change
        this.saveSettings();
      },
      deep: true // Cette option n'est pas nécessaire ici car il s'agit d'une chaîne de caractères
    },

    'transcriptionLive.text': function(newVal) {
    if (newVal) {
      // Ajouter le nouveau texte à la transcription accumulée
      this.accumulatedTranscription += newVal + "\n";
      }
    },

    accumulatedTranscription() {
      // Défilement automatique du textarea vers le bas
      this.$nextTick(() => {
        const textarea = this.$refs.transcriptionArea;
        if (textarea) {
          textarea.scrollTop = textarea.scrollHeight;
        }
      });
    }
  },


  data() {
    return {

      initializedModels: false, // Indique si les modèles sont bien chargés

      activeTab: 'streaming', // Par défaut, mode streaming (upload + progressif)

      volumeHeight: 0,        // Niveau de volume (0 à 100)
      maxHeight: 10,          // Nombre maximum de blocs dans la barre
      asciiVolumeBar: '',     // Contient la barre en ASCII
      analyser: null,
      asciiSpectrogram: "▁▁▁▁", // Pour afficher le spectromètre en ASCII

      switch1: true,

      transcriptionLive: { "text": "", "chunks": [ { "text": "", "timestamp": [ 0, 0.1 ] }] },
      accumulatedTranscription: "", // Variable pour stocker la transcription accumulée

      audioBuffer: [],      // Tableau pour accumuler les données audio
      bufferDuration: 0,    // Durée accumulée en secondes

      isRecording: false, // État de l'enregistrement
      mediaRecorder: null, // Instance du MediaRecorder
      audioChunks: [],
      recordingTime: 0,
      timerInterval: null,
      stream: null,

      showSettingsModal: false,
      showStatisticsModal: false,

      settings: {
        task: "transcribe", // valeur par défaut
        model: "openai/whisper-large-v3-turbo",
        lang: "auto",
        chat_model: 'gpt-4'
      },

      availableModels: [
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-base",
        "openai/whisper-large",
      ],

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
      currentSegmentIndex: null, // Pour garder une trace du segment en cours
      
      // Variables pour le mode API Simple
      isProcessing: false, // Indique si un traitement est en cours
      transcriptionResult: null, // Résultat de transcription pour le mode simple
      isDragOver: false, // Pour l'effet de drag & drop
    };
  },

  async created() {
    // Appeler /initialize/ avant que l'application ne soit affichée complètement
    await this.initializeModels();

    // Event listeners keep-alive supprimés - non nécessaires
  },

  beforeUnmount() {
    // Event listeners keep-alive supprimés - nettoyage non nécessaire
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
    async initializeModels() {
      try {
        // Vérifier l'état des modèles via l'endpoint /health/
        const response = await axios.get("/health/");
        console.log("État des modèles:", response.data);
        
        // Vérifier si tous les modèles critiques sont chargés
        const modelsLoaded = response.data.models_loaded;
        if (modelsLoaded.whisper && modelsLoaded.diarization) {
          this.initializedModels = true;
          console.log("✅ Tous les modèles sont initialisés");
        } else {
          console.warn("⚠️ Certains modèles ne sont pas encore chargés:", modelsLoaded);
          // Réessayer après un délai
          setTimeout(() => this.initializeModels(), 2000);
        }
      } catch (error) {
        console.error("Erreur lors de l'initialisation des modèles :", error);
        // Réessayer après un délai en cas d'erreur
        setTimeout(() => this.initializeModels(), 3000);
      }
    },

    openSwaggerAPI() {
      // Ouvrir l'API Swagger dans un nouvel onglet
      const apiUrl = process.env.VUE_APP_API_URL || 'http://localhost:8000';
      window.open(`${apiUrl}/docs`, '_blank');
      
      // Afficher une info pour l'utilisateur
      console.log('🔧 Mode Simple API ouvert dans Swagger');
      console.log('📋 Utilisez l\'endpoint /transcribe_simple/ pour l\'API pure');
    },

    // Fonction keep-alive supprimée - non nécessaire pour cette application


    toggleTask() {
        // Basculer entre "translate" et "transcribe" en fonction de l'état du switch
        this.settings.task = this.settings.task === 'translate' ? 'transcribe' : 'translate';
      },

      toggleModel() {
        // Basculer entre "translate" et "transcribe" en fonction de l'état du switch
        this.settings.chat_model = this.settings.chat_model === 'gpt-4' ? 'chocolatine' : 'gpt-4';
      },

    onToggleChange(newValue) {
      console.log('Statut du toggle changé à :', newValue);
    },

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

          // Initialiser le WebSocket
          console.log("WebSocket URL:", process.env.VUE_APP_WEBSOCKET_URL);
          this.socket = new WebSocket(process.env.VUE_APP_WEBSOCKET_URL);

          this.socket.onopen = () => {
            console.log("WebSocket connection opened");
          };

          this.socket.onmessage = (event) => {
            try {
              const message = JSON.parse(event.data);

              if (message.chunk_duration !== undefined) {
                console.log("Chunk duration received:", message.chunk_duration, "seconds");
                this.chunkDuration = message.chunk_duration;
              }

              if (message.transcription_live !== undefined) {
                console.log("Transcription received:", message.transcription_live);
                this.transcriptionLive = message.transcription_live;

              }

            } catch (error) {
              console.error("Erreur, je n'ai pas réussi à parser le message:", error);
            }
          };

          this.socket.onerror = (error) => {
            console.error("WebSocket error:", error);
          };

          // Utiliser l'API Web Audio
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          this.audioContext = new AudioContext({ sampleRate: 16000 });
          const bufferSize = 4096;

          // Créer l'AnalyserNode
          this.analyser = this.audioContext.createAnalyser();
          this.analyser.fftSize = 256; // Taille du FFT (affecte la résolution)

          
          this.scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
          this.input = this.audioContext.createMediaStreamSource(this.stream);

          // Créer l'AnalyserNode
          this.analyser = this.audioContext.createAnalyser();
          this.analyser.fftSize = 256; // La taille FFT affecte la résolution des données analysées

          // Connecter l'input à l'analyser
          this.input.connect(this.analyser);

          // Fonction pour analyser les données
          this.analyzeAudio();


          this.input.connect(this.scriptNode);
          this.scriptNode.connect(this.audioContext.destination);

          this.audioBuffer = [];
          this.bufferDuration = 0;
          this.fullAudioBuffer = []; // Pour accumuler l'audio complet

          this.scriptNode.onaudioprocess = (audioProcessingEvent) => {
            const inputBuffer = audioProcessingEvent.inputBuffer;
            const inputData = inputBuffer.getChannelData(0);

            // Convertir en Int16Array
            const int16Data = this.convertFloat32ToInt16(inputData);

            // Accumuler les données pour les chunks de 2 secondes
            this.audioBuffer.push(int16Data);
            this.bufferDuration += inputBuffer.duration;

            // Accumuler l'intégralité de l'audio
            this.fullAudioBuffer.push(int16Data);

            // Vérifier si nous avons accumulé au moins 1 secondes d'audio
            if (this.bufferDuration >= 1) {
              const mergedBuffer = this.mergeBuffers(this.audioBuffer);

              // Envoyer les données au backend
              if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(mergedBuffer.buffer);
              }

              // Réinitialiser le buffer et la durée pour les chunks
              this.audioBuffer = [];
              this.bufferDuration = 0;
            }
          };

          // Démarrer l'enregistrement
          this.isRecording = true;
          this.startTime = Date.now();
          this.startTimer();

        } catch (err) {
          console.error('Erreur lors de l\'accès au microphone:', err);
          alert('Impossible d\'accéder au microphone. Veuillez vérifier les permissions.', err);
        }
      } else {
        // Arrêt de l'enregistrement
        this.asciiSpectrogram = '';

        if (this.scriptNode) {
          // Avant de déconnecter, envoyer les données restantes s'il y en a
          if (this.audioBuffer.length > 0) {
            const mergedChunkBuffer = this.mergeBuffers(this.audioBuffer);
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
              this.socket.send(mergedChunkBuffer.buffer);
            }
            this.audioBuffer = [];
            this.bufferDuration = 0;
          }

          this.scriptNode.disconnect();
          this.scriptNode = null;
        }
        if (this.input) {
          this.input.disconnect();
          this.input = null;
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
          await this.audioContext.close();
        }
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
        }

        // Traiter l'audio complet
        if (this.fullAudioBuffer.length > 0) {
          // Fusionner l'audio complet
          const fullMergedBuffer = this.mergeBuffers(this.fullAudioBuffer);


          // Créer un fichier WAV valide
          const audioBlob = this.createWAVFile(fullMergedBuffer, 16000);
          // Créer un Blob à partir du buffer fusionné
          // const audioBlob = new Blob([fullMergedBuffer.buffer], { type: 'audio/wav' });

          // Si vous avez besoin de créer un objet File
          const audioFile = new File([audioBlob], 'recording.wav', {
            type: 'audio/wav',
            lastModified: Date.now()
          });

          // Ajouter la durée aux métadonnées du fichier
          Object.defineProperty(audioFile, 'duration', {
            value: this.recordingTime,
            writable: false
          });

          // Appeler la fonction pour traiter l'audio
          this.handleRecordedAudio(audioFile);

          // Réinitialiser le buffer complet
          this.fullAudioBuffer = [];
        }

        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.close();
        }
        this.isRecording = false;
        this.stopTimer();
      }
    },

    analyzeAudio() {
      // Créer un tableau de données pour stocker les valeurs de fréquence
      const bufferLength = this.analyser.frequencyBinCount; // Nombre de valeurs de fréquence
      const dataArray = new Uint8Array(bufferLength); // Tableau pour stocker les valeurs

      const logFrequencyData = () => {
        // Obtenir les données de fréquence de l'AnalyserNode
        this.analyser.getByteFrequencyData(dataArray);

        // Transformer les valeurs en spectromètre ASCII
        this.asciiSpectrogram = this.generateAsciiSpectrogram(dataArray);

        // Appeler cette fonction à nouveau pour continuer à enregistrer les valeurs
        requestAnimationFrame(logFrequencyData);
      };

      // Démarrer l'analyse
      logFrequencyData();
    },

    generateAsciiSpectrogram(dataArray) {
  // Définir les bandes de fréquence intéressantes pour la voix humaine
  const bands = [
    { start: 0, end: 8, label: "20-100 Hz" }, // Basses fréquences
    { start: 8, end: 16, label: "100-400 Hz" }, // Moyennes-basses
    { start: 16, end: 32, label: "400-1000 Hz" }, // Moyennes
    { start: 32, end: 64, label: "1000-4000 Hz" }, // Aigus
  ];

  // Définir les caractères représentant les niveaux d'amplitude
  const levels = ['▁', '▂', '▃', '▅', '▆', '▇'];

  let ascii = "";

  // Pour chaque bande, calculer l'intensité moyenne et choisir le caractère correspondant
  for (const band of bands) {
    let bandIntensity = 0;

    // Calculer l'intensité moyenne de la bande
    for (let i = band.start; i < band.end; i++) {
      bandIntensity += dataArray[i];
    }
    bandIntensity /= (band.end - band.start);

    // Normaliser la valeur entre 0 et le nombre de niveaux disponibles
    const levelIndex = Math.min(
      levels.length - 1, 
      Math.floor((bandIntensity / 255) * (levels.length))
    );

    // Ajouter le caractère correspondant à l'intensité moyenne
    ascii += levels[levelIndex];
  }

  return ascii;
},


  // Méthode 1: Utiliser la FFT pour calculer le niveau en dB
  getDecibelLevelFromFFT() {
      this.analyser.getByteFrequencyData(this.dataArray);
      
      // Calculer la moyenne des amplitudes
      const average = this.dataArray.reduce((acc, val) => acc + val, 0) / this.bufferLength;
      
      // Convertir en dB (référence arbitraire à 0dB = amplitude maximale)
      const db = 20 * Math.log10(average / 255);
      
      return Math.max(db, -100); // Limiter à -100dB minimum
  },




    convertFloat32ToInt16(buffer) {
      const l = buffer.length;
      const buf = new Int16Array(l);
      for (let i = 0; i < l; i++) {
        let s = Math.max(-1, Math.min(1, buffer[i]));
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return buf;
    },

    mergeBuffers(bufferArray) {
      let totalLength = 0;
      bufferArray.forEach(buf => {
        totalLength += buf.length;
      });

      const result = new Int16Array(totalLength);
      let offset = 0;
      bufferArray.forEach(buf => {
        result.set(buf, offset);
        offset += buf.length;
      });

      return result;
    },

    createWAVFile(int16Data, sampleRate) {
      const buffer = new ArrayBuffer(44 + int16Data.length * 2);
      const view = new DataView(buffer);

      /* RIFF identifier */
      this.writeString(view, 0, 'RIFF');
      /* file length */
      view.setUint32(4, 36 + int16Data.length * 2, true);
      /* RIFF type */
      this.writeString(view, 8, 'WAVE');
      /* format chunk identifier */
      this.writeString(view, 12, 'fmt ');
      /* format chunk length */
      view.setUint32(16, 16, true);
      /* sample format (raw) */
      view.setUint16(20, 1, true);
      /* channel count */
      view.setUint16(22, 1, true);
      /* sample rate */
      view.setUint32(24, sampleRate, true);
      /* byte rate (sample rate * block align) */
      view.setUint32(28, sampleRate * 2, true);
      /* block align (channel count * bytes per sample) */
      view.setUint16(32, 2, true);
      /* bits per sample */
      view.setUint16(34, 16, true);
      /* data chunk identifier */
      this.writeString(view, 36, 'data');
      /* data chunk length */
      view.setUint32(40, int16Data.length * 2, true);

      // Write the PCM samples
      for (let i = 0; i < int16Data.length; i++) {
        view.setInt16(44 + i * 2, int16Data[i], true);
      }

      return new Blob([view], { type: 'audio/wav' });
    },

    writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
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

    beforeUnmount() {
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
      this.progressMessage = "Extraction audio terminée!"; // Mettre à jour le message affiché dans l'UI
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
      // Vérifier si l'URL audio est valide
      if (!audioUrl || audioUrl === null || audioUrl === 'null') {
        console.warn("Aucune URL audio disponible pour charger les métadonnées");
        return;
      }

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
      // Vérifier si l'URL audio est valide
      if (!segment.audio_url || segment.audio_url === null || segment.audio_url === 'null') {
        console.warn("Aucune URL audio disponible pour ce segment");
        return;
      }

      // Si le même segment est déjà en cours de lecture, l'arrêter
      if (this.playingIndex === index && this.audio) {
        this.audio.pause();
        this.audio = null;
        this.playingIndex = null;
        return;
      }

      // Arrêter tout audio en cours
      if (this.audio) {
        this.audio.pause();
        this.audio = null;
      }

      // Démarrer la lecture du nouveau segment
      try {
        this.audio = new Audio(segment.audio_url);
        this.playingIndex = index;

        // Gérer les événements de l'audio
        this.audio.onended = () => {
          this.playingIndex = null;
        };

        this.audio.onerror = (e) => {
          console.error("Erreur de lecture audio:", e);
          this.playingIndex = null;
          this.audio = null;
        };

        // Lancer la lecture avec gestion d'erreur
        const playPromise = this.audio.play();
        if (playPromise !== undefined) {
          playPromise.catch(error => {
            console.error("Erreur lors du play():", error);
            this.playingIndex = null;
            this.audio = null;
          });
        }

      } catch (error) {
        console.error("Erreur création Audio:", error);
        this.playingIndex = null;
        this.audio = null;
      }
    },

    // Ouvrir les paramètres (à personnaliser)
    openSettings() {
      this.showSettingsModal = true;
    },
    closeSettings() {
      this.showSettingsModal = false; // Fermer la fenêtre modale de statistiques
    },

    // Ouvrir les statistiques
    openStatistics() {
      this.showStatisticsModal = true;
    },
    closeStatistics() {
      this.showStatisticsModal = false; // Fermer la fenêtre modale de statistiques
    },


    async saveSettings() {
      try {
        console.log('Settings envoyés:', this.settings);
        const response = await axios.post('/settings/', this.settings);
        console.log("chat_model dans App.vue:", this.settings.chat_model);

        console.log('Response:', response.data);
      } catch (error) {
        console.log('Full error:', error);
        console.log('Request config:', error.config);
        console.log('Request URL:', error.config.url);
        console.error('Error saving settings:', error);
      }
      // this.closeSettings();
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
      // Vérifier si l'URL audio est valide
      if (!audioUrl || audioUrl === null || audioUrl === 'null') {
        console.warn("Aucune URL audio disponible pour ce segment");
        return;
      }

      // Si le même chunk est déjà en cours de lecture, l'arrêter
      if (this.currentAudio && this.currentChunkIndex === chunkIndex) {
        this.currentAudio.pause();
        this.currentAudio = null;
        this.currentChunkIndex = null;
        return;
      }

      // Arrêter tout audio en cours
      if (this.currentAudio) {
        this.currentAudio.pause();
        this.currentAudio = null;
      }

      try {
        // Créer un nouvel objet Audio pour jouer le nouveau chunk
        const audio = new Audio(audioUrl);
        
        // Gérer les événements d'erreur
        audio.onerror = (e) => {
          console.error("Erreur de lecture audio chunk:", e);
          this.currentAudio = null;
          this.currentChunkIndex = null;
        };

        // Définir le temps de départ
        audio.currentTime = startTime;
        
        // Lancer la lecture avec gestion d'erreur
        const playPromise = audio.play();
        if (playPromise !== undefined) {
          playPromise.then(() => {
            // Définir un timeout pour arrêter l'audio à la fin du chunk
            setTimeout(() => {
              if (audio && !audio.paused) {
                audio.pause();
              }
            }, (endTime - startTime) * 1000);
          }).catch(error => {
            console.error("Erreur lors du play() chunk:", error);
            this.currentAudio = null;
            this.currentChunkIndex = null;
          });
        }

        // Stocker la référence de l'audio en cours et l'index du chunk
        this.currentAudio = audio;
        this.currentChunkIndex = chunkIndex;

      } catch (error) {
        console.error("Erreur création Audio chunk:", error);
        this.currentAudio = null;
        this.currentChunkIndex = null;
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
      // Vérifier si l'URL audio est valide
      if (!audioUrl || audioUrl === null || audioUrl === 'null') {
        console.warn("Aucune URL audio disponible");
        return;
      }

      try {
        const audio = new Audio(audioUrl);
        
        // Gérer les erreurs
        audio.onerror = (e) => {
          console.error("Erreur de lecture audio:", e);
        };
        
        // Lancer la lecture avec gestion d'erreur
        const playPromise = audio.play();
        if (playPromise !== undefined) {
          playPromise.catch(error => {
            console.error("Erreur lors du play():", error);
          });
        }
        
      } catch (error) {
        console.error("Erreur création Audio:", error);
      }
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

    // Méthode optimisée pour lire un segment avec timestamps
    playSegmentWithTimestamps(audioUrl, startTime, endTime, segmentIndex) {
      // Vérifier si l'URL audio est valide
      if (!audioUrl || audioUrl === null || audioUrl === 'null') {
        console.warn("Aucune URL audio disponible pour ce segment");
        return;
      }

      // Convertir l'URL relative en URL absolue vers le backend
      const fullAudioUrl = audioUrl.startsWith('http') ? audioUrl : `${process.env.VUE_APP_API_URL}${audioUrl}`;
      console.log('URL audio complète:', fullAudioUrl);

      // Si le même segment est en cours, l'arrêter
      if (this.currentAudio && this.currentSegmentIndex === segmentIndex) {
        this.currentAudio.pause();
        this.currentAudio = null;
        this.currentSegmentIndex = null;
        return;
      }

      // Arrêter tout audio en cours
      if (this.currentAudio) {
        this.currentAudio.pause();
        this.currentAudio = null;
      }

      try {
        const audio = new Audio(fullAudioUrl);
        this.currentAudio = audio;
        this.currentSegmentIndex = segmentIndex;

        // Gérer les événements
        audio.onerror = (e) => {
          console.error("Erreur de lecture audio segment:", e);
          this.currentAudio = null;
          this.currentSegmentIndex = null;
        };

        audio.onloadeddata = () => {
          // Positionner au début du segment
          audio.currentTime = startTime;
          
          // Lancer la lecture
          const playPromise = audio.play();
          if (playPromise !== undefined) {
            playPromise.then(() => {
              // Programmer l'arrêt à la fin du segment
              const segmentDuration = (endTime - startTime) * 1000;
              setTimeout(() => {
                if (audio && !audio.paused && this.currentSegmentIndex === segmentIndex) {
                  audio.pause();
                  this.currentAudio = null;
                  this.currentSegmentIndex = null;
                }
              }, segmentDuration);
            }).catch(error => {
              console.error("Erreur lors du play() segment:", error);
              this.currentAudio = null;
              this.currentSegmentIndex = null;
            });
          }
        };

        // Fallback si les métadonnées ne se chargent pas rapidement
        setTimeout(() => {
          if (audio.readyState >= 2) { // HAVE_CURRENT_DATA
            audio.currentTime = startTime;
          }
        }, 100);

      } catch (error) {
        console.error("Erreur création Audio segment:", error);
        this.currentAudio = null;
        this.currentSegmentIndex = null;
      }
    },

    // Méthode pour lire un chunk spécifique (ancienne version gardée pour compatibilité)
    playChunk(audioUrl, start, end) {
      this.playSegmentWithTimestamps(audioUrl, start, end, Math.random());
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
      if (!this.audio) return;
      
      try {
        if (this.isPlaying) {
          this.audio.pause();
          this.isPlaying = false;
        } else {
          const playPromise = this.audio.play();
          if (playPromise !== undefined) {
            playPromise.then(() => {
              this.isPlaying = true;
            }).catch(error => {
              console.error("Erreur lors du play() toggle:", error);
              this.isPlaying = false;
            });
          } else {
            this.isPlaying = true;
          }
        }
      } catch (error) {
        console.error("Erreur togglePlay:", error);
        this.isPlaying = false;
      }
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
        // Préparer les données pour l'upload via FormData en POST
        const uploadResponse = await fetch(`${process.env.VUE_APP_API_URL}/transcribe_streaming/`, {
          method: 'POST',
          body: formData
        });

        if (!uploadResponse.ok) {
          console.error("Erreur dans la réponse du serveur :", uploadResponse.statusText);
          return;
        }

        // Lire la réponse streaming
        const reader = uploadResponse.body.getReader();
        const decoder = new TextDecoder();

        // Arrêter la boucle de progression initiale
        this.stopProgressLoop();

        // Traiter le flux de données
        try {
          // eslint-disable-next-line no-constant-condition
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  console.log("Données reçues:", data);

                  // Traiter selon le status
                  if (data.status === 'started') {
                    this.progressMessage = data.message;
                  } else if (data.status === 'audio_ready') {
                    this.progressMessage = data.message;
                  } else if (data.status === 'diarization_start') {
                    this.progressMessage = data.message;
                  } else if (data.status === 'diarization_done') {
                    this.diarization = data.diarization;
                    this.progressMessage = "Diarisation terminée - Transcription en cours...";
                  } else if (data.status === 'transcribing') {
                    this.transcriptionProgress = data.progress;
                    this.progressMessage = `Transcription segment ${data.segment}/${data.total} (${data.progress.toFixed(1)}%)`;
                  } else if (data.status === 'segment_done') {
                    // Ajouter le segment transcrit
                    const segment = data.segment;
                    this.transcriptions.push({
                      speaker: segment.speaker,
                      text: { chunks: [{ text: segment.text, timestamp: [segment.start_time, segment.end_time] }] },
                      start_time: segment.start_time,
                      end_time: segment.end_time,
                      audio_url: null // Pas d'URL audio en mode streaming pour l'instant
                    });
                  } else if (data.status === 'completed') {
                    this.transcriptionProgress = 100;
                    this.progressMessage = "Transcription terminée !";
                    
                    // Calculer les statistiques finales
                    if (this.diarization) {
                      this.calculateSpeechStats();
                    }
                    
                    console.log("Transcription streaming terminée avec succès");
                    break;
                  } else if (data.status === 'error') {
                    console.error("Erreur du serveur:", data.message);
                    this.progressMessage = `Erreur: ${data.message}`;
                  }

                } catch (parseError) {
                  console.log("Ligne non-JSON ignorée:", line);
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } catch (error) {
        console.error("Erreur lors de l'upload ou récupération des transcriptions", error);
      }
    },

    // Fonctions utilitaires pour le mode Simple
    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    formatDuration(seconds) {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);
      
      if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
      } else {
        return `${minutes}m ${secs}s`;
      }
    },

    formatTimestamp(seconds) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      const millisecs = Math.floor((seconds % 1) * 100);
      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millisecs.toString().padStart(2, '0')}`;
    },

    handleDrop(event) {
      event.preventDefault();
      this.isDragOver = false;
      const files = event.dataTransfer.files;
      if (files.length > 0) {
        this.file = files[0];
      }
    },

    handleFileSelect(event) {
      const files = event.target.files;
      if (files.length > 0) {
        this.file = files[0];
      }
    },

    clearFile() {
      this.file = null;
      this.transcriptionResult = null;
    },

    // Nouvelle fonction pour le mode API Simple
    async uploadFileSimple() {
      this.isProcessing = true;
      this.transcriptionResult = null;

      const formData = new FormData();
      formData.append('file', this.file);

      try {
        console.log('Envoi vers /transcribe_simple/');
        const response = await fetch(`${process.env.VUE_APP_API_URL}/transcribe_simple/`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const result = await response.json();
        console.log('Réponse reçue:', result);
        
        // Adaptation pour le mode simple
        const adaptedResult = {
          audio_url: result.full_audio_url,
          chunks: result.transcriptions.map(item => ({
            text: item.text,
            timestamp: [item.start_time, item.end_time],
            speaker: item.speaker
          })),
          statistics: {
            total_duration: result.transcriptions.length > 0 ? 
              result.transcriptions[result.transcriptions.length - 1].end_time : 0,
            processing_time: "N/A",
            detected_language: "français"
          }
        };
        
        this.transcriptionResult = adaptedResult;
        
        // Validation des URLs audio
        if (adaptedResult.audio_url) {
          console.log('URL audio disponible:', adaptedResult.audio_url);
        } else {
          console.warn('Aucune URL audio dans la réponse');
        }

      } catch (error) {
        console.error('Erreur lors du traitement:', error);
        alert('Erreur lors du traitement du fichier: ' + error.message);
      } finally {
        this.isProcessing = false;
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

/* Hover par défaut déplacé vers les classes spécifiques audio-available et no-audio */

.chunk::after {
  content: "🖱️ Clic pour écouter";
  position: absolute;
  top: +120%; /* Positionne l'infobulle juste au-dessous du chunk */
  /* left: 50%; */
  right: 50%;
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

/* Styles pour les chunks avec audio disponible */
.chunk.audio-available {
  cursor: pointer;
}

.chunk.audio-available:hover {
  background-color: yellow;
}

/* Styles pour les chunks sans audio */
.chunk.no-audio {
  cursor: default;
  opacity: 0.7;
}

.chunk.no-audio:hover {
  background-color: #f5f5f5;
}

.chunk.no-audio::after {
  content: "🚫 Audio non disponible";
  background-color: #999;
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
  width: 48px;
  height: 48px;
  border-radius: 50%;
  padding: 0;
  border: none;
  background-color: #f70505;
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

.recording-status {
  color: red;
  animation: blink 1s infinite;
}

@keyframes blink {
  50% { opacity: 0; }
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

.switch input
{
  display: none;
}

.switch 
{
  display: inline-block;
  width: 60px; /*=w*/
  height: 30px; /*=h*/
  margin: 4px;
  transform: translateY(50%);
  position: relative;
  margin-bottom: 8px; /* Ajustez la valeur pour augmenter l'espace sous le bouton */

}

.slider
{
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  border-radius: 30px;
  box-shadow: 0 0 0 2px #777, 0 0 4px #777;
  cursor: pointer;
  border: 4px solid transparent;
  overflow: hidden;
  transition: 0.2s;
}

.slider:before
{
  position: absolute;
  content: "";
  width: 100%;
  height: 100%;
  background-color: #777;
  border-radius: 30px;
  transform: translateX(-30px); /*translateX(-(w-h))*/
  transition: 0.2s;
}

input:checked + .slider:before
{
  transform: translateX(30px); /*translateX(w-h)*/
  background-color: limeGreen;
}

input:checked + .slider
{
  box-shadow: 0 0 0 2px limeGreen, 0 0 8px limeGreen;
}

.switch200 .slider:before
{
  width: 200%;
  transform: translateX(-82px); /*translateX(-(w-h))*/
}

.switch200 input:checked + .slider:before
{
  background-color: red;
}

.switch200 input:checked + .slider
{
  box-shadow: 0 0 0 2px red, 0 0 8px red;
}

/* Classe pour le texte en gras */
.bold {
  font-weight: bold;
}

/* Style pour les onglets verticaux */
.vertical-tabs {
  display: flex;
  flex-direction: column;
  position: fixed;
  left: 0;
  top: 0;
  padding: 10px;
  background-color: #f0f0f0;
  height: 100vh;
}

.vertical-tabs button {
  padding: 10px;
  cursor: pointer;
  margin-bottom: 10px;
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 5px 0 0 5px;
  transition: background-color 0.3s;
}

.vertical-tabs button:hover {
  background-color: #45a049;
}

.vertical-tabs .active {
  font-weight: bold;
  background-color: #388e3c;
}

.tab-content {
  padding: 8px;  /* Réduit de 16px à 8px */
  border: 1px solid #e2e8f0;
  border-top: none;
  flex: 1;
}

.tabs-container {
  display: flex;
  flex-direction: column;
}

.tabs-header {
  display: inline-flex;  /* Pour que les onglets ne prennent que l'espace nécessaire */
  border-bottom: 1px solid #e2e8f0;
}

.tab-button {
  padding: 6px 12px;
  font-size: 14px;
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  margin-bottom: -1px;
  border-radius: 6px 6px 0 0;
  cursor: pointer;
  color: #666;
}

.tab-button:hover {
  color: #333;
}

.tab-button.active {
  background: white;
  border-color: #e2e8f0;
  border-bottom: 1px solid white;
  color: #000;
}

.tab-title {
  font-weight: bold;
  padding-bottom: 5px;
  margin-bottom: 10px;
  display: inline-block;
}

.tab-subtitle {
  display: block;
  font-size: 10px;
  font-weight: normal;
  color: #888;
  margin-top: 2px;
}

/* Styles pour les nouveaux modes */
.api-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  border: 1px solid #667eea !important;
}

.api-button:hover {
  background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
  color: white !important;
}

.mode-features {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.feature {
  background: #f0f9ff;
  border: 1px solid #0ea5e9;
  border-radius: 12px;
  padding: 4px 8px;
  font-size: 12px;
  color: #0369a1;
}

.dark .feature {
  background: #0c4a6e;
  border-color: #0ea5e9;
  color: #7dd3fc;
}

/* Mode Live spécifique */
.live-mode-container {
  max-width: 900px;
  margin: 0 auto;
}

.live-recording-container {
  text-align: center;
  padding: 20px;
}

.live-spectrogram {
  font-family: monospace;
  font-size: 24px;
  letter-spacing: 2px;
  color: #0ea5e9;
  margin: 10px 0;
}

.live-textarea {
  min-height: 200px;
  font-size: 16px;
  line-height: 1.6;
}

.tab-content {
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-top: none;
  flex: 1;
}

pre {
  font-family: monospace, "Courier New", Courier, "Lucida Console", Consolas;
  white-space: pre;
  line-height: 1.2em;
}

/* Styles pour le mode API Simple */
.upload-area {
  border: 2px dashed #e2e8f0;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  margin: 20px 0;
  transition: all 0.3s ease;
  background: #fafafa;
}

.upload-area.drag-over {
  border-color: #0ea5e9;
  background: #f0f9ff;
}

.upload-placeholder {
  color: #64748b;
}

.upload-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.file-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f1f5f9;
  padding: 16px;
  border-radius: 8px;
  margin: 16px 0;
}

.file-details {
  text-align: left;
}

.upload-controls {
  text-align: center;
  margin: 20px 0;
}

.btn-large {
  padding: 12px 24px;
  font-size: 16px;
  font-weight: bold;
}

.processing-status {
  text-align: center;
  padding: 20px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e2e8f0;
  border-top: 4px solid #0ea5e9;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.results-container {
  margin-top: 20px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 20px;
  background: #fafafa;
}

.stats-simple {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 16px;
  background: #f1f5f9;
  border-radius: 8px;
  flex-wrap: wrap;
}

.stat-item {
  font-size: 14px;
  color: #475569;
}

.chunks-container {
  max-height: 500px;
  overflow-y: auto;
}

.chunk-item {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-bottom: 12px;
  background: white;
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.timestamp {
  font-family: monospace;
  font-size: 12px;
  color: #64748b;
  background: #e2e8f0;
  padding: 4px 8px;
  border-radius: 4px;
}

.chunk-text {
  padding: 16px;
  line-height: 1.6;
}

.btn-play {
  background: #10b981;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: background 0.2s;
}

.btn-play:hover {
  background: #059669;
}

.btn-play.audio-available {
  background: #10b981;
}

.audio-unavailable {
  color: #ef4444;
  font-size: 12px;
  font-style: italic;
}

/* Mode sombre pour le mode Simple */
.dark .upload-area {
  background: #1e293b;
  border-color: #475569;
}

.dark .upload-area.drag-over {
  border-color: #0ea5e9;
  background: #0c4a6e;
}

.dark .file-info {
  background: #334155;
}

.dark .results-container {
  background: #1e293b;
  border-color: #475569;
}

.dark .stats-simple {
  background: #334155;
}

.dark .chunk-item {
  background: #334155;
  border-color: #475569;
}

.dark .chunk-header {
  background: #475569;
  border-color: #64748b;
}

</style>
# TranscriMate 🎙️

TranscriMate est une application de transcription audio/vidéo intelligente qui utilise l'IA pour séparer les voix, transcrire le contenu et permettre d'interagir avec les transcriptions via un chatbot.

## 🚀 Fonctionnalités

- **Transcription audio/vidéo** : Conversion automatique de fichiers audio et vidéo en texte
- **Séparation des locuteurs** : Diarisation automatique via pyannote (qui parle, et quand)
- **Traduction** : Traduction automatique vers l'anglais (mode `translate` de Whisper)
- **Chatbot IA** : Questions/réponses sur la transcription (Chocolatine en local ou GPT-4o-mini via API)
- **Transcription en temps réel** : Enregistrement et transcription live via microphone (WebSocket)
- **Interface moderne** : Interface web responsive avec mode sombre/clair

## 🧠 Modèles utilisés

Les modèles sont **chargés au démarrage du backend** et fixés dans le code (`load_core_models()` dans `backend/main.py`) :

| Rôle | Modèle |
|------|--------|
| Transcription de fichiers | `openai/whisper-large-v3-turbo` |
| Transcription live (micro) | `openai/whisper-base` |
| Diarisation | `pyannote/speaker-diarization-3.1` |
| Chatbot local | `jpacifico/Chocolatine-3B-Instruct-DPO-v1.2` (optionnel) |
| Chatbot API | `gpt-4o-mini` (optionnel, nécessite une clé OpenAI) |

Sur GPU, Whisper et Chocolatine tournent en FP16 ; sinon en FP32 sur CPU. Si Chocolatine ne peut pas être chargé, l'application démarre quand même : seul le chatbot local est désactivé.

## 📋 Prérequis

### Backend
- Python 3.10+
- ffmpeg installé sur le système (requis par pydub / moviepy)
- CUDA 12.6 recommandé (les dépendances installent `torch==2.8.0+cu126`)

### Frontend
- Node.js 18.3.0+
- npm

### Clés API requises
- `HF_TOKEN` : **obligatoire** — jeton Hugging Face, requis pour Whisper, pyannote et Chocolatine. Le backend refuse de démarrer sans lui. Le modèle `pyannote/speaker-diarization-3.1` étant *gated*, il faut aussi en accepter les conditions sur huggingface.co avec le compte associé au jeton.
- `OPENAI_API_KEY` : optionnel — uniquement pour le chatbot `gpt-4o-mini`.

## 🛠️ Installation et Configuration

### 1. Cloner le projet
```bash
git clone https://github.com/famibelle/TranscriMate.git
cd TranscriMate
```

### 2. Configuration des variables d'environnement

Créez un fichier `.env` dans le dossier `backend/` :

```bash
# backend/.env
HF_TOKEN=votre_jeton_huggingface
OPENAI_API_KEY=votre_clé_openai   # optionnel
```

Le frontend lit ses variables depuis `frontend/.env.development` (mode `npm run serve`) et `frontend/.env.production` (mode `npm run build`) :

```bash
VUE_APP_API_URL=http://localhost:8000
VUE_APP_WEBSOCKET_URL=ws://localhost:8000/live_transcription/
```

Ces variables sont injectées à la compilation par webpack : après modification, il faut relancer `npm run serve` ou reconstruire.

## 💻 Démarrage en ligne de commande (Développement)

### Backend

```bash
cd backend

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur (depuis backend/ : main.py importe temp_manager et session_manager en imports plats)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Le backend sera accessible sur http://localhost:8000. Le premier démarrage télécharge plusieurs Go de modèles ; l'application répond `status: "loading"` sur `/health/` tant que le chargement n'est pas terminé.

### Frontend

```bash
cd frontend

# Installer les dépendances
npm install

# Lancer en mode développement
npm run serve

# Construire pour la production (sortie dans frontend/dist)
npm run build

# Linter
npm run lint
```

Le frontend sera accessible sur http://localhost:8080

## 🐳 Démarrage avec Docker

```bash
# Construire et lancer tous les services
docker-compose up --build

# En arrière-plan
docker-compose up -d --build

# Arrêter les services
docker-compose down
```

L'application sera accessible sur :
- Frontend : http://localhost:8080
- Backend API : http://localhost:8000

> ℹ️ Le service backend charge `backend/.env` via `env_file` : ce fichier doit exister avant `docker-compose up`, sinon Compose s'arrête avec une erreur.

### Docker individuel

```bash
# Backend (image basée sur nvidia/cuda)
cd backend
docker build -t transcrimate-backend .
docker run -p 8000:8000 --gpus all --env-file .env transcrimate-backend

# Frontend
cd frontend
docker build -t transcrimate-frontend .
docker run -p 8080:8080 transcrimate-frontend
```

## 📱 Utilisation

L'interface est organisée en onglets (`🔄 Mode Streaming`, `🎤 Mode Live`, `🤖 AKABot`, `📄 API Simple`). **Seul l'onglet Streaming est actuellement accessible** : les boutons des trois autres sont commentés dans le template de `frontend/src/App.vue`, même si leurs vues existent toujours dans le code. Décommentez-les pour les réactiver.

### 1. Transcription de fichiers (🔄 Mode Streaming)

1. Glissez-déposez un fichier audio/vidéo ou utilisez le bouton "Sélectionner un fichier"
2. Choisissez la tâche : **Transcrire** (langue d'origine) ou **Traduire** (vers l'anglais)
3. Le traitement se déroule en streaming : préparation audio → diarisation → transcription segment par segment, affichée au fur et à mesure
4. Consultez les résultats par locuteur, réécoutez chaque segment, renommez les locuteurs et copiez la transcription complète

### 2. ChatBot (🤖 AKABot)

1. Choisissez le modèle : **Chocolatine 🍫🥖** (local) ou **OpenAI**
2. Posez vos questions : la transcription courante est envoyée comme contexte

### 3. Transcription en temps réel (🎤 Mode Live)

1. Cliquez sur le bouton microphone pour démarrer l'enregistrement
2. L'audio (PCM 16 kHz mono) est envoyé au backend par WebSocket et transcrit par tranches de 2 secondes avec `whisper-base`

## 📖 Documentation API

Documentation interactive une fois le backend lancé :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

L'API est organisée autour de **trois modes de transcription** :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health/` | GET | État des modèles, CUDA, paramètres courants |
| `/settings/` | POST | Met à jour la tâche (`transcribe` / `translate`) |
| `/transcribe_simple/` | POST | **Mode 1** — traitement complet, réponse JSON unique |
| `/transcribe_streaming/` | POST | **Mode 2** — traitement progressif en flux `data: {json}` (utilisé par l'interface) |
| `/live_transcription/` | WebSocket | **Mode 3** — transcription temps réel depuis le micro |
| `/progress/` | WebSocket | Progression des étapes de diarisation |
| `/temp_audio/{filename}` | GET | Récupération d'un fichier audio de session |
| `/session/create` | POST | Crée une session (fichiers temporaires) |
| `/session/{id}/info` | GET | Informations d'une session |
| `/session/{id}` | DELETE | Supprime une session et ses fichiers |
| `/sessions/list` | GET | Liste des sessions actives |
| `/ask_question/` | POST | Question au chatbot sur une transcription |
| `/chatbot/models` | GET | Modèles de chat disponibles et leur état |

Le fichier [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) décrit une version antérieure de l'API et n'est plus à jour : référez-vous à `/docs`.

### Exemples

```bash
# État des modèles et du GPU
curl http://localhost:8000/health/

# Modèles de chat disponibles
curl http://localhost:8000/chatbot/models

# Transcription complète d'un fichier (Mode 1)
curl -F "file=@mon_audio.mp3" http://localhost:8000/transcribe_simple/

# Transcription en streaming (Mode 2)
curl -N -F "file=@mon_audio.mp3" http://localhost:8000/transcribe_streaming/

# Passer en mode traduction
curl -X POST http://localhost:8000/settings/ \
  -H "Content-Type: application/json" \
  -d '{"task":"translate","model":"openai/whisper-large-v3-turbo","lang":"auto"}'

# Question au chatbot
curl -X POST http://localhost:8000/ask_question/ \
  -H "Content-Type: application/json" \
  -d '{"question":"Résume la discussion","transcription":"...","chat_model":"gpt-4o-mini"}'
```

> ℹ️ `/settings/` accepte les champs `model` et `lang` pour compatibilité, mais seul `task` est réellement appliqué : les modèles Whisper sont fixés au chargement du backend.

## 🗂️ Gestion des fichiers temporaires

Deux mécanismes cohabitent :

- `backend/temp_manager.py` : fichiers de travail dans le répertoire temporaire système, supprimés à la fin de chaque requête.
- `backend/session_manager.py` : un répertoire par session sous `backend/temp/<uuid>/`, qui survit à la requête pour permettre la relecture des segments audio depuis l'interface. Un thread de fond purge les sessions inactives depuis plus de 24 h ainsi que les répertoires orphelins.

Le chemin `backend/temp/` étant relatif, lancez uvicorn depuis le dossier `backend/` pour que les fichiers soient créés au bon endroit.

## 📁 Structure du projet

```
TranscriMate/
├── backend/
│   ├── main.py              # API FastAPI (3 modes, sessions, chatbot)
│   ├── RAG.py               # Index FAISS sur Multimedia/Use_Cases (CLI autonome)
│   ├── session_manager.py   # Sessions et fichiers persistants par utilisateur
│   ├── temp_manager.py      # Fichiers temporaires cross-platform
│   ├── requirements.txt     # Dépendances Python
│   ├── Dockerfile           # Image backend (base CUDA)
│   └── Multimedia/
│       └── Use_Cases/       # Base de connaissances pour le RAG
├── frontend/
│   ├── src/
│   │   ├── App.vue          # Application complète (onglets, WebSockets, lecture audio)
│   │   ├── main.js          # Point d'entrée
│   │   └── components/      # QuestionForm, MyDictaphone, MarkdownRenderer…
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yaml      # Orchestration Docker
├── k8s-config/              # Manifests Kubernetes (registre Scaleway)
├── nginx/nginx.conf         # Reverse proxy (routes à réaligner sur l'API actuelle)
├── uvicorn.service          # Unité systemd backend (déploiement Azure)
├── npm_frontend.service     # Unité systemd frontend (déploiement Azure)
└── API_DOCUMENTATION.md     # Documentation API historique (obsolète)
```

## 🧪 Tests

Le projet n'a pas encore de suite de tests automatisée. Les scripts `test_*.py` à la racine sont des utilitaires manuels :

- `test_mode2.py` : envoie un fichier à `/transcribe_streaming/` sur un serveur déjà lancé et affiche les événements reçus (le chemin de fichier y est au format Windows, à adapter sous Linux).
- `test_minimal.py`, `test_streaming_syntax.py` : scripts de mise au point d'anciennes versions du streaming, `test_streaming_syntax.py` référence une fonction qui n'existe plus.

Vérifiez donc les modifications avec de vraies requêtes (voir les exemples `curl` ci-dessus) et `npm run lint` côté frontend.

## 🐛 Dépannage

1. **Le backend s'arrête au démarrage** : `HF_TOKEN` absent du `backend/.env`, ou conditions d'utilisation de `pyannote/speaker-diarization-3.1` non acceptées sur Hugging Face.
2. **Erreur CUDA / mémoire insuffisante** : `whisper-large-v3-turbo` + pyannote + Chocolatine demandent beaucoup de VRAM. Sur GPU limité, laissez Chocolatine échouer au chargement (l'API continue de fonctionner) ou passez sur CPU.
3. **Le chatbot répond « modèle non supporté »** : seules les valeurs `chocolatine` et `gpt-4o-mini` sont acceptées par `/ask_question/`.
4. **Les segments audio ne se relisent pas** : les fichiers sont servis par `/temp_audio/{filename}` depuis le répertoire de session ; vérifiez que le backend a bien été lancé depuis `backend/`.
5. **Problème de CORS ou d'URL** : le backend autorise toutes les origines ; vérifiez `VUE_APP_API_URL` et rebuild du frontend après modification.

### Logs
```bash
# Logs Docker Compose
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f backend
docker-compose logs -f frontend
```

## 📊 Formats supportés

### Audio
- MP3, WAV, AAC, OGG, FLAC, M4A

### Vidéo
- MP4, MOV, 3GP, MKV

Tout fichier est converti en WAV mono 16 kHz avant traitement.

## ⚡ Performances

### Recommandations système
- **CPU** : 8+ cœurs recommandés
- **RAM** : 16 GB minimum, 32 GB recommandé
- **GPU** : NVIDIA avec CUDA — 5 à 10× plus rapide que le CPU

### VRAM
| VRAM | Comportement attendu |
|------|----------------------|
| ≥ 16 GB | Whisper large-v3-turbo + pyannote + Chocolatine simultanément |
| 8–12 GB | Whisper + pyannote confortables ; Chocolatine peut échouer à se charger |
| ≤ 6 GB | Chargement possible mais tendu ; privilégier le CPU pour Chocolatine |
| CPU seul | Fonctionnel mais lent 🐌 |

## 🤝 Contribution

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Committez vos changes (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**Développé avec ❤️ par medhi**

# TranscriMate 🎙️

TranscriMate est une application de transcription audio/vidéo intelligente qui utilise l'IA pour séparer les voix, transcrire le contenu et permettre d'interagir avec les transcriptions via un chatbot.

## 🚀 Fonctionnalités

- **Transcription audio/vidéo** : Conversion automatique de fichiers audio et vidéo en texte
- **Séparation des locuteurs** : Identification et séparation automatique des différentes voix
- **Traduction** : Traduction automatique vers l'anglais
- **Chatbot IA** : Interaction avec les transcriptions via AKABot (GPT-4 ou Chocolatine)
- **Transcription en temps réel** : Enregistrement et transcription live via microphone
- **Interface moderne** : Interface web responsive avec mode sombre/clair

## 📋 Prérequis

### Backend
- Python 3.10+
- CUDA (recommandé pour les performances GPU)
- ffmpeg installé sur le système

### Frontend  
- Node.js 18.3.0+
- npm

### Clés API requises
- `OPENAI_API_KEY` : Pour GPT-4 et Whisper
- `HuggingFace_API_KEY` : Pour les modèles Hugging Face

## 🛠️ Installation et Configuration

### 1. Cloner le projet
```bash
git clone https://github.com/famibelle/TranscriMate.git
cd TranscriMate
```

### 2. Configuration des variables d'environnement

Créez un fichier `.env` dans le dossier `backend/` :

```bash
# Backend/.env
OPENAI_API_KEY_MCF=votre_clé_openai
HuggingFace_API_KEY=votre_clé_huggingface
SERVER_URL=http://localhost:8000
```

Créez un fichier `.env` dans le dossier `frontend/` :

```bash
# Frontend/.env
VUE_APP_API_URL=http://localhost:8000
VUE_APP_WEBSOCKET_URL=ws://localhost:8000/streaming_audio/
```

## 🚀 Démarrage avec Docker (Recommandé)

### Option 1: Docker Compose (Le plus simple)

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

### Option 2: Docker individuel

#### Backend
```bash
cd backend
docker build -t transcrimate-backend .
docker run -p 8000:8000 --gpus all transcrimate-backend
```

#### Frontend
```bash
cd frontend
docker build -t transcrimate-frontend .
docker run -p 8080:8080 transcrimate-frontend
```

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

# Lancer le serveur
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Le backend sera accessible sur http://localhost:8000

### Frontend

```bash
cd frontend

# Installer les dépendances
npm install

# Lancer en mode développement
npm run serve

# Construire pour la production
npm run build
```

Le frontend sera accessible sur http://localhost:8080

## 📱 Utilisation

### 1. Transcription de fichiers

1. Accédez à l'onglet **Transcription 🎙️**
2. Glissez-déposez un fichier audio/vidéo ou utilisez le bouton "Sélectionner un fichier"
3. Configurez les paramètres :
   - **Transcrire** : Transcription dans la langue originale
   - **Traduire** : Traduction vers l'anglais
4. Attendez le traitement automatique (extraction audio → séparation des voix → transcription)
5. Consultez les résultats par locuteur et copiez la transcription complète

### 2. Chatbot AKABot

1. Accédez à l'onglet **AKABot 🤖**
2. Choisissez le modèle :
   - **OpenAI GPT** : GPT-4 pour des réponses de haute qualité
   - **Chocolatine 🍫🥖** : Modèle français local
3. Posez vos questions sur AKABI ou analysez une transcription

### 3. Transcription en temps réel

1. Accédez à l'onglet **Traduction 🗣️**
2. Configurez le mode (transcription/traduction)
3. Cliquez sur le bouton microphone pour démarrer l'enregistrement
4. Parlez et visualisez la transcription en temps réel

## 🔧 Configuration avancée

### Modèles Whisper disponibles
- `openai/whisper-large-v3-turbo` (recommandé)
- `openai/whisper-large-v3`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `openai/whisper-tiny`

### Paramètres système

Le backend utilise automatiquement :
- GPU CUDA si disponible (recommandé)
- CPU sinon (plus lent)
- Gestion automatique de la mémoire avec déchargement des modèles après inactivité

## 📁 Structure du projet

```
TranscriMate/
├── backend/
│   ├── main.py              # API FastAPI principale  
│   ├── RAG.py               # Système de recherche pour AKABot
│   ├── requirements.txt     # Dépendances Python
│   ├── Dockerfile          # Configuration Docker backend
│   └── Multimedia/
│       └── Use_Cases/      # Base de connaissances AKABI
├── frontend/
│   ├── src/
│   │   ├── App.vue         # Composant principal
│   │   ├── main.js         # Point d'entrée
│   │   └── components/     # Composants Vue.js
│   ├── package.json        # Dépendances Node.js
│   └── Dockerfile         # Configuration Docker frontend
├── docker-compose.yaml    # Orchestration Docker
└── README.md             # Ce fichier
```

## 🛠️ Scripts utiles

### Backend
```bash
# Vérifier l'état des modèles
curl http://localhost:8000/device_type/

# Initialiser les modèles
curl http://localhost:8000/initialize/

# Maintenir les modèles en vie
curl http://localhost:8000/keep_alive/
```

### Développement
```bash
# Lancer les tests backend
cd backend && python -m pytest

# Linter frontend  
cd frontend && npm run lint

# Construire pour la production
cd frontend && npm run build
```

## 🐛 Dépannage

### Problèmes courants

1. **Erreur CUDA** : Vérifiez que CUDA est installé et compatible
2. **Modèles trop lents** : Utilisez un modèle Whisper plus petit (tiny/base/small)
3. **Erreur de mémoire** : Réduisez la taille des fichiers ou utilisez CPU
4. **Problème de CORS** : Vérifiez les URLs dans les fichiers .env

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

## ⚡ Performances

### Recommandations système
- **CPU** : 8+ cœurs recommandés
- **RAM** : 16GB minimum, 32GB recommandé
- **GPU** : NVIDIA avec CUDA pour l'accélération
- **Stockage** : SSD recommandé pour les gros fichiers

### Optimisations
- Utilisez Docker avec `--gpus all` pour l'accélération GPU
- Les modèles se déchargent automatiquement après 10 minutes d'inactivité
- La transcription temps réel optimise automatiquement la qualité vs latence

## 🤝 Contribution

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Committez vos changes (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- OpenAI pour Whisper et GPT-4
- Pyannote pour la diarisation des locuteurs  
- Hugging Face pour les modèles de transformers
- L'équipe AKABI pour les cas d'usage

---

**Développé avec ❤️ par l'équipe AKABI**
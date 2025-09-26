# 📚 **Documentation API TranscriMate**

Une API FastAPI complète pour la transcription audio/vidéo avec diarisation (séparation des locuteurs) et analyse par IA.

## 🌐 **URL de base**
```
http://localhost:8001
```

## 📋 **Vue d'ensemble des endpoints**

| Méthode | Endpoint | Catégorie | Description |
|---------|----------|-----------|-------------|
| `GET` | `/device_type/` | 🔧 Système | Informations détaillées GPU/CPU |
| `GET` | `/gpu_test/` | 🔧 Système | Test de performance GPU |
| `GET` | `/initialize/` | 🔧 Système | Chargement des modèles IA |
| `GET` | `/keep_alive/` | 🔧 Système | Maintien de session |
| `POST` | `/settings/` | ⚙️ Configuration | Mise à jour des paramètres |
| `POST` | `/diarization/` | 🎵 Audio | Séparation des locuteurs uniquement |
| `POST` | `/uploadfile/` | 🎵 Audio | Transcription complète avec streaming |
| `POST` | `/process_audio/` | 🎵 Audio | Traitement audio/vidéo standard |
| `GET` | `/generate_audio_url/{filename}` | 📁 Fichiers | Génération d'URL pour segments |
| `GET` | `/segment_audio/{filename}` | 📁 Fichiers | Téléchargement de segments audio |
| `WebSocket` | `/streaming_audio/` | 🔴 Temps réel | Transcription en temps réel |
| `POST` | `/ask_question/` | 🤖 IA | Questions sur transcription |

---

## 🔧 **Endpoints Système**

### `GET /device_type/`
**Configuration GPU et recommandations système**

**Réponse :**
```json
{
  "device": "cuda:0",
  "cuda_available": true,
  "pytorch_version": "2.8.0+cu126",
  "cuda_version": "12.6",
  "gpu_count": 1,
  "current_gpu": 0,
  "gpu_details": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "memory_total_gb": 8.0,
      "compute_capability": "8.9",
      "multi_processor_count": 24
    }
  ],
  "memory_info": {
    "allocated_mb": 0.0,
    "reserved_mb": 0.0,
    "total_mb": 8192.0,
    "usage_percent": 0.0
  },
  "recommendations": [
    "GPU puissant détecté (≥8GB) - Modèles Whisper large recommandés",
    "Traitement de longs fichiers audio possible",
    "Diarisation en temps réel optimale"
  ],
  "pytorch_cuda_support": "✅ PyTorch CUDA activé"
}
```

---

### `GET /gpu_test/`
**Test de performance GPU avec les modèles**

**Réponse :**
```json
{
  "status": "success",
  "cuda_available": true,
  "gpu_test_time_ms": 12.34,
  "memory_after_test": {
    "allocated_mb": 8.6,
    "reserved_mb": 20.0
  },
  "models_status": {
    "diarization": {
      "loaded": true,
      "device": "cuda",
      "status": "✅ Modèle pyannote sur GPU"
    },
    "whisper": {
      "loaded": true,
      "device": "cuda",
      "model": "openai/whisper-large-v3-turbo",
      "status": "✅ Whisper sur cuda"
    }
  },
  "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "recommendations": [
    "GPU fonctionnel pour les calculs PyTorch",
    "Chargez vos modèles avec /initialize/ pour les tester sur GPU"
  ]
}
```

---

### `GET /initialize/`
**Chargement des modèles IA (Whisper, pyannote, Chocolatine)**

**Réponse :**
```json
{
  "message": "Modèles chargés avec succès"
}
```

⚠️ **Note :** Premier chargement peut prendre plusieurs minutes (téléchargement des modèles)

---

### `GET /keep_alive/`
**Maintien de la session active (évite le déchargement automatique des modèles)**

**Réponse :**
```json
{
  "message": "Timestamp d'activité mis à jour"
}
```

---

## ⚙️ **Configuration**

### `POST /settings/`
**Mise à jour des paramètres de transcription**

**Corps de requête :**
```json
{
  "task": "transcribe",
  "model": "openai/whisper-large-v3-turbo",
  "lang": "auto",
  "chat_model": "chocolatine"
}
```

**Paramètres valides :**
- **task** : `"transcribe"` | `"translate"`
- **model** : 
  - `"openai/whisper-large-v3-turbo"` (recommandé)
  - `"openai/whisper-large-v3"`
  - `"openai/whisper-medium"`
  - `"openai/whisper-base"`
  - `"openai/whisper-small"`
  - `"openai/whisper-tiny"`
- **lang** : `"auto"` | `"fr"` | `"en"`
- **chat_model** : `"chocolatine"` | `"gpt4o-mini"`

**Réponse :**
```json
{
  "message": "Paramètres mis à jour avec succès"
}
```

---

## 🎵 **Endpoints Audio**

### `POST /diarization/`
**Séparation des locuteurs uniquement (sans transcription)**

**Corps de requête :** `multipart/form-data`
- **file** : Fichier audio/vidéo (MP3, WAV, MP4, MOV, etc.)

**Réponse :**
```json
[
  {
    "speaker": "SPEAKER_00",
    "start_time": 0.5,
    "end_time": 3.2
  },
  {
    "speaker": "SPEAKER_01", 
    "start_time": 3.5,
    "end_time": 7.1
  }
]
```

---

### `POST /uploadfile/`
**Transcription complète avec diarisation en streaming**

**Corps de requête :** `multipart/form-data`
- **file** : Fichier audio/vidéo

**Réponse :** `text/plain` (Server-Sent Events)
```
{"extraction_audio_status": "extraction_audio_ongoing", "message": "Extraction audio en cours ..."}
{"status": "diarization_processing", "message": "Séparation des voix en cours ..."}
{"status": "diarization_done", "message": "Séparation des voix terminée."}
{"diarization": [...]}
{"speaker": "SPEAKER_00", "text": {"text": "Bonjour tout le monde"}, "start_time": 0.5, "end_time": 3.2, "audio_url": "http://localhost:8001/segment_audio/segment_123.wav"}
{"speaker": "SPEAKER_01", "text": {"text": "Salut comment ça va ?"}, "start_time": 3.5, "end_time": 7.1, "audio_url": "http://localhost:8001/segment_audio/segment_124.wav"}
```

**Formats supportés :** MP3, WAV, AAC, OGG, FLAC, M4A, MP4, MOV, 3GP, MKV

---

### `POST /process_audio/`
**Traitement audio/vidéo standard (sans streaming)**

**Corps de requête :** `multipart/form-data`
- **file** : Fichier audio/vidéo

**Réponse :**
```json
{
  "transcriptions": [
    {
      "speaker": "SPEAKER_00",
      "text": {
        "text": "Bonjour tout le monde",
        "chunks": [...]
      },
      "start_time": 0.5,
      "end_time": 3.2
    }
  ]
}
```

---

## 📁 **Gestion des fichiers**

### `GET /generate_audio_url/{filename}`
**Génération d'URL pour segment audio**

**Paramètres :**
- **filename** : Nom du fichier segment

**Réponse :**
```json
{
  "url": "http://localhost:8001/segment_audio/segment_123.wav"
}
```

---

### `GET /segment_audio/{filename}`
**Téléchargement direct du segment audio**

**Paramètres :**
- **filename** : Nom du fichier segment

**Réponse :** Fichier audio WAV

---

## 🔴 **Temps réel**

### `WebSocket /streaming_audio/`
**Transcription audio en temps réel**

**Protocole WebSocket :**

**Envoi (Client → Serveur) :**
- Données audio brutes : 16-bit, 16kHz, mono
- Format : chunks de ~1 seconde

**Réception (Serveur → Client) :**
```json
{
  "chunk_duration": 2.5,
  "transcription_live": {
    "text": "Bonjour comment allez-vous ?"
  }
}
```

**Silence détecté :**
```json
{
  "chunk_duration": 0,
  "transcription_live": {
    "text": "...\n"
  }
}
```

---

## 🤖 **IA et Analysis**

### `POST /ask_question/`
**Questions sur une transcription avec IA**

**Corps de requête :**
```json
{
  "question": "Quels sont les points clés de cette réunion ?",
  "transcription": "Transcript complet de la conversation...",
  "chat_model": "chocolatine"
}
```

**Paramètres :**
- **question** : Question à poser
- **transcription** : Texte transcrit à analyser
- **chat_model** : `"chocolatine"` (local) | `"gpt4o-mini"` (OpenAI)

**Réponse :**
```json
{
  "response": "Voici les **points clés** identifiés dans cette réunion:\n\n1. **Objectif principal** : ...\n2. **Décisions prises** : ..."
}
```

**Fonctionnalités spéciales :**
- Réponses en **Markdown**
- RAG (Retrieval-Augmented Generation) si mention "AKABI"
- Support multilingue

---

## 🔐 **Authentification et CORS**

**CORS :** Activé pour tous les domaines (`allow_origins=["*"]`)

**Authentification :** Aucune (API publique)

---

## 📊 **Codes de statut HTTP**

| Code | Signification |
|------|---------------|
| `200` | Succès |
| `400` | Erreur de requête (paramètres invalides) |
| `404` | Fichier non trouvé |
| `422` | Erreur de validation |
| `500` | Erreur serveur interne |

---

## 🚀 **Exemples d'utilisation**

### Transcription complète d'un fichier
```bash
curl -X POST "http://localhost:8001/uploadfile/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mon_audio.mp3"
```

### Configuration des paramètres
```bash
curl -X POST "http://localhost:8001/settings/" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "transcribe",
    "model": "openai/whisper-medium",
    "lang": "fr"
  }'
```

### Vérification GPU
```bash
curl "http://localhost:8001/device_type/"
```

### Question IA
```bash
curl -X POST "http://localhost:8001/ask_question/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Résume cette conversation",
    "transcription": "Bonjour...",
    "chat_model": "chocolatine"
  }'
```

---

## ⚡ **Optimisations GPU**

**Configuration recommandée :**
- GPU NVIDIA avec ≥4GB VRAM
- PyTorch avec CUDA
- Modèles Whisper : medium/large pour GPU puissant
- FP16 automatiquement activé sur GPU compatible

**Performances attendues (RTX 4060) :**
- Transcription : ~5-10x plus rapide qu'en CPU
- Diarisation : ~3-5x plus rapide
- Temps réel : Possible avec modèles optimisés

---

## 📝 **Notes techniques**

- **Gestion des fichiers** : Automatique avec nettoyage cross-platform
- **Modèles** : Déchargement automatique après 10 min d'inactivité
- **Formats audio** : Conversion automatique en 16kHz mono
- **Streaming** : Server-Sent Events pour `/uploadfile/`
- **WebSocket** : Temps réel avec détection de silence
- **Sécurité** : Validation des extensions de fichiers
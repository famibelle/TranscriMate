import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Generator, Tuple

from temp_manager import TempFileManager, async_temp_manager_context, async_temp_file_context

import filetype
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from moviepy.editor import VideoFileClip
from openai import OpenAI
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydantic import BaseModel, StrictStr
from pydub import AudioSegment
from RAG import setup_rag_pipeline
# from silero_vad import get_speech_timestamps
# from pysilero_vad import SileroVoiceActivityDetector
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from starlette.websockets import WebSocketDisconnect

# Configuration du logging avec plus de verbosité
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ]
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.info(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
logging.info(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

import gc
import threading
import warnings

from pydub import AudioSegment
from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# Configuration cross-platform des répertoires
TEMP_FOLDER = tempfile.gettempdir()
HF_cache = '/mnt/.cache/' if os.name != 'nt' else os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
Model_dir = '/mnt/Models' if os.name != 'nt' else os.path.join(os.path.expanduser('~'), 'Models')

HUGGING_FACE_KEY = os.environ.get("HuggingFace_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_MCF")

server_url = os.getenv("SERVER_URL")

# Configuration cross-platform des répertoires
if os.name != 'nt':  # Unix/Linux
    directories = ["/mnt/Models", "/mnt/logs", "/mnt/.cache"]
    for directory in directories:
        if not os.path.exists(directory):
            os.system(f"sudo mkdir -p {directory}")
    os.system("sudo chmod -R 755 /mnt/Models /mnt/.cache /mnt/logs")
    os.system("sudo chown -R $USER:$USER /mnt/Models /mnt/.cache /mnt/logs")
else:  # Windows
    directories = [HF_cache, Model_dir, os.path.join(os.path.expanduser('~'), 'logs')]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

os.environ['HF_HOME'] = HF_cache
os.environ['TRANSFORMERS_CACHE'] = HF_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logging.info(f"GPU trouvé! on utilise CUDA. Device: {device}")

else:
    logging.info(f"Pas de GPU de disponible... Device: {device}")




# Vérifier le type par défaut de tensor
print(f"Type de donnée par défaut : {torch.get_default_dtype()}")

# Vérifier la précision pour les calculs en FP16
print(f"Disponibilité de la précision FP16 : {torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)}")

# Initialisation de `current_settings` avec des valeurs par défaut
# global current_settings
current_settings = {
    "task": "transcribe",
    "model": "openai/whisper-large-v3-turbo",
    "lang": "auto",
    "chat_model": "chocolatine"
}

full_transcription =  []

# Charger les modèles
# diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization")
def load_core_models():
    """Charge uniquement les modèles de base : Whisper + Diarisation (pas Chocolatine)"""
    global Transcriber_Whisper, Transcriber_Whisper_live, last_activity_timestamp, diarization_model
    import time
    
    # Compter seulement les modèles core (pas Chocolatine)
    core_models_to_load = sum([1 for model in [Transcriber_Whisper, Transcriber_Whisper_live] if model is None])
    current_model = 0
    
    if core_models_to_load > 0:
        logging.info(f"🎤 Initialisation des modèles de base : {core_models_to_load} modèle(s) Whisper...")
        logging.info("⏱️ Temps estimé: 2-5 minutes (modèles STT)")
    
    # Vérifier que diarisation est chargé (normalement chargé au démarrage)
    if diarization_model is None:
        logging.error("❌ Modèle de diarisation non disponible")
        raise RuntimeError("Le modèle de diarisation doit être chargé au démarrage")

    if Transcriber_Whisper is None:
        current_model += 1
        logging.info(f"📦 [{current_model}/{core_models_to_load}] Démarrage du chargement de Whisper Principal...")
        logging.info(f"🔄 Modèle: {model_settings} - Transcription de haute qualité")
        logging.info("💾 Taille estimée: ~1-3 GB - Temps estimé: 1-3 minutes")
        
        start_time = time.time()
        try:
            Transcriber_Whisper = pipeline(
                    "automatic-speech-recognition",
                    model = model_settings,
                    chunk_length_s=30,
                    device=device
                )
            load_time = time.time() - start_time
            logging.info(f"✅ [{current_model}/{core_models_to_load}] Whisper Principal chargé en {load_time:.1f}s")
            logging.info(f"🎯 GPU utilisé: {device}")
        except Exception as e:
            logging.error(f"❌ Erreur lors du chargement de Whisper Principal: {str(e)}")
            raise

    if Transcriber_Whisper_live is None:
        current_model += 1
        logging.info(f"📦 [{current_model}/{core_models_to_load}] Démarrage du chargement de Whisper Live...")
        logging.info("🔄 Modèle: openai/whisper-medium - Transcription en temps réel")
        logging.info("💾 Taille estimée: ~1.5 GB - Temps estimé: 1-2 minutes")
        
        start_time = time.time()
        try:
            Transcriber_Whisper_live = pipeline(
                    "automatic-speech-recognition",
                    model = "openai/whisper-medium",
                    chunk_length_s=30,
                    device=device
                )
            load_time = time.time() - start_time
            logging.info(f"✅ [{current_model}/{core_models_to_load}] Whisper Live chargé en {load_time:.1f}s")
            logging.info(f"🎯 GPU utilisé: {device}")
        except Exception as e:
            logging.error(f"❌ Erreur lors du chargement de Whisper Live: {str(e)}")
            raise

    # Résumé final du chargement des modèles core
    if core_models_to_load > 0:
        logging.info("🎉 MODÈLES DE BASE CHARGÉS AVEC SUCCÈS!")
        logging.info("📊 Résumé des modèles STT disponibles:")
        if Transcriber_Whisper: logging.info(f"  ✅ Whisper Principal ({model_settings}): Transcription haute qualité")
        if Transcriber_Whisper_live: logging.info("  ✅ Whisper Live (medium): Transcription temps réel")
        if diarization_model: logging.info("  ✅ Diarisation: Séparation des locuteurs")
        logging.info(f"🚀 Système STT prêt sur {device}!")

    # Optimisation GPU pour les modèles Whisper
    if device.type == "cuda":
        logging.info("⚡ Optimisation GPU: Conversion des modèles Whisper en FP16...")
        if Transcriber_Whisper:
            model_post = Transcriber_Whisper.model
            model_post = model_post.half()
            Transcriber_Whisper.model = model_post
            logging.info("  ✅ Whisper Principal optimisé (FP16)")

        if Transcriber_Whisper_live:
            model_live = Transcriber_Whisper_live.model
            model_live = model_live.half()
            Transcriber_Whisper_live.model = model_live
            logging.info("  ✅ Whisper Live optimisé (FP16)")
        
        logging.info("⚡ Optimisation GPU terminée - Vitesse accrue!")

    # Mettre à jour le timestamp d'activité
    last_activity_timestamp = time.time()

def load_ai_model():
    """Charge uniquement le modèle IA : Chocolatine"""
    global Chocolatine, last_activity_timestamp
    import time
    
    if Chocolatine is None:
        logging.info("🤖 Démarrage du chargement de Chocolatine pour l'IA...")
        logging.info("🔄 Chocolatine (14B paramètres) - Téléchargement depuis Hugging Face...")
        logging.info("💾 Taille estimée: ~8-12 GB - Temps estimé: 3-8 minutes")
        
        start_time = time.time()
        try:
            Chocolatine = pipeline(
                "text-generation", 
                model="jpacifico/Chocolatine-2-14B-Instruct-v2.0",
                device=device
            )
            load_time = time.time() - start_time
            logging.info(f"✅ Chocolatine chargé en {load_time:.1f}s")
            logging.info(f"🎯 GPU utilisé: {device}")
            logging.info("🤖 IA prête pour l'analyse de transcriptions!")
        except Exception as e:
            logging.error(f"❌ Erreur lors du chargement de Chocolatine: {str(e)}")
            raise
    else:
        logging.info("✅ Chocolatine déjà chargé")
        
    # Mettre à jour le timestamp d'activité
    last_activity_timestamp = time.time()

def load_pipeline_diarization(model):
    pipeline_diarization = Pipeline.from_pretrained(
        model,
        cache_dir= HF_cache,
        use_auth_token = HUGGING_FACE_KEY,
    )

    if torch.cuda.is_available():
        pipeline_diarization.to(torch.device("cuda"))

    logging.info(f"Pipeline Diarization déplacée sur {device}")

    return pipeline_diarization

diarization_model = load_pipeline_diarization("pyannote/speaker-diarization-3.1")
# Charger le modèle Chocolatine


model_selected  = [
        "openai/whisper-large-v3-turbo", #Hot
        "openai/whisper-large-v3",
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-base",
        "openai/whisper-large",
    ]

# whisper_task_transcribe = "transcribe" 
# generate_kwargs={"language": "french"}), 
# generate_kwargs={"task": "translate"}),
# generate_kwargs={"language": "french", "task": "translate"})

model_settings = current_settings.get("model", "openai/whisper-large-v3-turbo")  # Valeur par défaut si non définie

# Initialisation des modèles à None
Transcriber_Whisper = None
Transcriber_Whisper_live = None
last_activity_timestamp = None
Chocolatine = None
timeout_seconds = 600  # Timeout en secondes de 10 minutes d'inactivité

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code de démarrage
    def monitor_inactivity():
        global last_activity_timestamp
        while True:
            if last_activity_timestamp and (time.time() - last_activity_timestamp > timeout_seconds):
                print("Inactivité détectée. Déchargement des modèles...")
                unload_models()
                last_activity_timestamp = None
            time.sleep(10)  # Vérifie toutes les 10 secondes

    # Démarrer le thread de surveillance d'inactivité
    threading.Thread(target=monitor_inactivity, daemon=True).start()

    yield

    # Code d'arrêt
    unload_models()

# Créer l'application FastAPI avec le gestionnaire de contexte
app = FastAPI(
    title="TranscriMate API",
    description="""
    🎵 **API complète de transcription audio/vidéo avec IA**
    
    TranscriMate offre **3 modes de transcription** adaptés à différents besoins :
    
    ## 🚀 **3 Modes Disponibles**
    
    ### **1️⃣ Mode Simple API** (`/transcribe_simple/`)
    * **Usage :** Intégrations tierces, développeurs, applications externes
    * **Fonctionnalités :** Diarisation + Transcription complète
    * **Interface :** Swagger uniquement (pas d'UI frontend)
    * **Retour :** JSON structuré avec métadonnées complètes
    
    ### **2️⃣ Mode Streaming** (`/transcribe_streaming/`)  
    * **Usage :** Interface utilisateur avec feedback temps réel
    * **Fonctionnalités :** Diarisation + Transcription avec affichage progressif
    * **Interface :** Frontend Vue.js avec Server-Sent Events
    * **Retour :** Segments progressifs par locuteur
    
    ### **3️⃣ Mode Live** (`/live_transcription/`)
    * **Usage :** Transcription microphone en temps réel
    * **Fonctionnalités :** Transcription continue (sans diarisation)
    * **Interface :** WebSocket bidirectionnel
    * **Retour :** Flux audio → texte instantané
    
    ## 🛠️ **Technologies**
    
    * **🎯 Diarisation** - pyannote.audio pour séparation des locuteurs
    * **📝 Transcription** - Whisper (OpenAI) haute qualité & temps réel
    * **🤖 IA** - Chocolatine/GPT pour analyse des transcriptions
    * **⚡ GPU** - PyTorch + CUDA pour performances optimales
    * **🌐 Cross-platform** - Support Windows/Linux
    """,
    version="1.0.0",
    contact={
        "name": "TranscriMate Support",
        "email": "medhi@transcrimate.ai",
    },
    license_info={
        "name": "MIT License",
    },
    lifespan=lifespan
)

def load_models():
    """Charge TOUS les modèles : Whisper + Diarisation + Chocolatine"""
    global Transcriber_Whisper, Transcriber_Whisper_live, last_activity_timestamp, Chocolatine
    import time
    
    total_models = sum([1 for model in [Chocolatine, Transcriber_Whisper, Transcriber_Whisper_live] if model is None])
    
    if total_models > 0:
        logging.info(f"🚀 Initialisation COMPLÈTE de {total_models} modèle(s)...")
        logging.info("⏱️ Temps estimé total: 5-15 minutes (premier chargement)")

    # Charger les modèles de base (STT)
    logging.info("🎤 === CHARGEMENT DES MODÈLES STT ===")
    load_core_models()
    
    # Charger le modèle IA
    logging.info("🤖 === CHARGEMENT DU MODÈLE IA ===")
    load_ai_model()

    # Résumé final complet
    logging.info("🎉 === TOUS LES MODÈLES CHARGÉS AVEC SUCCÈS ===")
    logging.info("📊 Résumé complet des modèles disponibles:")
    if Chocolatine: logging.info("  ✅ Chocolatine: Génération de texte IA")
    if Transcriber_Whisper: logging.info(f"  ✅ Whisper Principal ({model_settings}): Transcription haute qualité")
    if Transcriber_Whisper_live: logging.info("  ✅ Whisper Live (medium): Transcription temps réel")
    if diarization_model: logging.info("  ✅ Diarisation: Séparation des locuteurs")
    logging.info(f"🚀 Système COMPLET prêt sur {device} - Toutes les fonctionnalités disponibles!")

    # Mettre à jour le timestamp d'activité
    last_activity_timestamp = time.time()



    # Si un GPU est disponible, convertir le modèle Whisper en FP16
    if device.type == "cuda":
        logging.info("⚡ Optimisation GPU: Conversion des modèles Whisper en FP16...")
        if Transcriber_Whisper:
            model_post = Transcriber_Whisper.model
            model_post = model_post.half()  # Convertir en FP16
            Transcriber_Whisper.model = model_post  # Réassigner le modèle à la pipeline
            logging.info("  ✅ Whisper Principal optimisé (FP16)")

        if Transcriber_Whisper_live:
            model_live = Transcriber_Whisper_live.model
            model_live = model_live.half()  # Convertir en FP16
            Transcriber_Whisper_live.model = model_live  # Réassigner le modèle à la pipeline
            logging.info("  ✅ Whisper Live optimisé (FP16)")
        
        logging.info("⚡ Optimisation GPU terminée - Vitesse accrue!")

    # Mettre à jour le timestamp d'activité
    last_activity_timestamp = time.time()

# Vérifie la configuration GPU complète
@app.get(
    "/device_type/", 
    tags=["🔧 Système"],
    summary="Configuration GPU détaillée",
    description="Retourne des informations complètes sur la configuration GPU/CPU et les recommandations d'optimisation"
)
async def device_type():
    """
    Endpoint amélioré pour vérifier la configuration GPU complète
    Retourne des informations détaillées sur le GPU et PyTorch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gpu_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": None,
        "gpu_count": 0,
        "current_gpu": None,
        "gpu_details": [],
        "memory_info": {},
        "recommendations": []
    }
    
    if torch.cuda.is_available():
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["gpu_count"] = torch.cuda.device_count()
        gpu_info["current_gpu"] = torch.cuda.current_device()
        
        # Informations détaillées pour chaque GPU
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_detail = {
                "id": i,
                "name": gpu_props.name,
                "memory_total_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count
            }
            gpu_info["gpu_details"].append(gpu_detail)
        
        # Informations mémoire GPU actuelle
        current_gpu = torch.cuda.current_device()
        allocated_mb = torch.cuda.memory_allocated(current_gpu) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(current_gpu) / (1024**2)
        total_mb = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**2)
        
        gpu_info["memory_info"] = {
            "allocated_mb": round(allocated_mb, 2),
            "reserved_mb": round(reserved_mb, 2),
            "total_mb": round(total_mb, 2),
            "usage_percent": round((allocated_mb / total_mb) * 100, 2)
        }
        
        # Recommandations basées sur la configuration
        main_gpu = gpu_info["gpu_details"][0]
        if main_gpu["memory_total_gb"] >= 8:
            gpu_info["recommendations"] = [
                "GPU puissant détecté (≥8GB) - Modèles Whisper large recommandés",
                "Traitement de longs fichiers audio possible",
                "Diarisation en temps réel optimale"
            ]
        elif main_gpu["memory_total_gb"] >= 4:
            gpu_info["recommendations"] = [
                "GPU moyen détecté (4-8GB) - Modèles Whisper medium/base recommandés",
                "Attention aux très longs fichiers audio"
            ]
        else:
            gpu_info["recommendations"] = [
                "GPU léger détecté (<4GB) - Modèles Whisper small/tiny recommandés",
                "Traitement par segments recommandé"
            ]
            
        # Vérification si PyTorch utilise bien CUDA
        if "+cu" in torch.__version__:
            gpu_info["pytorch_cuda_support"] = "✅ PyTorch CUDA activé"
        else:
            gpu_info["pytorch_cuda_support"] = "⚠️ PyTorch CPU uniquement - installer version CUDA"
            
    else:
        gpu_info["recommendations"] = [
            "Aucun GPU CUDA détecté",
            "Modèles Whisper tiny/base pour de bonnes performances CPU",
            "Traitement par petits segments recommandé"
        ]
        gpu_info["pytorch_cuda_support"] = "❌ CUDA non disponible"
    
    return gpu_info


@app.get(
    "/gpu_test/", 
    tags=["🔧 Système"],
    summary="Test de performance GPU",
    description="Teste les performances GPU et vérifie que les modèles IA utilisent bien le GPU"
)
async def gpu_test():
    """
    Endpoint pour tester les performances GPU avec les modèles TranscriMate
    """
    if not torch.cuda.is_available():
        return {
            "status": "error", 
            "message": "GPU non disponible",
            "cuda_available": False
        }
    
    try:
        # Test basique GPU
        device = torch.device("cuda")
        
        # Test de multiplication matricielle
        start_time = time.time()
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        gpu_test_time = time.time() - start_time
        
        # Informations mémoire après test
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        
        # Test des modèles si chargés
        models_status = {}
        
        if 'diarization_model' in globals() and diarization_model is not None:
            try:
                # Vérifier sur quel device est le modèle de diarisation
                # Le modèle pyannote utilise automatiquement CUDA s'il est disponible
                models_status["diarization"] = {
                    "loaded": True,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "status": "✅ Modèle pyannote sur GPU"
                }
            except Exception as e:
                models_status["diarization"] = {
                    "loaded": True,
                    "error": str(e)
                }
        else:
            models_status["diarization"] = {"loaded": False}
            
        if Transcriber_Whisper is not None:
            try:
                # Vérifier le device du modèle Whisper
                model_device = str(Transcriber_Whisper.device)
                models_status["whisper"] = {
                    "loaded": True,
                    "device": model_device,
                    "model": model_settings,
                    "status": f"✅ Whisper sur {model_device}"
                }
            except Exception as e:
                models_status["whisper"] = {
                    "loaded": True,
                    "error": str(e)
                }
        else:
            models_status["whisper"] = {"loaded": False}
            
        # Nettoyer la mémoire de test
        del a, b, c
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "cuda_available": True,
            "gpu_test_time_ms": round(gpu_test_time * 1000, 2),
            "memory_after_test": {
                "allocated_mb": round(allocated_mb, 2),
                "reserved_mb": round(reserved_mb, 2)
            },
            "models_status": models_status,
            "gpu_name": torch.cuda.get_device_name(),
            "recommendations": [
                "GPU fonctionnel pour les calculs PyTorch",
                "Chargez vos modèles avec /initialize/ pour les tester sur GPU",
                "Utilisez /device_type/ pour plus de détails"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors du test GPU: {str(e)}",
            "cuda_available": torch.cuda.is_available()
        }


# Charger les modèles à la demande via la route /initialize/
@app.get(
    "/initialize/", 
    tags=["🔧 Système"],
    summary="Chargement des modèles IA",
    description="Charge tous les modèles IA (Whisper, pyannote, Chocolatine) en mémoire GPU/CPU"
)
async def initialize_models():
    import time
    
    logging.info("🚀 === INITIALISATION DES MODÈLES DEMANDÉE ===")
    logging.info(f"🕐 Heure de début: {time.strftime('%H:%M:%S')}")
    logging.info(f"💻 Appareil utilisé: {device}")
    
    # Vérifier l'état actuel des modèles
    models_already_loaded = sum([1 for model in [Chocolatine, Transcriber_Whisper, Transcriber_Whisper_live] if model is not None])
    models_to_load = sum([1 for model in [Chocolatine, Transcriber_Whisper, Transcriber_Whisper_live] if model is None])
    
    if models_to_load == 0:
        logging.info("✅ Tous les modèles sont déjà chargés!")
        return {"message": "Tous les modèles sont déjà chargés", "status": "already_loaded"}
    
    logging.info(f"📊 État: {models_already_loaded} modèle(s) déjà chargé(s), {models_to_load} à charger")
    
    start_total = time.time()
    try:
        load_models()
        total_time = time.time() - start_total
        
        logging.info("🎉 === INITIALISATION TERMINÉE ===")
        logging.info(f"⏱️ Temps total: {total_time:.1f} secondes")
        logging.info(f"🕐 Heure de fin: {time.strftime('%H:%M:%S')}")
        
        return {
            "message": "Modèles chargés avec succès", 
            "status": "loaded",
            "total_time_seconds": round(total_time, 1),
            "device": str(device)
        }
        
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
        return {"message": f"Erreur lors du chargement: {str(e)}", "status": "error"}

@app.get(
    "/model-status/", 
    tags=["🔧 Système"],
    summary="État des modèles",
    description="Affiche l'état actuel de tous les modèles IA et leur statut de chargement"
)
async def get_models_status():
    models_status = {
        "device": str(device),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "models": {
            "chocolatine": {
                "loaded": Chocolatine is not None,
                "type": "text-generation",
                "model_name": "jpacifico/Chocolatine-2-14B-Instruct-v2.0" if Chocolatine else None
            },
            "whisper_principal": {
                "loaded": Transcriber_Whisper is not None,
                "type": "speech-recognition",
                "model_name": model_settings if Transcriber_Whisper else None
            },
            "whisper_live": {
                "loaded": Transcriber_Whisper_live is not None,
                "type": "speech-recognition-live", 
                "model_name": "openai/whisper-medium" if Transcriber_Whisper_live else None
            },
            "diarization": {
                "loaded": diarization_model is not None,
                "type": "speaker-diarization",
                "model_name": "pyannote/speaker-diarization-3.1" if diarization_model else None
            }
        }
    }
    
    total_loaded = sum([1 for model in models_status["models"].values() if model["loaded"]])
    models_status["summary"] = {
        "total_models": 4,
        "loaded_models": total_loaded,
        "all_loaded": total_loaded == 4,
        "percentage": round((total_loaded / 4) * 100, 1)
    }
    
    logging.info(f"📊 État des modèles demandé - {total_loaded}/4 chargés ({models_status['summary']['percentage']}%)")
    
    return models_status

@app.get(
    "/keep_alive/", 
    tags=["🔧 Système"],
    summary="Maintien de session",
    description="Met à jour le timestamp d'activité pour éviter le déchargement automatique des modèles"
)
async def keep_alive():
    global last_activity_timestamp, Transcriber_Whisper, Transcriber_Whisper_live

    # Charger le modèle uniquement s'il n'est pas déjà chargé
    if Transcriber_Whisper is None or Transcriber_Whisper_live is None:
        load_models()

    # Mettre à jour le timestamp d'activité chaque fois que le frontend fait un ping
    last_activity_timestamp = time.time()
    return {"message": "Timestamp d'activité mis à jour"}


# Décharger les modèles pour économiser la mémoire
def unload_models():
    global Transcriber_Whisper, Transcriber_Whisper_live
    if Transcriber_Whisper:
        del Transcriber_Whisper
        Transcriber_Whisper = None
    if Transcriber_Whisper_live:
        del Transcriber_Whisper_live
        Transcriber_Whisper_live = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Modèles déchargés pour économiser la mémoire.")

# Surveiller l'inactivité pour décharger les modèles si nécessaire
def monitor_inactivity():
    global last_activity_timestamp
    while True:
        if last_activity_timestamp and (time.time() - last_activity_timestamp > timeout_seconds):
            print("Inactivité détectée. Déchargement des modèles...")
            unload_models()
            last_activity_timestamp = None
        time.sleep(10)  # Vérifie toutes les 10 secondes


generate_kwargs_live = {
    "max_new_tokens": 224,  # Limiter la taille pour accélérer les prédictions en streaming.
    "num_beams": 1,  # Décodage rapide (greedy decoding).
    "condition_on_prev_tokens": False,  # Désactiver le contexte entre les segments pour réduire la latence.
    "compression_ratio_threshold": 1.35,  # Standard pour filtrer les segments improbables.
    "temperature": 0.0,  # Préférer des transcriptions conservatrices en temps réel.
    "logprob_threshold": -1.0,  # Filtrer les tokens peu probables.
    "no_speech_threshold": 0.6,  # Garder une tolérance moyenne pour les silences.
}

generate_kwargs_aposteriori = {
    "max_new_tokens": 336,  # Autoriser des transcriptions plus longues (max 448)
    "num_beams": 4,  # Beam search pour améliorer la qualité de la transcription.
    "condition_on_prev_tokens": True,  # Maintenir le contexte entre les segments.
    "compression_ratio_threshold": 2.4,  # Tolérer des segments compressés.
    "temperature": 0.4,  # Équilibre entre diversité et précision.
    "logprob_threshold": -1.5,  # Filtrer les tokens improbables.
    "no_speech_threshold": 0.4,  # Accepter plus de segments contenant des silences.
}


app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:8080/"],  # URL de Vue.js
    # allow_origins=[f"{server_url}:8080"],  # URL de Vue.js
    allow_origins=["*"],  # URL de Vue.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/diarization/", 
    tags=["🎵 Audio"],
    summary="Diarisation (séparation des locuteurs)",
    description="Analyse un fichier audio/vidéo pour identifier et séparer les différents locuteurs sans transcription"
)
async def upload_file(file: UploadFile = File(...)):
    # Vérifier et initialiser les modèles si nécessaire
    global diarization_model
    if diarization_model is None:
        logging.info("Initialisation des modèles pour diarization...")
        await load_models()
        
        # Vérifier que l'initialisation a réussi
        if diarization_model is None:
            logging.error("Échec de l'initialisation du modèle de diarisation")
            raise HTTPException(status_code=500, detail="Impossible d'initialiser le modèle de diarisation. Vérifiez les logs serveur.")
    
    async with async_temp_manager_context("diarization") as temp_manager:
        # Lire le fichier uploadé
        file_data = await file.read()
        file_path = temp_manager.create_temp_file(file.filename, file_data)

        full_transcription_text = "\n"

        # Détection de l'extension du fichier (sécurisée)
        file_extension = os.path.splitext(file.filename)[1].lower()
        logging.info(f"Extension détectée {file_extension}.")
        logging.info(f"Fichier {file.filename} sauvegardé avec succès.")
        
        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.info(f"fichier audio détecté: {file_extension}.")
            # Charger le fichier audio avec Pydub
            audio = AudioSegment.from_file(file_path)

        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.info(f"fichier vidéo détecté: {file_extension}.")
            VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file.filename)

        logging.info(f"Conversion du {file.filename} en mono 16kHz.")

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Créer un chemin pour le fichier audio converti
        audio_path = temp_manager.get_temp_path_with_suffix(".wav")
        
        logging.info(f"Sauvegarde de la piste audio dans {audio_path}.")

        audio.export(audio_path, format="wav")

        # Vérification si le fichier existe
        if not os.path.exists(audio_path):
            logging.error(f"Le fichier {audio_path} n'existe pas.")
            raise HTTPException(status_code=404, detail=f"Le fichier {audio_path} n'existe pas.")

        with ProgressHook() as hook:
            logging.debug(f"Diarization démarrée")
            diarization = diarization_model(audio_path, hook=hook)
            logging.debug(f"Diarization terminée {diarization}")

        diarization_json = convert_tracks_to_json(diarization)
        logging.debug(f"Résultat de la diarization {diarization_json}")
        return diarization_json


class Settings(BaseModel):
    task: StrictStr = "transcribe"
    model: StrictStr = "openai/whisper-large-v3-turbo"  # Valeur par défaut
    lang: StrictStr = "auto"  # Valeur par défaut
    chat_model: StrictStr = "chocolatine"

    def __init__(self, **data):
        super().__init__(**data)
        self.validate()

    def validate(self):
        if self.task not in ["transcribe", "translate"]:
            raise ValueError("Task must be 'transcribe' or 'translate'")

        allowed_models = [
            "openai/whisper-large-v3-turbo",
            "openai/whisper-large-v3",
            "openai/whisper-tiny",
            "openai/whisper-small",
            "openai/whisper-medium",
            "openai/whisper-base",
            "openai/whisper-large"
        ]
        if self.model not in allowed_models:
            raise ValueError("Invalid model")

        if self.lang not in ["auto", "fr", "en"]:
            raise ValueError("Language must be 'fr', 'en', or 'auto'")


@app.post(
    "/settings/", 
    tags=["⚙️ Configuration"],
    summary="Mise à jour des paramètres",
    description="Configure les paramètres de transcription : modèle Whisper, langue, tâche (transcription/traduction)"
)
def update_settings(settings: Settings):
    global current_settings
    # Logique de mise à jour des paramètres côté backend
    # Enregistrez les paramètres dans une base de données ou un fichier de configuration, par exemple
    current_settings = settings.model_dump()  # Mettez à jour la variable globale
    logging.info(f"Settings: {current_settings}, task: {current_settings['task']}")

    return {"message": "Paramètres mis à jour avec succès"}


async def process_streaming_audio(file_path: str, file_extension: str, filename: str):
    """Générateur async pour le traitement streaming avec Server-Sent Events"""
    logging.info("🚀 === DÉBUT DU STREAMING process_streaming_audio() ===")
    start_total = time.time()
    
    # Envoyer le statut de début d'extraction
    extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_ongoing', 'message': 'Extraction audio en cours ...'})
    yield f"{extraction_status}\n"
    logging.info(f"📤 Message envoyé au frontend: {extraction_status}")

    # Faire l'extraction audio dans le streaming pour un feedback temps réel
    audio_path = None
    try:
        start_extraction = time.time()
        
        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.info(f"🎵 Fichier audio détecté: {file_extension}")

            logging.debug("🔄 Chargement du fichier audio avec pydub...")
            start_load = time.time()
            audio = AudioSegment.from_file(file_path)
            end_load = time.time()
            logging.info(f"🎵 Audio chargé en {end_load - start_load:.2f}s - Durée: {len(audio)/1000:.2f}s, Channels: {audio.channels}, Sample Rate: {audio.frame_rate}Hz")
            
            logging.debug(f"🔄 Conversion du {filename} en mono 16kHz...")
            start_convert = time.time()
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            end_convert = time.time()
            logging.info(f"🔄 Conversion terminée en {end_convert - start_convert:.2f}s")
            
            # Créer un chemin pour le fichier audio converti
            import tempfile
            audio_path = tempfile.mktemp(suffix=".wav")
            logging.info(f"💾 Sauvegarde de la piste audio dans {audio_path}")
            start_export = time.time()
            audio.export(audio_path, format="wav")
            end_export = time.time()
            logging.info(f"💾 Exportation terminée en {end_export - start_export:.2f}s")

        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.info(f"🎬 Fichier vidéo détecté: {file_extension}")
            
            logging.debug("🔄 Chargement de la vidéo avec MoviePy...")
            start_video_load = time.time()
            video_clip = VideoFileClip(file_path)
            end_video_load = time.time()
            logging.info(f"🎬 Vidéo chargée en {end_video_load - start_video_load:.2f}s - Durée: {video_clip.duration:.2f}s")
            
            logging.debug("🔄 Extraction audio de la vidéo...")
            start_audio_extract = time.time()
            
            # Utiliser le type détecté ou l'extension du fichier
            file_type = filetype.guess(file_path)
            format_to_use = file_type.extension if file_type else file_extension[1:]  # Enlever le point
            
            audio = AudioSegment.from_file(file_path, format=format_to_use)
            end_audio_extract = time.time()
            logging.info(f"🎵 Audio extrait en {end_audio_extract - start_audio_extract:.2f}s - Durée: {len(audio)/1000:.2f}s")

            logging.debug(f"🔄 Conversion du {filename} en mono 16kHz...")
            start_convert_video = time.time()
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            end_convert_video = time.time()
            logging.info(f"🔄 Conversion vidéo terminée en {end_convert_video - start_convert_video:.2f}s")
            
            # Créer un chemin pour le fichier audio converti (utiliser tempfile pour éviter le race condition)
            import tempfile
            audio_path = tempfile.mktemp(suffix=".wav")
            logging.info(f"💾 Sauvegarde de la piste audio dans {audio_path}")
            start_export_video = time.time()
            audio.export(audio_path, format="wav")
            end_export_video = time.time()
            logging.info(f"💾 Exportation vidéo terminée en {end_export_video - start_export_video:.2f}s")
            
            # Libérer la mémoire
            video_clip.close()
            logging.debug("🗑️ Ressources vidéo libérées")
            
        else:
            # Format de fichier non supporté
            logging.error(f"❌ Format de fichier non supporté: {file_extension}")
            logging.info("📋 Formats supportés: .mp3, .wav, .aac, .ogg, .flac, .m4a, .mp4, .mov, .3gp, .mkv")
            error_msg = json.dumps({'error': 'unsupported_format', 'message': f'Format de fichier non supporté: {file_extension}'})
            yield f"{error_msg}\n"
            logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
            return

        end_extraction = time.time()
        total_extraction_time = end_extraction - start_extraction
        logging.info(f"⏱️ Extraction audio totale terminée en {total_extraction_time:.2f}s")

        # Vérification si le fichier existe
        if not os.path.exists(audio_path):
            logging.error(f"❌ Le fichier audio converti n'existe pas: {audio_path}")
            error_msg = json.dumps({'error': 'file_not_found', 'message': f'Le fichier {audio_path} n\'existe pas.'})
            yield f"{error_msg}\n"
            logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
            return
        
        # Vérification supplémentaire de l'existence du fichier original
        if not os.path.exists(file_path):
            logging.error(f"❌ Le fichier original n'existe plus: {file_path}")
            error_msg = json.dumps({'error': 'original_file_not_found', 'message': f'Le fichier original {file_path} n\'existe plus.'})
            yield f"{error_msg}\n"
            logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
            return

        # Vérifier la taille du fichier créé
        file_size = os.path.getsize(audio_path)
        logging.info(f"✅ Fichier audio converti créé avec succès - Taille: {file_size} bytes")

        # Envoyer le statut de fin d'extraction
        extraction_done = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})
        yield f"{extraction_done}\n"
        logging.info(f"📤 ✅ Message de fin d'extraction envoyé: {extraction_done}")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"❌ ERREUR lors de l'extraction audio:")
        logging.error(f"❌ Type d'erreur: {type(e).__name__}")
        logging.error(f"❌ Message: {str(e)}")
        logging.error(f"❌ Stack trace complet:\n{error_details}")
        
        error_msg = json.dumps({
            'error': 'extraction_failed', 
            'message': f'Erreur lors de l\'extraction audio: {str(e)}',
            'error_type': type(e).__name__
        })
        yield f"{error_msg}\n"
        logging.debug(f"📤 Message d'erreur détaillé envoyé: {error_msg}")
        return

    # Étape 1 : Diarisation
    logging.info(f"🎯 === DÉBUT DE LA DIARISATION ===")
    logging.info(f"🎯 Fichier audio à traiter: {audio_path}")
    
    # Vérification finale avant diarisation
    if not os.path.exists(audio_path):
        logging.error(f"❌ CRITIQUE: Fichier audio inexistant juste avant diarisation: {audio_path}")
        error_msg = json.dumps({
            'error': 'audio_file_missing_before_diarization', 
            'message': f'Le fichier audio {audio_path} n\'existe plus avant la diarisation.',
            'error_type': 'FileNotFoundError'
        })
        yield f"{error_msg}\n"
        return
        
    logging.info(f"🎯 Taille du fichier: {os.path.getsize(audio_path)} bytes")
    
    start_diarization_total = time.time()

    # Envoi du statut "en cours"
    start_diarization = json.dumps({'status': 'diarization_processing', 'message': 'Séparation des voix en cours, patience est mère de vertu ...'})
    yield f"{start_diarization}\n"
    await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
    logging.info(start_diarization)

    logging.debug(f"Diarization démarrée pour le fichier {audio_path}")

    try:
        with ProgressHook() as hook:
            diarization = diarization_model(audio_path, hook=hook)
        # diarization = diarization_model(audio_path)
    except Exception as e:
        logging.error(f"Erreur pendant la diarisation : {str(e)}")

    # Envoi final du statut pour indiquer la fin
    end_diarization = json.dumps({'status': 'diarization_done', 'message': 'Séparation des voix terminée.'})
    yield f"{end_diarization}\n"

    logging.debug(f"Diarization terminée {diarization}")

    try:
        diarization_json = convert_tracks_to_json(diarization)
        logging.info(f"Taille des données de diarisation en JSON : {len(json.dumps(diarization_json))} octets")
    except Exception as e:
        logging.error(f"Erreur pendant la conversion de la diarisation en JSON : {str(e)}")
        yield json.dumps({"status": "error", "message": f"Erreur pendant la conversion en JSON : {str(e)}"}) + "\n"
        return

    logging.debug(f"Résultat de la diarization {diarization_json}")

    await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
    logging.info(end_diarization)

    diarization_json = convert_tracks_to_json(diarization)

    # Envoyer la diarisation complète d'abord
    logging.info(f"{json.dumps({'diarization': diarization_json})}")
    yield f"{json.dumps({'diarization': diarization_json})}\n"
    await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse

    # Exporter les segments pour chaque locuteur
    total_chunks = len(list(diarization.itertracks(yield_label=True))) 
    logging.info(f"total_turns: {total_chunks}")
    
    turn_number = 0
    full_transcription = []
    # for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), total=total_turns, desc="Processing turns"):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_number += 1
        logging.info(f"Tour {turn_number}/{total_chunks}")

        # Étape 2 : Transcription pour chaque segment
        start_ms = int(turn.start * 1000)  # Convertir de secondes en millisecondes
        end_ms = int(turn.end * 1000)

        # Extraire le segment audio correspondant au speaker
        segment_audio = audio[start_ms:end_ms]

        # Sauvegarder le segment temporairement pour Whisper
        segment_path = tempfile.mktemp(suffix=".wav")
        segment_audio.export(segment_path, format="wav")

        logging.info(f"----> Transcription démarée avec le model et la task <----")

        task = current_settings.get("task", "transcribe")
        if task != "transcribe":
            generate_kwargs = {"language": "english"} 
        else:
            generate_kwargs = {} 

        # Transcrire ce segment avec Whisper
        transcription = Transcriber_Whisper(
            segment_path,
            return_timestamps=True,
            generate_kwargs=generate_kwargs
        )

        # Supprimer le fichier de segment une fois transcrit
        # os.remove(segment_path)

        segment = {
            "speaker": speaker,
            "text": transcription,
            "start_time": turn.start,
            "end_time": turn.end,
            "audio_url": f"{server_url}/segment_audio/{os.path.basename(segment_path)}"  # URL du fichier audio
        }

        logging.info(f"Transcription du speaker {speaker} du segment de {turn.start} à {turn.end} terminée\n Résultat de la transcription {segment}")
        full_transcription.append(segment)

        yield f"{json.dumps(segment)}\n"  # Envoi du segment de transcription en JSON
        await asyncio.sleep(0)  # Forcer l'envoi de chaque chunk

        logging.info(f"Transcription du speaker {speaker} pour le segment de {turn.start} à {turn.end} terminée")

    # Fin du streaming
    logging.info(f"->> fin de transcription <<")
    logging.info(full_transcription)


@app.post(
    "/transcribe_streaming/", 
    tags=["🎵 Audio"],
    summary="🔄 Interface Progressive - Streaming Temps Réel",
    description="**Mode 2/3** - Traitement complet avec diarisation + transcription et affichage progressif (Server-Sent Events pour frontend Vue.js)"
)
async def upload_file_streaming(file: UploadFile = File(...)):
    logging.info("=== DÉBUT UPLOAD_FILE_STREAMING ===")
    logging.info(f"📁 Fichier reçu: {file.filename}, Type: {file.content_type}, Taille: {file.size if hasattr(file, 'size') else 'inconnue'}")
    
    # Vérifier et initialiser les modèles si nécessaire
    global Transcriber_Whisper, diarization_model
    logging.debug(f"🔍 État des modèles - Transcriber_Whisper: {'✅ Chargé' if Transcriber_Whisper is not None else '❌ Non chargé'}")
    logging.debug(f"🔍 État des modèles - diarization_model: {'✅ Chargé' if diarization_model is not None else '❌ Non chargé'}")
    
    if Transcriber_Whisper is None or diarization_model is None:
        logging.warning("⚠️ Modèles STT non initialisés, démarrage du chargement des modèles de base...")
        start_init = time.time()
        load_core_models()  # Charger seulement les modèles STT (Whisper + diarisation)
        end_init = time.time()
        logging.info(f"⏱️ Modèles STT initialisés en {end_init - start_init:.2f}s")
        
        # Vérifier que l'initialisation a réussi
        if Transcriber_Whisper is None or diarization_model is None:
            logging.error("❌ ÉCHEC de l'initialisation des modèles")
            raise HTTPException(status_code=500, detail="Impossible d'initialiser les modèles. Vérifiez les logs serveur.")
        else:
            logging.info("✅ Modèles initialisés avec succès")
    
    logging.info("📂 Initialisation du gestionnaire de fichiers temporaires...")
    async with async_temp_manager_context("transcribe_streaming") as temp_manager:
        # Lire le fichier uploadé
        logging.debug("📖 Lecture du fichier uploadé...")
        start_read = time.time()
        file_data = await file.read()
        end_read = time.time()
        logging.info(f"📖 Fichier lu en {end_read - start_read:.2f}s - Taille: {len(file_data)} bytes")
        
        # Créer le fichier temporaire
        logging.debug("💾 Création du fichier temporaire...")
        file_path = temp_manager.create_temp_file(file.filename, file_data)
        logging.info(f"💾 Fichier temporaire créé: {file_path}")
        
        # Vérification immédiate de l'existence du fichier
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logging.info(f"✅ Fichier temporaire confirmé - Taille: {file_size} bytes")
        else:
            logging.error(f"❌ ERREUR CRITIQUE: Le fichier temporaire n'a pas été créé: {file_path}")
            raise HTTPException(status_code=500, detail="Impossible de créer le fichier temporaire")
        
        # Détection de l'extension du fichier (sécurisée)
        file_extension = os.path.splitext(file.filename)[1].lower()
        logging.info(f"🔍 Extension détectée: {file_extension}")

        # Le nettoyage est automatique avec async_temp_manager_context
        # Retourner la réponse streaming dans le contexte pour éviter le race condition
        return StreamingResponse(
            process_streaming_audio(file_path, file_extension, file.filename), 
            media_type="application/json"
        )


# Endpoint pour générer les URLs des segments audio
        audio_path = None
        try:
            start_extraction = time.time()
            
            # Si le fichier est un fichier audio (formats courants)
            if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
                logging.info(f"🎵 Fichier audio détecté: {file_extension}")

                logging.debug("🔄 Chargement du fichier audio avec pydub...")
                start_load = time.time()
                audio = AudioSegment.from_file(file_path)
                end_load = time.time()
                logging.info(f"🎵 Audio chargé en {end_load - start_load:.2f}s - Durée: {len(audio)/1000:.2f}s, Channels: {audio.channels}, Sample Rate: {audio.frame_rate}Hz")
                
                logging.debug(f"🔄 Conversion du {file.filename} en mono 16kHz...")
                start_convert = time.time()
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                end_convert = time.time()
                logging.info(f"🔄 Conversion terminée en {end_convert - start_convert:.2f}s")
                    
                    # Créer un chemin pour le fichier audio converti (utiliser tempfile pour éviter le race condition)
                import tempfile
                audio_path = tempfile.mktemp(suffix=".wav")
                logging.info(f"💾 Sauvegarde de la piste audio dans {audio_path}")
                start_export = time.time()
                audio.export(audio_path, format="wav")
                end_export = time.time()
                logging.info(f"💾 Exportation terminée en {end_export - start_export:.2f}s")

            elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
                logging.info(f"🎬 Fichier vidéo détecté: {file_extension}")
                
                logging.debug("🔄 Chargement de la vidéo avec MoviePy...")
                start_video_load = time.time()
                video_clip = VideoFileClip(file_path)
                end_video_load = time.time()
                logging.info(f"🎬 Vidéo chargée en {end_video_load - start_video_load:.2f}s - Durée: {video_clip.duration:.2f}s")
                
                logging.debug("🔄 Extraction audio de la vidéo...")
                start_audio_extract = time.time()
                
                # Utiliser le type détecté ou l'extension du fichier
                format_to_use = file_type.extension if file_type else file_extension[1:]  # Enlever le point
                logging.debug(f"🔄 Format utilisé pour l'extraction: {format_to_use}")
                
                audio = AudioSegment.from_file(file_path, format=format_to_use)
                end_audio_extract = time.time()
                logging.info(f"🎵 Audio extrait en {end_audio_extract - start_audio_extract:.2f}s - Durée: {len(audio)/1000:.2f}s")

                logging.debug(f"🔄 Conversion du {file.filename} en mono 16kHz...")
                start_convert_video = time.time()
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                end_convert_video = time.time()
                logging.info(f"🔄 Conversion vidéo terminée en {end_convert_video - start_convert_video:.2f}s")
                
                # Créer un chemin pour le fichier audio converti (utiliser tempfile pour éviter le race condition)
                import tempfile
                audio_path = tempfile.mktemp(suffix=".wav")
                logging.info(f"💾 Sauvegarde de la piste audio dans {audio_path}")
                start_export_video = time.time()
                audio.export(audio_path, format="wav")
                end_export_video = time.time()
                logging.info(f"💾 Exportation vidéo terminée en {end_export_video - start_export_video:.2f}s")
                
                # Libérer la mémoire
                video_clip.close()
                logging.debug("🗑️ Ressources vidéo libérées")
                
            else:
                # Format de fichier non supporté
                logging.error(f"❌ Format de fichier non supporté: {file_extension}")
                logging.info("📋 Formats supportés: .mp3, .wav, .aac, .ogg, .flac, .m4a, .mp4, .mov, .3gp, .mkv")
                error_msg = json.dumps({'error': 'unsupported_format', 'message': f'Format de fichier non supporté: {file_extension}'})
                yield f"{error_msg}\n"
                logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
                return

            end_extraction = time.time()
            total_extraction_time = end_extraction - start_extraction
            logging.info(f"⏱️ Extraction audio totale terminée en {total_extraction_time:.2f}s")

            # Vérification si le fichier existe
            if not os.path.exists(audio_path):
                logging.error(f"❌ Le fichier audio converti n'existe pas: {audio_path}")
                error_msg = json.dumps({'error': 'file_not_found', 'message': f'Le fichier {audio_path} n\'existe pas.'})
                yield f"{error_msg}\n"
                logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
                return
                
            # Vérification supplémentaire de l'existence du fichier original
            if not os.path.exists(file_path):
                logging.error(f"❌ Le fichier original n'existe plus: {file_path}")
                error_msg = json.dumps({'error': 'original_file_not_found', 'message': f'Le fichier original {file_path} n\'existe plus.'})
                yield f"{error_msg}\n"
                logging.debug(f"📤 Message d'erreur envoyé: {error_msg}")
                return

            # Vérifier la taille du fichier créé
            file_size = os.path.getsize(audio_path)
            logging.info(f"✅ Fichier audio converti créé avec succès - Taille: {file_size} bytes")

            # Envoyer le statut de fin d'extraction
            extraction_done = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})
            yield f"{extraction_done}\n"
            logging.info(f"📤 ✅ Message de fin d'extraction envoyé: {extraction_done}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"❌ ERREUR lors de l'extraction audio:")
            logging.error(f"❌ Type d'erreur: {type(e).__name__}")
            logging.error(f"❌ Message: {str(e)}")
            logging.error(f"❌ Stack trace complet:\n{error_details}")
            
            error_msg = json.dumps({
                'error': 'extraction_failed', 
                'message': f'Erreur lors de l\'extraction audio: {str(e)}',
                'error_type': type(e).__name__
            })
            yield f"{error_msg}\n"
            logging.debug(f"📤 Message d'erreur détaillé envoyé: {error_msg}")
            return

        # Étape 1 : Diarisation
        logging.info(f"🎯 === DÉBUT DE LA DIARISATION ===")
        logging.info(f"🎯 Fichier audio à traiter: {audio_path}")
        
        # Vérification finale avant diarisation
        if not os.path.exists(audio_path):
            logging.error(f"❌ CRITIQUE: Fichier audio inexistant juste avant diarisation: {audio_path}")
            error_msg = json.dumps({
                'error': 'audio_file_missing_before_diarization', 
                'message': f'Le fichier audio {audio_path} n\'existe plus avant la diarisation.',
                'error_type': 'FileNotFoundError'
            })
            yield f"{error_msg}\n"
            return
            
        logging.info(f"🎯 Taille du fichier: {os.path.getsize(audio_path)} bytes")
        
        start_diarization_total = time.time()

        # Envoi du statut "en cours"
        start_diarization = json.dumps({'status': 'diarization_processing', 'message': 'Séparation des voix en cours, patience est mère de vertu ...'})
        yield f"{start_diarization}\n"
        await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
        logging.info(start_diarization)

        logging.debug(f"Diarization démarrée pour le fichier {audio_path}")

        try:
            with ProgressHook() as hook:
                diarization = diarization_model(audio_path, hook=hook)
            # diarization = diarization_model(audio_path)
        except Exception as e:
            logging.error(f"Erreur pendant la diarisation : {str(e)}")

        # Envoi final du statut pour indiquer la fin
        end_diarization = json.dumps({'status': 'diarization_done', 'message': 'Séparation des voix terminée.'})
        yield f"{end_diarization}\n"

        logging.debug(f"Diarization terminée {diarization}")

            # diarization_json = convert_tracks_to_json(diarization)

        try:
            diarization_json = convert_tracks_to_json(diarization)
            logging.info(f"Taille des données de diarisation en JSON : {len(json.dumps(diarization_json))} octets")
        except Exception as e:
            logging.error(f"Erreur pendant la conversion de la diarisation en JSON : {str(e)}")
            yield json.dumps({"status": "error", "message": f"Erreur pendant la conversion en JSON : {str(e)}"}) + "\n"
            return

        logging.debug(f"Résultat de la diarization {diarization_json}")

        await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
        logging.info(end_diarization)

        diarization_json = convert_tracks_to_json(diarization)

        # Envoyer la diarisation complète d'abord
        logging.info(f"{json.dumps({'diarization': diarization_json})}")
        yield f"{json.dumps({'diarization': diarization_json})}\n"
        await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse

        # Exporter les segments pour chaque locuteur
        total_chunks = len(list(diarization.itertracks(yield_label=True))) 
        logging.info(f"total_turns: {total_chunks}")
        
        turn_number = 0
        # for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), total=total_turns, desc="Processing turns"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turn_number += 1
            logging.info(f"Tour {turn_number}/{total_chunks}")

            # Étape 2 : Transcription pour chaque segment
            start_ms = int(turn.start * 1000)  # Convertir de secondes en millisecondes
            end_ms = int(turn.end * 1000)

            # Extraire le segment audio correspondant au speaker
            segment_audio = audio[start_ms:end_ms]

            # Sauvegarder le segment temporairement pour Whisper
            segment_path = temp_manager.get_temp_path_with_suffix(".wav")
            segment_audio.export(segment_path, format="wav")

            logging.info(f"----> Transcription démarée avec le model <{model_settings}> et la task <{task}> <----")

                # generate_kwargs = {
                #     "max_new_tokens": 448,
                #     "num_beams": 1,
                #     "condition_on_prev_tokens": False, # Si activé, Whisper prend en compte les tokens précédemment générés pour conditionner la génération des tokens actuels.
                #     "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                #     "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                #     "logprob_threshold": -1.0,
                #     "no_speech_threshold": 0.6,
                #     "return_timestamps": True,
                # }

            if current_settings['task'] != "transcribe":
                generate_kwargs={
                    "language": "english", 
                } 

            else:
                generate_kwargs={
                } 

            # Transcrire ce segment avec Whisper
            transcription = Transcriber_Whisper(
                segment_path,
                return_timestamps = True,
                generate_kwargs= generate_kwargs |  generate_kwargs_aposteriori
            )

            # Supprimer le fichier de segment une fois transcrit
            # os.remove(segment_path)

            segment = {
                "speaker": speaker,
                "text": transcription,
                "start_time": turn.start,
                "end_time": turn.end,
                "audio_url": f"{server_url}/segment_audio/{os.path.basename(segment_path)}"  # URL du fichier audio
            }

            logging.info(f"Transcription du speaker {speaker} du segment de {turn.start} à {turn.end} terminée\n Résultat de la transcription {segment}")
            full_transcription.append(segment)

            yield f"{json.dumps(segment)}\n"  # Envoi du segment de transcription en JSON
            await asyncio.sleep(0)  # Forcer l'envoi de chaque chunk

            logging.info(f"Transcription du speaker {speaker} pour le segment de {turn.start} à {turn.end} terminée")

            # Fin du streaming
            logging.info(f"->> fin de transcription <<")
            logging.info(full_transcription)

        # Le nettoyage est automatique avec async_temp_manager_context
        # Retourner la réponse streaming dans le contexte pour éviter le race condition
        return StreamingResponse(live_process_audio(), media_type="application/json")


# Endpoint pour générer les URLs des segments audio
@app.get(
    "/generate_audio_url/{filename}", 
    tags=["📁 Fichiers"],
    summary="Génération d'URL audio",
    description="Génère l'URL d'accès pour un segment audio spécifique"
)
def generate_audio_url(filename: str):
    return {"url": f"{server_url}/segment_audio/{filename}"}

# Endpoint pour servir les segments audio
@app.get(
    "/segment_audio/{filename}", 
    tags=["📁 Fichiers"],
    summary="Téléchargement segment audio",
    description="Télécharge directement un segment audio généré lors de la transcription"
)
async def get_segment_audio(filename: str):
    # Utiliser le répertoire temporaire système de manière cross-platform
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "Fichier non trouvé"}

@app.post(
    "/transcribe_simple/", 
    tags=["🎵 Audio"],
    summary="🔧 API Simple - Diarisation + Transcription",
    description="**Mode 1/3** - API pure pour intégrations tierces. Traitement complet avec diarisation et transcription (accessible uniquement via Swagger/API, sans interface utilisateur)"
)
async def transcribe_simple(file: UploadFile = File(...)):
    """
    Mode Simple : API uniquement pour intégrations tierces
    - Diarisation complète (séparation des locuteurs)
    - Transcription haute qualité
    - Retour JSON structuré
    - Aucune interface utilisateur (Swagger seulement)
    """
    logging.info("=== DÉBUT TRANSCRIBE_SIMPLE (API ONLY) ===")
    logging.info(f"📁 Fichier API reçu: {file.filename}, Type: {file.content_type}")
    
    # Vérifier et initialiser les modèles STT si nécessaire
    global Transcriber_Whisper, diarization_model
    if Transcriber_Whisper is None or diarization_model is None:
        logging.info("Initialisation des modèles STT pour transcribe_simple...")
        load_core_models()  # Charger seulement les modèles STT (Whisper + diarisation)
        
        # Vérifier que l'initialisation a réussi
        if Transcriber_Whisper is None or diarization_model is None:
            logging.error("Échec de l'initialisation des modèles STT")
            raise HTTPException(status_code=500, detail="Impossible d'initialiser les modèles de transcription. Vérifiez les logs serveur.")
    
    async with async_temp_manager_context("transcribe_simple") as temp_manager:
        # Lire le fichier uploadé
        file_data = await file.read()
        file_path = temp_manager.create_temp_file(file.filename, file_data)

        # Détection de l'extension du fichier
        file_extension = os.path.splitext(file.filename)[1].lower()
        logging.info(f"🎯 Extension détectée: {file_extension}")

        # Traitement audio/vidéo
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.info(f"🎵 Fichier audio - traitement direct")
            audio = AudioSegment.from_file(file_path)
        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.info(f"🎬 Fichier vidéo - extraction audio")
            VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file_extension)
        else:
            logging.error(f"❌ Format non supporté: {file_extension}")
            raise HTTPException(status_code=400, detail=f"Format de fichier non supporté: {file_extension}")

        # Normalisation audio
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Créer un chemin pour le fichier audio converti
        audio_path = temp_manager.get_temp_path_with_suffix(".wav")
        audio.export(audio_path, format="wav")
        logging.info(f"✅ Audio normalisé sauvegardé: {audio_path}")
        
        # Étape 1 : Diarisation (séparation des locuteurs)
        logging.info("🎯 === DÉBUT DIARISATION ===")
        with ProgressHook() as hook:
            diarization = diarization_model(audio_path, hook=hook)
        logging.info("✅ Diarisation terminée")

        # Étape 2 : Transcription par segment
        logging.info("📝 === DÉBUT TRANSCRIPTION ===")
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Extraire le segment audio correspondant au speaker
            start_ms = int(turn.start * 1000)  # Convertir en millisecondes
            end_ms = int(turn.end * 1000)
            segment_audio = audio[start_ms:end_ms]

            # Sauvegarder le segment temporairement pour Whisper
            segment_path = temp_manager.get_temp_path_with_suffix(".wav")
            segment_audio.export(segment_path, format="wav")

            # Transcrire ce segment avec Whisper
            transcription = Transcriber_Whisper(
                segment_path, 
                return_timestamps="word",
                generate_kwargs={"task": current_settings.get("task", "transcribe")}
            )
            
            # Ajouter le segment transcrit
            segments.append({
                "speaker": speaker,
                "text": transcription,
                "start_time": turn.start,
                "end_time": turn.end
            })
            logging.info(f"✅ Transcrit {speaker}: {turn.start:.1f}s-{turn.end:.1f}s")

        logging.info(f"🎉 API Simple terminé - {len(segments)} segments transcrits")
        
        # Retour structuré pour l'API
        return {
            "mode": "simple_api",
            "status": "completed",
            "file_info": {
                "filename": file.filename,
                "format": file_extension,
                "duration_seconds": len(audio) / 1000.0
            },
            "diarization": {
                "total_speakers": len(set([seg["speaker"] for seg in segments])),
                "total_segments": len(segments)
            },
            "transcriptions": segments,
            "settings_used": {
                "model": current_settings.get("model", "default"),
                "task": current_settings.get("task", "transcribe"),
                "language": current_settings.get("lang", "auto")
            }
        }

@app.post(
    "/transcribe_file/", 
    tags=["🎵 Audio"],
    summary="🔄 Interface Frontend - Traitement Standard", 
    description="**Mode 2/3** - Traitement complet audio/vidéo avec diarisation et transcription pour interface utilisateur (sans streaming temps réel)"
)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Mode Frontend Standard : Pour l'interface utilisateur actuelle
    - Diarisation complète (séparation des locuteurs)
    - Transcription haute qualité
    - Retour optimisé pour le frontend Vue.js
    - Compatible avec l'interface utilisateur existante
    """
    # Vérifier et initialiser les modèles STT si nécessaire
    global Transcriber_Whisper, diarization_model
    if Transcriber_Whisper is None or diarization_model is None:
        logging.info("Initialisation des modèles STT pour transcribe_file...")
        load_core_models()  # Charger seulement les modèles STT (Whisper + diarisation)
        
        # Vérifier que l'initialisation a réussi
        if Transcriber_Whisper is None or diarization_model is None:
            logging.error("Échec de l'initialisation des modèles STT")
            raise HTTPException(status_code=500, detail="Impossible d'initialiser les modèles de transcription. Vérifiez les logs serveur.")
    
    async with async_temp_manager_context("transcribe_file") as temp_manager:
        # Lire le fichier uploadé
        file_data = await file.read()
        file_path = temp_manager.create_temp_file(file.filename, file_data)

        # Détection de l'extension du fichier
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            # Charger le fichier audio avec Pydub
            audio = AudioSegment.from_file(file_path)

        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file_extension)

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Créer un chemin pour le fichier audio converti
        audio_path = temp_manager.get_temp_path_with_suffix(".wav")
        audio.export(audio_path, format="wav")

        logging.info(f"Fichier {file.filename} sauvegardé avec succès.")
        
        # Démarrer la diarisation
        logging.info("Démarrage de la diarisation")

        # Étape 1 : Diarisation
        with ProgressHook() as hook:
            logging.debug(f"Diarization démarrée")
            diarization = diarization_model(audio_path, hook=hook)
            logging.debug(f"Diarization terminée {diarization}")

        segments = []

        # total_turns = len(list(diarization.itertracks(yield_label=True))) 
        # for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), total=total_turns, desc="Processing turns"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Étape 2 : Transcription pour chaque segment
            start_ms = int(turn.start * 1000)  # Convertir de secondes en millisecondes
            end_ms = int(turn.end * 1000)

            # Extraire le segment audio correspondant au speaker
            segment_audio = audio[start_ms:end_ms]

            # Sauvegarder le segment temporairement pour Whisper
            segment_path = temp_manager.get_temp_path_with_suffix(".wav")
            segment_audio.export(segment_path, format="wav")

            # Transcrire ce segment avec Whisper
            transcription = Transcriber_Whisper(
                segment_path, 
                # return_timestamps = True, 
                return_timestamps="word",
                generate_kwargs={"task": "transcribe"}
                )
            # Étape 2 : Transcription pour chaque segment
            logging.info(f"Transcription du speaker {speaker} du segment de {turn.start} à {turn.end} terminée")
            
            segments.append({
                "speaker": speaker,
                "text": transcription,
                "start_time": turn.start,
                "end_time": turn.end
            })
            logging.info("Transcription terminée.")

        return {"transcriptions": segments}
        # Le nettoyage est automatique avec async_temp_manager_context


import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

# Charger le modèle pré-entraîné et le déplacer sur le GPU
model_denoiser = pretrained.dns64().to(device)

# # Charger le modèle Silero VAD
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# get_speech_timestamps = utils['get_speech_timestamps'] if isinstance(utils, dict) else utils[0]
model_vad = load_silero_vad()

initial_maxlen = 1  # Début du buffer à 1
target_maxlen = 5  # Valeur cible définie pour le buffer

buffer = deque(maxlen=initial_maxlen)  # Buffer pour stocker 30 secondes

@app.websocket("/live_transcription/")
async def websocket_live_transcription(websocket: WebSocket):
    global buffer# Ajout de buffer comme variable globale pour éviter l'erreur
    await websocket.accept()
    logging.debug("Client connecté, en attente des données")

    try:
        while True:
            data = await websocket.receive_bytes()

            # Convertir les données reçues en AudioSegment
            audio_segment = AudioSegment(
                data=data,
                sample_width=2,  # 16 bits
                frame_rate=16000,
                channels=1
            )

            # Ajouter le segment audio au buffer
            buffer.append(audio_segment)

            # Augmente dynamiquement la taille de maxlen jusqu'à atteindre target_maxlen
            if len(buffer) == buffer.maxlen and buffer.maxlen < target_maxlen:
                # Augmente maxlen du buffer
                buffer = deque(buffer, maxlen=buffer.maxlen + 1)
                logging.debug(f"buffer.maxlen: {buffer.maxlen}")

            # Calculer la durée du chunk
            chunk_duration = audio_segment.duration_seconds
            logging.debug(f"Chunk duration: {chunk_duration} seconds")
            logging.debug(f"current_settings: {current_settings['task']}")
            print(f"Chunk duration: {chunk_duration} seconds")

            print(f"len(buffer)={len(buffer)}; buffer.maxlen={buffer.maxlen}")

            if len(buffer) == (buffer.maxlen):
                combined_audio = sum(buffer)  # Combine tous les AudioSegments en un seul

                # Créer un fichier temporaire pour l'audio combiné
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    # Exporter combined_audio dans le fichier temporaire
                    combined_audio.export(tmp_file.name, format="wav")
                    tmp_file_path = tmp_file.name
                
                wav_vad = read_audio(tmp_file_path)

                speech_timestamps = get_speech_timestamps(wav_vad, model_vad)
                logging.debug(f"Speech Analysis: {speech_timestamps}")

                # Effectuer la détection
                if speech_timestamps:
                    print("Speech détecté ...")

                    # Charger et préparer le fichier audio
                    waveform, sample_rate = torchaudio.load(tmp_file_path)

                    # Convertir l'audio au bon format
                    waveform = convert_audio(waveform, sample_rate, model_denoiser.sample_rate, model_denoiser.chin)

                    # Déplacer les données sur le GPU
                    waveform = waveform.to(device)

                    # Appliquer la suppression de bruit
                    denoised_waveform = model_denoiser(waveform[None])[0]

                    # Déplacer les données nettoyées sur le CPU pour les sauvegarder (détacher du graphe de calcul)
                    denoised_waveform = denoised_waveform.detach().cpu()

                    # Sauvegarder le fichier audio nettoyé
                    torchaudio.save(tmp_file_path, denoised_waveform, model_denoiser.sample_rate)

                    if current_settings['task'] != "transcribe":
                        generate_kwargs={
                            "task": "translate",
                            "return_timestamps": False
                        }
                    else:
                        generate_kwargs={
                            "return_timestamps": False
                        }

                    # S'assurer que le modèle est chargé
                    if Transcriber_Whisper_live is None:
                        load_models()

                    transcription_live = Transcriber_Whisper_live(
                        tmp_file_path,
                        generate_kwargs = generate_kwargs | generate_kwargs_live
                    )

                    print(f"Transcription: {transcription_live}")
                    logging.debug(f"Transcription: {transcription_live}")

                    # Envoyer les données au frontend
                    await websocket.send_json({
                        'chunk_duration': combined_audio.duration_seconds,
                        'transcription_live': transcription_live
                    })

                else:
                    print("Silence")
                    await websocket.send_json({
                        'chunk_duration': 0,
                        'transcription_live': {'text': "...\n"}
                    })

                # Supprimer le fichier temporaire après transcription
                os.remove(tmp_file_path)

    except WebSocketDisconnect:
        logging.debug("Client déconnecté")


def convert_tracks_to_json(tracks):
    # Liste pour stocker les segments formatés
    formatted_segments = []

    # Itérer sur les segments avec leurs labels
    for turn, _, speaker in tracks.itertracks(yield_label=True):
        segment = {"speaker": speaker, "start_time": turn.start, "end_time": turn.end}
        formatted_segments.append(segment)

    # Convertir la liste de segments en JSON
    # return json.dumps(formatted_segments)
    return formatted_segments

# Libérer la mémoire GPU une fois que vous avez terminé
def release_whisper_memory():
    global whisper_pipeline
    
    try:
        del Transcriber_Whisper  # Supprime la référence au modèle
    except Exception as e:
        logging.error(f"Impossible de supprimer le modèle : {e}")

    torch.cuda.empty_cache()  # Vide le cache GPU pour libérer la mémoire
    print("Le modèle Whisper a été libéré de la mémoire GPU.")


def extract_audio(file_path):
    """
    Extrait l'audio d'un fichier média (audio ou vidéo) et retourne un AudioSegment.
    """
    # Détecter le type de fichier
    file_type = filetype.guess(file_path)
    
    if file_type is None:
        logging.error(f"Type de fichier non reconnu pour : {file_path}")
        return None
        
    logging.info(f"Type de fichier détecté : {file_type.mime}, Extension : {file_type.extension}")
    
    try:
        # Gestion des fichiers audio
        if file_type.mime.startswith('audio/'):
            logging.info(f"Fichier audio détecté : {file_type.mime}")
            return AudioSegment.from_file(file_path)
                
        # Gestion des fichiers vidéo
        elif file_type.mime.startswith('video/'):
            logging.info(f"Fichier vidéo détecté : {file_type.mime}")
            logging.info("Extraction Audio démarrée ...")
            
            # Créer un fichier temporaire pour l'audio extrait
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
            try:
                # Extraire l'audio de la vidéo
                video = VideoFileClip(file_path)
                video.audio.write_audiofile(temp_audio_path, logger=None)
                video.close()
                
                # Charger l'audio extrait
                audio = AudioSegment.from_wav(temp_audio_path)
                
                # Nettoyer le fichier temporaire
                os.unlink(temp_audio_path)
                
                return audio
                
            except Exception as e:
                # Nettoyer en cas d'erreur
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                raise e
                
        else:
            logging.warning(f"Type de fichier non supporté : {file_type.mime}")
            return None
            
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction audio : {e}")
        return None

def process_audio_chunk(data):
    # Créer un AudioSegment à partir des données brutes
    audio_chunk = AudioSegment.from_raw(io.BytesIO(data), sample_width=2, frame_rate=16000, channels=1)
    # Ici, vous pouvez ajouter des fonctionnalités pour traiter ou analyser l'audio
    print("Processed audio chunk of size:", len(data))

    return audio_chunk


# Fonction pour exécuter la commande `ollama run` et obtenir la réponse du modèle
def run_chocolatine_model(prompt):
    # command_3b = [
    #     "ollama", "run", "jpacifico/chocolatine-3b",
    #     prompt
    # ]

    # command = [
    #     "ollama" "run" "chocolatine-128k:latest",
    #     prompt
    # ]

    # result = subprocess.run(command, capture_output=True, text=True)
    # logging.info("Command output:", result.stdout)  # Affiche la sortie pour vérification
    # logging.info("Command error:", result.stderr)  # Affiche les erreurs éventuelles
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    response = Chocolatine(messages)
    
    return response

# Modèle de données pour la requête POST
class QuestionWithTranscription(BaseModel):
    question: str
    transcription: str
    chat_model: str

def stream_output(process: subprocess.Popen) -> Generator[Tuple[str, str], None, None]:
    """
    Génère la sortie du processus en temps réel.
    Retourne des tuples (type, line) où type est 'stdout' ou 'stderr'
    """
    while True:
        # Lecture de stdout
        stdout_line = process.stdout.readline() if process.stdout else ''
        if stdout_line:
            yield 'stdout', stdout_line.strip()
        
        # Lecture de stderr
        stderr_line = process.stderr.readline() if process.stderr else ''
        if stderr_line:
            yield 'stderr', stderr_line.strip()
        
        # Vérification si le processus est terminé
        if process.poll() is not None and not stdout_line and not stderr_line:
            break


# Fonction pour exécuter la commande en mode streaming
def run_chocolatine_streaming(prompt: str) -> Generator[str, None, None]:
    # Indique le début du streaming
    yield "event: start\ndata: Le streaming a commencé\n\n"

      # Commande pour lancer le modèle
    command = ["ollama", "run", "jpacifico/chocolatine-3b", prompt]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Envoie chaque ligne produite par le modèle
    for line in process.stdout:
        yield f"{line.strip()}\n"  # Envoie chaque ligne de sortie avec un saut de ligne
        # await asyncio.sleep(0)  # Forcer l'envoi de chaque chunk

    # Termine le processus
    process.stdout.close()
    process.wait()

    # Indique la fin du streaming
    yield "event: end\ndata: Le streaming est terminé\n\n"

# Fonction pour exécuter la commande
def run_chocolatine(prompt: str) -> str:
    """Version synchrone optimisée de run_chocolatine"""
    logging.debug(f"Démarrage run_chocolatine avec prompt: {prompt}")
    
    try:
        # Commande pour lancer le modèle
        command = ["ollama", "run", "jpacifico/chocolatine-3b", prompt]
        
        # Utilisation de check_output pour une capture simple et fiable
        result = subprocess.check_output(
            command,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        logging.debug(f"Résultat brut reçu: {result}")
        return result.strip()

    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution: {e.stderr}")
        raise HTTPException(status_code=500, detail=str(e.stderr))
    except Exception as e:
        logging.error(f"Erreur inattendue: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Fonction pour exécuter la commande en mode streaming avec gpt4o-mini
def run_gpt4o_mini_streaming(prompt: str) -> Generator[str, None, None]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Création de la réponse
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            stream=False
        )
 
        logging.debug(f"Réponse complète: {response}")
        return response
    
    except Exception as e:
        logging.error(f"Erreur GPT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Route POST pour le streaming
@app.post(
    "/ask_question/", 
    tags=["🤖 Intelligence Artificielle"],
    summary="Questions sur transcription",
    description="Pose des questions sur une transcription avec les modèles IA (Chocolatine local ou GPT-4o-mini)"
)
async def ask_question(data: QuestionWithTranscription):
    # Vérifier et charger le modèle IA si nécessaire (seulement pour Chocolatine)
    global Chocolatine
    if data.chat_model == "chocolatine" and Chocolatine is None:
        logging.info("Chargement du modèle IA (Chocolatine) pour ask_question...")
        load_ai_model()  # Charger seulement le modèle IA
        
        if Chocolatine is None:
            logging.error("Échec du chargement du modèle Chocolatine")
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle Chocolatine. Vérifiez les logs serveur.")
    
    # Crée le prompt pour le modèle à partir de la question et de la transcription
    prompt_chocolatine = f""" 
Vous êtes un assistant qui répond aux questions et demandes basées sur une transcription de conversation.

Voici les instructions à suivre pour la réponse à apporter à la demande:
    les réponses doivent être au format markdown, avec les points importants en **gras**, les extraits pris dans la conversation en *italique*
    la réponse doit être inférieure à 500 mots
    utilisez uniquement les informations contenues dans la transcription pour répondre.

Voici ci dessous la transcription de la conversation entre crochets:
    [\n\n{data.transcription}]
    
Voici la demande de l'utilisateur : {data.question}
"""
    
    prompt_gpt = [ 
        {"role": "system", "content": "Vous êtes l'assistant AKABI qui répond aux questions basées sur une transcription de conversation."},
        {"role": "user", "content": f"Voici la transcription de la conversation :\n\n{data.transcription}"},
        {"role": "assistant", "content": "Les réponses doivent être au format markdown, avec les points importants en **gras**, les extraits pris dans la conversation en *italique* et **AKABI** sera toujours écrit en majuscule et en gras"},
        # {"role": "assistant", "content": "Les réponses doivent être formatées en texte brut (donc sans symbole markdown) parce que l'affichage coté frontend ne gère pas le markdown, pour une lecture agréable avec des retours à la ligne fréquents.  Utilisez uniquement les informations contenues dans la transcription pour répondre"},
        {"role": "user", "content": f"Voici la demande de l'utilisateur : {data.question}"},
    ]

    logging.debug(f"Modèle utilisé: {data.chat_model}")
    embedding_model, use_cases, use_case_files, index = setup_rag_pipeline()
    
    # Sélection du prompt et de la fonction de streaming en fonction du modèle
    if data.chat_model == "chocolatine":
        if "AKABI" in (data.question).upper():  # pour ignorer la casse
            question_embedding = embedding_model.encode(data.question).astype("float32")
            # Recherche dans l'index FAISS
            _, indices = index.search(question_embedding.reshape(1, -1), k=5)
            relevant_texts = [use_cases[i] for i in indices[0]]

            # Préparer le contexte pour la réponse GPT
            context = " ".join(relevant_texts)
            prompt_chocolatine = f"{prompt_chocolatine}\nVoici également des cas d'usage réalisés chez AKABI: {context}"


        logging.debug(f"Prompt envoyé à {data.chat_model}: {prompt_chocolatine}")

        response = run_chocolatine(prompt_chocolatine)

        json_response =  {"response": response}
        logging.debug(f"Réponse envoyée: {json_response}")
        
        return json_response

    else:
        if "AKABI" in (data.question).upper():  # pour ignorer la casse
            question_embedding = embedding_model.encode(data.question).astype("float32")
            # Recherche dans l'index FAISS
            _, indices = index.search(question_embedding.reshape(1, -1), k=5)
            relevant_texts = [use_cases[i] for i in indices[0]]
            
            # Préparer le contexte pour la réponse GPT
            context = " ".join(relevant_texts)
            prompt_gpt.append({"role": "system", "content": f"Voici des cas d'usage réalisé chez AKABI: {context}"})
            # response = run_gpt4o_mini_streaming(prompt_gpt)

        response = run_gpt4o_mini_streaming(prompt_gpt)

        # Récupérer le contenu de la réponse
        full_content = response.choices[0].message.content
        # json_response = jsonable_encoder(full_content.content)
        
        json_response =  {"response": full_content}
        logging.debug(f"Réponse envoyée: {json_response}")
        
        return json_response
        # return JSONResponse(content= json_response, media_type="application/json")


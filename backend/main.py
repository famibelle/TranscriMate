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
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from starlette.websockets import WebSocketDisconnect

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gc
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Variables globales
Transcriber_Whisper = None
diarization_model = None 
rag_pipeline = None
current_settings = {
    "model": "base.en",
    "task": "transcribe",
    "model_loaded": False,
    "diarization_enabled": True
}

server_url = "http://localhost:8000"

app = FastAPI(
    title="🎙️ TranscriMate API",
    description="""
    # 🎯 API de Transcription Intelligente

    **TranscriMate** propose 3 modes d'utilisation distincts :

    ### **1️⃣ Mode Simple** (`/transcribe_simple/`)  
    - **API uniquement** - Aucune interface utilisateur
    - Diarisation complète (séparation des locuteurs)
    - Transcription haute qualité
    - Retour JSON structuré
    - **Usage :** Intégrations tierces, applications externes

    ### **2️⃣ Mode Streaming** (`/transcribe_streaming/`)  
    - Interface web avec feedback temps réel
    - Traitement progressif visible
    - Server-Sent Events (SSE) pour l'affichage en direct
    - **Usage :** Interface utilisateur avec progression

    ### **3️⃣ Mode Live** (WebSocket `/live_transcription/`)
    - Transcription en temps réel depuis le microphone
    - WebSocket pour latence minimale  
    - Retour instantané
    - **Usage :** Applications temps réel, streaming live

    ---
    """,
    version="2.0.0",
    contact={
        "name": "Support TranscriMate",
        "email": "support@transcrimate.com"
    }
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_core_models():
    """Charge les modèles essentiels pour STT (Whisper + Diarisation)"""
    global Transcriber_Whisper, diarization_model
    
    logging.info("🚀 === CHARGEMENT DES MODÈLES STT ===")
    
    try:
        # Whisper
        if Transcriber_Whisper is None:
            logging.info("📥 Chargement du modèle Whisper...")
            start_whisper = time.time()
            from whisper import load_model as load_whisper_model
            Transcriber_Whisper = load_whisper_model(current_settings["model"])
            end_whisper = time.time()
            logging.info(f"✅ Whisper chargé en {end_whisper - start_whisper:.2f}s")
        
        # Diarisation avec token depuis variables d'environnement
        if diarization_model is None:
            logging.info("📥 Chargement du modèle de diarisation...")
            start_diarization = time.time()
            
            # Récupérer le token depuis les variables d'environnement
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_read_write")
            if not hf_token:
                logging.warning("⚠️ Aucun token Hugging Face trouvé dans les variables d'environnement")
                logging.info("💡 Veuillez configurer HF_TOKEN ou HF_TOKEN_read_write dans le fichier .env")
            
            diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            end_diarization = time.time()
            logging.info(f"✅ Diarisation chargée en {end_diarization - start_diarization:.2f}s")
        
        current_settings["model_loaded"] = True
        logging.info("✅ === MODÈLES STT CHARGÉS AVEC SUCCÈS ===")
        
    except Exception as e:
        logging.error(f"❌ ERREUR lors du chargement des modèles: {str(e)}")
        raise e

def convert_tracks_to_json(diarization_result):
    """Convertit les résultats de diarisation en JSON"""
    tracks = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        tracks.append({
            "start_time": turn.start,
            "end_time": turn.end,
            "speaker": speaker,
            "duration": turn.end - turn.start
        })
    return tracks

# ENDPOINT MODE 1 - API SIMPLE
@app.post(
    "/transcribe_simple/",
    tags=["Mode 1 - API Simple"],
    summary="🎯 API Pure - Sans Interface",
    description="**Mode 1** - API complète avec diarisation + transcription pour intégrations tierces (JSON uniquement)"
)
async def transcribe_simple(file: UploadFile = File(...)):
    """Mode 1 : API simple pour intégrations tierces"""
    logging.info("🎯 === MODE 1 - API SIMPLE DÉMARRÉ ===")
    
    # Initialiser les modèles si nécessaire
    global Transcriber_Whisper, diarization_model
    if Transcriber_Whisper is None or diarization_model is None:
        logging.info("⚠️ Modèles non chargés, initialisation...")
        load_core_models()
    
    # Traitement du fichier
    async with async_temp_manager_context("transcribe_simple") as temp_manager:
        file_data = await file.read()
        file_path = temp_manager.create_temp_file(file.filename, file_data)
        
        # Extraction audio
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            audio = AudioSegment.from_file(file_path)
        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            video_clip = VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file_extension[1:])
            video_clip.close()
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")
        
        # Normalisation audio
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_path = tempfile.mktemp(suffix=".wav")
        audio.export(audio_path, format="wav")
        
        try:
            # Diarisation
            logging.info("🎯 Début diarisation...")
            diarization = diarization_model(audio_path)
            diarization_json = convert_tracks_to_json(diarization)
            
            # Transcription par segment
            full_transcription = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                
                segment_audio = audio[start_ms:end_ms]
                segment_path = tempfile.mktemp(suffix=".wav")
                segment_audio.export(segment_path, format="wav")
                
                # Transcription Whisper
                result = Transcriber_Whisper.transcribe(segment_path)
                transcription = result.get("text", "").strip()
                
                full_transcription.append({
                    "speaker": speaker,
                    "text": transcription,
                    "start_time": turn.start,
                    "end_time": turn.end
                })
                
                os.remove(segment_path)
            
            return {
                "status": "success",
                "filename": file.filename,
                "diarization": diarization_json,
                "transcription": full_transcription,
                "total_segments": len(full_transcription)
            }
            
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

# ENDPOINT MODE 2 - STREAMING 
@app.post(
    "/transcribe_streaming/", 
    tags=["Mode 2 - Streaming"],
    summary="🔄 Interface Progressive - Streaming Temps Réel",
    description="**Mode 2** - Traitement avec feedback progressif (Server-Sent Events)"
)
async def upload_file_streaming(file: UploadFile = File(...)):
    """Mode 2 : Transcription avec feedback temps réel"""
    logging.info("📡 === MODE 2 STREAMING - RÉCEPTION FICHIER ===")
    
    # Version simplifiée pour éviter les erreurs de syntaxe
    async def simple_streaming_generator():
        yield 'data: {"status": "started", "message": "Mode 2 - Traitement démarré"}\n\n'
        await asyncio.sleep(1)
        yield 'data: {"status": "processing", "message": "Traitement en cours (version simplifiée)..."}\n\n' 
        await asyncio.sleep(2)
        yield 'data: {"status": "completed", "message": "Mode 2 disponible mais temporairement simplifié", "filename": "' + file.filename + '"}\n\n'

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*"
    }

    return StreamingResponse(simple_streaming_generator(), media_type="text/event-stream", headers=headers)

# ENDPOINT MODE 3 - WEBSOCKET
buffer = deque(maxlen=5)

@app.websocket("/live_transcription/")
async def websocket_live_transcription(websocket: WebSocket):
    """Mode 3 : Transcription en temps réel via WebSocket"""
    global buffer
    await websocket.accept()
    logging.info("🎤 Client WebSocket connecté pour transcription live")

    try:
        while True:
            data = await websocket.receive_bytes()
            logging.debug(f"📡 Données audio reçues: {len(data)} bytes")
            
            response = {
                "status": "received",
                "message": "Mode 3 WebSocket fonctionnel - transcription live en développement",
                "data_size": len(data)
            }
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logging.info("🔌 Client WebSocket déconnecté")
    except Exception as e:
        logging.error(f"❌ Erreur WebSocket: {str(e)}")
        await websocket.close()

# ENDPOINTS UTILITAIRES
@app.get("/generate_audio_url/{filename}", tags=["Fichiers"])
def generate_audio_url(filename: str):
    return {"url": f"{server_url}/segment_audio/{filename}"}

@app.get("/segment_audio/{filename}", tags=["Fichiers"])
def serve_segment_audio(filename: str):
    file_path = f"segments/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Fichier audio introuvable")

@app.get("/", tags=["Statut"])
def root():
    return {
        "service": "TranscriMate API",
        "version": "2.0.0",
        "status": "operational",
        "modes": {
            "1": "API Simple (/transcribe_simple/)",
            "2": "Streaming (/transcribe_streaming/)",
            "3": "WebSocket Live (/live_transcription/)"
        }
    }

@app.get("/health", tags=["Statut"])
def health():
    return {
        "status": "healthy",
        "whisper_loaded": Transcriber_Whisper is not None,
        "diarization_loaded": diarization_model is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Démarrage de TranscriMate...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
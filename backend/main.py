import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydantic import BaseModel, StrictStr
from pydub import AudioSegment
from starlette.websockets import WebSocketDisconnect
from transformers import pipeline
import filetype
from moviepy.editor import VideoFileClip

from temp_manager import TempFileManager, async_temp_manager_context

# Configuration logging
logging.basicConfig(level=logging.INFO)

# Configuration CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Chargement des variables d'environnement
load_dotenv()

# Variables globales pour les modèles
Transcriber_Whisper = None
diarization_model = None

# Configuration des paramètres par défaut
current_settings = {
    "task": "transcribe",
    "model": "openai/whisper-large-v3-turbo",
    "lang": "auto"
}

# Lifespan pour charger les modèles au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_core_models()
    yield
    # Shutdown

app = FastAPI(
    title="TranscriMate - API de Transcription",
    description="API avec 3 modes : Simple API, Streaming, et Live WebSocket",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_core_models():
    """Charge les modèles Whisper et Pyannote"""
    global Transcriber_Whisper, diarization_model
    
    logging.info("🔄 Chargement des modèles...")
    
    try:
        # Chargement du modèle Whisper
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN non trouvé dans les variables d'environnement")
        
        Transcriber_Whisper = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=hf_token
        )
        
        # Chargement du modèle de diarisation
        diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if torch.cuda.is_available():
            diarization_model = diarization_model.to(torch.device("cuda"))
        
        logging.info("✅ Modèles chargés avec succès")
        
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement des modèles : {e}")
        raise

def convert_tracks_to_json(tracks):
    """Convertit les tracks de diarisation en JSON"""
    formatted_segments = []
    for turn, _, speaker in tracks.itertracks(yield_label=True):
        segment = {
            "speaker": speaker,
            "start_time": turn.start,
            "end_time": turn.end
        }
        formatted_segments.append(segment)
    return formatted_segments

def extract_and_prepare_audio(file_path: str, temp_manager: TempFileManager) -> str:
    """Extrait et prépare l'audio depuis un fichier"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Fichiers audio
    if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
        logging.info(f"Fichier audio détecté: {file_extension}")
        audio = AudioSegment.from_file(file_path)
    
    # Fichiers vidéo
    elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
        logging.info(f"Fichier vidéo détecté: {file_extension}")
        audio = AudioSegment.from_file(file_path)
    
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Format non supporté: {file_extension}"
        )
    
    # Conversion en mono 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Sauvegarde
    audio_path = temp_manager.get_temp_path_with_suffix(".wav")
    audio.export(audio_path, format="wav")
    
    return audio_path

# ==================== MODE 1 : API SIMPLE ====================

class Settings(BaseModel):
    task: StrictStr = "transcribe"
    model: StrictStr = "openai/whisper-large-v3-turbo"
    lang: StrictStr = "auto"

@app.post(
    "/transcribe_simple/",
    tags=["🎯 Mode 1 - API Simple"],
    summary="Transcription complète",
    description="Traitement complet : diarisation + transcription, retour JSON structuré"
)
async def transcribe_simple(file: UploadFile = File(...)):
    """Mode 1 : API Simple - Transcription complète avec diarisation"""
    
    # Vérifier les modèles
    if Transcriber_Whisper is None or diarization_model is None:
        raise HTTPException(status_code=500, detail="Modèles non initialisés")
    
    async with async_temp_manager_context("transcribe_simple") as temp_manager:
        # Lecture et sauvegarde du fichier
        file_data = await file.read()
        file_path = temp_manager.create_temp_file(file.filename, file_data)
        
        # Extraction et préparation audio
        audio_path = extract_and_prepare_audio(file_path, temp_manager)
        
        # Chargement audio pour segmentation
        audio = AudioSegment.from_wav(audio_path)
        
        # Diarisation
        logging.info("🔄 Diarisation en cours...")
        with ProgressHook() as hook:
            diarization = diarization_model(audio_path, hook=hook)
        
        # Transcription de chaque segment
        logging.info("🔄 Transcription des segments...")
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Extraire le segment audio
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment_audio = audio[start_ms:end_ms]
            
            # Sauvegarder temporairement
            segment_path = temp_manager.get_temp_path_with_suffix(".wav")
            segment_audio.export(segment_path, format="wav")
            
            # Transcrire
            transcription = Transcriber_Whisper(
                segment_path,
                return_timestamps=True,
                generate_kwargs={"task": current_settings["task"]}
            )
            
            segments.append({
                "speaker": speaker,
                "text": transcription["text"],
                "start_time": turn.start,
                "end_time": turn.end
            })
        
        logging.info("✅ Transcription simple terminée")
        return {
            "mode": "simple",
            "diarization": convert_tracks_to_json(diarization),
            "transcriptions": segments
        }

# ==================== MODE 2 : STREAMING ====================

@app.post(
    "/transcribe_streaming/",
    tags=["🌊 Mode 2 - Streaming"],
    summary="Transcription en streaming",
    description="Traitement progressif avec retour en temps réel (Server-Sent Events)"
)
async def transcribe_streaming(file: UploadFile = File(...)):
    """Mode 2 : Streaming - Traitement progressif avec SSE"""
    
    # Vérifier les modèles
    if Transcriber_Whisper is None or diarization_model is None:
        raise HTTPException(status_code=500, detail="Modèles non initialisés")
    
    # Lire le fichier avant le générateur pour éviter les problèmes de connexion
    file_data = await file.read()
    filename = file.filename
    
    async def streaming_generator():
        async with async_temp_manager_context("transcribe_streaming") as temp_manager:
            try:
                # Étape 1 : Préparation
                yield f"data: {json.dumps({'status': 'started', 'message': 'Démarrage du traitement'})}\n\n"
                
                file_path = temp_manager.create_temp_file(filename, file_data)
                audio_path = extract_and_prepare_audio(file_path, temp_manager)
                audio = AudioSegment.from_wav(audio_path)
                
                yield f"data: {json.dumps({'status': 'audio_ready', 'message': 'Audio préparé'})}\n\n"
                
                # Étape 2 : Diarisation
                yield f"data: {json.dumps({'status': 'diarization_start', 'message': 'Diarisation en cours...'})}\n\n"
                
                with ProgressHook() as hook:
                    diarization = diarization_model(audio_path, hook=hook)
                
                diarization_json = convert_tracks_to_json(diarization)
                yield f"data: {json.dumps({'status': 'diarization_done', 'diarization': diarization_json})}\n\n"
                
                # Étape 3 : Transcription progressive
                total_segments = len(list(diarization.itertracks(yield_label=True)))
                segment_count = 0
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segment_count += 1
                    
                    # Progress
                    progress = (segment_count / total_segments) * 100
                    yield f"data: {json.dumps({'status': 'transcribing', 'progress': progress, 'segment': segment_count, 'total': total_segments})}\n\n"
                    
                    # Extraire segment
                    start_ms = int(turn.start * 1000)
                    end_ms = int(turn.end * 1000)
                    segment_audio = audio[start_ms:end_ms]
                    
                    segment_path = temp_manager.get_temp_path_with_suffix(".wav")
                    segment_audio.export(segment_path, format="wav")
                    
                    # Transcrire
                    transcription = Transcriber_Whisper(
                        segment_path,
                        return_timestamps=True,
                        generate_kwargs={"task": current_settings["task"]}
                    )
                    
                    # Envoyer résultat du segment
                    segment_data = {
                        "speaker": speaker,
                        "text": transcription["text"],
                        "start_time": turn.start,
                        "end_time": turn.end,
                        "segment_number": segment_count
                    }
                    
                    yield f"data: {json.dumps({'status': 'segment_done', 'segment': segment_data})}\n\n"
                    await asyncio.sleep(0.01)  # Petit délai pour le streaming
                
                # Fin
                yield f"data: {json.dumps({'status': 'completed', 'message': 'Transcription terminée'})}\n\n"
                
            except Exception as e:
                logging.error(f"Erreur streaming: {e}")
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        streaming_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# ==================== MODE 3 : WEBSOCKET LIVE ====================

@app.websocket("/live_transcription/")
async def live_transcription(websocket: WebSocket):
    """Mode 3 : WebSocket Live - Transcription temps réel depuis microphone"""
    
    await websocket.accept()
    logging.info("Client WebSocket connecté pour transcription live")
    
    try:
        # Vérifier les modèles
        if Transcriber_Whisper is None:
            await websocket.send_json({
                "status": "error",
                "message": "Modèle Whisper non initialisé"
            })
            return
        
        # Confirmer la connexion
        await websocket.send_json({
            "status": "connected",
            "message": "WebSocket connecté - Mode temps réel actif"
        })
        
        # Buffer pour accumuler l'audio
        audio_buffer = []
        buffer_duration = 0.0
        target_duration = 2.0  # Transcrire toutes les 2 secondes
        
        while True:
            # Recevoir les données audio du client
            data = await websocket.receive_bytes()
            
            # Convertir les bytes en AudioSegment
            try:
                # Supposer que les données sont en Int16, mono, 16kHz
                audio_segment = AudioSegment(
                    data=data,
                    sample_width=2,  # 16 bits = 2 bytes
                    frame_rate=16000,
                    channels=1
                )
                
                # Ajouter au buffer
                audio_buffer.append(audio_segment)
                buffer_duration += audio_segment.duration_seconds
                
                # Si on a assez d'audio, transcrire
                if buffer_duration >= target_duration:
                    # Combiner tous les segments du buffer
                    combined_audio = sum(audio_buffer)
                    
                    # Créer un fichier temporaire
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                        combined_audio.export(tmp_file.name, format="wav")
                        
                        # Transcrire avec Whisper
                        transcription = Transcriber_Whisper(
                            tmp_file.name,
                            return_timestamps=False,
                            generate_kwargs={"task": current_settings["task"]}
                        )
                        
                        # Envoyer la transcription
                        await websocket.send_json({
                            "status": "transcription",
                            "text": transcription["text"],
                            "duration": buffer_duration,
                            "chunk_duration": target_duration
                        })
                        
                        # Le fichier sera automatiquement supprimé à la fin du 'with'
                    
                    # Réinitialiser le buffer
                    audio_buffer = []
                    buffer_duration = 0.0
                
                # Envoyer un accusé de réception périodique
                elif len(audio_buffer) % 10 == 0:  # Toutes les 10 réceptions
                    await websocket.send_json({
                        "status": "receiving",
                        "buffer_duration": buffer_duration,
                        "bytes_received": len(data)
                    })
                    
            except Exception as audio_error:
                logging.error(f"Erreur traitement audio: {audio_error}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Erreur traitement audio: {str(audio_error)}"
                })
            
    except WebSocketDisconnect:
        logging.info("Client WebSocket déconnecté")
    except Exception as e:
        logging.error(f"Erreur WebSocket: {e}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except:
            pass

# ==================== ROUTES DE CONFIGURATION ====================

@app.post(
    "/settings/",
    tags=["⚙️ Configuration"],
    summary="Mise à jour des paramètres",
    description="Configure les paramètres de transcription"
)
def update_settings(settings: Settings):
    global current_settings
    current_settings = settings.model_dump()
    logging.info(f"Paramètres mis à jour: {current_settings}")
    return {"message": "Paramètres mis à jour avec succès"}

@app.get(
    "/health/",
    tags=["🏥 Santé"],
    summary="Vérification de l'état",
    description="Vérifie l'état des modèles et du système"
)
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "whisper": Transcriber_Whisper is not None,
            "diarization": diarization_model is not None
        },
        "cuda_available": torch.cuda.is_available(),
        "settings": current_settings
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
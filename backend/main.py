import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydantic import BaseModel, StrictStr, Field
from pydub import AudioSegment
from starlette.websockets import WebSocketDisconnect
from transformers import pipeline
import filetype
from moviepy.editor import VideoFileClip

from temp_manager import TempFileManager, async_temp_manager_context

# Configuration logging
logging.basicConfig(level=logging.INFO)

# Classe pour capturer la progression de pyannote
class ProgressCapture:
    def __init__(self):
        self.current_stage = ""
        self.progress = 0
        self.duration = 0
        self.callbacks = []
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def update(self, stage, progress, duration):
        self.current_stage = stage
        self.progress = progress
        self.duration = duration
        for callback in self.callbacks:
            try:
                callback(stage, progress, duration)
            except Exception as e:
                logging.warning(f"Erreur callback progression: {e}")

# Instance globale pour la progression
progress_capture = ProgressCapture()

# Hook personnalisé pour capturer la progression pyannote
class CustomProgressHook(ProgressHook):
    def __init__(self, callback_func=None):
        super().__init__()
        self.callback_func = callback_func
        self.start_time = time.time()
        
    def __call__(self, step_name: str, step_artifact, file=None, total=None, completed=None):
        # Appeler le hook parent avec la bonne signature
        result = super().__call__(step_name, step_artifact, file=file, total=total, completed=completed)
        
        # Calculer le pourcentage si on a des valeurs
        if total is not None and completed is not None and total > 0:
            progress_percent = int((completed / total) * 100)
        else:
            # Progression approximative selon l'étape
            progress_percent = 50 if step_artifact is not None else 0
            
        # Calculer le temps écoulé depuis le début
        elapsed = time.time() - self.start_time
        
        # Mettre à jour la capture globale
        progress_capture.update(step_name, progress_percent, elapsed)
        
        return result

# Configuration CUDA pour performance maximale
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimise pour tailles fixes
torch.backends.cuda.enable_flash_sdp(True)  # Flash Attention si disponible

# Optimisations mémoire
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Supprimer les warnings verbeux
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")  # TorchAudio warnings
warnings.filterwarnings("ignore", message=".*AudioMetaData has been deprecated.*")  # Dépréciation spécifique
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Éviter les warnings

# Chargement des variables d'environnement
load_dotenv()

# Variables globales pour les modèles
Transcriber_Whisper = None
Transcriber_Whisper_Light = None  # Modèle léger pour le mode live
diarization_model = None
Chocolatine_pipeline = None

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
    """Charge les modèles Whisper, Pyannote et Chocolatine"""
    global Transcriber_Whisper, Transcriber_Whisper_Light, diarization_model, Chocolatine_pipeline
    
    logging.info("🔄 Chargement des modèles...")
    
    try:
        # Chargement du modèle Whisper-base (léger et multilingue)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN non trouvé dans les variables d'environnement")
        
        logging.info("🔄 Chargement de Whisper-Large-v3-Turbo (haute qualité)...")
        Transcriber_Whisper = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",  # Meilleure qualité disponible
            torch_dtype=torch.float16,  # FP16 pour optimisation GPU
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=hf_token
        )
        
        # Chargement du modèle Whisper léger pour le mode live
        logging.info("🔄 Chargement de Whisper-base (léger pour live)...")
        Transcriber_Whisper_Light = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",  # Modèle léger pour temps réel
            torch_dtype=torch.float16,  # FP16 pour optimisation GPU
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
        
        # Chargement du modèle Chocolatine
        logging.info("🔄 Chargement du modèle Chocolatine...")
        try:
            pipeline_kwargs = {
                "task": "text-generation",
                "model": "jpacifico/Chocolatine-3B-Instruct-DPO-v1.2", 
                "trust_remote_code": True,
                "token": hf_token
            }
            
            # Configuration GPU optimisée (version compatible)
            if torch.cuda.is_available():
                pipeline_kwargs.update({
                    "device_map": "auto",  # Distribution automatique
                    "torch_dtype": torch.float16  # Half precision
                })
                logging.info("🚀 Configuration GPU pour Chocolatine")
            else:
                pipeline_kwargs.update({
                    "device": "cpu",
                    "torch_dtype": torch.float32
                })
                logging.info("💻 Configuration CPU pour Chocolatine")
            
            Chocolatine_pipeline = pipeline(**pipeline_kwargs)
            logging.info("✅ Modèle Chocolatine chargé avec succès")
        except Exception as e:
            logging.warning(f"⚠️ Impossible de charger Chocolatine: {e}")
            Chocolatine_pipeline = None
        
        # Vérification de la configuration FP16
        if torch.cuda.is_available():
            logging.info("✅ Configuration GPU haute qualité:")
            logging.info(f"   🔹 CUDA Device: {torch.cuda.get_device_name(0)}")
            logging.info(f"   🔹 Whisper Large-v3-Turbo (principal): FP16 - Qualité maximale")
            logging.info(f"   🔹 Whisper Base (live): FP16 - Optimisé temps réel")
            if Chocolatine_pipeline:
                logging.info(f"   🔹 Chocolatine: FP16 (torch.float16)")
        else:
            logging.info("💻 Configuration CPU: FP32 (torch.float32)")
        
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
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        logging.info(f"Traitement fichier: {file_path} (extension: {file_extension})")
        
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
            
        logging.info(f"Audio chargé: durée={len(audio)}ms, canaux={audio.channels}, fréquence={audio.frame_rate}Hz")
        
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur traitement fichier: {str(e)}")
    
    # Conversion en mono 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Sauvegarde
    audio_path = temp_manager.get_temp_path_with_suffix(".wav")
    audio.export(audio_path, format="wav")
    
    return audio_path

# ==================== FONCTION CHOCOLATINE ====================

def run_chocolatine(prompt: str) -> str:
    """Fonction optimisée pour utiliser le modèle Chocolatine via Transformers"""
    global Chocolatine_pipeline
    
    if Chocolatine_pipeline is None:
        return "Le modèle Chocolatine n'est pas disponible. Vérifiez les logs de démarrage."
    
    try:
        # Optimisation: limiter la longueur du prompt pour accélération
        if len(prompt) > 2000:
            prompt = prompt[:1800] + "... [texte tronqué pour optimisation]"
            
        logging.info(f"🍫 Chocolatine génération démarrée (prompt: {len(prompt)} chars)")
        start_time = time.time()
        
        # Génération optimisée pour la vitesse
        response = Chocolatine_pipeline(
            prompt,  # Utiliser directement le prompt string
            max_new_tokens=150,  # Réduire pour accélérer
            temperature=0.3,     # Moins de randomness = plus rapide
            do_sample=True,
            top_p=0.9,          # Nucleus sampling pour qualité
            repetition_penalty=1.1,  # Éviter les répétitions
            pad_token_id=Chocolatine_pipeline.tokenizer.eos_token_id,
            eos_token_id=Chocolatine_pipeline.tokenizer.eos_token_id,
            return_full_text=False,  # Ne retourner que le texte généré
            clean_up_tokenization_spaces=True
        )
        
        # Extraction optimisée du texte généré
        elapsed_time = time.time() - start_time
        
        if response and len(response) > 0:
            generated_text = response[0]['generated_text']
            word_count = len(generated_text.split())
            tokens_per_sec = word_count / elapsed_time if elapsed_time > 0 else 0
            
            logging.info(f"✅ Chocolatine terminé: {elapsed_time:.2f}s, {word_count} mots, {tokens_per_sec:.1f} tok/s")
            # Avec return_full_text=False, on a directement la réponse
            return generated_text.strip()
        else:
            logging.warning(f"⚠️ Chocolatine aucune réponse en {elapsed_time:.2f}s")
            return "Aucune réponse générée par Chocolatine."
            
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        logging.error(f"❌ Erreur Chocolatine après {elapsed_time:.2f}s: {e}")
        return f"Erreur lors de la génération: {str(e)}"

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
        
        # Diarisation avec progression test
        logging.info("🔄 Diarisation en cours...")
        
        # Test simple : envoyer quelques progressions manuellement
        progress_capture.update("segmentation", 25, 1.0)
        progress_capture.update("segmentation", 50, 2.0)
        progress_capture.update("segmentation", 100, 3.0)
        progress_capture.update("speaker_counting", 100, 4.0)  
        progress_capture.update("embeddings", 50, 5.0)
        
        # Faire la diarisation réelle
        diarization = diarization_model(audio_path)
        
        # Finaliser la progression
        progress_capture.update("embeddings", 100, 8.0)
        
        # Sauvegarder le fichier audio complet pour accès ultérieur
        full_audio_path = temp_manager.get_temp_path_with_suffix(".wav")
        audio.export(full_audio_path, format="wav")
        
        # Le fichier est déjà dans le bon répertoire temporaire
        audio_filename = os.path.basename(full_audio_path)
        
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
                "end_time": turn.end,
                "audio_url": f"/temp_audio/{audio_filename}",  # URL pour le fichier complet
                "segment_start": turn.start,  # Timestamps pour extraction côté client
                "segment_end": turn.end
            })
        
        logging.info("✅ Transcription simple terminée")
        return {
            "mode": "simple",
            "diarization": convert_tracks_to_json(diarization),
            "transcriptions": segments,
            "full_audio_url": f"/temp_audio/{audio_filename}"  # URL du fichier complet
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
                
                try:
                    # Progression simulée pour streaming
                    async def simulate_diarization_progress():
                        stages = ['segmentation', 'speaker_counting', 'embeddings']
                        durations = [2, 1, 6]
                        
                        for i, stage in enumerate(stages):
                            start_time = time.time()
                            for progress in range(0, 101, 20):
                                elapsed = time.time() - start_time
                                progress_capture.update(stage, progress, elapsed)
                                await asyncio.sleep(durations[i] / 5)
                    
                    # Lancer progression en arrière-plan
                    progress_task = asyncio.create_task(simulate_diarization_progress())
                    
                    # Diarisation réelle
                    diarization = diarization_model(audio_path)
                    
                    # Arrêter progression
                    progress_task.cancel()
                    
                    diarization_json = convert_tracks_to_json(diarization)
                    yield f"data: {json.dumps({'status': 'diarization_done', 'diarization': diarization_json})}\n\n"
                except Exception as diar_error:
                    logging.error(f"Erreur diarisation streaming: {diar_error}")
                    yield f"data: {json.dumps({'status': 'error', 'message': f'Erreur diarisation: {str(diar_error)}'})}\n\n"
                    return
                
                # Étape 3 : Transcription progressive
                try:
                    # Convertir en liste pour éviter les problèmes d'itérateur
                    segments_list = list(diarization.itertracks(yield_label=True))
                    total_segments = len(segments_list)
                    segment_count = 0
                    
                    for turn, _, speaker in segments_list:
                        try:
                            segment_count += 1
                            
                            # Progress
                            progress = (segment_count / total_segments) * 100 if total_segments > 0 else 0
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
                                "start_time": float(turn.start),
                                "end_time": float(turn.end),
                                "segment_number": segment_count
                            }
                            
                            yield f"data: {json.dumps({'status': 'segment_done', 'segment': segment_data})}\n\n"
                            await asyncio.sleep(0.01)  # Petit délai pour le streaming
                            
                        except Exception as seg_error:
                            logging.error(f"Erreur segment {segment_count}: {seg_error}")
                            yield f"data: {json.dumps({'status': 'segment_error', 'segment': segment_count, 'message': str(seg_error)})}\n\n"
                            continue
                            
                except Exception as trans_error:
                    logging.error(f"Erreur transcription streaming: {trans_error}")
                    yield f"data: {json.dumps({'status': 'error', 'message': f'Erreur transcription: {str(trans_error)}'})}\n\n"
                    return
                
                # Génération des URLs audio pour les segments
                yield f"data: {json.dumps({'status': 'generating_audio_urls', 'message': 'Génération des URLs audio...'})}\n\n"
                
                # Sauvegarder le fichier audio complet pour accès ultérieur
                full_audio_path = temp_manager.get_temp_path_with_suffix(".wav")
                audio.export(full_audio_path, format="wav")
                
                # Créer le nom de fichier unique pour l'URL
                import uuid
                audio_filename = f"streaming_{uuid.uuid4().hex[:8]}.wav"
                
                # Copier vers le dossier temporaire accessible
                final_audio_path = f"backend/temp/{audio_filename}"
                os.makedirs("backend/temp", exist_ok=True)
                audio.export(final_audio_path, format="wav")
                
                # Envoyer les informations audio pour chaque segment
                audio_info = {
                    "full_audio_url": f"/temp_audio/{audio_filename}",
                    "segments_audio_info": []
                }
                
                # Pour chaque segment dans diarization, ajouter les infos audio
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    audio_info["segments_audio_info"].append({
                        "speaker": speaker,
                        "start_time": float(turn.start),
                        "end_time": float(turn.end),
                        "audio_url": f"/temp_audio/{audio_filename}"
                    })
                
                yield f"data: {json.dumps({'status': 'audio_urls_ready', 'audio_info': audio_info})}\n\n"
                
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

# ==================== WEBSOCKET PROGRESSION ====================

@app.websocket("/progress/")
async def websocket_progress(websocket: WebSocket):
    """WebSocket pour suivre la progression des traitements"""
    await websocket.accept()
    logging.info("Client WebSocket connecté pour progression")
    
    # Callback pour envoyer la progression
    async def send_progress(stage, progress, duration):
        try:
            await websocket.send_json({
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "duration": duration,
                "timestamp": time.time()
            })
        except Exception as e:
            logging.warning(f"Erreur envoi progression: {e}")
    
    # Ajouter le callback à la capture globale
    def sync_callback(stage, progress, duration):
        # Créer une tâche asyncio dans l'event loop courant
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(send_progress(stage, progress, duration))
        else:
            loop.run_until_complete(send_progress(stage, progress, duration))
    
    progress_capture.add_callback(sync_callback)
    
    try:
        # Garder la connexion ouverte
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logging.info("Client WebSocket progression déconnecté")

# ==================== MODE 3 : WEBSOCKET LIVE ====================

@app.websocket("/live_transcription/")
async def live_transcription(websocket: WebSocket):
    """Mode 3 : WebSocket Live - Transcription temps réel depuis microphone"""
    
    await websocket.accept()
    logging.info("Client WebSocket connecté pour transcription live")
    
    try:
        # Vérifier les modèles (priorité au modèle léger pour le live)
        whisper_model = Transcriber_Whisper_Light if Transcriber_Whisper_Light is not None else Transcriber_Whisper
        if whisper_model is None:
            await websocket.send_json({
                "status": "error",
                "message": "Aucun modèle Whisper disponible"
            })
            return
        
        # Confirmer la connexion avec informations sur le modèle
        model_info = "whisper-base-light" if whisper_model is Transcriber_Whisper_Light else "whisper-large"
        await websocket.send_json({
            "status": "connected",
            "message": f"WebSocket connecté - Mode temps réel actif avec {model_info}",
            "model": model_info,
            "supports_multilingual": True,
            "buffer_duration_seconds": 2.0
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
                        
                        # Transcrire avec Whisper léger
                        transcription = whisper_model(
                            tmp_file.name,
                            return_timestamps=True,  # Garder les timestamps pour plus d'infos
                            generate_kwargs={"task": current_settings["task"]}
                        )
                        
                        # Envoyer la transcription avec plus d'informations
                        response_data = {
                            "status": "transcription",
                            "text": transcription["text"].strip(),
                            "duration": round(buffer_duration, 2),
                            "chunk_duration": target_duration,
                            "model_used": "whisper-base-light" if whisper_model is Transcriber_Whisper_Light else "whisper-large",
                            "timestamp": round(asyncio.get_event_loop().time(), 2)
                        }
                        
                        # Ajouter les chunks si disponibles
                        if "chunks" in transcription:
                            response_data["chunks"] = transcription["chunks"]
                            
                        await websocket.send_json(response_data)
                        
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

# ==================== ROUTES AUDIO ====================

# Stockage temporaire des fichiers audio (en production, utilisez Redis ou un cache approprié)
temp_audio_cache = {}

@app.get(
    "/temp_audio/{filename}",
    tags=["🎵 Audio"],
    summary="Récupération d'un fichier audio temporaire",
    description="Retourne un fichier audio temporaire par son nom"
)
async def get_temp_audio(filename: str):
    """Retourne un fichier audio temporaire"""
    
    # Essayer d'abord dans backend/temp (pour les nouveaux fichiers streaming)
    local_temp_path = f"backend/temp/{filename}"
    if os.path.exists(local_temp_path):
        file_path = local_temp_path
    else:
        # Construire le chemin du fichier (dans /temp ou votre répertoire temporaire)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        # Vérifier que le fichier existe
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
    
    # Retourner le fichier
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Cache-Control": "public, max-age=3600"}  # Cache 1 heure
    )

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
    models_status = {
        "whisper": Transcriber_Whisper is not None,
        "whisper_light": Transcriber_Whisper_Light is not None,
        "diarization": diarization_model is not None,
        "chocolatine": Chocolatine_pipeline is not None
    }
    
    # Déterminer le statut global
    critical_models = ["whisper", "diarization"]  # Modèles critiques pour le fonctionnement
    all_critical_loaded = all(models_status[model] for model in critical_models)
    
    return {
        "status": "ready" if all_critical_loaded else "loading",
        "models_loaded": models_status,
        "critical_models_ready": all_critical_loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "settings": current_settings,
        "message": "Tous les modèles critiques sont chargés" if all_critical_loaded else "Chargement en cours..."
    }

# ==================== CHATBOT ====================

class QuestionRequest(BaseModel):
    question: str = Field(
        default="Présente-toi",
        description="La question à poser à l'IA",
        example="Présente-toi"
    )
    transcription: str = Field(
        default="Comment on appel un pain au chocolat dans le sud ?",
        description="Le texte de transcription sur lequel baser la réponse",
        example="Comment on appel un pain au chocolat dans le sud ?"
    )
    chat_model: str = Field(
        default="chocolatine",
        description="Le modèle IA à utiliser (chocolatine ou gpt-4)",
        example="chocolatine"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "Présente-toi",
                "transcription": "Comment on appel un pain au chocolat dans le sud ?",
                "chat_model": "chocolatine"
            }
        }

@app.post(
    "/ask_question/",
    tags=["🤖 Chatbot"],
    summary="Répondre à une question",
    description="Utilise l'IA pour répondre à une question basée sur la transcription"
)
async def ask_question(request: QuestionRequest):
    """Endpoint pour le chatbot - Réponse à une question basée sur la transcription"""
    
    try:
        # Construire le prompt complet
        prompt = f"""Voici une transcription d'une conversation:

{request.transcription}

Question: {request.question}

Réponds à la question en te basant sur le contenu de la transcription. Sois précis et structure ta réponse."""

        # Utiliser le modèle demandé
        if request.chat_model == "chocolatine":
            response_text = run_chocolatine(prompt)
        elif request.chat_model == "gpt-4":
            # Pour GPT-4, on simule pour l'instant (nécessiterait une API key OpenAI)
            response_text = f"""[Simulation GPT-4] Basé sur la transcription fournie, voici ma réponse à votre question "{request.question}":

La transcription montre une conversation entre deux locuteurs discutant d'une journée d'apprentissage du français. 

Analyse détaillée:
- SPEAKER_01 semble être un enseignant ou guide
- SPEAKER_00 est un étudiant partageant son expérience
- La conversation porte sur des interactions multilingues et des activités quotidiennes
- L'étudiant a eu une journée productive avec des cours et des rencontres

La conversation révèle un environnement d'apprentissage positif et interactif."""
        else:
            response_text = f"Modèle '{request.chat_model}' non supporté. Modèles disponibles: chocolatine, gpt-4"

        return {
            "response": response_text,
            "model_used": request.chat_model,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
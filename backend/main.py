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

# logging.basicConfig(level=logging.DEBUG)

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

TEMP_FOLDER = '/tmp/'
HF_cache = '/mnt/.cache/'
Model_dir = '/mnt/Models'

HUGGING_FACE_KEY = os.environ.get("HuggingFace_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_MCF")

server_url = os.getenv("SERVER_URL")

# Dossier pour les transcriptions
output_dir = "Transcriptions"
os.makedirs(output_dir, exist_ok=True)

directories = [ "/mnt/Models", "/mnt/logs" , "/mnt/.cache"]
for directory in directories:
    if not os.path.exists(directory):
        os.system(f"sudo mkdir {directory}")

os.system("sudo chmod -R 755 /mnt/Models /mnt/.cache /mnt/logs")
os.system("sudo chown -R $USER:$USER /mnt/Models /mnt/.cache /mnt/logs")

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
def load_pipeline_diarization(model):
    pipeline_diarization = Pipeline.from_pretrained(
        model,
        cache_dir= HF_cache,
        use_auth_token = HUGGING_FACE_KEY,
    )

    if torch.cuda.is_available():
        pipeline_diarization.to(torch.device("cuda"))

    logging.info(f"Piepline Diarization déplacée sur {device}")

    return pipeline_diarization

diarization_model = load_pipeline_diarization("pyannote/speaker-diarization-3.1")

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
app = FastAPI(lifespan=lifespan)

def load_models():
    global Transcriber_Whisper, Transcriber_Whisper_live, last_activity_timestamp

    if Transcriber_Whisper is None:
        logging.info("Chargement du modèle Transcriber_Whisper...")
        Transcriber_Whisper = pipeline(
                "automatic-speech-recognition",
                # model = model_selected[0],
                model = model_settings,
                chunk_length_s=30,
                # stride_length_s=(4, 2),
                # torch_dtype="torch.float16",    
                device=device
            )
        logging.info("Modèle Transcriber_Whisper chargé")


    if Transcriber_Whisper_live is None:
        logging.info("Chargement du modèle Transcriber_Whisper_live...")
        Transcriber_Whisper_live = pipeline(
                "automatic-speech-recognition",
                model = "openai/whisper-medium",
                chunk_length_s=30,
                # stride_length_s=(4, 2),
                # torch_dtype="torch.float16",    
                device=device
            )
        logging.info("Modèle Transcriber_Whisper_live chargé")


    # Si un GPU est disponible, convertir le modèle Whisper en FP16
    if device.type == "cuda":
        model_post = Transcriber_Whisper.model
        model_post = model_post.half()  # Convertir en FP16
        Transcriber_Whisper.model = model_post  # Réassigner le modèle à la pipeline

        model_live = Transcriber_Whisper_live.model
        model_live = model_live.half()  # Convertir en FP16
        Transcriber_Whisper_live.model = model_live  # Réassigner le modèle à la pipeline

    # Mettre à jour le timestamp d'activité
    last_activity_timestamp = time.time()


# Charger les modèles à la demande via la route /initialize/
@app.get("/initialize/")
async def initialize_models():
    # yield {"message": "Modèles initialisation démarrée"}

    load_models()
    return {"message": "Modèles chargés avec succès"}

@app.get("/keep_alive/")
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

@app.post("/diarization/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"

    full_transcription_text = "\n"

    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.info(f"Extension détectée {file_extension}.")

    try:
        # Sauvegarder temporairement le fichier uploadé
        with open(file_path, "wb") as f:
            f.write(await file.read())
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
        
        audio_path = f"{file_path}.wav"
        
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
        
    finally:
        logging.info(f"->> fin de diarization <<")
        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Fichier temporaire {file_path} supprimé.")


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


@app.post("/settings/")
def update_settings(settings: Settings):
    global current_settings
    # Logique de mise à jour des paramètres côté backend
    # Enregistrez les paramètres dans une base de données ou un fichier de configuration, par exemple
    current_settings = settings.model_dump()  # Mettez à jour la variable globale
    logging.info(f"Settings: {current_settings}, task: {current_settings['task']}")

    return {"message": "Paramètres mis à jour avec succès"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.info(f"Extension détectée {file_extension}.")

    # Utilise les paramètres de transcription
    task = current_settings.get("transcribe")  # Prend la valeur par défaut si non définie


    try:
        # Sauvegarder temporairement le fichier uploadé
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"Fichier {file.filename} sauvegardé avec succès.")

        file_type = filetype.guess(file_path)
        logging.info(f"Type de fichier : {file_type.mime}, Extension : {file_type.extension}")

        extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_ongoing', 'message': 'Extraction audio en cours ...'})

        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.info(f"fichier audio détecté: {file_extension}.")

            audio = AudioSegment.from_file(file_path)


        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.info(f"fichier vidéo détecté: {file_extension}.")
            VideoFileClip(file_path)
            logging.info("Extraction Audio démarrée ...")

            logging.info(extraction_status)

            audio = AudioSegment.from_file(file_path, format=file_type.extension)

            logging.info(extraction_status)

        logging.info(f"Conversion du {file.filename} en mono 16kHz.")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio_path = f"{file_path}.wav"
        logging.info(f"Sauvegarde de la piste audio dans {audio_path}.")
        audio.export(audio_path, format="wav")

        # Vérification si le fichier existe
        if not os.path.exists(audio_path):
            logging.error(f"Le fichier {audio_path} n'existe pas.")
            raise HTTPException(status_code=404, detail=f"Le fichier {audio_path} n'existe pas.")

        # Étape 1 : Diarisation
        logging.info(f"Démarrage de la diarisation du fichier {audio_path}")

        async def live_process_audio():
            extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})
            yield f"{extraction_status}\n"
            logging.info(extraction_status)

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
                segment_path = f"/tmp/segment_{start_ms}_{end_ms}.wav"
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

        # Retourner les résultats en streaming
        logging.info(f"->> fin de transcription <<")
        logging.info(full_transcription)
        return StreamingResponse(live_process_audio(), media_type="application/json")


    finally:
        logging.info(f"->> fin de transcription <<")
        logging.info(full_transcription)

        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Fichier temporaire {file_path} supprimé.")


# Endpoint pour générer les URLs des segments audio
@app.get("/generate_audio_url/{filename}")
def generate_audio_url(filename: str):
    return {"url": f"{server_url}/segment_audio/{filename}"}

# Endpoint pour servir les segments audio
@app.get("/segment_audio/{filename}")
async def get_segment_audio(filename: str):
    file_path = f"/tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "Fichier non trouvé"}

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"

    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()

    # Si le fichier est un fichier audio (formats courants)
    if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
        # Charger le fichier audio avec Pydub
        audio = AudioSegment.from_file(file_path)

    elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
        VideoFileClip(file_path)
        audio = AudioSegment.from_file(file_path, format=file.filename)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    audio_path = f"{file_path}.wav"

    audio.export(audio_path, format="wav")

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

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
            segment_path = f"/tmp/segment_{start_ms}_{end_ms}.wav"
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

    finally:
        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info(f"Fichier temporaire {audio_path} supprimé.")


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

@app.websocket("/streaming_audio/")
async def websocket_audio_receiver(websocket: WebSocket):
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

                    # Déplacer les données nettoyées sur le CPU pour les sauvegarder
                    denoised_waveform = denoised_waveform.cpu()

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
    command_3b = [
        "ollama", "run", "jpacifico/chocolatine-3b",
        prompt
    ]

    command = [
        "ollama" "run" "chocolatine-128k:latest",
        prompt
    ]

    
    result = subprocess.run(command, capture_output=True, text=True)
    logging.info("Command output:", result.stdout)  # Affiche la sortie pour vérification
    logging.info("Command error:", result.stderr)  # Affiche les erreurs éventuelles

    return result.stdout.strip()

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
@app.post("/ask_question/")
async def ask_question(data: QuestionWithTranscription):
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
        {"role": "system", "content": "Vous êtes un assistant qui répond aux questions basées sur une transcription de conversation."},
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
            response = run_gpt4o_mini_streaming(prompt_gpt)

        response = run_gpt4o_mini_streaming(prompt_gpt)

        # Récupérer le contenu de la réponse
        full_content = response.choices[0].message.content
        # json_response = jsonable_encoder(full_content.content)
        
        json_response =  {"response": full_content}
        logging.debug(f"Réponse envoyée: {json_response}")
        
        return json_response
        # return JSONResponse(content= json_response, media_type="application/json")


import asyncio
from rich.progress import Progress
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from fastapi.middleware.cors import CORSMiddleware

from json import dumps

import os
import json
import re
import time

import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)

import sys

from dotenv import load_dotenv
import torch
from moviepy.editor import *
from pydub import AudioSegment
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

TEMP_FOLDER = '/tmp/'
HF_cache = '/mnt/.cache/'
Model_dir = '/mnt/Models'

HUGGING_FACE_KEY =  os.environ.get("HuggingFace_API_KEY")

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
    print(f"GPU trouvé! on utilise CUDA. Device: {device}")
else:
    print(f"Pas de GPU de disponible... Device: {device}")

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

Transcriber_Whisper = pipeline(
        "automatic-speech-recognition",
        model = model_selected[0],
        chunk_length_s=30,
        device=device    
    )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:8080/"],  # URL de Vue.js
    allow_origins=["*"],  # URL de Vue.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/diarization/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"

    full_transcription =  []
    full_transcription_text = "\n"

    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.debug(f"Extension détectée {file_extension}.")

    try:
        # Sauvegarder temporairement le fichier uploadé
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.debug(f"Fichier {file.filename} sauvegardé avec succès.")
        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.debug(f"fichier audio détecté: {file_extension}.")
            # Charger le fichier audio avec Pydub
            audio = AudioSegment.from_file(file_path)

        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.debug(f"fichier vidéo détecté: {file_extension}.")
            video_clip = VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file.filename)

        logging.debug(f"Conversion du {file.filename} en mono 16kHz.")

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        audio_path = f"{file_path}.wav"
        
        logging.debug(f"Sauvegarde de la piste audio dans {audio_path}.")

        audio.export(audio_path, format="wav")

        # Vérification si le fichier existe
        if not os.path.exists(audio_path):
            logging.error(f"Le fichier {audio_path} n'existe pas.")
            raise HTTPException(status_code=404, detail=f"Le fichier {audio_path} n'existe pas.")

        with ProgressHook() as hook:
            diarization = diarization_model(audio_path, hook=hook)

        diarization_json = convert_tracks_to_json(diarization)
        logging.debug(f"Résultat de la diarization {diarization_json}")
        return diarization_json
        
    finally:
        logging.debug(f"->> fin de diarization <<")
        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.debug(f"Fichier temporaire {file_path} supprimé.")

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"

    full_transcription =  []
    full_transcription_text = "\n"

    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.debug(f"Extension détectée {file_extension}.")

    try:
        # Sauvegarder temporairement le fichier uploadé
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.debug(f"Fichier {file.filename} sauvegardé avec succès.")
        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.debug(f"fichier audio détecté: {file_extension}.")
            # Charger le fichier audio avec Pydub
            audio = AudioSegment.from_file(file_path)

        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.debug(f"fichier vidéo détecté: {file_extension}.")
            video_clip = VideoFileClip(file_path)
            audio = AudioSegment.from_file(file_path, format=file.filename)

        logging.debug(f"Conversion du {file.filename} en mono 16kHz.")
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio_path = f"{file_path}.wav"
        logging.debug(f"Sauvegarde de la piste audio dans {audio_path}.")
        audio.export(audio_path, format="wav")

        # Vérification si le fichier existe
        if not os.path.exists(audio_path):
            logging.error(f"Le fichier {audio_path} n'existe pas.")
            raise HTTPException(status_code=404, detail=f"Le fichier {audio_path} n'existe pas.")

        # Étape 1 : Diarisation
        logging.debug(f"Démarrage de la diarisation du fichier {audio_path}")

        # Variable pour stocker la progression
        progress_data = {"progress": 0}
        
        def monitor_progress():
            """Thread pour surveiller la progression."""
            while not progress_data["done"]:
                task_progress = []
                for task in hook.progress.tasks:
                    progress_percentage = (task.completed / task.total) * 100 if task.total else 100
                    task_progress.append({
                        "task": task.description,
                        "progress": progress_percentage
                    })
                progress_data["progress"] = task_progress
                yield f"{json.dumps({'progress_data': progress_data})}\n"
                time.sleep(0.5)  # Mettre à jour toutes les 500 ms

        with ProgressHook() as hook:
            # Démarrer le thread de suivi de la progression
            progress_data["done"] = False
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.start()

            diarization = diarization_model(audio_path, hook=hook)

            # La tâche de diarisation est terminée, donc arrêter le suivi
            progress_data["done"] = True
            monitor_thread.join()

            # Envoyer la progression du hook
            # yield f"data: {json.dumps({'progress': hook.progress})}\n\n"
            # Calculer le pourcentage de progression
            # progress_percentage = (hook.progress.completed / hook.progress.total) * 100

        diarization_json = convert_tracks_to_json(diarization)
        logging.debug(f"Résultat de la diarization {diarization_json}")

        # Streaming des résultats
        async def live_process_audio():
            # Envoyer la diarisation complète d'abord
            yield f"{json.dumps({'diarization': diarization_json})}\n"
            await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
    
            # Exporter les segments pour chaque locuteur
            total_chunks = len(list(diarization.itertracks(yield_label=True))) 
            logging.debug(f"total_turns: {total_chunks}")
            
            turn_number = 0
            # for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), total=total_turns, desc="Processing turns"):
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turn_number += 1
                logging.debug(f"Tour {turn_number}/{total_chunks}")

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
                    # input_features= segment_path, 
                    return_timestamps = True, 
                    # return_timestamps="word",
                    generate_kwargs={"task": "transcribe"}
                )

                # Supprimer le fichier de segment une fois transcrit
                # os.remove(segment_path)

                segment = {
                    "speaker": speaker,
                    "text": transcription,
                    "start_time": turn.start,
                    "end_time": turn.end,
                    "audio_url": f"http://127.0.0.1:8000/segment_audio/{os.path.basename(segment_path)}"  # URL du fichier audio
                }

                logging.debug(f"Transcription du speaker {speaker} du segment de {turn.start} à {turn.end} terminée\n Résultat de la transcription {segment}")
                full_transcription.append(segment)

                yield f"{json.dumps(segment)}\n"  # Envoi du segment de transcription en JSON
                await asyncio.sleep(0)  # Forcer l'envoi de chaque chunk

                logging.debug(f"Transcription du speaker {speaker} pour le segment de {turn.start} à {turn.end} terminée")

        # Retourner les résultats en streaming
        return StreamingResponse(live_process_audio(), media_type="application/json")


    finally:
        logging.debug(f"->> fin de transcription <<")
        print(full_transcription)

        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.debug(f"Fichier temporaire {file_path} supprimé.")


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
        video_clip = VideoFileClip(file_path)
        audio = AudioSegment.from_file(file_path, format=file.filename)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    audio_path = f"{file_path}.wav"

    audio.export(audio_path, format="wav")

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logging.debug(f"Fichier {file.filename} sauvegardé avec succès.")
        
        # Démarrer la diarisation
        logging.debug("Démarrage de la diarisation")

        # Étape 1 : Diarisation
        with ProgressHook() as hook:
            diarization = diarization_model(audio_path, hook=hook)

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
            logging.debug(f"Transcription du speaker {speaker} du segment de {turn.start} à {turn.end} terminée")
            
            segments.append({
                "speaker": speaker,
                "text": transcription,
                "start_time": turn.start,
                "end_time": turn.end
            })
        logging.debug("Transcription terminée.")

        return {"transcriptions": segments}

    finally:
        # Nettoyage : supprimer le fichier temporaire après traitement
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logging.debug(f"Fichier temporaire {audio_path} supprimé.")

def convert_tracks_to_json(tracks):
    # Liste pour stocker les segments formatés
    formatted_segments = []

    # Itérer sur les segments avec leurs labels
    for turn, _, speaker in tracks.itertracks(yield_label=True):
        segment = {"speaker": speaker, "start_time": turn.start, "end_time": turn.end}
        formatted_segments.append(segment)

    # Convertir la liste de segments en JSON
    return json.dumps(formatted_segments)

import asyncio
from rich.progress import Progress
import threading

import filetype

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
from pydantic import BaseModel, StrictStr, StrictBool



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
    print(f"GPU trouvé! on utilise CUDA. Device: {device}")
else:
    print(f"Pas de GPU de disponible... Device: {device}")

# Initialisation de `current_settings` avec des valeurs par défaut
# global current_settings
current_settings = {
    "task": "transcribe",
    "model": "openai/whisper-large-v3-turbo",
    "lang": "auto"
}

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

# whisper_task_transcribe = "transcribe" 
# generate_kwargs={"language": "french"}), 
# generate_kwargs={"task": "translate"}),
# generate_kwargs={"language": "french", "task": "translate"})

model_settings = current_settings.get("model", "openai/whisper-large-v3-turbo")  # Valeur par défaut si non définie

Transcriber_Whisper = pipeline(
        "automatic-speech-recognition",
        # model = model_selected[0],
        model = model_settings,
        chunk_length_s=30,
        # stride_length_s=(4, 2),
        device=device    
    )

app = FastAPI()
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


class Settings(BaseModel):
    task: StrictStr = "transcribe"
    model: StrictStr = "openai/whisper-large-v3-turbo"  # Valeur par défaut
    lang: StrictStr = "auto"  # Valeur par défaut

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
    logging.debug(f"Settings: {current_settings}, task: {current_settings['task']}")

    return {"message": "Paramètres mis à jour avec succès"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    full_transcription =  []
    # Détection de l'extension du fichier
    file_extension = os.path.splitext(file_path)[1].lower()
    logging.debug(f"Extension détectée {file_extension}.")

    # audio = extract_audio(file_path)
    
    # if audio is not None:
    #     print(f"Audio extrait avec succès. Durée : {len(audio) / 1000} secondes")
    #     # Vous pouvez maintenant utiliser l'objet audio (AudioSegment) comme vous le souhaitez
    #     # Par exemple : audio.export("sortie.mp3", format="mp3")    

    # Utilise les paramètres de transcription
    task = current_settings.get("transcribe")  # Prend la valeur par défaut si non définie
    # generate_kwargs={"language": "english"}
    # generate_kwargs={"task": "translate"})

    try:
        # Sauvegarder temporairement le fichier uploadé
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.debug(f"Fichier {file.filename} sauvegardé avec succès.")

        file_type = filetype.guess(file_path)
        logging.debug(f"Type de fichier : {file_type.mime}, Extension : {file_type.extension}")

        extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_ongoing', 'message': 'Extraction audio en cours ...'})

        # Si le fichier est un fichier audio (formats courants)
        if file_extension in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']:
            logging.debug(f"fichier audio détecté: {file_extension}.")

            # Charger le fichier audio avec Pydub
            # extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_ongoing', 'message': 'Extraction audio en cours ...'})
            audio = AudioSegment.from_file(file_path)
            # extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})


        elif file_extension in ['.mp4', '.mov', '.3gp', '.mkv']:
            logging.debug(f"fichier vidéo détecté: {file_extension}.")
            video_clip = VideoFileClip(file_path)
            logging.debug("Extraction Audio démarrée ...")

            # extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_ongoing', 'message': 'Extraction audio en cours ...'})
            logging.debug(extraction_status)
            # yield f"{extraction_status}\n"

            audio = AudioSegment.from_file(file_path, format=file_type.extension)

            # extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})
            logging.debug(extraction_status)
            # yield f"{extraction_status}\n"

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

        async def live_process_audio():
            extraction_status = json.dumps({'extraction_audio_status': 'extraction_audio_done', 'message': 'Extraction audio terminée!'})
            yield f"{extraction_status}\n"
            logging.debug(extraction_status)

            # Envoi du statut "en cours"
            start_diarization = json.dumps({'status': 'diarization_processing', 'message': 'Séparation des voix en cours...'})
            yield f"{start_diarization}\n"
            await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
            logging.debug(start_diarization)

            with ProgressHook() as hook:
                diarization = diarization_model(audio_path, hook=hook)

            # Envoi final du statut pour indiquer la fin
            end_diarization = json.dumps({'status': 'diarization_done', 'message': 'Séparation des voix terminée.'})
            yield f"{end_diarization}\n"
            await asyncio.sleep(0.1)  # Petit délai pour forcer l'envoi de la première réponse
            logging.debug(end_diarization)

            diarization_json = convert_tracks_to_json(diarization)

            # Envoyer la diarisation complète d'abord
            logging.debug(f"{json.dumps({'diarization': diarization_json})}")
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

                logging.debug(f"----> Transcription démarée avec le model <{model_settings}> et la task <{task}> <----")

                if current_settings['task'] != "transcribe":
                    transcription = Transcriber_Whisper(
                        segment_path,
                        return_timestamps = True,
                        generate_kwargs={"language": "french"} 
                        # generate_kwargs={"language": "english"} 
                        )
                else:
                    transcription = Transcriber_Whisper(
                        segment_path,
                        return_timestamps = True
                    )
                # Transcrire ce segment avec Whisper
                # Supprimer le fichier de segment une fois transcrit
                # os.remove(segment_path)

                segment = {
                    "speaker": speaker,
                    "text": transcription,
                    "start_time": turn.start,
                    "end_time": turn.end,
                    "audio_url": f"{server_url}/segment_audio/{os.path.basename(segment_path)}"  # URL du fichier audio
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
    # return json.dumps(formatted_segments)
    return formatted_segments



import filetype
import logging
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os
import tempfile

def extract_audio(file_path):
    """
    Extrait l'audio d'un fichier média (audio ou vidéo) et retourne un AudioSegment.
    """
    # Détecter le type de fichier
    file_type = filetype.guess(file_path)
    
    if file_type is None:
        logging.error(f"Type de fichier non reconnu pour : {file_path}")
        return None
        
    logging.debug(f"Type de fichier détecté : {file_type.mime}, Extension : {file_type.extension}")
    
    try:
        # Gestion des fichiers audio
        if file_type.mime.startswith('audio/'):
            logging.debug(f"Fichier audio détecté : {file_type.mime}")
            return AudioSegment.from_file(file_path)
                
        # Gestion des fichiers vidéo
        elif file_type.mime.startswith('video/'):
            logging.debug(f"Fichier vidéo détecté : {file_type.mime}")
            logging.debug("Extraction Audio démarrée ...")
            
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


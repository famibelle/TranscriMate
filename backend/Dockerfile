# Utilise une image Python légère
FROM python:3.10-slim

FROM nvidia/cuda:12.6.0-base-ubuntu22.04
# Mettre à jour les paquets et installer ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Étape 2 : Installer Python et pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean
    
# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires pour installer les dépendances
COPY requirements.txt .

# Installe les dépendances avec pip
# Mettre à jour pip avant d'installer les dépendances
# RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de l'application
COPY . .

# Expose le port utilisé par FastAPI
EXPOSE 8000

# Définit la commande par défaut pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Utilise une image Python légère
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires pour installer les dépendances
COPY requirements.txt .

# Installe les dépendances avec pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de l'application
COPY . .

# Expose le port utilisé par FastAPI
EXPOSE 8000

# Définit la commande par défaut pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

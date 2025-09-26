"""
Gestionnaire de fichiers temporaires cross-platform pour TranscriMate
Gère la création, validation et nettoyage des fichiers temporaires de manière sécurisée
"""

import os
import tempfile
import time
import uuid
import logging
from pathlib import Path
from typing import List, Optional
import contextlib
import asyncio


class TempFileManager:
    """Gestionnaire de fichiers temporaires sécurisé et cross-platform"""
    
    def __init__(self, prefix: str = "transcrimate"):
        self.temp_dir = tempfile.gettempdir()
        self.prefix = prefix
        self.created_files: List[str] = []
        logging.info(f"TempFileManager initialisé avec répertoire: {self.temp_dir}")
    
    def safe_filename(self, original_name: str) -> str:
        """
        Génère un nom de fichier sécurisé à partir du nom original
        
        Args:
            original_name: Le nom de fichier original
            
        Returns:
            str: Nom de fichier sécurisé avec timestamp et UUID
        """
        # Extraire l'extension de manière sécurisée
        suffix = Path(original_name).suffix.lower()
        
        # Valider l'extension (liste blanche des extensions autorisées)
        allowed_extensions = {
            '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a',  # Audio
            '.mp4', '.mov', '.3gp', '.mkv', '.avi', '.webm'   # Vidéo
        }
        
        if suffix not in allowed_extensions:
            logging.warning(f"Extension non autorisée: {suffix}. Utilisation de .tmp")
            suffix = '.tmp'
        
        # Créer un nom unique
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        safe_name = f"{self.prefix}_{timestamp}_{unique_id}{suffix}"
        
        return safe_name
    
    def create_temp_file(self, original_filename: str, data: bytes) -> str:
        """
        Crée un fichier temporaire sécurisé avec les données fournies
        
        Args:
            original_filename: Nom original du fichier
            data: Données binaires à écrire
            
        Returns:
            str: Chemin complet vers le fichier temporaire créé
        """
        safe_name = self.safe_filename(original_filename)
        file_path = os.path.join(self.temp_dir, safe_name)
        
        try:
            with open(file_path, 'wb') as f:
                f.write(data)
            
            self.created_files.append(file_path)
            logging.info(f"Fichier temporaire créé: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Erreur lors de la création du fichier temporaire: {e}")
            raise
    
    def create_temp_path(self, original_filename: str) -> str:
        """
        Génère un chemin temporaire sans créer le fichier
        
        Args:
            original_filename: Nom original du fichier
            
        Returns:
            str: Chemin complet vers le fichier temporaire
        """
        safe_name = self.safe_filename(original_filename)
        file_path = os.path.join(self.temp_dir, safe_name)
        self.created_files.append(file_path)
        return file_path
    
    def get_temp_path_with_suffix(self, suffix: str) -> str:
        """
        Génère un chemin temporaire avec un suffixe spécifique
        
        Args:
            suffix: Extension du fichier (ex: '.wav', '.mp3')
            
        Returns:
            str: Chemin complet vers le fichier temporaire
        """
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        safe_name = f"{self.prefix}_{timestamp}_{unique_id}{suffix}"
        file_path = os.path.join(self.temp_dir, safe_name)
        self.created_files.append(file_path)
        return file_path
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        Nettoie un fichier spécifique
        
        Args:
            file_path: Chemin du fichier à supprimer
            
        Returns:
            bool: True si supprimé avec succès, False sinon
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Fichier temporaire supprimé: {file_path}")
                if file_path in self.created_files:
                    self.created_files.remove(file_path)
                return True
            return False
        except Exception as e:
            logging.warning(f"Impossible de supprimer {file_path}: {e}")
            return False
    
    def cleanup(self) -> int:
        """
        Nettoie tous les fichiers temporaires créés par ce gestionnaire
        
        Returns:
            int: Nombre de fichiers supprimés avec succès
        """
        deleted_count = 0
        files_to_remove = self.created_files.copy()
        
        for file_path in files_to_remove:
            if self.cleanup_file(file_path):
                deleted_count += 1
        
        logging.info(f"Nettoyage terminé: {deleted_count} fichiers supprimés")
        return deleted_count
    
    def __del__(self):
        """Nettoyage automatique lors de la destruction de l'objet"""
        self.cleanup()


@contextlib.contextmanager
def temp_file_context(original_filename: str, data: bytes, prefix: str = "transcrimate"):
    """
    Gestionnaire de contexte pour fichier temporaire unique
    
    Args:
        original_filename: Nom original du fichier
        data: Données binaires
        prefix: Préfixe pour le nom du fichier
        
    Yields:
        str: Chemin du fichier temporaire
    """
    manager = TempFileManager(prefix=prefix)
    temp_path = None
    
    try:
        temp_path = manager.create_temp_file(original_filename, data)
        yield temp_path
    finally:
        if temp_path:
            manager.cleanup_file(temp_path)


@contextlib.contextmanager
def temp_manager_context(prefix: str = "transcrimate"):
    """
    Gestionnaire de contexte synchrone pour TempFileManager
    
    Args:
        prefix: Préfixe pour les noms de fichiers
        
    Yields:
        TempFileManager: Instance du gestionnaire
    """
    manager = TempFileManager(prefix=prefix)
    try:
        yield manager
    finally:
        manager.cleanup()


@contextlib.asynccontextmanager
async def async_temp_manager_context(prefix: str = "transcrimate"):
    """
    Gestionnaire de contexte asynchrone pour TempFileManager
    
    Args:
        prefix: Préfixe pour les noms de fichiers
        
    Yields:
        TempFileManager: Instance du gestionnaire
    """
    manager = TempFileManager(prefix=prefix)
    try:
        yield manager
    finally:
        manager.cleanup()


@contextlib.asynccontextmanager
async def async_temp_file_context(original_filename: str, data: bytes, prefix: str = "transcrimate"):
    """
    Gestionnaire de contexte asynchrone pour fichier temporaire unique
    
    Args:
        original_filename: Nom original du fichier
        data: Données binaires
        prefix: Préfixe pour le nom du fichier
        
    Yields:
        str: Chemin du fichier temporaire
    """
    manager = TempFileManager(prefix=prefix)
    temp_path = None
    
    try:
        temp_path = manager.create_temp_file(original_filename, data)
        yield temp_path
    finally:
        if temp_path:
            manager.cleanup_file(temp_path)


def get_system_temp_dir() -> str:
    """
    Retourne le répertoire temporaire système cross-platform
    
    Returns:
        str: Chemin du répertoire temporaire
    """
    return tempfile.gettempdir()
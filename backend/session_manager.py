"""
Gestionnaire de sessions pour TranscriMate
Gère les fichiers temporaires par session utilisateur
"""
import os
import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import shutil
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, base_temp_dir: str = "backend/temp", session_timeout_hours: int = 24):
        """
        Initialise le gestionnaire de sessions
        
        Args:
            base_temp_dir: Répertoire de base pour les fichiers temporaires
            session_timeout_hours: Durée de vie d'une session en heures
        """
        self.base_temp_dir = base_temp_dir
        self.session_timeout_hours = session_timeout_hours
        self.sessions: Dict[str, dict] = {}
        self.cleanup_interval = 3600  # Nettoyage toutes les heures
        
        # Créer le répertoire de base s'il n'existe pas
        os.makedirs(base_temp_dir, exist_ok=True)
        
        # Démarrer le thread de nettoyage automatique
        self._start_cleanup_thread()
    
    def create_session(self) -> str:
        """
        Crée une nouvelle session
        
        Returns:
            ID de la session
        """
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(self.base_temp_dir, session_id)
        
        # Créer le répertoire de session
        os.makedirs(session_dir, exist_ok=True)
        
        # Enregistrer la session
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_access': datetime.now(),
            'directory': session_dir,
            'files': []
        }
        
        logger.info(f"Session créée: {session_id}")
        return session_id
    
    def get_session_dir(self, session_id: str) -> str:
        """
        Obtient le répertoire d'une session
        
        Args:
            session_id: ID de la session
            
        Returns:
            Chemin du répertoire de session
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} introuvable")
        
        # Mettre à jour le dernier accès
        self.sessions[session_id]['last_access'] = datetime.now()
        
        return self.sessions[session_id]['directory']
    
    def add_file_to_session(self, session_id: str, filename: str) -> str:
        """
        Ajoute un fichier à une session
        
        Args:
            session_id: ID de la session
            filename: Nom du fichier
            
        Returns:
            Chemin complet du fichier
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} introuvable")
        
        session_dir = self.get_session_dir(session_id)
        file_path = os.path.join(session_dir, filename)
        
        # Ajouter à la liste des fichiers de la session
        self.sessions[session_id]['files'].append(filename)
        
        logger.info(f"Fichier ajouté à la session {session_id}: {filename}")
        return file_path
    
    def get_file_path(self, session_id: str, filename: str) -> str:
        """
        Obtient le chemin complet d'un fichier dans une session
        
        Args:
            session_id: ID de la session
            filename: Nom du fichier
            
        Returns:
            Chemin complet du fichier
        """
        session_dir = self.get_session_dir(session_id)
        return os.path.join(session_dir, filename)
    
    def cleanup_session(self, session_id: str):
        """
        Nettoie une session et ses fichiers
        
        Args:
            session_id: ID de la session à nettoyer
        """
        if session_id not in self.sessions:
            return
        
        session_info = self.sessions[session_id]
        session_dir = session_info['directory']
        
        try:
            # Supprimer tous les fichiers du répertoire
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            
            # Supprimer la session de la mémoire
            del self.sessions[session_id]
            
            logger.info(f"Session nettoyée: {session_id}")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage de la session {session_id}: {e}")
    
    def cleanup_expired_sessions(self):
        """
        Nettoie les sessions expirées
        """
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session_info in self.sessions.items():
            last_access = session_info['last_access']
            if now - last_access > timedelta(hours=self.session_timeout_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Nettoyage de la session expirée: {session_id}")
            self.cleanup_session(session_id)
        
        # Nettoyer aussi les répertoires orphelins
        self._cleanup_orphaned_directories()
    
    def _cleanup_orphaned_directories(self):
        """
        Nettoie les répertoires qui ne correspondent à aucune session active
        """
        try:
            if not os.path.exists(self.base_temp_dir):
                return
            
            for item in os.listdir(self.base_temp_dir):
                item_path = os.path.join(self.base_temp_dir, item)
                
                # Ignorer les fichiers (ne nettoyer que les répertoires)
                if not os.path.isdir(item_path):
                    continue
                
                # Si le répertoire ne correspond à aucune session active
                if item not in self.sessions:
                    # Vérifier si c'est un UUID valide (format de session)
                    try:
                        uuid.UUID(item)
                        logger.info(f"Suppression du répertoire orphelin: {item}")
                        shutil.rmtree(item_path)
                    except ValueError:
                        # Ce n'est pas un UUID, probablement un autre type de fichier
                        pass
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des répertoires orphelins: {e}")
    
    def _start_cleanup_thread(self):
        """
        Démarre le thread de nettoyage automatique
        """
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Erreur dans le thread de nettoyage: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Thread de nettoyage automatique démarré")
    
    def get_session_info(self, session_id: str) -> dict:
        """
        Obtient les informations d'une session
        
        Args:
            session_id: ID de la session
            
        Returns:
            Informations de la session
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} introuvable")
        
        session_info = self.sessions[session_id].copy()
        
        # Ajouter des informations calculées
        session_info['age_hours'] = (datetime.now() - session_info['created_at']).total_seconds() / 3600
        session_info['file_count'] = len(session_info['files'])
        
        # Calculer la taille totale des fichiers
        total_size = 0
        session_dir = session_info['directory']
        if os.path.exists(session_dir):
            for filename in session_info['files']:
                file_path = os.path.join(session_dir, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        session_info['total_size_mb'] = total_size / (1024 * 1024)
        
        return session_info
    
    def list_sessions(self) -> Dict[str, dict]:
        """
        Liste toutes les sessions actives avec leurs informations
        
        Returns:
            Dictionnaire des sessions avec leurs informations
        """
        sessions_info = {}
        for session_id in self.sessions:
            try:
                sessions_info[session_id] = self.get_session_info(session_id)
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des infos de session {session_id}: {e}")
        
        return sessions_info

# Instance globale du gestionnaire de sessions
session_manager = SessionManager()
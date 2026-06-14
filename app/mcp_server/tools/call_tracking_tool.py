"""
call Tracking Tool — Call Logger automatique.

Ce tool N'EST PAS dans le registre MCP et N'EST JAMAIS
choisi par Gemini. Il se déclenche automatiquement dans
api.py après chaque réponse du bot.

Rôle unique : écrire dans la table call_logs via repository.log_call()
"""

import logging
import time
from typing import Optional

from app.database import repository as repo

logger = logging.getLogger(__name__)


class CallLogger:
    """
    Logger automatique de chaque interaction voicebot.
    Instancié une seule fois dans api.py (comme MCPServer).
    """

    def log(
        self,
        session_id: str,
        transcript: str,
        tool_utilise: str,
        reponse: str,
        duree_ms: int,
        succes: bool = True
    ) -> None:
        """
        Enregistre une interaction dans call_logs.

        Paramètres :
          session_id   : identifiant WebSocket (str(id(websocket)))
          transcript   : ce que le client a dit (Darija)
          tool_utilise : "rag_search" ou "ctm_db_query"
          reponse      : réponse finale envoyée au client
          duree_ms     : temps total de traitement en millisecondes
          succes       : False si une erreur technique s'est produite

        Cette méthode est silencieuse — elle ne lève jamais d'exception
        pour ne pas bloquer la réponse au client.
        """
        repo.log_call(
            session_id=session_id,
            transcript=transcript,
            tool_utilise=tool_utilise,
            reponse=reponse,
            duree_ms=duree_ms,
            succes=succes
        )


# Singleton — une seule instance dans tout le projet
_call_logger_instance: Optional[CallLogger] = None


def get_call_logger() -> CallLogger:
    """Retourne l'instance unique du CallLogger."""
    global _call_logger_instance
    if _call_logger_instance is None:
        _call_logger_instance = CallLogger()
        logger.info("✅ CallLogger initialisé")
    return _call_logger_instance

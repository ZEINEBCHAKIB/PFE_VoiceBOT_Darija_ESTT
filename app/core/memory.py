"""
Gestion de la mémoire conversationnelle du VoiceBot.
Chaque session WebSocket a sa propre instance ConversationMemory.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Mémoire d'une session de conversation.
    Stocke l'historique des échanges user/assistant.
    """

    def __init__(self, max_turns: int = 10, session_id: Optional[str] = None):
        """
        Args:
            max_turns: Nombre maximum de tours à conserver (1 tour = 1 user + 1 assistant)
            session_id: Identifiant optionnel de la session
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_turns = max_turns
        self._history: List[Dict] = []
        logger.info(f"🧠 Mémoire initialisée — session: {self.session_id}")

    # ──────────────────────────────────────────────
    # Ajout de messages
    # ──────────────────────────────────────────────

    def add_user_message(self, text: str):
        """Ajoute un message de l'utilisateur"""
        self._history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        self._truncate()
        logger.debug(f"👤 User ajouté — historique: {len(self._history)} messages")

    def add_assistant_message(self, text: str):
        """Ajoute une réponse de l'assistant"""
        self._history.append({
            "role": "assistant",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        self._truncate()
        logger.debug(f"🤖 Assistant ajouté — historique: {len(self._history)} messages")

    def add_turn(self, user_text: str, assistant_text: str):
        """Ajoute un tour complet (user + assistant) en une seule opération"""
        self.add_user_message(user_text)
        self.add_assistant_message(assistant_text)

    # ──────────────────────────────────────────────
    # Accès à l'historique
    # ──────────────────────────────────────────────

    def get_history(self) -> List[Dict]:
        """Retourne l'historique complet"""
        return self._history.copy()

    def get_history_for_llm(self) -> List[Dict]:
        """
        Retourne l'historique formaté pour le LLM (sans timestamps).
        Format: [{"role": "user"/"assistant", "content": "..."}]
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._history
        ]

    def format_as_text(self) -> str:
        """
        Formate l'historique en texte lisible pour l'injection dans un prompt.
        """
        if not self._history:
            return ""

        lines = ["HISTORIQUE DE LA CONVERSATION:"]
        for msg in self._history:
            role = "Client" if msg["role"] == "user" else "Agent CTM"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines) + "\n"

    def get_last_n_turns(self, n: int) -> List[Dict]:
        """Retourne les n derniers tours"""
        messages = self._history[-(n * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # ──────────────────────────────────────────────
    # Gestion
    # ──────────────────────────────────────────────

    def _truncate(self):
        """Garde uniquement les max_turns derniers tours"""
        max_messages = self.max_turns * 2  # 1 tour = user + assistant
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
            logger.debug(f"✂️ Historique tronqué à {max_messages} messages")

    def clear(self):
        """Vide l'historique (nouvelle conversation)"""
        self._history.clear()
        logger.info(f"🗑️ Mémoire vidée — session: {self.session_id}")

    def is_empty(self) -> bool:
        return len(self._history) == 0

    def turn_count(self) -> int:
        return len(self._history) // 2

    def __repr__(self):
        return (
            f"ConversationMemory("
            f"session={self.session_id}, "
            f"turns={self.turn_count()}/{self.max_turns})"
        )


def get_memory(max_turns: int = 10, session_id: Optional[str] = None) -> ConversationMemory:
    """Factory — crée une nouvelle instance de mémoire par session"""
    return ConversationMemory(max_turns=max_turns, session_id=session_id)
"""
Gestion du LLM (OpenAI) pour le RAG
"""
import logging
import os
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Client pour OpenAI API"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY manquante dans les variables d'environnement")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self._initialized = True
        logger.info(f"✅ OpenAI {self.model} configuré")

    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Génère une réponse avec OpenAI"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Erreur OpenAI après {max_retries} tentatives: {e}")
                    return None
        return None

    def rag_generate(self, query: str, context: str) -> str:
        """Génère une réponse RAG en darija"""
        prompt = f"""Tu es un assistant du centre d'appel CTM (transport et logistique au Maroc).
Tu dois répondre en darija marocaine UNIQUEMENT en utilisant les informations du contexte ci-dessous.

RÈGLES IMPORTANTES:
1. Réponds UNIQUEMENT en darija marocaine (lettres arabes pures)
2. Utilise un ton professionnel et courtois comme un call center
3. Sois concis (2-3 phrases maximum)
4. Si l'information n'est pas dans le contexte, dis: Désolé، ما عنديش هاد المعلومة

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE EN DARIJA MAROCAINE:"""

        response = self.generate(prompt)
        return response or "Désolé، وقع مشكل تقني."

    def translate_to_french(self, darija_query: str) -> str:
        """Traduit une requête darija vers français"""
        prompt = f"""Traduis la phrase suivante du darija marocain vers le français.
Réponds UNIQUEMENT par la traduction, sans aucune explication.

Phrase en darija:
{darija_query}

Traduction française:"""

        response = self.generate(prompt)
        if response:
            return response.strip().strip('"').strip("'")
        return darija_query


def get_llm_client() -> LLMClient:
    """Obtenir l'instance du client LLM"""
    return LLMClient()

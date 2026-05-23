"""
Gestion du LLM (Gemini) pour le RAG
"""

import logging
import time
from typing import Optional, Dict, Any

import google.generativeai as genai

from app.config.settings import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client pour Gemini API"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self._initialized = True
        logger.info(f"✅ Gemini {config.GEMINI_MODEL} configuré")
    
    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Génère une réponse avec Gemini
        
        Args:
            prompt: Prompt à envoyer
            max_retries: Nombre de tentatives en cas d'erreur
        
        Returns:
            Réponse générée ou None en cas d'erreur
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Tentative {attempt+1}/{max_retries} échouée: {e}")
                if "quota" in str(e).lower():
                    wait_time = 60
                    logger.info(f"Quota dépassé, pause de {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Erreur Gemini après {max_retries} tentatives: {e}")
                    return None
        
        return None
    
    def rag_generate(self, query: str, context: str) -> str:
        """
        Génère une réponse RAG en darija
        
        Args:
            query: Question en français
            context: Contexte extrait des documents
        
        Returns:
            Réponse en darija
        """
        prompt = f"""Tu es un assistant du centre d'appel CTM (transport et logistique au Maroc).
Tu dois répondre en darija marocaine UNIQUEMENT en utilisant les informations du contexte ci-dessous.

RÈGLES IMPORTANTES:
1. Réponds UNIQUEMENT en darija marocaine (lettres arabes pures)
2. Utilise un ton professionnel et courtois comme un call center
3. Sois concis (2-3 phrases maximum)
4. Si l'information n'est pas dans le contexte, dis "Désolé, je n'ai pas cette information"

CONTEXTE:
{context}

QUESTION: {query}

RÉPONSE EN DARIJA MAROCAINE:"""
        
        response = self.generate(prompt)
        return response or "Désolé, une erreur technique s'est produite."
    
    def translate_to_french(self, darija_query: str) -> str:
        """
        Traduit une requête darija vers français
        
        Args:
            darija_query: Requête en darija
        
        Returns:
            Traduction en français
        """
        prompt = f"""Traduis la phrase suivante du darija marocain vers le français.
Réponds UNIQUEMENT par la traduction, sans aucune explication.

Phrase en darija: {darija_query}

Traduction française:"""
        
        response = self.generate(prompt)
        
        if response:
            # Nettoyage de la réponse
            translation = response.strip()
            # Enlever les guillemets si présents
            translation = translation.strip('"').strip("'")
            return translation
        
        return darija_query  # Fallback: retourner la requête originale


def get_llm_client() -> LLMClient:
    """Obtenir l'instance du client LLM (singleton)"""
    return LLMClient()
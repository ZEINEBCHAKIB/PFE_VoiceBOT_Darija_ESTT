"""
Gestion du LLM (Google Gemini) pour le RAG
3 clés API indépendantes — une par usage
"""
import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class LLMClient:
    """Client Gemini avec 3 clés API indépendantes"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # ── Clé 1 : Traduction ──
        key_translation = os.getenv("GEMINI_API_KEY_TRANSLATION")
        if not key_translation:
            raise ValueError("❌ GEMINI_API_KEY_TRANSLATION manquante")
        self.client_translation = genai.Client(api_key=key_translation)

        # ── Clé 2 : Génération RAG ──
        key_generation = os.getenv("GEMINI_API_KEY_GENERATION")
        if not key_generation:
            raise ValueError("❌ GEMINI_API_KEY_GENERATION manquante")
        self.client_generation = genai.Client(api_key=key_generation)

        self.model = "gemini-3.1-flash-lite"
        self._initialized = True
        logger.info(f"✅ LLMClient initialisé — 2 clés (traduction + génération)")

    def _call(self, client: genai.Client, prompt: str, temperature: float = 0.3, max_retries: int = 3) -> Optional[str]:
        """Méthode interne générique pour appeler un client Gemini"""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature)
                )
                return response.text
            except Exception as e:
                logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Erreur après {max_retries} tentatives: {e}")
                    return None
        return None

    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Génération générale — utilise client_generation"""
        return self._call(self.client_generation, prompt, max_retries=max_retries)

    def translate_to_french(self, darija_query: str) -> str:
        """Traduit darija → français — utilise GEMINI_API_KEY_TRANSLATION"""
        prompt = f"""Traduis la phrase suivante du darija marocain vers le français.
Réponds UNIQUEMENT par la traduction, sans aucune explication.

Phrase en darija:
{darija_query}

Traduction française:"""

        response = self._call(self.client_translation, prompt, temperature=0.1)
        if response:
            return response.strip().strip('"').strip("'")
        return darija_query


    def rag_generate_with_history(self, query: str, context: str, history_text: str = "") -> str:
        is_first_turn = len(history_text.strip()) == 0
        prompt = f"""Tu es un agent professionnel du centre d'appel CTM (transport au Maroc).
Réponds en darija marocaine (lettres arabes uniquement).

RÈGLES:
1. Réponds UNIQUEMENT en darija marocaine
2. Ton professionnel et courtois comme un vrai agent call center
3. Tiens compte de l'historique pour une conversation naturelle
4. Si l'info manque: سمحلي، ما عنديش هاد المعلومة
5. {"يمكنك البدء بتحية قصيرة" if is_first_turn else "⚠️ NE PAS commencer par مرحبا ou أهلا — continue la conversation directement"}
6. Ne répète JAMAIS une salutation si elle existe déjà dans l'historique
7. LANGUE : Réponds UNIQUEMENT en Darija écrite en alphabet arabe. N'utilise JAMAIS d'arabe classique (Fusha).
8. PHONÉTIQUE ET TTS : Ton texte va être lu par une synthèse vocale. Fais des phrases courtes. Évite les répétitions de lettres inutiles et n'utilise pas de voyelles courtes (Tachkeel/Harakat) sauf si c'est indispensable pour lever une ambiguïté de prononciation.
9. TON : Sois naturel, poli et utilise des expressions marocaines courantes et accueillantes (ex: "مرحبا", "شنو حب الخاطر").
10. VOCABULAIRE : Si tu dois utiliser des mots technologiques ou modernes, utilise l'équivalent le plus naturel en Darija (ex: "صيفط ميساج" au lieu de "أرسل رسالة").
{history_text}
CONTEXTE CTM:
{context}

QUESTION ACTUELLE:
{query}

RÉPONSE EN DARIJA:"""

        response = self._call(self.client_generation, prompt, temperature=0.3)
        return response or "سمحلي، وقع مشكل تقني."

    def generate_welcome(self) -> str:
        """Génère un message de bienvenue naturel et varié"""
        prompt = """Tu es un agent du centre d'appel CTM (transport au Maroc).
Génère UN message de bienvenue court et naturel en darija marocaine (lettres arabes).

RÈGLES:
1. Varie le message à chaque fois (ne répète pas toujours la même phrase)
2. Sois chaleureux et professionnel comme un vrai agent call center
3. Maximum 2 phrases
4. Mentionne CTM naturellement
5. Termine par une invitation à parler

Exemples de variations possibles:
- صباح الخير، أهلا بيك في CTM، كيفاش نقدر نخدمك؟
- مرحبا، وصلتي لـ CTM، أنا هنا باش نساعدك، واش عندك شي سؤال؟
- أهلا وسهلا في CTM، كيف نقدر نكون مفيد ليك اليوم؟

GÉNÈRE UN NOUVEAU MESSAGE (différent des exemples):"""

        response = self._call(self.client_generation, prompt, temperature=0.9)
        return response or "مرحبا بك في CTM، كيفاش نقدر نساعدك؟"

def get_llm_client() -> LLMClient:
    """Retourne l'instance unique du client LLM"""
    return LLMClient()
"""
Orchestrateur principal du VoiceBot CTM Darija.

Pipeline:
    Audio → ASR → Orchestrator → MCP (→ RAG) → LLM → Response (→ TTS futur)

Règles d'architecture:
    - L'orchestrateur n'accède JAMAIS directement au RAG
    - Toute récupération de knowledge passe par MCP
    - ASR = transcription uniquement
    - LLM reçoit uniquement: transcript + contexte MCP
"""

import logging
from typing import Optional

from app.config.settings import config
from app.mcp_server.server import get_mcp_server
from app.core.llm import get_llm_client

logger = logging.getLogger(__name__)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)


class VoiceBotOrchestrator:
    """
    Orchestrateur principal — coordonne le pipeline complet.

    Composants:
        - ASR: audio → texte (chargé à la demande)
        - MCP: gateway vers les tools (RAG, etc.)
        - LLM: génération de réponse
        - TTS: texte → audio (futur, placeholder)

    ⚠️ Aucun import de app.rag.* ici — RAG via MCP uniquement
    """

    def __init__(self):
        self.mcp = get_mcp_server()
        self.llm = get_llm_client()
        self._asr = None  # Lazy loading — pas besoin si mode texte
        logger.info("✅ VoiceBot Orchestrateur initialisé")

    @property
    def asr(self):
        """Lazy loading de l'ASR — chargé uniquement si audio reçu"""
        if self._asr is None:
            from app.core.asr import get_asr_engine
            self._asr = get_asr_engine()
        return self._asr

    def process_audio(self, audio_path: str) -> str:
        """
        Pipeline complet: Audio → ASR → MCP → LLM → Réponse

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Réponse en Darija (texte)
        """
        logger.info(f"🎙️ Pipeline audio démarré: {audio_path}")

        # ── Étape 1: ASR (transcription audio → texte) ──
        transcript = self.asr.transcribe(audio_path)
        logger.info(f"📝 Transcript ASR: {transcript[:100]}...")

        # ── Étape 2-4: Pipeline texte ──
        return self.process_text(transcript)

    def process_text(self, text: str) -> str:
        """
        Pipeline texte: Texte → MCP → LLM → Réponse
        (Même pipeline que process_audio, sans l'étape ASR)

        Args:
            text: Texte d'entrée (Darija ou Français)

        Returns:
            Réponse en Darija (texte)
        """
        logger.info(f"💬 Pipeline texte démarré: {text[:100]}...")

        # ── Étape 2: MCP → RAG (récupération contexte) ──
        mcp_result = self.mcp.call_tool(text)

        if not mcp_result["success"]:
            logger.error(f"❌ Erreur MCP: {mcp_result.get('error')}")
            return "عذراً، وقع مشكل تقني. عاود حاول من بعد."

        response = mcp_result["data"].get("answer", "سمحلي، ما لقيتش معلومات على هاد السؤال.")

        logger.info(f"✅ Réponse générée ({len(response)} chars)")

        # ── Étape 5 (futur): TTS ──
        # response_audio = self.tts.synthesize(response)

        return response

    def get_system_status(self) -> dict:
        """Status du système complet"""
        status = {
            "orchestrator": "ok",
            "mcp_tools": [t["name"] for t in self.mcp.list_tools()],
            "asr_loaded": self._asr is not None,
            "llm": "ok"
        }

        try:
            rag_stats = self.mcp.call_tool("عطيني الإحصائيات")
            if rag_stats["success"]:
                status["rag"] = rag_stats["data"]
        except Exception:
            status["rag"] = "error"

        return status


# ──────────────────────────────────────────────
# Point d'entrée CLI
# ──────────────────────────────────────────────

def main():
    """Mode interactif CLI pour tester le pipeline"""
    print("=" * 60)
    print("🤖 VoiceBot CTM Darija")
    print("   Pipeline: Texte → MCP → RAG → LLM → Réponse")
    print("=" * 60)
    print("Tapez 'quit' pour quitter, 'status' pour le statut système\n")

    bot = VoiceBotOrchestrator()

    while True:
        try:
            user_input = input("👤 Vous: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Au revoir!")
                break

            if user_input.lower() == "status":
                status = bot.get_system_status()
                print(f"\n📊 Statut système:")
                for key, value in status.items():
                    print(f"   {key}: {value}")
                print()
                continue

            # Pipeline texte
            response = bot.process_text(user_input)
            print(f"\n🤖 Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")
            print(f"\n❌ Erreur: {e}\n")


if __name__ == "__main__":
    main()

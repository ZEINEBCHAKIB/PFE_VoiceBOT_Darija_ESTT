"""
Orchestrateur principal du VoiceBot CTM Darija.

Pipeline:
    Audio → ASR → Orchestrator → MCP Host → MCP Servers → LLM → Réponse

Architecture MCP (vraie) :
    [Orchestrator]
         ↓
    [MCP Host — client MCP + Gemini routing]
         ↓ JSON-RPC 2.0 sur stdio
    [MCP Server RAG]   [MCP Server DB]    ← subprocess indépendants
         ↓                    ↓
    [RAGTool]           [CTMDatabaseTool]

Règles d'architecture:
    - L'orchestrateur n'accède JAMAIS directement au RAG
    - Toute récupération de knowledge passe par MCP (protocole officiel)
    - ASR = transcription uniquement
"""

import asyncio
import logging
from typing import Optional

from app.config.settings import config
from app.mcp_host import get_mcp_host
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
        - MCP Host: client MCP qui parle aux servers RAG/DB via JSON-RPC
        - LLM: génération de réponse
        - TTS: texte → audio (futur, placeholder)

    ⚠️ Aucun import de app.rag.* ici — RAG via MCP uniquement
    """

    def __init__(self):
        # MCP Host — lazy-start des subprocess au 1er call_tool
        self.mcp = get_mcp_host()
        self.llm = get_llm_client()
        self._asr = None  # Lazy loading — pas besoin si mode texte
        logger.info("✅ VoiceBot Orchestrateur initialisé (MCP Host configuré)")

    @property
    def asr(self):
        """Lazy loading de l'ASR — chargé uniquement si audio reçu"""
        if self._asr is None:
            from app.core.asr import get_asr_engine
            self._asr = get_asr_engine()
        return self._asr

    async def process_audio(self, audio_path: str) -> str:
        """
        Pipeline complet: Audio → ASR → MCP → LLM → Réponse

        Args:
            audio_path: Chemin vers le fichier audio

        Returns:
            Réponse en Darija (texte)
        """
        logger.info(f"🎙️ Pipeline audio démarré: {audio_path}")

        # ── Étape 1: ASR (transcription audio → texte) ──
        transcript = await self.asr.transcribe(audio_path)
        logger.info(f"📝 Transcript ASR: {transcript[:100]}...")

        # ── Étape 2-4: Pipeline texte ──
        return await self.process_text(transcript)

    async def process_text(self, text: str) -> str:
        """
        Pipeline texte: Texte → MCP Host → MCP Server → Réponse
        (MCP Host route via Gemini function_calling, exécute via JSON-RPC)

        Args:
            text: Texte d'entrée (Darija ou Français)

        Returns:
            Réponse en Darija (texte)
        """
        logger.info(f"💬 Pipeline texte démarré: {text[:100]}...")

        # ── Étape 2: MCP Host → MCP Server (RAG ou DB) ──
        mcp_result = await self.mcp.call_tool(text)

        if not mcp_result["success"]:
            logger.error(f"❌ Erreur MCP: {mcp_result.get('error')}")
            return "عذراً، وقع مشكل تقني. عاود حاول من بعد."

        response = mcp_result["data"].get("answer", "سمحلي، ما لقيتش معلومات على هاد السؤال.")

        logger.info(f"✅ Réponse générée ({len(response)} chars)")

        # ── Étape 5 (futur): TTS ──
        # response_audio = self.tts.synthesize(response)

        return response

    async def get_system_status(self) -> dict:
        """Status du système complet — inclut les MCP servers découverts"""
        status = {
            "orchestrator": "ok",
            "asr_loaded": self._asr is not None,
            "llm": "ok"
        }

        try:
            # list_tools() interroge les MCP servers via protocole MCP
            tools = await self.mcp.list_tools()
            status["mcp_tools"] = [
                f"{t['name']} (server: {t['server']})" for t in tools
            ]

            # Stats RAG via MCP call_tool
            rag_stats = await self.mcp.call_tool("عطيني الإحصائيات")
            if rag_stats["success"]:
                status["rag"] = rag_stats["data"]
        except Exception as e:
            status["mcp_tools"] = f"error: {e}"
            status["rag"] = "error"

        return status

    async def cleanup(self):
        """Ferme proprement les subprocess MCP — à appeler avant exit"""
        try:
            await self.mcp.stop()
        except Exception as e:
            logger.warning(f"⚠️ Erreur cleanup MCP: {e}")


# ──────────────────────────────────────────────
# Point d'entrée CLI
# ──────────────────────────────────────────────

async def _async_main():
    """Boucle CLI asynchrone — nécessaire car MCP SDK est async-only"""
    print("=" * 60)
    print("🤖 VoiceBot CTM Darija — Architecture MCP (vrai protocole)")
    print("   Pipeline: Texte → MCP Host → MCP Servers → LLM → Réponse")
    print("=" * 60)
    print("Tapez 'quit' pour quitter, 'status' pour le statut système\n")

    bot = VoiceBotOrchestrator()

    try:
        while True:
            try:
                # input() est bloquant — exécuté dans un thread via run_in_executor
                # pour ne pas bloquer la event loop
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("👤 Vous: ").strip()
                )

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    print("👋 Au revoir!")
                    break

                if user_input.lower() == "status":
                    status = await bot.get_system_status()
                    print(f"\n📊 Statut système:")
                    for key, value in status.items():
                        print(f"   {key}: {value}")
                    print()
                    continue

                # Pipeline texte — async car MCP est async
                response = await bot.process_text(user_input)
                print(f"\n🤖 Bot: {response}\n")

            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                logger.error(f"Erreur: {e}", exc_info=True)
                print(f"\n❌ Erreur: {e}\n")
    finally:
        # Cleanup propre des subprocess MCP
        await bot.cleanup()


def main():
    """Wrapper sync → async pour le CLI"""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

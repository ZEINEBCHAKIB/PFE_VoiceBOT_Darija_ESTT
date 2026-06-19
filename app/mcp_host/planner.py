"""
Planner Agent — décide quels tools appeler pour une question donnée.

Contrairement à l'ancien routing mono-tool, le Planner peut retourner
PLUSIEURS plans d'exécution (ex: RAG + DB pour une question multi-source).

Le Planner ne génère PAS la réponse finale — il choisit juste les tools.
C'est le Synthesizer (synthesizer.py) qui fusionnera les résultats.

Architecture multi-tools :
    Question → Planner → [plan1, plan2, ...] → Executor → Synthesizer → Réponse
"""

import logging
import os
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types

from app.core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Planner — analyse la question et retourne une liste de plans.

    Un plan = {"name": str, "args": dict}

    Cas possibles :
      - 0 plan  → réponse directe (salutation, remerciement)
      - 1 plan  → question simple (mono-tool)
      - N plans → question composée (multi-tools, exécutés en parallèle)
    """

    def __init__(self, client: Optional[genai.Client] = None, model: Optional[str] = None):
        """
        Args:
            client: client Gemini existant (réutilise celui du MCPHost pour éviter
                    de créer plusieurs clients)
            model:  modèle Gemini à utiliser (default: gemini-2.0-flash-lite)
        """
        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY_ROUTER")
            if not api_key:
                raise ValueError("❌ GEMINI_API_KEY_ROUTER manquante")
            self.client = genai.Client(api_key=api_key)

        # Flash-lite suffit pour du routing/planning
        self.model = model or os.getenv("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")

    # ──────────────────────────────────────────────
    # Prompt système — explicite les règles de décision
    # ──────────────────────────────────────────────

    def _build_system_instruction(self) -> str:
        return (
            "Tu es le Planner Agent d'un call center CTM (transport au Maroc).\n\n"
            "═══ TON RÔLE ═══\n"
            "Analyser la question de l'utilisateur (en Darija marocain ou français) et décider "
            "quels tools appeler pour récupérer les informations nécessaires.\n\n"
            "Tu NE génères PAS la réponse finale. Tu choisis seulement les tools.\n\n"
            "═══ TOOLS DISPONIBLES ═══\n"
            "Tu reçois la liste des tools via function_calling. Voici leur usage :\n"
            "  - rag_search(query, history_text)        → questions générales, procédures,\n"
            "                                              politiques (bagages, annulation, FAQ)\n"
            "  - ctm_db_query(action, ...)              → infos précises : horaires, tarifs,\n"
            "                                              agences, réclamations, suivi colis\n"
            "    actions possibles :\n"
            "      • horaires            (ville_depart, ville_arrivee)\n"
            "      • tarifs              (ville_depart, ville_arrivee)\n"
            "      • agence              (ville)\n"
            "      • reclamation_get     (reference | telephone)\n"
            "      • reclamation_create  (telephone, description)\n"
            "      • colis               (numero_suivi | telephone)\n"
            "  - rag_stats()                            → statistiques du système RAG (rare)\n\n"
            "═══ RÈGLES DE DÉCISION ═══\n\n"
            "1. MULTI-SOURCES :\n"
            "   Si la question contient plusieurs sous-questions séparées par 'و' (et),\n"
            "   'puis', 'et aussi', ou une virgule → appelle TOUS les tools pertinents\n"
            "   en parallèle dans la même réponse.\n\n"
            "   Exemples :\n"
            "   • 'مواعيد الرباط-طنجة و شروط الإلغاء'  → ctm_db_query(horaires) + rag_search(annulation)\n"
            "   • 'ثمن تذكرة كازا-فاس و واش كاينين وكالات فمكناس' → ctm_db_query(tarifs) + ctm_db_query(agence)\n\n"
            "2. MONO-SOURCE :\n"
            "   Si une seule info est demandée → appelle 1 seul tool.\n\n"
            "3. SALUTATIONS / GÉNÉRAL :\n"
            "   Si la question est 'سلام', 'شكرا', 'بغني ندير شي حاجة' sans info précise\n"
            "   → n'appelle AUCUN tool. Réponds juste en texte libre.\n\n"
            "4. CONTEXTE / ELLIPSES :\n"
            "   Si la question est courte ('الرباط؟', 'وشحال؟'), utilise l'historique fourni\n"
            "   pour comprendre. Ex: si historique = 'tarifs casa-rabat' et question = 'وكونفور؟'\n"
            "   → appelle ctm_db_query(tarifs, casa, rabat) — la nouvelle question complète\n"
            "   l'ancienne.\n\n"
            "5. PRIORITÉ DB sur RAG :\n"
            "   Pour les horaires, tarifs, agences, réclamations, colis → TOUJOURS ctm_db_query\n"
            "   (la DB contient les infos exactes, le RAG contient les procédures).\n"
            "   Pour les procédures, conditions, FAQ → rag_search.\n\n"
            "═══ FORMAT DE RÉPONSE ═══\n"
            "Appelle les tools via function_calling. Tu peux appeler plusieurs tools\n"
            "dans la même réponse — ils seront exécutés en parallèle.\n\n"
            "Ne génère PAS de texte explicatif, juste les function_calls.\n"
        )

    def _build_contents(
        self,
        user_query: str,
        memory: Optional[ConversationMemory]
    ) -> str:
        """Construit le contenu envoyé à Gemini (query + historique)."""
        if memory and not memory.is_empty():
            return (
                f"HISTORIQUE:\n{memory.format_as_text()}\n\n"
                f"QUESTION ACTUELLE: {user_query}"
            )
        return user_query

    # ──────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────

    async def plan(
        self,
        user_query: str,
        memory: Optional[ConversationMemory],
        gemini_tools: List[types.FunctionDeclaration]
    ) -> List[Dict[str, Any]]:
        """
        Analyse la question et retourne une liste de plans.

        Returns:
            Liste de plans : [{"name": str, "args": dict}, ...]
            - Liste vide  → réponse directe (sera remplie par le Synthesizer)
            - 1 plan      → question simple
            - N plans     → question composée (multi-tools)

        Cas spécial : si la question est une salutation et Gemini répond en texte,
        on retourne [{"name": "__direct__", "args": {"text": <réponse>}}]
        """
        contents = self._build_contents(user_query, memory)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=gemini_tools)],
                    temperature=0.1,
                    system_instruction=self._build_system_instruction()
                )
            )
        except Exception as e:
            logger.error(f"❌ Planner Gemini call échoué: {e}", exc_info=True)
            # Fallback sûr : 1 tool RAG avec la question brute
            return [{"name": "rag_search", "args": {"query": user_query}}]

        if not response.candidates:
            logger.warning("⚠️ Planner : aucun candidat retourné — fallback rag_search")
            return [{"name": "rag_search", "args": {"query": user_query}}]

        # ── Collecte tous les function_calls dans la réponse ──
        plans: List[Dict[str, Any]] = []
        direct_text = ""

        for part in response.candidates[0].content.parts:
            # Cas A : function_call (1 ou plusieurs)
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                plan = {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {}
                }
                plans.append(plan)
                logger.info(f"📋 Planner a plannifié : {fc.name}({plan['args']})")

            # Cas B : texte direct (salutation, général)
            elif hasattr(part, "text") and part.text:
                direct_text += part.text

        # ── Décision finale ──
        if plans:
            # 1 ou plusieurs tools à appeler
            logger.info(f"✅ Planner retourne {len(plans)} plan(s)")
            return plans

        if direct_text:
            # Salutation / question générale sans tool — on garde le texte
            logger.info(f"💬 Planner : réponse directe (pas de tool) — '{direct_text[:50]}...'")
            return [{"name": "__direct__", "args": {"text": direct_text}}]

        # Aucun function_call, aucun texte → fallback RAG
        logger.warning("⚠️ Planner : réponse vide — fallback rag_search")
        return [{"name": "rag_search", "args": {"query": user_query}}]

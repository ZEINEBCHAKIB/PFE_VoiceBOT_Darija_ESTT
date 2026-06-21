"""
Synthesizer Agent — fusionne les résultats de plusieurs tools
en une seule réponse cohérente en Darija marocain.

Le Synthesizer est appelé APRÈS l'Executor (qui a exécuté les plans du Planner).
Il reçoit la question originale + les N résultats tools et génère UNE réponse.

Architecture multi-tools :
    Question → Planner → [plan1, plan2, ...] → Executor → Synthesizer → Réponse

⚠️ Stratégie anti-quota :
    - 1 tool avec answer → COURT-CIRCUIT (pas de LLM call)
    - 0 tool → réponse d'accueil (pas de LLM call)
    - N tools → LLM call (mais avec le modèle du Planner qui a du quota)
"""

import logging
import os
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class SynthesizerAgent:
    """
    Synthesizer — prend la question + N résultats tools
    et génère UNE réponse fusionnée en Darija.

    Cas particuliers :
      - 0 résultat tool + 1 direct_text → renvoie direct_text tel quel
      - 1 résultat tool avec champ "answer" → court-circuit (pas de LLM call)
      - N résultats tools → LLM call pour fusionner
    """

    def __init__(self, client: Optional[genai.Client] = None, model: Optional[str] = None):
        """
        Args:
            client: client Gemini existant (réutilise celui du MCPHost)
            model:  modèle Gemini — par défaut on réutilise GEMINI_ROUTER_MODEL
                    (le même que le Planner) pour éviter d'épuiser un autre quota.
                    Override via GEMINI_SYNTHESIZER_MODEL si tu veux un modèle
                    plus puissant (ex: gemini-2.0-flash).
        """
        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY_ROUTER")
            if not api_key:
                raise ValueError("❌ GEMINI_API_KEY_ROUTER manquante")
            self.client = genai.Client(api_key=api_key)

        # ⚠️ Par défaut : on utilise le MÊME modèle que le Planner
        # (évite d'épuiser le quota d'un autre modèle inutilement)
        # Si tu veux un modèle plus puissant, définis GEMINI_SYNTHESIZER_MODEL
        self.model = (
            model
            or os.getenv("GEMINI_SYNTHESIZER_MODEL")
            or os.getenv("GEMINI_ROUTER_MODEL", "gemini-3.1-flash-lite")
        )

    # ──────────────────────────────────────────────
    # Prompt système
    # ──────────────────────────────────────────────

    def _build_system_instruction(self) -> str:
        return (
            "Tu es le Synthesizer Agent du call center CTM (Maroc). Génère UNE SEULE réponse courte (2-3 phrases max) en Darija marocain fluide à partir des outils.\n"
            "Tiens compte de l'historique des messages pour une conversation naturelle .\n"
            "RÈGLES CRUCIALES POUR LE TTS :\n"
            "1. Écris 'CTM' en texte. Pas de listes à puces ni tirets (fais des phrases liées par 'و' ou 'من بعد').\n"
            "2. Prix : Écris 'درهم' en toutes lettres après le chiffre (ex: 90 درهم), JAMAIS 'DH'.\n"
            "3. Horaires/Villes : Utilise le langage oral naturel (ex: 'مع 10 د الصباح', 'كازا', 'مراكش').\n"
            "4. Zéro Fusha/Français : Pas de mots comme 'هناك', 'لأن', 'رحلة'. Utilise 'كاين', 'علاش', 'كار'.\n"
            "Si salutation ou aucun outil : réponds brièvement et poliment en Darija."
        )

    def _build_user_prompt(
        self,
        original_query: str,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """Construit le prompt utilisateur avec les résultats des tools."""
        parts = [f"QUESTION DE L'UTILISATEUR:\n{original_query}\n"]

        if not tool_results:
            parts.append("\nAucun tool n'a été appelé. Réponds directement.")
            return "\n".join(parts)

        parts.append(f"\nRÉSULTATS DES {len(tool_results)} TOOL(S) APPELÉ(S):\n")
        for i, r in enumerate(tool_results, 1):
            parts.append(f"\n--- Tool {i}: {r['name']} ---")
            parts.append(f"Arguments: {r.get('args', {})}")
            if r.get("error"):
                parts.append(f"ERREUR: {r['error']}")
            else:
                result = r.get("result", {})
                parts.append(f"Résultat: {result}")
            parts.append("---")

        parts.append(
            "\nGénère maintenant UNE réponse en Darija marocain "
            "qui synthétise toutes ces informations de façon naturelle."
        )
        return "\n".join(parts)

    # ──────────────────────────────────────────────
    # Extraction robuste du champ "answer"
    # ──────────────────────────────────────────────

    def _extract_answer(self, result: Any) -> Optional[str]:
        """
        Cherche un champ 'answer' dans le résultat d'un tool.

        FastMCP peut wrapper le résultat dans différentes structures :
          - {"success": True, "answer": "...", ...}            ← direct
          - {"result": {"answer": "..."}}                      ← wrapped
          - {"content": [{"text": "..."}]}                     ← string only

        Cette méthode cherche à tous les niveaux probables.
        """
        if result is None:
            return None

        # Cas 1 : result est directement un dict avec "answer"
        if isinstance(result, dict):
            if "answer" in result and result["answer"]:
                return str(result["answer"])

            # Cas 2 : result est wrappé dans "result" ou "data"
            for wrapper_key in ("result", "data", "output"):
                inner = result.get(wrapper_key)
                if isinstance(inner, dict) and "answer" in inner and inner["answer"]:
                    return str(inner["answer"])

            # Cas 3 : il y a juste un "message" ou "text"
            for text_key in ("message", "text", "response"):
                if text_key in result and result[text_key]:
                    return str(result[text_key])

        # Cas 4 : result est une string
        if isinstance(result, str) and result.strip():
            return result

        return None

    # ──────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────

    async def synthesize(
        self,
        original_query: str,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Fusionne les résultats en une réponse Darija.

        Args:
            original_query: question originale de l'utilisateur
            tool_results: liste de {name, args, result, error}

        Returns:
            Une seule réponse en Darija marocain
        """
        # ── Cas 1 : 0 tool, réponse directe (salutation) ──
        if not tool_results:
            logger.info("⚡ Synthèse court-circuit : 0 tool (réponse directe)")
            return "أهلا وسهلا بيك ف CTM. كيفاش نقدر نعاونك اليوم؟"

        # ── Cas 2 : 1 seul tool sans erreur → court-circuit avec "answer" ──
        # Sauf pour la DB : on force le passage au LLM pour une reformulation naturelle
        if len(tool_results) == 1 and not tool_results[0].get("error"):
            r = tool_results[0]

            # Cas spécial : "__direct__" (salutation détectée par le Planner)
            if r["name"] == "__direct__":
                logger.info("⚡ Synthèse court-circuit : réponse directe du Planner")
                return r["args"].get("text", "كيفاش نعاونك؟")

            # ✅ COURT-CIRCUIT UNIQUEMENT pour le RAG (réponse déjà naturelle)
            # La DB passe TOUJOURS par le LLM pour une reformulation conversationnelle
            if r["name"] == "rag_search":
                answer = self._extract_answer(r.get("result"))
                if answer:
                    logger.info(f"⚡ Synthèse court-circuit : RAG (answer déjà en darija)")
                    return answer

            else:
                logger.warning(
                    f"⚠️ Pas de 'answer' trouvé dans le résultat de {r['name']} "
                    f"— fallback LLM call. Result keys: "
                    f"{list(r.get('result', {}).keys()) if isinstance(r.get('result'), dict) else type(r.get('result'))}"
                )

        # ── Cas 3 : N tools (ou 1 tool sans "answer") → LLM call pour fusionner ──
        logger.info(
            f"🧠 Synthèse LLM : fusion de {len(tool_results)} résultat(s) tool(s) "
            f"avec modèle={self.model}"
        )

        prompt = self._build_user_prompt(original_query, tool_results)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    system_instruction=self._build_system_instruction()
                )
            )
        except Exception as e:
            logger.error(f"❌ Synthesizer LLM call échoué: {e}")
            # Fallback : concaténer les answers bruts
            fallback = self._fallback_concat(tool_results)
            logger.info(f"🔄 Fallback concat utilisé — réponse={len(fallback)} chars")
            return fallback

        if not response.candidates:
            logger.warning("⚠️ Synthesizer : aucun candidat — fallback concat")
            return self._fallback_concat(tool_results)

        text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text

        if not text:
            logger.warning("⚠️ Synthesizer : réponse vide — fallback concat")
            return self._fallback_concat(tool_results)

        logger.info(
            f"✅ Synthèse LLM générée ({len(text)} chars) à partir de "
            f"{len(tool_results)} tool(s) avec {self.model}"
        )
        return text

    def _fallback_concat(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Fallback de secours : concatène les answers bruts des tools.
        Utilisé si le LLM call échoue.
        """
        parts = []
        for r in tool_results:
            if r.get("error"):
                continue
            answer = self._extract_answer(r.get("result"))
            if answer:
                parts.append(answer)

        if not parts:
            return "عذراً، ما عنديش معلومات كافية. عاود من فضلك."

        return "\n\n".join(parts)

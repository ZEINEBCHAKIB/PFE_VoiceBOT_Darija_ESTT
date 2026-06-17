"""
MCP Server — Gateway vers tous les tools du VoiceBot.
Routing intelligent par LLM (Gemini).
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from app.core.memory import ConversationMemory

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class MCPServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        api_key = os.getenv("GEMINI_API_KEY_ROUTER")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY_ROUTER manquante")

        self.client = genai.Client(api_key=api_key)
        self._tools: Dict[str, dict] = {}
        self._register_default_tools()

        self._initialized = True
        logger.info(f"✅ MCP Server initialisé — {len(self._tools)} tool(s) enregistré(s)")

    def _register_default_tools(self):

        self.register_tool(
            name="rag_search",
            description=(
                "Cherche dans la base de connaissances CTM : "
                "politique bagages, conditions annulation, documents requis, "
                "informations générales sur les services CTM, FAQ. "
                "À utiliser pour toute question générale ou procédurale."
            ),
            handler=self._handle_rag_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La question de l'utilisateur en darija ou français"
                    }
                },
                "required": ["query"]
            }
        )

        self.register_tool(
            name="ctm_db_query",
            description=(
                "Interroge la base de données structurée CTM pour des informations précises. "
                "À utiliser pour : horaires exacts de départ, tarifs par classe (standard/confort), "
                "localisation et contacts des agences CTM, statut d'une réclamation, "
                "création d'une nouvelle réclamation client, suivi de colis CTM Messagerie. "
                "NE PAS utiliser pour des questions générales ou procédurales — utiliser rag_search."
            ),
            handler=self._handle_ctm_db_query,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "horaires",
                            "tarifs",
                            "agence",
                            "reclamation_get",
                            "reclamation_create",
                            "colis"
                        ],
                        "description": (
                            "L'action à effectuer : "
                            "'horaires' pour les horaires de départ, "
                            "'tarifs' pour les prix, "
                            "'agence' pour localiser une agence CTM, "
                            "'reclamation_get' pour consulter une réclamation existante, "
                            "'reclamation_create' pour créer une nouvelle réclamation, "
                            "'colis' pour suivre un colis CTM Messagerie"
                        )
                    },
                    "ville_depart": {
                        "type": "string",
                        "description": "Ville de départ (pour horaires et tarifs)"
                    },
                    "ville_arrivee": {
                        "type": "string",
                        "description": "Ville d'arrivée (pour horaires et tarifs)"
                    },
                    "ville": {
                        "type": "string",
                        "description": "Nom de la ville (pour agence)"
                    },
                    "reference": {
                        "type": "string",
                        "description": "Numéro de référence de la réclamation (ex: REC-2025-0001)"
                    },
                    "telephone": {
                        "type": "string",
                        "description": "Numéro de téléphone du client"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description du problème (pour reclamation_create)"
                    },
                    "numero_suivi": {
                        "type": "string",
                        "description": "Numéro de suivi du colis (ex: CTM-COL-2025-0001)"
                    }
                },
                "required": ["action"]
            }
        )

        self.register_tool(
            name="rag_stats",
            description="Statistiques du système RAG (nombre de documents indexés, etc.)",
            handler=self._handle_rag_stats,
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )

    def register_tool(self, name: str, description: str, handler, parameters: dict):
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters
        }
        logger.debug(f"📦 Tool enregistré: {name}")

    def _build_tools_schema_for_prompt(self) -> str:
        schema = []
        for tool in self._tools.values():
            schema.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _select_tool(self, user_query: str, memory=None) -> dict:
        """Le LLM router reçoit aussi l'historique pour comprendre le contexte"""

        history_text = ""
        if memory and not memory.is_empty():
            history_text = f"""
HISTORIQUE DE LA CONVERSATION (pour comprendre le contexte):
{memory.format_as_text()}
"""

        prompt = f"""Tu es un router intelligent pour un call center CTM (transport au Maroc).
Ton rôle est UNIQUEMENT de choisir le bon tool selon la question ET le contexte de la conversation.

TOOLS DISPONIBLES:
{self._build_tools_schema_for_prompt()}
{history_text}
QUESTION ACTUELLE: {user_query}

IMPORTANT: Si la question est courte ou incomplète (ex: "الرباط", "وشحال؟", "ومن فاس؟"),
utilise l'historique pour comprendre ce que le client veut vraiment.

Réponds UNIQUEMENT en JSON valide, sans markdown:
{{"tool": "nom_du_tool", "params": {{...paramètres...}}}}

Exemples avec contexte:
- Historique: "horaires من كازا" → Question: "الرباط"
  → {{"tool": "ctm_db_query", "params": {{"action": "horaires", "ville_depart": "Casablanca", "ville_arrivee": "Rabat"}}}}
- Historique: "tarifs طنجة-كازا" → Question: "وكونفور؟"
  → {{"tool": "ctm_db_query", "params": {{"action": "tarifs", "ville_depart": "Tanger", "ville_arrivee": "Casablanca"}}}}
- Historique: vide → Question: "شحال الثمن من فاس لوجدة؟"
  → {{"tool": "ctm_db_query", "params": {{"action": "tarifs", "ville_depart": "Fès", "ville_arrivee": "Oujda"}}}}
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )

            raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            result = json.loads(raw)

            logger.info(f"🧭 Tool sélectionné : {result['tool']} | params : {result['params']}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur parsing JSON router: {e}")
            return {"tool": "rag_search", "params": {"query": user_query}}

        except Exception as e:
            logger.error(f"❌ Erreur router LLM: {e}")
            return {"tool": "rag_search", "params": {"query": user_query}}

    def call_tool(self, user_query: str, memory=None) -> Dict[str, Any]:
        try:
            routing = self._select_tool(user_query, memory)
            tool_name = routing["tool"]
            params = routing["params"]
            params["memory"] = memory

            if tool_name not in self._tools:
                available = ", ".join(self._tools.keys())
                logger.error(f"❌ Tool inconnu: {tool_name}. Disponibles: {available}")
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' non trouvé",
                    "tool": tool_name
                }

            tool = self._tools[tool_name]
            logger.info(f"🔧 MCP → {tool_name}({params})")
            result = tool["handler"](params)

            return {"success": True, "tool": tool_name, "data": result}

        except Exception as e:
            logger.error(f"❌ Erreur MCP call_tool: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "unknown",
                "data": {"answer": "وقع مشكل تقني، عاود من فضلك"}
            }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self._tools.values()
        ]

    def _get_rag_tool(self):
        if not hasattr(self, "_rag_tool_instance"):
            from app.mcp_server.tools.rag_tool import RAGTool
            self._rag_tool_instance = RAGTool()
            logger.info("✅ RAG Tool initialisé (lazy)")
        return self._rag_tool_instance

    def _get_ctm_db_tool(self):
        if not hasattr(self, "_ctm_db_tool_instance"):
            from app.mcp_server.tools.ctm_db_tool import CTMDatabaseTool
            self._ctm_db_tool_instance = CTMDatabaseTool()
            logger.info("✅ CTM DB Tool initialisé (lazy)")
        return self._ctm_db_tool_instance

    def _handle_rag_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        memory = params.get("memory")
        if not query:
            return {"answer": "Le paramètre 'query' est requis"}
        rag = self._get_rag_tool()
        return rag.answer_darija(query, memory)

    def _handle_rag_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rag = self._get_rag_tool()
        return rag.get_stats()

    def _handle_ctm_db_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.pop("action", "")
        memory = params.pop("memory", None)
        ctm = self._get_ctm_db_tool()
        return ctm.query(action=action, params=params, memory=memory)


def get_mcp_server() -> MCPServer:
    return MCPServer()
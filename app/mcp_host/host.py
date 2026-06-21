"""
MCP Host — Client MCP qui orchestre plusieurs MCP servers.

Implémente le VRAI protocole MCP :
  - Lance les servers comme subprocess indépendants
  - Communique avec eux via JSON-RPC 2.0 sur stdio
  - Découvre leurs tools via list_tools()
  - Les appelle via call_tool()

Côté routing, on garde Gemini function_calling (natif) pour choisir
quel tool appeler — mais l'exécution se fait via MCP, pas en local.

Interface publique (async) :
  - await host.start()                          — lance les subprocess
  - await host.call_tool(query, memory=None)    — route + exécute
  - await host.list_tools()                     — liste les tools MCP
  - await host.stop()                           — ferme les subprocess
"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class MCPHost:
    """
    Host MCP — coordonne plusieurs MCP servers via JSON-RPC 2.0 sur stdio.

    Lifecycle :
        host = get_mcp_host()
        await host.call_tool(...)   # lazy-start automatique au 1er appel
        await host.stop()           # optionnel — ferme les subprocess
    """

    _instance: Optional["MCPHost"] = None

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

        # ── Client Gemini pour le routing ──
        self.client = genai.Client(api_key=api_key)
        # Note : l'ancien code utilisait "gemini-3.1-flash-lite" qui n'existe pas
        self.router_model = os.getenv("GEMINI_ROUTER_MODEL", "gemini-3.1-flash-lite")

        # ── Commande Python pour lancer les subprocess ──
        # Par défaut "python" — hérite du venv actif.
        # Override via MCP_PYTHON_CMD si besoin (ex: "/path/to/venv/bin/python")
        self._python_cmd = os.getenv("MCP_PYTHON_CMD", "python")

        # ── Configuration des MCP servers ──
        # Chaque server est lancé comme subprocess indépendant
        self._servers_config: Dict[str, StdioServerParameters] = {
            "rag": StdioServerParameters(
                command=self._python_cmd,
                args=["-m", "app.mcp_servers.rag_server"],
                env=None  # hérite de l'environnement parent
            ),
            "db": StdioServerParameters(
                command=self._python_cmd,
                args=["-m", "app.mcp_servers.db_server"],
                env=None
            ),
        }

        # ── Sessions MCP (initialisées lazy au 1er call_tool) ──
        self._sessions: Dict[str, ClientSession] = {}
        self._exit_stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()

        self._initialized = True
        logger.info(
            f"✅ MCP Host initialisé — router={self.router_model} — "
            f"{len(self._servers_config)} server(s) configuré(s)"
        )

    # ──────────────────────────────────────────────
    # Lifecycle : démarrage / arrêt des subprocess
    # ──────────────────────────────────────────────

    async def _ensure_started(self):
        """Lance les subprocess MCP si pas déjà fait (lazy)."""
        if self._sessions:
            return

        async with self._lock:
            if self._sessions:
                return

            logger.info("🚀 Démarrage des MCP servers (subprocess + JSON-RPC stdio)...")
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            for name, params in self._servers_config.items():
                try:
                    # 1. Lance le subprocess + crée les pipes stdio
                    stdio_transport = await self._exit_stack.enter_async_context(
                        stdio_client(params)
                    )
                    read, write = stdio_transport

                    # 2. Crée la session MCP (client JSON-RPC)
                    session = await self._exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )

                    # 3. Handshake MCP — initialize()
                    await session.initialize()
                    self._sessions[name] = session

                    # 4. Découverte des tools via protocole MCP officiel
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    logger.info(
                        f"   ✅ MCP server '{name}' connecté — "
                        f"{len(tool_names)} tool(s) découvert(s) : {tool_names}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Échec démarrage MCP server '{name}': {e}",
                        exc_info=True
                    )
                    # Nettoyer ce qui a été démarré
                    await self.stop()
                    raise

            logger.info(f"✅ {len(self._sessions)} MCP server(s) actifs")

    async def stop(self):
        """Ferme proprement les subprocess MCP."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture exit_stack: {e}")
            self._exit_stack = None
        self._sessions.clear()
        logger.info("🛑 MCP servers arrêtés")

    # ──────────────────────────────────────────────
    # Découverte des tools via protocole MCP
    # ──────────────────────────────────────────────

    async def _collect_all_tools(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des tools depuis tous les MCP servers
        via session.list_tools() — protocole MCP officiel.

        Returns:
            liste de dicts avec : server_name, name, description, input_schema
        """
        all_tools = []
        for server_name, session in self._sessions.items():
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                all_tools.append({
                    "server_name": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema or {
                        "type": "object",
                        "properties": {}
                    }
                })
        return all_tools

    # ──────────────────────────────────────────────
    # Conversion MCP → Gemini
    # ──────────────────────────────────────────────

    def _mcp_to_gemini(
        self,
        mcp_tools: List[Dict[str, Any]]
    ) -> List[types.FunctionDeclaration]:
        """
        Convertit les schemas d'outils MCP (JSON Schema) en
        FunctionDeclaration Gemini — pour le routing natif.
        """
        declarations = []
        for t in mcp_tools:
            schema = t["input_schema"]

            # Construire les properties Gemini
            properties = {}
            for prop_name, prop_schema in (schema.get("properties") or {}).items():
                properties[prop_name] = self._json_schema_to_gemini(prop_schema)

            decl = types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema.get("required", [])
                )
            )
            declarations.append(decl)

        return declarations

    def _json_schema_to_gemini(self, schema: dict) -> types.Schema:
        """Convertit une property JSON Schema en Schema Gemini."""
        type_map = {
            "string":  types.Type.STRING,
            "number":  types.Type.NUMBER,
            "integer": types.Type.INTEGER,
            "boolean": types.Type.BOOLEAN,
            "array":   types.Type.ARRAY,
            "object":  types.Type.OBJECT,
        }

        gemini_type = type_map.get(schema.get("type", "string"), types.Type.STRING)

        kwargs: Dict[str, Any] = {"type": gemini_type}
        if "description" in schema:
            kwargs["description"] = schema["description"]
        if "enum" in schema:
            kwargs["enum"] = schema["enum"]

        return types.Schema(**kwargs)

    # ──────────────────────────────────────────────
    # Construction des inputs Gemini
    # ──────────────────────────────────────────────

    def _build_system_instruction(self) -> str:
        """Instruction système pour Gemini — remplace le préambule du prompt custom."""
        return (
            "Tu es un router intelligent pour un call center CTM (transport au Maroc). "
            "Ton rôle UNIQUEMENT est de choisir le bon tool selon la question ET le contexte. "
            "Si la question est courte ou incomplète (ex: 'الرباط', 'وشحال؟', 'ومن فاس؟'), "
            "utilise l'historique fourni pour comprendre ce que le client veut vraiment. "
            "Tu DOIS toujours appeler un tool — ne réponds jamais en texte libre."
        )

    def _build_contents(
        self,
        user_query: str,
        memory: Optional[ConversationMemory]
    ) -> str:
        """Construit le contenu envoyé à Gemini (query + historique)."""
        if memory and not memory.is_empty():
            history_text = memory.format_as_text()
            return (
                f"HISTORIQUE DE LA CONVERSATION (pour comprendre le contexte):\n"
                f"{history_text}\n\n"
                f"QUESTION ACTUELLE: {user_query}"
            )
        return user_query

    # ──────────────────────────────────────────────
    # Parsing du résultat MCP
    # ──────────────────────────────────────────────

    def _parse_mcp_result(self, result) -> Dict[str, Any]:
        """
        Parse un CallToolResult MCP en dict Python.
        FastMCP retourne généralement structuredContent quand le tool
        renvoie un dict, sinon content (liste de TextContent JSON).
        """
        # Cas 1 : structuredContent (FastMCP avec return type dict)
        if hasattr(result, "structuredContent") and result.structuredContent:
            sc = result.structuredContent
            if isinstance(sc, dict):
                return sc
            if isinstance(sc, str):
                try:
                    return json.loads(sc)
                except json.JSONDecodeError:
                    return {"answer": sc}

        # Cas 2 : content (liste de TextContent)
        if hasattr(result, "content") and result.content:
            text_parts = []
            for c in result.content:
                if hasattr(c, "text") and c.text:
                    text_parts.append(c.text)
            if text_parts:
                try:
                    return json.loads(text_parts[0])
                except json.JSONDecodeError:
                    return {"answer": "\n".join(text_parts)}

        return {"answer": "Réponse vide du tool"}

    # ──────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────

    async def call_tool(
        self,
        user_query: str,
        memory: Optional[ConversationMemory] = None
    ) -> Dict[str, Any]:
        """
        Route la requête via Gemini function_calling natif, puis exécute
        le tool sur le bon MCP server via JSON-RPC 2.0 sur stdio.

        Retourne le même format que l'ancien MCPServer.call_tool() :
            {"success": bool, "tool": str, "data": dict}
        """
        try:
            # 1. Lazy-start des subprocess MCP
            await self._ensure_started()

            # 2. Découverte des tools via protocole MCP
            all_mcp_tools = await self._collect_all_tools()
            if not all_mcp_tools:
                return self._error_response("Aucun tool MCP disponible")

            # 3. Conversion MCP → Gemini FunctionDeclarations
            gemini_declarations = self._mcp_to_gemini(all_mcp_tools)

            # 4. Appel Gemini pour routing (function_calling natif)
            contents = self._build_contents(user_query, memory)

            response = self.client.models.generate_content(
                model=self.router_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=gemini_declarations)],
                    temperature=0.1,
                    system_instruction=self._build_system_instruction()
                )
            )

            # 5. Parser la réponse Gemini
            if not response.candidates:
                return self._error_response("Aucune réponse du routeur")

            candidate = response.candidates[0]

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_name = fc.name
                    params = dict(fc.args) if fc.args else {}

                    logger.info(
                        f"🧭 Gemini a choisi : {tool_name}({params}) — "
                        f"routing natif"
                    )

                    # 6. Trouver le server qui possède ce tool
                    target_server = None
                    for t in all_mcp_tools:
                        if t["name"] == tool_name:
                            target_server = t["server_name"]
                            break

                    if not target_server:
                        return self._error_response(
                            f"Tool '{tool_name}' non trouvé dans les MCP servers"
                        )

                    # 7. Injecter history_text si le tool le supporte
                    #    (rag_search utilise la mémoire conversationnelle)
                    if (
                        memory
                        and not memory.is_empty()
                        and "history_text" not in params
                        and tool_name == "rag_search"
                    ):
                        params["history_text"] = memory.format_as_text()

                    # 8. Appeler le tool sur le server MCP — protocole officiel
                    session = self._sessions[target_server]
                    logger.info(
                        f"📤 Appel MCP → server '{target_server}' / "
                        f"tool '{tool_name}' via JSON-RPC"
                    )

                    mcp_result = await session.call_tool(
                        tool_name, arguments=params
                    )

                    # 9. Parser le résultat MCP
                    data = self._parse_mcp_result(mcp_result)

                    logger.info(
                        f"✅ Tool {tool_name} exécuté via MCP server "
                        f"'{target_server}' — success={data.get('success', True)}"
                    )

                    return {"success": True, "tool": tool_name, "data": data}

            # ── Pas de function_call — fallback ──

            # Cas A : Gemini a répondu en texte (rare, notre system_instruction l'interdit)
            text_response = ""
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_response += part.text

            if text_response:
                logger.info(
                    f"💬 Réponse directe Gemini (sans tool) : "
                    f"{text_response[:80]}..."
                )
                return {
                    "success": True,
                    "tool": "direct",
                    "data": {"answer": text_response}
                }

            # Cas B : Réponse vide ou bloquée par safety — fallback rag_search via MCP
            logger.warning(
                f"⚠️ Réponse vide — finish_reason={candidate.finish_reason} — "
                f"fallback rag_search via MCP"
            )
            if "rag" in self._sessions:
                session = self._sessions["rag"]
                history_text = memory.format_as_text() if memory else ""
                mcp_result = await session.call_tool(
                    "rag_search",
                    arguments={"query": user_query, "history_text": history_text}
                )
                data = self._parse_mcp_result(mcp_result)
                return {"success": True, "tool": "rag_search", "data": data}

            return self._error_response("Aucun tool appelé par Gemini")

        except Exception as e:
            logger.error(f"❌ Erreur MCP Host call_tool: {e}", exc_info=True)
            return self._error_response(str(e))

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Format standard d'erreur — compatible avec api.py et main.py."""
        return {
            "success": False,
            "error": error_msg,
            "tool": "unknown",
            "data": {"answer": "وقع مشكل تقني، عاود من فضلك"}
        }

    async def orchestrate(
        self,
        user_query: str,
        memory: Optional[ConversationMemory] = None
    ) -> Dict[str, Any]:
        """
        Nouveau point d'entrée multi-tools :
        Planner → Executor (parallèle) → Synthesizer

        Retourne un format compatible avec call_tool() :
            {"success": bool, "tool": str (CSV si plusieurs), "data": {"answer": str}}
        """
        from app.mcp_host.planner import PlannerAgent
        from app.mcp_host.synthesizer import SynthesizerAgent

        try:
            # 1. Lazy-start subprocess MCP
            await self._ensure_started()

            # 2. Découverte des tools via protocole MCP
            all_mcp_tools = await self._collect_all_tools()
            if not all_mcp_tools:
                return self._error_response("Aucun tool MCP disponible")

            # 3. Conversion MCP → Gemini
            gemini_declarations = self._mcp_to_gemini(all_mcp_tools)

            # 4. PLANNER — décide quels tools appeler
            # ✅ Passer directement le client au constructeur
            planner = PlannerAgent(client=self.client, model=self.router_model)
            plans = await planner.plan(user_query, memory, gemini_declarations)

            # ✅ Une SEULE instanciation du Synthesizer, avec client + model
            # corrects. (Avant : une 2e instanciation sans paramètres plus bas
            # écrasait celle-ci et retombait sur un fallback de modèle cassé.)
            synthesizer = SynthesizerAgent(client=self.client, model=self.router_model)

            if not plans:
                return {
                    "success": True,
                    "tool": "none",
                    "data": {"answer": "عافاك، عاونني نفهم السؤال ديالك."}
                }

            # Cas direct (salutation) — pas de tool, pas de synthèse
            if len(plans) == 1 and plans[0]["name"] == "__direct__":
                return {
                    "success": True,
                    "tool": "direct",
                    "data": {"answer": plans[0]["args"]["text"]}
                }

            # 5. EXECUTOR — exécute les N plans en parallèle
            tool_results = await self._execute_plans(plans, memory, all_mcp_tools)

            # 6. SYNTHESIZER — fusionne les résultats
            # (réutilise l'instance créée à l'étape 4, avec client + model corrects)
            final_answer = await synthesizer.synthesize(user_query, tool_results)

            # 7. Format de retour compatible avec api.py
            tools_csv = ", ".join(r["name"] for r in tool_results if r["name"] != "__direct__")
            return {
                "success": True,
                "tool": tools_csv or "none",
                "data": {"answer": final_answer},
                "tools_called": [r["name"] for r in tool_results],
                "plans": plans
            }

        except Exception as e:
            logger.error(f"❌ Erreur orchestrate: {e}", exc_info=True)
            return self._error_response(str(e))

    async def _execute_plans(
        self,
        plans: List[Dict[str, Any]],
        memory: Optional[ConversationMemory],
        all_mcp_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Exécute N plans en parallèle via asyncio.gather."""
        import asyncio

        async def _execute_one(plan: dict) -> dict:
            tool_name = plan["name"]
            args = dict(plan["args"])

            # Trouver le server MCP cible
            target_server = None
            for t in all_mcp_tools:
                if t["name"] == tool_name:
                    target_server = t["server_name"]
                    break

            if not target_server:
                return {
                    "name": tool_name, "args": args,
                    "result": None, "error": f"Tool {tool_name} non trouvé"
                }

            # Injecter history_text pour rag_search
            if (
                memory and not memory.is_empty()
                and "history_text" not in args
                and tool_name == "rag_search"
            ):
                args["history_text"] = memory.format_as_text()

            try:
                session = self._sessions[target_server]
                logger.info(f"📤 Appel parallèle → {target_server}/{tool_name}")
                mcp_result = await session.call_tool(tool_name, arguments=args)
                data = self._parse_mcp_result(mcp_result)
                return {"name": tool_name, "args": args, "result": data, "error": None}
            except Exception as e:
                logger.error(f"❌ Tool {tool_name} a échoué: {e}")
                return {"name": tool_name, "args": args, "result": None, "error": str(e)}

        # Exécution parallèle
        tasks = [_execute_one(p) for p in plans]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Gestion des exceptions retournées par gather
        final = []
        for plan, res in zip(plans, results):
            if isinstance(res, Exception):
                final.append({
                    "name": plan["name"], "args": plan["args"],
                    "result": None, "error": str(res)
                })
            else:
                final.append(res)
        return final

    # ──────────────────────────────────────────────
    # API publique : list_tools (compatibilité main.py)
    # ──────────────────────────────────────────────

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Liste tous les tools MCP disponibles — utilise list_tools()
        sur chaque session (protocole MCP officiel).
        """
        await self._ensure_started()
        all_tools = await self._collect_all_tools()
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "server": t["server_name"],
                "parameters": t["input_schema"]
            }
            for t in all_tools
        ]


# ──────────────────────────────────────────────
# Factory singleton — remplace get_mcp_server()
# ──────────────────────────────────────────────

_mcp_host_instance: Optional[MCPHost] = None


def get_mcp_host() -> MCPHost:
    """Retourne l'instance unique du MCPHost."""
    global _mcp_host_instance
    if _mcp_host_instance is None:
        _mcp_host_instance = MCPHost()
    return _mcp_host_instance


def get_mcp_server() -> MCPHost:
    """Alias — compatibilité avec l'ancien nom de factory."""
    return get_mcp_host()
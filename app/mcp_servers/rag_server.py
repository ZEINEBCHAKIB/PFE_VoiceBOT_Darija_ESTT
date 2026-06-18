"""
MCP Server — RAG (base de connaissances CTM).

C'est un VRAI server MCP : processus Python indépendant qui expose
ses tools via JSON-RPC 2.0 sur stdio (le protocole officiel MCP).

Démarrage standalone (test) :
    python -m app.mcp_servers.rag_server

Démarrage par le host :
    Lancé automatiquement comme subprocess par app.mcp_host.host.MCPHost
    via StdioServerParameters(command="python", args=["-m", "app.mcp_servers.rag_server"])

Tools exposés :
    - rag_search(query, history_text="") : recherche + génération Darija
    - rag_stats()                        : statistiques du RAG
"""

import logging
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Déclaration du server MCP
# ──────────────────────────────────────────────

mcp = FastMCP("ctm-rag-server")


# ──────────────────────────────────────────────
# Proxy mémoire — permet de passer l'historique
# conversationnel au RAGTool via le protocole MCP
# ──────────────────────────────────────────────

class _MemoryProxy:
    """
    RAGTool.answer_darija() n'utilise que memory.format_as_text()
    et memory.is_empty(). On recrée un proxy minimal qui se
    comporte comme ConversationMemory côté server.
    """
    def __init__(self, history_text: str):
        self._text = history_text or ""

    def format_as_text(self) -> str:
        return self._text

    def is_empty(self) -> bool:
        return not self._text


# ──────────────────────────────────────────────
# Tools exposés via MCP
# ──────────────────────────────────────────────

@mcp.tool()
def rag_search(query: str, history_text: str = "") -> Dict[str, Any]:
    """
    Cherche dans la base de connaissances CTM : politique bagages,
    conditions annulation, documents requis, informations générales
    sur les services CTM, FAQ.

    À utiliser pour toute question générale ou procédurale.

    Args:
        query: La question de l'utilisateur en darija ou français
        history_text: Historique conversationnel formaté (optionnel,
                      pour résoudre les ellipses comme "وشحال؟")

    Returns:
        dict avec les clés : query_darija, query_french, answer,
        sources, scores, themes, success
    """
    from app.mcp_server.tools.rag_tool import RAGTool

    if not query:
        return {
            "success": False,
            "answer": "Le paramètre 'query' est requis",
            "sources": []
        }

    # Lazy-init du RAGTool (charge Qdrant + LLM)
    rag = RAGTool()

    # Reconstruire un proxy mémoire depuis le texte sérialisé
    memory = _MemoryProxy(history_text) if history_text else None

    try:
        result = rag.answer_darija(query, memory)
        logger.info(
            f"✅ rag_search exécuté — query='{query[:50]}...' — "
            f"success={result.get('success')}"
        )
        return result
    except Exception as e:
        logger.error(f"❌ Erreur rag_search: {e}", exc_info=True)
        return {
            "success": False,
            "answer": f"Erreur RAG: {str(e)}",
            "sources": []
        }


@mcp.tool()
def rag_stats() -> Dict[str, Any]:
    """
    Retourne les statistiques du système RAG
    (nombre de documents indexés, taille de l'index, etc.).

    Returns:
        dict avec les stats du vector store
    """
    from app.mcp_server.tools.rag_tool import RAGTool

    rag = RAGTool()
    return rag.get_stats()


# ──────────────────────────────────────────────
# Point d'entrée standalone (test sans host)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logger.info("🚀 Démarrage MCP Server RAG sur stdio...")
    logger.info("   En attente de commandes JSON-RPC depuis le host...")
    mcp.run(transport='stdio')

"""
MCP Server — CTM Database.

Vrai server MCP (processus indépendant, JSON-RPC 2.0 sur stdio).

Démarrage standalone (test) :
    python -m app.mcp_servers.db_server

Démarrage par le host :
    Lancé automatiquement par app.mcp_host.host.MCPHost

Tools exposés :
    - ctm_db_query(action, ...) : horaires, tarifs, agences,
                                  réclamations, suivi colis
"""

import logging
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Déclaration du server MCP
# ──────────────────────────────────────────────

mcp = FastMCP("ctm-db-server")


# ──────────────────────────────────────────────
# Tool exposé via MCP
# ──────────────────────────────────────────────

@mcp.tool()
def ctm_db_query(
    action: str,
    ville_depart: str = "",
    ville_arrivee: str = "",
    ville: str = "",
    reference: str = "",
    telephone: str = "",
    description: str = "",
    numero_suivi: str = ""
) -> Dict[str, Any]:
    """
    Interroge la base de données structurée CTM pour des informations précises.

    À utiliser pour :
      - horaires exacts de départ
      - tarifs par classe (standard/confort)
      - localisation et contacts des agences CTM
      - statut d'une réclamation existante
      - création d'une nouvelle réclamation client
      - suivi d'un colis CTM Messagerie

    NE PAS utiliser pour des questions générales ou procédurales —
    utiliser rag_search dans ce cas.

    Args:
        action: Une parmi :
            'horaires'           — horaires d'une liaison
            'tarifs'             — prix d'une liaison
            'agence'             — infos d'une agence CTM
            'reclamation_get'    — consulter une réclamation
            'reclamation_create' — créer une nouvelle réclamation
            'colis'              — suivre un colis
        ville_depart:  Ville de départ (pour horaires et tarifs)
        ville_arrivee: Ville d'arrivée (pour horaires et tarifs)
        ville:         Nom de la ville (pour agence)
        reference:     Numéro de référence de la réclamation (ex: REC-2025-0001)
        telephone:     Numéro de téléphone du client
        description:   Description du problème (pour reclamation_create)
        numero_suivi:  Numéro de suivi du colis (ex: CTM-COL-2025-0001)

    Returns:
        dict avec success, action, data, answer
    """
    from app.mcp_server.tools.ctm_db_tool import CTMDatabaseTool

    if not action:
        return {
            "success": False,
            "answer": "Le paramètre 'action' est requis"
        }

    db = CTMDatabaseTool()

    # Construire le dict params attendu par CTMDatabaseTool.query()
    # (on ne garde que les params non vides + on exclut 'action')
    params = {
        "ville_depart":  ville_depart,
        "ville_arrivee": ville_arrivee,
        "ville":         ville,
        "reference":     reference,
        "telephone":     telephone,
        "description":   description,
        "numero_suivi":  numero_suivi,
    }
    params = {k: v for k, v in params.items() if v}

    try:
        result = db.query(action=action, params=params, memory=None)
        logger.info(
            f"✅ ctm_db_query exécuté — action={action} — "
            f"success={result.get('success')}"
        )
        return result
    except Exception as e:
        logger.error(f"❌ Erreur ctm_db_query: {e}", exc_info=True)
        return {
            "success": False,
            "answer": f"Erreur DB: {str(e)}"
        }


# ──────────────────────────────────────────────
# Point d'entrée standalone (test sans host)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logger.info("🚀 Démarrage MCP Server DB sur stdio...")
    logger.info("   En attente de commandes JSON-RPC depuis le host...")
    mcp.run(transport='stdio')

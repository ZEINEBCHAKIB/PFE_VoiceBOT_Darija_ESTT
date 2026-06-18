"""
MCP Host — Client MCP qui orchestre plusieurs MCP servers.

Architecture :
    [User query + memory]
           ↓
    [MCP Host — Gemini function_calling natif pour le routing]
           ↓                                ↓
    [MCP Server RAG]              [MCP Server DB]
     (subprocess, JSON-RPC stdio)  (subprocess, JSON-RPC stdio)
           ↓                                ↓
    [RAGTool]                     [CTMDatabaseTool]

Le host :
  1. Lance les 2 MCP servers comme subprocess (JSON-RPC 2.0 sur stdio)
  2. Découvre les tools via session.list_tools() — protocole MCP officiel
  3. Convertit les schemas MCP (JSON Schema) → FunctionDeclaration Gemini
  4. Appelle Gemini pour choisir le tool
  5. Appelle le tool sur le bon server via session.call_tool() — protocole MCP officiel
"""

from app.mcp_host.host import MCPHost, get_mcp_host, get_mcp_server

__all__ = ["MCPHost", "get_mcp_host", "get_mcp_server"]

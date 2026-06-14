"""
Tool RAG pour le MCP Server
"""

import logging
from typing import Dict, Any, Optional

from app.rag.retrieval import get_retriever
from app.core.llm import get_llm_client
from app.config.settings import config
from app.core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class RAGTool:
    """Tool pour la recherche et génération RAG"""

    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm_client()

    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Recherche dans la base de connaissances sans génération LLM"""
        top_k = top_k or config.TOP_K
        results = self.retriever.retrieve(query, top_k)
        return {
            "query": query,
            "results": [
                {
                    "text": r["text"],
                    "theme": r["theme"],
                    "score": r["score"],
                    "similarity_percent": f"{r['score'] * 100:.1f}%"
                }
                for r in results
            ],
            "count": len(results)
        }

    def answer(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """Répond à une question en français"""
        if use_rag:
            retrieval_result = self.retriever.retrieve_with_context(query)

            if not retrieval_result["has_results"]:
                return {
                    "query": query,
                    "answer": "سمحلي، ما لقيتش معلومات على هاد السؤال.",
                    "sources": [],
                    "rag_used": True
                }

            # 👇 utilise rag_generate_with_history sans historique
            response = self.llm.rag_generate_with_history(
                query=query,
                context=retrieval_result["context"],
                history_text=""
            )

            return {
                "query": query,
                "answer": response,
                "sources": retrieval_result["results"],
                "scores": retrieval_result["scores"],
                "themes": retrieval_result["themes"],
                "rag_used": True
            }
        else:
            response = self.llm.generate(f"Réponds à cette question en français: {query}")
            return {
                "query": query,
                "answer": response or "سمحلي، وقع مشكل تقني.",
                "sources": [],
                "rag_used": False
            }

    def answer_darija(self, query_darija: str, memory: ConversationMemory = None) -> Dict[str, Any]:
        """Répond en darija avec pipeline complet + mémoire conversationnelle"""

        # Étape 1: Traduction darija → français
        french_query = self.llm.translate_to_french(query_darija)
        logger.info(f"🌐 Traduction: '{query_darija}' → '{french_query}'")

        # Étape 2: Récupération contexte Qdrant
        retrieval_result = self.retriever.retrieve_with_context(french_query)

        if not retrieval_result["has_results"]:
            return {
                "query_darija": query_darija,
                "query_french": french_query,
                "answer": "سمحلي، ما لقيتش معلومات على هاد السؤال.",
                "sources": [],
                "success": False
            }

        # Étape 3: Historique formaté depuis la mémoire
        history_text = memory.format_as_text() if memory else ""

        # Étape 4: Génération réponse Darija avec historique
        response = self.llm.rag_generate_with_history(
            query=french_query,
            context=retrieval_result["context"],
            history_text=history_text
        )

        return {
            "query_darija": query_darija,
            "query_french": french_query,
            "answer": response,
            "sources": retrieval_result["results"],
            "scores": retrieval_result["scores"],
            "themes": retrieval_result["themes"],
            "success": True
        }

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du système RAG"""
        return self.retriever.store.get_stats()
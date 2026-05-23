"""
Tool RAG pour le MCP Server
"""

import logging
from typing import Dict, Any, Optional

from app.rag.retrieval import get_retriever
from app.core.llm import get_llm_client
from app.config.settings import config

logger = logging.getLogger(__name__)


class RAGTool:
    """Tool pour la recherche et génération RAG"""
    
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm_client()
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Recherche dans la base de connaissances
        
        Args:
            query: Requête en français
            top_k: Nombre de résultats
        
        Returns:
            Résultats de recherche
        """
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
        """
        Répond à une question en utilisant le RAG
        
        Args:
            query: Question en français
            use_rag: Utiliser le RAG ou juste le LLM
        
        Returns:
            Réponse avec contexte
        """
        if use_rag:
            # Récupérer le contexte
            retrieval_result = self.retriever.retrieve_with_context(query)
            
            if not retrieval_result["has_results"]:
                return {
                    "query": query,
                    "response": "Désolé, je n'ai pas trouvé d'information pertinente dans ma base de connaissances.",
                    "sources": [],
                    "rag_used": True
                }
            
            # Générer la réponse
            response = self.llm.rag_generate(query, retrieval_result["context"])
            
            return {
                "query": query,
                "response": response,
                "sources": retrieval_result["results"],
                "scores": retrieval_result["scores"],
                "themes": retrieval_result["themes"],
                "rag_used": True
            }
        else:
            # Réponse directe du LLM sans contexte
            response = self.llm.generate(f"Réponds à cette question en français: {query}")
            
            return {
                "query": query,
                "response": response or "Désolé, je n'ai pas pu générer de réponse.",
                "sources": [],
                "rag_used": False
            }
    
    def answer_darija(self, query_darija: str) -> Dict[str, Any]:
        """
        Répond à une question en darija avec pipeline complet
        
        Args:
            query_darija: Question en darija
        
        Returns:
            Réponse en darija
        """
        # Étape 1: Traduction darija → français
        french_query = self.llm.translate_to_french(query_darija)
        
        # Étape 2: Récupération du contexte
        retrieval_result = self.retriever.retrieve_with_context(french_query)
        
        if not retrieval_result["has_results"]:
            return {
                "query_darija": query_darija,
                "query_french": french_query,
                "response": "Désolé, je n'ai pas trouvé d'information pertinente.",
                "sources": [],
                "success": False
            }
        
        # Étape 3: Génération réponse en darija
        response = self.llm.rag_generate(french_query, retrieval_result["context"])
        
        return {
            "query_darija": query_darija,
            "query_french": french_query,
            "response": response,
            "sources": retrieval_result["results"],
            "scores": retrieval_result["scores"],
            "themes": retrieval_result["themes"],
            "success": True
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques du système RAG"""
        return self.retriever.store.get_stats()


# Instance globale
rag_tool = RAGTool()
"""
Retrieval RAG - Recherche et contexte
"""

from app.rag.qdrant_store import get_qdrant_store
from app.config.settings import config


class Retriever:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.store = get_qdrant_store()
        self._initialized = True
    
    def retrieve(self, query: str, top_k: int = None) -> list:
        """Recherche les chunks similaires"""
        top_k = top_k or config.TOP_K
        return self.store.search(query, top_k)
    
    def build_context(self, results: list) -> str:
        """Construit le contexte à partir des résultats"""
        if not results:
            return ""
        
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i} - {r['theme']} - Score: {r['score']:.3f}]\n{r['text']}")
        
        return "\n\n".join(context_parts)
    
    def retrieve_with_context(self, query: str) -> dict:
        """Recherche + contexte"""
        results = self.retrieve(query)
        context = self.build_context(results)
        
        return {
            "results": results,
            "context": context,
            "has_results": len(results) > 0,
            "scores": [r["score"] for r in results],
            "themes": [r["theme"] for r in results]
        }


def get_retriever():
    return Retriever()
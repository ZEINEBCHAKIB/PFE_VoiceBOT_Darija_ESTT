"""
Gestion des embeddings avec BGE-M3
"""

import logging
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer

from app.config.settings import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper pour le modèle d'embedding BGE-M3"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Chargement du modèle {config.EMBEDDING_MODEL} sur {self.device}")
        
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        self.model.to(self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self._initialized = True
        logger.info(f"✅ Modèle chargé - Dimension: {self.dimension}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """Encoder un texte ou une liste de textes"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def encode_query(self, query: str) -> List[float]:
        """Encoder une requête pour la recherche"""
        return self.encode(query)[0]


def get_embedding_model() -> EmbeddingModel:
    """Obtenir l'instance du modèle d'embedding (singleton)"""
    return EmbeddingModel()
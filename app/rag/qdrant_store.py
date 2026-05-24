"""
Qdrant Store - Chargement de l'index existant
"""

import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from app.config.settings import config
from app.rag.embeddings import get_embedding_model


class QdrantStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.client = QdrantClient(path="data/qdrant")
        self.collection_name = config.QDRANT_COLLECTION
        self.embedding_model = get_embedding_model()
        
        # Créer la collection si elle n'existe pas
        self._init_collection()
        
        self._initialized = True
    
    def _init_collection(self):
        """Initialise la collection (vide au départ)"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection créée: {self.collection_name}")
    
    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.55) -> list:
        """Recherche dans l'index"""
        query_embedding = self.embedding_model.encode_query(query)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        return [
            {
                "text": point.payload["text"],
                "theme": point.payload["theme"],
                "score": point.score,
                "chunk_id": point.payload.get("chunk_id", 0)
            }
            for point in results.points
        ]
    
    def load_from_metadata(self, metadata_path: str):
        """Charge les chunks depuis le JSON et les indexe dans Qdrant"""
        print(f"📂 Chargement des métadonnées depuis {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Si c'est le format avec wrapper
        if "chunks" in metadata:
            chunks = metadata["chunks"]
        else:
            chunks = metadata
        
        print(f"📊 {len(chunks)} chunks trouvés")
        
        # Encoder et uploader
        texts = [chunk["text"] for chunk in chunks]
        print("🔄 Encodage des chunks...")
        embeddings = self.embedding_model.encode(texts)
        
        from qdrant_client.http.models import PointStruct
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "theme": chunk["theme"],
                    "chunk_id": chunk.get("chunk_id", i),
                    "length": chunk.get("length", 0)
                }
            ))
        
        # Upload par lots
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  📤 Uploadé {len(batch)} points...")
        
        print(f"✅ Index chargé: {len(points)} points")
        return {"total_chunks": len(chunks)}
    
    def get_stats(self):
        """Statistiques de l'index"""
        if not self.client.collection_exists(self.collection_name):
            return {"exists": False}
        
        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "collection": self.collection_name
        }


def get_qdrant_store():
    return QdrantStore()
"""
Script d'indexation complet pour CTM Voicebot
Lit les fichiers TXT, crée les chunks, encode avec BGE-M3, et construit l'index Qdrant
À exécuter UNE SEULE FOIS
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Ajouter le parent au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import config
from app.rag.embeddings import get_embedding_model
from app.rag.chunking import chunk_documents
from app.rag.qdrant_store import get_qdrant_store

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents() -> dict:
    """
    Charge les documents depuis les fichiers TXT dans data/raw/
    
    Returns:
        Dictionnaire {theme: contenu}
    """
    documents = {}
    
    logger.info("📂 Chargement des documents depuis data/raw/...")
    
    for theme, filepath in config.DOCUMENTS.items():
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                documents[theme] = f.read()
            logger.info(f"  ✅ {theme}: {len(documents[theme])} caractères")
        else:
            logger.error(f"  ❌ Fichier non trouvé: {filepath}")
            raise FileNotFoundError(f"Fichier manquant: {filepath}")
    
    return documents


def save_metadata(chunks: list, output_path: Path) -> None:
    """
    Sauvegarde les métadonnées des chunks au format JSON
    
    Args:
        chunks: Liste des chunks
        output_path: Chemin de sauvegarde
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "collection": config.QDRANT_COLLECTION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "num_chunks": len(chunks),
        "chunks": chunks
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Métadonnées sauvegardées: {output_path}")


def main():
    """Fonction principale d'indexation"""
    print("\n" + "="*60)
    print("🚀 CTM VOICEBOT - INDEXATION RAG")
    print("="*60)
    
    start_time = time.time()
    
    # ========== ÉTAPE 1: Chargement des documents ==========
    print("\n📁 ÉTAPE 1/5: Chargement des documents")
    print("-"*40)
    
    try:
        documents = load_documents()
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        print("\nAssure-toi d'avoir les 3 fichiers dans data/raw/:")
        print("  - data/raw/colis.txt")
        print("  - data/raw/voyages.txt")
        print("  - data/raw/politiques.txt")
        return
    
    print(f"\n✅ {len(documents)}/3 documents chargés")
    
    # ========== ÉTAPE 2: Découpage en chunks ==========
    print("\n✂️  ÉTAPE 2/5: Découpage des documents en chunks")
    print("-"*40)
    
    chunks = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    if not chunks:
        print("❌ Erreur: Aucun chunk créé")
        return
    
    print(f"\n✅ {len(chunks)} chunks créés")
    
    # Statistiques par thème
    themes_count = {}
    for chunk in chunks:
        theme = chunk["theme"]
        themes_count[theme] = themes_count.get(theme, 0) + 1
    
    for theme, count in themes_count.items():
        print(f"  📌 {theme}: {count} chunks")
    
    # ========== ÉTAPE 3: Initialisation du modèle BGE-M3 ==========
    print("\n🧠 ÉTAPE 3/5: Chargement du modèle BGE-M3")
    print("-"*40)
    
    embedding_start = time.time()
    embedding_model = get_embedding_model()
    print(f"✅ Modèle chargé - Dimension: {embedding_model.dimension}")
    
    # ========== ÉTAPE 4: Encodage des chunks ==========
    print("\n🔄 ÉTAPE 4/5: Encodage des chunks")
    print("-"*40)
    
    texts = [chunk["text"] for chunk in chunks]
    print(f"📝 {len(texts)} textes à encoder")
    print(f"📏 Longueur moyenne: {sum(len(t) for t in texts)/len(texts):.0f} caractères")
    
    encode_start = time.time()
    embeddings = embedding_model.encode(texts, batch_size=16)
    encode_time = time.time() - encode_start
    
    print(f"✅ Encodage terminé: {len(embeddings)} vecteurs de dimension {len(embeddings[0])}")
    print(f"⏱️  Temps d'encodage: {encode_time:.2f} secondes")
    
    # ========== ÉTAPE 5: Création de l'index Qdrant ==========
    print("\n💾 ÉTAPE 5/5: Création de l'index Qdrant")
    print("-"*40)
    
    store = get_qdrant_store()
    
    # Supprimer l'ancienne collection si elle existe
    if store.client.collection_exists(config.QDRANT_COLLECTION):
        print(f"⚠️ Collection {config.QDRANT_COLLECTION} existe déjà, suppression...")
        store.client.delete_collection(config.QDRANT_COLLECTION)
    
    # Créer la nouvelle collection
    store._init_collection()
    
    # Préparer et uploader les points
    print("📤 Upload des vecteurs vers Qdrant...")
    
    from qdrant_client.http.models import PointStruct
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={
                "text": chunk["text"],
                "theme": chunk["theme"],
                "chunk_id": chunk["chunk_id"],
                "length": chunk["length"],
                "num_sentences": chunk.get("num_sentences", 0),
                "timestamp": datetime.now().isoformat()
            }
        ))
    
    # Upload par lots de 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        store.client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=batch
        )
        print(f"  📤 Uploadé {len(batch)} points ({i+len(batch)}/{len(points)})")
    
    # ========== SAUVEGARDE DES MÉTADONNÉES ==========
    print("\n💾 Sauvegarde des métadonnées...")
    save_metadata(chunks, config.METADATA_PATH)
    
    # ========== VÉRIFICATION FINALE ==========
    print("\n🔍 Vérification de l'index...")
    
    collection_info = store.client.get_collection(config.QDRANT_COLLECTION)
    print(f"✅ Collection: {config.QDRANT_COLLECTION}")
    print(f"📊 Points dans Qdrant: {collection_info.points_count}")
    print(f"📊 Vecteurs dimension: {collection_info.config.params.vectors.size}")
    print(f"📊 Distance: {collection_info.config.params.vectors.distance}")
    
    # Test de recherche
    print("\n🧪 Test de recherche...")
    test_query = "Comment réserver un billet de voyage?"
    print(f"   Requête: '{test_query}'")
    
    test_embedding = embedding_model.encode_query(test_query)
    test_results = store.client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=test_embedding,
        limit=3
    )
    
    if test_results.points:
        print(f"   ✅ Recherche OK - {len(test_results.points)} résultats trouvés")
        for point in test_results.points[:2]:
            print(f"      • [{point.payload['theme']}] Score: {point.score:.3f}")
            print(f"        {point.payload['text'][:80]}...")
    else:
        print("   ⚠️ Aucun résultat - vérifie l'index")
    
    # ========== STATISTIQUES FINALES ==========
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ INDEXATION TERMINÉE AVEC SUCCÈS !")
    print("="*60)
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Documents: {len(documents)}")
    print(f"   • Chunks: {len(chunks)}")
    print(f"   • Points dans Qdrant: {collection_info.points_count}")
    print(f"   • Dimension vecteurs: {embedding_model.dimension}")
    print(f"\n⏱️  TEMPS TOTAL: {total_time:.2f} secondes")
    print(f"   • Encodage: {encode_time:.2f}s")
    print(f"   • Vitesse: {len(chunks)/encode_time:.2f} chunks/seconde")
    print(f"\n💾 FICHIERS GÉNÉRÉS:")
    print(f"   • Métadonnées: {config.METADATA_PATH}")
    print(f"\n🚀 Pour tester, lance:")
    print(f"   python scripts/chat.py")
    print("="*60)


if __name__ == "__main__":
    main()
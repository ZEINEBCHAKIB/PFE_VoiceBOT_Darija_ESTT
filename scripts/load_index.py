"""
Script pour charger chunks_metadata.json dans Qdrant
À exécuter UNE SEULE FOIS après avoir copié le fichier JSON
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import config
from app.rag.qdrant_store import get_qdrant_store


def main():
    print("="*50)
    print("📂 Chargement de l'index RAG")
    print("="*50)
    
    metadata_path = config.METADATA_PATH
    
    if not metadata_path.exists():
        print(f"❌ Fichier non trouvé: {metadata_path}")
        print("\nCopie ton chunks_metadata.json depuis Colab:")
        print("  /content/drive/MyDrive/pfe/index/chunks_metadata.json")
        print(f"  → {metadata_path}")
        return
    
    store = get_qdrant_store()
    stats = store.load_from_metadata(str(metadata_path))
    
    print("\n" + "="*50)
    print("✅ Index chargé avec succès!")
    print(f"   📊 Chunks: {stats['total_chunks']}")
    print("="*50)


if __name__ == "__main__":
    main()
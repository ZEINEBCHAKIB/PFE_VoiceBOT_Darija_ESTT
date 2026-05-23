"""
Script de test pour poser des questions au RAG
C'est ICI que tu poses tes questions !
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.retrieval import get_retriever
from app.core.llm import get_llm_client
from app.config.settings import config


def ask_question(question: str, verbose: bool = True):
    """
    Pose une question au système RAG
    
    Args:
        question: Question en français ou darija
        verbose: Afficher les détails
    """
    retriever = get_retriever()
    llm = get_llm_client()
    
    if verbose:
        print("\n" + "="*60)
        print(f"🔍 Question: {question}")
        print("="*60)
    
    # 1. Recherche dans l'index
    retrieval = retriever.retrieve_with_context(question)
    
    if not retrieval["has_results"]:
        print("\n⚠️ Aucun résultat trouvé")
        return {"response": "Désolé, je n'ai pas trouvé d'information."}
    
    if verbose:
        print(f"\n📊 {len(retrieval['results'])} documents trouvés:")
        for r in retrieval["results"]:
            print(f"   • [{r['theme']}] Score: {r['score']:.3f}")
            print(f"     {r['text'][:100]}...\n")
    
    # 2. Génération de la réponse
    response = llm.generate_response(question, retrieval["context"])
    
    if verbose:
        print(f"\n💬 RÉPONSE:\n{'-'*40}")
        print(response)
        print("-"*40)
    
    return {
        "question": question,
        "response": response,
        "sources": retrieval["results"],
        "themes": retrieval["themes"]
    }


def main():
    """Boucle interactive pour poser des questions"""
    print("\n" + "="*60)
    print("🤖 CTM VOICEBOT - RAG READY")
    print("="*60)
    print("\nPose tes questions sur CTM (voyages, colis, politiques)")
    print("Tape 'quit' ou 'exit' pour quitter\n")
    
    # Vérifier que l'index est chargé
    from app.rag.qdrant_store import get_qdrant_store
    store = get_qdrant_store()
    stats = store.get_stats()
    
    if stats.get("points_count", 0) == 0:
        print("⚠️ Index vide ! Exécute d'abord: python scripts/load_index.py")
        print("   pour charger chunks_metadata.json\n")
        return
    
    print(f"✅ Index chargé: {stats['points_count']} chunks disponibles\n")
    
    while True:
        try:
            question = input("\n❓ Vous: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir!")
                break
            
            if not question:
                continue
            
            ask_question(question, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")


if __name__ == "__main__":
    main()
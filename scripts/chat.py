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


def is_darija(text: str) -> bool:
    """
    Détecte si le texte est en darija (arabe ou latin translittéré).
    - Arabe : caractères Unicode arabic block
    - Latin darija : mots-clés typiques (bghit, wach, fin, kifash, etc.)
    """
    # Caractères arabes
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars > 0:
        return True

    # Mots-clés darija en latin
    darija_keywords = [
        "bghit", "bgha", "wach", "wash", "fin", "feen", "kifash", "kifach",
        "chhal", "chno", "mnin", "imta", "ach", "ach", "ndir", "nsift",
        "walo", "zwina", "mzyan", "dyal", "dial", "kayn", "makaynch",
        "l ", "b ", "f ", "w ", "m ", "casa", "rbat", "marrakech"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in darija_keywords)


def ask_question(question: str, verbose: bool = True):
    """
    Pose une question au système RAG.
    Traduit automatiquement la darija en français avant la recherche.

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

    # 1. Traduction darija → français si nécessaire
    search_query = question
    if is_darija(question):
        translated = llm.translate_to_french(question)
        if translated and translated != question:
            search_query = translated
            if verbose:
                print(f"\n🔄 Traduit en: {search_query}")

    # 2. Recherche dans l'index (toujours en français)
    retrieval = retriever.retrieve_with_context(search_query)

    if not retrieval["has_results"]:
        print("\n⚠️ Aucun résultat trouvé")
        return {"response": "ما لقيت حتى معلومة على هاد السؤال."}

    if verbose:
        print(f"\n📊 {len(retrieval['results'])} documents trouvés:")
        for r in retrieval["results"]:
            print(f"   • [{r['theme']}] Score: {r['score']:.3f}")
            print(f"     {r['text'][:100]}...\n")

    # 3. Génération de la réponse (toujours en darija, question originale)
    response = llm.rag_generate(question, retrieval["context"])

    if verbose:
        print(f"\n💬 RÉPONSE:\n{'-'*40}")
        print(response)
        print("-"*40)

    return {
        "question": question,
        "search_query": search_query,
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
    print("Darija acceptée 🇲🇦 | Tape 'quit' pour quitter\n")

    # Vérifier que l'index est chargé
    from app.rag.qdrant_store import get_qdrant_store
    store = get_qdrant_store()
    stats = store.get_stats()

    if stats.get("points_count", 0) == 0:
        print("⚠️ Index vide ! Exécute d'abord: python scripts/load_index.py")
        return

    print(f"✅ Index chargé: {stats['points_count']} chunks disponibles\n")

    while True:
        try:
            question = input("\n❓ Vous: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 بسلامة!")
                break

            if not question:
                continue

            ask_question(question, verbose=True)

        except KeyboardInterrupt:
            print("\n\n👋 بسلامة!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")


if __name__ == "__main__":
    main()
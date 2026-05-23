"""
Découpage intelligent des documents en chunks
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Découpe un texte en chunks avec chevauchement
    
    Args:
        text: Texte à découper
        chunk_size: Taille maximale du chunk en caractères
        overlap: Nombre de mots à garder en chevauchement
    
    Returns:
        Liste de chunks avec métadonnées
    """
    chunks = []
    
    # Nettoyage du texte
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Séparation par phrases
    sentences = re.split(r'(?<=[.!?:؟\n])\s+', text)
    
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "length": len(current_chunk),
                    "num_sentences": len(current_chunk.split('. '))
                })
                chunk_id += 1
                
                # Overlap: garder les derniers mots
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence + " "
            else:
                current_chunk = sentence + " "
    
    # Dernier chunk
    if current_chunk:
        chunks.append({
            "id": chunk_id,
            "text": current_chunk.strip(),
            "length": len(current_chunk),
            "num_sentences": len(current_chunk.split('. '))
        })
    
    return chunks


def chunk_documents(documents: Dict[str, str], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Découpe tous les documents en chunks
    
    Args:
        documents: Dictionnaire {theme: contenu}
        chunk_size: Taille des chunks
        overlap: Chevauchement
    
    Returns:
        Liste de tous les chunks avec métadonnées (theme, chunk_id, etc.)
    """
    all_chunks = []
    
    for theme, content in documents.items():
        chunks = chunk_text(content, chunk_size, overlap)
        logger.info(f"  📌 {theme}: {len(chunks)} chunks")
        
        for chunk in chunks:
            all_chunks.append({
                "id": len(all_chunks),
                "text": chunk["text"],
                "theme": theme,
                "chunk_id": chunk["id"],
                "length": chunk["length"],
                "num_sentences": chunk.get("num_sentences", 0)
            })
    
    return all_chunks
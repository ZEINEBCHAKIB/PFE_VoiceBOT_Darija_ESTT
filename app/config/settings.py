"""
Configuration du projet voicebot-ctm
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = BASE_DIR / "models"

# Créer les dossiers si nécessaire
for d in [DATA_DIR, RAW_DATA_DIR, INDEX_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Config:
    """Configuration principale"""

    # ========== MODÈLES ==========
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ========== RAG ==========
    VECTOR_SIZE = 1024  # Dimension BGE-M3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.55  # Seuil ajusté pour darija

    # ========== QDRANT ==========
    QDRANT_COLLECTION = "ctm_documents"
    QDRANT_MODE = os.getenv("QDRANT_MODE", "memory")  # "memory", "local", "cloud"
    QDRANT_PATH = str(INDEX_DIR / "qdrant_storage")

    # ========== API KEYS ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # ========== FICHIERS ==========
    DOCUMENTS = {
        "colis": RAW_DATA_DIR / "colis.txt",
        "voyages": RAW_DATA_DIR / "voyages.txt",
        "politiques": RAW_DATA_DIR / "politiques.txt",
    }

    METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

    @classmethod
    def validate(cls):
        """Vérifier la configuration"""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY non définie")

        for name, path in cls.DOCUMENTS.items():
            if not path.exists():
                errors.append(f"Fichier {name} non trouvé: {path}")

        return errors


class DevelopmentConfig(Config):
    """Configuration de développement"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Configuration de production"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    QDRANT_MODE = "local"  # Mode persistant en prod


def get_config():
    """Retourne la configuration selon l'environnement"""
    env = os.getenv("ENV", "development").lower()
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()


config = get_config()
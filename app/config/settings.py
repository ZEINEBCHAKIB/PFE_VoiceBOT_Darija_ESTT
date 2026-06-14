"""
Configuration du projet voicebot-ctm
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = BASE_DIR / "models"

# ✅ NOUVEAU — dossier base de données
DB_DIR = BASE_DIR / "app" / "database"

# Créer les dossiers si nécessaire
for d in [DATA_DIR, RAW_DATA_DIR, INDEX_DIR, MODELS_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Config:
    """Configuration principale"""

    # ========== MODÈLES ==========
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ========== RAG ==========
    VECTOR_SIZE = 1024
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.55

    # ========== QDRANT ==========
    QDRANT_COLLECTION = "ctm_documents"
    QDRANT_MODE = os.getenv("QDRANT_MODE", "memory")
    QDRANT_PATH = str(INDEX_DIR / "qdrant_storage")

    # ========== API KEYS ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # ========== FICHIERS RAG ==========
    DOCUMENTS = {
        "colis": RAW_DATA_DIR / "colis.txt",
        "voyages": RAW_DATA_DIR / "voyages.txt",
        "politiques": RAW_DATA_DIR / "politiques.txt",
    }
    METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

    # ========== BASE DE DONNÉES SQLite ========== ✅ NOUVEAU
    # Le fichier .db sera créé automatiquement au premier lancement
    # dans app/database/ctm_data.db
    DB_PATH = str(DB_DIR / "ctm_data.db")
    DB_URL = f"sqlite:///{DB_PATH}"

    @classmethod
    def validate(cls):
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY non définie")
        for name, path in cls.DOCUMENTS.items():
            if not path.exists():
                errors.append(f"Fichier {name} non trouvé: {path}")
        return errors


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"
    QDRANT_MODE = "local"


def get_config():
    env = os.getenv("ENV", "development").lower()
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()


config = get_config()

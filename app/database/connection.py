"""
Connexion SQLite via SQLAlchemy.

Ce fichier fait UNE seule chose :
  - créer le moteur de connexion (engine)
  - fournir des sessions propres avec fermeture garantie
  - créer toutes les tables au premier lancement

Usage dans le reste du code :
    from app.database.connection import get_db
    with get_db() as db:
        db.query(Ligne).all()
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config.settings import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Base pour tous les modèles SQLAlchemy
# Tous les models.py héritent de cette classe
# ──────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────
# Moteur SQLite
# check_same_thread=False → nécessaire pour FastAPI (multi-thread)
# ──────────────────────────────────────────────────────────────
engine = create_engine(
    config.DB_URL,
    connect_args={"check_same_thread": False},
    echo=False  # Passe à True pour voir toutes les requêtes SQL en debug
)


# ──────────────────────────────────────────────────────────────
# Activer les clés étrangères dans SQLite
# SQLite les désactive par défaut — on les force à chaque connexion
# Sans ça, l'intégrité référentielle n'est pas garantie
# ──────────────────────────────────────────────────────────────
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ──────────────────────────────────────────────────────────────
# Factory de sessions
# autocommit=False → on contrôle nous-mêmes les transactions
# autoflush=False  → on choisit quand envoyer les changements
# ──────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)


# ──────────────────────────────────────────────────────────────
# Context manager de session
# Garantit la fermeture même en cas d'erreur
# ──────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    """
    Fournit une session DB avec fermeture garantie.

    Exemple d'utilisation :
        with get_db() as db:
            lignes = db.query(Ligne).all()
            # session fermée automatiquement ici
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Erreur DB, rollback effectué : {e}")
        raise
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────
# Création des tables
# Appelé une seule fois au démarrage de l'application
# Si les tables existent déjà → rien ne se passe (checkfirst=True)
# ──────────────────────────────────────────────────────────────
def init_db():
    """
    Crée toutes les tables dans ctm_data.db si elles n'existent pas.
    À appeler au démarrage dans app/api.py ou app/main.py.
    """
    # Import obligatoire ici pour que SQLAlchemy connaisse les modèles
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine, checkfirst=True)
    logger.info(f"✅ Base de données initialisée → {config.DB_PATH}")

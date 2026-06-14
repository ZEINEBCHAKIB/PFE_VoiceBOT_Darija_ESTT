"""
Modèles SQLAlchemy — Tables de la base de données CTM.

Tables :
  - Ligne        : les lignes de bus CTM
  - Horaire      : départs par ligne
  - Tarif        : prix par classe et ligne
  - Agence       : agences CTM par ville
  - Reclamation  : réclamations clients
  - Colis        : suivi des envois CTM Messagerie
  - CallLog      : journal de bord de chaque interaction bot
"""

from datetime import datetime, date, time
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Date, Time, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship

from app.database.connection import Base


# ══════════════════════════════════════════════════════════════
# TABLE : Ligne
# Table centrale — Horaires et Tarifs en dépendent
# ══════════════════════════════════════════════════════════════
class Ligne(Base):
    __tablename__ = "lignes"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    code            = Column(String(20), nullable=False, unique=True)   # ex: "TNG-CAS"
    ville_depart    = Column(String(100), nullable=False)
    ville_arrivee   = Column(String(100), nullable=False)
    duree_minutes   = Column(Integer, nullable=False)                   # durée estimée du trajet
    distance_km     = Column(Integer, nullable=True)
    active          = Column(Boolean, default=True)

    # Relations : une ligne a plusieurs horaires et plusieurs tarifs
    horaires        = relationship("Horaire", back_populates="ligne", cascade="all, delete-orphan")
    tarifs          = relationship("Tarif",   back_populates="ligne", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Ligne {self.code}: {self.ville_depart} → {self.ville_arrivee}>"


# ══════════════════════════════════════════════════════════════
# TABLE : Horaire
# Dépend de Ligne (clé étrangère)
# ══════════════════════════════════════════════════════════════
class Horaire(Base):
    __tablename__ = "horaires"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    ligne_id        = Column(Integer, ForeignKey("lignes.id", ondelete="CASCADE"), nullable=False)
    heure_depart    = Column(String(5), nullable=False)    # format "HH:MM"
    jours_service   = Column(String(50), default="QUOTIDIEN")  # "QUOTIDIEN", "LUN-VEN", "WEEKEND"
    gare_depart     = Column(String(200), nullable=True)

    ligne           = relationship("Ligne", back_populates="horaires")

    def __repr__(self):
        return f"<Horaire ligne={self.ligne_id} départ={self.heure_depart}>"


# ══════════════════════════════════════════════════════════════
# TABLE : Tarif
# Dépend de Ligne (clé étrangère)
# ══════════════════════════════════════════════════════════════
class Tarif(Base):
    __tablename__ = "tarifs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    ligne_id        = Column(Integer, ForeignKey("lignes.id", ondelete="CASCADE"), nullable=False)
    classe          = Column(String(20), nullable=False)    # "STANDARD", "CONFORT"
    prix_dh         = Column(Float, nullable=False)

    ligne           = relationship("Ligne", back_populates="tarifs")

    def __repr__(self):
        return f"<Tarif ligne={self.ligne_id} classe={self.classe} prix={self.prix_dh}DH>"


# ══════════════════════════════════════════════════════════════
# TABLE : Agence
# Table indépendante — pas de lien avec Ligne
# ══════════════════════════════════════════════════════════════
class Agence(Base):
    __tablename__ = "agences"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    ville               = Column(String(100), nullable=False, unique=True)
    adresse             = Column(Text, nullable=False)
    telephone           = Column(String(20), nullable=True)
    horaires_ouverture  = Column(String(100), default="08h00 - 20h00")
    latitude            = Column(Float, nullable=True)   # pour navigation future
    longitude           = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Agence {self.ville}>"


# ══════════════════════════════════════════════════════════════
# TABLE : Reclamation
# Table indépendante — liée au client, pas aux lignes
# ══════════════════════════════════════════════════════════════
class Reclamation(Base):
    __tablename__ = "reclamations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    reference       = Column(String(20), nullable=False, unique=True)  # "REC-2025-0001"
    telephone       = Column(String(20), nullable=False)
    description     = Column(Text, nullable=False)
    # Statuts possibles : OUVERTE → EN_COURS → RESOLUE
    statut          = Column(String(20), nullable=False, default="OUVERTE")
    date_creation   = Column(DateTime, default=datetime.utcnow)
    date_resolution = Column(DateTime, nullable=True)
    commentaire     = Column(Text, nullable=True)  # note interne opérateur

    def __repr__(self):
        return f"<Reclamation {self.reference} statut={self.statut}>"


# ══════════════════════════════════════════════════════════════
# TABLE : Colis
# Suivi CTM Messagerie
# Table indépendante
# ══════════════════════════════════════════════════════════════
class Colis(Base):
    __tablename__ = "colis"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    numero_suivi        = Column(String(30), nullable=False, unique=True)  # "CTM-COL-2025-0001"
    expediteur_nom      = Column(String(100), nullable=False)
    expediteur_tel      = Column(String(20), nullable=False)
    destinataire_nom    = Column(String(100), nullable=False)
    destinataire_tel    = Column(String(20), nullable=False)
    ville_depart        = Column(String(100), nullable=False)
    ville_arrivee       = Column(String(100), nullable=False)
    poids_kg            = Column(Float, nullable=True)
    # Statuts : EN_PREPARATION → EN_TRANSIT → ARRIVE_AGENCE → LIVRE → RETOURNE
    statut              = Column(String(30), nullable=False, default="EN_PREPARATION")
    date_envoi          = Column(DateTime, nullable=True)
    date_livraison_prevue = Column(String(20), nullable=True)  # ex: "12/06/2025"
    date_livraison_reelle = Column(DateTime, nullable=True)
    localisation_actuelle = Column(String(100), nullable=True)  # ex: "Agence CTM Rabat"

    def __repr__(self):
        return f"<Colis {self.numero_suivi} statut={self.statut}>"


# ══════════════════════════════════════════════════════════════
# TABLE : CallLog
# Journal de bord du bot — rempli automatiquement par call_logger
# Jamais appelé par Gemini, jamais vu du client
# ══════════════════════════════════════════════════════════════
class CallLog(Base):
    __tablename__ = "call_logs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(String(100), nullable=False)   # id WebSocket
    transcript      = Column(Text, nullable=True)           # ce que le client a dit
    tool_utilise    = Column(String(50), nullable=True)     # "rag_search" ou "ctm_db_query"
    reponse         = Column(Text, nullable=True)           # ce que le bot a répondu
    duree_ms        = Column(Integer, nullable=True)        # temps de traitement en ms
    succes          = Column(Boolean, default=True)         # False si erreur technique
    timestamp       = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CallLog session={self.session_id} tool={self.tool_utilise} durée={self.duree_ms}ms>"

"""
Repository CTM — Couche d'accès aux données.

Principe : tout le SQL est ici, nulle part ailleurs.
Les tools MCP appellent ces fonctions, pas SQLAlchemy directement.

Fonctions disponibles :
  Lecture  → get_horaires, get_tarifs, get_agence,
             get_reclamation_by_ref, get_reclamations_by_tel,
             get_colis, get_stats_reclamations
  Écriture → create_reclamation
  Logging  → log_call
"""

import logging
from datetime import datetime
from typing import Optional

from app.database.connection import get_db
from app.database.models import (
    Ligne, Horaire, Tarif, Agence,
    Reclamation, Colis, CallLog
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# HORAIRES
# ══════════════════════════════════════════════════════════════

def get_horaires(ville_depart: str, ville_arrivee: str) -> dict:
    """
    Retourne tous les horaires pour une liaison donnée.
    Recherche insensible à la casse et aux accents partiels.
    """
    with get_db() as db:
        lignes = db.query(Ligne).filter(
            Ligne.ville_depart.ilike(f"%{ville_depart}%"),
            Ligne.ville_arrivee.ilike(f"%{ville_arrivee}%"),
            Ligne.active == True
        ).all()

        if not lignes:
            return {
                "found": False,
                "message": f"Aucune ligne trouvée entre {ville_depart} et {ville_arrivee}"
            }

        result = []
        for ligne in lignes:
            horaires = db.query(Horaire).filter(
                Horaire.ligne_id == ligne.id
            ).order_by(Horaire.heure_depart).all()

            result.append({
                "ligne": ligne.code,
                "depart": ligne.ville_depart,
                "arrivee": ligne.ville_arrivee,
                "duree_minutes": ligne.duree_minutes,
                "distance_km": ligne.distance_km,
                "horaires": [
                    {
                        "heure": h.heure_depart,
                        "jours": h.jours_service,
                        "gare": h.gare_depart
                    }
                    for h in horaires
                ]
            })

        return {"found": True, "lignes": result}


# ══════════════════════════════════════════════════════════════
# TARIFS
# ══════════════════════════════════════════════════════════════

def get_tarifs(ville_depart: str, ville_arrivee: str) -> dict:
    """
    Retourne les tarifs standard et confort pour une liaison.
    """
    with get_db() as db:
        lignes = db.query(Ligne).filter(
            Ligne.ville_depart.ilike(f"%{ville_depart}%"),
            Ligne.ville_arrivee.ilike(f"%{ville_arrivee}%"),
            Ligne.active == True
        ).all()

        if not lignes:
            return {
                "found": False,
                "message": f"Aucun tarif trouvé pour {ville_depart} → {ville_arrivee}"
            }

        result = []
        for ligne in lignes:
            tarifs = db.query(Tarif).filter(
                Tarif.ligne_id == ligne.id
            ).all()

            result.append({
                "ligne": ligne.code,
                "depart": ligne.ville_depart,
                "arrivee": ligne.ville_arrivee,
                "tarifs": [
                    {"classe": t.classe, "prix_dh": t.prix_dh}
                    for t in tarifs
                ]
            })

        return {"found": True, "lignes": result}


# ══════════════════════════════════════════════════════════════
# AGENCES
# ══════════════════════════════════════════════════════════════

def get_agence(ville: str) -> dict:
    """
    Retourne les infos de l'agence CTM d'une ville.
    """
    with get_db() as db:
        agence = db.query(Agence).filter(
            Agence.ville.ilike(f"%{ville}%")
        ).first()

        if not agence:
            return {
                "found": False,
                "message": f"Aucune agence CTM trouvée à {ville}"
            }

        return {
            "found": True,
            "ville": agence.ville,
            "adresse": agence.adresse,
            "telephone": agence.telephone,
            "horaires_ouverture": agence.horaires_ouverture
        }


# ══════════════════════════════════════════════════════════════
# RÉCLAMATIONS — Lecture
# ══════════════════════════════════════════════════════════════

def get_reclamation_by_ref(reference: str) -> dict:
    """
    Cherche une réclamation par son numéro de référence.
    Ex: "REC-2025-0001"
    """
    with get_db() as db:
        rec = db.query(Reclamation).filter(
            Reclamation.reference.ilike(f"%{reference}%")
        ).first()

        if not rec:
            return {
                "found": False,
                "message": f"Aucune réclamation trouvée avec la référence {reference}"
            }

        return {
            "found": True,
            "reference": rec.reference,
            "statut": rec.statut,
            "description": rec.description,
            "date_creation": rec.date_creation.strftime("%d/%m/%Y %H:%M") if rec.date_creation else None,
            "date_resolution": rec.date_resolution.strftime("%d/%m/%Y") if rec.date_resolution else "En attente",
            "commentaire": rec.commentaire
        }


def get_reclamations_by_tel(telephone: str) -> dict:
    """
    Cherche toutes les réclamations d'un client par son numéro de téléphone.
    """
    with get_db() as db:
        recs = db.query(Reclamation).filter(
            Reclamation.telephone == telephone
        ).order_by(Reclamation.date_creation.desc()).all()

        if not recs:
            return {
                "found": False,
                "message": f"Aucune réclamation trouvée pour le numéro {telephone}"
            }

        return {
            "found": True,
            "count": len(recs),
            "reclamations": [
                {
                    "reference": r.reference,
                    "statut": r.statut,
                    "description": r.description[:80] + "..." if len(r.description) > 80 else r.description,
                    "date_creation": r.date_creation.strftime("%d/%m/%Y") if r.date_creation else None
                }
                for r in recs
            ]
        }


# ══════════════════════════════════════════════════════════════
# RÉCLAMATIONS — Écriture (action agentique)
# ══════════════════════════════════════════════════════════════

def create_reclamation(telephone: str, description: str) -> dict:
    """
    Crée une nouvelle réclamation et retourne sa référence.
    C'est ici que le bot AGIT sur la base de données.
    """
    with get_db() as db:
        # Générer un numéro de référence unique
        count = db.query(Reclamation).count()
        reference = f"REC-{datetime.utcnow().year}-{str(count + 1).zfill(4)}"

        nouvelle = Reclamation(
            reference=reference,
            telephone=telephone,
            description=description,
            statut="OUVERTE",
            date_creation=datetime.utcnow()
        )
        db.add(nouvelle)
        # Le commit est géré par get_db()

        logger.info(f"✅ Réclamation créée : {reference} pour {telephone}")

        return {
            "success": True,
            "reference": reference,
            "statut": "OUVERTE",
            "message": f"Réclamation {reference} enregistrée avec succès"
        }


# ══════════════════════════════════════════════════════════════
# COLIS
# ══════════════════════════════════════════════════════════════

def get_colis(numero_suivi: str) -> dict:
    """
    Retourne le statut d'un colis par son numéro de suivi.
    Ex: "CTM-COL-2025-0001"
    """
    with get_db() as db:
        colis = db.query(Colis).filter(
            Colis.numero_suivi.ilike(f"%{numero_suivi}%")
        ).first()

        if not colis:
            return {
                "found": False,
                "message": f"Aucun colis trouvé avec le numéro {numero_suivi}"
            }

        return {
            "found": True,
            "numero_suivi": colis.numero_suivi,
            "statut": colis.statut,
            "expediteur": colis.expediteur_nom,
            "destinataire": colis.destinataire_nom,
            "destinataire_tel": colis.destinataire_tel,
            "trajet": f"{colis.ville_depart} → {colis.ville_arrivee}",
            "date_envoi": colis.date_envoi.strftime("%d/%m/%Y") if colis.date_envoi else None,
            "livraison_prevue": colis.date_livraison_prevue,
            "localisation": colis.localisation_actuelle
        }


def get_colis_by_tel(telephone: str) -> dict:
    """
    Cherche les colis d'un destinataire par téléphone.
    """
    with get_db() as db:
        resultats = db.query(Colis).filter(
            Colis.destinataire_tel == telephone
        ).order_by(Colis.date_envoi.desc()).limit(5).all()

        if not resultats:
            return {
                "found": False,
                "message": f"Aucun colis trouvé pour le numéro {telephone}"
            }

        return {
            "found": True,
            "count": len(resultats),
            "colis": [
                {
                    "numero_suivi": c.numero_suivi,
                    "statut": c.statut,
                    "trajet": f"{c.ville_depart} → {c.ville_arrivee}",
                    "localisation": c.localisation_actuelle
                }
                for c in resultats
            ]
        }


# ══════════════════════════════════════════════════════════════
# LOGGING — Appelé automatiquement après chaque réponse du bot
# ══════════════════════════════════════════════════════════════

def log_call(
    session_id: str,
    transcript: str,
    tool_utilise: str,
    reponse: str,
    duree_ms: int,
    succes: bool = True
) -> None:
    """
    Enregistre une interaction dans call_logs.
    Silencieux — ne lève pas d'exception si ça échoue
    pour ne pas bloquer la réponse au client.
    """
    try:
        with get_db() as db:
            log = CallLog(
                session_id=session_id,
                transcript=transcript,
                tool_utilise=tool_utilise,
                reponse=reponse[:500] if reponse else None,  # tronquer si trop long
                duree_ms=duree_ms,
                succes=succes,
                timestamp=datetime.utcnow()
            )
            db.add(log)
        logger.debug(f"📝 CallLog enregistré — session={session_id} tool={tool_utilise} durée={duree_ms}ms")

    except Exception as e:
        # On logge l'erreur mais on ne la propage pas
        # Le client doit toujours recevoir sa réponse même si le log échoue
        logger.error(f"⚠️ Échec log_call (non bloquant) : {e}")

"""
Données synthétiques CTM — à lancer UNE SEULE FOIS.

Lance avec : python -m app.database.seed_data
depuis la racine du projet voicebot-ctm.

Ce script :
  1. Crée toutes les tables (si pas encore fait)
  2. Vérifie que la DB est vide (évite les doublons)
  3. Insère toutes les données de test
"""

import logging
from datetime import datetime
from app.database.connection import init_db, get_db
from app.database.models import Ligne, Horaire, Tarif, Agence, Reclamation, Colis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed():
    logger.info("🌱 Démarrage du seeding des données CTM...")

    # Créer les tables d'abord
    init_db()

    with get_db() as db:

        # Vérifier si déjà seedé
        if db.query(Ligne).count() > 0:
            logger.info("⚠️  La base contient déjà des données. Seeding annulé.")
            return

        # ══════════════════════════════════════════════════
        # LIGNES CTM (10 lignes réelles)
        # ══════════════════════════════════════════════════
        lignes_data = [
            {"code": "TNG-CAS", "ville_depart": "Tanger",    "ville_arrivee": "Casablanca", "duree_minutes": 270, "distance_km": 340},
            {"code": "TNG-FES", "ville_depart": "Tanger",    "ville_arrivee": "Fès",        "duree_minutes": 210, "distance_km": 250},
            {"code": "TTN-CAS", "ville_depart": "Tétouan",   "ville_arrivee": "Casablanca", "duree_minutes": 300, "distance_km": 380},
            {"code": "TTN-TNG", "ville_depart": "Tétouan",   "ville_arrivee": "Tanger",     "duree_minutes":  60, "distance_km":  60},
            {"code": "RBA-CAS", "ville_depart": "Rabat",     "ville_arrivee": "Casablanca", "duree_minutes":  90, "distance_km":  90},
            {"code": "CAS-MAR", "ville_depart": "Casablanca","ville_arrivee": "Marrakech",  "duree_minutes": 210, "distance_km": 240},
            {"code": "CAS-AGD", "ville_depart": "Casablanca","ville_arrivee": "Agadir",     "duree_minutes": 330, "distance_km": 460},
            {"code": "FES-OUJ", "ville_depart": "Fès",       "ville_arrivee": "Oujda",      "duree_minutes": 270, "distance_km": 320},
            {"code": "MAR-AGD", "ville_depart": "Marrakech", "ville_arrivee": "Agadir",     "duree_minutes": 180, "distance_km": 200},
            {"code": "CAS-TNG", "ville_depart": "Casablanca","ville_arrivee": "Tanger",     "duree_minutes": 270, "distance_km": 340},
        ]

        lignes_obj = {}
        for ld in lignes_data:
            l = Ligne(**ld, active=True)
            db.add(l)
            db.flush()  # pour récupérer l.id
            lignes_obj[ld["code"]] = l

        logger.info(f"  ✅ {len(lignes_data)} lignes insérées")

        # ══════════════════════════════════════════════════
        # HORAIRES (3 à 5 départs par ligne)
        # ══════════════════════════════════════════════════
        horaires_data = {
            "TNG-CAS": [
                ("07:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("10:30", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("14:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("18:30", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("23:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
            ],
            "TNG-FES": [
                ("08:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("13:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
                ("19:00", "QUOTIDIEN", "Gare CTM Tanger, Av. Louis Van Beethoven"),
            ],
            "TTN-CAS": [
                ("06:30", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("09:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("13:30", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("17:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
            ],
            "TTN-TNG": [
                ("07:30", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("10:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("12:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("15:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("18:00", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
                ("20:30", "QUOTIDIEN", "Gare CTM Tétouan, Av. Hassan II"),
            ],
            "RBA-CAS": [
                ("06:00", "QUOTIDIEN", "Gare CTM Rabat, Av. Hassan II"),
                ("08:00", "QUOTIDIEN", "Gare CTM Rabat, Av. Hassan II"),
                ("11:00", "QUOTIDIEN", "Gare CTM Rabat, Av. Hassan II"),
                ("15:00", "QUOTIDIEN", "Gare CTM Rabat, Av. Hassan II"),
                ("19:00", "QUOTIDIEN", "Gare CTM Rabat, Av. Hassan II"),
            ],
            "CAS-MAR": [
                ("07:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("11:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("16:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("21:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
            ],
            "CAS-AGD": [
                ("08:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("14:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("22:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
            ],
            "FES-OUJ": [
                ("07:30", "QUOTIDIEN", "Gare CTM Fès, Av. Mohammed V"),
                ("13:00", "QUOTIDIEN", "Gare CTM Fès, Av. Mohammed V"),
                ("20:00", "QUOTIDIEN", "Gare CTM Fès, Av. Mohammed V"),
            ],
            "MAR-AGD": [
                ("08:30", "QUOTIDIEN", "Gare CTM Marrakech, Av. Hassan II"),
                ("14:00", "QUOTIDIEN", "Gare CTM Marrakech, Av. Hassan II"),
                ("19:30", "QUOTIDIEN", "Gare CTM Marrakech, Av. Hassan II"),
            ],
            "CAS-TNG": [
                ("07:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("10:30", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("15:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
                ("20:00", "QUOTIDIEN", "Gare CTM Casablanca, Rue Léon L'Africain"),
            ],
        }

        count_h = 0
        for code, horaires in horaires_data.items():
            ligne = lignes_obj[code]
            for heure, jours, gare in horaires:
                db.add(Horaire(ligne_id=ligne.id, heure_depart=heure, jours_service=jours, gare_depart=gare))
                count_h += 1

        logger.info(f"  ✅ {count_h} horaires insérés")

        # ══════════════════════════════════════════════════
        # TARIFS (standard + confort par ligne)
        # ══════════════════════════════════════════════════
        tarifs_data = {
            "TNG-CAS": [("STANDARD", 110.0), ("CONFORT", 150.0)],
            "TNG-FES": [("STANDARD",  90.0), ("CONFORT", 125.0)],
            "TTN-CAS": [("STANDARD", 120.0), ("CONFORT", 165.0)],
            "TTN-TNG": [("STANDARD",  30.0), ("CONFORT",  45.0)],
            "RBA-CAS": [("STANDARD",  40.0), ("CONFORT",  60.0)],
            "CAS-MAR": [("STANDARD",  90.0), ("CONFORT", 130.0)],
            "CAS-AGD": [("STANDARD", 150.0), ("CONFORT", 210.0)],
            "FES-OUJ": [("STANDARD", 100.0), ("CONFORT", 140.0)],
            "MAR-AGD": [("STANDARD",  75.0), ("CONFORT", 110.0)],
            "CAS-TNG": [("STANDARD", 110.0), ("CONFORT", 150.0)],
        }

        count_t = 0
        for code, tarifs in tarifs_data.items():
            ligne = lignes_obj[code]
            for classe, prix in tarifs:
                db.add(Tarif(ligne_id=ligne.id, classe=classe, prix_dh=prix))
                count_t += 1

        logger.info(f"  ✅ {count_t} tarifs insérés")

        # ══════════════════════════════════════════════════
        # AGENCES (8 grandes villes)
        # ══════════════════════════════════════════════════
        agences_data = [
            {
                "ville": "Casablanca",
                "adresse": "Rue Léon L'Africain, Quartier des Hôpitaux, Casablanca",
                "telephone": "0522541010",
                "horaires_ouverture": "06h00 - 23h00",
                "latitude": 33.5731, "longitude": -7.5898
            },
            {
                "ville": "Tanger",
                "adresse": "Av. Louis Van Beethoven, près gare ferroviaire, Tanger",
                "telephone": "0539931234",
                "horaires_ouverture": "06h00 - 22h00",
                "latitude": 35.7595, "longitude": -5.8340
            },
            {
                "ville": "Tétouan",
                "adresse": "Av. Hassan II, Centre-ville, Tétouan",
                "telephone": "0539961212",
                "horaires_ouverture": "06h00 - 21h00",
                "latitude": 35.5785, "longitude": -5.3684
            },
            {
                "ville": "Rabat",
                "adresse": "Av. Hassan II, Quartier Hassan, Rabat",
                "telephone": "0537707070",
                "horaires_ouverture": "06h00 - 22h00",
                "latitude": 34.0209, "longitude": -6.8416
            },
            {
                "ville": "Fès",
                "adresse": "Av. Mohammed V, Centre-ville, Fès",
                "telephone": "0535621717",
                "horaires_ouverture": "06h00 - 22h00",
                "latitude": 34.0181, "longitude": -5.0078
            },
            {
                "ville": "Marrakech",
                "adresse": "Av. Hassan II, Guéliz, Marrakech",
                "telephone": "0524435525",
                "horaires_ouverture": "06h00 - 23h00",
                "latitude": 31.6295, "longitude": -7.9811
            },
            {
                "ville": "Agadir",
                "adresse": "Av. du Général Kettani, Centre-ville, Agadir",
                "telephone": "0528821020",
                "horaires_ouverture": "07h00 - 22h00",
                "latitude": 30.4278, "longitude": -9.5981
            },
            {
                "ville": "Oujda",
                "adresse": "Av. Mohammed V, Centre-ville, Oujda",
                "telephone": "0536681818",
                "horaires_ouverture": "07h00 - 21h00",
                "latitude": 34.6814, "longitude": -1.9086
            },
        ]

        for ad in agences_data:
            db.add(Agence(**ad))

        logger.info(f"  ✅ {len(agences_data)} agences insérées")

        # ══════════════════════════════════════════════════
        # RÉCLAMATIONS (5 exemples avec statuts variés)
        # ══════════════════════════════════════════════════
        reclamations_data = [
            {
                "reference": "REC-2025-0001",
                "telephone": "0661234567",
                "description": "Bus TNG-CAS du 01/06/2025 à 07h00 avait 2h30 de retard sans explication. J'ai raté un rendez-vous important.",
                "statut": "RESOLUE",
                "date_creation": datetime(2025, 6, 1, 10, 30),
                "date_resolution": datetime(2025, 6, 5, 14, 0),
                "commentaire": "Retard confirmé suite à incident routier. Bon de réduction 50DH accordé."
            },
            {
                "reference": "REC-2025-0002",
                "telephone": "0662345678",
                "description": "Climatisation en panne sur le bus TTN-CAS du 03/06/2025, trajet très difficile par forte chaleur.",
                "statut": "EN_COURS",
                "date_creation": datetime(2025, 6, 3, 16, 0),
                "date_resolution": None,
                "commentaire": "Signalé au responsable technique. En attente de vérification."
            },
            {
                "reference": "REC-2025-0003",
                "telephone": "0663456789",
                "description": "Bagage perdu sur le trajet Casablanca-Marrakech du 05/06/2025. Valise noire avec étiquette rouge.",
                "statut": "EN_COURS",
                "date_creation": datetime(2025, 6, 5, 20, 15),
                "date_resolution": None,
                "commentaire": "Recherche en cours dans les dépôts de Casablanca et Marrakech."
            },
            {
                "reference": "REC-2025-0004",
                "telephone": "0664567890",
                "description": "Remboursement non reçu suite à annulation de billet REF-2024-8821 il y a 3 semaines.",
                "statut": "OUVERTE",
                "date_creation": datetime(2025, 6, 8, 9, 0),
                "date_resolution": None,
                "commentaire": None
            },
            {
                "reference": "REC-2025-0005",
                "telephone": "0665678901",
                "description": "Chauffeur du bus FES-OUJ du 10/06 conduisait de façon dangereuse et utilisait son téléphone.",
                "statut": "OUVERTE",
                "date_creation": datetime(2025, 6, 10, 15, 45),
                "date_resolution": None,
                "commentaire": None
            },
        ]

        for rd in reclamations_data:
            db.add(Reclamation(**rd))

        logger.info(f"  ✅ {len(reclamations_data)} réclamations insérées")

        # ══════════════════════════════════════════════════
        # COLIS (5 exemples avec statuts variés)
        # ══════════════════════════════════════════════════
        colis_data = [
            {
                "numero_suivi": "CTM-COL-2025-0001",
                "expediteur_nom": "Ahmed Benali",
                "expediteur_tel": "0661111111",
                "destinataire_nom": "Fatima Zahra Alaoui",
                "destinataire_tel": "0662222222",
                "ville_depart": "Tanger",
                "ville_arrivee": "Casablanca",
                "poids_kg": 3.5,
                "statut": "LIVRE",
                "date_envoi": datetime(2025, 6, 8, 9, 0),
                "date_livraison_prevue": "10/06/2025",
                "date_livraison_reelle": datetime(2025, 6, 10, 14, 30),
                "localisation_actuelle": "Livré — Agence CTM Casablanca"
            },
            {
                "numero_suivi": "CTM-COL-2025-0002",
                "expediteur_nom": "Karim Tazi",
                "expediteur_tel": "0663333333",
                "destinataire_nom": "Youssef El Mansouri",
                "destinataire_tel": "0664444444",
                "ville_depart": "Fès",
                "ville_arrivee": "Marrakech",
                "poids_kg": 8.0,
                "statut": "EN_TRANSIT",
                "date_envoi": datetime(2025, 6, 12, 11, 0),
                "date_livraison_prevue": "14/06/2025",
                "date_livraison_reelle": None,
                "localisation_actuelle": "En transit — Agence CTM Rabat"
            },
            {
                "numero_suivi": "CTM-COL-2025-0003",
                "expediteur_nom": "Nadia Berrada",
                "expediteur_tel": "0665555555",
                "destinataire_nom": "Hassan Chraibi",
                "destinataire_tel": "0666666666",
                "ville_depart": "Tétouan",
                "ville_arrivee": "Agadir",
                "poids_kg": 1.2,
                "statut": "ARRIVE_AGENCE",
                "date_envoi": datetime(2025, 6, 11, 8, 30),
                "date_livraison_prevue": "13/06/2025",
                "date_livraison_reelle": None,
                "localisation_actuelle": "Disponible — Agence CTM Agadir, à retirer avec CIN"
            },
            {
                "numero_suivi": "CTM-COL-2025-0004",
                "expediteur_nom": "Salma Idrissi",
                "expediteur_tel": "0667777777",
                "destinataire_nom": "Omar Bensouda",
                "destinataire_tel": "0668888888",
                "ville_depart": "Casablanca",
                "ville_arrivee": "Oujda",
                "poids_kg": 5.0,
                "statut": "EN_PREPARATION",
                "date_envoi": None,
                "date_livraison_prevue": "16/06/2025",
                "date_livraison_reelle": None,
                "localisation_actuelle": "En préparation — Agence CTM Casablanca"
            },
            {
                "numero_suivi": "CTM-COL-2025-0005",
                "expediteur_nom": "Rachid Amrani",
                "expediteur_tel": "0669999999",
                "destinataire_nom": "Zineb El Fassi",
                "destinataire_tel": "0660000000",
                "ville_depart": "Rabat",
                "ville_arrivee": "Tanger",
                "poids_kg": 2.8,
                "statut": "EN_TRANSIT",
                "date_envoi": datetime(2025, 6, 13, 7, 0),
                "date_livraison_prevue": "15/06/2025",
                "date_livraison_reelle": None,
                "localisation_actuelle": "En transit — Agence CTM Tanger (arrivée prévue ce soir)"
            },
        ]

        for cd in colis_data:
            db.add(Colis(**cd))

        logger.info(f"  ✅ {len(colis_data)} colis insérés")

    logger.info("🎉 Seeding terminé avec succès !")
    logger.info(f"   📁 Base de données : {__import__('app.config.settings', fromlist=['config']).config.DB_PATH}")


if __name__ == "__main__":
    seed()

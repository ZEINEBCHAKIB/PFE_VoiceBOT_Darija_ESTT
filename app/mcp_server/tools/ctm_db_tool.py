"""
Tool MCP : ctm_db_query
Interroge la base de données CTM structurée.

Gemini router l'appelle pour :
  - Horaires précis (heure, gare, jours)
  - Tarifs (standard / confort)
  - Localisation et contacts des agences
  - Statut d'une réclamation
  - Création d'une réclamation (action agentique)
  - Suivi d'un colis CTM Messagerie

Structure calquée sur rag_tool.py pour cohérence.
"""

import logging
from typing import Dict, Any, Optional

from app.database import repository as repo
from app.core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class CTMDatabaseTool:
    """
    Tool MCP pour interroger et modifier la base de données CTM.
    Chaque méthode correspond à une intention utilisateur détectée.
    """

    # ──────────────────────────────────────────────
    # Point d'entrée principal appelé par le handler MCP
    # ──────────────────────────────────────────────

    def query(self, action: str, params: Dict[str, Any], memory: ConversationMemory = None) -> Dict[str, Any]:
        """
        Dispatch vers la bonne fonction selon l'action demandée.

        Actions disponibles :
          - horaires         : horaires d'une liaison
          - tarifs           : prix d'une liaison
          - agence           : infos d'une agence CTM
          - reclamation_get  : statut d'une réclamation (par ref ou tel)
          - reclamation_create : créer une nouvelle réclamation
          - colis            : suivi d'un colis (par numéro ou tel)
        """
        logger.info(f"🗄️  CTMDatabaseTool.query → action={action} params={params}")

        dispatch = {
            "horaires":             self._get_horaires,
            "tarifs":               self._get_tarifs,
            "agence":               self._get_agence,
            "reclamation_get":      self._get_reclamation,
            "reclamation_create":   self._create_reclamation,
            "colis":                self._get_colis,
        }

        handler = dispatch.get(action)
        if not handler:
            return {
                "success": False,
                "answer": f"Action '{action}' non reconnue. Actions disponibles : {list(dispatch.keys())}"
            }

        try:
            return handler(params)
        except Exception as e:
            logger.error(f"❌ CTMDatabaseTool erreur action={action} : {e}")
            return {
                "success": False,
                "answer": "عندي مشكل تقني فالقاعدة ديال المعطيات، عاود من فضلك."
            }

    # ──────────────────────────────────────────────
    # Horaires
    # ──────────────────────────────────────────────

    def _get_horaires(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ville_depart  = params.get("ville_depart", "")
        ville_arrivee = params.get("ville_arrivee", "")

        if not ville_depart or not ville_arrivee:
            return {
                "success": False,
                "answer": "من فضلك عطيني مدينة الانطلاق والوصول باش نقدر نعطيك المواعيد."
            }

        result = repo.get_horaires(ville_depart, ville_arrivee)

        if not result["found"]:
            return {
                "success": False,
                "answer": f"ما لقيتش رحلات بين {ville_depart} و{ville_arrivee} فالنظام."
            }

        # Formater la réponse pour Gemini ou réponse directe
        return {
            "success": True,
            "action": "horaires",
            "data": result,
            # answer = texte prêt à envoyer si Gemini ne reformule pas
            "answer": self._format_horaires(result)
        }

    def _format_horaires(self, result: dict) -> str:
        lines = []
        for l in result["lignes"]:
            heures = ", ".join([h["heure"] for h in l["horaires"]])
            lines.append(
                f"الخط {l['ligne']} ({l['depart']} ← {l['arrivee']}) — "
                f"المدة: {l['duree_minutes'] // 60}س{l['duree_minutes'] % 60:02d}د — "
                f"مواعيد الانطلاق: {heures}"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Tarifs
    # ──────────────────────────────────────────────

    def _get_tarifs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ville_depart  = params.get("ville_depart", "")
        ville_arrivee = params.get("ville_arrivee", "")

        if not ville_depart or not ville_arrivee:
            return {
                "success": False,
                "answer": "عطيني مدينة الانطلاق والوصول باش نعطيك الثمن."
            }

        result = repo.get_tarifs(ville_depart, ville_arrivee)

        if not result["found"]:
            return {
                "success": False,
                "answer": f"ما لقيتش أثمنة بين {ville_depart} و{ville_arrivee}."
            }

        return {
            "success": True,
            "action": "tarifs",
            "data": result,
            "answer": self._format_tarifs(result)
        }

    def _format_tarifs(self, result: dict) -> str:
        lines = []
        for l in result["lignes"]:
            tarifs_str = " | ".join([f"{t['classe']}: {t['prix_dh']} درهم" for t in l["tarifs"]])
            lines.append(f"الخط {l['ligne']}: {tarifs_str}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Agences
    # ──────────────────────────────────────────────

    def _get_agence(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ville = params.get("ville", "")

        if not ville:
            return {
                "success": False,
                "answer": "عطيني اسم المدينة باش نعطيك عنوان الوكالة."
            }

        result = repo.get_agence(ville)

        if not result["found"]:
            return {
                "success": False,
                "answer": f"ما لقيتش وكالة CTM ف{ville}."
            }

        return {
            "success": True,
            "action": "agence",
            "data": result,
            "answer": (
                f"وكالة CTM ف{result['ville']}:\n"
                f"العنوان: {result['adresse']}\n"
                f"الهاتف: {result['telephone']}\n"
                f"أوقات الفتح: {result['horaires_ouverture']}"
            )
        }

    # ──────────────────────────────────────────────
    # Réclamations — Lecture
    # ──────────────────────────────────────────────

    def _get_reclamation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        reference = params.get("reference", "")
        telephone = params.get("telephone", "")

        if reference:
            result = repo.get_reclamation_by_ref(reference)
            if not result["found"]:
                return {"success": False, "answer": f"ما لقيتش شكاية برقم {reference}."}

            statut_ar = {"OUVERTE": "مفتوحة", "EN_COURS": "قيد المعالجة", "RESOLUE": "محلولة"}.get(result["statut"], result["statut"])
            return {
                "success": True,
                "action": "reclamation_get",
                "data": result,
                "answer": (
                    f"الشكاية {result['reference']}:\n"
                    f"الحالة: {statut_ar}\n"
                    f"تاريخ التسجيل: {result['date_creation']}\n"
                    f"تاريخ الحل: {result['date_resolution']}"
                )
            }

        elif telephone:
            result = repo.get_reclamations_by_tel(telephone)
            if not result["found"]:
                return {"success": False, "answer": f"ما لقيتش شكايات لرقم {telephone}."}

            return {
                "success": True,
                "action": "reclamation_get",
                "data": result,
                "answer": f"لقيت {result['count']} شكاية لهاد الرقم."
            }

        else:
            return {
                "success": False,
                "answer": "عطيني رقم الشكاية أو رقم الهاتف باش نقدر نقلب عليها."
            }

    # ──────────────────────────────────────────────
    # Réclamations — Création (action agentique)
    # ──────────────────────────────────────────────

    def _create_reclamation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        telephone   = params.get("telephone", "")
        description = params.get("description", "")

        if not telephone or not description:
            return {
                "success": False,
                "answer": "باش نسجل الشكاية، محتاج رقم الهاتف ديالك ووصف المشكلة."
            }

        result = repo.create_reclamation(telephone, description)

        return {
            "success": True,
            "action": "reclamation_create",
            "data": result,
            "answer": (
                f"تسجلات الشكاية ديالك براقم {result['reference']}.\n"
                f"غادي نتواصلو معاك على {telephone} باش نحلو المشكلة."
            )
        }

    # ──────────────────────────────────────────────
    # Colis
    # ──────────────────────────────────────────────

    def _get_colis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        numero_suivi = params.get("numero_suivi", "")
        telephone    = params.get("telephone", "")

        statut_ar_map = {
            "EN_PREPARATION":  "قيد التحضير",
            "EN_TRANSIT":      "في الطريق",
            "ARRIVE_AGENCE":   "وصل للوكالة، ينتظر الاستلام",
            "LIVRE":           "تم التسليم",
            "RETOURNE":        "رجع للمرسل"
        }

        if numero_suivi:
            result = repo.get_colis(numero_suivi)
            if not result["found"]:
                return {"success": False, "answer": f"ما لقيتش طرد برقم {numero_suivi}."}

            statut_ar = statut_ar_map.get(result["statut"], result["statut"])
            return {
                "success": True,
                "action": "colis",
                "data": result,
                "answer": (
                    f"الطرد {result['numero_suivi']}:\n"
                    f"الحالة: {statut_ar}\n"
                    f"المسار: {result['trajet']}\n"
                    f"الموقع الحالي: {result['localisation']}\n"
                    f"موعد التسليم المتوقع: {result['livraison_prevue']}"
                )
            }

        elif telephone:
            result = repo.get_colis_by_tel(telephone)
            if not result["found"]:
                return {"success": False, "answer": f"ما لقيتش طرود لرقم {telephone}."}

            return {
                "success": True,
                "action": "colis",
                "data": result,
                "answer": f"لقيت {result['count']} طرد لهاد الرقم."
            }

        else:
            return {
                "success": False,
                "answer": "عطيني رقم التتبع ديال الطرد أو رقم الهاتف باش نقلب عليه."
            }

"""
tts/tts.py
-----------------
Client TTS cote VS Code (projet local Windows).

Envoie le texte darija genere par le bot (gen_text) vers le service Habibi-TTS MAR
qui tourne sur Google Colab (GPU T4), recupere l'audio clone et le sauvegarde en local.

Dependance :
    pip install gradio_client

Configuration :
    Renseignez l'URL publique *.gradio.live affichee par le notebook Colab,
    soit via la variable d'environnement COLAB_TTS_URL, soit en editant COLAB_TTS_URL ci-dessous.
"""

import os
import shutil
from pathlib import Path

from gradio_client import Client

# URL publique du service Colab (onglet "share" du notebook, ex: https://abc123.gradio.live)
COLAB_TTS_URL = os.environ.get(
    "COLAB_TTS_URL",
    "https://5c9e9e3126efb61403.gradio.live",  # <-- collez ici l'URL affichee par Colab
)

# Dossiers locaux du projet
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Client reutilise (connexion etablie une seule fois)
_client = None


def _get_client() -> Client:
    global _client
    if _client is None:
        if "CHANGE_ME" in COLAB_TTS_URL:
            raise RuntimeError(
                "COLAB_TTS_URL non configuree. Collez l'URL *.gradio.live du notebook Colab "
                "dans tts/tts_client.py ou exportez COLAB_TTS_URL."
            )
        _client = Client(COLAB_TTS_URL)
    return _client


def generate_tts(gen_text: str, out_name: str = "response_tts.wav") -> str:
    """
    Prend une reponse texte en darija generee localement par le bot.
    Envoie ce texte vers Habibi-TTS MAR (Colab), recupere le wav genere,
    le copie dans tts/outputs/ et retourne le chemin local du fichier audio.
    """
    if not gen_text or not gen_text.strip():
        raise ValueError("gen_text vide.")

    client = _get_client()
    # /synthesize correspond a l'api_name defini cote Gradio dans le notebook
    result_path = client.predict(gen_text, api_name="/synthesize")

    dest = OUTPUT_DIR / out_name
    shutil.copy(result_path, dest)
    return str(dest)


if __name__ == "__main__":
    # Test manuel : python tts/tts_client.py
    phrase = "مرحبا، الكار غادي يخرج من المحطة فالعاشرة ديال الصباح، شكرا على اتصالك."
    chemin = generate_tts(phrase)
    print("Audio TTS sauvegarde dans :", chemin)
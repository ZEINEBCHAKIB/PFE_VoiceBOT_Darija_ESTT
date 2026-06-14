"""
Module ASR (Automatic Speech Recognition) - Transcription audio → texte pour le Darija Marocain.
Utilise Wav2Vec2 ("boumehdi/wav2vec2-large-xlsr-moroccan-darija") et Silero-VAD pour la détection de parole.

⚠️ Ce module fait UNIQUEMENT de la transcription et de la détection de parole (VAD).
   Aucune logique métier, aucun appel direct à la base de données.
"""

import logging
import asyncio
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Optional

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Moteur de transcription audio → texte à basse latence pour le Darija.
    
    Fonctionnalités :
    - Détection de parole via Silero VAD (pour éliminer les silences et bruits).
    - Resampling rapide 16kHz via torchaudio.transforms.Resample.
    - Inférence Wav2Vec2 optimisée sur GPU/CPU (eval, no_grad, normalisation).
    - Inférence non-bloquante via asyncio.to_thread().
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ASR Device: {self.device}")

        # 1. Chargement de Wav2Vec2 Darija
        self.model_name = "boumehdi/wav2vec2-large-xlsr-moroccan-darija"
        logger.info(f"Chargement du modèle Wav2Vec2: {self.model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()  # Mode évaluation obligatoire pour la prod

        # 2. Chargement de Silero VAD
        logger.info("Chargement du modèle Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.get_speech_timestamps = self.vad_utils[0]

        self._initialized = True
        logger.info("✅ ASREngine initialisé (Wav2Vec2 + Silero VAD)")

    def is_speech(self, audio_tensor: torch.Tensor, sampling_rate: int = 16000) -> bool:
        """
        Détermine si un tenseur audio contient de la parole via Silero VAD.
        
        Args:
            audio_tensor: Tenseur 1D (float32) normalisé.
            sampling_rate: Fréquence d'échantillonnage (16000 ou 8000).
            
        Returns:
            True si de la parole est détectée, sinon False.
        """
        with torch.no_grad():
            timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=sampling_rate,
                threshold=0.5
            )
        return len(timestamps) > 0

    def resample_audio(self, audio_tensor: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Rééchantillonne rapidement l'audio à 16kHz mono via torchaudio transforms.
        """
        if orig_sr == 16000:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
        return resampler(audio_tensor)

    def _transcribe_sync(self, audio_tensor: torch.Tensor) -> str:
        """
        Méthode synchrone interne de transcription Wav2Vec2 (s'exécute dans un thread séparé).
        
        L'audio d'entrée doit être à 16kHz, mono, float32 et normalisé.
        """
        # S'assurer que le tenseur est 1D
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        # 1. Normalisation de l'audio entre -1.0 et 1.0 (float32) avant le VAD
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            audio_tensor = audio_tensor / max_val

        # 2. Détecter si l'audio contient de la parole (VAD)
        if not self.is_speech(audio_tensor):
            logger.debug("Silero VAD : Aucun signal de parole détecté dans le segment.")
            return ""

        # 3. Traitement avec le processeur Wav2Vec2
        inputs = self.processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        # 4. Inférence avec optimisation evaluation + no_grad
        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    async def transcribe(self, audio_tensor: torch.Tensor, orig_sr: int) -> str:
        """
        Transcrit de façon asynchrone l'audio en déchargeant l'inférence lourde
        vers un thread de travail (ThreadPoolExecutor via asyncio.to_thread).
        
        Args:
            audio_tensor: Tenseur audio d'entrée (peut être à n'importe quelle fréquence d'échantillonnage).
            orig_sr: Fréquence d'échantillonnage d'origine de l'audio.
            
        Returns:
            Transcription textuelle en Darija.
        """
        # Rééchantillonner en 16kHz mono (très rapide)
        resampled_audio = self.resample_audio(audio_tensor, orig_sr)
        
        # Lancer la transcription dans un thread séparé pour ne pas bloquer FastAPI
        return await asyncio.to_thread(self._transcribe_sync, resampled_audio)


def get_asr_engine() -> ASREngine:
    """Obtenir l'instance unique du moteur ASR"""
    return ASREngine()

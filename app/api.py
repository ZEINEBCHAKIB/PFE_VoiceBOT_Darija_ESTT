"""
FastAPI Server — Simulation Call Center Temps Réel.
Expose un WebSocket (/ws/call) pour recevoir le flux audio du microphone,
détecter la parole avec Silero VAD, et transcrire avec Wav2Vec2.
"""
# Ajouter ces lignes EN HAUT du fichier, avant tout import
from aiohttp import http_websocket
from dotenv import load_dotenv
load_dotenv()  # doit être appelé AVANT get_mcp_host()
from app.core.llm import get_llm_client
from app.core.tts import generate_tts
import io
import uuid
import logging
import time
from pathlib import Path
from app.database.connection import get_db
from app.database.models import CallLog
import asyncio
import numpy as np
import torch
import soundfile as sf
import av
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.core.asr import get_asr_engine
# AJOUTER après les imports existants
from app.mcp_server.tools.call_tracking_tool import get_call_logger
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simulation VoiceBot CTM Darija",
    description="Simulation call center temps réel avec Wav2Vec2 et Silero VAD"
)

# ────────────────────────────────────────────────────────
# Dossier des audios TTS générés, exposé en statique pour le navigateur
# ────────────────────────────────────────────────────────
try:
    from app.core.tts import OUTPUT_DIR as AUDIO_DIR   # source unique de vérité
except ImportError:
    AUDIO_DIR = Path(__file__).resolve().parent / "core" / "outputs"
AUDIO_DIR = Path(AUDIO_DIR)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Chargement du moteur ASR unique
asr_engine = get_asr_engine()



# ────────────────────────────────────────────────────────
# Décodage Audio Multi-format (WAV, PCM, WebM/Opus)
# ────────────────────────────────────────────────────────
def decode_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """
    Décode les octets audio (WAV, PCM ou WebM/Opus) en un tenseur PyTorch.
    Puisque le client de simulation envoie du PCM 16-bit brut (16kHz, mono),
    on décode directement en PCM 16-bit en premier pour éviter que soundfile/PyAV
    n'interprètent mal les octets de données comme un autre format (comme MP3/MPEG).

    Returns:
        tuple (tensor_audio, sample_rate)
    """
    # 1. Tenter d'abord la lecture directe en PCM 16-bit brut
    try:
        if len(audio_bytes) % 2 == 0:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return torch.tensor(data, dtype=torch.float32), 16000
    except Exception:
        pass

    # 2. Repli vers soundfile (WAV standard)
    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor, samplerate
    except Exception:
        # 3. Repli vers PyAV (WebM/Opus, etc.)
        try:
            container = av.open(io.BytesIO(audio_bytes))
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='fltp', layout='mono')

            frames = []
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                for f in resampled_frames:
                    frames.append(f.to_ndarray())

            if frames:
                audio_data = np.concatenate(frames, axis=1).squeeze()
                return torch.tensor(audio_data, dtype=torch.float32), stream.rate
            raise ValueError("Aucune trame audio n'a pu être décodée.")
        except Exception as e:
            raise ValueError(f"Impossible de décoder les données audio : {e}")


# ────────────────────────────────────────────────────────
# Interface Graphique (HTML Dashboard Premium)
# ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Retourne l'interface web interactive pour tester le microphone."""
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CTM VoiceBot — Poste Opérateur</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0B0E14;
    --panel: #141822;
    --panel-2: #191E2A;
    --border: rgba(255,255,255,0.07);
    --border-strong: rgba(255,255,255,0.12);
    --accent: #FF9800;
    --accent-dim: rgba(255,152,0,0.14);
    --success: #22C55E;
    --success-dim: rgba(34,197,94,0.14);
    --danger: #EF4444;
    --danger-dim: rgba(239,68,68,0.14);
    --text: #F8FAFC;
    --text-mid: #B6BECC;
    --text-dim: #6B7384;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Inter', sans-serif;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    background-image:
      radial-gradient(ellipse 800px 400px at 15% -10%, rgba(255,152,0,0.07), transparent),
      radial-gradient(ellipse 600px 400px at 100% 10%, rgba(34,197,94,0.04), transparent);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 28px 16px 60px;
  }

  /* ── Top bar ───────────────────────────────────────────── */
  .topbar {
    width: 100%;
    max-width: 980px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4px 22px;
  }

  .brand { display: flex; align-items: center; gap: 12px; }

  .brand-mark {
    width: 48px; height: 42px;
    border-radius: 10px;
    background: #0E1118;
    border: 1px solid var(--border-strong);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    flex-shrink: 0;
    overflow: hidden;
  }

  .brand-mark svg { width: 38px; height: 32px; }

  .brand-text h1 {
    font-size: 15px;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.2px;
  }

  .brand-text p {
    margin: 1px 0 0;
    font-size: 11.5px;
    color: var(--text-dim);
    font-family: var(--mono);
  }

  .conn-indicator {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    padding: 7px 14px;
    border: 1px solid var(--border);
    border-radius: 100px;
    background: var(--panel);
  }

  .conn-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--text-dim);
    transition: all .3s ease;
  }

  .conn-indicator.live .conn-dot {
    background: var(--success);
    box-shadow: 0 0 0 3px var(--success-dim);
  }

  .conn-indicator.live { color: var(--success); border-color: rgba(34,197,94,0.25); }

  /* ── Main grid ─────────────────────────────────────────── */
  .console {
    width: 100%;
    max-width: 980px;
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 16px;
  }

  @media (max-width: 760px) {
    .console { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
  }

  /* ── Left: call status panel ──────────────────────────── */
  .call-panel { display: flex; flex-direction: column; gap: 20px; }

  .call-state {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    padding: 6px 11px;
    border-radius: 7px;
    width: fit-content;
    background: var(--panel-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
  }

  .call-state .blip {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  .call-state.idle { color: var(--text-dim); }
  .call-state.listening {
    color: var(--accent);
    background: var(--accent-dim);
    border-color: rgba(255,152,0,0.25);
  }
  .call-state.listening .blip { animation: blip-pulse 1.1s infinite; }
  .call-state.processing {
    color: #60A5FA;
    background: rgba(96,165,250,0.12);
    border-color: rgba(96,165,250,0.25);
  }
  .call-state.processing .blip { animation: blip-pulse 0.6s infinite; }

  @keyframes blip-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.7); }
  }

  .call-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .call-meta .session-id {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  .call-timer {
    font-family: var(--mono);
    font-size: 34px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--text);
  }

  .call-timer span { color: var(--text-dim); font-size: 16px; }

  /* ── Visualizer ────────────────────────────────────────── */
  .visualizer {
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    padding: 0 4px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 12px;
  }

  .visualizer .bar {
    width: 4px;
    min-height: 4px;
    border-radius: 3px;
    background: var(--text-dim);
    transition: height .09s ease, background .2s ease;
  }

  .visualizer.active .bar { background: var(--accent); }

  /* ── Mic button ────────────────────────────────────────── */
  .mic-zone { display: flex; flex-direction: column; align-items: center; gap: 10px; padding-top: 4px; }

  .mic-btn {
    width: 64px; height: 64px;
    border-radius: 50%;
    border: none;
    background: var(--panel-2);
    border: 1.5px solid var(--border-strong);
    color: var(--text-mid);
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all .25s cubic-bezier(.2,.9,.3,1.2);
  }

  .mic-btn svg { width: 26px; height: 26px; fill: currentColor; }

  .mic-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }

  .mic-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  .mic-btn.recording {
    background: var(--accent);
    border-color: var(--accent);
    color: #1a1206;
    box-shadow: 0 0 0 8px var(--accent-dim);
  }

  .mic-hint {
    font-size: 11.5px;
    color: var(--text-dim);
    font-family: var(--mono);
  }

  /* ── Stats row ─────────────────────────────────────────── */
  .stat-row { display: flex; flex-direction: column; gap: 10px; margin-top: auto; padding-top: 16px; border-top: 1px solid var(--border); }

  .stat {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 12px;
  }

  .stat-label { color: var(--text-dim); font-family: var(--mono); }
  .stat-value { color: var(--text-mid); font-family: var(--mono); font-weight: 500; }

  /* ── Right: transcript panel ──────────────────────────── */
  .transcript-panel { display: flex; flex-direction: column; padding: 0; overflow: hidden; }

  .transcript-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 22px;
    border-bottom: 1px solid var(--border);
  }

  .transcript-head h2 {
    font-size: 13px;
    font-weight: 600;
    margin: 0;
    color: var(--text-mid);
  }

  .transcript-head .count {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  .transcript-body {
    flex: 1;
    min-height: 420px;
    max-height: 420px;
    overflow-y: auto;
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .transcript-body::-webkit-scrollbar { width: 6px; }
  .transcript-body::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }

  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--text-dim);
    font-size: 12.5px;
    font-family: var(--mono);
    max-width: 220px;
    line-height: 1.6;
  }

  .msg-row { display: flex; flex-direction: column; gap: 6px; animation: rise .25s ease-out; }
  .msg-row.user { align-items: flex-end; }
  .msg-row.bot { align-items: flex-start; }

  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .msg-tag {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 0 2px;
  }

  .msg-bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 13px;
    font-size: 14px;
    line-height: 1.55;
  }

  .msg-row.user .msg-bubble {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-right-radius: 4px;
  }

  .msg-row.bot .msg-bubble {
    background: var(--accent-dim);
    border: 1px solid rgba(255,152,0,0.22);
    color: #FFD699;
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
    border-bottom-left-radius: 4px;
  }

  .msg-audio {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    padding: 0 2px;
  }

  .msg-audio svg { width: 11px; height: 11px; fill: var(--success); }

  /* ── Transcript footer (input hint) ───────────────────── */
  .transcript-foot {
    padding: 12px 22px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .transcript-foot .key {
    background: var(--panel-2);
    border: 1px solid var(--border-strong);
    border-radius: 5px;
    padding: 2px 7px;
    color: var(--text-mid);
  }
</style>
</head>
<body>

  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 120 90" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Swoosh rouge -->
          <path d="M4 28C24 18 46 11 68 13C84 14.3 98 19 116 27"
                stroke="#E23B3B" stroke-width="3.4" stroke-linecap="round" fill="none"/>
          <!-- Swoosh bleu clair -->
          <path d="M2 34C26 23 50 16 72 18.5C90 20.5 104 26 118 35"
                stroke="#3DA5E0" stroke-width="9" stroke-linecap="round" fill="none" opacity="0.95"/>
          <!-- Monogramme CTM simplifié -->
          <text x="60" y="72" text-anchor="middle"
                font-family="Inter, sans-serif" font-weight="800" font-size="34"
                fill="#1E3A8A" letter-spacing="-1">CTM</text>
        </svg>
      </div>
      <div class="brand-text">
        <h1>Poste Opérateur — VoiceBot Darija</h1>
        <p>Simulation centre d'appel · ASR + Agents LLM + TTS</p>
      </div>
    </div>
    <div class="conn-indicator" id="connIndicator">
      <div class="conn-dot"></div>
      <span id="connLabel">Hors ligne</span>
    </div>
  </div>

  <div class="console">

    <!-- ── Left panel : call status ───────────────────────── -->
    <div class="panel call-panel">

      <div class="call-state idle" id="callState">
        <div class="blip"></div>
        <span id="callStateLabel">En attente</span>
      </div>

      <div class="call-meta">
        <div class="session-id" id="sessionId">SESSION — — — —</div>
        <div class="call-timer" id="callTimer">00<span>:</span>00</div>
      </div>

      <div class="visualizer" id="visualizer"></div>

      <div class="mic-zone">
        <button class="mic-btn" id="micBtn" disabled aria-label="Activer le micro">
          <svg viewBox="0 0 24 24"><path d="M12,14A3,3 0 0,0 15,11V5A3,3 0 0,0 12,2A3,3 0 0,0 9,5V11A3,3 0 0,0 12,14M17.3,11C17.3,14 14.76,16.1 12,16.1C9.24,16.1 6.7,14 6.7,11H5C5,14.41 7.72,17.23 11,17.72V21H13V17.72C16.28,17.23 19,14.41 19,11H17.3Z"/></svg>
        </button>
        <span class="mic-hint" id="micHint">Connexion en cours…</span>
      </div>

      <div class="stat-row">
        <div class="stat"><span class="stat-label">Échanges</span><span class="stat-value" id="statTurns">0</span></div>
        <div class="stat"><span class="stat-label">Dernier outil</span><span class="stat-value" id="statTool">—</span></div>
        <div class="stat"><span class="stat-label">Latence</span><span class="stat-value" id="statLatency">—</span></div>
      </div>

    </div>

    <!-- ── Right panel : transcript ────────────────────────── -->
    <div class="panel transcript-panel">
      <div class="transcript-head">
        <h2>Transcript en direct</h2>
        <span class="count" id="msgCount">0 message(s)</span>
      </div>

      <div class="transcript-body" id="chat">
        <div class="empty-state" id="emptyState">
          EN ATTENTE D'APPEL<br>Appuyez sur le micro pour démarrer une simulation
        </div>
      </div>

      <div class="transcript-foot">
        <span class="key">●</span> Parlez naturellement — la détection de silence lance la transcription automatiquement
      </div>
    </div>

  </div>

<script>
  // ── DOM refs ──────────────────────────────────────────────
  const connIndicator = document.getElementById('connIndicator');
  const connLabel = document.getElementById('connLabel');
  const callState = document.getElementById('callState');
  const callStateLabel = document.getElementById('callStateLabel');
  const callTimer = document.getElementById('callTimer');
  const sessionIdEl = document.getElementById('sessionId');
  const micBtn = document.getElementById('micBtn');
  const micHint = document.getElementById('micHint');
  const visualizer = document.getElementById('visualizer');
  const chat = document.getElementById('chat');
  const emptyState = document.getElementById('emptyState');
  const msgCount = document.getElementById('msgCount');
  const statTurns = document.getElementById('statTurns');
  const statTool = document.getElementById('statTool');
  const statLatency = document.getElementById('statLatency');

  let ws;
  let audioContext, processor, globalStream;
  let isRecording = false;
  let turns = 0;
  let callStartTs = null;
  let timerInterval = null;

  // ── Visualizer bars (build once) ────────────────────────
  const BAR_COUNT = 28;
  const bars = [];
  for (let i = 0; i < BAR_COUNT; i++) {
    const b = document.createElement('div');
    b.className = 'bar';
    visualizer.appendChild(b);
    bars.push(b);
  }
  function setVisualizer(level) {
    // level: 0..1
    visualizer.classList.toggle('active', level > 0.03);
    bars.forEach((b, i) => {
      const jitter = Math.sin(Date.now() / 90 + i) * 0.25 + 0.75;
      const h = Math.max(4, Math.min(54, level * 54 * jitter * (0.5 + Math.random() * 0.6)));
      b.style.height = h + 'px';
    });
  }
  function decayVisualizer() {
    bars.forEach(b => { b.style.height = '4px'; });
    visualizer.classList.remove('active');
  }

  // ── Call timer ───────────────────────────────────────────
  function startTimer() {
    callStartTs = Date.now();
    timerInterval = setInterval(() => {
      const s = Math.floor((Date.now() - callStartTs) / 1000);
      const mm = String(Math.floor(s / 60)).padStart(2, '0');
      const ss = String(s % 60).padStart(2, '0');
      callTimer.innerHTML = `${mm}<span>:</span>${ss}`;
    }, 500);
  }
  function stopTimer() {
    clearInterval(timerInterval);
    callTimer.innerHTML = '00<span>:</span>00';
  }

  // ── State helpers ────────────────────────────────────────
  function setCallState(mode, label) {
    callState.className = 'call-state ' + mode;
    callStateLabel.textContent = label;
  }

  function setConnected(on) {
    connIndicator.classList.toggle('live', on);
    connLabel.textContent = on ? 'En ligne' : 'Hors ligne';
  }

  // ── Chat rendering ───────────────────────────────────────
  function addMessage(sender, text, withAudio) {
    if (emptyState) emptyState.remove();
    const row = document.createElement('div');
    row.className = 'msg-row ' + sender;

    const tag = document.createElement('div');
    tag.className = 'msg-tag';
    tag.textContent = sender === 'user' ? 'Appelant' : 'Agent CTM';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;

    row.appendChild(tag);
    row.appendChild(bubble);

    if (withAudio) {
      const audioTag = document.createElement('div');
      audioTag.className = 'msg-audio';
      audioTag.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg> réponse vocale jouée';
      row.appendChild(audioTag);
    }

    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;

    if (sender === 'bot') {
      turns++;
      statTurns.textContent = turns;
    }
    msgCount.textContent = chat.querySelectorAll('.msg-row').length + ' message(s)';
  }

  // ── WebSocket ─────────────────────────────────────────────
  function connect() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws/call`);

    ws.onopen = () => {
      setConnected(true);
      micBtn.disabled = false;
      micHint.textContent = 'Appuyez pour parler';
      sessionIdEl.textContent = 'SESSION — ' + Math.random().toString(36).slice(2, 8).toUpperCase();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'welcome') {
        addMessage('bot', data.message, false);
      } else if (data.type === 'status') {
        if (data.message === 'Listening...') {
          setCallState('listening', 'Écoute en cours');
        } else if (data.message === 'Processing...') {
          setCallState('processing', 'Traitement…');
          decayVisualizer();
        }
      } else if (data.type === 'response') {
        setCallState('idle', 'En attente');
        statTool.textContent = data.tool || '—';
        addMessage('user', data.transcript, false);
        addMessage('bot', data.response, !!data.audio);

        if (data.audio) {
          try {
            const botAudio = new Audio(data.audio);
            botAudio.play().catch(err => console.warn('Lecture audio bloquée:', err));
          } catch (err) {
            console.warn('Audio TTS indisponible:', err);
          }
        }
      } else if (data.type === 'error') {
        setCallState('idle', 'En attente');
        addMessage('bot', data.message, false);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      micBtn.disabled = true;
      micHint.textContent = 'Reconnexion…';
      setCallState('idle', 'Déconnecté');
      setTimeout(connect, 3000);
    };
  }

  // ── Mic capture (logic unchanged from original) ─────────
  async function startRecording() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(globalStream);

    processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);

      // Niveau RMS pour le visualiseur
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
      const rms = Math.sqrt(sum / inputData.length);
      setVisualizer(Math.min(1, rms * 6));

      const buffer = new ArrayBuffer(inputData.length * 2);
      const view = new DataView(buffer);
      for (let i = 0; i < inputData.length; i++) {
        const sample = Math.max(-1, Math.min(1, inputData[i]));
        view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      }

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(buffer);
      }
    };

    isRecording = true;
    micBtn.classList.add('recording');
    micHint.textContent = 'Micro actif — parlez';
    setCallState('listening', 'Ligne ouverte');
    startTimer();
  }

  function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micHint.textContent = 'Appuyez pour parler';

    if (processor) processor.disconnect();
    if (audioContext) audioContext.close();
    if (globalStream) globalStream.getTracks().forEach(t => t.stop());

    decayVisualizer();
    setCallState('idle', 'En attente');
    stopTimer();
  }

  micBtn.addEventListener('click', () => {
    if (!isRecording) startRecording();
    else stopRecording();
  });

  connect();
</script>
</body>
</html>
"""


# ────────────────────────────────────────────────────────
# WebSocket Endpoint pour la Simulation Temps Réel
# ────────────────────────────────────────────────────────
@app.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Connexion WebSocket acceptée pour le call center.")

    state = "IDLE"
    audio_buffer = bytearray()
    pre_speech_chunks = []
    silence_start_time = None

    SILENCE_THRESHOLD_SECONDS = 1.5
    NOISE_GATE_THRESHOLD = 0.05

    # Mémoire conversationnelle — une instance par session WebSocket
    from app.core.memory import get_memory
    memory = get_memory(max_turns=10, session_id=str(id(websocket)))
    call_logger = get_call_logger()
    session_id = str(id(websocket))

    # Message de bienvenue dynamique
    llm = get_llm_client()
    welcome_msg = llm.generate_welcome()
    await websocket.send_json({
        "type": "welcome",
        "message": welcome_msg
    })

    try:
        while True:
            data = await websocket.receive_bytes()

            if state == "PROCESSING":
                continue

            try:
                chunk_tensor, sr = decode_audio_bytes(data)
                resampled_chunk = asr_engine.resample_audio(chunk_tensor, sr)

                max_val = torch.max(torch.abs(resampled_chunk))
                if max_val > NOISE_GATE_THRESHOLD:
                    resampled_chunk = resampled_chunk / max_val
                    has_speech = asr_engine.is_speech(resampled_chunk, sampling_rate=16000)
                else:
                    has_speech = False

                if state == "IDLE":
                    if has_speech:
                        logger.info("🎙️ Détection de parole - Début de l'enregistrement")
                        state = "LISTENING"
                        await websocket.send_json({"type": "status", "message": "Listening..."})
                        audio_buffer = bytearray()
                        for p_data in pre_speech_chunks:
                            audio_buffer.extend(p_data)
                        audio_buffer.extend(data)
                        silence_start_time = None
                    else:
                        pre_speech_chunks.append(data)
                        if len(pre_speech_chunks) > 2:
                            pre_speech_chunks.pop(0)

                elif state == "LISTENING":
                    audio_buffer.extend(data)
                    if has_speech:
                        silence_start_time = None
                    else:
                        if silence_start_time is None:
                            silence_start_time = asyncio.get_event_loop().time()
                        elif asyncio.get_event_loop().time() - silence_start_time >= SILENCE_THRESHOLD_SECONDS:
                            logger.info("🤫 Silence détecté - Lancement de la transcription")
                            state = "PROCESSING"
                            await websocket.send_json({"type": "status", "message": "Processing..."})

                            # ──────────────────────────────────────────────
                            # Keep-alive GLOBAL : couvre TOUT le traitement
                            # (ASR + MCP + TTS), pas seulement l'appel MCP.
                            # Évite que le navigateur/proxy ne ferme le
                            # WebSocket par timeout pendant les étapes
                            # longues : transcription CPU (peut prendre
                            # plusieurs secondes) et appel TTS distant
                            # sur Colab (latence réseau + inférence GPU).
                            # ──────────────────────────────────────────────
                            async def _keepalive():
                                while True:
                                    await asyncio.sleep(2)
                                    try:
                                        await websocket.send_json({"type": "status", "message": "Processing..."})
                                    except Exception:
                                        break

                            keepalive_task = asyncio.create_task(_keepalive())
                            start_ts = time.time()

                            try:
                                full_tensor, full_sr = decode_audio_bytes(bytes(audio_buffer))
                                transcript = await asr_engine.transcribe(full_tensor, full_sr)

                                if transcript:
                                    # ── Filtre transcription corrompue ──
                                    word_count = len(transcript.split())
                                    if word_count > 30:
                                        logger.warning(f"⚠️ Transcription suspecte ({word_count} mots) — ignorée : {transcript[:80]}...")
                                        keepalive_task.cancel()
                                        await websocket.send_json({
                                            "type": "error",
                                            "message": "سمحلي ما فهمتكش مزيان، تقدر تعاود بجملة قصيرة ؟"
                                        })
                                    else:
                                        logger.info(f"Transcription finale : {transcript}")

                                        from app.mcp_host import get_mcp_host
                                        mcp = get_mcp_host()

                                        # ── Appel MCP protégé ──
                                        try:
                                            rag_result = await mcp.orchestrate(transcript, memory)

                                            if rag_result.get("success"):
                                                response_text = rag_result["data"].get("answer", "معذرة ما قدرتش نلقى جواب.")
                                                tool_utilise = rag_result.get("tool", "unknown")
                                                succes = True
                                            else:
                                                logger.error(f"Erreur MCP : {rag_result.get('error')}")
                                                response_text = "معذرة، صعيب نجاوبك دابا."
                                                tool_utilise = "error"
                                                succes = False

                                        except Exception as mcp_err:
                                            logger.error(f"❌ Erreur MCP critique: {mcp_err}")
                                            response_text = "معذرة، وقع مشكل تقني، عاود من فضلك"
                                            tool_utilise = "error"
                                            succes = False

                                        duration_ms = int((time.time() - start_ts) * 1000)

                                        # ── Logger automatiquement — silencieux, non bloquant ──
                                        try:
                                            call_logger.log(
                                                session_id=session_id,
                                                transcript=transcript,
                                                tool_utilise=tool_utilise,
                                                reponse=response_text,
                                                duree_ms=duration_ms,
                                                succes=succes
                                            )
                                            logger.info(f"📝 Call logged | tool={tool_utilise} | {duration_ms}ms | succes={succes}")
                                        except Exception as log_err:
                                            logger.warning(f"⚠️ Call logger échoué (non bloquant): {log_err}")

                                        # ── Sauvegarder dans la mémoire ──
                                        memory.add_turn(transcript, response_text)
                                        logger.info(f"🧠 {memory}")

                                        # ── TTS Habibi distant via Colab (non bloquant) ──
                                        audio_url = None
                                        try:
                                            # Nom unique : évite le cache navigateur et l'écrasement
                                            out_name = f"response_{uuid.uuid4().hex[:12]}.wav"
                                            # generate_tts est synchrone → on le sort de la boucle asyncio
                                            audio_path = await asyncio.to_thread(
                                                generate_tts, response_text, out_name=out_name
                                            )
                                            # On expose seulement l'URL web, pas le chemin disque
                                            audio_url = f"/audio/{Path(audio_path).name}"
                                            logger.info(f"🔊 Audio TTS généré : {audio_url}")
                                        except Exception as tts_err:
                                            logger.warning(f"⚠️ TTS échoué, réponse texte conservée : {tts_err}")

                                        # ── Arrêt du keep-alive juste avant l'envoi final ──
                                        keepalive_task.cancel()
                                        await websocket.send_json({
                                            "type": "response",
                                            "transcript": transcript,
                                            "response": response_text,
                                            "audio": audio_url
                                        })

                                else:
                                    logger.info("Transcription vide")
                                    keepalive_task.cancel()
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": "سمحلي ما سمعتكش مزيان، تقدر تعاود؟"
                                    })

                            finally:
                                # Filet de sécurité : si une exception saute toutes
                                # les annulations explicites ci-dessus, on s'assure
                                # quand même que la tâche de keep-alive ne tourne
                                # pas indéfiniment en arrière-plan.
                                if not keepalive_task.done():
                                    keepalive_task.cancel()

                            # ── Reset systématique après chaque traitement ──
                            audio_buffer.clear()
                            pre_speech_chunks.clear()
                            state = "IDLE"
                            silence_start_time = None

            except Exception as e:
                logger.error(f"Erreur traitement chunk audio : {e}")
                # Reset en cas d'erreur pour éviter de rester bloqué
                audio_buffer.clear()
                pre_speech_chunks.clear()
                state = "IDLE"
                silence_start_time = None

    except WebSocketDisconnect:
        logger.info(f"WebSocket déconnecté — {memory}")
        memory.clear()
    except Exception as e:
        logger.error(f"Erreur WebSocket fatale : {e}")
        memory.clear()
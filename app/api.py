"""
FastAPI Server — Simulation Call Center Temps Réel.
Expose un WebSocket (/ws/call) pour recevoir le flux audio du microphone,
détecter la parole avec Silero VAD, et transcrire avec Wav2Vec2.
"""
# Ajouter ces lignes EN HAUT du fichier, avant tout import
from aiohttp import http_websocket
from dotenv import load_dotenv
load_dotenv()  # doit être appelé AVANT get_mcp_server()
from app.core.llm import get_llm_client
import io
import logging
import time
from app.database.connection import get_db
from app.database.models import CallLog
import asyncio
import numpy as np
import torch
import soundfile as sf
import av
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
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
    <title>CTM VoiceBot — Simulation Call Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #FF9800;
            --primary-dark: #E65100;
            --bg-gradient: linear-gradient(135deg, #100e17, #241c30);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --text: #ffffff;
            --text-muted: #b0a8ba;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: var(--text);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            width: 90%;
            max-width: 600px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border-radius: 28px;
            padding: 35px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
            text-align: center;
        }

        h1 {
            font-weight: 700;
            margin-bottom: 5px;
            font-size: 2.2rem;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #ff9800, #ffc107);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 30px;
            transition: all 0.3s ease;
            letter-spacing: 0.5px;
        }

        .status-badge.connected {
            color: #4CAF50;
            background: rgba(76, 175, 80, 0.15);
            border: 1px solid rgba(76, 175, 80, 0.25);
        }

        .status-badge.listening {
            color: #FF9800;
            background: rgba(255, 152, 0, 0.15);
            border: 1px solid rgba(255, 152, 0, 0.25);
            animation: pulse-shadow 1.5s infinite;
        }

        .status-badge.processing {
            color: #2196F3;
            background: rgba(33, 150, 243, 0.15);
            border: 1px solid rgba(33, 150, 243, 0.25);
        }

        .mic-container {
            position: relative;
            width: 150px;
            height: 150px;
            margin: 0 auto 35px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .mic-button {
            position: relative;
            z-index: 10;
            width: 100px;
            height: 100px;
            background: var(--primary);
            border: none;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            outline: none;
            box-shadow: 0 10px 30px rgba(255, 152, 0, 0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .mic-button:hover {
            transform: scale(1.08);
            background: var(--primary-dark);
            box-shadow: 0 15px 35px rgba(255, 152, 0, 0.6);
        }

        .mic-button:active {
            transform: scale(0.95);
        }

        .mic-button svg {
            fill: #ffffff;
            width: 46px;
            height: 46px;
            transition: transform 0.3s ease;
        }

        .mic-button.recording svg {
            transform: scale(0.9);
        }

        .pulse-ring {
            position: absolute;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: rgba(255, 152, 0, 0.25);
            animation: pulse 1.8s infinite cubic-bezier(0.215, 0.61, 0.355, 1);
            opacity: 0;
            pointer-events: none;
        }

        .chat-container {
            text-align: left;
            margin-top: 30px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: 20px;
            padding: 20px;
            height: 250px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .chat-message {
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
            animation: slide-in 0.3s ease-out;
        }

        .chat-message.user {
            align-items: flex-end;
        }

        .chat-message.bot {
            align-items: flex-start;
        }

        .chat-bubble {
            max-width: 80%;
            padding: 14px 20px;
            border-radius: 20px;
            font-size: 0.95rem;
            line-height: 1.45;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        .chat-message.user .chat-bubble {
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }

        .chat-message.bot .chat-bubble {
            background: var(--primary);
            color: #ffffff;
            border-bottom-left-radius: 4px;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }

        .chat-label {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.4);
            margin-bottom: 5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        @keyframes pulse {
            0% {
                transform: scale(0.85);
                opacity: 0.8;
            }
            100% {
                transform: scale(1.4);
                opacity: 0;
            }
        }

        @keyframes pulse-shadow {
            0% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(255, 152, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0); }
        }

        @keyframes slide-in {
            from { transform: translateY(10px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simulation Call Center CTM</h1>
        <div class="subtitle">Wav2Vec2 + VAD Silero + FastAPI WebSockets (Temps Réel)</div>

        <div class="status-badge" id="statusBadge">Déconnecté</div>

        <div class="mic-container">
            <div class="pulse-ring" id="pulseRing" style="animation-play-state: paused;"></div>
            <button class="mic-button" id="micBtn" disabled>
                <svg viewBox="0 0 24 24">
                    <path d="M12,14A3,3 0 0,0 15,11V5A3,3 0 0,0 12,2A3,3 0 0,0 9,5V11A3,3 0 0,0 12,14M17.3,11C17.3,14 14.76,16.1 12,16.1C9.24,16.1 6.7,14 6.7,11H5C5,14.41 7.72,17.23 11,17.72V21H13V17.72C16.28,17.23 19,14.41 19,11H17.3Z" />
                </svg>
            </button>
        </div>

        <div class="chat-container" id="chat">
            <!-- Messages injectés en temps réel -->
        </div>
    </div>

    <script>
        const statusBadge = document.getElementById('statusBadge');
        const pulseRing = document.getElementById('pulseRing');
        const micBtn = document.getElementById('micBtn');
        const chat = document.getElementById('chat');

        let ws;
        let audioContext;
        let processor;
        let globalStream;
        let isRecording = false;

        function connect() {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${window.location.host}/ws/call`);

            ws.onopen = () => {
                statusBadge.textContent = 'CONNECTÉ (PRÊT)';
                statusBadge.className = 'status-badge connected';
                micBtn.disabled = false;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'welcome') {
                    addMessage('bot', data.message);
                } else if (data.type === 'status') {
                    statusBadge.textContent = data.message.toUpperCase();
                    if (data.message === 'Listening...') {
                        statusBadge.className = 'status-badge listening';
                    } else if (data.message === 'Processing...') {
                        statusBadge.className = 'status-badge processing';
                    }
                } else if (data.type === 'response') {
                    statusBadge.textContent = 'CONNECTÉ (PRÊT)';
                    statusBadge.className = 'status-badge connected';
                    
                    addMessage('user', data.transcript);
                    addMessage('bot', data.response);
                } else if (data.type === 'error') {
                    statusBadge.textContent = 'CONNECTÉ (PRÊT)';
                    statusBadge.className = 'status-badge connected';
                    addMessage('bot', data.message);
                }
            };

            ws.onclose = () => {
                statusBadge.textContent = 'DÉCONNECTÉ (RECONNEXION...)';
                statusBadge.className = 'status-badge';
                micBtn.disabled = true;
                setTimeout(connect, 3000);
            };
        }

        function addMessage(sender, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `chat-message ${sender}`;
            
            const label = document.createElement('div');
            label.className = 'chat-label';
            label.textContent = sender === 'user' ? 'Vous' : 'Agent CTM';
            
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble';
            bubble.textContent = text;
            
            msgDiv.appendChild(label);
            msgDiv.appendChild(bubble);
            chat.appendChild(msgDiv);
            chat.scrollTop = chat.scrollHeight;
        }

        async function startRecording() {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = audioContext.createMediaStreamSource(globalStream);
            
            // ScriptProcessor pour extraire 4096 échantillons par cycle
            processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Conversion Float32 -> PCM 16-bit Int
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
            pulseRing.style.animationPlayState = 'running';
            statusBadge.textContent = 'ÉCOUTE...';
            statusBadge.className = 'status-badge listening';
        }

        function stopRecording() {
            isRecording = false;
            micBtn.classList.remove('recording');
            pulseRing.style.animationPlayState = 'paused';
            
            if (processor) processor.disconnect();
            if (audioContext) audioContext.close();
            if (globalStream) {
                globalStream.getTracks().forEach(track => track.stop());
            }
            
            statusBadge.textContent = 'CONNECTÉ (PRÊT)';
            statusBadge.className = 'status-badge connected';
        }

        micBtn.addEventListener('click', () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
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

                            full_tensor, full_sr = decode_audio_bytes(bytes(audio_buffer))
                            transcript = await asr_engine.transcribe(full_tensor, full_sr)

                            if transcript:
                                # ── Filtre transcription corrompue ──
                                word_count = len(transcript.split())
                                if word_count > 30:
                                    logger.warning(f"⚠️ Transcription suspecte ({word_count} mots) — ignorée : {transcript[:80]}...")
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": "سمحلي ما فهمتكش مزيان، تقدر تعاود بجملة قصيرة ؟"
                                    })
                                else:
                                    logger.info(f"Transcription finale : {transcript}")

                                    from app.mcp_server.server import get_mcp_server
                                    mcp = get_mcp_server()

                                    # ── Appel MCP protégé ──
                                    try:
                                        start_ts = time.time()
                                        rag_result = mcp.call_tool(transcript, memory)
                                        duration_ms = int((time.time() - start_ts) * 1000)

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
                                        duration_ms = 0

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

                                    await websocket.send_json({
                                        "type": "response",
                                        "transcript": transcript,
                                        "response": response_text
                                    })

                            else:
                                logger.info("Transcription vide")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "سمحلي ما سمعتكش مزيان، تقدر تعاود؟"
                                })

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
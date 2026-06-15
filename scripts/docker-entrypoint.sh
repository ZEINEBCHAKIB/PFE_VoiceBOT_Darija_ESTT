#!/bin/bash
set -e

echo "========================================"
echo "   VoiceBot CTM — Démarrage"
echo "========================================"

# Vérifier GPU
echo "🎮 GPU détecté :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  ⚠️ nvidia-smi non disponible"

# ── Étape 1 : Init base de données ──
echo ""
echo "📦 Initialisation base de données..."
python -c "
from app.database.connection import init_db
init_db()
print('  ✅ Tables créées')
"

# ── Étape 2 : Seed données si vide ──
echo ""
echo "🌱 Vérification données synthétiques..."
python -c "
from app.database.connection import get_db
from app.database.models import Ligne
with get_db() as db:
    count = db.query(Ligne).count()
    if count == 0:
        print('  → Base vide, seeding...')
        import subprocess
        subprocess.run(['python', '-m', 'app.database.seed_data'], check=True)
        print('  ✅ Données insérées')
    else:
        print(f'  ✅ {count} lignes déjà présentes')
"

# ── Étape 3 : Vérifier Qdrant ──
echo ""
echo "🔍 Vérification index Qdrant..."
python -c "
import os
path = 'data/qdrant'
if not os.path.exists(path) or not os.listdir(path):
    print('  ⚠️  Index Qdrant vide')
    print('  → Lance: docker exec voicebot-ctm-gpu python -m scripts.indexer')
else:
    print('  ✅ Index Qdrant présent')
"

# ── Étape 4 : Télécharger Wav2Vec2 si absent ──
echo ""
echo "🤖 Vérification modèle Wav2Vec2..."
python -c "
import os
local = './models/wav2vec2-darija'
if not os.path.exists(local) or not os.listdir(local):
    print('  📥 Téléchargement depuis HuggingFace (~1.2 GB)...')
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    p = Wav2Vec2Processor.from_pretrained('boumehdi/wav2vec2-large-xlsr-moroccan-darija')
    m = Wav2Vec2ForCTC.from_pretrained('boumehdi/wav2vec2-large-xlsr-moroccan-darija')
    p.save_pretrained(local)
    m.save_pretrained(local)
    print('  ✅ Modèle sauvegardé')
else:
    print('  ✅ Modèle déjà présent')
"

# ── Étape 5 : Lancer Uvicorn ──
echo ""
echo "========================================"
echo "✅ Prêt — Lancement sur http://0.0.0.0:8000"
echo "========================================"

exec uvicorn app.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
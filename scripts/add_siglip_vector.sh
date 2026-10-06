#!/usr/bin/env bash
# Adds the siglip vector to an index that currently holds only jina, then
# re-runs the dev configs so the second visual vector becomes its own ablation.
#
# Waits for the asr/ocr work to finish first: re-embedding 54,802 keyframes is a
# GPU job and Whisper is already using the card.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
LOG="$DATA_ROOT/siglip.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
py()  { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a

log "=== waiting for the GPU: asr/ocr and any eval must finish ==="
exec 9>"$DATA_ROOT/everything.lock"; flock 9; flock -u 9
exec 8>"$DATA_ROOT/pipeline.lock"; flock 8; flock -u 8
log "GPU free"

log "re-embedding 54,802 keyframes with siglip + jina"
t0=$SECONDS
py -m offline.run --stages embed --force >> "$LOG" 2>&1
log "embed rc=$? in $(( (SECONDS-t0)/60 ))min"

log "upserting both named vectors into Qdrant"
t0=$SECONDS
py -m offline.run --stages index --force >> "$LOG" 2>&1
log "index rc=$? in $(( (SECONDS-t0)/60 ))min"

# Confirm the collection really carries two vectors now.
py - <<'PY' >> "$LOG" 2>&1
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
name = os.getenv("QDRANT_COLLECTION", "videomme_keyframes")
info = c.get_collection(name)
pts = c.scroll(collection_name=name, limit=1, with_vectors=True)[0]
have = sorted((pts[0].vector or {}).keys()) if pts else []
print(f"points={info.points_count} vectors_on_a_point={have}")
PY

log "re-running the dev configs with both vectors"
rm -f "$REPO"/results/eval/{visual-only,+asr-gt,+ocr,+asr-whisper,full}-dev.json
rm -f "$DATA_ROOT/pipeline.lock"
setsid nohup "$REPO/scripts/run_full_pipeline.sh" >/dev/null 2>&1 < /dev/null &
disown
log "=== handed back to run_full_pipeline.sh ==="

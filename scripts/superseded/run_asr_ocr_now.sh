#!/usr/bin/env bash
# Runs the Whisper and OCR branches while the Qwen weights are still downloading.
#
# Neither needs the VLM, and they use resources the download is not touching:
# Whisper wants the idle GPU, OCR is ~428 batched Gemini requests. Doing them now
# means run_full_pipeline.sh can skip straight to the evaluations — its phase B
# and C both check for completed work before repeating it.
#
# Sequenced rather than parallel: two concurrent `offline.run --stages index`
# would upsert the same Qdrant points twice. Harmless, since point ids are
# deterministic, but pointless work on a busy card.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
LOG="$DATA_ROOT/asr_ocr.log"
LOCK="$DATA_ROOT/asr_ocr.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another asr/ocr run holds the lock; exiting" >&2
  exit 0
fi

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
py()  { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a

log "=== start ==="

# --- OCR first: minutes, and it frees the API branch early ---
done_ocr=$(sqlite3 "$DATA_ROOT/manifest.db" \
  "SELECT COUNT(*) FROM stage_state WHERE stage='ocr' AND status='done';" 2>/dev/null || echo 0)
if [ "$done_ocr" -lt 200 ]; then
  log "ocr: $done_ocr/200 done, running"
  t0=$SECONDS
  py -m offline.run --stages ocr >> "$LOG" 2>&1
  log "ocr finished rc=$? in $(( (SECONDS-t0)/60 ))min"
else
  log "ocr already complete"
fi

log "indexing OCR documents (subtitle ASR source)"
py -m offline.run --stages index --force >> "$LOG" 2>&1
log "index rc=$?"

# --- Whisper: ~40min at the 40x realtime measured on video 001 ---
if [ ! -f "$DATA_ROOT/asr_whisper/200.json" ]; then
  log "whisper: transcribing 200 videos"
  t0=$SECONDS
  ASR_SOURCE=whisper py -m offline.run --stages asr --force >> "$LOG" 2>&1
  log "whisper finished rc=$? in $(( (SECONDS-t0)/60 ))min"

  log "indexing whisper transcripts into their own index"
  ASR_SOURCE=whisper py -m offline.run --stages index --force >> "$LOG" 2>&1
  log "whisper index rc=$?"
else
  log "whisper transcripts already present"
fi

# --- a WER-style comparison against the ground-truth subtitles, free to compute ---
log "comparing whisper against the ground-truth subtitles"
py - <<'PY' >> "$LOG" 2>&1
import json, re
from pathlib import Path
from data.videomme import load_subtitle_segments

root = Path("/media/nekoartisan/Lexar/vqa-data")
ann = root / "annotations" / "subtitle.zip"
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())

rows, gt_words, wh_words = [], 0, 0
for p in sorted((root / "asr_whisper").glob("*.json")):
    vid = p.stem
    wh = json.loads(p.read_text())["segments"]
    gt = load_subtitle_segments(ann, vid)
    g = len(norm(" ".join(s.text for s in gt)).split())
    w = len(norm(" ".join(s["text"] for s in wh)).split())
    gt_words += g
    wh_words += w
    rows.append((vid, len(gt), len(wh), g, w))

print(f"videos compared: {len(rows)}")
print(f"GT words {gt_words}, Whisper words {wh_words}, ratio {wh_words/max(1,gt_words):.3f}")
print("A ratio far from 1.0 means Whisper is dropping or hallucinating speech.")
PY

log "=== done ==="

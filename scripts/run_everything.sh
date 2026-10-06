#!/usr/bin/env bash
# Drives the remaining work to completion in dependency order.
#
#   1  push ASR+OCR text to Elasticsearch (fast path, ES only)
#   2  Whisper over 200 videos, then push its transcripts to their own index
#   3  hand off to run_full_pipeline.sh for the evaluations
#
# Elasticsearch had to be restarted first: a disk-watermark reroute task wedged
# the cluster-state queue for 9.6 hours, so every put-mapping behind it timed out
# and 30,251 OCR documents landed as zero. Watermarks are now 95/97/98 because the
# indices total a few megabytes while /mnt/data sits at 91% from unrelated data.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
LOG="$DATA_ROOT/everything.log"
LOCK="$DATA_ROOT/everything.lock"

exec 9>"$LOCK"
flock -n 9 || { echo "another run holds the lock" >&2; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
py()  { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a

log "=== step 1: push subtitle ASR + OCR to Elasticsearch ==="
py -m offline.run --stages index_text --force >> "$LOG" 2>&1
log "step 1 rc=$?"
for i in asr_data ocr_data; do
  log "  $i: $(curl -s "http://localhost:9200/$i/_count" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("count",0))' 2>/dev/null)"
done

log "=== step 2: Whisper over 200 videos (~40min at 40x realtime) ==="
if [ ! -f "$DATA_ROOT/asr_whisper/200.json" ]; then
  ASR_SOURCE=whisper py -m offline.run --stages asr --force >> "$LOG" 2>&1
  log "whisper rc=$?"
  ASR_SOURCE=whisper py -m offline.run --stages index_text --force >> "$LOG" 2>&1
  log "whisper index rc=$?; asr_data_whisper: $(curl -s 'http://localhost:9200/asr_data_whisper/_count' | python3 -c 'import json,sys;print(json.load(sys.stdin).get("count",0))' 2>/dev/null)"
else
  log "whisper transcripts already present"
fi

log "=== step 3: evaluations ==="
flock -u 9
rm -f "$DATA_ROOT/pipeline.lock"
setsid nohup "$REPO/scripts/run_full_pipeline.sh" >/dev/null 2>&1 < /dev/null &
disown
log "=== handed off to run_full_pipeline.sh ==="

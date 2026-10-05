#!/usr/bin/env bash
# Waits for a running `offline.run` to exit, then completes embed + index.
#
# Chained rather than left to the watcher because the watcher invokes
# `--stages all`, which would put a second process on the same GPU alongside the
# shots/keyframes run already in flight.
#
#   nohup ./scripts/finish_indexing.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
LOG="$DATA_ROOT/index.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] finisher: $*" >> "$LOG"; }

log "waiting for the in-flight offline.run to finish"
while pgrep -f 'm offline[.]run' > /dev/null; do
  sleep 30
done
log "upstream run finished; starting embed + index"

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a

conda run --no-capture-output -n "$CONDA_ENV" \
  python -m offline.run --stages embed,index >> "$LOG" 2>&1
rc=$?

indexed=$(sqlite3 "$DATA_ROOT/manifest.db" \
  "SELECT COUNT(DISTINCT video_id) FROM stage_state WHERE stage='index' AND status='done';" \
  2>/dev/null || echo 0)
log "embed+index finished rc=$rc, $indexed video(s) indexed"

# The eval needs the VLM. Check for the actual weight files and their total
# size: an earlier version tested only that the directory existed, which passed
# on a 12MB config-and-tokenizer-only download and burned three hours in failed
# fetch attempts before reporting "VLM unavailable".
QWEN_DIR="${QWEN_VL_MODEL:-/mnt/data/vqa-models/qwen2.5-vl-3b}"
qwen_bytes=0
if [ -d "$QWEN_DIR" ]; then
  qwen_bytes=$(find "$QWEN_DIR" -name '*.safetensors' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')
fi
# fp16 3B is ~7.3GB across two shards; require 7GB before trusting it.
if [ "$qwen_bytes" -gt 7000000000 ]; then
  log "Qwen weights present ($((qwen_bytes/1024/1024))MB); running dev-split eval"
  for cfg in baseline-uniform visual-only; do
    conda run --no-capture-output -n "$CONDA_ENV" \
      python -m eval.run_eval --config "$cfg" --split dev >> "$DATA_ROOT/eval.log" 2>&1
    log "eval $cfg done"
  done
else
  log "Qwen weights incomplete ($((qwen_bytes/1024/1024))MB of ~7300MB); skipping eval"
fi
log "=== finisher done ==="

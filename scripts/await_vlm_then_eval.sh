#!/usr/bin/env bash
# Polls until the Qwen weights are complete, then runs the dev-split evals.
#
# Separate from finish_indexing.sh because that script runs once and exits,
# whereas the weights arrive over a CDN that is currently serving ~2KB/s and
# will need many hours. Everything upstream is already indexed, so this is the
# only thing standing between the project and its first numbers.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
QWEN_DIR="${QWEN_VL_MODEL:-/mnt/data/vqa-models/qwen2.5-vl-3b}"
FETCHER="$REPO/scripts/fetch_model_curl.sh"
LOG="$DATA_ROOT/eval.log"
NEEDED=7000000000      # fp16 3B is ~7.3GB across two shards
POLL=300

log() { echo "[$(date '+%m-%d %H:%M:%S')] await-vlm: $*" >> "$LOG"; }

weight_bytes() {
  find "$QWEN_DIR" -name '*.safetensors' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}

log "=== waiting for Qwen weights in $QWEN_DIR ==="
last=-1
while :; do
  have=$(weight_bytes)
  [ "$have" -ge "$NEEDED" ] && break

  # Respawn the fetcher if it died; curl resumes from wherever it stopped.
  if ! pgrep -f 'fetch_model_curl' > /dev/null; then
    log "fetcher not running at $((have/1024/1024))MB - respawning"
    setsid nohup "$FETCHER" Qwen/Qwen2.5-VL-3B-Instruct "$QWEN_DIR" \
      >/dev/null 2>&1 < /dev/null &
    disown
  elif [ "$have" != "$last" ]; then
    log "$((have/1024/1024))MB of $((NEEDED/1024/1024))MB"
  fi
  last=$have
  sleep "$POLL"
done

log "weights complete ($(( $(weight_bytes) /1024/1024))MB); starting eval"

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a

for cfg in baseline-uniform visual-only +asr-gt; do
  log "eval $cfg (dev)"
  conda run --no-capture-output -n "$CONDA_ENV" \
    python -m eval.run_eval --config "$cfg" --split dev >> "$LOG" 2>&1
  log "eval $cfg exited rc=$?"
done

log "=== results in $REPO/results/eval ==="
ls -la "$REPO/results/eval" >> "$LOG" 2>&1

#!/usr/bin/env bash
# Keeps the Video-MME-v2 download alive and indexes videos as they land.
#
# Two jobs, both needed because the link is unreliable (~1.75MB/s, observed full
# network drops and curl giving up after exhausting retries):
#   1. Respawn the downloader whenever it dies with zips still missing.
#   2. Index each new batch of videos as soon as it arrives, instead of waiting
#      for all 10 zips — the GPU is otherwise idle for hours.
#
# Safe to run more than once: it holds a lock and exits if another copy is live.
#
#   nohup ./scripts/watch_download_and_index.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
FETCHER="$REPO/scripts/fetch_videomme2.sh"
TARGET_ZIPS="${TARGET_ZIPS:-10}"
CONDA_ENV="${CONDA_ENV:-jina_env}"

LOG="$DATA_ROOT/watcher.log"
LOCK="$DATA_ROOT/watcher.lock"
POLL_SECONDS=60

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another watcher is already running" >&2
  exit 0
fi

log "=== watcher up (target ${TARGET_ZIPS} zips) ==="

zips_done()   { ls "$DATA_ROOT/_done" 2>/dev/null | wc -l; }
videos_have() { find "$DATA_ROOT/videos" -name '*.mp4' 2>/dev/null | wc -l; }
fetcher_live() { pgrep -f 'fetch_videomme2\.sh' > /dev/null; }

# Count videos already carried through the final `index` stage, so a restart of
# this watcher does not redo work. Absent manifest == nothing indexed yet.
videos_indexed() {
  local db="$DATA_ROOT/manifest.db"
  [ -f "$db" ] || { echo 0; return; }
  sqlite3 "$db" \
    "SELECT COUNT(DISTINCT video_id) FROM stage_state WHERE stage='index' AND status='done';" \
    2>/dev/null || echo 0
}

last_indexed=-1

while :; do
  done_n=$(zips_done)
  have_n=$(videos_have)

  # --- job 1: keep the download going ---
  if [ "$done_n" -lt "$TARGET_ZIPS" ] && ! fetcher_live; then
    log "downloader not running with $done_n/$TARGET_ZIPS zips done - respawning"
    if [ -x "$FETCHER" ]; then
      setsid nohup "$FETCHER" "$TARGET_ZIPS" >> "$DATA_ROOT/download.stdout" 2>&1 < /dev/null &
      disown
    else
      log "FETCHER missing at $FETCHER - cannot respawn"
    fi
  fi

  # --- job 2: index whatever has arrived ---
  if [ -f "$REPO/offline/run.py" ]; then
    indexed_n=$(videos_indexed)
    if [ "$have_n" -gt "$indexed_n" ]; then
      log "indexing: $have_n videos on disk, $indexed_n indexed - running offline pipeline"
      (
        cd "$REPO" || exit 1
        set -a; [ -f .env ] && . ./.env; set +a
        # Must be the conda env: av, torch and transformers<5 are only there.
        # Calling plain `python` picked up the system interpreter and failed
        # every video with ModuleNotFoundError: No module named 'av'.
        conda run --no-capture-output -n "$CONDA_ENV" \
          python -m offline.run --stages all >> "$DATA_ROOT/index.log" 2>&1
      )
      rc=$?
      indexed_n=$(videos_indexed)
      log "indexing finished rc=$rc, now $indexed_n indexed"
    elif [ "$indexed_n" != "$last_indexed" ]; then
      log "nothing new to index ($indexed_n indexed)"
    fi
    last_indexed=$indexed_n
  else
    log "waiting for offline/run.py (phase 2) - $have_n videos ready"
  fi

  # --- exit when the job is actually complete ---
  if [ "$done_n" -ge "$TARGET_ZIPS" ] && [ -f "$REPO/offline/run.py" ]; then
    if [ "$(videos_indexed)" -ge "$(videos_have)" ] && [ "$(videos_have)" -gt 0 ]; then
      log "=== all $done_n zips downloaded and $(videos_have) videos indexed - watcher done ==="
      exit 0
    fi
  fi

  sleep "$POLL_SECONDS"
done

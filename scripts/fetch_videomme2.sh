#!/usr/bin/env bash
# Download Video-MME-v2 videos one zip at a time: fetch -> extract -> delete zip.
# Keeps peak disk ~20GB instead of ~40GB. Resumable at byte level: re-run to continue.
#
# Uses curl, not `hf download`: on this ~2MB/s link hf's HTTP client times out and then
# raises "Cannot send a request, as the client has been closed" for every later file,
# so 9 of 10 zips failed without ever transferring a byte. curl resumes with -C -.
set -uo pipefail

ROOT=/media/nekoartisan/Lexar/vqa-data
REPO_URL="https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/main/videos"

VIDEO_DIR="$ROOT/videos"
ZIP_DIR="$ROOT/_zips"
MARK_DIR="$ROOT/_done"
LOG="$ROOT/download.log"
MIN_FREE_GB=15

# How many zips to fetch (1..40). Each zip is ~20 videos.
MAX_ZIPS="${1:-40}"

mkdir -p "$VIDEO_DIR" "$ZIP_DIR" "$MARK_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
free_gb() { df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9'; }

log "=== start (curl): zips 001..$(printf '%03d' "$MAX_ZIPS") -> $VIDEO_DIR ==="

for i in $(seq -w 1 "$MAX_ZIPS"); do
  id=$(printf '%03d' "$((10#$i))")

  if [ -f "$MARK_DIR/$id" ]; then
    log "$id done already, skip"
    continue
  fi

  avail=$(free_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}GB free, need >${MIN_FREE_GB}GB"
    exit 1
  fi

  zip="$ZIP_DIR/$id.zip"
  log "$id fetching (${avail}GB free)"

  # -C -            resume from whatever is already on disk
  # --retry 999     keep retrying; the link drops often
  # --speed-limit   treat <10KB/s for 120s as a stall and trigger a retry
  # no auth header: the repo is public, and a stale header breaks the signed CDN redirect
  if curl -sSL --fail \
        -C - \
        --retry 999 --retry-delay 10 --retry-all-errors \
        --connect-timeout 30 \
        --speed-limit 10240 --speed-time 120 \
        -o "$zip" \
        "$REPO_URL/$id.zip" >>"$LOG" 2>&1; then
    :
  else
    log "$id curl gave up, leaving partial file for next run"
    continue
  fi

  # A truncated zip must not be treated as success
  if ! unzip -tqq "$zip" >>"$LOG" 2>&1; then
    log "$id zip corrupt/incomplete, keeping for resume"
    continue
  fi

  log "$id extracting"
  if unzip -o -q -j "$zip" -d "$VIDEO_DIR" >>"$LOG" 2>&1; then
    rm -f "$zip"
    touch "$MARK_DIR/$id"
    n=$(find "$VIDEO_DIR" -name '*.mp4' | wc -l)
    log "$id OK - $n mp4 total, $(free_gb)GB free"
  else
    log "$id EXTRACT FAILED, keeping zip"
  fi
done

n=$(find "$VIDEO_DIR" -name '*.mp4' | wc -l)
done_n=$(ls "$MARK_DIR" | wc -l)
log "=== finished: $done_n/$MAX_ZIPS zips, $n mp4 files, $(free_gb)GB free ==="

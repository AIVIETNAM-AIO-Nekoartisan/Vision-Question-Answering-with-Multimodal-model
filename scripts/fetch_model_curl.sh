#!/usr/bin/env bash
# Fetch a HF model repo into a plain directory with curl.
#
# `hf download` cannot complete here: it dies with "I/O error: error decoding
# response body" and, once a transfer times out, its client raises "Cannot send
# a request, as the client has been closed" for every file after it. curl with
# --retry 999 -C - is the combination that already pulled 20GB of dataset zips
# over this same link.
#
# The result is loaded with from_pretrained("<dir>"), which needs no HF cache and
# no network at all.
#
#   ./scripts/fetch_model_curl.sh Qwen/Qwen2.5-VL-3B-Instruct /mnt/data/vqa-models/qwen2.5-vl-3b
set -uo pipefail

REPO="${1:?usage: fetch_model_curl.sh <hf-repo-id> <dest-dir>}"
DEST="${2:?usage: fetch_model_curl.sh <hf-repo-id> <dest-dir>}"
TOKEN="$(cat /media/nekoartisan/Lexar/vqa-data/hf_home/token 2>/dev/null)"
MIN_FREE_GB=3

mkdir -p "$DEST"
LOG="$DEST/fetch.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== $REPO -> $DEST ==="

files=$(curl -sS --retry 10 --retry-delay 5 --retry-all-errors \
  "https://huggingface.co/api/models/$REPO/tree/main?recursive=1" \
  | python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f.get('type') == 'file':
        print(f['path'])
" 2>/dev/null)

if [ -z "$files" ]; then
  log "could not list $REPO"
  exit 1
fi

for path in $files; do
  case "$path" in
    .gitattributes|README.md|LICENSE|*.md) continue ;;
  esac

  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"

  avail=$(df --output=avail -BG "$DEST" | tail -1 | tr -dc '0-9')
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${avail}GB free"
    exit 1
  fi

  # Remote size, so a complete file can be skipped on re-run.
  remote=$(curl -sSIL --retry 5 --retry-all-errors \
    ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
    "https://huggingface.co/$REPO/resolve/main/$path" 2>/dev/null \
    | grep -iE '^(content-length|x-linked-size):' | tail -1 \
    | tr -dc '0-9')
  local_size=$([ -f "$out" ] && stat -c%s "$out" || echo 0)

  if [ -n "$remote" ] && [ "$local_size" = "$remote" ]; then
    log "$path already complete ($((remote/1024/1024))MB)"
    continue
  fi

  log "$path fetching (have $((local_size/1024/1024))MB of $((${remote:-0}/1024/1024))MB)"
  curl -sSL --fail -C - \
    --retry 999 --retry-delay 10 --retry-all-errors \
    --connect-timeout 30 --speed-limit 10240 --speed-time 120 \
    ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
    -o "$out" \
    "https://huggingface.co/$REPO/resolve/main/$path" >> "$LOG" 2>&1

  got=$([ -f "$out" ] && stat -c%s "$out" || echo 0)
  if [ -n "$remote" ] && [ "$got" != "$remote" ]; then
    log "$path INCOMPLETE ($got of $remote) - re-run to resume"
  else
    log "$path OK ($((got/1024/1024))MB)"
  fi
done

log "=== done: $(du -sh "$DEST" | cut -f1) ==="

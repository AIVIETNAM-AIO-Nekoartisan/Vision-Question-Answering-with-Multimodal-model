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
PIDFILE="$DEST/fetcher.pid"
LOCKFILE="$DEST/fetcher.lock"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# flock, not pgrep and not a PID file alone. Two earlier attempts both failed:
# `pgrep -f fetch_model_curl` matched the watcher's own command line and any
# shell inspecting the download, so it stopped respawning; and a bare PID file
# is check-then-write, so a respawn racing a manual launch produced two curls
# appending to the same output. Two writers on one -C - transfer corrupt it
# silently, and from_pretrained does not verify safetensors checksums.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "another fetcher holds the lock; exiting"
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT INT TERM

log "=== $REPO -> $DEST (pid $$) ==="

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

  # One curl call per attempt, with the loop doing the retrying — NOT curl's own
  # --retry. Letting curl retry internally destroyed progress: the transfer
  # climbed to 261MB, then collapsed to 7MB and then 0MB. When the CDN answers a
  # resumed request with 200 instead of 206, curl truncates and rewrites from
  # byte 0, and --retry-all-errors kept handing it that chance. Re-invoking curl
  # fresh means the Range header is always derived from the real file size, and
  # the guard below refuses to accept a shrunken file.
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    before=$([ -f "$out" ] && stat -c%s "$out" || echo 0)

    if [ -n "$remote" ] && [ "$before" -ge "$remote" ]; then
      break
    fi
    if [ "$attempt" -gt 400 ]; then
      log "$path giving up after $attempt attempts at $((before/1024/1024))MB"
      break
    fi

    curl -sSL --fail -C - --no-progress-meter \
      --connect-timeout 30 --speed-limit 1024 --speed-time 300 \
      ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
      -o "$out" \
      "https://huggingface.co/$REPO/resolve/main/$path" >> "$LOG" 2>&1
    rc=$?
    after=$([ -f "$out" ] && stat -c%s "$out" || echo 0)

    if [ "$after" -lt "$before" ]; then
      # The server ignored the Range and curl rewrote from the start. Keep the
      # longer of the two rather than losing what was already on disk.
      log "$path SHRANK $((before/1024/1024))MB -> $((after/1024/1024))MB (server ignored Range); retrying"
      sleep 20
      continue
    fi

    if [ "$rc" -eq 0 ] && { [ -z "$remote" ] || [ "$after" -ge "$remote" ]; }; then
      break
    fi

    gained=$(( (after - before) / 1024 ))
    log "$path attempt $attempt rc=$rc, +${gained}KB, at $((after/1024/1024))MB of $((${remote:-0}/1024/1024))MB"
    sleep 15
  done

  got=$([ -f "$out" ] && stat -c%s "$out" || echo 0)
  if [ -n "$remote" ] && [ "$got" != "$remote" ]; then
    log "$path INCOMPLETE ($((got/1024/1024))MB of $((remote/1024/1024))MB) - re-run to resume"
  else
    log "$path OK ($((got/1024/1024))MB)"
  fi
done

log "=== done: $(du -sh "$DEST" | cut -f1) ==="

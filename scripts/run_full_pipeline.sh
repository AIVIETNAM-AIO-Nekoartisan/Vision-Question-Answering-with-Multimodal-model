#!/usr/bin/env bash
# Runs the whole evaluation programme once the Qwen weights land.
#
# Phases are ordered so the cheap, informative results arrive first and the
# expensive branches cannot invalidate them:
#
#   A  dev: baseline-uniform, visual-only, +asr-gt       (free, answers the
#                                                         research question)
#   B  OCR over all 200 videos, then dev: +ocr           (~428 Gemini requests)
#   C  Whisper over all 200 videos, then dev: +asr-whisper (~5h GPU)
#   D  dev: full                                          (adds DeepSeek expand)
#   E  test split for every config, but only if dev passes a sanity gate
#
# Resumable: a config whose results JSON already exists is skipped, so an
# interrupted run picks up where it stopped.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
QWEN_DIR="${QWEN_VL_MODEL:-/mnt/data/vqa-models/qwen2.5-vl-3b}"
LOG="$DATA_ROOT/pipeline.log"
LOCK="$DATA_ROOT/pipeline.lock"
RESULTS="$REPO/results/eval"
NEEDED=7000000000

# flock, not pgrep: a pattern like 'run_full_pipeline' matches the command line
# of any shell inspecting the run, which is how an earlier watcher convinced
# itself a dead fetcher was alive.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another pipeline run holds the lock; exiting" >&2
  exit 0
fi

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
py()  { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p "$RESULTS"

weight_bytes() {
  find "$QWEN_DIR" -name '*.safetensors' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}

# --------------------------------------------------------------------------- #
log "=== full pipeline start ==="

log "waiting for Qwen weights"
while [ "$(weight_bytes)" -lt "$NEEDED" ]; do
  sleep 120
done
log "weights complete ($(( $(weight_bytes) /1024/1024))MB)"

# --- smoke test: one question, before committing hours to a broken setup ---
log "smoke test: answering question 001-1"
if ! py -m eval.run_eval --config visual-only --split dev --limit 1 \
      --out "$DATA_ROOT/smoke" >> "$LOG" 2>&1; then
  log "SMOKE TEST FAILED - stopping before the expensive phases"
  exit 1
fi
log "smoke test passed"

run_eval() {   # run_eval <config> <split>
  local cfg="$1" split="$2"
  local out="$RESULTS/$cfg-$split.json"
  if [ -f "$out" ]; then
    log "skip $cfg/$split (already have $(basename "$out"))"
    return 0
  fi
  log "eval $cfg/$split starting"
  local t0=$SECONDS
  if py -m eval.run_eval --config "$cfg" --split "$split" >> "$LOG" 2>&1; then
    log "eval $cfg/$split done in $(( (SECONDS-t0)/60 ))min"
  else
    log "eval $cfg/$split FAILED (rc=$?)"
  fi
}

# --- Phase A: the configs that need nothing new ---
log "--- phase A: dev, subtitle-based ---"
for cfg in baseline-uniform visual-only +asr-gt; do
  run_eval "$cfg" dev
done

# --- Phase B: OCR. gemini-2.0-flash batches 128 images per request across 30
# keys, so 54,802 keyframes is ~428 requests, not 54,802. ---
log "--- phase B: Gemini OCR over 200 videos ---"
if [ "$(sqlite3 "$DATA_ROOT/manifest.db" \
        "SELECT COUNT(*) FROM stage_state WHERE stage='ocr' AND status='done';" \
        2>/dev/null || echo 0)" -lt 200 ]; then
  py -m offline.run --stages ocr >> "$LOG" 2>&1
  log "ocr stage rc=$?; reindexing to push the OCR documents"
  py -m offline.run --stages index --force >> "$LOG" 2>&1
else
  log "ocr already done for 200 videos"
fi
run_eval "+ocr" dev

# --- Phase C: Whisper. Writes to asr_whisper/ and asr_data_whisper so the
# subtitle corpus survives intact and the comparison stays honest. ---
log "--- phase C: Whisper over 200 videos (~5h GPU) ---"
if [ ! -f "$DATA_ROOT/asr_whisper/200.json" ]; then
  ASR_SOURCE=whisper py -m offline.run --stages asr --force >> "$LOG" 2>&1
  log "whisper asr rc=$?; indexing into the whisper index"
  ASR_SOURCE=whisper py -m offline.run --stages index --force >> "$LOG" 2>&1
else
  log "whisper transcripts already present"
fi
run_eval "+asr-whisper" dev

# --- Phase D ---
log "--- phase D: dev, full ---"
run_eval full dev

# --- Phase E: test split, gated on dev being sane ---
log "--- phase E: gate check on dev ---"
gate=$(py - <<'PY' 2>/dev/null
import json, pathlib, sys
p = pathlib.Path("results/eval/visual-only-dev.json")
if not p.is_file():
    print("no-dev-result"); sys.exit()
s = json.loads(p.read_text())["summary"]
if s["unparsed_rate"] > 0.05:
    print(f"unparsed-{s['unparsed_rate']:.3f}")
elif s["accuracy"] < 0.20:
    print(f"accuracy-{s['accuracy']:.3f}")
else:
    print("pass")
PY
)
log "gate: $gate"

if [ "$gate" = "pass" ]; then
  log "--- phase E: test split, all configs ---"
  for cfg in baseline-uniform visual-only +asr-gt +ocr +asr-whisper full; do
    run_eval "$cfg" test
  done
else
  log "test split SKIPPED (gate=$gate). Dev numbers are not trustworthy yet:"
  log "  unparsed above 5% means the prompt needs fixing before accuracy means anything;"
  log "  accuracy under 20% is near the 12.8% random baseline for 8 options."
fi

# --- summary table ---
log "--- writing summary ---"
py - <<'PY' >> "$LOG" 2>&1
import json, pathlib
rows = []
for p in sorted(pathlib.Path("results/eval").glob("*.json")):
    if p.parent.name != "eval":
        continue
    try:
        s = json.loads(p.read_text())["summary"]
    except Exception:
        continue
    rows.append((p.stem, s))

out = ["# Results\n",
       "| config | split | n | acc | group | unparsed | L1 | L2 | L3 |",
       "|---|---|---|---|---|---|---|---|---|"]
for name, s in rows:
    cfg, _, split = name.rpartition("-")
    lv = s.get("by_level", {})
    out.append(
        f"| {cfg} | {split} | {s['n']} | {s['accuracy']:.3f} | "
        f"{s['group_score']:.3f} | {s['unparsed_rate']:.3f} | "
        f"{lv.get('1', float('nan')):.3f} | {lv.get('2', float('nan')):.3f} | "
        f"{lv.get('3', float('nan')):.3f} |"
    )
out.append("\nRandom baseline is 12.8% (8 options).")
text = "\n".join(out)
pathlib.Path("results/RESULTS.md").write_text(text + "\n")
print(text)
PY

log "=== full pipeline done; see $REPO/results/RESULTS.md ==="

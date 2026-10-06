#!/usr/bin/env bash
# One process, one step at a time. Replaces the chain of scripts that handed off
# to each other via flock.
#
# That chain had a race: run_everything.sh released its lock and only then
# launched the pipeline with `setsid nohup &`. add_siglip_vector.sh acquired both
# locks inside that window, so an embed and an eval ran on the card together.
# Qwen's 7.5GB plus the embedders' 4GB does not fit in 11.63GB, and both died —
# the embed never wrote a siglip vector, and five dev configs came back with
# unparsed=1.000 from CUDA OOM after their good results had been deleted.
#
# Sequential execution removes the possibility rather than guarding against it.
# Resumable: a config whose JSON exists is skipped.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/media/nekoartisan/Lexar/vqa-data
CONDA_ENV="${CONDA_ENV:-jina_env}"
RESULTS="$REPO/results/eval"
LOG="$DATA_ROOT/sequential.log"

exec 9>"$DATA_ROOT/sequential.lock"
flock -n 9 || { echo "already running" >&2; exit 0; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
py()  { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }

cd "$REPO" || exit 1
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p "$RESULTS"
# Fragmentation was part of why the three-model load sat 0.23GB from the edge.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "=== sequential run start ==="

# --------------------------------------------------------------------------- #
# Step 1: add the siglip vector. Embedders need the GPU here and nothing else
# may touch it, which is the whole point of running in one process.
# --------------------------------------------------------------------------- #
has_siglip=$(py - <<'PY' 2>/dev/null
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
pts = c.scroll(collection_name=os.getenv("QDRANT_COLLECTION", "videomme_keyframes"),
               limit=1, with_vectors=True)[0]
print("yes" if pts and "siglip" in (pts[0].vector or {}) else "no")
PY
)
log "siglip vector already in Qdrant: $has_siglip"

# SKIP_SIGLIP=1 keeps the second visual vector off the critical path. Adding it
# means re-embedding 54,802 keyframes and re-upserting every point, about 2.5
# hours before any evaluation can start, and it only widens the visual branch —
# it is not what answers the research question. The jina-only numbers are already
# a complete result; the ensemble is a follow-up.
if [ "${SKIP_SIGLIP:-0}" = "1" ]; then
  log "SKIP_SIGLIP=1: going straight to the evaluations with jina only"
elif [ "$has_siglip" != "yes" ]; then
  log "step 1a: embedding 54,802 keyframes with siglip + jina (GPU)"
  t0=$SECONDS
  unset EMBED_DEVICE
  py -m offline.run --stages embed --force >> "$LOG" 2>&1
  log "embed rc=$? in $(( (SECONDS-t0)/60 ))min"

  log "step 1b: upserting both vectors into Qdrant"
  t0=$SECONDS
  py -m offline.run --stages index --force >> "$LOG" 2>&1
  log "index rc=$? in $(( (SECONDS-t0)/60 ))min"

  py - <<'PY' >> "$LOG" 2>&1
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
name = os.getenv("QDRANT_COLLECTION", "videomme_keyframes")
pts = c.scroll(collection_name=name, limit=1, with_vectors=True)[0]
print(f"points={c.get_collection(name).points_count} "
      f"vectors={sorted((pts[0].vector or {}).keys()) if pts else []}")
PY
fi

# --------------------------------------------------------------------------- #
# Step 2: evaluations. Embedders go to CPU: they only encode one query string per
# question, while keeping them on the card left 0.23GB of headroom after an
# 18-image prefill. On CPU the same load peaks at 8.16GB of 12.49GB.
# --------------------------------------------------------------------------- #
export EMBED_DEVICE=cpu

run_eval() {   # run_eval <config> <split>
  local cfg="$1" split="$2" out="$RESULTS/$1-$2.json"
  if [ -f "$out" ]; then
    log "skip $cfg/$split (have results)"
    return 0
  fi
  log "eval $cfg/$split starting"
  local t0=$SECONDS
  py -m eval.run_eval --config "$cfg" --split "$split" >> "$LOG" 2>&1
  local rc=$?
  log "eval $cfg/$split rc=$rc in $(( (SECONDS-t0)/60 ))min"

  # An OOM run records every question as unparsed. Keeping that file would make
  # the next pass skip it and treat 0.000 as a measurement.
  if [ -f "$out" ]; then
    local bad
    bad=$(py - "$out" <<'PY' 2>/dev/null
import json, sys
s = json.loads(open(sys.argv[1]).read())["summary"]
print("bad" if s["unparsed_rate"] > 0.5 else "ok")
PY
)
    if [ "$bad" = "bad" ]; then
      log "  DISCARDING $cfg/$split: unparsed above 50%, not a measurement"
      mv "$out" "$out.rejected"
    fi
  fi
}

# Five configs, not six: +asr-whisper is dropped from the sweep. It stays in
# eval/configs.py and its Elasticsearch index (asr_data_whisper, 17,408 docs)
# is built, so it can be run later with one command. Dropping it also drops the
# ASR-error measurement — +asr-gt uses ground-truth subtitles, which do not
# exist in a deployed system, so the gap between the two was the only number
# saying what real transcription costs. It measured 2.0 points.
log "--- dev split, five configs ---"
for cfg in baseline-uniform visual-only +asr-gt +ocr full; do
  run_eval "$cfg" dev
done

# --------------------------------------------------------------------------- #
# Step 3: test split, gated on dev being trustworthy.
# --------------------------------------------------------------------------- #
gate=$(py - <<'PY' 2>/dev/null
import json, pathlib, sys
p = pathlib.Path("results/eval/visual-only-dev.json")
if not p.is_file():
    print("no-dev-result"); sys.exit()
s = json.loads(p.read_text())["summary"]
print("pass" if s["unparsed_rate"] <= 0.05 and s["accuracy"] >= 0.20
      else f"fail(unparsed={s['unparsed_rate']:.3f},acc={s['accuracy']:.3f})")
PY
)
log "--- gate: $gate ---"
if [ "$gate" = "pass" ]; then
  for cfg in baseline-uniform visual-only +asr-gt +ocr full; do
    run_eval "$cfg" test
  done
else
  log "test split skipped; dev is not trustworthy yet"
fi

# --------------------------------------------------------------------------- #
log "--- summary ---"
py - <<'PY' >> "$LOG" 2>&1
import json, pathlib
rows = []
for p in sorted(pathlib.Path("results/eval").glob("*.json")):
    try:
        rows.append((p.stem, json.loads(p.read_text())["summary"]))
    except Exception:
        pass
out = ["# Results", "",
       "| config | split | n | acc | group | unparsed | L1 | L2 | L3 |",
       "|---|---|---|---|---|---|---|---|---|"]
for name, s in rows:
    cfg, _, split = name.rpartition("-")
    lv = s.get("by_level", {})
    g = lambda k: f"{lv[k]:.3f}" if k in lv else "-"
    out.append(f"| {cfg} | {split} | {s['n']} | {s['accuracy']:.3f} | "
               f"{s['group_score']:.3f} | {s['unparsed_rate']:.3f} | "
               f"{g('1')} | {g('2')} | {g('3')} |")
out += ["", "Random baseline is 12.8% (8 options).",
        "Accuracy alone is not a result: use the paired McNemar test, since the",
        "configs answer the same questions."]
text = "\n".join(out)
pathlib.Path("results/RESULTS.md").write_text(text + "\n")
print(text)
PY

log "=== done; see $REPO/results/RESULTS.md ==="

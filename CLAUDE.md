# CLAUDE.md

Retrieval-augmented VQA on Video-MME-v2. Built on the AIC 2025 codebase at
`/mnt/data/AIC_2025-main` (read-only: it sits on branch `asr-full-b1` with 48
uncommitted files — copy out, never write in).

Design: `docs/superpowers/specs/2026-10-04-vqa-videomme-design.md`
Plan: `docs/superpowers/plans/2026-10-04-vqa-videomme.md`

## Hard constraints

Violating any of these breaks the build in ways that are slow to diagnose.

- **`transformers>=4.55,<5`.** jina-clip-v2 ships remote code that breaks on 5.x;
  SigLIP2 and Qwen2.5-VL both need ≥4.49. Use the **`jina_env`** conda env
  (transformers 4.57.6). `base` and `aio2026_env2` have transformers 5.x and
  will fail.
- **Run everything from the repo root** with `python -m`. Running a module by
  path breaks `core.*` imports.
- **Load `.env` first**: `set -a; . ./.env; set +a`.
- **Qdrant payloads stay nested** — `video.name`, `frame.timestamp_seconds`,
  `shot_id`, `file_path`. `fusion.py` keys on `file_path`, the shot rollup keys
  on `shot_id`, and `parsing.py` reads `video.name`. Flattening breaks all three.
- **Elasticsearch documents put their text in `content`**, not `text`.
  `ElasticsearchService.search` queries `content`, `content.ngram` and
  `content.exact`; a `text` field indexes fine and matches nothing.
- **Tests are stdlib `unittest`**, never pytest, no linter:
  `python3 -m unittest discover -s tests -v`
- **12GB VRAM.** Online holds SigLIP2 (~1.6GB) + jina-clip-v2 (~1.8GB) +
  Qwen2.5-VL-3B (~7GB) ≈ 10.4GB. Never load Whisper in the same process as
  Qwen. On OOM set `QWEN_VL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
- **SigLIP2 text needs lowercase + padding to exactly 64 tokens.**
  `core/models/SigLIP_embedding.py` already does this; do not "simplify" it.

## Filesystem layout, and why it is split

`/media/nekoartisan/Lexar` is **exFAT**: no POSIX ownership, no symlinks, no
Lucene-compatible file locking.

| What | Where | Why |
|---|---|---|
| Videos, keyframes, artifacts | `/media/nekoartisan/Lexar/vqa-data` (exFAT) | Plain file reads; exFAT is fine |
| Elasticsearch + Qdrant data | `/mnt/data/vqa-stores` (ext4) | ES writes `node.lock`; exFAT cannot |
| `HF_HOME` | `/mnt/data/vqa-hf` (ext4) | The HF blob cache uses symlinks |
| Repo | `/` (ext4, ~17GB free) | Keep large files off it |

## Environment facts that cost time to rediscover

- **pypi.org and the npm registry are unreachable** from this network. Four
  packages were replaced rather than installed, and `requirements.txt` records
  each substitution:
  `opencv-python` → ffmpeg CLI + PyAV (the videos are HEVC, which cv2 often
  cannot decode anyway); `librosa` → faster-whisper's own PyAV decoding;
  `imagehash` → `offline/phash.py`; `qwen-vl-utils` → `AutoProcessor` takes PIL
  images directly. `express` is reused from AIC's `node_modules`.
- **The Docker registry CDN is unreachable.** `docker-compose.yml` pins
  Elasticsearch `8.17.0` because that tag is already in the local image cache.
- **`hf download` is unusable for bulk fetches.** After one timeout its client
  raises `Cannot send a request, as the client has been closed` for every
  subsequent file — it failed 9 of 10 dataset zips without transferring a byte.
  `scripts/fetch_videomme2.sh` uses `curl --retry 999 -C -` instead. For single
  model repos `hf download` works, but **`HF_HUB_DISABLE_XET=1` is required**:
  hf_xet hangs indefinitely here.
- Link speed is ~1.75 MB/s and drops intermittently. Parallel connections do
  not help; that is the whole pipe.

## Commands

```bash
docker compose up -d                      # Qdrant :6333, Elasticsearch :9200
set -a; . ./.env; set +a

conda run -n jina_env python -m offline.run --stages all
conda run -n jina_env python -m offline.run --stages shots,keyframes --limit 2
conda run -n jina_env python -m offline.run --stages ocr --dev-only

conda run -n jina_env python -m online.backend.api_server       # :8000
cd online/frontend && npm start                                  # :3000

conda run -n jina_env python -m eval.run_eval --config visual-only --split dev
conda run -n jina_env python -m unittest discover -s tests -v
```

Watch an overnight run: `scripts/watch_download_and_index.sh` respawns the
fetcher if it dies and indexes each batch as it lands.

## Dev/test discipline

The split is by **video**, not by question — four questions share a video, so
splitting on questions leaks.

| Split | Videos | Questions | Use |
|---|---|---|---|
| dev | `001`–`050` | 200 | Tune RRF weights, `top_k`, thresholds; measure OCR's value |
| test | `051`–`200` | 600 | One run per config, for reporting |

`eval/run_eval.py --split` defaults to `dev` deliberately. Repeatedly touching
test to pick parameters invalidates every number in the report.

## What was dropped from AIC, and do not re-add

BEiT-3, BLIP2, OpenCLIP, YOLOE, GPT4o, `core/events/*`, `guided*`, `agent*`,
`search_{temporal,composed,events,intelligent}`, `submission_checker`, DRES, and
the caption/BGE-M3 collection. `enrich_results_metadata` went with the caption
collection it hydrated.

`routes.py`, `resources.py` and `schemas.py` are **written fresh**, not trimmed:
AIC's routes.py holds 39 references to `events`, 36 to `guided`, 18 to BEiT-3
and 14 to the agent layer.

`parsing.py`'s `_VIDEO_NAME_RE` was widened: it matched only AIC names
(`L01_V001`), so every Video-MME-v2 result (`001`–`200`) was flagged as needing
metadata repair.

## Research question

Not "how high can accuracy go" but:

> Does retrieval-augmented VQA beat uniform frame sampling, and on which kinds
> of question?

So `baseline-uniform` is mandatory, and results are always reported per Level.
Level 1 (Retrieval & Aggregation) should favour retrieval; Level 3 (complex
reasoning over the whole video) is expected to lose to the baseline. That is a
valid finding, not a bug — provided the per-Level breakdown is shown.

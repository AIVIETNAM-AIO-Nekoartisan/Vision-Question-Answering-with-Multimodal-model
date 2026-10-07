# Retrieval-Augmented VQA on Video-MME-v2

Answers multiple-choice questions about long videos by retrieving evidence -
keyframes, speech, on-screen text - and giving it to a local vision-language
model. Everything runs on one 12GB GPU.



## Result

On the held-out test split of 600 questions, retrieval is **directionally
positive but not statistically significant**:

| Config | Accuracy | Non-Lin | L1 | L2 | L3 | Δ vs baseline ||
|---|---|---|---|---|---|---|-
| `baseline-uniform` | 0.208 | 0.094 | 0.314 | 0.124 | 0.196 | — |
| `visual-only` | 0.215 | 0.099 | 0.314 | 0.124 | 0.211 | +0.007 |
| `+asr-gt` | 0.218 | 0.100 | 0.314 | 0.143 | 0.207 | +0.010 |
| `+ocr` | **0.227** | **0.101** | 0.314 | 0.143 | 0.225 | +0.018 |
| `full` | 0.223 | 0.099 | 0.296 | **0.174** | 0.211 | +0.015 |

Random guessing scores 0.128 (most questions have eight options).
`unparsed_rate` is 0.000 for all ten runs, so every figure is a real measurement
rather than a parsing artefact. `Non-Lin` is the benchmark's official group
metric: `(N/4)²` for consistency groups and the longest leading run of correct
answers for coherence groups.

The dev split told a much more optimistic story — `+ocr` reached 0.325 with
p=0.013 — which the test split did not reproduce. The gap between those two
numbers is the main experimental finding, and the reason the test split exists.

Two things the dev split got wrong, corrected by test:

- **Level 1 is not retrieval's strength.** Dev showed 0.483 → 0.552, but dev has
  only 29 Level 1 questions, so that was a two-question difference. On test, all
  four retrieval configs score exactly 50/159 — identical to the baseline.
- **Level 2 is where retrieval helps.** Test goes 0.124 → 0.143 → 0.174, the most
  consistent signal in the table, and `full` (with query expansion) is best there.

Full numbers: [`results/RESULTS.md`](results/RESULTS.md).
Write-up: [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

## Pipeline

**Offline** — each stage runs across all videos so models load once, and every
stage writes artifacts to disk before anything reaches a store. Changing fusion
weights or an index mapping then costs a rerun of `index_text`, not of any model.

```
200 × .mp4 (1080p HEVC)
  ├─ TransNetV2 ─────────────► shots/{vid}.json            200 videos
  ├─ PyAV + perceptual hash ─► keyframes/{vid}/*.jpg       54,802 frames
  ├─ SigLIP2 (1152-d) ┐
  │  jina-clip-v2 (1024-d) ──► Qdrant                      54,802 points, 2 vectors
  ├─ Whisper large-v3 (VAD) ─► Elasticsearch asr_data      12,846 / 17,408 docs
  └─ Gemini 2.5 Flash OCR ───► Elasticsearch ocr_data      30,259 docs
```

**Online**

```
browser :3000 ──► FastAPI :8000
  (Express proxy)      │
       query ──► DeepSeek expansion (optional)
                       │
        ┌──────────────┼──────────────┬──────────────┐
   Qdrant siglip   Qdrant jina    ES asr_data    ES ocr_data
        └──────────────┴──────────────┴──────────────┘
                       │  collapse to one hit per shot per source
                       │  weighted RRF, then aggregate by shot
                       │  expand to temporal windows, ordered by time
                       ▼
             Qwen2.5-VL-3B ──► letter A–H
   prompt: 8 uniform frames + condensed transcript
         + windows of before/during/after frames with [mm:ss] ASR and OCR
```

## Quickstart

Needs a conda env with `transformers>=4.55,<5` — jina-clip-v2's remote code
breaks on 5.x while SigLIP2 and Qwen2.5-VL both need ≥4.49.

```bash
docker compose up -d                  # Qdrant :6333, Elasticsearch :9200
cp .env.example .env                  # then fill in the API keys and paths
set -a; . ./.env; set +a

python -m offline.run --stages all    # index everything
python -m online.backend.api_server   # :8000
cd online/frontend && npm start       # :3000
```

Run every command from the repo root via `python -m`; launching a module by path
breaks the `core.*` imports.

## Models to download

About **16 GB** of weights. Nothing here is trained — every model is used
zero-shot, so these are the only artifacts needed to reproduce the results.

| Model | Repo | Size | Points at | Used for |
|---|---|---|---|---|
| SigLIP2 so400m | `google/siglip2-so400m-patch14-384` | 4.3 GB | `SIGLIP_MODEL` | visual embedding, 1152-d |
| jina-clip-v2 | `jinaai/jina-clip-v2` | 1.7 GB | `JINA_MODEL` | visual embedding, 1024-d | 
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | 7.1 GB | `QWEN_VL_MODEL` | answering |
| TransNetV2 | not on the Hub — see below | 30 MB | `TRANSNET_WEIGHTS` | shot boundaries |

```bash
export HF_HOME=/path/to/model/cache
export HF_HUB_DISABLE_XET=1          # see the note below

hf download google/siglip2-so400m-patch14-384
hf download jinaai/jina-clip-v2
hf download jinaai/xlm-roberta-flash-implementation   # jina-clip-v2's remote code
hf download Systran/faster-whisper-large-v3
hf download Qwen/Qwen2.5-VL-3B-Instruct
```

**TransNetV2** has no Hugging Face repo. Take
`transnetv2-pytorch-weights.pth` from the
[TransNetV2 release](https://github.com/soCzech/TransNetV2) and set
`TRANSNET_WEIGHTS` to it.

Each of `SIGLIP_MODEL`, `JINA_MODEL`, `QWEN_VL_MODEL` and `WHISPER_MODEL`
accepts either a Hub id or a local directory, so an existing copy can be reused
without re-downloading.

Three things worth knowing before the download:

- **`HF_HUB_DISABLE_XET=1` may be required.** On some networks `hf_xet` hangs
  indefinitely with no output. If a download sits at zero bytes, set this.
  `scripts/fetch_model_curl.sh <repo> <dir>` is a curl-based fallback that
  resumes; it exists because `hf download` cannot recover once its client has
  timed out — every later file then fails with "Cannot send a request, as the
  client has been closed".
- **Whisper is the CTranslate2 build**, `Systran/faster-whisper-large-v3`, not
  `openai/whisper-large-v3`. It is roughly 40× realtime at int8 and decodes
  audio through PyAV, so no librosa is needed.
- **jina-clip-v2 needs `trust_remote_code=True`** and pulls a second repo for
  its model code. Its licence is CC-BY-NC-4.0.

Two models are called over an API and need no download, only keys in `.env`:
`gemini-2.5-flash` for OCR (`OCR_MODEL`) and `deepseek-chat` for query
expansion (`DEEPSEEK_MODEL`).


## Evaluation

```bash
python -m eval.run_eval --config visual-only --split dev
./scripts/run_all_sequential.sh       # the whole sweep, resumable
```

`--split` defaults to `dev` deliberately. The test split is meant for one run per
config; defaulting to it would contaminate it during tuning. The dev/test split
is by **video**, not by question — four questions share a video, so splitting on
questions would leak evidence between the two sets.

| Config | What it isolates |
|---|---|
| `baseline-uniform` | 10 evenly spaced frames, no retrieval at all |
| `visual-only` | SigLIP2 + jina-clip-v2 |
| `+asr-gt` | plus ground-truth subtitles — the ASR branch's ceiling |
| `+asr-whisper` | plus real Whisper output (built, not in the default sweep) |
| `+ocr` | plus Gemini OCR |
| `full` | plus DeepSeek query expansion |




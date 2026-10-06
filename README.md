# Retrieval-Augmented VQA on Video-MME-v2

Answers multiple-choice questions about long videos by retrieving evidence —
keyframes, speech, on-screen text — and giving it to a local vision-language
model. Everything runs on one 12GB GPU.

The question this was built to answer is not "how high can accuracy go" but:

> Does retrieval-augmented VQA beat uniform frame sampling, and on which kinds
> of question?

That is why a uniform-sampling baseline is a mandatory config and every result is
reported per difficulty level with a paired significance test.

## Result

On the held-out test split of 600 questions, retrieval is **directionally
positive but not statistically significant**:

| Config | Accuracy | Non-Lin | L1 | L2 | L3 | Δ vs baseline | p |
|---|---|---|---|---|---|---|---|
| `baseline-uniform` | 0.208 | 0.094 | 0.314 | 0.124 | 0.196 | — | — |
| `visual-only` | 0.215 | 0.099 | 0.314 | 0.124 | 0.211 | +0.007 | 0.689 |
| `+asr-gt` | 0.218 | 0.100 | 0.314 | 0.143 | 0.207 | +0.010 | 0.504 |
| `+ocr` | **0.227** | **0.101** | 0.314 | 0.143 | 0.225 | +0.018 | 0.193 |
| `full` | 0.223 | 0.099 | 0.296 | **0.174** | 0.211 | +0.015 | 0.281 |

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

## Tests

```bash
python -m unittest discover -s tests -v
```

158 tests, no GPU or network needed. The integration tests skip themselves unless
the backend is running.

## What this reuses

17 modules are copied from a prior AIC 2025 codebase — the SigLIP2 and
jina-clip-v2 wrappers, `QdrantService`, `ElasticsearchService`,
`rrf_weighted_fuse`, Whisper+VAD, the Gemini OCR key rotation, and the TransNetV2
network. They carry fixes that were expensive to learn, such as SigLIP2 needing
lowercase text padded to exactly 64 tokens.

`routes.py`, `resources.py`, `schemas.py`, the Express server, the Video-MME-v2
loader, the stage orchestrator, shot detection over PyAV, the retrieval layer,
Qwen answering and the eval harness are written for this project.

See [`CLAUDE.md`](CLAUDE.md) for the environment constraints and the failure modes
already diagnosed — the exFAT/ext4 split, the unreachable package registries and
the substitutions they forced, and an Elasticsearch disk-watermark deadlock that
silently dropped 30,251 documents.

## Licence note

`jina-clip-v2` is CC-BY-NC-4.0 (non-commercial). Video-MME-v2 is MIT.

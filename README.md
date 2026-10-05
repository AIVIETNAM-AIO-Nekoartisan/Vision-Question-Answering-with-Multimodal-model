# VQA on Video-MME-v2

Answers multiple-choice questions about videos by retrieving evidence —
keyframes, speech, on-screen text — and handing it to a local VLM.

The question this is built to answer is not "how high can accuracy go" but:

> Does retrieval-augmented VQA beat uniform frame sampling, and on which kinds
> of question?

That is why `baseline-uniform` is a mandatory config and every result is
reported per difficulty Level.

## Pipeline

**Offline** — each stage runs across all videos so models load once, and every
stage writes artifacts to disk before anything reaches a store. Changing RRF
weights or an index mapping then costs a rerun of `index` alone.

```
videos/*.mp4 ─┬─ TransNetV2 ──► shots/*.json
              ├─ PyAV + phash ─► keyframes/*.jpg
              ├─ SigLIP2 + jina-clip-v2 ──► embeds/*.npz ──► Qdrant :6333
              ├─ subtitles | Whisper ──► asr/*.json ──────► Elasticsearch asr_data
              └─ Gemini OCR ──────────► ocr/*.json ──────► Elasticsearch ocr_data
```

**Online**

```
browser :3000 ──► FastAPI :8000 ──► Qdrant (siglip, jina)      ─┐
  (proxy /api,                      Elasticsearch (asr, ocr)   ─┴─► RRF
   /files, /gif)                                                     │
                                            top-k shots, time-ordered│
                                                       Qwen2.5-VL-3B ◄┘
                                                              │
                                                        letter A–H
```

## Data

[Video-MME-v2](https://hf.co/datasets/MME-Benchmarks/Video-MME-v2), MIT licence,
English, 1080p HEVC. 200 videos are indexed here (10 of 40 zips, ~20GB), giving
800 questions; the full set is 800 videos and 3,200 questions.

| | |
|---|---|
| Questions | 800 (dev 200 / test 600) |
| Options per question | 8 for 97% of them |
| Random baseline | 12.8% |
| Subtitle granularity | per word — grouped into segments by `data/videomme.py` |

The dev/test split is by **video**, not by question: four questions share a
video, so splitting on questions would leak.

## Quickstart

Requires the `jina_env` conda env (transformers 4.57.6 — see `CLAUDE.md` for why
the `<5` bound matters).

```bash
docker compose up -d                  # Qdrant :6333, Elasticsearch :9200
set -a; . ./.env; set +a

conda run -n jina_env python -m offline.run --stages all
conda run -n jina_env python -m online.backend.api_server     # :8000
cd online/frontend && npm start                                # :3000
```

Everything runs from the repo root via `python -m`; launching by path breaks the
`core.*` imports.

## Evaluation

```bash
conda run -n jina_env python -m eval.run_eval --config visual-only --split dev
```

`--split` defaults to `dev` on purpose: the test split gets one run per config,
and defaulting to test would contaminate it while tuning.

| Config | What it isolates |
|---|---|
| `baseline-uniform` | 10 evenly spaced frames, no retrieval at all |
| `visual-only` | SigLIP2 + jina-clip-v2 |
| `+asr-gt` | plus ground-truth subtitles — the ASR branch's ceiling |
| `+asr-whisper` | plus real Whisper output; the gap to `+asr-gt` is ASR error |
| `+ocr` | plus Gemini OCR |
| `full` | plus DeepSeek query expansion |

Results land in `results/eval/<config>-<split>.json` with accuracy, per-Level and
per-`second_head` breakdowns, the official group score with first-error
truncation, and the `unparsed` rate. If `unparsed` exceeds 5% the harness warns:
the prompt needs fixing before any accuracy number means anything.

## Tests

```bash
conda run -n jina_env python -m unittest discover -s tests -v
```

104 unit tests, no network or GPU needed. The integration tests in
`tests/test_integration.py` skip themselves unless the backend is up.

## What this reuses from AIC_2025

17 modules are copied as-is from `/mnt/data/AIC_2025-main` — SigLIP2 and
jina-clip-v2 embedding wrappers, `QdrantService`, `ElasticsearchService`,
`rrf_weighted_fuse`, Whisper+VAD, the Gemini OCR key rotation, and the TransNetV2
network. They carry fixes that were expensive to learn, such as SigLIP2's
lowercase-and-pad-to-64-tokens requirement.

`routes.py`, `resources.py`, `schemas.py` and the Express server are written
fresh rather than trimmed: AIC's versions are wired into the competition
features this project drops (DRES, events, guided search, BEiT-3, agents).

Six modules are new: the Video-MME-v2 loader, the stage orchestrator and
manifest, shot detection over PyAV, the retrieval/fusion layer, Qwen2.5-VL
answering, and the eval harness.

See `CLAUDE.md` for the environment constraints — the exFAT/ext4 split, the
unreachable package registries and the substitutions they forced.

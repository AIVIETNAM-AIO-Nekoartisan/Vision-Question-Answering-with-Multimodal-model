# Data Documentation

This document describes the exact data used by the retrieval-augmented VideoQA experiments, including the official source, pinned dataset revision, local subset, evaluation split, preprocessing procedure, derived artifacts, and reproduction commands.

## 1. Official Dataset

| Field | Value |
|---|---|
| Dataset | Video-MME-v2 |
| Official dataset page | [MME-Benchmarks/Video-MME-v2 on Hugging Face](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2) |
| Official repository | [MME-Benchmarks/Video-MME-v2 on GitHub](https://github.com/MME-Benchmarks/Video-MME-v2) |
| Project page and leaderboard | [video-mme-v2.netlify.app](https://video-mme-v2.netlify.app/) |
| Paper | [Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding](https://arxiv.org/abs/2604.05015) |
| Dataset version | Video-MME-v2, H.265 re-encoded release announced on 2026-06-11 |
| Pinned upstream revision | `6e4bebb03202e1ddbf3d37703e560e51c5aa2d64` |
| Revision last verified | 2026-10-07 |
| License | MIT, as declared in the official dataset card |
| Language | English |
| Task | Multiple-choice long-video question answering |


The complete official release contains:

- 800 videos at 1080p, distributed in 40 ZIP archives;
- 3,200 multiple-choice questions, with four questions for each video;
- `test.parquet`, containing questions, options, answers, hierarchy labels, and group metadata;
- `subtitle.zip`, containing one word-level timestamped JSONL file for each video.

Direct links to the pinned annotation files are:

- [`test.parquet`](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/6e4bebb03202e1ddbf3d37703e560e51c5aa2d64/test.parquet)
- [`subtitle.zip`](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/6e4bebb03202e1ddbf3d37703e560e51c5aa2d64/subtitle.zip)

## 2. Data Used in This Project

The project uses the first 200 videos rather than the complete 800-video benchmark. These videos are contained in official archives `videos/001.zip` through `videos/010.zip` and have identifiers `001` through `200`.

| Item | Full official release | Used in this project |
|---|---:|---:|
| Video archives | 40 | 10 (`001.zip`–`010.zip`) |
| Videos | 800 | 200 (`001.mp4`–`200.mp4`) |
| Questions | 3,200 | 800 |
| Questions per video | 4 | 4 |
| Raw video size | Not required locally in full | 18.39 GiB |
| Video encoding | H.265/HEVC, 1080p | H.265/HEVC, 1080p |

The experiment subset contains the following reasoning-level distribution:

| Reasoning level | Development | Test | Total |
|---|---:|---:|---:|
| Level 1 — Retrieval and aggregation | 29 | 159 | 188 |
| Level 2 — Temporal understanding | 51 | 161 | 212 |
| Level 3 — Complex reasoning | 120 | 280 | 400 |
| **Total** | **200** | **600** | **800** |

Of the 800 questions, 504 belong to relevance/consistency groups and 296 belong to logic/coherence groups.

## 3. Internal Data Split

Video-MME-v2 publishes its benchmark questions as one official `test` split. This project creates an internal development/test partition for controlled experimentation. It does not create a training split and does not train or fine-tune the answer model on Video-MME-v2.

| Internal split | Video IDs | Videos | Questions | Usage |
|---|---|---:|---:|---|
| Development | `001`–`050` | 50 | 200 | Debugging, prompt design, and configuration selection |
| Test | `051`–`200` | 150 | 600 | Final evaluation after freezing the pipeline |

The split is implemented in [`data/videomme.py`](data/videomme.py) through `DEV_VIDEO_IDS`. It is performed by video rather than by question because all four questions associated with one video share the same visual, speech, and OCR evidence. A question-level random split would leak evidence from the same video across development and test sets.

The split is deterministic and does not use a random seed:

```python
DEV_VIDEO_IDS = {"001", "002", ..., "050"}
TEST_VIDEO_IDS = {"051", "052", ..., "200"}
```

## 4. Raw Data Format

### 4.1 Question annotations

`test.parquet` contains the following columns:

| Column | Description |
|---|---|
| `video_id` | Three-digit identifier matching `<video_id>.mp4` |
| `url` | Original source-video URL |
| `group_type` | `relevance` or `logic`, used by the Non-Linear metric |
| `group_structure` | Ordering of the four related questions |
| `question_id` | Identifier such as `051-1` |
| `question` | Natural-language question |
| `options` | Candidate answers labeled from A to H when eight options are present |
| `answer` | Gold answer label |
| `level` | Reasoning level 1, 2, or 3 |
| `second_head` | Intermediate capability category |
| `third_head` | Fine-grained task category |

The loader parses the option string while preserving answer labels and restricts annotations to video files that are present locally.

### 4.2 Subtitles

`subtitle.zip` contains `subtitle/<video_id>.jsonl`. Each JSONL record represents one word with `text`, `start_time`, and `end_time`. The project converts these word-level entries into longer timestamped segments before indexing.

### 4.3 Videos

Each archive contains approximately 20 MP4 files. The local experiment directory contains `001.mp4` through `200.mp4`. The files are decoded with PyAV because the current release uses long-GOP H.265/HEVC video.

## 5. Integrity Checks

The local annotation files match the objects stored at the pinned Hugging Face revision.

| File | Bytes | SHA-256 |
|---|---:|---|
| `test.parquet` | 1,185,975 | `8dc7f8c8830aa49dd08a82592f8276899472a145155dde3bea5dd6914a65a9b4` |
| `subtitle.zip` | 7,854,449 | `adbd3cfd98bd03756398d1c8b63c7bcddf0e5c2494b6a0736ed890456021c287` |

Verify the files with:

```bash
sha256sum "$DATA_ROOT/annotations/test.parquet" \
  "$DATA_ROOT/annotations/subtitle.zip"

find "$DATA_ROOT/videos" -maxdepth 1 -name '*.mp4' | wc -l
```

The expected video count for this project is `200`.

## 6. Download Procedure

Set a data directory outside the Git repository because the videos and generated keyframes are large:

```bash
export DATA_ROOT=/path/to/vqa-data
export VIDEOMME_REVISION=6e4bebb03202e1ddbf3d37703e560e51c5aa2d64
mkdir -p "$DATA_ROOT/annotations" "$DATA_ROOT/videos" "$DATA_ROOT/_zips"
```

Download the pinned annotations:

```bash
curl -L --fail --retry 10 \
  -o "$DATA_ROOT/annotations/test.parquet" \
  "https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/$VIDEOMME_REVISION/test.parquet"

curl -L --fail --retry 10 \
  -o "$DATA_ROOT/annotations/subtitle.zip" \
  "https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/$VIDEOMME_REVISION/subtitle.zip"
```

Download and extract the ten video archives used by the experiments:

```bash
for n in $(seq 1 10); do
  archive=$(printf '%03d' "$n")
  curl -L --fail -C - --retry 999 --retry-all-errors \
    -o "$DATA_ROOT/_zips/$archive.zip" \
    "https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/resolve/$VIDEOMME_REVISION/videos/$archive.zip"
  unzip -tqq "$DATA_ROOT/_zips/$archive.zip"
  unzip -o -q -j "$DATA_ROOT/_zips/$archive.zip" -d "$DATA_ROOT/videos"
  rm "$DATA_ROOT/_zips/$archive.zip"
done
```

The repository also provides [`scripts/fetch_videomme2.sh`](scripts/fetch_videomme2.sh), which downloads, validates, extracts, and deletes one archive at a time to limit peak disk usage. The exact project subset can be downloaded with:

```bash
./scripts/fetch_videomme2.sh 10
```

That script currently uses the upstream `main` branch and has its data root configured near the top of the file. For strict reproduction, use the revision-pinned commands above or confirm that `main` still resolves to the documented commit before running the script.

## 7. Preprocessing Procedure

Preprocessing is implemented as resumable stages in [`offline/run.py`](offline/run.py) and [`offline/stages.py`](offline/stages.py). Completion state for each `(video_id, stage)` pair is stored in `manifest.db`.

### 7.1 Shot detection

1. PyAV decodes every video frame and resizes it to `48 × 27` RGB pixels.
2. TransNetV2 runs in 100-frame windows.
3. Each window uses 25 context frames on both sides, so 50 new frames are scored per step.
4. A transition threshold of `0.5` is applied.
5. Consecutive above-threshold frames are collapsed into one transition.
6. Shot boundaries and timestamps are written to `shots/<video_id>.json`.

### 7.2 Keyframe extraction and deduplication

Keyframes are selected according to shot duration:

- shorter than 1 second: one midpoint frame;
- from 1 to 10 seconds: frames at approximately 10%, 50%, and 90%;
- longer than 10 seconds: approximately one frame every 3 seconds, capped at 40 frames per shot.

Selected frames are:

- decoded sequentially with PyAV;
- resized so the longest side is at most 448 pixels;
- converted to RGB;
- saved as optimized JPEG with quality 70;
- deduplicated within each shot using a 64-bit perceptual hash, rejecting a frame when its Hamming distance from an already retained frame is at most 4.

The retained frame metadata is written to `keyframes/<video_id>/index.json`.

### 7.3 Visual embeddings

Every retained keyframe is embedded by:

- SigLIP2, producing a 1,152-dimensional vector;
- jina-clip-v2, producing a 1,024-dimensional vector.

Embedding uses chunks of 64 frames to bound GPU memory. Compressed vectors are written to `embeds/<video_id>.npz`, then inserted into Qdrant as named vectors with video, shot, frame, and timestamp metadata.

### 7.4 Ground-truth subtitle segmentation

The official word-level subtitles are converted into sentence-like segments. A new segment begins when:

- the silence between consecutive words exceeds 0.6 seconds; or
- the running segment would exceed 15 seconds.

The generated files use the same `{text, start, end}` representation as Whisper output and are stored under `asr/<video_id>.json`.

### 7.5 Optional Whisper transcription

The deployable ASR variant uses `Systran/faster-whisper-large-v3` with:

- language: English;
- device: CUDA;
- compute type: `int8_float16`;
- beam size: 5;
- voice activity detection enabled;
- minimum silence duration: 500 ms;
- `condition_on_previous_text=False` to reduce repetition loops.

Whisper output is stored separately under `asr_whisper/<video_id>.json` and indexed in `asr_data_whisper`, preventing it from overwriting ground-truth subtitle artifacts.

### 7.6 OCR

Gemini 2.5 Flash processes the retained keyframes. Only non-empty recognized text is kept. Each OCR record contains the relative keyframe path, shot number, timestamp, and text, and is stored in `ocr/<video_id>.json`. Because Gemini is an external service, exact OCR strings may vary if the provider updates the model behind the API name.

### 7.7 Text and vector indexes

- Qdrant stores one point per retained keyframe with SigLIP2 and jina-clip-v2 named vectors.
- Elasticsearch `asr_data` stores ground-truth subtitle segments.
- Elasticsearch `asr_data_whisper` stores Whisper segments.
- Elasticsearch `ocr_data` stores non-empty OCR records.

Text documents are filtered by `video_id` during retrieval. ASR records retain start and end timestamps, while OCR records retain shot and keyframe timestamps.

### 7.8 Evaluation-time evidence construction

The stored data is converted into a bounded answer context at evaluation time:

- eight uniformly sampled global frames, taken from the midpoint of eight equal temporal intervals;
- a transcript digest capped at 1,800 characters and sampled across the timeline;
- up to three retrieved temporal windows;
- at most three ordered keyframes per window, representing the beginning, middle, and end of the local span;
- a maximum model input of 20 images.

These context files are caches used by the experiment pipeline and do not alter the gold questions or labels.

## 8. Derived Artifact Inventory

The completed 200-video preprocessing run produced:

| Artifact | Files or records |
|---|---:|
| Shot JSON files | 200 |
| Detected shots | 20,831 |
| Keyframe index files | 200 |
| Retained keyframes / Qdrant points | 54,802 |
| Embedding NPZ files | 200 |
| Ground-truth subtitle segments | 12,846 |
| Whisper segments | 17,408 |
| Non-empty OCR records | 30,258 |

The counts above refer to the disk artifacts, which are the canonical inputs to indexing. A reused Elasticsearch volume should be cleared or use new index names before reproduction so stale documents from earlier runs do not affect database counts.

This project does not introduce new human annotations or redistribute a new dataset. Shot files, keyframes, embeddings, ASR segments, OCR records, and database indexes are derived experiment caches generated from the official Video-MME-v2 inputs. They are intentionally excluded from Git because of their size and are reproduced using the scripts below; the official Hugging Face dataset page is the public download source for the underlying data.

## 9. Reproducing the Processed Data

### 9.1 Environment

Install the Python dependencies and start the two local data stores:

```bash
pip install -r requirements.txt
docker compose up -d
cp .env.example .env
```

The reference preprocessing environment used Python 3.12.13, PyTorch 2.13.0 with CUDA 13.0, PyAV 18.1.0, and Transformers 4.57.6. PyTorch must be installed separately with a CUDA build compatible with the host driver. The model identifiers used by the code are `google/siglip2-so400m-patch14-384`, `jinaai/jina-clip-v2`, and `Systran/faster-whisper-large-v3`; equivalent complete local snapshots may be supplied through the environment variables below.

At minimum, configure these variables in `.env`:

```dotenv
DATA_ROOT=/path/to/vqa-data
VIDEO_DIRS=/path/to/vqa-data/videos
ANNOTATIONS_DIR=/path/to/vqa-data/annotations
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=videomme_keyframes
ES_HOST=localhost
ES_PORT=9200
ES_ASR_INDEX=asr_data
ES_OCR_INDEX=ocr_data
ASR_SOURCE=subtitle
TRANSNET_WEIGHTS=/path/to/transnetv2-pytorch-weights.pth
SIGLIP_MODEL=/path/to/siglip2
JINA_MODEL=jinaai/jina-clip-v2
WHISPER_MODEL=large-v3
OCR_MODEL=gemini-2.5-flash
```

OCR additionally requires a valid Gemini or Vertex AI Express API key as described in `.env.example`. Whisper processing does not require an external API.

Load the environment before running project modules:

```bash
set -a
. ./.env
set +a
```

### 9.2 Main preprocessing run

Run commands from the repository root:

```bash
python -m offline.run --stages shots
python -m offline.run --stages keyframes
python -m offline.run --stages embed
ASR_SOURCE=subtitle python -m offline.run --stages asr
python -m offline.run --stages ocr
ASR_SOURCE=subtitle python -m offline.run --stages index
```

The equivalent resumable command is:

```bash
python -m offline.run --stages all
```

The explicit form is preferred in a reproduction log because it makes the ASR source and stage order visible.

### 9.3 Optional Whisper branch

The manifest tracks `asr` as a stage name rather than tracking separate ASR sources. Therefore `--force` is required when creating Whisper artifacts after ground-truth subtitle artifacts:

```bash
ASR_SOURCE=whisper python -m offline.run --stages asr --force
ASR_SOURCE=whisper python -m offline.run --stages index_text --force
```

Return to the ground-truth index for the reported `+asr-gt`, `+ocr`, and `full` configurations with:

```bash
ASR_SOURCE=subtitle python -m offline.run --stages index_text --force
```

### 9.4 Evaluation

Run an individual configuration with:

```bash
python -m eval.run_eval --config +ocr --split dev
python -m eval.run_eval --config +ocr --split test
```

Run the five-configuration sequential sweep with:

```bash
./scripts/run_all_sequential.sh
```

The evaluation loader automatically restricts questions to videos present on disk. Consequently, a machine with only videos `001`–`200` evaluates exactly the 800-question project subset documented above.

## 10. Required Reproduction Files

| File | Role |
|---|---|
| [`scripts/fetch_videomme2.sh`](scripts/fetch_videomme2.sh) | Resumable video archive download and extraction |
| [`data/videomme.py`](data/videomme.py) | Annotation loading, option parsing, subtitle grouping, and internal split |
| [`offline/run.py`](offline/run.py) | Resumable preprocessing-stage orchestrator |
| [`offline/shots.py`](offline/shots.py) | TransNetV2 inference and keyframe-position selection |
| [`offline/decode.py`](offline/decode.py) | H.265 decoding, resizing, and JPEG output |
| [`offline/phash.py`](offline/phash.py) | Per-shot perceptual-hash deduplication |
| [`offline/stages.py`](offline/stages.py) | Keyframe, embedding, ASR, OCR, and indexing stages |
| [`core/models/whisper_fast.py`](core/models/whisper_fast.py) | Optional faster-whisper transcription |
| [`eval/configs.py`](eval/configs.py) | Experimental configurations and modality weights |
| [`eval/run_eval.py`](eval/run_eval.py) | Split selection, inference, and metric computation |
| [`scripts/run_all_sequential.sh`](scripts/run_all_sequential.sh) | Reproducible sequential evaluation sweep |



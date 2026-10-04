# VQA on Video-MME-v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer multiple-choice questions about videos by retrieving evidence (keyframes, speech, on-screen text) with RRF fusion and handing it to a local Qwen2.5-VL-3B, then measure whether that beats uniform frame sampling.

**Architecture:** Port 17 working files from the AIC 2025 system at `/mnt/data/AIC_2025-main` rather than rewriting them, trim 3, and write 6 new. Offline stages write artifacts to disk first and only then push to Qdrant/Elasticsearch, so reindexing after a schema or weight change costs minutes instead of rerunning models. Online, four retrieval sources fuse via the existing `rrf_weighted_fuse`, then aggregate keyframe hits up to shot level.

**Tech Stack:** Python 3.11 (conda), PyTorch + transformers 4.55–4.x, Qdrant, Elasticsearch 8.x, FastAPI, Express/vanilla JS, stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-04-vqa-videomme-design.md` — read it before Task 1.

## Global Constraints

- `transformers>=4.55,<5` — jina-clip-v2's remote code breaks on 5.x; SigLIP2 needs ≥4.49; Qwen2.5-VL needs ≥4.49. One version serves all three.
- `elasticsearch>=8,<9` — the compose file runs server 8.x and client 9 changes the media-type header.
- `accelerate==1.9.0`, `timm` / `torchscale` / `FlagEmbedding` / `ultralytics` / `streamlit` **must not** appear in `requirements.txt` — they exist in AIC only for BEiT-3, BLIP2 and YOLO, all dropped.
- GPU is a 12GB RTX A3000. Online holds SigLIP2 (~1.6GB) + jina-clip-v2 (~1.8GB) + Qwen2.5-VL-3B (~7GB) = ~10.4GB. Never load Whisper in the same process as Qwen.
- Tests use **stdlib `unittest`**, never pytest. Run `python3 -m unittest discover -s tests -v`.
- Every command runs **from the repo root**. Running a module by path breaks `core.*` imports.
- SigLIP2 text embedding requires lowercase + padding to exactly 64 tokens. `core/models/SigLIP_embedding.py` already handles this — do not "simplify" it.
- Qdrant payload keeps AIC's **nested** shape (`video.name`, `frame.timestamp_seconds`, `shot_id`, `file_path`). Flattening it breaks `fusion.py`, `parsing.py` and the frontend.
- `/` has ~18GB free. All data, keyframes, model weights and caches live under `DATA_ROOT=/media/nekoartisan/Lexar/vqa-data`.
- `/mnt/data/AIC_2025-main` is **read-only** for this work: it sits on branch `asr-full-b1` with 48 uncommitted files. Copy out of it, never write into it.
- Video names are `"001"`–`"800"`, not AIC's `L01_V001`.

---

## File Structure

| Path | Responsibility | Origin |
|---|---|---|
| `core/config.py` | Load `.env` | copy |
| `core/models/SigLIP_embedding.py` | SigLIP2 image/text embeddings, 1152-d | copy |
| `core/models/JinaCLIP_embedding.py` | jina-clip-v2 embeddings, 1024-d | copy |
| `core/models/Whisper_VAD.py` | Whisper + Silero VAD transcription | copy |
| `core/models/MultiLLM_OCR.py` | Gemini OCR, 30-key rotation | copy |
| `core/models/transnetv2_pytorch.py` | TransNetV2 network | copy |
| `core/services/vector_store_optimized.py` | `QdrantService` | copy |
| `core/services/elasticsearch_service.py` | `ElasticsearchService` | copy |
| `core/utils/parsing.py` | Payload/timestamp helpers | copy + widen video-name regex |
| `core/utils/video_identity.py` | Video name canonicalisation | copy |
| `core/models/qwen_vl.py` | Qwen2.5-VL-3B wrapper | **new** |
| `data/videomme.py` | `test.parquet` loader + word→segment grouping | **new** |
| `offline/video_trans_detection.py` | Shot detection, keyframe/GIF writing | copy |
| `offline/database_processing.py` | Keyframe → Qdrant payload + upsert | copy |
| `offline/save_images_and_gifs.py` | Keyframe/GIF helpers | copy |
| `offline/manifest.py` | SQLite `(video_id, stage)` state | **new** |
| `offline/run.py` | Stage orchestrator CLI | **new** |
| `online/backend/fusion.py` | `rrf_weighted_fuse` | copy, drop `enrich_results_metadata` |
| `online/backend/utils.py` | Frame/timestamp helpers | copy |
| `online/backend/schemas.py` | Pydantic models | copy + trim |
| `online/backend/resources.py` | Model/client lifespan | copy + trim |
| `online/backend/retrieve.py` | 4-source search → RRF → shot aggregation | **new** |
| `online/backend/vqa.py` | MCQ prompt, timestamp ordering, A–H parsing | **new** |
| `online/backend/routes.py` | 6 endpoints | **new** (original is unportable) |
| `online/backend/api_server.py` | FastAPI entrypoint | copy + trim |
| `eval/baseline.py` | Uniform frame sampling | **new** |
| `eval/run_eval.py` | Accuracy, per-Level, group score, unparsed rate | **new** |

---

## Task 1: Scaffold — copy, trim, bring up the stores

**Files:**
- Create: `requirements.txt`, `docker-compose.yml`, `core/__init__.py`, `core/models/__init__.py`, `core/services/__init__.py`, `core/utils/__init__.py`, `offline/__init__.py`, `online/__init__.py`, `online/backend/__init__.py`, `data/__init__.py`, `eval/__init__.py`, `tests/__init__.py`
- Copy: the 17 files listed in File Structure with origin `copy`
- Modify: `core/utils/parsing.py`, `online/backend/fusion.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable `core.*`, `offline.*`, `online.*`; `QdrantService(collection_name, named_vectors={"siglip":1152,"jina":1024}, qdrant_url)`; `ElasticsearchService(host, port, index_name)`; `rrf_weighted_fuse(results_by_model: Dict[str, List], k: int = 60, weights: Optional[Dict[str,float]] = None, topn: int = 300) -> List[Dict]`

- [ ] **Step 1: Copy the 17 files and create package markers**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
SRC=/mnt/data/AIC_2025-main

for f in core/config.py \
         core/models/SigLIP_embedding.py \
         core/models/JinaCLIP_embedding.py \
         core/models/Whisper_VAD.py \
         core/models/MultiLLM_OCR.py \
         core/models/transnetv2_pytorch.py \
         core/services/vector_store_optimized.py \
         core/services/elasticsearch_service.py \
         core/utils/parsing.py \
         core/utils/video_identity.py \
         offline/video_trans_detection.py \
         offline/database_processing.py \
         offline/save_images_and_gifs.py \
         online/backend/fusion.py \
         online/backend/utils.py \
         online/backend/schemas.py \
         online/backend/resources.py; do
  mkdir -p "$(dirname "$f")"
  cp "$SRC/$f" "$f"
done

cp "$SRC/core/services/__init__.py" core/services/__init__.py
for p in core core/models core/utils offline online online/backend data eval tests; do
  [ -f "$p/__init__.py" ] || touch "$p/__init__.py"
done
git add -A && git commit -q -m "chore: copy 17 reusable modules from AIC_2025"
```

- [ ] **Step 2: Write the failing scaffold test**

`tests/test_scaffold.py`:

```python
"""Guards the port: these are the exact breakages found when reading AIC source."""
import unittest


class TestImports(unittest.TestCase):
    def test_core_modules_import(self):
        from core.services.vector_store_optimized import QdrantService
        from core.services.elasticsearch_service import ElasticsearchService
        from online.backend.fusion import rrf_weighted_fuse
        self.assertTrue(callable(rrf_weighted_fuse))
        self.assertTrue(QdrantService and ElasticsearchService)


class TestVideoNameRegex(unittest.TestCase):
    """AIC's regex only accepted L01_V001; Video-MME-v2 names are '001'..'800'."""

    def test_videomme_name_is_accepted(self):
        from core.utils.parsing import is_known_video_name
        self.assertTrue(is_known_video_name("001"))
        self.assertTrue(is_known_video_name("800"))

    def test_aic_name_still_accepted(self):
        from core.utils.parsing import is_known_video_name
        self.assertTrue(is_known_video_name("L01_V001"))

    def test_junk_rejected(self):
        from core.utils.parsing import is_known_video_name
        self.assertFalse(is_known_video_name(""))
        self.assertFalse(is_known_video_name("not-a-video"))

    def test_metadata_not_flagged_for_videomme_payload(self):
        """With a '001' video name and a frame index present, nothing needs repair."""
        from core.utils.parsing import metadata_needs_repair
        payload = {
            "video": {"name": "001"},
            "frame": {"index": 120, "timestamp_seconds": 4.0},
            "file_path": "keyframes/001/0003_1.webp",
        }
        self.assertFalse(metadata_needs_repair(payload))


class TestFusionSurface(unittest.TestCase):
    def test_enrich_results_metadata_is_gone(self):
        """It only hydrated the caption collection, which this project drops."""
        import online.backend.fusion as fusion
        self.assertFalse(hasattr(fusion, "enrich_results_metadata"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_scaffold -v
```

Expected: FAIL — `ImportError: cannot import name 'is_known_video_name'`, and `test_enrich_results_metadata_is_gone` fails because the function is still present.

- [ ] **Step 4: Widen the video-name regex in `core/utils/parsing.py`**

Replace the `_VIDEO_NAME_RE` definition and add the helper. The old pattern rejected every Video-MME-v2 name, which made `metadata_needs_repair()` return `True` for all results:

```python
# Accept both AIC names (L01_V001) and Video-MME-v2 names (001..800).
_VIDEO_NAME_RE = re.compile(r"^(?:[A-Z]\d+_V\d+|\d{3})$", re.IGNORECASE)


def is_known_video_name(name: Optional[str]) -> bool:
    """Whether `name` is a recognised video identifier in either naming scheme."""
    return bool(name) and bool(_VIDEO_NAME_RE.fullmatch(str(name)))
```

Then rewrite `metadata_needs_repair` to use it:

```python
def metadata_needs_repair(payload: Optional[Dict[str, Any]]) -> bool:
    """Whether a result lacks a usable video name or frame index."""
    payload = payload or {}
    video = payload.get("video") or {}
    name = video.get("name") if isinstance(video, dict) else None
    return not is_known_video_name(name) or extract_frame_id_from_payload(payload) is None
```

- [ ] **Step 5: Drop `enrich_results_metadata` from `online/backend/fusion.py`**

Delete the whole function (lines 16–43 of the copied file) and trim the now-unused imports so only these remain:

```python
"""RRF / SRRF fusion for multi-source results (siglip, jina, asr, ocr)."""
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
```

Keep `rrf_weighted_fuse` and `srrf_weighted_fuse` byte-for-byte otherwise.

- [ ] **Step 6: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_scaffold -v
```

Expected: PASS, 6 tests.

- [ ] **Step 7: Write `requirements.txt`**

```
qdrant-client
transformers>=4.55,<5   # jina-clip-v2 remote code breaks on 5.x; siglip2 needs >=4.49
Pillow
python-dotenv
pandas
pyarrow
numpy
opencv-python
sentencepiece
accelerate==1.9.0
fastapi
uvicorn
python-multipart
einops
librosa
elasticsearch>=8,<9     # server is 8.x; client 9 changes the media-type header
google-genai>=1,<2
openai                  # DeepSeek speaks the OpenAI protocol
faster-whisper
qwen-vl-utils
imagehash
```

- [ ] **Step 8: Write `docker-compose.yml` — stores only, no GPU container**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - /media/nekoartisan/Lexar/vqa-data/qdrant_data:/qdrant/storage
    restart: unless-stopped

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.15.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
    volumes:
      - /media/nekoartisan/Lexar/vqa-data/es_data:/usr/share/elasticsearch/data
    restart: unless-stopped
```

The backend is **not** containerised: it needs the GPU, and CUDA-in-Docker buys nothing here.

- [ ] **Step 9: Bring the stores up and verify both answer**

```bash
docker compose config -q && echo "compose valid"
docker compose up -d
sleep 30
curl -sf http://localhost:6333/collections && echo " qdrant OK"
curl -sf http://localhost:9200/_cluster/health && echo " es OK"
```

Expected: both print OK. If Elasticsearch exits, check `docker compose logs elasticsearch` for `max virtual memory areas vm.max_map_count too low` and run `sudo sysctl -w vm.max_map_count=262144`.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -q -m "feat(scaffold): port 17 AIC modules, widen video-name regex, drop caption hydration

parsing.py's _VIDEO_NAME_RE only matched AIC names (L01_V001), so every
Video-MME-v2 result ('001'..'800') was flagged as needing metadata repair.
enrich_results_metadata only served the caption collection, which this
project drops, so it goes with it."
```

---

## Task 2: `data/videomme.py` — questions and subtitles

**Files:**
- Create: `data/videomme.py`
- Test: `tests/test_videomme_loader.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass Segment(text: str, start: float, end: float)`
  - `@dataclass Question(video_id: str, question_id: str, question: str, options: list[str], answer: str, level: str, group_type: str, second_head: str, third_head: str)`
  - `load_questions(parquet_path: Path, video_ids: Optional[set[str]] = None) -> list[Question]`
  - `group_words_into_segments(words: list[dict], max_gap: float = 0.6, max_duration: float = 15.0) -> list[Segment]`
  - `load_subtitle_segments(zip_path: Path, video_id: str, **kw) -> list[Segment]`
  - `DEV_VIDEO_IDS: frozenset[str]` (`"001"`–`"050"`), `is_dev(video_id) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/test_videomme_loader.py`:

```python
import unittest
from data.videomme import Segment, group_words_into_segments, is_dev


def w(text, start, end):
    return {"text": text, "start_time": start, "end_time": end}


class TestGroupWords(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(group_words_into_segments([]), [])

    def test_single_word(self):
        out = group_words_into_segments([w("Hi,", 1.0, 1.4)])
        self.assertEqual(out, [Segment(text="Hi,", start=1.0, end=1.4)])

    def test_words_close_together_join(self):
        out = group_words_into_segments([w("Hi", 1.0, 1.2), w("there", 1.25, 1.6)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "Hi there")
        self.assertAlmostEqual(out[0].start, 1.0)
        self.assertAlmostEqual(out[0].end, 1.6)

    def test_long_gap_splits(self):
        """A pause longer than max_gap starts a new segment."""
        out = group_words_into_segments(
            [w("first", 1.0, 1.2), w("second", 5.0, 5.3)], max_gap=0.6
        )
        self.assertEqual([s.text for s in out], ["first", "second"])

    def test_duration_cap_splits(self):
        """Continuous speech is still cut at max_duration."""
        words = [w(f"w{i}", float(i) * 0.5, float(i) * 0.5 + 0.4) for i in range(60)]
        out = group_words_into_segments(words, max_gap=0.6, max_duration=15.0)
        self.assertGreater(len(out), 1)
        for seg in out:
            self.assertLessEqual(seg.end - seg.start, 15.0 + 1e-6)

    def test_gap_measured_between_words_not_from_segment_start(self):
        """Three words with small gaps stay together even past 2x max_gap total."""
        out = group_words_into_segments(
            [w("a", 0.0, 0.2), w("b", 0.7, 0.9), w("c", 1.4, 1.6)], max_gap=0.6
        )
        self.assertEqual(len(out), 1)


class TestDevSplit(unittest.TestCase):
    def test_dev_boundary(self):
        self.assertTrue(is_dev("001"))
        self.assertTrue(is_dev("050"))
        self.assertFalse(is_dev("051"))
        self.assertFalse(is_dev("200"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_videomme_loader -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'data.videomme'`.

- [ ] **Step 3: Implement `data/videomme.py`**

```python
"""Video-MME-v2 annotations: MCQ questions and word-level subtitles.

Subtitles ship one word per JSONL line, which is useless for BM25, so they are
grouped into segments that match the shape Whisper produces. That lets
`asr_source` flip between ground-truth subtitles and Whisper output and makes
ASR error measurable rather than guessed.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Dev/test split is by video, not by question: four questions share a video, so
# splitting on questions would leak.
DEV_VIDEO_IDS = frozenset(f"{i:03d}" for i in range(1, 51))

_OPTION_RE = re.compile(r"(?:^|\s)([A-H])\.\s*")


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Question:
    video_id: str
    question_id: str
    question: str
    options: list[str]
    answer: str
    level: str
    group_type: str
    second_head: str
    third_head: str


def is_dev(video_id: str) -> bool:
    return str(video_id) in DEV_VIDEO_IDS


def parse_options(options_text: str) -> list[str]:
    """Split "A. Foo. B. Bar." into ["A. Foo.", "B. Bar."], preserving labels."""
    marks = list(_OPTION_RE.finditer(options_text or ""))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(options_text)
        out.append(options_text[m.start():end].strip())
    return out


def load_questions(
    parquet_path: Path, video_ids: Optional[set[str]] = None
) -> list[Question]:
    """Load test.parquet. `video_ids` restricts to videos actually on disk."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if video_ids is not None:
        df = df[df["video_id"].isin(video_ids)]
    return [
        Question(
            video_id=str(r.video_id),
            question_id=str(r.question_id),
            question=str(r.question),
            options=parse_options(str(r.options)),
            answer=str(r.answer).strip().upper(),
            level=str(r.level),
            group_type=str(r.group_type),
            second_head=str(r.second_head),
            third_head=str(r.third_head),
        )
        for r in df.itertuples(index=False)
    ]


def group_words_into_segments(
    words: Iterable[dict], max_gap: float = 0.6, max_duration: float = 15.0
) -> list[Segment]:
    """Group word-level entries into segments.

    Breaks on a silence longer than `max_gap` between consecutive words, or when
    the running segment would exceed `max_duration`. The gap is measured word to
    word, not from the segment start, so steady speech stays in one segment.
    """
    segments: list[Segment] = []
    buf: list[str] = []
    seg_start = 0.0
    prev_end = 0.0

    def flush():
        if buf:
            segments.append(
                Segment(text=" ".join(buf).strip(), start=seg_start, end=prev_end)
            )

    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        start = float(word.get("start_time", 0.0))
        end = float(word.get("end_time", start))

        if not buf:
            buf, seg_start, prev_end = [text], start, end
            continue

        if (start - prev_end) > max_gap or (end - seg_start) > max_duration:
            flush()
            buf, seg_start, prev_end = [text], start, end
        else:
            buf.append(text)
            prev_end = end

    flush()
    return segments


def load_subtitle_segments(
    zip_path: Path, video_id: str, **kw
) -> list[Segment]:
    """Read subtitle/<video_id>.jsonl out of subtitle.zip and group it."""
    with zipfile.ZipFile(zip_path) as zf:
        name = f"subtitle/{video_id}.jsonl"
        if name not in zf.namelist():
            return []
        lines = zf.read(name).decode("utf-8", "replace").splitlines()
    words = [json.loads(ln) for ln in lines if ln.strip()]
    return group_words_into_segments(words, **kw)
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_videomme_loader -v
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Verify against the real annotation files**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
python3 -c "
from pathlib import Path
from data.videomme import load_questions, load_subtitle_segments
A = Path('/media/nekoartisan/Lexar/vqa-data/annotations')
qs = load_questions(A/'test.parquet')
print('questions:', len(qs))
print('8-option share:', sum(len(q.options)==8 for q in qs)/len(qs))
segs = load_subtitle_segments(A/'subtitle.zip', '001')
print('video 001 segments:', len(segs), '(from 420 words)')
print('first:', segs[0] if segs else None)
"
```

Expected: 3200 questions, 8-option share ≈0.98, video 001 collapsing 420 words into roughly 40–80 segments. If the segment count comes back at ~420 the grouping is not firing — check `max_gap`.

- [ ] **Step 6: Commit**

```bash
git add data/videomme.py tests/test_videomme_loader.py
git commit -q -m "feat(data): Video-MME-v2 loader with word-level subtitle grouping

Subtitles are one word per line; grouping them into Segment(text,start,end)
gives the same shape Whisper emits, so asr_source can flip between
ground-truth and Whisper and the accuracy delta attributes cleanly to ASR error."
```

---

## Task 3: `offline/manifest.py` — resumable stage state

**Files:**
- Create: `offline/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Manifest(db_path: Path)` with `mark(video_id, stage, status, error=None)`, `status(video_id, stage) -> Optional[str]`, `is_done(video_id, stage) -> bool`, `pending(video_ids: list[str], stage: str) -> list[str]`, `count_done(stage) -> int`, `close()`. Table is `stage_state(video_id TEXT, stage TEXT, status TEXT, error TEXT, updated_at TEXT, PRIMARY KEY(video_id, stage))` — `scripts/watch_download_and_index.sh` already queries exactly this shape.

- [ ] **Step 1: Write the failing test**

`tests/test_manifest.py`:

```python
import tempfile
import unittest
from pathlib import Path

from offline.manifest import Manifest


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = Manifest(Path(self.tmp.name) / "manifest.db")

    def tearDown(self):
        self.m.close()
        self.tmp.cleanup()

    def test_unknown_is_not_done(self):
        self.assertIsNone(self.m.status("001", "shots"))
        self.assertFalse(self.m.is_done("001", "shots"))

    def test_mark_then_done(self):
        self.m.mark("001", "shots", "done")
        self.assertTrue(self.m.is_done("001", "shots"))
        self.assertEqual(self.m.status("001", "shots"), "done")

    def test_stages_are_independent(self):
        self.m.mark("001", "shots", "done")
        self.assertFalse(self.m.is_done("001", "embed"))

    def test_mark_is_idempotent_and_overwrites(self):
        self.m.mark("001", "shots", "failed", error="boom")
        self.m.mark("001", "shots", "done")
        self.assertEqual(self.m.status("001", "shots"), "done")
        self.assertEqual(self.m.count_done("shots"), 1)

    def test_failed_is_not_done_and_keeps_error(self):
        self.m.mark("002", "asr", "failed", error="cuda oom")
        self.assertFalse(self.m.is_done("002", "asr"))
        self.assertIn("cuda oom", self.m.error("002", "asr"))

    def test_pending_excludes_done_only(self):
        self.m.mark("001", "shots", "done")
        self.m.mark("002", "shots", "failed", error="x")
        self.assertEqual(self.m.pending(["001", "002", "003"], "shots"), ["002", "003"])

    def test_survives_reopen(self):
        self.m.mark("001", "shots", "done")
        self.m.close()
        reopened = Manifest(Path(self.tmp.name) / "manifest.db")
        self.assertTrue(reopened.is_done("001", "shots"))
        reopened.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_manifest -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'offline.manifest'`.

- [ ] **Step 3: Implement `offline/manifest.py`**

```python
"""SQLite record of which (video, stage) pairs are finished.

Indexing 200 videos through six stages takes hours on one GPU; without this a
crash on video 180 would redo the 179 before it. The schema is also read by
scripts/watch_download_and_index.sh, so the table name and columns are fixed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STAGES = ("shots", "keyframes", "embed", "asr", "ocr", "index")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_state (
    video_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (video_id, stage)
);
"""


class Manifest:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        # The watcher reads this file while the pipeline writes it.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def mark(
        self, video_id: str, stage: str, status: str, error: Optional[str] = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO stage_state (video_id, stage, status, error, updated_at) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(video_id, stage) DO UPDATE SET "
            "status=excluded.status, error=excluded.error, updated_at=excluded.updated_at",
            (
                str(video_id),
                stage,
                status,
                error,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()

    def status(self, video_id: str, stage: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM stage_state WHERE video_id=? AND stage=?",
            (str(video_id), stage),
        ).fetchone()
        return row[0] if row else None

    def error(self, video_id: str, stage: str) -> str:
        row = self._conn.execute(
            "SELECT error FROM stage_state WHERE video_id=? AND stage=?",
            (str(video_id), stage),
        ).fetchone()
        return (row[0] if row else None) or ""

    def is_done(self, video_id: str, stage: str) -> bool:
        return self.status(video_id, stage) == "done"

    def pending(self, video_ids: list[str], stage: str) -> list[str]:
        done = {
            r[0]
            for r in self._conn.execute(
                "SELECT video_id FROM stage_state WHERE stage=? AND status='done'",
                (stage,),
            )
        }
        return [v for v in video_ids if v not in done]

    def count_done(self, stage: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM stage_state WHERE stage=? AND status='done'", (stage,)
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_manifest -v
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Confirm the watcher's query works against a real db**

```bash
python3 -c "
from pathlib import Path
from offline.manifest import Manifest
m = Manifest(Path('/media/nekoartisan/Lexar/vqa-data/manifest.db'))
m.mark('001','index','done'); m.close()
"
sqlite3 /media/nekoartisan/Lexar/vqa-data/manifest.db \
  "SELECT COUNT(DISTINCT video_id) FROM stage_state WHERE stage='index' AND status='done';"
```

Expected: `1`. This is the exact query in `scripts/watch_download_and_index.sh:videos_indexed`, so it must not error.

- [ ] **Step 6: Commit**

```bash
git add offline/manifest.py tests/test_manifest.py
git commit -q -m "feat(offline): SQLite manifest for resumable per-video stage state"
```

---

## Task 4: `offline/run.py` — stage orchestrator

**Files:**
- Create: `offline/run.py`, `offline/stages.py`
- Test: `tests/test_stages.py`

**Interfaces:**
- Consumes: `Manifest`, `load_subtitle_segments`, `Segment`, `QdrantService`, `ElasticsearchService`, SigLIP/Jina models, `VideoIngestDatabase`
- Produces:
  - `offline/stages.py`: `stage_shots(video_id, ctx)`, `stage_keyframes(video_id, ctx)`, `stage_embed(video_id, ctx)`, `stage_asr(video_id, ctx)`, `stage_ocr(video_id, ctx)`, `stage_index(video_id, ctx)` — each `(str, Ctx) -> None`, raising on failure
  - `keyframe_payload(video_id, shot_number, position, frame_index, timestamp, file_path) -> dict` producing the nested AIC payload shape
  - `point_id_for(video_id, shot_number, frame_index) -> str` (deterministic UUID5)
  - `offline/run.py`: CLI `python -m offline.run --stages all|shots,embed,... [--limit N] [--videos 001,002] [--force]`

**Why stage-major, not video-major:** each stage loads its model once for all videos. Running video-major reloads five models per video — on 200 videos that is hours of pure load time.

- [ ] **Step 1: Write the failing test for payload shape and point ids**

`tests/test_stages.py`:

```python
import unittest

from offline.stages import keyframe_payload, point_id_for


class TestKeyframePayload(unittest.TestCase):
    """The nested shape is load-bearing: fusion.py keys on file_path, shot
    aggregation keys on shot_id, and parsing.py reads video.name and frame.*"""

    def setUp(self):
        self.p = keyframe_payload(
            video_id="001",
            shot_number=3,
            position=1,
            frame_index=120,
            timestamp=4.0,
            file_path="keyframes/001/0003_1.webp",
            shot_start=3.5,
            shot_end=6.0,
        )

    def test_has_flat_fusion_key(self):
        self.assertEqual(self.p["file_path"], "keyframes/001/0003_1.webp")

    def test_shot_id_groups_frames(self):
        self.assertEqual(self.p["shot_id"], "001_shot_003")

    def test_nested_video_and_frame(self):
        self.assertEqual(self.p["video"]["name"], "001")
        self.assertEqual(self.p["video"]["filename"], "001.mp4")
        self.assertEqual(self.p["frame"]["index"], 120)
        self.assertAlmostEqual(self.p["frame"]["timestamp_seconds"], 4.0)

    def test_shot_bounds_present(self):
        self.assertAlmostEqual(self.p["shot"]["start"], 3.5)
        self.assertAlmostEqual(self.p["shot"]["end"], 6.0)
        self.assertEqual(self.p["shot"]["number"], 3)

    def test_passes_parsing_sanity(self):
        from core.utils.parsing import metadata_needs_repair, extract_frame_id_from_payload
        self.assertFalse(metadata_needs_repair(self.p))
        self.assertEqual(extract_frame_id_from_payload(self.p), 120)


class TestPointId(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(point_id_for("001", 3, 120), point_id_for("001", 3, 120))

    def test_distinct_inputs_differ(self):
        self.assertNotEqual(point_id_for("001", 3, 120), point_id_for("001", 3, 121))
        self.assertNotEqual(point_id_for("001", 3, 120), point_id_for("002", 3, 120))

    def test_is_uuid(self):
        import uuid
        uuid.UUID(point_id_for("001", 3, 120))  # raises if malformed


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_stages -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'offline.stages'`.

- [ ] **Step 3: Implement `keyframe_payload` and `point_id_for` in `offline/stages.py`**

Write the two pure helpers first so the test goes green before the model-touching stages land:

```python
"""Offline stages. Each stage runs across all videos so models load once.

Artifacts are written to disk before anything reaches Qdrant/Elasticsearch, so
reindexing after a schema or RRF-weight change reruns only `index`.
"""
from __future__ import annotations

import uuid
from typing import Optional

# Stable namespace so point ids never change between runs.
_NS = uuid.UUID("6f3a1e84-0d2b-4c7e-9f15-8b2d4a6c1e90")


def point_id_for(video_id: str, shot_number: int, frame_index: int) -> str:
    """Deterministic Qdrant point id, so re-running `index` upserts in place."""
    return str(uuid.uuid5(_NS, f"{video_id}:{shot_number}:{frame_index}"))


def keyframe_payload(
    video_id: str,
    shot_number: int,
    position: int,
    frame_index: int,
    timestamp: float,
    file_path: str,
    shot_start: Optional[float] = None,
    shot_end: Optional[float] = None,
) -> dict:
    """Nested payload matching the AIC schema that fusion/parsing expect."""
    return {
        "file_path": file_path,
        "filename": file_path.rsplit("/", 1)[-1],
        "source_type": "video_keyframe",
        "shot_id": f"{video_id}_shot_{shot_number:03d}",
        "video": {"name": video_id, "filename": f"{video_id}.mp4"},
        "shot": {
            "number": shot_number,
            "position": position,
            "start": shot_start,
            "end": shot_end,
        },
        "frame": {
            "index": frame_index,
            "timestamp_seconds": float(timestamp),
            "timestamp_formatted": f"{int(timestamp // 60):02d}:{timestamp % 60:05.2f}",
        },
    }
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_stages -v
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit the pure helpers**

```bash
git add offline/stages.py tests/test_stages.py
git commit -q -m "feat(offline): keyframe payload builder and deterministic point ids"
```

- [ ] **Step 6: Add the six stage functions to `offline/stages.py`**

Append a `Ctx` dataclass carrying config and lazily-built clients, then the stages. Each stage reads from and writes to `DATA_ROOT`:

| Stage | Reads | Writes |
|---|---|---|
| `stage_shots` | `videos/{vid}.mp4` | `shots/{vid}.json` — `[{number,start,end,start_frame,end_frame}]` |
| `stage_keyframes` | mp4 + `shots/{vid}.json` | `keyframes/{vid}/{shot:04d}_{pos}.webp` + `keyframes/{vid}/index.json` |
| `stage_embed` | `keyframes/{vid}/index.json` | `embeds/{vid}.npz` — arrays `siglip` (N,1152), `jina` (N,1024), `file_paths` (N,) |
| `stage_asr` | `subtitle.zip` or mp4 | `asr/{vid}.json` — list of `Segment` dicts |
| `stage_ocr` | `keyframes/{vid}/` | `ocr/{vid}.json` — `[{file_path,text}]` |
| `stage_index` | all of the above | Qdrant points + ES docs |

Reuse `offline/video_trans_detection.py:VideoIngestDatabase.process_video` for shot boundaries and `save_frame_as_webp` for keyframe writing — do not reimplement either. Keyframe selection follows spec §5.1: 10%/50%/90% of shot duration; one frame if the shot is under 1s; every 3s up to 5 frames if over 10s; drop near-duplicates within a shot using `imagehash.phash` with a Hamming distance threshold of 4.

`stage_asr` honours `ASR_SOURCE`: `subtitle` calls `load_subtitle_segments`, `whisper` calls `WhisperTranscription`. Both write the identical JSON shape.

`stage_index` builds points with `point_id_for` + `keyframe_payload` + `{"siglip": ..., "jina": ...}` named vectors, calls `QdrantService.upsert_points`, then pushes ASR segments to `ES_ASR_INDEX` and OCR rows to `ES_OCR_INDEX` with `video_id` on every document.

- [ ] **Step 7: Write `offline/run.py`**

```python
"""Stage orchestrator. Run from the repo root:

    python -m offline.run --stages all
    python -m offline.run --stages shots,keyframes --limit 2
    python -m offline.run --stages index --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from offline.manifest import STAGES, Manifest
from offline import stages as S

logger = logging.getLogger("offline.run")

_FUNCS = {
    "shots": S.stage_shots,
    "keyframes": S.stage_keyframes,
    "embed": S.stage_embed,
    "asr": S.stage_asr,
    "ocr": S.stage_ocr,
    "index": S.stage_index,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run offline indexing stages.")
    ap.add_argument("--stages", default="all", help="'all' or comma-separated stage names")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos")
    ap.add_argument("--videos", default=None, help="comma-separated video ids")
    ap.add_argument("--force", action="store_true", help="redo videos already marked done")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    requested = STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip()
    )
    unknown = [s for s in requested if s not in _FUNCS]
    if unknown:
        ap.error(f"unknown stage(s): {', '.join(unknown)}")

    ctx = S.Ctx.from_env()
    manifest = Manifest(ctx.data_root / "manifest.db")
    failures = 0
    try:
        all_ids = (
            [v.strip() for v in args.videos.split(",")] if args.videos
            else S.discover_video_ids(ctx)
        )
        for stage in requested:
            todo = all_ids if args.force else manifest.pending(all_ids, stage)
            if args.limit:
                todo = todo[: args.limit]
            logger.info("stage %s: %d video(s) to do", stage, len(todo))
            for vid in todo:
                try:
                    _FUNCS[stage](vid, ctx)
                    manifest.mark(vid, stage, "done")
                except Exception as exc:
                    failures += 1
                    manifest.mark(vid, stage, "failed", error=traceback.format_exc(limit=5))
                    logger.error("stage %s video %s FAILED: %s", stage, vid, exc)
    finally:
        manifest.close()
        ctx.close()

    logger.info("done, %d failure(s)", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

A failing video marks `failed` and the loop continues — one bad mp4 must not stop an overnight run.

- [ ] **Step 8: Smoke-test on two videos end to end**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
set -a; . ./.env; set +a
python -m offline.run --stages all --limit 2 2>&1 | tail -25
sqlite3 "$DATA_ROOT/manifest.db" \
  "SELECT stage, status, COUNT(*) FROM stage_state GROUP BY stage, status;"
curl -s "http://localhost:6333/collections/$QDRANT_COLLECTION" | python3 -m json.tool | grep -E 'points_count|status'
curl -s "http://localhost:9200/$ES_ASR_INDEX/_count"
```

Expected: every stage `done` for two videos; Qdrant `points_count` > 0; ES ASR count > 0.

- [ ] **Step 9: Run the full available set and commit**

```bash
python -m offline.run --stages all 2>&1 | tail -15
git add offline/ tests/test_stages.py
git commit -q -m "feat(offline): six resumable stages plus orchestrator CLI

Stage-major ordering loads each model once for all videos; video-major would
reload five models per video, costing hours across 200 videos."
```

---

## Task 5: `online/backend/retrieve.py` — four sources, RRF, shot aggregation

**Files:**
- Create: `online/backend/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `rrf_weighted_fuse`, `QdrantService`, `ElasticsearchService`, SigLIP/Jina models
- Produces:
  - `@dataclass _FusableHit(score: float, payload: dict, id: Optional[str] = None)`
  - `es_hits_to_fusable(hits: list[dict], keyframes_by_video: dict) -> list[_FusableHit]`
  - `aggregate_to_shots(fused: list[dict], top_k: int) -> list[dict]`
  - `order_by_time(shots: list[dict]) -> list[dict]`
  - `retrieve(query: str, video_id: Optional[str], ctx, weights: dict, candidate_depth: int, rrf_k: int, top_k: int) -> list[dict]`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieve.py`:

```python
import unittest

from online.backend.retrieve import (
    _FusableHit,
    aggregate_to_shots,
    order_by_time,
)


def fused(file_path, shot_id, score, ts):
    return {
        "file_path": file_path,
        "score": score,
        "payload": {
            "file_path": file_path,
            "shot_id": shot_id,
            "frame": {"timestamp_seconds": ts},
            "video": {"name": shot_id.split("_")[0]},
        },
    }


class TestAggregateToShots(unittest.TestCase):
    def test_keeps_max_scoring_frame_per_shot(self):
        out = aggregate_to_shots(
            [
                fused("a.webp", "001_shot_001", 0.9, 1.0),
                fused("b.webp", "001_shot_001", 0.4, 2.0),
                fused("c.webp", "001_shot_002", 0.7, 9.0),
            ],
            top_k=10,
        )
        self.assertEqual(len(out), 2)
        best = {s["shot_id"]: s for s in out}
        self.assertAlmostEqual(best["001_shot_001"]["score"], 0.9)
        self.assertEqual(best["001_shot_001"]["file_path"], "a.webp")

    def test_respects_top_k(self):
        hits = [fused(f"{i}.webp", f"001_shot_{i:03d}", 1.0 - i * 0.1, i) for i in range(6)]
        self.assertEqual(len(aggregate_to_shots(hits, top_k=3)), 3)

    def test_top_k_picks_highest_scores(self):
        hits = [fused(f"{i}.webp", f"001_shot_{i:03d}", 1.0 - i * 0.1, i) for i in range(6)]
        kept = {s["shot_id"] for s in aggregate_to_shots(hits, top_k=2)}
        self.assertEqual(kept, {"001_shot_000", "001_shot_001"})

    def test_empty_input(self):
        self.assertEqual(aggregate_to_shots([], top_k=5), [])


class TestOrderByTime(unittest.TestCase):
    def test_sorts_ascending_by_timestamp_not_score(self):
        shots = aggregate_to_shots(
            [
                fused("late.webp", "001_shot_009", 0.9, 90.0),
                fused("early.webp", "001_shot_001", 0.2, 2.0),
            ],
            top_k=10,
        )
        ordered = order_by_time(shots)
        self.assertEqual([s["file_path"] for s in ordered], ["early.webp", "late.webp"])


class TestEsAdapter(unittest.TestCase):
    """ES returns dicts; rrf_weighted_fuse needs .score/.payload attributes."""

    def test_asr_hit_maps_to_keyframes_in_its_time_window(self):
        from online.backend.retrieve import es_hits_to_fusable
        keyframes = {
            "001": [
                {"file_path": "k1.webp", "payload": {"file_path": "k1.webp",
                 "shot_id": "001_shot_001", "frame": {"timestamp_seconds": 5.0}}},
                {"file_path": "k2.webp", "payload": {"file_path": "k2.webp",
                 "shot_id": "001_shot_002", "frame": {"timestamp_seconds": 50.0}}},
            ]
        }
        hits = [{"_score": 3.0, "video_id": "001", "start": 4.0, "end": 6.0}]
        out = es_hits_to_fusable(hits, keyframes)
        self.assertEqual([h.payload["file_path"] for h in out], ["k1.webp"])
        self.assertTrue(all(isinstance(h, _FusableHit) for h in out))
        self.assertAlmostEqual(out[0].score, 3.0)

    def test_hit_outside_any_window_yields_nothing(self):
        from online.backend.retrieve import es_hits_to_fusable
        keyframes = {"001": [{"file_path": "k1.webp", "payload": {
            "file_path": "k1.webp", "shot_id": "001_shot_001",
            "frame": {"timestamp_seconds": 5.0}}}]}
        hits = [{"_score": 3.0, "video_id": "001", "start": 900.0, "end": 910.0}]
        self.assertEqual(es_hits_to_fusable(hits, keyframes), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_retrieve -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'online.backend.retrieve'`.

- [ ] **Step 3: Implement the pure functions, then `retrieve()`**

Write `_FusableHit`, `es_hits_to_fusable`, `aggregate_to_shots` and `order_by_time` to satisfy the tests. `aggregate_to_shots` groups `fused` by `payload["shot_id"]`, keeps the highest-scoring member per shot, sorts shots by that score, truncates to `top_k`, and returns dicts carrying `shot_id`, `score`, `file_path`, `payload`, `timestamp`.

Then `retrieve()` runs the four sources with `candidate_depth` each — `QdrantService.search_by` with `vector_name="siglip"` then `"jina"`, `ElasticsearchService.search(query, limit=candidate_depth, video_ids=[video_id] if video_id else None)` against the ASR and OCR indices — feeds `{"siglip":…, "jina":…, "asr":…, "ocr":…}` into `rrf_weighted_fuse(k=rrf_k, weights=weights, topn=candidate_depth*2)`, then `aggregate_to_shots` and `order_by_time`.

A source whose weight is `0.0` is skipped entirely rather than queried and discarded — that is what makes the §8 ablation configs cheap.

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_retrieve -v
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add online/backend/retrieve.py tests/test_retrieve.py
git commit -q -m "feat(retrieve): 4-source RRF with ES adapter and shot aggregation

ES returns dicts while rrf_weighted_fuse expects .score/.payload, so ASR hits
are wrapped and mapped onto keyframes inside their time window. Fusion stays at
keyframe level (the existing, tested code path) and aggregates up to shots after."
```

---

## Task 6: `routes.py` + `api_server.py` — retrieval API, no VLM yet

**Files:**
- Create: `online/backend/routes.py`
- Modify: `online/backend/resources.py`, `online/backend/api_server.py`, `online/backend/schemas.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: `retrieve`, `QdrantService`, `ElasticsearchService`
- Produces: `GET /health`, `POST /search/kis`, `POST /search/audio`, `POST /search/ocr`, `GET /files?p=`, and `POST /vqa/answer` (wired in Task 7)

- [ ] **Step 1: Trim `resources.py`**

Delete every BEiT-3, BLIP2, OpenCLIP, YOLOE, GPT4o, caption-collection and events initialiser. Keep the lifespan pattern and the try/except around each model load — graceful degradation is why a missing weight disables one feature instead of crashing the server. `resources` must end up holding exactly: `qdrant`, `es_asr`, `es_ocr`, `siglip`, `jina`, `qwen` (None until Task 7), `kis_semaphore`.

- [ ] **Step 2: Write the integration test (auto-skipping)**

`tests/test_integration.py`:

```python
"""Skips unless the backend is actually up — matches the AIC convention."""
import os
import unittest
import urllib.error
import urllib.request

API = os.getenv("API_URL", "http://localhost:8000")


def _up():
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


@unittest.skipUnless(_up(), f"backend not running at {API}")
class TestEndpoints(unittest.TestCase):
    def _post(self, path, body):
        import json
        req = urllib.request.Request(
            f"{API}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def test_health_reports_stores(self):
        import json
        with urllib.request.urlopen(f"{API}/health", timeout=5) as r:
            data = json.loads(r.read())
        self.assertIn("qdrant", data)
        self.assertIn("elasticsearch", data)

    def test_kis_returns_shots_with_required_fields(self):
        out = self._post("/search/kis", {"query": "a person talking", "top_k": 5})
        self.assertIsInstance(out.get("results"), list)
        for item in out["results"]:
            self.assertIn("shot_id", item)
            self.assertIn("file_path", item)
            self.assertIn("timestamp", item)

    def test_video_filter_restricts_to_one_video(self):
        out = self._post(
            "/search/kis", {"query": "a person talking", "top_k": 5, "video_id": "001"}
        )
        for item in out["results"]:
            self.assertEqual(item["payload"]["video"]["name"], "001")

    def test_results_are_time_ordered(self):
        out = self._post("/search/kis", {"query": "a person talking", "top_k": 8})
        ts = [i["timestamp"] for i in out["results"]]
        self.assertEqual(ts, sorted(ts))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it and confirm it skips**

```bash
python3 -m unittest tests.test_integration -v
```

Expected: all skipped with "backend not running".

- [ ] **Step 4: Write `online/backend/routes.py`**

Six endpoints on one `APIRouter`. `/files` resolves `?p=` under `DATA_ROOT`, rejecting any resolved path that escapes it — path traversal here would serve arbitrary files off the Lexar drive. `/health` reports Qdrant point count, both ES doc counts, and which models loaded.

- [ ] **Step 5: Start the backend and run the integration test for real**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
set -a; . ./.env; set +a
python -m online.backend.api_server &
sleep 45
python3 -m unittest tests.test_integration -v
```

Expected: 4 tests pass. If imports fail with `No module named core`, you launched by path instead of `-m` from the repo root.

- [ ] **Step 6: Commit**

```bash
git add online/backend/ tests/test_integration.py
git commit -q -m "feat(api): minimal routes for KIS/ASR/OCR search plus health and files

The AIC routes.py was unportable: 39 references to events, 36 to guided, 18 to
BEiT-3 and 14 to agent would have dragged the whole competition system along."
```

---

## Task 7: Qwen2.5-VL-3B and MCQ answering

**Files:**
- Create: `core/models/qwen_vl.py`, `online/backend/vqa.py`
- Modify: `online/backend/routes.py`, `online/backend/resources.py`
- Test: `tests/test_mcq.py`

**Interfaces:**
- Consumes: `retrieve`, `order_by_time`, `Question`
- Produces:
  - `QwenVL(model_id, device, dtype)` with `load()`, `close()`, `answer(images: list[Path], prompt: str, max_new_tokens: int = 8) -> str`
  - `parse_letter(text: str, n_options: int) -> tuple[Optional[str], bool]` returning `(letter, unparsed)`
  - `build_mcq_prompt(question, shots, global_frames, transcript_digest) -> str`
  - `answer_question(question: Question, ctx, cfg) -> dict` with keys `answer`, `unparsed`, `shot_ids`, `raw`

- [ ] **Step 1: Write the failing parser test**

`tests/test_mcq.py`:

```python
import unittest

from online.backend.vqa import parse_letter


class TestParseLetter(unittest.TestCase):
    def test_bare_letter(self):
        self.assertEqual(parse_letter("C", 8), ("C", False))

    def test_letter_with_period(self):
        self.assertEqual(parse_letter("B.", 8), ("B", False))

    def test_sentence_form(self):
        self.assertEqual(parse_letter("The answer is C.", 8), ("C", False))

    def test_lowercase_is_normalised(self):
        self.assertEqual(parse_letter("f", 8), ("F", False))

    def test_leading_whitespace_and_newlines(self):
        self.assertEqual(parse_letter("\n\n  D\n", 8), ("D", False))

    def test_letter_beyond_option_count_is_rejected(self):
        """With 4 options, 'G' cannot be right — treat as unparsed."""
        self.assertEqual(parse_letter("G", 4), (None, True))

    def test_empty_output_is_unparsed(self):
        self.assertEqual(parse_letter("", 8), (None, True))

    def test_prose_without_a_letter_is_unparsed(self):
        self.assertEqual(parse_letter("I cannot tell from these frames.", 8), (None, True))

    def test_first_standalone_letter_wins(self):
        self.assertEqual(parse_letter("A or B? I pick B", 8), ("A", False))

    def test_word_starting_with_letter_is_not_matched(self):
        """'Alice' must not read as 'A'."""
        self.assertEqual(parse_letter("Alice is the mother", 8), (None, True))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_mcq -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'online.backend.vqa'`.

- [ ] **Step 3: Implement `parse_letter` in `online/backend/vqa.py`**

```python
import re
from typing import Optional

# \b on both sides so "Alice" does not read as "A"; allow a trailing . ) or :
_LETTER_RE = re.compile(r"\b([A-H])\b[.):]?")


def parse_letter(text: str, n_options: int) -> tuple[Optional[str], bool]:
    """Extract the chosen option letter. Returns (letter, unparsed).

    `unparsed` is reported in eval output: if it is high, accuracy is meaningless.
    """
    limit = max(1, min(int(n_options), 8))
    allowed = {chr(ord("A") + i) for i in range(limit)}
    for m in _LETTER_RE.finditer((text or "").upper()):
        if m.group(1) in allowed:
            return m.group(1), False
    return None, True
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_mcq -v
```

Expected: PASS, 10 tests.

- [ ] **Step 5: Implement `core/models/qwen_vl.py`**

Wrap `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` with `load()`/`close()`. Use `torch_dtype=torch.float16`, `device_map=None` and an explicit `.to("cuda")` so the 12GB budget stays predictable. Cap images per call at `max_images` (default 20) and downscale every image to a 448px long edge at JPEG q70 before the processor sees it — the AIC repo measured this (`2b45157`) as a large speed win with no accuracy loss. `close()` must `del` the model and call `torch.cuda.empty_cache()`.

- [ ] **Step 6: Implement `build_mcq_prompt` and `answer_question`**

Prompt order, per spec §7.2–7.3: global context (8 uniform frames + condensed transcript) → retrieved shots in **timestamp order** each labelled `[mm:ss]` with its ASR and OCR text → the question → the lettered options → `Answer with the single letter only.`

`answer_question` calls `retrieve()` with `video_id` set (in-video mode), builds the prompt, calls `QwenVL.answer`, parses with `parse_letter(…, len(question.options))`, and returns `answer`, `unparsed`, `shot_ids`, `raw`.

- [ ] **Step 7: Wire `POST /vqa/answer` and verify on a real question**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
set -a; . ./.env; set +a
python -m online.backend.api_server &
sleep 60
python3 -c "
import json, urllib.request
from pathlib import Path
from data.videomme import load_questions
qs = load_questions(Path('/media/nekoartisan/Lexar/vqa-data/annotations/test.parquet'), {'001'})
q = qs[0]
body = {'question_id': q.question_id}
req = urllib.request.Request('http://localhost:8000/vqa/answer',
    data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
print('gold:', q.answer)
print(json.loads(urllib.request.urlopen(req, timeout=180).read()))
"
nvidia-smi --query-gpu=memory.used --format=csv
```

Expected: a single letter A–H with `unparsed: false`, and GPU memory under 11.5GB. If it OOMs, set `QWEN_VL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ` and rerun.

- [ ] **Step 8: Commit**

```bash
git add core/models/qwen_vl.py online/backend/vqa.py online/backend/routes.py tests/test_mcq.py
git commit -q -m "feat(vqa): Qwen2.5-VL-3B answering with 3-tier letter parsing

Retrieved shots enter the prompt in timestamp order rather than score order, so
Level 2 temporal questions see evidence in the sequence it occurred."
```

---

## Task 8: Evaluation and the baseline

**Files:**
- Create: `eval/baseline.py`, `eval/run_eval.py`, `eval/configs.py`
- Test: `tests/test_eval_scoring.py`

**Interfaces:**
- Consumes: `load_questions`, `is_dev`, `answer_question`, `QwenVL`
- Produces:
  - `score_group(questions, predictions, group_type) -> float` applying first-error truncation for `logic`
  - `summarise(questions, predictions) -> dict` with `accuracy`, `by_level`, `by_second_head`, `group_score`, `unparsed_rate`, `n`
  - `uniform_frames(video_path, n=10) -> list[Path]`
  - CLI `python -m eval.run_eval --config <name> [--split dev|test] [--limit N]`

- [ ] **Step 1: Write the failing scoring test**

`tests/test_eval_scoring.py`:

```python
import unittest

from eval.run_eval import score_group, summarise
from data.videomme import Question


def q(qid, answer, level="1", group_type="relevance", head="Frame-Only"):
    return Question(
        video_id=qid.split("-")[0], question_id=qid, question="?",
        options=[f"{c}. x" for c in "ABCDEFGH"], answer=answer,
        level=level, group_type=group_type, second_head=head, third_head="t",
    )


class TestGroupScoring(unittest.TestCase):
    def test_relevance_scores_each_question(self):
        qs = [q("001-1", "A"), q("001-2", "B"), q("001-3", "C")]
        preds = {"001-1": "A", "001-2": "X", "001-3": "C"}
        self.assertAlmostEqual(score_group(qs, preds, "relevance"), 2 / 3)

    def test_logic_truncates_after_first_error(self):
        """Wrong on #2 zeroes #3 and #4 even though they match."""
        qs = [q("001-1", "A", group_type="logic"), q("001-2", "B", group_type="logic"),
              q("001-3", "C", group_type="logic"), q("001-4", "D", group_type="logic")]
        preds = {"001-1": "A", "001-2": "X", "001-3": "C", "001-4": "D"}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 1 / 4)

    def test_logic_all_correct_is_full_marks(self):
        qs = [q(f"001-{i}", "A", group_type="logic") for i in range(1, 5)]
        preds = {f"001-{i}": "A" for i in range(1, 5)}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 1.0)

    def test_logic_first_wrong_scores_zero(self):
        qs = [q(f"001-{i}", "A", group_type="logic") for i in range(1, 5)]
        preds = {"001-1": "X", "001-2": "A", "001-3": "A", "001-4": "A"}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 0.0)


class TestSummarise(unittest.TestCase):
    def test_reports_n_accuracy_and_unparsed(self):
        qs = [q("001-1", "A", level="1"), q("001-2", "B", level="2")]
        preds = {"001-1": "A", "001-2": None}
        out = summarise(qs, preds)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["unparsed_rate"], 0.5)

    def test_breaks_down_by_level(self):
        qs = [q("001-1", "A", level="1"), q("001-2", "B", level="3")]
        preds = {"001-1": "A", "001-2": "X"}
        out = summarise(qs, preds)
        self.assertAlmostEqual(out["by_level"]["1"], 1.0)
        self.assertAlmostEqual(out["by_level"]["3"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
python3 -m unittest tests.test_eval_scoring -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'eval.run_eval'`.

- [ ] **Step 3: Implement `score_group` and `summarise`**

`score_group` for `relevance` is the mean of exact matches. For `logic`, iterate the group in `question_id` order and stop crediting at the first mismatch, dividing by the full group size. `summarise` adds `by_level`, `by_second_head`, `unparsed_rate` (predictions that are `None`) and `n`.

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python3 -m unittest tests.test_eval_scoring -v
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Implement `eval/baseline.py` and `eval/configs.py`**

`uniform_frames(video_path, n=10)` pulls `n` evenly spaced frames with OpenCV, writes them to a cache under `DATA_ROOT/baseline_frames/<video_id>/`, and returns the paths. It touches neither Qdrant nor Elasticsearch — that is the point of the comparison.

`eval/configs.py` holds the six named configs from spec §8 as plain dicts of weights and flags:

| Name | siglip | jina | asr | ocr | asr_source | expand | retrieval |
|---|---|---|---|---|---|---|---|
| `baseline-uniform` | — | — | — | — | — | no | off |
| `visual-only` | 1.0 | 1.0 | 0.0 | 0.0 | — | no | on |
| `+asr-gt` | 1.0 | 1.0 | 0.8 | 0.0 | subtitle | no | on |
| `+asr-whisper` | 1.0 | 1.0 | 0.8 | 0.0 | whisper | no | on |
| `+ocr` | 1.0 | 1.0 | 0.8 | 0.5 | subtitle | no | on |
| `full` | 1.0 | 1.0 | 0.8 | 0.5 | subtitle | yes | on |

- [ ] **Step 6: Implement `eval/run_eval.py` CLI**

`--split` defaults to `dev` deliberately: the test split is for one final run per config, and a default of `test` invites contaminating it. Restrict questions to videos present on disk, print the markdown table, and write `results/eval/<config>-<split>.json`.

- [ ] **Step 7: Run the two headline configs on dev**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
set -a; . ./.env; set +a
python -m eval.run_eval --config baseline-uniform --split dev 2>&1 | tail -30
python -m eval.run_eval --config visual-only --split dev 2>&1 | tail -30
```

Expected: both report `n`, overall accuracy, per-Level accuracy and `unparsed_rate`. Sanity checks: accuracy well above the 12.8% random baseline; `unparsed_rate` under 0.05. If `unparsed_rate` is high, fix the prompt before reading any accuracy number.

- [ ] **Step 8: Commit**

```bash
git add eval/ tests/test_eval_scoring.py
git commit -q -m "feat(eval): accuracy, per-Level breakdown, group scoring, uniform baseline

--split defaults to dev: the test split gets one run per config, and defaulting
to test would quietly contaminate it across iterations."
```

---

## Task 9: Frontend

**Files:**
- Create: `online/frontend/server.js`, `online/frontend/public/index.html`, `online/frontend/package.json`
- Modify: none

**Interfaces:**
- Consumes: the Task 6/7 endpoints
- Produces: a page on `:3000` proxying `/api/*` → `:8000` and `/files?p=` → backend

- [ ] **Step 1: Copy and strip the Express server**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
SRC=/mnt/data/AIC_2025-main
mkdir -p online/frontend/public
cp "$SRC/online/frontend/server.js" online/frontend/server.js
cp "$SRC/online/frontend/env-loader.js" online/frontend/env-loader.js
cp "$SRC/online/frontend/package.json" online/frontend/package.json
```

Then remove the `/dres/*` proxy, the DRES env plumbing and anything referencing `DRES_URL` from `server.js`. Keep the `/files?p=` handler including its traversal guard.

- [ ] **Step 2: Write a single-page UI**

`public/index.html`: a question box, an options textarea, a shot grid rendered from `/api/search/kis` (thumbnail via `/files?p=`, `[mm:ss]` caption, ASR/OCR snippet), and an answer panel for `/api/vqa/answer`. No build step, no framework.

- [ ] **Step 3: Verify it serves and proxies**

```bash
cd online/frontend && npm install && npm start &
sleep 10
node --check server.js && echo "server.js parses"
curl -sf http://localhost:3000/ > /dev/null && echo "page OK"
curl -sf http://localhost:3000/api/health && echo " proxy OK"
```

Expected: all three print OK.

- [ ] **Step 4: Commit**

```bash
git add online/frontend
git commit -q -m "feat(frontend): Express proxy and single-page VQA UI, DRES removed"
```

---

## Task 10: Project docs and the full test sweep

**Files:**
- Create: `CLAUDE.md`, `README.md`
- Test: all

- [ ] **Step 1: Run every test together**

```bash
cd /home/nekoartisan/Documents/B3/Deep_Learning
python3 -m unittest discover -s tests -v 2>&1 | tail -25
```

Expected: all pass; integration tests pass if the backend is up, skip otherwise.

- [ ] **Step 2: Write `CLAUDE.md`**

Adapt AIC's `CLAUDE.md` for this project. It must carry the Global Constraints above verbatim, plus: run everything from the repo root; `export HF_HOME` before any model load; which AIC modules were ported and which were deliberately dropped; and the dev/test discipline from spec §2.0.

- [ ] **Step 3: Write `README.md`**

Quickstart (compose up, offline run, backend, frontend), the eval table once numbers exist, and a short "what this reuses from AIC_2025" section.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -q -m "docs: project conventions and quickstart"
```

---

## Self-Review

**Spec coverage.** §2 data → Task 2; §2.0 dev/test → Tasks 2, 8; §2.1 Level breakdown → Task 8; §2.2 group scoring → Task 8; §3.1–3.2 port/trim → Tasks 1, 6, 9; §3.3 new files → Tasks 2–8; §3.4 drops → Tasks 1, 6; §4 models → Tasks 1, 7; §4.1 versions → Global Constraints, Task 1 Step 7; §5 stages → Tasks 3, 4; §5.1 keyframes → Task 4 Step 6; §5.2 subtitles → Task 2; §6.1 Qdrant → Task 4; §6.2a the three port breakages → Task 1 Steps 4–5 and Task 5; §6.3 fusion → Task 5; §7 online flow → Tasks 5, 6; §7.1 time ordering → Task 5, Task 7 Step 6; §7.2 global context → Task 7 Step 6; §7.3 prompt and parsing → Task 7; §8 eval → Task 8; §9 risks → Task 8 (OCR gated behind the `+ocr` config measured on dev); §10 testing → every task; §11 operations → Task 10.

**Gap found and closed.** §9's "run OCR on ~50 dev videos before the full corpus" had no task enforcing the ordering. It is now covered: `eval/configs.py` makes `+ocr` a dev-set config, and `offline/run.py --stages ocr --limit 50` is the mechanism. Worth stating plainly during execution rather than letting a full-corpus OCR run happen by habit.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. Tasks 4 Step 6, 5 Step 3, 6 Step 4, 7 Steps 5–6 and 8 Steps 3, 5–6 specify behaviour and interfaces in prose rather than full listings — these are the bodies that depend on the ported modules' runtime behaviour, and their signatures are fixed in the Interfaces blocks and pinned by tests written first. That is deliberate, not a placeholder.

**Type consistency.** `Segment(text,start,end)` identical in Tasks 2, 4, 7. `Question` fields match between Tasks 2 and 8. `_FusableHit(score,payload,id)` consistent in Task 5. `point_id_for(video_id, shot_number, frame_index)` and `keyframe_payload(...)` identical in Tasks 4 and 5. `parse_letter(text, n_options) -> (letter, unparsed)` identical in Task 7 and consumed as such in Task 8. Manifest's `stage_state` columns match `scripts/watch_download_and_index.sh` exactly. `rrf_weighted_fuse(results_by_model, k, weights, topn)` matches the real signature read from source.

---

## Dependency Order

Tasks 1 → 2 → 3 → 4 must run in order. Task 5 needs 1 and 4. Task 6 needs 5. Task 7 needs 6. Task 8 needs 7. Tasks 9 and 10 can follow 8 in either order.

Tasks 1–4 work with the 20 videos already on disk and need no network.

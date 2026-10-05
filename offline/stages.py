"""Offline stages. Each stage runs across all videos so models load once.

Running video-major instead would reload five models per video — hours of pure
load time across 200 videos.

Every stage writes artifacts to disk before anything reaches Qdrant or
Elasticsearch. Changing the RRF weights or an index mapping then costs a rerun
of `index` alone, not of any model.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from data.videomme import Segment, discover_video_ids, load_subtitle_segments
from offline.decode import extract_frames, probe, save_jpeg
from offline.phash import hamming, phash
from offline.shots import ShotDetector, select_keyframe_positions

logger = logging.getLogger(__name__)

# Stable namespace so point ids never change between runs.
_NS = uuid.UUID("6f3a1e84-0d2b-4c7e-9f15-8b2d4a6c1e90")

PHASH_THRESHOLD = 4
KEYFRAME_MAX_SIDE = 448


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


@dataclass
class Ctx:
    """Config plus lazily-built clients, shared by every stage in a run."""

    data_root: Path
    videos_dir: Path
    annotations_dir: Path
    transnet_weights: Path
    asr_source: str = "subtitle"
    device: str = "cuda"
    qdrant_collection: str = "videomme_keyframes"
    qdrant_url: str = "http://localhost:6333"
    es_host: str = "localhost"
    es_port: int = 9200
    es_asr_index: str = "asr_data"
    es_ocr_index: str = "ocr_data"
    _cache: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Ctx":
        root = Path(os.getenv("DATA_ROOT", "/media/nekoartisan/Lexar/vqa-data"))
        return cls(
            data_root=root,
            videos_dir=Path(os.getenv("VIDEO_DIRS", str(root / "videos"))),
            annotations_dir=Path(
                os.getenv("ANNOTATIONS_DIR", str(root / "annotations"))
            ),
            transnet_weights=Path(
                os.getenv(
                    "TRANSNET_WEIGHTS",
                    "/mnt/data/AIC_2025-main/model_weights/transnetv2-pytorch-weights.pth",
                )
            ),
            asr_source=os.getenv("ASR_SOURCE", "subtitle"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "videomme_keyframes"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            es_host=os.getenv("ES_HOST", "localhost"),
            es_port=int(os.getenv("ES_PORT", "9200")),
            es_asr_index=os.getenv("ES_ASR_INDEX", "asr_data"),
            es_ocr_index=os.getenv("ES_OCR_INDEX", "ocr_data"),
        )

    # --- paths ---
    def video(self, vid: str) -> Path:
        return self.videos_dir / f"{vid}.mp4"

    def shots_json(self, vid: str) -> Path:
        return self.data_root / "shots" / f"{vid}.json"

    def keyframe_dir(self, vid: str) -> Path:
        return self.data_root / "keyframes" / vid

    def keyframe_index(self, vid: str) -> Path:
        return self.keyframe_dir(vid) / "index.json"

    def embeds_npz(self, vid: str) -> Path:
        return self.data_root / "embeds" / f"{vid}.npz"

    @property
    def asr_dir_name(self) -> str:
        """Artifacts are kept per source so both can exist at once.

        Sharing one path would make the +asr-gt vs +asr-whisper comparison
        meaningless: the Elasticsearch _id derives from file_path, so whisper
        segment i would overwrite subtitle segment i, and because the two sources
        produce different segment counts the index would end up a mix of both
        with neither intact.
        """
        return "asr" if self.asr_source == "subtitle" else f"asr_{self.asr_source}"

    @property
    def es_asr_index_for_source(self) -> str:
        if self.asr_source == "subtitle":
            return self.es_asr_index
        return f"{self.es_asr_index}_{self.asr_source}"

    def asr_json(self, vid: str) -> Path:
        return self.data_root / self.asr_dir_name / f"{vid}.json"

    def ocr_json(self, vid: str) -> Path:
        return self.data_root / "ocr" / f"{vid}.json"

    # --- lazily built, shared resources ---
    def shot_detector(self) -> ShotDetector:
        if "shots" not in self._cache:
            det = ShotDetector(self.transnet_weights, device=self.device)
            det.load()
            self._cache["shots"] = det
        return self._cache["shots"]

    def embedders(self) -> dict:
        """Whichever visual embedders actually load, keyed by vector name.

        Returning a partial set on purpose: the weights arrive over a link that
        drops, and because artifacts are written per-stage, re-running `embed`
        later to add a second vector costs nothing upstream.
        """
        if "embed" not in self._cache:
            available = {}
            try:
                from core.models.SigLIP_embedding import MultimodalEmbeddingSigLIP

                available["siglip"] = MultimodalEmbeddingSigLIP(fp16=True)
            except Exception as exc:
                logger.warning("siglip unavailable, skipping that vector: %s", exc)
            try:
                from core.models.JinaCLIP_embedding import MultimodalEmbeddingJinaCLIP

                available["jina"] = MultimodalEmbeddingJinaCLIP(fp16=True)
            except Exception as exc:
                logger.warning("jina unavailable, skipping that vector: %s", exc)
            if not available:
                raise RuntimeError("no visual embedder could be loaded")
            logger.info("embedders ready: %s", ", ".join(sorted(available)))
            self._cache["embed"] = available
        return self._cache["embed"]

    def qdrant(self):
        if "qdrant" not in self._cache:
            from core.services.vector_store_optimized import QdrantService

            svc = QdrantService(
                collection_name=self.qdrant_collection,
                qdrant_url=self.qdrant_url,
                named_vectors={"siglip": 1152, "jina": 1024},
            )
            svc.setup_collection()
            self._cache["qdrant"] = svc
        return self._cache["qdrant"]

    def es(self, index_name: str):
        key = f"es:{index_name}"
        if key not in self._cache:
            from core.services.elasticsearch_service import ElasticsearchService

            svc = ElasticsearchService(
                host=self.es_host, port=self.es_port, index_name=index_name
            )
            svc.setup_index()
            self._cache[key] = svc
        return self._cache[key]

    def close(self) -> None:
        det = self._cache.get("shots")
        if det is not None:
            det.close()
        self._cache.clear()


def discover_video_ids_ctx(ctx: Ctx) -> list[str]:
    return discover_video_ids(ctx.videos_dir)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def stage_shots(video_id: str, ctx: Ctx) -> None:
    out = ctx.shots_json(video_id)
    shots = ctx.shot_detector().detect(ctx.video(video_id))
    if not shots:
        raise RuntimeError(f"no shots detected for {video_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shots))


def stage_keyframes(video_id: str, ctx: Ctx) -> None:
    shots = json.loads(ctx.shots_json(video_id).read_text())
    info = probe(ctx.video(video_id))
    fps = info["fps"]

    wanted: list[tuple[int, int, int]] = []  # (frame_index, shot_number, position)
    for shot in shots:
        positions = select_keyframe_positions(shot["start"], shot["end"], fps)
        for pos, frame_index in enumerate(positions, start=1):
            wanted.append((frame_index, shot["number"], pos))

    frames = extract_frames(
        ctx.video(video_id), [w[0] for w in wanted], max_side=KEYFRAME_MAX_SIDE
    )

    kf_dir = ctx.keyframe_dir(video_id)
    kf_dir.mkdir(parents=True, exist_ok=True)
    by_shot: dict[int, list[int]] = {}
    records = []

    for frame_index, shot_number, pos in wanted:
        arr = frames.get(frame_index)
        if arr is None:
            continue
        # Reject frames that are near-identical to one already kept for this
        # shot: they cost embedding time and VLM context for no new evidence.
        h = phash(arr)
        seen = by_shot.setdefault(shot_number, [])
        if any(hamming(h, prev) <= PHASH_THRESHOLD for prev in seen):
            continue
        seen.append(h)

        rel = f"keyframes/{video_id}/{shot_number:04d}_{pos}.jpg"
        save_jpeg(arr, ctx.data_root / rel)
        shot = next(s for s in shots if s["number"] == shot_number)
        records.append(
            {
                "file_path": rel,
                "shot_number": shot_number,
                "position": pos,
                "frame_index": frame_index,
                "timestamp": round(frame_index / fps, 3),
                "shot_start": shot["start"],
                "shot_end": shot["end"],
            }
        )

    if not records:
        raise RuntimeError(f"no keyframes written for {video_id}")
    ctx.keyframe_index(video_id).write_text(json.dumps(records))


def stage_embed(video_id: str, ctx: Ctx) -> None:
    records = json.loads(ctx.keyframe_index(video_id).read_text())
    paths = [str(ctx.data_root / r["file_path"]) for r in records]

    arrays = {"file_paths": np.array([r["file_path"] for r in records])}
    for name, model in ctx.embedders().items():
        vectors = np.asarray(
            model.get_batch_image_embeddings(paths), dtype=np.float32
        )
        if vectors.shape[0] != len(records):
            raise RuntimeError(
                f"{video_id}: {name} returned {vectors.shape[0]} vectors "
                f"for {len(records)} keyframes"
            )
        arrays[name] = vectors

    out = ctx.embeds_npz(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)


def stage_asr(video_id: str, ctx: Ctx) -> None:
    """Writes the same JSON shape whichever source is configured."""
    if ctx.asr_source == "subtitle":
        segments = load_subtitle_segments(
            ctx.annotations_dir / "subtitle.zip", video_id
        )
    elif ctx.asr_source == "whisper":
        # core/models/whisper_fast.py, not AIC's Whisper_VAD.py: that module
        # imports librosa (uninstallable here) and pulls openai/whisper-large-v3
        # through transformers (~3GB, uncached), while the CTranslate2 build is
        # already on disk.
        from core.models.whisper_fast import FastWhisper

        if "whisper" not in ctx._cache:
            model = FastWhisper(language=os.getenv("WHISPER_LANGUAGE", "en"))
            model.load()
            ctx._cache["whisper"] = model
        segments = ctx._cache["whisper"].transcribe(ctx.video(video_id))
    else:
        raise ValueError(f"unknown ASR_SOURCE: {ctx.asr_source}")

    out = ctx.asr_json(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"source": ctx.asr_source, "segments": [s.to_dict() for s in segments]}
        )
    )


def stage_ocr(video_id: str, ctx: Ctx) -> None:
    """Gemini OCR over this video's keyframes.

    Gated deliberately: none of Video-MME-v2's 33 task types involve reading
    on-screen text, so this runs on the dev split first and its contribution is
    measured before any full-corpus spend.
    """
    from core.models.MultiLLM_OCR import MultiLLMOCR

    records = json.loads(ctx.keyframe_index(video_id).read_text())
    if "ocr" not in ctx._cache:
        # Three defaults in MultiLLMOCR have to be overridden here:
        #
        #  model_name: gemini-2.0-flash returns 404 for these keys, which are
        #    Vertex AI Express keys bound to project 26275598820 in
        #    asia-southeast1. Probed directly, gemini-2.5-flash answers.
        #  batch_size/max_output_tokens: the defaults are 128 images against 500
        #    output tokens, about 4 tokens per image, while each needs a
        #    {"text", "caption"} object of 50-60. The reply was truncated
        #    mid-string, json.loads failed, and the parser returned all-None —
        #    so OCR "succeeded" with no text at all. 2.5-flash also spends output
        #    budget on thinking, so the headroom is deliberately large.
        #  num_llms: the default of 6 would leave 24 of the 30 keys idle.
        ctx._cache["ocr"] = MultiLLMOCR(
            model_name=os.getenv("OCR_MODEL", "gemini-2.5-flash"),
            num_llms=int(os.getenv("OCR_NUM_LLMS", "30")),
            batch_size=int(os.getenv("OCR_BATCH_SIZE", "24")),
            max_output_tokens=int(os.getenv("OCR_MAX_TOKENS", "16384")),
        )
    ocr = ctx._cache["ocr"]

    rows = []
    for result in ocr.process_directory_iter(str(ctx.keyframe_dir(video_id))):
        # The text is nested under "ocr_result", not at the top level. Reading
        # result["text"] silently yielded nothing for every keyframe even though
        # Gemini was returning "Omeleto", "DRESS UP" and so on.
        payload = result.get("ocr_result") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        text = str((payload or {}).get("text") or "").strip()
        name = Path(str(result.get("file_path") or "")).name
        if not text or not name:
            continue
        match = next((r for r in records if Path(r["file_path"]).name == name), None)
        if match is None:
            continue
        rows.append(
            {
                "file_path": match["file_path"],
                "shot_number": match["shot_number"],
                "timestamp": match["timestamp"],
                "text": text,
            }
        )

    # Fail loudly on a total blank. A truncated reply makes the parser return
    # all-None, which previously wrote an empty file and marked the stage done —
    # the OCR branch would have scored zero for a parsing bug rather than for the
    # genuine reason the spec predicted. Some videos really do have no on-screen
    # text, so the guard trips only when nothing at all came back.
    if not rows and len(records) >= 20:
        raise RuntimeError(
            f"{video_id}: OCR returned no text for any of {len(records)} keyframes; "
            "check the model id, batch size and max_output_tokens"
        )

    out = ctx.ocr_json(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))


def stage_index(video_id: str, ctx: Ctx) -> None:
    from qdrant_client.models import PointStruct

    records = json.loads(ctx.keyframe_index(video_id).read_text())
    data = np.load(ctx.embeds_npz(video_id), allow_pickle=False)
    stored_paths = [str(p) for p in data["file_paths"]]
    if stored_paths != [r["file_path"] for r in records]:
        raise RuntimeError(
            f"{video_id}: embeds out of sync with keyframe index; rerun embed"
        )
    # Whichever vectors stage_embed managed to produce. Qdrant accepts a subset
    # of a collection's named vectors, so a missing model costs one source, not
    # the whole index.
    vector_names = [n for n in ("siglip", "jina") if n in data.files]
    if not vector_names:
        raise RuntimeError(f"{video_id}: embeds file has no vectors")

    points = []
    for i, r in enumerate(records):
        payload = keyframe_payload(
            video_id=video_id,
            shot_number=r["shot_number"],
            position=r["position"],
            frame_index=r["frame_index"],
            timestamp=r["timestamp"],
            file_path=r["file_path"],
            shot_start=r["shot_start"],
            shot_end=r["shot_end"],
        )
        points.append(
            PointStruct(
                id=point_id_for(video_id, r["shot_number"], r["frame_index"]),
                vector={n: data[n][i].tolist() for n in vector_names},
                payload=payload,
            )
        )
    ctx.qdrant().upsert_points(points)

    asr_path = ctx.asr_json(video_id)
    if asr_path.exists():
        blob = json.loads(asr_path.read_text())
        # The field must be named `content`: ElasticsearchService.search queries
        # content, content.ngram and content.exact. A `text` field would index
        # fine and never match anything.
        source = blob.get("source", ctx.asr_source)
        docs = [
            {
                # The source is part of the path, so the derived _id cannot
                # collide across sources even inside one index.
                "file_path": f"{ctx.asr_dir_name}/{video_id}/{i}",
                "video_id": video_id,
                "content": s["text"],
                "start": s["start"],
                "end": s["end"],
                "source": source,
            }
            for i, s in enumerate(blob.get("segments", []))
            if s.get("text", "").strip()
        ]
        if docs:
            ctx.es(ctx.es_asr_index_for_source).upsert_documents(docs)

    ocr_path = ctx.ocr_json(video_id)
    if ocr_path.exists():
        rows = json.loads(ocr_path.read_text())
        docs = [
            {
                "file_path": r["file_path"],
                "video_id": video_id,
                "content": r["text"],
                "shot_number": r["shot_number"],
                "timestamp": r["timestamp"],
            }
            for r in rows
            if r.get("text", "").strip()
        ]
        if docs:
            ctx.es(ctx.es_ocr_index).upsert_documents(docs)

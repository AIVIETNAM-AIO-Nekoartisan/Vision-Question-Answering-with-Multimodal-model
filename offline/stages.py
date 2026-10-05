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

    def asr_json(self, vid: str) -> Path:
        return self.data_root / "asr" / f"{vid}.json"

    def ocr_json(self, vid: str) -> Path:
        return self.data_root / "ocr" / f"{vid}.json"

    # --- lazily built, shared resources ---
    def shot_detector(self) -> ShotDetector:
        if "shots" not in self._cache:
            det = ShotDetector(self.transnet_weights, device=self.device)
            det.load()
            self._cache["shots"] = det
        return self._cache["shots"]

    def embedders(self):
        if "embed" not in self._cache:
            from core.models.JinaCLIP_embedding import MultimodalEmbeddingJinaCLIP
            from core.models.SigLIP_embedding import MultimodalEmbeddingSigLIP

            self._cache["embed"] = (
                MultimodalEmbeddingSigLIP(fp16=True),
                MultimodalEmbeddingJinaCLIP(fp16=True),
            )
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
    siglip, jina = ctx.embedders()
    paths = [str(ctx.data_root / r["file_path"]) for r in records]

    sig = np.asarray(siglip.get_batch_image_embeddings(paths), dtype=np.float32)
    jin = np.asarray(jina.get_batch_image_embeddings(paths), dtype=np.float32)
    if sig.shape[0] != len(records) or jin.shape[0] != len(records):
        raise RuntimeError(
            f"{video_id}: embedding count mismatch "
            f"siglip={sig.shape} jina={jin.shape} records={len(records)}"
        )

    out = ctx.embeds_npz(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        siglip=sig,
        jina=jin,
        file_paths=np.array([r["file_path"] for r in records]),
    )


def stage_asr(video_id: str, ctx: Ctx) -> None:
    """Writes the same JSON shape whichever source is configured."""
    if ctx.asr_source == "subtitle":
        segments = load_subtitle_segments(
            ctx.annotations_dir / "subtitle.zip", video_id
        )
    elif ctx.asr_source == "whisper":
        from core.models.Whisper_VAD import WhisperTranscription

        if "whisper" not in ctx._cache:
            ctx._cache["whisper"] = WhisperTranscription()
        raw = ctx._cache["whisper"].transcribe_file_with_sliding_window(
            str(ctx.video(video_id))
        )
        segments = [
            Segment(
                text=s.get("text", ""),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
            )
            for s in (raw or [])
        ]
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
        ctx._cache["ocr"] = MultiLLMOCR()
    ocr = ctx._cache["ocr"]

    rows = []
    for result in ocr.process_directory_iter(str(ctx.keyframe_dir(video_id))):
        text = (result.get("text") or result.get("ocr_text") or "").strip()
        name = Path(str(result.get("file_path") or result.get("image") or "")).name
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

    out = ctx.ocr_json(video_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))


def stage_index(video_id: str, ctx: Ctx) -> None:
    from qdrant_client.models import PointStruct

    records = json.loads(ctx.keyframe_index(video_id).read_text())
    data = np.load(ctx.embeds_npz(video_id), allow_pickle=False)
    sig, jin = data["siglip"], data["jina"]
    stored_paths = [str(p) for p in data["file_paths"]]
    if stored_paths != [r["file_path"] for r in records]:
        raise RuntimeError(
            f"{video_id}: embeds out of sync with keyframe index; rerun embed"
        )

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
                vector={"siglip": sig[i].tolist(), "jina": jin[i].tolist()},
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
        docs = [
            {
                "file_path": f"asr/{video_id}/{i}",
                "video_id": video_id,
                "content": s["text"],
                "start": s["start"],
                "end": s["end"],
                "source": blob.get("source", ctx.asr_source),
            }
            for i, s in enumerate(blob.get("segments", []))
            if s.get("text", "").strip()
        ]
        if docs:
            ctx.es(ctx.es_asr_index).upsert_documents(docs)

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

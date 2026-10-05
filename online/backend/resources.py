"""Process-wide models and store clients, built once at startup.

Rewritten rather than trimmed: AIC's 393-line version branches on BEiT-3,
BLIP2, the caption collection and the events index, none of which this project
builds. What is kept is the part that earned its place — each model loads inside
its own try/except, so a missing weight disables one feature instead of taking
down the server, and `snapshot()` reports exactly what came up.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AppResources:
    def __init__(self) -> None:
        self.data_root = Path(
            os.getenv("DATA_ROOT", "/media/nekoartisan/Lexar/vqa-data")
        )
        self.videos_dir = Path(
            os.getenv("VIDEO_DIRS", str(self.data_root / "videos"))
        )
        self.annotations_dir = Path(
            os.getenv("ANNOTATIONS_DIR", str(self.data_root / "annotations"))
        )

        self.qdrant = None
        self.es_asr = None
        self.es_ocr = None
        self.siglip = None
        self.jina = None
        self.qwen = None

        self.status: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self._video_ids: Optional[list[str]] = None
        self.kis_semaphore = threading.Semaphore(
            int(os.getenv("KIS_CONCURRENCY", "2"))
        )

    # --- lifecycle ---

    def _mark(self, name: str, ok: bool, error: Exception | None = None) -> None:
        self.status[name] = "ready" if ok else "failed"
        if error is not None:
            self.errors[name] = f"{type(error).__name__}: {error}"
            logger.warning("%s unavailable: %s", name, error)
        else:
            logger.info("%s ready", name)

    def initialize_datastores(self) -> None:
        from core.services.elasticsearch_service import ElasticsearchService
        from core.services.vector_store_optimized import QdrantService

        try:
            self.qdrant = QdrantService(
                collection_name=os.getenv("QDRANT_COLLECTION", "videomme_keyframes"),
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                named_vectors={"siglip": 1152, "jina": 1024},
            )
            self.qdrant.setup_collection()
            self._mark("qdrant", True)
        except Exception as exc:
            self.qdrant = None
            self._mark("qdrant", False, exc)

        # ASR_SOURCE selects which index to read: subtitle and whisper segments
        # live in separate indices so both stay intact and the +asr-gt vs
        # +asr-whisper comparison is between two clean corpora.
        asr_source = os.getenv("ASR_SOURCE", "subtitle")
        asr_index = os.getenv("ES_ASR_INDEX", "asr_data")
        if asr_source != "subtitle":
            asr_index = f"{asr_index}_{asr_source}"

        for attr, index_name in (
            ("es_asr", asr_index),
            ("es_ocr", os.getenv("ES_OCR_INDEX", "ocr_data")),
        ):
            try:
                svc = ElasticsearchService(
                    host=os.getenv("ES_HOST", "localhost"),
                    port=int(os.getenv("ES_PORT", "9200")),
                    index_name=index_name,
                )
                svc.setup_index()
                setattr(self, attr, svc)
                self._mark(attr, True)
            except Exception as exc:
                setattr(self, attr, None)
                self._mark(attr, False, exc)

    def initialize_models(self) -> None:
        try:
            from core.models.SigLIP_embedding import MultimodalEmbeddingSigLIP

            self.siglip = MultimodalEmbeddingSigLIP(fp16=True)
            self._mark("siglip", True)
        except Exception as exc:
            self.siglip = None
            self._mark("siglip", False, exc)

        try:
            from core.models.JinaCLIP_embedding import MultimodalEmbeddingJinaCLIP

            self.jina = MultimodalEmbeddingJinaCLIP(fp16=True)
            self._mark("jina", True)
        except Exception as exc:
            self.jina = None
            self._mark("jina", False, exc)

        if env_flag("ENABLE_VLM", True):
            try:
                from core.models.qwen_vl import QwenVL

                self.qwen = QwenVL(
                    model_id=os.getenv(
                        "QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
                    )
                )
                self.qwen.load()
                self._mark("qwen", True)
            except Exception as exc:
                self.qwen = None
                self._mark("qwen", False, exc)
        else:
            self.status["qwen"] = "disabled"

    # --- accessors used by retrieve() ---

    def embedder(self, name: str):
        return {"siglip": self.siglip, "jina": self.jina}.get(name)

    def all_video_ids(self) -> list[str]:
        if self._video_ids is None:
            self._video_ids = sorted(p.stem for p in self.videos_dir.glob("*.mp4"))
        return self._video_ids

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "models": dict(self.status),
            "errors": dict(self.errors),
            "videos_on_disk": len(self.all_video_ids()),
        }
        if self.qdrant is not None:
            try:
                out["qdrant"] = {
                    "collection": self.qdrant.collection_name,
                    "points": int(
                        self.qdrant.client.count(
                            collection_name=self.qdrant.collection_name, exact=True
                        ).count
                    ),
                }
            except Exception as exc:
                out["qdrant"] = {"error": str(exc)}
        else:
            out["qdrant"] = None

        es: dict[str, Any] = {}
        for label, svc in (("asr", self.es_asr), ("ocr", self.es_ocr)):
            if svc is None:
                es[label] = None
                continue
            try:
                es[label] = {
                    "index": svc.index_name,
                    "docs": int(
                        svc.client.count(index=svc.index_name).get("count", 0)
                    ),
                }
            except Exception as exc:
                es[label] = {"error": str(exc)}
        out["elasticsearch"] = es
        return out


resources = AppResources()

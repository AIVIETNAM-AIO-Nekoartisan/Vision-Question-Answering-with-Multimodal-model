"""Application resources with non-blocking model initialization."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from qdrant_client import models

from core.services.elasticsearch_service import ElasticsearchService
from core.services.vector_store_optimized import QdrantService

logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AppResources:
    """Own model/database handles and expose an observable startup state."""

    def __init__(self) -> None:
        self.qdrant = None
        self.qdrant_caption = None
        self.es_audio = None
        self.es_ocr = None
        self.gpt_service = None
        self.embedding_model_siglip = None
        self.embedding_model_jina = None
        self.embedding_model_beit3 = None
        self.embedding_model_bge = None
        self.reranker = None
        self.qdrant_point_count = 0
        self.qdrant_vector_counts: dict[str, int] = {}
        self.qdrant_caption_point_count = 0
        self.qdrant_events = None
        self.es_events = None
        self.event_store_status = "disabled" if not env_flag("ENABLE_EVENT_SEARCH", False) else "pending"
        self.event_counts = {"qdrant": 0, "es": 0}

        self.started_at = time.time()
        self.ready_at = None
        self.phase = "created"
        self.errors: dict[str, str] = {}
        self.model_status: dict[str, str] = {
            "siglip": "pending" if env_flag("ENABLE_SIGLIP", True) else "disabled",
            "jina": "pending" if env_flag("ENABLE_JINA", True) else "disabled",
            "bge": "pending" if env_flag("ENABLE_BGE", False) else "disabled",
            "beit3": "pending" if env_flag("ENABLE_BEIT3", False) else "disabled",
            "blip2": "pending" if env_flag("ENABLE_BLIP2", False) else "disabled",
            "gpt": "pending" if env_flag("ENABLE_GPT_EXPANSION", False) else "disabled",
        }
        self._lock = threading.RLock()
        self.kis_semaphore = threading.BoundedSemaphore(
            max(1, int(os.getenv("KIS_MAX_CONCURRENCY", "1"))),
        )

    def _set_model_status(self, name: str, status: str, error: Exception | None = None) -> None:
        with self._lock:
            self.model_status[name] = status
            if error is not None:
                self.errors[name] = str(error)

    def initialize_datastores(self) -> None:
        """Connect lightweight stores before yielding FastAPI startup."""
        self.phase = "datastores"
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = os.getenv("QDRANT_COLLECTION", "AIC_data")
        named_vectors = {"siglip": 1152}
        if env_flag("ENABLE_JINA", True):
            named_vectors["jina"] = 1024
        if env_flag("ENABLE_BEIT3", False):
            named_vectors["beit3"] = int(os.getenv("BEIT3_DIM", "1024"))

        self.qdrant = QdrantService(
            collection_name=collection,
            named_vectors=named_vectors,
            qdrant_url=qdrant_url,
        )
        qdrant_health = self.qdrant.health_check()
        if (
            qdrant_health.get("status") != "healthy"
            or qdrant_health.get("connection_type") == "memory"
        ):
            raise RuntimeError(f"Persistent Qdrant is unavailable: {qdrant_health}")
        if not self.qdrant.setup_collection():
            raise RuntimeError(f"Could not initialize Qdrant collection {collection}")
        self.qdrant_point_count = int(
            self.qdrant.client.count(
                collection_name=collection,
                exact=True,
            ).count,
        )
        for vector_name in named_vectors:
            self.qdrant_vector_counts[vector_name] = int(
                self.qdrant.client.count(
                    collection_name=collection,
                    count_filter=models.Filter(
                        must=[models.HasVectorCondition(has_vector=vector_name)],
                    ),
                    exact=True,
                ).count,
            )

        if env_flag("ENABLE_BGE", False):
            caption_collection = os.getenv(
                "QDRANT_CAPTION_COLLECTION",
                "keyframe-captions",
            )
            self.qdrant_caption = QdrantService(
                collection_name=caption_collection,
                vector_dim=1024,
                qdrant_url=qdrant_url,
            )
            if not self.qdrant_caption.setup_collection():
                self.errors["qdrant_caption"] = (
                    f"Could not initialize caption collection {caption_collection}"
                )
                self.qdrant_caption = None
            else:
                self.qdrant_caption_point_count = int(
                    self.qdrant_caption.client.count(
                        collection_name=caption_collection,
                        exact=True,
                    ).count,
                )

        es_host = os.getenv("ES_HOST", "localhost")
        es_port = int(os.getenv("ES_PORT", "9200"))
        try:
            self.es_audio = ElasticsearchService(
                host=es_host,
                port=es_port,
                index_name=os.getenv("ES_ASR_INDEX", "asr_data"),
            )
            self.es_audio.setup_index()
            self.es_ocr = ElasticsearchService(
                host=es_host,
                port=es_port,
                index_name=os.getenv("ES_OCR_INDEX", "ocr_data"),
            )
            self.es_ocr.setup_index()
        except Exception as error:
            self.errors["elasticsearch"] = str(error)
            self.es_audio = None
            self.es_ocr = None
            logger.exception("Elasticsearch initialization failed")

        if env_flag("ENABLE_EVENT_SEARCH", False):
            try:
                from core.services.event_store import EventVectorStore, EventTextStore
                from elasticsearch import Elasticsearch

                event_coll = os.getenv("QDRANT_EVENT_COLLECTION", "video-events")
                event_idx = os.getenv("ES_EVENT_INDEX", "video-events-text")
                self.qdrant_events = EventVectorStore(self.qdrant.client, collection_name=event_coll)

                es_raw_client = getattr(self.es_audio, "client", None) or Elasticsearch(f"http://{es_host}:{es_port}")
                self.es_events = EventTextStore(es_raw_client, index_name=event_idx)

                schema_ver = int(os.getenv("EVENT_SCHEMA_VERSION", "1"))
                q_ok = self.qdrant_events.verify(schema_version=schema_ver)
                es_ok = self.es_events.verify(schema_version=schema_ver)
                q_count = self.qdrant_events.count() if q_ok else 0
                es_count = self.es_events.count() if es_ok else 0
                self.event_counts["qdrant"] = q_count
                self.event_counts["es"] = es_count

                if not (q_ok and es_ok):
                    self.event_store_status = "unavailable"
                    self.errors["events"] = f"Event store schema verification failed: qdrant={q_ok}, es={es_ok}"
                elif q_count == 0 or es_count == 0:
                    self.event_store_status = "unavailable"
                    self.errors["events"] = f"Event store is empty: qdrant_count={q_count}, es_count={es_count}"
                elif q_count != es_count:
                    self.event_store_status = "unavailable"
                    self.errors["events"] = f"Event store count mismatch: qdrant_count={q_count}, es_count={es_count}"
                else:
                    q_chk = self.qdrant_events.compute_event_id_checksum()
                    es_chk = self.es_events.compute_event_id_checksum()
                    if q_chk != es_chk:
                        self.event_store_status = "unavailable"
                        self.errors["events"] = f"Event store checksum mismatch: qdrant_chk={q_chk[:12]}, es_chk={es_chk[:12]}"
                    else:
                        self.event_store_status = "ready"
            except Exception as error:
                self.event_store_status = "unavailable"
                self.errors["events"] = str(error)
                logger.exception("Event store initialization failed")

    def initialize_models(self) -> None:
        """Load heavy models in a worker thread after the HTTP server is live."""
        self.phase = "models"

        if env_flag("ENABLE_SIGLIP", True):
            self._set_model_status("siglip", "loading")
            try:
                from core.models.SigLIP_embedding import get_embedding_model

                self.embedding_model_siglip = get_embedding_model()
                self._set_model_status("siglip", "ready")
            except Exception as error:
                self.embedding_model_siglip = None
                self._set_model_status("siglip", "error", error)
                logger.exception("SigLIP initialization failed")

        if env_flag("ENABLE_JINA", True):
            self._set_model_status("jina", "loading")
            try:
                from core.models.JinaCLIP_embedding import get_jina_model

                self.embedding_model_jina = get_jina_model()
                self._set_model_status("jina", "ready")
            except Exception as error:
                self.embedding_model_jina = None
                self._set_model_status("jina", "error", error)
                logger.exception("Jina initialization failed")

        if env_flag("ENABLE_BGE", False):
            self._set_model_status("bge", "loading")
            try:
                from FlagEmbedding import BGEM3FlagModel

                self.embedding_model_bge = BGEM3FlagModel(
                    os.getenv("BGE_MODEL", "BAAI/bge-m3"),
                    use_fp16=os.getenv("BGE_DEVICE", "cpu").startswith("cuda"),
                    devices=os.getenv("BGE_DEVICE", "cpu"),
                )
                self._set_model_status("bge", "ready")
            except Exception as error:
                self.embedding_model_bge = None
                self._set_model_status("bge", "error", error)
                logger.exception("BGE-M3 initialization failed")

        if env_flag("ENABLE_BEIT3", False):
            self._set_model_status("beit3", "loading")
            try:
                from core.models.Beit3_embedding_optimized import (
                    get_embedding_model as get_beit3_model,
                )

                checkpoint = os.environ["BEIT3_CKPT"]
                tokenizer = os.environ["BEIT3_TOKENIZER"]
                self.embedding_model_beit3 = get_beit3_model(
                    checkpoint_path=checkpoint,
                    tokenizer_path=tokenizer,
                    device=os.getenv("BEIT3_DEVICE", "cpu"),
                    fp16=os.getenv("BEIT3_DEVICE", "cpu").startswith("cuda"),
                    input_size=384,
                    warmup=False,
                    aggressive_optimization=False,
                )
                self._set_model_status("beit3", "ready")
            except Exception as error:
                self.embedding_model_beit3 = None
                self._set_model_status("beit3", "error", error)
                logger.exception("BEiT-3 initialization failed")

        if env_flag("ENABLE_GPT_EXPANSION", False) or env_flag("ENABLE_EVENT_SEARCH", False) or env_flag("ENABLE_GUIDED_EVENT_SEARCH", False):
            self._set_model_status("gpt", "loading")
            try:
                from core.models.GPT4o_service import GPT4oService

                self.gpt_service = GPT4oService()
                self._set_model_status("gpt", "ready")
            except Exception as error:
                self.gpt_service = None
                self._set_model_status("gpt", "error", error)
                logger.exception("GPT query expansion initialization failed")

        if env_flag("ENABLE_BLIP2", False):
            self._set_model_status("blip2", "loading")
            try:
                from core.models.BLIP2_rerank import BLIP2Reranker

                self.reranker = BLIP2Reranker(
                    model_path=os.getenv("BLIP2_MODEL", "Salesforce/blip2-itm-vit-g"),
                    device=os.getenv("BLIP2_DEVICE", "cuda"),
                )
                self._set_model_status("blip2", "ready")
            except Exception as error:
                self.reranker = None
                self._set_model_status("blip2", "error", error)
                logger.exception("BLIP2 initialization failed")

        self.ready_at = time.time()
        self.phase = "ready" if self.kis_ready else "degraded"
        logger.info("Model initialization finished with phase=%s", self.phase)

    @property
    def kis_ready(self) -> bool:
        return (
            self.qdrant is not None
            and self.qdrant_point_count > 0
            and self.embedding_model_siglip is not None
        )

    @property
    def model_loading(self) -> bool:
        return any(
            status in {"pending", "loading"}
            for status in self.model_status.values()
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            guided_enabled = env_flag("ENABLE_GUIDED_EVENT_SEARCH", False)
            guided_ready = False
            guided_report_dict: dict[str, Any] = {}
            try:
                from online.backend.guided_readiness import verify_guided_readiness
                from dataclasses import asdict
                rep = verify_guided_readiness(self)
                guided_ready = bool(guided_enabled and rep.ready)
                guided_report_dict = asdict(rep)
            except Exception as e:
                guided_report_dict = {"ready": False, "error": str(e)}

            return {
                "phase": self.phase,
                "kis_ready": self.kis_ready,
                "guided_events_ready": guided_ready,
                "model_loading": self.model_loading,
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "models": dict(self.model_status),
                "services": {
                    "qdrant": self.qdrant is not None,
                    "qdrant_caption": self.qdrant_caption is not None,
                    "es_audio": self.es_audio is not None,
                    "es_ocr": self.es_ocr is not None,
                    "siglip": self.embedding_model_siglip is not None,
                    "jina": self.embedding_model_jina is not None,
                    "beit3": self.embedding_model_beit3 is not None,
                    "bge": self.embedding_model_bge is not None,
                    "gpt": self.gpt_service is not None,
                    "reranker": self.reranker is not None,
                    "event_qdrant": self.qdrant_events is not None and self.event_store_status == "ready",
                    "event_es": self.es_events is not None and self.event_store_status == "ready",
                    "guided_events": guided_ready,
                },
                "data": {
                    "qdrant_collection": (
                        self.qdrant.collection_name if self.qdrant is not None else None
                    ),
                    "qdrant_points_at_startup": self.qdrant_point_count,
                    "qdrant_vector_counts_at_startup": dict(
                        self.qdrant_vector_counts,
                    ),
                    "caption_points_at_startup": self.qdrant_caption_point_count,
                    "asr_index": (
                        self.es_audio.index_name if self.es_audio is not None else None
                    ),
                    "ocr_index": (
                        self.es_ocr.index_name if self.es_ocr is not None else None
                    ),
                    "event_collection": os.getenv("QDRANT_EVENT_COLLECTION", "video-events"),
                    "event_index": os.getenv("ES_EVENT_INDEX", "video-events-text"),
                    "event_status": self.event_store_status,
                    "event_counts": dict(self.event_counts),
                    "guided_events_report": guided_report_dict,
                },
                "errors": dict(self.errors),
            }


resources = AppResources()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing datastores")
    try:
        await asyncio.to_thread(resources.initialize_datastores)
    except Exception as error:
        resources.phase = "datastore_error"
        resources.errors["datastores"] = str(error)
        logger.exception("Datastore initialization failed")

    model_task = asyncio.create_task(asyncio.to_thread(resources.initialize_models))
    app.state.model_init_task = model_task
    logger.info("HTTP startup released; models continue loading in background")
    yield
    logger.info("Shutting down services")

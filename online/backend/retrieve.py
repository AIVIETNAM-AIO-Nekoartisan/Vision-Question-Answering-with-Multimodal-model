"""Four retrieval sources fused with RRF, then rolled up to shot level.

`rrf_weighted_fuse` is reused unchanged because it is the tested code path from
AIC. It keys on `payload["file_path"]`, i.e. fusion happens at keyframe level;
`aggregate_to_shots` then rolls keyframes up by `payload["shot_id"]`. Two steps,
no rewrite of fusion.

Elasticsearch returns plain dicts while `rrf_weighted_fuse` reads `.score` and
`.payload` off its inputs, so ASR and OCR hits are wrapped in `_FusableHit` and
mapped onto the keyframes they cover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from online.backend.fusion import rrf_weighted_fuse

logger = logging.getLogger(__name__)


@dataclass
class _FusableHit:
    """Shaped like a Qdrant ScoredPoint, enough for rrf_weighted_fuse."""

    score: float
    payload: dict
    id: Optional[str] = None


def _hit_score(hit: dict) -> float:
    for key in ("_score", "score", "rrf_score"):
        if hit.get(key) is not None:
            return float(hit[key])
    return 0.0


def es_hits_to_fusable(
    hits: list[dict], keyframes_by_video: dict[str, list[dict]]
) -> list[_FusableHit]:
    """Map Elasticsearch hits onto keyframes so they can join the fusion.

    An OCR row names its keyframe directly. An ASR segment covers a time span,
    so it maps onto every keyframe whose timestamp falls inside it — that is how
    spoken evidence reaches the frames it was spoken over.
    """
    out: list[_FusableHit] = []
    for hit in hits:
        video_id = str(hit.get("video_id") or "")
        keyframes = keyframes_by_video.get(video_id)
        if not keyframes:
            continue
        score = _hit_score(hit)

        file_path = hit.get("file_path")
        if file_path and hit.get("start") is None:
            for kf in keyframes:
                if kf.get("file_path") == file_path:
                    out.append(_FusableHit(score=score, payload=kf["payload"]))
                    break
            continue

        start = hit.get("start")
        end = hit.get("end")
        if start is None or end is None:
            continue
        for kf in keyframes:
            ts = kf.get("timestamp")
            if ts is None:
                ts = (kf.get("payload", {}).get("frame") or {}).get("timestamp_seconds")
            if ts is not None and float(start) <= float(ts) <= float(end):
                out.append(_FusableHit(score=score, payload=kf["payload"]))
    return out


def aggregate_to_shots(fused: list[dict], top_k: int) -> list[dict]:
    """Collapse fused keyframes to shots, keeping each shot's best frame."""
    best: dict[str, dict] = {}
    for item in fused:
        payload = item.get("payload") or {}
        shot_id = payload.get("shot_id")
        if not shot_id:
            continue
        score = float(item.get("score", 0.0))
        current = best.get(shot_id)
        if current is None or score > current["score"]:
            frame = payload.get("frame") or {}
            best[shot_id] = {
                "shot_id": shot_id,
                "score": score,
                "file_path": item.get("file_path") or payload.get("file_path"),
                "timestamp": frame.get("timestamp_seconds"),
                "payload": payload,
                "rrf_components": item.get("rrf_components", []),
            }
    ranked = sorted(best.values(), key=lambda s: s["score"], reverse=True)
    return ranked[:top_k]


def order_by_time(shots: list[dict]) -> list[dict]:
    """Sort by timestamp, not score.

    The VLM sees evidence in the order it occurred, which is what Level 2
    questions about ordering and temporal reasoning depend on.
    """
    return sorted(shots, key=lambda s: (s.get("timestamp") is None, s.get("timestamp")))


def _keyframes_for_videos(qdrant, video_ids: list[str]) -> dict[str, list[dict]]:
    """All keyframes of the given videos, for mapping text hits onto frames."""
    from qdrant_client import models as qmodels

    out: dict[str, list[dict]] = {}
    for video_id in video_ids:
        records = []
        offset = None
        while True:
            points, offset = qdrant.client.scroll(
                collection_name=qdrant.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="video.name",
                            match=qmodels.MatchValue(value=video_id),
                        )
                    ]
                ),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                frame = payload.get("frame") or {}
                records.append(
                    {
                        "file_path": payload.get("file_path"),
                        "shot_id": payload.get("shot_id"),
                        "timestamp": frame.get("timestamp_seconds"),
                        "payload": payload,
                    }
                )
            if offset is None:
                break
        out[video_id] = records
    return out


def retrieve(
    query: str,
    ctx: Any,
    video_id: Optional[str] = None,
    weights: Optional[dict[str, float]] = None,
    candidate_depth: int = 50,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Run the enabled sources, fuse, roll up to shots, order by time.

    A source whose weight is 0 is skipped rather than queried and discarded,
    which is what makes the ablation configs in spec section 8 cheap to run.
    """
    weights = weights or {"siglip": 1.0, "jina": 1.0, "asr": 0.8, "ocr": 0.5}
    results_by_model: dict[str, list] = {}

    video_filter = [video_id] if video_id else None

    # --- visual sources ---
    # search_by takes a Qdrant Filter, not a list of ids. video.name is already
    # payload-indexed by setup_collection().
    qdrant_filter = None
    if video_id:
        from qdrant_client import models as qmodels

        qdrant_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="video.name", match=qmodels.MatchValue(value=video_id)
                )
            ]
        )

    for name in ("siglip", "jina"):
        if weights.get(name, 0.0) <= 0.0:
            continue
        model = ctx.embedder(name)
        if model is None:
            logger.warning("%s embedder unavailable, skipping that source", name)
            continue
        hits = ctx.qdrant.search_by(
            vector_name=name,
            vector=model.get_text_embedding(query),
            limit=candidate_depth,
            filter_condition=qdrant_filter,
        )
        if hits:
            results_by_model[name] = hits

    # --- text sources ---
    need_keyframes = any(
        weights.get(n, 0.0) > 0.0 and getattr(ctx, f"es_{n}", None) is not None
        for n in ("asr", "ocr")
    )
    keyframes_by_video: dict[str, list[dict]] = {}
    if need_keyframes:
        videos = [video_id] if video_id else ctx.all_video_ids()
        keyframes_by_video = _keyframes_for_videos(ctx.qdrant, videos)

    for name in ("asr", "ocr"):
        if weights.get(name, 0.0) <= 0.0:
            continue
        service = getattr(ctx, f"es_{name}", None)
        if service is None:
            continue
        raw = service.search(
            query, limit=candidate_depth, video_ids=video_filter
        )
        fusable = es_hits_to_fusable(raw or [], keyframes_by_video)
        if fusable:
            results_by_model[name] = fusable

    if not results_by_model:
        return []

    fused = rrf_weighted_fuse(
        results_by_model,
        k=rrf_k,
        weights=weights,
        topn=max(candidate_depth * 2, top_k),
    )
    return order_by_time(aggregate_to_shots(fused, top_k))

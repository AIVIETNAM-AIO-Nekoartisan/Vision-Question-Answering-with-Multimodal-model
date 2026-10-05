"""API routes.

Written from scratch: AIC's routes.py could not be ported. It carries 39
references to `events`, 36 to `guided`, 18 to BEiT-3 and 14 to the agent layer,
so porting it would have dragged the whole competition system along.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from online.backend.resources import resources
from online.backend.retrieve import retrieve
from online.backend.schemas import (
    SearchRequest,
    SearchResponse,
    VQARequest,
    VQAResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _weights_from_env() -> dict[str, float]:
    return {
        "siglip": float(os.getenv("W_SIGLIP", "1.0")),
        "jina": float(os.getenv("W_JINA", "1.0")),
        "asr": float(os.getenv("W_ASR", "0.8")),
        "ocr": float(os.getenv("W_OCR", "0.5")),
    }


@router.get("/health")
def health() -> dict:
    return resources.snapshot()


def _search(req: SearchRequest, only: list[str] | None = None) -> SearchResponse:
    if resources.qdrant is None:
        raise HTTPException(503, "Qdrant unavailable")

    weights = dict(req.weights or _weights_from_env())
    if only is not None:
        weights = {k: (v if k in only else 0.0) for k, v in weights.items()}

    expanded = None
    query = req.query
    if req.expand:
        from online.backend.expand import expand_query

        expanded = expand_query(query)
        if expanded:
            query = expanded

    shots = retrieve(
        query=query,
        ctx=resources,
        video_id=req.video_id,
        weights=weights,
        candidate_depth=req.candidate_depth,
        rrf_k=req.rrf_k,
        top_k=req.top_k,
    )
    return SearchResponse(
        query=req.query,
        expanded_query=expanded,
        count=len(shots),
        results=shots,
    )


@router.post("/search/kis", response_model=SearchResponse)
def search_kis(req: SearchRequest) -> SearchResponse:
    """Full fusion over all enabled sources."""
    return _search(req)


@router.post("/search/audio", response_model=SearchResponse)
def search_audio(req: SearchRequest) -> SearchResponse:
    """Speech only, for inspecting what the ASR branch contributes."""
    return _search(req, only=["asr"])


@router.post("/search/ocr", response_model=SearchResponse)
def search_ocr(req: SearchRequest) -> SearchResponse:
    """On-screen text only, for inspecting what the OCR branch contributes."""
    return _search(req, only=["ocr"])


@router.post("/vqa/answer", response_model=VQAResponse)
def vqa_answer(req: VQARequest) -> VQAResponse:
    from online.backend.vqa import answer_request

    if resources.qwen is None:
        raise HTTPException(
            503, f"VLM unavailable: {resources.errors.get('qwen', 'not loaded')}"
        )
    try:
        return answer_request(req, resources, _weights_from_env())
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/files")
def files(p: str = Query(..., description="path relative to DATA_ROOT")) -> FileResponse:
    """Serve a keyframe or video from under DATA_ROOT.

    The resolved path is checked against DATA_ROOT because traversal here would
    hand out arbitrary files from the drive.
    """
    root = resources.data_root.resolve()
    target = (root / p).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(403, "path escapes DATA_ROOT")
    if not target.is_file():
        raise HTTPException(404, f"not found: {p}")
    return FileResponse(target)


@router.get("/gif")
def gif(
    video_id: str,
    start: float,
    end: float,
    fps: int = Query(default=8, ge=1, le=24),
    width: int = Query(default=320, ge=64, le=960),
) -> FileResponse:
    """Build a shot preview GIF on demand and cache it.

    Generated lazily rather than during indexing: a user looks at ~10 shots per
    query, while pre-rendering every shot would cost ~96GB at full corpus size.
    """
    from online.backend.gif import build_gif

    try:
        path = build_gif(
            resources.data_root, video_id, start, end, fps=fps, width=width
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return FileResponse(path, media_type="image/gif")

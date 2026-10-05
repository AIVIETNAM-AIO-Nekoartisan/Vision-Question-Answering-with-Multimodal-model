"""Request and response models.

Rewritten rather than trimmed: AIC's version carried 21 classes for agent chat,
composed search, temporal combination, events and guided refinement — all
features this project drops.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    video_id: Optional[str] = None
    candidate_depth: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    weights: Optional[dict[str, float]] = None
    expand: bool = False


class ShotHit(BaseModel):
    shot_id: str
    score: float
    file_path: Optional[str] = None
    timestamp: Optional[float] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    rrf_components: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    expanded_query: Optional[str] = None
    count: int
    results: list[ShotHit]


class VQARequest(BaseModel):
    """Either give a question_id from test.parquet, or a free-form question.

    question_id is the normal path for evaluation; the free-form fields exist so
    the UI can ask arbitrary questions about a video.
    """

    question_id: Optional[str] = None
    video_id: Optional[str] = None
    question: Optional[str] = None
    options: Optional[list[str]] = None
    top_k: int = Field(default=10, ge=1, le=40)
    weights: Optional[dict[str, float]] = None
    include_global_context: bool = True


class VQAResponse(BaseModel):
    question_id: Optional[str] = None
    video_id: str
    answer: Optional[str]
    unparsed: bool
    gold: Optional[str] = None
    correct: Optional[bool] = None
    shot_ids: list[str] = Field(default_factory=list)
    raw: str = ""

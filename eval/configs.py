"""The six comparison configurations from spec section 8.

Each isolates one branch so its contribution is a measured number rather than an
assumption. A weight of 0.0 means the source is never queried, not queried and
discarded, which is what makes the sweep cheap.
"""
from __future__ import annotations

from typing import Any

_VISUAL = {"siglip": 1.0, "jina": 1.0}


def _w(asr: float = 0.0, ocr: float = 0.0) -> dict[str, float]:
    return {**_VISUAL, "asr": asr, "ocr": ocr}


CONFIGS: dict[str, dict[str, Any]] = {
    # No Qdrant, no Elasticsearch: 10 evenly spaced frames straight to the VLM.
    # Every other number is read against this one.
    "baseline-uniform": {
        "retrieval": False,
        "weights": {"siglip": 0.0, "jina": 0.0, "asr": 0.0, "ocr": 0.0},
        "asr_source": None,
        "expand": False,
        "global_context": True,
    },
    "visual-only": {
        "retrieval": True,
        "weights": _w(),
        "asr_source": None,
        "expand": False,
        "global_context": True,
    },
    # Ground-truth subtitles: the upper bound on what the speech branch can give.
    "+asr-gt": {
        "retrieval": True,
        "weights": _w(asr=0.8),
        "asr_source": "subtitle",
        "expand": False,
        "global_context": True,
    },
    # Same weights, real transcripts. The gap to +asr-gt is the cost of ASR error.
    "+asr-whisper": {
        "retrieval": True,
        "weights": _w(asr=0.8),
        "asr_source": "whisper",
        "expand": False,
        "global_context": True,
    },
    # OCR weight stays low and this runs on dev first: none of Video-MME-v2's 33
    # task types involve reading on-screen text, so its value is unproven.
    "+ocr": {
        "retrieval": True,
        "weights": _w(asr=0.8, ocr=0.5),
        "asr_source": "subtitle",
        "expand": False,
        "global_context": True,
    },
    "full": {
        "retrieval": True,
        "weights": _w(asr=0.8, ocr=0.5),
        "asr_source": "subtitle",
        "expand": True,
        "global_context": True,
    },
}


def get_config(name: str) -> dict[str, Any]:
    if name not in CONFIGS:
        raise KeyError(
            f"unknown config {name!r}; choose from {', '.join(sorted(CONFIGS))}"
        )
    return CONFIGS[name]

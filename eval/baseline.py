"""Uniform frame sampling — the baseline every retrieval number is read against.

Touches neither Qdrant nor Elasticsearch. That is the point: without it an
accuracy figure says nothing, because the question is not "how high" but
"does retrieval beat just looking at the whole video evenly".
"""
from __future__ import annotations

import logging
from pathlib import Path

from offline.decode import extract_frames, probe, save_jpeg

logger = logging.getLogger(__name__)

MAX_SIDE = 448


def uniform_frames(
    video_path: Path,
    n: int = 10,
    cache_root: Path | None = None,
    max_side: int = MAX_SIDE,
) -> list[Path]:
    """Return paths to `n` frames spaced evenly across the video, cached on disk."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no video at {video_path}")

    video_id = video_path.stem
    out_dir = (cache_root or video_path.parent.parent / "baseline_frames") / video_id
    expected = [out_dir / f"u{n}_{i:02d}.jpg" for i in range(n)]
    if all(p.is_file() for p in expected):
        return expected

    info = probe(video_path)
    total = int(info["frames"])
    if total <= 0:
        raise RuntimeError(f"could not determine frame count for {video_path}")

    # Sample at the midpoint of n equal spans rather than at 0 and the last
    # frame, which are often black or a title card.
    indices = [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]
    frames = extract_frames(video_path, indices, max_side=max_side)

    out: list[Path] = []
    for i, idx in enumerate(indices):
        arr = frames.get(idx)
        if arr is None:
            continue
        path = out_dir / f"u{n}_{i:02d}.jpg"
        save_jpeg(arr, path)
        out.append(path)
    if not out:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return out

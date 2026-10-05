"""Lazy shot-preview GIFs via the ffmpeg CLI.

Generated on request and cached, not during indexing: a user looks at roughly 10
shots per query, so pre-rendering every shot would have cost ~96GB at full
corpus size and an entire offline stage.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_SECONDS = 6.0


def build_gif(
    data_root: Path,
    video_id: str,
    start: float,
    end: float,
    fps: int = 8,
    width: int = 320,
) -> Path:
    video = data_root / "videos" / f"{video_id}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"no video {video_id}")

    duration = max(0.2, min(float(end) - float(start), MAX_SECONDS))
    out = (
        data_root
        / "gif_cache"
        / video_id
        / f"{float(start):.2f}_{duration:.2f}_{fps}_{width}.gif"
    )
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    # palettegen/paletteuse keeps the file small enough to be worth caching.
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        f"split[a][b];[a]palettegen[p];[b][p]paletteuse"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{float(start):.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(video),
        "-vf", vf,
        "-loop", "0",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
    return out

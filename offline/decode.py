"""Video decoding via PyAV.

OpenCV is not available here (pypi unreachable) and would struggle anyway: the
Video-MME-v2 videos are HEVC/H.265, which cv2 frequently cannot decode without a
custom build. PyAV ships with faster-whisper, decodes HEVC, and resizes inside
`to_ndarray`, so no separate resize library is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

DECODE_THREADS = 8


def probe(path: Path) -> dict:
    """Return fps, frame count and duration without decoding the whole file."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        frames = int(stream.frames or 0)
        duration = float(container.duration / 1_000_000) if container.duration else 0.0
        if not frames and duration:
            frames = int(duration * fps)
        return {
            "fps": fps,
            "frames": frames,
            "duration": duration,
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
        }


def iter_small_frames(
    path: Path, width: int = 48, height: int = 27
) -> Iterator[np.ndarray]:
    """Yield every frame as uint8 RGB at `height`x`width` — TransNetV2's input."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = DECODE_THREADS
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24", width=width, height=height)


def extract_frames(
    path: Path, frame_indices: list[int], max_side: Optional[int] = None
) -> dict[int, np.ndarray]:
    """Decode the requested frame indices in one sequential pass.

    Sequential decoding beats seeking: these are long-GOP HEVC files where each
    seek costs a keyframe re-decode, and the indices are spread across the video.
    """
    import av

    wanted = set(frame_indices)
    if not wanted:
        return {}
    last = max(wanted)
    out: dict[int, np.ndarray] = {}

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = DECODE_THREADS
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                if max_side:
                    w, h = frame.width, frame.height
                    scale = min(1.0, max_side / max(w, h))
                    out[i] = frame.to_ndarray(
                        format="rgb24",
                        width=max(1, int(w * scale)),
                        height=max(1, int(h * scale)),
                    )
                else:
                    out[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    return out


def save_jpeg(arr: np.ndarray, path: Path, quality: int = 70) -> None:
    """Write an RGB array as JPEG. q70 follows the AIC measurement in 2b45157."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=quality, optimize=True)

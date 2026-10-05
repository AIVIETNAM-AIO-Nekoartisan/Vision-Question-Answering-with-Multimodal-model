"""Shot boundary detection with TransNetV2, and keyframe position selection.

Written against `core/models/transnetv2_pytorch.py` directly rather than reusing
AIC's `offline/video_trans_detection.py`: that module calls cv2 on every path,
including its PyAV branch, and cv2 is not installable here.

Unlike AIC's non-overlapping batches, inference uses the official sliding window
(50 new frames with 25 frames of context either side), which keeps the network
from guessing at window edges where most false cuts would otherwise appear.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from offline.decode import iter_small_frames, probe

logger = logging.getLogger(__name__)

WINDOW = 100
CONTEXT = 25
STRIDE = WINDOW - 2 * CONTEXT  # 50 scored frames per window


def predictions_to_shots(
    preds: np.ndarray, fps: float, threshold: float = 0.5
) -> list[dict]:
    """Turn per-frame transition probabilities into contiguous shots.

    A run of consecutive above-threshold frames is one boundary, not several: a
    dissolve or fade spans many frames but is still a single cut.
    """
    n = int(len(preds))
    if n == 0:
        return []

    above = preds > threshold
    # Keep only the first frame of each run of high predictions.
    cut_at = [
        i for i in range(n) if above[i] and (i == 0 or not above[i - 1])
    ]
    cut_at = [i for i in cut_at if 0 < i < n]

    bounds = [0, *cut_at, n]
    shots = []
    for number, (start_f, end_f) in enumerate(zip(bounds, bounds[1:]), start=1):
        last = end_f - 1
        if last < start_f:
            continue
        shots.append(
            {
                "number": number,
                "start_frame": int(start_f),
                "end_frame": int(last),
                "start": round(start_f / fps, 3),
                "end": round(last / fps, 3),
            }
        )
    # Renumber in case a degenerate span was skipped.
    for i, shot in enumerate(shots, start=1):
        shot["number"] = i
    return shots


def select_keyframe_positions(
    start: float, end: float, fps: float
) -> list[int]:
    """Frame indices to sample for one shot, per spec section 5.1.

    Under 1s: one frame. Over 10s: every 3s, capped at 5. Otherwise 10/50/90%.
    """
    duration = max(0.0, end - start)
    start_f = int(round(start * fps))
    end_f = max(start_f, int(round(end * fps)))

    if duration < 1.0:
        fracs = [0.5]
    elif duration > 10.0:
        step = 3.0
        count = min(5, max(1, int(duration // step)))
        fracs = [min(0.98, (i * step) / duration) for i in range(count)]
    else:
        fracs = [0.1, 0.5, 0.9]

    span = end_f - start_f
    positions = sorted({start_f + int(round(f * span)) for f in fracs})
    return [min(max(p, start_f), end_f) for p in positions]


class ShotDetector:
    """TransNetV2, loaded once per stage."""

    def __init__(self, weights_path: Path, device: str = "cuda"):
        self.weights_path = Path(weights_path)
        self.device = device
        self.model = None

    def load(self) -> None:
        import torch

        from core.models.transnetv2_pytorch import TransNetV2

        model = TransNetV2()
        state = torch.load(
            str(self.weights_path), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model
        logger.info("TransNetV2 loaded from %s on %s", self.weights_path, self.device)

    def close(self) -> None:
        import torch

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, video_path: Path, threshold: float = 0.5) -> list[dict]:
        """Return the shot list for one video."""
        import torch

        if self.model is None:
            raise RuntimeError("ShotDetector.load() must be called first")

        info = probe(video_path)
        fps = info["fps"]
        frames = np.stack(list(iter_small_frames(video_path)))
        n = len(frames)
        if n == 0:
            return []

        # Pad both ends by CONTEXT so the first and last real frames get the
        # same amount of context as everything in between.
        padded = np.concatenate(
            [
                np.repeat(frames[:1], CONTEXT, axis=0),
                frames,
                np.repeat(frames[-1:], CONTEXT + WINDOW, axis=0),
            ]
        )

        scores = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for offset in range(0, n, STRIDE):
                window = padded[offset: offset + WINDOW]
                if len(window) < WINDOW:
                    break
                batch = torch.from_numpy(window).unsqueeze(0).to(self.device)
                logits, _ = self.model(batch.byte())
                probs = torch.sigmoid(logits)[0, :, 0].cpu().numpy()
                middle = probs[CONTEXT: CONTEXT + STRIDE]
                take = min(STRIDE, n - offset)
                scores[offset: offset + take] = middle[:take]

        shots = predictions_to_shots(scores, fps=fps, threshold=threshold)
        logger.info(
            "%s: %d frames @ %.2ffps -> %d shots", video_path.name, n, fps, len(shots)
        )
        return shots

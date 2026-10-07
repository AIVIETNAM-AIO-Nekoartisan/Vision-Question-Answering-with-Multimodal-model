"""Whisper transcription via faster-whisper, emitting the project's Segment shape.

Replaces `core/models/Whisper_VAD.py` for this project. That module cannot run
here for two independent reasons:

  * it imports librosa at module level, and librosa is not installable on this
    network (pypi is unreachable);
  * it loads `openai/whisper-large-v3` through transformers, a ~3GB download that
    is not cached, whereas `Systran/faster-whisper-large-v3` already is.

faster-whisper also decodes audio itself through PyAV and bundles Silero VAD as
ONNX, so neither librosa nor `torch.hub.load("snakers4/silero-vad")` is needed,
and CTranslate2 int8 is several times faster than the transformers path on a
12GB card.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional

from data.videomme import Segment

logger = logging.getLogger(__name__)

DEFAULT_REPO = "Systran/faster-whisper-large-v3"


def _local_snapshot(repo: str = DEFAULT_REPO) -> Optional[str]:
    """Find a complete local snapshot so no network access is attempted.

    Checks the project's HF_HOME first, then the user cache, which is where the
    2.9GB CTranslate2 model already sits.
    """
    slug = "models--" + repo.replace("/", "--")
    roots = [
        os.getenv("HF_HOME", "/mnt/data/vqa-hf") + "/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    for root in roots:
        for snap in sorted(glob.glob(f"{root}/{slug}/snapshots/*")):
            if Path(snap, "model.bin").exists():
                return snap
    return None


class FastWhisper:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "en",
        beam_size: int = 5,
    ):
        # WHISPER_MODEL takes a Hub id or a local directory. A bare size such as
        # "large-v3" is not a repo, so it is mapped onto the CTranslate2 build;
        # faster-whisper cannot load openai/whisper-large-v3.
        env = (os.getenv("WHISPER_MODEL") or "").strip()
        if env and "/" not in env and not Path(env).exists():
            env = f"Systran/faster-whisper-{env}"
        # Prefer a complete local snapshot of whatever was asked for, so a bare
        # size or a Hub id still resolves offline when the weights are cached.
        self.model_path = (
            model_path
            or _local_snapshot(env or DEFAULT_REPO)
            or env
            or DEFAULT_REPO
        )
        self.device = device
        # int8_float16 keeps Whisper near 1.5GB so it can share the card, and
        # keeps a margin for the embedders if a run overlaps.
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_path, device=self.device, compute_type=self.compute_type
        )
        logger.info(
            "faster-whisper loaded from %s (%s, %s)",
            self.model_path, self.device, self.compute_type,
        )

    def close(self) -> None:
        import torch

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, video_path: Path) -> list[Segment]:
        """Transcribe one file into Segments, VAD-filtered.

        The VAD matters on this benchmark: many videos have long music or silence
        stretches where Whisper otherwise emits hallucinated repetitions, which
        would pollute the BM25 index with text nobody said.
        """
        if self.model is None:
            raise RuntimeError("FastWhisper.load() must be called first")

        segments, _info = self.model.transcribe(
            str(video_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # curbs repetition loops
        )

        out: list[Segment] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                out.append(
                    Segment(text=text, start=float(seg.start), end=float(seg.end))
                )
        return out

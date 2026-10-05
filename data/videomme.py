"""Video-MME-v2 annotations: MCQ questions and word-level subtitles.

Subtitles ship one word per JSONL line, which is useless for BM25, so they are
grouped into segments shaped exactly like Whisper's output. That lets
`ASR_SOURCE` flip between ground-truth subtitles and Whisper and makes ASR error
measurable as a config change rather than a guess.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# The dev/test split is by video, not by question: four questions share a video,
# so splitting on questions would leak evidence between the two sets.
DEV_VIDEO_IDS = frozenset(f"{i:03d}" for i in range(1, 51))

_OPTION_RE = re.compile(r"(?:^|\s)([A-H])\.\s*")


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]))


@dataclass(frozen=True)
class Question:
    video_id: str
    question_id: str
    question: str
    options: list[str]
    answer: str
    level: str
    group_type: str
    second_head: str
    third_head: str


def is_dev(video_id: str) -> bool:
    return str(video_id) in DEV_VIDEO_IDS


def parse_options(options_text: str) -> list[str]:
    """Split "A. Foo. B. Bar." into ["A. Foo.", "B. Bar."], keeping the labels.

    Labels are kept because the VLM is asked to reply with a letter, so the
    letter must appear next to its text in the prompt.
    """
    text = options_text or ""
    marks = list(_OPTION_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append(text[m.start():end].strip())
    return out


def load_questions(
    parquet_path: Path, video_ids: Optional[set[str]] = None
) -> list[Question]:
    """Load test.parquet. `video_ids` restricts to videos actually on disk."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if video_ids is not None:
        df = df[df["video_id"].isin(video_ids)]
    return [
        Question(
            video_id=str(r.video_id),
            question_id=str(r.question_id),
            question=str(r.question),
            options=parse_options(str(r.options)),
            answer=str(r.answer).strip().upper(),
            level=str(r.level),
            group_type=str(r.group_type),
            second_head=str(r.second_head),
            third_head=str(r.third_head),
        )
        for r in df.itertuples(index=False)
    ]


def group_words_into_segments(
    words: Iterable[dict], max_gap: float = 0.6, max_duration: float = 15.0
) -> list[Segment]:
    """Group word-level subtitle entries into sentence-ish segments.

    Breaks on a silence longer than `max_gap` between consecutive words, or when
    the running segment would exceed `max_duration`. The gap is measured word to
    word, not from the segment start, so steady speech stays in one segment.
    """
    segments: list[Segment] = []
    buf: list[str] = []
    seg_start = 0.0
    prev_end = 0.0

    def flush() -> None:
        if buf:
            segments.append(
                Segment(text=" ".join(buf).strip(), start=seg_start, end=prev_end)
            )

    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        start = float(word.get("start_time", 0.0))
        end = float(word.get("end_time", start))

        if not buf:
            buf, seg_start, prev_end = [text], start, end
            continue

        if (start - prev_end) > max_gap or (end - seg_start) > max_duration:
            flush()
            buf, seg_start, prev_end = [text], start, end
        else:
            buf.append(text)
            prev_end = end

    flush()
    return segments


def load_subtitle_segments(zip_path: Path, video_id: str, **kw) -> list[Segment]:
    """Read subtitle/<video_id>.jsonl out of subtitle.zip and group it."""
    with zipfile.ZipFile(zip_path) as zf:
        name = f"subtitle/{video_id}.jsonl"
        if name not in zf.namelist():
            return []
        raw = zf.read(name).decode("utf-8", "replace").splitlines()
    words = [json.loads(line) for line in raw if line.strip()]
    return group_words_into_segments(words, **kw)


def discover_video_ids(videos_dir: Path) -> list[str]:
    """Sorted ids of videos present on disk, e.g. ['001', '002', ...]."""
    return sorted(p.stem for p in Path(videos_dir).glob("*.mp4"))

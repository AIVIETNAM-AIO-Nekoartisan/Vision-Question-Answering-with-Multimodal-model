"""MCQ prompt assembly and answer parsing.

Two deliberate choices, both from spec section 7:

- Retrieved shots enter the prompt in timestamp order, not score order, so the
  VLM sees evidence in the sequence it happened. That is what Level 2 questions
  about ordering and temporal reasoning depend on.
- A global context layer (uniformly sampled frames plus a condensed transcript)
  always accompanies the retrieved shots, because Level 3 questions about plot
  and character cannot be answered from ten disconnected frames.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# \b on both sides so "Alice" does not read as "A"; allow a trailing . ) or :
_LETTER_RE = re.compile(r"\b([A-H])\b[.):]?")

GLOBAL_FRAMES = 8
TRANSCRIPT_CHARS = 1800


def parse_letter(text: Optional[str], n_options: int) -> tuple[Optional[str], bool]:
    """Extract the chosen option letter. Returns (letter, unparsed).

    `unparsed` is reported by the eval harness: if it is high, accuracy numbers
    are meaningless and the prompt needs fixing first.
    """
    limit = max(1, min(int(n_options or 8), 8))
    allowed = {chr(ord("A") + i) for i in range(limit)}
    for match in _LETTER_RE.finditer((text or "").upper()):
        if match.group(1) in allowed:
            return match.group(1), False
    return None, True


def _mmss(seconds: Optional[float]) -> str:
    s = float(seconds or 0.0)
    return f"[{int(s // 60):02d}:{int(s % 60):02d}]"


def condense_transcript(segments: list[dict], max_chars: int = TRANSCRIPT_CHARS) -> str:
    """Render the transcript inside a character budget, keeping time order.

    When it does not fit, segments are sampled evenly across the video rather
    than truncated at the front: a question about the ending must not lose the
    ending.
    """
    if not segments:
        return ""
    lines = [
        f"{_mmss(s.get('start'))} {str(s.get('text', '')).strip()}"
        for s in segments
        if str(s.get("text", "")).strip()
    ]
    if not lines:
        return ""

    joined = " ".join(lines)
    if len(joined) <= max_chars:
        return joined

    keep = max(1, max_chars // max(1, (len(joined) // len(lines)) + 1))
    step = max(1, len(lines) // keep)
    sampled = lines[::step]
    out = " ".join(sampled)
    return out[:max_chars]


def build_mcq_prompt(
    question: str,
    options: list[str],
    shots: list[dict],
    transcript_digest: str,
    n_global_frames: int,
) -> str:
    parts: list[str] = []

    if n_global_frames or transcript_digest:
        parts.append(
            f"The first {n_global_frames} images are frames sampled evenly across "
            "the whole video, for overall context."
        )
        if transcript_digest:
            parts.append(f"Transcript overview:\n{transcript_digest}")

    if shots:
        parts.append(
            "The remaining images are the moments most relevant to the question, "
            "in chronological order. Each labelled span gives several frames "
            "from before, during and after that moment:"
        )
        for shot in shots:
            # A window spans time; label the span so before/after is explicit.
            if shot.get("start_time") is not None and shot.get("end_time") is not None:
                line = f"{_mmss(shot['start_time'])}-{_mmss(shot['end_time'])}"
            else:
                line = f"{_mmss(shot.get('timestamp'))}"
            asr = str(shot.get("asr") or "").strip()
            ocr = str(shot.get("ocr") or "").strip()
            if asr:
                line += f" speech: {asr}"
            if ocr:
                line += f" on-screen text: {ocr}"
            parts.append(line)

    parts.append(f"Question: {question}")
    parts.append("Options:\n" + "\n".join(options))
    parts.append("Answer with the single letter only.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #


def _load_asr(data_root: Path, video_id: str) -> list[dict]:
    path = data_root / "asr" / f"{video_id}.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("segments", [])
    except (OSError, json.JSONDecodeError):
        return []


ASR_PAD_S = 1.5


def _load_ocr_by_shot(data_root: Path, video_id: str) -> dict[int, str]:
    """All on-screen text per shot, deduplicated, not just the chosen frame.

    Looking OCR up by the representative frame's file_path dropped text that
    appeared on any other keyframe of the same shot. With 55% of keyframes
    carrying text, most of a shot's text was being discarded.
    """
    path = data_root / "ocr" / f"{video_id}.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    per_shot: dict[int, list[str]] = {}
    for r in rows:
        text = str(r.get("text") or "").strip()
        num = r.get("shot_number")
        if not text or num is None:
            continue
        lines = per_shot.setdefault(int(num), [])
        # Watermarks repeat on nearly every frame of a shot; keep one copy.
        if text.lower() not in {x.lower() for x in lines}:
            lines.append(text)
    return {num: " | ".join(lines) for num, lines in per_shot.items()}


def _overlapping_speech(
    segments: list[dict], start: Optional[float], end: Optional[float],
    pad: float = ASR_PAD_S,
) -> str:
    """Speech overlapping a time span, not speech containing a single instant.

    The previous rule tested `segment.start <= frame_ts <= segment.end` against
    the representative frame's timestamp alone, so a line spoken at the start of
    a shot was lost whenever the chosen keyframe sat at the end of it. Interval
    overlap with a small pad keeps the line attached to the shot it belongs to.
    """
    if start is None and end is None:
        return ""
    lo = float(start if start is not None else end) - pad
    hi = float(end if end is not None else start) + pad
    spoken = [
        str(s.get("text", "")).strip()
        for s in segments
        if float(s.get("end", 0.0)) >= lo and float(s.get("start", 0.0)) <= hi
    ]
    return " ".join(x for x in spoken if x)


def _attach_text(
    shots: list[dict], segments: list[dict], ocr_by_shot: dict[int, str]
) -> list[dict]:
    """Attach the speech overlapping each shot and all of its on-screen text."""
    out = []
    for shot in shots:
        payload = shot.get("payload") or {}
        sh = payload.get("shot") or {}
        start = sh.get("start")
        end = sh.get("end")
        if start is None and end is None:
            start = end = shot.get("timestamp")
        num = sh.get("number")
        out.append(
            {
                **shot,
                "asr": _overlapping_speech(segments, start, end),
                "ocr": ocr_by_shot.get(int(num), "") if num is not None else "",
            }
        )
    return out


def answer_question(
    video_id: str,
    question: str,
    options: list[str],
    res: Any,
    weights: dict[str, float],
    top_k: int = 10,
    include_global_context: bool = True,
    expand: bool = False,
    windows: bool = True,
    n_windows: int = 3,
) -> dict:
    """Retrieve evidence, prompt the VLM, parse the letter."""
    from online.backend.retrieve import retrieve

    # The `full` config sets expand=True, but nothing here used to read it, so
    # `full` returned predictions identical to `+ocr` on all 200 dev questions —
    # 0 differences — and was not a measurement of query expansion at all.
    query = question
    if expand:
        from online.backend.expand import expand_query

        expanded = expand_query(question)
        if expanded:
            query = expanded

    shots = retrieve(
        query=query,
        ctx=res,
        video_id=video_id,
        weights=weights,
        top_k=top_k,
        build_windows=windows,
        n_windows=n_windows,
    )

    segments = _load_asr(res.data_root, video_id)
    ocr_by_shot = _load_ocr_by_shot(res.data_root, video_id)
    shots = _attach_text(shots, segments, ocr_by_shot)

    images: list[Path] = []
    digest = ""
    n_global = 0
    if include_global_context:
        from eval.baseline import uniform_frames

        try:
            images = uniform_frames(
                res.data_root / "videos" / f"{video_id}.mp4",
                n=GLOBAL_FRAMES,
                cache_root=res.data_root / "baseline_frames",
            )
            n_global = len(images)
        except Exception as exc:
            logger.warning("global context unavailable for %s: %s", video_id, exc)
        digest = condense_transcript(segments)

    for s in shots:
        frames = s.get("frames")
        if frames:                       # a temporal window: every frame, in order
            images += [res.data_root / f["file_path"] for f in frames
                       if f.get("file_path")]
        elif s.get("file_path"):         # a single shot
            images.append(res.data_root / s["file_path"])

    prompt = build_mcq_prompt(question, options, shots, digest, n_global)
    raw = res.qwen.answer(images, prompt)
    letter, unparsed = parse_letter(raw, len(options))

    return {
        "video_id": video_id,
        "answer": letter,
        "unparsed": unparsed,
        "shot_ids": [s["shot_id"] for s in shots],
        "raw": raw,
    }


def answer_request(req: Any, res: Any, env_weights: dict[str, float]) -> Any:
    """Serve a VQARequest, resolving question_id against test.parquet if given."""
    from data.videomme import load_questions
    from online.backend.schemas import VQAResponse

    weights = dict(req.weights or env_weights)
    gold = None

    if req.question_id:
        parquet = Path(
            os.getenv(
                "ANNOTATIONS_DIR", str(res.data_root / "annotations")
            )
        ) / "test.parquet"
        match = next(
            (q for q in load_questions(parquet) if q.question_id == req.question_id),
            None,
        )
        if match is None:
            raise LookupError(f"unknown question_id {req.question_id}")
        video_id, question, options, gold = (
            match.video_id,
            match.question,
            match.options,
            match.answer,
        )
    else:
        if not (req.video_id and req.question and req.options):
            raise LookupError("give question_id, or video_id + question + options")
        video_id, question, options = req.video_id, req.question, req.options

    result = answer_question(
        video_id=video_id,
        question=question,
        options=options,
        res=res,
        weights=weights,
        top_k=req.top_k,
        include_global_context=req.include_global_context,
        expand=getattr(req, "expand", False),
    )
    return VQAResponse(
        question_id=req.question_id,
        gold=gold,
        correct=(result["answer"] == gold) if gold else None,
        **result,
    )

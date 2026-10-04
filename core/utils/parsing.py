# utils/parsing.py
# Pure helpers dùng chung cho api_server và vector_store (stdlib-only).
import os
import re
from typing import Any, Dict, List, Mapping, Optional

_FRAME_RE = re.compile(r"__frame_(\d+)")
# Accepts AIC names (L01_V001) and Video-MME-v2 names (001..800). The original
# pattern matched only the former, which made metadata_needs_repair() return
# True for every Video-MME result.
_VIDEO_NAME_RE = re.compile(r"^(?:[A-Z]\d+_V\d+|\d{3})$", re.IGNORECASE)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def parse_hms_to_seconds(s: str) -> Optional[float]:
    """Parse 'HH:MM:SS[.fff]' hoặc 'MM:SS[.fff]' -> seconds. Trả None nếu không parse được."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60 + sec
    except (AttributeError, ValueError):
        pass
    return None


def extract_frame_id_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    """Lấy frame index từ payload: ưu tiên frame.index, fallback parse '__frame_000123' trong filename/file_path."""
    payload = payload or {}
    frame = payload.get("frame") or {}
    if isinstance(frame, dict) and frame.get("index") is not None:
        try:
            return int(frame["index"])
        except (TypeError, ValueError):
            pass
    for key in ("filename", "file_path"):
        m = _FRAME_RE.search(payload.get(key) or "")
        if m:
            return int(m.group(1))
    return None


def infer_video_name_from_path(file_path: str) -> Optional[str]:
    """Infer an AIC video name from either a UUID path or a legacy keyframe name."""
    path = str(file_path or "").strip()
    if not path:
        return None

    parent = os.path.basename(os.path.dirname(path))
    if _VIDEO_NAME_RE.fullmatch(parent):
        return parent

    stem = os.path.splitext(os.path.basename(path))[0]
    legacy = re.match(r"^([A-Z]\d+_V\d+)(?:__|$)", stem, re.IGNORECASE)
    return legacy.group(1) if legacy else None


def keyframe_id_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the UUID used by the canonical AIC_data point for a keyframe."""
    payload = payload or {}
    for value in (payload.get("filename"), payload.get("file_path")):
        stem = os.path.splitext(os.path.basename(str(value or "")))[0]
        if _UUID_RE.fullmatch(stem):
            return stem.lower()
    return None


def is_known_video_name(name: Optional[str]) -> bool:
    """Whether `name` is a recognised video identifier in either naming scheme."""
    return bool(name) and bool(_VIDEO_NAME_RE.fullmatch(str(name)))


def metadata_needs_repair(payload: Optional[Dict[str, Any]]) -> bool:
    """Whether a result lacks a usable video name or frame index."""
    payload = payload or {}
    video = payload.get("video") or {}
    name = video.get("name") if isinstance(video, dict) else None
    return not is_known_video_name(name) or extract_frame_id_from_payload(payload) is None


def repair_result_metadata(
    results: List[Dict[str, Any]],
    canonical_payloads: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace broken caption metadata with payloads from canonical AIC_data points."""
    repaired = []
    for result in results:
        source_payload = result.get("payload") or {}
        point_id = keyframe_id_from_payload(source_payload)
        canonical = canonical_payloads.get(point_id or "")
        if not canonical or not metadata_needs_repair(source_payload):
            repaired.append(result)
            continue

        payload = {**source_payload, **canonical}
        if source_payload.get("caption"):
            payload["caption"] = source_payload["caption"]
        item = {**result, "payload": payload}
        if payload.get("file_path"):
            item["file_path"] = payload["file_path"]
        frame_id = extract_frame_id_from_payload(payload)
        if frame_id is not None:
            item["frame_id"] = frame_id
        repaired.append(item)
    return repaired

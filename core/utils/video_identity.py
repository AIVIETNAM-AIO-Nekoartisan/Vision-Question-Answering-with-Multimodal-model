from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

# Pattern to remove keyframe/shot metadata suffixes from file/video names
# e.g. L01_V001__shot_001__pos_middle__frame_00010__ts_0.500
SUFFIX_PATTERN = re.compile(r"__(?:shot|pos|frame|ts)_.*$", re.IGNORECASE)

KNOWN_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".webm",
    ".mov",
    ".flv",
    ".wmv",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".json",
}


def _clean_candidate(raw: Any) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() == "unknown":
        return None

    # Strip directory components if it's a file path
    base = os.path.basename(s)

    # Strip recognized media extensions
    root, ext = os.path.splitext(base)
    while ext.lower() in KNOWN_EXTENSIONS:
        base = root
        root, ext = os.path.splitext(base)

    # Remove shot/frame/ts suffixes
    cleaned = SUFFIX_PATTERN.sub("", base).strip()
    if not cleaned or cleaned.lower() == "unknown":
        return None
    return cleaned


def canonical_video_id(
    value: Optional[str] = None,
    *,
    document: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Derives canonical video_id from value or heterogeneous document fields.

    Precedence order:
    1. document["video_id"]
    2. document["video_name"] or document["video"]["name"]
    3. document["file_name"]
    4. document["file_path"]
    5. raw value parameter
    """
    if document and isinstance(document, dict):
        # 1. video_id
        if "video_id" in document:
            cleaned = _clean_candidate(document["video_id"])
            if cleaned:
                return cleaned

        # 2. video_name / video.name
        if "video_name" in document:
            cleaned = _clean_candidate(document["video_name"])
            if cleaned:
                return cleaned
        video_obj = document.get("video")
        if isinstance(video_obj, dict):
            cleaned = _clean_candidate(video_obj.get("name") or video_obj.get("id"))
            if cleaned:
                return cleaned

        # 3. file_name
        if "file_name" in document:
            cleaned = _clean_candidate(document["file_name"])
            if cleaned:
                return cleaned

        # 4. file_path
        if "file_path" in document:
            cleaned = _clean_candidate(document["file_path"])
            if cleaned:
                return cleaned

    # 5. raw value
    if value is not None:
        return _clean_candidate(value)

    return None

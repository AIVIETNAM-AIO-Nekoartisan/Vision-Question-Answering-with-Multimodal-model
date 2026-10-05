"""SQLite record of which (video, stage) pairs are finished.

Indexing 200 videos through six stages takes hours on one GPU; without this a
crash on video 180 would redo the 179 before it. The table name and columns are
also read by scripts/watch_download_and_index.sh, so they are fixed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STAGES = ("shots", "keyframes", "embed", "asr", "ocr", "index")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_state (
    video_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (video_id, stage)
);
"""


class Manifest:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        # The watcher reads this file while the pipeline writes it.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def mark(
        self, video_id: str, stage: str, status: str, error: Optional[str] = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO stage_state (video_id, stage, status, error, updated_at) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(video_id, stage) DO UPDATE SET "
            "status=excluded.status, error=excluded.error, updated_at=excluded.updated_at",
            (
                str(video_id),
                stage,
                status,
                error,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()

    def status(self, video_id: str, stage: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM stage_state WHERE video_id=? AND stage=?",
            (str(video_id), stage),
        ).fetchone()
        return row[0] if row else None

    def error(self, video_id: str, stage: str) -> str:
        row = self._conn.execute(
            "SELECT error FROM stage_state WHERE video_id=? AND stage=?",
            (str(video_id), stage),
        ).fetchone()
        return (row[0] if row else None) or ""

    def is_done(self, video_id: str, stage: str) -> bool:
        return self.status(video_id, stage) == "done"

    def pending(self, video_ids: list[str], stage: str) -> list[str]:
        """Ids from `video_ids`, in the given order, that are not yet done."""
        done = {
            r[0]
            for r in self._conn.execute(
                "SELECT video_id FROM stage_state WHERE stage=? AND status='done'",
                (stage,),
            )
        }
        return [v for v in video_ids if v not in done]

    def count_done(self, stage: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM stage_state WHERE stage=? AND status='done'", (stage,)
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()

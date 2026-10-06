"""Stage orchestrator. Run from the repo root:

    python -m offline.run --stages all
    python -m offline.run --stages shots,keyframes --limit 2
    python -m offline.run --stages ocr --dev-only
    python -m offline.run --stages index --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from data.videomme import is_dev
from offline import stages as S
from offline.manifest import STAGES, Manifest

logger = logging.getLogger("offline.run")

_FUNCS = {
    "shots": S.stage_shots,
    "keyframes": S.stage_keyframes,
    "embed": S.stage_embed,
    "asr": S.stage_asr,
    "ocr": S.stage_ocr,
    "index": S.stage_index,
    "index_text": S.stage_index_text,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run offline indexing stages.")
    ap.add_argument(
        "--stages", default="all", help="'all' or comma-separated stage names"
    )
    ap.add_argument("--limit", type=int, default=None, help="at most N videos")
    ap.add_argument("--videos", default=None, help="comma-separated video ids")
    ap.add_argument(
        "--dev-only",
        action="store_true",
        help="restrict to the dev split (videos 001-050)",
    )
    ap.add_argument(
        "--force", action="store_true", help="redo videos already marked done"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    requested = (
        STAGES
        if args.stages == "all"
        else tuple(s.strip() for s in args.stages.split(",") if s.strip())
    )
    unknown = [s for s in requested if s not in _FUNCS]
    if unknown:
        ap.error(f"unknown stage(s): {', '.join(unknown)}")

    ctx = S.Ctx.from_env()
    manifest = Manifest(ctx.data_root / "manifest.db")
    failures = 0
    try:
        if args.videos:
            all_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
        else:
            all_ids = S.discover_video_ids_ctx(ctx)
        if args.dev_only:
            all_ids = [v for v in all_ids if is_dev(v)]
        logger.info("%d video(s) in scope", len(all_ids))

        for stage in requested:
            todo = all_ids if args.force else manifest.pending(all_ids, stage)
            if args.limit:
                todo = todo[: args.limit]
            logger.info("stage %s: %d video(s) to do", stage, len(todo))
            for n, vid in enumerate(todo, start=1):
                try:
                    _FUNCS[stage](vid, ctx)
                    manifest.mark(vid, stage, "done")
                    logger.info("[%s] %s ok (%d/%d)", stage, vid, n, len(todo))
                except Exception as exc:
                    failures += 1
                    manifest.mark(
                        vid, stage, "failed", error=traceback.format_exc(limit=6)
                    )
                    logger.error("[%s] %s FAILED: %s", stage, vid, exc)
    finally:
        manifest.close()
        ctx.close()

    logger.info("done, %d failure(s)", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

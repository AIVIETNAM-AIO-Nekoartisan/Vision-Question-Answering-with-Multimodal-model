"""Evaluation harness.

    python -m eval.run_eval --config visual-only --split dev

--split defaults to dev on purpose. The test split gets one run per config;
defaulting to test would quietly contaminate it across iterations of tuning.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from data.videomme import Question, is_dev, load_questions
from eval.configs import CONFIGS, get_config

logger = logging.getLogger("eval")


def score_group(
    questions: list[Question], predictions: dict[str, Optional[str]], group_type: str
) -> float:
    """Score one group of questions about one video.

    `relevance` groups score each question independently. `logic` groups use
    first-error truncation: once an answer is wrong, the rest of the chain
    scores zero regardless of what was predicted, because the later steps were
    reasoned from a broken premise.
    """
    if not questions:
        return 0.0
    ordered = sorted(questions, key=lambda q: q.question_id)
    total = len(ordered)

    if group_type == "logic":
        credited = 0
        for q in ordered:
            if predictions.get(q.question_id) == q.answer:
                credited += 1
            else:
                break
        return credited / total

    return sum(predictions.get(q.question_id) == q.answer for q in ordered) / total


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(
    questions: list[Question], predictions: dict[str, Optional[str]]
) -> dict:
    """Overall accuracy plus the breakdowns the research question needs."""
    n = len(questions)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "unparsed_rate": 0.0,
            "by_level": {},
            "by_second_head": {},
            "group_score": 0.0,
        }

    correct = [predictions.get(q.question_id) == q.answer for q in questions]
    unparsed = [predictions.get(q.question_id) is None for q in questions]

    by_level: dict[str, list[float]] = defaultdict(list)
    by_head: dict[str, list[float]] = defaultdict(list)
    for q, ok in zip(questions, correct):
        by_level[q.level].append(float(ok))
        by_head[q.second_head].append(float(ok))

    # Groups are (video, group_type) pairs: the four questions of a video.
    groups: dict[tuple[str, str], list[Question]] = defaultdict(list)
    for q in questions:
        groups[(q.video_id, q.group_type)].append(q)
    group_scores = [
        score_group(qs, predictions, gtype) for (_, gtype), qs in groups.items()
    ]

    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "unparsed_rate": sum(unparsed) / n,
        "by_level": {k: _mean(v) for k, v in sorted(by_level.items())},
        "by_second_head": {k: _mean(v) for k, v in sorted(by_head.items())},
        "group_score": _mean(group_scores),
        "n_groups": len(groups),
    }


def render_table(name: str, split: str, summary: dict) -> str:
    lines = [
        f"### {name} ({split})",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| questions (n) | {summary['n']} |",
        f"| accuracy | {summary['accuracy']:.3f} |",
        f"| group score | {summary['group_score']:.3f} |",
        f"| unparsed rate | {summary['unparsed_rate']:.3f} |",
        "",
        "| level | accuracy |",
        "|---|---|",
    ]
    for level, acc in summary["by_level"].items():
        lines.append(f"| {level} | {acc:.3f} |")
    lines += ["", "| group | accuracy |", "|---|---|"]
    for head, acc in summary["by_second_head"].items():
        lines.append(f"| {head} | {acc:.3f} |")
    return "\n".join(lines)


def _predict_all(
    questions: list[Question], cfg: dict, res, top_k: int
) -> tuple[dict[str, Optional[str]], list[dict]]:
    from online.backend.vqa import answer_question

    predictions: dict[str, Optional[str]] = {}
    details = []
    started = time.time()

    for i, q in enumerate(questions, start=1):
        try:
            out = answer_question(
                video_id=q.video_id,
                question=q.question,
                options=q.options,
                res=res,
                weights=cfg["weights"],
                top_k=top_k if cfg["retrieval"] else 0,
                include_global_context=cfg["global_context"],
            )
            predictions[q.question_id] = out["answer"]
            details.append(
                {
                    "question_id": q.question_id,
                    "video_id": q.video_id,
                    "gold": q.answer,
                    "pred": out["answer"],
                    "unparsed": out["unparsed"],
                    "level": q.level,
                    "shot_ids": out["shot_ids"],
                    "raw": out["raw"],
                }
            )
        except Exception as exc:
            logger.error("%s failed: %s", q.question_id, exc)
            predictions[q.question_id] = None
            details.append(
                {"question_id": q.question_id, "error": str(exc), "pred": None}
            )

        if i % 25 == 0 or i == len(questions):
            rate = i / max(1e-9, time.time() - started)
            logger.info(
                "%d/%d answered (%.2f q/s, %.0fs elapsed)",
                i, len(questions), rate, time.time() - started,
            )
    return predictions, details


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a config on Video-MME-v2.")
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "10")))
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = get_config(args.config)

    from online.backend.resources import resources

    resources.initialize_datastores()
    resources.initialize_models()
    if resources.qwen is None:
        logger.error(
            "VLM unavailable (%s); cannot evaluate",
            resources.errors.get("qwen", "not loaded"),
        )
        return 2

    videos = set(resources.all_video_ids())
    annotations = Path(
        os.getenv("ANNOTATIONS_DIR", str(resources.data_root / "annotations"))
    )
    questions = load_questions(annotations / "test.parquet", videos)
    if args.split == "dev":
        questions = [q for q in questions if is_dev(q.video_id)]
    elif args.split == "test":
        questions = [q for q in questions if not is_dev(q.video_id)]
    questions.sort(key=lambda q: q.question_id)
    if args.limit:
        questions = questions[: args.limit]

    logger.info(
        "config=%s split=%s questions=%d videos_on_disk=%d",
        args.config, args.split, len(questions), len(videos),
    )
    if not questions:
        logger.error("no questions in scope")
        return 2

    predictions, details = _predict_all(questions, cfg, resources, args.top_k)
    summary = summarise(questions, predictions)
    summary.update(
        {"config": args.config, "split": args.split, "top_k": args.top_k,
         "weights": cfg["weights"], "retrieval": cfg["retrieval"]}
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.config}-{args.split}.json"
    out_path.write_text(json.dumps({"summary": summary, "details": details}, indent=2))

    print()
    print(render_table(args.config, args.split, summary))
    print()
    print(f"written to {out_path}")
    if summary["unparsed_rate"] > 0.05:
        print(
            f"\nWARNING: unparsed rate {summary['unparsed_rate']:.1%} exceeds 5%. "
            "Fix the prompt before reading the accuracy number."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

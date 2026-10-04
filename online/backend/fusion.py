"""RRF / SRRF fusion for multi-source results (siglip, jina, asr, ocr).

enrich_results_metadata was dropped with the caption collection: it hydrated
broken caption payloads from the canonical AIC_data collection, which this
project does not build.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def rrf_weighted_fuse(results_by_model: Dict[str, List], k: int = 60, weights: Optional[Dict[str, float]] = None, topn: int = 300):
    """
    results_by_model: {"siglip":[qdrant_points...], "beit3":[qdrant_points...]}
    Trả về danh sách hợp nhất theo RRF (có trọng số).
    """
    weights = weights or {}
    fused = {}
    best_src = {}
    components_by_path = defaultdict(set)

    for model_name, res_list in results_by_model.items():
        w = float(weights.get(model_name, 1.0))
        # sort desc theo score (càng cao càng tốt)
        sorted_list = sorted(res_list, key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(sorted_list, start=1):
            path = (r.payload or {}).get("file_path")
            if not path:
                continue
            components_by_path[path].add(model_name)
            score_add = w * (1.0 / (k + rank))
            fused[path] = fused.get(path, 0.0) + score_add
            # lưu best payload từ nguồn nào cũng được (ưu tiên điểm cao)
            if (path not in best_src) or (r.score > best_src[path].score):
                best_src[path] = r

    # sort kết quả theo fused score
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topn]
    out = []
    for path, fscore in ordered:
        r = best_src[path]
        out.append({
            "point_id": getattr(r, "id", None),
            "file_path": path,
            "score": float(fscore),          # RRF score
            "payload": r.payload,
            "rrf_score": float(fscore),
            "rrf_components": sorted(list(components_by_path.get(path, set()))),
        })
    return out

def srrf_weighted_fuse(
    results_by_model: Dict[str, List],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    topn: int = 300,
    beta: float = 0.5
):
    """
    Kết hợp kết quả bằng Score-Reflected RRF (SRRF) có trọng số. (PHIÊN BẢN CUỐI CÙNG)

    Hàm này tránh việc thay đổi đối tượng ScoredPoint gốc bằng cách lưu điểm
    chuẩn hóa trong một tuple (point, normalized_score).
    """
    weights = weights or {}
    fused_scores = {}
    best_src = {}
    components_by_path = defaultdict(set)

    # normalized_results giờ sẽ lưu một list các tuple: [(point1, norm_score1), (point2, norm_score2), ...]
    normalized_results = defaultdict(list)
    for model_name, res_list in results_by_model.items():
        if not res_list:
            continue

        scores = [r.score for r in res_list]
        min_score, max_score = min(scores), max(scores)
        score_range = max(max_score - min_score, 1e-9)

        for r in res_list:
            normalized_score = (r.score - min_score) / score_range
            # Lưu đối tượng và điểm chuẩn hóa vào một tuple
            normalized_results[model_name].append((r, normalized_score))

    for model_name, res_list_with_scores in normalized_results.items():
        w = float(weights.get(model_name, 1.0))

        # Sắp xếp list các tuple dựa trên score của đối tượng gốc (item[0])
        sorted_list = sorted(res_list_with_scores, key=lambda item: item[0].score, reverse=True)

        for rank, (r, normalized_score) in enumerate(sorted_list, start=1):
            path = (r.payload or {}).get("file_path")
            if not path:
                continue

            components_by_path[path].add(model_name)
            rank_score_part = 1.0 / (k + rank)
            norm_score_part = normalized_score
            srrf_score = (1 - beta) * rank_score_part + beta * norm_score_part

            fused_scores[path] = fused_scores.get(path, 0.0) + (w * srrf_score)

            if (path not in best_src) or (r.score > best_src[path].score):
                best_src[path] = r

    if not fused_scores:
        return []

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    out = []
    for path, fscore in ordered:
        r = best_src[path]
        out.append({
            "point_id": getattr(r, "id", None),
            "file_path": path,
            "score": float(fscore),
            "payload": r.payload,
            "srrf_score": float(fscore),
            "original_score": r.score,
            "rrf_components": sorted(list(components_by_path.get(path, set()))),
        })
    return out

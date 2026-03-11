"""통합 스코어러: 헝가리안 매칭 + NED 기반 채점."""
from __future__ import annotations

from typing import Any

from .matcher import LeafPair, flatten_to_pairs
from .metrics import compare_leaf
from .schema_traversal import unwrap_root


def score_result(
    extracted: dict | None,
    ground_truth: dict,
    schema: dict,
) -> dict:
    """구조화 추출 결과를 GT와 비교하여 점수 산출.

    Args:
        extracted: LLM이 추출한 dict (None이면 0점)
        ground_truth: 정답 dict
        schema: JSON Schema dict

    Returns:
        {"total": float, "max": float, "pct": float, "field_scores": dict}
    """
    if extracted is None:
        leaf_count = _count_leaves(ground_truth)
        return {
            "total": 0.0,
            "max": float(max(leaf_count, 1)),
            "pct": 0.0,
            "field_scores": {},
        }

    root_schema = unwrap_root(schema)

    # 1. GT와 Predicted를 리프 페어로 분해 (gt, pred 순서)
    pairs = flatten_to_pairs(ground_truth, extracted, root_schema, root_schema)

    if not pairs:
        # GT와 extracted 모두 비어있지 않은데 pairs가 없으면 0점
        if extracted and ground_truth:
            return {"total": 0.0, "max": 1.0, "pct": 0.0, "field_scores": {}}
        return {"total": 0.0, "max": 0.0, "pct": 100.0, "field_scores": {}}

    # 2. 각 리프 페어에 타입 기반 메트릭 적용
    field_scores: dict[str, float] = {}
    total = 0.0

    for pair in pairs:
        score = compare_leaf(pair.actual, pair.predicted, pair.field_type)
        field_scores[pair.path] = round(score, 4)
        total += score

    max_score = float(len(pairs))
    pct = round((total / max_score) * 100, 1) if max_score > 0 else 100.0

    return {
        "total": round(total, 2),
        "max": max_score,
        "pct": pct,
        "field_scores": field_scores,
    }


def _count_leaves(data: Any, _root: bool = True) -> int:
    """dict/list 내 리프 필드 수. _root=True일 때만 최소 1 보장."""
    if isinstance(data, dict):
        count = sum(_count_leaves(v, False) for v in data.values())
        return max(count, 1) if _root else count
    if isinstance(data, list):
        count = sum(_count_leaves(v, False) for v in data)
        return max(count, 1) if _root else count
    return 1

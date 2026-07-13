"""통합 스코어러: 헝가리안 매칭 + NED 기반 채점."""
from __future__ import annotations

from .matcher import flatten_to_pairs
from .metrics import compare_leaf, compare_leaf_exact
from .schema_traversal import unwrap_root


def score_result(
    extracted: dict | None,
    ground_truth: dict,
    schema: dict,
) -> dict:
    """구조화 추출 결과를 GT와 비교하여 점수 산출.

    extracted가 None/빈 값이어도 flatten_to_pairs가 GT 리프를 모두 (gt, None) 페어로
    만들어 정확히 0점 처리하므로 별도 조기 반환은 두지 않는다(양쪽 모두 비어있을 때만 100%).

    Returns:
        {"total", "max", "pct" (NED 기반), "pct_exact" (엄격 일치 기반), "field_scores"}
    """
    root_schema = unwrap_root(schema)

    # GT와 Predicted를 리프 페어로 분해 (gt, pred 순서)
    pairs = flatten_to_pairs(ground_truth, extracted, root_schema, root_schema)

    if not pairs:
        # 리프 페어가 없다는 것은 비교 가능한 값이 없다는 뜻(양쪽 빈 구조 또는 전부 null-null
        # 일치)이며, 어느 쪽도 이의가 없으므로 완전일치 100점이다. GT에 값이 있는데 예측이
        # 비었거나 그 반대면 flatten이 (값, None) 페어를 만들어 여기까지 오지 않는다.
        return {"total": 0.0, "max": 0.0, "pct": 100.0, "pct_exact": 100.0, "field_scores": {}}

    # 각 리프 페어에 타입 기반 메트릭 적용 (NED + 엄격 일치 동시 산출)
    field_scores: dict[str, float] = {}
    total = 0.0
    total_exact = 0.0

    for pair in pairs:
        score = compare_leaf(pair.actual, pair.predicted, pair.field_type)
        field_scores[pair.path] = round(score, 4)
        total += score
        total_exact += compare_leaf_exact(pair.actual, pair.predicted, pair.field_type)

    max_score = float(len(pairs))
    pct = round((total / max_score) * 100, 1) if max_score > 0 else 100.0
    pct_exact = round((total_exact / max_score) * 100, 1) if max_score > 0 else 100.0

    return {
        "total": round(total, 2),
        "max": max_score,
        "pct": pct,
        "pct_exact": pct_exact,
        "field_scores": field_scores,
    }

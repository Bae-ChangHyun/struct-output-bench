"""ExtractBench 채점 함수. evaluation_config를 참고한 필드별 정확도 평가."""
from __future__ import annotations

from difflib import SequenceMatcher


def _normalize_str(s: str) -> str:
    return s.strip().lower()


def _compare_string_exact(extracted: str | None, gold: str | None) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    return 1.0 if str(extracted).strip() == str(gold).strip() else 0.0


def _compare_string_case_insensitive(extracted: str | None, gold: str | None) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    return 1.0 if _normalize_str(str(extracted)) == _normalize_str(str(gold)) else 0.0


def _compare_string_fuzzy(extracted: str | None, gold: str | None) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    ratio = SequenceMatcher(None, _normalize_str(str(extracted)), _normalize_str(str(gold))).ratio()
    return ratio


def _compare_string_semantic(extracted: str | None, gold: str | None) -> float:
    """Semantic 비교 - LLM 없이 fuzzy match로 대체."""
    return _compare_string_fuzzy(extracted, gold)


def _compare_number_exact(extracted, gold) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    try:
        return 1.0 if float(extracted) == float(gold) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compare_number_tolerance(extracted, gold, tolerance: float = 0.05) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    try:
        e, g = float(extracted), float(gold)
        if g == 0:
            return 1.0 if e == 0 else 0.0
        return 1.0 if abs(e - g) / abs(g) <= tolerance else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compare_integer_exact(extracted, gold) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    try:
        return 1.0 if int(extracted) == int(gold) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compare_boolean_exact(extracted, gold) -> float:
    if extracted is None and gold is None:
        return 1.0
    if extracted is None or gold is None:
        return 0.0
    return 1.0 if bool(extracted) == bool(gold) else 0.0


def _compare_array(extracted_arr, gold_arr, item_schema: dict | None = None) -> float:
    """배열 비교: 개수 + 내용 매칭."""
    if not isinstance(extracted_arr, list):
        extracted_arr = []
    if not isinstance(gold_arr, list):
        gold_arr = []
    if not gold_arr and not extracted_arr:
        return 1.0
    if not gold_arr or not extracted_arr:
        return 0.0

    # 개수 점수 (50%)
    count_score = min(len(extracted_arr), len(gold_arr)) / max(len(extracted_arr), len(gold_arr))

    # 내용 점수 (50%): 각 gold 항목에 대해 best match 찾기
    content_scores = []
    for gold_item in gold_arr:
        best = 0.0
        for ext_item in extracted_arr:
            s = _compare_values(ext_item, gold_item, item_schema)
            if s > best:
                best = s
        content_scores.append(best)

    content_score = sum(content_scores) / len(content_scores) if content_scores else 0.0

    return 0.5 * count_score + 0.5 * content_score


def _compare_values(extracted, gold, schema: dict | None = None) -> float:
    """두 값을 비교. schema가 있으면 evaluation_config 참고."""
    if schema is None:
        schema = {}

    eval_config = schema.get("evaluation_config", "")

    # dict 형태의 evaluation_config 처리 (number_tolerance 등)
    if isinstance(eval_config, dict):
        metrics = eval_config.get("metrics", [])
        if metrics:
            metric = metrics[0]
            metric_id = metric.get("metric_id", "")
            params = metric.get("params", {})
            if metric_id == "number_tolerance":
                return _compare_number_tolerance(
                    extracted, gold, params.get("tolerance", 0.05)
                )
        eval_config = ""

    # 문자열 evaluation_config 처리
    if eval_config == "string_exact":
        return _compare_string_exact(extracted, gold)
    if eval_config == "string_case_insensitive":
        return _compare_string_case_insensitive(extracted, gold)
    if eval_config == "string_fuzzy":
        return _compare_string_fuzzy(extracted, gold)
    if eval_config == "string_semantic":
        return _compare_string_semantic(extracted, gold)
    if eval_config == "number_exact":
        return _compare_number_exact(extracted, gold)
    if eval_config == "integer_exact":
        return _compare_integer_exact(extracted, gold)
    if eval_config == "boolean_exact":
        return _compare_boolean_exact(extracted, gold)
    if eval_config in ("array_llm", "array_exact"):
        items_schema = schema.get("items")
        return _compare_array(extracted, gold, items_schema)

    # evaluation_config 없는 경우: 타입에 따라 자동 판단
    if isinstance(gold, bool):
        return _compare_boolean_exact(extracted, gold)
    if isinstance(gold, (int, float)):
        return _compare_number_tolerance(extracted, gold, 0.05)
    if isinstance(gold, str):
        return _compare_string_fuzzy(extracted, gold)
    if isinstance(gold, list):
        return _compare_array(extracted, gold)
    if isinstance(gold, dict):
        return _score_object(extracted, gold, schema)

    return 1.0 if extracted == gold else 0.0


def _score_object(extracted: dict | None, gold: dict, schema: dict) -> float:
    """object 비교: 재귀적으로 각 필드 비교."""
    if not isinstance(extracted, dict):
        return 0.0

    props_schema = schema.get("properties", {})
    # schema_definition 래퍼 처리
    if "schema_definition" in schema:
        props_schema = schema["schema_definition"].get("properties", {})

    scores = []
    for key, gold_val in gold.items():
        if key == "evaluation_config":
            continue
        ext_val = extracted.get(key)
        field_schema = props_schema.get(key, {})
        scores.append(_compare_values(ext_val, gold_val, field_schema))

    return sum(scores) / len(scores) if scores else 1.0


def score_result(
    extracted: dict | None,
    ground_truth: dict,
    schema_dict: dict,
) -> dict:
    """100점 만점. 필드별 정확도를 재귀적으로 평가.

    Returns:
        {"total": float, "max": 100, "pct": float, "field_scores": {field: score}}
    """
    if not extracted:
        return {"total": 0, "max": 100, "pct": 0.0, "field_scores": {}}

    # schema_definition 래퍼 처리
    if "schema_definition" in schema_dict:
        root_schema = schema_dict["schema_definition"]
    else:
        root_schema = schema_dict

    props_schema = root_schema.get("properties", {})
    field_scores: dict[str, float] = {}
    total_fields = 0

    for key, gold_val in ground_truth.items():
        if key == "evaluation_config":
            continue
        total_fields += 1
        ext_val = extracted.get(key)
        field_schema = props_schema.get(key, {})
        field_scores[key] = _compare_values(ext_val, gold_val, field_schema)

    if total_fields == 0:
        return {"total": 100, "max": 100, "pct": 100.0, "field_scores": {}}

    avg = sum(field_scores.values()) / total_fields
    pct = round(avg * 100, 1)

    return {
        "total": round(avg * 100, 1),
        "max": 100,
        "pct": pct,
        "field_scores": field_scores,
    }

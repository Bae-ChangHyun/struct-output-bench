"""JSON 구조 매칭: GT와 Predicted를 헝가리안 알고리즘으로 정렬하여 리프 페어 생성."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hungarian import max_weight_matching
from .metrics import _levenshtein
from .schema_traversal import (
    get_field_type,
    get_items_schema,
    get_properties,
    infer_type,
    resolve_schema,
)


@dataclass
class LeafPair:
    path: str
    actual: Any
    predicted: Any
    field_type: str


def _stable_str(obj: Any) -> str:
    """값을 정렬 가능한 문자열로 변환."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        parts = []
        for k in sorted(obj.keys()):
            parts.append(f"{k}:{_stable_str(obj[k])}")
        return "{" + ",".join(parts) + "}"
    if isinstance(obj, list):
        return "[" + ",".join(_stable_str(x) for x in obj) + "]"
    return str(obj)


def _extract_fields(obj: Any) -> dict[str, str]:
    """객체의 모든 필드를 key→string 형태로 추출 (비어있지 않은 값만)."""
    if not isinstance(obj, dict):
        return {}
    result = {}
    for k, v in obj.items():
        s = _stable_str(v)
        if s:
            result[k] = s
    return result


def _match_object_array(
    arr_gt: list, arr_pred: list, path: str, items_schema: dict, root_schema: dict | None
) -> list[LeafPair]:
    """객체 배열을 헝가리안 매칭으로 정렬 후 리프 페어 생성."""
    if not arr_gt and not arr_pred:
        return []

    n, m = len(arr_gt), len(arr_pred)

    # score matrix 구축: 공통 필드 수 + 공통 값 길이 - Levenshtein 페널티
    fields_gt = [_extract_fields(g) for g in arr_gt]
    fields_pred = [_extract_fields(p) for p in arr_pred]

    score_matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            fg, fp = fields_gt[i], fields_pred[j]
            common_keys = set(fg.keys()) & set(fp.keys())
            common_count = sum(1 for k in common_keys if fg[k] == fp[k])
            common_len = sum(len(fg[k]) for k in common_keys if fg[k] == fp[k])

            str_i = _stable_str(arr_gt[i])
            str_j = _stable_str(arr_pred[j])
            lev = _levenshtein(str_i, str_j)

            score_matrix[i][j] = (common_count * 1000.0) + (common_len * 1.0) + (-lev * 0.1)

    matched = max_weight_matching(score_matrix) if n > 0 and m > 0 else []
    matched_gt = {i for i, _ in matched}
    matched_pred = {j for _, j in matched}

    pairs: list[LeafPair] = []

    # 매칭된 쌍: 재귀 비교
    for i, j in matched:
        child_path = f"{path}[{i}]"
        pairs.extend(
            flatten_to_pairs(arr_gt[i], arr_pred[j], items_schema, root_schema, child_path)
        )

    # GT에만 있는 항목
    for i in range(n):
        if i not in matched_gt:
            child_path = f"{path}[{i}]"
            pairs.extend(
                flatten_to_pairs(arr_gt[i], None, items_schema, root_schema, child_path)
            )

    # Predicted에만 있는 항목 (GT=None)
    for j in range(m):
        if j not in matched_pred:
            child_path = f"{path}[{n + j}]"
            pairs.extend(
                flatten_to_pairs(None, arr_pred[j], items_schema, root_schema, child_path)
            )

    return pairs


def _match_primitive_array(
    arr_gt: list, arr_pred: list, path: str, field_type: str
) -> list[LeafPair]:
    """원시 타입 배열을 헝가리안 매칭으로 정렬 후 리프 페어 생성."""
    if not arr_gt and not arr_pred:
        return []

    n, m = len(arr_gt), len(arr_pred)

    # score matrix: NED 기반
    score_matrix = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            si = str(arr_gt[i])
            sj = str(arr_pred[j])
            max_len = max(len(si), len(sj), 1)
            dist = _levenshtein(si, sj)
            score_matrix[i][j] = 1.0 - (dist / max_len)

    matched = max_weight_matching(score_matrix) if n > 0 and m > 0 else []
    matched_gt = {i for i, _ in matched}
    matched_pred = {j for _, j in matched}

    pairs: list[LeafPair] = []

    for i, j in matched:
        pairs.append(LeafPair(f"{path}[{i}]", arr_gt[i], arr_pred[j], field_type))

    for i in range(n):
        if i not in matched_gt:
            pairs.append(LeafPair(f"{path}[{i}]", arr_gt[i], None, field_type))

    for j in range(m):
        if j not in matched_pred:
            pairs.append(LeafPair(f"{path}[{n + j}]", None, arr_pred[j], field_type))

    return pairs


def flatten_to_pairs(
    gt: Any,
    pred: Any,
    schema: dict,
    root_schema: dict | None,
    path: str = "$",
) -> list[LeafPair]:
    """GT와 Predicted를 JSON Schema 기반으로 재귀 순회하여 리프 페어 리스트 생성."""
    if root_schema:
        schema = resolve_schema(schema, root_schema)

    field_type = get_field_type(schema, root_schema)

    # 둘 다 None
    if gt is None and pred is None:
        return []

    # object
    if field_type == "object" or isinstance(gt, dict) or isinstance(pred, dict):
        return _flatten_object(gt, pred, schema, root_schema, path)

    # array
    if field_type == "array" or isinstance(gt, list) or isinstance(pred, list):
        return _flatten_array(gt, pred, schema, root_schema, path)

    # leaf
    if field_type in ("string", "number", "integer", "boolean"):
        return [LeafPair(path, gt, pred, field_type)]

    # fallback: 값에서 타입 추론
    inferred = infer_type(gt if gt is not None else pred)
    if inferred == "object":
        return _flatten_object(gt, pred, schema, root_schema, path)
    if inferred == "array":
        return _flatten_array(gt, pred, schema, root_schema, path)

    return [LeafPair(path, gt, pred, inferred)]


def _flatten_object(
    gt: Any, pred: Any, schema: dict, root_schema: dict | None, path: str
) -> list[LeafPair]:
    """object 타입 재귀 순회."""
    gt_dict = gt if isinstance(gt, dict) else {}
    pred_dict = pred if isinstance(pred, dict) else {}
    props = get_properties(schema, root_schema)

    pairs: list[LeafPair] = []
    # GT 키 기준 순회 (스키마에 없는 키도 포함)
    all_keys = set(gt_dict.keys())
    # pred에만 있는 키도 추가
    all_keys |= set(pred_dict.keys())
    # evaluation_config 같은 메타 키 제외
    all_keys -= {"evaluation_config"}

    for key in sorted(all_keys):
        child_path = f"{path}.{key}"
        child_schema = props.get(key, {})
        if root_schema and "$ref" in child_schema:
            child_schema = resolve_schema(child_schema, root_schema)
        if root_schema and "anyOf" in child_schema:
            child_schema = resolve_schema(child_schema, root_schema)

        gt_val = gt_dict.get(key)
        pred_val = pred_dict.get(key)
        pairs.extend(
            flatten_to_pairs(gt_val, pred_val, child_schema, root_schema, child_path)
        )

    return pairs


def _flatten_array(
    gt: Any, pred: Any, schema: dict, root_schema: dict | None, path: str
) -> list[LeafPair]:
    """array 타입 매칭."""
    gt_list = gt if isinstance(gt, list) else []
    pred_list = pred if isinstance(pred, list) else []

    if not gt_list and not pred_list:
        return []

    items_schema = get_items_schema(schema, root_schema)
    items_type = get_field_type(items_schema, root_schema)

    # object 배열: 헝가리안 매칭 후 재귀
    if items_type == "object" or (gt_list and isinstance(gt_list[0], dict)):
        return _match_object_array(gt_list, pred_list, path, items_schema, root_schema)

    # primitive 배열: 헝가리안 매칭
    return _match_primitive_array(gt_list, pred_list, path, items_type)

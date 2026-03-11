"""JSON Schema 탐색 유틸리티: $ref 해석, anyOf 처리, 타입 추론."""
from __future__ import annotations

from typing import Any


def unwrap_root(schema: dict) -> dict:
    """schema_definition 래퍼가 있으면 벗겨낸다."""
    if "schema_definition" in schema:
        return schema["schema_definition"]
    return schema


def resolve_ref(ref: str, root_schema: dict) -> dict:
    """#/$defs/xxx 형식의 $ref를 실제 스키마로 해석."""
    parts = ref.lstrip("#/").split("/")
    node = root_schema
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return {}
        node = node[p]
    return node


def resolve_schema(schema: dict, root_schema: dict) -> dict:
    """$ref와 anyOf를 해석하여 실제 스키마를 반환."""
    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], root_schema)
        merged = {**resolved}
        if "description" in schema:
            merged["description"] = schema["description"]
        return merged

    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            inner = {**non_null[0]}
            for key in ("description",):
                if key in schema and key not in inner:
                    inner[key] = schema[key]
            return resolve_schema(inner, root_schema)
        # 복잡한 anyOf (union type): 첫 번째 non-null 타입으로 fallback
        # 실제 비교 시 compare_leaf에서 타입 변환을 시도하므로 대부분 동작함
        if non_null:
            return resolve_schema(non_null[0], root_schema)

    return schema


def get_field_type(schema: dict, root_schema: dict | None = None) -> str:
    """스키마 노드의 타입 문자열 반환."""
    if root_schema and "$ref" in schema:
        schema = resolve_schema(schema, root_schema)
    if root_schema and "anyOf" in schema:
        schema = resolve_schema(schema, root_schema)
    return schema.get("type", "string")


def get_properties(schema: dict, root_schema: dict | None = None) -> dict:
    """스키마의 properties를 반환. $ref 해석 포함."""
    if root_schema:
        schema = resolve_schema(schema, root_schema)
    return schema.get("properties", {})


def get_items_schema(schema: dict, root_schema: dict | None = None) -> dict:
    """배열 스키마의 items를 반환. $ref 해석 포함."""
    if root_schema:
        schema = resolve_schema(schema, root_schema)
    items = schema.get("items", {})
    if root_schema and "$ref" in items:
        items = resolve_schema(items, root_schema)
    return items


def infer_type(value: Any) -> str:
    """값에서 타입 추론 (스키마 없을 때 fallback)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"

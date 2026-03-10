"""JSON Schema를 Pydantic BaseModel로 동적 변환."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model


def _resolve_ref(ref: str, root_schema: dict) -> dict:
    """#/$defs/xxx 형식의 $ref를 실제 스키마로 해석."""
    parts = ref.lstrip("#/").split("/")
    node = root_schema
    for p in parts:
        node = node[p]
    return node


def _resolve_schema(schema: dict, root_schema: dict) -> dict:
    """$ref가 있으면 해석하고, 원본 필드(description, evaluation_config 등)와 병합."""
    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], root_schema)
        merged = {**resolved}
        for key in ("description", "evaluation_config"):
            if key in schema:
                merged[key] = schema[key]
        return merged
    return schema


def _get_type_annotation(
    schema: dict,
    root_schema: dict,
    with_descriptions: bool,
    model_name_prefix: str,
    counter: list[int],
) -> Any:
    """JSON Schema 노드 하나를 Python type annotation으로 변환."""

    schema = _resolve_schema(schema, root_schema)

    # anyOf로 nullable 처리: [{"type": "string"}, {"type": "null"}] 등
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        has_null = any(s.get("type") == "null" for s in schema["anyOf"])
        if len(non_null) == 1:
            inner = {**non_null[0]}
            # 원본 schema의 description/evaluation_config 유지
            for key in ("description", "evaluation_config"):
                if key in schema and key not in inner:
                    inner[key] = schema[key]
            inner_type = _get_type_annotation(
                inner, root_schema, with_descriptions, model_name_prefix, counter
            )
            if has_null:
                return Optional[inner_type]
            return inner_type
        # 복잡한 anyOf: 여러 non-null 타입 -> Any
        if has_null:
            return Optional[Any]
        return Any

    schema_type = schema.get("type")

    # enum 처리
    if "enum" in schema:
        values = tuple(schema["enum"])
        return Literal[values]  # type: ignore[valid-type]

    # 기본 타입
    if schema_type == "string":
        return str
    if schema_type == "number":
        return float
    if schema_type == "integer":
        return int
    if schema_type == "boolean":
        return bool

    # array 처리
    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _get_type_annotation(
            items, root_schema, with_descriptions, model_name_prefix, counter
        )
        return list[item_type]

    # object -> nested BaseModel
    if schema_type == "object":
        return _build_model(
            schema, root_schema, with_descriptions, model_name_prefix, counter
        )

    # fallback
    return Any


def _build_model(
    schema: dict,
    root_schema: dict,
    with_descriptions: bool,
    model_name_prefix: str,
    counter: list[int],
) -> type[BaseModel]:
    """object 타입의 JSON Schema를 Pydantic BaseModel로 변환."""
    props = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    field_definitions: dict[str, Any] = {}

    for field_name, field_schema in props.items():
        # evaluation_config 필드는 Pydantic 모델에서 무시
        if field_name == "evaluation_config":
            continue

        resolved = _resolve_schema(field_schema, root_schema)
        field_type = _get_type_annotation(
            resolved, root_schema, with_descriptions, model_name_prefix, counter
        )

        desc = resolved.get("description")
        is_required = field_name in required_fields

        if with_descriptions and desc:
            if is_required:
                field_definitions[field_name] = (field_type, Field(description=desc))
            else:
                field_definitions[field_name] = (
                    Optional[field_type],
                    Field(default=None, description=desc),
                )
        else:
            if is_required:
                field_definitions[field_name] = (field_type, ...)
            else:
                field_definitions[field_name] = (Optional[field_type], None)

    counter[0] += 1
    name = f"{model_name_prefix}_{counter[0]}"
    return create_model(name, **field_definitions)


def json_schema_to_pydantic(
    schema: dict,
    with_descriptions: bool = True,
    model_name: str = "ExtractModel",
) -> type[BaseModel]:
    """JSON Schema를 Pydantic BaseModel로 동적 변환.

    Args:
        schema: JSON Schema dict. resume-schema처럼 schema_definition 래퍼가 있으면 자동 처리.
        with_descriptions: True이면 Field(description=...) 포함, False이면 제거.
        model_name: 생성될 루트 모델 이름.
    """
    # resume-schema.json처럼 schema_definition 래퍼 처리
    if "schema_definition" in schema:
        root_schema = schema["schema_definition"]
    else:
        root_schema = schema

    counter = [0]
    props = root_schema.get("properties", {})
    required_fields = set(root_schema.get("required", []))
    field_definitions: dict[str, Any] = {}

    for field_name, field_schema in props.items():
        if field_name == "evaluation_config":
            continue

        resolved = _resolve_schema(field_schema, root_schema)
        field_type = _get_type_annotation(
            resolved, root_schema, with_descriptions, model_name, counter
        )

        desc = resolved.get("description")
        is_required = field_name in required_fields

        if with_descriptions and desc:
            if is_required:
                field_definitions[field_name] = (field_type, Field(description=desc))
            else:
                field_definitions[field_name] = (
                    Optional[field_type],
                    Field(default=None, description=desc),
                )
        else:
            if is_required:
                field_definitions[field_name] = (field_type, ...)
            else:
                field_definitions[field_name] = (Optional[field_type], None)

    return create_model(model_name, **field_definitions)

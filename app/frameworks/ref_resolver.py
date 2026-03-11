"""JSON Schema $ref inline 유틸리티.

vLLM 등 OpenAI-compatible 서버는 $ref를 해석하지 못하므로,
Pydantic의 .model_json_schema()가 생성하는 $defs/$ref를 inline으로 풀어준다.
"""
from __future__ import annotations

import copy
from typing import Any


def resolve_refs(schema: dict) -> dict:
    """JSON Schema의 $ref/$defs를 재귀적으로 inline 처리.

    Pydantic v2의 .model_json_schema()는 중첩 모델을 $defs + $ref로 분리하는데,
    vLLM 등은 이를 해석하지 못한다. 이 함수로 $ref를 실제 정의로 치환한다.

    Args:
        schema: JSON Schema dict (원본을 변경하지 않으려면 .copy() 후 전달)

    Returns:
        $ref가 모두 inline된 JSON Schema dict
    """
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].rsplit("/", 1)[-1]
                if ref_name in defs:
                    return _resolve(defs[ref_name])
                return node
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    resolved = _resolve(schema)
    resolved.pop("$defs", None)
    _add_additional_properties_false(resolved)
    return resolved


def _add_additional_properties_false(node: Any) -> None:
    """모든 object 타입 노드에 additionalProperties: false 추가 (strict 모드 호환)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _add_additional_properties_false(v)
    elif isinstance(node, list):
        for item in node:
            _add_additional_properties_false(item)

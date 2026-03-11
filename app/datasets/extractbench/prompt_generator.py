"""JSON Schema의 description을 추출하여 rich 프롬프트 동적 생성."""
from __future__ import annotations

from app.datasets.shared.schema_converter import _resolve_ref


def _collect_descriptions(schema: dict, root_schema: dict | None = None, path: str = "", depth: int = 0) -> list[str]:
    """재귀적으로 properties의 description을 계층 구조로 추출."""
    lines: list[str] = []
    indent = "  " * depth

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            if name == "evaluation_config":
                continue

            # $ref 해석
            resolved_prop = prop
            if "$ref" in prop and root_schema:
                resolved_prop = _resolve_ref(prop["$ref"], root_schema)
                # 원본 description 유지
                if "description" in prop:
                    resolved_prop = {**resolved_prop, "description": prop["description"]}

            # anyOf 해석
            if "anyOf" in resolved_prop and root_schema:
                non_null = [s for s in resolved_prop["anyOf"] if s.get("type") != "null"]
                if non_null:
                    inner = non_null[0]
                    if "$ref" in inner:
                        inner = _resolve_ref(inner["$ref"], root_schema)
                    resolved_prop = {**inner}
                    if "description" in prop:
                        resolved_prop["description"] = prop["description"]

            desc = resolved_prop.get("description", "")
            type_str = resolved_prop.get("type", "")

            if desc:
                lines.append(f"{indent}- {name} ({type_str}): {desc}")
            else:
                lines.append(f"{indent}- {name} ({type_str})")

            # 재귀: nested object
            if resolved_prop.get("type") == "object" and "properties" in resolved_prop:
                lines.extend(_collect_descriptions(resolved_prop, root_schema, f"{path}.{name}", depth + 1))

            # 재귀: array of objects
            if resolved_prop.get("type") == "array" and "items" in resolved_prop:
                items = resolved_prop["items"]
                if "$ref" in items and root_schema:
                    items = _resolve_ref(items["$ref"], root_schema)
                if isinstance(items, dict) and items.get("type") == "object":
                    lines.extend(
                        _collect_descriptions(items, root_schema, f"{path}.{name}[]", depth + 1)
                    )

    return lines


def generate_rich_prompt(schema_dict: dict) -> str:
    """JSON Schema의 description들을 추출하여 상세 프롬프트 생성."""
    # schema_definition 래퍼 처리
    if "schema_definition" in schema_dict:
        schema = schema_dict["schema_definition"]
    else:
        schema = schema_dict

    field_lines = _collect_descriptions(schema, root_schema=schema)
    fields_block = "\n".join(field_lines)

    schema_name = schema_dict.get("name", "Document")

    return f"""You are a precise data extraction assistant.
Extract structured information from the given document text.

Follow these field definitions carefully:
{fields_block}

Rules:
- Extract values exactly as they appear in the document.
- If a field is not present in the document, use null.
- For dates, follow the format specified in each field description.
- For numeric values, extract the exact number without units.
- For arrays, include all matching items found in the document.
- For enum fields, use only one of the allowed values.
"""

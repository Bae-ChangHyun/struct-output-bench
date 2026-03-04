"""JSON Schema의 description을 추출하여 rich 프롬프트 동적 생성."""
from __future__ import annotations


def _collect_descriptions(schema: dict, path: str = "", depth: int = 0) -> list[str]:
    """재귀적으로 properties의 description을 계층 구조로 추출."""
    lines: list[str] = []
    indent = "  " * depth

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            if name == "evaluation_config":
                continue
            desc = prop.get("description", "")
            type_str = prop.get("type", "")

            # anyOf에서 실제 타입 추출
            if "anyOf" in prop:
                non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
                if non_null:
                    type_str = non_null[0].get("type", "")

            if desc:
                lines.append(f"{indent}- {name} ({type_str}): {desc}")
            else:
                lines.append(f"{indent}- {name} ({type_str})")

            # 재귀: nested object
            if prop.get("type") == "object" and "properties" in prop:
                lines.extend(_collect_descriptions(prop, f"{path}.{name}", depth + 1))

            # 재귀: array of objects
            if prop.get("type") == "array" and "items" in prop:
                items = prop["items"]
                if isinstance(items, dict) and items.get("type") == "object":
                    lines.extend(
                        _collect_descriptions(items, f"{path}.{name}[]", depth + 1)
                    )

    return lines


def generate_rich_prompt(schema_dict: dict) -> str:
    """JSON Schema의 description들을 추출하여 상세 프롬프트 생성."""
    # schema_definition 래퍼 처리
    if "schema_definition" in schema_dict:
        schema = schema_dict["schema_definition"]
    else:
        schema = schema_dict

    field_lines = _collect_descriptions(schema)
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

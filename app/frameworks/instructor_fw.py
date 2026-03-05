from __future__ import annotations

from typing import Any, TYPE_CHECKING

import instructor
from instructor.processing import schema as _instructor_schema
from instructor.providers.openai import utils as _instructor_openai_utils
from openai import AsyncOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_MODE_MAP = {
    "default": instructor.Mode.TOOLS,
    "tools": instructor.Mode.TOOLS,
    "tools_strict": instructor.Mode.TOOLS_STRICT,
    "json": instructor.Mode.JSON,
    "json_schema": instructor.Mode.JSON_SCHEMA,
    "md_json": instructor.Mode.MD_JSON,
}


def _resolve_refs(schema: dict) -> dict:
    """Recursively inline $ref/$defs in a JSON Schema for vLLM compatibility."""
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].rsplit("/", 1)[-1]
                if ref_name in defs:
                    return _resolve(defs[ref_name])
                return node
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


# Monkeypatch instructor's generate_openai_schema to resolve $ref.
# instructor sends $defs/$ref to the API, which vLLM models cannot interpret.
_original_generate_openai_schema = _instructor_schema.generate_openai_schema.__wrapped__


def _patched_generate_openai_schema(model: type) -> dict[str, Any]:
    result = _original_generate_openai_schema(model)
    if "$defs" in result.get("parameters", {}):
        result = {
            **result,
            "parameters": _resolve_refs(result["parameters"].copy()),
        }
    return result


_instructor_schema.generate_openai_schema = _patched_generate_openai_schema  # type: ignore[assignment]
# Also patch the already-bound reference in openai utils (from ... import)
_instructor_openai_utils.generate_openai_schema = _patched_generate_openai_schema  # type: ignore[assignment]


@FrameworkRegistry.register("instructor")
class InstructorAdapter(BaseFrameworkAdapter):
    name = "instructor"
    supported_modes = list(_MODE_MAP.keys())

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        mode = _MODE_MAP.get(self.mode, instructor.Mode.TOOLS)

        client = instructor.from_openai(
            AsyncOpenAI(base_url=self.base_url, api_key=self.api_key),
            mode=mode,
        )

        result = await client.chat.completions.create(
            model=self.model,
            response_model=schema_class,
            max_retries=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import v1 as pv1

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_PROMPT_TPL = PromptTemplate("{system_prompt}\n\n{text}")

_V1_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _json_schema_to_v1_model(schema: dict[str, Any], model_name: str = "Model") -> type:
    """Pydantic v2 JSON Schema → Pydantic v1 동적 모델 생성 (LlamaIndex 호환)."""
    defs = schema.get("$defs", {})
    created: dict[str, type] = {}

    def _resolve_type(prop_schema: dict) -> Any:
        if "$ref" in prop_schema:
            ref_name = prop_schema["$ref"].rsplit("/", 1)[-1]
            if ref_name not in created:
                _build(ref_name, defs[ref_name])
            return created[ref_name]
        t = prop_schema.get("type", "string")
        if t == "array":
            return List[_resolve_type(prop_schema.get("items", {}))]
        if t == "object" and "properties" in prop_schema:
            sub = prop_schema.get("title", "Sub")
            _build(sub, prop_schema)
            return created[sub]
        return _V1_TYPE_MAP.get(t, str)

    def _build(name: str, obj_schema: dict) -> None:
        if name in created:
            return
        created[name] = None  # type: ignore  (placeholder)
        props = obj_schema.get("properties", {})
        fields = {}
        for fname, fschema in props.items():
            ft = _resolve_type(fschema)
            desc = fschema.get("description")
            fields[fname] = (Optional[ft], pv1.Field(None, description=desc) if desc else None)
        model = pv1.create_model(name, **fields)
        model.__doc__ = obj_schema.get("description", "Extracted data")
        created[name] = model

    for def_name, def_schema in defs.items():
        _build(def_name, def_schema)
    _build(model_name, schema)
    return created[model_name]


def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON Schema의 $ref를 inline으로 풀어준다."""
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
    return resolved


@FrameworkRegistry.register("llamaindex")
class LlamaIndexAdapter(BaseFrameworkAdapter):
    name = "llamaindex"
    supported_modes = ["default", "text", "function_calling"]

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        if self.mode == "function_calling":
            return await self._extract_function_calling(text, schema_class, system_prompt)
        return await self._extract_text(text, schema_class, system_prompt)

    async def _extract_text(
        self, text: str, schema_class: type[BaseModel], system_prompt: str,
    ) -> ExtractionResult:
        """LLMTextCompletionProgram (JSON text mode)."""
        llm = OpenAILike(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        v2_schema = schema_class.model_json_schema()
        v1_cls = _json_schema_to_v1_model(v2_schema, schema_class.__name__)

        program = LLMTextCompletionProgram.from_defaults(
            output_cls=v1_cls,
            llm=llm,
            prompt=_PROMPT_TPL,
        )
        v1_result = await asyncio.to_thread(
            program, system_prompt=system_prompt, text=text,
        )

        data = v1_result.dict()
        return ExtractionResult(success=True, data=data)

    async def _extract_function_calling(
        self, text: str, schema_class: type[BaseModel], system_prompt: str,
    ) -> ExtractionResult:
        """$ref inline + OpenAI tool call."""
        if not schema_class.__doc__:
            schema_class.__doc__ = "Extracted structured data"

        raw_schema = schema_class.model_json_schema()
        inlined_schema = _dereference_schema(raw_schema)

        tool = {
            "type": "function",
            "function": {
                "name": schema_class.__name__,
                "description": schema_class.__doc__,
                "parameters": inlined_schema,
            },
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        def _call():
            from openai import OpenAI
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "dummy",
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": schema_class.__name__}},
            )
            tc = resp.choices[0].message.tool_calls
            if not tc:
                raise ValueError("No tool calls in response")
            return json.loads(tc[0].function.arguments)

        data = await asyncio.to_thread(_call)
        return ExtractionResult(success=True, data=data)

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

import instructor
from openai import AsyncOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry
from app.frameworks.ref_resolver import resolve_refs

if TYPE_CHECKING:
    from pydantic import BaseModel

_MODE_MAP = {
    # vLLM/guided-decoding 엔드포인트는 tool-call 파서가 없어 plain TOOLS에서 tool_calls를
    # 반환하지 않는다(응답은 content의 JSON). cvfit과 동일하게 기본을 JSON으로 둔다.
    # 명시적 tool-call 벤치마크는 "tools"/"tools_strict" 모드로 지정한다.
    "default": instructor.Mode.JSON,
    "tools": instructor.Mode.TOOLS,
    "tools_strict": instructor.Mode.TOOLS_STRICT,
    "json": instructor.Mode.JSON,
    "json_schema": instructor.Mode.JSON_SCHEMA,
    "md_json": instructor.Mode.MD_JSON,
}


@FrameworkRegistry.register("instructor")
class InstructorAdapter(BaseFrameworkAdapter):
    name = "instructor"
    supported_modes = tuple(_MODE_MAP.keys())

    def __init__(self, model, base_url=None, api_key=None, mode="default", **kwargs):
        super().__init__(model, base_url, api_key, mode, **kwargs)
        inst_mode = _MODE_MAP.get(self.mode, instructor.Mode.JSON)
        base_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

        _orig_create = base_client.chat.completions.create

        @functools.wraps(_orig_create)
        async def _vllm_compat_create(*args: Any, **kwargs: Any) -> Any:
            tools = kwargs.get("tools")
            if tools:
                for tool in tools:
                    func = tool.get("function", {})
                    if func.get("description") is None:
                        func["description"] = "Extract structured data"
                    params = func.get("parameters", {})
                    if "$defs" in params:
                        func["parameters"] = resolve_refs(params)
            return await _orig_create(*args, **kwargs)

        base_client.chat.completions.create = _vllm_compat_create  # type: ignore[assignment]
        self._client = instructor.from_openai(base_client, mode=inst_mode)

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        if not schema_class.__doc__:
            schema_class.__doc__ = "Extracted structured data"

        result = await self._client.chat.completions.create(
            model=self.model,
            response_model=schema_class,
            max_retries=0,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )

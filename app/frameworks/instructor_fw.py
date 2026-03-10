from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

import instructor
from openai import AsyncOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry
from app.frameworks.schema_utils import resolve_refs

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


@FrameworkRegistry.register("instructor")
class InstructorAdapter(BaseFrameworkAdapter):
    name = "instructor"
    supported_modes = tuple(_MODE_MAP.keys())

    def __init__(self, model, base_url=None, api_key=None, mode="default"):
        super().__init__(model, base_url, api_key, mode)
        inst_mode = _MODE_MAP.get(self.mode, instructor.Mode.TOOLS)
        base_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

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
            schema_class = type(
                schema_class.__name__,
                (schema_class,),
                {"__doc__": "Extracted structured data"},
            )

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

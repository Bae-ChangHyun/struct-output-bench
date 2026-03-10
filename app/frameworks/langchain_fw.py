from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_MODE_MAP = {
    "default": "json_schema",
    "json_schema": "json_schema",
    "function_calling": "function_calling",
    "json_mode": "json_mode",
}

# vLLM does not support these OpenAI-specific parameters
_VLLM_DISABLED_PARAMS = {
    "parallel_tool_calls": None,
    "stream_options": None,
}


@FrameworkRegistry.register("langchain")
class LangChainAdapter(BaseFrameworkAdapter):
    name = "langchain"
    supported_modes = list(_MODE_MAP.keys())

    def __init__(self, model, base_url=None, api_key=None, mode="default"):
        super().__init__(model, base_url, api_key, mode)
        self._llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
            disabled_params=_VLLM_DISABLED_PARAMS,
        )

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        method = _MODE_MAP.get(self.mode, "json_schema")
        llm = self._llm

        strict = True if method == "json_schema" else None
        structured_llm = llm.with_structured_output(
            schema_class,
            method=method,
            include_raw=True,
            strict=strict,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        if method == "json_mode":
            schema_text = json.dumps(schema_class.model_json_schema(), ensure_ascii=False)
            messages[0] = {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"You must respond in JSON matching this schema:\n{schema_text}"
                ),
            }

        response = await structured_llm.ainvoke(messages)

        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")

        if parsing_error is not None:
            return ExtractionResult(
                success=False,
                error=f"Parsing error: {parsing_error}",
            )

        if parsed is None:
            raw = response.get("raw")
            return ExtractionResult(
                success=False,
                error=f"Model returned None. Raw: {raw}",
            )

        data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        return ExtractionResult(
            success=True,
            data=data,
        )

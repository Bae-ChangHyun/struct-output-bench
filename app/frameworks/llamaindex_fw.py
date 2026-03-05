from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry
from app.frameworks.schema_utils import resolve_refs

if TYPE_CHECKING:
    from pydantic import BaseModel

_PROMPT_TPL = PromptTemplate("{system_prompt}\n\n{text}")


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
        """LLMTextCompletionProgram (JSON text mode) — Pydantic v2 직접 사용."""
        llm = OpenAILike(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            is_chat_model=True,
            is_function_calling_model=True,
        )

        program = LLMTextCompletionProgram.from_defaults(
            output_cls=schema_class,
            llm=llm,
            prompt=_PROMPT_TPL,
        )
        result = await program.acall(system_prompt=system_prompt, text=text)

        data = result.model_dump()
        return ExtractionResult(success=True, data=data)

    async def _extract_function_calling(
        self, text: str, schema_class: type[BaseModel], system_prompt: str,
    ) -> ExtractionResult:
        """$ref inline + OpenAI tool call."""
        if not schema_class.__doc__:
            schema_class.__doc__ = "Extracted structured data"

        raw_schema = schema_class.model_json_schema()
        inlined_schema = resolve_refs(raw_schema)

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

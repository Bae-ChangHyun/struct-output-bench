from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic_ai import Agent, NativeOutput, TextOutput, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("pydantic_ai")
class PydanticAIAdapter(BaseFrameworkAdapter):
    name = "pydantic_ai"
    supported_modes = ("default", "tool", "json", "text")

    def __init__(self, model, base_url=None, api_key=None, mode="default", **kwargs):
        super().__init__(model, base_url, api_key, mode, **kwargs)
        self._model = OpenAIChatModel(
            self.model,
            provider=OpenAIProvider(
                base_url=self.base_url,
                api_key=self.api_key,
            ),
        )

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        model = self._model

        output_type = self._build_output_type(schema_class)
        effective_prompt = system_prompt
        if self.mode == "text":
            schema_json = json.dumps(schema_class.model_json_schema(), ensure_ascii=False)
            effective_prompt = (
                f"{system_prompt}\n\n"
                f"You MUST respond with ONLY valid JSON matching this schema:\n{schema_json}"
            )
        agent = Agent(
            model,
            system_prompt=effective_prompt,
            output_type=output_type,
            retries=0,
        )
        result = await agent.run(text, model_settings={"temperature": 0})

        return ExtractionResult(
            success=True,
            data=result.output.model_dump(),
        )

    def _build_output_type(self, schema_class: type[BaseModel]):
        if self.mode == "tool":
            return ToolOutput(
                schema_class,
                name="extract_data",
                description="Extract structured data from the input text.",
            )
        if self.mode in ("json", "default"):
            return NativeOutput(schema_class)
        if self.mode == "text":
            def _parse_json(text: str):
                data = json.loads(text)
                return schema_class.model_validate(data)
            return TextOutput(_parse_json)
        return NativeOutput(schema_class)

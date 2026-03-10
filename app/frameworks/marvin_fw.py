from __future__ import annotations

from typing import TYPE_CHECKING

import marvin
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("marvin")
class MarvinAdapter(BaseFrameworkAdapter):
    name = "marvin"
    supported_modes = ["cast", "extract"]

    def __init__(self, model, base_url=None, api_key=None, mode="default"):
        super().__init__(model, base_url, api_key, mode)
        provider = OpenAIProvider(
            base_url=self.base_url,
            api_key=self.api_key or "dummy",
        )
        self._model = OpenAIModel(self.model, provider=provider)

    def _build_agent(self, system_prompt: str) -> marvin.Agent:
        return marvin.Agent(model=self._model, instructions=system_prompt)

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        agent = self._build_agent(system_prompt)

        if self.mode == "extract":
            results = await marvin.extract_async(
                data=text,
                target=schema_class,
                instructions=system_prompt,
                agent=agent,
            )
            if not results:
                return ExtractionResult(
                    success=False,
                    error="extract_async returned empty list",
                )
            result = results[0]
        else:
            result = await marvin.cast_async(
                data=text,
                target=schema_class,
                instructions=system_prompt,
                agent=agent,
            )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )

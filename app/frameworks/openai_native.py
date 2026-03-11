from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry
from app.frameworks.schema_utils import resolve_refs

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("openai")
class OpenAINativeAdapter(BaseFrameworkAdapter):
    name = "openai"
    supported_modes = ("default", "tool_calling", "json_object")

    def __init__(self, model, base_url=None, api_key=None, mode="default", **kwargs):
        super().__init__(model, base_url, api_key, mode, **kwargs)
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        client = self._client

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        if self.mode == "tool_calling":
            return await self._extract_tool_calling(client, messages, schema_class)
        elif self.mode == "json_object":
            return await self._extract_json_object(client, messages, schema_class)
        else:
            return await self._extract_default(client, messages, schema_class)

    async def _extract_default(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        schema_class: type[BaseModel],
    ) -> ExtractionResult:
        completion = await client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema_class,
            temperature=0,
        )
        message = completion.choices[0].message

        if message.parsed:
            return ExtractionResult(
                success=True,
                data=message.parsed.model_dump(),
            )
        return ExtractionResult(
            success=False,
            error=message.refusal or "No parsed response",
        )

    async def _extract_tool_calling(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        schema_class: type[BaseModel],
    ) -> ExtractionResult:
        schema = resolve_refs(schema_class.model_json_schema())
        tool_name = schema_class.__name__

        completion = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Extract structured data as {tool_name}",
                        "parameters": schema,
                        "strict": True,
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": tool_name},
            },
        )
        message = completion.choices[0].message

        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            parsed = schema_class.model_validate(args)
            return ExtractionResult(
                success=True,
                data=parsed.model_dump(),
            )
        return ExtractionResult(
            success=False,
            error="No tool call in response",
        )

    async def _extract_json_object(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        schema_class: type[BaseModel],
    ) -> ExtractionResult:
        schema = schema_class.model_json_schema()
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        json_messages = list(messages)
        json_messages[0] = {
            "role": "system",
            "content": (
                f"{messages[0]['content']}\n\n"
                f"You MUST respond with a valid JSON object that conforms to this schema:\n"
                f"```json\n{schema_str}\n```"
            ),
        }

        completion = await client.chat.completions.create(
            model=self.model,
            messages=json_messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = completion.choices[0].message.content

        if content:
            data = json.loads(content)
            parsed = schema_class.model_validate(data)
            return ExtractionResult(
                success=True,
                data=parsed.model_dump(),
            )
        return ExtractionResult(
            success=False,
            error="Empty response content",
        )

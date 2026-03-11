from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class ExtractionResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    framework: str = ""
    model: str = ""
    mode: str = ""


class BaseFrameworkAdapter(ABC):
    name: str = ""
    supported_modes: ClassVar[tuple[str, ...]] = ("default",)

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        mode: str = "default",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.mode = mode
        self.timeout = timeout

    @abstractmethod
    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult: ...

    async def run(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        start = time.perf_counter()
        try:
            result = await self.extract(text, schema_class, system_prompt)
        except Exception as e:
            result = ExtractionResult(success=False, error=str(e))
        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        result.framework = self.name
        result.model = self.model
        result.mode = self.mode
        return result

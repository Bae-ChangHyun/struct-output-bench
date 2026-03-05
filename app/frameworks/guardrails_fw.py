from __future__ import annotations

from typing import TYPE_CHECKING

from guardrails import AsyncGuard

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("guardrails")
class GuardrailsAdapter(BaseFrameworkAdapter):
    name = "guardrails"
    supported_modes = ["default"]

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        guard = AsyncGuard.for_pydantic(output_class=schema_class)

        # guardrails는 내부적으로 litellm 사용
        # vLLM 등 커스텀 서버: hosted_vllm/ provider 사용
        model_name = f"hosted_vllm/{self.model}"
        api_key = self.api_key or "dummy"
        api_base = self.base_url

        result = await guard(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            num_reasks=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        if result.validation_passed and result.validated_output:
            data = (
                result.validated_output.model_dump()
                if hasattr(result.validated_output, "model_dump")
                else dict(result.validated_output)
            )
            return ExtractionResult(success=True, data=data)

        if result.error:
            error = str(result.error)
        elif result.reask and result.reask.fail_results:
            error = "; ".join(
                fr.error_message
                for fr in result.reask.fail_results
                if fr.error_message
            )
        else:
            error = "Validation failed"
        return ExtractionResult(success=False, error=error)

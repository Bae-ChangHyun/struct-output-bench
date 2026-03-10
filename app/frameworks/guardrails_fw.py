from __future__ import annotations

from typing import TYPE_CHECKING

from guardrails import AsyncGuard

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry
from app.frameworks.schema_utils import resolve_refs

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("guardrails")
class GuardrailsAdapter(BaseFrameworkAdapter):
    name = "guardrails"
    supported_modes = ["default"]

    def __init__(self, model, base_url=None, api_key=None, mode="default"):
        super().__init__(model, base_url, api_key, mode)
        self._model_name = f"hosted_vllm/{self.model}"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        guard = AsyncGuard.for_pydantic(output_class=schema_class)

        # guardrails는 내부적으로 litellm 사용
        # vLLM 등 커스텀 서버: hosted_vllm/ provider 사용
        model_name = self._model_name
        api_key = self.api_key or "dummy"
        api_base = self.base_url

        # $ref를 inline으로 풀어서 response_format에 전달
        # vLLM은 $ref를 해석하지 못하므로 json_schema 모드로 구조화 출력 강제
        raw_schema = schema_class.model_json_schema()
        inlined_schema = resolve_refs(raw_schema)

        result = await guard(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            num_reasks=0,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_class.__name__,
                    "strict": True,
                    "schema": inlined_schema,
                },
            },
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

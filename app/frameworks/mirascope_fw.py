from __future__ import annotations

from typing import TYPE_CHECKING

from mirascope import llm
from mirascope.llm.providers.openai import model_id as _oai_model_id
from mirascope.llm.providers.openai.completions._utils import (
    encode as _oai_comp_encode,
    decode as _oai_comp_decode,
)
from mirascope.llm.providers.openai.completions import (
    base_provider as _oai_comp_base,
    provider as _oai_comp_provider,
)
from mirascope.llm.providers.openai.responses._utils import (
    encode as _oai_resp_encode,
)
from mirascope.llm.providers.openai.responses import (
    provider as _oai_resp_provider,
)

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from mirascope.llm import FormattingMode
    from pydantic import BaseModel

_MODE_MAP: dict[str, FormattingMode] = {
    "default": "tool",
    "tool": "tool",
    "json": "json",
    "strict": "strict",
}


# vLLM 서버의 모델명에 슬래시가 포함될 수 있으므로 (예: "openai/gpt-oss-120b")
# Mirascope의 model_name이 split("/")[1]로 자르는 것을 split("/", 1)[1]로 패치
def _patched_model_name(
    model_id: str, api_mode: _oai_model_id.ApiMode | None
) -> str:
    base_name = model_id.split("/", 1)[1].removesuffix(":responses").removesuffix(":completions")
    if api_mode is None:
        return base_name
    return f"{base_name}:{api_mode}"


# 모든 사용처에 패치 적용 (각 모듈이 from import로 바인딩하므로 개별 패치 필요)
_oai_model_id.model_name = _patched_model_name
_oai_comp_encode.model_name = _patched_model_name
_oai_comp_decode.model_name = _patched_model_name
_oai_comp_provider.openai_model_name = _patched_model_name
_oai_comp_base.openai_model_name = _patched_model_name
_oai_resp_encode.model_name = _patched_model_name
_oai_resp_provider.model_name = _patched_model_name


@FrameworkRegistry.register("mirascope")
class MirascopeAdapter(BaseFrameworkAdapter):
    name = "mirascope"
    supported_modes = list(_MODE_MAP.keys())

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        mode = _MODE_MAP.get(self.mode, "tool")

        # vLLM은 OpenAI-compatible API이므로 openai provider 사용
        # :completions suffix로 Chat Completions API 강제 (vLLM은 Responses API 미지원)
        model_id = f"openai/{self.model}:completions"

        llm.register_provider(
            "openai",
            scope=model_id,
            base_url=self.base_url,
            api_key=self.api_key or "no-key",
        )

        fmt = llm.format(schema_class, mode=mode)

        @llm.call(model_id, format=fmt)
        async def do_extract(resume_text: str, sys_prompt: str) -> str:
            return f"{sys_prompt}\n\n{resume_text}"

        response = await do_extract(text, system_prompt)
        parsed = response.parse()

        return ExtractionResult(
            success=True,
            data=parsed.model_dump(),
        )

"""벤치마크 설정 상수."""
from __future__ import annotations

# 3가지 조합 (A/B/C)
COMBINATIONS = [
    {"id": "A_desc", "label": "Schema(desc)+Prompt(min)", "use_desc": True, "use_rich": False},
    {"id": "B_nodesc", "label": "Schema(nodesc)+Prompt(min)", "use_desc": False, "use_rich": False},
    {"id": "C_rich", "label": "Schema(nodesc)+Prompt(rich)", "use_desc": False, "use_rich": True},
]

# 기본 프레임워크/모드
ALL_FW_MODES = [
    ("instructor", "tools"),
    ("instructor", "json_schema"),
    ("openai", "default"),
    ("langchain", "json_schema"),
    ("langchain", "function_calling"),
    ("marvin", "default"),
    ("pydantic_ai", "default"),
    ("mirascope", "default"),
    ("guardrails", "default"),
    ("llamaindex", "default"),
    ("llamaindex", "function_calling"),
]

# 공통 minimal prompt (데이터셋별 prompt가 없을 때 fallback)
DEFAULT_MINIMAL_PROMPT = (
    "You are a data extraction assistant.\n"
    "Extract structured information from the given text into the specified JSON schema.\n"
    "The output must be a valid JSON object conforming to the provided schema.\n"
    "Preserve original values as they appear in the text (numbers, dates, names, etc.).\n"
    "If a field is not present in the text, use null or the appropriate default value."
)

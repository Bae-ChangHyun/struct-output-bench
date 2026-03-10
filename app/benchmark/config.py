"""벤치마크 설정 상수."""
from __future__ import annotations

# 4가지 조합 (A/B/C/D)
COMBINATIONS = [
    {"id": "A_desc", "label": "Schema(desc)+Prompt(min)", "use_desc": True, "use_rich": False},
    {"id": "B_nodesc", "label": "Schema(nodesc)+Prompt(min)", "use_desc": False, "use_rich": False},
    {"id": "C_rich", "label": "Schema(nodesc)+Prompt(rich)", "use_desc": False, "use_rich": True},
    {"id": "D_both", "label": "Schema(desc)+Prompt(rich)", "use_desc": True, "use_rich": True},
]

# 전체 프레임워크/모드 (alias 제외, 고유 코드 경로만)
ALL_FW_MODES = [
    # instructor: "default"는 "tools" alias
    ("instructor", "tools"),
    ("instructor", "tools_strict"),
    ("instructor", "json"),
    ("instructor", "json_schema"),
    ("instructor", "md_json"),
    # openai: "default"는 parse API (고유)
    ("openai", "default"),
    ("openai", "tool_calling"),
    ("openai", "json_object"),
    # langchain: "default"는 "json_schema" alias
    ("langchain", "json_schema"),
    ("langchain", "function_calling"),
    ("langchain", "json_mode"),
    # marvin: "cast"와 "extract"만 지원
    ("marvin", "cast"),
    ("marvin", "extract"),
    # pydantic_ai: "default"는 "json" alias (동일 NativeOutput 분기)
    ("pydantic_ai", "tool"),
    ("pydantic_ai", "json"),
    ("pydantic_ai", "text"),
    # mirascope: "default"는 "tool" alias
    ("mirascope", "tool"),
    ("mirascope", "json"),
    ("mirascope", "strict"),
    # guardrails: 단일 모드
    ("guardrails", "default"),
    # llamaindex: "default"는 "text" alias (동일 _extract_text 경로)
    ("llamaindex", "text"),
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

# struct-output-bench

> LLM에서 구조화된 출력(Structured Output)을 만들기 위한 프레임워크는 Instructor, LangChain, Marvin, PydanticAI 등 다양하게 존재한다.
> **struct-output-bench**는 이들을 **동일한 조건**(같은 모델, 같은 스키마, 같은 프롬프트, temperature=0)에서 비교 테스트하는 벤치마크 도구이다.

Pydantic 스키마와 프롬프트를 정의하고, 여러 프레임워크에 동일한 입력을 넣어 결과 품질을 Ground Truth 기반으로 정량 비교한다. FastAPI 서버로도 개별 프레임워크를 즉시 테스트할 수 있다.

### Supported Frameworks

| 프레임워크 | 모드 | 구조화 방식 | Docs |
|-----------|------|-----------|------|
| **[Instructor](https://python.useinstructor.com/)** | [tools, tools_strict](https://python.useinstructor.com/modes-comparison/), [json](https://python.useinstructor.com/concepts/patching/#json-mode), [json_schema](https://python.useinstructor.com/concepts/patching/#json-schema-mode), [md_json](https://python.useinstructor.com/concepts/patching/#markdown-json-mode) | Tool Calling / JSON Schema | [modes](https://python.useinstructor.com/modes-comparison/) |
| **[OpenAI Native](https://platform.openai.com/)** | [default](https://platform.openai.com/docs/guides/structured-outputs) (parse), [tool_calling](https://platform.openai.com/docs/guides/function-calling), [json_object](https://platform.openai.com/docs/api-reference/chat/create) | JSON Schema / Tool Calling / JSON Object | [guide](https://platform.openai.com/docs/guides/structured-outputs) |
| **[LangChain](https://python.langchain.com/)** | [json_schema, function_calling, json_mode](https://python.langchain.com/docs/concepts/structured_outputs/) | JSON Schema / Tool Calling / JSON Mode | [guide](https://python.langchain.com/docs/concepts/structured_outputs/) |
| **[Marvin](https://askmarvin.ai/)** | [cast](https://askmarvin.ai/functions/cast), [extract](https://askmarvin.ai/functions/extract) | Tool Calling (cast_async / extract_async) | [docs](https://askmarvin.ai/) |
| **[PydanticAI](https://ai.pydantic.dev/)** | [tool](https://ai.pydantic.dev/output/#tooloutput) (ToolOutput), [json](https://ai.pydantic.dev/output/#nativeoutput) (NativeOutput), [text](https://ai.pydantic.dev/output/#textoutput) (TextOutput) | Tool Calling / JSON Schema / Text | [output](https://ai.pydantic.dev/output/) |
| **[Mirascope](https://mirascope.com/)** | [tool, json, strict](https://mirascope.com/docs/mirascope/learn/response_models) | Tool Calling / JSON Mode / Strict Mode | [guide](https://mirascope.com/docs/mirascope/learn/response_models) |
| **[Guardrails](https://www.guardrailsai.com/)** | [default](https://www.guardrailsai.com/docs/how_to_guides/structured_data_with_guardrails) | litellm 경유 (JSON Schema) | [guide](https://www.guardrailsai.com/docs/how_to_guides/structured_data_with_guardrails) |
| **[LlamaIndex](https://docs.llamaindex.ai/)** | [text](https://docs.llamaindex.ai/en/stable/examples/output_parsing/llm_program/) (LLMTextCompletionProgram), [function_calling](https://docs.llamaindex.ai/en/stable/examples/output_parsing/function_program/) (FunctionCallingProgram) | Text Completion / Tool Calling | [guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/) |

---

## Table of Contents

- [Motivation](#motivation)
- [Background](#background)
- [Experiment Design](#experiment-design)
- [Scoring](#scoring)
- [How Each Framework Works](#how-each-framework-works)
- [Benchmark Results](#benchmark-results)
- [Analysis](#analysis)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [References](#references)

---

## Motivation

LLM에서 Pydantic 모델 형태의 구조화된 출력을 얻기 위한 프레임워크가 많아졌다. Instructor, LangChain, Marvin, PydanticAI, Mirascope, Guardrails, LlamaIndex 등 각각 다른 방식으로 동일한 문제를 풀고 있다. 하지만 **같은 모델, 같은 스키마, 같은 프롬프트를 넣었을 때 과연 결과가 동일한가?**

struct-output-bench는 이 질문에 답하기 위한 도구이다:

- **동일 조건 비교**: 같은 입력(텍스트, 스키마, 프롬프트)으로 여러 프레임워크를 실행하고 결과를 비교
- **정량 평가**: Ground Truth 기반 100점 만점 채점으로 프레임워크 간 성능 차이를 수치화
- **API 서버**: FastAPI 기반으로 개별 프레임워크를 즉시 테스트 가능
- **확장 가능**: 새 프레임워크는 `BaseFrameworkAdapter`를 상속하고 `@register` 데코레이터만 붙이면 추가

### Key Questions

1. 프레임워크마다 Pydantic 스키마의 `Field(description=...)` 을 LLM에 전달하는 방식이 다른가?
2. 프롬프트에 필드 설명을 명시하는 것과 스키마의 description에 의존하는 것 중 무엇이 더 효과적인가?
3. 프레임워크 선택이 최종 결과 품질에 얼마나 영향을 미치는가?

---

## Background

### 3 Approaches to Structured Output

LLM에서 구조화된 출력을 얻는 방식은 크게 3가지로 나뉜다:

| 방식 | 동작 원리 | 대표 라이브러리 | 출력 보장 |
|------|----------|---------------|----------|
| **Prompting** | 프롬프트에 원하는 형식을 설명하고 LLM이 따르도록 유도 | spacy-llm | 보장 안됨 |
| **Function Calling (Tool Calling)** | Pydantic 스키마를 tool definition에 넣어 전달. LLM이 함수 인자 형태로 응답 | Instructor, Marvin, Mirascope | 거의 보장 |
| **Constrained Token Sampling** | 문법(CFG)으로 제약을 정의하고, 해당 제약을 만족하는 토큰만 샘플링 | Outlines, Guidance, **xgrammar** | 완전 보장 |

각 프레임워크는 이 방식 중 하나 이상을 사용한다. Instructor는 Function Calling과 JSON Schema 모드를 모두 지원하고, OpenAI Native SDK는 `response_format`으로 JSON Schema를 전달하여 xgrammar가 토큰을 제약한다. LangChain 역시 두 방식을 모두 제공한다.

### The Problem: vLLM Ignores Schema Descriptions

이 프로젝트는 vLLM 환경에서 Pydantic 스키마의 `Field(description=...)` 을 꼼꼼히 작성하여 structured output을 추출했는데, **예상과 다르게 description이 전혀 반영되지 않는 결과**가 나오면서 시작되었다. 분명히 같은 모델인데 왜 프레임워크마다 결과가 다른지 의심을 갖고 조사한 결과, 핵심 원인을 발견하였다:

| 방식 | vLLM에서의 동작 | Description 전달 |
|------|---------------|-----------------|
| **JSON Schema (response_format)** | xgrammar가 `type`, `properties`, `required` 등 **구조적 제약만** 사용 | **description 무시** |
| **Tool Calling** | tool definition에 description 포함하여 LLM에 전달 | **description 전달됨** |

vLLM의 xgrammar는 JSON Schema의 description 필드를 완전히 무시한다. OpenAI SDK는 Pydantic 스키마를 `response_format`으로 전달할 뿐 프롬프트에 스키마를 자동 주입하지 않는다. 따라서 JSON Schema 방식을 사용하는 프레임워크에서는 아무리 description을 정성껏 작성해도 LLM에 도달하지 않는다. 이 차이가 프레임워크 간 성능 차이를 만드는 핵심 원인이었다.

---

## Experiment Design

동일한 Pydantic 스키마를 두 가지 버전으로 준비하였다:

- **Description 포함 스키마**: 모든 필드에 `Field(description="...")` 포함
- **Description 없는 스키마**: 동일 구조이나 description 없이 타입과 기본값만 정의

**DeepJSONEval** 데이터셋(다국어, 깊은 중첩 JSON 추출 벤치마크) 100건의 샘플과 Ground Truth를 사용하여, 8개 프레임워크(22개 모드) × 3가지 조합 = 총 **6,600건**의 실험을 수행하였다.

### 4 Test Combinations

| 조합 | 스키마 | 프롬프트 | 실험 의도 |
|------|--------|---------|----------|
| **A: Schema(desc) + Prompt(minimal)** | description 포함 | 필드 설명 없는 최소 프롬프트 | description 필드에만 의존했을 때의 성능 측정 |
| **B: Schema(no desc) + Prompt(minimal)** | description 없음 | 필드 설명 없는 최소 프롬프트 | 아무 설명도 없을 때의 기저(baseline) 성능 측정 |
| **C: Schema(no desc) + Prompt(rich)** | description 없음 | 모든 필드 설명이 포함된 상세 프롬프트 | 프롬프트로만 설명을 제공했을 때의 성능 측정 |
| **D: Schema(desc) + Prompt(rich)** | description 포함 | 모든 필드 설명이 포함된 상세 프롬프트 | 스키마 description과 프롬프트를 모두 제공했을 때의 상한 측정 |

### Prompt Design

**Minimal Prompt**: 최소한의 지시만 포함 ("Extract structured information from the given text.")

**Rich Prompt**: 각 섹션과 필드에 대한 상세한 설명을 포함. Literal 타입의 허용 값, 각 필드의 용도, 날짜 형식 등을 명시한 약 80줄 분량의 가이드.

---

## Scoring

Ground Truth 기반 100점 만점 채점 방식을 사용한다. 추출 결과의 모든 리프 필드를 GT와 1:1 비교하여 필드별 점수(0.0~1.0)를 산출하고, 전체 평균을 백분율로 환산한다.

### Scoring Pipeline

```
Predicted + Ground Truth + JSON Schema
  → flatten_to_pairs()  : 재귀적으로 리프 필드 페어 추출
  → Hungarian Matching   : 배열 요소 순서 무관 최적 매칭 (scipy)
  → compare_leaf()       : 타입별 메트릭 적용
  → 평균 → 백분율(%)
```

### Leaf Comparison Metrics

| 타입 | 메트릭 | 설명 |
|------|--------|------|
| `string` | **NED** (1 - Normalized Edit Distance) | Levenshtein 기반 문자열 유사도. 1.0 = 완벽 일치 |
| `number` / `integer` | **Relative Error** | 상대 오차 ≤ 5% → 1.0, 초과 → 0.0. GT=0일 때는 절대 오차 기반 |
| `boolean` | **Exact Match** | 일치 = 1.0, 불일치 = 0.0 |
| `null` | **Exact Match** | 양쪽 모두 null = 1.0, 한쪽만 null = 0.0 |

### Array Matching

배열 요소의 순서가 GT와 다를 수 있으므로, **헝가리안 알고리즘**으로 최적 매칭을 수행한다. GT 배열의 각 요소와 Predicted 배열의 각 요소 간 유사도 행렬을 구성하고, 최대 유사도 합을 갖는 매칭을 찾는다.

---

## How Each Framework Works

<details>
<summary><b>Instructor</b> — Tool Calling / JSON Schema, 자동 retry 및 validation 내장</summary>

```python
client = instructor.from_openai(
    AsyncOpenAI(base_url=..., api_key=...),
    mode=instructor.Mode.TOOLS,  # or TOOLS_STRICT, JSON, JSON_SCHEMA, MD_JSON
)
result = await client.chat.completions.create(
    model=model,
    response_model=schema_class,
    max_retries=0,
    messages=[...],
)
```

`instructor.from_openai()`에 `AsyncOpenAI` 클라이언트를 전달하여 vLLM 등 OpenAI 호환 서버에 연결한다. `mode` 파라미터로 Tool Calling(TOOLS, TOOLS_STRICT) 또는 JSON Schema(JSON, JSON_SCHEMA, MD_JSON) 방식을 선택할 수 있다. Tool Calling 방식에서는 description 필드가 tool definition에 포함되므로 LLM이 필드의 의미를 파악할 수 있다. nested Pydantic 모델의 `$ref`는 자동으로 inline 처리된다.

</details>

<details>
<summary><b>OpenAI Native</b> — 3가지 모드 지원 (default / tool_calling / json_object)</summary>

```python
# default: response_format + parse()
client.chat.completions.parse(
    model=model, messages=[...], response_format=schema_class,
)

# tool_calling: tools + tool_choice
client.chat.completions.create(
    model=model, messages=[...],
    tools=[{"type": "function", "function": {"name": "...", "parameters": schema}}],
    tool_choice={"type": "function", "function": {"name": "..."}},
)

# json_object: response_format={"type": "json_object"} + 스키마를 프롬프트에 포함
client.chat.completions.create(
    model=model, messages=[...], response_format={"type": "json_object"},
)
```

AsyncOpenAI 기반으로 3가지 structured output 모드를 지원한다. **default** 모드는 `response_format`으로 Pydantic 스키마를 전달하여 xgrammar가 구조를 강제한다. **tool_calling** 모드는 스키마를 function tool로 등록하여 description을 LLM에 직접 전달한다. **json_object** 모드는 JSON Schema를 시스템 프롬프트에 포함하고 `response_format={"type": "json_object"}`로 JSON 응답을 강제한다.

</details>

<details>
<summary><b>LangChain</b> — json_schema / function_calling / json_mode 3가지 모드 지원</summary>

```python
llm = ChatOpenAI(model=model, base_url=...)
structured_llm = llm.with_structured_output(schema_class, method="json_schema")  # or "function_calling", "json_mode"
result = await structured_llm.ainvoke(messages)
```

`json_schema` 모드에서는 `response_format` 기반으로 동작하고, `function_calling` 모드에서는 Tool Calling 방식으로 동작한다. `json_mode`는 `response_format={"type": "json_object"}`와 프롬프트 내 스키마 주입으로 동작한다.

</details>

<details>
<summary><b>Marvin</b> — pydantic_ai 기반 cast_async / extract_async, Tool Calling으로 description 전달</summary>

```python
provider = OpenAIProvider(base_url=..., api_key=...)
model = OpenAIModel(model_name, provider=provider)
agent = marvin.Agent(model=model, instructions=system_prompt)

# cast 모드: 텍스트를 target 타입으로 변환 (단일 객체 반환)
result = await marvin.cast_async(data=text, target=schema_class, instructions=system_prompt, agent=agent)

# extract 모드: 텍스트에서 엔티티 추출 (리스트 반환)
results = await marvin.extract_async(data=text, target=schema_class, instructions=system_prompt, agent=agent)
```

Marvin 3.x는 pydantic-ai 기반으로 재작성되었다. `cast_async`는 텍스트를 target 타입으로 변환하고, `extract_async`는 여러 엔티티를 리스트로 추출한다. `OpenAIModel`/`OpenAIProvider`로 커스텀 엔드포인트를 설정하고, Agent의 `instructions`로 system prompt를 전달한다. 내부적으로 Tool Calling 방식을 사용하므로 description이 전달된다.

</details>

<details>
<summary><b>PydanticAI</b> — Agent + output_type, 4가지 모드 지원 (default/tool/json/text)</summary>

```python
model = OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=...))
# default/json: NativeOutput (JSON Schema 기반)
agent = Agent(model, system_prompt=prompt, output_type=NativeOutput(schema_class))
# tool: ToolOutput (Tool Calling 기반)
agent = Agent(model, system_prompt=prompt, output_type=ToolOutput(schema_class))
# text: plain text 응답 후 JSON 파싱
agent = Agent(model, system_prompt=prompt, output_type=str)
result = await agent.run(text)
```

pydantic-ai v1.0.5 기준 `OpenAIChatModel`/`OpenAIProvider`로 모델을 설정한다. **default/json** 모드는 `NativeOutput`으로 JSON Schema 기반 structured output을 사용하고, **tool** 모드는 `ToolOutput`으로 Tool Calling 방식을 사용한다. **text** 모드는 plain text 응답을 받아 JSON으로 파싱한다.

</details>

<details>
<summary><b>Mirascope</b> — openai provider 등록, @call 데코레이터 방식 (tool / json / strict)</summary>

```python
llm.register_provider("openai", scope="openai/model:completions", base_url=..., api_key=...)
fmt = llm.format(schema_class, mode="tool")  # or "json", "strict"

@llm.call("openai/model:completions", format=fmt)
async def do_extract(text: str, sys_prompt: str) -> str:
    return f"{sys_prompt}\n\n{text}"
```

`mirascope.llm.call` 데코레이터와 `format=llm.format(schema_class, mode=...)`를 사용한다. openai provider로 vLLM에 연결하며, `:completions` suffix로 Chat Completions API를 강제한다. `tool` 모드는 Tool Calling, `json` 모드는 JSON Mode, `strict` 모드는 Strict JSON Schema로 동작한다.

</details>

<details>
<summary><b>Guardrails</b> — AsyncGuard + litellm 경유, hosted_vllm provider</summary>

```python
guard = AsyncGuard.for_pydantic(output_class=schema_class)
result = await guard(
    model="hosted_vllm/model",
    api_base=base_url,
    api_key=api_key,
    num_reasks=0,
    messages=[...],
)
```

내부적으로 litellm을 사용하며, `hosted_vllm/` provider로 vLLM에 연결한다. `AsyncGuard`로 네이티브 async를 지원하고, `num_reasks=0`으로 벤치마크 공정성을 위해 재시도 없이 단일 호출만 수행한다.

</details>

<details>
<summary><b>LlamaIndex</b> — FunctionCallingProgram / LLMTextCompletionProgram 2가지 모드 지원</summary>

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.program import FunctionCallingProgram, LLMTextCompletionProgram

llm = OpenAILike(model=model, api_base=base_url, api_key=api_key,
                 is_chat_model=True, is_function_calling_model=True)

# function_calling 모드: Tool Calling 기반
program = FunctionCallingProgram.from_defaults(output_cls=schema_class, llm=llm, prompt=prompt_tpl)

# text 모드: LLM이 텍스트로 JSON을 생성하고 Pydantic으로 파싱
program = LLMTextCompletionProgram.from_defaults(output_cls=schema_class, llm=llm, prompt=prompt_tpl)

result = await program.acall(system_prompt=prompt, text=text)
```

`FunctionCallingProgram`은 Pydantic 스키마를 Function Calling의 tool definition으로 변환하여 전달한다. `LLMTextCompletionProgram`은 LLM에게 텍스트로 JSON을 생성하도록 지시하고 Pydantic으로 파싱한다. `OpenAILike`로 vLLM 등 OpenAI 호환 서버에 연결할 수 있다.

</details>

---

## Benchmark Results

> **Model**: `openai/gpt-oss-120b` (vLLM, temperature=0) | **Dataset**: DeepJSONEval 100 samples | **Total**: 6,600 tests (A/B/C)

> [!IMPORTANT]
> 아래 결과는 **특정 모델(`gpt-oss-120b`) + 특정 서빙 환경(vLLM)** 에서의 측정값이다. 모델이 달라지면 결과가 전혀 다를 수 있다.
> - **기본적으로 모델 성능이 우선**이다. Tool Calling을 안정적으로 지원하는 모델(GPT-4o, Claude 등)에서는 Tool Calling 모드의 실패율이 크게 낮아질 수 있다.
> - **서빙 엔진에 따라 동작이 달라진다.** vLLM의 xgrammar는 JSON Schema의 description을 무시하지만, OpenAI API나 다른 엔진에서는 다를 수 있다.
> - 이 벤치마크는 프레임워크 간 **상대적 특성 차이**를 파악하기 위한 것이며, 절대적 성능 수치로 일반화해서는 안 된다.

### Result Matrix

| Framework / Mode | A_desc | B_nodesc | C_rich | D_both | Overall |
|-----------------|--------|----------|--------|--------|---------|
| **instructor**/tools | 91.8% (86F) | 91.1% (87F) | 93.6% (17F) | | 93.1% |
| **instructor**/tools_strict | 90.8% (92F) | 86.7% (90F) | 94.3% (16F) | | 93.2% |
| **instructor**/json | 93.5% (2F) | 93.0% (3F) | 93.7% (3F) | | **93.4%** |
| **instructor**/json_schema | 80.6% (52F) | 75.4% (49F) | 93.5% (3F) | | 85.6% |
| **instructor**/md_json | 93.8% (3F) | 93.0% (4F) | 93.8% (3F) | | **93.5%** |
| **openai**/default | 89.1% | 86.9% | 93.5% | | 89.8% |
| **openai**/tool_calling | 91.6% (73F) | 95.5% (73F) | 93.4% (11F) | | 93.5% |
| **openai**/json_object | 94.0% (4F) | 93.3% (2F) | 93.8% (4F) | | **93.7%** |
| **langchain**/json_schema | 86.9% | 86.5% | 92.5% | | 88.6% |
| **langchain**/function_calling | 96.9% (92F) | 87.4% (85F) | 94.2% (12F) | | 93.5% |
| **langchain**/json_mode | 94.1% (2F) | 93.1% (3F) | 94.0% (3F) | | **93.8%** |
| **marvin**/cast | 93.6% (3F) | 93.1% (7F) | 93.9% (1F) | | **93.5%** |
| **marvin**/extract | 93.8% (2F) | 93.7% (2F) | 93.7% (2F) | | **93.7%** |
| **pydantic_ai**/tool | 57.1% (99F) | ALL FAIL | 94.0% (38F) | | 93.4% |
| **pydantic_ai**/json | 77.8% (48F) | 74.8% (49F) | 92.8% | | 84.4% |
| **pydantic_ai**/text | 94.5% (2F) | 93.0% (2F) | 93.9% (2F) | | **93.8%** |
| **mirascope**/tool | 93.0% (11F) | 93.3% (9F) | 93.0% (4F) | | **93.1%** |
| **mirascope**/json | 94.7% (9F) | 93.4% (7F) | 93.6% (14F) | | **93.9%** |
| **mirascope**/strict | 78.3% (45F) | 80.3% (46F) | 93.3% | | 86.0% |
| **guardrails**/default | 72.9% (6F) | 72.3% (7F) | 93.1% | | 79.7% |
| **llamaindex**/text | 94.3% (2F) | 93.4% (3F) | 94.0% (2F) | | **93.9%** |
| **llamaindex**/function_calling | ALL FAIL | ALL FAIL | 95.9% (55F) | | 95.9% |

> `(NF)` = 100건 중 N건 실패 (파싱 에러, tool call 미생성 등). `ALL FAIL` = 100건 모두 실패. 성공한 샘플만으로 점수 계산.

---

## Analysis

### 1. Rich Prompt → 모든 프레임워크가 ~93%로 수렴

프롬프트에 필드 설명을 상세히 넣은 조합 C에서 **대부분의 프레임워크가 92~95%로 수렴**하며, 실패율도 대폭 감소한다.

- guardrails: 72% → **93.1%** (실패 0건)
- pydantic_ai/json: 74~78% → **92.8%** (실패 0건)
- mirascope/strict: 78~80% → **93.3%** (실패 0건)
- llamaindex/function_calling: ALL FAIL → **95.9%** (실패에서 성공으로)

프롬프트에 필드 설명을 명시하면 프레임워크 간 성능 차이가 사실상 사라진다.

### 2. 안정적인 모드 vs 불안정한 모드

**안정적 (실패율 < 5%)**: 프롬프트에 스키마를 직접 포함하거나 JSON 기반으로 동작하는 모드
- instructor/json, md_json, langchain/json_mode, openai/json_object, pydantic_ai/text, llamaindex/text, marvin/*, mirascope/tool

**불안정 (실패율 > 30%)**: Tool Calling 방식으로 LLM이 tool call 자체를 생성하지 못하는 경우
- instructor/tools, tools_strict, openai/tool_calling, langchain/function_calling, pydantic_ai/tool, llamaindex/function_calling

vLLM 환경에서 Tool Calling은 `tool_choice`로 강제해도 모델이 content로 JSON을 반환하거나 multiple tool calls를 생성하는 등 불안정한 동작을 보인다.

### 3. JSON Schema (xgrammar) 모드의 한계

- **openai/default**: 86.9~89.1% (실패 0건, 안정적이지만 점수 하락)
- **langchain/json_schema**: 86.5~86.9% (실패 0건, 동일)
- **instructor/json_schema**: 75.4~80.6% (높은 실패율)
- **pydantic_ai/json**: 74.8~77.8% (높은 실패율)

`response_format`으로 JSON Schema를 전달하면 xgrammar가 **구조만 강제**하고 description을 무시한다. openai/default와 langchain/json_schema는 실패 없이 안정적이지만 점수가 낮고, instructor/json_schema와 pydantic_ai/json은 높은 실패율을 동반한다.

### 4. Schema Description 효과 (A vs B) → 미미

A(desc)와 B(nodesc)의 점수 차이는 대부분 1~3%p로 미미하다. JSON Schema 방식에서는 xgrammar가 description을 무시하고, Tool Calling 방식에서도 필드명만으로 어느 정도 추론이 가능하기 때문이다.

### 5. Tool Calling 실패 원인 분석

vLLM에서 Tool Calling 실패는 크게 두 가지 원인으로 나뉜다:

1. **tool call 미생성**: `tool_choice`로 함수를 강제 지정해도 모델이 tool call 대신 content에 JSON을 반환 (instructor/tools_strict, pydantic_ai/tool에서 빈번)
2. **multiple tool calls**: 단일 스키마 추출인데 모델이 여러 개의 tool call을 생성하여 프레임워크가 거부 (instructor에서 발견)

이는 프레임워크 코드 문제가 아닌 **모델의 tool calling 능력 한계**이다.

---

## Conclusion

```
Key Findings:
  1. 프롬프트 엔지니어링 >> 프레임워크 선택
  2. Rich Prompt 사용 시 대부분의 프레임워크가 ~93%로 수렴
  3. JSON 기반 모드(json, md_json, json_mode, json_object, text)가 Tool Calling보다 안정적
  4. vLLM의 xgrammar는 JSON Schema의 description을 완전히 무시
  5. Tool Calling은 vLLM 환경에서 높은 실패율을 보임 (tool call 미생성, multiple tool calls)
```

**vLLM 환경에서 Structured Output 품질을 높이려면:**
1. **프롬프트에 필드 설명을 명시** — 어떤 프레임워크를 사용하든 결과가 ~93%로 수렴
2. **JSON 기반 모드를 우선 사용** — Tool Calling 대비 실패율이 현저히 낮음
3. **Tool Calling은 주의** — 모델이 tool call을 안정적으로 생성하지 못하는 경우가 많음

결론적으로, **프레임워크의 선택보다 프롬프트 엔지니어링이 성능에 훨씬 더 큰 영향을 미치며**, vLLM 환경에서는 JSON 기반 모드가 Tool Calling보다 안정적이다.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API 호환 서버 (vLLM 등)

### Install

```bash
# 전체 설치 (모든 프레임워크 + 서버 + 대시보드)
uv sync --extra dev

# 코어 + 특정 프레임워크만
uv sync --extra instructor --extra openai

# 코어만 (프레임워크 없이)
uv sync
```

<details>
<summary>Optional dependency groups</summary>

| Group | 포함 패키지 |
|-------|-----------|
| `instructor` | instructor |
| `langchain` | langchain-openai |
| `llamaindex` | llama-index-llms-openai-like, llama-index-program-openai |
| `marvin` | marvin |
| `mirascope` | mirascope[openai] |
| `guardrails` | guardrails-ai |
| `pydantic-ai` | pydantic-ai |
| `all` | 위 프레임워크 전체 |
| `server` | fastapi, uvicorn |
| `dashboard` | streamlit, matplotlib, pandas |
| `datasets` | pymupdf |
| `dev` | all + server + dashboard + datasets + python-dotenv |

</details>

### Configuration

```bash
cp .env.example .env
# .env 파일에서 BASE_URL, MODEL, API_KEY 설정
```

### Run Benchmark

```bash
# 전체 벤치마크 (모든 프레임워크 × 모든 조합)
uv run python run_benchmark.py --dataset deepjsoneval

# 특정 프레임워크만
uv run python run_benchmark.py --dataset deepjsoneval --frameworks instructor/tools openai/default

# 특정 조합만
uv run python run_benchmark.py --dataset deepjsoneval --combos A_desc C_rich D_both

# 샘플 수 제한
uv run python run_benchmark.py --dataset deepjsoneval --max-samples 10

# 이전 실행 이어하기 (완료된 프레임워크 건너뛰기)
uv run python run_benchmark.py --resume results/deepjsoneval_20260304_163132

# 서버 설정 직접 지정
uv run python run_benchmark.py --dataset deepjsoneval --base-url http://localhost:8001/v1 --model my-model
```

### Run API Server

```bash
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "framework": "instructor",
    "mode": "tools",
    "markdown": "...",
    "schema_name": "SchemaName",
    "prompt_name": "prompt_name",
    "model": "your-model",
    "base_url": "http://your-server/v1"
  }'
```

### Dashboard

```bash
uv run streamlit run dashboard.py
```

---

## Project Structure

```
struct-output-bench/
├── run_benchmark.py          # CLI 진입점
├── dashboard.py              # Streamlit 대시보드
├── app/
│   ├── benchmark/
│   │   ├── config.py         # 조합(A/B/C/D), 프레임워크/모드 정의
│   │   ├── runner.py         # 벤치마크 실행 엔진
│   │   └── datasets.py       # 데이터셋 어댑터 레지스트리
│   ├── frameworks/           # 프레임워크 어댑터 (8개)
│   │   ├── base.py           # BaseFrameworkAdapter
│   │   ├── registry.py       # @FrameworkRegistry.register 데코레이터
│   │   ├── instructor_fw.py
│   │   ├── openai_native.py
│   │   ├── langchain_fw.py
│   │   ├── marvin_fw.py
│   │   ├── pydantic_ai_fw.py
│   │   ├── mirascope_fw.py
│   │   ├── guardrails_fw.py
│   │   └── llamaindex_fw.py
│   ├── scoring/              # 통합 채점 시스템
│   │   ├── scorer.py         # score_result() 진입점
│   │   ├── matcher.py        # 재귀 flatten + 헝가리안 매칭
│   │   ├── metrics.py        # NED, 숫자 비교, 부울 비교
│   │   ├── hungarian.py      # 헝가리안 알고리즘
│   │   └── schema_utils.py   # JSON Schema 유틸리티
│   ├── datasets/             # 데이터셋별 로더/스키마 생성
│   ├── schemas/              # Pydantic 스키마 정의
│   ├── prompts/              # 프롬프트 템플릿
│   └── api/                  # FastAPI 라우터
├── results/                  # 벤치마크 결과 (gitignored)
└── pyproject.toml
```

### Adding a New Framework

`BaseFrameworkAdapter`를 상속하고 `@FrameworkRegistry.register` 데코레이터를 붙이면 자동 등록된다.

```python
from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

@FrameworkRegistry.register("my_framework")
class MyAdapter(BaseFrameworkAdapter):
    name = "my_framework"
    supported_modes = ["default"]

    async def extract(self, text, schema_class, system_prompt) -> ExtractionResult:
        # 구현
        return ExtractionResult(success=True, data={...})
```

---

## References

### Articles & Tools
- [The best library for structured LLM output](https://simmering.dev/blog/structured_output/) — Paul Simmering
- [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) — Stephen Leo

### Benchmark Datasets
- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) — 10K real-world JSON Schemas for constrained decoding evaluation
- [ExtractBench](https://arxiv.org/abs/2602.12247) — PDF-to-JSON structured extraction, 35 docs + JSON Schema + human-annotated GT (12,867 fields)
- [DeepJSONEval](https://arxiv.org/abs/2509.25922) — Multilingual deep-nested JSON extraction benchmark with schema + input + GT (2,100 instances)

---

## License

MIT License

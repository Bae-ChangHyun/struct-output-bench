# struct-bench

> LLM에서 구조화된 출력(Structured Output)을 만들기 위한 프레임워크는 Instructor, LangChain, Marvin, PydanticAI 등 다양하게 존재한다.
> **struct-bench**는 이들을 동일한 조건에서 비교 테스트하는 벤치마크 도구이다.

Pydantic 스키마와 프롬프트를 정의하고, 여러 프레임워크에 동일한 입력을 넣어 결과 품질을 Ground Truth 기반으로 정량 비교한다. FastAPI 서버로도 개별 프레임워크를 즉시 테스트할 수 있다.

### Supported Frameworks

| 프레임워크 | 모드 | 구조화 방식 | Docs |
|-----------|------|-----------|------|
| **Instructor** | tools, json_schema | Tool Calling / JSON Schema | [docs](https://python.useinstructor.com/) |
| **OpenAI Native** | default, tool_calling, json_object | JSON Schema / Tool Calling / JSON Object | [docs](https://platform.openai.com/docs/guides/structured-outputs) |
| **LangChain** | json_schema, function_calling, json_mode | JSON Schema / Tool Calling / JSON Mode | [docs](https://python.langchain.com/docs/how_to/structured_output/) |
| **Marvin** | cast, extract | Tool Calling (cast_async / extract_async) | [docs](https://askmarvin.ai/docs/text/extraction/) |
| **PydanticAI** | default | Tool Calling | [docs](https://ai.pydantic.dev/output/) |
| **Mirascope** | default | Tool Calling | [docs](https://mirascope.com/docs/mirascope/guides/getting-started/structured-outputs/) |
| **Guardrails** | default | litellm 경유 | [docs](https://www.guardrailsai.com/docs/how_to_guides/generate_structured_data) |
| **LlamaIndex** | default | Tool Calling (OpenAIPydanticProgram) | [docs](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/) |

---

## Table of Contents

- [Motivation](#motivation)
- [Background](#background)
- [Experiment Design](#experiment-design)
- [How Each Framework Works](#how-each-framework-works)
- [Benchmark Results](#benchmark-results)
- [Analysis](#analysis)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [References](#references)

---

## Motivation

LLM에서 Pydantic 모델 형태의 구조화된 출력을 얻기 위한 프레임워크가 많아졌다. Instructor, LangChain, Marvin, PydanticAI, Mirascope, Guardrails, LlamaIndex 등 각각 다른 방식으로 동일한 문제를 풀고 있다. 하지만 **같은 모델, 같은 스키마, 같은 프롬프트를 넣었을 때 과연 결과가 동일한가?**

struct-bench는 이 질문에 답하기 위한 도구이다:

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

- **Description 포함 스키마**: 모든 필드에 `Field(description="...")` 포함 (60+ 필드, `Literal` 타입 제약 포함)
- **Description 없는 스키마**: 동일 구조이나 description 없이 타입과 기본값만 정의

10건의 테스트 문서와 Ground Truth를 준비하고, 7개 프레임워크(9개 모드) × 3가지 조합 = 총 **270건**의 실험을 수행하였다.

### 3 Test Combinations

| 조합 | 스키마 | 프롬프트 | 실험 의도 |
|------|--------|---------|----------|
| **A: Schema(desc) + Prompt(minimal)** | description 포함 | 필드 설명 없는 최소 프롬프트 | description 필드에만 의존했을 때의 성능 측정 |
| **B: Schema(no desc) + Prompt(minimal)** | description 없음 | 필드 설명 없는 최소 프롬프트 | 아무 설명도 없을 때의 기저(baseline) 성능 측정 |
| **C: Schema(no desc) + Prompt(rich)** | description 없음 | 모든 필드 설명이 포함된 상세 프롬프트 | 프롬프트로만 설명을 제공했을 때의 성능 측정 |

### Prompt Design

**Minimal Prompt**: 최소한의 지시만 포함 ("Extract structured information from the given text.")

**Rich Prompt**: 각 섹션과 필드에 대한 상세한 설명을 포함. Literal 타입의 허용 값, 각 필드의 용도, 날짜 형식 등을 명시한 약 80줄 분량의 가이드.

---

## How Each Framework Works

<details>
<summary><b>Instructor</b> — Tool Calling / JSON Schema, 자동 retry 및 validation 내장</summary>

```python
client = instructor.from_provider("ollama/model", base_url=...)
result = client.chat.completions.create(
    response_model=schema_class,
    messages=[...],
)
```

`instructor.from_provider()`는 Tool Calling 방식으로 Pydantic 스키마를 tool definition에 넣어 전달한다. description 필드가 tool definition에 포함되므로 LLM이 필드의 의미를 파악할 수 있다. 자동 retry 및 validation이 내장되어 있다. `json_schema` 모드에서는 `response_format` 기반으로 동작한다.

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
<summary><b>LangChain</b> — json_schema / function_calling 두 가지 모드 지원</summary>

```python
llm = ChatOpenAI(model=model, base_url=...)
structured_llm = llm.with_structured_output(schema_class, method="json_schema")
result = await structured_llm.ainvoke(messages)
```

`json_schema` 모드에서는 `response_format` 기반으로 동작하고, `function_calling` 모드에서는 Tool Calling 방식으로 동작한다.

</details>

<details>
<summary><b>Marvin</b> — pydantic_ai 기반 Tool Calling, description 전달됨</summary>

```python
provider = OpenAIProvider(base_url=..., api_key=...)
model = OpenAIModel(model_name, provider=provider)
agent = marvin.Agent(model=model, instructions=system_prompt)
result = agent.run(text, result_type=schema_class)
```

pydantic_ai의 `OpenAIModel`/`OpenAIProvider`로 모델을 주입한 뒤 `marvin.Agent`를 통해 추출한다. 내부적으로 Tool Calling 방식을 사용하므로 description이 전달된다.

</details>

<details>
<summary><b>PydanticAI</b> — Agent + output_type, Tool Calling 방식</summary>

```python
model = OpenAIModel(model_name, provider=OpenAIProvider(base_url=...))
agent = Agent(model, system_prompt=prompt, output_type=schema_class)
result = await agent.run(text)
```

`Agent`에 `output_type`으로 스키마를 전달하며, Tool Calling 방식으로 동작한다.

</details>

<details>
<summary><b>Mirascope</b> — ollama provider 등록, @call 데코레이터 방식</summary>

```python
register_provider("ollama", scope="ollama/", base_url=...)

@call("ollama/model", format=schema_class)
def do_extract(text, sys_prompt):
    return f"{sys_prompt}\n\n{text}"
```

`mirascope.llm.call` 데코레이터와 `format=schema_class`를 사용한다. ollama provider로 등록하여 vLLM에 연결하며, Tool Calling 방식으로 동작한다.

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
<summary><b>LlamaIndex</b> — OpenAIPydanticProgram, Function Calling 기반 구조화 추출</summary>

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.program.openai import OpenAIPydanticProgram

llm = OpenAILike(model=model, api_base=base_url, api_key=api_key,
                 is_chat_model=True, is_function_calling_model=True)
program = OpenAIPydanticProgram.from_defaults(
    output_cls=schema_class,
    prompt_template_str="{system_prompt}\n\n{text}",
    llm=llm,
)
result = program(system_prompt=prompt, text=text)
```

`OpenAIPydanticProgram`은 Pydantic 스키마를 Function Calling의 tool definition으로 변환하여 전달한다. `OpenAILike`로 vLLM 등 OpenAI 호환 서버에 연결할 수 있다. description이 tool definition에 포함되므로 LLM이 필드의 의미를 파악할 수 있다.

</details>

---

## Benchmark Results

### Result Matrix

| Framework / Mode | A_desc | B_nodesc | C_rich | Overall |
|-----------------|--------|----------|--------|---------|
| [Instructor](https://python.useinstructor.com/) / tools | 94.6% | 94.9% | 95.5% | **95.0%** |
| [Instructor](https://python.useinstructor.com/) / json_schema | 94.9% | 94.7% | 96.0% | **95.2%** |
| [OpenAI Native](https://platform.openai.com/docs/guides/structured-outputs) / default | 82.5% | 85.2% | 96.0% | 87.9% |
| [LangChain](https://python.langchain.com/docs/how_to/structured_output/) / json_schema | 84.0% (1F) | 81.1% | 96.0% | 87.1% |
| [LangChain](https://python.langchain.com/docs/how_to/structured_output/) / function_calling | ALL FAIL | ALL FAIL | 96.0% | 96.0% |
| [Marvin](https://askmarvin.ai/docs/text/extraction/) / default | 93.5% | 94.4% | 95.8% | **94.6%** |
| [PydanticAI](https://ai.pydantic.dev/output/) / default | 38.7% | 40.5% (2F) | 96.0% | 59.7% |
| [Mirascope](https://mirascope.com/docs/mirascope/guides/getting-started/structured-outputs/) / default | 35.0% (5F) | 34.2% (5F) | 96.0% | 65.3% |
| [Guardrails](https://www.guardrailsai.com/docs/how_to_guides/generate_structured_data) / default | 7.8% (6F) | 6.0% (7F) | 96.0% | 59.4% |
| **AVG** | 73.6% (22F) | 76.0% (24F) | 95.9% (0F) | — |

> `(NF)` = 10건 중 N건 실패 (파싱 에러 또는 validation 실패). `ALL FAIL` = 10건 모두 실패.

### Scoring

Ground Truth 기반 100점 만점 채점 방식을 사용하였다. 12개 카테고리로 나뉘며, 각 항목은 키워드 매칭, 개수 일치, 정확도 등을 종합적으로 평가한다.

---

## Analysis

### 1. Rich Prompt → 모든 프레임워크가 ~96%로 수렴

가장 두드러진 결과는, 프롬프트에 필드 설명을 상세히 넣은 조합 C에서 **모든 프레임워크가 95.5~96.0%로 수렴**했다는 점이다.

- Guardrails: 6~8% -> **96%** (12배 이상 향상)
- Mirascope: 34~35% -> **96%** (약 3배 향상)
- PydanticAI: 38~40% -> **96%** (약 2.5배 향상)
- LangChain function_calling: ALL FAIL -> **96%** (실패에서 성공으로)

프롬프트에 필드 설명을 명시하면 프레임워크 간 성능 차이가 사실상 사라진다. 이는 LLM이 "무엇을 추출해야 하는가"에 대한 정보를 프롬프트에서 직접 얻을 수 있기 때문이다.

### 2. Tool Calling 방식만 A/B에서 고성능

프롬프트에 설명이 없는 조합 A/B에서도 높은 점수를 유지한 프레임워크들이 있다:

- **Instructor** (94.6~95.5%): `from_provider`가 tool definition에 description을 포함하여 전달
- **Marvin** (93.5~94.4%): pydantic_ai 기반 Tool Calling으로 description 전달

공통점은 **Tool Calling 방식으로 description을 LLM에 직접 전달**한다는 것이다.

### 3. JSON Schema (xgrammar) → A/B에서 중간 성능

- **OpenAI Native**: 82.5~85.2%
- **LangChain json_schema**: 81.1~84.0%

`response_format`으로 JSON Schema를 전달하지만, vLLM의 xgrammar가 description을 무시하기 때문에 구조만 강제되고 의미 정보가 부족하다.

### 4. A/B에서 저성능 프레임워크

- **PydanticAI** (38~40%): Tool Calling 방식이지만, NoDesc 스키마에서 description이 없으면 tool definition에도 설명이 포함되지 않아 낮은 성능
- **Mirascope** (34~35%): Literal 타입 validation 실패가 빈번하게 발생 (10건 중 5건 실패)
- **Guardrails** (6~8%): litellm을 경유하면서 description 전달이 불안정하고, 10건 중 6~7건이 실패

### 5. Schema Description 효과 (A vs B) → 미미

| 조합 | 전체 평균 | 실패 수 |
|------|----------|---------|
| A (desc + minimal) | 73.6% | 22건 실패 |
| B (no desc + minimal) | 76.0% | 24건 실패 |

A와 B의 차이는 2.4%p로 거의 없다. JSON Schema 방식에서는 xgrammar가 무시하고, Tool Calling 방식에서는 필드명만으로도 어느 정도 추론이 가능하기 때문이다.

### 6. Literal Type Issue

스키마에 포함된 `Literal` 타입 제약이 실패의 주요 원인이었다. LLM이 정확한 Literal 값 대신 유사한 값을 생성하는 경우가 빈번하였으며, 이로 인해 Pydantic validation error가 발생한다. `langchain/function_calling`이 조합 A/B에서 전부 실패한 주요 원인이기도 하다.

---

## Conclusion

```
Key Findings:
  1. 프롬프트 엔지니어링 >> 프레임워크 선택
  2. Rich Prompt 사용 시 모든 프레임워크가 ~96%로 동일한 성능
  3. 스키마 description에만 의존하면 Tool Calling 계열에서만 효과 있음
  4. vLLM의 xgrammar는 JSON Schema의 description을 완전히 무시
```

**vLLM 환경에서 Structured Output 품질을 높이려면 프롬프트에 필드 설명을 명시하는 것이 가장 확실하고 효과적인 방법이다.** 이 경우 어떤 프레임워크를 사용하든 결과는 동일하다 (~96%).

스키마의 `Field(description=...)` 에만 의존하는 전략은, Tool Calling 방식으로 description을 LLM에 직접 전달하는 프레임워크(Instructor, Marvin)에서만 유효하다. JSON Schema 방식(OpenAI native, LangChain json_schema)에서는 xgrammar가 description을 무시하므로 효과가 없다.

결론적으로, **프레임워크의 선택보다 프롬프트 엔지니어링이 성능에 훨씬 더 큰 영향을 미친다.**

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API 호환 서버 (vLLM 등)

### Install

```bash
uv sync
```

### Run Benchmark

```bash
cd tests
uv run run_multi_resume_benchmark.py
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
